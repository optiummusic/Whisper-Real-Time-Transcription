// ─── lxdb/builder.rs ──────────────────────────────────────────────────────────
// Compiles a parsed TOML source into a well-formed .lxdb binary blob.
//
// Pipeline:
//   1. Intern all strings into a string pool (deduplication via HashMap).
//   2. Lay out the LANG TABLE with pointers into the pool.
//   3. Build the CONCEPT TABLE: for each concept, store one word_off per lang.
//   4. Build the WORD INDEX: (hash, concept_id, lang_id) triples, then sort.
//   5. Emit the HEADER with computed section offsets.
//   6. Concatenate: header ++ lang_table ++ concept_table ++ word_index ++ pool.
// ─────────────────────────────────────────────────────────────────────────────

use std::collections::HashMap;

use crate::format::{
    self, HEADER_SIZE, LANG_ENTRY_SIZE, WORD_ENTRY_SIZE,
    write_u16, write_u32, write_u64, word_key_hash, MAGIC, VERSION,
};
use crate::toml_schema::LxdbToml;
use crate::LxdbError;

/// Compiles an [`LxdbToml`] into raw bytes ready to write to a `.lxdb` file.
pub fn compile(source: &LxdbToml) -> Result<Vec<u8>, LxdbError> {
    Builder::new(source)?.build()
}

// ─── internal ─────────────────────────────────────────────────────────────────

struct Builder<'a> {
    source:    &'a LxdbToml,
    lang_ids:  HashMap<&'a str, u16>,   // lang_code → index in lang table
    lang_codes: Vec<&'a str>,           // ordered list of lang codes
}

impl<'a> Builder<'a> {
    fn new(source: &'a LxdbToml) -> Result<Self, LxdbError> {
        if source.languages.is_empty() {
            return Err(LxdbError::Build("no languages defined".into()));
        }
        // Deterministic language order: sort codes alphabetically.
        let mut lang_codes: Vec<&str> = source.languages.keys().map(String::as_str).collect();
        lang_codes.sort_unstable();

        let lang_ids: HashMap<&str, u16> = lang_codes
            .iter()
            .enumerate()
            .map(|(i, &code)| (code, i as u16))
            .collect();

        Ok(Self { source, lang_ids, lang_codes })
    }

    fn build(self) -> Result<Vec<u8>, LxdbError> {
        let lang_count     = self.lang_codes.len() as u16;
        let concept_count  = self.source.concepts.len() as u32;

        // ── 1. String pool ────────────────────────────────────────────────────
        let mut pool: Vec<u8> = vec![0u8];  // offset 0 = null sentinel
        let mut pool_map: HashMap<String, u32> = HashMap::new();

        let intern = |pool: &mut Vec<u8>, pool_map: &mut HashMap<String, u32>, s: &str| -> u32 {
            if s.is_empty() { return 0; }   // map empty → null sentinel
            if let Some(&off) = pool_map.get(s) { return off; }
            let off = pool.len() as u32;
            pool.extend_from_slice(s.as_bytes());
            pool.push(0);
            pool_map.insert(s.to_string(), off);
            off
        };

        // ── 2. Language table ─────────────────────────────────────────────────
        // Pre-intern all lang codes + names before building the table bytes so
        // that the pool is populated before we serialise.
        let mut lang_entries: Vec<(u32, u32)> = Vec::with_capacity(self.lang_codes.len());
        for &code in &self.lang_codes {
            let name = self.source.languages.get(code).map(String::as_str).unwrap_or(code);
            let code_off = intern(&mut pool, &mut pool_map, code);
            let name_off = intern(&mut pool, &mut pool_map, name);
            lang_entries.push((code_off, name_off));
        }

        // ── 3. Concept table + word index ─────────────────────────────────────
        // concept_table: concept_count × lang_count × u32
        let row_size        = self.lang_codes.len();
        let mut concept_tbl: Vec<u32> = vec![0u32; concept_count as usize * row_size];

        struct IndexEntry {
            hash:       u64,
            concept_id: u32,
            lang_id:    u16,
        }
        let mut word_index: Vec<IndexEntry> = Vec::new();

        for (cid, concept) in self.source.concepts.iter().enumerate() {
            let cid_u32 = cid as u32;

            use std::collections::HashSet;
            let mut seen: HashSet<(u16, String)> = HashSet::new();

            //BASE
            for (lang, word) in &concept.forms {
                let Some(&lang_id) = self.lang_ids.get(lang.as_str()) else {
                    continue;
                };

                let norm = word.trim().to_lowercase();
                if norm.is_empty() {
                    continue;
                }

                if !seen.insert((lang_id, norm.clone())) {
                    continue;
                }

                let word_off = intern(&mut pool, &mut pool_map, &norm);

                concept_tbl[cid * row_size + lang_id as usize] = word_off;

                word_index.push(IndexEntry {
                    hash: word_key_hash(lang, &norm),
                    concept_id: cid_u32,
                    lang_id,
                });
            }

            //GENERATED
            for (lang, forms) in &concept.generated {
                let Some(&lang_id) = self.lang_ids.get(lang.as_str()) else {
                    continue;
                };

                for word in forms {
                    let norm = word.trim().to_lowercase();
                    if norm.is_empty() {
                        continue;
                    }

                    if !seen.insert((lang_id, norm.clone())) {
                        continue;
                    }

                    let _ = intern(&mut pool, &mut pool_map, &norm);

                    word_index.push(IndexEntry {
                        hash: word_key_hash(lang, &norm),
                        concept_id: cid_u32,
                        lang_id,
                    });
                }
            }

            //CUSTOM
            for (lang, forms) in &concept.custom {
                let Some(&lang_id) = self.lang_ids.get(lang.as_str()) else {
                    continue;
                };

                for word in forms {
                    let norm = word.trim().to_lowercase();
                    if norm.is_empty() {
                        continue;
                    }

                    if !seen.insert((lang_id, norm.clone())) {
                        continue;
                    }

                    let _ = intern(&mut pool, &mut pool_map, &norm);

                    word_index.push(IndexEntry {
                        hash: word_key_hash(lang, &norm),
                        concept_id: cid_u32,
                        lang_id,
                    });
                }
            }
        }

        // Sort the word index by hash for binary search at runtime.
        word_index.sort_unstable_by_key(|e| e.hash);
        let word_count = word_index.len() as u32;

        // ── 4. Compute section offsets ────────────────────────────────────────
        let lang_table_off   = HEADER_SIZE as u32;
        let concept_table_off= lang_table_off   + lang_count as u32 * LANG_ENTRY_SIZE as u32;
        let word_index_off   = concept_table_off + concept_count * row_size as u32 * 4;
        let string_pool_off  = word_index_off   + word_count * WORD_ENTRY_SIZE as u32;

        // ── 5. Serialise everything ───────────────────────────────────────────
        let capacity = string_pool_off as usize + pool.len();
        let mut out: Vec<u8> = Vec::with_capacity(capacity);

        // Header
        out.extend_from_slice(&MAGIC);
        out.push(VERSION);
        out.push(0); // flags
        write_u16(&mut out, lang_count);
        write_u32(&mut out, concept_count);
        write_u32(&mut out, word_count);
        write_u32(&mut out, lang_table_off);
        write_u32(&mut out, concept_table_off);
        write_u32(&mut out, word_index_off);
        write_u32(&mut out, string_pool_off);
        debug_assert_eq!(out.len(), HEADER_SIZE);

        // Language table
        for (code_off, name_off) in lang_entries {
            write_u32(&mut out, code_off);
            write_u32(&mut out, name_off);
            write_u16(&mut out, 0); // reserved
        }

        // Concept table
        for &word_off in &concept_tbl {
            write_u32(&mut out, word_off);
        }

        // Word index
        for e in &word_index {
            write_u64(&mut out, e.hash);
            write_u32(&mut out, e.concept_id);
            write_u16(&mut out, e.lang_id);
        }

        // String pool
        out.extend_from_slice(&pool);

        Ok(out)
    }
}

// ─── Stats helper ─────────────────────────────────────────────────────────────

/// Human-readable summary printed after a successful compile.
pub struct CompileStats {
    pub languages:    usize,
    pub concepts:     usize,
    pub word_entries: usize,
    pub file_bytes:   usize,
}

impl std::fmt::Display for CompileStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "compiled: {} languages, {} concepts, {} index entries, {} bytes on disk",
            self.languages, self.concepts, self.word_entries, self.file_bytes,
        )
    }
}

pub fn stats_from(blob: &[u8], source: &LxdbToml) -> CompileStats {
    let word_count = format::read_u32(blob, crate::format::HDR_WORD_COUNT);
    CompileStats {
        languages:    source.languages.len(),
        concepts:     source.concepts.len(),
        word_entries: word_count as usize,
        file_bytes:   blob.len(),
    }
}
