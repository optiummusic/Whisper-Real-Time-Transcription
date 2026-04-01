// ─── lxdb/reader.rs ───────────────────────────────────────────────────────────
// Zero-copy reader that operates directly over the raw bytes of a .lxdb file.
//
// Lookup: O(log W) binary search over the sorted word index.
// Translate: O(1) concept-table read + O(1) string-pool dereference.
//
// No heap allocations during translate() — all strings are slices into the
// original byte slice.
// ─────────────────────────────────────────────────────────────────────────────

use crate::format::{
    read_u16, read_u32, read_u64,
    HEADER_SIZE, LANG_ENTRY_SIZE, WORD_ENTRY_SIZE,
    HDR_LANG_COUNT, HDR_CONCEPT_COUNT, HDR_WORD_COUNT,
    HDR_LANG_TABLE_OFF, HDR_CONCEPT_OFF, HDR_WORD_INDEX_OFF, HDR_STRING_POOL_OFF,
    word_key_hash, MAGIC, VERSION,
};
use crate::LxdbError;

/// A validated, zero-copy view over a `.lxdb` file's raw bytes.
///
/// The lifetime `'a` is tied to whatever owns the bytes (e.g. `Vec<u8>`,
/// `Arc<[u8]>`, or a memory-mapped file).
pub struct LxdbReader<'a> {
    data:         &'a [u8],
    lang_count:   u16,
    concept_count:u32,
    word_count:   u32,
    lang_tbl_off: u32,
    concept_off:  u32,
    word_idx_off: u32,
    pool_off:     u32,
}

impl<'a> LxdbReader<'a> {
    /// Parse and validate a raw byte slice.  Fails fast on magic / version mismatch.
    pub fn new(data: &'a [u8]) -> Result<Self, LxdbError> {
        if data.len() < HEADER_SIZE {
            return Err(LxdbError::InvalidMagic);
        }
        if &data[0..4] != &MAGIC {
            return Err(LxdbError::InvalidMagic);
        }
        if data[4] != VERSION {
            return Err(LxdbError::UnsupportedVersion(data[4]));
        }

        Ok(Self {
            data,
            lang_count:    read_u16(data, HDR_LANG_COUNT),
            concept_count: read_u32(data, HDR_CONCEPT_COUNT),
            word_count:    read_u32(data, HDR_WORD_COUNT),
            lang_tbl_off:  read_u32(data, HDR_LANG_TABLE_OFF),
            concept_off:   read_u32(data, HDR_CONCEPT_OFF),
            word_idx_off:  read_u32(data, HDR_WORD_INDEX_OFF),
            pool_off:      read_u32(data, HDR_STRING_POOL_OFF),
        })
    }

    // ── Language resolution ────────────────────────────────────────────────────

    /// Returns the numeric ID for `lang_code`, or None if not present.
    pub fn lang_id(&self, lang_code: &str) -> Option<u16> {
        let code = lang_code.to_lowercase();
        for i in 0..self.lang_count {
            if self.lang_code(i) == code.as_str() {
                return Some(i);
            }
        }
        None
    }

    /// BCP-47 code for the i-th language entry.
    pub fn lang_code(&self, i: u16) -> &str {
        let entry_off = self.lang_tbl_off as usize + i as usize * LANG_ENTRY_SIZE;
        let code_off  = read_u32(self.data, entry_off) as usize;
        self.pool_str(code_off)
    }

    /// Human-readable name for the i-th language entry.
    pub fn lang_name(&self, i: u16) -> &str {
        let entry_off = self.lang_tbl_off as usize + i as usize * LANG_ENTRY_SIZE;
        let name_off  = read_u32(self.data, entry_off + 4) as usize;
        self.pool_str(name_off)
    }

    /// All (code, name) pairs in index order.
    pub fn languages(&self) -> impl Iterator<Item = (&str, &str)> {
        (0..self.lang_count).map(move |i| (self.lang_code(i), self.lang_name(i)))
    }

    // ── Core lookup ───────────────────────────────────────────────────────────

    /// Looks up `word` in `src_lang` and translates it to `tgt_lang`.
    ///
    /// Returns `None` if the word is not in the index or has no translation
    /// in the target language.
    ///
    /// No allocation: the returned `&str` is a slice into `self.data`.
    pub fn translate<'b>(&'b self, word: &str, src_lang: &str, tgt_lang: &str) -> Option<&'b str>
    where 'a: 'b
    {
        let src_id  = self.lang_id(src_lang)?;
        let tgt_id  = self.lang_id(tgt_lang)?;
        let cid     = self.find_concept(word, src_id)?;
        let word_off= self.concept_word_off(cid, tgt_id);
        if word_off == 0 { return None; }  // no translation in target lang
        Some(self.pool_str(word_off as usize))
    }

    /// Like `translate` but accepts already-resolved numeric language IDs.
    /// Useful in hot loops where you've already called `lang_id` once.
    pub fn translate_by_id<'b>(&'b self, word: &str, src_id: u16, tgt_id: u16) -> Option<&'b str>
    where 'a: 'b
    {
        let cid      = self.find_concept(word, src_id)?;
        let word_off = self.concept_word_off(cid, tgt_id);
        if word_off == 0 { return None; }
        Some(self.pool_str(word_off as usize))
    }

    pub fn translate_concept(&self, concept_id: u32, lang_code: &str) -> Option<&str> {
        let lang_id  = self.lang_id(lang_code)?;
        let word_off = self.concept_word_off(concept_id, lang_id);
        if word_off == 0 { return None; }
        Some(self.pool_str(word_off as usize))
    }

    // ── Word index binary search ───────────────────────────────────────────────

    fn find_concept(&self, word: &str, lang_id: u16) -> Option<u32> {
        let target_hash = word_key_hash(self.lang_code(lang_id), word);

        // Binary search over the sorted word-index.
        let base = self.word_idx_off as usize;
        let n    = self.word_count as usize;

        let mut lo = 0usize;
        let mut hi = n;
        while lo < hi {
            let mid     = lo + (hi - lo) / 2;
            let off     = base + mid * WORD_ENTRY_SIZE;
            let mid_h   = read_u64(self.data, off);

            match mid_h.cmp(&target_hash) {
                std::cmp::Ordering::Equal => {
                    // Verify lang_id in case of hash collision.
                    let mid_lang = read_u16(self.data, off + 12);
                    if mid_lang == lang_id {
                        return Some(read_u32(self.data, off + 8));
                    }
                    // Hash collision with different lang — scan neighbours.
                    return self.scan_hash_collision(base, n, target_hash, lang_id);
                }
                std::cmp::Ordering::Less    => lo = mid + 1,
                std::cmp::Ordering::Greater => hi = mid,
            }
        }
        None
    }

    /// Linear scan around a hash-collision cluster (extremely rare in practice).
    fn scan_hash_collision(&self, base: usize, n: usize, h: u64, lang_id: u16) -> Option<u32> {
        for i in 0..n {
            let off = base + i * WORD_ENTRY_SIZE;
            if read_u64(self.data, off) == h && read_u16(self.data, off + 12) == lang_id {
                return Some(read_u32(self.data, off + 8));
            }
        }
        None
    }

    // ── Concept table ─────────────────────────────────────────────────────────

    /// Returns the string-pool offset for concept `cid` in language `lang_id`.
    /// 0 means "no translation stored".
    #[inline]
    fn concept_word_off(&self, concept_id: u32, lang_id: u16) -> u32 {
        let row_off = self.concept_off as usize
            + concept_id as usize * self.lang_count as usize * 4
            + lang_id as usize * 4;
        read_u32(self.data, row_off)
    }

    // ── String pool ───────────────────────────────────────────────────────────

    #[inline]
    fn pool_str(&self, offset: usize) -> &str {
        let start = self.pool_off as usize + offset;
        let end   = self.data[start..]
            .iter()
            .position(|&b| b == 0)
            .map(|n| start + n)
            .unwrap_or(self.data.len());
        std::str::from_utf8(&self.data[start..end]).unwrap_or("")
    }

    // ── Metadata ──────────────────────────────────────────────────────────────

    pub fn lang_count(&self)    -> u16  { self.lang_count }
    pub fn concept_count(&self) -> u32  { self.concept_count }
    pub fn word_count(&self)    -> u32  { self.word_count }
    pub fn file_size(&self)     -> usize { self.data.len() }
}
