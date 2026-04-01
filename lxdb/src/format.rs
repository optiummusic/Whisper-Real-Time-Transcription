// ─── lxdb/format.rs ───────────────────────────────────────────────────────────
// Physical layout constants and zero-copy read helpers.
// All integers are little-endian throughout the format.
// ─────────────────────────────────────────────────────────────────────────────

/// File magic: "LXDB"
pub const MAGIC: [u8; 4] = *b"LXDB";
pub const VERSION: u8 = 1;

// ── Section sizes ──────────────────────────────────────────────────────────────

/// Header is exactly 32 bytes.
pub const HEADER_SIZE: usize = 32;

/// Each language-table entry: code_off(4) + name_off(4) + reserved(2) = 10 bytes.
pub const LANG_ENTRY_SIZE: usize = 10;

/// Each word-index entry: hash(8) + concept_id(4) + lang_id(2) = 14 bytes.
pub const WORD_ENTRY_SIZE: usize = 14;

// ── Header offsets (byte positions inside the 32-byte header) ─────────────────

pub const HDR_MAGIC:          usize = 0;   // [u8; 4]
pub const HDR_VERSION:        usize = 4;   // u8
pub const HDR_FLAGS:          usize = 5;   // u8  (reserved, always 0)
pub const HDR_LANG_COUNT:     usize = 6;   // u16 LE
pub const HDR_CONCEPT_COUNT:  usize = 8;   // u32 LE
pub const HDR_WORD_COUNT:     usize = 12;  // u32 LE
pub const HDR_LANG_TABLE_OFF: usize = 16;  // u32 LE → offset of LANG TABLE
pub const HDR_CONCEPT_OFF:    usize = 20;  // u32 LE → offset of CONCEPT TABLE
pub const HDR_WORD_INDEX_OFF: usize = 24;  // u32 LE → offset of WORD INDEX
pub const HDR_STRING_POOL_OFF:usize = 28;  // u32 LE → offset of STRING POOL

// ── Zero-copy read helpers ─────────────────────────────────────────────────────

#[inline]
pub fn read_u16(data: &[u8], off: usize) -> u16 {
    u16::from_le_bytes(data[off..off + 2].try_into().unwrap())
}

#[inline]
pub fn read_u32(data: &[u8], off: usize) -> u32 {
    u32::from_le_bytes(data[off..off + 4].try_into().unwrap())
}

#[inline]
pub fn read_u64(data: &[u8], off: usize) -> u64 {
    u64::from_le_bytes(data[off..off + 8].try_into().unwrap())
}

// ── Write helpers ──────────────────────────────────────────────────────────────

#[inline]
pub fn write_u16(buf: &mut Vec<u8>, v: u16) { buf.extend_from_slice(&v.to_le_bytes()); }

#[inline]
pub fn write_u32(buf: &mut Vec<u8>, v: u32) { buf.extend_from_slice(&v.to_le_bytes()); }

#[inline]
pub fn write_u64(buf: &mut Vec<u8>, v: u64) { buf.extend_from_slice(&v.to_le_bytes()); }

// ── FNV-1a 64-bit hash ─────────────────────────────────────────────────────────
// Used to key the word index: hash("{lang_code}:{normalized_word}").
// The colon separator makes collisions across languages impossible even
// for identical surface forms.

const FNV_OFFSET: u64 = 14_695_981_039_346_656_037;
const FNV_PRIME:  u64 = 1_099_511_628_211;

pub fn fnv1a_64(s: &str) -> u64 {
    let mut h = FNV_OFFSET;
    for b in s.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(FNV_PRIME);
    }
    h
}

// ── Word key helper ────────────────────────────────────────────────────────────
// Produces the canonical index key for any (lang, word) pair.
// Normalisation: lowercase, trimmed.  Multi-word phrases are left intact
// so ngram entries ("good morning") hash as a unit.

pub fn word_key(lang_code: &str, word: &str) -> String {
    format!("{}:{}", lang_code.to_lowercase(), word.trim().to_lowercase())
}

pub fn word_key_hash(lang_code: &str, word: &str) -> u64 {
    fnv1a_64(&word_key(lang_code, word))
}
