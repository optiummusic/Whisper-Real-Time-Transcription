// ─── lxdb/tests/integration.rs ────────────────────────────────────────────────

use lxdb::{compile_toml, LxdbReader};

const SAMPLE_TOML: &str = r#"
[meta]
version     = 1
description = "Test dictionary"

[languages]
en = "English"
uk = "Ukrainian"
de = "German"

[[concepts]]
en = "hello"
uk = "привіт"
de = "hallo"

[[concepts]]
en = "world"
uk = "світ"
de = "welt"

[[concepts]]
en = "good morning"
uk = "доброго ранку"
de = "guten morgen"

[[concepts]]
en = "cat"
uk = "кіт"
de = "katze"

[[concepts]]
en = "house"
uk = "будинок"
"#;

fn make_reader(toml: &str) -> (Vec<u8>, ) {
    let blob = compile_toml(toml).expect("compile failed");
    (blob,)
}

#[test]
fn round_trip_basic() {
    let (blob,) = make_reader(SAMPLE_TOML);
    let r = LxdbReader::new(&blob).expect("reader failed");

    assert_eq!(r.lang_count(), 3);
    assert_eq!(r.concept_count(), 5);
}

#[test]
fn translate_en_to_uk() {
    let (blob,) = make_reader(SAMPLE_TOML);
    let r = LxdbReader::new(&blob).unwrap();

    assert_eq!(r.translate("hello", "en", "uk"), Some("привіт"));
    assert_eq!(r.translate("world", "en", "uk"), Some("світ"));
    assert_eq!(r.translate("cat",   "en", "uk"), Some("кіт"));
}

#[test]
fn translate_uk_to_en() {
    let (blob,) = make_reader(SAMPLE_TOML);
    let r = LxdbReader::new(&blob).unwrap();

    assert_eq!(r.translate("привіт", "uk", "en"), Some("hello"));
    assert_eq!(r.translate("кіт",    "uk", "en"), Some("cat"));
}

#[test]
fn translate_en_to_de() {
    let (blob,) = make_reader(SAMPLE_TOML);
    let r = LxdbReader::new(&blob).unwrap();

    assert_eq!(r.translate("hallo",       "de", "en"), Some("hello"));
    assert_eq!(r.translate("guten morgen","de", "en"), Some("good morning"));
}

#[test]
fn ngram_phrase_lookup() {
    let (blob,) = make_reader(SAMPLE_TOML);
    let r = LxdbReader::new(&blob).unwrap();

    // "good morning" is stored as a single multi-word concept
    assert_eq!(r.translate("good morning", "en", "uk"), Some("доброго ранку"));
    assert_eq!(r.translate("good morning", "en", "de"), Some("guten morgen"));
}

#[test]
fn missing_translation_returns_none() {
    let (blob,) = make_reader(SAMPLE_TOML);
    let r = LxdbReader::new(&blob).unwrap();

    // "house" has no German translation in the fixture
    assert_eq!(r.translate("house", "en", "de"), None);
}

#[test]
fn unknown_word_returns_none() {
    let (blob,) = make_reader(SAMPLE_TOML);
    let r = LxdbReader::new(&blob).unwrap();

    assert_eq!(r.translate("spaceship", "en", "uk"), None);
}

#[test]
fn case_insensitive_lookup() {
    let (blob,) = make_reader(SAMPLE_TOML);
    let r = LxdbReader::new(&blob).unwrap();

    // Normalisation lowercases input
    assert_eq!(r.translate("Hello", "en", "uk"), Some("привіт"));
    assert_eq!(r.translate("HELLO", "en", "uk"), Some("привіт"));
}

#[test]
fn language_metadata() {
    let (blob,) = make_reader(SAMPLE_TOML);
    let r = LxdbReader::new(&blob).unwrap();

    // Languages are stored alphabetically (de, en, uk)
    let langs: Vec<(&str, &str)> = r.languages().collect();
    assert!(langs.contains(&("en", "English")));
    assert!(langs.contains(&("uk", "Ukrainian")));
    assert!(langs.contains(&("de", "German")));
}

#[test]
fn hot_loop_translate_by_id() {
    let (blob,) = make_reader(SAMPLE_TOML);
    let r = LxdbReader::new(&blob).unwrap();

    let en = r.lang_id("en").expect("en not found");
    let uk = r.lang_id("uk").expect("uk not found");

    // Pre-resolved IDs bypass the code→id lookup
    let words = ["hello", "world", "cat", "house"];
    let expected = [Some("привіт"), Some("світ"), Some("кіт"), Some("будинок")];

    for (word, exp) in words.iter().zip(expected.iter()) {
        assert_eq!(r.translate_by_id(word, en, uk), *exp, "failed on '{word}'");
    }
}

#[test]
fn empty_concepts_are_valid() {
    let toml = r#"
[languages]
en = "English"
uk = "Ukrainian"
"#;
    let blob = compile_toml(toml).expect("compile failed");
    let r    = LxdbReader::new(&blob).unwrap();
    assert_eq!(r.concept_count(), 0);
    assert_eq!(r.word_count(), 0);
}
