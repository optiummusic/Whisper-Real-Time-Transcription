// ─── lxdb/src/main.rs ─────────────────────────────────────────────────────────
//
// CLI-инструмент для работы с .lxdb файлами.
// Это БИНАРНЫЙ таргет крейта lxdb (не библиотека).
//
// Использование:
//   cargo run -p lxdb -- compile dictionary/main.lxdb.toml
//   cargo run -p lxdb -- compile dictionary/main.lxdb.toml out.lxdb
//   cargo run -p lxdb -- info    dictionary/main.lxdb
//   cargo run -p lxdb -- lookup  dictionary/main.lxdb uk привіт en
//   cargo run -p lxdb -- dump    dictionary/main.lxdb
// ─────────────────────────────────────────────────────────────────────────────

use std::path::PathBuf;

// Используем библиотечную часть ЭТОГО ЖЕ крейта.
// Cargo автоматически линкует lib и bin внутри одного [package].
use lxdb::{compile_file, load_file, builder::stats_from, LxdbReader, LxdbToml};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();

    match args.as_slice() {
        [cmd, rest @ ..] if cmd == "compile" => cmd_compile(rest),
        [cmd, rest @ ..] if cmd == "info"    => cmd_info(rest),
        [cmd, rest @ ..] if cmd == "lookup"  => cmd_lookup(rest),
        [cmd, rest @ ..] if cmd == "dump"    => cmd_dump(rest),
        _ => print_usage(),
    }
}

fn print_usage() {
    eprintln!("lxdb — Lexical Database toolchain\n");
    eprintln!("  compile <input.toml> [output.lxdb]      compile TOML source → binary");
    eprintln!("  info    <file.lxdb>                     print file metadata");
    eprintln!("  lookup  <file.lxdb> <src> <word> <tgt>  translate one word");
    eprintln!("  dump    <file.lxdb>                     print all concepts");
    std::process::exit(1);
}

// ── compile ───────────────────────────────────────────────────────────────────
// TOML → .lxdb бинарный файл.
// Выходной путь по умолчанию: заменяем .toml → .lxdb в имени входного файла.

fn cmd_compile(args: &[String]) {
    let (input, output) = match args {
        [i]    => {
            let out = i.strip_suffix(".toml")
                .map(|s| format!("{s}.lxdb"))
                .unwrap_or_else(|| format!("{i}.lxdb"));
            (PathBuf::from(i), PathBuf::from(out))
        }
        [i, o] => (PathBuf::from(i), PathBuf::from(o)),
        _ => {
            eprintln!("usage: lxdb compile <input.toml> [output.lxdb]");
            std::process::exit(1);
        }
    };

    // Компиляция
    let blob = compile_file(&input).unwrap_or_else(|e| {
        eprintln!("compile error: {e}");
        std::process::exit(1);
    });

    // Запись на диск
    std::fs::write(&output, &blob).unwrap_or_else(|e| {
        eprintln!("write error: {e}");
        std::process::exit(1);
    });

    // Статистика — читаем TOML ещё раз только чтобы получить source-level счётчики
    let toml_text = std::fs::read_to_string(&input).unwrap();
    let source: LxdbToml = toml::from_str(&toml_text).unwrap();
    let stats = stats_from(&blob, &source);

    println!("✓  {stats}");
    println!("   → {output:?}");
}

// ── info ──────────────────────────────────────────────────────────────────────
// Метаданные заголовка без полного чтения файла.

fn cmd_info(args: &[String]) {
    let path = match args {
        [p] => PathBuf::from(p),
        _   => { eprintln!("usage: lxdb info <file.lxdb>"); std::process::exit(1); }
    };

    let data = load(&path);
    let r    = open(&data);

    println!("File         : {path:?}");
    println!("Size         : {} bytes", r.file_size());
    println!("Languages    : {}", r.lang_count());
    for (code, name) in r.languages() {
        println!("               {code:>4}  {name}");
    }
    println!("Concepts     : {}", r.concept_count());
    println!("Index entries: {}", r.word_count());
}

// ── lookup ────────────────────────────────────────────────────────────────────
// Одиночный перевод слова (или фразы в кавычках).

fn cmd_lookup(args: &[String]) {
    let (path, src, word, tgt) = match args {
        [p, s, w, t] => (PathBuf::from(p), s.as_str(), w.as_str(), t.as_str()),
        _ => {
            eprintln!("usage: lxdb lookup <file.lxdb> <src_lang> <word> <tgt_lang>");
            std::process::exit(1);
        }
    };

    let data = load(&path);
    let r    = open(&data);

    match r.translate(word, src, tgt) {
        Some(t) => println!("{src}:{word}  →  {tgt}:{t}"),
        None    => {
            eprintln!("(not found: '{word}' in lang '{src}')");
            std::process::exit(2);
        }
    }
}

// ── dump ──────────────────────────────────────────────────────────────────────
// Печатает все концепты в виде таблицы — полезно для дебага словаря.

fn cmd_dump(args: &[String]) {
    let path = match args {
        [p] => PathBuf::from(p),
        _   => { eprintln!("usage: lxdb dump <file.lxdb>"); std::process::exit(1); }
    };

    let data  = load(&path);
    let r     = open(&data);
    let langs: Vec<(&str, &str)> = r.languages().collect();

    // Заголовок таблицы
    let header: Vec<String> = langs.iter().map(|(code, _)| format!("{code:>12}")).collect();
    println!("cid   {}", header.join("  "));
    println!("{}", "─".repeat(6 + langs.len() * 14));

    // Каждый концепт
    for cid in 0..r.concept_count() {
        let cells: Vec<String> = langs.iter().map(|(code, _)| {
            let word = r.translate_concept(cid, code).unwrap_or("—");
            format!("{word:>12}")
        }).collect();
        println!("{cid:<6}{}", cells.join("  "));
    }
}

// ── helpers ───────────────────────────────────────────────────────────────────

fn load(path: &PathBuf) -> Vec<u8> {
    load_file(path).unwrap_or_else(|e| {
        eprintln!("error reading {path:?}: {e}");
        std::process::exit(1);
    })
}

fn open(data: &[u8]) -> LxdbReader<'_> {
    LxdbReader::new(data).unwrap_or_else(|e| {
        eprintln!("invalid lxdb file: {e}");
        std::process::exit(1);
    })
}
