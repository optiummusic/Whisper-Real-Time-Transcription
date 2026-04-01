// ─── lxdb/src/lib.rs ──────────────────────────────────────────────────────────
//
// Публичный API крейта lxdb.
// Этот файл — единственная точка входа для внешних пользователей крейта.
// Все детали реализации скрыты в подмодулях.
//
// Секции .lxdb файла (в порядке следования в файле):
//
//   ┌─────────────┐
//   │   HEADER    │  32 bytes — магик, версия, счётчики, оффсеты секций
//   ├─────────────┤
//   │  LANG TABLE │  10 bytes × lang_count
//   ├─────────────┤
//   │CONCEPT TABLE│  4 × lang_count bytes × concept_count
//   ├─────────────┤
//   │ WORD INDEX  │  14 bytes × word_count  (отсортирован по FNV-1a хэшу)
//   ├─────────────┤
//   │ STRING POOL │  все строки, null-terminated
//   └─────────────┘
//
// Перевод: word → O(log W) бинарный поиск → concept_id → O(1) чтение → слово
// ─────────────────────────────────────────────────────────────────────────────

// ── Подмодули (приватная реализация) ─────────────────────────────────────────

pub mod builder;       // TOML → бинарный блоб
pub mod format;        // константы физического макета + FNV-1a + read/write
pub mod reader;        // zero-copy чтение и translate()
pub mod toml_schema;   // serde-структуры для .lxdb.toml источника

// ── Публичный реэкспорт ───────────────────────────────────────────────────────

pub use builder::compile;       // compile(&LxdbToml) -> Result<Vec<u8>>
pub use reader::LxdbReader;     // основной объект для lookup/translate
pub use toml_schema::LxdbToml;  // структура десериализованного TOML

// ── Тип ошибки ────────────────────────────────────────────────────────────────

#[derive(Debug)]
pub enum LxdbError {
    /// Файл не начинается с магических байт "LXDB".
    InvalidMagic,
    /// Версия в заголовке новее, чем поддерживает эта версия крейта.
    UnsupportedVersion(u8),
    /// Ошибка при компиляции из TOML (напр., неизвестный язык).
    Build(String),
    /// I/O ошибка при чтении/записи файла.
    Io(std::io::Error),
    /// TOML не удалось распарсить.
    Toml(String),
}

impl std::fmt::Display for LxdbError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidMagic          => write!(f, "not a valid LXDB file (bad magic bytes)"),
            Self::UnsupportedVersion(v) => write!(f, "unsupported LXDB version {v}"),
            Self::Build(msg)            => write!(f, "build error: {msg}"),
            Self::Io(e)                 => write!(f, "I/O error: {e}"),
            Self::Toml(msg)             => write!(f, "TOML parse error: {msg}"),
        }
    }
}

impl std::error::Error for LxdbError {}

// Автоматическое преобразование std::io::Error → LxdbError::Io
// чтобы можно было писать `?` в функциях возвращающих Result<_, LxdbError>
impl From<std::io::Error> for LxdbError {
    fn from(e: std::io::Error) -> Self { Self::Io(e) }
}

// ── Convenience API ───────────────────────────────────────────────────────────
// Обёртки для самых частых сценариев использования.

/// Парсит TOML-строку и компилирует её в .lxdb байты за один вызов.
///
/// ```no_run
/// let bytes = lxdb::compile_toml(r#"
///     [languages]
///     en = "English"
///     uk = "Ukrainian"
///     [[concepts]]
///     en = "hello"
///     uk = "привіт"
/// "#).unwrap();
/// ```
pub fn compile_toml(toml_str: &str) -> Result<Vec<u8>, LxdbError> {
    let source: LxdbToml = toml::from_str(toml_str)
        .map_err(|e| LxdbError::Toml(e.to_string()))?;
    compile(&source)
}

/// Читает TOML-файл с диска и компилирует его в .lxdb байты.
pub fn compile_file(path: &std::path::Path) -> Result<Vec<u8>, LxdbError> {
    let text = std::fs::read_to_string(path)?;
    compile_toml(&text)
}

/// Загружает .lxdb файл в память.
/// Результат передаётся в `LxdbReader::new()`.
pub fn load_file(path: &std::path::Path) -> Result<Vec<u8>, LxdbError> {
    Ok(std::fs::read(path)?)
}
