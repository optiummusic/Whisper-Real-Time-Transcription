// ─── lxdb/src/toml_schema.rs ──────────────────────────────────────────────────
// Serde структуры для десериализации исходного .lxdb.toml словаря.
// ─────────────────────────────────────────────────────────────────────────────

use serde::Deserialize;
use std::collections::BTreeMap;

/// Корневая структура .lxdb.toml файла.
#[derive(Debug, Deserialize, Clone)]
pub struct LxdbToml {
    /// Метаданные словаря (версия, описание).
    pub meta: Meta,
    
    /// Карта языков: BCP-47 код -> Человекочитаемое название.
    /// Например: "uk" -> "Ukrainian".
    pub languages: BTreeMap<String, String>,
    
    /// Список концептов. 
    /// Каждый концепт — это словарь маппинга: код языка -> переведенное слово или фраза.
    /// Используем #[serde(default)] на случай, если словарь пока пуст.
    #[serde(default)]
    pub concepts: Vec<BTreeMap<String, String>>,
}

/// Метаданные заголовка файла.
#[derive(Debug, Deserialize, Clone)]
pub struct Meta {
    /// Версия формата (ожидается 1, проверяется в reader).
    pub version: u8,
    /// Произвольное текстовое описание словаря.
    pub description: String,
}