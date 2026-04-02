use serde::Deserialize;
use std::collections::BTreeMap;

#[derive(Debug, Deserialize, Clone)]
pub struct LxdbToml {
    #[serde(default)]
    pub meta: Meta,

    pub languages: BTreeMap<String, String>,

    #[serde(default)]
    pub concepts: Vec<Concept>,
}

#[derive(Debug, Deserialize, Clone)]
pub struct Concept {
    #[serde(flatten)]
    pub forms: BTreeMap<String, String>,

    /// [concepts.generated]
    #[serde(default)]
    pub generated: BTreeMap<String, Vec<String>>,

    /// [concepts.custom]
    #[serde(default)]
    pub custom: BTreeMap<String, Vec<String>>,
}

#[derive(Debug, Deserialize, Clone, Default)]
pub struct Meta {
    #[serde(default = "default_version")]
    pub version: u8,
    #[serde(default)]
    pub description: String,
}

fn default_version() -> u8 { 1 }