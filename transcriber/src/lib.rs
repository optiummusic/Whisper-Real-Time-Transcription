pub mod audio;
pub mod config;
pub mod display_task;
pub mod translation;
pub mod types;
pub mod utility;
pub mod vad;
pub mod whisper;
pub mod ui;
pub mod prelude;
pub use crate::types::{AudioPacket, PhraseChunk, TranscriptEvent};
use std::sync::Arc;

pub struct PhraseData {
    pub text: String,
    pub is_final: bool,
    pub duration_s: f32,
    pub rtf: f32,
}

pub struct Pass1Job {
    phrase_id: u32,
    chunk_id: u32,
    short: bool,
    audio: Vec<f32>,
    is_last: bool,
}

pub struct Pass2Job {
    phrase_id: u32,
    audio: Vec<f32>,
    context: Arc<String>,
}

pub struct Pass1Result {
    pub phrase_id: u32,
    pub chunk_id: u32,
    pub text: String,
    pub short: bool,
    pub is_last: bool,
    pub duration_s: f32,
    pub rtf: f32,
}
