pub mod types;
pub mod audio;
pub mod display_task;
pub mod vad;
pub mod utils;
pub mod whisper;
pub mod config;
pub mod translation;

pub use crate::types::{PhraseChunk, TranscriptEvent, AudioPacket};
use std::sync::Arc;

pub struct PhraseData {
    pub text: String,
    pub is_final: bool,
    pub duration_s: f32,
    pub rtf: f32,
}

pub struct Pass1Job {
    phrase_id: u32,
    chunk_id:  u32,
    short: bool,
    audio:     Vec<f32>,
}

pub struct Pass2Job {
    phrase_id: u32,
    audio:     Vec<f32>,
    context:   Arc<String>,
}

pub struct Pass1Result {
    pub phrase_id:  u32,
    pub chunk_id:   u32,
    pub text:       String,
    pub short:      bool,
    pub duration_s: f32,
    pub rtf:        f32,
}