use std::sync::Arc;
pub type AudioPacket = Vec<f32>;

#[derive(Debug)]
pub enum TranscriptEvent {
    Partial { phrase_id: u32, chunk_id: u32, text: String },
    Final   { phrase_id: u32, text: String, duration_s: f32, rtf: f32 },
}

#[derive(Clone)]
pub struct PhraseChunk {
    pub phrase_id: u32,
    pub chunk_id:  u32,
    pub is_last:   bool,
    pub short: bool,
    pub data:      Arc<Vec<f32>>,
}