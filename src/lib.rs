pub mod types;
pub mod audio;
pub mod display_task;
pub mod vad;
pub mod utils;
pub mod whisper;

pub use crate::types::{PhraseChunk, TranscriptEvent, AudioPacket};
use std::sync::Arc;
use parking_lot::{Mutex, Condvar};
use std::sync::atomic::{AtomicBool, Ordering};
use tokio::sync::mpsc;
use tracing::debug;

pub const TARGET_SAMPLE_RATE: u32 = 16_000;
pub const VAD_CHUNK_SIZE:      usize = 480;
pub const MAX_SILENCE_CHUNKS:  usize = 12;
pub const STREAM_CHUNK_SAMPLES: usize = TARGET_SAMPLE_RATE as usize;
pub const MIN_PHRASE_SAMPLES:   usize = TARGET_SAMPLE_RATE as usize / 2;
pub const MAX_PHRASE_SAMPLES:   usize = TARGET_SAMPLE_RATE as usize * 12;
pub const PASS1_MIN_SAMPLES:    usize = TARGET_SAMPLE_RATE as usize * 1;
pub const FAST_TRACK_THRESHOLD_S: f32 = 3.0;
pub const MAX_WINDOW: usize = TARGET_SAMPLE_RATE as usize * 10;
pub const MIN_WINDOW: usize = TARGET_SAMPLE_RATE as usize * 4;
pub const PREROLL_CHUNKS: usize = 5;
pub const STITCH_MIN_SAMPLES: usize = (TARGET_SAMPLE_RATE as f32 * 1.5) as usize;
pub const STITCH_MAX_SILENCE: f32 = 1.2;  
pub const VAD_CHUNK_DURATION_S: f32 = VAD_CHUNK_SIZE as f32 / TARGET_SAMPLE_RATE as f32;
pub const STITCH_MAX_CHUNKS: usize = (STITCH_MAX_SILENCE / VAD_CHUNK_DURATION_S) as usize;
pub const LANGUAGE: &str = "en";
pub const USE_GPU_ACC: bool = true;
pub const USE_GPU_FAST: bool = true;
pub const SPEECH_PROBABILITY: f32 = 0.5; // Range: 0..1
pub const DUMP_AUDIO: bool = true;

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

pub struct JobSlot {
    job:  Mutex<Option<Pass1Job>>,
    cv:   Condvar,
    done: AtomicBool,
}

impl JobSlot {
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            job:  Mutex::new(None),
            cv:   Condvar::new(),
            done: AtomicBool::new(false),
        })
    }

    pub fn put(&self, job: Pass1Job) {
        let old_job = {
            let mut guard = self.job.lock();
            std::mem::replace(&mut *guard, Some(job))
        };
        self.cv.notify_one();
        drop(old_job); 
    }

    pub fn take_blocking(&self) -> Option<Pass1Job> {
        let mut guard = self.job.lock();
        loop {
            if let Some(j) = guard.take() { return Some(j); }
            if self.done.load(Ordering::Relaxed) { return None; }
            self.cv.wait(&mut guard);
        }
    }

    pub fn shutdown(&self) {
        self.done.store(true, Ordering::Relaxed);
        self.cv.notify_all();
    }
}
