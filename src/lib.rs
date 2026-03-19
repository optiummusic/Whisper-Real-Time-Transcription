pub mod types;
pub mod audio;
pub mod display_task;
pub mod vad;
pub mod utils;
pub mod whisper;
pub mod config;

pub use crate::types::{PhraseChunk, TranscriptEvent, AudioPacket};
use std::sync::Arc;
use parking_lot::{Mutex, Condvar};
use std::sync::atomic::{AtomicBool, Ordering};
use tokio::sync::mpsc;
use tracing::debug;

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
