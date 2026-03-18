pub mod engine;

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::collections::HashMap;
use std::ffi::c_void;
use tokio::sync::mpsc;
use tracing::{info, debug, trace, warn};
use whisper_rs::{WhisperContext, WhisperContextParameters};
use parking_lot::{Mutex, Condvar};

use crate::types::{PhraseChunk, TranscriptEvent};
use crate::utils::append_context;
use crate::{TARGET_SAMPLE_RATE, STREAM_CHUNK_SAMPLES, PASS1_MIN_SAMPLES, MIN_PHRASE_SAMPLES, MAX_WINDOW, MIN_WINDOW};
use self::engine::{run_whisper, WhisperConfig};

struct Pass1Job {
    phrase_id: u32,
    chunk_id:  u32,
    short: bool,
    audio:     Vec<f32>,
}

struct Pass2Job {
    phrase_id: u32,
    audio:     Vec<f32>,
    context:   Arc<String>,
}

struct JobSlot {
    job:  Mutex<Option<Pass1Job>>,
    cv:   Condvar,
    done: AtomicBool,
}

impl JobSlot {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            job:  Mutex::new(None),
            cv:   Condvar::new(),
            done: AtomicBool::new(false),
        })
    }

    fn put(&self, job: Pass1Job) {
        let old_job = {
            let mut guard = self.job.lock();
            std::mem::replace(&mut *guard, Some(job))
        };
        self.cv.notify_one();
        drop(old_job); 
    }

    fn take_blocking(&self) -> Option<Pass1Job> {
        let mut guard = self.job.lock();
        loop {
            if let Some(j) = guard.take() { return Some(j); }
            if self.done.load(Ordering::Relaxed) { return None; }
            self.cv.wait(&mut guard);
        }
    }

    fn shutdown(&self) {
        self.done.store(true, Ordering::Relaxed);
        self.cv.notify_all();
    }
}

pub async fn pass1_task(
    mut rx:   mpsc::Receiver<PhraseChunk>,
    event_tx: mpsc::Sender<TranscriptEvent>,
) {
    let slot    = JobSlot::new();
    let slot_tx = slot.clone();
 
    let (res_tx, mut res_rx) = mpsc::channel::<(u32, u32, String, bool, f32, f32)>(8);
 
    std::thread::spawn(move || {
        let mut ctx_params = WhisperContextParameters::default();
        ctx_params.use_gpu(true);

        let whisper_path = crate::utils::find_first_file_in_dir("models/whisper-fast", "bin")
        .expect("No Whisper model found in models/whisper-fast");

        let ctx = WhisperContext::new_with_params(
            &whisper_path,
            ctx_params,
        ).expect("Pass1: error {whisper_path}");
        let mut state = ctx.create_state().expect("Pass1: state init failed");
 
        while let Some(job) = slot_tx.take_blocking() {
            let duration_s = job.audio.len() as f32 / TARGET_SAMPLE_RATE as f32;
            let t0 = std::time::Instant::now();
            
            let (text, _) = run_whisper(&mut state, &job.audio, &WhisperConfig::fast());
            
            let rtf = t0.elapsed().as_secs_f32() / duration_s.max(0.001);

            if !text.is_empty() {
                if res_tx.blocking_send((job.phrase_id, job.chunk_id, text, job.short, duration_s, rtf)).is_err() {
                    break;
                }
            }
        }
    });
 
    let mut current_id:  u32      = u32::MAX;
    let mut current_buf: Vec<f32> = Vec::with_capacity(STREAM_CHUNK_SAMPLES * 16);
    let mut closed_id: u32 = u32::MAX;
 
    loop {
        tokio::select! {
            biased;
 
            maybe_chunk = rx.recv() => {
                let Some(mut chunk) = maybe_chunk else { break };
 
                debug!(pid = chunk.phrase_id, cid = chunk.chunk_id,
                       len = chunk.data.len(), is_last = chunk.is_last, "pass1 got chunk");
 
                let mut drained = 0usize;
                while let Ok(next) = rx.try_recv() {
                    drained += 1;
                    if next.phrase_id > chunk.phrase_id {
                        if chunk.phrase_id != current_id {
                        } else {
                            current_buf.clear();
                            trace!(pid = chunk.phrase_id, cid = chunk.chunk_id, "pass 1 cleared buffer");
                            closed_id = chunk.phrase_id;
                        }
                    }
                    chunk = next;
                }
                if drained > 0 {
                    debug!(pid = chunk.phrase_id, count = drained, "pass1 drained multiple chunks from queue");
                }
 
                if chunk.phrase_id < current_id && current_id != u32::MAX {
                    trace!(pid = chunk.phrase_id, cid = chunk.chunk_id, curID = current_id, "Declined past ID");
                    continue;
                }
 
                if chunk.phrase_id > current_id {
                    current_buf.clear();
                    current_id = chunk.phrase_id;
                } else if current_id == u32::MAX {
                    current_id = chunk.phrase_id;
                }
 
                if !chunk.data.is_empty() {
                    current_buf.extend_from_slice(&chunk.data);
                }
 
                if chunk.is_last {
                    closed_id = chunk.phrase_id;
                    current_id = u32::MAX;
                }
 
                if current_buf.len() < PASS1_MIN_SAMPLES && !chunk.is_last {
                    continue;
                }
 
                let window = if chunk.is_last {
                    MAX_WINDOW
                } else {
                    MIN_WINDOW
                };

                let audio  = if current_buf.len() > window {
                    current_buf[current_buf.len() - window..].to_vec()
                } else {
                    current_buf.to_vec()
                };
 
                let phrase_id = chunk.phrase_id;
                let chunk_id  = chunk.chunk_id;
 
                if chunk.is_last { current_buf.clear(); }
 
                slot.put(Pass1Job { phrase_id, chunk_id, short: chunk.short, audio });
            }
 
            Some((phrase_id, chunk_id, text, short, duration_s, rtf)) = res_rx.recv() => {
                if !short && (phrase_id == closed_id || (current_id != u32::MAX && phrase_id < current_id)) {
                    trace!(phrase_id, "pass1: discarding result for closed phrase");
                    continue;
                }
                
                if short {
                    info!(phrase_id, duration_s, rtf, "pass1 fast track -> Final");
                    let _ = event_tx.send(TranscriptEvent::Final {
                        phrase_id, text, duration_s, rtf,
                    }).await;
                } else {
                    let _ = event_tx.send(TranscriptEvent::Partial {
                        phrase_id, chunk_id, text,
                    }).await;
                }
            }
        }
    }
 
    slot.shutdown();
}

pub async fn pass2_task(
    mut rx:   mpsc::Receiver<PhraseChunk>,
    event_tx: mpsc::Sender<TranscriptEvent>,
) {
    let (job_tx, job_rx) = std::sync::mpsc::sync_channel::<Pass2Job>(16);
    let (res_tx, mut res_rx) = mpsc::channel::<TranscriptEvent>(8);
 
    std::thread::spawn(move || {
        let mut ctx_params = WhisperContextParameters::default();
        ctx_params.use_gpu(true);
        let whisper_path = crate::utils::find_first_file_in_dir("models/whisper-accurate", "bin")
        .expect("No Whisper model found in models/whisper-accurate");

        let ctx = WhisperContext::new_with_params(
            &whisper_path,
            ctx_params,
        ).expect("Pass2: error loading {whisper_path}");
        let mut state = ctx.create_state().expect("Pass2: state init failed");
 
        for job in job_rx {
            let duration_s = job.audio.len() as f32 / TARGET_SAMPLE_RATE as f32;
            let t0 = std::time::Instant::now();
 
            let (text, _) = run_whisper(&mut state, &job.audio, &WhisperConfig::accurate(&*job.context));
 
            let dur_ms = t0.elapsed().as_millis();
            let rtf    = t0.elapsed().as_secs_f32() / duration_s.max(0.001);
            info!(pid = job.phrase_id, dur_ms, duration_s, rtf,
                  out_len = text.len(), "pass2 final inference");
 
            let event = TranscriptEvent::Final {
                phrase_id:  job.phrase_id,
                text:       text.trim().to_string(),
                duration_s,
                rtf,
            };
            if res_tx.blocking_send(event).is_err() { break; }
        }
    });
 
    let mut phrases: HashMap<u32, Vec<f32>> = HashMap::new();
    let mut last_context: Arc<String> = Arc::new(String::new());
 
    loop {
        tokio::select! {
            maybe_chunk = rx.recv() => {
                let Some(chunk) = maybe_chunk else { break };
 
                debug!(pid = chunk.phrase_id, cid = chunk.chunk_id,
                       len = chunk.data.len(), is_last = chunk.is_last, "pass2 got chunk");
 
                let acc = phrases.entry(chunk.phrase_id)
                    .or_insert_with(|| Vec::with_capacity(STREAM_CHUNK_SAMPLES * 8));
 
                if !chunk.data.is_empty() {
                    acc.extend_from_slice(&chunk.data);
                }
                debug!(pid = chunk.phrase_id, acc_len = acc.len(), "accumulated length");
 
                if !chunk.is_last { continue; }
 
                let audio = phrases.remove(&chunk.phrase_id).unwrap_or_default();
                if audio.len() < MIN_PHRASE_SAMPLES {
                    debug!(pid = chunk.phrase_id, "pass2: phrase too short after accumulation, skipping");
                    continue;
                }
 
                let job = Pass2Job { phrase_id: chunk.phrase_id, audio, context: Arc::clone(&last_context), };
                if let Err(e) = job_tx.try_send(job) {
                    warn!(%e, pid = chunk.phrase_id, "pass2 job queue full — dropping phrase");
                }
            }
 
            Some(event) = res_rx.recv() => {
                if let TranscriptEvent::Final { ref text, .. } = event {
                    let mut new_ctx = (*last_context).clone();
                    append_context(&mut new_ctx, text, 100);
                    last_context = Arc::new(new_ctx);
                }
                let _ = event_tx.send(event).await;
            }
        }
    }
}

pub fn disable_whisper_log() {
    unsafe { whisper_rs::set_log_callback(Some(whisper_log_callback), std::ptr::null_mut()); }
}

unsafe extern "C" fn whisper_log_callback(
    _level: u32,
    _msg: *const std::os::raw::c_char,
    _user_data: *mut c_void,
) {}
