pub mod engine;

use std::sync::mpsc::TrySendError;
use std::sync::Arc;
use std::collections::{ HashMap };
use tokio::sync::{ mpsc, oneshot };
use tracing::{debug, error, info, trace };
use whisper_rs::{WhisperContext, WhisperContextParameters};

use crate::types::{PhraseChunk, TranscriptEvent};
use crate::utils::{ append_context, dump_audio_to_file} ;
use self::engine::{run_whisper, WhisperConfig};
use crate::{Pass1Job, Pass2Job, Pass1Result};

use crate::config::{
    TARGET_SAMPLE_RATE, STREAM_CHUNK_SAMPLES, PASS1_MIN_SAMPLES
};
use crate::config;

pub async fn pass1_task(
    ready_tx: oneshot::Sender<()>,
    mut rx:   mpsc::Receiver<PhraseChunk>,
    event_tx: mpsc::Sender<TranscriptEvent>,
) {
    let mut stream = StreamInfo::new();
    let (job_tx, mut job_rx) = tokio::sync::mpsc::channel::<Pass1Job>(3);
    let (res_tx, mut res_rx) = mpsc::channel::<Pass1Result>(8);
 
    std::thread::spawn(move || {
        let mut s: Option<whisper_rs::WhisperState> = None;
        let mut _ctx: Option<WhisperContext> = None;
        
        ready_tx.send(()).ok();
 
        while let Some(job) = job_rx.blocking_recv() {
            if s.is_none() {
                println!("DEBUG: Pass1: First sound received. Lazy loading model...");
                let current_cfg = config::startup();
                
                let mut ctx_params = WhisperContextParameters::default();
                ctx_params.use_gpu(current_cfg.use_gpu_fast);
                ctx_params.gpu_device(current_cfg.gpu_device_fast);

                let whisper_path = match crate::utils::find_first_file_in_dir("models/whisper-fast", "bin") {
                    Some(path) => path,
                    None => {
                        println!("CRITICAL: Pass1: No model found in models/whisper-fast!");
                        panic!("Pass1 missing model");
                    }
                };

                println!("DEBUG: Pass1: Creating WhisperContext from {:?}", whisper_path);
                
                let c = match WhisperContext::new_with_params(&whisper_path, ctx_params) {
                    Ok(ctx) => ctx,
                    Err(e) => {
                        println!("CRITICAL: Pass1: Context failed to load! Error: {:?}", e);
                        panic!("Pass1 context error");
                    }
                };
                
                println!("DEBUG: Pass1: Context loaded. Creating state...");
                
                let state = match c.create_state() {
                    Ok(st) => st,
                    Err(e) => {
                        println!("CRITICAL: Pass1: State init failed! Error: {:?}", e);
                        panic!("Pass1 state error");
                    }
                };

                s = Some(state);
                _ctx = Some(c); 
                println!("DEBUG: Pass1: Fully initialized and ready!");
            }
            let mut state = s.as_mut().unwrap();
            let duration_s = job.audio.len() as f32 / TARGET_SAMPLE_RATE as f32;
            let t0 = std::time::Instant::now();

            let debug_filename = format!("p1_debug_{}_{}.wav", job.phrase_id, job.chunk_id);
            dump_audio_to_file(&job.audio, &debug_filename);
            
            let (text, _) = run_whisper(&mut state, &job.audio, &WhisperConfig::fast());
            let elapsed = t0.elapsed();
            let rtf = t0.elapsed().as_secs_f32() / duration_s.max(0.001);
            tracing::info!(
                target: "PERFORMANCE",
                pid = job.phrase_id,
                cid = job.chunk_id,
                pass = 1,
                ms = elapsed.as_millis(),
                rtf = format!("{:.3}", rtf),
                "Inference finished"
            );

            if !text.is_empty() {
                if res_tx.blocking_send(Pass1Result {
                        phrase_id: job.phrase_id,
                        chunk_id:  job.chunk_id,
                        text,
                        short:      job.short,
                        duration_s,
                        rtf,
                    }).is_err() {
                    break;
                }
            }
        }
    });
     
    loop {
        tokio::select! {
            biased;

            Some(res) = res_rx.recv() => {
                let now = std::time::Instant::now();
                let valid = stream.is_result_valid(res.phrase_id, res.short);
                debug!(pid = res.phrase_id, cid = res.chunk_id, valid, "Pass1 got result from Whisper");
                if !valid { continue; }

                debug!(pid = res.phrase_id, "Pass1 sending to Transcript Event...");

                let event = if res.short {
                    TranscriptEvent::Final {
                        phrase_id:  res.phrase_id,
                        text:       res.text,
                        duration_s: res.duration_s,
                        rtf:        res.rtf,
                        sent_at: now,
                    }
                } else {
                    TranscriptEvent::Partial {
                        phrase_id: res.phrase_id,
                        chunk_id:  res.chunk_id,
                        text:      res.text,
                        sent_at: now,
                    }
                };
                let _ = event_tx.send(event).await;
                debug!(pid = res.phrase_id, "Pass1 send complete to Transcript Event complete.");
            }

            maybe_chunk = rx.recv() => {
                let Some(chunk) = maybe_chunk else { break };
                debug!(pid = chunk.phrase_id, cid = chunk.chunk_id,
                    len = chunk.data.len(), is_last = chunk.is_last, "pass1 got chunk");
                if let Some(job) = stream.process_incoming(chunk, &mut rx) {
                    if let Err(e) = job_tx.try_send(job) {
                        match e {
                            mpsc::error::TrySendError::Full(_) => {
                                debug!("Pass1 Worker busy, dropping newest chunk");
                            }
                            mpsc::error::TrySendError::Closed(_) => {
                                error!("Worker thread died! Shutting down async task.");
                                break;
                            }
                        }
                    }
                }
            }
        }
    }
}

pub async fn pass2_task(
    ready_tx: oneshot::Sender<()>,
    mut rx:   mpsc::Receiver<PhraseChunk>,
    event_tx: mpsc::Sender<TranscriptEvent>,
) {
    let (job_tx, job_rx) = std::sync::mpsc::sync_channel::<Pass2Job>(12);
    let (res_tx, mut res_rx) = mpsc::channel::<TranscriptEvent>(8);
 
    std::thread::spawn(move || {
        let mut s: Option<whisper_rs::WhisperState> = None;
        let mut _ctx: Option<WhisperContext> = None;
        ready_tx.send(()).ok();

        for job in job_rx {
            if s.is_none() {
                println!("DEBUG: Pass2: First sound received. Lazy loading model...");
                let current_cfg = config::startup();

                let mut ctx_params = WhisperContextParameters::default();
                ctx_params.use_gpu(current_cfg.use_gpu_acc);
                ctx_params.gpu_device(current_cfg.gpu_device_acc);

                let whisper_path = match crate::utils::find_first_file_in_dir("models/whisper-accurate", "bin") {
                    Some(path) => path,
                    None => {
                        println!("CRITICAL: Pass2: No model found in models/whisper-accurate!");
                        panic!("Pass2 missing model");
                    }
                };
                
                println!("DEBUG: Pass2: Creating WhisperContext from {:?}", whisper_path);
                let c = WhisperContext::new_with_params(&whisper_path, ctx_params)
                    .unwrap_or_else(|e| {
                        println!("CRITICAL: Pass2: Context failed to load! Error: {:?}", e);
                        panic!("Pass2 context error");
                    });
                
                println!("DEBUG: Pass2: Context loaded. Creating state...");
                s = Some(c.create_state().unwrap_or_else(|e| {
                    println!("CRITICAL: Pass2: State init failed! Error: {:?}", e);
                    panic!("Pass2 state error");
                }));
                _ctx = Some(c);
                println!("DEBUG: Pass2: Fully initialized and ready!");
            }
            let mut state = s.as_mut().unwrap();
            let duration_s = job.audio.len() as f32 / TARGET_SAMPLE_RATE as f32;
            let t0 = std::time::Instant::now();

            let debug_filename = format!("pass2_debug_phrase_{}.wav", job.phrase_id);
            dump_audio_to_file(&job.audio, &debug_filename);
 
            let (text, _) = run_whisper(&mut state, &job.audio, &WhisperConfig::accurate(&*job.context));
 
            let dur_ms = t0.elapsed().as_millis();
            let rtf    = t0.elapsed().as_secs_f32() / duration_s.max(0.001);
            tracing::info!(
                target: "PERFORMANCE",
                pid = job.phrase_id,
                pass = 2,
                ms = dur_ms,
                rtf = format!("{:.3}", rtf),
                text_len = text.len(),
                "Inference finished"
            );
 
            let event = TranscriptEvent::Final {
                phrase_id:  job.phrase_id,
                text:       text.trim().to_string(),
                duration_s,
                rtf,
                sent_at: std::time::Instant::now(),
            };
            if res_tx.blocking_send(event).is_err() { break; }
        }
    });
    let mut phrases: HashMap<u32, Vec<f32>> = HashMap::new();
    let mut last_context: Arc<String> = Arc::new(String::new());
 
    loop {
        tokio::select! { 
            biased;
            Some(event) = res_rx.recv() => {
                let pid = match &event {
                    TranscriptEvent::Final { phrase_id, .. } => *phrase_id,
                    TranscriptEvent::Partial { phrase_id, .. } => *phrase_id,
                };

                debug!(pid, "Pass2: received result from worker thread");

                if let TranscriptEvent::Final { ref text, .. } = event {
                    trace!(pid, text_len = text.len(), "Pass2: updating context with new final text");
                    let mut new_ctx = (*last_context).clone();
                    append_context(&mut new_ctx, text, 100);
                    last_context = Arc::new(new_ctx);
                }

                if let Err(e) = event_tx.send(event).await {
                    error!(pid, err = %e, "Pass2: failed to send event to downstream");
                } else {
                    debug!(pid, "Pass2: event successfully dispatched to event_tx");
                }
            }

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
 
                if chunk.is_last {
                    let audio = phrases.remove(&chunk.phrase_id).unwrap_or_default();
                    let audio_len = audio.len();
                    if audio_len < config::min_phrase_samples() {
                        debug!(pid = chunk.phrase_id, "pass2: phrase too short after accumulation, skipping");
                        continue;
                    }
                    
                    let job = Pass2Job { phrase_id: chunk.phrase_id, audio, context: Arc::clone(&last_context), };
                    if let Err(e) = job_tx.try_send(job) {
                        match e {
                            TrySendError::Full(_) => debug!("Pass2 Worker busy, skipping chunk"),
                            TrySendError::Disconnected(_) => {
                                error!("Worker thread died! Shutting down async task.");
                                break;
                            }
                        }
                    }
                } else { continue; }
            }
        }
    }
}

pub fn disable_whisper_log() {
    return;
    unsafe { whisper_rs::set_log_callback(Some(whisper_log_callback), std::ptr::null_mut()); }
}

unsafe extern "C" fn whisper_log_callback(
    // For some reason compilation for windows breaks if level value is unsigned, and it asks for integer.
    #[cfg(target_os = "windows")]
    _level: i32,
    #[cfg(target_os = "linux")]
    _level: u32,
    _msg: *const std::os::raw::c_char,
    _user_data: *mut std::ffi::c_void,
) {}

pub struct StreamInfo {
    current_id: Option<u32>,
    closed_id: Option<u32>,
    buffer: Vec<f32>,
}

impl StreamInfo {
    pub fn new() -> Self {
        Self {
            current_id: None,
            closed_id: None,
            buffer: Vec::with_capacity(STREAM_CHUNK_SAMPLES * 16)
        }
    }

    pub fn process_incoming(&mut self, mut chunk: PhraseChunk, rx: &mut mpsc::Receiver<PhraseChunk>) -> Option<Pass1Job> {
        while let Ok(next) = rx.try_recv() {
            if next.phrase_id > chunk.phrase_id && Some(chunk.phrase_id) == self.current_id {
                debug!("Drained");
                self.reset(next.phrase_id);
            }
            chunk = next;
        }

        if let Some(cur) = self.current_id {
            if chunk.phrase_id < cur {
                return None;
            }
        }
        if Some(chunk.phrase_id) != self.current_id {
            self.reset(chunk.phrase_id);
        }

        if !chunk.data.is_empty() {
            self.buffer.extend_from_slice(&chunk.data);
        }

        if chunk.is_last {
            self.closed_id = Some(chunk.phrase_id);
            self.current_id = None;
        }
        self.make_job(&chunk)
    }

    fn is_result_valid(&self, phrase_id: u32, short: bool) -> bool {
        if short { return true; }
        if Some(phrase_id) == self.closed_id { return false; }
        if let Some(cur) = self.current_id {
            if phrase_id < cur { return false; }
        }
        true
    }

    fn reset(&mut self, new_id: u32) {
        self.buffer.clear();
        self.current_id = Some(new_id);
        self.closed_id = None;
    }
    
    fn make_job(&mut self, chunk: &PhraseChunk) -> Option<Pass1Job> {
        let len = self.buffer.len();
        if len < PASS1_MIN_SAMPLES && !chunk.is_last {
            info!("{} chunk is less than {}, {}", chunk.chunk_id, PASS1_MIN_SAMPLES, chunk.data.len());
            return None;
        }

        let window = if chunk.is_last { config::max_window() } else { config::min_window() };

        let audio  = if len > window {
            self.buffer[len - window..].to_vec()
        } else {
            self.buffer.to_vec()
        };

        let phrase_id = chunk.phrase_id;
        let chunk_id  = chunk.chunk_id;

        if chunk.is_last { self.buffer.clear(); }
        trace!("[MAKE JOB]{} phrase, {} chunk is sent", phrase_id, chunk_id);
        Some(Pass1Job { phrase_id, chunk_id, short: chunk.short, audio })
    }
}