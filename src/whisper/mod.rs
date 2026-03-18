pub mod engine;

use std::path::Path;
use std::fs;
use std::sync::Arc;
use std::collections::HashMap;
use std::ffi::c_void;
use tokio::sync::mpsc;
use tracing::{info, debug, warn};
use whisper_rs::{WhisperContext, WhisperContextParameters};

use crate::types::{PhraseChunk, TranscriptEvent};
use crate::utils::append_context;
use crate::{TARGET_SAMPLE_RATE, STREAM_CHUNK_SAMPLES, MIN_PHRASE_SAMPLES, PASS1_MIN_SAMPLES, MAX_WINDOW, MIN_WINDOW};
use self::engine::{run_whisper, WhisperConfig};
use crate::{JobSlot, Pass1Job, Pass2Job};

pub async fn pass1_task(
    mut rx:   mpsc::Receiver<PhraseChunk>,
    event_tx: mpsc::Sender<TranscriptEvent>,
) {
    let slot    = JobSlot::new();
    let slot_tx = slot.clone();
    let mut stream = StreamInfo::new();
 
    let (res_tx, mut res_rx) = mpsc::channel::<(u32, u32, String, bool, f32, f32)>(8);
 
    std::thread::spawn(move || {
        let mut ctx_params = WhisperContextParameters::default();
        ctx_params.use_gpu(crate::USE_GPU_FAST);

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

            let debug_filename = format!("p1_debug_{}_{}.wav", job.phrase_id, job.chunk_id);
            dump_audio_to_file(&job.audio, &debug_filename);
            
            let (text, _) = run_whisper(&mut state, &job.audio, &WhisperConfig::fast());
            
            let rtf = t0.elapsed().as_secs_f32() / duration_s.max(0.001);

            if !text.is_empty() {
                if res_tx.blocking_send((job.phrase_id, job.chunk_id, text, job.short, duration_s, rtf)).is_err() {
                    break;
                }
            }
        }
    });
 
    loop {
        tokio::select! {
            biased;

            maybe_chunk = rx.recv() => {
                let Some(chunk) = maybe_chunk else { break };
                debug!(pid = chunk.phrase_id, cid = chunk.chunk_id,
                    len = chunk.data.len(), is_last = chunk.is_last, "pass1 got chunk");
                if let Some(job) = stream.process_incoming(chunk, &mut rx) {
                    slot.put(job);

                }
            }
            
            Some((pid, cid, text, short, dur, rtf)) = res_rx.recv() => {
                if !stream.is_result_valid(pid, short) {
                    continue;
                }

                let event = if short {
                    TranscriptEvent::Final { phrase_id: pid, text, duration_s: dur, rtf }
                } else {
                    TranscriptEvent::Partial { phrase_id: pid, chunk_id: cid, text }
                };
                let _ = event_tx.send(event).await;
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
        ctx_params.use_gpu(crate::USE_GPU_ACC);
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

            let debug_filename = format!("pass2_debug_phrase_{}.wav", job.phrase_id);
            dump_audio_to_file(&job.audio, &debug_filename);
 
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
 
                if chunk.is_last {
                    let audio = phrases.remove(&chunk.phrase_id).unwrap_or_default();
                    let audio_len = audio.len();
                    if audio_len < MIN_PHRASE_SAMPLES {
                        debug!(pid = chunk.phrase_id, "pass2: phrase too short after accumulation, skipping");
                        continue;
                    }
                    
                    let job = Pass2Job { phrase_id: chunk.phrase_id, audio, context: Arc::clone(&last_context), };
                    match job_tx.try_send(job) {
                        Ok(_) => debug!(pid = chunk.phrase_id, "SUCCESS: Job sent to worker thread"),
                        Err(e) => warn!(pid = chunk.phrase_id, err = %e, "ERROR: Could not send job to Pass 2 worker!"),
                    }
                } else { continue; }
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

pub fn dump_audio_to_file(samples: &[f32], filename: &str) {
    if crate::DUMP_AUDIO == false { return; }
    let dir = "debug_audio";
    
    if let Err(e) = fs::create_dir_all(dir) {
        eprintln!("Failed to create debug directory: {}", e);
        return;
    }

    let path = Path::new(dir).join(filename);

    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: 16000,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };

    let mut writer = hound::WavWriter::create(path, spec).unwrap();
    for &sample in samples {
        writer.write_sample(sample).unwrap();
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

struct StreamInfo {
    current_id: Option<u32>,
    closed_id: Option<u32>,
    buffer: Vec<f32>,
}

impl StreamInfo {
    fn new() -> Self {
        Self {
            current_id: None,
            closed_id: None,
            buffer: Vec::with_capacity(STREAM_CHUNK_SAMPLES * 16)
        }
    }

    fn process_incoming(&mut self, mut chunk: PhraseChunk, rx: &mut mpsc::Receiver<PhraseChunk>) -> Option<Pass1Job> {
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
    }
    
    fn make_job(&mut self, chunk: &PhraseChunk) -> Option<Pass1Job> {
        let len = self.buffer.len();
        if len < PASS1_MIN_SAMPLES && !chunk.is_last {
            return None;
        }

        let window = if chunk.is_last { MAX_WINDOW } else { MIN_WINDOW };

        let audio  = if len > window {
            self.buffer[len - window..].to_vec()
        } else {
            self.buffer.to_vec()
        };

        let phrase_id = chunk.phrase_id;
        let chunk_id  = chunk.chunk_id;

        if chunk.is_last { self.buffer.clear(); }

        Some(Pass1Job { phrase_id, chunk_id, short: chunk.short, audio })
    }
}