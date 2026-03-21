pub mod engine;

use std::path::Path;
use std::fs;
use std::sync::Arc;
use std::collections::{ HashMap, VecDeque };
use std::ffi::c_void;
use std::sync::{Mutex, LazyLock};
use tokio::sync::{ mpsc, oneshot };
use tracing::{info, debug, warn};
use whisper_rs::{WhisperContext, WhisperContextParameters};

use crate::types::{PhraseChunk, TranscriptEvent};
use crate::utils::append_context;
use self::engine::{run_whisper, WhisperConfig};
use crate::{JobSlot, Pass1Job, Pass2Job, Pass1Result};

use crate::config::{
    TARGET_SAMPLE_RATE, VAD_CHUNK_SIZE, STREAM_CHUNK_SAMPLES, STITCH_MIN_SAMPLES, PASS1_MIN_SAMPLES, VAD_CHUNK_DURATION_S 
};
use crate::config;

pub async fn pass1_task(
    ready_tx: oneshot::Sender<()>,
    mut rx:   mpsc::Receiver<PhraseChunk>,
    event_tx: mpsc::Sender<TranscriptEvent>,
) {
    let mut stream = StreamInfo::new();

    let (job_tx, job_rx) = std::sync::mpsc::sync_channel::<Pass1Job>(1);
    let (res_tx, mut res_rx) = mpsc::channel::<Pass1Result>(8);
 
    std::thread::spawn(move || {
        let mut ctx_params = WhisperContextParameters::default();
        ctx_params.use_gpu(config::startup().use_gpu_fast);

        let whisper_path = crate::utils::find_first_file_in_dir("models/whisper-fast", "bin")
        .expect("No Whisper model found in models/whisper-fast");

        let ctx = WhisperContext::new_with_params(
            &whisper_path,
            ctx_params,
        ).expect(&format!("Pass1: error {whisper_path:?}"));
        let mut state = ctx.create_state().expect("Pass1: state init failed");
        ready_tx.send(()).ok();
 
        for job in job_rx {
            let duration_s = job.audio.len() as f32 / TARGET_SAMPLE_RATE as f32;
            let t0 = std::time::Instant::now();

            let debug_filename = format!("p1_debug_{}_{}.wav", job.phrase_id, job.chunk_id);
            dump_audio_to_file(&job.audio, &debug_filename);
            
            let (text, _) = run_whisper(&mut state, &job.audio, &WhisperConfig::fast());
            
            let rtf = t0.elapsed().as_secs_f32() / duration_s.max(0.001);

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

            maybe_chunk = rx.recv() => {
                let Some(chunk) = maybe_chunk else { break };
                debug!(pid = chunk.phrase_id, cid = chunk.chunk_id,
                    len = chunk.data.len(), is_last = chunk.is_last, "pass1 got chunk");
                if let Some(job) = stream.process_incoming(chunk, &mut rx) {
                   if let Err(e) = job_tx.try_send(job) {
                        debug!("Pass1: skipping job (worker busy or channel full): {}", e);
                    }
                }
            }
            
            Some(res) = res_rx.recv() => {
                let valid = stream.is_result_valid(res.phrase_id, res.short);
                info!(pid = res.phrase_id, cid = res.chunk_id, valid, "pass1 result from whisper");
                if !valid { continue; }

                info!(pid = res.phrase_id, "pass1 sending to event_tx");

                let event = if res.short {
                    TranscriptEvent::Final {
                        phrase_id:  res.phrase_id,
                        text:       res.text,
                        duration_s: res.duration_s,
                        rtf:        res.rtf,
                        sent_at: std::time::Instant::now(),
                    }
                } else {
                    TranscriptEvent::Partial {
                        phrase_id: res.phrase_id,
                        chunk_id:  res.chunk_id,
                        text:      res.text,
                        sent_at: std::time::Instant::now(),
                    }
                };
                let _ = event_tx.send(event).await;
                info!(pid = res.phrase_id, "pass1 event_tx send complete");
            }
        }
    }
}

pub async fn pass2_task(
    ready_tx: oneshot::Sender<()>,
    mut rx:   mpsc::Receiver<PhraseChunk>,
    event_tx: mpsc::Sender<TranscriptEvent>,
) {
    let (job_tx, job_rx) = std::sync::mpsc::sync_channel::<Pass2Job>(4);
    let (res_tx, mut res_rx) = mpsc::channel::<TranscriptEvent>(8);
 
    std::thread::spawn(move || {
        let mut ctx_params = WhisperContextParameters::default();
        ctx_params.use_gpu(config::startup().use_gpu_acc);
        let whisper_path = crate::utils::find_first_file_in_dir("models/whisper-accurate", "bin")
        .expect("No Whisper model found in models/whisper-accurate");
        let ctx = WhisperContext::new_with_params(
            &whisper_path,
            ctx_params,
        ).expect(&format!("Pass2: error loading {whisper_path:?}"));
        let mut state = ctx.create_state().expect("Pass2: state init failed");
        ready_tx.send(()).ok();

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
                sent_at: std::time::Instant::now(),
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
                    if audio_len < config::min_phrase_samples() {
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

static DUMP_HISTORY: LazyLock<Mutex<VecDeque<String>>> = LazyLock::new(|| {
    Mutex::new(VecDeque::new())
});

const MAX_DUMP_FILES: usize = 20;
pub fn dump_audio_to_file(samples: &[f32], filename: &str) {
    if !config::dump_audio() { return; }
    
    let dir = "debug_audio";
    if let Err(e) = fs::create_dir_all(dir) {
        eprintln!("Failed to create debug directory: {}", e);
        return;
    }

    let file_path = Path::new(dir).join(filename);
    let file_path_str = file_path.to_string_lossy().into_owned();

    {
        let mut history = DUMP_HISTORY.lock().unwrap();
        
        history.retain(|x| x != &file_path_str);

        if history.len() >= MAX_DUMP_FILES {
            if let Some(old_file) = history.pop_front() {
                let _ = fs::remove_file(old_file);
            }
        }
        
        history.push_back(file_path_str);
    }

    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: 16000,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };

    if let Ok(mut writer) = hound::WavWriter::create(&file_path, spec) {
        for &sample in samples {
            let _ = writer.write_sample(sample);
        }
    }
}

pub fn disable_whisper_log() {
    // unsafe { whisper_rs::set_log_callback(Some(whisper_log_callback), std::ptr::null_mut()); }
}

// unsafe extern "C" fn whisper_log_callback(
//     _level: u32,
//     _msg: *const std::os::raw::c_char,
//     _user_data: *mut c_void,
// ) {}

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
        info!("{} phrase, {} chunk is sent", phrase_id, chunk_id);
        Some(Pass1Job { phrase_id, chunk_id, short: chunk.short, audio })
    }
}