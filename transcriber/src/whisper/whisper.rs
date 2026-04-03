use crate::prelude::*;
use std::sync::mpsc::TrySendError;
use whisper_rs::{WhisperContext, WhisperContextParameters};
use crate::utility::{stats, utils::
    {
        append_context, dump_audio_to_file, find_first_file_in_dir
    },
};
use crate::whisper::engine::{WhisperConfig, run_whisper};
use crate::{Pass1Job, Pass1Result, Pass2Job};
use crate::whisper::stream_info::StreamInfo;
struct WhisperWorker {
    _ctx: WhisperContext,
    state: whisper_rs::WhisperState,
}

impl WhisperWorker {
    fn load(model_dir: &str, use_gpu: bool, gpu_device: i32) -> Self {
        let mut ctx_params = WhisperContextParameters::default();
        ctx_params.use_gpu(use_gpu);
        if use_gpu { ctx_params.gpu_device(gpu_device); }

        let path = find_first_file_in_dir(model_dir, "bin")
            .unwrap_or_else(|| panic!("No model found in {model_dir}"));

        let _ctx = WhisperContext::new_with_params(&path, ctx_params)
            .unwrap_or_else(|e| panic!("Context failed: {e:?}"));

        let state = _ctx.create_state()
            .unwrap_or_else(|e| panic!("State failed: {e:?}"));

        tracing::info!("WhisperWorker loaded from {:?}", path);
        Self { _ctx, state }
    }
}
pub async fn pass1_task(
    ready_tx: oneshot::Sender<()>,
    mut rx: mpsc::Receiver<PhraseChunk>,
    event_tx: mpsc::Sender<TranscriptEvent>,
) {
    let mut stream = StreamInfo::new();
    let (job_tx, mut job_rx) = tokio::sync::mpsc::channel::<Pass1Job>(1);
    let (res_tx, mut res_rx) = mpsc::channel::<Pass1Result>(8);

    std::thread::spawn(move || {
        let mut worker: Option<WhisperWorker> = None;
        let cfg = config::startup();
        ready_tx.send(()).ok();

        while let Some(job) = job_rx.blocking_recv() {
            let w = worker.get_or_insert_with(|| {
            WhisperWorker::load("models/whisper-fast",
                    cfg.use_gpu_fast, cfg.gpu_device_fast)
            });
            let duration_s = job.audio.len() as f32 / TARGET_SAMPLE_RATE as f32;
            let t0 = std::time::Instant::now();

            let debug_filename = format!("p1_debug_{}_{}.wav", job.phrase_id, job.chunk_id);
            dump_audio_to_file(&job.audio, &debug_filename);

            let (text, _) = run_whisper(&mut w.state, &job.audio, &WhisperConfig::fast());

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
            if job.is_last {
                stats::get().record_pass1_done(job.phrase_id, rtf, duration_s, text.len());
            }
            if !text.is_empty()
                && res_tx
                    .blocking_send(Pass1Result {
                        phrase_id: job.phrase_id,
                        chunk_id: job.chunk_id,
                        text,
                        short: job.short,
                        is_last: job.is_last,
                        duration_s,
                        rtf,
                    })
                    .is_err()
                {
                    break;
                }
        }
    });

    loop {
        tokio::select! {
            biased;

            Some(res) = res_rx.recv() => {
                let now = std::time::Instant::now();
                let single_pass = config::SINGLE_PASS.load(Ordering::Relaxed);
                let valid = stream.is_result_valid(res.phrase_id, res.short, res.is_last, single_pass);
                debug!(pid = res.phrase_id, cid = res.chunk_id, valid, "Pass1 got result from Whisper");
                if !valid { continue; }

                debug!(pid = res.phrase_id, "Pass1 sending to Transcript Event...");

                let is_final_event = res.short || (single_pass && res.is_last);

                let event = if is_final_event {
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
                    stats::get().record_pass1_start(job.phrase_id);
                    if let Err(e) = job_tx.try_send(job) {
                        match e {
                            mpsc::error::TrySendError::Full(_) => {
                                debug!("Pass1 Worker busy, dropping newest chunk");
                                stats::get().pass1_dropped.fetch_add(1, Ordering::Relaxed);
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
    mut rx: mpsc::Receiver<PhraseChunk>,
    event_tx: mpsc::Sender<TranscriptEvent>,
) {
    let (job_tx, job_rx) = std::sync::mpsc::sync_channel::<Pass2Job>(12);
    let (res_tx, mut res_rx) = mpsc::channel::<TranscriptEvent>(8);

    std::thread::spawn(move || {
        let mut worker: Option<WhisperWorker> = None;
        let cfg = config::startup();
        ready_tx.send(()).ok();

        for job in job_rx {
            let w = worker.get_or_insert_with(|| {
            WhisperWorker::load("models/whisper-accurate",
                    cfg.use_gpu_acc, cfg.gpu_device_acc)
            });
            let duration_s = job.audio.len() as f32 / TARGET_SAMPLE_RATE as f32;
            let t0 = std::time::Instant::now();

            let debug_filename = format!("pass2_debug_phrase_{}.wav", job.phrase_id);
            dump_audio_to_file(&job.audio, &debug_filename);

            let (text, _) = run_whisper(
                &mut w.state,
                &job.audio,
                &WhisperConfig::accurate(&job.context),
            );

            let dur_ms = t0.elapsed().as_millis();
            let rtf = t0.elapsed().as_secs_f32() / duration_s.max(0.001);
            tracing::info!(
                target: "PERFORMANCE",
                pid = job.phrase_id,
                pass = 2,
                ms = dur_ms,
                rtf = format!("{:.3}", rtf),
                text_len = text.len(),
                "Inference finished"
            );
            stats::get().record_pass2_done(job.phrase_id, rtf, text.len());

            let event = TranscriptEvent::Final {
                phrase_id: job.phrase_id,
                text: text.trim().to_string(),
                duration_s,
                rtf,
                sent_at: std::time::Instant::now(),
            };
            if res_tx.blocking_send(event).is_err() {
                break;
            }
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
                if phrases.len() > 32 {
                    let min_id = phrases.keys().min().copied().unwrap_or(0);
                    phrases.remove(&min_id);
                    tracing::warn!("pass2: force-dropped stale phrase {}", min_id);
                }
                if chunk.is_last {
                    let audio = phrases.remove(&chunk.phrase_id).unwrap_or_default();
                    let audio_len = audio.len();
                    if audio_len < config::min_phrase_samples() {
                        debug!(pid = chunk.phrase_id, "pass2: phrase too short after accumulation, skipping");
                        continue;
                    }

                    stats::get().record_pass2_start(chunk.phrase_id);
                    let job = Pass2Job { phrase_id: chunk.phrase_id, audio, context: Arc::clone(&last_context), };
                    if let Err(e) = job_tx.try_send(job) {
                        match e {
                            TrySendError::Full(_) => {
                                debug!("Pass2 Worker busy, skipping chunk");
                                stats::get().pass2_dropped.fetch_add(1, Ordering::Relaxed);
                            },
                            TrySendError::Disconnected(_) => {
                                error!("Worker thread died! Shutting down async task.");
                                break;
                            }
                        }
                    }
                } else {
                    continue;
                }
            }
        }
    }
}

pub fn disable_whisper_log() {
    #[cfg(target_os = "windows")]
    return;
    unsafe { whisper_rs::set_log_callback(Some(whisper_log_callback), std::ptr::null_mut()); }
}

unsafe extern "C" fn whisper_log_callback(
    // For some reason compilation for windows breaks if level value is unsigned, and it asks for integer.
    #[cfg(target_os = "windows")]
    _level: i32,
    #[cfg(not(target_os = "windows"))]
    _level: u32,
    _msg: *const std::os::raw::c_char,
    _user_data: *mut std::ffi::c_void,
) {}

