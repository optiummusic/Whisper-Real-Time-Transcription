use std::collections::VecDeque;
use std::sync::Arc;
use tokio::sync::mpsc;
use ort::{inputs, session::Session, value::Value};
use ndarray::{Array, Array2, Array3};
use tracing::{info, debug, trace, warn};
use std::fmt::Write;
use ort::memory::AllocatorType;

use crate::types::{AudioPacket, PhraseChunk};
use crate::{
    TARGET_SAMPLE_RATE, VAD_CHUNK_SIZE, MAX_PHRASE_SAMPLES, 
    MAX_SILENCE_CHUNKS, MIN_PHRASE_SAMPLES, STREAM_CHUNK_SAMPLES, FAST_TRACK_THRESHOLD_S, PREROLL_CHUNKS,
    STITCH_MIN_SAMPLES, STITCH_MAX_CHUNKS
};

pub async fn vad_task(
    mut rx:   mpsc::Receiver<AudioPacket>,
    pass1_tx: mpsc::Sender<PhraseChunk>,
    pass2_tx: mpsc::Sender<PhraseChunk>,
) {
    let vad_path = crate::utils::find_first_file_in_dir("models/vad", "onnx")
    .expect("No VAD model found in models/vad/");

    let mut model = Session::builder().unwrap()
        .commit_from_file(&vad_path).unwrap();

    info!("Warming up VAD model...");
    let warmup_input = Array2::<f32>::zeros((1, VAD_CHUNK_SIZE));
    let warmup_h = Array3::<f32>::zeros((2, 1, 64));
    let warmup_c = Array3::<f32>::zeros((2, 1, 64));
    let warmup_sr = Array::from_elem((1,), 16_000_i64);

    let _ = model.run(inputs![
        "input" => Value::from_array(warmup_input).unwrap(),
        "sr"    => Value::from_array(warmup_sr).unwrap(),
        "h"     => Value::from_array(warmup_h).unwrap(),
        "c"     => Value::from_array(warmup_c).unwrap()
    ]).unwrap();
    info!("Warmup done!");

    let mut h  = Array3::<f32>::zeros((2, 1, 64));
    let mut c  = Array3::<f32>::zeros((2, 1, 64));
    let sr     = Value::from_array(Array::from_elem((1,), 16_000_i64)).unwrap();
    let mut input_buf = [0f32; VAD_CHUNK_SIZE];

    let mut stream_buf      = Vec::with_capacity(STREAM_CHUNK_SAMPLES);
    let mut phrase_id:      u32   = 0;
    let mut chunk_id:       u32   = 0;
    let mut phrase_total:   usize = 0;
    let mut is_speaking:    bool  = false;
    let mut silence_chunks: usize = 0;
    let mut phrase_start_ts = std::time::Instant::now();

    let mut preroll: VecDeque<Arc<Vec<f32>>> = VecDeque::with_capacity(PREROLL_CHUNKS + 1);

    let mut stitch_buf:     Vec<f32> = Vec::with_capacity(STITCH_MIN_SAMPLES * 2);
    let mut stitch_id:      u32      = u32::MAX;
    let mut stitch_silence: usize    = 0;


    while let Some(audio_data) = rx.recv().await {
        preroll.push_back(Arc::new(audio_data.clone()));
        if preroll.len() > PREROLL_CHUNKS + 1 {
            preroll.pop_front();
        }

        input_buf.copy_from_slice(&audio_data);
        let input_array = Array2::from_shape_vec((1, VAD_CHUNK_SIZE), input_buf.to_vec()).unwrap();

        let outputs = model.run(inputs![
            "input" => Value::from_array(input_array).unwrap(),
            "sr"    => sr.clone(),
            "h"     => Value::from_array(h.clone()).unwrap(),
            "c"     => Value::from_array(c.clone()).unwrap(),
        ]).unwrap();

        let probability =
            *outputs["output"].try_extract_tensor::<f32>().unwrap().1.first().unwrap();

        let hn = outputs["hn"].try_extract_tensor::<f32>().unwrap();
        let cn = outputs["cn"].try_extract_tensor::<f32>().unwrap();

        h.as_slice_mut().unwrap().copy_from_slice(hn.1);
        c.as_slice_mut().unwrap().copy_from_slice(cn.1);

        if probability > 0.5 {
            if !is_speaking {
                is_speaking = true;
                phrase_start_ts = std::time::Instant::now();

                if stitch_id != u32::MAX {
                    trace!(
                        prev_pid = stitch_id, 
                        next_pid = phrase_id, 
                        stitch_samples = stitch_buf.len(), 
                        "phrase successfully stitched: merged {} into {}", 
                        stitch_id, 
                        phrase_id
                    );
                    let preroll_samples = inject_preroll(&preroll, &mut stream_buf);
                    stream_buf.extend_from_slice(&stitch_buf);
                    stitch_buf.clear();
                    phrase_total = stream_buf.len();
                    if preroll_samples > 0 {
                        trace!(preroll_samples, "pre-roll injected before stitch");
                    }
                } else {
                    chunk_id = 0;
                    let preroll_samples = inject_preroll(&preroll, &mut stream_buf);
                    phrase_total = stream_buf.len();
                    if preroll_samples > 0 {
                        debug!(preroll_samples, "pre-roll injected at phrase start");
                    }
                }
                
                let active_id = if stitch_id != u32::MAX { stitch_id } else { phrase_id };
                info!(active_id, "phrase started");
            }
            silence_chunks = 0;
        } else if is_speaking {
            silence_chunks += 1;
        }

        if is_speaking && phrase_total < MAX_PHRASE_SAMPLES {
            let active_id = if stitch_id != u32::MAX { stitch_id } else { phrase_id };
            
            let mut src = &audio_data[..];
            while !src.is_empty() {
                let current_len = stream_buf.len();
                
                if current_len >= STREAM_CHUNK_SAMPLES {
                    let data = std::mem::take(&mut stream_buf);
                    stream_buf.reserve(STREAM_CHUNK_SAMPLES);
                    
                    let chunk = PhraseChunk { 
                        phrase_id: active_id, 
                        chunk_id, 
                        is_last: false, 
                        short: false, 
                        data: Arc::new(data) 
                    };
                    
                    let _ = pass1_tx.try_send(chunk.clone());
                    let _ = pass2_tx.try_send(chunk);
                    chunk_id += 1;
                    continue;
                }

                let space = STREAM_CHUNK_SAMPLES - current_len;
                let take = src.len().min(space);
                stream_buf.extend_from_slice(&src[..take]);
                phrase_total += take;
                src = &src[take..];
            }
        }

        let phrase_ended = is_speaking
            && (silence_chunks > MAX_SILENCE_CHUNKS || phrase_total >= MAX_PHRASE_SAMPLES);

        if !is_speaking && stitch_id != u32::MAX {
            stitch_silence += 1;
        }

        if stitch_id != u32::MAX && stitch_silence > STITCH_MAX_CHUNKS {
            let dur_secs = stitch_buf.len() as f32 / TARGET_SAMPLE_RATE as f32;
            let use_fast_track = dur_secs < FAST_TRACK_THRESHOLD_S;

            debug!(stitch_id, dur_secs, use_fast_track, "stitch timeout — flushing");
            let arc   = Arc::new(std::mem::take(&mut stitch_buf));
            let chunk = PhraseChunk { phrase_id: stitch_id, chunk_id: 0,
                                      is_last: true, short: use_fast_track, data: arc };
            let _ = pass1_tx.send(chunk.clone()).await;
            if !use_fast_track {
                let _ = pass2_tx.send(chunk).await;
            }
            stitch_id      = u32::MAX;
            stitch_silence = 0;
        }

        if phrase_ended {
            is_speaking = false;
            preroll.clear();

            let dur = phrase_start_ts.elapsed().as_secs_f32();
            info!(phrase_id, dur, samples_total = phrase_total, "phrase ended");
    
            if phrase_total >= MIN_PHRASE_SAMPLES {
                let tail = std::mem::take(&mut stream_buf);
                stream_buf.reserve(STREAM_CHUNK_SAMPLES);
 
                if phrase_total < STITCH_MIN_SAMPLES {
                    if stitch_id == u32::MAX {
                        stitch_id      = phrase_id;
                        stitch_silence = 0;
                    }
                    stitch_buf.extend_from_slice(&tail);
                    debug!(phrase_id, stitch_len = stitch_buf.len(), "phrase too short — stitching");
                } else {
                    let emit_id = if stitch_id != u32::MAX {
                        let id = stitch_id;
                        stitch_id = u32::MAX;
                        stitch_silence = 0;
                        id
                    } else {
                        phrase_id
                    };

                    let dur_secs = phrase_total as f32 / TARGET_SAMPLE_RATE as f32;
                    let use_fast_track = dur_secs < FAST_TRACK_THRESHOLD_S;

                    let arc   = Arc::new(tail);
                    let chunk = PhraseChunk { 
                        phrase_id: emit_id, 
                        chunk_id,
                        is_last: true, 
                        short: use_fast_track, 
                        data: arc 
                    };
                    
                    let _ = pass1_tx.send(chunk.clone()).await;
                    
                    if !use_fast_track {
                        let _ = pass2_tx.send(chunk).await;
                    }
                }
            } else {
                stream_buf.clear();
                info!(phrase_id, "phrase too short — skipping");
            }
 
            phrase_id      += 1;
            silence_chunks  = 0;
        }
    }
}

fn inject_preroll(preroll: &VecDeque<Arc<Vec<f32>>>, dst: &mut Vec<f32>) -> usize {
    let count = preroll.len().saturating_sub(1);
    let mut total = 0usize;
    for chunk in preroll.iter().take(count) {
        dst.extend_from_slice(chunk);
        total += chunk.len();
    }
    total
}