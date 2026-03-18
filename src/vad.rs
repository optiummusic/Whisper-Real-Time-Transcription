use std::collections::VecDeque;
use std::sync::Arc;
use tokio::sync::mpsc;
use ort::{inputs, session::Session, value::Value, value::TensorValueType};
use ndarray::{Array, Array2, Array3};
use tracing::{info, debug, trace};

use crate::types::{AudioPacket, PhraseChunk};
use crate::{
    TARGET_SAMPLE_RATE, VAD_CHUNK_SIZE, MAX_PHRASE_SAMPLES, 
    MAX_SILENCE_CHUNKS, MIN_PHRASE_SAMPLES, STREAM_CHUNK_SAMPLES, FAST_TRACK_THRESHOLD_S, PREROLL_CHUNKS,
    STITCH_MIN_SAMPLES, STITCH_MAX_CHUNKS, SPEECH_PROBABILITY
};

pub struct VadEngine {
    model: Session,
    h: Array3<f32>,
    c: Array3<f32>,
    sr: Value<TensorValueType<i64>>,
    
    stream_buf: Vec<f32>,
    phrase_id: u32,
    chunk_id: u32,
    phrase_total: usize,
    is_speaking: bool,
    silence_chunks: usize,
    phrase_start_ts: std::time::Instant,
    preroll: VecDeque<Arc<Vec<f32>>>,
    
    stitch_buf: Vec<f32>,
    stitch_id: u32,
    stitch_silence: usize,
}

impl VadEngine {
    pub fn new(model_path: &std::path::Path) -> Self {
        let model = Session::builder().unwrap()
            .commit_from_file(model_path).unwrap();

        let h = Array3::<f32>::zeros((2, 1, 64));
        let c = Array3::<f32>::zeros((2, 1, 64));
        let sr = Value::from_array(Array::from_elem((1,), 16_000_i64)).unwrap();

        Self {
            model,
            h,
            c,
            sr,
            stream_buf: Vec::with_capacity(STREAM_CHUNK_SAMPLES),
            phrase_id: 0,
            chunk_id: 0,
            phrase_total: 0,
            is_speaking: false,
            silence_chunks: 0,
            phrase_start_ts: std::time::Instant::now(),
            preroll: VecDeque::with_capacity(PREROLL_CHUNKS + 1),
            stitch_buf: Vec::with_capacity(STITCH_MIN_SAMPLES * 2),
            stitch_id: u32::MAX,
            stitch_silence: 0,
        }
    }

    pub fn process(&mut self, audio_data: Vec<f32>) -> Vec<PhraseChunk> {
        let mut results = Vec::new();
        
        self.preroll.push_back(Arc::new(audio_data.clone()));
        if self.preroll.len() > PREROLL_CHUNKS + 1 {
            self.preroll.pop_front();
        }
        
        let (probability, hn_vec, cn_vec) = {
            let input_array = Array2::from_shape_vec((1, VAD_CHUNK_SIZE), audio_data.clone()).unwrap();
            let outputs = self.model.run(inputs![
                "input" => Value::from_array(input_array).unwrap(),
                "sr"    => self.sr.clone(),
                "h"     => Value::from_array(self.h.clone()).unwrap(),
                "c"     => Value::from_array(self.c.clone()).unwrap(),
            ]).unwrap();

            let prob = *outputs["output"].try_extract_tensor::<f32>().unwrap().1.first().unwrap();
            
            // Копируем данные из тензоров в обычные векторы, чтобы разорвать связь с SessionOutputs
            let hn = outputs["hn"].try_extract_tensor::<f32>().unwrap().1.to_vec();
            let cn = outputs["cn"].try_extract_tensor::<f32>().unwrap().1.to_vec();
            
            (prob, hn, cn)
        };
        self.h.as_slice_mut().unwrap().copy_from_slice(&hn_vec);
        self.c.as_slice_mut().unwrap().copy_from_slice(&cn_vec);

        if probability > SPEECH_PROBABILITY {
            if !self.is_speaking {
                self.is_speaking = true;
                self.phrase_start_ts = std::time::Instant::now();

                if self.stitch_id != u32::MAX {
                    inject_preroll(&self.preroll, &mut self.stream_buf);
                    self.stream_buf.extend_from_slice(&self.stitch_buf);
                    self.stitch_buf.clear();
                    self.phrase_total = self.stream_buf.len();
                } else {
                    self.chunk_id = 0;
                    inject_preroll(&self.preroll, &mut self.stream_buf);
                    self.phrase_total = self.stream_buf.len();
                }
            }
            self.silence_chunks = 0;
        } else if self.is_speaking {
            self.silence_chunks += 1;
        }

        if self.is_speaking && self.phrase_total < MAX_PHRASE_SAMPLES {
            let active_id = if self.stitch_id != u32::MAX { self.stitch_id } else { self.phrase_id };
            let mut src = &audio_data[..];
            
            while !src.is_empty() {
                if self.stream_buf.len() >= STREAM_CHUNK_SAMPLES {
                    let data = std::mem::take(&mut self.stream_buf);
                    self.stream_buf.reserve(STREAM_CHUNK_SAMPLES);
                    results.push(PhraseChunk { 
                        phrase_id: active_id, 
                        chunk_id: self.chunk_id, 
                        is_last: false, 
                        short: false, 
                        data: Arc::new(data) 
                    });
                    self.chunk_id += 1;
                }
                let space = STREAM_CHUNK_SAMPLES - self.stream_buf.len();
                let take = src.len().min(space);
                self.stream_buf.extend_from_slice(&src[..take]);
                self.phrase_total += take;
                src = &src[take..];
            }
        }

        let phrase_ended = self.is_speaking 
            && (self.silence_chunks > MAX_SILENCE_CHUNKS || self.phrase_total >= MAX_PHRASE_SAMPLES);

        if !self.is_speaking && self.stitch_id != u32::MAX {
            self.stitch_silence += 1;
            if self.stitch_silence > STITCH_MAX_CHUNKS {
                results.push(self.flush_stitch());
            }
        }

        if phrase_ended {
            self.is_speaking = false;
            self.preroll.clear();

            if self.phrase_total >= MIN_PHRASE_SAMPLES {
                let tail = std::mem::take(&mut self.stream_buf);
                self.stream_buf.reserve(STREAM_CHUNK_SAMPLES);

                if self.phrase_total < STITCH_MIN_SAMPLES {
                    if self.stitch_id == u32::MAX {
                        self.stitch_id = self.phrase_id;
                        self.stitch_silence = 0;
                    }
                    self.stitch_buf.extend_from_slice(&tail);
                } else {
                    let emit_id = if self.stitch_id != u32::MAX {
                        let id = self.stitch_id;
                        self.stitch_id = u32::MAX;
                        id
                    } else {
                        self.phrase_id
                    };
                    results.push(self.create_final_chunk(emit_id, tail));
                }
            } else {
                self.stream_buf.clear();
            }
            self.phrase_id += 1;
            self.silence_chunks = 0;
        }

        results
    }

    fn create_final_chunk(&self, id: u32, data: Vec<f32>) -> PhraseChunk {
        let dur_secs = self.phrase_total as f32 / TARGET_SAMPLE_RATE as f32;
        PhraseChunk {
            phrase_id: id,
            chunk_id: self.chunk_id,
            is_last: true,
            short: dur_secs < FAST_TRACK_THRESHOLD_S,
            data: Arc::new(data),
        }
    }

    fn flush_stitch(&mut self) -> PhraseChunk {
        let data = std::mem::take(&mut self.stitch_buf);
        let id = self.stitch_id;
        self.stitch_id = u32::MAX;
        self.stitch_silence = 0;
        self.create_final_chunk(id, data)
    }
}

pub async fn vad_task(
    mut rx:   mpsc::Receiver<AudioPacket>,
    pass1_tx: mpsc::Sender<PhraseChunk>,
    pass2_tx: mpsc::Sender<PhraseChunk>,
) {
    let vad_path = crate::utils::find_first_file_in_dir("models/vad", "onnx")
        .expect("No VAD model found in models/vad/");

    let mut engine = VadEngine::new(&vad_path);
    info!("VAD Engine initialized");

    while let Some(audio_data) = rx.recv().await {
        let chunks = engine.process(audio_data);
        
        for chunk in chunks {
            let _ = pass1_tx.try_send(chunk.clone());
            if !chunk.short || !chunk.is_last {
                let _ = pass2_tx.try_send(chunk);
            }
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