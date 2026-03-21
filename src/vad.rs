use std::collections::VecDeque;
use std::path::Path;
use std::sync::Arc;

use ndarray::{Array, Array2, Array3};
use ort::{inputs, session::Session, value::TensorValueType, value::Value};
use tokio::sync:: { oneshot, mpsc };
use tracing::{info, warn};

use crate::config::{
    self, STITCH_MIN_SAMPLES, STREAM_CHUNK_SAMPLES, TARGET_SAMPLE_RATE, VAD_CHUNK_SIZE,
};
use crate::types::{AudioPacket, PhraseChunk};

pub struct VadEngine {
    model: Session,
    h: Array3<f32>,
    c: Array3<f32>,
    sr: Value<TensorValueType<i64>>,
    input_buf: Vec<f32>,

    stream_buf: Vec<f32>,
    phrase_id: u32,
    chunk_id: u32,
    phrase_total: usize,
    is_speaking: bool,
    silence_chunks: usize,
    preroll: VecDeque<Arc<Vec<f32>>>,

    stitch_buf: Vec<f32>,
    stitch_id: Option<u32>,
    stitch_silence: usize,
}

impl VadEngine {
    pub fn new(model_path: &Path) -> Self {
        let model = Session::builder()
            .expect("Failed to create ONNX session builder")
            .with_intra_threads(1)
            .expect("Failed intra")
            .with_inter_threads(1)
            .expect("Failed inter")
            .commit_from_file(model_path)
            .expect(&format!("Failed to load VAD model from {:?}", model_path));

        let h = Array3::<f32>::zeros((2, 1, 64));
        let c = Array3::<f32>::zeros((2, 1, 64));
        let sr = Value::from_array(Array::from_elem((1,), 16_000_i64))
            .expect("Failed to create sample rate tensor");

        Self {
            model,
            h,
            c,
            sr,
            input_buf: vec![0.0; VAD_CHUNK_SIZE],
            stream_buf: Vec::with_capacity(STREAM_CHUNK_SAMPLES),
            phrase_id: 0,
            chunk_id: 0,
            phrase_total: 0,
            is_speaking: false,
            silence_chunks: 0,
            preroll: VecDeque::with_capacity(config::preroll_chunks() + 1),
            stitch_buf: Vec::with_capacity(STITCH_MIN_SAMPLES * 2),
            stitch_id: None,
            stitch_silence: 0,
        }
    }

    pub fn process(&mut self, audio_data: Vec<f32>, results: &mut Vec<PhraseChunk>) {
        tracing::trace!("VAD process: received {} samples", audio_data.len());
        let audio = Arc::new(audio_data);
        let Some(prob) = self.run_vad(audio.as_ref()) else {
            tracing::warn!("VAD run_vad returned None! Check audio buffer size: {}", audio.len());
            return;
        };
        self.push_preroll(Arc::clone(&audio));

        if prob > config::speech_probability() {
            if !self.is_speaking {
                self.begin_speech();
            }
            self.silence_chunks = 0;
        } else if self.is_speaking {
            self.silence_chunks += 1;
        }
        tracing::info!("VAD Probability: {:.4}", prob);

        if self.is_speaking && self.phrase_total < config::max_phrase_samples() {
            let active_id = self.stitch_id.unwrap_or(self.phrase_id);
            self.push_stream_audio(audio.as_ref(), active_id, results);
        }

        let phrase_ended = self.is_speaking
            && (self.silence_chunks > config::max_silence_chunks()
                || self.phrase_total >= config::max_phrase_samples());

        if phrase_ended {
            self.finish_phrase(results);
        } else if !self.is_speaking && self.stitch_id.is_some() {
            self.stitch_silence += 1;
            if self.stitch_silence > config::stitch_max_chunks() {
                results.push(self.flush_stitch());
            }
        }

        return;
    }

    pub fn run_vad(&mut self, audio: &[f32]) -> Option<f32> {
        if audio.len() != VAD_CHUNK_SIZE {
            return None;
        }
        self.input_buf.copy_from_slice(audio);

        let input_array = Array2::from_shape_vec((1, VAD_CHUNK_SIZE), self.input_buf.clone()).ok()?;
        let input_val = Value::from_array(input_array).ok()?;
        
        let h_val = Value::from_array(self.h.clone()).ok()?;
        let c_val = Value::from_array(self.c.clone()).ok()?;

        let outputs = self.model.run(inputs![
            "input" => &input_val,
            "sr"    => &self.sr,
            "h"     => &h_val,
            "c"     => &c_val,
        ]).ok()?;

        let prob = *outputs["output"].try_extract_tensor::<f32>().ok()?.1.first()?;

        if let (Ok(hn), Ok(cn)) = (outputs["hn"].try_extract_tensor::<f32>(), outputs["cn"].try_extract_tensor::<f32>()) {
            if let Some(s) = self.h.as_slice_mut() { s.copy_from_slice(hn.1); }
            if let Some(s) = self.c.as_slice_mut() { s.copy_from_slice(cn.1); }
        } else {
            tracing::error!("VAD: Failed to extract hidden states (hn/cn)!");
        }

        Some(prob)
    }

    fn push_preroll(&mut self, audio: Arc<Vec<f32>>) {
        self.preroll.push_back(audio);
        while self.preroll.len() > config::preroll_chunks() + 1 {
            self.preroll.pop_front();
        }
    }

    fn begin_speech(&mut self) {
        self.is_speaking = true;
        self.silence_chunks = 0;
        self.phrase_total = 0;

        if self.stitch_id.is_some() {
            self.stitch_silence = 0;
        } else {
            self.chunk_id = 0;
        }

        inject_preroll(&self.preroll, &mut self.stream_buf);

        if self.stitch_id.is_some() {
            self.stream_buf.extend_from_slice(&self.stitch_buf);
            self.stitch_buf.clear();
        }

        self.phrase_total = self.stream_buf.len();
    }

    fn push_stream_audio(&mut self, audio: &[f32], phrase_id: u32, results: &mut Vec<PhraseChunk>) {
        let mut src = audio;

        while !src.is_empty() {
            if self.stream_buf.len() == STREAM_CHUNK_SAMPLES {
                self.emit_stream_chunk(phrase_id, results);
                continue;
            }

            let space = STREAM_CHUNK_SAMPLES - self.stream_buf.len();
            let take = src.len().min(space);

            self.stream_buf.extend_from_slice(&src[..take]);
            self.phrase_total += take;
            src = &src[take..];

            if self.stream_buf.len() == STREAM_CHUNK_SAMPLES {
                self.emit_stream_chunk(phrase_id, results);
            }
        }
    }

    fn emit_stream_chunk(&mut self, phrase_id: u32, results: &mut Vec<PhraseChunk>) {
        if self.stream_buf.is_empty() {
            return;
        }

        let data = self.stream_buf.split_off(0);

        results.push(PhraseChunk {
            phrase_id,
            chunk_id: self.chunk_id,
            is_last: false,
            short: false,
            data: Arc::new(data),
        });

        self.chunk_id += 1;
    }

    fn finish_phrase(&mut self, results: &mut Vec<PhraseChunk>) {
        self.is_speaking = false;
        self.preroll.clear();

        if self.phrase_total < config::min_phrase_samples() {
            self.stream_buf.clear();
            self.phrase_total = 0;
            self.silence_chunks = 0;
            self.phrase_id += 1;
            return;
        }

        let tail = self.stream_buf.split_off(0);

        if self.phrase_total < STITCH_MIN_SAMPLES {
            if self.stitch_id.is_none() {
                self.stitch_id = Some(self.phrase_id);
                self.stitch_silence = 0;
            }
            self.stitch_buf.extend_from_slice(&tail);
        } else {
            let emit_id = self.stitch_id.take().unwrap_or(self.phrase_id);
            results.push(self.build_final_chunk(emit_id, self.chunk_id, tail));
        }

        self.phrase_total = 0;
        self.silence_chunks = 0;
        self.phrase_id += 1;
    }

    fn build_final_chunk(&self, id: u32, chunk_id: u32, data: Vec<f32>) -> PhraseChunk {
        let dur_secs = self.phrase_total as f32 / TARGET_SAMPLE_RATE as f32;
        PhraseChunk {
            phrase_id: id,
            chunk_id,
            is_last: true,
            short: dur_secs < config::fast_track_threshold_s(),
            data: Arc::new(data),
        }
    }

    fn flush_stitch(&mut self) -> PhraseChunk {
        let data = std::mem::take(&mut self.stitch_buf);
        let id = self.stitch_id.take().unwrap_or(self.phrase_id);
        self.stitch_silence = 0;
        self.build_final_chunk(id, self.chunk_id, data)
    }
}

pub async fn vad_task(
    ready_tx: oneshot::Sender<()>,
    mut rx: mpsc::Receiver<AudioPacket>,
    pass1_tx: mpsc::Sender<PhraseChunk>,
    pass2_tx: mpsc::Sender<PhraseChunk>,
) {
    let vad_path = crate::utils::find_first_file_in_dir("models/vad", "onnx")
        .expect("No VAD model found in models/vad/");

    let mut engine = VadEngine::new(&vad_path);
    let mut results: Vec<PhraseChunk> = Vec::with_capacity(4);
    ready_tx.send(()).ok();

    while let Some(audio_data) = rx.recv().await {
        results.clear();
        engine.process(audio_data, &mut results);
        for chunk in results.drain(..) {
            if chunk.is_last {
                if pass1_tx.send(chunk.clone()).await.is_err() { break; }
            } else {
                if let Err(e) = pass1_tx.try_send(chunk.clone()) {
                    warn!("VAD->pass1 chunk dropped: {}", e);
                }
            }

            let send_to_pass2 = !chunk.short || !chunk.is_last;
            if send_to_pass2 {
                if chunk.is_last {
                    if pass2_tx.send(chunk).await.is_err() { break; }
                } else {
                    if let Err(e) = pass2_tx.try_send(chunk) {
                        warn!("VAD->pass2 chunk dropped: {}", e);
                    }
                }
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