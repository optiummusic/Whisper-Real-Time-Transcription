use crate::prelude::*;
use crate::{Pass1Job};
use crate::config::PASS1_MIN_SAMPLES;

pub struct StreamInfo {
    current_id: Option<u32>,
    closed_id: Option<u32>,
    buffer: Vec<f32>,
}

impl Default for StreamInfo {
    fn default() -> Self {
        Self::new()
    }
}

impl StreamInfo {
    pub fn new() -> Self {
        Self {
            current_id: None,
            closed_id: None,
            buffer: Vec::with_capacity(STREAM_CHUNK_SAMPLES * 16),
        }
    }

    pub fn process_incoming(
        &mut self,
        mut chunk: PhraseChunk,
        rx: &mut mpsc::Receiver<PhraseChunk>,
    ) -> Option<Pass1Job> {
        let mut drained = 0;
        while drained < 8 {
            match rx.try_recv() {
                Ok(next) => {
                    if next.phrase_id > chunk.phrase_id && Some(chunk.phrase_id) == self.current_id
                    {
                        debug!("Drained");
                        self.reset(next.phrase_id);
                    }
                    chunk = next;
                    drained += 1;
                }
                Err(_) => break,
            }
        }

        if let Some(cur) = self.current_id
            && chunk.phrase_id < cur {
                return None;
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

    pub fn is_result_valid(
        &self,
        phrase_id: u32,
        short: bool,
        is_last: bool,
        single_pass: bool,
    ) -> bool {
        if short {
            return true;
        }
        if is_last && single_pass {
            return true;
        }
        if Some(phrase_id) == self.closed_id {
            println!("Phrase closed id");
            return false;
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
            info!(
                "{} chunk is less than {}, {}",
                chunk.chunk_id,
                PASS1_MIN_SAMPLES,
                chunk.data.len()
            );
            return None;
        }

        let window = if chunk.is_last {
            config::max_window()
        } else {
            config::min_window()
        };

        let audio = if len > window {
            self.buffer[len - window..].to_vec()
        } else {
            self.buffer.to_vec()
        };

        let phrase_id = chunk.phrase_id;
        let chunk_id = chunk.chunk_id;

        if chunk.is_last {
            self.buffer.clear();
        }
        trace!("[MAKE JOB]{} phrase, {} chunk is sent", phrase_id, chunk_id);
        Some(Pass1Job {
            phrase_id,
            chunk_id,
            short: chunk.short,
            is_last: chunk.is_last,
            audio,
        })
    }
}
