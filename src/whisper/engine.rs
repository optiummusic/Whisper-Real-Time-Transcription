use std::time::Instant;

use crate::config;
use whisper_rs::{FullParams, SamplingStrategy};

pub struct WhisperConfig<'a> {
    n_threads: i32,
    no_speech_thold: f32,
    temperature: f32,
    context_prompt: &'a str,
    pass: i32,
}

impl WhisperConfig<'_> {
    pub fn fast() -> Self {
        Self {
            n_threads: 2,
            no_speech_thold: 0.5,
            temperature: 0.0,
            context_prompt: "",
            pass: 1,
        }
    }

    pub fn accurate(context_prompt: &str) -> WhisperConfig<'_> {
        WhisperConfig {
            n_threads: 1,
            no_speech_thold: 0.4,
            temperature: -1.0,
            context_prompt,
            pass: 2,
        }
    }
}

pub fn run_whisper(
    state: &mut whisper_rs::WhisperState,
    audio: &[f32],
    cfg: &WhisperConfig,
) -> (String, i32) {
    let strategy = if cfg.temperature < 0.0 {
        SamplingStrategy::Greedy { best_of: 1 }
    } else {
        SamplingStrategy::Greedy { best_of: 1 }
    };
    let mut params = FullParams::new(strategy);
    let cfg_start = config::startup();
    params.set_language(Some(&cfg_start.language));
    params.set_n_threads(cfg.n_threads);
    params.set_print_special(false);
    params.set_print_progress(false);
    params.set_print_realtime(false);
    params.set_print_timestamps(false);
    params.set_translate(false);
    params.set_no_speech_thold(cfg.no_speech_thold);
    params.set_token_timestamps(false);

    if cfg.temperature >= 0.0 {
        params.set_temperature(cfg.temperature);
        params.set_n_max_text_ctx(128);
        params.set_offset_ms(0);
        params.set_duration_ms(0);
    }
    if cfg.context_prompt.is_empty() {
        params.set_no_context(true);
    } else {
        params.set_initial_prompt(cfg.context_prompt);
    }
    let t0 = Instant::now();
    tracing::info!("ENTER full pass={}", cfg.pass);
    if state.full(params, audio).is_err() {
        return (String::new(), 0);
    }
    tracing::info!(
        "EXIT full pass={} ms={}",
        cfg.pass,
        t0.elapsed().as_millis()
    );
    let total = state.full_n_segments();
    let mut text = String::with_capacity(256);
    for i in 0..total {
        if let Some(seg) = state.get_segment(i) {
            if i > 0 {
                text.push(' ');
            }
            text.push_str(&seg.to_string());
        }
    }
    (text.trim().to_string(), total)
}
