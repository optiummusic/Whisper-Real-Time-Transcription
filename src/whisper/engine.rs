use whisper_rs::{FullParams, SamplingStrategy};

pub struct WhisperConfig<'a> {
    n_threads:       i32,
    no_speech_thold: f32,
    temperature:     f32,
    context_prompt:  &'a str,
}
 
impl WhisperConfig<'_> {
    pub fn fast() -> Self {
        Self { n_threads: 2, no_speech_thold: 0.5, temperature: 0.0, context_prompt: "" }
    }
 
    pub fn accurate(context_prompt: &str) -> WhisperConfig<'_> {
        WhisperConfig { n_threads: 6, no_speech_thold: 0.4, temperature: -1.0, context_prompt }
    }
}

pub fn run_whisper(
    state:  &mut whisper_rs::WhisperState,
    audio:  &[f32],
    cfg:    &WhisperConfig,
) -> (String, i32) {
    let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
    params.set_language(Some(crate::LANGUAGE));
    params.set_n_threads(cfg.n_threads);
    params.set_print_special(false);
    params.set_print_progress(false);
    params.set_print_realtime(false);
    params.set_print_timestamps(false);
    params.set_translate(false);
    params.set_no_speech_thold(cfg.no_speech_thold);
 
    if cfg.temperature >= 0.0 {
        params.set_temperature(cfg.temperature);
    }
    if !cfg.context_prompt.is_empty() {
        params.set_initial_prompt(cfg.context_prompt);
    }
 
    if state.full(params, audio).is_err() { return (String::new(), 0); }
 
    let total = state.full_n_segments();
    let mut text = String::new();
    for i in 0..total {
        if let Some(seg) = state.get_segment(i) {
            if i > 0 { text.push(' '); }
            text.push_str(&seg.to_string());
        }
    }
    (text.trim().to_string(), total)
}