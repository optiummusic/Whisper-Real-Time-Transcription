use std::sync::atomic::{AtomicBool, AtomicU32, AtomicUsize, Ordering};
use std::sync::OnceLock;

pub const TARGET_SAMPLE_RATE:   u32   = 16_000;
pub const VAD_CHUNK_SIZE:       usize = 480;

pub const STREAM_CHUNK_SAMPLES: usize = TARGET_SAMPLE_RATE as usize;
pub const PASS1_MIN_SAMPLES:    usize = TARGET_SAMPLE_RATE as usize;
pub const STITCH_MIN_SAMPLES:   usize = (TARGET_SAMPLE_RATE as f32 * 1.5) as usize;
pub const VAD_CHUNK_DURATION_S: f32   = VAD_CHUNK_SIZE as f32 / TARGET_SAMPLE_RATE as f32;

static SPEECH_PROB_X1000:  AtomicU32   = AtomicU32::new(500);
static MAX_SILENCE_CHK:    AtomicUsize = AtomicUsize::new(12);
static STITCH_MAX_SIL_MS:  AtomicU32   = AtomicU32::new(1200);
static FAST_TRACK_MS:      AtomicU32   = AtomicU32::new(3000);
static PREROLL_CHK:        AtomicUsize = AtomicUsize::new(5);
static DUMP_AUDIO_FLAG:    AtomicBool  = AtomicBool::new(false);
static MIN_WINDOW_SEC_X10: AtomicU32   = AtomicU32::new(40);
static MAX_WINDOW_SEC_X10: AtomicU32   = AtomicU32::new(100);
static MAX_PHRASE_SEC_X10: AtomicU32   = AtomicU32::new(120);
static MIN_PHRASE_SEC_X10: AtomicU32   = AtomicU32::new(20);

pub fn speech_probability()     -> f32   { SPEECH_PROB_X1000.load(Ordering::Relaxed) as f32 / 1000.0 }
pub fn max_silence_chunks()     -> usize { MAX_SILENCE_CHK.load(Ordering::Relaxed) }
pub fn stitch_max_silence()     -> f32   { STITCH_MAX_SIL_MS.load(Ordering::Relaxed) as f32 / 1000.0 }
pub fn stitch_max_chunks()      -> usize { (stitch_max_silence() / VAD_CHUNK_DURATION_S) as usize }
pub fn fast_track_threshold_s() -> f32   { FAST_TRACK_MS.load(Ordering::Relaxed) as f32 / 1000.0 }
pub fn preroll_chunks()         -> usize { PREROLL_CHK.load(Ordering::Relaxed) }
pub fn dump_audio()             -> bool  { DUMP_AUDIO_FLAG.load(Ordering::Relaxed) }
pub fn min_window()             -> usize { (MIN_WINDOW_SEC_X10.load(Ordering::Relaxed) as f32 / 10.0 * TARGET_SAMPLE_RATE as f32) as usize }
pub fn max_window()             -> usize { (MAX_WINDOW_SEC_X10.load(Ordering::Relaxed) as f32 / 10.0 * TARGET_SAMPLE_RATE as f32) as usize }
pub fn max_phrase_samples()     -> usize { (MAX_PHRASE_SEC_X10.load(Ordering::Relaxed) as f32 / 10.0 * TARGET_SAMPLE_RATE as f32) as usize }
pub fn min_phrase_samples()     -> usize { (MIN_PHRASE_SEC_X10.load(Ordering::Relaxed) as f32 / 10.0 * TARGET_SAMPLE_RATE as f32) as usize }

pub fn set_speech_probability(v: f32)    { SPEECH_PROB_X1000.store((v * 1000.0) as u32, Ordering::Relaxed); }
pub fn set_max_silence_chunks(v: usize)  { MAX_SILENCE_CHK.store(v, Ordering::Relaxed); }
pub fn set_stitch_max_silence(v: f32)    { STITCH_MAX_SIL_MS.store((v * 1000.0) as u32, Ordering::Relaxed); }
pub fn set_fast_track_threshold(v: f32)  { FAST_TRACK_MS.store((v * 1000.0) as u32, Ordering::Relaxed); }
pub fn set_preroll_chunks(v: usize)      { PREROLL_CHK.store(v, Ordering::Relaxed); }
pub fn set_dump_audio(v: bool)           { DUMP_AUDIO_FLAG.store(v, Ordering::Relaxed); }
pub fn set_min_window_secs(v: f32)       { MIN_WINDOW_SEC_X10.store((v * 10.0) as u32, Ordering::Relaxed); }
pub fn set_max_window_secs(v: f32)       { MAX_WINDOW_SEC_X10.store((v.max(min_window() as f32 / TARGET_SAMPLE_RATE as f32 + 1.0) * 10.0) as u32, Ordering::Relaxed); }
pub fn set_max_phrase_secs(v: f32)       { MAX_PHRASE_SEC_X10.store((v * 10.0) as u32, Ordering::Relaxed); }
pub fn set_min_phrase_secs(v: f32)       { MIN_PHRASE_SEC_X10.store((v * 10.0) as u32, Ordering::Relaxed); }

#[derive(Debug)]
pub struct StartupConfig {
    pub language:     String,
    pub use_gpu_fast: bool,
    pub use_gpu_acc:  bool,
}

static STARTUP: OnceLock<StartupConfig> = OnceLock::new();
static DEFAULT_CONFIG: StartupConfig = StartupConfig {
    language: String::new(), // String нельзя в константах, поэтому используем небольшую хитрость ниже
    use_gpu_fast: true,
    use_gpu_acc: true,
};
pub fn startup() -> &'static StartupConfig {
    // Вместо создания временного объекта, используем get_or_init
    STARTUP.get_or_init(|| StartupConfig {
        language: "en".to_string(),
        use_gpu_fast: true,
        use_gpu_acc: true,
    })
}

pub fn init(language: String, use_gpu_fast: bool, use_gpu_acc: bool) {
    STARTUP.set(StartupConfig { language, use_gpu_fast, use_gpu_acc })
        .expect("init() called twice");
}

// ── TOML ──────────────────────────────────────────────────────────────────────
#[derive(serde::Deserialize, Default)]
#[serde(default)]
pub struct TomlConfig {
    pub language:             Option<String>,
    pub use_gpu_fast:         Option<bool>,
    pub use_gpu_acc:          Option<bool>,
    pub speech_probability:   Option<f32>,
    pub max_silence_chunks:   Option<usize>,
    pub stitch_max_silence:   Option<f32>,
    pub fast_track_threshold: Option<f32>,
    pub preroll_chunks:       Option<usize>,
    pub dump_audio:           Option<bool>,
    pub min_window_secs:      Option<f32>, 
    pub max_window_secs:      Option<f32>,
    pub min_phrase_secs:      Option<f32>,
    pub max_phrase_secs:      Option<f32>, 
}

pub fn load_from_toml(path: &str) {
    let cfg: TomlConfig = std::fs::read_to_string(path)
        .ok()
        .and_then(|content| match toml::from_str(&content) {
            Ok(c) => Some(c),
            Err(e) => {
                eprintln!("Warning: failed to parse {path}: {e}");
                None
            }
        })
        .unwrap_or_default();

    if let Some(v) = cfg.speech_probability   { set_speech_probability(v); }
    if let Some(v) = cfg.max_silence_chunks   { set_max_silence_chunks(v); }
    if let Some(v) = cfg.stitch_max_silence   { set_stitch_max_silence(v); }
    if let Some(v) = cfg.fast_track_threshold { set_fast_track_threshold(v); }
    if let Some(v) = cfg.preroll_chunks       { set_preroll_chunks(v); }
    if let Some(v) = cfg.dump_audio           { set_dump_audio(v); }
    if let Some(v) = cfg.min_window_secs      { set_min_window_secs(v); }
    if let Some(v) = cfg.max_window_secs      { set_max_window_secs(v); }
    if let Some(v) = cfg.min_phrase_secs      { set_min_phrase_secs(v); }
    if let Some(v) = cfg.max_phrase_secs      { set_max_phrase_secs(v); }

    init(
        cfg.language.unwrap_or_else(|| "en".to_string()),
        cfg.use_gpu_fast.unwrap_or(true),
        cfg.use_gpu_acc.unwrap_or(true),
    );
}

pub fn save_to_toml(path: &str) {
    let content = format!(
        r#"language = "{}"
        use_gpu_fast = {}
        use_gpu_acc = {}
        speech_probability = {:.3}
        max_silence_chunks = {}
        min_window_secs = {:.1}
        max_window_secs = {:.1}
        min_phrase_secs = {:.1}
        max_phrase_secs = {:.1}
        dump_audio = {}
        "#,
        startup().language,
        startup().use_gpu_fast,
        startup().use_gpu_acc,
        speech_probability(),
        max_silence_chunks(),
        min_window() as f32 / TARGET_SAMPLE_RATE as f32,
        max_window() as f32 / TARGET_SAMPLE_RATE as f32,
        min_phrase_samples() as f32 / TARGET_SAMPLE_RATE as f32,
        max_phrase_samples() as f32 / TARGET_SAMPLE_RATE as f32,
        dump_audio(),
    );
    let _ = std::fs::write(path, content);
}