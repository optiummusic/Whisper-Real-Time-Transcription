use std::sync::atomic::{AtomicBool, AtomicU32, AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock, RwLock};

use lxdb::LxdbReader;

// GUARDS
/// When true, VAD task drains audio packets and emits nothing downstream.
pub static AUDIO_MUTED: AtomicBool = AtomicBool::new(false);

/// When true, Translator skips processing and sends no TranslationEvents.
pub static TRANSLATION_MUTED: AtomicBool = AtomicBool::new(false);

// Skip if whisper models are the same
pub static SINGLE_PASS: AtomicBool = AtomicBool::new(false);

pub const TARGET_SAMPLE_RATE: u32 = 16_000;
pub const VAD_CHUNK_SIZE: usize = 512;

pub static CONFIG_VERSION: AtomicU32 = AtomicU32::new(0);
pub static TARGET_LANG: OnceLock<RwLock<Arc<str>>> = OnceLock::new();
pub static SOURCE_LANG: OnceLock<RwLock<Arc<str>>> = OnceLock::new();

pub const STREAM_CHUNK_SAMPLES: usize = TARGET_SAMPLE_RATE as usize;
pub const PASS1_MIN_SAMPLES: usize = TARGET_SAMPLE_RATE as usize;
pub const STITCH_MIN_SAMPLES: usize = (TARGET_SAMPLE_RATE as f32 / 2.5) as usize;
pub const VAD_CHUNK_DURATION_S: f32 = VAD_CHUNK_SIZE as f32 / TARGET_SAMPLE_RATE as f32;

static SPEECH_PROB_X1000: AtomicU32 = AtomicU32::new(500);
static MAX_SILENCE_CHK: AtomicUsize = AtomicUsize::new(12);
static STITCH_MAX_SIL_MS: AtomicU32 = AtomicU32::new(1200);
static FAST_TRACK_MS: AtomicU32 = AtomicU32::new(3000);
static PREROLL_CHK: AtomicUsize = AtomicUsize::new(5);
static DUMP_AUDIO_FLAG: AtomicBool = AtomicBool::new(false);
static MIN_WINDOW_SEC_X10: AtomicU32 = AtomicU32::new(40);
static MAX_WINDOW_SEC_X10: AtomicU32 = AtomicU32::new(100);
static MAX_PHRASE_SEC_X10: AtomicU32 = AtomicU32::new(120);
static MIN_PHRASE_SEC_X10: AtomicU32 = AtomicU32::new(20);
static SELECTED_DEVICE: OnceLock<RwLock<String>> = OnceLock::new();
static AUDIO_GAIN_X100: AtomicU32 = AtomicU32::new(100);

fn device_storage() -> &'static RwLock<String> {
    SELECTED_DEVICE.get_or_init(|| RwLock::new(String::new()))
}

pub fn speech_probability() -> f32 {
    SPEECH_PROB_X1000.load(Ordering::Relaxed) as f32 / 1000.0
}
pub fn max_silence_chunks() -> usize {
    MAX_SILENCE_CHK.load(Ordering::Relaxed)
}
pub fn stitch_max_silence() -> f32 {
    STITCH_MAX_SIL_MS.load(Ordering::Relaxed) as f32 / 1000.0
}
pub fn stitch_max_chunks() -> usize {
    (stitch_max_silence() / VAD_CHUNK_DURATION_S) as usize
}
pub fn fast_track_threshold_s() -> f32 {
    FAST_TRACK_MS.load(Ordering::Relaxed) as f32 / 1000.0
}
pub fn preroll_chunks() -> usize {
    PREROLL_CHK.load(Ordering::Relaxed)
}
pub fn dump_audio() -> bool {
    DUMP_AUDIO_FLAG.load(Ordering::Relaxed)
}
pub fn min_window() -> usize {
    (MIN_WINDOW_SEC_X10.load(Ordering::Relaxed) as f32 / 10.0 * TARGET_SAMPLE_RATE as f32) as usize
}
pub fn max_window() -> usize {
    (MAX_WINDOW_SEC_X10.load(Ordering::Relaxed) as f32 / 10.0 * TARGET_SAMPLE_RATE as f32) as usize
}
pub fn max_phrase_samples() -> usize {
    (MAX_PHRASE_SEC_X10.load(Ordering::Relaxed) as f32 / 10.0 * TARGET_SAMPLE_RATE as f32) as usize
}
pub fn min_phrase_samples() -> usize {
    (MIN_PHRASE_SEC_X10.load(Ordering::Relaxed) as f32 / 10.0 * TARGET_SAMPLE_RATE as f32) as usize
}
pub fn get_device() -> String {
    device_storage().read().unwrap().clone()
}
pub fn audio_gain() -> f32 {
    AUDIO_GAIN_X100.load(Ordering::Relaxed) as f32 / 100.0
}

pub fn target_lang() -> Arc<str> {
    TARGET_LANG.get_or_init(|| RwLock::new(Arc::from("en")))
        .read().unwrap().clone()
}

pub fn source_lang() -> Arc<str> {
    SOURCE_LANG.get_or_init(|| RwLock::new(Arc::from("ru")))
        .read().unwrap().clone()
}

pub fn get_config_version() -> u32 {
    CONFIG_VERSION.load(Ordering::Relaxed)
}

pub fn get_available_languages() -> Vec<(String, String)> {
    let path = crate::utility::utils::get_base_dir()
        .join("dictionary")
        .join("main.lxdb");

    if let Ok(data) = std::fs::read(path) {
        if let Ok(r) = LxdbReader::new(&data) {
            return r.languages()
                .map(|(code, name)| (code.to_string(), name.to_string()))
                .collect();
        }
    }
    // Если файла нет или он сломан, возвращаем дефолт, чтобы UI не был пустым
    vec![("en".into(), "English".into()), ("uk".into(), "Ukrainian".into())]
}

pub fn set_src_lang(v: String) {
    {
        let mut src = SOURCE_LANG.get_or_init(|| RwLock::new(Arc::from("ru")))
            .write().unwrap();
        *src = Arc::from(v);
    }
    // Сигнализируем об изменении
    CONFIG_VERSION.fetch_add(1, Ordering::SeqCst);
}

pub fn set_tgt_lang(v: String) {
    {
        let mut tgt = TARGET_LANG.get_or_init(|| RwLock::new(Arc::from("en")))
            .write().unwrap();
        *tgt = Arc::from(v);
    }
    // Сигнализируем об изменении
    CONFIG_VERSION.fetch_add(1, Ordering::SeqCst);
}

pub fn set_speech_probability(v: f32) {
    SPEECH_PROB_X1000.store((v * 1000.0) as u32, Ordering::Relaxed);
}
pub fn set_max_silence_chunks(v: usize) {
    MAX_SILENCE_CHK.store(v, Ordering::Relaxed);
}
pub fn set_stitch_max_silence(v: f32) {
    STITCH_MAX_SIL_MS.store((v * 1000.0) as u32, Ordering::Relaxed);
}
pub fn set_fast_track_threshold(v: f32) {
    FAST_TRACK_MS.store((v * 1000.0) as u32, Ordering::Relaxed);
}
pub fn set_preroll_chunks(v: usize) {
    PREROLL_CHK.store(v, Ordering::Relaxed);
}
pub fn set_dump_audio(v: bool) {
    DUMP_AUDIO_FLAG.store(v, Ordering::Relaxed);
}
pub fn set_min_window_secs(v: f32) {
    MIN_WINDOW_SEC_X10.store((v * 10.0) as u32, Ordering::Relaxed);
}
pub fn set_max_window_secs(v: f32) {
    MAX_WINDOW_SEC_X10.store(
        (v.max(min_window() as f32 / TARGET_SAMPLE_RATE as f32 + 1.0) * 10.0) as u32,
        Ordering::Relaxed,
    );
}
pub fn set_max_phrase_secs(v: f32) {
    MAX_PHRASE_SEC_X10.store((v * 10.0) as u32, Ordering::Relaxed);
}
pub fn set_min_phrase_secs(v: f32) {
    MIN_PHRASE_SEC_X10.store((v * 10.0) as u32, Ordering::Relaxed);
}
pub fn set_device(name: String) {
    let mut lock = device_storage().write().unwrap();
    *lock = name;
}
pub fn set_audio_gain(v: f32) {
    AUDIO_GAIN_X100.store((v * 100.0) as u32, Ordering::Relaxed);
}

#[derive(Debug, Clone)]
pub struct StartupConfig {
    pub language: String,
    pub use_gpu_fast: bool,
    pub use_gpu_acc: bool,
    pub gpu_device_fast: i32,
    pub gpu_device_acc: i32,
    pub audio_gain: f32,
}

static STARTUP_DATA: OnceLock<RwLock<StartupConfig>> = OnceLock::new();

fn startup_storage() -> &'static RwLock<StartupConfig> {
    STARTUP_DATA.get_or_init(|| {
        RwLock::new(StartupConfig {
            language: "en".to_string(),
            use_gpu_fast: true,
            use_gpu_acc: true,
            gpu_device_fast: 0,
            gpu_device_acc: 0,
            audio_gain: 1.0,
        })
    })
}

pub fn startup() -> StartupConfig {
    startup_storage().read().unwrap().clone()
}

pub fn init(
    language: String,
    use_gpu_fast: bool,
    use_gpu_acc: bool,
    gpu_fast: i32,
    gpu_acc: i32,
    audio_gain: f32,
) {
    let mut lock = startup_storage().write().unwrap();
    *lock = StartupConfig {
        language,
        use_gpu_fast,
        use_gpu_acc,
        gpu_device_fast: gpu_fast,
        gpu_device_acc: gpu_acc,
        audio_gain,
    };
    set_audio_gain(audio_gain);
}

#[derive(serde::Serialize, serde::Deserialize, Default)]
#[serde(default)]
pub struct TomlConfig {
    pub language: Option<String>,
    pub device: Option<String>,
    pub use_gpu_fast: Option<bool>,
    pub use_gpu_acc: Option<bool>,
    pub speech_probability: Option<f32>,
    pub max_silence_chunks: Option<usize>,
    pub stitch_max_silence: Option<f32>,
    pub fast_track_threshold: Option<f32>,
    pub preroll_chunks: Option<usize>,
    pub dump_audio: Option<bool>,
    pub min_window_secs: Option<f32>,
    pub max_window_secs: Option<f32>,
    pub min_phrase_secs: Option<f32>,
    pub max_phrase_secs: Option<f32>,
    pub gpu_device_fast: Option<i32>,
    pub gpu_device_acc: Option<i32>,

    pub audio_gain: Option<f32>,
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

    if let Some(v) = cfg.speech_probability {
        set_speech_probability(v);
    }
    if let Some(v) = cfg.max_silence_chunks {
        set_max_silence_chunks(v);
    }
    if let Some(v) = cfg.stitch_max_silence {
        set_stitch_max_silence(v);
    }
    if let Some(v) = cfg.fast_track_threshold {
        set_fast_track_threshold(v);
    }
    if let Some(v) = cfg.preroll_chunks {
        set_preroll_chunks(v);
    }
    if let Some(v) = cfg.dump_audio {
        set_dump_audio(v);
    }
    if let Some(v) = cfg.min_window_secs {
        set_min_window_secs(v);
    }
    if let Some(v) = cfg.max_window_secs {
        set_max_window_secs(v);
    }
    if let Some(v) = cfg.min_phrase_secs {
        set_min_phrase_secs(v);
    }
    if let Some(v) = cfg.max_phrase_secs {
        set_max_phrase_secs(v);
    }
    if let Some(v) = cfg.audio_gain {
        set_audio_gain(v);
    }

    if let Some(d) = cfg.device {
        set_device(d);
    }

    init(
        cfg.language.unwrap_or_else(|| "en".to_string()),
        cfg.use_gpu_fast.unwrap_or(true),
        cfg.use_gpu_acc.unwrap_or(true),
        cfg.gpu_device_fast.unwrap_or(0),
        cfg.gpu_device_acc.unwrap_or(0),
        cfg.audio_gain.unwrap_or(1.0),
    );
}

pub fn save_to_toml(path: &str) {
    let current = startup();
    let cfg = TomlConfig {
        language: Some(current.language),
        device: Some(get_device()),
        use_gpu_fast: Some(current.use_gpu_fast),
        use_gpu_acc: Some(current.use_gpu_acc),
        gpu_device_fast: Some(current.gpu_device_fast),
        gpu_device_acc: Some(current.gpu_device_acc),
        speech_probability: Some(speech_probability()),
        max_silence_chunks: Some(max_silence_chunks()),
        dump_audio: Some(dump_audio()),
        min_window_secs: Some(min_window() as f32 / TARGET_SAMPLE_RATE as f32),
        max_window_secs: Some(max_window() as f32 / TARGET_SAMPLE_RATE as f32),
        min_phrase_secs: Some(min_phrase_samples() as f32 / TARGET_SAMPLE_RATE as f32),
        max_phrase_secs: Some(max_phrase_samples() as f32 / TARGET_SAMPLE_RATE as f32),
        audio_gain: Some(audio_gain()),
        ..Default::default()
    };
    match toml::to_string_pretty(&cfg) {
        Ok(content) => {
            let _ = std::fs::write(path, content);
        }
        Err(e) => eprintln!("Failed to serialize config: {e}"),
    }
}
