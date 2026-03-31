use crate::config;
use std::collections::VecDeque;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use tokio::sync::mpsc;
use std::process::Command;
use tokio::fs::{File, OpenOptions, create_dir_all};
use tokio::io::AsyncWriteExt;

pub fn append_context(ctx: &mut String, text: &str, max_words: usize) {
    if text.is_empty() {
        return;
    }

    if !ctx.is_empty() {
        ctx.push(' ');
    }
    ctx.push_str(text.trim());

    let words: Vec<&str> = ctx.split_whitespace().collect();
    if words.len() > max_words {
        *ctx = words[words.len() - max_words..].join(" ");
    }
}

pub fn merge_strings(old: &str, new: &str) -> String {
    if old.is_empty() {
        return new.to_string();
    }
    if new.is_empty() {
        return old.to_string();
    }

    // Fast path: new literally contains the whole old string.
    if new.to_lowercase().contains(&old.trim().to_lowercase()) {
        return new.to_string();
    }

    let old_words: Vec<&str> = old.split_whitespace().collect();
    let new_words: Vec<&str> = new.split_whitespace().collect();

    // Normalise a word for overlap comparison: lowercase + strip non-alphanumeric
    let norm = |w: &str| -> String {
        w.chars()
            .filter(|c| c.is_alphanumeric())
            .collect::<String>()
            .to_lowercase()
    };

    let old_norm: Vec<String> = old_words.iter().map(|w| norm(w)).collect();
    let new_norm: Vec<String> = new_words.iter().map(|w| norm(w)).collect();

    let max_overlap = old_norm.len().min(new_norm.len());

    // Try longest overlap first.
    for overlap in (1..=max_overlap).rev() {
        let old_suffix = &old_norm[old_norm.len() - overlap..];
        let new_prefix = &new_norm[..overlap];

        // Skip trivial single-word overlaps on very common words
        // (avoids spurious merges on filler like "the", "a", "i").
        if overlap == 1 && old_suffix[0].len() <= 2 {
            continue;
        }

        if old_suffix == new_prefix {
            let mut result = old_words[..old_words.len() - overlap].to_vec();
            result.extend_from_slice(&new_words);
            return result.join(" ");
        }
    }

    // No clean overlap found.  Prefer whichever result contains more words.
    if new_words.len() >= old_words.len() {
        new.to_string()
    } else {
        // produced a truncated window.  Keep old to avoid visual regression.
        old.to_string()
    }
}

pub fn models_are_identical() -> bool {
    let fast = find_first_file_in_dir("models/whisper-fast", "bin");
    let acc = find_first_file_in_dir("models/whisper-accurate", "bin");

    let (Some(fast_path), Some(acc_path)) = (fast, acc) else {
        return false;
    };

    if fast_path == acc_path {
        tracing::info!("Model identity check: same path → single-pass mode");
        return true;
    }

    let fast_size = fs::metadata(&fast_path).map(|m| m.len()).unwrap_or(0);
    let acc_size = fs::metadata(&acc_path).map(|m| m.len()).unwrap_or(0);

    if fast_size == 0 || fast_size != acc_size {
        tracing::info!(
            "Model identity check: different sizes ({} vs {}) → two-pass mode",
            fast_size,
            acc_size
        );
        return false;
    }

    use std::io::Read;
    const SAMPLE: usize = 8192;
    let mut buf_f = [0u8; SAMPLE];
    let mut buf_a = [0u8; SAMPLE];

    let n_f = fs::File::open(&fast_path)
        .and_then(|mut f| f.read(&mut buf_f))
        .unwrap_or(0);
    let n_a = fs::File::open(&acc_path)
        .and_then(|mut f| f.read(&mut buf_a))
        .unwrap_or(0);

    let identical = n_f == n_a && buf_f[..n_f] == buf_a[..n_a];
    tracing::info!(
        "Model identity check: header comparison → {} (same_size={}, same_bytes={})",
        if identical { "single-pass" } else { "two-pass" },
        true,
        identical,
    );
    identical
}

pub fn get_base_dir() -> PathBuf {
    if let Some(pkg_dir) = option_env!("CARGO_MANIFEST_DIR") {
        PathBuf::from(pkg_dir)
    } else {
        std::env::current_exe()
            .ok()
            .and_then(|p| p.parent().map(|p| p.to_path_buf()))
            .expect("Failed to get exe path")
    }
}

pub fn get_model_path(relative_path: &str) -> PathBuf {
    let exe_path = std::env::current_exe().expect("Failed to get current exe path");
    let exe_dir = exe_path.parent().expect("Failed to get exe parent");

    let path_near_exe = exe_dir.join(relative_path);
    if path_near_exe.exists() {
        return path_near_exe;
    }

    if let Some(project_root) = exe_dir.parent().and_then(|p| p.parent()) {
        let path_in_root = project_root.join(relative_path);
        if path_in_root.exists() {
            return path_in_root;
        }
    }

    PathBuf::from(relative_path)
}

pub fn find_first_file_in_dir(relative_dir: &str, extension: &str) -> Option<PathBuf> {
    let dir_path = get_model_path(relative_dir);

    if let Ok(entries) = fs::read_dir(dir_path) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_file() && path.extension().and_then(|s| s.to_str()) == Some(extension) {
                return Some(path);
            }
        }
    }
    None
}

pub fn prepare_debug_dir() {
    if !config::dump_audio() {
        return;
    }
    let dir = "debug_audio";

    if Path::new(dir).exists() {
        let _ = fs::remove_dir_all(dir);
    }

    let _ = fs::create_dir_all(dir);
}

pub fn performance(elapsed: f32, func_name: String) {
    tracing::info!(
        target: "PERFORMANCE",
        ms = elapsed,
        name = func_name,
        "Block finished"
    );
}

struct DumpRequest {
    samples: Vec<f32>,
    filename: String,
}
static DUMP_TX: OnceLock<mpsc::UnboundedSender<DumpRequest>> = OnceLock::new();
pub fn init_audio_dumper() {
    let (tx, mut rx) = mpsc::unbounded_channel::<DumpRequest>();
    let _ = DUMP_TX.set(tx);

    tokio::spawn(async move {
        let dir = "debug_audio";
        let _ = fs::create_dir_all(dir);
        let mut history: VecDeque<String> = VecDeque::new();
        const MAX_DUMP_FILES: usize = 20;

        tracing::info!("Audio dumper task started");

        while let Some(req) = rx.recv().await {
            let file_path = Path::new(dir).join(&req.filename);
            let file_path_str = file_path.to_string_lossy().into_owned();

            history.retain(|x| x != &file_path_str);
            if history.len() >= MAX_DUMP_FILES
                && let Some(old_file) = history.pop_front() {
                    let _ = fs::remove_file(old_file);
                }
            history.push_back(file_path_str.clone());

            let spec = hound::WavSpec {
                channels: 1,
                sample_rate: 16000,
                bits_per_sample: 32,
                sample_format: hound::SampleFormat::Float,
            };

            let _ = tokio::task::spawn_blocking(move || {
                if let Ok(mut writer) = hound::WavWriter::create(file_path, spec) {
                    for &sample in &req.samples {
                        let _ = writer.write_sample(sample);
                    }
                    let _ = writer.finalize();
                }
            })
            .await;
        }
    });
}

pub fn dump_audio_to_file(samples: &[f32], filename: &str) {
    if !config::dump_audio() {
        return;
    }

    if let Some(tx) = DUMP_TX.get() {
        let _ = tx.send(DumpRequest {
            samples: samples.to_vec(),
            filename: filename.to_string(),
        });
    }
}

pub async fn recording_task(
    mut rx: mpsc::Receiver<String>,
    path: PathBuf,
    should_save: std::sync::Arc<std::sync::atomic::AtomicBool>,
) {
    let mut count = 0;
    let mut file: Option<File> = None;
    tracing::info!("Path is: {:?}", path);
    while let Some(text) = rx.recv().await {
        if should_save.load(std::sync::atomic::Ordering::Relaxed) {
            if file.is_none() {
                if let Some(parent) = Path::new(&path).parent() {
                    let _ = create_dir_all(parent).await;
                }
                let f = OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(&path)
                    .await
                    .expect("Failed to open transcription file");
                file = Some(f);
            }
            if let Some(ref mut f) = file {
                count += 1;
                let suffix = if count % 4 == 0 { ".\n" } else { " " };
                let output = format!("{}{}", text, suffix);
                if let Err(e) = f.write_all(output.as_bytes()).await {
                    eprintln!("Write error: {}", e);
                }
                let _ = f.flush().await;
            }
        } else {
            file = None;
        }
    }
}

pub fn add_to_custom_dict(word: &str, translation: &str) {
    let dict_dir = get_model_path("dictionary");
    let _ = std::fs::create_dir_all(&dict_dir);
    let custom_path = dict_dir.join("custom.toml");

    let content = std::fs::read_to_string(&custom_path).unwrap_or_else(|_| "[rules]\n".to_string());
    let mut new_content = content.trim_end().to_string();

    if !new_content.contains("[rules]") {
        new_content.push_str("\n[rules]");
    }

    new_content.push_str(&format!("\n\"{}\" = \"{}\"\n", word, translation));

    if let Err(e) = std::fs::write(&custom_path, new_content) {
        tracing::error!("Failed to write to custom dict: {}", e);
    } else {
        tracing::info!("Added '{}' -> '{}' to custom.toml", word, translation);
    }
}

pub enum TestState {
    Idle,
    Running(std::sync::mpsc::Receiver<Result<String, String>>),
    Done(Result<String, String>),
}

pub enum ModelType {
    VAD,
    WFast,
    WAcc,
}

pub struct ModelInfo {
    pub name: &'static str,
    pub url: &'static str,
    pub path: PathBuf,
    pub size_str: &'static str,
}

impl ModelInfo {
    pub fn get(t: &ModelType) -> ModelInfo {
        match t {
            ModelType::VAD => {
                let dir = "models/vad";
                Self {
                    name: "Silero VAD",
                    url: "https://github.com/snakers4/silero-vad/raw/v4.0/files/silero_vad.onnx",
                    path: find_first_file_in_dir(dir, "onnx")
                        .unwrap_or_else(|| get_model_path(dir).join("silero_vad.onnx")),
                    size_str: "~2 MB",
                }
            }
            ModelType::WFast => {
                let dir = "models/whisper-fast";
                Self {
                    name: "Whisper Fast (tiny)",
                    url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny-q8_0.bin",
                    path: find_first_file_in_dir(dir, "bin")
                        .unwrap_or_else(|| get_model_path(dir).join("ggml-tiny-q8_0.bin")),
                    size_str: "~43 MB",
                }
            }
            ModelType::WAcc => {
                let dir = "models/whisper-accurate";
                Self {
                    name: "Whisper Accurate (small)",
                    url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small-q5_1.bin",
                    path: find_first_file_in_dir(dir, "bin")
                        .unwrap_or_else(|| get_model_path(dir).join("ggml-small-q5_1.bin")),
                    size_str: "~190 MB",
                }
            }
        }
    }
}

pub fn start_test() -> std::sync::mpsc::Receiver<Result<String, String>> {
    let (tx, rx) = std::sync::mpsc::channel();

    std::thread::Builder::new()
        .name("pipeline-test".into())
        .spawn(move || {
            let current_exe = std::env::current_exe().expect("Failed to get self path");

            let output = Command::new(current_exe)
                .env("RUN_PIPELINE_TEST", "1")
                .stdout(std::process::Stdio::piped())
                .stderr(std::process::Stdio::piped())
                .output();

            match output {
                Ok(out) => {
                    let stdout = String::from_utf8_lossy(&out.stdout).trim().to_string();
                    let stderr = String::from_utf8_lossy(&out.stderr).trim().to_string();

                    if out.status.success() {
                        if let Some(res) = stdout.strip_prefix("TEST_OK:") {
                            let _ = tx.send(Ok(res.to_string()));
                        } else {
                            let _ = tx.send(Ok(stdout));
                        }
                    } else {
                        let error_msg = if let Some(panic_idx) = stderr.find("panic") {
                            stderr[panic_idx..].to_string()
                        } else if let Some(last_line) = stderr.lines().last() {
                            format!("Process exited with error. Last log: {}", last_line)
                        } else {
                            "Unknown process crash".to_string()
                        };

                        let _ = tx.send(Err(error_msg));
                    }
                }
                Err(e) => {
                    let _ = tx.send(Err(format!("Failed to spawn test: {}", e)));
                }
            }
        })
        .expect("Failed to spawn test thread");

    rx
}

// Mini runtime testing before starting the app.
pub async fn run_test_inner() -> Result<String, String> {
    static TEST_WAV: &[u8] = include_bytes!("./assets/test.wav");
    let samples = decode_wav(TEST_WAV)?;

    let vad_path = find_first_file_in_dir("models/vad", "onnx").ok_or("VAD model not found")?;
    let mut vad = crate::vad::VadEngine::new(&vad_path);
    let mut chunks: Vec<crate::types::PhraseChunk> = Vec::new();

    for window in samples.chunks(crate::config::VAD_CHUNK_SIZE) {
        if window.len() == crate::config::VAD_CHUNK_SIZE {
            vad.process(window.to_vec(), &mut chunks);
        }
    }

    if chunks.is_empty() {
        return Err("VAD produced no chunks — no speech detected".into());
    }

    use crate::types::TranscriptEvent;
    use std::sync::Arc;
    use tokio::sync::{mpsc, oneshot};

    let (pass1_tx, pass1_rx) = mpsc::channel(32);
    let (pass2_tx, pass2_rx) = mpsc::channel(32);
    let (event_tx1, mut event_rx1) = mpsc::channel::<TranscriptEvent>(32);
    let (event_tx2, mut event_rx2) = mpsc::channel::<TranscriptEvent>(32);
    let (ready_tx1, ready_rx1) = oneshot::channel();
    let (ready_tx2, ready_rx2) = oneshot::channel();

    std::thread::Builder::new()
        .name("test-whisper-pass1".into())
        .spawn(move || {
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap();
            rt.block_on(crate::whisper::whisper::pass1_task(
                ready_tx1, pass1_rx, event_tx1,
            ));
        })
        .map_err(|e| format!("Failed to spawn pass1 thread: {}", e))?;

    std::thread::Builder::new()
        .name("test-whisper-pass2".into())
        .spawn(move || {
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap();
            rt.block_on(crate::whisper::whisper::pass2_task(
                ready_tx2, pass2_rx, event_tx2,
            ));
        })
        .map_err(|e| format!("Failed to spawn pass2 thread: {}", e))?;

    ready_rx1.await.map_err(|_| "pass1 failed to signal ready".to_string())?;
    ready_rx2.await.map_err(|_| "pass2 failed to signal ready".to_string())?;

    for chunk in chunks {
        let is_last = chunk.is_last;
        pass1_tx.send(chunk).await.map_err(|_| "pass1 channel closed".to_string())?;
        if !is_last {
            tokio::time::sleep(tokio::time::Duration::from_millis(25)).await;
        }
    }

    let pass1_timeout = tokio::time::Duration::from_secs(30);
    let pass1_text = match tokio::time::timeout(pass1_timeout, event_rx1.recv()).await {
        Ok(Some(TranscriptEvent::Final { text, .. } | TranscriptEvent::Partial { text, .. })) => text,
        _ => String::new(),
    };

    let full_audio = Arc::new(samples);
    pass2_tx.send(crate::types::PhraseChunk {
        phrase_id: 0,
        chunk_id: 0,
        is_last: true,
        short: false,
        data: full_audio,
    }).await.map_err(|_| "pass2 channel closed".to_string())?;

    let pass2_timeout = tokio::time::Duration::from_secs(120);
    match tokio::time::timeout(pass2_timeout, event_rx2.recv()).await {
        Ok(Some(TranscriptEvent::Final { text, .. })) => {
            let clean = text.replace("[BLANK_AUDIO]", "").trim().to_string();
            if clean.is_empty() && !pass1_text.is_empty() {
                Ok(format!("[pass1 fallback] {}", pass1_text))
            } else if clean.is_empty() {
                Err("Both passes returned empty".into())
            } else {
                Ok(text)
            }
        }
        Ok(Some(TranscriptEvent::Partial { text, .. })) => Ok(format!("[partial] {}", text)),
        Ok(None) => Err("pass2 channel closed without result".into()),
        Err(_) => Err("pass2 timeout — accurate model took too long".into()),
    }
}

fn decode_wav(bytes: &[u8]) -> Result<Vec<f32>, String> {
    let cursor = std::io::Cursor::new(bytes);
    let mut reader =
        hound::WavReader::new(cursor).map_err(|e| format!("WAV decode error: {}", e))?;

    let samples: Vec<f32> = match reader.spec().sample_format {
        hound::SampleFormat::Float => reader.samples::<f32>().map(|s| s.unwrap_or(0.0)).collect(),
        hound::SampleFormat::Int => reader
            .samples::<i16>()
            .map(|s| s.unwrap_or(0) as f32 / 32768.0)
            .collect(),
    };

    let max_amp = samples.iter().cloned().map(f32::abs).fold(0.0f32, f32::max);
    let samples = if max_amp > 0.01 && max_amp < 0.5 {
        samples.iter().map(|s| s / max_amp * 0.7).collect()
    } else {
        samples
    };

    Ok(samples)
}
