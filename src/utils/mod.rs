use std::collections::VecDeque;
use std::path::{ Path, PathBuf };
use std::fs;
use std::sync::OnceLock;
use crate::config;
use tokio::sync::{ mpsc };
use tokio::io::AsyncWriteExt;

pub fn append_context(ctx: &mut String, text: &str, max_words: usize) {
    if text.is_empty() { return; }
 
    if !ctx.is_empty() { ctx.push(' '); }
    ctx.push_str(text.trim());
 
    let words: Vec<&str> = ctx.split_whitespace().collect();
    if words.len() > max_words {
        *ctx = words[words.len() - max_words..].join(" ");
    }
}

pub fn merge_strings(old: &str, new: &str) -> String {
    if old.is_empty() { return new.to_string(); }
    if new.is_empty() { return old.to_string(); }
    if new.contains(old.trim()) { return new.to_string(); }
 
    let old_words: Vec<&str> = old.split_whitespace().collect();
    let new_words: Vec<&str> = new.split_whitespace().collect();
 
    let max_overlap = old_words.len().min(new_words.len());
 
    for overlap in (1..=max_overlap).rev() {
        let old_suffix = &old_words[old_words.len() - overlap..];
        let new_prefix = &new_words[..overlap];
        if old_suffix == new_prefix {
            let mut result = old_words[..old_words.len() - overlap].to_vec();
            result.extend_from_slice(&new_words);
            return result.join(" ");
        }
    }
    new.to_string()
}

pub fn get_model_path(relative_path: &str) -> PathBuf {
    let exe_path = std::env::current_exe().expect("Failed to get current exe path");
    let exe_dir = exe_path.parent().expect("Failed to get exe parent");

    let path_near_exe = exe_dir.join(relative_path);
    if path_near_exe.exists() { return path_near_exe; }

    if let Some(project_root) = exe_dir.parent().and_then(|p| p.parent()) {
        let path_in_root = project_root.join(relative_path);
        if path_in_root.exists() { return path_in_root; }
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
    if !config::dump_audio() { return; }
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
            if history.len() >= MAX_DUMP_FILES {
                if let Some(old_file) = history.pop_front() {
                    let _ = fs::remove_file(old_file);
                }
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
            }).await;
        }
    });
}

pub fn dump_audio_to_file(samples: &[f32], filename: &str) {
    if !config::dump_audio() { return; }
    
    if let Some(tx) = DUMP_TX.get() {
        let _ = tx.send(DumpRequest {
            samples: samples.to_vec(),
            filename: filename.to_string(),
        });
    }
}

pub async fn recording_task(
    mut rx: mpsc::Receiver<String>, 
    path: String, 
    should_save: std::sync::Arc<std::sync::atomic::AtomicBool>
) {
    let _ = tokio::fs::create_dir_all("transcriptions").await;
    let mut file = tokio::fs::OpenOptions::new()
        .create(true).append(true).open(&path).await
        .expect("Failed to open transcription file");

    let mut count = 0;
    while let Some(text) = rx.recv().await {
        if should_save.load(std::sync::atomic::Ordering::Relaxed) {
            count += 1;
            let suffix = if count % 4 == 0 { ".\n" } else { " " };
            let _ = file.write_all(format!("{}{}", text, suffix).as_bytes()).await;
            let _ = file.flush().await; 
        }
    }
}

pub fn add_to_custom_dict(word: &str, translation: &str) {
    let dict_dir = crate::utils::get_model_path("dictionary");
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