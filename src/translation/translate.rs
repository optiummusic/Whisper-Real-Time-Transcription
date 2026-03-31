use crate::types::{TranscriptEvent, TranslationBuffer, TranslationEvent};
use crate::utility::utils::{get_base_dir, get_model_path};
use serde::Deserialize;
use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};
use tokio::sync::mpsc;

#[derive(Deserialize, Default)]
struct DictFile {
    #[serde(default)]
    rules: HashMap<String, String>,
}

pub struct Translator {
    event: mpsc::Receiver<TranscriptEvent>,
    send: mpsc::Sender<TranslationEvent>,
    dictionary: Arc<RwLock<HashMap<String, String>>>,
    unresolved_tx: tokio::sync::mpsc::UnboundedSender<String>,
}

impl Translator {
    pub fn new(
        event: mpsc::Receiver<TranscriptEvent>,
        send: mpsc::Sender<TranslationEvent>,
    ) -> Self {
        let dict_dir = get_model_path("dictionary");
        tracing::info!("Resolved dictionary path: {:?}", dict_dir);

        let initial_dict = Self::load_dicts(&dict_dir);
        let dictionary = Arc::new(RwLock::new(initial_dict));

        let (unresolved_tx, unresolved_rx) = tokio::sync::mpsc::unbounded_channel();
        Self::spawn_unresolved_worker(unresolved_rx);

        let translator = Self {
            event,
            send,
            dictionary,
            unresolved_tx,
        };

        translator.spawn_dict_watcher(dict_dir);
        translator
    }

    fn spawn_unresolved_worker(mut rx: tokio::sync::mpsc::UnboundedReceiver<String>) {
        tokio::spawn(async move {
            let unresolved_dir = get_base_dir().join("dictionary").join("unresolved");
            let _ = tokio::fs::create_dir_all(&unresolved_dir).await;
            let file_path = unresolved_dir.join("unresolved.toml");

            let mut known_unresolved = HashSet::new();
            if let Ok(content) = tokio::fs::read_to_string(&file_path).await {
                for line in content.lines() {
                    if line.trim().starts_with('[') || line.trim().is_empty() {
                        continue;
                    }
                    if let Some(key) = line.split('=').next() {
                        let clean_key = key.trim().trim_matches('"');
                        known_unresolved.insert(clean_key.to_string());
                    }
                }
            }

            if !file_path.exists() {
                let _ = tokio::fs::write(&file_path, "[rules]\n").await;
            }

            let mut file = tokio::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(&file_path)
                .await
                .expect("Failed to open unresolved.toml");

            while let Some(word) = rx.recv().await {
                if known_unresolved.insert(word.clone()) {
                    use tokio::io::AsyncWriteExt;
                    let line = format!("\"{}\" = \"\"\n", word);
                    if let Err(e) = file.write_all(line.as_bytes()).await {
                        tracing::error!("Failed to write unresolved word: {}", e);
                    }
                    let _ = file.flush().await;
                }
            }
        });
    }

    fn spawn_dict_watcher(&self, dir_path: PathBuf) {
        let dict_clone = Arc::clone(&self.dictionary);

        std::thread::Builder::new()
            .name("dict-watcher".to_string())
            .spawn(move || {
                let mut last_mtime = None;
                loop {
                    let mut current_max_mtime = None;
                    if let Ok(entries) = fs::read_dir(&dir_path) {
                        for entry in entries.flatten() {
                            if entry.path().extension().and_then(|s| s.to_str()) == Some("toml")
                                && let Ok(meta) = entry.metadata()
                                    && let Ok(mtime) = meta.modified() {
                                        current_max_mtime =
                                            Some(current_max_mtime.unwrap_or(mtime).max(mtime));
                                    }
                        }
                    }

                    if current_max_mtime.is_some() && current_max_mtime != last_mtime {
                        last_mtime = current_max_mtime;
                        let new_rules = Self::load_dicts(&dir_path);
                        if let Ok(mut w) = dict_clone.write() {
                            *w = new_rules;
                            tracing::info!("Translator: Dictionaries reloaded from {:?}", dir_path);
                        }
                    }
                    std::thread::sleep(std::time::Duration::from_secs(2));
                }
            })
            .expect("Failed to spawn dict-watcher thread");
    }

    fn load_dicts(dir_path: &Path) -> HashMap<String, String> {
        tracing::info!("Loading dictionaries from: {:?}", dir_path);
        let mut combined_rules = HashMap::new();

        if let Ok(entries) = fs::read_dir(dir_path) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.extension().and_then(|s| s.to_str()) == Some("toml") {
                    let content = fs::read_to_string(&path).unwrap_or_default();
                    if let Ok(dict) = toml::from_str::<DictFile>(&content) {
                        for (k, v) in dict.rules {
                            let key = k.to_lowercase();
                            let val = v.to_lowercase();

                            combined_rules.insert(key.clone(), v.clone());
                            combined_rules.entry(val).or_insert(k);
                        }
                    }
                }
            }
        }

        tracing::info!(
            "Dictionaries loaded. Unique bidirectional rules: {}",
            combined_rules.len()
        );
        combined_rules
    }

    fn lookup(&self, key: &str) -> Option<String> {
        let dict = self.dictionary.read().unwrap();
        let result = dict.get(key).cloned();

        if let Some(ref val) = result {
            tracing::debug!("Lookup hit: '{}' -> '{}'", key, val);
        } else {
            //tracing::trace!("Lookup miss: '{}'", key);
        }

        result
    }

    pub async fn translate(mut self, buffer: Arc<TranslationBuffer>) {
        tracing::info!("Translator loop started");
        while let Some(evt) = self.event.recv().await {
            match evt {
                TranscriptEvent::Final {
                    phrase_id, text, ..
                } => {
                    if crate::config::TRANSLATION_MUTED.load(std::sync::atomic::Ordering::Relaxed) {
                        tracing::debug!(pid = phrase_id, "Translation muted — skipping phrase");
                        continue;
                    }
                    tracing::debug!(pid = phrase_id, "Processing final transcript: '{}'", text);
                    self.process_final_text(phrase_id, &text, Arc::clone(&buffer))
                        .await;
                }
                TranscriptEvent::Partial { .. } => {
                    continue;
                }
            }
        }
        tracing::warn!("Translator loop finished");
    }

    async fn process_final_text(&self, phrase_id: u32, text: &str, buffer: Arc<TranslationBuffer>) {
        let words: Vec<&str> = text.split_whitespace().collect();
        let max_ngram = 3;
        let mut i = 0;
        let t0 = std::time::Instant::now();
        
        let ui_notify = buffer.register(phrase_id).await;
        let mut translated: Vec<TranslationEvent> = Vec::new();

        while i < words.len() {
            let mut match_found = None;

            for window_size in (1..=max_ngram).rev() {
                if i + window_size <= words.len() {
                    let mut candidate_words = Vec::with_capacity(window_size);
                    for j in 0..window_size {
                        let cleaned = Self::clean_word(words[i + j]);
                        if !cleaned.is_empty() {
                            candidate_words.push(cleaned);
                        }
                    }

                    if candidate_words.len() == window_size {
                        let candidate_phrase = candidate_words.join(" ");
                        if let Some(val) = self.lookup(&candidate_phrase) {
                            match_found = Some((val, window_size));
                            break;
                        }
                    }
                }
            }

            let (translated_text, consumed) = match match_found {
                Some((val, n)) => (val, n),
                None => {
                    let (f_text, n) = self.fallback(Some(words[i]));
                    (f_text, n)
                }
            };

            translated.push(TranslationEvent::Translate {
                phrase_id,
                word_index: i,
                span: consumed,
                text: translated_text,
            });

            i += consumed;
        }

        for evt in translated {
            let _ = self.send.send(evt).await;
        }

        ui_notify.notify_one();

        let elapsed_ms = t0.elapsed().as_secs_f32() * 1000.0;
        crate::utility::utils::performance(elapsed_ms, format!("translate_phrase_{}", phrase_id));
    }
    
    fn fallback(&self, raw_word: Option<&str>) -> (String, usize) {
        if let Some(word) = raw_word {
            let cleaned = Self::clean_word(word);
            if !cleaned.is_empty() {
                let _ = self.unresolved_tx.send(cleaned);
            }
        }
        ("".to_string(), 1)
    }

    fn clean_word(word: &str) -> String {
        word.trim_matches(|c: char| !c.is_alphanumeric())
            .to_lowercase()
    }
}
