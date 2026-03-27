use std::collections::HashMap;
use std::fs;
use std::sync::{Arc, RwLock};
use crate::utils::get_model_path;
use std::path::{Path, PathBuf};
use serde::Deserialize;
use tokio::sync::mpsc;
use crate::types::{TranscriptEvent, TranslationEvent, TranslationBuffer};

#[derive(Deserialize, Default)]
struct DictFile {
    #[serde(default)]
    rules: HashMap<String, String>,
}

pub struct Translator {
    event: mpsc::Receiver<TranscriptEvent>,
    send: mpsc::Sender<TranslationEvent>,
    dictionary: Arc<RwLock<HashMap<String, String>>>,
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

        let translator = Self { 
            event, 
            send,
            dictionary
        };

        translator.spawn_dict_watcher(dict_dir);
        translator
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
                            if entry.path().extension().and_then(|s| s.to_str()) == Some("toml") {
                                if let Ok(meta) = entry.metadata() {
                                    if let Ok(mtime) = meta.modified() {
                                        current_max_mtime = Some(current_max_mtime.unwrap_or(mtime).max(mtime));
                                    }
                                }
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
                            if !combined_rules.contains_key(&val) {
                                combined_rules.insert(val, k); 
                            }
                        }
                    }
                }
            }
        }
            
        tracing::info!("Dictionaries loaded. Unique bidirectional rules: {}", combined_rules.len());
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
                TranscriptEvent::Final { phrase_id, text, .. } => {
                    tracing::debug!(pid = phrase_id, "Processing final transcript: '{}'", text);
                    self.process_final_text(phrase_id, &text, Arc::clone(&buffer)).await;
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
            let mut consumed = 0;
            let mut translated_text = String::new();

            for window_size in (1..=max_ngram).rev() {
                if i + window_size <= words.len() {
                    
                    let mut candidate_words = Vec::with_capacity(window_size);
                    for j in 0..window_size {
                        let cleaned = Self::clean_word(words[i + j]);
                        if cleaned.is_empty() {
                            break;
                        }
                        candidate_words.push(cleaned);
                    }

                    if candidate_words.len() == window_size {
                        let candidate_phrase = candidate_words.join(" ");
                        
                        if let Some(val) = self.lookup(&candidate_phrase) {
                            tracing::info!(pid = phrase_id, "Match found for {}-gram: '{}'", window_size, candidate_phrase);
                            translated_text = val;
                            consumed = window_size;
                            break;
                        }
                    }
                }
            }

            if consumed == 0 {
                consumed = 1;
                let w = Self::clean_word(words[i]);
                if !w.is_empty() {
                    translated_text = words[i].to_uppercase(); 
                    //tracing::trace!(pid = phrase_id, "No rule for '{}', using fallback", w);
                } else {
                    translated_text = words[i].to_string(); 
                }
            }

            if consumed > 1 || Self::is_valid_word(&translated_text) {
                //tracing::trace!(pid = phrase_id, "Sending translation: '{}'", translated_text);
                
                translated.push(TranslationEvent::Translate {
                    phrase_id,
                    word_index: i,
                    span: consumed,
                    text: translated_text,
                    });
            } else {
                //tracing::debug!(pid = phrase_id, "Word '{}' filtered out", translated_text);
            }
            i += consumed;
        }
        ui_notify.notified().await;
        for evt in translated {
            let _ = self.send.send(evt).await;
        }

        let elapsed_ms = t0.elapsed().as_secs_f32() * 1000.0;
        crate::utils::performance(elapsed_ms, format!("translate_phrase_{}", phrase_id));
    }

    fn clean_word(word: &str) -> String {
        word.trim_matches(|c: char| !c.is_alphanumeric()).to_lowercase()
    }

    fn is_valid_word(word: &str) -> bool {
        word.chars().filter(|c| c.is_alphabetic()).count() >= 2
    }
}