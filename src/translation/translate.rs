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
        let relative_path = "dictionary/rules.toml";
        let dict_path = get_model_path(relative_path);
        tracing::info!("Resolved dictionary path: {:?}", dict_path);
        
        let initial_dict = Self::load_dict(&dict_path);
        let dictionary = Arc::new(RwLock::new(initial_dict));

        let translator = Self { 
            event, 
            send,
            dictionary
        };

        translator.spawn_dict_watcher(dict_path);
        translator
    }

    fn spawn_dict_watcher(&self, path: PathBuf) {
        let dict_clone = Arc::clone(&self.dictionary);
        
        std::thread::Builder::new()
            .name("dict-watcher".to_string())
            .spawn(move || {
                let mut last_mtime = None;
                loop {
                    if let Ok(meta) = fs::metadata(&path) {
                        let mtime = meta.modified().ok();
                        if mtime != last_mtime {
                            last_mtime = mtime;
                            let new_rules = Self::load_dict(&path);
                            if let Ok(mut w) = dict_clone.write() {
                                *w = new_rules;
                                tracing::info!("Translator: Dictionary reloaded from {:?}", path);
                            }
                        }
                    }
                    std::thread::sleep(std::time::Duration::from_secs(2));
                }
            })
            .expect("Failed to spawn dict-watcher thread");
    }

    fn load_dict(path: &Path) -> HashMap<String, String> {
        tracing::info!("Loading dictionary from: {:?}", path);
        
        let content = fs::read_to_string(path).unwrap_or_else(|e| {
            tracing::error!("Failed to read dictionary file {:?}: {}", path, e);
            String::new()
        });
        
        let dict: DictFile = toml::from_str(&content).unwrap_or_else(|e| {
            tracing::error!("Failed to parse TOML dictionary: {}", e);
            DictFile::default()
        });

        let rules_count = dict.rules.len();
        let processed = dict.rules.into_iter()
            .map(|(k, v)| (k.to_lowercase(), v))
            .collect();
            
        tracing::info!("Dictionary loaded. Rules count: {}", rules_count);
        processed
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