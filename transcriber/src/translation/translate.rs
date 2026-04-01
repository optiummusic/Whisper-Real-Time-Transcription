
// Architecture:
//   TranscriptEvent::Final  →  process_final_text
//     ngram scan (1..=MAX_NGRAM, greedy longest-match)
//       → LxdbDict::lookup(phrase)   O(log W) binary search + O(1) read
//     → TranslationEvent::Translate per token span
//


use crate::config::{self, TRANSLATION_MUTED};
use crate::types::{TranscriptEvent, TranslationBuffer, TranslationEvent};
use crate::utility::utils::{get_base_dir, performance};

use lxdb::{load_file, LxdbError, LxdbReader};
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::{atomic::Ordering, Arc};
use tokio::sync::mpsc;

const MAX_NGRAM: usize = 3;

// ── LxdbDict ──────────────────────────────────────────────────────────────────

#[derive(Clone)]
pub struct LxdbDict {
    data:    Arc<Vec<u8>>,
    src_id:  u16,
    tgt_id:  u16,
}

impl LxdbDict {
    fn open(path: &Path, src: &str, tgt: &str) -> Result<Self, LxdbError> {
        let data = load_file(path)?;
        let r    = LxdbReader::new(&data)?;

        let src_id = r.lang_id(src).ok_or_else(||
            LxdbError::Build(format!("source lang '{src}' not in dictionary")))?;
        let tgt_id = r.lang_id(tgt).ok_or_else(||
            LxdbError::Build(format!("target lang '{tgt}' not in dictionary")))?;

        tracing::info!(
            "LxdbDict: {:?} — {} concepts, {} entries, {src}→{tgt}",
            path, r.concept_count(), r.word_count(),
        );

        Ok(Self {
            data: Arc::new(data),
            src_id, tgt_id,
        })
    }

    #[inline]
    fn lookup(&self, word: &str) -> Option<String> {
        let r = LxdbReader::new(&self.data).ok()?;
        r.translate_by_id(word, self.src_id, self.tgt_id)
            .map(str::to_owned)
    }
}

// ── Translator ────────────────────────────────────────────────────────────────

pub struct Translator {
    event:         mpsc::Receiver<TranscriptEvent>,
    send:          mpsc::Sender<TranslationEvent>,
    dict:          Arc<parking_lot::RwLock<Option<LxdbDict>>>,
    unresolved_tx: tokio::sync::mpsc::UnboundedSender<String>,
}

impl Translator {
    pub fn new(
        event: mpsc::Receiver<TranscriptEvent>,
        send:  mpsc::Sender<TranslationEvent>,
    ) -> Self {
        let dict_path = Self::dict_path();
        tracing::info!("LXDB path: {:?}", dict_path);

        let initial = Self::try_load(&dict_path, "uk", "en");
        let dict    = Arc::new(parking_lot::RwLock::new(initial));

        let (unresolved_tx, unresolved_rx) = tokio::sync::mpsc::unbounded_channel();
        Self::spawn_unresolved_worker(unresolved_rx);

        let t = Self { event, send, dict: Arc::clone(&dict), unresolved_tx };
        t.spawn_dict_watcher(dict_path);
        t
    }

    fn dict_path() -> PathBuf {
        get_base_dir().join("dictionary").join("main.lxdb")
    }

    fn try_load(path: &Path, src: &str, tgt: &str) -> Option<LxdbDict> {
        match LxdbDict::open(path, src, tgt) {
            Ok(d)  => Some(d),
            Err(e) => { tracing::warn!("LxdbDict load failed: {e}"); None }
        }
    }

    // ── Hot-reload watcher ────────────────────────────────────────────────────

    fn spawn_dict_watcher(&self, path: PathBuf) {
        let dict = Arc::clone(&self.dict);
        std::thread::Builder::new()
            .name("lxdb-watcher".into())
            .spawn(move || {
                let mut last_mtime = None;
                let mut last_version = config::get_config_version();
                loop {
                    std::thread::sleep(std::time::Duration::from_secs(2));

                    let current_version = config::get_config_version();
                    let mtime = std::fs::metadata(&path).ok()
                        .and_then(|m| m.modified().ok());

                    if (mtime.is_some() && mtime != last_mtime) || current_version != last_version {
                        last_mtime = mtime;
                        last_version = current_version;
                        let src = config::source_lang();
                        let tgt = config::target_lang();

                        if let Some(fresh) = Self::try_load(&path, &src, &tgt) {
                            *dict.write() = Some(fresh);
                            tracing::info!("LXDB Hot-Reloaded: {} -> {}", src, tgt);
                        }
                    }
                }
            })
            .expect("failed to spawn lxdb-watcher");
    }

    // ── Unresolved logger ─────────────────────────────────────────────────────

    fn spawn_unresolved_worker(mut rx: tokio::sync::mpsc::UnboundedReceiver<String>) {
        tokio::spawn(async move {
            use tokio::io::AsyncWriteExt;

            let dir       = get_base_dir().join("dictionary").join("unresolved");
            let _         = tokio::fs::create_dir_all(&dir).await;
            let file_path = dir.join("unresolved.toml");

            let mut known: HashSet<String> = HashSet::new();
            if let Ok(content) = tokio::fs::read_to_string(&file_path).await {
                for line in content.lines() {
                    if let Some(k) = line.split('=').next() {
                        let k = k.trim().trim_matches('"').to_string();
                        if !k.is_empty() && !k.starts_with('[') { known.insert(k); }
                    }
                }
            }

            if !file_path.exists() {
                let _ = tokio::fs::write(&file_path, "[rules]\n").await;
            }

            let mut file = tokio::fs::OpenOptions::new()
                .create(true).append(true)
                .open(&file_path).await
                .expect("failed to open unresolved.toml");

            while let Some(word) = rx.recv().await {
                if known.insert(word.clone()) {
                    let line = format!("\"{}\" = \"\"\n", word);
                    if let Err(e) = file.write_all(line.as_bytes()).await {
                        tracing::error!("unresolved write: {e}");
                    }
                    let _ = file.flush().await;
                }
            }
        });
    }

    // ── Main translate loop ───────────────────────────────────────────────────

    pub async fn translate(mut self, buffer: Arc<TranslationBuffer>) {
        tracing::info!("Translator started (LXDB backend)");
        while let Some(evt) = self.event.recv().await {
            match evt {
                TranscriptEvent::Final { phrase_id, text, .. } => {
                    if TRANSLATION_MUTED.load(Ordering::Relaxed) { continue; }
                    tracing::debug!(pid = phrase_id, "final: '{}'", text);
                    self.process_final_text(phrase_id, &text, Arc::clone(&buffer)).await;
                }
                TranscriptEvent::Partial { .. } => continue,
            }
        }
        tracing::warn!("Translator loop ended");
    }

    async fn process_final_text(
        &self,
        phrase_id: u32,
        text:      &str,
        buffer:    Arc<TranslationBuffer>,
    ) {
        let t0 = std::time::Instant::now();
        
        let raw_words: Vec<&str> = text.split_whitespace().collect();
        if raw_words.is_empty() { return; }

        let cleaned_words: Vec<String> = raw_words.iter()
            .map(|w| Self::clean(w))
            .collect();

        tracing::debug!(
            pid = phrase_id, 
            word_count = raw_words.len(), 
            "Start translation: '{}'", text
        );

        let ui_notify = buffer.register(phrase_id).await;
        let mut out: Vec<TranslationEvent> = Vec::with_capacity(raw_words.len());

        let dict_snap = self.dict.read().clone();
        
        let mut i = 0;
        let mut hits = 0;
        let mut misses = 0;

        while i < raw_words.len() {
            if cleaned_words[i].is_empty() {
                out.push(TranslationEvent::Translate { 
                    phrase_id, word_index: i, span: 1, text: String::new() 
                });
                i += 1;
                continue;
            }

            let (translated_text, span) = self.find_best_ngram(&dict_snap, &cleaned_words, i);
            
            if !translated_text.is_empty() {
                hits += 1;
            } else {
                misses += 1;
                self.record_miss(&cleaned_words[i]);
            }

            out.push(TranslationEvent::Translate { 
                phrase_id, 
                word_index: i, 
                span, 
                text: translated_text 
            });
            
            i += span;
        }

        for evt in out { let _ = self.send.send(evt).await; }
        ui_notify.notify_one();

        let elapsed = t0.elapsed();
        tracing::info!(
            pid = phrase_id,
            "Finished translation in {:?}. Hits: {}, Misses: {}", 
            elapsed, hits, misses
        );
        performance(elapsed.as_secs_f32() * 1000.0, format!("translate_phrase_{phrase_id}"));
    }

    fn find_best_ngram(&self, dict: &Option<LxdbDict>, cleaned_words: &[String], start_idx: usize) -> (String, usize) {
        let Some(d) = dict else { return (String::new(), 1); };

        for window in (1..=MAX_NGRAM).rev() {
            if start_idx + window > cleaned_words.len() { continue; }

            let slice = &cleaned_words[start_idx..start_idx + window];
            
            if slice.iter().any(|s| s.is_empty()) { continue; }

            let phrase = slice.join(" ");
            
            tracing::trace!("Trying n-gram (len {}): '{}'", window, phrase);

            if let Some(translation) = d.lookup(&phrase) {
                tracing::debug!("HIT: '{}' -> '{}' (span: {})", phrase, translation, window);
                return (translation, window);
            }
        }

        (String::new(), 1)
    }

    fn record_miss(&self, cleaned_word: &str) {
        if !cleaned_word.is_empty() {
            let _ = self.unresolved_tx.send(cleaned_word.to_string());
        }
    }

    fn clean(w: &str) -> String {
        w.trim_matches(|c: char| !c.is_alphanumeric()).to_lowercase()
    }
}