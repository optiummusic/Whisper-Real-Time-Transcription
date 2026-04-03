
// Architecture:
//   TranscriptEvent::Final  →  process_final_text
//     ngram scan (1..=MAX_NGRAM, greedy longest-match)
//       → LxdbDict::lookup(phrase)   O(log W) binary search + O(1) read
//     → TranslationEvent::Translate per token span
//

use crate::prelude::*;
use crate::config::{self, TRANSLATION_MUTED};
use crate::utility::utils::{get_base_dir, performance};
use lxdb::{load_file, LxdbError, LxdbReader};
use std::path::{Path, PathBuf};

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
    dicts:         Arc<parking_lot::RwLock<Vec<LxdbDict>>>,
    unresolved_tx: tokio::sync::mpsc::UnboundedSender<String>,
}

impl Translator {
    pub fn new(
        event: mpsc::Receiver<TranscriptEvent>,
        send:  mpsc::Sender<TranslationEvent>,
    ) -> Self {
        let dict_dir = Self::dict_dir();
        tracing::info!("LXDB directory: {:?}", dict_dir);

        let initial = Self::load_all_dicts(&dict_dir, &config::source_lang(), &config::target_lang());
        let dicts   = Arc::new(parking_lot::RwLock::new(initial));

        let (unresolved_tx, unresolved_rx) = tokio::sync::mpsc::unbounded_channel();
        Self::spawn_unresolved_worker(unresolved_rx);

        let t = Self { event, send, dicts: Arc::clone(&dicts), unresolved_tx };
        t.spawn_dict_watcher(dict_dir);
        t
    }

    fn dict_dir() -> PathBuf {
        get_base_dir().join("dictionary")
    }

    fn load_all_dicts(dir: &Path, src: &str, tgt: &str) -> Vec<LxdbDict> {
        let mut loaded = Vec::new();
        
        let Ok(entries) = std::fs::read_dir(dir) else {
            tracing::warn!("Could not read dictionary directory: {:?}", dir);
            return loaded;
        };

        let mut paths: Vec<_> = entries.filter_map(|e| e.ok()).map(|e| e.path()).collect();
        paths.sort_by(|a, b| {
            let a_is_main = a.file_name().map_or(false, |n| n == "main.lxdb");
            let b_is_main = b.file_name().map_or(false, |n| n == "main.lxdb");
            if a_is_main { std::cmp::Ordering::Less }
            else if b_is_main { std::cmp::Ordering::Greater }
            else { a.cmp(b) }
        });

        for path in paths {
            if path.extension().map_or(false, |ext| ext == "lxdb") {
                if let Ok(d) = LxdbDict::open(&path, src, tgt) {
                    loaded.push(d);
                }
            }
        }
        loaded
    }

    // ── Hot-reload watcher ────────────────────────────────────────────────────

    fn spawn_dict_watcher(&self, dir: PathBuf) {
        let dicts_ptr = Arc::clone(&self.dicts);
        std::thread::Builder::new()
            .name("lxdb-watcher".into())
            .spawn(move || {
                let mut last_version = config::get_config_version();
                let mut last_scan_time: Option<std::time::SystemTime> = None;
                loop {
                    std::thread::sleep(std::time::Duration::from_secs(3));

                    let current_version = config::get_config_version();
                    let current_mtime = std::fs::metadata(&dir).ok().and_then(|m| m.modified().ok());
                    
                    if current_version != last_version || current_mtime != last_scan_time {
                        last_version = current_version;
                        last_scan_time = current_mtime;

                        let src = config::source_lang();
                        let tgt = config::target_lang();

                        let fresh = Self::load_all_dicts(&dir, &src, &tgt);
                        if !fresh.is_empty() {
                            *dicts_ptr.write() = fresh;
                            tracing::info!("LXDB Buffer reloaded. Total dicts: {}", dicts_ptr.read().len());
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

        let cleaned_words: Vec<String> = raw_words.iter().map(|w| Self::clean(w)).collect();
        let ui_notify = buffer.register(phrase_id).await;
        
        let dicts_snap = self.dicts.read().clone();
        
        let mut out: Vec<TranslationEvent> = Vec::with_capacity(raw_words.len());
        let mut i = 0;

        while i < raw_words.len() {
            if cleaned_words[i].is_empty() {
                out.push(TranslationEvent::Translate { phrase_id, word_index: i, span: 1, text: String::new() });
                i += 1;
                continue;
            }

            let (translated_text, span) = self.find_best_ngram(&dicts_snap, &cleaned_words, i);
            
            if translated_text.is_empty() {
                self.record_miss(&cleaned_words[i]);
            }

            out.push(TranslationEvent::Translate { phrase_id, word_index: i, span, text: translated_text });
            i += span;
        }

        for evt in out { let _ = self.send.send(evt).await; }
        ui_notify.notify_one();

        performance(t0.elapsed().as_secs_f32() * 1000.0, format!("translate_phrase_{phrase_id}"));
    }

    fn find_best_ngram(&self, dicts: &[LxdbDict], cleaned_words: &[String], start_idx: usize) -> (String, usize) {
        if dicts.is_empty() { return (String::new(), 1); }

        for window in (1..=MAX_NGRAM).rev() {
            if start_idx + window > cleaned_words.len() { continue; }

            let slice = &cleaned_words[start_idx..start_idx + window];
            if slice.iter().any(|s| s.is_empty()) { continue; }
            let phrase = slice.join(" ");
            
            for (idx, dict) in dicts.iter().enumerate() {
                if let Some(translation) = dict.lookup(&phrase) {
                    tracing::debug!(
                        "HIT: '{}' -> '{}' (span: {}, dict_idx: {})", 
                        phrase, translation, window, idx
                    );
                    return (translation, window);
                }
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