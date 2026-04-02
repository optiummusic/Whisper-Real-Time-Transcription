use std::sync::Arc;
pub type AudioPacket = Vec<f32>;
use std::collections::HashMap;
use tokio::sync::{Mutex, Notify, mpsc, oneshot};

#[derive(Debug, Clone)]
pub enum TranscriptEvent {
    Partial {
        phrase_id: u32,
        chunk_id: u32,
        text: String,
        sent_at: std::time::Instant,
    },
    Final {
        phrase_id: u32,
        text: String,
        duration_s: f32,
        rtf: f32,
        sent_at: std::time::Instant,
    },
}

#[derive(Clone)]
pub enum TranslationEvent {
    Translate {
        phrase_id: u32,
        word_index: usize,
        span: usize,
        text: String,
    },
}

pub struct TranslationBuffer {
    notifiers: Mutex<HashMap<u32, Arc<Notify>>>,
}

impl TranslationBuffer {
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            notifiers: Mutex::new(HashMap::new()),
        })
    }

    pub async fn register(&self, phrase_id: u32) -> Arc<Notify> {
        let notify = Arc::new(Notify::new());
        self.notifiers
            .lock()
            .await
            .insert(phrase_id, Arc::clone(&notify));
        notify
    }

    pub async fn signal_ready(&self, phrase_id: u32) {
        if let Some(notify) = self.notifiers.lock().await.remove(&phrase_id) {
            notify.notify_one();
        }
    }
}

#[derive(Clone)]
pub struct PhraseChunk {
    pub phrase_id: u32,
    pub chunk_id: u32,
    pub is_last: bool,
    pub short: bool,
    pub data: Arc<Vec<f32>>,
}

pub struct BackendArgs {
    pub startup_rx: oneshot::Receiver<()>,
    pub device_rx: oneshot::Receiver<String>,
    pub event_tx: mpsc::Sender<TranscriptEvent>,
    pub event_rx_main: mpsc::Receiver<TranscriptEvent>,
    pub event_tx_ui: mpsc::Sender<TranscriptEvent>,
    pub event_tx_translator: mpsc::Sender<TranscriptEvent>,
    pub event_rx_translator: mpsc::Receiver<TranscriptEvent>,
    pub translation_tx: mpsc::Sender<TranslationEvent>,
    pub translation_buffer: Arc<TranslationBuffer>,
}

pub struct AppArgs {
    pub event_rx: mpsc::Receiver<TranscriptEvent>,
    pub translation_rx: mpsc::Receiver<TranslationEvent>,
    pub translation_buffer: Arc<TranslationBuffer>,
    pub device_tx: oneshot::Sender<String>,
    pub handle: tokio::runtime::Handle,
    pub startup_tx: oneshot::Sender<()>,
}