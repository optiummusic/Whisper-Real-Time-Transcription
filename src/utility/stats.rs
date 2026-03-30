use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};
use std::collections::VecDeque;
use std::time::Instant;

const MAX_PIPELINE_EVENTS: usize = 80;

#[derive(Clone, Debug)]
pub enum PipelineEventKind {
    Pass1Start,
    Pass1End { rtf: f32, text_len: usize },
    Pass2Start,
    Pass2End  { rtf: f32, text_len: usize },
}

#[derive(Clone, Debug)]
pub struct PipelineEvent {
    pub phrase_id: u32,
    pub kind:      PipelineEventKind,
    pub at:        Instant,
}

#[derive(Default)]
pub struct PerfStats {
    // Drop / overflow counters
    pub audio_overflow:    AtomicU64,
    pub vad_pass1_dropped: AtomicU64,
    pub vad_pass2_dropped: AtomicU64,
    pub pass1_dropped:     AtomicU64,
    pub pass2_dropped:     AtomicU64,
 
    // Phrase counts
    pub phrases_p1: AtomicU64,
    pub phrases_p2: AtomicU64,
 
    // RTF accumulators (guarded by Mutex; rarely contested)
    pub pass1_rtf_acc: Mutex<f64>,
    pub pass1_count: AtomicU64,
    pub pass2_rtf_acc: Mutex<f64>,
    pub pass2_count: AtomicU64,
 
    // Audio duration accumulator
    pub audio_dur_acc:   Mutex<f64>,
    pub audio_dur_count: AtomicU64,
 
    // VAD live state
    pub is_speaking: AtomicBool,
 
    // Pipeline event log (ring buffer)
    pub pipeline_events: Mutex<VecDeque<PipelineEvent>>,
}

impl PerfStats {
    fn new() -> Self {
        Self {
            pipeline_events: Mutex::new(VecDeque::with_capacity(MAX_PIPELINE_EVENTS)),
            ..Default::default()
        } 
    }

    pub fn record_pass1_start(&self, phrase_id: u32) {
        self.push_event(PipelineEvent {
            phrase_id,
            kind: PipelineEventKind::Pass1Start,
            at:   Instant::now(),
        });
    }

    pub fn record_pass1_done(&self, phrase_id: u32, rtf: f32, audio_dur_s: f32, text_len: usize) {
        if let Ok(mut acc) = self.pass1_rtf_acc.lock() { *acc += rtf as f64; }
        self.pass1_count.fetch_add(1, Ordering::Relaxed);
        if let Ok(mut acc) = self.audio_dur_acc.lock()  { *acc += audio_dur_s as f64; }
        self.audio_dur_count.fetch_add(1, Ordering::Relaxed);
        self.phrases_p1.fetch_add(1, Ordering::Relaxed);
        self.push_event(PipelineEvent {
            phrase_id,
            kind: PipelineEventKind::Pass1End { rtf, text_len },
            at:   Instant::now(),
        });
    }

    pub fn record_pass2_start(&self, phrase_id: u32) {
        self.push_event(PipelineEvent {
            phrase_id,
            kind: PipelineEventKind::Pass2Start,
            at:   Instant::now(),
        });
    }
 
    pub fn record_pass2_done(&self, phrase_id: u32, rtf: f32, text_len: usize) {
        if let Ok(mut acc) = self.pass2_rtf_acc.lock() { *acc += rtf as f64; }
        self.pass2_count.fetch_add(1, Ordering::Relaxed);
        self.phrases_p2.fetch_add(1, Ordering::Relaxed);
        self.push_event(PipelineEvent {
            phrase_id,
            kind: PipelineEventKind::Pass2End { rtf, text_len },
            at:   Instant::now(),
        });
    }

    pub fn avg_pass1_rtf(&self) -> f32 {
        let n = self.pass1_count.load(Ordering::Relaxed);
        if n == 0 { return 0.0; }
        self.pass1_rtf_acc.lock().map(|a| (*a / n as f64) as f32).unwrap_or(0.0)
    }
 
    pub fn avg_pass2_rtf(&self) -> f32 {
        let n = self.pass2_count.load(Ordering::Relaxed);
        if n == 0 { return 0.0; }
        self.pass2_rtf_acc.lock().map(|a| (*a / n as f64) as f32).unwrap_or(0.0)
    }
 
    pub fn avg_audio_dur_s(&self) -> f32 {
        let n = self.audio_dur_count.load(Ordering::Relaxed);
        if n == 0 { return 0.0; }
        self.audio_dur_acc.lock().map(|a| (*a / n as f64) as f32).unwrap_or(0.0)
    }

    fn push_event(&self, ev: PipelineEvent) {
        if let Ok(mut q) = self.pipeline_events.lock() {
            if q.len() >= MAX_PIPELINE_EVENTS { q.pop_front(); }
            q.push_back(ev);
        }
    }
 
    /// Snapshot of the event ring for the UI (cheap clone).
    pub fn events_snapshot(&self) -> Vec<PipelineEvent> {
        self.pipeline_events.lock()
            .map(|q| q.iter().cloned().collect())
            .unwrap_or_default()
    }

    pub fn reset(&self) {
        self.audio_overflow.store(0, Ordering::Relaxed);
        self.vad_pass1_dropped.store(0, Ordering::Relaxed);
        self.vad_pass2_dropped.store(0, Ordering::Relaxed);
        self.pass1_dropped.store(0, Ordering::Relaxed);
        self.pass2_dropped.store(0, Ordering::Relaxed);
        self.phrases_p1.store(0, Ordering::Relaxed);
        self.phrases_p2.store(0, Ordering::Relaxed);
        self.pass1_count.store(0, Ordering::Relaxed);
        self.pass2_count.store(0, Ordering::Relaxed);
        self.audio_dur_count.store(0, Ordering::Relaxed);
        if let Ok(mut a) = self.pass1_rtf_acc.lock()  { *a = 0.0; }
        if let Ok(mut a) = self.pass2_rtf_acc.lock()  { *a = 0.0; }
        if let Ok(mut a) = self.audio_dur_acc.lock()   { *a = 0.0; }
        if let Ok(mut q) = self.pipeline_events.lock() { q.clear(); }
    }
}

static STATS: OnceLock<PerfStats> = OnceLock::new();
 
pub fn get() -> &'static PerfStats {
    STATS.get_or_init(PerfStats::new)
}