mod types;
mod audio;
mod display_task;
mod vad;
mod utils;
mod whisper;

use std::collections::BTreeMap;
use eframe::egui;
use tokio::sync::mpsc;
use tracing_subscriber::EnvFilter;
use crate::types::{AudioPacket, PhraseChunk, TranscriptEvent};
use crate::utils::merge_strings;

pub const TARGET_SAMPLE_RATE: u32 = 16_000;
pub const VAD_CHUNK_SIZE:      usize = 480;
pub const MAX_SILENCE_CHUNKS:  usize = 12;
pub const STREAM_CHUNK_SAMPLES: usize = TARGET_SAMPLE_RATE as usize;
pub const MIN_PHRASE_SAMPLES:   usize = TARGET_SAMPLE_RATE as usize / 2;
pub const MAX_PHRASE_SAMPLES:   usize = TARGET_SAMPLE_RATE as usize * 12;
pub const PASS1_MIN_SAMPLES:    usize = TARGET_SAMPLE_RATE as usize * 1;
pub const FAST_TRACK_THRESHOLD_S: f32 = 3.0;
pub const MAX_WINDOW: usize = TARGET_SAMPLE_RATE as usize * 10;
pub const MIN_WINDOW: usize = TARGET_SAMPLE_RATE as usize * 4;
pub const PREROLL_CHUNKS: usize = 5;
pub const STITCH_MIN_SAMPLES: usize = (TARGET_SAMPLE_RATE as f32 * 1.5) as usize;
pub const STITCH_MAX_SILENCE: f32 = 1.2;  
pub const VAD_CHUNK_DURATION_S: f32 = VAD_CHUNK_SIZE as f32 / TARGET_SAMPLE_RATE as f32;
pub const STITCH_MAX_CHUNKS: usize = (STITCH_MAX_SILENCE / VAD_CHUNK_DURATION_S) as usize;
pub const LANGUAGE: &str = "en";
pub const USE_GPU_ACC: bool = true;
pub const USE_GPU_FAST: bool = true;


fn init_logging() {
    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("info"));
 
    #[cfg(tokio_unstable)]
    {
        use tracing_subscriber::prelude::*;
        tracing_subscriber::registry()
            .with(console_subscriber::spawn())
            .with(tracing_subscriber::fmt::layer().with_filter(filter))
            .init();
    }
 
    #[cfg(not(tokio_unstable))]
    {
        tracing_subscriber::fmt()
            .with_env_filter(filter)
            .init();
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    init_logging();
    whisper::disable_whisper_log();
    
    let (audio_tx, audio_rx) = mpsc::channel::<AudioPacket>(1000);
    let (pass1_tx, pass1_rx) = mpsc::channel::<PhraseChunk>(32);
    let (pass2_tx, pass2_rx) = mpsc::channel::<PhraseChunk>(32);
    let (event_tx, event_rx) = mpsc::channel::<TranscriptEvent>(64);

    tokio::spawn(vad::vad_task(audio_rx, pass1_tx, pass2_tx));
    tokio::time::sleep(std::time::Duration::from_millis(500)).await;

    tokio::spawn(whisper::pass1_task(pass1_rx, event_tx.clone()));
    tokio::time::sleep(std::time::Duration::from_secs(2)).await;

    tokio::spawn(whisper::pass2_task(pass2_rx, event_tx.clone()));
    tokio::time::sleep(std::time::Duration::from_secs(1)).await;
    //tokio::spawn(display_task::display_task(event_rx));

    let _stream = audio::start_listening(audio_tx)?;

    if std::env::args().any(|arg| arg == "--bench") {
        tracing::info!("Benchmark mode: models loaded, exiting.");
        return Ok(()); 
    }

    let native_options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([600.0, 400.0])
            .with_title("Arch Translator"),
        ..Default::default()
    };

    eframe::run_native(
        "translator_ui",
        native_options,
        Box::new(|_cc| Ok(Box::new(App::new(event_rx)))),
    ).map_err(|e| format!("eframe error: {}", e))?;
    
    tokio::signal::ctrl_c().await?;
    Ok(())
}

struct PhraseData {
    text: String,
    is_final: bool,
    duration_s: f32,
    rtf: f32,
}

struct App {
    event_rx: mpsc::Receiver<TranscriptEvent>,
    transcription: BTreeMap<u32, PhraseData>,
}
impl App {
    pub fn new(event_rx: mpsc::Receiver<TranscriptEvent>) -> Self {
        Self {
            event_rx,
            transcription: BTreeMap::new(),
        }
    }

    pub fn update_data(&mut self) {
        while let Ok(event) = self.event_rx.try_recv() {
            match event {
                TranscriptEvent::Partial { phrase_id, text, .. } => {
                    let entry = self.transcription.entry(phrase_id).or_insert(PhraseData {
                        text: String::new(),
                        is_final: false,
                        duration_s: 0.0,
                        rtf: 0.0,
                    });
                    if !entry.is_final {
                        entry.text = merge_strings(&entry.text, &text);
                    }
                }
                TranscriptEvent::Final { phrase_id, text, duration_s, rtf } => {
                    self.transcription.insert(phrase_id, PhraseData {
                        text,
                        is_final: true,
                        duration_s,
                        rtf,
                    });
                }
            }
        }
    }
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        ctx.set_zoom_factor(1.5);
        self.update_data();

        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading(":) Live Transcription");
            ui.separator();

            egui::ScrollArea::vertical().stick_to_bottom(true).show(ui, |ui| {
                for (id, data) in &self.transcription {
                    ui.horizontal(|ui| {
                        ui.weak(format!("[{id}]"));

                        if data.is_final {
                            ui.colored_label(egui::Color32::LIGHT_GREEN, &data.text);
                            
                            ui.weak(format!("({:.1}s | RTF: {:.2})", data.duration_s, data.rtf));
                        } else {
                            ui.colored_label(egui::Color32::GRAY, format!("{} ⏳", data.text));
                        }
                    });
                }
            });
        });
        ctx.request_repaint_after(std::time::Duration::from_millis(50));
    }
}