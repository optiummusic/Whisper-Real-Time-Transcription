use std::collections::BTreeMap;
use eframe::egui;
use tokio::sync:: { oneshot, mpsc };
use tracing_subscriber::EnvFilter;
use mimalloc::MiMalloc;
use thread_priority::*;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

use translator::{
    whisper, vad, audio, utils, config,
    config::{ TARGET_SAMPLE_RATE },
    types::{AudioPacket, PhraseChunk, TranscriptEvent},
    PhraseData,
};
use crate::utils::merge_strings;

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
    tracing::info!("Logger initialized");

    config::load_from_toml("config.toml");
    tracing::info!("Config loaded: {:?}", config::startup());

    //whisper::disable_whisper_log();

    crate::utils::prepare_debug_dir();
    
    let (audio_tx, audio_rx) = mpsc::channel::<AudioPacket>(64);
    let (pass1_tx, pass1_rx) = mpsc::channel::<PhraseChunk>(32);
    let (pass2_tx, pass2_rx) = mpsc::channel::<PhraseChunk>(32);
    let (event_tx, event_rx) = mpsc::channel::<TranscriptEvent>(64);

    let (vad_ready_tx,   vad_ready_rx)   = oneshot::channel();
    let (pass1_ready_tx, pass1_ready_rx) = oneshot::channel();
    let (pass2_ready_tx, pass2_ready_rx) = oneshot::channel();

    
    tokio::spawn(vad::vad_task(vad_ready_tx, audio_rx, pass1_tx, pass2_tx));
    vad_ready_rx.await.expect("vad failed to start");

    let pass1_event_tx = event_tx.clone();
    std::thread::Builder::new()
        .name("whisper-pass1".to_string())
        .spawn_with_priority(ThreadPriority::Min, move |result| {
            if result.is_err() { tracing::warn!("Failed to set Min priority for Pass 1"); }
            
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap();
                
            rt.block_on(whisper::pass1_task(pass1_ready_tx, pass1_rx, pass1_event_tx));
        })?;
    pass1_ready_rx.await.expect("pass1 failed to start");

    std::thread::Builder::new()
        .name("whisper-pass2".to_string())
        .spawn_with_priority(ThreadPriority::Crossplatform(50u8.try_into().unwrap()), move |result| {
            if result.is_err() { tracing::warn!("Failed to set priority for Pass 2"); }
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap();
                
            rt.block_on(whisper::pass2_task(pass2_ready_tx, pass2_rx, event_tx));
        })?;
    pass2_ready_rx.await.expect("pass2 failed to start");

    //tokio::spawn(display_task::display_task(event_rx)); ----Terminal display, as of now disabled

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
        Box::new(|_cc| {
            _cc.egui_ctx.set_zoom_factor(1.5);
            Ok(Box::new(App::new(event_rx)))
        }),
    ).map_err(|e| format!("eframe error: {}", e))?;
    
    Ok(())
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

    pub fn update_data(&mut self, ctx: &egui::Context) {
        let mut got_event = false;
        while let Ok(event) = self.event_rx.try_recv() {
            got_event = true;
            match &event {
                TranscriptEvent::Partial { sent_at, phrase_id, .. } => {
                    tracing::info!(pid = phrase_id, delay_ms = sent_at.elapsed().as_millis(), "UI got partial");
                }
                TranscriptEvent::Final { sent_at, phrase_id, .. } => {
                    tracing::info!(pid = phrase_id, delay_ms = sent_at.elapsed().as_millis(), "UI got final");
                }
            };
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
                TranscriptEvent::Final { phrase_id, text, duration_s, rtf, .. } => {
                    self.transcription.insert(phrase_id, PhraseData {
                        text,
                        is_final: true,
                        duration_s,
                        rtf,
                    });
                }
            }
        }
        if got_event {
            ctx.request_repaint();
        }
    }
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.update_data(ctx);

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
        egui::SidePanel::right("settings").show(ctx, |ui| {
            ui.heading("Settings");

            let mut prob = config::speech_probability();
            if ui.add(egui::Slider::new(&mut prob, 0.1..=0.9)
                .text("VAD sensitivity")).changed() {
                config::set_speech_probability(prob);
            }

            let mut dump = config::dump_audio();
            if ui.checkbox(&mut dump, "Dump audio").changed() {
                config::set_dump_audio(dump);
            }
            let mut min_w = config::min_window() as f32 / TARGET_SAMPLE_RATE as f32;
            if ui.add(egui::Slider::new(&mut min_w, 1.0..=8.0)
                .step_by(0.5)
                .suffix("s")
                .text("Min context window")).changed() {
                config::set_min_window_secs(min_w);
            }

            let mut max_w = config::max_window() as f32 / TARGET_SAMPLE_RATE as f32;
            if ui.add(egui::Slider::new(&mut max_w, 4.0..=20.0)
                .step_by(0.5)
                .suffix("s")
                .text("Max context window")).changed() {
                config::set_max_window_secs(max_w.max(min_w + 1.0)); // max всегда > min
            }
            let mut max_phrase = config::max_phrase_samples() as f32 / TARGET_SAMPLE_RATE as f32;
            if ui.add(egui::Slider::new(&mut max_phrase, 5.0..=30.0)
                .step_by(0.5)
                .suffix("s")
                .text("Max phrase length")).changed() {
                config::set_max_phrase_secs(max_phrase);
            }
            let mut min_phrase = config::min_phrase_samples() as f32 / TARGET_SAMPLE_RATE as f32;
            if ui.add(egui::Slider::new(&mut min_phrase, 0.1..=10.0)
                .step_by(0.5)
                .suffix("s")
                .text("Min phrase length")).changed() {
                config::set_min_phrase_secs(min_phrase);
            }
        });
        ctx.request_repaint_after(std::time::Duration::from_millis(50));
    }
    fn on_exit(&mut self, _gl: Option<&eframe::glow::Context>) {
        config::save_to_toml("config.toml");
    }
}