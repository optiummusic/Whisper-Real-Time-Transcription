use std::collections::BTreeMap;
use eframe::egui;
use tokio::sync:: { oneshot, mpsc };
use tracing_subscriber::EnvFilter;
use mimalloc::MiMalloc;
use thread_priority::*;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

use translator::{
    PhraseData, audio, config::{self, TARGET_SAMPLE_RATE}, 
    types::{AudioPacket, PhraseChunk, TranscriptEvent, TranslationEvent}, 
    utils::merge_strings, 
    vad, 
    whisper, 
    translation::Translator
};

fn init_logging() {
    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| {
            EnvFilter::new("info,ort=warn,onnxruntime=warn")
        });
 
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

    whisper::disable_whisper_log();

    translator::utils::prepare_debug_dir();
    
    let (audio_tx, audio_rx) = mpsc::channel::<AudioPacket>(64);
    let (pass1_tx, pass1_rx) = mpsc::channel::<PhraseChunk>(32);
    let (pass2_tx, pass2_rx) = mpsc::channel::<PhraseChunk>(32);
    let (event_tx, mut event_rx_main) = mpsc::channel::<TranscriptEvent>(64);
    let (event_tx_ui, event_rx_ui) = mpsc::channel::<TranscriptEvent>(64);
    let (event_tx_translator, event_rx_translator) = mpsc::channel::<TranscriptEvent>(64);
    let (translation_tx, mut translation_rx) = mpsc::channel::<TranslationEvent>(64);

    let (vad_ready_tx,   vad_ready_rx)   = oneshot::channel();
    let (pass1_ready_tx, pass1_ready_rx) = oneshot::channel();
    let (pass2_ready_tx, pass2_ready_rx) = oneshot::channel();
    let (device_tx, device_rx) = oneshot::channel::<String>();

    tokio::spawn(async move {
        while let Some(evt) = event_rx_main.recv().await {
            let _ = event_tx_ui.send(evt.clone()).await;
            let _ = event_tx_translator.send(evt).await;
        }
    });

    let audio_tx_clone = audio_tx.clone();
    std::thread::Builder::new()
        .name("audio-capture".to_string())
        .spawn(move || {
            tracing::debug!("Audio thread: waiting for device signal...");
            let device_name = device_rx.blocking_recv().expect("UI closed without selecting device");
            tracing::debug!("Audio thread: attempting to open '{}'", device_name);
            let _stream = audio::start_listening(&device_name, audio_tx_clone)
                .expect("Failed to start audio stream");

            tracing::info!("Audio thread: Stream is now ACTIVE");
            loop { std::thread::park(); }
        })?;
    
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
    let translator = Translator::new(event_rx_translator, translation_tx);
    tokio::spawn(translator.translate());

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
            Ok(Box::new(App::new(event_rx_ui, device_tx)))
        }),
    ).map_err(|e| format!("eframe error: {}", e))?;
    
    Ok(())
}

struct App {
    event_rx: mpsc::Receiver<TranscriptEvent>,
    transcription: BTreeMap<u32, PhraseData>,

    device_tx: Option<oneshot::Sender<String>>,
    available_devices: Vec<String>,
    selected_device: String,
    is_running: bool,

    preview_stream: Option<cpal::Stream>,
    last_selected: String,
}
impl App {
    pub fn new(event_rx: mpsc::Receiver<TranscriptEvent>, device_tx: oneshot::Sender<String>) -> Self {
        let devices = audio::get_input_devices();
        let saved_device = config::get_device();

        let selected = if devices.contains(&saved_device) {
            saved_device
        } else {
            devices.first().cloned().unwrap_or_default()
        };
        let preview = audio::start_preview(&selected);

        Self {
            event_rx,
            transcription: BTreeMap::new(),
            device_tx: Some(device_tx),
            available_devices: devices,
            last_selected: selected.clone(),
            selected_device: selected,
            is_running: false,
            preview_stream: preview,
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
    fn select_device(&mut self, ctx: &egui::Context) {
        if self.selected_device != self.last_selected {
            self.preview_stream = audio::start_preview(&self.selected_device);
            self.last_selected = self.selected_device.clone();
        }

        egui::CentralPanel::default().show(ctx, |ui| {
            ui.vertical_centered(|ui| {
                ui.add_space(50.0);
                ui.heading("Audio");
                ui.add_space(20.0);

                egui::ComboBox::from_label("Input device")
                    .width(300.0)
                    .selected_text(&self.selected_device)
                    .show_ui(ui, |ui| {
                        for dev in &self.available_devices {
                            ui.selectable_value(&mut self.selected_device, dev.clone(), dev);
                        }
                    });
                let level = audio::get_ui_level();
                ui.add(egui::ProgressBar::new(level).desired_width(ui.available_width()));

                if ui.button("🔄").on_hover_text("Refresh devices").clicked() {
                    self.refresh_devices();
                }
                ui.add_space(20.0);

                if ui.button("Start transcription").clicked() {
                    self.preview_stream.take();
                    std::thread::sleep(std::time::Duration::from_millis(200));
                    config::set_device(self.selected_device.clone());
                    if let Some(tx) = self.device_tx.take() {
                        let _ = tx.send(self.selected_device.clone());
                    }
                    self.is_running = true;
                }
            });
        });
    }
    fn refresh_devices(&mut self) {
        self.available_devices = audio::get_input_devices();
    }
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        ctx.request_repaint_after(std::time::Duration::from_millis(33));
        if !self.is_running { 
            self.select_device(ctx);
            return; 
        }
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
    }

    fn on_exit(&mut self, _gl: Option<&eframe::glow::Context>) {
        config::save_to_toml("config.toml");
    }
}