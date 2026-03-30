use std::{collections::BTreeMap, sync::Arc};
use std::collections::{HashMap, HashSet};

use eframe::egui;
use tokio::{runtime::Handle, sync:: { mpsc, oneshot }};
use tracing_subscriber::EnvFilter;
use mimalloc::MiMalloc;
use translator::utils::{TestState };
use wgpu::{Instance, InstanceDescriptor, Backends, Adapter, DeviceType};
use std::sync::atomic::AtomicBool;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

use translator::{
    PhraseData, 
    audio, 
    config::{self, TARGET_SAMPLE_RATE}, 
    translation::Translator, 
    types::{AudioPacket, PhraseChunk, TranscriptEvent, TranslationBuffer, TranslationEvent}, 
    utils::{ merge_strings, init_audio_dumper, ModelType, ModelInfo }, 
    vad, 
    whisper
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

async fn start_backend(
    startup_rx: oneshot::Receiver<()>,
    device_rx: oneshot::Receiver<String>,
    event_tx: mpsc::Sender<TranscriptEvent>,
    mut event_rx_main: mpsc::Receiver<TranscriptEvent>,
    event_tx_ui: mpsc::Sender<TranscriptEvent>,
    event_tx_translator: mpsc::Sender<TranscriptEvent>,
    event_rx_translator: mpsc::Receiver<TranscriptEvent>,
    translation_tx: mpsc::Sender<TranslationEvent>,
    translation_buffer: Arc<TranslationBuffer>,
) {
    if startup_rx.await.is_err() {
        tracing::info!("UI closed before starting the backend.");
        return;
    }
    tracing::info!("Startup signal received. Initializing channels and models...");

    let (audio_tx, audio_rx) = mpsc::channel::<AudioPacket>(128);
    let (pass1_tx, pass1_rx) = mpsc::channel::<PhraseChunk>(128);
    let (pass2_tx, pass2_rx) = mpsc::channel::<PhraseChunk>(128);

    let (vad_ready_tx, vad_ready_rx) = oneshot::channel();
    let (pass1_ready_tx, pass1_ready_rx) = oneshot::channel();
    let (pass2_ready_tx, pass2_ready_rx) = oneshot::channel();

    tokio::spawn(async move {
        while let Some(evt) = event_rx_main.recv().await {
            let _ = event_tx_ui.try_send(evt.clone()); 
            let tx_tr = event_tx_translator.clone();
            tokio::spawn(async move {
                let _ = tx_tr.send(evt).await;
            });
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
        }).expect("Failed to spawn audio capture thread");
    
    tokio::spawn(vad::vad_task(vad_ready_tx, audio_rx, pass1_tx, pass2_tx));
    vad_ready_rx.await.expect("vad failed to start");

    let pass1_event_tx = event_tx.clone();
    std::thread::Builder::new()
        .name("whisper-pass1".to_string())
        .spawn(move || {
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("Failed to build whisper-pass1 runtime");
                
            rt.block_on(crate::whisper::whisper::pass1_task(pass1_ready_tx, pass1_rx, pass1_event_tx));
        }).expect("Failed to spawn whisper-pass1 thread");
    pass1_ready_rx.await.expect("pass1 failed to start");

    std::thread::Builder::new()
        .name("whisper-pass2".to_string())
        .spawn(move || {
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("Failed to build whisper-pass2 runtime");
                
            rt.block_on(crate::whisper::whisper::pass2_task(pass2_ready_tx, pass2_rx, event_tx));
        }).expect("Failed to spawn whisper-pass2 thread");
    pass2_ready_rx.await.expect("pass2 failed to start");
    
    #[cfg(target_os = "linux")]
    audio::setup_linux_virtual_sink("Whisper Monitor");
    
    let translator = Translator::new(event_rx_translator, translation_tx);
    tokio::spawn(translator.translate(std::sync::Arc::clone(&translation_buffer)));

    tracing::info!("Backend initialization complete!");
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    init_logging();
    tracing::info!("Logger initialized");

    #[cfg(not(target_os = "windows"))]
    {
        rustls::crypto::ring::default_provider().install_default().expect("Failed to install rustls crypto provider");
    }

    let _cleanup_guard = audio::LinuxCleanupGuard;
    let default_panic = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        #[cfg(target_os = "linux")]
        audio::cleanup_linux_virtual_sink();
        default_panic(info);
    }));

    #[cfg(target_os = "linux")]
    {
        tokio::spawn(async move {
            if tokio::signal::ctrl_c().await.is_ok() {
                tracing::info!("Shutdown signal received...");
                audio::cleanup_linux_virtual_sink();
                std::process::exit(0);
            }
        });
    }

    config::load_from_toml("config.toml");
    tracing::info!("Config loaded: {:?}", config::startup());

    crate::whisper::whisper::disable_whisper_log();
    init_audio_dumper();
    
    let (event_tx, event_rx_main) = mpsc::channel::<TranscriptEvent>(64);
    let (event_tx_ui, event_rx_ui) = mpsc::channel::<TranscriptEvent>(64);
    let (event_tx_translator, event_rx_translator) = mpsc::channel::<TranscriptEvent>(64);
    let (translation_tx, translation_rx) = mpsc::channel::<TranslationEvent>(64);
    let translation_buffer = translator::types::TranslationBuffer::new();

    let (device_tx, device_rx) = oneshot::channel::<String>();
    
    let (startup_tx, startup_rx) = oneshot::channel::<()>();

    tokio::spawn(start_backend(
        startup_rx,
        device_rx,
        event_tx,
        event_rx_main,
        event_tx_ui,
        event_tx_translator,
        event_rx_translator,
        translation_tx,
        Arc::clone(&translation_buffer)
    ));

    if std::env::args().any(|arg| arg == "--bench") {
        tracing::info!("Benchmark mode: models loaded, exiting.");
        return Ok(()); 
    }

    let native_options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([600.0, 400.0])
            .with_min_inner_size([600.0, 700.0])
            .with_title("Live ASR to Translation"),
        ..Default::default()
    };
    
    let handle = Handle::current();
    eframe::run_native(
        "translator_ui",
        native_options,
        Box::new(|_cc| {
            _cc.egui_ctx.set_zoom_factor(1.25);
            Ok(Box::new(App::new(
                event_rx_ui,
                translation_rx,
                translation_buffer,
                device_tx,
                handle,
                startup_tx,
            )))
        }),
    ).map_err(|e| format!("eframe error: {}", e))?;
    
    Ok(())
}

struct App {
    event_rx:           mpsc::Receiver<TranscriptEvent>,
    translation_rx:     mpsc::Receiver<TranslationEvent>,
    translation_buffer: Arc<TranslationBuffer>,
    handle:             Handle,

    transcription:      BTreeMap<u32, PhraseData>,
    translations:       HashMap<u32, Vec<(usize, usize, String)>>,
    phrase_rects:       HashMap<u32, Vec<egui::Rect>>,
    last_available_width: f32,
    phrases_signaled:   HashSet<u32>,

    device_tx:          Option<oneshot::Sender<String>>,
    available_devices:  Vec<String>,
    selected_device:    String,
    last_selected:      String,
    is_running:         bool,
    preview_stream:     Option<cpal::Stream>,
    pending_config:     config::StartupConfig,
    available_languages: Vec<(&'static str, &'static str)>,
    available_gpus:     Vec<(i32, String)>,

    save_transcription: Arc<AtomicBool>,
    transcript_path: String,
    save_tx: Option<mpsc::Sender<String>>,

    dict_new_word: String,
    dict_new_trans: String,

    test_state: TestState,

    startup_tx: Option<oneshot::Sender<()>>,
    downloading_models: std::sync::Arc<std::sync::Mutex<std::collections::HashSet<String>>>,
    download_errors: std::sync::Arc<std::sync::Mutex<std::collections::HashMap<String, String>>>,

    show_settings: bool,
}

fn get_available_gpus() -> Vec<(i32, String)> {
    
    let fallback_with_error = |reason: &str| {
        vec![(0, format!("Default Device (Auto-detect) [{}]", reason))]
    };

    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let instance = Instance::new(InstanceDescriptor::new_without_display_handle());
        pollster::block_on(instance.enumerate_adapters(Backends::VULKAN))
    }));

    let adapters = match result {
        Ok(adapters) => adapters,
        Err(panic_payload) => {
            let reason = if let Some(s) = panic_payload.downcast_ref::<&str>() {
                s.to_string()
            } else if let Some(s) = panic_payload.downcast_ref::<String>() {
                s.clone()
            } else {
                "unknown panic".to_string()
            };

            tracing::warn!("Vulkan panic: {}", reason);
            return fallback_with_error(&reason);
        }
    };
    let mut temp_gpus = Vec::new();
    let mut seen_pci_ids = std::collections::HashSet::new();

    for adapter in adapters {
        let info = adapter.get_info();
        
        if info.device_type == DeviceType::Cpu {
            continue;
        }

        let pci_id = (info.vendor, info.device);
        if !seen_pci_ids.insert(pci_id) {
            continue;
        }
        
        let device_type_str = match info.device_type {
            DeviceType::DiscreteGpu   => "(Discrete)",
            DeviceType::IntegratedGpu => "(Integrated)",
            DeviceType::VirtualGpu    => "(Virtual)",
            _                         => "(Unknown)",
        };

        let display_name = format!("{} {}", info.name, device_type_str);
        temp_gpus.push((info.device_type, display_name, info.device_pci_bus_id));
    }

    temp_gpus.sort_by(|a, b| {
        match (a.0, b.0) {
            (DeviceType::IntegratedGpu, DeviceType::DiscreteGpu) => std::cmp::Ordering::Less,
            (DeviceType::DiscreteGpu, DeviceType::IntegratedGpu) => std::cmp::Ordering::Greater,
            _ => std::cmp::Ordering::Equal,
        }
    });

    let mut gpus = Vec::new();
    for (index, (_, display_name, bus_id)) in temp_gpus.into_iter().enumerate() {
        tracing::info!("Validated GPU: index={}, name={}, id={}", index, display_name, bus_id);
        gpus.push((index as i32, display_name));
    }

    if gpus.is_empty() {
        gpus.push((0, "Default Device (Auto-detect)".to_string()));
    }

    gpus
}

impl App {
    pub fn new(
        event_rx: mpsc::Receiver<TranscriptEvent>,
        translation_rx: mpsc::Receiver<TranslationEvent>,
        translation_buffer: Arc<TranslationBuffer>,
        device_tx: oneshot::Sender<String>,
        handle: Handle,
        startup_tx: oneshot::Sender<()>,
    ) -> Self {
        let devices = audio::get_input_devices();
        let saved_device = config::get_device();
        let cfg = config::startup();
        let selected = if devices.contains(&saved_device) { saved_device }
                       else { devices.first().cloned().unwrap_or_default() };
        let preview = audio::start_preview(&selected);
        let save_flag = Arc::new(AtomicBool::new(false));

        Self {
            event_rx,
            translation_rx,
            translation_buffer,
            handle,
            transcription: BTreeMap::new(),
            translations: HashMap::new(),
            phrase_rects: HashMap::new(),
            phrases_signaled: HashSet::new(),
            last_available_width: 0.0,
            device_tx: Some(device_tx),
            available_devices: devices,
            last_selected: selected.clone(),
            selected_device: selected,
            is_running: false,
            preview_stream: preview,
            pending_config: cfg,
            available_languages: vec![
                ("en", "English"), ("uk", "Ukrainian"), ("ru", "Russian"), ("auto", "Auto-detect"),
            ],
            available_gpus: get_available_gpus(),
            save_transcription: save_flag,
            save_tx: None,
            transcript_path: format!("transcriptions/{}.txt", chrono::Local::now().format("%Y_%m_%d_%H_%M")),
            dict_new_word: String::new(),
            dict_new_trans: String::new(),
            test_state: TestState::Idle,
            startup_tx: Some(startup_tx),
            downloading_models: std::sync::Arc::new(std::sync::Mutex::new(std::collections::HashSet::new())),
            download_errors: Arc::new(std::sync::Mutex::new(HashMap::new())),
            show_settings: true,
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
                        text: text.clone(),
                        is_final: true,
                        duration_s,
                        rtf,
                    });
                    if let Some(tx) = &self.save_tx {
                        let _ = tx.try_send(text.clone());
                    }
                }
            }
        }
        while let Ok(evt) = self.translation_rx.try_recv() {
            got_event = true;
            match evt {
                TranslationEvent::Translate { phrase_id, word_index, span, text } => {
                    self.translations
                        .entry(phrase_id)
                        .or_default()
                        .push((word_index, span, text));
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
        egui::SidePanel::right("model_status").show(ctx, |ui| {
            ui.heading("Models");
            
            self.show_model_row(ui, ctx, ModelType::VAD);
            self.show_model_row(ui, ctx, ModelType::WFast);
            self.show_model_row(ui, ctx, ModelType::WAcc);

            ui.separator();

            match &self.test_state {
                TestState::Idle => {
                    if ui.button("▶ Test Pipeline").clicked() {
                        let rx = translator::utils::start_test();
                        self.test_state = TestState::Running(rx);
                    }
                }
                TestState::Running(rx) => {
                    if let Ok(result) = rx.try_recv() {
                        self.test_state = TestState::Done(result);
                    } else {
                        ui.spinner();
                        ui.label("Testing...");
                        ctx.request_repaint_after(std::time::Duration::from_millis(100));
                    }
                }
                TestState::Done(Ok(text)) => {
                    ui.colored_label(egui::Color32::GREEN, "✅ Pipeline OK");
                    ui.weak(text);
                    if ui.small_button("Reset").clicked() {
                        self.test_state = TestState::Idle;
                    }
                }
                TestState::Done(Err(e)) => {
                    ui.colored_label(egui::Color32::RED, "❌ Failed");
                    ui.label(e);
                    if ui.small_button("Retry").clicked() {
                        self.test_state = TestState::Idle;
                    }
                }
            }
        });
        egui::CentralPanel::default().show(ctx, |ui| {
            egui::ScrollArea::both()
            .auto_shrink([false; 2])
            .show(ui, |ui| {
                ui.add_space(20.0);
                ui.heading("Translator Setup");
                ui.add_space(20.0);

                ui.group(|ui| {
                    ui.label(egui::RichText::new("Audio Input").strong());
                    egui::ComboBox::from_label("Device")
                        .width(300.0)
                        .selected_text(&self.selected_device)
                        .show_ui(ui, |ui| {
                            for dev in &self.available_devices {
                                ui.selectable_value(&mut self.selected_device, dev.clone(), dev);
                            }
                        });

                    ui.add_space(5.0);

                    ui.horizontal(|ui| {
                        ui.label("Mic Boost:");
                        if ui.add(egui::Slider::new(&mut self.pending_config.audio_gain, 1.0..=10.0)
                            .step_by(0.1))
                            .changed() 
                        {
                            config::set_audio_gain(self.pending_config.audio_gain);
                        }
                    });

                    let level = audio::get_ui_level() * self.pending_config.audio_gain;
                    ui.add(egui::ProgressBar::new(level.min(1.0)).desired_width(300.0));
                    ui.add_space(5.0);
                    if ui.button("🔄").on_hover_text("Refresh devices").clicked() {
                        self.refresh_devices();
                    }
                });

                ui.add_space(10.0);

                ui.group(|ui| {
                    ui.label(egui::RichText::new("Model Engine Settings").strong());
                    
                    ui.horizontal(|ui| {
                        ui.label("Target Language:");
                        egui::ComboBox::from_id_salt("lang_select")
                            .selected_text(self.pending_config.language.clone())
                            .show_ui(ui, |ui| {
                                for (code, name) in &self.available_languages {
                                    ui.selectable_value(&mut self.pending_config.language, code.to_string(), *name);
                                }
                            });
                    });

                    ui.separator();

                    ui.vertical(|ui| {
                        ui.label(egui::RichText::new("Fast Model (Pass 1)").underline());
                        ui.checkbox(&mut self.pending_config.use_gpu_fast, "Use GPU acceleration");
                        
                        if self.pending_config.use_gpu_fast {
                            let selected_name = self.available_gpus.iter()
                                .find(|(id, _)| *id == self.pending_config.gpu_device_fast)
                                .map(|(_, name)| name.clone())
                                .unwrap_or_else(|| "Select GPU...".to_string());

                            egui::ComboBox::from_id_salt("gpu_fast_select")
                                .width(ui.available_width() - 20.0)
                                .selected_text(selected_name)
                                .show_ui(ui, |ui| {
                                    for (id, name) in &self.available_gpus {
                                        ui.selectable_value(
                                            &mut self.pending_config.gpu_device_fast, 
                                            *id, 
                                            name
                                        );
                                    }
                                });
                        }

                        ui.add_space(10.0);
                        ui.separator();
                        ui.add_space(5.0);

                        ui.label(egui::RichText::new("Accurate Model (Pass 2)").underline());
                        ui.checkbox(&mut self.pending_config.use_gpu_acc, "Use GPU acceleration");
                        
                        if self.pending_config.use_gpu_acc {
                            let selected_name = self.available_gpus.iter()
                                .find(|(id, _)| *id == self.pending_config.gpu_device_acc)
                                .map(|(_, name)| name.clone())
                                .unwrap_or_else(|| "Select GPU...".to_string());

                            egui::ComboBox::from_id_salt("gpu_acc_select")
                                .width(ui.available_width() - 20.0)
                                .selected_text(selected_name)
                                .show_ui(ui, |ui| {
                                    for (id, name) in &self.available_gpus {
                                        ui.selectable_value(
                                            &mut self.pending_config.gpu_device_acc, 
                                            *id, 
                                            name
                                        );
                                    }
                                });
                        }
                    });
                });

                ui.add_space(30.0);

                if ui.add(egui::Button::new(egui::RichText::new("START").heading())
                    .min_size(egui::vec2(200.0, 50.0))).clicked() 
                {
                    self.preview_stream = None;
                    let (tx, rx) = mpsc::channel(100);
                    self.save_tx = Some(tx);
                    let should_save = Arc::clone(&self.save_transcription);
                    let path = self.transcript_path.clone();
                    self.handle.spawn(translator::utils::recording_task(rx, path, should_save));
                    config::init(
                        self.pending_config.language.clone(),
                        self.pending_config.use_gpu_fast,
                        self.pending_config.use_gpu_acc,
                        self.pending_config.gpu_device_fast,
                        self.pending_config.gpu_device_acc,
                        self.pending_config.audio_gain,
                    );
                    config::set_device(self.selected_device.clone());
                    config::save_to_toml("config.toml");
                    if let Some(tx) = self.startup_tx.take() {
                        let _ = tx.send(());
                    }

                    if let Some(tx) = self.device_tx.take() {
                        let _ = tx.send(self.selected_device.clone());
                    }
                    self.is_running = true;
                }
            });
        });
    }
    fn show_model_row(&mut self, ui: &mut egui::Ui, ctx: &egui::Context, t: ModelType) {
        let info = ModelInfo::get(&t);
        let found = info.path.exists();

        let is_downloading = self.downloading_models.lock().unwrap().contains(info.name);
        let error_msg = self.download_errors.lock().unwrap().get(info.name).cloned();

        ui.vertical(|ui| {
            ui.horizontal(|ui| {
                if found {
                    ui.colored_label(egui::Color32::GREEN, "✅");
                    ui.label(info.name);
                } else {
                    ui.colored_label(egui::Color32::RED, "❌");
                    ui.label(info.name);
                    ui.weak(format!("({})", info.size_str));

                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if is_downloading {
                            ui.spinner();
                            ctx.request_repaint();
                        } else {
                            if ui.button("Download").clicked() {
                                self.downloading_models.lock().unwrap().insert(info.name.to_string());
                                self.download_model(ctx.clone(), info.name, info.url, info.path.clone());
                            }
                        }
                    });
                }
            });

            if let Some(err) = error_msg {
                ui.add_space(-4.0);
                ui.horizontal(|ui| {
                    ui.add_space(24.0);
                    ui.colored_label(egui::Color32::LIGHT_RED, egui::RichText::new(format!("{}", err)).small());
                });
            }
        });
    }

    fn download_model(&self, ctx: egui::Context, name: &'static str, url: &'static str, target: std::path::PathBuf) {
        let url = url.to_string();
        let name = name.to_string();
        let downloading_models = self.downloading_models.clone();
        let download_errors = self.download_errors.clone(); // Клонируем доступ к ошибкам

        std::thread::spawn(move || {
            download_errors.lock().unwrap().remove(&name);
            let result: Result<(), String> = (|| {
                if let Some(parent) = target.parent() {
                    std::fs::create_dir_all(parent).map_err(|e| format!("Folder error: {}", e))?;
                }

                let response = reqwest::blocking::get(&url)
                    .map_err(|e| format!("Network error: {}", e))?;
                
                if !response.status().is_success() {
                    return Err(format!("Server returned {}", response.status()));
                }

                let bytes = response.bytes().map_err(|e| format!("Read error: {}", e))?;
                std::fs::write(&target, bytes).map_err(|e| format!("Save error: {}", e))?;
                
                Ok(())
            })();

            if let Err(e) = result {
                tracing::error!("Download failed for {}: {}", name, e);
                download_errors.lock().unwrap().insert(name.clone(), e);
            }

            downloading_models.lock().unwrap().remove(&name);
            ctx.request_repaint();
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
        if self.show_settings {
        egui::SidePanel::right("settings")
            .min_width(330.0)
            .max_width(330.0)  
            .resizable(false).show(ctx, |ui| {
                egui::ScrollArea::both()
                .id_salt("settings_scroll")
                .auto_shrink([false; 2])
                .show(ui, |ui| {
                    ui.heading("Settings");
                    ui.add_space(8.0);
                    
                    ui.group(|ui| {
                        ui.label(egui::RichText::new("Audio Input").strong());
                        
                        let mut gain = config::audio_gain();
                        if ui.add(egui::Slider::new(&mut gain, 1.0..=10.0)
                            .text("Mic Boost"))
                            .changed() 
                        {
                            config::set_audio_gain(gain);
                        }
                        
                        let level = audio::get_ui_level() * gain;
                        ui.add(egui::ProgressBar::new(level.min(1.0)).desired_width(ui.available_width()));
                    });
                    
                    ui.add_space(8.0);

                    ui.group(|ui| {
                        ui.label(egui::RichText::new("VAD (Voice Activation Detection)").strong());
                        
                        let mut prob = config::speech_probability();
                        if ui.add(egui::Slider::new(&mut prob, 0.1..=0.9)
                            .text("Sensitivity")).changed() {
                            config::set_speech_probability(prob);
                        }

                        let mut max_silence = config::max_silence_chunks();
                        if ui.add(egui::Slider::new(&mut max_silence, 1..=50)
                            .text("Max Silence Chunks")).changed() {
                            config::set_max_silence_chunks(max_silence);
                        }

                        let mut preroll = config::preroll_chunks();
                        if ui.add(egui::Slider::new(&mut preroll, 0..=20)
                            .text("Preroll Chunks")).changed() {
                            config::set_preroll_chunks(preroll);
                        }
                    });

                    ui.add_space(8.0);

                    ui.group(|ui| {
                        ui.label(egui::RichText::new("Engine & Context").strong());

                        let mut stitch_sil = config::stitch_max_silence();
                        if ui.add(egui::Slider::new(&mut stitch_sil, 0.1..=5.0)
                            .step_by(0.1)
                            .suffix("s")
                            .text("Stitch Max Silence")).changed() {
                            config::set_stitch_max_silence(stitch_sil);
                        }

                        let mut fast_track = config::fast_track_threshold_s();
                        if ui.add(egui::Slider::new(&mut fast_track, 0.5..=10.0)
                            .step_by(0.1)
                            .suffix("s")
                            .text("Fast Track Threshold")).changed() {
                            config::set_fast_track_threshold(fast_track);
                        }

                        let mut min_w = config::min_window() as f32 / TARGET_SAMPLE_RATE as f32;
                        if ui.add(egui::Slider::new(&mut min_w, 1.0..=8.0)
                            .step_by(0.5)
                            .suffix("s")
                            .text("Min Context Window")).changed() {
                            config::set_min_window_secs(min_w);
                        }

                        let mut max_w = config::max_window() as f32 / TARGET_SAMPLE_RATE as f32;
                        if ui.add(egui::Slider::new(&mut max_w, 4.0..=30.0)
                            .step_by(0.5)
                            .suffix("s")
                            .text("Max Context Window")).changed() {
                            config::set_max_window_secs(max_w);
                        }
                    });

                    ui.add_space(8.0);

                    ui.group(|ui| {
                        ui.label(egui::RichText::new("Phrase Limits").strong());

                        let mut min_phrase = config::min_phrase_samples() as f32 / TARGET_SAMPLE_RATE as f32;
                        if ui.add(egui::Slider::new(&mut min_phrase, 0.1..=5.0)
                            .step_by(0.1)
                            .suffix("s")
                            .text("Min Phrase Length")).changed() {
                            config::set_min_phrase_secs(min_phrase);
                        }

                        let mut max_phrase = config::max_phrase_samples() as f32 / TARGET_SAMPLE_RATE as f32;
                        if ui.add(egui::Slider::new(&mut max_phrase, 5.0..=60.0)
                            .step_by(1.0)
                            .suffix("s")
                            .text("Max Phrase Length")).changed() {
                            config::set_max_phrase_secs(max_phrase);
                        }
                    });

                    ui.add_space(8.0);

                    let mut dump = config::dump_audio();
                    if ui.checkbox(&mut dump, "Dump audio").changed() {
                        config::set_dump_audio(dump);
                    }

                    let mut check_val = self.save_transcription.load(std::sync::atomic::Ordering::Relaxed);
                    if ui.checkbox(&mut check_val, "Save transcription to disk").changed() {
                        self.save_transcription.store(check_val, std::sync::atomic::Ordering::Relaxed);
                    }
                    
                    if ui.button("Reset Transcription").clicked() {
                        self.transcription.clear();
                        self.translations.clear();
                        self.phrases_signaled.clear();
                        self.phrase_rects.clear();
                    }

                    ui.add_space(8.0);

                    ui.group(|ui| {
                        ui.label(egui::RichText::new("Dictionary (Live)").strong());
                        
                        ui.horizontal(|ui| {
                            ui.add(egui::TextEdit::singleline(&mut self.dict_new_word).desired_width(120.0).hint_text("Word"));
                            ui.label("->");
                            ui.add(egui::TextEdit::singleline(&mut self.dict_new_trans).desired_width(120.0).hint_text("Translation"));
                        });
                        
                        if ui.button("Add to custom.toml").clicked() {
                            if !self.dict_new_word.is_empty() && !self.dict_new_trans.is_empty() {
                                translator::utils::add_to_custom_dict(&self.dict_new_word, &self.dict_new_trans);
                                self.dict_new_word.clear();
                                self.dict_new_trans.clear();
                            }
                        }
                    
                    ui.add_space(16.0);
                    });
                });
                    
        });
        } // IF SHOW SETTINGS BRACKET
        

        egui::CentralPanel::default().show(ctx, |ui| {
            ui.horizontal(|ui| {
                let btn_text = if self.show_settings { "Hide Settings" } else { "Show Settings" };
                if ui.button(btn_text).clicked() {
                    self.show_settings = !self.show_settings;
                
                }
                ui.add_space(8.0);
                ui.heading(":) Live Transcription");
            });
            ui.separator();
            let available_width = ui.available_width();
            if (available_width - self.last_available_width).abs() > 1.0 {
                self.phrase_rects.clear();
                self.last_available_width = available_width;
            }
            egui::ScrollArea::both().stick_to_bottom(true).show(ui, |ui| {
                for (id, data) in &self.transcription {
                    if data.is_final {
                        let words: Vec<&str> = data.text.split_whitespace().collect();
                        let trans = self.translations.get(id).cloned().unwrap_or_default();

                        let row_groups: Vec<Vec<usize>> = if let Some(prev_rects) = self.phrase_rects.get(id) {
                            let mut rows: BTreeMap<i32, Vec<usize>> = BTreeMap::new();
                            for (i, r) in prev_rects.iter().enumerate() {
                                rows.entry(r.min.y as i32).or_default().push(i);
                            }
                            rows.into_values().collect()
                        } else {
                            vec![(0..words.len()).collect()]
                        };

                        let row_count = row_groups.len();
                        let mut new_rects: Vec<egui::Rect> = vec![egui::Rect::NOTHING; words.len()];

                        for (row_idx, row_word_indices) in row_groups.iter().enumerate() {
                            let row_rects = ui.horizontal_wrapped(|ui| {
                                if row_idx == 0 { ui.weak(format!("[{id}]")); }
                                let mut rects = vec![];
                                for &wi in row_word_indices {
                                    if wi < words.len() {
                                        let r = ui.colored_label(egui::Color32::LIGHT_GREEN, words[wi]);
                                        rects.push((wi, r.rect));
                                    }
                                }
                                if row_idx == row_count - 1 {
                                    ui.weak(format!("({:.1}s | RTF: {:.2})", data.duration_s, data.rtf));
                                }
                                rects
                            }).inner;

                            for (wi, rect) in row_rects {
                                if wi < new_rects.len() { new_rects[wi] = rect; }
                            }

                            let row_set: HashSet<usize> = row_word_indices.iter().copied().collect();
                            let row_trans: Vec<_> = trans.iter()
                                .filter(|(wi, _, _)| row_set.contains(wi))
                                .collect();

                            if !row_trans.is_empty() {
                                ui.horizontal_wrapped(|ui| {
                                    for (_, _, text) in &row_trans {
                                        ui.colored_label(egui::Color32::RED, text);
                                    }
                                });
                            }
                        }

                        self.phrase_rects.insert(*id, new_rects);

                        if !self.phrases_signaled.contains(id) && !words.is_empty() {
                            self.phrases_signaled.insert(*id);
                            let buf = Arc::clone(&self.translation_buffer);
                            let pid = *id;
                            self.handle.spawn(async move { buf.signal_ready(pid).await; });
                        }
                    } else {
                        ui.horizontal(|ui| {
                            ui.weak(format!("[{id}]"));
                            ui.colored_label(egui::Color32::GRAY, format!("{} ⏳", data.text));
                        });
                    }
                }
            });
        });
    }

    fn on_exit(&mut self, _gl: Option<&eframe::glow::Context>) {
        #[cfg(target_os = "linux")]
        audio::cleanup_linux_virtual_sink();
        config::save_to_toml("config.toml");
    }
}