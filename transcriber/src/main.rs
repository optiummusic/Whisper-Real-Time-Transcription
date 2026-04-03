use translator::prelude::*;
use eframe::egui::{self};
use mimalloc::MiMalloc;
use tokio::runtime::Handle;
use tracing_subscriber::EnvFilter;
use translator::{
    audio,
    translation::Translator,
    types::{BackendArgs, AppArgs},
    utility::utils::init_audio_dumper,
    vad, whisper,
    ui::App,
};

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

fn init_logging() {
    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("info,ort=warn,onnxruntime=warn"));

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
        tracing_subscriber::fmt().with_env_filter(filter).init();
    }
}

async fn start_backend(mut args: BackendArgs) {
    if args.startup_rx.await.is_err() {
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
        while let Some(evt) = args.event_rx_main.recv().await {
            let _ = args.event_tx_ui.try_send(evt.clone());
            let tx_tr = args.event_tx_translator.clone();
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
            let device_name = args.device_rx
                .blocking_recv()
                .expect("UI closed without selecting device");
            tracing::debug!("Audio thread: attempting to open '{}'", device_name);
            let _stream = audio::start_listening(&device_name, audio_tx_clone)
                .expect("Failed to start audio stream");

            tracing::info!("Audio thread: Stream is now ACTIVE");
            loop {
                std::thread::park();
            }
        })
        .expect("Failed to spawn audio capture thread");

    tokio::spawn(vad::vad_task(vad_ready_tx, audio_rx, pass1_tx, pass2_tx));
    vad_ready_rx.await.expect("vad failed to start");

    let pass1_event_tx = args.event_tx.clone();
    std::thread::Builder::new()
        .name("whisper-pass1".to_string())
        .spawn(move || {
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("Failed to build whisper-pass1 runtime");

            rt.block_on(crate::whisper::whisper::pass1_task(
                pass1_ready_tx,
                pass1_rx,
                pass1_event_tx,
            ));
        })
        .expect("Failed to spawn whisper-pass1 thread");
    pass1_ready_rx.await.expect("pass1 failed to start");

    std::thread::Builder::new()
        .name("whisper-pass2".to_string())
        .spawn(move || {
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("Failed to build whisper-pass2 runtime");

            rt.block_on(crate::whisper::whisper::pass2_task(
                pass2_ready_tx,
                pass2_rx,
                args.event_tx,
            ));
        })
        .expect("Failed to spawn whisper-pass2 thread");
    pass2_ready_rx.await.expect("pass2 failed to start");

    let translator = Translator::new(args.event_rx_translator, args.translation_tx);
    tokio::spawn(translator.translate(std::sync::Arc::clone(&args.translation_buffer)));

    tracing::info!("Backend initialization complete!");
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // ENV VARIABLE CHECK FOR TEST RUN
    if std::env::var("RUN_PIPELINE_TEST").is_ok() {
        std::panic::set_hook(Box::new(|panic_info| {
            eprintln!("CRITICAL PANIC: {}", panic_info);
            std::process::exit(101);
        }));
        
        match translator::utility::utils::run_test_inner().await {
            Ok(res) => {
                println!("TEST_OK:{}", res);
                std::process::exit(0);
            }
            Err(e) => {
                eprintln!("TEST_ERR:{}", e);
                std::process::exit(1);
            }
        }
    }
    init_logging();
    tracing::info!("Logger initialized");
    #[cfg(target_os = "linux")]
    audio::setup_linux_virtual_sink("Whisper Monitor");
    #[cfg(not(target_os = "windows"))]
    {
        rustls::crypto::ring::default_provider()
            .install_default()
            .expect("Failed to install rustls crypto provider");
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

    tokio::spawn(start_backend(BackendArgs {
        startup_rx,
        device_rx,
        event_tx,
        event_rx_main,
        event_tx_ui,
        event_tx_translator,
        event_rx_translator,
        translation_tx,
        translation_buffer: Arc::clone(&translation_buffer),
    }));

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
        Box::new(move |_cc| {
            _cc.egui_ctx.set_zoom_factor(1.15);
            // Создаем App, передавая структуру
            Ok(Box::new(App::new(AppArgs {
                event_rx: event_rx_ui,
                translation_rx,
                translation_buffer,
                device_tx,
                handle,
                startup_tx,
            })))
        }),
    )
    .map_err(|e| format!("eframe error: {}", e))?;

    Ok(())
}