use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::{collections::BTreeMap, sync::Arc};

use eframe::egui::{self, Ui};
use std::sync::atomic::{AtomicBool, Ordering};
use tokio::{
    runtime::Handle,
    sync::{mpsc, oneshot},
};
use crate::utility::stats::PipelineEventKind;
use crate::utility::utils::{TestState, models_are_identical};


use crate::{
    PhraseData, audio,
    config::{self, TARGET_SAMPLE_RATE},
    types::{TranscriptEvent, TranslationBuffer, TranslationEvent, AppArgs},
    utility::{
        stats,
        utils::{ModelInfo, ModelType, merge_strings, start_test, get_available_gpus},
    },
};

#[derive(Default)]
struct AppState {
    transcription: BTreeMap<u32, PhraseData>,
    translations: HashMap<u32, Vec<(usize, usize, String)>>,
    phrase_rects: HashMap<u32, Vec<egui::Rect>>,
    last_available_width: f32,
    phrases_signaled: HashSet<u32>,
    selected_device: String,
    last_selected: String,
    is_running: bool,
    save_transcription: Arc<AtomicBool>,
    save_tx: Option<mpsc::Sender<String>>,
    dict_new_word: String,
    dict_new_trans: String,
    downloading_models: std::sync::Arc<std::sync::Mutex<std::collections::HashSet<String>>>,
    download_errors: std::sync::Arc<std::sync::Mutex<std::collections::HashMap<String, String>>>,
    show_settings: bool,
    transcript_path: PathBuf,
    available_languages: Vec<(String, String)>,
    available_gpus: Vec<(i32, String)>,
    test_state: TestState,
    available_devices: Vec<String>,
}
pub struct App {
    event_rx: mpsc::Receiver<TranscriptEvent>,
    translation_rx: mpsc::Receiver<TranslationEvent>,
    translation_buffer: Arc<TranslationBuffer>,
    handle: Handle,
    startup_tx: Option<oneshot::Sender<()>>,
    device_tx: Option<oneshot::Sender<String>>,
    pending_config: config::StartupConfig,
    preview_stream: Option<cpal::Stream>,
    state: AppState,
}

impl App {
    pub fn new(args: AppArgs) -> Self {
        let devices = audio::get_input_devices();
        let saved_device = config::get_device();
        let cfg = config::startup();
        let selected = if devices.contains(&saved_device) {
            saved_device
        } else {
            devices.first().cloned().unwrap_or_default()
        };
        let preview = audio::start_preview(&selected);
        let transcript_filename = chrono::Local::now().format("%Y_%m_%d_%H_%M").to_string();

        let mut state = AppState::default();
        state.selected_device = selected.clone();
        state.last_selected = selected;
        state.transcript_path = crate::utility::utils::get_base_dir()
                .join("transcriptions")
                .join(format!("{}.txt", transcript_filename));
        state.available_languages = config::get_available_languages();
        state.available_gpus = get_available_gpus();
        state.available_devices = devices;
        Self {
            event_rx: args.event_rx,
            translation_rx: args.translation_rx,
            translation_buffer: args.translation_buffer,
            handle: args.handle,
            device_tx: Some(args.device_tx),
            startup_tx: Some(args.startup_tx),
            preview_stream: preview,
            pending_config: cfg,
            state,
        }
    }

    pub fn update_data(&mut self, ctx: &egui::Context) {
        let mut got_event = false;
        while let Ok(event) = self.event_rx.try_recv() {
            got_event = true;
            match &event {
                TranscriptEvent::Partial {
                    sent_at, phrase_id, ..
                } => {
                    tracing::debug!(
                        pid = phrase_id,
                        delay_ms = sent_at.elapsed().as_millis(),
                        "UI got partial"
                    );
                }
                TranscriptEvent::Final {
                    sent_at, phrase_id, ..
                } => {
                    tracing::debug!(
                        pid = phrase_id,
                        delay_ms = sent_at.elapsed().as_millis(),
                        "UI got final"
                    );
                }
            };
            match event {
                TranscriptEvent::Partial {
                    phrase_id, text, ..
                } => {
                    let entry = self.state.transcription.entry(phrase_id).or_insert(PhraseData {
                        text: String::new(),
                        is_final: false,
                        duration_s: 0.0,
                        rtf: 0.0,
                    });
                    if !entry.is_final {
                        entry.text = merge_strings(&entry.text, &text);
                    }
                }
                TranscriptEvent::Final {
                    phrase_id,
                    text,
                    duration_s,
                    rtf,
                    ..
                } => {
                    self.state.transcription.insert(
                        phrase_id,
                        PhraseData {
                            text: text.clone(),
                            is_final: true,
                            duration_s,
                            rtf,
                        },
                    );
                    if let Some(tx) = &self.state.save_tx {
                        let _ = tx.try_send(text.clone());
                    }
                }
            }
        }
        while let Ok(evt) = self.translation_rx.try_recv() {
            got_event = true;
            match evt {
                TranslationEvent::Translate {
                    phrase_id,
                    word_index,
                    span,
                    text,
                } => {
                    self.state.translations
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
    fn draw_model_status(&mut self, ctx: &egui::Context, ui: &mut Ui) {
        ui.heading("Models");
        self.show_model_row(ui, ctx, ModelType::VAD);
        self.show_model_row(ui, ctx, ModelType::WFast);
        self.show_model_row(ui, ctx, ModelType::WAcc);

        ui.separator();

        match &self.state.test_state {
            TestState::Idle => {
                if ui.button("▶ Test Pipeline").clicked() {
                    let rx = start_test();
                    self.state.test_state = TestState::Running(rx);
                }
            }
            TestState::Running(rx) => {
                if let Ok(result) = rx.try_recv() {
                    self.state.test_state = TestState::Done(result);
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
                    self.state.test_state = TestState::Idle;
                }
            }
            TestState::Done(Err(e)) => {
                ui.colored_label(egui::Color32::RED, "❌ Failed");
                ui.label(e);
                if ui.small_button("Retry").clicked() {
                    self.state.test_state = TestState::Idle;
                }
            }
        }
    }
    fn draw_audio_input(&mut self, ui: &mut Ui) {
        ui.group(|ui| {
        ui.label(egui::RichText::new("Audio Input").strong());
        egui::ComboBox::from_label("Device")
            .width(300.0)
            .selected_text(&self.state.selected_device)
            .show_ui(ui, |ui| {
                for dev in &self.state.available_devices {
                    ui.selectable_value(
                        &mut self.state.selected_device,
                        dev.clone(),
                        dev,
                    );
                }
            });

        ui.add_space(5.0);

        ui.horizontal(|ui| {
            ui.label("Mic Boost:");
            if ui
                .add(
                    egui::Slider::new(
                        &mut self.pending_config.audio_gain,
                        1.0..=10.0,
                    )
                    .step_by(0.1),
                )
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
        }});
    }
    fn draw_language(&mut self, ui: &mut Ui) {
        ui.group(|ui| {
            ui.label(egui::RichText::new("Language Settings").strong());
            ui.add_space(4.0);

            ui.horizontal(|ui| {
                ui.label("Whisper Source:");
                egui::ComboBox::from_id_salt("whisper_lang_select")
                    .selected_text(config::source_lang().to_string())
                    .show_ui(ui, |ui| {
                        for (code, name) in &self.state.available_languages {
                            if ui.selectable_label(config::source_lang().as_ref() == code, name).clicked() {
                                config::set_src_lang(code.clone());
                            }
                        }
                    });
            });

            ui.horizontal(|ui| {
                ui.label("Translation Target:");
                egui::ComboBox::from_id_salt("target_lang_select")
                    .selected_text(config::target_lang().to_string())
                    .show_ui(ui, |ui| {
                        for (code, name) in &self.state.available_languages {
                            if ui.selectable_label(config::target_lang().as_ref() == code, name).clicked() {
                                config::set_tgt_lang(code.clone());
                            }
                        }
                    });
            });
        });
    }
    fn draw_gpu(&mut self, ui: &mut Ui) {
        ui.vertical(|ui| {
            ui.label(egui::RichText::new("Fast Model (Pass 1)").underline());
            ui.checkbox(
                &mut self.pending_config.use_gpu_fast,
                "Use GPU acceleration",
            );

            if self.pending_config.use_gpu_fast {
                let selected_name = self
                    .state.available_gpus
                    .iter()
                    .find(|(id, _)| *id == self.pending_config.gpu_device_fast)
                    .map(|(_, name)| name.clone())
                    .unwrap_or_else(|| "Select GPU...".to_string());

                egui::ComboBox::from_id_salt("gpu_fast_select")
                    .width(ui.available_width() - 20.0)
                    .selected_text(selected_name)
                    .show_ui(ui, |ui| {
                        for (id, name) in &self.state.available_gpus {
                            ui.selectable_value(
                                &mut self.pending_config.gpu_device_fast,
                                *id,
                                name,
                            );
                        }
                    });
            }

            ui.add_space(10.0);
            ui.separator();
            ui.add_space(5.0);

            ui.label(egui::RichText::new("Accurate Model (Pass 2)").underline());
            ui.checkbox(
                &mut self.pending_config.use_gpu_acc,
                "Use GPU acceleration",
            );

            if self.pending_config.use_gpu_acc {
                let selected_name = self
                    .state.available_gpus
                    .iter()
                    .find(|(id, _)| *id == self.pending_config.gpu_device_acc)
                    .map(|(_, name)| name.clone())
                    .unwrap_or_else(|| "Select GPU...".to_string());

                egui::ComboBox::from_id_salt("gpu_acc_select")
                    .width(ui.available_width() - 20.0)
                    .selected_text(selected_name)
                    .show_ui(ui, |ui| {
                        for (id, name) in &self.state.available_gpus {
                            ui.selectable_value(
                                &mut self.pending_config.gpu_device_acc,
                                *id,
                                name,
                            );
                        }
                    });
            }
        });
    }
    fn main_draw_language(&mut self, ui: &mut Ui) {
        let current_src = config::source_lang();
        let current_tgt = config::target_lang();
        ui.group(|ui| {
            ui.label("Whisper Language (Source):");
            ui.horizontal_wrapped(|ui| {
                for (code, name) in &self.state.available_languages {
                    if ui.selectable_label(current_src.as_ref() == code, name).clicked() {
                        config::set_src_lang(code.clone());
                    }
                }
            });
        });

        ui.group(|ui| {
            ui.label("Translation Language (Target):");
            ui.horizontal_wrapped(|ui| {
                for (code, name) in &self.state.available_languages {
                    if ui.selectable_label(current_tgt.as_ref() == code, name).clicked() {
                        config::set_tgt_lang(code.clone());
                    }
                }
            });
        });
    }
    fn main_draw_audio(&mut self, ui: &mut Ui) {
        ui.label(egui::RichText::new("Audio Input").strong());
        let mut gain = config::audio_gain();
        if ui.add(egui::Slider::new(&mut gain, 1.0..=10.0).text("Mic Boost"))
            .changed()
        {
            config::set_audio_gain(gain);
        }

        let level = audio::get_ui_level() * gain;
        ui.add(egui::ProgressBar::new(level.min(1.0))
            .desired_width(ui.available_width()),
        );

        ui.add_space(4.0);

        let muted = config::AUDIO_MUTED.load(Ordering::Relaxed);
        let (btn_label, btn_color) = if muted {
            ("Unmute Audio", egui::Color32::RED)
        } else {
            ("Mute Audio", egui::Color32::LIGHT_GRAY)
        };
        if ui.add(egui::Button::new(egui::RichText::new(btn_label).color(btn_color),
            )).clicked()
        {
            config::AUDIO_MUTED.store(!muted, Ordering::Relaxed);
        }
    }
    fn main_draw_vad(&mut self, ui: &mut Ui) {
        ui.label(egui::RichText::new("VAD (Voice Activation Detection)")
            .strong(),
        );

        let mut prob = config::speech_probability();
        if ui.add(egui::Slider::new(&mut prob, 0.1..=0.9).text("Sensitivity"),)
            .changed()
        {
            config::set_speech_probability(prob);
        }

        let mut max_silence = config::max_silence_chunks();
        if ui.add(egui::Slider::new(&mut max_silence, 1..=50).text("Max Silence Chunks"),)
            .changed()
        {
            config::set_max_silence_chunks(max_silence);
        }

        let mut preroll = config::preroll_chunks();
        if ui.add(egui::Slider::new(&mut preroll, 0..=20).text("Preroll Chunks"),)
            .changed()
        {
            config::set_preroll_chunks(preroll);
        }
    }
    fn main_draw_engine(&mut self, ui: &mut Ui) {
        ui.label(egui::RichText::new("Engine & Context").strong());
        let mut stitch_sil = config::stitch_max_silence();
        if ui.add(egui::Slider::new(&mut stitch_sil, 0.1..=5.0)
            .step_by(0.1)
            .suffix("s").text("Stitch Max Silence"),)
            .changed()
        {
            config::set_stitch_max_silence(stitch_sil);
        }

        let mut fast_track = config::fast_track_threshold_s();
        if ui.add(egui::Slider::new(&mut fast_track, 0.5..=10.0)
            .step_by(0.1)
            .suffix("s").text("Fast Track Threshold"),)
            .changed()
        {
            config::set_fast_track_threshold(fast_track);
        }

        let mut min_w =
            config::min_window() as f32 / TARGET_SAMPLE_RATE as f32;
        if ui.add(egui::Slider::new(&mut min_w, 1.0..=8.0)
            .step_by(0.5)
            .suffix("s").text("Min Context Window"),)
            .changed()
        {
            config::set_min_window_secs(min_w);
        }

        let mut max_w =
            config::max_window() as f32 / TARGET_SAMPLE_RATE as f32;
        if ui.add(egui::Slider::new(&mut max_w, 4.0..=30.0).step_by(0.5)
            .suffix("s").text("Max Context Window"),)
            .changed()
        {
            config::set_max_window_secs(max_w);
        }
    }
    fn main_draw_phrases(&mut self, ui: &mut Ui) {
        ui.label(egui::RichText::new("Phrase Limits").strong());
        let mut min_phrase = config::min_phrase_samples() as f32 / TARGET_SAMPLE_RATE as f32;
        if ui.add(egui::Slider::new(&mut min_phrase, 0.1..=5.0)
            .step_by(0.1)
            .suffix("s").text("Min Phrase Length"),)
            .changed()
        {
            config::set_min_phrase_secs(min_phrase);
        }

        let mut max_phrase = config::max_phrase_samples() as f32 / TARGET_SAMPLE_RATE as f32;
        if ui.add(egui::Slider::new(&mut max_phrase, 5.0..=60.0)
            .step_by(1.0)
            .suffix("s")
            .text("Max Phrase Length"),)
            .changed()
        {
            config::set_max_phrase_secs(max_phrase);
        }
    }
    fn draw_transcription(&self, ui: &mut Ui, words: &Vec<&str>, data: &PhraseData, id: &u32) {
        let trans = self.state.translations.get(id).cloned().unwrap_or_default();
        ui.spacing_mut().item_spacing = egui::vec2(8.0, 4.0);
        ui.weak(format!("[{id}]"));

        let mut i = 0;
        while i < words.len() {
            let translation_match = trans.iter().find(|(wi, _, _)| *wi == i);

            if let Some((_wi, span, translated_text)) = translation_match {
                let span = *span;
                ui.vertical(|ui| {
                    ui.horizontal(|ui| {
                        for j in 0..span {
                            if i + j < words.len() {
                                ui.colored_label(egui::Color32::LIGHT_GREEN, words[i + j]);
                            }
                        }
                    });
                    ui.colored_label(egui::Color32::RED, translated_text);
                });
                i += span;
            } else {
                ui.vertical(|ui| {
                    ui.colored_label(egui::Color32::LIGHT_GREEN, words[i]);
                    ui.colored_label(egui::Color32::TRANSPARENT, " "); 
                });
                i += 1;
            }
        }

        ui.weak(format!("({:.1}s | RTF: {:.2})", data.duration_s, data.rtf));
    }
    fn reset_transcription(&mut self) {
        self.state.transcription.clear();
        self.state.translations.clear();
        self.state.phrases_signaled.clear();
        self.state.phrase_rects.clear();
    }
    fn main_draw_misc(&mut self, ui: &mut Ui) {
        let mut dump = config::dump_audio();
        if ui.checkbox(&mut dump, "Dump audio").changed() {
            config::set_dump_audio(dump);
        }

        let mut check_val = self
            .state.save_transcription
            .load(std::sync::atomic::Ordering::Relaxed);
        if ui.checkbox(&mut check_val, "Save transcription to disk")
            .changed()
        {
            self.state.save_transcription
                .store(check_val, std::sync::atomic::Ordering::Relaxed);
        }

        let trans_muted = config::TRANSLATION_MUTED.load(Ordering::Relaxed);
        let (tr_label, tr_color) = if trans_muted {
            ("📝 Enable Translation", egui::Color32::LIGHT_GREEN)
        } else {
            ("📝 Disable Translation", egui::Color32::LIGHT_GRAY)
        };
        if ui.add(egui::Button::new(egui::RichText::new(tr_label).color(tr_color),))
            .clicked()
        {
            config::TRANSLATION_MUTED.store(!trans_muted, Ordering::Relaxed);
        }

        if ui.button("Reset Transcription").clicked() {
            self.reset_transcription();
        }

        ui.add_space(8.0);

        ui.group(|ui| {
            ui.label(egui::RichText::new("Dictionary (Live)").strong());

            ui.horizontal(|ui| {
                ui.add(egui::TextEdit::singleline(&mut self.state.dict_new_word)
                    .desired_width(120.0)
                    .hint_text("Word"),
                );
                ui.label("->");
                ui.add(egui::TextEdit::singleline(&mut self.state.dict_new_trans)
                    .desired_width(120.0)
                    .hint_text("Translation"),
                );
            });

            if ui.button("Add to custom.toml").clicked()
                && !self.state.dict_new_word.is_empty()
                    && !self.state.dict_new_trans.is_empty()
                {
                    crate::utility::utils::add_to_custom_dict(
                        &self.state.dict_new_word,
                        &self.state.dict_new_trans,
                    );
                    self.state.dict_new_word.clear();
                    self.state.dict_new_trans.clear();
                }

            ui.add_space(16.0);
        });
    }
    fn draw_analytics(&mut self, ui: &mut egui::Ui) {
        let s = stats::get();

        ui.add_space(8.0);

        let speaking = s.is_speaking.load(Ordering::Relaxed);
        ui.horizontal(|ui| {
            if speaking {
                let (rect, _) =
                    ui.allocate_exact_size(egui::vec2(14.0, 14.0), egui::Sense::hover());
                ui.painter()
                    .circle_filled(rect.center(), 6.0, egui::Color32::RED);
                ui.colored_label(egui::Color32::RED, "VAD: SPEAKING");
            } else {
                let (rect, _) =
                    ui.allocate_exact_size(egui::vec2(14.0, 14.0), egui::Sense::hover());
                ui.painter()
                    .circle_filled(rect.center(), 6.0, egui::Color32::DARK_GRAY);
                ui.weak("VAD: silent");
            }
        });

        if config::SINGLE_PASS.load(Ordering::Relaxed) {
            ui.horizontal(|ui| {
                ui.colored_label(egui::Color32::YELLOW, "⚡");
                ui.colored_label(egui::Color32::YELLOW, "Single-Pass Mode (models identical)");
            });
        }

        ui.add_space(6.0);

        ui.label(egui::RichText::new("Pipeline Events").strong());

        let events = s.events_snapshot();
        let now = std::time::Instant::now();

        egui::ScrollArea::vertical()
            .id_salt("pipeline_events_scroll")
            .max_height(130.0)
            .stick_to_bottom(true)
            .show(ui, |ui| {
                if events.is_empty() {
                    ui.weak("(no events yet)");
                }
                for ev in events.iter().rev().take(30) {
                    let ago_ms = now.duration_since(ev.at).as_millis();
                    ui.horizontal(|ui| {
                        match &ev.kind {
                            PipelineEventKind::Pass1Start => {
                                ui.colored_label(egui::Color32::from_rgb(255, 200, 60), "[P1]");
                                ui.weak(format!("pid={} started", ev.phrase_id));
                            }
                            PipelineEventKind::Pass1End { rtf, text_len } => {
                                ui.colored_label(egui::Color32::from_rgb(255, 200, 60), "[P1]");
                                ui.label(format!(
                                    "pid={} done  RTF:{:.2}  {}ch",
                                    ev.phrase_id, rtf, text_len
                                ));
                            }
                            PipelineEventKind::Pass2Start => {
                                ui.colored_label(egui::Color32::from_rgb(100, 180, 255), "[P2]");
                                ui.weak(format!("pid={} started", ev.phrase_id));
                            }
                            PipelineEventKind::Pass2End { rtf, text_len } => {
                                ui.colored_label(egui::Color32::from_rgb(100, 180, 255), "[P2]");
                                ui.label(format!(
                                    "pid={} done  RTF:{:.2}  {}ch",
                                    ev.phrase_id, rtf, text_len
                                ));
                            }
                        }
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            ui.weak(format!("{:.1}s ago", ago_ms as f32 / 1000.0));
                        });
                    });
                }
            });

        ui.add_space(6.0);
        ui.separator();

        ui.label(egui::RichText::new("Drop Counters").strong());
        egui::Grid::new("drop_counters")
            .num_columns(2)
            .spacing([10.0, 2.0])
            .show(ui, |ui| {
                ui.weak("Audio overflow:");
                ui.label(s.audio_overflow.load(Ordering::Relaxed).to_string());
                ui.end_row();

                ui.weak("VAD -> P1 drop:");
                ui.label(s.vad_pass1_dropped.load(Ordering::Relaxed).to_string());
                ui.end_row();

                ui.weak("VAD -> P2 drop:");
                ui.label(s.vad_pass2_dropped.load(Ordering::Relaxed).to_string());
                ui.end_row();

                ui.weak("P1 worker drop:");
                ui.label(s.pass1_dropped.load(Ordering::Relaxed).to_string());
                ui.end_row();

                ui.weak("P2 worker drop:");
                ui.label(s.pass2_dropped.load(Ordering::Relaxed).to_string());
                ui.end_row();
            });

        ui.add_space(4.0);

        ui.label(egui::RichText::new("Averages").strong());
        egui::Grid::new("avg_stats")
            .num_columns(2)
            .spacing([10.0, 2.0])
            .show(ui, |ui| {
                let p1_count = s.pass1_count.load(Ordering::Relaxed);
                let p2_count = s.pass2_count.load(Ordering::Relaxed);

                ui.weak("Pass 1 RTF:");
                ui.label(format!("{:.3}  ({})", s.avg_pass1_rtf(), p1_count));
                ui.end_row();

                ui.weak("Pass 2 RTF:");
                ui.label(format!("{:.3}  ({})", s.avg_pass2_rtf(), p2_count));
                ui.end_row();

                ui.weak("Avg audio dur:");
                ui.label(format!("{:.2}s", s.avg_audio_dur_s()));
                ui.end_row();

                ui.weak("Phrases P1:");
                ui.label(s.phrases_p1.load(Ordering::Relaxed).to_string());
                ui.end_row();

                ui.weak("Phrases P2:");
                ui.label(s.phrases_p2.load(Ordering::Relaxed).to_string());
                ui.end_row();
            });

        ui.add_space(4.0);
        if ui.small_button("Reset Stats").clicked() {
            stats::get().reset();
        }
        ui.add_space(8.0);
    }
    fn select_device(&mut self, ctx: &egui::Context) {
        if self.state.selected_device != self.state.last_selected {
            self.preview_stream = audio::start_preview(&self.state.selected_device);
            self.state.last_selected = self.state.selected_device.clone();
        }
        egui::SidePanel::right("model_status").show(ctx, |ui| {

            self.draw_model_status(ctx, ui);

        });
        egui::CentralPanel::default().show(ctx, |ui| {
            egui::ScrollArea::both()
                .auto_shrink([false; 2])
                .show(ui, |ui| {
                    ui.add_space(20.0);
                    ui.heading("Translator Setup");
                    ui.add_space(20.0);

                    self.draw_audio_input(ui);
                    
                    ui.add_space(10.0);

                    ui.group(|ui| {
                        self.draw_language(ui);

                        ui.separator();

                        self.draw_gpu(ui);
                    });

                    ui.add_space(30.0);

                    if ui
                    .add(egui::Button::new(egui::RichText::new("START").heading())
                        .min_size(egui::vec2(200.0, 50.0)),
                    ).clicked() {
                        self.preview_stream = None;
                        let (tx, rx) = mpsc::channel(100);
                        self.state.save_tx = Some(tx);
                        let should_save = Arc::clone(&self.state.save_transcription);
                        let path = self.state.transcript_path.clone();
                        self.handle
                            .spawn(crate::utility::utils::recording_task(
                                rx,
                                path,
                                should_save,
                            ));
                        config::init(
                            self.pending_config.use_gpu_fast,
                            self.pending_config.use_gpu_acc,
                            self.pending_config.gpu_device_fast,
                            self.pending_config.gpu_device_acc,
                            self.pending_config.audio_gain,
                        );
                        config::set_device(self.state.selected_device.clone());
                        config::save_to_toml("config.toml");

                        let identical = models_are_identical();
                        config::SINGLE_PASS.store(identical, Ordering::Relaxed);
                        if identical {
                            tracing::info!("Models are identical — enabling single-pass mode");
                        }
                        if let Some(tx) = self.startup_tx.take() {
                            let _ = tx.send(());
                        }

                        if let Some(tx) = self.device_tx.take() {
                            let _ = tx.send(self.state.selected_device.clone());
                        }
                        self.state.is_running = true;
                    }
                });
        });
    }
    fn show_model_row(&mut self, ui: &mut egui::Ui, ctx: &egui::Context, t: ModelType) {
        let info = ModelInfo::get(&t);
        let found = info.path.exists();

        let is_downloading = self.state.downloading_models.lock().unwrap().contains(info.name);
        let error_msg = self.state.download_errors.lock().unwrap().get(info.name).cloned();

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
                                self.state.downloading_models
                                    .lock()
                                    .unwrap()
                                    .insert(info.name.to_string());
                                self.download_model(
                                    ctx.clone(),
                                    info.name,
                                    info.url,
                                    info.path.clone(),
                                );
                            }
                        }
                    });
                }
            });

            if let Some(err) = error_msg {
                ui.add_space(-4.0);
                ui.horizontal(|ui| {
                    ui.add_space(24.0);
                    ui.colored_label(
                        egui::Color32::LIGHT_RED,
                        egui::RichText::new(err.to_string()).small(),
                    );
                });
            }
        });
    }
    fn download_model(
        &self,
        ctx: egui::Context,
        name: &'static str,
        url: &'static str,
        target: std::path::PathBuf,
    ) {
        let url = url.to_string();
        let name = name.to_string();
        let downloading_models = self.state.downloading_models.clone();
        let download_errors = self.state.download_errors.clone(); // Клонируем доступ к ошибкам

        std::thread::spawn(move || {
            download_errors.lock().unwrap().remove(&name);
            let result: Result<(), String> = (|| {
                if let Some(parent) = target.parent() {
                    std::fs::create_dir_all(parent).map_err(|e| format!("Folder error: {}", e))?;
                }

                let response =
                    reqwest::blocking::get(&url).map_err(|e| format!("Network error: {}", e))?;

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
        self.state.available_devices = audio::get_input_devices();
    }
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        ctx.request_repaint_after(std::time::Duration::from_millis(33));
        if !self.state.is_running {
            self.select_device(ctx);
            return;
        }
        self.update_data(ctx);
        if self.state.show_settings {
            egui::SidePanel::right("settings")
                .min_width(330.0)
                .max_width(330.0)
                .resizable(false)
                .show(ctx, |ui| {
                    egui::ScrollArea::both()
                        .id_salt("settings_scroll")
                        .auto_shrink([false; 2])
                        .show(ui, |ui| {
                            ui.heading("Settings");
                            ui.add_space(8.0);

                            self.main_draw_language(ui);

                            ui.group(|ui| {
                                self.main_draw_audio(ui);
                            });

                            ui.add_space(8.0);

                            ui.group(|ui| {
                                self.main_draw_vad(ui);
                            });

                            ui.add_space(8.0);

                            ui.group(|ui| {
                                self.main_draw_engine(ui);
                            });

                            ui.add_space(8.0);

                            ui.group(|ui| {
                                self.main_draw_phrases(ui);
                            });

                            ui.add_space(8.0);

                            self.main_draw_misc(ui);

                            ui.add_space(8.0);
                            let header = egui::RichText::new("Analytics").strong();
                            egui::CollapsingHeader::new(header)
                                .id_salt("analytics_header")
                                .default_open(false)
                                .show(ui, |ui| {
                                    self.draw_analytics(ui);
                                });
                        });
                });
        }

        egui::CentralPanel::default().show(ctx, |ui| {
            ui.horizontal(|ui| {
                let btn_text = if self.state.show_settings {
                    "Hide Settings"
                } else {
                    "Show Settings"
                };
                if ui.button(btn_text).clicked() {
                    self.state.show_settings = !self.state.show_settings;
                }
                ui.add_space(8.0);
                ui.heading(":) Live Transcription");

                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    let speaking = stats::get().is_speaking.load(Ordering::Relaxed);
                    let (dot_color, label) = if speaking {
                        (egui::Color32::RED, "⏺")
                    } else {
                        (egui::Color32::DARK_GRAY, "⏺")
                    };
                    ui.colored_label(dot_color, label)
                        .on_hover_text(if speaking {
                            "VAD: Speaking"
                        } else {
                            "VAD: Silent"
                        });

                    if config::AUDIO_MUTED.load(Ordering::Relaxed) {
                        ui.colored_label(egui::Color32::RED, "🔇 MUTED");
                    }
                    if config::TRANSLATION_MUTED.load(Ordering::Relaxed) {
                        ui.colored_label(egui::Color32::from_rgb(180, 120, 0), "📝 OFF");
                    }
                });
            });
            ui.separator();
            let available_width = ui.available_width();
            if (available_width - self.state.last_available_width).abs() > 1.0 {
                self.state.phrase_rects.clear();
                self.state.last_available_width = available_width;
            }
            egui::ScrollArea::both()
                .stick_to_bottom(true)
                .show(ui, |ui| {
                    for (id, data) in &self.state.transcription {
                        if data.is_final {
                            let words: Vec<&str> = data.text.split_whitespace().collect();
                            
                            ui.horizontal_wrapped(|ui| {
                                self.draw_transcription(ui, &words, &data, &id);
                            });

                            if !self.state.phrases_signaled.contains(id) && !words.is_empty() {
                                self.state.phrases_signaled.insert(*id);
                                let buf = Arc::clone(&self.translation_buffer);
                                let pid = *id;
                                self.handle.spawn(async move {
                                    buf.signal_ready(pid).await;
                                });
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
