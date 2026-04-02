use eframe::egui;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;

// ── Language config ──────────────────────────────────────────────────────────

const LANGS: &[(&str, &str)] = &[
    ("en", "English"),
    ("uk", "Ukrainian"),
    ("de", "German"),
    ("fr", "French"),
    ("es", "Spanish"),
    ("pl", "Polish"),
    ("ru", "Russian"),
];

fn flag(lang: &str) -> &'static str {
    match lang {
        "en" => "🇬🇧",
        "uk" => "🇺🇦",
        "de" => "🇩🇪",
        "fr" => "🇫🇷",
        "es" => "🇪🇸",
        "pl" => "🇵🇱",
        "ru" => "🇷🇺",
        _ => "🌐",
    }
}

// ── TOML data model ──────────────────────────────────────────────────────────

/// One [[concepts]] block.
/// `base` is flattened into the top-level keys (en = "…", uk = "…", …).
/// `custom` and `generated` are sub-tables.
#[derive(Serialize, Deserialize, Default, Clone, Debug)]
struct Concept {
    #[serde(flatten)]
    base: HashMap<String, String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    custom: Option<HashMap<String, Vec<String>>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    generated: Option<HashMap<String, Vec<String>>>,
}

#[derive(Serialize, Deserialize, Clone, Default)]
struct LxdbToml {
    #[serde(default)]
    meta: HashMap<String, toml::Value>,
    #[serde(default)]
    languages: HashMap<String, String>,
    #[serde(default)]
    concepts: Vec<Concept>,
}

// ── Edit pane (UI mirror of one Concept) ────────────────────────────────────

struct EditPane {
    base: HashMap<String, String>,
    /// Each value is comma-joined for single-line editing
    custom: HashMap<String, String>,
    generated: HashMap<String, String>,
}

impl EditPane {
    fn blank() -> Self {
        let mk = || {
            LANGS
                .iter()
                .map(|(l, _)| (l.to_string(), String::new()))
                .collect::<HashMap<_, _>>()
        };
        Self { base: mk(), custom: mk(), generated: mk() }
    }

    fn from_concept(c: &Concept) -> Self {
        let mut p = Self::blank();
        for (lang, _) in LANGS {
            if let Some(v) = c.base.get(*lang) {
                p.base.insert(lang.to_string(), v.clone());
            }
            if let Some(cm) = &c.custom {
                if let Some(v) = cm.get(*lang) {
                    p.custom.insert(lang.to_string(), v.join(", "));
                }
            }
            if let Some(gm) = &c.generated {
                if let Some(v) = gm.get(*lang) {
                    p.generated.insert(lang.to_string(), v.join(", "));
                }
            }
        }
        p
    }

    fn to_concept(&self) -> Concept {
        let parse_csv = |m: &HashMap<String, String>| -> HashMap<String, Vec<String>> {
            m.iter()
                .filter(|(_, v)| !v.trim().is_empty())
                .map(|(k, v)| {
                    let list = v
                        .split(',')
                        .map(|s| s.trim().to_string())
                        .filter(|s| !s.is_empty())
                        .collect();
                    (k.clone(), list)
                })
                .collect()
        };

        let base: HashMap<String, String> = self
            .base
            .iter()
            .filter(|(_, v)| !v.trim().is_empty())
            .map(|(k, v)| (k.clone(), v.trim().to_string()))
            .collect();

        let custom = parse_csv(&self.custom);
        let generated = parse_csv(&self.generated);

        Concept {
            base,
            custom: if custom.is_empty() { None } else { Some(custom) },
            generated: if generated.is_empty() { None } else { Some(generated) },
        }
    }
}

// ── App state ────────────────────────────────────────────────────────────────

/// What's visible in the right panel.
#[derive(Clone, Copy, PartialEq)]
enum Panel {
    Empty,
    Edit(Option<usize>), // None = new concept
}

/// Deferred actions to avoid borrow conflicts in closures.
enum Act {
    Refresh,
    OpenFile(PathBuf),
    SelectConcept(usize),
    NewConcept,
    Save,
    Delete(usize),
    CancelEdit,
}

struct App {
    dict_path: PathBuf,
    toml_files: Vec<PathBuf>,
    sel_file: Option<PathBuf>,
    data: Option<LxdbToml>,
    panel: Panel,
    edit: EditPane,
    search: String,
    status: String,
}

impl App {
    fn new(_cc: &eframe::CreationContext<'_>) -> Self {
        let dict_path = PathBuf::from("../dictionary");
        println!("Path is: {:?}", dict_path);
        let toml_files = Self::scan_toml(&dict_path);
        Self {
            dict_path,
            toml_files,
            sel_file: None,
            data: None,
            panel: Panel::Empty,
            edit: EditPane::blank(),
            search: String::new(),
            status: "Select a .toml file to begin".into(),
        }
    }

    fn scan_toml(path: &PathBuf) -> Vec<PathBuf> {
        fs::read_dir(path)
            .map(|rd| {
                let mut files: Vec<PathBuf> = rd
                    .filter_map(|e| e.ok())
                    .map(|e| e.path())
                    .filter(|p| p.extension().map_or(false, |e| e == "toml"))
                    .collect();
                files.sort();
                files
            })
            .unwrap_or_default()
    }

    fn concept_title(c: &Concept) -> String {
        c.base
            .get("en")
            .cloned()
            .or_else(|| c.base.values().next().cloned())
            .unwrap_or_else(|| "—".to_string())
    }

    fn flush(&mut self) {
        if let (Some(path), Some(data)) = (&self.sel_file.clone(), &self.data) {
            match toml::to_string(data) {
                Ok(dst) => {
                    if let Err(e) = fs::write(path, dst) {
                        self.status = format!("Write error: {e}");
                    } else {
                        self.status = "✓ Saved (Compact mode)".into();
                    }
                }
                Err(e) => self.status = format!("Serialization error: {e}"),
            }
        }
    }

    fn apply(&mut self, act: Act) {
        match act {
            Act::Refresh => {
                self.toml_files = Self::scan_toml(&self.dict_path);
                self.status = format!("Found {} file(s)", self.toml_files.len());
            }
            Act::OpenFile(p) => {
                match fs::read_to_string(&p)
                    .map_err(|e| e.to_string())
                    .and_then(|s| toml::from_str::<LxdbToml>(&s).map_err(|e| e.to_string()))
                {
                    Ok(data) => {
                        let n = data.concepts.len();
                        self.data = Some(data);
                        self.sel_file = Some(p);
                        self.panel = Panel::Empty;
                        self.status = format!("Loaded — {n} concepts");
                    }
                    Err(e) => self.status = format!("Error: {e}"),
                }
            }
            Act::NewConcept => {
                self.edit = EditPane::blank();
                self.panel = Panel::Edit(None);
            }
            Act::SelectConcept(i) => {
                if let Some(data) = &self.data {
                    self.edit = EditPane::from_concept(&data.concepts[i]);
                    self.panel = Panel::Edit(Some(i));
                }
            }
            Act::Save => {
                let concept = self.edit.to_concept();
                if concept.base.is_empty() {
                    self.status = "Error: at least one base word is required".into();
                    return;
                }
                if let Some(data) = &mut self.data {
                    match self.panel {
                        Panel::Edit(Some(i)) => {
                            data.concepts[i] = concept;
                        }
                        Panel::Edit(None) => {
                            data.concepts.push(concept);
                            let new_i = data.concepts.len() - 1;
                            self.panel = Panel::Edit(Some(new_i));
                        }
                        _ => {}
                    }
                }
                self.flush();
            }
            Act::Delete(i) => {
                if let Some(data) = &mut self.data {
                    data.concepts.remove(i);
                }
                self.panel = Panel::Empty;
                self.flush();
            }
            Act::CancelEdit => {
                self.panel = Panel::Empty;
            }
        }
    }
}

// ── UI helpers ────────────────────────────────────────────────────────────────

/// Render a language grid (label | text field) for a given HashMap.
fn lang_grid(
    ui: &mut egui::Ui,
    grid_id: &str,
    map: &mut HashMap<String, String>,
    multiline: bool,
) {
    egui::Grid::new(grid_id)
        .num_columns(2)
        .min_col_width(100.0)
        .spacing([10.0, 5.0])
        .show(ui, |ui| {
            for (lang, lang_name) in LANGS {
                ui.label(
                    egui::RichText::new(format!("{} {}", flag(lang), lang_name)).small(),
                );
                if let Some(val) = map.get_mut(*lang) {
                    let widget = if multiline {
                        egui::TextEdit::multiline(val)
                            .desired_rows(2)
                            .desired_width(f32::INFINITY)
                    } else {
                        egui::TextEdit::singleline(val).desired_width(f32::INFINITY)
                    };
                    ui.add(widget);
                }
                ui.end_row();
            }
        });
}

// ── eframe::App ───────────────────────────────────────────────────────────────

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        let mut act: Option<Act> = None;

        // ── Pre-compute display data (avoids borrow conflicts inside closures) ─
        let editing: Option<Option<usize>> = match self.panel {
            Panel::Empty => None,
            Panel::Edit(idx) => Some(idx),
        };

        // Clone labels once so the left panel closure doesn't borrow self.data
        let concept_labels: Vec<String> = self
            .data
            .as_ref()
            .map(|d| d.concepts.iter().map(Self::concept_title).collect())
            .unwrap_or_default();

        let concept_count = concept_labels.len();

        let sel_file_name = self
            .sel_file
            .as_ref()
            .and_then(|p| p.file_name())
            .and_then(|n| n.to_str())
            .unwrap_or("— choose file —")
            .to_string();

        // ── Top bar ───────────────────────────────────────────────────────────
        egui::TopBottomPanel::top("top_bar").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.strong("LXDB Editor");
                ui.separator();

                if ui
                    .small_button("🔄")
                    .on_hover_text("Refresh file list")
                    .clicked()
                {
                    act = Some(Act::Refresh);
                }

                egui::ComboBox::from_id_salt("file_picker")
                    .selected_text(&sel_file_name)
                    .width(260.0)
                    .show_ui(ui, |ui| {
                        for path in &self.toml_files.clone() {
                            let name = path
                                .file_name()
                                .and_then(|n| n.to_str())
                                .unwrap_or("?");
                            if ui
                                .selectable_label(
                                    self.sel_file.as_ref() == Some(path),
                                    name,
                                )
                                .clicked()
                            {
                                act = Some(Act::OpenFile(path.clone()));
                            }
                        }
                        if self.toml_files.is_empty() {
                            ui.label(
                                egui::RichText::new(format!(
                                    "No .toml files in {:?}",
                                    self.dict_path
                                ))
                                .italics()
                                .weak(),
                            );
                        }
                    });

                ui.with_layout(
                    egui::Layout::right_to_left(egui::Align::Center),
                    |ui| {
                        ui.label(egui::RichText::new(&self.status).small().weak());
                    },
                );
            });
        });

        // ── No file loaded splash ─────────────────────────────────────────────
        if self.data.is_none() {
            egui::CentralPanel::default().show(ctx, |ui| {
                ui.centered_and_justified(|ui| {
                    ui.label("Select a .toml dictionary file from the dropdown above.");
                });
            });
            if let Some(a) = act {
                self.apply(a);
            }
            return;
        }

        // ── LEFT PANEL — concept list ─────────────────────────────────────────
        egui::SidePanel::left("concept_list")
            .resizable(true)
            .default_width(220.0)
            .min_width(140.0)
            .show(ctx, |ui| {
                ui.add_space(6.0);

                if ui
                    .add_sized([ui.available_width(), 0.0], egui::Button::new("+ New Concept"))
                    .clicked()
                {
                    act = Some(Act::NewConcept);
                }

                ui.add_space(6.0);
                ui.separator();
                ui.add_space(4.0);

                ui.horizontal(|ui| {
                    ui.label("🔍");
                    ui.add(
                        egui::TextEdit::singleline(&mut self.search)
                            .hint_text("Filter…")
                            .desired_width(f32::INFINITY),
                    );
                });

                ui.add_space(6.0);

                let q = self.search.to_lowercase();

                egui::ScrollArea::vertical().show(ui, |ui| {
                    for (idx, label) in concept_labels.iter().enumerate() {
                        if !q.is_empty() && !label.to_lowercase().contains(&q) {
                            continue;
                        }

                        let selected = editing == Some(Some(idx));

                        // Number badge + label
                        let text = format!("{:>4}.  {}", idx + 1, label);
                        let resp = ui.selectable_label(selected, text);

                        if resp.clicked() {
                            act = Some(Act::SelectConcept(idx));
                        }
                    }

                    ui.add_space(8.0);
                    ui.separator();
                    ui.add_space(4.0);
                    ui.label(
                        egui::RichText::new(format!("{concept_count} concept(s)"))
                            .small()
                            .weak(),
                    );
                });
            });

        // ── RIGHT PANEL — editor ──────────────────────────────────────────────
        egui::CentralPanel::default().show(ctx, |ui| {
            let Some(idx_opt) = editing else {
                // Nothing selected
                ui.centered_and_justified(|ui| {
                    ui.label("<- Select a concept to edit, or create a new one");
                });
                return;
            };

            // Heading
            let heading = match idx_opt {
                Some(i) => format!(
                    "Concept #{}  —  {}",
                    i + 1,
                    concept_labels.get(i).map(String::as_str).unwrap_or("?")
                ),
                None => "New Concept".into(),
            };
            ui.heading(&heading);
            ui.separator();

            egui::ScrollArea::vertical().show(ui, |ui| {
                // ── Base words ────────────────────────────────────────────────
                egui::CollapsingHeader::new("Base Words")
                    .default_open(true)
                    .show(ui, |ui| {
                        ui.add_space(4.0);
                        lang_grid(ui, "grid_base", &mut self.edit.base, false);
                        ui.add_space(4.0);
                    });

                ui.add_space(4.0);

                // ── Custom variants ───────────────────────────────────────────
                egui::CollapsingHeader::new("Custom Variants")
                    .default_open(false)
                    .show(ui, |ui| {
                        ui.add_space(2.0);
                        ui.label(
                            egui::RichText::new(
                                "Manually curated alternatives (comma-separated)",
                            )
                            .small()
                            .weak(),
                        );
                        ui.add_space(4.0);
                        lang_grid(ui, "grid_custom", &mut self.edit.custom, false);
                        ui.add_space(4.0);
                    });

                ui.add_space(4.0);

                // ── Generated forms ───────────────────────────────────────────
                egui::CollapsingHeader::new("Generated Forms")
                    .default_open(false)
                    .show(ui, |ui| {
                        ui.add_space(2.0);
                        ui.label(
                            egui::RichText::new(
                                "Inflected / morphological variants (comma-separated)",
                            )
                            .small()
                            .weak(),
                        );
                        ui.add_space(4.0);
                        lang_grid(ui, "grid_gen", &mut self.edit.generated, true);
                        ui.add_space(4.0);
                    });

                ui.add_space(14.0);
                ui.separator();
                ui.add_space(8.0);

                // ── Action buttons ────────────────────────────────────────────
                ui.horizontal(|ui| {
                    let save_label = if idx_opt.is_some() {
                        "Save"
                    } else {
                        "+ Add Concept"
                    };

                    if ui.button(save_label).clicked() {
                        act = Some(Act::Save);
                    }

                    if let Some(i) = idx_opt {
                        ui.add_space(8.0);
                        if ui
                            .button(
                                egui::RichText::new("Delete")
                                    .color(egui::Color32::from_rgb(210, 70, 70)),
                            )
                            .clicked()
                        {
                            act = Some(Act::Delete(i));
                        }
                    }

                    ui.with_layout(
                        egui::Layout::right_to_left(egui::Align::Center),
                        |ui| {
                            if ui.button("Cancel").clicked() {
                                act = Some(Act::CancelEdit);
                            }
                        },
                    );
                });

                ui.add_space(8.0);
            });
        });

        // ── Apply deferred action ─────────────────────────────────────────────
        if let Some(a) = act {
            self.apply(a);
        }
    }
}

// ── Entry point ───────────────────────────────────────────────────────────────

fn main() -> eframe::Result<()> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([960.0, 660.0])
            .with_min_inner_size([640.0, 420.0])
            .with_title("LXDB Editor"),
        ..Default::default()
    };

    eframe::run_native(
        "LXDB Editor",
        options,
        Box::new(|cc| Ok(Box::new(App::new(cc)))),
    )
}