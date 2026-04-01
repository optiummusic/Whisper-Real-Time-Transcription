use crate::types::TranscriptEvent;
use crate::utility::utils::merge_strings;
use std::io::Write;
use tokio::sync::mpsc;
use tracing::{debug, info};

pub async fn display_task(mut rx: mpsc::Receiver<TranscriptEvent>) {
    let mut last_partial_id: Option<u32> = None;
    let mut finalized = std::collections::HashSet::<u32>::new();

    let mut session_text: std::collections::HashMap<u32, String> = std::collections::HashMap::new();

    while let Some(event) = rx.recv().await {
        match event {
            TranscriptEvent::Partial {
                phrase_id,
                chunk_id: _,
                text,
                ..
            } => {
                if finalized.contains(&phrase_id) {
                    debug!(phrase_id, "got partial for finalized phrase — ignoring");
                    continue;
                }
                let current_entry = session_text.entry(phrase_id).or_default();
                *current_entry = merge_strings(current_entry, &text);
                let display_text = current_entry.clone();

                print!("\r\x1b[2K\x1b[90m[{phrase_id}] ⏳ {}\x1b[0m", display_text);
                std::io::stdout().flush().ok();
                last_partial_id = Some(phrase_id);
            }
            TranscriptEvent::Final {
                phrase_id,
                text,
                duration_s,
                rtf,
                ..
            } => {
                finalized.insert(phrase_id);
                info!(phrase_id, duration_s, rtf, "final -> display");
                session_text.remove(&phrase_id);

                if last_partial_id == Some(phrase_id) {
                    print!("\r\x1b[2K");
                } else {
                    println!();
                }
                println!("\x1b[1m[{phrase_id}]\x1b[0m \x1b[32m✓\x1b[0m {text}");
                println!("\x1b[90m    └─ {duration_s:.1}s | RTF={rtf:.2}\x1b[0m");
                last_partial_id = None;
            }
        }
    }
}
