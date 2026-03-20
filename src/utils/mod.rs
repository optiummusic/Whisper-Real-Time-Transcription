use std::path::{ Path, PathBuf };
use std::fs;
use crate::config;

pub fn append_context(ctx: &mut String, text: &str, max_words: usize) {
    if text.is_empty() { return; }
 
    if !ctx.is_empty() { ctx.push(' '); }
    ctx.push_str(text.trim());
 
    let words: Vec<&str> = ctx.split_whitespace().collect();
    if words.len() > max_words {
        *ctx = words[words.len() - max_words..].join(" ");
    }
}

pub fn merge_strings(old: &str, new: &str) -> String {
    if old.is_empty() { return new.to_string(); }
    if new.is_empty() { return old.to_string(); }
    if new.contains(old.trim()) { return new.to_string(); }
 
    let old_words: Vec<&str> = old.split_whitespace().collect();
    let new_words: Vec<&str> = new.split_whitespace().collect();
 
    let max_overlap = old_words.len().min(new_words.len());
 
    for overlap in (1..=max_overlap).rev() {
        let old_suffix = &old_words[old_words.len() - overlap..];
        let new_prefix = &new_words[..overlap];
        if old_suffix == new_prefix {
            let mut result = old_words[..old_words.len() - overlap].to_vec();
            result.extend_from_slice(&new_words);
            return result.join(" ");
        }
    }
    new.to_string()
}

pub fn get_model_path(relative_path: &str) -> PathBuf {
    let exe_path = std::env::current_exe().expect("Failed to get current exe path");
    let exe_dir = exe_path.parent().expect("Failed to get exe parent");

    let path_near_exe = exe_dir.join(relative_path);
    if path_near_exe.exists() { return path_near_exe; }

    if let Some(project_root) = exe_dir.parent().and_then(|p| p.parent()) {
        let path_in_root = project_root.join(relative_path);
        if path_in_root.exists() { return path_in_root; }
    }

    PathBuf::from(relative_path)
}

pub fn find_first_file_in_dir(relative_dir: &str, extension: &str) -> Option<PathBuf> {
    let dir_path = get_model_path(relative_dir);
    
    if let Ok(entries) = fs::read_dir(dir_path) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_file() && path.extension().and_then(|s| s.to_str()) == Some(extension) {
                return Some(path);
            }
        }
    }
    None
}

pub fn prepare_debug_dir() {
    if !config::dump_audio() { return; }
    let dir = "debug_audio";
    
    if Path::new(dir).exists() {
        let _ = fs::remove_dir_all(dir);
    }
    
    let _ = fs::create_dir_all(dir);
}