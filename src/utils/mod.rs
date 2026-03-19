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
    if new.contains(old) { return new.to_string(); }
    
    let old_words: Vec<&str> = old.split_whitespace().collect();
    let new_words: Vec<&str> = new.split_whitespace().collect();
    
    for i in 0..old_words.len() {
        if new_words.starts_with(&old_words[i..]) {
            let mut res = old_words[0..i].to_vec();
            res.extend(new_words);
            return res.join(" ");
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