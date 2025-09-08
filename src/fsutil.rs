use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use walkdir::WalkDir;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileMeta {
    pub path: String,
    pub size: u64,
    pub ext: Option<String>,
}

pub fn file_inventory(root: &Path) -> Result<Vec<FileMeta>> {
    let mut out = Vec::new();
    for entry in WalkDir::new(root)
        .follow_links(false)
        .into_iter()
        .filter_entry(|e| {
            let p = e.path();
            let name = p.file_name().and_then(|s| s.to_str()).unwrap_or("");
            !name.starts_with('.')
                && name != "target"
                && name != "node_modules"
                && name != "dist"
                && name != "build"
        })
    {
        let entry = match entry {
            Ok(e) => e,
            Err(_) => continue,
        };
        let p = entry.path();
        if p.is_file() {
            if let Ok(md) = p.metadata() {
                let rel = diff_paths(p, root);
                out.push(FileMeta {
                    path: rel.to_string_lossy().to_string(),
                    size: md.len(),
                    ext: p
                        .extension()
                        .and_then(|s| s.to_str())
                        .map(|s| s.to_string()),
                });
            }
        }
    }
    Ok(out)
}

pub fn atomic_write(path: &Path, content: &str) -> Result<()> {
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    std::fs::create_dir_all(parent)?;
    let tmp = path.with_extension("tmp.write");
    fs::write(&tmp, content)?;
    fs::rename(&tmp, path)?;
    Ok(())
}

pub fn read_to_string(p: &Path) -> Result<String> {
    Ok(fs::read_to_string(p)?)
}

fn diff_paths(p: &Path, root: &Path) -> PathBuf {
    pathdiff::diff_paths(p, root).unwrap_or_else(|| p.to_path_buf())
}

/// Append ignore patterns to .gitignore (dedup).
pub fn merge_ignore_patterns(patterns: &[&str]) {
    let path = Path::new(".gitignore");
    let mut existing = String::new();
    if path.exists() {
        if let Ok(s) = fs::read_to_string(path) {
            existing = s;
        }
    }
    let mut lines: Vec<String> = existing.lines().map(|s| s.to_string()).collect();
    let mut changed = false;
    for pat in patterns {
        if !lines.iter().any(|l| l.trim() == *pat) {
            lines.push(pat.to_string());
            changed = true;
        }
    }
    if changed {
        if let Ok(mut f) = fs::OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(path)
        {
            let _ = writeln!(f, "{}", lines.join("\n"));
        }
    }
}

/// Remove a file or directory recursively.
pub fn remove_path(p: &Path) -> Result<()> {
    if p.is_dir() {
        fs::remove_dir_all(p)?;
    } else if p.is_file() {
        fs::remove_file(p)?;
    }
    Ok(())
}
