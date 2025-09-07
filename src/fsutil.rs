use anyhow::Result;
use ignore::WalkBuilder;
use once_cell::sync::Lazy;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

static IGNORE_PATTERNS: Lazy<Mutex<Vec<String>>> = Lazy::new(|| {
    Mutex::new(
        vec![
            "node_modules", "dist", "build", "target", ".git", ".next", ".turbo",
        ]
        .into_iter()
        .map(String::from)
        .collect(),
    )
});

#[derive(Debug, Clone, serde::Serialize)]
pub struct FileMeta {
    pub path: String,
    pub size: u64,
    pub ext: Option<String>,
}

pub fn file_inventory(root: &Path) -> Result<Vec<FileMeta>> {
    let mut out = vec![];
    let walker = WalkBuilder::new(root)
        .hidden(false)
        .git_ignore(true)
        .git_global(true)
        .git_exclude(true)
        .build();

    let ignore = IGNORE_PATTERNS.lock().unwrap();

    for dent in walker {
        let dent = match dent {
            Ok(d) => d,
            Err(_) => continue,
        };
        let p = dent.path();
        if !p.is_file() { continue; }
        if contains_any_segment_str(p, &ignore) { continue; }

        let rel = pathdiff::diff_paths(p, root).unwrap_or_else(|| p.to_path_buf());
        let size = std::fs::metadata(p).map(|m| m.len()).unwrap_or(0);
        let ext = p.extension().map(|e| e.to_string_lossy().to_string());
        out.push(FileMeta {
            path: rel.to_string_lossy().to_string(),
            size,
            ext,
        });
    }
    out.sort_by(|a, b| a.path.cmp(&b.path));
    Ok(out)
}

pub fn list_project_files(root: &Path) -> Result<Vec<PathBuf>> {
    let inv = file_inventory(root)?;
    Ok(inv.into_iter().map(|m| root.join(m.path)).collect())
}

fn contains_any_segment_str(p: &Path, segs: &[String]) -> bool {
    p.components().any(|c| {
        let s = c.as_os_str().to_string_lossy();
        segs.iter().any(|needle| s == needle.as_str())
    })
}

pub fn read_to_string(p: &Path) -> Result<String> {
    Ok(std::fs::read_to_string(p)?)
}

pub fn atomic_write(p: &Path, content: &str) -> Result<()> {
    if let Some(parent) = p.parent() {
        fs::create_dir_all(parent)?;
    }
    let tmp = p.with_extension("tmp~agent");
    {
        let mut f = fs::File::create(&tmp)?;
        f.write_all(content.as_bytes())?;
        f.sync_all()?;
    }
    fs::rename(tmp, p)?;
    Ok(())
}

/// Merge additional ignore patterns at runtime.
pub fn merge_ignore_patterns(new: &[&str]) {
    let mut guard = IGNORE_PATTERNS.lock().unwrap();
    for &pat in new {
        if !guard.iter().any(|existing| existing == pat) {
            guard.push(pat.to_string());
        }
    }
}
