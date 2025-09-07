use anyhow::{bail, Context, Result};
use std::{env, fs, io::Write, process::Command, thread, time::Duration};
use tempfile::NamedTempFile;
use which::which;

use crate::fsutil;

/// Returns true if DRY_RUN is truthy.
fn is_dry_run() -> bool {
    match env::var("DRY_RUN") {
        Ok(val) => {
            let v = val.to_ascii_lowercase();
            v == "1" || v == "true" || v == "yes"
        }
        Err(_) => false,
    }
}

#[allow(dead_code)]
pub fn guess_editor() -> String {
    env::var("VISUAL")
        .or_else(|_| env::var("EDITOR"))
        .ok()
        .filter(|s| !s.trim().is_empty())
        .or_else(|| {
            for cand in ["nvim", "vim", "nano", "code", "notepad"] {
                if which(cand).is_ok() {
                    return Some(cand.to_string());
                }
            }
            None
        })
        .unwrap_or_else(|| "vi".into())
}

#[allow(dead_code)]
pub fn open_in_editor(initial: &str, _hint_name: &str) -> Result<String> {
    if is_dry_run() {
        return Ok(initial.to_string());
    }

    let mut tmp = NamedTempFile::new()?;
    tmp.write_all(initial.as_bytes())?;
    let path = tmp.path().to_path_buf();

    let editor = guess_editor();
    let mut cmd = Command::new(&editor);
    if editor.contains("code") {
        cmd.arg("--wait");
    }
    cmd.arg(&path);

    let status = cmd
        .status()
        .with_context(|| format!("launching editor: {}", editor))?;
    if !status.success() {
        bail!("editor exited with non-zero status");
    }

    let edited = fs::read_to_string(&path)?;
    Ok(edited)
}

/// `/ignore` support — appends patterns to .gitignore (deduplicated).
#[allow(dead_code)]
pub fn handle_ignore_command(arg_str: &str) -> Result<()> {
    if is_dry_run() {
        return Ok(());
    }
    let patterns: Vec<String> = arg_str.split_whitespace().map(|s| s.to_string()).collect();
    if patterns.is_empty() {
        bail!("no ignore patterns provided");
    }
    let refs: Vec<&str> = patterns.iter().map(|s| s.as_str()).collect();
    fsutil::merge_ignore_patterns(&refs);
    Ok(())
}

/// Execute ad-hoc code snippets or files (shebang or quick Rust).
pub fn execute_code(code: &str) -> Result<String, std::io::Error> {
    if is_dry_run() {
        return Ok(String::new());
    }

    let dir = tempfile::tempdir()?;
    let src_path = dir.path().join("code.tmp");
    fs::write(&src_path, code)?;

    let first_line = code.lines().next().unwrap_or("");
    if first_line.starts_with("#!") {
        let interpreter_line = first_line[2..].trim();
        let mut parts = interpreter_line.split_whitespace();
        let interpreter = match parts.next() {
            Some(p) => p,
            None => return Ok(String::new()),
        };
        let args: Vec<&str> = parts.collect();

        let output = std::process::Command::new(interpreter)
            .args(&args)
            .arg(&src_path)
            .output()?;

        let mut combined = String::new();
        combined.push_str(&String::from_utf8_lossy(&output.stdout));
        combined.push_str(&String::from_utf8_lossy(&output.stderr));
        return Ok(combined);
    }

    let bin_name = if cfg!(windows) { "code_bin.exe" } else { "code_bin" };
    let bin_path = dir.path().join(bin_name);

    let compile_output = std::process::Command::new("rustc")
        .arg(&src_path)
        .arg("-o")
        .arg(&bin_path)
        .output()?;

    if !compile_output.status.success() {
        let mut combined = String::new();
        combined.push_str(&String::from_utf8_lossy(&compile_output.stdout));
        combined.push_str(&String::from_utf8_lossy(&compile_output.stderr));
        return Ok(combined);
    }

    let run_output = std::process::Command::new(&bin_path).output()?;
    let mut combined = String::new();
    combined.push_str(&String::from_utf8_lossy(&run_output.stdout));
    combined.push_str(&String::from_utf8_lossy(&run_output.stderr));
    Ok(combined)
}

/// Apply a unified diff patch using the `patch` command.
pub fn apply_patch(patch: &str) -> Result<()> {
    if is_dry_run() {
        return Ok(());
    }

    let mut tmp = NamedTempFile::new().context("creating temporary file for patch")?;
    tmp.write_all(patch.as_bytes())
        .context("writing patch to temporary file")?;
    let patch_path = tmp.path();

    let status = Command::new("patch")
        .arg("-p0")
        .arg("-i")
        .arg(patch_path)
        .status()
        .context("executing patch command")?;

    if !status.success() {
        bail!("patch command failed with status: {}", status);
    }
    Ok(())
}

/// Simple self-healing loop that retries a task, optionally applying patches in-between.
#[allow(dead_code)]
pub fn run_with_fixer<F, G>(mut task: F, max_attempts: usize, propose_patch: G) -> anyhow::Result<()>
where
    F: FnMut() -> anyhow::Result<()>,
    G: Fn(&str) -> Option<String>,
{
    if max_attempts == 0 {
        anyhow::bail!("max_attempts must be greater than zero");
    }

    let mut attempt = 0usize;
    loop {
        attempt += 1;
        match task() {
            Ok(_) => return Ok(()),
            Err(err) => {
                if attempt >= max_attempts {
                    anyhow::bail!("Exhausted after {attempt} attempts: {err}");
                }
                eprintln!("Attempt {attempt} failed: {err}");
                if let Some(patch) = propose_patch(&format!("{err}")) {
                    let _ = apply_patch(&patch);
                }
                thread::sleep(Duration::from_millis(300));
            }
        }
    }
}
