use anyhow::{bail, Context, Result};
use std::{env, fs, io::Write, process::Command, thread, time::Duration};
use tempfile::NamedTempFile;
use which::which;

use crate::fsutil;
use crate::planner::PlannerAgent;

/// Returns true if the `DRY_RUN` environment variable is set to a truthy value.
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

// This function is intentionally kept for future editor integration.
// It may not be used currently, so we silence dead‑code warnings.
#[allow(dead_code)]
pub fn open_in_editor(initial: &str, _hint_name: &str) -> Result<String> {
    if is_dry_run() {
        // In dry‑run mode we simply return the original content.
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

/// `/ignore` support
// This function is intentionally kept for future editor integration.
// It may not be used currently, so we silence dead‑code warnings.
#[allow(dead_code)]
pub fn handle_ignore_command(arg_str: &str) -> Result<()> {
    if is_dry_run() {
        // Skip actual merging in dry‑run mode.
        return Ok(());
    }

    let patterns: Vec<String> = arg_str.split_whitespace().map(|s| s.to_string()).collect();
    if patterns.is_empty() {
        bail!("no ignore patterns provided");
    }

    // convert Vec<String> → Vec<&str>
    let refs: Vec<&str> = patterns.iter().map(|s| s.as_str()).collect();
    fsutil::merge_ignore_patterns(&refs);
    Ok(())
}

/// Execute ad‑hoc code snippets or files.
/// Shebang → runs interpreter; otherwise compiles as Rust via `rustc`.
pub fn execute_code(code: &str) -> Result<String, std::io::Error> {
    if is_dry_run() {
        // In dry‑run mode we avoid filesystem side‑effects and return an empty mock output.
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

    let bin_name = if cfg!(windows) {
        "code_bin.exe"
    } else {
        "code_bin"
    };
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

/// Apply a unified diff patch to the current working directory.
/// In dry‑run mode the function becomes a no‑op.
pub fn apply_patch(patch: &str) -> Result<()> {
    if is_dry_run() {
        // Skip actual patching when dry‑run is enabled.
        return Ok(());
    }

    // Write the patch to a temporary file.
    let mut tmp = NamedTempFile::new().context("creating temporary file for patch")?;
    tmp.write_all(patch.as_bytes())
        .context("writing patch to temporary file")?;
    let patch_path = tmp.path();

    // Run the `patch` command. We use `-p0` to apply paths as‑is.
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

/// Runs a given task (e.g., build or test) and, on failure, engages the
/// `PlannerAgent` to generate a fix, applies the resulting patch, and retries
/// until the task succeeds or the maximum number of attempts is reached.
///
/// The `task` closure should return `Ok(())` on success or an `anyhow::Error`
/// describing the failure.  `max_attempts` caps the retry loop to avoid infinite
/// retries.
///
/// This function embodies the **FixerAgent** self‑healing loop.
pub fn run_with_fixer<F>(mut task: F, max_attempts: usize) -> Result<()>
where
    F: FnMut() -> Result<()>,
{
    if max_attempts == 0 {
        bail!("max_attempts must be greater than zero");
    }

    let mut attempt = 0usize;

    loop {
        attempt += 1;
        match task() {
            Ok(_) => {
                // Task succeeded; exit the loop.
                return Ok(());
            }
            Err(err) => {
                if attempt >= max_attempts {
                    bail!(
                        "FixerAgent exhausted after {} attempts. Last error: {}",
                        attempt,
                        err
                    );
                }

                // Log the failure (could be replaced with a proper logger).
                eprintln!("Attempt {} failed: {}", attempt, err);

                // Ask the PlannerAgent for a patch that fixes the error.
                let planner = PlannerAgent::new();
                let error_msg = format!("{}", err);
                let patch = planner
                    .generate_fix(&error_msg)
                    .with_context(|| "PlannerAgent failed to generate a fix")?;

                // Apply the generated patch.
                apply_patch(&patch).with_context(|| "Failed to apply patch from PlannerAgent")?;

                // Optional back‑off before retrying.
                thread::sleep(Duration::from_millis(200));
                // Continue the loop to retry the task.
            }
        }
    }
}

/// Trait representing a summarizer provider. Implementors should attempt to
/// produce a summary for the given input and return an `anyhow::Result`.
pub trait SummarizerProvider {
    fn summarize(&self, input: &str) -> Result<String>;
}

/// Attempts to summarize `input` using the supplied list of `providers` in order.
/// If a provider succeeds, its result is returned immediately. If all providers
/// fail, a detailed error containing each provider's failure reason is returned,
/// allowing the UI to surface comprehensive diagnostics.
pub fn summarize_with_fallback(
    input: &str,
    providers: &[Box<dyn SummarizerProvider>],
) -> Result<String> {
    let mut error_messages = Vec::new();

    for provider in providers {
        match provider.summarize(input) {
            Ok(summary) => return Ok(summary),
            Err(e) => error_messages.push(e.to_string()),
        }
    }

    bail!(
        "All summarizer providers failed. Details: {}",
        error_messages.join("; ")
    );
}
