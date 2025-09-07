use anyhow::{Context, Result};
use portable_pty::{native_pty_system, CommandBuilder, PtySize};
use std::fs::OpenOptions;
use std::io::Write;
use std::path::Path;

/// What we return to the caller; we don’t try to decode the raw status into a code
/// because portable_pty’s API varies by version.
#[derive(Debug)]
pub struct PtyRunResult {
    pub raw_status: portable_pty::ExitStatus,
    pub last_output: String,
}

/// Minimal PTY runner that works across portable_pty versions without relying on
/// ExitStatus enum variants or methods that might not exist.
///
/// Strategy:
/// - Spawn `/bin/bash -lc "cd <workdir> && <program> <args...>"` inside a pty
/// - Stream the master fd to a log file (best-effort)
/// - Wait for exit and return the raw ExitStatus
pub fn run_with_pty(
    program: &str,
    args: &[String],
    workdir: &Path,
    _env: &[(String, String)],
    log_path: &Path,
) -> Result<PtyRunResult> {
    let pty_system = native_pty_system();
    let pair = pty_system
        .openpty(PtySize {
            rows: 30,
            cols: 120,
            pixel_width: 0,
            pixel_height: 0,
        })
        .context("openpty failed")?;

    // Build a safe shell command: cd into workdir, then run program with args
    let mut cmd_line = String::new();
    cmd_line.push_str("cd ");
    cmd_line.push_str(&shell_quote_path(workdir));
    cmd_line.push_str(" && ");
    cmd_line.push_str(&shell_quote(program));
    for a in args {
        cmd_line.push(' ');
        cmd_line.push_str(&shell_quote(a));
    }

    // Spawn a shell that runs our command
    let mut cmd = CommandBuilder::new("/bin/bash");
    cmd.arg("-lc");
    cmd.arg(cmd_line);

    // Attach the slave end to the command and spawn
    let mut child = pair
        .slave
        .spawn_command(cmd)
        .context("spawn_command failed")?;

    // We will log output from the master end
    let mut reader = pair
        .master
        .try_clone_reader()
        .context("failed to clone pty reader")?;

    let mut log_file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(log_path)
        .with_context(|| format!("open log file {}", log_path.display()))?;

    // Drain output until the child exits.
    // We avoid fancy polling on try_wait() API differences. Instead, loop reading
    // and break once read returns 0 and wait() no longer blocks.
    let mut buf = [0u8; 8192];
    let mut last_output = String::new();

    loop {
        // Non-blocking-ish read: if the child is done, read may return 0 at some point.
        match reader.read(&mut buf) {
            Ok(0) => {
                // EOF from the pty; child likely exited or closed output. We’ll break after wait.
                break;
            }
            Ok(n) => {
                let chunk = String::from_utf8_lossy(&buf[..n]);
                // print to our stdout
                print!("{}", chunk);
                let _ = log_file.write_all(chunk.as_bytes());
                // keep a short tail
                last_output.push_str(&chunk);
                if last_output.len() > 5000 {
                    let cut = last_output.len() - 5000;
                    last_output.drain(..cut);
                }
            }
            Err(_e) => {
                // If read fails, we’ll still try to wait on child and return
                break;
            }
        }
    }

    // Wait for child to exit
    let status = child.wait().context("pty child wait failed")?;

    Ok(PtyRunResult {
        raw_status: status,
        last_output,
    })
}

// --- helpers ---

fn shell_quote(s: &str) -> String {
    if s.is_empty() {
        "''".to_string()
    } else if s.chars().all(|c| c.is_ascii_alphanumeric() || "-_./:@".contains(c)) {
        s.to_string()
    } else {
        let mut out = String::from("'");
        for ch in s.chars() {
            if ch == '\'' {
                out.push_str("'\"'\"'");
            } else {
                out.push(ch);
            }
        }
        out.push('\'');
        out
    }
}

fn shell_quote_path(p: &Path) -> String {
    shell_quote(&p.to_string_lossy())
}

// Bring in Read for reader.read()
use std::io::Read;
