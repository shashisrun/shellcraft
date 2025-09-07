use anyhow::{anyhow, Context, Result};
use portable_pty::{native_pty_system, CommandBuilder, ExitStatus, PtySize};
use std::fs::OpenOptions;
use std::io::{Read, Write};
use std::path::Path;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};
use std::thread;
use std::time::{Duration, Instant};

#[cfg(unix)]
use libc::{ioctl, winsize, STDOUT_FILENO, TIOCGWINSZ};

/// Result of a PTY run with additional guard‑rail information.
#[derive(Debug)]
pub struct PtyRunResult {
    /// Raw exit status from the PTY child.
    pub raw_status: ExitStatus,
    /// Tail of the captured output (subject to `max_output_bytes` limit).
    pub last_output: String,
    /// Whether the process was terminated because it exceeded the timeout.
    pub timed_out: bool,
    /// Optional error message captured from the runner itself (e.g., I/O errors).
    pub error: Option<String>,
}

/// Run a program inside a PTY with safety guardrails.
///
/// * `program` – executable to run (must pass `enforce_command_safety`).
/// * `args` – arguments passed to the program.
/// * `workdir` – directory in which the command is executed.
/// * `_env` – currently unused; kept for future extension.
/// * `log_path` – path to a file where all PTY output is appended.
/// * `timeout` – maximum wall‑clock time the command may run.
/// * `max_output_bytes` – maximum number of bytes retained in `last_output`.
pub fn run_with_pty(
    program: &str,
    args: &[String],
    workdir: &Path,
    _env: &[(String, String)],
    log_path: &Path,
    timeout: Duration,
    max_output_bytes: usize,
) -> Result<PtyRunResult> {
    // -------------------------------------------------------------------------
    // Guardrails: deny destructive commands and allowlist safe ones
    // -------------------------------------------------------------------------
    enforce_command_safety(program, args)?;

    let pty_system = native_pty_system();
    let pair = pty_system
        .openpty(PtySize {
            rows: 30,
            cols: 120,
            pixel_width: 0,
            pixel_height: 0,
        })
        .context("openpty failed")?;

    // -------------------------------------------------------------------------
    // Set up SIGWINCH handling to resize the PTY when the terminal changes size
    // -------------------------------------------------------------------------
    let resize_requested = Arc::new(AtomicBool::new(false));
    #[cfg(unix)]
    {
        // Register the flag; ignore errors – resizing is a best‑effort feature
        let _ = signal_hook::flag::register(signal_hook::consts::SIGWINCH, resize_requested.clone());
    }

    // -------------------------------------------------------------------------
    // Build a safe shell command: cd into workdir, then run program with args
    // -------------------------------------------------------------------------
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

    // -------------------------------------------------------------------------
    // Logging: open per‑task log file (caller supplies the correct path)
    // -------------------------------------------------------------------------
    let mut log_file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(log_path)
        .with_context(|| format!("open log file {}", log_path.display()))?;

    // -------------------------------------------------------------------------
    // Drain output until the child exits or we hit the timeout.
    // -------------------------------------------------------------------------
    let mut reader = pair
        .master
        .try_clone_reader()
        .context("failed to clone pty reader")?;

    let mut buf = [0u8; 8192];
    let mut last_output = String::new();
    let mut error: Option<String> = None;
    let start = Instant::now();
    let mut timed_out = false;
    let mut exit_status: Option<ExitStatus> = None;

    loop {
        // Check for a pending resize request before blocking on read.
        if resize_requested.swap(false, Ordering::SeqCst) {
            if let Ok(new_size) = get_current_terminal_size() {
                let _ = pair.master.resize(new_size);
            }
        }

        // Non‑blocking read with a short timeout to allow us to check the child.
        match reader.read(&mut buf) {
            Ok(0) => {
                // EOF – child likely exited.
                // We'll still check the child's status below.
            }
            Ok(n) => {
                let chunk = String::from_utf8_lossy(&buf[..n]);
                // Echo to the current stdout (live PTY)
                print!("{}", chunk);
                // Also tee to the log file
                if let Err(e) = log_file.write_all(chunk.as_bytes()) {
                    error = Some(format!("Failed to write to log file: {}", e));
                }

                // Append to the tail buffer respecting the size limit.
                last_output.push_str(&chunk);
                if last_output.len() > max_output_bytes {
                    let cut = last_output.len() - max_output_bytes;
                    last_output.drain(..cut);
                }
            }
            Err(e) => {
                // Capture read errors but continue; they usually indicate the PTY is closed.
                error = Some(format!("PTY read error: {}", e));
                break;
            }
        }

        // Check if the child has exited.
        match child.try_wait() {
            Ok(Some(status)) => {
                exit_status = Some(status);
                break;
            }
            Ok(None) => {
                // Still running – check timeout.
                if start.elapsed() > timeout {
                    // Timeout exceeded: attempt to kill the child.
                    let _ = child.kill();
                    timed_out = true;
                    // After killing, wait for the final status.
                    match child.wait() {
                        Ok(s) => exit_status = Some(s),
                        Err(e) => {
                            error = Some(format!("Failed to wait after kill: {}", e));
                        }
                    }
                    break;
                }
                // No exit yet; continue looping.
            }
            Err(e) => {
                error = Some(format!("try_wait error: {}", e));
                break;
            }
        }

        // Small sleep to avoid a tight loop if no data is available.
        thread::sleep(Duration::from_millis(10));
    }

    // Final resize check in case a SIGWINCH arrived just before exit.
    if resize_requested.swap(false, Ordering::SeqCst) {
        if let Ok(new_size) = get_current_terminal_size() {
            let _ = pair.master.resize(new_size);
        }
    }

    // Ensure we have a status; if the child never reported one, try a final wait.
    let raw_status = match exit_status {
        Some(s) => s,
        None => child.wait().unwrap_or_else(|e| {
            error = Some(format!("Final wait failed: {}", e));
            // Construct a generic failure status; portable_pty's ExitStatus does not have a public ctor,
            // but we can fallback to a zeroed status via Default if available.
            ExitStatus::default()
        }),
    };

    Ok(PtyRunResult {
        raw_status,
        last_output,
        timed_out,
        error,
    })
}

// -----------------------------------------------------------------------------
// Helper: enforce simple allow‑/deny‑list safety checks
// -----------------------------------------------------------------------------
fn enforce_command_safety(program: &str, args: &[String]) -> Result<()> {
    // Simple denylist for obviously destructive commands
    let denylist = [
        "rm", "sudo", "shutdown", "reboot", "halt", "poweroff", "mkfs", "dd", "chmod",
        "chown", "kill", "killall", "pkill", "passwd", "useradd", "usermod", "userdel",
    ];

    // Very naive detection of dangerous patterns (e.g., `rm -rf /`)
    if denylist.iter().any(|&d| program.ends_with(d)) {
        // Additional check for rm -rf patterns
        if program.ends_with("rm") && args.iter().any(|a| a == "-rf" || a == "-r" || a == "-f") {
            return Err(anyhow!(
                "Destructive command '{}' with arguments {:?} is blocked",
                program,
                args
            ));
        }
        return Err(anyhow!(
            "Command '{}' is on the denylist and is blocked",
            program
        ));
    }

    // Allowlist of common safe development commands
    let allowlist = [
        "cargo", "make", "npm", "yarn", "go", "python", "python3", "node", "git", "bash",
        "sh", "zsh", "ls", "cat", "echo", "grep", "sed", "awk", "gcc", "g++", "clang",
        "clang++", "rustc", "rustup", "cargo-build", "cargo-test", "cargo-run",
    ];

    let prog_name = Path::new(program)
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or(program);

    if allowlist.iter().any(|&a| a == prog_name) {
        return Ok(());
    }

    // If not explicitly allowed, require an explicit opt‑in via env var
    match std::env::var("PTY_ALLOW_UNSAFE") {
        Ok(v) if v == "1" => Ok(()),
        _ => Err(anyhow!(
            "Command '{}' is not in the allowlist. Set PTY_ALLOW_UNSAFE=1 to override."
        )),
    }
}

// -----------------------------------------------------------------------------
// Helper: obtain the current terminal size (rows/cols) using ioctl
// -----------------------------------------------------------------------------
fn get_current_terminal_size() -> Result<PtySize> {
    #[cfg(unix)]
    unsafe {
        let mut ws: winsize = std::mem::zeroed();
        if ioctl(STDOUT_FILENO, TIOCGWINSZ, &mut ws) == -1 {
            return Err(anyhow!("ioctl TIOCGWINSZ failed"));
        }
        Ok(PtySize {
            rows: ws.ws_row as u16,
            cols: ws.ws_col as u16,
            pixel_width: ws.ws_xpixel as u16,
            pixel_height: ws.ws_ypixel as u16,
        })
    }
    #[cfg(not(unix))]
    {
        // On non‑Unix platforms we fall back to a static size.
        Ok(PtySize {
            rows: 30,
            cols: 120,
            pixel_width: 0,
            pixel_height: 0,
        })
    }
}

// --- helpers ---

fn shell_quote(s: &str) -> String {
    if s.is_empty() {
        "''".to_string()
    } else if s
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || "-_./:@".contains(c))
    {
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