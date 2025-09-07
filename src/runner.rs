use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::Instant;
use std::{io, process::Command as StdCommand};

use once_cell::sync::Lazy;
use tokio::fs::OpenOptions;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, Command};
use tokio::sync::Mutex;

/// Global handle to the currently running child process (if any).
pub static CURRENT_CHILD: Lazy<Mutex<Option<Child>>> = Lazy::new(|| Mutex::new(None));

/// Result of a spawned run.
#[derive(Debug, Clone)]
pub struct RunResult {
    /// Exit code of the process (`-1` if unavailable).
    pub exit_code: i32,
    /// Duration of the run in milliseconds.
    pub duration_ms: u128,
    /// The last ~4000 characters of combined stdout+stderr.
    pub log_tail: String,
    /// Full path to the log file that received the output.
    pub full_log_path: PathBuf,
    /// The exact command line that was executed.
    pub command_line: String,
}

/// Execute a shell command synchronously using `std::process::Command`,
/// capturing both stdout and stderr as UTF‑8 strings.
///
/// Returns a tuple `(stdout, stderr)`. Errors from spawning or waiting for the
/// process are propagated as `io::Error`.
pub fn exec_shell(cmd: &str, args: &[&str]) -> io::Result<(String, String)> {
    let output = StdCommand::new(cmd)
        .args(args)
        .output()?;

    let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
    let stderr = String::from_utf8_lossy(&output.stderr).into_owned();

    Ok((stdout, stderr))
}

/// Run an arbitrary command string using the system shell and return its
/// standard output on success. On failure, returns the standard error as a
/// `String`.
///
/// This function is platform‑agnostic: on Unix it invokes `sh -c <cmd>`,
/// while on Windows it uses `cmd /C <cmd>`. It captures the command's output
/// synchronously via `std::process::Command`.
///
/// # Arguments
///
/// * `cmd` – The full command line to execute (as a single string).
///
/// # Returns
///
/// * `Ok(stdout)` – The command succeeded and its stdout is returned.
/// * `Err(message)` – The command failed; `message` contains the stderr or
///   a description of the spawning error.
pub fn run_command(cmd: &str) -> Result<String, String> {
    // Build the appropriate shell command based on the target OS.
    #[cfg(target_os = "windows")]
    let mut command = {
        let mut c = StdCommand::new("cmd");
        c.arg("/C");
        c
    };
    #[cfg(not(target_os = "windows"))]
    let mut command = {
        let mut c = StdCommand::new("sh");
        c.arg("-c");
        c
    };

    command.arg(cmd);

    // Execute and capture output.
    let output = command.output().map_err(|e| e.to_string())?;

    if output.status.success() {
        Ok(String::from_utf8_lossy(&output.stdout).into_owned())
    } else {
        Err(String::from_utf8_lossy(&output.stderr).into_owned())
    }
}

/// Run an arbitrary shell command and return both its stdout and stderr.
///
/// This function invokes the system shell (`sh -c` on Unix, `cmd /C` on Windows)
/// via `std::process::Command`. It captures the command's standard output and
/// standard error as UTF‑8 strings and returns them as a tuple. Errors that
/// occur while spawning or waiting for the process are propagated as `io::Error`.
///
/// # Arguments
///
/// * `cmd` – The full command line to execute (as a single string).
///
/// # Returns
///
/// * `Ok((stdout, stderr))` – The command was executed; both streams are returned.
/// * `Err(e)` – An I/O error occurred while spawning or waiting for the process.
pub fn run_shell_command(cmd: &str) -> io::Result<(String, String)> {
    // Choose the appropriate shell based on the operating system.
    #[cfg(target_os = "windows")]
    let mut command = {
        let mut c = StdCommand::new("cmd");
        c.arg("/C");
        c
    };
    #[cfg(not(target_os = "windows"))]
    let mut command = {
        let mut c = StdCommand::new("sh");
        c.arg("-c");
        c
    };

    command.arg(cmd);

    let output = command.output()?;

    let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
    let stderr = String::from_utf8_lossy(&output.stderr).into_owned();

    Ok((stdout, stderr))
}

/// Run a command synchronously (blocking) but wrapped in a Tokio task so it
/// doesn't block the async runtime. The output is written to the provided log
/// file and a `RunResult` is returned.
///
/// This function re‑uses the same logging format as `spawn_and_stream`.
pub async fn run_sync_and_log(
    cmd: &str,
    args: &[&str],
    workdir: &Path,
    env: &[(String, String)],
    log_path: &Path,
) -> RunResult {
    // Prepare log file (append mode, create if missing)
    let mut log_file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(log_path)
        .await
        .expect("Unable to open log file");

    // Build a human‑readable command line for logging / display
    let command_line = format!("{} {}", cmd, args.join(" "));

    // Write session header
    let timestamp = chrono::Utc::now().to_rfc3339();
    let header = format!(
        "==== RUN @ {} (cmd: {}) ====\\n",
        timestamp, command_line
    );
    log_file
        .write_all(header.as_bytes())
        .await
        .expect("Unable to write header to log");

    // Measure time
    let start = Instant::now();

    // Run the command in a blocking task
    let (stdout, stderr, exit_code) = tokio::task::spawn_blocking({
        let cmd = cmd.to_string();
        let args = args.iter().map(|s| s.to_string()).collect::<Vec<_>>();
        let workdir = workdir.to_path_buf();
        let env = env.to_vec();
        move || {
            let mut command = StdCommand::new(&cmd);
            command.args(&args);
            command.current_dir(&workdir);
            for (k, v) in &env {
                command.env(k, v);
            }
            // Capture output
            let output = command.output()?;
            let code = output.status.code().unwrap_or(-1);
            let out = String::from_utf8_lossy(&output.stdout).into_owned();
            let err = String::from_utf8_lossy(&output.stderr).into_owned();
            Ok::<_, io::Error>((out, err, code))
        }
    })
    .await
    .expect("Blocking task panicked")
    .expect("Failed to execute command");

    // Write captured output to log and build tail buffer
    let mut tail_buffer = String::new();

    for line in stdout.lines().chain(stderr.lines()) {
        // Print to terminal
        println!("{}", line);

        // Write to log
        let mut line_with_nl = line.to_string();
        line_with_nl.push('\n');
        let _ = log_file.write_all(line_with_nl.as_bytes()).await;

        // Update tail buffer
        tail_buffer.push_str(line);
        tail_buffer.push('\n');
        if tail_buffer.len() > 4000 {
            let excess = tail_buffer.len() - 4000;
            tail_buffer.drain(0..excess);
        }
    }

    let duration_ms = start.elapsed().as_millis();

    RunResult {
        exit_code,
        duration_ms,
        log_tail: tail_buffer,
        full_log_path: log_path.to_path_buf(),
        command_line,
    }
}

/// Spawn a command, stream its output line‑by‑line to both the terminal and a log file,
/// and return a `RunResult` with summary information.
///
/// * `cmd` – executable name (e.g., `"cargo"`).
/// * `args` – arguments passed to the executable.
/// * `workdir` – directory in which the command is executed.
/// * `env` – additional environment variables (key, value) to set for the child.
/// * `log_path` – file to which the combined output is appended.
///
/// The function merges stdout and stderr, writes a timestamped session header,
/// and keeps the last ~4000 characters of output for later diagnostic use.
pub async fn spawn_and_stream(
    cmd: &str,
    args: &[&str],
    workdir: &Path,
    env: &[(String, String)],
    log_path: &Path,
) -> RunResult {
    // Prepare log file (append mode, create if missing)
    let mut log_file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(log_path)
        .await
        .expect("Unable to open log file");

    // Build a human‑readable command line for logging / display
    let command_line = format!("{} {}", cmd, args.join(" "));

    // Write session header
    let timestamp = chrono::Utc::now().to_rfc3339();
    let header = format!(
        "==== RUN @ {} (cmd: {}) ====\\n",
        timestamp, command_line
    );
    log_file
        .write_all(header.as_bytes())
        .await
        .expect("Unable to write header to log");

    // Start measuring time
    let start = Instant::now();

    // Configure the command
    let mut command = Command::new(cmd);
    command.args(args);
    command.current_dir(workdir);
    command.stdout(Stdio::piped());
    command.stderr(Stdio::piped());

    // Insert extra environment variables (do not overwrite existing ones)
    for (k, v) in env {
        command.env(k, v);
    }

    // Spawn the child process
    let mut child = command.spawn().expect("Failed to spawn child process");

    // Store the child globally so `/kill` can access it
    {
        let mut guard = CURRENT_CHILD.lock().await;
        *guard = Some(child);
    }

    // Take the child back for processing
    let mut child = {
        let mut guard = CURRENT_CHILD.lock().await;
        guard.take().expect("Child disappeared unexpectedly")
    };

    // Prepare readers for stdout and stderr
    let stdout = child
        .stdout
        .take()
        .expect("Child stdout not captured");
    let stderr = child
        .stderr
        .take()
        .expect("Child stderr not captured");

    let mut stdout_reader = BufReader::new(stdout).lines();
    let mut stderr_reader = BufReader::new(stderr).lines();

    // Buffer for the tail (last ~4000 characters)
    let mut tail_buffer = String::new();

    // Helper to write a line to both terminal and log, and update tail buffer
    async fn handle_line(
        line: &str,
        log_file: &mut tokio::fs::File,
        tail_buffer: &mut String,
    ) {
        // Print to terminal
        println!("{}", line);

        // Write to log file (add newline)
        let mut line_with_nl = line.to_string();
        line_with_nl.push('\n');
        let _ = log_file.write_all(line_with_nl.as_bytes()).await;

        // Update tail buffer (keep only last 4000 chars)
        tail_buffer.push_str(line);
        tail_buffer.push('\n');
        if tail_buffer.len() > 4000 {
            // Trim from the front
            let excess = tail_buffer.len() - 4000;
            tail_buffer.drain(0..excess);
        }
    }

    // Stream both stdout and stderr concurrently, preserving line order as they become ready
    loop {
        tokio::select! {
            maybe_line = stdout_reader.next_line() => {
                match maybe_line {
                    Ok(Some(line)) => {
                        handle_line(&line, &mut log_file, &mut tail_buffer).await;
                    },
                    Ok(None) => {
                        // stdout closed
                        stdout_reader = futures::stream::empty();
                    },
                    Err(_) => {}
                }
            },
            maybe_line = stderr_reader.next_line() => {
                match maybe_line {
                    Ok(Some(line)) => {
                        handle_line(&line, &mut log_file, &mut tail_buffer).await;
                    },
                    Ok(None) => {
                        // stderr closed
                        stderr_reader = futures::stream::empty();
                    },
                    Err(_) => {}
                }
            },
            // Both streams finished
            else => break,
        }
    }

    // Wait for the child to exit and capture its status
    let exit_status = child
        .wait()
        .await
        .expect("Failed to wait on child process");

    // Clean up the global handle (ensure it's None)
    {
        let mut guard = CURRENT_CHILD.lock().await;
        *guard = None;
    }

    let duration_ms = start.elapsed().as_millis();
    let exit_code = exit_status.code().unwrap_or(-1);

    RunResult {
        exit_code,
        duration_ms,
        log_tail: tail_buffer,
        full_log_path: log_path.to_path_buf(),
        command_line,
    }
}

/// Attempt to terminate the currently running child process, if any.
///
/// Returns `true` if a process was found and a kill signal was sent,
/// `false` otherwise.
pub async fn kill_current() -> bool {
    let mut guard = CURRENT_CHILD.lock().await;
    if let Some(child) = guard.as_mut() {
        // Ignore errors – the process may have already exited.
        let _ = child.kill();
        true
    } else {
        false
    }
}