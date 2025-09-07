use std::collections::HashMap;
use std::io;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::thread::sleep;
use std::time::{Duration, SystemTime};

use log::{error, info, warn};
use walkdir::WalkDir;

/// Configuration for autonomous command execution.
///
/// * `max_retries` – Number of additional attempts after the initial execution
///   (e.g., `max_retries = 2` results in up to three total attempts).
/// * `base_delay_ms` – Base delay in milliseconds used for exponential back‑off
///   between retries. The actual delay for attempt *n* is
///   `base_delay_ms * 2.pow(n)`.
///
/// The defaults are chosen to be safe for most environments; they can be
/// overridden by constructing a custom `CommandRunner`.
#[derive(Debug, Clone, Copy)]
pub struct CommandRunner {
    pub max_retries: u32,
    pub base_delay_ms: u64,
}

impl CommandRunner {
    /// Creates a new `CommandRunner` with the given retry policy.
    pub fn new(max_retries: u32, base_delay_ms: u64) -> Self {
        Self {
            max_retries,
            base_delay_ms,
        }
    }

    /// Executes a shell command with automatic retries, exponential back‑off,
    /// and structured logging.
    ///
    /// The command is run via the system's default shell (`sh -c`). On each
    /// attempt the function logs:
    ///
    /// * **INFO** – the command being executed.
    /// * **INFO** – the captured stdout when the command succeeds.
    /// * **WARN** – non‑zero exit status together with stderr.
    /// * **ERROR** – I/O errors that prevent the command from being spawned.
    ///
    /// If the command exits successfully (`status.success()`), its stdout is
    /// returned. Otherwise the function retries according to the configured
    /// policy. After exhausting all attempts, the last error (or a generic
    /// `Other` error if the process ran but never succeeded) is returned.
    pub fn run(&self, command: &str) -> Result<String, io::Error> {
        let mut attempt: u32 = 0;
        let mut last_error: Option<io::Error> = None;

        loop {
            info!("Attempt {}: executing command: {}", attempt + 1, command);
            let output_result = Command::new("sh").arg("-c").arg(command).output();

            match output_result {
                Ok(output) => {
                    if output.status.success() {
                        let stdout = String::from_utf8_lossy(&output.stdout).to_string();
                        info!(
                            "Command succeeded on attempt {}. Output: {}",
                            attempt + 1,
                            stdout
                        );
                        return Ok(stdout);
                    } else {
                        let stderr = String::from_utf8_lossy(&output.stderr).to_string();
                        warn!(
                            "Command returned non‑zero exit code ({:?}) on attempt {}. Stderr: {}",
                            output.status.code(),
                            attempt + 1,
                            stderr
                        );
                    }
                }
                Err(e) => {
                    error!(
                        "I/O error while spawning command on attempt {}: {}",
                        attempt + 1,
                        e
                    );
                    last_error = Some(e);
                }
            }

            // Determine whether we should retry.
            if attempt >= self.max_retries {
                break;
            }

            // Exponential back‑off before the next attempt.
            let backoff = self.base_delay_ms.saturating_mul(2u64.pow(attempt));
            info!(
                "Waiting {} ms before next retry (attempt {}/{})",
                backoff,
                attempt + 2,
                self.max_retries + 1
            );
            sleep(Duration::from_millis(backoff));

            attempt += 1;
        }

        // All attempts exhausted; return the most relevant error.
        Err(last_error.unwrap_or_else(|| {
            io::Error::new(
                io::ErrorKind::Other,
                "Command failed after all retry attempts",
            )
        }))
    }
}

/// Executes a shell command and returns its standard output as a `String`.
///
/// This is a thin wrapper around `CommandRunner::run` with a default,
/// non‑retrying configuration, preserving the original simple behaviour.
///
/// # Arguments
///
/// * `command` – The command line to execute. It will be passed to the system's
///   default shell (`sh -c`) for interpretation.
///
/// # Returns
///
/// * `Ok(String)` containing the command's stdout on success.
/// * `Err(io::Error)` if the command could not be spawned, its output could not
///   be read, or it exited with a non‑zero status.
pub fn run_command(command: &str) -> Result<String, io::Error> {
    // Default runner: no retries, minimal back‑off.
    let runner = CommandRunner::new(0, 0);
    runner.run(command)
}

/// A helper that watches a directory (recursively) for any file modification
/// timestamps changes. It stores the last known modification times and can
/// report whether any file has changed since the previous check.
#[derive(Debug)]
struct FileWatcher {
    root: PathBuf,
    timestamps: HashMap<PathBuf, SystemTime>,
}

impl FileWatcher {
    fn new<P: AsRef<Path>>(root: P) -> io::Result<Self> {
        let root_path = root.as_ref().to_path_buf();
        let timestamps = Self::collect_timestamps(&root_path)?;
        Ok(Self {
            root: root_path,
            timestamps,
        })
    }

    /// Walk the directory tree and record the modification time of each file.
    fn collect_timestamps(root: &Path) -> io::Result<HashMap<PathBuf, SystemTime>> {
        let mut map = HashMap::new();
        for entry in WalkDir::new(root)
            .into_iter()
            .filter_map(Result::ok)
            .filter(|e| e.file_type().is_file())
        {
            let meta = entry.metadata()?;
            if let Ok(mtime) = meta.modified() {
                map.insert(entry.path().to_path_buf(), mtime);
            }
        }
        Ok(map)
    }

    /// Returns `true` if any file under `root` has a newer modification time
    /// compared to the stored snapshot. The internal snapshot is updated to the
    /// latest state before returning.
    fn has_changed(&mut self) -> io::Result<bool> {
        let current = Self::collect_timestamps(&self.root)?;
        let changed = current.iter().any(|(path, mtime)| {
            self.timestamps
                .get(path)
                .map_or(true, |prev| *prev != *mtime)
        }) || self.timestamps.len() != current.len();

        // Update stored timestamps for the next check.
        self.timestamps = current;
        Ok(changed)
    }
}

/// Orchestrates continuous autonomous operation:
/// * Watches a source directory for changes.
/// * Re‑executes the planner command when changes are detected.
/// * Runs the pipeline command after a successful planner run.
/// * Monitors failures and restarts the pipeline as needed.
///
/// The loop runs indefinitely; it can be stopped by terminating the process.
pub struct AutonomousRunner {
    planner_cmd: String,
    pipeline_cmd: String,
    watcher: FileWatcher,
    runner: CommandRunner,
    poll_interval: Duration,
}

impl AutonomousRunner {
    /// Creates a new `AutonomousRunner`.
    ///
    /// * `planner_cmd` – Command that generates or updates the plan.
    /// * `pipeline_cmd` – Command that consumes the plan and performs the work.
    /// * `watch_path` – Directory to monitor for source changes.
    /// * `poll_interval` – How often to poll the filesystem for changes.
    /// * `runner` – `CommandRunner` used for executing both commands (retries, etc.).
    pub fn new<P: AsRef<Path>>(
        planner_cmd: &str,
        pipeline_cmd: &str,
        watch_path: P,
        poll_interval: Duration,
        runner: CommandRunner,
    ) -> io::Result<Self> {
        let watcher = FileWatcher::new(watch_path)?;
        Ok(Self {
            planner_cmd: planner_cmd.to_string(),
            pipeline_cmd: pipeline_cmd.to_string(),
            watcher,
            runner,
            poll_interval,
        })
    }

    /// Starts the autonomous loop. This function blocks forever (or until an
    /// unrecoverable I/O error occurs).
    pub fn run(&mut self) -> io::Result<()> {
        loop {
            // 1. Detect source changes.
            match self.watcher.has_changed() {
                Ok(true) => {
                    info!("Source changes detected – re‑executing planner.");
                    if let Err(e) = self.execute_planner() {
                        error!("Planner failed: {}", e);
                        // Continue looping; we will retry on next change detection.
                        continue;
                    }
                }
                Ok(false) => {
                    // No changes – nothing to do right now.
                }
                Err(e) => {
                    error!("Failed to scan watch directory: {}", e);
                }
            }

            // 2. Run (or re‑run) the pipeline.
            if let Err(e) = self.execute_pipeline() {
                error!("Pipeline execution failed: {}", e);
                // The loop will retry after the next poll interval.
            }

            // 3. Wait before the next poll.
            sleep(self.poll_interval);
        }
    }

    fn execute_planner(&self) -> Result<String, io::Error> {
        self.runner.run(&self.planner_cmd)
    }

    fn execute_pipeline(&self) -> Result<String, io::Error> {
        self.runner.run(&self.pipeline_cmd)
    }
}

/// Convenience entry‑point used by external callers. It constructs an
/// `AutonomousRunner` with sensible defaults (e.g., a 2‑second poll interval
/// and a retry policy of three attempts with 500 ms base back‑off) and starts
/// the autonomous loop.
///
/// # Arguments
///
/// * `planner_cmd` – Command that builds the plan.
/// * `pipeline_cmd` – Command that consumes the plan.
/// * `watch_path` – Path to the source directory that should trigger replanning
///   when modified.
///
/// # Errors
///
/// Returns an `io::Error` if the watcher cannot be initialised or if any
/// subsequent I/O operation fails.
pub fn start_autonomous_mode(
    planner_cmd: &str,
    pipeline_cmd: &str,
    watch_path: &str,
) -> io::Result<()> {
    let runner = CommandRunner::new(2, 500); // up to 3 attempts, 500 ms base delay
    let poll_interval = Duration::from_secs(2);
    let mut autonomous = AutonomousRunner::new(
        planner_cmd,
        pipeline_cmd,
        watch_path,
        poll_interval,
        runner,
    )?;
    autonomous.run()
}

// The `walkdir` crate is used for recursive directory traversal. If the project
// does not already depend on it, add `walkdir = "2"` to Cargo.toml. This comment
// is left here to remind maintainers of the required dependency.