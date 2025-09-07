use std::collections::HashMap;
use std::io::{self, BufRead, Write};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::thread::{self, sleep};
use std::time::{Duration, SystemTime};

use log::{error, info, warn};
use once_cell::sync::Lazy;
use std::sync::{mpsc, Arc, Mutex};
use walkdir::WalkDir;

use crate::editor;
use crate::llm;

/// Guardrail configuration.
///
/// * `require_confirmation` – If `true`, any command that is not explicitly
///   allow‑listed will prompt the user for confirmation before execution.
pub struct GuardConfig {
    pub require_confirmation: bool,
}

static GLOBAL_GUARD: Lazy<Mutex<GuardConfig>> = Lazy::new(|| {
    Mutex::new(GuardConfig {
        require_confirmation: false,
    })
});

/// Set the global `require_confirmation` flag.
///
/// This can be called by the application (e.g., based on a CLI flag or config
/// file) to enforce interactive confirmation for non‑allow‑listed commands.
pub fn set_require_confirmation(val: bool) {
    let mut cfg = GLOBAL_GUARD.lock().unwrap();
    cfg.require_confirmation = val;
}

/// Global dry‑run flag. When enabled, no external commands are executed and
/// no files are written; instead a report of intended actions is collected.
static GLOBAL_DRY_RUN: Lazy<Mutex<bool>> = Lazy::new(|| Mutex::new(false));

/// Set the global dry‑run mode.
pub fn set_dry_run(val: bool) {
    let mut dr = GLOBAL_DRY_RUN.lock().unwrap();
    *dr = val;
}

/// Collect a textual description of each action that would have been performed
/// in dry‑run mode.
static DRY_RUN_REPORT: Lazy<Mutex<Vec<String>>> = Lazy::new(|| Mutex::new(Vec::new()));

fn add_dry_run_report(entry: String) {
    let mut report = DRY_RUN_REPORT.lock().unwrap();
    report.push(entry);
}

/// Retrieve the current dry‑run report.
pub fn get_dry_run_report() -> Vec<String> {
    DRY_RUN_REPORT.lock().unwrap().clone()
}

/// List of destructive patterns that are denied by default.
static DENYLIST: &[&str] = &["rm -rf", "sudo", "shutdown", "reboot", "init 0", "poweroff"];

/// Common safe commands that are allowed without confirmation.
static ALLOWLIST: &[&str] = &[
    "cargo", "npm", "pytest", "go", "mvn", "rustfmt", "prettier", "black", "gofmt", "clippy",
    "eslint", "flake8", "git", "grep", "rg",
];

/// Perform guardrail checks on a raw command string.
///
/// Returns `Ok(())` if the command is permitted, otherwise an `io::Error` with
/// `PermissionDenied`. If the global `require_confirmation` flag is set and the
/// command is not in the allowlist, the user is prompted for confirmation.
fn guard_check(command: &str) -> Result<(), io::Error> {
    // Denylist check – simple substring match.
    for &bad in DENYLIST {
        if command.contains(bad) {
            return Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                format!("Command contains denied pattern '{}'", bad),
            ));
        }
    }

    // Allowlist check.
    let first_token = command.split_whitespace().next().unwrap_or("");
    let is_allowed = ALLOWLIST.iter().any(|&good| good == first_token);

    if !is_allowed {
        let cfg = GLOBAL_GUARD.lock().unwrap();
        if cfg.require_confirmation {
            eprint!(
                "Command '{}' is not in the allowlist. Execute? (y/N): ",
                command
            );
            io::stderr().flush()?;
            let stdin = io::stdin();
            let mut line = String::new();
            stdin.lock().read_line(&mut line)?;
            let resp = line.trim().to_ascii_lowercase();
            if resp != "y" && resp != "yes" {
                return Err(io::Error::new(
                    io::ErrorKind::PermissionDenied,
                    "User declined execution of non‑allowlisted command",
                ));
            }
        } else {
            return Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                format!("Command '{}' is not in the allowlist", command),
            ));
        }
    }

    Ok(())
}

/// Write command output to a per‑task log file under `./.agent/logs/`.
///
/// The log file is named `<task>.log`, where `task` is the first token of the
/// command (e.g., `cargo` → `cargo.log`). Both stdout and stderr are appended,
/// prefixed with a timestamp.
fn tee_log(task: &str, stdout: &str, stderr: &str) -> io::Result<()> {
    if *GLOBAL_DRY_RUN.lock().unwrap() {
        add_dry_run_report(format!(
            "Dry-run: Would write log for task '{}' (stdout {} bytes, stderr {} bytes)",
            task,
            stdout.len(),
            stderr.len()
        ));
        return Ok(());
    }

    let log_dir = Path::new("./.agent/logs");
    std::fs::create_dir_all(log_dir)?;
    let log_path = log_dir.join(format!("{}.log", task));
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(log_path)?;

    let ts = chrono::Utc::now().to_rfc3339();
    writeln!(file, "[{}] STDOUT:", ts)?;
    writeln!(file, "{}", stdout)?;
    writeln!(file, "[{}] STDERR:", ts)?;
    writeln!(file, "{}", stderr)?;
    Ok(())
}

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
        // Guardrail check before any attempt.
        guard_check(command)?;

        if *GLOBAL_DRY_RUN.lock().unwrap() {
            add_dry_run_report(format!("Dry-run: Would execute command '{}'", command));
            return Ok(String::new());
        }

        let mut attempt: u32 = 0;
        let mut last_error: Option<io::Error> = None;

        loop {
            info!("Attempt {}: executing command: {}", attempt + 1, command);
            let output_result = Command::new("sh").arg("-c").arg(command).output();

            match output_result {
                Ok(output) => {
                    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
                    let stderr = String::from_utf8_lossy(&output.stderr).to_string();

                    // Tee to log file.
                    let task_name = command.split_whitespace().next().unwrap_or("unknown");
                    let _ = tee_log(task_name, &stdout, &stderr);

                    if output.status.success() {
                        info!(
                            "Command succeeded on attempt {}. Output: {}",
                            attempt + 1,
                            stdout
                        );
                        return Ok(stdout);
                    } else {
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
///   be read, or it exited with a non-zero status.
pub fn run_command(command: &str) -> Result<String, io::Error> {
    // Default runner: no retries, minimal back‑off.
    let runner = CommandRunner::new(0, 0);
    runner.run(command)
}

/* -------------------------------------------------------------------------- */
/*                     Tool Registry – Portable, Project‑Aware               */
/* -------------------------------------------------------------------------- */

/// Safety configuration for a tool. Empty allowlist means “any argument is
/// allowed”; a non‑empty denylist blocks specific arguments.
pub struct Safety {
    pub allowlist: &'static [&'static str],
    pub denylist: &'static [&'static str],
}

/// Core descriptor for a tool.
pub struct Tool {
    pub name: &'static str,
    pub detect: fn(&Path) -> bool,
    pub run: fn(&[String], &Path) -> Result<String, io::Error>,
    pub safety: Safety,
}

/// Generic runner that spawns a command with the given arguments in `cwd`.
fn generic_run(args: &[String], cwd: &Path) -> Result<String, io::Error> {
    if args.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "No command provided",
        ));
    }

    // Guardrail check on the executable name.
    guard_check(&args[0])?;

    if *GLOBAL_DRY_RUN.lock().unwrap() {
        add_dry_run_report(format!(
            "Dry-run: Would run executable '{}' with args {:?} in cwd '{}'",
            args[0],
            &args[1..],
            cwd.display()
        ));
        return Ok(String::new());
    }

    let mut cmd = Command::new(&args[0]);
    if args.len() > 1 {
        cmd.args(&args[1..]);
    }
    cmd.current_dir(cwd);
    let output = cmd.output()?;
    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();

    // Tee to log.
    let _ = tee_log(&args[0], &stdout, &stderr);

    if output.status.success() {
        Ok(stdout)
    } else {
        Err(io::Error::new(
            io::ErrorKind::Other,
            format!(
                "Command failed with status {:?}: {}",
                output.status.code(),
                stderr
            ),
        ))
    }
}

/* Detection helpers -------------------------------------------------------- */

fn detect_cargo(path: &Path) -> bool {
    path.join("Cargo.toml").exists()
}
fn detect_npm(path: &Path) -> bool {
    path.join("package.json").exists()
}
fn detect_pytest(path: &Path) -> bool {
    path.join("pytest.ini").exists() || path.join("tests").is_dir()
}
fn detect_go(path: &Path) -> bool {
    path.join("go.mod").exists()
}
fn detect_maven(path: &Path) -> bool {
    path.join("pom.xml").exists()
}

/* Built‑in tool implementations -------------------------------------------- */

fn cargo_build_run(_args: &[String], cwd: &Path) -> Result<String, io::Error> {
    generic_run(&vec!["cargo".to_string(), "build".to_string()], cwd)
}
fn cargo_test_run(_args: &[String], cwd: &Path) -> Result<String, io::Error> {
    generic_run(&vec!["cargo".to_string(), "test".to_string()], cwd)
}
fn npm_build_run(_args: &[String], cwd: &Path) -> Result<String, io::Error> {
    generic_run(
        &vec!["npm".to_string(), "run".to_string(), "build".to_string()],
        cwd,
    )
}
fn npm_test_run(_args: &[String], cwd: &Path) -> Result<String, io::Error> {
    generic_run(&vec!["npm".to_string(), "test".to_string()], cwd)
}
fn pytest_run(_args: &[String], cwd: &Path) -> Result<String, io::Error> {
    generic_run(&vec!["pytest".to_string()], cwd)
}
fn go_test_run(_args: &[String], cwd: &Path) -> Result<String, io::Error> {
    generic_run(&vec!["go".to_string(), "test".to_string()], cwd)
}
fn mvn_test_run(_args: &[String], cwd: &Path) -> Result<String, io::Error> {
    generic_run(&vec!["mvn".to_string(), "test".to_string()], cwd)
}
fn rustfmt_run(_args: &[String], cwd: &Path) -> Result<String, io::Error> {
    generic_run(&vec!["rustfmt".to_string()], cwd)
}
fn prettier_run(_args: &[String], cwd: &Path) -> Result<String, io::Error> {
    generic_run(
        &vec![
            "prettier".to_string(),
            "--write".to_string(),
            ".".to_string(),
        ],
        cwd,
    )
}
fn black_run(_args: &[String], cwd: &Path) -> Result<String, io::Error> {
    generic_run(&vec!["black".to_string(), ".".to_string()], cwd)
}
fn gofmt_run(_args: &[String], cwd: &Path) -> Result<String, io::Error> {
    generic_run(
        &vec!["gofmt".to_string(), "-w".to_string(), ".".to_string()],
        cwd,
    )
}
fn clippy_run(_args: &[String], cwd: &Path) -> Result<String, io::Error> {
    generic_run(&vec!["cargo".to_string(), "clippy".to_string()], cwd)
}
fn eslint_run(_args: &[String], cwd: &Path) -> Result<String, io::Error> {
    generic_run(&vec!["eslint".to_string(), ".".to_string()], cwd)
}
fn flake8_run(_args: &[String], cwd: &Path) -> Result<String, io::Error> {
    generic_run(&vec!["flake8".to_string()], cwd)
}
fn grep_run(_args: &[String], cwd: &Path) -> Result<String, io::Error> {
    generic_run(_args, cwd)
}
fn ripgrep_run(_args: &[String], cwd: &Path) -> Result<String, io::Error> {
    generic_run(_args, cwd)
}
fn git_diff_run(_args: &[String], cwd: &Path) -> Result<String, io::Error> {
    generic_run(_args, cwd)
}

/* Registry ----------------------------------------------------------------- */

static TOOL_REGISTRY: Lazy<HashMap<&'static str, Tool>> = Lazy::new(|| {
    let mut m = HashMap::new();

    // Build / test tools
    m.insert(
        "cargo_build",
        Tool {
            name: "cargo_build",
            detect: detect_cargo,
            run: cargo_build_run,
            safety: Safety {
                allowlist: &[],
                denylist: &[],
            },
        },
    );
    m.insert(
        "cargo_test",
        Tool {
            name: "cargo_test",
            detect: detect_cargo,
            run: cargo_test_run,
            safety: Safety {
                allowlist: &[],
                denylist: &[],
            },
        },
    );
    m.insert(
        "npm_build",
        Tool {
            name: "npm_build",
            detect: detect_npm,
            run: npm_build_run,
            safety: Safety {
                allowlist: &[],
                denylist: &[],
            },
        },
    );
    m.insert(
        "npm_test",
        Tool {
            name: "npm_test",
            detect: detect_npm,
            run: npm_test_run,
            safety: Safety {
                allowlist: &[],
                denylist: &[],
            },
        },
    );
    m.insert(
        "pytest",
        Tool {
            name: "pytest",
            detect: detect_pytest,
            run: pytest_run,
            safety: Safety {
                allowlist: &[],
                denylist: &[],
            },
        },
    );
    m.insert(
        "go_test",
        Tool {
            name: "go_test",
            detect: detect_go,
            run: go_test_run,
            safety: Safety {
                allowlist: &[],
                denylist: &[],
            },
        },
    );
    m.insert(
        "mvn_test",
        Tool {
            name: "mvn_test",
            detect: detect_maven,
            run: mvn_test_run,
            safety: Safety {
                allowlist: &[],
                denylist: &[],
            },
        },
    );

    // Formatters
    m.insert(
        "rustfmt",
        Tool {
            name: "rustfmt",
            detect: detect_cargo,
            run: rustfmt_run,
            safety: Safety {
                allowlist: &[],
                denylist: &[],
            },
        },
    );
    m.insert(
        "prettier",
        Tool {
            name: "prettier",
            detect: detect_npm,
            run: prettier_run,
            safety: Safety {
                allowlist: &[],
                denylist: &[],
            },
        },
    );
    m.insert(
        "black",
        Tool {
            name: "black",
            detect: detect_pytest,
            run: black_run,
            safety: Safety {
                allowlist: &[],
                denylist: &[],
            },
        },
    );
    m.insert(
        "gofmt",
        Tool {
            name: "gofmt",
            detect: detect_go,
            run: gofmt_run,
            safety: Safety {
                allowlist: &[],
                denylist: &[],
            },
        },
    );

    // Linters
    m.insert(
        "clippy",
        Tool {
            name: "clippy",
            detect: detect_cargo,
            run: clippy_run,
            safety: Safety {
                allowlist: &[],
                denylist: &[],
            },
        },
    );
    m.insert(
        "eslint",
        Tool {
            name: "eslint",
            detect: detect_npm,
            run: eslint_run,
            safety: Safety {
                allowlist: &[],
                denylist: &[],
            },
        },
    );
    m.insert(
        "flake8",
        Tool {
            name: "flake8",
            detect: detect_pytest,
            run: flake8_run,
            safety: Safety {
                allowlist: &[],
                denylist: &[],
            },
        },
    );

    // Search helpers
    m.insert(
        "grep",
        Tool {
            name: "grep",
            detect: |_| true,
            run: grep_run,
            safety: Safety {
                allowlist: &[],
                denylist: &[],
            },
        },
    );
    m.insert(
        "ripgrep",
        Tool {
            name: "ripgrep",
            detect: |_| true,
            run: ripgrep_run,
            safety: Safety {
                allowlist: &[],
                denylist: &[],
            },
        },
    );

    // Git diff helpers
    m.insert(
        "git_diff",
        Tool {
            name: "git_diff",
            detect: |_| true,
            run: git_diff_run,
            safety: Safety {
                allowlist: &[],
                denylist: &[],
            },
        },
    );

    m
});

/// Look up a tool by its name.
pub fn get_tool(name: &str) -> Option<&'static Tool> {
    TOOL_REGISTRY.get(name)
}

/// Execute a registered tool with the supplied arguments and working directory,
/// applying safety checks (allowlist / denylist) before execution.
pub fn execute_tool(name: &str, args: &[&str], cwd: &Path) -> Result<String, io::Error> {
    let tool = get_tool(name).ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::NotFound,
            format!("Tool '{}' not found", name),
        )
    })?;

    // Safety checks
    for &arg in args {
        if tool.safety.denylist.iter().any(|&d| d == arg) {
            return Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                format!("Argument '{}' is denied for tool '{}'", arg, name),
            ));
        }
        if !tool.safety.allowlist.is_empty() && !tool.safety.allowlist.iter().any(|&a| a == arg) {
            return Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                format!("Argument '{}' not allowed for tool '{}'", arg, name),
            ));
        }
    }

    let args_vec: Vec<String> = args.iter().map(|s| s.to_string()).collect();
    (tool.run)(&args_vec, cwd)
}

/* -------------------------------------------------------------------------- */
/*                     Task Graph & Executor Agent                            */
/* -------------------------------------------------------------------------- */

/// Represents a single unit of work.
///
/// * `id` – Unique identifier for the task.
/// * `tool` – Name of a registered tool (e.g., `cargo_build`) or a raw shell
///   command if the tool is not found in the registry.
/// * `args` – Arguments passed to the tool. Ignored when `tool` is a raw command.
/// * `deps` – List of task IDs that must complete successfully before this
///   task can run.
#[derive(Clone, Debug)]
pub struct Task {
    pub id: String,
    pub tool: String,
    pub args: Vec<String>,
    pub deps: Vec<String>,
}

impl Task {
    pub fn new<S: Into<String>>(id: S, tool: S, args: Vec<String>, deps: Vec<String>) -> Self {
        Self {
            id: id.into(),
            tool: tool.into(),
            args,
            deps,
        }
    }
}

/// A directed acyclic graph of tasks.
///
/// The graph does **not** enforce acyclicity on insertion; `validate` must be
/// called before execution.
#[derive(Clone, Debug, Default)]
pub struct TaskGraph {
    pub tasks: HashMap<String, Task>,
}

impl TaskGraph {
    pub fn new() -> Self {
        Self {
            tasks: HashMap::new(),
        }
    }

    /// Insert a task into the graph. Overwrites any existing task with the same ID.
    pub fn add_task(&mut self, task: Task) {
        self.tasks.insert(task.id.clone(), task);
    }

    /// Verify that the graph is a DAG (no cycles) and that all dependencies refer
    /// to existing tasks.
    pub fn validate(&self) -> Result<(), io::Error> {
        // Ensure all dependencies exist.
        for task in self.tasks.values() {
            for dep in &task.deps {
                if !self.tasks.contains_key(dep) {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidInput,
                        format!("Task '{}' depends on unknown task '{}'", task.id, dep),
                    ));
                }
            }
        }

        // Detect cycles via Kahn's algorithm.
        let mut indegree: HashMap<&String, usize> = HashMap::new();
        for (id, task) in &self.tasks {
            indegree.entry(id).or_insert(0);
            for dep in &task.deps {
                *indegree.entry(dep).or_insert(0) += 1;
            }
        }

        let mut queue: Vec<&String> = indegree
            .iter()
            .filter_map(|(id, &deg)| if deg == 0 { Some(*id) } else { None })
            .collect();

        let mut visited = 0usize;
        while let Some(node) = queue.pop() {
            visited += 1;
            if let Some(task) = self.tasks.get(node) {
                for dependent in self
                    .tasks
                    .values()
                    .filter(|t| t.deps.contains(node))
                    .map(|t| &t.id)
                {
                    if let Some(cnt) = indegree.get_mut(dependent) {
                        *cnt -= 1;
                        if *cnt == 0 {
                            queue.push(dependent);
                        }
                    }
                }
            }
        }

        if visited != self.tasks.len() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "Task graph contains a cycle",
            ));
        }

        Ok(())
    }
}

/// Executes a `TaskGraph` respecting dependencies and a configurable concurrency
/// limit. It integrates the `ToolRegistry` and applies guardrails for each
/// execution.
///
/// The executor returns `Ok(())` when all tasks succeed; the first failure aborts
/// the whole run and propagates the error.
pub struct ExecutorAgent {
    runner: CommandRunner,
    concurrency: usize,
}

impl ExecutorAgent {
    /// Create a new executor.
    ///
    /// * `runner` – The `CommandRunner` used for raw command execution.
    /// * `concurrency` – Maximum number of tasks to run in parallel. Values
    ///   less than 1 are treated as 1.
    pub fn new(runner: CommandRunner, concurrency: usize) -> Self {
        let cap = if concurrency == 0 { 1 } else { concurrency };
        Self {
            runner,
            concurrency: cap,
        }
    }

    /// Execute the provided `TaskGraph`.
    pub fn execute(&self, graph: TaskGraph) -> Result<(), io::Error> {
        graph.validate()?;

        // Build indegree map and dependents list.
        let mut indegree: HashMap<String, usize> = HashMap::new();
        let mut dependents: HashMap<String, Vec<String>> = HashMap::new();

        for (id, task) in &graph.tasks {
            indegree.entry(id.clone()).or_insert(0);
            for dep in &task.deps {
                *indegree.entry(id.clone()).or_insert(0) += 1;
                dependents
                    .entry(dep.clone())
                    .or_insert_with(Vec::new)
                    .push(id.clone());
            }
        }

        // Channel for ready tasks.
        let (tx, rx) = mpsc::channel::<String>();
        let tx_arc = Arc::new(tx);
        let rx = Arc::new(Mutex::new(rx));

        // Seed initial ready tasks (indegree == 0).
        for (id, &deg) in indegree.iter() {
            if deg == 0 {
                let _ = tx_arc.send(id.clone());
            }
        }

        // Shared state for tracking completion.
        let indegree_arc = Arc::new(Mutex::new(indegree));
        let dependents_arc = Arc::new(dependents);
        let tasks_arc = Arc::new(graph.tasks);
        let runner_arc = Arc::new(self.runner);
        let error_flag = Arc::new(Mutex::new(None));

        // Worker threads.
        let mut handles = Vec::new();
        for _ in 0..self.concurrency {
            let rx = Arc::clone(&rx);
            let indegree = Arc::clone(&indegree_arc);
            let dependents = Arc::clone(&dependents_arc);
            let tasks = Arc::clone(&tasks_arc);
            let runner = Arc::clone(&runner_arc);
            let tx = Arc::clone(&tx_arc);
            let err_flag = Arc::clone(&error_flag);

            let handle = thread::spawn(move || {
                loop {
                    let task_id = {
                        let lock = rx.lock().unwrap();
                        lock.recv()
                    };
                    let task_id = match task_id {
                        Ok(id) => id,
                        Err(_) => break,
                    };

                    // Early exit if an earlier task failed.
                    if err_flag.lock().unwrap().is_some() {
                        break;
                    }

                    let task = match tasks.get(&task_id) {
                        Some(t) => t.clone(),
                        None => {
                            let mut err = err_flag.lock().unwrap();
                            *err = Some(io::Error::new(
                                io::ErrorKind::NotFound,
                                format!("Task '{}' not found in registry", task_id),
                            ));
                            break;
                        }
                    };

                    // Execute the task.
                    let exec_res = if let Some(_tool) = get_tool(&task.tool) {
                        // Use the tool registry.
                        let arg_refs: Vec<&str> = task.args.iter().map(|s| s.as_str()).collect();
                        execute_tool(&task.tool, &arg_refs, &Path::new("."))
                    } else {
                        // Fallback to raw command execution.
                        let mut cmd = task.tool.clone();
                        for a in &task.args {
                            cmd.push(' ');
                            cmd.push_str(a);
                        }
                        runner.run(&cmd)
                    };

                    if let Err(e) = exec_res {
                        // Record first error and stop further processing.
                        let mut err = err_flag.lock().unwrap();
                        if err.is_none() {
                            *err = Some(e);
                        }
                        break;
                    }

                    // Update dependents' indegree.
                    if let Some(children) = dependents.get(&task_id) {
                        let mut indeg = indegree.lock().unwrap();
                        for child in children {
                            if let Some(cnt) = indeg.get_mut(child) {
                                *cnt -= 1;
                                if *cnt == 0 {
                                    let _ = tx.send(child.clone());
                                }
                            }
                        }
                    }
                }
            });
            handles.push(handle);
        }

        // Wait for workers.
        for h in handles {
            let _ = h.join();
        }

        // Propagate any error.
        let maybe_err = {
            let mut guard = error_flag.lock().unwrap();
            guard.take()
        };
        if let Some(e) = maybe_err {
            Err(e)
        } else {
            Ok(())
        }
    }
}

/* -------------------------------------------------------------------------- */
/*                     Existing Autonomous Runner Logic                        */
/* -------------------------------------------------------------------------- */

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
    /// Maximum self‑healing attempts per failing command.
    max_heal_iters: u32,
}

impl AutonomousRunner {
    /// Creates a new `AutonomousRunner`.
    ///
    /// * `planner_cmd` – Command that generates or updates the plan.
    /// * `pipeline_cmd` – Command that consumes the plan and performs the work.
    /// * `watch_path` – Directory to monitor for source changes.
    /// * `poll_interval` – How often to poll the filesystem for changes.
    /// * `runner` – `CommandRunner` used for executing both commands (retries, etc.).
    /// * `max_heal_iters` – Upper bound for the self‑healing loop (e.g., 3).
    pub fn new<P: AsRef<Path>>(
        planner_cmd: &str,
        pipeline_cmd: &str,
        watch_path: P,
        poll_interval: Duration,
        runner: CommandRunner,
        max_heal_iters: u32,
    ) -> io::Result<Self> {
        let watcher = FileWatcher::new(watch_path)?;
        Ok(Self {
            planner_cmd: planner_cmd.to_string(),
            pipeline_cmd: pipeline_cmd.to_string(),
            watcher,
            runner,
            poll_interval,
            max_heal_iters,
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
        run_with_self_healing(&self.planner_cmd, &self.runner, self.max_heal_iters)
    }

    fn execute_pipeline(&self) -> Result<String, io::Error> {
        run_with_self_healing(&self.pipeline_cmd, &self.runner, self.max_heal_iters)
    }
}

/* -------------------------------------------------------------------------- */
/*                     Self‑Healing Helper Function                           */
/* -------------------------------------------------------------------------- */

/// Timeline entry used for observability.
#[derive(Debug, Clone)]
pub struct TimelineEntry {
    pub task: String,
    pub start: SystemTime,
    pub end: SystemTime,
    pub duration: Duration,
    pub agent: String,
    pub llm_provider: String,
    pub tokens_used: u64,
    pub verdict: String,
}

/// Global timeline collector.
static GLOBAL_TIMELINE: Lazy<Mutex<Vec<TimelineEntry>>> = Lazy::new(|| Mutex::new(Vec::new()));

/// Record a timeline entry.
fn record_timeline(entry: TimelineEntry) {
    let mut timeline = GLOBAL_TIMELINE.lock().unwrap();
    timeline.push(entry);
}

/// Runs a command using the provided `CommandRunner`. If the command fails,
/// attempts up to `max_heal` automatic fixes:
///   1. Capture the latest log for the command.
///   2. Obtain a `git diff` of the repository.
///   3. Ask the LLM to propose a minimal patch.
///   4. Apply the patch via the editor module.
///   5. Retry the original command.
///
/// If all attempts are exhausted, a new corrective task is enqueued via the
/// `PlannerAgent` and an error is returned.
///
/// Returns the command's stdout on success.
fn run_with_self_healing(
    command: &str,
    runner: &CommandRunner,
    max_heal: u32,
) -> Result<String, io::Error> {
    // Initial attempt (may be retried by the runner's own retry policy).
    let mut attempt = 0;
    let start_time = SystemTime::now();

    if *GLOBAL_DRY_RUN.lock().unwrap() {
        add_dry_run_report(format!(
            "Dry-run: Would run self‑healing command '{}'",
            command
        ));
        return Ok(String::new());
    }

    loop {
        match runner.run(command) {
            Ok(out) => {
                let end_time = SystemTime::now();
                let duration = end_time
                    .duration_since(start_time)
                    .unwrap_or_else(|_| Duration::from_secs(0));
                record_timeline(TimelineEntry {
                    task: command.to_string(),
                    start: start_time,
                    end: end_time,
                    duration,
                    agent: "runner".to_string(),
                    llm_provider: "none".to_string(),
                    tokens_used: 0,
                    verdict: "success".to_string(),
                });
                return Ok(out);
            }
            Err(err) => {
                attempt += 1;
                error!(
                    "Command '{}' failed (attempt {}): {}",
                    command, attempt, err
                );

                if attempt > max_heal {
                    // Exhausted self‑healing attempts – hand off to planner.
                    let end_time = SystemTime::now();
                    let duration = end_time
                        .duration_since(start_time)
                        .unwrap_or_else(|_| Duration::from_secs(0));
                    record_timeline(TimelineEntry {
                        task: command.to_string(),
                        start: start_time,
                        end: end_time,
                        duration,
                        agent: "runner".to_string(),
                        llm_provider: "none".to_string(),
                        tokens_used: 0,
                        verdict: "failure".to_string(),
                    });

                    let task_desc = format!(
                        "Self‑healing exhausted for command '{}'. \
                         Consider adding missing crates, fixing imports, or other manual fixes.",
                        command
                    );
                    // TODO: enqueue corrective task when PlannerAgent is available.
                    info!("Self‑healing exhausted for command '{}'. Manual intervention may be required.", command);
                    return Err(io::Error::new(
                        io::ErrorKind::Other,
                        format!(
                            "Command '{}' failed after {} self‑healing attempts",
                            command, max_heal
                        ),
                    ));
                }

                // --- Gather context for the LLM ---
                // 1. Full log.
                let task_name = command.split_whitespace().next().unwrap_or("unknown");
                let log_path = Path::new("./.agent/logs").join(format!("{}.log", task_name));
                let log_content = std::fs::read_to_string(&log_path).unwrap_or_else(|e| {
                    warn!("Unable to read log file '{}': {}", log_path.display(), e);
                    String::new()
                });

                // 2. Current diff.
                let diff = match run_command("git diff") {
                    Ok(d) => d,
                    Err(e) => {
                        warn!("Failed to obtain git diff: {}", e);
                        String::new()
                    }
                };

                // 3. Ask LLM for a minimal patch.
                let patch = match llm::propose_patch(&log_content, &diff) {
                    Ok(p) => p,
                    Err(e) => {
                        warn!("LLM failed to propose a patch: {}", e);
                        // Skip to next iteration (which will eventually hit max_heal).
                        continue;
                    }
                };

                // 4. Apply the patch.
                if let Err(e) = editor::apply_patch(&patch) {
                    warn!("Failed to apply patch from LLM: {}", e);
                    // Continue to next iteration; maybe another attempt will work.
                    continue;
                } else {
                    info!(
                        "Applied LLM‑generated patch (attempt {} of {}).",
                        attempt, max_heal
                    );
                }

                // Loop will retry the original command after the patch.
            }
        }
    }
}

/* -------------------------------------------------------------------------- */
/*                     Convenience Entry‑point                                 */
/* -------------------------------------------------------------------------- */

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
    let max_heal_iters = 3;
    let mut autonomous = AutonomousRunner::new(
        planner_cmd,
        pipeline_cmd,
        watch_path,
        poll_interval,
        runner,
        max_heal_iters,
    )?;
    autonomous.run()
}

// The `walkdir` crate is used for recursive directory traversal. If the project
// does not already depend on it, add `walkdir = "2"` to Cargo.toml. This comment
// is left here to remind maintainers of the required dependency.
