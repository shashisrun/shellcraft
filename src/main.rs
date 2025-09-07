// src/main.rs

use anyhow::{bail, Context, Result};
use dialoguer::{theme::ColorfulTheme, Input};
use std::env;
use std::fs::{self, read_to_string, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::{SystemTime, UNIX_EPOCH};

use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};

// Minimal stub for the `ctrlc` crate to allow compilation without the external dependency.
mod ctrlc {
    pub fn set_handler<F: FnMut() + Send + 'static>(mut _f: F) -> Result<(), ()> {
        // In a real implementation this would register a signal handler.
        // Here we simply do nothing and report success.
        Ok(())
    }
}

use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::Command;
use tokio::time::{sleep, Duration};

mod diff;
mod editor;
mod fsutil;
mod llm;
mod planner;
mod runner;
mod ui;

const VERSION: &str = env!("CARGO_PKG_VERSION");

#[allow(dead_code)]
#[derive(Clone, Debug)]
struct RunResult {
    exit_code: i32,
    duration_ms: u128,
    log_tail: String,
    full_log_path: PathBuf,
    command_line: String,
}

#[derive(Clone, Debug)]
struct CommandSpec {
    cmd: String,
    args: Vec<String>,
    workdir: PathBuf,
    env: Vec<(String, String)>,
    log_path: PathBuf,
}

#[tokio::main]
async fn main() -> Result<()> {
    // graceful shutdown flag
    let shutdown = Arc::new(AtomicBool::new(false));
    {
        let shutdown = shutdown.clone();
        ctrlc::set_handler(move || {
            shutdown.store(true, Ordering::SeqCst);
        })
        .expect("Error setting Ctrl-C handler");
    }

    // optional flags
    let mut export_patch = false;
    let mut patch_dir = PathBuf::from("diffs");
    let mut autonomous = false;
    let mut subcommand: Option<String> = None;
    let mut sub_args: Vec<String> = Vec::new();

    // parse arguments
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        if arg == "--export-patch" {
            export_patch = true;
        } else if let Some(dir) = arg.strip_prefix("--patch-dir=") {
            patch_dir = PathBuf::from(dir);
        } else if arg == "--autonomous" {
            autonomous = true;
        } else if subcommand.is_none() {
            subcommand = Some(arg);
            sub_args.extend(args.map(|a| a));
            break;
        }
    }

    // handle subcommands before entering autonomous mode
    if let Some(cmd) = subcommand {
        match cmd.as_str() {
            "run" => {
                if sub_args.is_empty() {
                    bail!("Usage: <program> run <code snippet or file path>");
                }
                let first = &sub_args[0];
                let code = if Path::new(first).exists() {
                    read_to_string(first)
                        .with_context(|| format!("Failed to read code file {}", first))?
                } else {
                    sub_args.join(" ")
                };
                let output =
                    editor::execute_code(&code).with_context(|| "Failed to execute code")?;
                ui::print(&output);
                return Ok(());
            }
            _ => bail!("Unknown subcommand: {}", cmd),
        }
    }

    if export_patch {
        fs::create_dir_all(&patch_dir)
            .with_context(|| format!("Failed to create patch directory {}", patch_dir.display()))?;
    }

    ui::banner();

    // choose project once
    let root: String = Input::with_theme(&ColorfulTheme::default())
        .with_prompt("Project folder")
        .default(std::env::current_dir()?.to_string_lossy().to_string())
        .interact_text()?;
    let root = PathBuf::from(root);
    if !root.is_dir() {
        bail!("Not a directory: {}", root.display());
    }

    // Load .agent.env if present
    let mut agent_env = read_agent_env(&root)?;
    // Export .agent.env into current process so llm can read it
    for (k, v) in &agent_env {
        env::set_var(k, v);
    }
    // If MODEL_ID is present, tell llm module
    if let Ok(mid) = env::var("MODEL_ID") {
        if !mid.trim().is_empty() {
            llm::set_model_id(&mid);
        }
    }

    if autonomous {
        // Spawn autonomous self‑assessment loop (version checks, health checks, etc.)
        let self_assess_handle = {
            let shutdown_clone = shutdown.clone();
            let root_clone = root.clone();
            tokio::spawn(async move {
                self_assessment_loop(root_clone, shutdown_clone).await;
            })
        };

        ui::info("\nAutonomous mode: running periodic planning and execution cycles. Press Ctrl‑C to stop.");

        // Autonomous execution loop
        autonomous_loop(
            root.clone(),
            export_patch,
            patch_dir.clone(),
            &mut agent_env,
            shutdown.clone(),
        )
        .await?;

        // Ensure self‑assessment task finishes
        if let Err(e) = self_assess_handle.await {
            ui::error(&format!("Self‑assessment task failed: {e:?}"));
        }
    } else {
        ui::info("\nAutonomous mode not enabled. Exiting.");
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Autonomous execution loop – periodically invoke planner & runner
// ---------------------------------------------------------------------------
async fn autonomous_loop(
    root: PathBuf,
    export_patch: bool,
    patch_dir: PathBuf,
    env: &mut Vec<(String, String)>,
    shutdown: Arc<AtomicBool>,
) -> Result<()> {
    // Define the interval between autonomous cycles
    let interval = Duration::from_secs(30);

    loop {
        if shutdown.load(Ordering::SeqCst) {
            ui::info("\nReceived Ctrl‑C, exiting autonomous mode.");
            break;
        }

        // Fixed request describing the autonomous intent
        let request = "Perform autonomous maintenance: analyze the repository, generate needed changes, and apply them.".to_string();

        if let Err(e) = high_level_orchestrate(&root, &request, export_patch, &patch_dir, env).await {
            ui::error(&format!("Autonomous orchestrator error: {e}"));
            log_diagnostic(&root, &format!("Autonomous orchestrator error: {e}")).ok();
        }

        // Wait for the next cycle or early exit on shutdown
        let mut elapsed = Duration::from_secs(0);
        while elapsed < interval {
            if shutdown.load(Ordering::SeqCst) {
                ui::info("\nReceived Ctrl‑C during wait, exiting.");
                return Ok(());
            }
            let remaining = interval - elapsed;
            let sleep_dur = if remaining > Duration::from_secs(1) {
                Duration::from_secs(1)
            } else {
                remaining
            };
            sleep(sleep_dur).await;
            elapsed += sleep_dur;
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// High‑level orchestrator (think → understand → plan → implement → verify)
// ---------------------------------------------------------------------------
async fn high_level_orchestrate(
    root: &Path,
    request: &str,
    export_patch: bool,
    patch_dir: &Path,
    env: &mut Vec<(String, String)>,
) -> Result<()> {
    // 1) Think
    ui::info("🤔 Thinking about the request...");

    // 2) Understand – simple project inspection
    ui::info("🔍 Understanding the project structure...");
    let mut top_files = Vec::new();
    for entry in fs::read_dir(root)? {
        let entry = entry?;
        if entry.file_type()?.is_file() {
            if let Some(name) = entry.file_name().to_str() {
                top_files.push(name.to_string());
            }
        }
    }
    ui::info(&format!(
        "Found {} top‑level files (showing up to 10):",
        top_files.len()
    ));
    for f in top_files.iter().take(10) {
        ui::print(&format!("- {}", f));
    }

    // 3) Retry loop for planning & implementation
    let mut last_err: Option<anyhow::Error> = None;
    for attempt in 1..=3 {
        ui::info(&format!("🚀 Attempt {}/3", attempt));
        match orchestrate(root, request, export_patch, patch_dir, env).await {
            Ok(_) => {
                ui::success("✅ Orchestration succeeded");
                // 4) Verify – run build/tests if applicable
                match verify_project(root).await {
                    Ok(_) => {
                        ui::success("✅ Verification passed");
                        // 5) Summary
                        ui::info("\n📋 Summary of changes:");
                        ui::print(&format!("Request: {}", request));
                        ui::print(&format!("Project root: {}", root.display()));
                        ui::print(&format!("Export patches: {}", export_patch));
                        return Ok(());
                    }
                    Err(e) => {
                        ui::error(&format!("Verification failed: {e}"));
                        log_diagnostic(root, &format!("Verification error: {e}")).ok();
                        last_err = Some(e);
                    }
                }
            }
            Err(e) => {
                ui::error(&format!("Orchestration error: {e}"));
                log_diagnostic(
                    root,
                    &format!("Orchestration attempt {} error: {e}", attempt),
                )
                .ok();
                last_err = Some(e);
            }
        }
    }
    Err(last_err.unwrap_or_else(|| anyhow::anyhow!("Orchestration failed after retries")))
}

// ---------------------------------------------------------------------------
// Verification step – run cargo test if a Rust project is detected
// ---------------------------------------------------------------------------
async fn verify_project(root: &Path) -> Result<()> {
    if root.join("Cargo.toml").exists() {
        ui::info("Running `cargo test` for verification...");
        let spec = CommandSpec {
            cmd: "cargo".to_string(),
            args: vec!["test".to_string(), "--quiet".to_string()],
            workdir: root.to_path_buf(),
            env: env::vars().collect(),
            log_path: root.join("cargo_test.log"),
        };
        let res = run_and_capture(spec).await?;
        if res.exit_code != 0 {
            bail!("cargo test failed with exit code {}", res.exit_code);
        }
    } else {
        ui::info("No Cargo.toml found, skipping verification.");
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Orchestrator (autopilot per cycle)
// ---------------------------------------------------------------------------
async fn orchestrate(
    root: &Path,
    request: &str,
    export_patch: bool,
    patch_dir: &Path,
    env: &mut Vec<(String, String)>,
) -> Result<()> {
    let mut patch_counter: usize = 1;

    loop {
        // 1) Planning
        let pb = ui::spinner("Planning…");
        let plan = match planner::plan_changes(root, request).await {
            Ok(p) => p,
            Err(e) => {
                pb.finish_and_clear();
                bail!("Planner failed: {e}");
            }
        };
        pb.finish_and_clear();

        // 2) Informational chat? Summarize the picked context files
        if plan.edit.is_empty() && plan.actions.is_empty() {
            let mut picked = plan.read.clone();
            if picked.is_empty() {
                // fallback picks
                for probe in &[
                    "README.md",
                    "readme.md",
                    "Cargo.toml",
                    "package.json",
                    "pyproject.toml",
                    "src/main.rs",
                ] {
                    if root.join(probe).exists() {
                        picked.push(probe.to_string());
                    }
                }
            }
            // read up to ~80KB total
            let mut total = 0usize;
            let mut blobs = Vec::new();
            for rel in picked.iter().take(8) {
                let abs = root.join(rel);
                if let Ok(mut s) = fs::read_to_string(&abs) {
                    if s.len() > 20 * 1024 {
                        s.truncate(20 * 1024);
                        s.push_str("\n…(truncated)…");
                    }
                    total += s.len();
                    blobs.push(format!("--- FILE: {}\n{}", rel, s));
                    if total > 80 * 1024 {
                        break;
                    }
                }
            }

            let system = r#"You are a senior engineer. Summarize the project for a teammate who just joined.
Be concise and practical:
- What is this project? What does it do?
- Key entry points / binaries / scripts.
- How to run (dev) & build/test commands inferred from files.
- Important dependencies / integrations.
- Any missing config/env secrets you can infer.
- 3–6 concrete next steps or fixes if something looks off."#;

            let user = format!(
                "User asked:\n{}\n\nHere are relevant files from the repo root:\n{}\n",
                request,
                blobs.join("\n")
            );

            match llm::chat_text(system, &user).await {
                Ok(summary) => {
                    ui::print("\nProject summary:");
                    ui::print(&summary.trim());
                }
                Err(e) => {
                    if !plan.notes.trim().is_empty() {
                        ui::print("\nPlanner notes:");
                        ui::print(&plan.notes);
                    }
                    ui::error(&format!("Summarizer error: {e}"));
                    log_diagnostic(root, &format!("Summarizer error: {e}")).ok();
                }
            }
            return Ok(());
        }

        if !plan.notes.trim().is_empty() {
            ui::print("\nPlanner notes:");
            ui::print(&plan.notes);
        }

        // 3) Proposed edits
        if !plan.edit.is_empty() {
            let mut proposals: Vec<(PathBuf, String, String, bool)> = vec![];
            for ep in &plan.edit {
                let abs = root.join(&ep.path);
                let (current, is_new) = match fsutil::read_to_string(&abs) {
                    Ok(c) => (c, false),
                    Err(_) => (String::new(), true),
                };
                let pb2 = ui::spinner(&format!(
                    "Proposing edit: {}{}",
                    ep.path,
                    if is_new { " (new file)" } else { "" }
                ));
                let instr = if current.is_empty() {
                    format!(
                        "{request}\n\nThis file does not yet exist. Create it with appropriate contents.\nFile-specific intent: {}",
                        &ep.intent
                    )
                } else {
                    format!("{request}\n\nFile-specific intent: {}", &ep.intent)
                };
                let proposed = match llm::propose_edit(llm::EditReq {
                    file_path: ep.path.clone(),
                    file_content: current.clone(),
                    instruction: instr,
                    ..Default::default()
                })
                .await
                {
                    Ok(p) => p,
                    Err(e) => {
                        pb2.finish_and_clear();
                        bail!("LLM failed on {}: {e}", ep.path);
                    }
                };
                pb2.finish_and_clear();
                proposals.push((abs, current, proposed, is_new));
            }

            // Show and apply all (autopilot)
            ui::print("\n=== Applying proposals ===");
            for (abs, cur, prop, _is_new) in &proposals {
                let rel = path_rel(root, abs);
                ui::print(&format!("\n## {}", rel));
                ui::print(&diff::unified_colored(cur, prop, &rel));

                if export_patch {
                    let patch_name = format!("{:03}.patch", patch_counter);
                    patch_counter += 1;
                    let patch_path = patch_dir.join(patch_name);
                    let diff_text = diff::unified_colored(cur, prop, &rel);
                    fs::write(&patch_path, diff_text).with_context(|| {
                        format!("Failed to write patch {}", patch_path.display())
                    })?;
                    ui::success(&format!("Exported patch {}", patch_path.display()));
                } else {
                    fsutil::atomic_write(abs, prop)?;
                    let now = SystemTime::now()
                        .duration_since(UNIX_EPOCH)
                        .unwrap()
                        .as_secs();
                    let mut log = OpenOptions::new()
                        .create(true)
                        .append(true)
                        .open("agent.log")?;
                    writeln!(log, "{} {}", now, abs.display())?;
                    ui::success(&format!("Saved {}", path_rel(root, abs)));
                }
            }
        }

        // 4) Execute actions (build/test/run/etc.)
        for action in &plan.actions {
            match action {
                planner::Action::Run {
                    program,
                    args,
                    workdir,
                    log_hint,
                    ..
                } => {
                    let wd = workdir
                        .as_ref()
                        .map(PathBuf::from)
                        .unwrap_or_else(|| root.to_path_buf());
                    let log_name = match log_hint.as_deref() {
                        Some("test") => "test.log",
                        Some("build") => "build.log",
                        Some("run") => "run.log",
                        _ => "command.log",
                    };
                    let spec = CommandSpec {
                        cmd: program.clone(),
                        args: args.clone(),
                        workdir: wd.clone(),
                        env: env.clone(),
                        log_path: root.join(log_name),
                    };
                    match run_and_capture(spec.clone()).await {
                        Ok(res) => {
                            ui::success(&format!(
                                "Command `{}` exited {} after {} ms",
                                res.command_line, res.exit_code, res.duration_ms
                            ));
                        }
                        Err(e) => {
                            ui::error(&format!("Command error: {e}"));
                            log_diagnostic(root, &format!("Command error: {e}")).ok();
                        }
                    }
                }
            }
        }

        // Normal completion – break out of retry loop
        break;
    }

    Ok(())
}

fn path_rel(root: &Path, p: &Path) -> String {
    pathdiff::diff_paths(p, root)
        .unwrap_or_else(|| p.to_path_buf())
        .to_string_lossy()
        .to_string()
}

// Reads .agent.env (key=value per line, ignores comments) into vector of pairs.
fn read_agent_env(root: &Path) -> Result<Vec<(String, String)>> {
    let env_path = root.join(".agent.env");
    if !env_path.exists() {
        return Ok(vec![]);
    }
    let content = fs::read_to_string(&env_path)?;
    let mut pairs = Vec::new();
    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if let Some(eq) = line.find('=') {
            let key = line[..eq].trim().to_string();
            let val = line[eq + 1..].trim().to_string();
            pairs.push((key, val));
        }
    }
    Ok(pairs)
}

// Upserts key=value pairs into .agent.env (creates file if missing).
fn upsert_agent_env(root: &Path, updates: &[(String, String)]) -> Result<()> {
    let env_path = root.join(".agent.env");
    // merge with existing
    let mut existing = read_agent_env(root)?;
    for (k, v) in updates {
        if let Some(pos) = existing.iter().position(|(kk, _)| kk == k) {
            existing[pos] = (k.clone(), v.clone());
        } else {
            existing.push((k.clone(), v.clone()));
        }
    }
    let mut file = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(&env_path)?;
    for (k, v) in existing {
        writeln!(file, "{}={}", k, v)?;
    }
    Ok(())
}

// Async runner with streaming & log capture.
async fn run_and_capture(spec: CommandSpec) -> Result<RunResult> {
    // Prepare log file with header
    let mut log_file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&spec.log_path)?;
    let timestamp = chrono::Utc::now().to_rfc3339();
    let cmd_line = format!("{} {}", spec.cmd, spec.args.join(" "));
    writeln!(
        log_file,
        "==== CMD @ {} (cmd: {}) ====",
        timestamp, cmd_line
    )?;

    // Spawn process
    let mut command = Command::new(&spec.cmd);
    command
        .args(&spec.args)
        .current_dir(&spec.workdir)
        .envs(spec.env.iter().map(|(k, v)| (k, v)))
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    let mut child = command.spawn().with_context(|| "Failed to spawn command")?;

    let start = std::time::Instant::now();

    // Capture stdout and stderr
    let stdout = child
        .stdout
        .take()
        .expect("Child process stdout not captured");
    let stderr = child
        .stderr
        .take()
        .expect("Child process stderr not captured");

    let mut stdout_reader = BufReader::new(stdout).lines();
    let mut stderr_reader = BufReader::new(stderr).lines();

    let mut combined_tail = String::new();
    let mut stdout_done = false;
    let mut stderr_done = false;

    while !(stdout_done && stderr_done) {
        tokio::select! {
            res = stdout_reader.next_line(), if !stdout_done => {
                match res {
                    Ok(Some(l)) => {
                        ui::print(&l);
                        writeln!(log_file, "{}", l)?;
                        combined_tail.push_str(&l);
                        combined_tail.push('\n');
                    }
                    Ok(None) => { stdout_done = true; }
                    Err(e) => { ui::error(&format!("stdout read error: {e}")); stdout_done = true; }
                }
            }
            res = stderr_reader.next_line(), if !stderr_done => {
                match res {
                    Ok(Some(l)) => {
                        ui::error(&l);
                        writeln!(log_file, "{}", l)?;
                        combined_tail.push_str(&l);
                        combined_tail.push('\n');
                    }
                    Ok(None) => { stderr_done = true; }
                    Err(e) => { ui::error(&format!("stderr read error: {e}")); stderr_done = true; }
                }
            }
        }
    }

    let status = child.wait().await?;
    let duration = start.elapsed().as_millis();

    // Trim tail to last 4000 chars
    if combined_tail.len() > 4000 {
        combined_tail = combined_tail[combined_tail.len() - 4000..].to_string();
    }

    Ok(RunResult {
        exit_code: status.code().unwrap_or(-1),
        duration_ms: duration,
        log_tail: combined_tail,
        full_log_path: spec.log_path,
        command_line: cmd_line,
    })
}

// Write a diagnostic entry to a central log for post‑mortem analysis.
fn log_diagnostic(root: &Path, msg: &str) -> Result<()> {
    let diag_path = root.join("agent_diagnostics.log");
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&diag_path)?;
    let ts = chrono::Utc::now().to_rfc3339();
    writeln!(file, "[{}] {}", ts, msg)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Self‑assessment loop – checks for newer versions and performs self‑updates
// ---------------------------------------------------------------------------
async fn self_assessment_loop(root: PathBuf, shutdown: Arc<AtomicBool>) {
    let interval = Duration::from_secs(300); // every 5 minutes
    loop {
        if shutdown.load(Ordering::SeqCst) {
            ui::info("\nSelf‑assessment loop exiting due to shutdown.");
            break;
        }

        if let Some(new_ver) = check_for_new_version(&root).await {
            if new_ver != VERSION {
                if let Err(e) = self_update(&new_ver, &root).await {
                    ui::error(&format!("Self‑update failed: {e}"));
                    log_diagnostic(&root, &format!("Self‑update error: {e}")).ok();
                }
            }
        }

        // Wait for next interval, respecting shutdown
        let mut elapsed = Duration::from_secs(0);
        while elapsed < interval {
            if shutdown.load(Ordering::SeqCst) {
                ui::info("\nSelf‑assessment loop exiting during wait.");
                return;
            }
            let remaining = interval - elapsed;
            let sleep_dur = if remaining > Duration::from_secs(1) {
                Duration::from_secs(1)
            } else {
                remaining
            };
            sleep(sleep_dur).await;
            elapsed += sleep_dur;
        }
    }
}

// Simple version check – reads a `.latest_version` file in the project root.
// In a real system this could query a remote service.
async fn check_for_new_version(root: &Path) -> Option<String> {
    let path = root.join(".latest_version");
    if path.exists() {
        if let Ok(v) = fs::read_to_string(&path) {
            let v = v.trim().to_string();
            if !v.is_empty() {
                return Some(v);
            }
        }
    }
    None
}

// Placeholder self‑update routine.
// In a production system this would download and replace the binary.
async fn self_update(new_version: &str, root: &Path) -> Result<()> {
    ui::info(&format!(
        "New version {} detected. Initiating self‑update.",
        new_version
    ));
    // Record the new version in .agent.env for visibility.
    upsert_agent_env(root, &[("AGENT_VERSION".to_string(), new_version.to_string())])?;
    ui::success(&format!("Self‑update simulated to version {}", new_version));
    Ok(())
}