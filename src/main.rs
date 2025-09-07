// src/main.rs

use anyhow::{bail, Context, Result};
use console::style;
use dialoguer::{theme::ColorfulTheme, Input};
use indicatif::{ProgressBar, ProgressStyle};
use std::env;
use std::fs::{self, read_to_string, OpenOptions};
use std::io::{self, BufRead, Write};
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::{SystemTime, UNIX_EPOCH};

use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::Command;

mod diff;
mod editor;
mod fsutil;
mod llm;
mod planner;

const VERSION: &str = env!("CARGO_PKG_VERSION");

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
    // optional flags
    let mut export_patch = false;
    let mut patch_dir = PathBuf::from("diffs");
    let mut subcommand: Option<String> = None;
    let mut sub_args: Vec<String> = Vec::new();

    // parse arguments
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        if arg == "--export-patch" {
            export_patch = true;
        } else if let Some(dir) = arg.strip_prefix("--patch-dir=") {
            patch_dir = PathBuf::from(dir);
        } else if subcommand.is_none() {
            subcommand = Some(arg);
            sub_args.extend(args.map(|a| a));
            break;
        }
    }

    // handle subcommands before entering interactive mode
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
                let output = editor::execute_code(&code).with_context(|| "Failed to execute code")?;
                println!("{}", output);
                return Ok(());
            }
            _ => bail!("Unknown subcommand: {}", cmd),
        }
    }

    if export_patch {
        fs::create_dir_all(&patch_dir).with_context(|| {
            format!("Failed to create patch directory {}", patch_dir.display())
        })?;
    }

    banner();

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

    println!(
        "{}",
        style("\nChat mode: Type what you want. Use /help for commands. End input with a line containing '/end'.")
            .dim()
    );

    // Main loop — chat-first + autopilot per turn
    loop {
        let prompt: String = read_multiline_input()?;
        let line = prompt.trim();

        // minimal commands for power users
        if line.starts_with('/') && !prompt.contains('\n') {
            match line {
                "/quit" | "/exit" => return Ok(()),
                "/help" => {
                    println!(
                        "{}",
                        style("\nCommands:\n  /version show version\n  /model   change LLM model at runtime\n  /env KEY=VAL ... set env vars\n  /quit    exit\n").dim()
                    );
                }
                "/version" => println!("{}", style(format!("Version: {}", VERSION)).dim()),
                "/model" => {
                    let new_id: String = Input::with_theme(&ColorfulTheme::default())
                        .with_prompt("Enter new MODEL_ID")
                        .interact_text()?;
                    llm::set_model_id(&new_id);
                    // persist + export
                    upsert_agent_env(&root, &[("MODEL_ID".into(), new_id.clone())])?;
                    env::set_var("MODEL_ID", &new_id);
                    if let Some(pos) = agent_env.iter().position(|(k, _)| k == "MODEL_ID") {
                        agent_env[pos].1 = new_id.clone();
                    } else {
                        agent_env.push(("MODEL_ID".into(), new_id.clone()));
                    }
                    println!("{}", style("Model updated").green());
                }
                other if other.starts_with("/env ") => {
                    let parts: Vec<&str> = other.split_whitespace().skip(1).collect();
                    let mut updates: Vec<(String, String)> = vec![];
                    for kv in parts {
                        if let Some(eq) = kv.find('=') {
                            let k = kv[..eq].to_string();
                            let v = kv[eq + 1..].to_string();
                            updates.push((k.clone(), v.clone()));
                        }
                    }
                    if updates.is_empty() {
                        println!("{}", style("No KEY=VAL pairs provided").red());
                    } else {
                        upsert_agent_env(&root, &updates)?;
                        for (k, v) in updates {
                            env::set_var(&k, &v);
                            if let Some(pos) = agent_env.iter().position(|(kk, _)| kk == &k) {
                                agent_env[pos] = (k, v);
                            } else {
                                agent_env.push((k, v));
                            }
                        }
                        println!("{}", style("Environment updated").green());
                    }
                }
                _ => println!("{}", style("Unknown command. Try /help").dim()),
            }
            continue;
        }

        // === Autopilot for every chat turn ===
        if let Err(e) = orchestrate(&root, &prompt, export_patch, &patch_dir, &mut agent_env).await
        {
            eprintln!("{}", style(format!("Autopilot error: {e}")).red());
        }
    }
}

// ---------------------------------------------------------------------------
// Orchestrator (autopilot per chat turn)
// ---------------------------------------------------------------------------
async fn orchestrate(
    root: &Path,
    request: &str,
    export_patch: bool,
    patch_dir: &Path,
    env: &mut Vec<(String, String)>,
) -> Result<()> {
    let mut patch_counter: usize = 1;

    // 1) Planning
    let pb = spinner("Planning…");
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
                println!("\n{}", style("Project summary:").bold());
                println!("{}", summary.trim());
            }
            Err(e) => {
                if !plan.notes.trim().is_empty() {
                    println!("{}", style("\nPlanner notes:").bold());
                    println!("{}", plan.notes);
                }
                eprintln!("{}", style(format!("Summarizer error: {e}")).red());
            }
        }
        return Ok(());
    }

    if !plan.notes.trim().is_empty() {
        println!("\n{}", style("Planner notes:").bold());
        println!("{}", plan.notes);
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
            let pb2 = spinner(&format!(
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
        println!("{}", style("\n=== Applying proposals ===").bold());
        for (abs, cur, prop, _is_new) in &proposals {
            let rel = path_rel(root, abs);
            println!("{}", style(format!("\n## {}", rel)).bold());
            println!("{}", diff::unified_colored(cur, prop, &rel));

            if export_patch {
                let patch_name = format!("{:03}.patch", patch_counter);
                patch_counter += 1;
                let patch_path = patch_dir.join(patch_name);
                let diff_text = diff::unified_colored(cur, prop, &rel);
                fs::write(&patch_path, diff_text)
                    .with_context(|| format!("Failed to write patch {}", patch_path.display()))?;
                println!("{}", style(format!("Exported patch {}", patch_path.display())).green());
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
                println!("{}", style(format!("Saved {}", path_rel(root, abs))).green());
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
                        println!(
                            "{}",
                            style(format!(
                                "Command `{}` exited {} after {} ms",
                                res.command_line, res.exit_code, res.duration_ms
                            ))
                            .green()
                        );
                    }
                    Err(e) => {
                        eprintln!("{}", style(format!("Command error: {e}")).red());
                    }
                }
            }
        }
    }

    Ok(())
}

fn banner() {
    println!("{}", style("\nShell Craft Coding Agent").cyan().bold());
    println!(
        "{}",
        style("OpenAI-compatible · multi-file plan → proposals · chat-first · new-file support · diff-first, safe save\n")
            .dim()
    );
    println!(
        "{}",
        style("Usage: <program> [--export-patch] [--patch-dir=DIR] [run <code|file>]").dim()
    );
}

fn spinner(msg: &str) -> ProgressBar {
    let pb = ProgressBar::new_spinner();
    pb.enable_steady_tick(std::time::Duration::from_millis(80));
    pb.set_style(ProgressStyle::with_template("{spinner} {msg}").unwrap());
    pb.set_message(msg.to_string());
    pb
}

fn read_multiline_input() -> Result<String> {
    print!("you@agent: ");
    io::stdout().flush()?;
    let stdin = io::stdin();
    let mut lines = Vec::new();
    for line_res in stdin.lock().lines() {
        let line = line_res?;
        if line.trim() == "/end" {
            break;
        }
        lines.push(line);
    }
    Ok(lines.join("\n"))
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
                        println!("{}", l);
                        writeln!(log_file, "{}", l)?;
                        combined_tail.push_str(&l);
                        combined_tail.push('\n');
                    }
                    Ok(None) => { stdout_done = true; }
                    Err(e) => { eprintln!("{}", style(format!("stdout read error: {e}")).red()); stdout_done = true; }
                }
            }
            res = stderr_reader.next_line(), if !stderr_done => {
                match res {
                    Ok(Some(l)) => {
                        println!("{}", style(&l).red());
                        writeln!(log_file, "{}", l)?;
                        combined_tail.push_str(&l);
                        combined_tail.push('\n');
                    }
                    Ok(None) => { stderr_done = true; }
                    Err(e) => { eprintln!("{}", style(format!("stderr read error: {e}")).red()); stderr_done = true; }
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
