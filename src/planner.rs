use anyhow::{Context, Result};
use regex::Regex;
use serde::{Deserialize, Serialize};
use serde_json::{self, Value};
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use crate::fsutil::{file_inventory, FileMeta};
use crate::llm;

// New crate for self‑updating capabilities.
use self_update::backends::github::Update;
use std::env;

/// High-level plan the planner returns.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Plan {
    #[serde(default)]
    pub read: Vec<String>,
    #[serde(default)]
    pub edit: Vec<EditPlan>,
    #[serde(default)]
    pub notes: String,
    #[serde(default)]
    pub actions: Vec<Action>,
    /// Optional signal that can influence execution flow.
    #[serde(default)]
    pub signal: Option<Signal>,
    /// Optional error message to surface to the UI when planning or execution fails.
    #[serde(default)]
    pub error: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct EditPlan {
    pub path: String,
    pub intent: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(tag = "kind")]
pub enum Action {
    #[serde(rename = "run")]
    Run {
        program: String,
        #[serde(default)]
        args: Vec<String>,
        #[serde(default)]
        workdir: Option<String>,
        #[serde(default)]
        log_hint: Option<String>,
        #[serde(default = "default_retries")]
        retries: u32,
        #[serde(default = "default_backoff_ms")]
        backoff_ms: u64,
    },
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub enum Signal {
    Retry,
    Abort,
    Continue,
}

#[derive(Debug, Serialize)]
struct PlanPrompt<'a> {
    user_request: &'a str,
    file_index: &'a [FileMeta],
    guidance: &'a str,
}

fn system_prompt() -> String {
    r#"You are the planning module of a self-upgrading coding agent.

Given: 
- The user's request (natural language)
- A list of project files (relative paths, size, extension)

Return a pure JSON plan describing:
- which files to READ (for context)
- which files to EDIT (with per-file intent)
- which ACTIONS to perform (like build, test, run, format, etc.)

IMPORTANT:
- Respond as pure JSON, no markdown fences.
- Prefer the **fewest** files needed.
- If the task is informational (no edits), return an empty `edit` array and put your summary in `notes`.
- If the task requires executing a command (e.g., build/test/start), include an action of kind `"run"`.

JSON schema:
{
  "read": string[],
  "edit": [ { "path": string, "intent": string } ],
  "actions": [
    { "kind": "run", "program": string, "args": string[], "workdir": string | null, "log_hint": string | null, "retries": number, "backoff_ms": number }
  ],
  "notes": string
}
"#
    .to_string()
}

fn planner_guidance() -> String {
    r#"Heuristics:
- Start with entry points (src/main.rs, index.ts, app routers) and config files.
- If the user asks to "understand current project", set edit=[] and include notes summary.
- Avoid vendor/private keys or secrets.
- Keep plan small (<= 6 files total across read+edit).
- Common actions: cargo build/test/run, npm/yarn/pnpm build/test/start, python -m pytest, etc.
"#
    .to_string()
}

pub async fn plan_changes(root: &Path, user_request: &str) -> Result<Plan> {
    let mut index = file_inventory(root)?;
    if index.len() > 800 {
        index = compact_index(index);
    }

    let prompt = PlanPrompt {
        user_request,
        file_index: &index,
        guidance: &planner_guidance(),
    };

    let llm_value: Result<Value> = llm::chat_json::<Value>(
        &system_prompt(),
        &serde_json::to_string(&prompt).unwrap(),
    )
    .await
    .context("planner LLM failed");

    if let Ok(v) = llm_value {
        if let Ok(mut plan) = serde_json::from_value::<Plan>(v.clone()) {
            if plan.read.is_empty()
                && plan.edit.is_empty()
                && plan.notes.trim().is_empty()
                && plan.actions.is_empty()
            {
                // fall through to fallback
            } else {
                plan.read.retain(|p| root.join(p).exists());
                plan.edit.retain(|e| root.join(&e.path).exists());
                return Ok(plan);
            }
        }
    }

    Ok(fallback_plan(root, user_request, &index))
}

fn fallback_plan(root: &Path, user_request: &str, _index: &[FileMeta]) -> Plan {
    let mut read_set: HashSet<String> = HashSet::new();
    let mut edit_set: HashSet<String> = HashSet::new();

    let token_re = Regex::new(r"[A-Za-z0-9_/\\.-]+\.[A-Za-z0-9]+").unwrap();
    for cap in token_re.captures_iter(user_request) {
        let candidate = cap.get(0).unwrap().as_str();
        let norm = candidate.replace('\\', "/");
        if root.join(&norm).exists() {
            if contains_any(
                user_request,
                &["edit", "modify", "change", "refactor", "add"],
            ) {
                edit_set.insert(norm);
            } else {
                read_set.insert(norm);
            }
        }
    }

    if read_set.is_empty() && edit_set.is_empty() {
        for probe in &[
            "src/main.rs",
            "Cargo.toml",
            "package.json",
            "pyproject.toml",
            "README.md",
            "readme.md",
        ] {
            if root.join(probe).exists() {
                read_set.insert(probe.to_string());
            }
        }
    }

    let mut actions: Vec<Action> = vec![];
    if contains_any(user_request, &["build", "compile"]) {
        if root.join("Cargo.toml").exists() {
            actions.push(Action::Run {
                program: "cargo".into(),
                args: vec!["build".into()],
                workdir: None,
                log_hint: Some("build".into()),
                retries: DEFAULT_RETRIES,
                backoff_ms: DEFAULT_BACKOFF_MS,
            });
        } else if root.join("package.json").exists() {
            actions.push(Action::Run {
                program: "npm".into(),
                args: vec!["run".into(), "build".into()],
                workdir: None,
                log_hint: Some("build".into()),
                retries: DEFAULT_RETRIES,
                backoff_ms: DEFAULT_BACKOFF_MS,
            });
        }
    }
    if contains_any(user_request, &["test", "unit test", "pytest", "cargo test"]) {
        if root.join("Cargo.toml").exists() {
            actions.push(Action::Run {
                program: "cargo".into(),
                args: vec!["test".into()],
                workdir: None,
                log_hint: Some("test".into()),
                retries: DEFAULT_RETRIES,
                backoff_ms: DEFAULT_BACKOFF_MS,
            });
        } else if root.join("pyproject.toml").exists() {
            actions.push(Action::Run {
                program: "python".into(),
                args: vec!["-m".into(), "pytest".into()],
                workdir: None,
                log_hint: Some("test".into()),
                retries: DEFAULT_RETRIES,
                backoff_ms: DEFAULT_BACKOFF_MS,
            });
        } else if root.join("package.json").exists() {
            actions.push(Action::Run {
                program: "npm".into(),
                args: vec!["test".into()],
                workdir: None,
                log_hint: Some("test".into()),
                retries: DEFAULT_RETRIES,
                backoff_ms: DEFAULT_BACKOFF_MS,
            });
        }
    }
    if contains_any(
        user_request,
        &["run", "start server", "start dev", "dev server"],
    ) {
        if root.join("Cargo.toml").exists() {
            actions.push(Action::Run {
                program: "cargo".into(),
                args: vec!["run".into()],
                workdir: None,
                log_hint: Some("run".into()),
                retries: DEFAULT_RETRIES,
                backoff_ms: DEFAULT_BACKOFF_MS,
            });
        } else if root.join("package.json").exists() {
            actions.push(Action::Run {
                program: "npm".into(),
                args: vec!["run".into(), "dev".into()],
                workdir: None,
                log_hint: Some("run".into()),
                retries: DEFAULT_RETRIES,
                backoff_ms: DEFAULT_BACKOFF_MS,
            });
        }
    }

    let mut read: Vec<String> = read_set.into_iter().collect();
    let mut edit: Vec<String> = edit_set.into_iter().collect();
    while read.len() + edit.len() > 6 {
        if !read.is_empty() {
            read.pop();
        } else if !edit.is_empty() {
            edit.pop();
        }
    }

    let edit_plans: Vec<EditPlan> = edit
        .into_iter()
        .map(|p| EditPlan {
            path: p.clone(),
            intent: format!("Apply changes inferred from request: \"{}\"", user_request),
        })
        .collect();

    Plan {
        read,
        edit: edit_plans.clone(),
        actions,
        notes: if edit_plans.is_empty() {
            "No explicit edit targets detected; providing context files and inferred actions."
                .to_string()
        } else {
            String::new()
        },
        signal: None,
        error: None,
    }
}

fn contains_any(hay: &str, needles: &[&str]) -> bool {
    let h = hay.to_lowercase();
    needles.iter().any(|n| h.contains(&n.to_lowercase()))
}

fn compact_index(mut v: Vec<FileMeta>) -> Vec<FileMeta> {
    fn weight(ext: &str) -> i32 {
        match ext {
            "rs" | "ts" | "tsx" | "js" | "jsx" | "py" => 10,
            "toml" | "json" | "yml" | "yaml" | "md" => 8,
            "go" | "rb" | "java" | "kt" | "c" | "h" | "cpp" | "hpp" => 7,
            _ => 1,
        }
    }
    v.sort_by_key(|m| {
        let w = m.ext.as_deref().map(weight).unwrap_or(1);
        let size_bucket = (m.size as i64 / 4096) as i64;
        (-(w as i64), size_bucket)
    });
    v.truncate(800);
    v
}

// -----------------------------------------------------------------------------
// Retry / back‑off configuration
// -----------------------------------------------------------------------------
const DEFAULT_RETRIES: u32 = 3;
const DEFAULT_BACKOFF_MS: u64 = 1_000; // 1 second

fn default_retries() -> u32 {
    DEFAULT_RETRIES
}

fn default_backoff_ms() -> u64 {
    DEFAULT_BACKOFF_MS
}

// -----------------------------------------------------------------------------
// Self‑improvement capabilities
// -----------------------------------------------------------------------------

/// Represents a plan that should be executed at a later time.
#[derive(Debug, Clone)]
pub struct ScheduledTask {
    pub execute_at: Instant,
    pub plan: Plan,
}

// Global in‑memory queue of scheduled tasks.
static TASK_QUEUE: Mutex<Vec<ScheduledTask>> = Mutex::new(Vec::new());

fn self_improvement_system_prompt() -> String {
    r#"You are an autonomous self‑improvement module. Based on developer feedback, produce a JSON plan that describes:
- Files to READ for context.
- Files to EDIT with clear intents.
- Any actions (build, test, run) needed to validate the changes.

Return ONLY the JSON plan, no explanations."#
        .to_string()
}

/// Generate a self‑improvement plan from raw LLM feedback.
pub async fn generate_self_improvement(root: &Path, feedback: &str) -> Result<Plan> {
    // Re‑use the same indexing logic as the normal planner.
    let mut index = file_inventory(root)?;
    if index.len() > 800 {
        index = compact_index(index);
    }

    // Prompt the LLM with the feedback.
    let llm_value: Result<Value> = llm::chat_json::<Value>(
        &self_improvement_system_prompt(),
        feedback,
    )
    .await
    .context("self‑improvement LLM failed");

    if let Ok(v) = llm_value {
        if let Ok(mut plan) = serde_json::from_value::<Plan>(v.clone()) {
            // Ensure paths actually exist.
            plan.read.retain(|p| root.join(p).exists());
            plan.edit.retain(|e| root.join(&e.path).exists());
            return Ok(plan);
        }
    }

    // Fallback: empty plan with a note.
    Ok(Plan {
        read: vec![],
        edit: vec![],
        actions: vec![],
        notes: "Self‑improvement feedback could not be parsed into a plan.".into(),
        signal: None,
        error: None,
    })
}

/// Schedule a self‑improvement plan to be executed after `delay`.
pub async fn schedule_self_improvement(
    root: &Path,
    feedback: &str,
    delay: Duration,
) -> Result<()> {
    let plan = generate_self_improvement(root, feedback).await?;
    let task = ScheduledTask {
        execute_at: Instant::now() + delay,
        plan,
    };
    let mut queue = TASK_QUEUE.lock().unwrap();
    queue.push(task);
    Ok(())
}

/// Retrieve a snapshot of all pending tasks.
pub fn pending_tasks() -> Vec<ScheduledTask> {
    let queue = TASK_QUEUE.lock().unwrap();
    queue.clone()
}

/// Safety check: ensure all paths in a plan stay within the project root.
fn safety_check_plan(root: &Path, plan: &Plan) -> Result<()> {
    let root_canon = root.canonicalize().context("cannot canonicalize root")?;
    for p in plan.read.iter().chain(plan.edit.iter().map(|e| &e.path)) {
        let full = root.join(p);
        let canon = full
            .canonicalize()
            .with_context(|| format!("invalid path {}", p))?;
        if !canon.starts_with(&root_canon) {
            anyhow::bail!("plan contains path outside project root: {}", p);
        }
    }
    Ok(())
}

/// Apply the edit portion of a plan to the filesystem.
/// For demonstration, we prepend a comment with the intent to each edited file.
fn apply_plan_edits(root: &Path, plan: &Plan) -> Result<()> {
    for edit in &plan.edit {
        let file_path = root.join(&edit.path);
        let content = std::fs::read_to_string(&file_path)
            .with_context(|| format!("reading {}", edit.path))?;
        // Simple safety: do not overwrite binary files.
        if content.contains('\0') {
            anyhow::bail!("refusing to edit binary file {}", edit.path);
        }
        let comment = if file_path.extension().and_then(|e| e.to_str()) == Some("rs") {
            format!("// SELF‑IMPROVEMENT: {}\n", edit.intent)
        } else if file_path.extension().and_then(|e| e.to_str()) == Some("py") {
            format!("# SELF‑IMPROVEMENT: {}\n", edit.intent)
        } else {
            format!("/* SELF‑IMPROVEMENT: {} */\n", edit.intent)
        };
        let new_content = format!("{}{}", comment, content);
        std::fs::write(&file_path, new_content)
            .with_context(|| format!("writing {}", edit.path))?;
    }
    Ok(())
}

/// Run the actions defined in a plan.
fn run_plan_actions(root: &Path, plan: &Plan) -> Result<()> {
    for action in &plan.actions {
        if let Action::Run {
            program,
            args,
            workdir,
            log_hint: _,
            retries,
            backoff_ms,
        } = action
        {
            let mut attempt = 0;
            loop {
                attempt += 1;
                let mut cmd = Command::new(program);
                cmd.args(args);
                if let Some(wd) = workdir {
                    cmd.current_dir(root.join(wd));
                }
                let status = cmd.status();
                match status {
                    Ok(s) if s.success() => break,
                    _ if attempt >= *retries => {
                        anyhow::bail!(
                            "action {:?} failed after {} attempts",
                            program,
                            retries
                        );
                    }
                    _ => {
                        std::thread::sleep(Duration::from_millis(*backoff_ms));
                    }
                }
            }
        }
    }
    Ok(())
}

/// Commit changes to git with a generated message using the command‑line git tool.
fn commit_changes(root: &Path, message: &str) -> Result<()> {
    // Stage all changes.
    let status = Command::new("git")
        .arg("add")
        .arg(".")
        .current_dir(root)
        .status()
        .context("git add failed")?;
    if !status.success() {
        anyhow::bail!("git add command failed");
    }

    // Create the commit.
    let status = Command::new("git")
        .arg("commit")
        .arg("-m")
        .arg(message)
        .current_dir(root)
        .status()
        .context("git commit failed")?;
    if !status.success() {
        anyhow::bail!("git commit command failed");
    }

    Ok(())
}

/// Execute any tasks whose scheduled time has arrived.
/// This function now applies edits, runs actions, performs safety checks,
/// and records a git commit for each successful plan.
pub async fn execute_due_tasks() -> Result<()> {
    let now = Instant::now();
    let mut due: Vec<ScheduledTask> = Vec::new();

    {
        let mut queue = TASK_QUEUE.lock().unwrap();
        let mut i = 0;
        while i < queue.len() {
            if queue[i].execute_at <= now {
                due.push(queue.remove(i));
            } else {
                i += 1;
            }
        }
    }

    for task in due {
        // Determine the project root.
        let root = std::env::current_dir().context("cannot determine current dir")?;
        // Safety first.
        safety_check_plan(&root, &task.plan)?;

        // Apply edits.
        apply_plan_edits(&root, &task.plan)?;

        // Run any actions.
        run_plan_actions(&root, &task.plan)?;

        // Versioning: commit the changes.
        let commit_msg = format!(
            "Self‑improvement: applied plan with {} edit(s) and {} action(s)",
            task.plan.edit.len(),
            task.plan.actions.len()
        );
        commit_changes(&root, &commit_msg)?;
    }

    Ok(())
}

// -----------------------------------------------------------------------------
// AutonomousPlanner – core of the autonomous upgrade workflow
// -----------------------------------------------------------------------------

/// Core struct that enables the planner to evaluate its own performance,
/// fetch updates from a remote repository, and trigger a self‑update.
pub struct AutonomousPlanner {
    /// Root directory of the project the planner operates on.
    pub root: PathBuf,
    /// GitHub repository owner (e.g., "my-org").
    pub repo_owner: String,
    /// GitHub repository name (e.g., "my-agent").
    pub repo_name: String,
    /// Name of the binary produced by the repository (used by self_update).
    pub binary_name: String,
}

impl AutonomousPlanner {
    /// Construct a new `AutonomousPlanner`.
    pub fn new(
        root: impl Into<PathBuf>,
        repo_owner: impl Into<String>,
        repo_name: impl Into<String>,
        binary_name: impl Into<String>,
    ) -> Self {
        Self {
            root: root.into(),
            repo_owner: repo_owner.into(),
            repo_name: repo_name.into(),
            binary_name: binary_name.into(),
        }
    }

    /// Simple performance metric based on pending self‑improvement tasks.
    /// Returns a value between 0.0 and 1.0 where 1.0 indicates no backlog.
    pub async fn evaluate_performance(&self) -> Result<f64> {
        let queue = TASK_QUEUE.lock().unwrap();
        let pending = queue.len() as f64;
        // In a real system this would incorporate success/failure rates.
        Ok(if pending == 0.0 { 1.0 } else { 1.0 / (1.0 + pending) })
    }

    /// Fetch the latest release from the configured GitHub repository.
    /// Uses the `self_update` crate to download and replace the current binary.
    pub async fn fetch_updates(&self) -> Result<()> {
        // The current version is taken from Cargo metadata.
        let current_version = env!("CARGO_PKG_VERSION");
        Update::configure()
            .repo_owner(&self.repo_owner)
            .repo_name(&self.repo_name)
            .bin_name(&self.binary_name)
            .show_download_progress(true)
            .current_version(current_version)
            .build()?
            .update()?;
        Ok(())
    }

    /// Perform a self‑update by fetching the latest binary and (optionally) restarting.
    /// For safety this implementation only fetches; restarting is left to the caller.
    pub async fn self_update(&self) -> Result<()> {
        self.fetch_updates().await?;
        // In a full implementation you might exec the new binary here.
        Ok(())
    }

    /// Run a full autonomous cycle:
    /// 1. Evaluate performance.
    /// 2. If performance is below a threshold, generate a self‑improvement plan.
    /// 3. Schedule and execute the plan.
    /// 4. Finally, attempt a self‑update.
    pub async fn run_autonomous_cycle(&self, feedback: &str) -> Result<()> {
        let performance = self.evaluate_performance().await?;
        if performance < 0.8 {
            // Generate and schedule a self‑improvement task.
            schedule_self_improvement(&self.root, feedback, Duration::from_secs(0)).await?;
            execute_due_tasks().await?;
        }

        // Attempt to update the binary if a newer version is available.
        self.self_update().await?;
        Ok(())
    }
}

// -----------------------------------------------------------------------------
// End of file
// -----------------------------------------------------------------------------