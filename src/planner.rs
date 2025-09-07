use anyhow::{Context, Result};
use regex::Regex;
use serde::{Deserialize, Serialize};
use serde_json::{self, Value};
use std::collections::{HashMap, HashSet};
use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::{LazyLock, Mutex};
use std::time::{Duration, Instant};

use crate::fsutil::{file_inventory, FileMeta};

use self_update::backends::github::Update;
use std::env;

/// ---------------------------------------------------------------------------
/// Observability: Timeline entry and collector
/// ---------------------------------------------------------------------------

/// One‑line entry describing a planning step.
#[derive(Debug)]
struct TimelineEntry {
    /// Human readable task name.
    task: String,
    /// When the task started.
    start: Instant,
    /// When the task finished.
    end: Instant,
    /// Agent performing the step (e.g., "Planner", "SelfImprovement").
    agent: String,
    /// LLM model used, if any.
    llm: String,
    /// Approximate token usage (placeholder if unknown).
    tokens: usize,
    /// Verdict or outcome description.
    verdict: String,
}

/// Global thread‑safe collector of timeline entries.
static TIMELINE: Mutex<Vec<TimelineEntry>> = Mutex::new(Vec::new());

/// Helper to record a completed timeline entry.
fn record_timeline(entry: TimelineEntry) {
    let mut timeline = TIMELINE.lock().unwrap();
    timeline.push(entry);
}

/// ---------------------------------------------------------------------------
/// Tooling Registry
/// ---------------------------------------------------------------------------

/// Safety configuration for a tool (currently unused but reserved for future checks).
#[derive(Debug)]
struct Safety {
    allowlist: Vec<String>,
    denylist: Vec<String>,
}

/// Descriptor for a tool family.
#[derive(Debug)]
struct Tool {
    /// Human readable name.
    name: &'static str,
    /// Detect whether this tool applies to the given project root.
    detect: fn(&Path) -> bool,
    /// Build an `Action::Run` for the given sub‑command and working directory.
    run: fn(args: &[String], cwd: &Path) -> Action,
    /// Safety configuration.
    safety: Safety,
}

/// Built‑in tool implementations.

fn detect_cargo(root: &Path) -> bool {
    root.join("Cargo.toml").exists()
}
fn run_cargo(args: &[String], cwd: &Path) -> Action {
    Action::Run {
        program: "cargo".into(),
        args: args.to_vec(),
        workdir: Some(cwd.to_string_lossy().into_owned()),
        log_hint: Some(args.first().cloned().unwrap_or_default()),
        retries: DEFAULT_RETRIES,
        backoff_ms: DEFAULT_BACKOFF_MS,
    }
}

fn detect_npm(root: &Path) -> bool {
    root.join("package.json").exists()
}
fn run_npm(args: &[String], cwd: &Path) -> Action {
    let mut full_args = vec!["run".to_string()];
    full_args.extend_from_slice(args);
    Action::Run {
        program: "npm".into(),
        args: full_args,
        workdir: Some(cwd.to_string_lossy().into_owned()),
        log_hint: Some(args.first().cloned().unwrap_or_default()),
        retries: DEFAULT_RETRIES,
        backoff_ms: DEFAULT_BACKOFF_MS,
    }
}

fn detect_yarn(root: &Path) -> bool {
    root.join("yarn.lock").exists()
}
fn run_yarn(args: &[String], cwd: &Path) -> Action {
    let mut full_args = vec!["run".to_string()];
    full_args.extend_from_slice(args);
    Action::Run {
        program: "yarn".into(),
        args: full_args,
        workdir: Some(cwd.to_string_lossy().into_owned()),
        log_hint: Some(args.first().cloned().unwrap_or_default()),
        retries: DEFAULT_RETRIES,
        backoff_ms: DEFAULT_BACKOFF_MS,
    }
}

fn detect_pnpm(root: &Path) -> bool {
    root.join("pnpm-lock.yaml").exists()
}
fn run_pnpm(args: &[String], cwd: &Path) -> Action {
    let mut full_args = vec!["run".to_string()];
    full_args.extend_from_slice(args);
    Action::Run {
        program: "pnpm".into(),
        args: full_args,
        workdir: Some(cwd.to_string_lossy().into_owned()),
        log_hint: Some(args.first().cloned().unwrap_or_default()),
        retries: DEFAULT_RETRIES,
        backoff_ms: DEFAULT_BACKOFF_MS,
    }
}

fn detect_python(root: &Path) -> bool {
    root.join("pyproject.toml").exists() || root.join("requirements.txt").exists()
}
fn run_python(args: &[String], cwd: &Path) -> Action {
    if args.first().map(|s| s.as_str()) == Some("test") {
        Action::Run {
            program: "python".into(),
            args: vec!["-m".into(), "pytest".into()],
            workdir: Some(cwd.to_string_lossy().into_owned()),
            log_hint: Some("test".into()),
            retries: DEFAULT_RETRIES,
            backoff_ms: DEFAULT_BACKOFF_MS,
        }
    } else {
        Action::Run {
            program: "python".into(),
            args: args.to_vec(),
            workdir: Some(cwd.to_string_lossy().into_owned()),
            log_hint: Some(args.first().cloned().unwrap_or_default()),
            retries: DEFAULT_RETRIES,
            backoff_ms: DEFAULT_BACKOFF_MS,
        }
    }
}

fn detect_go(root: &Path) -> bool {
    root.join("go.mod").exists()
}
fn run_go(args: &[String], cwd: &Path) -> Action {
    let mut full_args = vec![];
    full_args.extend_from_slice(args);
    Action::Run {
        program: "go".into(),
        args: full_args,
        workdir: Some(cwd.to_string_lossy().into_owned()),
        log_hint: Some(args.first().cloned().unwrap_or_default()),
        retries: DEFAULT_RETRIES,
        backoff_ms: DEFAULT_BACKOFF_MS,
    }
}

fn detect_maven(root: &Path) -> bool {
    root.join("pom.xml").exists()
}
fn run_maven(args: &[String], cwd: &Path) -> Action {
    Action::Run {
        program: "mvn".into(),
        args: args.to_vec(),
        workdir: Some(cwd.to_string_lossy().into_owned()),
        log_hint: Some(args.first().cloned().unwrap_or_default()),
        retries: DEFAULT_RETRIES,
        backoff_ms: DEFAULT_BACKOFF_MS,
    }
}

/// Registry of known tools.
static TOOLS: LazyLock<Vec<Tool>> = LazyLock::new(|| {
    vec![
        Tool {
            name: "cargo",
            detect: detect_cargo,
            run: run_cargo,
            safety: Safety {
                allowlist: vec!["build".into(), "test".into(), "run".into()],
                denylist: vec![],
            },
        },
        Tool {
            name: "npm",
            detect: detect_npm,
            run: run_npm,
            safety: Safety {
                allowlist: vec!["build".into(), "test".into(), "dev".into()],
                denylist: vec![],
            },
        },
        Tool {
            name: "yarn",
            detect: detect_yarn,
            run: run_yarn,
            safety: Safety {
                allowlist: vec!["build".into(), "test".into(), "dev".into()],
                denylist: vec![],
            },
        },
        Tool {
            name: "pnpm",
            detect: detect_pnpm,
            run: run_pnpm,
            safety: Safety {
                allowlist: vec!["build".into(), "test".into(), "dev".into()],
                denylist: vec![],
            },
        },
        Tool {
            name: "python",
            detect: detect_python,
            run: run_python,
            safety: Safety {
                allowlist: vec!["test".into()],
                denylist: vec![],
            },
        },
        Tool {
            name: "go",
            detect: detect_go,
            run: run_go,
            safety: Safety {
                allowlist: vec!["test".into(), "run".into()],
                denylist: vec![],
            },
        },
        Tool {
            name: "maven",
            detect: detect_maven,
            run: run_maven,
            safety: Safety {
                allowlist: vec!["test".into()],
                denylist: vec![],
            },
        },
    ]
});

/// Detect the appropriate tool for the given project root.
fn detect_tool(root: &Path) -> Option<&'static Tool> {
    TOOLS.iter().find(|t| (t.detect)(root))
}

/// Simple runner that delegates to a tool's `run` method.
struct Runner;

impl Runner {
    fn run_tool(&self, tool: &Tool, args: &[String], cwd: &Path) -> Action {
        (tool.run)(args, cwd)
    }
}

/// ---------------------------------------------------------------------------
/// Memory module – long‑term fact storage
/// ---------------------------------------------------------------------------

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct MemoryFact {
    /// Human‑readable fact description.
    pub fact: String,
    /// Citation – typically a file path (and optionally line number).
    pub source: String,
}

/// Returns the path to the memory JSON file inside `.agent`.
fn memory_path(root: &Path) -> Result<PathBuf> {
    let dir = root.join(".agent");
    fs::create_dir_all(&dir)
        .with_context(|| format!("creating .agent dir at {}", dir.display()))?;
    Ok(dir.join("memory.json"))
}

/// Load all stored facts, returning an empty vector if the file does not exist.
pub fn load_memory(root: &Path) -> Result<Vec<MemoryFact>> {
    let path = memory_path(root)?;
    if !path.exists() {
        return Ok(Vec::new());
    }
    let data = fs::read_to_string(&path)
        .with_context(|| format!("reading memory file {}", path.display()))?;
    let facts: Vec<MemoryFact> =
        serde_json::from_str(&data).context("deserializing memory JSON")?;
    Ok(facts)
}

/// Persist the provided facts vector to disk.
fn save_memory(root: &Path, facts: &[MemoryFact]) -> Result<()> {
    let path = memory_path(root)?;
    let json = serde_json::to_string_pretty(facts).context("serializing memory JSON")?;
    let mut file =
        File::create(&path).with_context(|| format!("creating memory file {}", path.display()))?;
    file.write_all(json.as_bytes())
        .with_context(|| format!("writing memory file {}", path.display()))?;
    Ok(())
}

/// Record a new fact with its source citation. Duplicate facts are ignored.
pub fn record_fact(root: &Path, fact: &str, source: &str) -> Result<()> {
    let mut facts = load_memory(root)?;
    if !facts.iter().any(|f| f.fact == fact && f.source == source) {
        facts.push(MemoryFact {
            fact: fact.to_string(),
            source: source.to_string(),
        });
        save_memory(root, &facts)?;
    }
    Ok(())
}

/// Retrieve all stored facts.
pub fn retrieve_facts(root: &Path) -> Result<Vec<MemoryFact>> {
    load_memory(root)
}

/// ---------------------------------------------------------------------------
/// Existing planner structures (unchanged)
/// ---------------------------------------------------------------------------

/// High‑level plan the planner returns.
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

// -----------------------------------------------------------------------------
// New Goal‑oriented planning structures
// -----------------------------------------------------------------------------

/// Represents a high‑level user goal.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Goal {
    pub description: String,
}

/// The type/kind of a task in the graph.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum TaskType {
    Codegen,
    Edit,
    Run,
    Test,
    Lint,
    Review,
}

/// A single node in the task graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Task {
    pub id: usize,
    pub title: String,
    #[serde(rename = "type")]
    pub task_type: TaskType,
    #[serde(default)]
    pub inputs: Vec<String>,
    #[serde(default)]
    pub outputs: Vec<String>,
    #[serde(default)]
    pub tool_calls: Vec<String>,
    #[serde(default)]
    pub depends_on: Vec<usize>,
}

/// Directed acyclic graph of tasks.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskGraph {
    pub goal: Goal,
    pub tasks: Vec<Task>,
    #[serde(default)]
    pub notes: String,
    #[serde(default)]
    pub risks: String,
}

// -----------------------------------------------------------------------------
// Helper functions for DAG creation & serialization
// -----------------------------------------------------------------------------

/// Ensure the `.agent` directory exists under the project root.
fn ensure_agent_dir(root: &Path) -> Result<PathBuf> {
    let dir = root.join(".agent");
    fs::create_dir_all(&dir)
        .with_context(|| format!("creating .agent dir at {}", dir.display()))?;
    Ok(dir)
}

/// Serialize the task graph to JSON and a human‑readable markdown file.
fn persist_taskgraph(root: &Path, graph: &TaskGraph) -> Result<()> {
    let dir = ensure_agent_dir(root)?;
    let json_path = dir.join("plan.json");
    let md_path = dir.join("plan.md");

    // JSON
    let json = serde_json::to_string_pretty(graph).context("serializing TaskGraph to JSON")?;
    let mut json_file =
        File::create(&json_path).with_context(|| format!("creating {}", json_path.display()))?;
    json_file
        .write_all(json.as_bytes())
        .with_context(|| format!("writing {}", json_path.display()))?;

    // Markdown (human readable)
    let mut md = String::new();
    md.push_str("# Plan Summary\n\n");
    md.push_str(&format!("**Goal:** {}\n\n", graph.goal.description));
    md.push_str("## Tasks\n\n");
    for task in &graph.tasks {
        md.push_str(&format!(
            "- **{}** (`{}`) – {}{}\n",
            task.id,
            format!("{:?}", task.task_type).to_lowercase(),
            task.title,
            if !task.depends_on.is_empty() {
                format!(" (depends on: {:?})", task.depends_on)
            } else {
                "".to_string()
            }
        ));
    }
    if !graph.risks.is_empty() {
        md.push_str("\n## Risks / Assumptions\n\n");
        md.push_str(&graph.risks);
        md.push('\n');
    }
    if !graph.notes.is_empty() {
        md.push_str("\n## Additional Notes\n\n");
        md.push_str(&graph.notes);
        md.push('\n');
    }

    let mut md_file =
        File::create(&md_path).with_context(|| format!("creating {}", md_path.display()))?;
    md_file
        .write_all(md.as_bytes())
        .with_context(|| format!("writing {}", md_path.display()))?;

    Ok(())
}

/// If the graph exceeds the safety budget (>20 tasks), compress it by merging
/// consecutive tasks of the same type. Returns true if compression happened.
fn enforce_safety_budget(graph: &mut TaskGraph) -> bool {
    const MAX_TASKS: usize = 20;
    if graph.tasks.len() <= MAX_TASKS {
        return false;
    }

    // Simple compression: merge sequential tasks with identical `task_type`.
    let mut compressed: Vec<Task> = Vec::new();
    let mut i = 0;
    while i < graph.tasks.len() {
        let mut current = graph.tasks[i].clone();
        let mut j = i + 1;
        while j < graph.tasks.len()
            && graph.tasks[j].task_type == current.task_type
            && graph.tasks[j].depends_on == vec![current.id]
        {
            current.title = format!("{} + {}", current.title, graph.tasks[j].title);
            current.outputs.extend(graph.tasks[j].outputs.clone());
            current.tool_calls.extend(graph.tasks[j].tool_calls.clone());
            current.depends_on = graph.tasks[j].depends_on.clone();
            j += 1;
        }
        compressed.push(current);
        i = j;
    }

    // Re‑assign IDs and dependencies to keep a valid DAG.
    let mut id_map: HashMap<usize, usize> = HashMap::new();
    for (new_id, task) in compressed.iter_mut().enumerate() {
        let old_id = task.id;
        id_map.insert(old_id, new_id);
        task.id = new_id;
    }
    for task in compressed.iter_mut() {
        let new_deps = task
            .depends_on
            .iter()
            .filter_map(|dep| id_map.get(dep).cloned())
            .collect();
        task.depends_on = new_deps;
    }

    graph.tasks = compressed;
    true
}

/// Generate a simple linear task list (5‑12 tasks) from a goal description.
/// This is a placeholder implementation; in a real system the LLM would be used.
fn generate_task_graph(goal: Goal) -> TaskGraph {
    let mut raw_steps: Vec<String> = goal
        .description
        .split(&[',', ';'][..])
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();

    if raw_steps.len() < 5 {
        let extra = vec![
            "Review generated code".to_string(),
            "Run tests".to_string(),
            "Lint the code".to_string(),
            "Deploy to staging".to_string(),
        ];
        raw_steps.extend(extra);
    }
    raw_steps.truncate(12);

    let mut tasks = Vec::new();
    for (idx, step) in raw_steps.iter().enumerate() {
        let task_type = if step.to_lowercase().contains("test") {
            TaskType::Test
        } else if step.to_lowercase().contains("lint") {
            TaskType::Lint
        } else if step.to_lowercase().contains("run") || step.to_lowercase().contains("deploy") {
            TaskType::Run
        } else if step.to_lowercase().contains("review") {
            TaskType::Review
        } else if step.to_lowercase().contains("code") || step.to_lowercase().contains("generate") {
            TaskType::Codegen
        } else {
            TaskType::Edit
        };

        let depends_on = if idx == 0 { vec![] } else { vec![idx - 1] };

        tasks.push(Task {
            id: idx,
            title: step.clone(),
            task_type,
            inputs: vec![],
            outputs: vec![],
            tool_calls: vec![],
            depends_on,
        });
    }

    TaskGraph {
        goal,
        tasks,
        notes: String::new(),
        risks: String::new(),
    }
}

/// Public entry point: accept a user goal, produce a plan summary, a TaskGraph,
/// risk/assumption notes, enforce safety budget, and persist the artifacts.
pub fn plan_goal(root: &Path, user_goal: &str) -> Result<()> {
    let start = Instant::now();
    let goal = Goal {
        description: user_goal.trim().to_string(),
    };

    // Generate the task graph directly (PlannerAgent removed).
    let mut graph = generate_task_graph(goal);

    if graph.notes.is_empty() {
        graph.notes = "Generated automatically; review before execution.".to_string();
    }
    if graph.risks.is_empty() {
        graph.risks =
            "Assumes all tools (cargo, npm, etc.) are installed and functional.".to_string();
    }

    if enforce_safety_budget(&mut graph) {
        graph
            .notes
            .push_str("\nNote: Task graph was compressed to respect safety budget.");
    }

    persist_taskgraph(root, &graph)?;

    let end = Instant::now();
    record_timeline(TimelineEntry {
        task: "plan_goal".to_string(),
        start,
        end,
        agent: "Planner".to_string(),
        llm: "N/A".to_string(),
        tokens: 0,
        verdict: "success".to_string(),
    });

    Ok(())
}

// -----------------------------------------------------------------------------
// Existing planner implementation (updated for multi‑LLM routing)
// -----------------------------------------------------------------------------

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

// NOTE: The routing client is omitted here to keep this file self‑contained.
// In a full application, replace the stub with the real router client.

pub async fn plan_changes(root: &Path, user_request: &str) -> Result<Plan> {
    let step_start = Instant::now();

    let mut index = file_inventory(root)?;
    if index.len() > 800 {
        index = compact_index(index);
    }

    let _prompt = PlanPrompt {
        user_request,
        file_index: &index,
        guidance: &planner_guidance(),
    };

    // Stubbed LLM call – in production this would use the router client.
    let llm_value: Result<Value> = Err(anyhow::anyhow!("LLM client not available"));

    if let Ok(v) = llm_value {
        if let Ok(mut plan) = serde_json::from_value::<Plan>(v.clone()) {
            let budget_note = format!("\n[Budget] LLM time: {:.2?}", Duration::from_secs(0));
            if plan.notes.is_empty() {
                plan.notes = budget_note.trim_start().to_string();
            } else {
                plan.notes.push_str(&budget_note);
            }

            if plan.read.is_empty()
                && plan.edit.is_empty()
                && plan.notes.trim().is_empty()
                && plan.actions.is_empty()
            {
                // fall through to fallback
            } else {
                plan.read.retain(|p| root.join(p).exists());
                plan.edit.retain(|e| root.join(&e.path).exists());

                let step_end = Instant::now();
                record_timeline(TimelineEntry {
                    task: "plan_changes".to_string(),
                    start: step_start,
                    end: step_end,
                    agent: "Planner".to_string(),
                    llm: "fallback".to_string(),
                    tokens: 0,
                    verdict: "success".to_string(),
                });

                return Ok(plan);
            }
        }
    }

    // Fallback plan.
    let mut fallback = fallback_plan(root, user_request, &index);
    fallback
        .notes
        .push_str("\n[Budget] LLM call failed or returned empty.");

    let step_end = Instant::now();
    record_timeline(TimelineEntry {
        task: "plan_changes".to_string(),
        start: step_start,
        end: step_end,
        agent: "Planner".to_string(),
        llm: "fallback".to_string(),
        tokens: 0,
        verdict: "fallback".to_string(),
    });

    Ok(fallback)
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

    // -----------------------------------------------------------------
    // Build actions using the tooling registry.
    // -----------------------------------------------------------------
    let mut actions: Vec<Action> = vec![];
    let runner = Runner;
    if let Some(tool) = detect_tool(root) {
        let fact = format!("repo uses {}", tool.name);
        let source = match tool.name {
            "cargo" => "Cargo.toml",
            "npm" => "package.json",
            "yarn" => "yarn.lock",
            "pnpm" => "pnpm-lock.yaml",
            "python" => "pyproject.toml or requirements.txt",
            "go" => "go.mod",
            "maven" => "pom.xml",
            _ => "unknown",
        };
        let _ = record_fact(root, &fact, source);

        let mut push_action = |cmd: &str| {
            let args = vec![cmd.to_string()];
            let act = runner.run_tool(tool, &args, root);
            actions.push(act);
        };

        if contains_any(user_request, &["build", "compile"]) {
            push_action("build");
        }
        if contains_any(user_request, &["test", "unit test", "pytest", "cargo test"]) {
            push_action("test");
        }
        if contains_any(
            user_request,
            &["run", "start server", "start dev", "dev server"],
        ) {
            let dev_cmd = if tool.name == "npm" || tool.name == "yarn" || tool.name == "pnpm" {
                "dev"
            } else {
                "run"
            };
            push_action(dev_cmd);
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
        let size_bucket = (m.size as i64 / 4096);
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
// Self‑improvement capabilities (updated to use summary model)
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
pub async fn generate_self_improvement(root: &Path, _feedback: &str) -> Result<Plan> {
    let step_start = Instant::now();

    let mut index = file_inventory(root)?;
    if index.len() > 800 {
        index = compact_index(index);
    }

    let llm_value: Result<Value> = Err(anyhow::anyhow!("LLM client not available"));

    if let Ok(v) = llm_value {
        if let Ok(mut plan) = serde_json::from_value::<Plan>(v.clone()) {
            plan.read.retain(|p| root.join(p).exists());
            plan.edit.retain(|e| root.join(&e.path).exists());

            let budget_note = format!("\n[Budget] LLM time: {:.2?}", Duration::from_secs(0));
            if plan.notes.is_empty() {
                plan.notes = budget_note.trim_start().to_string();
            } else {
                plan.notes.push_str(&budget_note);
            }

            let step_end = Instant::now();
            record_timeline(TimelineEntry {
                task: "generate_self_improvement".to_string(),
                start: step_start,
                end: step_end,
                agent: "SelfImprovement".to_string(),
                llm: "fallback".to_string(),
                tokens: 0,
                verdict: "success".to_string(),
            });

            return Ok(plan);
        }
    }

    let step_end = Instant::now();
    record_timeline(TimelineEntry {
        task: "generate_self_improvement".to_string(),
        start: step_start,
        end: step_end,
        agent: "SelfImprovement".to_string(),
        llm: "fallback".to_string(),
        tokens: 0,
        verdict: "fallback".to_string(),
    });

    Ok(Plan {
        read: vec![],
        edit: vec![],
        actions: vec![],
        notes: "Self‑improvement feedback could not be parsed into a plan.".to_string(),
        signal: None,
        error: None,
    })
}

/// Schedule a self‑improvement plan to be executed after `delay`.
pub async fn schedule_self_improvement(root: &Path, feedback: &str, delay: Duration) -> Result<()> {
    let step_start = Instant::now();

    let plan = generate_self_improvement(root, feedback).await?;
    let task = ScheduledTask {
        execute_at: Instant::now() + delay,
        plan,
    };
    {
        let mut queue = TASK_QUEUE.lock().unwrap();
        queue.push(task);
    }

    let step_end = Instant::now();
    record_timeline(TimelineEntry {
        task: "schedule_self_improvement".to_string(),
        start: step_start,
        end: step_end,
        agent: "SelfImprovementScheduler".to_string(),
        llm: "fallback".to_string(),
        tokens: 0,
        verdict: "queued".to_string(),
    });

    Ok(())
}

/// Retrieve a snapshot of all pending tasks.
pub fn pending_tasks() -> Vec<ScheduledTask> {
    let queue = TASK_QUEUE.lock().unwrap();
    queue.clone()
}

/// Add a high‑level healing task when automatic fixes are exhausted.
/// This task is scheduled for immediate execution.
pub fn add_healing_task(plan: Plan) -> Result<()> {
    let root = std::env::current_dir().context("cannot determine current directory")?;
    safety_check_plan(&root, &plan)?;

    let task = ScheduledTask {
        execute_at: Instant::now(),
        plan,
    };
    let mut queue = TASK_QUEUE.lock().unwrap();
    queue.push(task);
    Ok(())
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
                        anyhow::bail!("action {:?} failed after {} attempts", program, retries);
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
    let status = Command::new("git")
        .arg("add")
        .arg(".")
        .current_dir(root)
        .status()
        .context("git add failed")?;
    if !status.success() {
        anyhow::bail!("git add command failed");
    }

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
    let step_start = Instant::now();

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
        let root = std::env::current_dir().context("cannot determine current dir")?;
        safety_check_plan(&root, &task.plan)?;
        apply_plan_edits(&root, &task.plan)?;
        run_plan_actions(&root, &task.plan)?;

        let commit_msg = format!(
            "Self‑improvement: applied plan with {} edit(s) and {} action(s)",
            task.plan.edit.len(),
            task.plan.actions.len()
        );
        commit_changes(&root, &commit_msg)?;
    }

    let step_end = Instant::now();
    record_timeline(TimelineEntry {
        task: "execute_due_tasks".to_string(),
        start: step_start,
        end: step_end,
        agent: "Executor".to_string(),
        llm: "N/A".to_string(),
        tokens: 0,
        verdict: "completed".to_string(),
    });

    Ok(())
}

// -----------------------------------------------------------------------------
// One‑shot planning API for small edit requests
// -----------------------------------------------------------------------------

/// Attempt a lightweight, single‑edit plan without invoking the full DAG.
/// Returns a `Plan` that contains exactly one `EditPlan` if the request matches
/// a known simple pattern (e.g., rename a function). If the request cannot be
/// handled, falls back to the regular `plan_changes` logic.
///
/// The `skip_dag` flag signals callers that they want the fast path; the function
/// respects it and never calls the DAG‑based planner when `true`.
pub async fn plan_one_shot(root: &Path, request: &str, skip_dag: bool) -> Result<Plan> {
    if skip_dag {
        let rename_re = Regex::new(r"rename\s+fn\s+(\w+)\s+to\s+(\w+)").unwrap();
        if let Some(caps) = rename_re.captures(request) {
            let old_name = caps.get(1).unwrap().as_str();
            let new_name = caps.get(2).unwrap().as_str();

            let mut target_path = PathBuf::from("src/lib.rs");
            for entry in walkdir::WalkDir::new(root)
                .into_iter()
                .filter_map(|e| e.ok())
            {
                if entry.path().extension().and_then(|e| e.to_str()) == Some("rs") {
                    if let Ok(content) = std::fs::read_to_string(entry.path()) {
                        if content.contains(&format!("fn {}", old_name)) {
                            target_path = entry.path().strip_prefix(root).unwrap().to_path_buf();
                            break;
                        }
                    }
                }
            }

            let edit = EditPlan {
                path: target_path.to_string_lossy().into_owned(),
                intent: format!("Rename function `{}` to `{}`", old_name, new_name),
            };

            return Ok(Plan {
                read: vec![],
                edit: vec![edit],
                actions: vec![],
                notes: "One‑shot edit generated without full DAG planning.".to_string(),
                signal: None,
                error: None,
            });
        }

        return plan_changes(root, request).await;
    }

    plan_changes(root, request).await
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
        Ok(if pending == 0.0 {
            1.0
        } else {
            1.0 / (1.0 + pending)
        })
    }

    /// Fetch the latest release from the configured GitHub repository.
    /// Uses the `self_update` crate to download and replace the current binary.
    pub async fn fetch_updates(&self) -> Result<()> {
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
            schedule_self_improvement(&self.root, feedback, Duration::from_secs(0)).await?;
            execute_due_tasks().await?;
        }

        self.self_update().await?;
        Ok(())
    }
}

// -----------------------------------------------------------------------------
// PlannerAgent – public entry point for external modules
// -----------------------------------------------------------------------------
/// Public wrapper around planning capabilities. Made public so it can be
/// re‑exported from other modules.
pub struct PlannerAgent {
    /// Root directory of the project.
    pub root: PathBuf,
}

impl PlannerAgent {
    /// Construct a new `PlannerAgent` using the current working directory.
    pub fn new() -> Self {
        let root = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
        Self { root }
    }

    /// Generate a task graph for the given request, persisting it and returning the graph.
    pub async fn plan(&self, request: &str) -> Result<TaskGraph> {
        // Obtain the legacy Plan first.
        let plan = plan_changes(&self.root, request).await?;

        // Convert the Plan into a simple TaskGraph.
        let mut tasks = Vec::new();
        let mut next_id = 0usize;

        // Edit tasks.
        for edit in &plan.edit {
            tasks.push(Task {
                id: next_id,
                title: edit.intent.clone(),
                task_type: TaskType::Edit,
                inputs: vec![],
                outputs: vec![edit.path.clone()],
                tool_calls: vec![],
                depends_on: vec![],
            });
            next_id += 1;
        }

        // Action (run) tasks.
        for action in &plan.actions {
            if let Action::Run {
                program,
                args,
                workdir: _,
                log_hint: _,
                retries: _,
                backoff_ms: _,
            } = action
            {
                let title = format!("Run {} {}", program, args.join(" "));
                tasks.push(Task {
                    id: next_id,
                    title,
                    task_type: TaskType::Run,
                    inputs: vec![],
                    outputs: vec![],
                    tool_calls: vec![],
                    depends_on: vec![],
                });
                next_id += 1;
            }
        }

        let graph = TaskGraph {
            goal: Goal {
                description: request.to_string(),
            },
            tasks,
            notes: plan.notes.clone(),
            risks: String::new(),
        };

        // Persist the graph for downstream agents.
        persist_taskgraph(&self.root, &graph)?;

        Ok(graph)
    }

    /// Generate a goal‑oriented task graph and persist it.
    pub fn plan_goal(&self, user_goal: &str) -> Result<()> {
        plan_goal(&self.root, user_goal)
    }

    /// Serialize a given TaskGraph to a pretty‑printed JSON string.
    pub fn serialize_taskgraph(&self, graph: &TaskGraph) -> Result<String> {
        serde_json::to_string_pretty(graph).context("serializing TaskGraph")
    }

    /// Generate a fix patch based on an error message.
    /// This stub implementation returns an empty string; replace with real logic as needed.
    pub fn generate_fix(&self, _error_msg: &str) -> Result<String> {
        Ok(String::new())
    }
}

// -----------------------------------------------------------------------------
// End of file
// -----------------------------------------------------------------------------
