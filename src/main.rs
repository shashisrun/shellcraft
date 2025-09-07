// src/main.rs

use anyhow::{anyhow, bail, Context, Result};
use dialoguer::{theme::ColorfulTheme, Confirm, Input};
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

use once_cell::sync::OnceCell;
use std::collections::{HashMap, HashSet};

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

use serde::{Deserialize, Serialize};
use serde_json;

const VERSION: &str = env!("CARGO_PKG_VERSION");

// ---------------------------------------------------------------------------
// Model registry and budgeting
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct ProviderConfig {
    name: String,
    api_key: String,
    model: String,
    cost_per_token: f64,
    latency_ms: u64,
}

#[derive(Debug, Default)]
struct ClientRegistry {
    providers: Vec<ProviderConfig>,
}

impl ClientRegistry {
    fn new(env_vars: &[(String, String)]) -> Self {
        let mut providers = Vec::new();

        // Helper to find a value by key prefix
        let find = |prefix: &str| -> Option<String> {
            env_vars
                .iter()
                .find(|(k, _)| k.starts_with(prefix))
                .map(|(_, v)| v.clone())
        };

        // OpenAI
        if let Some(key) = find("OPENAI_API_KEY") {
            let model = env::var("OPENAI_MODEL").unwrap_or_else(|_| "gpt-4o".to_string());
            providers.push(ProviderConfig {
                name: "openai".to_string(),
                api_key: key,
                model,
                cost_per_token: 0.0005,
                latency_ms: 200,
            });
        }

        // Anthropic
        if let Some(key) = find("ANTHROPIC_API_KEY") {
            let model = env::var("ANTHROPIC_MODEL")
                .unwrap_or_else(|_| "claude-3-5-sonnet-20240620".to_string());
            providers.push(ProviderConfig {
                name: "anthropic".to_string(),
                api_key: key,
                model,
                cost_per_token: 0.0008,
                latency_ms: 300,
            });
        }

        // Groq
        if let Some(key) = find("GROQ_API_KEY") {
            let model = env::var("GROQ_MODEL").unwrap_or_else(|_| "mixtral-8x7b-32768".to_string());
            providers.push(ProviderConfig {
                name: "groq".to_string(),
                api_key: key,
                model,
                cost_per_token: 0.0003,
                latency_ms: 150,
            });
        }

        // Local (e.g., Ollama)
        if let Some(key) = find("LOCAL_MODEL") {
            providers.push(ProviderConfig {
                name: "local".to_string(),
                api_key: key,
                model: env::var("LOCAL_MODEL").unwrap_or_else(|_| "llama3".to_string()),
                cost_per_token: 0.0,
                latency_ms: 100,
            });
        }

        ClientRegistry { providers }
    }

    // Very simple routing based on task category
    fn select_provider(&self, task_category: &str) -> Option<&ProviderConfig> {
        // Priorities:
        // code -> first provider that supports code (we assume all do)
        // reasoning -> prefer higher latency (more capable) models
        // summary -> prefer low latency, cheap models
        match task_category {
            "code" => self.providers.first(),
            "reasoning" => self.providers.iter().max_by_key(|p| p.latency_ms),
            "summary" => self.providers.iter().min_by_key(|p| p.latency_ms),
            _ => self.providers.first(),
        }
    }
}

// Global registry (set once at startup)
static CLIENT_REGISTRY: OnceCell<Arc<ClientRegistry>> = OnceCell::new();

#[derive(Debug, Default, Serialize, Deserialize)]
struct Memory {
    /// Recent messages / artifacts for short‑term recall.
    short_term: Vec<String>,
    /// Persistent facts for long‑term recall.
    #[serde(default)]
    long_term: serde_json::Value,
}

impl Memory {
    fn new() -> Self {
        Self::default()
    }

    fn add_short_term(&mut self, entry: impl Into<String>) {
        self.short_term.push(entry.into());
        // Keep only the last N entries (e.g., 20)
        const MAX: usize = 20;
        if self.short_term.len() > MAX {
            self.short_term.drain(0..self.short_term.len() - MAX);
        }
    }

    fn add_long_term(&mut self, key: impl Into<String>, value: impl Serialize) {
        if self.long_term.as_object_mut().is_none() {
            self.long_term = serde_json::json!({});
        }
        let obj = self.long_term.as_object_mut().unwrap();
        obj.insert(key.into(), serde_json::to_value(value).unwrap_or_default());
    }

    fn save(&self, path: &Path) -> Result<()> {
        let json = serde_json::to_string_pretty(self)?;
        fs::write(path, json)?;
        Ok(())
    }
}

// Global memory (set once at startup)
static MEMORY: OnceCell<Arc<std::sync::Mutex<Memory>>> = OnceCell::new();

fn get_memory() -> &'static Arc<std::sync::Mutex<Memory>> {
    MEMORY.get().expect("Memory not initialized")
}

#[derive(Debug, Default)]
struct Budget {
    tokens: usize,
    time_ms: u128,
}

// Global budget tracker
static BUDGET: OnceCell<std::sync::Mutex<Budget>> = OnceCell::new();

fn record_usage(tokens: usize, time_ms: u128) {
    if let Some(mutex) = BUDGET.get() {
        let mut budget = mutex.lock().unwrap();
        budget.tokens += tokens;
        budget.time_ms += time_ms;
    }
}

// ---------------------------------------------------------------------------
// Observability structures
// ---------------------------------------------------------------------------

#[derive(Debug, Serialize, Deserialize)]
struct TimelineEntry {
    start: u128,
    end: u128,
    agent: String,
    llm: String,
    tokens: usize,
    verdict: String,
}

static TIMELINE: OnceCell<Arc<std::sync::Mutex<Vec<TimelineEntry>>>> = OnceCell::new();
static PROVIDER_USAGE: OnceCell<Arc<std::sync::Mutex<HashMap<String, (usize, u128)>>>> =
    OnceCell::new();
static CHANGED_FILES: OnceCell<Arc<std::sync::Mutex<HashSet<String>>>> = OnceCell::new();

fn add_timeline_entry(
    start: SystemTime,
    end: SystemTime,
    agent: &str,
    llm: &str,
    tokens: usize,
    verdict: &str,
) {
    let start_ms = start.duration_since(UNIX_EPOCH).unwrap().as_millis();
    let end_ms = end.duration_since(UNIX_EPOCH).unwrap().as_millis();
    if let Some(tl) = TIMELINE.get() {
        let mut vec = tl.lock().unwrap();
        vec.push(TimelineEntry {
            start: start_ms,
            end: end_ms,
            agent: agent.to_string(),
            llm: llm.to_string(),
            tokens,
            verdict: verdict.to_string(),
        });
    }
}

fn record_provider_usage(provider: &str, tokens: usize, time_ms: u128) {
    if let Some(map) = PROVIDER_USAGE.get() {
        let mut m = map.lock().unwrap();
        let entry = m.entry(provider.to_string()).or_insert((0, 0));
        entry.0 += tokens;
        entry.1 += time_ms;
    }
}

fn add_changed_file(path: &Path) {
    if let Some(set) = CHANGED_FILES.get() {
        let mut s = set.lock().unwrap();
        s.insert(path.to_string_lossy().to_string());
    }
}

// ---------------------------------------------------------------------------
// Tooling Registry
// ---------------------------------------------------------------------------

type DetectFn = Arc<dyn Fn(&Path) -> bool + Send + Sync>;
type RunFn = Arc<dyn Fn(&[String], &Path) -> Result<()> + Send + Sync>;

#[derive(Clone)]
struct Tool {
    name: String,
    detect: DetectFn,
    run: RunFn,
    // safety fields are placeholders for future use
    allowlist: Vec<String>,
    denylist: Vec<String>,
}

struct ToolRegistry {
    tools: Vec<Tool>,
}

impl ToolRegistry {
    fn new(root: &Path) -> Self {
        let mut tools = Vec::new();

        // Helper to add a tool
        let mut add = |name: &str,
                       detect: DetectFn,
                       run: RunFn,
                       allowlist: Vec<String>,
                       denylist: Vec<String>| {
            tools.push(Tool {
                name: name.to_string(),
                detect,
                run,
                allowlist,
                denylist,
            });
        };

        // Cargo tools
        add(
            "cargo_build",
            Arc::new(|p| p.join("Cargo.toml").exists()),
            Arc::new(|_args, cwd| {
                let status = std::process::Command::new("cargo")
                    .arg("build")
                    .current_dir(cwd)
                    .status()
                    .with_context(|| "Failed to run cargo build")?;
                if !status.success() {
                    bail!("cargo build failed");
                }
                Ok(())
            }),
            vec![],
            vec![],
        );

        add(
            "cargo_test",
            Arc::new(|p| p.join("Cargo.toml").exists()),
            Arc::new(|_args, cwd| {
                let status = std::process::Command::new("cargo")
                    .arg("test")
                    .arg("--quiet")
                    .current_dir(cwd)
                    .status()
                    .with_context(|| "Failed to run cargo test")?;
                if !status.success() {
                    bail!("cargo test failed");
                }
                Ok(())
            }),
            vec![],
            vec![],
        );

        // npm tools
        add(
            "npm_build",
            Arc::new(|p| p.join("package.json").exists()),
            Arc::new(|_args, cwd| {
                let status = std::process::Command::new("npm")
                    .arg("run")
                    .arg("build")
                    .current_dir(cwd)
                    .status()
                    .with_context(|| "Failed to run npm build")?;
                if !status.success() {
                    bail!("npm run build failed");
                }
                Ok(())
            }),
            vec![],
            vec![],
        );

        add(
            "npm_test",
            Arc::new(|p| p.join("package.json").exists()),
            Arc::new(|_args, cwd| {
                let status = std::process::Command::new("npm")
                    .arg("test")
                    .current_dir(cwd)
                    .status()
                    .with_context(|| "Failed to run npm test")?;
                if !status.success() {
                    bail!("npm test failed");
                }
                Ok(())
            }),
            vec![],
            vec![],
        );

        add(
            "npm_start",
            Arc::new(|p| p.join("package.json").exists()),
            Arc::new(|_args, cwd| {
                let status = std::process::Command::new("npm")
                    .arg("start")
                    .current_dir(cwd)
                    .status()
                    .with_context(|| "Failed to run npm start")?;
                if !status.success() {
                    bail!("npm start failed");
                }
                Ok(())
            }),
            vec![],
            vec![],
        );

        // pnpm tools
        add(
            "pnpm_build",
            Arc::new(|p| p.join("package.json").exists()),
            Arc::new(|_args, cwd| {
                let status = std::process::Command::new("pnpm")
                    .arg("run")
                    .arg("build")
                    .current_dir(cwd)
                    .status()
                    .with_context(|| "Failed to run pnpm build")?;
                if !status.success() {
                    bail!("pnpm run build failed");
                }
                Ok(())
            }),
            vec![],
            vec![],
        );

        // yarn tools
        add(
            "yarn_build",
            Arc::new(|p| p.join("package.json").exists()),
            Arc::new(|_args, cwd| {
                let status = std::process::Command::new("yarn")
                    .arg("build")
                    .current_dir(cwd)
                    .status()
                    .with_context(|| "Failed to run yarn build")?;
                if !status.success() {
                    bail!("yarn build failed");
                }
                Ok(())
            }),
            vec![],
            vec![],
        );

        // Python test (pytest)
        add(
            "pytest",
            Arc::new(|p| p.join("pyproject.toml").exists() || p.join("setup.py").exists()),
            Arc::new(|_args, cwd| {
                let status = std::process::Command::new("pytest")
                    .current_dir(cwd)
                    .status()
                    .with_context(|| "Failed to run pytest")?;
                if !status.success() {
                    bail!("pytest failed");
                }
                Ok(())
            }),
            vec![],
            vec![],
        );

        // Go test
        add(
            "go_test",
            Arc::new(|p| p.join("go.mod").exists()),
            Arc::new(|_args, cwd| {
                let status = std::process::Command::new("go")
                    .arg("test")
                    .current_dir(cwd)
                    .status()
                    .with_context(|| "Failed to run go test")?;
                if !status.success() {
                    bail!("go test failed");
                }
                Ok(())
            }),
            vec![],
            vec![],
        );

        // Maven test
        add(
            "mvn_test",
            Arc::new(|p| p.join("pom.xml").exists()),
            Arc::new(|_args, cwd| {
                let status = std::process::Command::new("mvn")
                    .arg("test")
                    .current_dir(cwd)
                    .status()
                    .with_context(|| "Failed to run mvn test")?;
                if !status.success() {
                    bail!("mvn test failed");
                }
                Ok(())
            }),
            vec![],
            vec![],
        );

        // Formatters
        add(
            "rustfmt",
            Arc::new(|p| p.join("Cargo.toml").exists()),
            Arc::new(|_args, cwd| {
                let status = std::process::Command::new("cargo")
                    .arg("fmt")
                    .current_dir(cwd)
                    .status()
                    .with_context(|| "Failed to run rustfmt")?;
                if !status.success() {
                    bail!("cargo fmt failed");
                }
                Ok(())
            }),
            vec![],
            vec![],
        );

        add(
            "prettier",
            Arc::new(|p| p.join("package.json").exists()),
            Arc::new(|_args, cwd| {
                let status = std::process::Command::new("npx")
                    .arg("prettier")
                    .arg("--write")
                    .arg(".")
                    .current_dir(cwd)
                    .status()
                    .with_context(|| "Failed to run prettier")?;
                if !status.success() {
                    bail!("prettier failed");
                }
                Ok(())
            }),
            vec![],
            vec![],
        );

        add(
            "black",
            Arc::new(|p| p.join("pyproject.toml").exists() || p.join("requirements.txt").exists()),
            Arc::new(|_args, cwd| {
                let status = std::process::Command::new("black")
                    .arg(".")
                    .current_dir(cwd)
                    .status()
                    .with_context(|| "Failed to run black")?;
                if !status.success() {
                    bail!("black failed");
                }
                Ok(())
            }),
            vec![],
            vec![],
        );

        add(
            "gofmt",
            Arc::new(|p| p.join("go.mod").exists()),
            Arc::new(|_args, cwd| {
                let status = std::process::Command::new("gofmt")
                    .arg("-w")
                    .arg(".")
                    .current_dir(cwd)
                    .status()
                    .with_context(|| "Failed to run gofmt")?;
                if !status.success() {
                    bail!("gofmt failed");
                }
                Ok(())
            }),
            vec![],
            vec![],
        );

        // Linters
        add(
            "clippy",
            Arc::new(|p| p.join("Cargo.toml").exists()),
            Arc::new(|_args, cwd| {
                let status = std::process::Command::new("cargo")
                    .arg("clippy")
                    .current_dir(cwd)
                    .status()
                    .with_context(|| "Failed to run clippy")?;
                if !status.success() {
                    bail!("cargo clippy failed");
                }
                Ok(())
            }),
            vec![],
            vec![],
        );

        add(
            "eslint",
            Arc::new(|p| p.join("package.json").exists()),
            Arc::new(|_args, cwd| {
                let status = std::process::Command::new("npx")
                    .arg("eslint")
                    .arg(".")
                    .current_dir(cwd)
                    .status()
                    .with_context(|| "Failed to run eslint")?;
                if !status.success() {
                    bail!("eslint failed");
                }
                Ok(())
            }),
            vec![],
            vec![],
        );

        add(
            "flake8",
            Arc::new(|p| p.join("setup.cfg").exists() || p.join("pyproject.toml").exists()),
            Arc::new(|_args, cwd| {
                let status = std::process::Command::new("flake8")
                    .arg(".")
                    .current_dir(cwd)
                    .status()
                    .with_context(|| "Failed to run flake8")?;
                if !status.success() {
                    bail!("flake8 failed");
                }
                Ok(())
            }),
            vec![],
            vec![],
        );

        // Search tools (always available)
        add(
            "grep",
            Arc::new(|_| true),
            Arc::new(|args, cwd| {
                if args.is_empty() {
                    bail!("grep requires a pattern argument");
                }
                let status = std::process::Command::new("grep")
                    .args(args)
                    .current_dir(cwd)
                    .status()
                    .with_context(|| "Failed to run grep")?;
                if !status.success() {
                    bail!("grep failed");
                }
                Ok(())
            }),
            vec![],
            vec![],
        );

        add(
            "ripgrep",
            Arc::new(|_| true),
            Arc::new(|args, cwd| {
                if args.is_empty() {
                    bail!("rg requires a pattern argument");
                }
                let status = std::process::Command::new("rg")
                    .args(args)
                    .current_dir(cwd)
                    .status()
                    .with_context(|| "Failed to run rg")?;
                if !status.success() {
                    bail!("rg failed");
                }
                Ok(())
            }),
            vec![],
            vec![],
        );

        // Git diff helper
        add(
            "git_diff",
            Arc::new(|_| true),
            Arc::new(|_args, cwd| {
                let status = std::process::Command::new("git")
                    .arg("diff")
                    .current_dir(cwd)
                    .status()
                    .with_context(|| "Failed to run git diff")?;
                if !status.success() {
                    bail!("git diff failed");
                }
                Ok(())
            }),
            vec![],
            vec![],
        );

        ToolRegistry { tools }
    }

    fn list(&self) -> Vec<String> {
        self.tools.iter().map(|t| t.name.clone()).collect()
    }

    fn get(&self, name: &str) -> Option<&Tool> {
        self.tools.iter().find(|t| t.name == name)
    }
}

// Global tooling registry
static TOOL_REGISTRY: OnceCell<Arc<ToolRegistry>> = OnceCell::new();

// ---------------------------------------------------------------------------
// Safety guardrails
// ---------------------------------------------------------------------------

static UNSAFE_MODE: OnceCell<bool> = OnceCell::new();
static LOGS_DIR: OnceCell<PathBuf> = OnceCell::new();

#[derive(Default)]
struct Config {
    ask_before_destructive: bool,
}
static CONFIG: OnceCell<Arc<std::sync::Mutex<Config>>> = OnceCell::new();

fn guard_command(spec: &CommandSpec) -> Result<()> {
    let unsafe_mode = *UNSAFE_MODE.get().unwrap_or(&false);
    if unsafe_mode {
        return Ok(());
    }

    // Denylist of obviously dangerous commands
    let denylist = [
        "rm",
        "sudo",
        "shutdown",
        "reboot",
        "poweroff",
        "halt",
        "ifconfig",
        "ip",
        "iptables",
        "systemctl",
    ];
    if denylist.iter().any(|d| spec.cmd == *d) {
        bail!(
            "Command '{}' is denied by safety guardrails. Use --unsafe to override.",
            spec.cmd
        );
    }

    // Simple destructive pattern for rm -rf etc.
    if spec.cmd == "rm"
        && spec
            .args
            .iter()
            .any(|a| a.contains("-rf") || a == "-r" || a == "-f")
    {
        // Power toggle: ask_before_destructive
        if let Some(cfg) = CONFIG.get() {
            let cfg = cfg.lock().unwrap();
            if cfg.ask_before_destructive {
                let confirm = Confirm::new()
                    .with_prompt("Destructive rm command detected. Proceed?")
                    .default(false)
                    .interact()?;
                if !confirm {
                    bail!("Destructive rm command aborted by user.");
                }
            }
        }
        // If not aborted, allow (or unsafe mode will have already blocked)
    }

    // Allowlist of common safe commands
    let allowlist = [
        "cargo", "npm", "pnpm", "yarn", "go", "mvn", "pytest", "black", "gofmt", "clang", "gcc",
        "make", "git", "rg", "grep", "npx", "python", "node", "bash", "sh", "zsh", "fish", "ls",
        "cat", "echo", "pwd", "whoami",
    ];
    if !allowlist.iter().any(|a| spec.cmd == *a) {
        bail!(
            "Command '{}' is not in the allowlist. Use --unsafe to override.",
            spec.cmd
        );
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Existing agent definitions (unchanged)
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Agent infrastructure
// ---------------------------------------------------------------------------

type TaskResult = Result<String>;

trait Agent {
    fn handle(&self, task: &str, context: &str) -> TaskResult;
}

// Re-export PlannerAgent for crate::planner_agent and crate::agents::planner_agent
pub mod planner_agent {
    use super::PlannerAgent;
}
pub mod agents {
    pub mod planner_agent {
        use super::super::PlannerAgent;
    }
}

#[derive(Default)]
struct PlannerAgent;
impl PlannerAgent {
    fn new() -> Self {
        PlannerAgent
    }
}
impl Agent for PlannerAgent {
    fn handle(&self, task: &str, _context: &str) -> TaskResult {
        Ok(format!("Planner handled task: {}", task))
    }
}

#[derive(Default)]
struct CodegenAgent;
impl CodegenAgent {
    fn new() -> Self {
        CodegenAgent
    }
}
impl Agent for CodegenAgent {
    fn handle(&self, task: &str, _context: &str) -> TaskResult {
        Ok(format!("Codegen handled task: {}", task))
    }
}

#[derive(Default)]
struct ExecutorAgent;
impl ExecutorAgent {
    fn new() -> Self {
        ExecutorAgent
    }
}
impl Agent for ExecutorAgent {
    fn handle(&self, task: &str, _context: &str) -> TaskResult {
        Ok(format!("Executor handled task: {}", task))
    }
}

#[derive(Default)]
struct TesterAgent;
impl TesterAgent {
    fn new() -> Self {
        TesterAgent
    }
}
impl Agent for TesterAgent {
    fn handle(&self, task: &str, _context: &str) -> TaskResult {
        Ok(format!("Tester handled task: {}", task))
    }
}

#[derive(Default)]
struct ReviewerAgent;
impl ReviewerAgent {
    fn new() -> Self {
        ReviewerAgent
    }
}
impl Agent for ReviewerAgent {
    fn handle(&self, task: &str, _context: &str) -> TaskResult {
        Ok(format!("Reviewer handled task: {}", task))
    }
}

#[derive(Default)]
struct FixerAgent;
impl FixerAgent {
    fn new() -> Self {
        FixerAgent
    }
}
impl Agent for FixerAgent {
    fn handle(&self, task: &str, _context: &str) -> TaskResult {
        Ok(format!("Fixer handled task: {}", task))
    }
}

#[derive(Serialize, Deserialize)]
struct AgentState {
    timestamp: String,
    summary: String,
}

// Helper to persist state JSON
fn persist_state(root: &Path, summary: &str) -> Result<()> {
    let state_dir = root.join(".agent");
    fs::create_dir_all(&state_dir)
        .with_context(|| format!("Failed to create .agent dir at {}", state_dir.display()))?;
    let state_path = state_dir.join("state.json");
    let state = AgentState {
        timestamp: chrono::Utc::now().to_rfc3339(),
        summary: summary.to_string(),
    };
    let json = serde_json::to_string_pretty(&state)?;
    fs::write(state_path, json)?;
    Ok(())
}

// Helper to append human‑readable notes to agent.log
fn append_log(root: &Path, note: &str) -> Result<()> {
    let log_path = root.join("agent.log");
    let mut log = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&log_path)?;
    let ts = chrono::Utc::now().to_rfc3339();
    writeln!(log, "[{}] {}", ts, note)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Intent detection
// ---------------------------------------------------------------------------

enum Intent {
    Goal,
    SimpleChange,
    Info,
}

fn detect_intent(msg: &str) -> Intent {
    let lower = msg.to_lowercase();
    if lower.starts_with("goal:") || lower.contains("i want to") || lower.contains("my goal is") {
        Intent::Goal
    } else if lower.contains("rename")
        || lower.contains("change")
        || lower.contains("add")
        || lower.contains("remove")
        || lower.contains("delete")
        || lower.contains("modify")
        || lower.contains("refactor")
    {
        Intent::SimpleChange
    } else {
        Intent::Info
    }
}

// ---------------------------------------------------------------------------
// Chat handling helpers
// ---------------------------------------------------------------------------

async fn handle_goal_message(
    msg: &str,
    root: &Path,
    env: &mut Vec<(String, String)>,
    export_patch: bool,
    patch_dir: &Path,
    dry_run: bool,
) -> Result<()> {
    // For simplicity we reuse the high‑level orchestrator which does planning,
    // edit proposals, actions, and verification.
    high_level_orchestrate(root, msg, export_patch, patch_dir, env, dry_run).await
}

async fn handle_simple_change_message(
    msg: &str,
    root: &Path,
    env: &mut Vec<(String, String)>,
    export_patch: bool,
    patch_dir: &Path,
    dry_run: bool,
) -> Result<()> {
    // Same path as goal – the orchestrator will do a one‑shot plan.
    high_level_orchestrate(root, msg, export_patch, patch_dir, env, dry_run).await
}

async fn handle_info_message(msg: &str, root: &Path) -> Result<()> {
    // Summarize context and suggest possible actions.
    let mut picked = Vec::new();
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
        msg,
        blobs.join("\n")
    );

    // Use the registry to select a summary‑optimized provider
    let provider_name = if let Some(reg) = CLIENT_REGISTRY.get() {
        reg.select_provider("summary")
            .map(|p| p.name.clone())
            .unwrap_or_else(|| "default".to_string())
    } else {
        "default".to_string()
    };
    ui::info(&format!(
        "Using provider '{}' for summarization",
        provider_name
    ));

    let sum_start = SystemTime::now();
    match llm::chat_text(system, &user).await {
        Ok(summary) => {
            let sum_end = SystemTime::now();
            ui::print("\nProject summary:");
            ui::print(&summary.trim());
            // Record dummy usage (in real code we'd parse token count)
            record_usage(100, 200);
            record_provider_usage(
                &provider_name,
                100,
                sum_end.duration_since(sum_start).unwrap().as_millis(),
            );
            add_timeline_entry(
                sum_start,
                sum_end,
                "Summarizer",
                &provider_name,
                100,
                "success",
            );
        }
        Err(e) => {
            ui::error(&format!("Summarizer error: {e}"));
            log_diagnostic(root, &format!("Summarizer error: {e}")).ok();
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Main entry point
// ---------------------------------------------------------------------------

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

    // Initialize global budget tracker
    BUDGET.set(std::sync::Mutex::new(Budget::default())).ok();

    // Initialize observability globals
    TIMELINE
        .set(Arc::new(std::sync::Mutex::new(Vec::new())))
        .ok();
    PROVIDER_USAGE
        .set(Arc::new(std::sync::Mutex::new(HashMap::new())))
        .ok();
    CHANGED_FILES
        .set(Arc::new(std::sync::Mutex::new(HashSet::new())))
        .ok();

    // optional flags
    let mut export_patch = false;
    let mut patch_dir = PathBuf::from("diffs");
    let mut autonomous = false;
    let mut unsafe_mode = false;
    let mut dry_run = false;
    let mut report_flag = false;
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
        } else if arg == "--unsafe" {
            unsafe_mode = true;
        } else if arg == "--dry-run" {
            dry_run = true;
        } else if arg == "--report" {
            report_flag = true;
        } else if arg == "--help" || arg == "-h" {
            ui::print("Usage: <program> [FLAGS] [SUBCOMMAND] [ARGS]");
            ui::print("Flags:");
            ui::print("  --export-patch          Export generated patches to a directory");
            ui::print("  --patch-dir=<DIR>       Directory for exported patches (default: diffs)");
            ui::print("  --autonomous            Run in autonomous mode (periodic cycles)");
            ui::print("  --unsafe                Disable safety guardrails");
            ui::print("  --dry-run               Simulate actions without making changes");
            ui::print("  --report                Generate a report and exit");
            ui::print("  --help, -h              Show this help message");
            ui::print("Subcommands:");
            ui::print("  run <code|file>         Execute a code snippet or file");
            ui::print("  goal <description>      Define a high‑level goal");
            ui::print("  tool <list|run> ...     List or run a tool");
            ui::print("  report                  Generate a markdown report");
            return Ok(());
        } else if subcommand.is_none() {
            subcommand = Some(arg);
            sub_args.extend(args.map(|a| a));
            break;
        }
    }

    // Store unsafe mode globally for guardrails
    UNSAFE_MODE.set(unsafe_mode).ok();

    // Initialize default config
    CONFIG
        .set(Arc::new(std::sync::Mutex::new(Config {
            ask_before_destructive: true,
        })))
        .ok();

    // handle subcommands before entering autonomous mode
    if let Some(cmd) = subcommand.clone() {
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
            "goal" => {
                // Handled later after project root is known.
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

    // -----------------------------------------------------------------------
    // Initialize ClientRegistry and make it globally available
    // -----------------------------------------------------------------------
    let registry = ClientRegistry::new(&agent_env);
    CLIENT_REGISTRY.set(Arc::new(registry)).ok();

    // -----------------------------------------------------------------------
    // Initialize ToolRegistry (project‑aware)
    // -----------------------------------------------------------------------
    let tool_registry = ToolRegistry::new(&root);
    TOOL_REGISTRY.set(Arc::new(tool_registry)).ok();

    // -----------------------------------------------------------------------
    // Initialize Memory (short‑term + long‑term) and make it globally available
    // -----------------------------------------------------------------------
    let agent_dir = root.join(".agent");
    fs::create_dir_all(&agent_dir).with_context(|| {
        format!(
            "Failed to create .agent directory at {}",
            agent_dir.display()
        )
    })?;

    let memory_path = agent_dir.join("memory.json");
    let memory = if memory_path.exists() {
        let data = fs::read_to_string(&memory_path)
            .with_context(|| format!("Failed to read memory file {}", memory_path.display()))?;
        serde_json::from_str(&data).unwrap_or_else(|_| Memory::new())
    } else {
        Memory::new()
    };
    MEMORY.set(Arc::new(std::sync::Mutex::new(memory))).ok();

    // Ensure .agent/logs directory exists for streaming logs
    let logs_dir = agent_dir.join("logs");
    fs::create_dir_all(&logs_dir)
        .with_context(|| format!("Failed to create logs directory at {}", logs_dir.display()))?;
    LOGS_DIR.set(logs_dir).ok();

    // -----------------------------------------------------------------------
    // Initialize agents
    // -----------------------------------------------------------------------
    let planner_agent = PlannerAgent::new();
    let codegen_agent = CodegenAgent::new();
    let executor_agent = ExecutorAgent::new();
    let tester_agent = TesterAgent::new();
    let reviewer_agent = ReviewerAgent::new();
    let fixer_agent = FixerAgent::new();

    // If --report flag was used, generate report now and exit
    if report_flag {
        generate_report(&root, "N/A", "N/A", "N/A")?;
        return Ok(());
    }

    // -------------------------------------------------
    // Handle "goal" subcommand: plan and persist output
    // -------------------------------------------------
    if let Some(cmd) = subcommand.as_deref() {
        if cmd == "goal" {
            if sub_args.is_empty() {
                bail!("Usage: <program> goal <goal description>");
            }
            let goal_desc = sub_args.join(" ");

            // Call the planner (ignore return value if it returns ())
            planner::plan_goal(&root, &goal_desc)?;

            // Placeholder values – in a full implementation these would be returned by the planner.
            let summary = String::new();
            let risks = String::new();
            let graph = serde_json::json!({});

            ui::print("\nPlan Summary:");
            ui::print(&summary);
            ui::print("\nRisks / Assumptions:");
            ui::print(&risks);

            // Persist the plan
            let json_path = agent_dir.join("plan.json");
            let json_str = serde_json::to_string_pretty(&graph)
                .with_context(|| "Failed to serialize TaskGraph to JSON")?;
            fs::write(&json_path, json_str)
                .with_context(|| format!("Failed to write plan JSON to {}", json_path.display()))?;

            let md_path = agent_dir.join("plan.md");
            let md_content = format!(
                "# Plan Summary\n\n{}\n\n## Risks / Assumptions\n\n{}\n",
                summary, risks
            );
            fs::write(&md_path, md_content).with_context(|| {
                format!("Failed to write plan markdown to {}", md_path.display())
            })?;

            // Save the goal description for reporting
            let goal_path = agent_dir.join("goal.txt");
            fs::write(&goal_path, &goal_desc)
                .with_context(|| format!("Failed to write goal file {}", goal_path.display()))?;

            ui::success(&format!(
                "Plan saved to {}\nJSON: {}\nMarkdown: {}",
                agent_dir.display(),
                json_path.display(),
                md_path.display()
            ));

            // Automatic report generation after goal completion
            generate_report(&root, &goal_desc, &summary, &risks)?;

            // Final budget report before exit
            if let Some(mutex) = BUDGET.get() {
                let budget = mutex.lock().unwrap();
                ui::info(&format!(
                    "Budget report – tokens used: {}, time (ms): {}",
                    budget.tokens, budget.time_ms
                ));
            }
            return Ok(());
        }
    }

    // -------------------------------------------------
    // Handle "tool" subcommand: list / run tools
    // -------------------------------------------------
    if let Some(cmd) = subcommand.as_deref() {
        if cmd == "tool" {
            if sub_args.is_empty() {
                bail!("Usage: tool <list|run> [args...]");
            }
            let tool_cmd = &sub_args[0];
            match tool_cmd.as_str() {
                "list" => {
                    let reg = TOOL_REGISTRY
                        .get()
                        .ok_or_else(|| anyhow!("Tool registry not initialized"))?;
                    for name in reg.list() {
                        ui::print(&name);
                    }
                }
                "run" => {
                    if sub_args.len() < 2 {
                        bail!("Usage: tool run <tool_name> [args...]");
                    }
                    let name = &sub_args[1];
                    let args = &sub_args[2..];
                    let reg = TOOL_REGISTRY
                        .get()
                        .ok_or_else(|| anyhow!("Tool registry not initialized"))?;
                    let tool = reg
                        .get(name)
                        .ok_or_else(|| anyhow!("Tool '{}' not found", name))?;
                    (tool.run)(args, &root)?;
                }
                _ => bail!("Unknown tool subcommand: {}", tool_cmd),
            }
            return Ok(());
        }
    }

    // -------------------------------------------------
    // Handle "report" subcommand: generate markdown report
    // -------------------------------------------------
    if let Some(cmd) = subcommand.as_deref() {
        if cmd == "report" {
            // Load stored goal and plan
            let goal_path = root.join(".agent/goal.txt");
            let goal = if goal_path.exists() {
                fs::read_to_string(&goal_path)?
            } else {
                "N/A".to_string()
            };
            let plan_md_path = root.join(".agent/plan.md");
            let plan_md = if plan_md_path.exists() {
                fs::read_to_string(&plan_md_path)?
            } else {
                String::new()
            };
            // For simplicity we reuse the plan markdown as both summary and risks sections.
            generate_report(&root, &goal, &plan_md, &plan_md)?;
            return Ok(());
        }
    }

    // -------------------------------------------------
    // Interactive chat loop (chat‑first UX)
    // -------------------------------------------------
    if !autonomous {
        ui::info("\nEnter chat messages. Type '/help' for commands, '/exit' to quit.");
        loop {
            let user_input: String = Input::with_theme(&ColorfulTheme::default())
                .with_prompt("User")
                .interact_text()?;
            let trimmed = user_input.trim();
            if trimmed.is_empty() {
                continue;
            }
            if trimmed == "/exit" || trimmed == "exit" {
                ui::info("Exiting chat.");
                break;
            }
            if trimmed == "/help" {
                ui::print("Available commands:");
                ui::print("/help               – Show this help");
                ui::print("/exit               – Exit the chat");
                ui::print(
                    "/set <key> <value>  – Set a power toggle (e.g., ask_before_destructive false)",
                );
                ui::print("/report             – Generate a markdown report");
                ui::print("Any other text is treated as a message and will be processed.");
                continue;
            }
            if trimmed.starts_with("/set ") {
                let parts: Vec<&str> = trimmed.split_whitespace().collect();
                if parts.len() != 3 {
                    ui::error("Usage: /set <key> <value>");
                    continue;
                }
                let key = parts[1];
                let value = parts[2];
                if let Some(cfg) = CONFIG.get() {
                    let mut cfg = cfg.lock().unwrap();
                    match key {
                        "ask_before_destructive" => {
                            cfg.ask_before_destructive = value.parse::<bool>().unwrap_or(true);
                            ui::info(&format!(
                                "Set ask_before_destructive = {}",
                                cfg.ask_before_destructive
                            ));
                        }
                        _ => ui::error("Unknown config key"),
                    }
                }
                continue;
            }
            if trimmed == "/report" {
                // Generate report on demand
                let goal_path = root.join(".agent/goal.txt");
                let goal = if goal_path.exists() {
                    fs::read_to_string(&goal_path)?
                } else {
                    "N/A".to_string()
                };
                let plan_md_path = root.join(".agent/plan.md");
                let plan_md = if plan_md_path.exists() {
                    fs::read_to_string(&plan_md_path)?
                } else {
                    String::new()
                };
                generate_report(&root, &goal, &plan_md, &plan_md)?;
                continue;
            }

            // Detect intent and route
            match detect_intent(trimmed) {
                Intent::Goal => {
                    if let Err(e) = handle_goal_message(
                        trimmed,
                        &root,
                        &mut agent_env,
                        export_patch,
                        &patch_dir,
                        dry_run,
                    )
                    .await
                    {
                        ui::error(&format!("Error handling goal: {e}"));
                    }
                }
                Intent::SimpleChange => {
                    if let Err(e) = handle_simple_change_message(
                        trimmed,
                        &root,
                        &mut agent_env,
                        export_patch,
                        &patch_dir,
                        dry_run,
                    )
                    .await
                    {
                        ui::error(&format!("Error handling change request: {e}"));
                    }
                }
                Intent::Info => {
                    if let Err(e) = handle_info_message(trimmed, &root).await {
                        ui::error(&format!("Error handling info request: {e}"));
                    }
                }
            }
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
            dry_run,
        )
        .await?;

        // Ensure self‑assessment task finishes
        if let Err(e) = self_assess_handle.await {
            ui::error(&format!("Self‑assessment task failed: {e:?}"));
        }
    } else {
        ui::info("\nAutonomous mode not enabled. Exiting.");
    }

    // Final shutdown hook – print accumulated budget
    if let Some(mutex) = BUDGET.get() {
        let budget = mutex.lock().unwrap();
        ui::info(&format!(
            "Final budget report – total tokens: {}, total time (ms): {}",
            budget.tokens, budget.time_ms
        ));
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
    dry_run: bool,
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

        if let Err(e) =
            high_level_orchestrate(&root, &request, export_patch, &patch_dir, env, dry_run).await
        {
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
    dry_run: bool,
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
        match orchestrate(root, request, export_patch, patch_dir, env, dry_run).await {
            Ok(_) => {
                ui::success("✅ Orchestration succeeded");
                // 4) Verify – run build/tests if applicable
                match verify_project(root, dry_run).await {
                    Ok(_) => {
                        ui::success("✅ Verification passed");
                        // 5) Summary
                        ui::info("\n📋 Summary of changes:");
                        ui::print(&format!("Request: {}", request));
                        ui::print(&format!("Project root: {}", root.display()));
                        ui::print(&format!("Export patches: {}", export_patch));
                        // Persist final state summary
                        persist_state(root, "Autonomous cycle completed successfully")?;
                        append_log(root, "Autonomous cycle completed successfully")?;
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
    Err(last_err.unwrap_or_else(|| anyhow!("Orchestration failed after retries")))
}

// ---------------------------------------------------------------------------
// Verification step – run cargo test if a Rust project is detected
// ---------------------------------------------------------------------------
async fn verify_project(root: &Path, dry_run: bool) -> Result<()> {
    if dry_run {
        ui::info("Dry-run: skipping verification steps.");
        return Ok(());
    }
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
    dry_run: bool,
) -> Result<()> {
    let mut patch_counter: usize = 1;

    // 1) Planning
    let plan_start = SystemTime::now();
    let pb = ui::spinner("Planning…");
    let plan = match planner::plan_changes(root, request).await {
        Ok(p) => p,
        Err(e) => {
            pb.finish_and_clear();
            bail!("Planner failed: {e}");
        }
    };
    pb.finish_and_clear();
    let plan_end = SystemTime::now();
    add_timeline_entry(plan_start, plan_end, "Planner", "N/A", 0, "success");
    // Persist state after planning
    persist_state(root, "Planning completed")?;
    append_log(root, "Planning completed")?;

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

        // Use the registry to select a summary‑optimized provider
        let provider_name = if let Some(reg) = CLIENT_REGISTRY.get() {
            reg.select_provider("summary")
                .map(|p| p.name.clone())
                .unwrap_or_else(|| "default".to_string())
        } else {
            "default".to_string()
        };
        ui::info(&format!(
            "Using provider '{}' for summarization",
            provider_name
        ));

        let sum_start = SystemTime::now();
        match llm::chat_text(system, &user).await {
            Ok(summary) => {
                let sum_end = SystemTime::now();
                ui::print("\nProject summary:");
                ui::print(&summary.trim());
                // Record dummy usage (in real code we'd parse token count)
                record_usage(100, 200);
                record_provider_usage(
                    &provider_name,
                    100,
                    sum_end.duration_since(sum_start).unwrap().as_millis(),
                );
                add_timeline_entry(
                    sum_start,
                    sum_end,
                    "Summarizer",
                    &provider_name,
                    100,
                    "success",
                );
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
        // Persist state after summarization
        persist_state(root, "Project summarization completed")?;
        append_log(root, "Project summarization completed")?;
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
            // Select a code‑capable provider
            let provider_name = if let Some(reg) = CLIENT_REGISTRY.get() {
                reg.select_provider("code")
                    .map(|p| p.name.clone())
                    .unwrap_or_else(|| "default".to_string())
            } else {
                "default".to_string()
            };
            ui::info(&format!(
                "Using provider '{}' for edit proposal",
                provider_name
            ));

            let edit_start = SystemTime::now();
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
            let edit_end = SystemTime::now();
            pb2.finish_and_clear();
            // Dummy usage accounting
            record_usage(200, 500);
            record_provider_usage(
                &provider_name,
                200,
                edit_end.duration_since(edit_start).unwrap().as_millis(),
            );
            add_timeline_entry(
                edit_start,
                edit_end,
                "Codegen",
                &provider_name,
                200,
                "success",
            );
            proposals.push((abs, current, proposed, is_new));
        }

        // Show and apply all (autopilot)
        ui::print("\n=== Applying proposals ===");
        for (abs, cur, prop, _is_new) in &proposals {
            let rel = path_rel(root, abs);
            ui::print(&format!("\n## {}", rel));
            ui::print(&diff::unified_colored(cur, prop, &rel));

            if dry_run {
                ui::info(&format!("Dry-run: would apply changes to {}", rel));
                continue;
            }

            if export_patch {
                let patch_name = format!("{:03}.patch", patch_counter);
                patch_counter += 1;
                let patch_path = patch_dir.join(patch_name);
                let patch_text = diff::unified_colored(cur, prop, &rel);
                fs::write(&patch_path, patch_text)
                    .with_context(|| format!("Failed to write patch {}", patch_path.display()))?;
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
                    .open(root.join("agent.log"))?;
                writeln!(log, "{} {}", now, abs.display())?;
                ui::success(&format!("Saved {}", path_rel(root, abs)));
                add_changed_file(abs);
            }
        }

        // Persist state after applying proposals
        persist_state(root, "Proposals applied")?;
        append_log(root, "Proposals applied")?;
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
                if dry_run {
                    ui::info(&format!(
                        "Dry-run: would run command `{}` with args {:?}",
                        program, args
                    ));
                    continue;
                }
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
                let exec_start = SystemTime::now();
                match run_and_capture(spec.clone()).await {
                    Ok(res) => {
                        let exec_end = SystemTime::now();
                        ui::success(&format!(
                            "Command `{}` exited {} after {} ms",
                            res.command_line, res.exit_code, res.duration_ms
                        ));
                        // Record dummy usage for command execution
                        record_usage(50, res.duration_ms);
                        add_timeline_entry(exec_start, exec_end, "Executor", "N/A", 0, "success");
                    }
                    Err(e) => {
                        ui::error(&format!("Command error: {e}"));
                        log_diagnostic(root, &format!("Command error: {e}")).ok();
                        let exec_end = SystemTime::now();
                        add_timeline_entry(exec_start, exec_end, "Executor", "N/A", 0, "failure");
                    }
                }
            }
        }
    }

    // Persist state after actions
    persist_state(root, "Actions executed")?;
    append_log(root, "Actions executed")?;

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
    // Guardrails
    guard_command(&spec)?;

    // Prepare log file with header
    let mut log_file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&spec.log_path)?;
    // Also open a tee file in the central logs directory, if available
    let mut tee_file_opt = None;
    if let Some(dir) = LOGS_DIR.get() {
        let tee_path = dir.join(format!("{}.log", spec.cmd));
        let tee = OpenOptions::new()
            .create(true)
            .append(true)
            .open(tee_path)?;
        tee_file_opt = Some(tee);
    }

    let timestamp = chrono::Utc::now().to_rfc3339();
    let cmd_line = format!("{} {}", spec.cmd, spec.args.join(" "));
    writeln!(
        log_file,
        "==== CMD @ {} (cmd: {}) ====",
        timestamp, cmd_line
    )?;
    if let Some(t) = tee_file_opt.as_mut() {
        writeln!(t, "==== CMD @ {} (cmd: {}) ====", timestamp, cmd_line)?;
    }

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
                        if let Some(t) = tee_file_opt.as_mut() {
                            writeln!(t, "{}", l)?;
                        }
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
                        if let Some(t) = tee_file_opt.as_mut() {
                            writeln!(t, "{}", l)?;
                        }
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

        // Wait for the next interval, respecting shutdown
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
    upsert_agent_env(
        root,
        &[("AGENT_VERSION".to_string(), new_version.to_string())],
    )?;
    ui::success(&format!("Self‑update simulated to version {}", new_version));
    Ok(())
}

// ---------------------------------------------------------------------------
// Report generation
// ---------------------------------------------------------------------------
fn generate_report(root: &Path, goal: &str, summary: &str, risks: &str) -> Result<()> {
    let mut md = String::new();
    md.push_str("# Automated Report\n\n");
    md.push_str("## Goal\n");
    md.push_str(&format!("{}\n\n", goal));
    md.push_str("## Plan Summary\n");
    md.push_str(&format!("{}\n\n", summary));
    md.push_str("## Risks / Manual Follow‑ups\n");
    md.push_str(&format!("{}\n\n", risks));

    // Timeline
    md.push_str("## Timeline\n\n");
    md.push_str("| Start | End | Agent | LLM Provider | Tokens | Verdict |\n");
    md.push_str("|---|---|---|---|---|---|\n");
    if let Some(tl) = TIMELINE.get() {
        let vec = tl.lock().unwrap();
        for e in vec.iter() {
            md.push_str(&format!(
                "| {} | {} | {} | {} | {} | {} |\n",
                e.start, e.end, e.agent, e.llm, e.tokens, e.verdict
            ));
        }
    }

    // Files changed
    md.push_str("\n## Files Changed\n\n");
    if let Some(set) = CHANGED_FILES.get() {
        let s = set.lock().unwrap();
        for f in s.iter() {
            md.push_str(&format!("- {}\n", f));
        }
    }

    // Provider usage
    md.push_str("\n## Runtime & Token Totals per Provider\n\n");
    md.push_str("| Provider | Tokens | Time (ms) |\n");
    md.push_str("|---|---|---|\n");
    if let Some(map) = PROVIDER_USAGE.get() {
        let m = map.lock().unwrap();
        for (prov, (tokens, time)) in m.iter() {
            md.push_str(&format!("| {} | {} | {} |\n", prov, tokens, time));
        }
    }

    // Open risks (re‑use risks section)
    md.push_str("\n## Open Risks / Manual Follow‑ups\n\n");
    md.push_str(risks);
    md.push('\n');

    let report_path = root.join("report.md");
    fs::write(&report_path, md)
        .with_context(|| format!("Failed to write report to {}", report_path.display()))?;
    ui::info(&format!("Report written to {}", report_path.display()));
    Ok(())
}
