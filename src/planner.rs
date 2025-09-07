use anyhow::{Context, Result};
use regex::Regex;
use serde::{Deserialize, Serialize};
use serde_json::{self, Value};
use std::collections::HashSet;
use std::path::Path;

use crate::fsutil::{file_inventory, FileMeta};
use crate::llm;

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
    },
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
    { "kind": "run", "program": string, "args": string[], "workdir": string | null, "log_hint": string | null }
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
            if plan.read.is_empty() && plan.edit.is_empty() && plan.notes.trim().is_empty() && plan.actions.is_empty() {
                // fall through
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
            if contains_any(user_request, &["edit", "modify", "change", "refactor", "add"]) {
                edit_set.insert(norm);
            } else {
                read_set.insert(norm);
            }
        }
    }

    if read_set.is_empty() && edit_set.is_empty() {
        for probe in &["src/main.rs", "Cargo.toml", "package.json", "pyproject.toml", "README.md", "readme.md"] {
            if root.join(probe).exists() {
                read_set.insert(probe.to_string());
            }
        }
    }

    let mut actions: Vec<Action> = vec![];
    if contains_any(user_request, &["build", "compile"]) {
        if root.join("Cargo.toml").exists() {
            actions.push(Action::Run { program: "cargo".into(), args: vec!["build".into()], workdir: None, log_hint: Some("build".into()) });
        } else if root.join("package.json").exists() {
            actions.push(Action::Run { program: "npm".into(), args: vec!["run".into(), "build".into()], workdir: None, log_hint: Some("build".into()) });
        }
    }
    if contains_any(user_request, &["test", "unit test", "pytest", "cargo test"]) {
        if root.join("Cargo.toml").exists() {
            actions.push(Action::Run { program: "cargo".into(), args: vec!["test".into()], workdir: None, log_hint: Some("test".into()) });
        } else if root.join("pyproject.toml").exists() {
            actions.push(Action::Run { program: "python".into(), args: vec!["-m".into(), "pytest".into()], workdir: None, log_hint: Some("test".into()) });
        } else if root.join("package.json").exists() {
            actions.push(Action::Run { program: "npm".into(), args: vec!["test".into()], workdir: None, log_hint: Some("test".into()) });
        }
    }
    if contains_any(user_request, &["run", "start server", "start dev", "dev server"]) {
        if root.join("Cargo.toml").exists() {
            actions.push(Action::Run { program: "cargo".into(), args: vec!["run".into()], workdir: None, log_hint: Some("run".into()) });
        } else if root.join("package.json").exists() {
            actions.push(Action::Run { program: "npm".into(), args: vec!["run".into(), "dev".into()], workdir: None, log_hint: Some("run".into()) });
        }
    }

    let mut read: Vec<String> = read_set.into_iter().collect();
    let mut edit: Vec<String> = edit_set.into_iter().collect();
    while read.len() + edit.len() > 6 {
        if !read.is_empty() { read.pop(); } else if !edit.is_empty() { edit.pop(); }
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
            "No explicit edit targets detected; providing context files and inferred actions.".to_string()
        } else {
            String::new()
        },
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
