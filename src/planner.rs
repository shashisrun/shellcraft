use anyhow::{anyhow, Context, Result};
use serde::{Deserialize, Serialize};
use std::path::Path;

use crate::capabilities::{can_run, system_preamble, Manifest};
use crate::fsutil::{file_inventory, FileMeta};
use crate::llm;

/// Final plan from planner
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Plan {
    #[serde(default)]
    pub read: Vec<String>,
    #[serde(default)]
    pub edit: Vec<EditPlan>,
    #[serde(default)]
    pub delete: Vec<String>,
    #[serde(default)]
    pub actions: Vec<Action>,
    #[serde(default)]
    pub notes: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EditPlan {
    pub path: String,
    pub intent: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind")]
pub enum Action {
    #[serde(rename = "run")]
    Run {
        program: String,
        args: Vec<String>,
        #[serde(default)]
        workdir: Option<String>,
        #[serde(default)]
        log_hint: Option<String>,
        #[serde(default = "default_retries")]
        retries: u32,
        #[serde(default = "default_backoff")]
        backoff_ms: u64,
    },
}

fn default_retries() -> u32 {
    1
}
fn default_backoff() -> u64 {
    1500
}

#[derive(Debug, Serialize)]
struct PlanPrompt<'a> {
    user_request: &'a str,
    file_index: &'a [FileMeta],
    guidance: &'a str,
    capabilities: &'a str,
}

fn guidance() -> String {
    r#"Heuristics:
- Do not call external tools like `repo_browser.print_tree`; the file index
  already contains the repository structure.
- Prefer touching the fewest files.
- If the ask is informational only, leave `edit=[]` and put a short answer in `notes`.
- Use actions only for tools that are enabled in the capabilities list.
- For Rust projects, typical actions are: `cargo build`, `cargo test`.
- Use `delete` for files or directories that should be removed.
- Always fill `retries` and `backoff_ms` (small numbers).
Schema:
{
  "read": string[],
  "edit": [{"path": string, "intent": string}],
  "delete": string[],
  "actions": [{"kind":"run","program":string,"args":string[],"workdir?":string,"log_hint?":string,"retries":number,"backoff_ms":number}],
  "notes": string
}
Return pure JSON, no markdown."#.to_string()
}

/// Build a plan using the LLM and preflight
pub async fn plan_changes(root: &Path, user_request: &str, manifest: &Manifest) -> Result<Plan> {
    let mut index = file_inventory(root)?;
    if index.len() > 800 {
        index = compact_index(index);
    }

    // Ask LLM with capability preamble
    let preamble = system_preamble(manifest);
    let prompt = PlanPrompt {
        user_request,
        file_index: &index,
        guidance: &guidance(),
        capabilities: &preamble,
    };

    let mut plan: Plan = llm::chat_json(
        &format!("You are a senior planner.\n{}\n", preamble),
        &serde_json::to_string(&prompt).unwrap(),
    )
    .await
    .context("planner LLM failed")?;
    if !validate_plan_paths(root, &plan) {
        return Err(anyhow!("LLM returned non-existent file paths"));
    }

    // Preflight: drop invalid actions & annotate notes
    preflight_actions(manifest, &mut plan);

    Ok(plan)
}

fn validate_plan_paths(root: &Path, plan: &Plan) -> bool {
    for p in plan
        .read
        .iter()
        .chain(plan.edit.iter().map(|e| &e.path))
        .chain(plan.delete.iter())
    {
        if !root.join(p).exists() {
            return false;
        }
    }
    true
}

pub fn preflight_actions(manifest: &Manifest, plan: &mut Plan) {
    let mut kept = vec![];
    let mut dropped = vec![];
    for a in &plan.actions {
        match a {
            Action::Run { program, .. } => {
                let (ok, why) = can_run(manifest, program);
                if ok {
                    kept.push(a.clone());
                } else {
                    dropped.push(format!(
                        "drop action `{}`: {}",
                        program,
                        why.unwrap_or_default()
                    ));
                }
            }
        }
    }
    plan.actions = kept;
    if !dropped.is_empty() {
        if !plan.notes.is_empty() {
            plan.notes.push_str("\n");
        }
        plan.notes.push_str(&format!(
            "Preflight removed actions not supported in this environment:\n- {}",
            dropped.join("\n- ")
        ));
    }
}

/// Keep top ~800 source-like files
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
