use anyhow::{anyhow, Context, Result};
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
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
- Prefer touching the fewest files.
- If the ask is informational only, leave `edit=[]` and put a short answer in `notes`.
- Use actions only for tools that are enabled in the capabilities list.
- For Rust projects, typical actions are: `cargo build`, `cargo test`.
- Always fill `retries` and `backoff_ms` (small numbers).
Schema:
{
  "read": string[],
  "edit": [{"path": string, "intent": string}],
  "actions": [{"kind":"run","program":string,"args":string[],"workdir?":string,"log_hint?":string,"retries":number,"backoff_ms":number}],
  "notes": string
}
Return pure JSON, no markdown."#.to_string()
}

/// Build a plan using the LLM + fallback + preflight
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
    .context("planner LLM failed")
    .and_then(|p: Plan| {
        if validate_plan_paths(root, &p) {
            Ok(p)
        } else {
            Err(anyhow!("LLM returned non-existent file paths"))
        }
    })
    .unwrap_or_else(|_| fallback_plan(root, user_request, &index));

    // Preflight: drop invalid actions & annotate notes
    preflight_actions(manifest, &mut plan);

    Ok(plan)
}

fn validate_plan_paths(root: &Path, plan: &Plan) -> bool {
    for p in plan.read.iter().chain(plan.edit.iter().map(|e| &e.path)) {
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
                    dropped.push(format!("drop action `{}`: {}", program, why.unwrap_or_default()));
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

/// Fallback heuristic when LLM fails
fn fallback_plan(root: &Path, user_request: &str, _index: &[FileMeta]) -> Plan {
    let mut read_set: HashSet<String> = HashSet::new();
    let mut edit_set: HashSet<String> = HashSet::new();
    let mut actions: Vec<Action> = Vec::new();

    let token_re = Regex::new(r"[A-Za-z0-9_/\\.-]+\.[A-Za-z0-9]+").unwrap();
    for cap in token_re.captures_iter(user_request) {
        let candidate = cap.get(0).unwrap().as_str().replace('\\', "/");
        if root.join(&candidate).exists() {
            if user_request.to_lowercase().contains("edit")
                || user_request.to_lowercase().contains("modify")
                || user_request.to_lowercase().contains("change")
            {
                edit_set.insert(candidate);
            } else {
                read_set.insert(candidate);
            }
        }
    }

    // Defaults
    if read_set.is_empty() && edit_set.is_empty() {
        for probe in ["src/main.rs", "Cargo.toml", "README.md", "readme.md"].iter() {
            if root.join(probe).exists() {
                read_set.insert((*probe).into());
            }
        }
    }

    // Infer simple actions
    let low = user_request.to_lowercase();
    if low.contains("build") && root.join("Cargo.toml").exists() {
        actions.push(Action::Run {
            program: "cargo".into(),
            args: vec!["build".into()],
            workdir: None,
            log_hint: Some("build".into()),
            retries: 1,
            backoff_ms: 1200,
        });
    }
    if low.contains("test") && root.join("Cargo.toml").exists() {
        actions.push(Action::Run {
            program: "cargo".into(),
            args: vec!["test".into()],
            workdir: None,
            log_hint: Some("test".into()),
            retries: 1,
            backoff_ms: 1200,
        });
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

    let edit_plans = edit
        .into_iter()
        .map(|p| EditPlan {
            path: p.clone(),
            intent: format!("Apply changes inferred from request: \"{}\"", user_request),
        })
        .collect::<Vec<_>>();

    Plan {
        read,
        edit: edit_plans,
        actions,
        notes: "Fallback plan generated heuristically.".into(),
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
