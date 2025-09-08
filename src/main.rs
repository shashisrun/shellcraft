use anyhow::Result;
use console::style;
use std::path::{Path, PathBuf};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};

mod agents;
mod capabilities;
mod fsutil;
mod llm;
mod models;
mod planner;
mod task_ui;
mod ui;

// We inline a tiny diff preview + atomic write so we don't depend on
// diff/editor symbols that may differ in your tree.
use similar::{ChangeTag, TextDiff};
use std::io::Write as _;
use tempfile::NamedTempFile;
use tokio::fs as tokio_fs;

#[tokio::main]
async fn main() -> Result<()> {
    // Ctrl+C handling
    let running = Arc::new(AtomicBool::new(true));
    {
        let r = running.clone();
        ctrlc::set_handler(move || {
            r.store(false, Ordering::SeqCst);
        })?;
    }

    println!(
        "{}",
        style("Welcome to shellcraft — type /help for commands").green()
    );

    repl().await
}

async fn repl() -> Result<()> {
    loop {
        let user = ui::read_message_singleline("✔ User · >")?;
        let trimmed = user.trim();

        match trimmed {
            "/quit" | "/exit" => break,
            "/help" => {
                println!("{}", HELP_TEXT);
                continue;
            }
            _ => {}
        }

        if !trimmed.is_empty() {
            if let Err(e) = orchestrate(&user).await {
                eprintln!("{} {e:#}", style("Error:").red());
            }
        }
    }
    Ok(())
}

async fn orchestrate(user_input: &str) -> Result<()> {
    let root = std::env::current_dir()?;
    let manifest = capabilities::build_manifest(&root); // signature: (&Path) -> Manifest

    // Planner agent chats with user and returns plan
    let planner = agents::PlannerAgent::default();
    let plan = planner.chat_and_plan(&root, user_input, &manifest).await?;

    if !plan.notes.is_empty() {
        println!("{} {}", style("Notes:").cyan(), plan.notes);
    }

    // Reads
    for path in plan.read.iter() {
        let abs = root.join(path);
        match tokio_fs::read_to_string(&abs).await {
            Ok(content) => {
                println!("{} {}", style("Read:").yellow(), path);
                println!("{content}");
            }
            Err(err) => eprintln!("{} {} ({err})", style("Failed to read:").red(), path),
        }
    }

    // Deletes
    for path in plan.delete.iter() {
        let abs = root.join(path);
        if abs.exists() {
            if let Err(err) = fsutil::remove_path(&abs) {
                eprintln!("{} {} ({err})", style("Failed to delete:").red(), path);
            } else {
                println!("{} {}", style("Deleted:").red(), path);
            }
        } else {
            eprintln!("{} {} (not found)", style("Failed to delete:").red(), path);
        }
    }

    // Edits
    for edit in plan.edit.iter() {
        let file_path: PathBuf = root.join(&edit.path);
        let old_content = tokio_fs::read_to_string(&file_path)
            .await
            .unwrap_or_default();

        // llm::propose_edit(EditReq)
        let req = llm::EditReq {
            file_path: edit.path.clone(),
            file_content: old_content.clone(),
            instruction: edit.intent.clone(),
        };
        let proposal = llm::propose_edit(req).await.unwrap_or_default();

        print_unified_diff(&edit.path, &old_content, &proposal);
        atomic_write(&file_path, proposal.as_bytes())?;
        println!("{} {}", style("Applied:").green(), edit.path);
    }

    // Actions (placeholder): avoid referencing fields of planner::Action.
    if !plan.actions.is_empty() {
        println!(
            "{} {}",
            style("Planned actions:").cyan(),
            plan.actions.len()
        );
        // Interactive task dashboard for planned actions
        let mut items: Vec<task_ui::TaskItem> = plan
            .actions
            .iter()
            .enumerate()
            .filter_map(|(i, a)| match a {
                planner::Action::Run { program, args, .. } => Some(task_ui::TaskItem {
                    id: i,
                    summary: format!("{} {}", program, args.join(" ")),
                    detail: format!("program: {}\nargs: {}", program, args.join(" ")),
                    status: task_ui::TaskStatus::Pending,
                    expanded: false,
                }),
            })
            .collect();

        if !items.is_empty() {
            task_ui::task_dashboard(&mut items)?;
        }
        // TODO: replace with your actual runner call, e.g.:
        // runner::run_and_capture(&root, &plan.actions).await?;
    }

    Ok(())
}

fn print_unified_diff(rel_path: &str, old: &str, new: &str) {
    let diff = TextDiff::from_lines(old, new);
    println!(
        "{}",
        style(format!("--- a/{rel_path}\n+++ b/{rel_path}")).dim()
    );
    for change in diff.iter_all_changes() {
        let (sign, s) = match change.tag() {
            ChangeTag::Delete => ("-", style(change).red()),
            ChangeTag::Insert => ("+", style(change).green()),
            ChangeTag::Equal => (" ", style(change).dim()),
        };
        print!("{sign}{s}");
    }
    println!();
}

fn atomic_write(path: &Path, bytes: &[u8]) -> Result<()> {
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    std::fs::create_dir_all(parent)?;
    let mut tmp = NamedTempFile::new_in(parent)?;
    tmp.write_all(bytes)?;
    tmp.flush()?;
    tmp.persist(path)?;
    Ok(())
}

const HELP_TEXT: &str = r#"
Input:
  • Enter submits
  • Shift+Enter inserts newline (best effort); Ctrl+Enter as fallback
  • Pasting preserves newlines and does not auto-submit
Commands:
  • /env KEY=VAL       – set & persist an env var
  • /model <MODEL_ID>  – switch model for this session
  • /capabilities      – show detected tools/providers
  • /help              – this message
  • /quit or /exit     – quit shellcraft
"#;
