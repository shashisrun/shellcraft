use std::io::{self, Write};
use std::path::Path;
use std::sync::Mutex;
use std::time::{Duration, Instant, SystemTime};

use crossterm::style::{Color, Stylize};

use indicatif::{ProgressBar, ProgressStyle};

use once_cell::sync::Lazy;

use crate::runner;

/// Convert an `Instant` to a `SystemTime` by offsetting from the current time.
/// This provides an approximate wall‑clock timestamp for reporting purposes.
fn instant_to_system_time(instant: Instant) -> SystemTime {
    let now_instant = Instant::now();
    let now_system = SystemTime::now();
    if now_instant >= instant {
        now_system
            .checked_sub(now_instant.duration_since(instant))
            .unwrap_or(now_system)
    } else {
        now_system
            .checked_add(instant.duration_since(now_instant))
            .unwrap_or(now_system)
    }
}

#[allow(dead_code)]

/// Prints a decorative boxed header showing the command being executed and the
/// current working directory. This is used by the `/run` and `/test` commands
/// to give the user a clear visual cue of what is happening.
///
/// Example output:
/// ```text
/// +------------------------------+
/// | Command: cargo run --example |
/// | Dir: /home/user/project      |
/// +------------------------------+
/// ```
pub fn boxed_header(cmd: &str, cwd: &Path) {
    let cwd_str = cwd.display().to_string();

    // Prepare the two lines that will appear inside the box.
    let line_cmd = format!(" Command: {}", cmd);
    let line_dir = format!(" Dir: {}", cwd_str);

    // Determine the width of the box (including a single space padding on each side).
    let inner_width = line_cmd.len().max(line_dir.len()) + 2; // 1 space left, 1 space right
    let top_bottom = format!("+{}+", "-".repeat(inner_width));

    // Helper to pad a line to the full inner width.
    let pad_line = |content: String| {
        let padding = inner_width - content.len();
        format!("|{}{}|", content, " ".repeat(padding))
    };

    // Render with colors
    println!("{}", top_bottom.clone().with(Color::Cyan));
    println!("{}", pad_line(line_cmd).with(Color::Green));
    println!("{}", pad_line(line_dir).with(Color::Yellow));
    println!("{}", top_bottom.clone().with(Color::Cyan));
}

/// Prints a concise status line after a command finishes, showing the exit
/// code and the elapsed time in seconds with three decimal places.
///
/// Example output:
/// ```text
/// Exit code: 0 | Elapsed: 1.237 s
/// ```
pub fn print_status(exit_code: i32, duration: Duration) {
    let secs = duration.as_secs_f64();
    let exit_col = if exit_code == 0 {
        Color::Green
    } else {
        Color::Red
    };
    println!(
        "{} | {}",
        format!("Exit code: {}", exit_code).with(exit_col),
        format!("Elapsed: {:.3}s", secs).with(Color::Cyan)
    );
}

/* -------------------------------------------------------------------------- */
/*                     Enhanced chat / UI helper functions                     */
/* -------------------------------------------------------------------------- */

/// Prints a visual separator between chat messages. The separator adapts to
/// the terminal width (fallback to 80 columns) and uses a simple line of `─`.
pub fn print_separator() {
    // Try to obtain the terminal width; if unavailable, default to 80.
    let width = match term_size::dimensions_stdout() {
        Some((w, _)) => w,
        None => 80,
    };
    println!("{}", "─".repeat(width).with(Color::DarkGrey));
}

/// Prints a message from the system (e.g., the tool) with a clear label and
/// optional indentation for multi‑line content.
pub fn print_system_message(message: &str) {
    print_separator();
    for (i, line) in message.lines().enumerate() {
        if i == 0 {
            println!("{} {}", "🤖".with(Color::Magenta), line);
        } else {
            println!("   {}", line);
        }
    }
    print_separator();
}

/// Prints a message from the user with a distinct label.
pub fn print_user_message(message: &str) {
    print_separator();
    for (i, line) in message.lines().enumerate() {
        if i == 0 {
            println!("{} {}", "🧑".with(Color::Blue), line);
        } else {
            println!("   {}", line);
        }
    }
    print_separator();
}

/// Renders a heading (e.g., section title) in bold cyan.
pub fn render_heading(text: &str) {
    println!("\n{}", text.bold().with(Color::Cyan));
}

/// Renders a generic status line in green.
pub fn render_status(text: &str) {
    println!("{}", text.with(Color::Green));
}

/// Renders a prompt string in yellow without a trailing newline and flushes stdout.
pub fn render_prompt(prompt: &str) {
    print!("{} ", prompt.bold().with(Color::Yellow));
    let _ = io::stdout().flush();
}

/// Prompts the user for input, displaying the given prompt text. The function
/// trims whitespace and repeats the prompt until a non‑empty line is entered,
/// handling `Ctrl‑C` gracefully by returning an empty string.
pub fn prompt_user(prompt: &str) -> String {
    loop {
        render_prompt(prompt);
        let mut input = String::new();
        match io::stdin().read_line(&mut input) {
            Ok(0) => {
                // EOF (Ctrl‑D) – treat as empty input.
                return String::new();
            }
            Ok(_) => {
                let trimmed = input.trim();
                if !trimmed.is_empty() {
                    return trimmed.to_string();
                }
                // Empty input; re‑prompt.
            }
            Err(_) => {
                // On error (including Ctrl‑C), return an empty string.
                return String::new();
            }
        }
    }
}

/* -------------------------------------------------------------------------- */
/*                         Simple UI convenience wrappers                     */
/* -------------------------------------------------------------------------- */

/// Simple wrapper that prints a line of text followed by a newline.
pub fn print(msg: &str) {
    println!("{}", msg);
}

/// Prints a banner shown at program start.
pub fn banner() {
    // A simple banner; feel free to replace with something fancier.
    println!("{}", "=== Shellcraft ===".with(Color::Cyan).bold());
}

/// Prints an informational message in cyan.
pub fn info(msg: &str) {
    println!("{}", msg.with(Color::Cyan));
}

/// Prints a success message in green.
pub fn success(msg: &str) {
    println!("{}", msg.with(Color::Green));
}

/// Prints an error message in red (to stderr).
pub fn error(msg: &str) {
    eprintln!("{}", msg.with(Color::Red));
}

/// Returns a spinner progress bar with the given message.
pub fn spinner(message: &str) -> ProgressBar {
    let pb = ProgressBar::new_spinner();
    pb.set_message(message.to_string());
    pb.enable_steady_tick(Duration::from_millis(100));
    pb.set_style(
        ProgressStyle::default_spinner()
            .template("{spinner} {msg}")
            .unwrap(),
    );
    pb
}

/// Reads multiline input from stdin until a line containing only `/end` is
/// entered. Returns the collected text (excluding the terminating line). An
/// empty string is returned on EOF or error.
pub fn read_multiline_input() -> io::Result<String> {
    let mut buffer = String::new();
    loop {
        let mut line = String::new();
        let bytes = io::stdin().read_line(&mut line)?;
        if bytes == 0 {
            // EOF
            break;
        }
        let trimmed = line.trim_end_matches(&['\n', '\r'][..]);
        if trimmed == "/end" {
            break;
        }
        buffer.push_str(trimmed);
        buffer.push('\n');
    }
    Ok(buffer)
}

/* -------------------------------------------------------------------------- */
/* -------------------------------------------------------------------------- */

/// Prompt the user for a terminal command, execute it via `runner::run_command`,
/// and display the resulting output or error in the UI.
///
/// This is a minimal integration point that can be wired to any existing UI
/// action (e.g., a menu entry or a command palette option).
pub fn prompt_and_run_terminal_command() {
    // Ask the user for a command.
    let cmd = prompt_user("Enter terminal command:");
    if cmd.is_empty() {
        // Nothing entered – simply return.
        return;
    }

    // Execute the command using the shared runner logic.
    match runner::run_command(&cmd) {
        Ok(output) => {
            // Show the command output.
            print_system_message(&format!("Command succeeded:\n{}", output));
        }
        Err(e) => {
            // Show the error.
            print_system_message(&format!("Command failed: {}", e));
        }
    }
}

/// Starts a simple interactive REPL loop. The user types a line, which is
/// echoed back as a system response. Typing `exit` (case‑insensitive) quits
/// the loop.
pub fn start_repl() {
    render_heading("Interactive REPL (type 'exit' to quit)");
    loop {
        let input = prompt_user(">>>");

        if input.eq_ignore_ascii_case("exit") {
            render_status("Exiting REPL.");
            break;
        }

        if input.is_empty() {
            continue;
        }

        // Show what the user typed.
        print_user_message(&input);

        // Placeholder for real processing – echo back for now.
        print_system_message(&format!("You said: {}", input));
    }
}

/* -------------------------------------------------------------------------- */
/*                         Observability / Reporting                         */
/* -------------------------------------------------------------------------- */

/// A single timeline entry describing a task execution.
#[derive(Debug, Clone)]
struct TimelineEntry {
    start: Instant,
    end: Instant,
    agent: String,
    llm: String,
    tokens: u64,
    verdict: String,
}

/// Global timeline storage.  Mutex protects concurrent writes from async contexts.
static TIMELINE: Lazy<Mutex<Vec<TimelineEntry>>> = Lazy::new(|| Mutex::new(Vec::new()));

/// Record a completed task in the timeline.
///
/// # Arguments
///
/// * `start` – When the task started.
/// * `end` – When the task finished.
/// * `agent` – Identifier of the agent that performed the task.
/// * `llm` – Name of the LLM used (if any).
/// * `tokens` – Number of tokens consumed.
/// * `verdict` – Short outcome description (e.g., "success", "failed").
pub fn record_task(
    start: Instant,
    end: Instant,
    agent: &str,
    llm: &str,
    tokens: u64,
    verdict: &str,
) {
    let entry = TimelineEntry {
        start,
        end,
        agent: agent.to_string(),
        llm: llm.to_string(),
        tokens,
        verdict: verdict.to_string(),
    };
    let mut timeline = TIMELINE.lock().unwrap();
    timeline.push(entry);
}

/// Generate a markdown report from the collected timeline.
///
/// The report includes:
/// * Goal and a snippet of the plan.
/// * A table of task outcomes.
/// * File change statistics (placeholder).
/// * Runtime & token totals per LLM provider.
/// * Open risks / manual follow‑ups (placeholder).
pub fn generate_report(goal: &str, plan_snippet: &str) -> String {
    let timeline = TIMELINE.lock().unwrap();

    // Compute aggregates.
    let total_duration = timeline
        .iter()
        .map(|e| e.end.duration_since(e.start))
        .fold(Duration::ZERO, |acc, d| acc + d);

    let mut tokens_per_llm = std::collections::HashMap::<String, u64>::new();
    for e in timeline.iter() {
        *tokens_per_llm.entry(e.llm.clone()).or_insert(0) += e.tokens;
    }

    // Header.
    let mut md = String::new();
    md.push_str(&format!("# Goal\n\n{}\n\n", goal));
    md.push_str("## Plan snippet\n\n```text\n");
    md.push_str(plan_snippet);
    md.push_str("\n```\n\n");

    // Timeline table.
    md.push_str("## Timeline\n\n");
    md.push_str("| Start | End | Duration (s) | Agent | LLM | Tokens | Verdict |\n");
    md.push_str("|-------|-----|--------------|-------|-----|--------|---------|\n");
    for e in timeline.iter() {
        let start = humantime::format_rfc3339(instant_to_system_time(e.start));
        let end = humantime::format_rfc3339(instant_to_system_time(e.end));
        let dur = e.end.duration_since(e.start).as_secs_f64();
        md.push_str(&format!(
            "| {} | {} | {:.3} | {} | {} | {} | {} |\n",
            start, end, dur, e.agent, e.llm, e.tokens, e.verdict
        ));
    }
    md.push('\n');

    // Placeholder for file changes.
    md.push_str("## Files changed\n\n");
    md.push_str("_File change statistics would appear here (e.g., `git diff --stat`)._\n\n");

    // Runtime & token totals.
    md.push_str("## Runtime & Token totals per provider\n\n");
    md.push_str("| LLM Provider | Tokens |\n");
    md.push_str("|--------------|--------|\n");
    for (llm, tokens) in tokens_per_llm.iter() {
        md.push_str(&format!("| {} | {} |\n", llm, tokens));
    }
    md.push_str(&format!(
        "\n**Total runtime:** {:.3} s\n\n",
        total_duration.as_secs_f64()
    ));

    // Open risks / manual follow‑ups.
    md.push_str("## Open risks / manual follow‑ups\n\n");
    md.push_str("_List any remaining risks or actions that require human attention._\n");

    md
}

/// Print the report to stdout and also copy it to the clipboard (if possible).
pub fn report_command(goal: &str, plan_snippet: &str) {
    let report = generate_report(goal, plan_snippet);
    // Print to stdout.
    println!("{}", report);

    // Attempt to copy to clipboard using the `arboard` crate.
    #[cfg(any(target_os = "windows", target_os = "macos", target_os = "linux"))]
    {
        if let Ok(mut clipboard) = arboard::Clipboard::new() {
            let _ = clipboard.set_text(report.clone());
        }
    }
}

/* -------------------------------------------------------------------------- */
/*                         Power‑toggle Settings                              */
/* -------------------------------------------------------------------------- */

/// Runtime‑adjustable settings that control the agent's behaviour.
#[derive(Debug, Clone)]
struct Settings {
    /// If true, the agent will ask for confirmation before executing
    /// potentially destructive commands (e.g., `git reset --hard`).
    ask_before_destructive: bool,
}

impl Default for Settings {
    fn default() -> Self {
        Settings {
            ask_before_destructive: true,
        }
    }
}

/// Global mutable settings protected by a mutex.
static SETTINGS: Lazy<Mutex<Settings>> = Lazy::new(|| Mutex::new(Settings::default()));

/// Retrieve a copy of the current settings.
pub fn get_settings() -> Settings {
    SETTINGS.lock().unwrap().clone()
}

/// Update the `ask_before_destructive` flag.
pub fn set_ask_before_destructive(value: bool) {
    let mut s = SETTINGS.lock().unwrap();
    s.ask_before_destructive = value;
}

/* -------------------------------------------------------------------------- */
/*                         UI Helper Functions                                 */
/* -------------------------------------------------------------------------- */

/// Print a minimal help message with the available power‑toggle commands.
pub fn print_help() {
    render_heading("Help – Power Toggles & Commands");
    println!(
        "{} – Toggle confirmation before destructive commands.\n    Usage: /toggle ask_before_destructive [on|off]",
        "/toggle".with(Color::Magenta)
    );
    println!("\nCurrent settings:");
    let s = get_settings();
    let status = if s.ask_before_destructive {
        "on"
    } else {
        "off"
    };
    println!("  ask_before_destructive: {}", status);
    println!("\nOther commands:");
    println!("  /report   – Generate a markdown report of the session.");
    println!("  /timeline – Show a concise timeline of executed tasks.");
    println!("  /review   – Run rustfmt and clippy on the current project.");
    println!("  /help     – Show this help message.");
}

/// Suggest possible actions for purely informational inputs.
pub fn suggest_actions() {
    render_status("Suggested actions: build, test, lint");
}

/* -------------------------------------------------------------------------- */
/*                         Reviewer Agent                                       */
/* -------------------------------------------------------------------------- */

/// Runs `cargo fmt -- --check` and `cargo clippy -- -D warnings` on the given
/// project directory, reports the results, and records a timeline entry.
///
/// This provides a quick code‑style and lint check without involving an LLM.
pub fn reviewer_agent(project_path: &Path) {
    let original_dir = match std::env::current_dir() {
        Ok(d) => d,
        Err(e) => {
            error(&format!("Failed to get current directory: {}", e));
            return;
        }
    };

    if let Err(e) = std::env::set_current_dir(project_path) {
        error(&format!(
            "Failed to change directory to {}: {}",
            project_path.display(),
            e
        ));
        return;
    }

    let start = Instant::now();
    let mut verdict = "success";

    // Run rustfmt check.
    match runner::run_command("cargo fmt -- --check") {
        Ok(output) => {
            print_system_message(&format!("rustfmt check passed:\n{}", output));
        }
        Err(e) => {
            verdict = "rustfmt failed";
            print_system_message(&format!("rustfmt check failed:\n{}", e));
        }
    }

    // Run clippy.
    match runner::run_command("cargo clippy -- -D warnings") {
        Ok(output) => {
            print_system_message(&format!("clippy passed:\n{}", output));
        }
        Err(e) => {
            verdict = if verdict == "success" {
                "clippy failed"
            } else {
                "rustfmt & clippy failed"
            };
            print_system_message(&format!("clippy failed:\n{}", e));
        }
    }

    let end = Instant::now();

    // Restore original working directory.
    let _ = std::env::set_current_dir(original_dir);

    record_task(start, end, "ReviewerAgent", "none", 0, verdict);
    render_status(&format!(
        "ReviewerAgent completed with verdict: {}",
        verdict
    ));
}

/* -------------------------------------------------------------------------- */
/*                         Timeline Display                                      */
/* -------------------------------------------------------------------------- */

/// Prints a concise, human‑readable timeline of all recorded tasks.
pub fn display_timeline() {
    let timeline = TIMELINE.lock().unwrap();
    if timeline.is_empty() {
        render_status("No timeline entries recorded yet.");
        return;
    }

    println!("\n{}", "=== Timeline ===".with(Color::Cyan).bold());
    for e in timeline.iter() {
        let start = humantime::format_rfc3339(instant_to_system_time(e.start));
        let end = humantime::format_rfc3339(instant_to_system_time(e.end));
        let dur = e.end.duration_since(e.start).as_secs_f64();
        println!(
            "{} → {} ({:.3}s) | Agent: {} | Verdict: {}",
            start, end, dur, e.agent, e.verdict
        );
    }
    println!();
}

/* -------------------------------------------------------------------------- */
/*                         Command Dispatcher                                   */
/* -------------------------------------------------------------------------- */

/// Simple command dispatcher for UI/CLI input. Recognises `/report`, `/help`,
/// `/timeline`, `/review`, and `/toggle` commands. Other inputs are ignored here
/// and can be handled elsewhere (e.g., by the chat‑first agent logic).
pub fn handle_ui_command(input: &str, goal: &str, plan_snippet: &str) {
    match input.trim() {
        "/report" => {
            report_command(goal, plan_snippet);
        }
        "/timeline" => {
            display_timeline();
        }
        "/review" => {
            // Run reviewer on the current working directory.
            if let Ok(cwd) = std::env::current_dir() {
                reviewer_agent(&cwd);
            } else {
                error("Unable to determine current directory for review.");
            }
        }
        "/help" => {
            print_help();
        }
        cmd if cmd.starts_with("/toggle") => {
            // Expected format: /toggle <key> <on|off>
            let parts: Vec<&str> = cmd.split_whitespace().collect();
            if parts.len() == 3 {
                let key = parts[1];
                let value = parts[2];
                match (key, value) {
                    ("ask_before_destructive", "on") => {
                        set_ask_before_destructive(true);
                        render_status("ask_before_destructive set to on");
                    }
                    ("ask_before_destructive", "off") => {
                        set_ask_before_destructive(false);
                        render_status("ask_before_destructive set to off");
                    }
                    _ => {
                        render_status(
                            "Unknown toggle or value. Use: /toggle ask_before_destructive [on|off]",
                        );
                    }
                }
            } else {
                render_status("Invalid toggle command. Usage: /toggle <key> <on|off>");
            }
        }
        _ => {
            // No special handling; could be routed to other UI actions.
        }
    }
}
