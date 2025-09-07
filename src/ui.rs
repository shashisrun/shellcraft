use std::io::{self, Write};
use std::path::Path;
use std::time::Duration;

use crossterm::style::{Color, Stylize};

use indicatif::{ProgressBar, ProgressStyle};

use term_size;

use crate::runner;

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
