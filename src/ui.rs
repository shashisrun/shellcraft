use std::path::Path;
use std::time::Duration;
use std::io::{self, Write};

use crate::runner;

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

    println!("{}", top_bottom);
    println!("{}", pad_line(line_cmd));
    println!("{}", pad_line(line_dir));
    println!("{}", top_bottom);
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
    println!("Exit code: {} | Elapsed: {:.3}s", exit_code, secs);
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
    println!("{}", "─".repeat(width));
}

/// Prints a message from the system (e.g., the tool) with a clear label and
/// optional indentation for multi‑line content.
pub fn print_system_message(message: &str) {
    print_separator();
    for (i, line) in message.lines().enumerate() {
        if i == 0 {
            println!("🤖 {}", line);
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
            println!("🧑 {}", line);
        } else {
            println!("   {}", line);
        }
    }
    print_separator();
}

/// Prompts the user for input, displaying the given prompt text. The function
/// trims whitespace and repeats the prompt until a non‑empty line is entered,
/// handling `Ctrl‑C` gracefully by returning an empty string.
pub fn prompt_user(prompt: &str) -> String {
    loop {
        // Show the prompt without a newline and flush stdout so the user sees it.
        print!("{} ", prompt);
        let _ = io::stdout().flush();

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
/*                         Optional dependency note                           */
/* -------------------------------------------------------------------------- */

/*
The `print_separator` function uses the `term_size` crate to adapt the line
length to the current terminal width. If you prefer not to add an external
dependency, you can replace the call with a fixed width, e.g.:

let width = 80;
println!("{}", "─".repeat(width));
*/

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