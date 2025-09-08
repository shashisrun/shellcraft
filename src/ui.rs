use std::io::{stdout, Write};
use crossterm::{
    cursor,
    event::{self, Event, KeyCode, KeyEvent, KeyModifiers},
    execute, queue,
    style,
    terminal::{Clear, ClearType, disable_raw_mode, enable_raw_mode},
};

/// Read a single message with:
/// - Enter submits
/// - Shift+Enter inserts newline (best effort); Ctrl+Enter as portable fallback
/// - Bracketed paste keeps multi-line content as-is
pub fn read_message_singleline(prompt: &str) -> anyhow::Result<String> {
    let mut out = stdout();
    enable_raw_mode()?;
    // Best effort: bracketed paste makes pastes arrive as Event::Paste(String)
    execute!(out, event::EnableBracketedPaste)?;

    let mut buf = String::new();
    render_prompt(&mut out, prompt, &buf)?;

    loop {
        if event::poll(std::time::Duration::from_millis(250))? {
            match event::read()? {
                Event::Key(KeyEvent { code: KeyCode::Enter, modifiers, .. }) => {
                    if modifiers.contains(KeyModifiers::SHIFT) || modifiers.contains(KeyModifiers::CONTROL) {
                        buf.push('\n');
                        render_prompt(&mut out, prompt, &buf)?;
                        continue;
                    }
                    break; // plain Enter submits
                }
                Event::Key(KeyEvent { code: KeyCode::Char(c), modifiers, .. }) => {
                    if modifiers.contains(KeyModifiers::CONTROL) {
                        match c {
                            'u' | 'U' => buf.clear(),        // Ctrl+U: clear
                            'w' | 'W' => {                   // Ctrl+W: delete word
                                let trimmed = buf.trim_end_matches(|ch: char| ch.is_whitespace());
                                let cut = trimmed.rfind(|ch: char| ch.is_whitespace()).map(|i| i + 1).unwrap_or(0);
                                buf.truncate(cut);
                            }
                            _ => {}
                        }
                    } else {
                        buf.push(c);
                    }
                    render_prompt(&mut out, prompt, &buf)?;
                }
                Event::Key(KeyEvent { code: KeyCode::Backspace, .. }) => {
                    buf.pop();
                    render_prompt(&mut out, prompt, &buf)?;
                }
                Event::Key(KeyEvent { code: KeyCode::Tab, .. }) => {
                    buf.push('\t');
                    render_prompt(&mut out, prompt, &buf)?;
                }
                Event::Key(KeyEvent { code: KeyCode::Esc, .. }) => {
                    // ESC clears current line (keeps REPL)
                    buf.clear();
                    render_prompt(&mut out, prompt, &buf)?;
                }
                Event::Paste(s) => {
                    buf.push_str(&s);
                    render_prompt(&mut out, prompt, &buf)?;
                }
                Event::Resize(_, _) => {
                    render_prompt(&mut out, prompt, &buf)?;
                }
                _ => {}
            }
        }
    }

    queue!(out, style::Print("\r\n"))?;
    execute!(out, event::DisableBracketedPaste)?;
    disable_raw_mode()?;
    Ok(buf)
}

fn render_prompt<W: Write>(out: &mut W, prompt: &str, buf: &str) -> anyhow::Result<()> {
    queue!(
        out,
        cursor::MoveToColumn(0),
        Clear(ClearType::CurrentLine),
        style::Print(prompt),
        style::Print(" "),
        style::Print(buf)
    )?;
    out.flush()?;
    Ok(())
}

pub fn handle_help() -> anyhow::Result<String> {
    let help_message = "/help: this help message
/quit: quit the application
/exit: quit the application";
    let mut out = stdout();
    queue!(out, style::Print(help_message))?;
    out.flush()?;
    Ok(String::new())
}

pub fn handle_quit() -> anyhow::Result<String> {
    let quit_message = "Goodbye!";
    let mut out = stdout();
    queue!(out, style::Print(quit_message))?;
    out.flush()?;
    std::process::exit(0);
}

pub fn handle_exit() -> anyhow::Result<String> {
    let exit_message = "Goodbye!";
    let mut out = stdout();
    queue!(out, style::Print(exit_message))?;
    out.flush()?;
    std::process::exit(0);
}