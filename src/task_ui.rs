use anyhow::Result;
use crossterm::{
    cursor,
    event::{self, Event, KeyCode},
    queue,
    style,
    terminal::{Clear, ClearType, disable_raw_mode, enable_raw_mode},
};
use std::io::{stdout, Write};
use std::time::Duration;

#[derive(Clone, Copy)]
pub enum TaskStatus {
    Pending,
    Running,
    Paused,
    Cancelled,
}

#[derive(Clone)]
pub struct TaskItem {
    pub id: usize,
    pub summary: String,
    pub detail: String,
    pub status: TaskStatus,
    pub expanded: bool,
}

pub fn task_dashboard(tasks: &mut [TaskItem]) -> Result<()> {
    enable_raw_mode()?;
    let mut out = stdout();
    let mut selected: usize = 0;

    loop {
        queue!(out, cursor::MoveTo(0, 0), Clear(ClearType::All))?;
        for (idx, task) in tasks.iter().enumerate() {
            let prefix = if idx == selected { ">" } else { " " };
            let status = match task.status {
                TaskStatus::Pending => "pending",
                TaskStatus::Running => "running",
                TaskStatus::Paused => "paused",
                TaskStatus::Cancelled => "cancelled",
            };
            queue!(
                out,
                style::Print(format!("{prefix} [{status}] {}\n", task.summary))
            )?;
            if task.expanded && idx == selected {
                queue!(out, style::Print(format!("    {}\n", task.detail)))?;
            }
        }
        queue!(out, style::Print("\nq: quit  Enter: expand  p: pause/resume  c: cancel"))?;
        out.flush()?;

        if event::poll(Duration::from_millis(250))? {
            match event::read()? {
                Event::Key(key) => match key.code {
                    KeyCode::Char('q') => break,
                    KeyCode::Up => {
                        if selected > 0 {
                            selected -= 1;
                        }
                    }
                    KeyCode::Down => {
                        if selected + 1 < tasks.len() {
                            selected += 1;
                        }
                    }
                    KeyCode::Enter => {
                        tasks[selected].expanded = !tasks[selected].expanded;
                    }
                    KeyCode::Char('c') => {
                        tasks[selected].status = TaskStatus::Cancelled;
                    }
                    KeyCode::Char('p') => {
                        tasks[selected].status = match tasks[selected].status {
                            TaskStatus::Paused => TaskStatus::Running,
                            TaskStatus::Running | TaskStatus::Pending => TaskStatus::Paused,
                            other => other,
                        };
                    }
                    _ => {}
                },
                _ => {}
            }
        }
    }
    disable_raw_mode()?;
    // Move to next line to avoid overwriting prompt
    queue!(out, cursor::MoveTo(0, (tasks.len() + 3) as u16), Clear(ClearType::CurrentLine))?;
    out.flush()?;
    Ok(())
}

