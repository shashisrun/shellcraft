use similar::{ChangeTag, TextDiff};

/// Human-friendly unified diff (with +/- markers).
pub fn unified_colored(old: &str, new: &str, path: &str) -> String {
    let diff = TextDiff::from_lines(old, new);
    let mut out = String::new();
    out.push_str(&format!("--- a/{path}\n+++ b/{path}\n"));
    for op in diff.ops() {
        for change in diff.iter_changes(op) {
            match change.tag() {
                ChangeTag::Delete => out.push_str(&format!("-{}", change)),
                ChangeTag::Insert => out.push_str(&format!("+{}", change)),
                ChangeTag::Equal => out.push_str(&format!(" {}", change)),
            }
        }
    }
    out
}
