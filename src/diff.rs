use console::style;
use similar::{ChangeTag, TextDiff};

/// Render a unified, colorized diff between `old` and `new` for display in the
/// terminal. `rel_path` is only used in the header lines.
pub fn unified_colored(old: &str, new: &str, rel_path: &str) -> String {
    let diff = TextDiff::from_lines(old, new);
    let mut out = String::new();

    // Unified diff style headers with colors
    out.push_str(&format!("{}--- a/{}\n", style(" ").on_blue(), rel_path));
    out.push_str(&format!("{}+++ b/{}\n", style(" ").on_green(), rel_path));

    for block in diff.grouped_ops(3) {
        // Compute a rough hunk header based on new ranges without moving `block`
        let (mut min, mut max) = (usize::MAX, 0usize);
        for op in &block {
            min = min.min(op.new_range().start);
            max = max.max(op.new_range().end);
        }
        let len = max.saturating_sub(min);
        out.push_str(&format!("@@ -{},{} +{},{} @@{}\n", min, len, min, len, style(" ").on_magenta()));

        // Iterate again by reference so we don't move `block`
        for op in &block {
            for change in diff.iter_inline_changes(op) {
                let sign = match change.tag() {
                    ChangeTag::Delete => "-",
                    ChangeTag::Insert => "+",
                    ChangeTag::Equal => " ",
                };

                let mut line = String::new();
                line.push_str(sign);

                // `iter_strings_lossy()` yields (emphasized, Cow<str>) pieces.
                // When not emphasized we must push &str, so use `.as_ref()`.
                for (emph, value) in change.iter_strings_lossy() {
                    match change.tag() {
                        ChangeTag::Delete => {
                            if emph {
                                line.push_str(&style(value).on_red().bold().to_string());
                            } else {
                                line.push_str(&style(value).on_red().to_string());
                            }
                        },
                        ChangeTag::Insert => {
                            if emph {
                                line.push_str(&style(value).on_green().bold().to_string());
                            } else {
                                line.push_str(&style(value).on_green().to_string());
                            }
                        },
                        ChangeTag::Equal => {
                            if emph {
                                line.push_str(&style(value).bold().to_string());
                            } else {
                                line.push_str(value.as_ref());
                            }
                        }
                    }
                }

                // Ensure each rendered change is its own line.
                if !line.ends_with('\n') {
                    line.push('\n');
                }
                out.push_str(&line);
            }
        }
    }

    out
}
