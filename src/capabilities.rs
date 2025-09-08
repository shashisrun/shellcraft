use serde::{Deserialize, Serialize};
use std::path::Path;
use which::which;

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Providers {
    pub openai: bool,
    pub groq: bool,
    pub local: bool,
    pub anthropic: bool,
    pub model: String,
    pub base_url: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Tools {
    pub fs: bool,
    pub cargo: bool,
    pub npm: bool,
    pub bun: bool,
    pub pnpm: bool,
    pub yarn: bool,
    pub pytest: bool,
    pub go: bool,
    pub mvn: bool,
    pub git: bool,
    pub github: bool,
    pub rg: bool,
    pub grep: bool,
    pub prettier: bool,
    pub eslint: bool,
    pub rustfmt: bool,
    pub clippy: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Manifest {
    pub providers: Providers,
    pub tools: Tools,
}

pub fn build_manifest(_root: &Path) -> Manifest {
    let openai = std::env::var("OPENAI_API_KEY").is_ok();
    let groq = std::env::var("GROQ_API_KEY").is_ok();
    let anthropic = std::env::var("ANTHROPIC_API_KEY").is_ok();
    let local = std::env::var("LOCAL_MODEL").is_ok();

    let base_url = std::env::var("OPENAI_BASE_URL")
        .or_else(|_| std::env::var("GROQ_BASE_URL"))
        .unwrap_or_else(|_| {
            if groq {
                "https://api.groq.com/openai/v1".to_string()
            } else {
                "https://api.openai.com/v1".to_string()
            }
        });

    let default_model = if groq {
        "llama-3.3-70b-versatile"
    } else {
        "gpt-4o-mini"
    };
    let model = std::env::var("MODEL_ID").unwrap_or_else(|_| default_model.to_string());

    Manifest {
        providers: Providers {
            openai,
            groq,
            local,
            anthropic,
            model,
            base_url,
        },
        tools: Tools {
            fs: true,
            cargo: which("cargo").is_ok(),
            npm: which("npm").is_ok(),
            pnpm: which("pnpm").is_ok(),
            yarn: which("yarn").is_ok(),
            pytest: which("pytest").is_ok(),
            go: which("go").is_ok(),
            mvn: which("mvn").is_ok(),
            git: which("git").is_ok(),
            github: which("gh").is_ok(),
            rg: which("rg").is_ok(),
            grep: which("grep").is_ok(),
            prettier: which("prettier").is_ok(),
            eslint: which("eslint").is_ok(),
            rustfmt: which("rustfmt").is_ok(),
            clippy: which("cargo-clippy").is_ok(),
            bun: which("bun").is_ok(),
        },
    }
}

/// Can we run this program? Returns (ok, why_not).
pub fn can_run(manifest: &Manifest, program: &str) -> (bool, Option<String>) {
    let t = &manifest.tools;
    let ok = match program {
        "cargo" => t.cargo,
        "npm" => t.npm,
        "bun" => t.bun,
        "pnpm" => t.pnpm,
        "yarn" => t.yarn,
        "pytest" => t.pytest,
        "go" => t.go,
        "mvn" => t.mvn,
        "git" => t.git,
        "gh" | "github" => t.github,
        "rg" => t.rg,
        "grep" => t.grep,
        "prettier" => t.prettier,
        "eslint" => t.eslint,
        "rustfmt" => t.rustfmt,
        "cargo-clippy" | "clippy" => t.clippy,
        other => which(other).is_ok(),
    };
    if ok {
        (true, None)
    } else {
        (false, Some(format!("binary `{}` not on PATH", program)))
    }
}

/// A short text the planner sees as capabilities preamble.
pub fn system_preamble(manifest: &Manifest) -> String {
    let t = &manifest.tools;
    let mut lines: Vec<String> = vec![
        "A file index listing project files is provided for a birds-eye view.".into(),
        "Use the `fs` capability for file operations:".into(),
        "- add paths to `read` to view file contents".into(),
        "- provide {path,intent} entries in `edit` to modify files".into(),
        "- list paths in `delete` to remove them".into(),
        "".into(),
        "You can also request actions to run other tools.\nEnabled tools:".into(),
    ];
    let mut add = |name: &str, ok: bool| {
        if ok {
            lines.push(format!("- {}", name));
        }
    };
    add("fs", t.fs);
    add("cargo", t.cargo);
    add("npm", t.npm);
    add("bun", t.bun);
    add("pnpm", t.pnpm);
    add("yarn", t.yarn);
    add("pytest", t.pytest);
    add("go", t.go);
    add("mvn", t.mvn);
    add("git", t.git);
    add("github", t.github);
    add("rg", t.rg);
    add("grep", t.grep);
    add("prettier", t.prettier);
    add("eslint", t.eslint);
    add("rustfmt", t.rustfmt);
    add("clippy", t.clippy);

    lines.push(format!(
        "\nLLM provider base_url = {}, model = {}",
        manifest.providers.base_url, manifest.providers.model
    ));
    lines.join("\n")
}
