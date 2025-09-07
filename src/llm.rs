use anyhow::{anyhow, Context, Result};
use once_cell::sync::{Lazy, OnceCell};
use reqwest::{header, Client};
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use serde_json::{self, json};
use std::collections::HashMap;
use std::env;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::RwLock;
use std::time::{Duration, Instant};
use tokio::time::sleep;

/// -------- HTTP client --------
static HTTP: Lazy<Client> = Lazy::new(|| {
    Client::builder()
        .timeout(Duration::from_secs(120))
        .build()
        .expect("failed to build HTTP client")
});

/// -------- Budgeting counters --------
static TOTAL_TOKENS: AtomicU64 = AtomicU64::new(0);
static TOTAL_TIME_MS: AtomicU64 = AtomicU64::new(0);

/// -------- Provider definition --------
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum Provider {
    OpenAI,
    Anthropic,
    Groq,
    Local,
}

impl Provider {
    fn env_prefix(&self) -> &'static str {
        match self {
            Provider::OpenAI => "OPENAI",
            Provider::Anthropic => "ANTHROPIC",
            Provider::Groq => "GROQ",
            Provider::Local => "LOCAL",
        }
    }
}

fn provider_from_str(s: &str) -> Option<Provider> {
    match s.to_ascii_lowercase().as_str() {
        "openai" => Some(Provider::OpenAI),
        "anthropic" => Some(Provider::Anthropic),
        "groq" => Some(Provider::Groq),
        "local" => Some(Provider::Local),
        _ => None,
    }
}

/// Configuration needed for a provider
#[derive(Debug, Clone, Deserialize, Serialize)]
struct ProviderConfig {
    base_url: String,
    api_key: String,
    model_id: String,
}

impl ProviderConfig {
    /// Validate that the configuration contains the minimal required data.
    fn validate(&self) -> Result<()> {
        if self.api_key.trim().is_empty() {
            Err(anyhow!("API key for provider is missing or empty"))
        } else {
            Ok(())
        }
    }
}

/// Per‑task configuration (primary provider + optional fallbacks)
#[derive(Debug, Clone, Deserialize, Serialize)]
struct TaskConfig {
    primary: String,
    fallbacks: Option<Vec<String>>,
}

/// Top‑level configuration loaded from `llm_config.toml`
#[derive(Debug, Clone, Deserialize, Serialize)]
struct Config {
    providers: HashMap<String, ProviderConfig>,
    tasks: HashMap<String, TaskConfig>,
}

/// Registry that holds all configured providers
struct ClientRegistry {
    providers: HashMap<Provider, ProviderConfig>,
}

impl ClientRegistry {
    fn load() -> Self {
        // Attempt to read the configuration file.
        // If it does not exist or is invalid, fall back to an empty configuration.
        let config_str = match std::fs::read_to_string("llm_config.toml") {
            Ok(s) => s,
            Err(e) => {
                eprintln!(
                    "Warning: Failed to read llm_config.toml ({}). Using default empty configuration.",
                    e
                );
                String::new()
            }
        };

        let cfg: Config = if config_str.is_empty() {
            Config {
                providers: HashMap::new(),
                tasks: HashMap::new(),
            }
        } else {
            match toml::from_str(&config_str) {
                Ok(c) => c,
                Err(e) => {
                    eprintln!(
                        "Warning: Invalid llm_config.toml ({}) . Using default empty configuration.",
                        e
                    );
                    Config {
                        providers: HashMap::new(),
                        tasks: HashMap::new(),
                    }
                }
            }
        };

        // Store task configuration for later lookup.
        // This will always succeed because we set it exactly once during startup.
        TASK_CONFIG
            .set(cfg.tasks.clone())
            .expect("Task configuration already set");

        // Build provider map from the (possibly empty) configuration.
        let mut providers = HashMap::new();
        for (name, pcfg) in cfg.providers {
            if let Some(p) = provider_from_str(&name) {
                if pcfg.validate().is_ok() {
                    providers.insert(p, pcfg);
                }
            }
        }

        ClientRegistry { providers }
    }

    fn get(&self, provider: Provider) -> Option<&ProviderConfig> {
        self.providers.get(&provider)
    }

    /// Return providers ordered by configuration for a given task.
    fn ordered_for(&self, task: TaskType) -> Vec<Provider> {
        let task_name = match task {
            TaskType::Code => "code",
            TaskType::Reasoning => "reasoning",
            TaskType::Summary => "summary",
        };

        let tasks = TASK_CONFIG.get().expect("Task config not initialized");
        if let Some(task_cfg) = tasks.get(task_name) {
            let mut list = Vec::new();

            if let Some(p) = provider_from_str(&task_cfg.primary) {
                list.push(p);
            }

            if let Some(fb) = &task_cfg.fallbacks {
                for name in fb {
                    if let Some(p) = provider_from_str(name) {
                        list.push(p);
                    }
                }
            }

            // Keep only providers we actually have config for
            list.into_iter()
                .filter(|p| self.providers.contains_key(p))
                .collect()
        } else {
            Vec::new()
        }
    }
}

/// Global registry (lazy loaded once)
static REGISTRY: Lazy<ClientRegistry> = Lazy::new(ClientRegistry::load);

/// Global task configuration
static TASK_CONFIG: OnceCell<HashMap<String, TaskConfig>> = OnceCell::new();

/// -------- Task classification --------
#[derive(Debug, Clone, Copy, PartialEq)]
enum TaskType {
    Code,
    Reasoning,
    Summary,
}

/// -------- Model ID (legacy fallback) --------
static MODEL_ID: Lazy<RwLock<String>> = Lazy::new(|| {
    let default = env::var("MODEL_ID")
        .ok()
        .filter(|s| !s.trim().is_empty())
        .unwrap_or_else(|| "gpt-4o-mini".to_string());
    RwLock::new(default)
});

fn current_model_id() -> String {
    MODEL_ID.read().expect("MODEL_ID poisoned").clone()
}

pub fn set_model_id(new_id: &str) {
    *MODEL_ID.write().expect("MODEL_ID poisoned") = new_id.to_string();
}

/// -------- Chat payloads --------
#[derive(Serialize)]
struct ChatRequest<'a> {
    model: &'a str,
    messages: Vec<ChatMessage<'a>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
}

#[derive(Serialize)]
struct ChatMessage<'a> {
    role: &'a str,
    content: &'a str,
}

#[derive(Deserialize)]
struct ChatResponse {
    choices: Vec<ChatChoice>,
}

#[derive(Deserialize)]
struct ChatChoice {
    message: ChatMsgOut,
}

#[derive(Deserialize)]
struct ChatMsgOut {
    content: String,
}

/// -------- Short‑term session memory --------
const MEMORY_CAPACITY: usize = 10;

#[derive(Clone)]
struct MemoryMessage {
    role: String,
    content: String,
}

static SESSION_MEMORY: Lazy<RwLock<Vec<MemoryMessage>>> = Lazy::new(|| RwLock::new(Vec::new()));

fn push_memory(role: &str, content: &str) {
    let mut mem = SESSION_MEMORY.write().unwrap();
    if mem.len() >= MEMORY_CAPACITY {
        mem.remove(0);
    }
    mem.push(MemoryMessage {
        role: role.to_string(),
        content: content.to_string(),
    });
}

fn recent_memory_formatted() -> String {
    let mem = SESSION_MEMORY.read().unwrap();
    if mem.is_empty() {
        return String::new();
    }
    let mut out = String::new();
    out.push_str("\n--- Recent Session Memory ---\n");
    for m in mem.iter() {
        out.push_str(&format!("{}: {}\n", m.role, m.content));
    }
    out
}

/// Public API for planner to retrieve recent context
pub fn get_recent_context() -> String {
    recent_memory_formatted()
}

/// Low‑level request that knows about a specific provider.
async fn provider_chat(
    cfg: &ProviderConfig,
    system: &str,
    user: &str,
    temperature: Option<f32>,
    max_tokens: Option<u32>,
) -> Result<String> {
    // Ensure the config is still valid at call time.
    cfg.validate()
        .with_context(|| format!("Invalid provider configuration for {}", cfg.base_url))?;

    let url = format!("{}/chat/completions", cfg.base_url);
    let req = ChatRequest {
        model: &cfg.model_id,
        messages: vec![
            ChatMessage {
                role: "system",
                content: system,
            },
            ChatMessage {
                role: "user",
                content: user,
            },
        ],
        temperature,
        max_tokens,
    };

    let start = Instant::now();

    let resp = HTTP
        .post(&url)
        .header(header::AUTHORIZATION, format!("Bearer {}", cfg.api_key))
        .header(header::CONTENT_TYPE, "application/json")
        .json(&req)
        .send()
        .await
        .context("LLM HTTP request failed")?;

    let elapsed = start.elapsed().as_millis() as u64;
    TOTAL_TIME_MS.fetch_add(elapsed, Ordering::Relaxed);

    let status = resp.status();
    let text = resp.text().await.unwrap_or_default();

    if !status.is_success() {
        // Estimate token usage from request size (rough)
        let est_tokens = (system.len() + user.len()) as u64;
        TOTAL_TOKENS.fetch_add(est_tokens, Ordering::Relaxed);
        return Err(anyhow!(
            "LLM error {} ({}): {}",
            status.as_u16(),
            status.canonical_reason().unwrap_or("unknown"),
            text
        ));
    }

    // Successful response – count tokens (very rough estimate)
    let est_tokens = (system.len() + user.len() + text.len()) as u64;
    TOTAL_TOKENS.fetch_add(est_tokens, Ordering::Relaxed);

    let parsed: ChatResponse = serde_json::from_str(&text).context("LLM JSON decode failed")?;
    let out = parsed
        .choices.first()
        .map(|c| c.message.content.clone())
        .unwrap_or_default();
    Ok(out)
}

/// Simple fallback summarizer used when all providers fail.
/// This implementation just returns the first 200 characters of the user input,
/// trimmed and suffixed with an ellipsis if truncated.
fn simple_fallback_summary(user: &str) -> String {
    const MAX_LEN: usize = 200;
    let trimmed = user.trim();
    if trimmed.len() <= MAX_LEN {
        trimmed.to_string()
    } else {
        format!("{}...", &trimmed[..MAX_LEN])
    }
}

/// Core routing logic with fallback and exponential backoff.
async fn routed_chat(
    task: TaskType,
    system: &str,
    user: &str,
    temperature: Option<f32>,
    max_tokens: Option<u32>,
) -> Result<String> {
    // Augment system prompt with recent memory
    let augmented_system = if recent_memory_formatted().is_empty() {
        system.to_string()
    } else {
        format!("{}{}", system, recent_memory_formatted())
    };

    let providers = REGISTRY.ordered_for(task);
    if providers.is_empty() {
        return Err(anyhow!("No providers configured for task"));
    }

    let mut error_log: Vec<String> = Vec::new();

    // Try each provider in order
    for provider in providers {
        let cfg = REGISTRY
            .get(provider)
            .expect("Provider should exist in ordered list");

        let mut attempt = 0usize;
        let max_attempts = 3usize;
        let mut backoff = Duration::from_millis(500);

        loop {
            attempt += 1;
            match provider_chat(cfg, &augmented_system, user, temperature, max_tokens).await {
                Ok(res) => {
                    // Store the exchange in short‑term memory
                    push_memory("user", user);
                    push_memory("assistant", &res);
                    return Ok(res);
                }
                Err(e) => {
                    error_log.push(format!(
                        "Provider {:?} attempt {}: {}",
                        provider, attempt, e
                    ));

                    // Look for rate‑limit (429) or auth (401) in the error message
                    let is_retryable = e.to_string().contains("429")
                        || e.to_string().contains("401")
                        || e.to_string().contains("rate limit")
                        || e.to_string().contains("unauthorized");

                    if is_retryable && attempt < max_attempts {
                        sleep(backoff).await;
                        backoff *= 2;
                        continue;
                    } else {
                        // Break out to try next provider
                        break;
                    }
                }
            }
        }
    }

    // All providers exhausted
    let combined_error = error_log.join("\n");
    let fallback_msg = if task == TaskType::Summary {
        // Use a lightweight local summarizer as a last resort.
        simple_fallback_summary(user)
    } else {
        String::new()
    };

    if !fallback_msg.is_empty() {
        Ok(fallback_msg)
    } else {
        Err(anyhow!(
            "All providers failed for the requested task.\nDetails:\n{}",
            combined_error
        ))
    }
}

/// Public API – chat returning plain text (defaults to reasoning‑optimized routing)
pub async fn chat_text(system: &str, user: &str) -> Result<String> {
    routed_chat(TaskType::Reasoning, system, user, Some(0.2), None).await
}

/// Public API – chat returning JSON‑deserialized value
pub async fn chat_json<T: DeserializeOwned>(system: &str, user: &str) -> Result<T> {
    let s = chat_text(system, user).await?;
    match serde_json::from_str::<T>(&s) {
        Ok(v) => Ok(v),
        Err(_) => {
            let trimmed = s
                .trim()
                .trim_start_matches("```json")
                .trim_start_matches("```")
                .trim_end_matches("```")
                .trim()
                .to_string();
            serde_json::from_str::<T>(&trimmed)
                .or_else(|_| serde_json::from_value::<T>(json!({ "notes": s })))
                .context("LLM returned non‑JSON for chat_json")
        }
    }
}

/// -------- Public types --------
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EditReq {
    pub file_path: String,
    pub file_content: String,
    pub instruction: String,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub max_tokens: Option<u32>,
}

/// Ask the model to propose a new file content for a single file (code‑capable routing)
pub async fn propose_edit(req: EditReq) -> Result<String> {
    let system = r#"You are a careful code transformation engine.
Given a single file's current contents and an instruction, output ONLY the full new file content.
- Do not include prose, backticks, or diff markers.
- Preserve license headers and formatting where possible."#;

    let user = format!(
        "FILE PATH: {}\n\nINSTRUCTION:\n{}\n\nCURRENT CONTENTS:\n{}\n\n---\nReturn only the new full content of the file.",
        req.file_path, req.instruction, req.file_content
    );

    routed_chat(
        TaskType::Code,
        system,
        &user,
        req.temperature.or(Some(0.2)),
        req.max_tokens,
    )
    .await
}

/// Stub implementation: propose a patch based on provided diff.
/// In a full implementation this would invoke the LLM to generate a patch,
/// but for now we simply return the diff unchanged.
pub fn propose_patch(_log_content: &str, diff: &str) -> Result<String> {
    Ok(diff.to_string())
}

/// -------- Budget report --------
pub fn budget_report() -> String {
    let tokens = TOTAL_TOKENS.load(Ordering::Relaxed);
    let time_ms = TOTAL_TIME_MS.load(Ordering::Relaxed);
    format!("Tokens used: {}, Time spent: {} ms", tokens, time_ms)
}
