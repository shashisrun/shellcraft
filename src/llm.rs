use anyhow::{anyhow, Context, Result};
use once_cell::sync::Lazy;
use reqwest::{header, Client};
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use serde_json::{self, json};
use std::env;
use std::sync::RwLock;
use std::time::Duration;

/// -------- HTTP client --------
static HTTP: Lazy<Client> = Lazy::new(|| {
    Client::builder()
        .timeout(Duration::from_secs(120))
        .build()
        .expect("failed to build HTTP client")
});

/// -------- Model ID (runtime mutable) --------
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
    *MODEL_ID
        .write()
        .expect("MODEL_ID poisoned") = new_id.to_string();
}

/// -------- Endpoint & key --------
fn base_url() -> String {
    env::var("BASE_URL")
        .ok()
        .filter(|s| !s.trim().is_empty())
        .unwrap_or_else(|| "https://api.openai.com/v1".to_string())
}

fn api_key() -> Result<String> {
    env::var("API_KEY")
        .ok()
        .filter(|s| !s.trim().is_empty())
        .ok_or_else(|| anyhow!("API_KEY not set. Put it in your shell env or .agent.env"))
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

/// Low-level: call chat.completions and return raw assistant string.
pub async fn chat_text(system: &str, user: &str) -> Result<String> {
    let key = api_key()?;
    let url = format!("{}/chat/completions", base_url());
    let model = current_model_id();

    let req = ChatRequest {
        model: &model,
        messages: vec![
            ChatMessage { role: "system", content: system },
            ChatMessage { role: "user", content: user },
        ],
        temperature: Some(0.2),
        max_tokens: None,
    };

    let res = HTTP
        .post(&url)
        .header(header::AUTHORIZATION, format!("Bearer {}", key))
        .header(header::CONTENT_TYPE, "application/json")
        .json(&req)
        .send()
        .await
        .context("LLM HTTP request failed")?;

    let status = res.status();
    let text = res.text().await.unwrap_or_default();

    if !status.is_success() {
        return Err(anyhow!("LLM error {}: {}", status.as_u16(), text));
    }

    let parsed: ChatResponse =
        serde_json::from_str(&text).context("LLM JSON decode failed")?;
    let out = parsed
        .choices
        .get(0)
        .map(|c| c.message.content.clone())
        .unwrap_or_default();
    Ok(out)
}

/// Parse the assistant string as JSON into `T`, or salvage if wrapped.
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
                .or_else(|_| serde_json::from_value::<T>(json!({"notes": s})))
                .context("LLM returned non-JSON for chat_json")
        }
    }
}

/// Ask the model to propose a new file content for a single file.
pub async fn propose_edit(req: EditReq) -> Result<String> {
    let system = r#"You are a careful code transformation engine.
Given a single file's current contents and an instruction, output ONLY the full new file content.
- Do not include prose, backticks, or diff markers.
- Preserve license headers and formatting where possible."#;

    let user = format!(
        "FILE PATH: {}\n\nINSTRUCTION:\n{}\n\nCURRENT CONTENTS:\n{}\n\n---\nReturn only the new full content of the file.",
        req.file_path, req.instruction, req.file_content
    );

    chat_text(system, &user).await
}
