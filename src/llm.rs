use anyhow::{anyhow, Context, Result};
use once_cell::sync::Lazy;
use reqwest::Client;
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use serde_json::{self, json};

use crate::models::{ModelInfo, ModelRegistry};

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EditReq {
    pub file_path: String,
    pub file_content: String,
    pub instruction: String,
}

static HTTP: Lazy<Client> = Lazy::new(|| {
    Client::builder()
        .user_agent("shellcraft/1.0")
        .build()
        .expect("reqwest client")
});

static MODEL_REGISTRY: Lazy<ModelRegistry> = Lazy::new(ModelRegistry::load);

fn pick_provider(model_override: Option<&str>) -> Result<(String, String, String)> {
    let registry = &*MODEL_REGISTRY;
    let model_id = model_override
        .map(|s| s.to_string())
        .or_else(|| std::env::var("MODEL_ID").ok())
        .unwrap_or_else(|| registry.default_model.clone());

    if let Some(ModelInfo {
        provider,
        api_key_env,
        ..
    }) = registry.get(&model_id).cloned()
    {
        let key = std::env::var(&api_key_env).map_err(|_| anyhow!("{} not set", api_key_env))?;
        let base = match provider.as_str() {
            "openai" => std::env::var("OPENAI_BASE_URL")
                .unwrap_or_else(|_| "https://api.openai.com/v1".to_string()),
            "groq" => std::env::var("GROQ_BASE_URL")
                .unwrap_or_else(|_| "https://api.groq.com/openai/v1".to_string()),
            other => std::env::var(format!("{}_BASE_URL", other.to_uppercase()))
                .unwrap_or_else(|_| String::new()),
        };
        return Ok((key, base, model_id));
    }

    if let Ok(key) = std::env::var("GROQ_API_KEY") {
        let base = std::env::var("OPENAI_BASE_URL")
            .or_else(|_| std::env::var("GROQ_BASE_URL"))
            .unwrap_or_else(|_| "https://api.groq.com/openai/v1".to_string());
        let model =
            std::env::var("MODEL_ID").unwrap_or_else(|_| "llama-3.3-70b-versatile".to_string());
        return Ok((key, base, model));
    }
    if let Ok(key) = std::env::var("OPENAI_API_KEY") {
        let base = std::env::var("OPENAI_BASE_URL")
            .unwrap_or_else(|_| "https://api.openai.com/v1".to_string());
        let model = std::env::var("MODEL_ID").unwrap_or_else(|_| "gpt-4o-mini".to_string());
        return Ok((key, base, model));
    }
    Err(anyhow!(
        "API_KEY not set. Set OPENAI_API_KEY or GROQ_API_KEY (and optional MODEL_ID / *_BASE_URL).",
    ))
}

#[derive(Serialize)]
struct ChatRequest<'a> {
    model: &'a str,
    messages: Vec<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    response_format: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
}

#[derive(Deserialize)]
struct ChatResponse {
    choices: Vec<Choice>,
}
#[derive(Deserialize)]
struct Choice {
    message: Message,
}
#[derive(Deserialize)]
struct Message {
    content: String,
}

pub async fn chat_text(system: &str, user: &str) -> Result<String> {
    let (key, base, model) = pick_provider(None)?;
    let url = format!("{}/chat/completions", base.trim_end_matches('/'));
    let req = ChatRequest {
        model: &model,
        messages: vec![
            json!({"role":"system","content":system}),
            json!({"role":"user","content":user}),
        ],
        response_format: None,
        temperature: Some(0.2),
    };
    let res = HTTP
        .post(&url)
        .bearer_auth(&key)
        .json(&req)
        .send()
        .await
        .context("LLM HTTP error")?;
    let status = res.status();
    let body = res.text().await.unwrap_or_default();
    if !status.is_success() {
        return Err(anyhow!("LLM error {status}: {body}"));
    }
    let parsed: ChatResponse = serde_json::from_str(&body).context("parse LLM response")?;
    let content = parsed
        .choices
        .get(0)
        .map(|c| c.message.content.clone())
        .unwrap_or_default();
    Ok(content)
}

pub async fn chat_json<T: DeserializeOwned>(system: &str, user_json: &str) -> Result<T> {
    let (key, base, model) = pick_provider(None)?;
    let url = format!("{}/chat/completions", base.trim_end_matches('/'));

    let req = ChatRequest {
        model: &model,
        messages: vec![
            json!({"role":"system","content":system}),
            json!({"role":"user","content":user_json}),
        ],
        response_format: Some(json!({"type":"json_object"})),
        temperature: Some(0.0),
    };

    let res = HTTP
        .post(&url)
        .bearer_auth(&key)
        .json(&req)
        .send()
        .await
        .context("LLM HTTP error")?;
    let status = res.status();
    let body = res.text().await.unwrap_or_default();
    if !status.is_success() {
        return Err(anyhow!("LLM error {status}: {body}"));
    }
    let parsed: ChatResponse = serde_json::from_str(&body).context("parse LLM response")?;
    let content = parsed
        .choices
        .get(0)
        .map(|c| c.message.content.clone())
        .unwrap_or_else(|| "{}".into());

    serde_json::from_str::<T>(&content)
        .or_else(|_| Err(anyhow!("LLM did not return valid JSON: {}", content)))
}

pub async fn propose_edit(req: EditReq) -> Result<String> {
    let system = r#"You are a code editor. Given a file path, the current full file, and an instruction, return the **entire new file content**. Do not add code fences or commentary. Output only the file content."#;
    let user = format!(
        "PATH: {}\n--- CURRENT FILE START ---\n{}\n--- CURRENT FILE END ---\nINSTRUCTION:\n{}\n",
        req.file_path, req.file_content, req.instruction
    );
    let content = chat_text(system, &user).await?;
    Ok(strip_code_fences(&content).to_string())
}

pub async fn propose_patch(log_tail: &str, _diff_hint: &str) -> Result<String> {
    let system = r#"You are a code fixer. The user will give you an error log snippet. Produce a minimal unified diff patch (git-style) that fixes the error. No explanations or fences, just the patch text."#;
    let user = format!("--- ERROR LOG (tail) ---\n{}\n", log_tail);
    let content = chat_text(system, &user).await?;
    Ok(strip_code_fences(&content).to_string())
}

fn strip_code_fences(s: &str) -> &str {
    let t = s.trim();
    if t.starts_with("```") {
        if let Some(pos) = t.find('\n') {
            let rest = &t[pos + 1..];
            if let Some(end) = rest.rfind("```") {
                return &rest[..end];
            }
        }
    }
    t
}

pub async fn robust_chat_text(system: &str, user: &str) -> Result<String> {
    let (key, base, model) = pick_provider(None)?;
    let url = format!("{}/chat/completions", base.trim_end_matches('/'));
    let req = ChatRequest {
        model: &model,
        messages: vec![
            json!({"role":"system","content":system}),
            json!({"role":"user","content":user}),
        ],
        response_format: None,
        temperature: Some(0.2),
    };
    let mut res = HTTP
        .post(&url)
        .bearer_auth(&key)
        .json(&req)
        .send()
        .await
        .context("LLM HTTP error")?;
    let mut status = res.status();
    let mut body = res.text().await.unwrap_or_default();
    if !status.is_success() {
        return Err(anyhow!("LLM error {status}: {body}"));
    }
    let mut parsed: ChatResponse = serde_json::from_str(&body).context("parse LLM response")?;
    let mut content = parsed
        .choices
        .get(0)
        .map(|c| c.message.content.clone())
        .unwrap_or_default();
    if content.contains("README") {
        let new_user = format!("{} Please do not return README file.", user);
        res = HTTP
            .post(&url)
            .bearer_auth(&key)
            .json(&ChatRequest {
                model: &model,
                messages: vec![
                    json!({"role":"system","content":system}),
                    json!({"role":"user","content":new_user}),
                ],
                response_format: None,
                temperature: Some(0.2),
            })
            .send()
            .await
            .context("LLM HTTP error")?;
        status = res.status();
        body = res.text().await.unwrap_or_default();
        if !status.is_success() {
            return Err(anyhow!("LLM error {status}: {body}"));
        }
        parsed = serde_json::from_str(&body).context("parse LLM response")?;
        content = parsed
            .choices
            .get(0)
            .map(|c| c.message.content.clone())
            .unwrap_or_default();
    }
    Ok(content)
}

pub async fn propose_edit_robust(req: EditReq) -> Result<String> {
    let system = r#"You are a code editor. Given a file path, the current full file, and an instruction, return the **entire new file content**. Do not add code fences or commentary. Output only the file content."#;
    let user = format!(
        "PATH: {}\n--- CURRENT FILE START ---\n{}\n--- CURRENT FILE END ---\nINSTRUCTION:\n{}\n",
        req.file_path, req.file_content, req.instruction
    );
    let content = robust_chat_text(system, &user).await?;
    Ok(strip_code_fences(&content).to_string())
}
