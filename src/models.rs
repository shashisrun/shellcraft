use serde::Deserialize;
use std::fs;

#[derive(Debug, Deserialize, Clone)]
pub struct ModelInfo {
    pub id: String,
    pub provider: String,
    #[serde(default)]
    pub api_key_env: String,
    #[serde(default)]
    pub tools: Vec<String>,
    #[serde(default)]
    pub specialty: String,
}

#[derive(Debug, Deserialize, Clone)]
pub struct ModelRegistry {
    pub default_model: String,
    #[serde(default)]
    pub models: Vec<ModelInfo>,
}

impl ModelRegistry {
    pub fn load() -> Self {
        let path = std::env::var("MODEL_CONFIG").unwrap_or_else(|_| "models.json".into());
        let data = fs::read_to_string(&path).unwrap_or_else(|_| "{}".into());
        serde_json::from_str(&data).unwrap_or_else(|_| ModelRegistry {
            default_model: "gpt-4o-mini".into(),
            models: vec![],
        })
    }

    pub fn get(&self, id: &str) -> Option<&ModelInfo> {
        self.models.iter().find(|m| m.id == id)
    }
}
