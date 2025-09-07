use std::path::Path;

use anyhow::Result;

use crate::capabilities::Manifest;
use crate::planner::{self, Plan};

/// Trait for all agents in the system.
pub trait Agent {
    fn name(&self) -> &str;
}

/// Planner agent chats with the user and produces a plan.
pub struct PlannerAgent {
    pub model: String,
}

impl PlannerAgent {
    pub fn new(model: String) -> Self {
        Self { model }
    }

    pub fn default() -> Self {
        let model = std::env::var("MODEL_ID").unwrap_or_default();
        Self { model }
    }

    pub async fn chat_and_plan(
        &self,
        root: &Path,
        user: &str,
        manifest: &Manifest,
    ) -> Result<Plan> {
        planner::plan_changes(root, user, manifest).await
    }
}

impl Agent for PlannerAgent {
    fn name(&self) -> &str {
        "planner"
    }
}

/// Worker agent placeholder.
pub struct WorkerAgent {
    pub model: String,
    pub tools: bool,
}

impl WorkerAgent {
    pub fn new(model: String, tools: bool) -> Self {
        Self { model, tools }
    }
}

impl Agent for WorkerAgent {
    fn name(&self) -> &str {
        "worker"
    }
}
