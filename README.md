# Shellcraft Multi-Agent Coding Assistant

Shellcraft is a tiny terminal agent focused on coding tasks. It uses a
chat-first, multi-agent architecture. A planner agent converses with you,
breaks requests into detailed steps, and can delegate work to additional
worker agents as needed.

## Model configuration

Available models live in `models.json`. Each entry specifies the provider
and an environment variable that holds the API key. A default model is used
when `MODEL_ID` is not set. If a model supports tool calling, it can list the
available tools in a `tools` array.

Example `models.json`:
```json
{
  "default_model": "gpt-4o-mini",
  "models": [
    {
      "id": "gpt-4o-mini",
      "provider": "openai",
      "api_key_env": "OPENAI_API_KEY",
      "tools": ["code_interpreter"],
      "specialty": "general coding"
    }
  ]
}
```

## Setup

```bash
cargo build
export OPENAI_API_KEY="sk-..."      # required for OpenAI models
# export GROQ_API_KEY="sk-..."       # optional for Groq models
# export MODEL_ID="gpt-4o-mini"      # override default model
```

Run the CLI and start chatting:

```bash
cargo run
```
