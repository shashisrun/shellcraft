# Rust CLI Coding Agent

Tiny terminal agent that proposes **full-file edits** with an OpenAI-compatible LLM.
Review a colored diff, tweak in `$EDITOR`, then save.

## Setup
```bash
cargo build
export OPENAI_API_KEY="sk-..."                          # required
# export OPENAI_BASE_URL="https://api.openai.com/v1"     # optional (default)
# export MODEL_ID="gpt-4o-mini"                           # optional
