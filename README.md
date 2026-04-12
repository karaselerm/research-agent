# Research Agent

A purple A2A-compatible agent for the AgentBeats Research track.

## What it does

This agent answers research-style and ML-engineering prompts with concise structured reasoning.

It supports:
- local execution without API keys via deterministic fallback logic
- OpenRouter-backed responses when `OPENROUTER_API_KEY` is provided
- A2A-compatible packaging for AgentBeats

## Local run

```bash
uv sync
uv run src/server.py
```

The agent will run on:
http://127.0.0.1:9009

Run tests
```
uv sync --extra test
uv run pytest --agent-url http://localhost:9009
Docker
docker build -t research-agent .
docker run -p 9009:9009 kartoshechko-research-agent
```
Optional environment variables
OPENROUTER_API_KEY
OPENROUTER_MODEL

If no API key is set, the agent uses local fallback logic so A2A tests still pass.


---

## 8) `.env.example`

```env
OPENROUTER_API_KEY=your_openrouter_key_here
OPENROUTER_MODEL=deepseek/deepseek-chat-v3-0324
