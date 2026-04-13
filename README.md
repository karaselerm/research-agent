# Research Agent

Purple A2A agent for AgentBeats MLE-bench runs.

## Project layout

```text
src/
  server.py                  # A2A server bootstrap and agent card
  executor.py                # Task execution lifecycle
  agent.py                   # Tabular ML pipeline + orchestration
  messenger.py               # A2A messaging helpers
tests/
  conftest.py
  test_agent.py
```

## RouterAI defaults

The agent is configured for RouterAI with model `openai/gpt-5.4-mini`.

Environment variables:

- `ROUTERAI_API_KEY`
- `ROUTERAI_BASE_URL` (default: `https://routerai.ru/api/v1`)
- `ROUTERAI_MODEL` (default: `openai/gpt-5.4-mini`)
- `AGENT_MODE` (`safe` | `fast` | `standard` | `heavy`, default: `standard`)
- `BINARY_STRATEGY` (`auto` | `stable` | `aggressive`, default: `auto`)

## Local run

```bash
uv sync
uv run src/server.py
```

Agent URL: `http://127.0.0.1:9009`

## Test

```bash
uv sync --extra test
uv run pytest --agent-url http://localhost:9009
```

Unit-only fast check:

```bash
uv run pytest tests/test_agent.py -k "submission_artifact or fallback_submission or run_submits_baseline" -q
```

## Docker

```bash
docker build -t research-agent .
docker run -p 9009:9009 research-agent
```
