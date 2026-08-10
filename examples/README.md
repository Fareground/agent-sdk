# Examples

Five runnable apps, ordered from smallest to a full web application. All run
without a database (in-memory or SQLite) — nothing to set up beyond a model.

| Example | What it shows |
|---|---|
| [`00_quickstart.py`](00_quickstart.py) | The `ask()` one-shot (model auto-detected from your env) and the `Agent` facade for conversations. Start here. |
| [`01_hello_agent.py`](01_hello_agent.py) | The same agent built from the four low-level objects (`AgentLLM`, `create_repository`, `ToolRegistry`, `AgentEngine`), streaming events. |
| [`02_custom_tools.py`](02_custom_tools.py) | Tool patterns: sync vs async, typed parameters, `ExecutionContext` injection, `register_many()`. |
| [`03_multi_agent.py`](03_multi_agent.py) | Orchestrator delegating to specialized sub-agents, with `subagent.started` / `subagent.completed` events. |
| [`04_web_app.py`](04_web_app.py) | Full FastAPI app via `create_app` with a browser chat UI streaming over SSE. Open http://localhost:8000. |
| [`frontend/`](frontend) | SSE clients for your own UI: `sse-client.js` (vanilla JS) and `use-agent.ts` (React hook). |

## Pick a model

Every example takes a provider-qualified model id. Two ways to run them:

**Hosted provider (fastest to set up).** Install the provider extra, export the
key, and set the model:

```bash
pip install "fg-agents[anthropic]"        # or [openai]
export ANTHROPIC_API_KEY="sk-ant-..."     # or OPENAI_API_KEY
```

Then use `model="anthropic:claude-sonnet-4-6"` (or `"openai:gpt-5.2"`, etc.).
Example 00 needs no `model=` at all — it auto-detects from your env vars
(Anthropic → OpenAI → Google → local Ollama); for the others, change the
`model=` line.

**Ollama (free, local, no API key).** Examples 01–04 default to
`ollama:qwen3:8b`:

```bash
ollama pull qwen3:8b
ollama serve        # if not already running
```

## Run

```bash
python examples/00_quickstart.py
python examples/01_hello_agent.py
python examples/02_custom_tools.py
python examples/03_multi_agent.py
python examples/04_web_app.py     # then open http://localhost:8000
```

More detail on what the events mean: [docs/streaming.md](../docs/streaming.md).
Full API reference: [docs/api.md](../docs/api.md).
