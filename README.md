<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/wordmark-dark.svg" />
    <img src="assets/wordmark.svg" alt="Fareground" width="320" />
  </picture>
</p>

# agent-framework

<p align="center">
  <em>An agent framework built for web applications — not laptops.</em>
</p>

<p align="center">
  <a href="https://github.com/Fareground/agent-framework/actions/workflows/ci.yml"><img alt="CI" src="https://img.shields.io/github/actions/workflow/status/Fareground/agent-framework/ci.yml?branch=main&style=flat-square&label=CI" /></a>
  <img src="https://img.shields.io/badge/python-3.11%2B-blue" alt="Python 3.11+" />
  <img src="https://img.shields.io/badge/status-alpha-orange" alt="Alpha" />
  <img src="https://img.shields.io/badge/license-Apache--2.0-lightgrey" alt="Apache-2.0" />
</p>

---

## Overview

Almost every agent framework — Claude Agent SDK, LangChain agents, AutoGen,
CrewAI — was built to run on your local machine. They read and write files on a
laptop, expect a terminal, and assume a single user. They're excellent for
personal automation and coding assistants, but none of that translates the
moment you want an agent **inside a real web product**: there's no filesystem,
there are many users each with their own data, replies must stream to a browser,
and sessions must survive a server restart.

This framework is designed from day one for that world. Drop it into a
FastAPI/Starlette app and you get a working AI agent backend — SSE streaming,
async persistence (PostgreSQL, SQLite, in-memory), multi-tenant isolation, and
multi-agent orchestration — without writing the plumbing yourself.

It is the runtime layer in Fareground's family of open agent building blocks,
alongside [`agent-id`](https://github.com/Fareground/agent-id) (identity),
[`agent-messaging`](https://github.com/Fareground/agent-messaging) (encrypted
agent-to-agent messaging), [`agent-memory`](https://github.com/Fareground/agent-memory)
(per-agent memory), and [`agent-knowledge`](https://github.com/Fareground/agent-knowledge)
(shared team knowledge).

What it deliberately does **not** do:

- **Authentication** — your app verifies users. You hand the router an
  `authenticator` that turns each request into a tenant + user, and the
  framework enforces it: a caller only ever reaches its own tenant's sessions
  and memory, and the tenant comes from the authenticator, never the request
  body. (Omit it and the API runs as a single shared tenant for local/dev — it
  warns loudly so you never ship that by accident.)
- **The model** — it talks to OpenAI, Anthropic, Google, and local models via
  Ollama. You bring the API key; pick any model at runtime, or let the
  framework detect a sensible default from your environment.
- **The frontend** — you build the chat UI; the framework streams events to it.

> **Package names.** The distribution is published as **`fg-agents`** and the
> import package is **`fg_agents`** (e.g. `from fg_agents import create_app`).
> These are the names dependents rely on and are intentionally left unchanged;
> the repository name is `agent-framework`.

## Install

```bash
pip install git+https://github.com/Fareground/agent-framework.git                 # core (FastAPI + PostgreSQL)
pip install "fg-agents[sqlite] @ git+https://github.com/Fareground/agent-framework.git"     # + SQLite backend
pip install "fg-agents[openai] @ git+https://github.com/Fareground/agent-framework.git"     # + OpenAI provider
pip install "fg-agents[anthropic] @ git+https://github.com/Fareground/agent-framework.git"  # + Anthropic provider
pip install "fg-agents[all] @ git+https://github.com/Fareground/agent-framework.git"        # everything (all providers + scheduler)
```

Requires Python 3.11+.

## Quickstart

With one API key env var set (`export ANTHROPIC_API_KEY=...`), this is the
whole program:

```python
from fg_agents import ask

print(await ask("What's 2+2?"))
```

`ask()` detects the provider from your environment — `ANTHROPIC_API_KEY`,
then `OPENAI_API_KEY`, then `GOOGLE_API_KEY`/`GEMINI_API_KEY`, then a local
[Ollama](https://ollama.com) server — and picks a current model for it. Pass
`model="provider:model"` to override. It takes tools, a system prompt, and a
memory backend too, and `stream()` is its streaming twin:

```python
from fg_agents import ask, stream

answer = await ask("Greet Ada.", tools=[greet], system_prompt="Be friendly.")

async for event in stream("Tell me a story"):
    print(event.type, event.data)
```

### Conversations: the `Agent` facade

`ask()` is one-shot. For multi-turn conversations, the `Agent` facade is one
object that wires the LLM client, tool registry, persistence, and engine —
usable as an async context manager. Tools are plain functions; the schema is
derived from the signature and docstring:

```python
def greet(name: str) -> str:
    """Greet someone by name."""
    return f"Hello, {name}!"

async with Agent(
    tools=[greet],                    # model= optional — auto-detected
    system_prompt="You are a friendly greeter.",
    memory="sqlite",   # persist sessions; "memory" (default) or "postgres:<url>"
) as agent:
    first = await agent.run("Please greet Ada.")
    second = await agent.run("Now greet Grace.")  # same conversation
```

`run()` returns an `AgentRunResult` (`.text`, `.session_id`, `.events` —
`str()` is the answer text), and `agent.stream()` yields raw `StreamEvent`s
as they happen. Any failure in `run()` — an engine error event or a raised exception — surfaces
as a single `AgentRunError` carrying the `session_id` and the partial `events`,
with the original exception chained as `__cause__`. Each `Agent` instance keeps
one conversation by default; pass `session_id=` per call to target another.

### Going lower level

The facade is sugar over four objects you can wire yourself when you need full
control (custom middleware, skills, shared registries across agents). Graduate
gradually via `agent.engine`, `agent.llm`, `agent.tools`, and
`agent.repository` — or build the stack directly:

```python
import asyncio
from fg_agents import AgentDefinition, AgentEngine, AgentLLM, ToolRegistry, create_repository, tool


@tool(description="Greet someone by name")
def greet(name: str) -> str:
    return f"Hello, {name}!"


async def main():
    llm = AgentLLM()
    repo = create_repository("memory")   # or "sqlite", "postgres"
    await repo.initialize()             # explicit here; the facade does this lazily

    tools = ToolRegistry()
    tools.register_function(greet)

    agent = AgentDefinition(
        name="greeter",
        model="anthropic:claude-sonnet-4-6",  # required — no default model
        system_prompt="You are a friendly greeter. Use the greet tool when asked.",
        tools=["greet"],
    )

    engine = AgentEngine(llm=llm, tool_registry=tools, repository=repo)
    async for event in engine.run("session-1", "Say hello to Alice", agent):
        if event.data.get("text"):
            print(event.data["text"], end="", flush=True)


asyncio.run(main())
```

## A full web backend

`create_app` returns a FastAPI application exposing routes to start sessions,
post messages, and stream replies over SSE.

```python
from fg_agents import AgentDefinition, ToolRegistry, create_app, tool


@tool(description="Search the knowledge base")
async def search(query: str) -> str:
    return f"Results for '{query}'..."


assistant = AgentDefinition(
    name="assistant",
    model="anthropic:claude-sonnet-4-6",
    system_prompt="You are a helpful assistant. Use your tools to answer.",
    tools=["search"],
)

tools = ToolRegistry()
tools.register_function(search)

app = create_app(
    agents={"assistant": assistant},
    tool_registry=tools,
    # db_url="postgresql+asyncpg://user:pass@localhost:5432/mydb",  # production
)
# uvicorn myapp:app
```

The frontend then creates a session and streams messages:

```
POST /api/agent/sessions                       -> { "session_id": "..." }
POST /api/agent/sessions/{id}/messages         -> text/event-stream (SSE)
```

See [`examples/`](examples/) for five runnable apps, from a two-line facade
hello-world to a complete chat UI ([`examples/04_web_app.py`](examples/04_web_app.py)),
plus frontend SSE clients in [`examples/frontend/`](examples/frontend).

## Documentation

- [`docs/api.md`](docs/api.md) — reference for the public API surface
- [`docs/streaming.md`](docs/streaming.md) — the SSE wire format and every event type (for frontend authors)
- [`docs/persistence.md`](docs/persistence.md) — memory / SQLite / PostgreSQL backends and initialization semantics

## Concepts / Building blocks

| Building block | What it is |
|---|---|
| **`Agent`** | The facade — one object that wires LLM, tools, persistence, and engine; `run()` returns an `AgentRunResult`, failures raise `AgentRunError`. |
| **`AgentDefinition`** | Declarative agent config — name, model, system prompt, tools, sub-agents. No default model; you choose the provider at runtime. |
| **`AgentEngine`** | The ReAct loop: model → tools → model → …, emitting stream events at each step. |
| **`Orchestrator`** | A coordinator agent that plans, delegates to sub-agents (in parallel), and synthesizes results. |
| **Tools** | Python functions exposed to the agent via the `@tool` decorator and a `ToolRegistry`; built-in tools ship in `tools/builtin.py`. |
| **Skills** | Reusable bundles of prompt templates + tools that give an agent domain knowledge (`SkillsManager`). |
| **Memory** | `WorkingMemory` per-session scratch space and a `ContextManager` with health scoring and progressive compression. |
| **Middleware** | Hooks into the agent loop: audit, token tracking, permissions, rate limiting, loop guarding, context compression. |
| **Persistence** | Async repositories behind one interface — `InMemoryRepository`, `SQLiteRepository`, `PostgresRepository` via `create_repository(...)`. |
| **Streaming** | `StreamEvent` objects serialized to SSE (`event.to_sse()`) with reconnection support. |
| **Scheduler** | Run an agent on a cron schedule (`AgentScheduler`, optional `croniter` dependency). |
| **API layer** | `create_app` / `create_agent_router` — FastAPI integration with pluggable authentication and per-tenant isolation. |

```mermaid
flowchart TB
    User([User in browser])
    subgraph Framework["Agent"]
        direction TB
        Skills["Skills<br/><sub>knowledge & playbooks</sub>"]
        Tools["Tools<br/><sub>actions the agent can take</sub>"]
        Subs["Sub-agents<br/><sub>specialists to delegate to</sub>"]
    end
    DB[("Your database<br/><sub>sessions, history</sub>")]
    LLM([Any model<br/>OpenAI · Anthropic · local])

    User <-->|SSE stream| Framework
    Framework <--> Skills
    Framework <--> Tools
    Framework <--> Subs
    Framework <--> LLM
    Framework <--> DB
```

## Project structure

```
src/fg_agents/
    agent.py       Agent facade — one-object setup (quickstart tier)
    core/          Engine, LLM client, types, errors
    tools/         @tool decorator, registry, built-in tools, handlers
    orchestrator/  Multi-agent orchestration + sub-agent runner
    skills/        Skill loading and injection
    memory/        Working memory, context management, health scoring
    middleware/    Audit, token tracking, permissions, rate limit, loop guard
    persistence/   Multi-backend: in-memory, SQLite, PostgreSQL
    streaming/     SSE event definitions
    prompts/       System prompt construction
    scheduler/     Cron-scheduled agents
    api/           Service layer, FastAPI router, auth, schemas
    app.py         create_app() application factory
examples/          Five runnable apps + frontend SSE clients
docs/              Reference docs: API, streaming wire format, persistence
tests/             Test suite
```

## Contributing

Development setup, tests, linting, and commit conventions live in
[CONTRIBUTING.md](CONTRIBUTING.md). See the [changelog](CHANGELOG.md) for what's
changed.

---

<p align="center">
  <sub>Built by <a href="https://github.com/Fareground">Fareground</a>.</sub><br/>
  <sub>Licensed under <a href="LICENSE">Apache-2.0</a>.</sub>
</p>
