# API reference

The headline public symbols, all importable from `fg_agents`. This covers the
surface most apps touch; the full export list is in `src/fg_agents/__init__.py`.

- [One-shot: `ask` / `stream`](#one-shot-ask--stream)
- [Model auto-detection: `resolve_default_model`](#model-auto-detection)
- [Agent (facade)](#agent-facade)
- [AgentRunResult / AgentRunError](#agentrunresult)
- [AgentDefinition](#agentdefinition)
- [AgentEngine](#agentengine)
- [AgentLLM](#agentllm)
- [Tools: `@tool` and ToolRegistry](#tools)
- [Persistence: `create_repository`](#persistence)
- [Web layer: `create_app`](#web-layer)
- [Streaming: StreamEvent / EventType](#streaming)
- [Middleware](#middleware)

---

## One-shot: `ask` / `stream`

Free functions for one-shot use — no client object. Each constructs an
ephemeral `Agent`, runs the prompt, and closes it.

```python
from fg_agents import ask

print(await ask("What's 2+2?"))
```

```python
async def ask(
    prompt: str,
    *,
    model: str | None = None,        # omit to auto-detect from the environment
    tools: list[Callable | RegisteredTool] | None = None,
    system_prompt: str = "",
    memory: str | BaseRepository = "memory",
    **kwargs,                        # anything else Agent accepts
) -> AgentRunResult
```

`str()` of the result is the answer text, so `print(await ask(...))` prints
just the answer.

`stream(prompt, ...)` takes the same arguments and yields `StreamEvent`s as
they happen; the ephemeral agent is closed when the stream ends:

```python
from fg_agents import stream

async for event in stream("Tell me a story"):
    print(event.type, event.data)
```

The ladder: `ask()` for one-shots → `Agent` for conversations →
`AgentEngine` and friends for full control.

---

## Model auto-detection

`resolve_default_model() -> str` picks a provider-qualified default model
from the environment. It runs whenever `ask()`, `stream()`, or `Agent()` is
called without a `model`. Detection order:

1. `ANTHROPIC_API_KEY` → `anthropic:claude-sonnet-4-6`
2. `OPENAI_API_KEY` → `openai:gpt-5.2`
3. `GOOGLE_API_KEY` or `GEMINI_API_KEY` → `google:gemini-2.5-flash`
4. A local Ollama server on `localhost:11434` → `ollama:qwen3:8b`

If none apply it raises `ModelDetectionError` (importable from `fg_agents`)
with instructions. An explicit `model=` always wins — detection never
overrides a caller's choice.

---

## Agent (facade)

One object that wires `AgentLLM` + `ToolRegistry` + a repository + `AgentEngine`.
The fastest way to run an agent; graduate to the low-level objects at any point.

```python
from fg_agents import Agent

agent = Agent(model="anthropic:claude-sonnet-4-6", tools=[my_func])
result = await agent.run("hello")
```

### Constructor

```python
Agent(
    model: str | None = None,                     # provider-qualified, e.g. "openai:gpt-5.2"; None = auto-detect
    *,
    tools: list[Callable | RegisteredTool] | None = None,
    system_prompt: str = "",
    name: str = "agent",
    memory: str | BaseRepository = "memory",      # "memory" | "sqlite" | "sqlite:<path>" | "postgres:<url>" | repository
    llm: AgentLLM | None = None,                  # pre-built client (e.g. with api_keys)
    api_keys: dict[str, str] | None = None,       # provider -> key; ignored when llm= is given
    session_id: str | None = None,                # default conversation id; fresh per instance otherwise
    **agent_kwargs,                               # extra AgentDefinition fields: max_turns, temperature, fallback_models, ...
)
```

`tools` accepts `@tool`-decorated functions, plain callables (auto-wrapped —
schema derived from the signature, description from the docstring), or
`RegisteredTool` instances. Duplicate tool names raise `ValueError`.

The repository is initialized lazily on the first `run()`/`stream()` call —
forgetting to initialize is never an error. (The low-level tier requires an
explicit `await repo.initialize()`.)

### Methods

| Method | Returns | Notes |
|---|---|---|
| `await run(message, *, session_id=None, metadata=None, variables=None)` | `AgentRunResult` | Runs to completion. Continues the instance's conversation unless `session_id` is passed. |
| `stream(message, *, session_id=None, metadata=None, variables=None)` | `AsyncIterator[StreamEvent]` | Yields raw events as they happen. |
| `await close()` | `None` | Closes the repository; the Agent is unusable afterwards. |

`Agent` is also an async context manager — `async with Agent(...) as agent:`
closes it on exit:

```python
async with Agent(tools=[greet]) as agent:
    result = await agent.run("hello")
```

### Failure contract

`run()` converts **any** failure — an engine-emitted `error` event or an
exception raised mid-run — into a single `AgentRunError` carrying the
`session_id` and the partial `events` collected so far; the original exception
(when there is one) is chained as `__cause__`.

`stream()` is lower-level by design: engine failures surface either as `error`
events in the stream or as raw exceptions from the iterator — handle both, or
use `run()` for the uniform contract.

### Graduation properties

`agent.engine`, `agent.llm`, `agent.tools` (the `ToolRegistry`),
`agent.repository`, `agent.definition`, `agent.session_id` — the exact objects
the facade built, for when you outgrow it.

---

## AgentRunResult

Frozen dataclass returned by `Agent.run`:

| Field | Type | Meaning |
|---|---|---|
| `text` | `str` | Final assistant output (`str(result)` returns it too). |
| `session_id` | `str` | Conversation the run belonged to. |
| `events` | `list[StreamEvent]` | Every event emitted during the run. |

## AgentRunError

`AgentRunError(message, session_id, events)` — subclass of
`AgentFrameworkError` (the root of the framework's exception hierarchy).
Attributes: `.session_id`, `.events` (partial events up to the failure),
`.__cause__` (original exception, if any).

---

## AgentDefinition

Declarative agent configuration (Pydantic model). Key fields:

```python
AgentDefinition(
    name="assistant",
    model="anthropic:claude-sonnet-4-6",   # required — no default model
    system_prompt="...",
    tools=["search", "greet"],             # names registered in the ToolRegistry
    # plus: max_turns, temperature, fallback_models, sub_agents, skills, ...
)
```

See `fg_agents.core.types.AgentDefinition` for the full field list.

---

## AgentEngine

The ReAct loop: model → tools → model → ..., emitting `StreamEvent`s at each
step. Runs are serialized per session (per-session lock) and support
cooperative cancellation.

```python
AgentEngine(
    llm: AgentLLM,
    tool_registry: ToolRegistry,
    repository: BaseRepository,
    skills_manager: SkillsManager | None = None,
    middleware: list[BaseMiddleware] | None = None,
)

async for event in engine.run(
    session_id: str,
    user_message: str,
    agent_def: AgentDefinition,
    metadata: dict | None = None,
    variables: dict | None = None,
):
    ...
```

`run()` creates or resumes the session, appends the user message, and runs the
loop until completion or `max_turns`. The repository must be initialized
(`await repo.initialize()`) before the first run.

---

## AgentLLM

Provider-agnostic LLM client with streaming and tool-use support. Native
Anthropic/OpenAI/Google clients plus 20+ OpenAI-compatible providers
(including `ollama:` for local models). Provider clients are created lazily.

```python
AgentLLM(
    api_keys: dict[str, str] | None = None,        # provider -> key; falls back to {PROVIDER}_API_KEY env vars
    custom_providers: dict[str, str] | None = None # provider name -> base URL, for providers not built in
)
```

Model ids are always provider-qualified: `"anthropic:claude-sonnet-4-6"`,
`"openai:gpt-5.2"`, `"ollama:qwen3:8b"`.

---

## Tools

### `@tool` decorator

```python
from fg_agents import tool

@tool(name="search_db", description="Search the case database")
async def search_db(query: str, limit: int = 10) -> dict:
    ...
```

```python
tool(
    name: str | None = None,              # default: function name
    description: str | None = None,       # default: first docstring line
    tool_type: ToolType = ToolType.FUNCTION,
    permission_level: str = "auto_approve",
    timeout_seconds: int = 30,
    retry_max: int = 0,
    tags: list[str] | None = None,
    json_schema: dict | None = None,      # explicit schema; otherwise derived from the signature
)
```

Sync functions are wrapped to run in a thread; async functions run natively.
The decorated function gains a `.tool_definition` attribute (a
`RegisteredTool`). A parameter named `ctx`/`context` typed `ExecutionContext`
is injected by the framework and hidden from the JSON schema.

### ToolRegistry

Central tool store handed to the engine or `create_app`.

| Method | Purpose |
|---|---|
| `register_function(fn)` | Register a `@tool`-decorated function. |
| `register(registered_tool)` | Register a `RegisteredTool` directly. |
| `register_many([...])` | Bulk registration. |
| `has(name)` / `get(name)` | Lookup. |
| `.tools` | `dict[str, RegisteredTool]` of everything registered. |

---

## Persistence

```python
from fg_agents import create_repository

repo = create_repository("memory")                                   # tests / throwaway
repo = create_repository("sqlite", db_path="agents.db")              # local dev (needs fg-agents[sqlite])
repo = create_repository("postgres", db_url="postgresql+asyncpg://…")# production
await repo.initialize()   # required before use (the Agent facade does this for you)
```

All backends implement `BaseRepository` (async session/message CRUD). Details
and dependency requirements: [persistence.md](persistence.md).

---

## Web layer

```python
from fg_agents import create_app

app = create_app(
    agents: dict[str, AgentDefinition],       # named agents exposed by the API
    db_url: str | None = None,                # postgres URL for production; SQLite file if None
    api_keys: dict[str, str] | None = None,
    tool_registry: ToolRegistry | None = None,
    cors_origins: list[str] | None = None,
    prefix: str = "/api/agent",
    title: str = "Fareground Agent API",
    admin_only: bool = True,                  # POST /tools and /agents return 403 when True
    **fastapi_kwargs,
) -> FastAPI
```

Returns a FastAPI app with sessions, messages, and SSE streaming wired;
persistence is initialized in the app lifespan. For mounting into an existing
app, use `create_agent_router(...)` with your own `authenticator` for
multi-tenant isolation (see `fg_agents/api/router.py`).

Routes (under `prefix`):

```
POST /sessions                    -> { "session_id": "..." }
POST /sessions/{id}/messages      -> text/event-stream (SSE)
```

---

## Streaming

`StreamEvent` (Pydantic model) is the single event shape flowing
Engine → iterator → SSE → frontend:

| Field | Type |
|---|---|
| `id` | `str` (uuid) |
| `type` | `EventType` (e.g. `llm.text_delta`) |
| `session_id` | `str` |
| `data` | `dict` (per-type payload) |
| `timestamp` | UTC datetime |
| `turn_number` | `int \| None` |

`event.to_sse()` renders the SSE wire frame; `event.to_dict()` a plain dict.
Every event type and its payload: [streaming.md](streaming.md).

---

## Middleware

Hooks into the agent loop, passed to `AgentEngine(middleware=[...])`. Implement
the `Middleware` protocol or subclass `BaseMiddleware` (pass-through defaults)
and override what you need:

| Hook | Signature (self omitted) | Can |
|---|---|---|
| `before_llm_call` | `(messages, tools, context) -> (messages, tools)` | Inject context, filter tools. |
| `after_llm_call` | `(response, context) -> response` | Inspect/modify the LLM response. |
| `before_tool_call` | `(tool_call, context) -> tool_call \| None` | Modify, or return `None` to **block** the call. |
| `after_tool_call` | `(tool_call, result, context) -> result` | Inspect/modify the tool result. |
| `on_turn_complete` | `(turn_number, context) -> None` | Per-turn bookkeeping. |
| `on_error` | `(error, context) -> None` | Observe loop errors. |

Middleware runs in registration order. Shipped implementations:
`AuditMiddleware`, `TokenTrackingMiddleware`, `PermissionMiddleware`,
`RateLimitMiddleware`, `LoopGuardMiddleware`, `ContextCompressionMiddleware`.
