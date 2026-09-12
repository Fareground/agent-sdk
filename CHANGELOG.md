# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.4.0] - 2026-08-10

### ⚠️ Breaking — slimmer core install

- **The web stack and PostgreSQL are now extras, not core dependencies.**
  `pip install fg-agents` ships only the engine, tools, `ask`/`Agent`, and
  in-memory persistence (pydantic, structlog, aiohttp). If you use:
  - `create_app` / `create_agent_router` / the HTTP+SSE API → install
    **`fg-agents[web]`** (fastapi + uvicorn)
  - `PostgresRepository` / `memory="postgres:<url>"` → install
    **`fg-agents[postgres]`** (sqlalchemy + asyncpg)
  - `fg-agents[all]` still installs everything.
  Missing extras fail at call time with an actionable install hint.
- **`Base` (SQLAlchemy declarative base) removed from the top-level public
  surface** (`fg_agents.Base`, `__all__`). It remains importable from
  `fg_agents.persistence` when the `postgres` extra is installed.
- **Bare model strings are rejected.** `model="gpt-5"` (no `provider:` prefix)
  now raises `ValueError` listing the expected `provider:model` format and the
  known providers, instead of being silently routed to a default provider.

### Added

- **Eager provider-dependency check.** `Agent(...)` (and therefore `ask()`)
  verifies at construction that the resolved provider's SDK is installed and
  raises a one-line, actionable error — e.g. *"The anthropic provider requires
  the 'anthropic' package — pip install 'fg-agents[anthropic]'"* — instead of a
  deep `ModuleNotFoundError` inside the engine. The same check guards client
  creation for low-level `AgentLLM` use.
- `SECURITY.md` and a core-only CI job that installs the bare package (no web,
  no SQLAlchemy, no provider SDKs) and runs the non-web test subset.

### Changed

- `Agent` validates its memory/tools arguments before model detection, so
  `Agent(memory="redis")` reports the bogus backend rather than a misleading
  `ModelDetectionError`.
- The engine no longer logs a full stack trace for clean, user-facing
  configuration errors (`AgentFrameworkError` subclasses); stack logging is
  reserved for genuine internal failures.
- README hello-world is now directly runnable (`asyncio.run`), with the bare
  `await` form kept only under an explicit async/REPL caption.

### Tool argument safety (T-541)

- Native Anthropic and OpenAI-compatible streamed calls share final JSON-object
  validation. Malformed native calls no longer execute with empty defaults.
- Failed calls retain content-free parse coordinates, fragment/character counts
  and stop reason; argument contents are not included in diagnostics. Native
  SDK parser errors that embed tool JSON are sanitized before propagation.
- Stream fragments are joined once rather than repeatedly copying the growing
  JSON string. Native calls complete only after final stop metadata is known;
  valid sibling calls and usage snapshots remain intact.

### Security — audit hardening (2026-07-23)

- **Tenant isolation is now enforced at the HTTP boundary (was: unenforced).**
  `create_agent_router(authenticator=...)` gates every route with a verified
  `AuthContext`; a caller only reaches its own tenant's sessions and memory,
  and a session's tenant/user come from the authenticator, never the request
  body (a client can no longer mint a session in another tenant). Cross-tenant
  access returns 404, not 403, so a probe cannot even confirm a session id
  exists. Ships `HeaderAuthenticator` (trusted-proxy) and `UnauthenticatedAccess`
  (single-tenant default, warns loudly). Runtime tool/agent registration now
  also requires `is_admin`.
- **Persistent agent memory is tenant-scoped (was: cross-tenant leak).**
  Objectives, knowledge docs, and skills are keyed by `(tenant, agent)` — two
  tenants running the same agent no longer share, or leak into each other's
  prompt, one memory. Empty tenant maps to the bare agent id (back-compatible).
- Rate-limiter now actually enforces (was a no-op) and pauses the session
  recoverably; `ask` permission fails closed (was auto-approve); tool args are
  validated at the boundary; SSRF hardening on the API tool handler (path-param
  encoding, host-pin, no redirects); read-only SQL enforced on every backend;
  Gemini multi-turn tool loop + Anthropic thinking-block replay fixed; bounded
  per-session middleware memory; code-exec subprocess no longer inherits the
  parent's secrets. See git history on this range for detail.

## [0.3.0] - 2026-03-24

### Added
- **Parallel sub-agent dispatch**: `manage_agent(action="spawn_parallel")` runs multiple sub-agents concurrently via `asyncio.gather`. Configurable via `max_parallel_agents` on `AgentDefinition` (default: 5).
- **Nuclear stop**: `orchestrator.cancel(session_id)` cancels a session and ALL child sessions. HTTP: `POST /sessions/{id}/cancel`. Running loops exit at the top of the next turn.
- **Session cancellation**: `engine.cancel(session_id, cascade=True)` with cascade to child sessions. New `get_child_sessions()` on all repository backends.
- **SSE production streaming**: Router now uses `sse_response()` from `streaming/sse.py` with proper queue-based heartbeat (fires during long LLM calls), Last-Event-ID reconnection, and client disconnect detection.
- **Auth guards**: `POST /tools` and `POST /agents` return 403 by default. Opt-in with `admin_only=False` on `create_agent_router()`.
- **Code execution safety gate**: `execute_code_tool()` requires `FG_ALLOW_CODE_EXECUTION=true` env var. Disabled by default.
- **Turn-budget awareness**: System prompt shows `Turn N of M` every turn. Prompts instruct agents to skip to SYNTHESIZE/DELIVER when <3 turns remain.
- **Sub-agent restrictions**: Task agents can only use `manage_agent(action="report")`. All other actions (spawn, message, list) return errors.
- **User/tenant auth context**: `ExecutionContext` now has typed `user_id` and `tenant_id` fields, wired from session metadata.
- **Objective management**: `manage_objective` tool for persistent mission/purpose (auto-loaded into prompt every turn).
- **Manual context compaction**: `manage_context` tool sets a flag → engine triggers LLM summarization on the next turn.
- **Scheduler**: DB-backed cron/one-time/event scheduling for automated agent runs.
- **235 tests** across 26 test files covering engine, orchestrator, delegation, SSE, security, router, prompts, persistence, middleware, scheduler.

### Changed
- **OPTIMIZE before DELIVER**: Task agent phases reordered (PREPARE → PLAN → INVESTIGATE → OPTIMIZE → REPORT). Both prompts warn: "Complete all persist calls BEFORE your final response."
- **Prompts enhanced**: spawn_parallel documented, tool failure retry guidance, interim vs final report labels, "only use listed tools" guardrail, "handle tool failures" principle.
- **SubAgentRunner encapsulation**: Uses `engine.repository` property instead of `engine._repo`.
- **Dead code removed**: Router's inline `_stream_events()` with broken heartbeat replaced by `sse_response()`.

### Fixed
- SSE heartbeat now fires during long LLM calls (was dead code in router).
- `manage_context` tool actually triggers compaction (was a no-op).
- Security section and troubleshooting section added to README.

## [0.2.0] - 2026-03-24

### Added
- **Multi-backend persistence**: `InMemoryRepository`, `SQLiteRepository` alongside existing PostgreSQL. Use `create_repository("memory"|"sqlite"|"postgres")`.
- **LLM retry & error recovery**: Automatic retry on rate limits, timeouts, and context overflow. Configurable via `max_llm_retries` on `AgentDefinition`.
- **Turn-level recovery**: Failed LLM turns set session to `WAITING_INPUT` instead of `FAILED`, allowing resume on next message.
- **AgentService**: Framework-agnostic service layer (`api/service.py`). No web framework imports — use with FastAPI, Flask, Django, or CLI.
- **Programmatic skills**: `SkillsManager.from_function()`, `from_url()`, `from_database()` for creating skills from code, URLs, or database callbacks.
- **Dynamic knowledge sources**: `knowledge` field now accepts `str` (file paths), `dict` (inline), or `Callable` (dynamic).
- **Runtime skill injection**: `ExecutionContext.add_skill()` / `remove_skill()` for mid-session skill changes.
- **BaseRepository ABC**: All persistence backends implement a common interface.
- **Repository factory**: `create_repository()` function for one-line backend selection.
- **Public Orchestrator properties**: `.repo`, `.tools`, `.middleware`, `.skills` accessors.
- `get_audit_log()` method on all repository backends.

### Changed
- **Model is now required**: `AgentDefinition.model` defaults to `""` — you must explicitly specify a model string. Clear error message if omitted.
- **PostgreSQL is optional**: `asyncpg` and `sqlalchemy` moved from core dependencies to the `postgres` extra. Core framework has zero database dependencies.
- **FastAPI router decoupled**: Router is now a thin adapter over `AgentService`. No private attribute access on Orchestrator.
- `Repository` renamed to `PostgresRepository` (backward-compatible alias `Repository` still works).

### Fixed
- Examples now use correct API (`create_repository`, explicit model, proper event access).
- Working memory tool examples use sync `.store()` / `.get()` (not async).

## [0.1.0] - 2025-12-01

### Added
- Initial release: ReAct engine, multi-agent orchestration, tool system, skills, middleware, streaming, PostgreSQL persistence.
