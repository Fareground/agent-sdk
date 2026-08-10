"""
Fareground Agent Framework — Agent Facade

Zero-config entry point that collapses the four-object setup
(AgentLLM + repository + ToolRegistry + AgentEngine) into one class.

Usage::

    from fg_agents import Agent

    agent = Agent(model="anthropic:claude-sonnet-4-6", tools=[my_func])
    result = await agent.run("hello")
    print(result.text)

    async for event in agent.stream("tell me more"):
        print(event.type)

Graduate to the low-level API at any point via the ``engine``, ``llm``,
``tools``, and ``repository`` properties.
"""

import asyncio
import uuid
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field

from fg_agents.core.engine import AgentEngine
from fg_agents.core.errors import AgentFrameworkError
from fg_agents.core.llm import AgentLLM
from fg_agents.core.types import AgentDefinition, EventType, RegisteredTool
from fg_agents.persistence.base import BaseRepository
from fg_agents.persistence.factory import create_repository
from fg_agents.streaming.events import StreamEvent
from fg_agents.tools.decorators import tool as tool_decorator
from fg_agents.tools.registry import ToolRegistry


class AgentRunError(AgentFrameworkError):
    """
    Raised by :meth:`Agent.run` when a run fails.

    Carries the session id and the partial events collected before the
    failure. The original exception (if any) is chained as ``__cause__``.
    """

    def __init__(self, message: str, session_id: str, events: list[StreamEvent]):
        super().__init__(message)
        self.session_id = session_id
        self.events = events


@dataclass(frozen=True)
class AgentRunResult:
    """Result of a completed :meth:`Agent.run` call."""

    text: str
    session_id: str
    events: list[StreamEvent] = field(default_factory=list)

    def __str__(self) -> str:
        return self.text


class Agent:
    """
    High-level facade over AgentLLM, ToolRegistry, a repository, and AgentEngine.

    Args:
        model: Provider-qualified model id, e.g. ``"anthropic:claude-sonnet-4-6"``.
        tools: Tools available to the agent. Accepts ``@tool``-decorated
            functions, plain callables (auto-wrapped via ``@tool``), or
            ``RegisteredTool`` instances.
        system_prompt: System prompt for the agent.
        name: Agent name (defaults to ``"agent"``).
        memory: Persistence backend — ``"memory"`` (default), ``"sqlite"``,
            ``"sqlite:<path>"``, or a ready :class:`BaseRepository` instance.
        llm: Optional pre-built :class:`AgentLLM` (e.g. with api_keys).
        api_keys: Provider → API key overrides (ignored when ``llm`` is given).
        session_id: Conversation id. Defaults to a fresh id per Agent
            instance; every run/stream continues the same conversation
            unless a per-call ``session_id`` is passed.
        **agent_kwargs: Extra :class:`AgentDefinition` fields passed through
            (``max_turns``, ``temperature``, ``fallback_models``, ...).

    Initialization is lazy: the repository is initialized on the first
    ``run``/``stream`` call — forgetting to initialize is never an error.
    """

    def __init__(
        self,
        model: str,
        *,
        tools: list[Callable | RegisteredTool] | None = None,
        system_prompt: str = "",
        name: str = "agent",
        memory: str | BaseRepository = "memory",
        llm: AgentLLM | None = None,
        api_keys: dict[str, str] | None = None,
        session_id: str | None = None,
        **agent_kwargs,
    ):
        self._llm = llm or AgentLLM(api_keys=api_keys)
        self._repository = self._resolve_repository(memory)
        self._registry = ToolRegistry()
        for t in tools or []:
            self._register_tool(t)

        self._definition = AgentDefinition(
            name=name,
            model=model,
            system_prompt=system_prompt,
            tools=list(self._registry.tools.keys()),
            **agent_kwargs,
        )
        self._engine = AgentEngine(
            llm=self._llm,
            tool_registry=self._registry,
            repository=self._repository,
        )
        self._session_id = session_id or f"agent-{uuid.uuid4().hex[:12]}"
        self._initialized = False
        self._closed = False
        self._init_lock = asyncio.Lock()

    # ── Setup helpers ─────────────────────────────────────────────────

    @staticmethod
    def _resolve_repository(memory: str | BaseRepository) -> BaseRepository:
        if isinstance(memory, BaseRepository):
            return memory
        if not isinstance(memory, str):
            raise TypeError(
                f"memory must be a backend string or BaseRepository, got {type(memory).__name__}"
            )
        backend, _, arg = memory.partition(":")
        backend = backend.lower().strip()
        if backend == "memory":
            return create_repository("memory")
        if backend == "sqlite":
            kwargs = {"db_path": arg} if arg else {}
            return create_repository("sqlite", **kwargs)
        if backend in ("postgres", "postgresql"):
            if not arg:
                raise ValueError(
                    "postgres memory requires a URL: 'postgres:postgresql+asyncpg://...'"
                )
            return create_repository("postgres", db_url=arg)
        raise ValueError(
            f"Unknown memory backend: '{memory}'. "
            "Use 'memory', 'sqlite[:path]', 'postgres:<url>', or pass a repository."
        )

    def _register_tool(self, t: Callable | RegisteredTool) -> None:
        if isinstance(t, RegisteredTool):
            registered = t
        elif callable(t):
            if not hasattr(t, "tool_definition"):
                t = tool_decorator()(t)  # wrap plain callables
            registered = t.tool_definition
        else:
            raise TypeError(f"Cannot register tool {t!r} — must be callable or RegisteredTool")
        if self._registry.has(registered.name):
            raise ValueError(
                f"Duplicate tool name '{registered.name}' in tools=[...] — "
                "tool names must be unique."
            )
        self._registry.register(registered)

    async def _ensure_initialized(self) -> None:
        """Initialize the repository exactly once, safely under concurrency."""
        if self._initialized:
            return
        async with self._init_lock:
            if not self._initialized:
                await self._repository.initialize()
                self._initialized = True

    # ── Execution ─────────────────────────────────────────────────────

    async def run(
        self,
        message: str,
        *,
        session_id: str | None = None,
        metadata: dict | None = None,
        variables: dict | None = None,
    ) -> AgentRunResult:
        """
        Run the agent to completion and return the final result.

        Continues the Agent's conversation by default; pass ``session_id``
        to target a different conversation.

        Failure contract: any failure — an engine-emitted ERROR event or an
        exception raised during the run — surfaces as :class:`AgentRunError`
        carrying ``session_id`` and the partial ``events`` collected so far.
        The original exception, when there is one, is chained as ``__cause__``.
        """
        if self._closed:
            raise RuntimeError("Agent is closed")
        events: list[StreamEvent] = []
        final_text = ""
        sid = session_id or self._session_id
        try:
            async for event in self.stream(
                message, session_id=sid, metadata=metadata, variables=variables
            ):
                events.append(event)
                if event.type == EventType.SESSION_COMPLETED:
                    final_text = event.data.get("final_output", "")
                elif event.type == EventType.ERROR:
                    raise AgentRunError(
                        f"Agent run failed ({event.data.get('error_type', 'error')}): "
                        f"{event.data.get('message', 'unknown error')}",
                        session_id=sid,
                        events=events,
                    )
        except AgentRunError:
            raise
        except Exception as e:
            raise AgentRunError(
                f"Agent run failed: {e}", session_id=sid, events=events
            ) from e
        return AgentRunResult(text=final_text, session_id=sid, events=events)

    async def stream(
        self,
        message: str,
        *,
        session_id: str | None = None,
        metadata: dict | None = None,
        variables: dict | None = None,
    ) -> AsyncIterator[StreamEvent]:
        """
        Run the agent and yield StreamEvents as they happen.

        Failure contract: engine failures surface either as ERROR events in
        the stream (the engine's in-loop handling) or as raw engine
        exceptions propagating out of the iterator — callers of ``stream()``
        handle both. Use :meth:`run` for a single, uniform ``AgentRunError``.
        """
        if self._closed:
            raise RuntimeError("Agent is closed")
        await self._ensure_initialized()
        async for event in self._engine.run(
            session_id or self._session_id,
            message,
            self._definition,
            metadata=metadata,
            variables=variables,
        ):
            yield event

    async def close(self) -> None:
        """Close the underlying repository. The Agent is unusable afterwards."""
        if self._closed:
            return
        self._closed = True
        if self._initialized:
            await self._repository.close()

    # ── Graduation to the low-level API ───────────────────────────────

    @property
    def engine(self) -> AgentEngine:
        return self._engine

    @property
    def llm(self) -> AgentLLM:
        return self._llm

    @property
    def tools(self) -> ToolRegistry:
        return self._registry

    @property
    def repository(self) -> BaseRepository:
        return self._repository

    @property
    def definition(self) -> AgentDefinition:
        return self._definition

    @property
    def session_id(self) -> str:
        return self._session_id
