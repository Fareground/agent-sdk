"""
Fareground Agent Framework
=====================

The agent framework built for web applications.

Drop into your FastAPI/Starlette app. Streams over SSE. Persists to
PostgreSQL. Knows about users, tenants, and sessions. Supports 23+
LLM providers — no default model, you choose at runtime.

Core concepts:
- Orchestrator: Brain agent that plans, delegates, and synthesizes
- AgentEngine: The ReAct loop (model → tools → model → ...)
- AgentDefinition: Declarative agent configuration (no default model)
- ToolRegistry: Central tool store with @tool decorator
- Skills: Reusable prompt template + tools bundles
- WorkingMemory: In-flight scratch space per session
- Middleware: Hook into the agent loop (audit, permissions, token tracking)
- Repository: Async DB persistence (PostgreSQL, SQLite, in-memory)
- StreamEvent: Real-time SSE events with reconnection support
- UserContext: Multi-tenant user/org context from HTTP headers

Quick start (one API key env var set is all it takes)::

    from fg_agents import ask

    print(await ask("What's 2+2?"))

Web app::

    from fg_agents import create_app, AgentDefinition, tool, ToolRegistry

    @tool(name="search", description="Search the database")
    async def search(query: str) -> dict:
        return {"results": [...]}

    tools = ToolRegistry()
    tools.register_function(search)

    assistant = AgentDefinition(
        name="assistant",
        model="openai:gpt-4.1",  # user provides model — no default
        tools=["search"],
    )

    app = create_app(
        agents={"assistant": assistant},
        db_url="postgresql://...",
        tool_registry=tools,
    )
    # Run with: uvicorn myapp:app

    # Or use programmatically:
    from fg_agents import Orchestrator, AgentLLM, create_repository

    llm = AgentLLM(api_keys={"openai": "sk-..."})
    repo = create_repository("postgres", db_url="postgresql+asyncpg://...")
    orchestrator = Orchestrator(llm, tools, repo)
    await orchestrator.initialize()

    async for event in orchestrator.run("session-1", "Analyze this", assistant):
        print(event.to_sse())  # SSE-formatted for streaming to frontend
"""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

try:
    __version__ = _pkg_version("fg-agents")
except PackageNotFoundError:  # editable/source checkout without install metadata
    __version__ = "0.0.0.dev0"

from fg_agents.agent import Agent, AgentRunError, AgentRunResult
from fg_agents.ask import ask, stream
from fg_agents.core.engine import AgentEngine
from fg_agents.core.errors import (
    AgentFrameworkError,
    ContextOverflowError,
    LLMError,
    LLMRateLimitError,
    MaxTurnsExceededError,
    MiddlewareError,
    SessionError,
    SessionNotFoundError,
    SkillNotFoundError,
    SubAgentError,
    ToolDeniedError,
    ToolExecutionError,
    ToolNotFoundError,
    ToolTimeoutError,
)
from fg_agents.core.llm import AgentLLM
from fg_agents.core.types import (
    AgentDefinition,
    AgentMessage,
    AgentSession,
    EventType,
    ExecutionContext,
    LLMResponse,
    LLMStreamChunk,
    LLMUsage,
    MessageRole,
    RegisteredTool,
    SessionStatus,
    Skill,
    StopReason,
    SubAgentResult,
    ToolCall,
    ToolResult,
    ToolSchema,
    ToolStatus,
    ToolType,
    UserContext,
)
from fg_agents.memory.context import ContextManager
from fg_agents.memory.health import ContextHealthConfig, assess_health, compute_health_score
from fg_agents.memory.working import WorkingMemory
from fg_agents.middleware.audit import AuditMiddleware
from fg_agents.middleware.base import BaseMiddleware, Middleware
from fg_agents.middleware.context_compression import (
    CompressionConfig,
    ContextCompressionMiddleware,
    progressive_compress,
)
from fg_agents.middleware.loop_guard import LoopGuardConfig, LoopGuardMiddleware
from fg_agents.middleware.permissions import PermissionMiddleware, PermissionRule
from fg_agents.middleware.rate_limit import RateLimitMiddleware
from fg_agents.middleware.token_tracking import TokenTrackingMiddleware
from fg_agents.model_detection import (
    ModelDetectionError,
    resolve_default_model,
    resolve_default_model_async,
)
from fg_agents.orchestrator.orchestrator import Orchestrator
from fg_agents.orchestrator.sub_agent import SubAgentRunner
from fg_agents.persistence.base import BaseRepository
from fg_agents.persistence.factory import create_repository
from fg_agents.persistence.memory import InMemoryRepository
from fg_agents.persistence.models import Base

try:
    from fg_agents.persistence.repository import PostgresRepository, Repository
except ImportError:

    def _postgres_not_installed(*args, **kwargs):
        raise ImportError(
            "PostgresRepository requires asyncpg and sqlalchemy (core dependencies "
            "of fg-agents). Install with: pip install 'sqlalchemy[asyncio]' asyncpg"
        )

    PostgresRepository = _postgres_not_installed  # type: ignore[assignment,misc]
    Repository = _postgres_not_installed  # type: ignore[assignment,misc]

try:
    from fg_agents.persistence.sqlite import SQLiteRepository
except ImportError:

    def _sqlite_not_installed(*args, **kwargs):
        raise ImportError(
            "SQLiteRepository requires aiosqlite. Install with: pip install 'fg-agents[sqlite]'"
        )

    SQLiteRepository = _sqlite_not_installed  # type: ignore[assignment,misc]
from fg_agents.prompts.templates import (
    DEFAULT_SYSTEM_PROMPT,
    ORCHESTRATOR_SYSTEM_PROMPT,
    TASK_AGENT_SYSTEM_PROMPT,
    build_system_prompt,
    load_knowledge,
)
from fg_agents.skills.manager import SkillsManager
from fg_agents.streaming.events import StreamEvent
from fg_agents.tools.builtin import BUILTIN_TOOLS
from fg_agents.tools.decorators import tool
from fg_agents.tools.handlers import TOOL_TYPE_HANDLERS, dispatch_tool
from fg_agents.tools.registry import ToolRegistry

try:
    from fg_agents.scheduler.scheduler import AgentScheduler, ScheduleEntry
except ImportError:
    AgentScheduler = None  # type: ignore[assignment,misc]
    ScheduleEntry = None  # type: ignore[assignment,misc]

try:
    from fg_agents.api.router import create_agent_router
    from fg_agents.api.service import AgentService
    from fg_agents.app import create_app
    from fg_agents.dependencies import get_service, get_user_context
except ImportError as e:
    _web_import_error = e

    def _fastapi_not_installed(*args, **kwargs):
        raise ImportError(
            "The web layer (create_app, create_agent_router, AgentService) requires "
            "FastAPI, a core dependency of fg-agents. Your environment is missing it — "
            "install with: pip install fastapi 'uvicorn[standard]'"
        ) from _web_import_error

    create_agent_router = _fastapi_not_installed  # type: ignore[assignment,misc]
    AgentService = _fastapi_not_installed  # type: ignore[assignment,misc]
    create_app = _fastapi_not_installed  # type: ignore[assignment,misc]
    get_service = _fastapi_not_installed  # type: ignore[assignment,misc]
    get_user_context = _fastapi_not_installed  # type: ignore[assignment,misc]

__all__ = [
    # One-shot
    "ask",
    "stream",
    "resolve_default_model",
    "resolve_default_model_async",
    "ModelDetectionError",
    # Facade
    "Agent",
    "AgentRunError",
    "AgentRunResult",
    # Core types
    "AgentSession",
    "AgentMessage",
    "AgentDefinition",
    "ToolCall",
    "ToolResult",
    "ToolSchema",
    "RegisteredTool",
    "Skill",
    "ExecutionContext",
    "SubAgentResult",
    "UserContext",
    "LLMStreamChunk",
    "LLMResponse",
    "LLMUsage",
    "SessionStatus",
    "MessageRole",
    "ToolStatus",
    "ToolType",
    "EventType",
    "StopReason",
    # Engine
    "AgentLLM",
    "AgentEngine",
    # Orchestrator
    "Orchestrator",
    "SubAgentRunner",
    # Tools
    "ToolRegistry",
    "tool",
    "BUILTIN_TOOLS",
    "dispatch_tool",
    "TOOL_TYPE_HANDLERS",
    # Skills
    "SkillsManager",
    # Memory
    "WorkingMemory",
    "ContextManager",
    # Middleware
    "Middleware",
    "BaseMiddleware",
    "AuditMiddleware",
    "TokenTrackingMiddleware",
    "PermissionMiddleware",
    "PermissionRule",
    "RateLimitMiddleware",
    "ContextCompressionMiddleware",
    "CompressionConfig",
    "progressive_compress",
    "LoopGuardMiddleware",
    "LoopGuardConfig",
    # Context Health
    "ContextHealthConfig",
    "assess_health",
    "compute_health_score",
    # Persistence
    "BaseRepository",
    "create_repository",
    "InMemoryRepository",
    "PostgresRepository",
    "SQLiteRepository",
    "Repository",
    "Base",
    # Streaming
    "StreamEvent",
    # Prompts
    "build_system_prompt",
    "load_knowledge",
    "DEFAULT_SYSTEM_PROMPT",
    "ORCHESTRATOR_SYSTEM_PROMPT",
    "TASK_AGENT_SYSTEM_PROMPT",
    # Scheduler
    "AgentScheduler",
    "ScheduleEntry",
    # API & Web
    "create_app",
    "create_agent_router",
    "AgentService",
    "get_service",
    "get_user_context",
    # Errors
    "AgentFrameworkError",
    "LLMError",
    "LLMRateLimitError",
    "ContextOverflowError",
    "ToolExecutionError",
    "ToolNotFoundError",
    "ToolDeniedError",
    "ToolTimeoutError",
    "SessionError",
    "SessionNotFoundError",
    "MaxTurnsExceededError",
    "SubAgentError",
    "MiddlewareError",
    "SkillNotFoundError",
]
