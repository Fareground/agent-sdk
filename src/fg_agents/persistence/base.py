"""
Fareground Agent Framework — Abstract Repository Interface

Defines the contract all persistence backends must implement.
"""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Any

from fg_agents.core.types import AgentMessage, AgentSession


def validate_message_batch_size(batch_size: int) -> None:
    if type(batch_size) is not int or not 1 <= batch_size <= 256:
        raise ValueError("message batch_size must be an integer from 1 to 256")


class BaseRepository(ABC):
    """
    Abstract base for all persistence backends.

    Implementations: PostgresRepository, InMemoryRepository,
    SQLiteRepository.
    """

    @abstractmethod
    async def initialize(self) -> None:
        """Set up storage (create tables, dirs, etc.). Call once on startup."""

    @abstractmethod
    async def close(self) -> None:
        """Release resources."""

    # ══════════════════════════════════════════════════════════════════
    # Sessions
    # ══════════════════════════════════════════════════════════════════

    @abstractmethod
    async def create_session(self, session: AgentSession) -> AgentSession: ...

    @abstractmethod
    async def get_session(self, session_id: str) -> AgentSession | None: ...

    @abstractmethod
    async def update_session(self, session_id: str, **kwargs) -> None: ...

    @abstractmethod
    async def get_sessions_for_user(
        self,
        user_id: str,
        tenant_id: str = "",
        status: str | None = None,
        limit: int = 50,
    ) -> list[AgentSession]: ...

    @abstractmethod
    async def delete_session(self, session_id: str) -> None: ...

    async def clear_session_data(self, session_id: str) -> None:
        """Wipe a session's message history + derived records and reset its
        running counters, WITHOUT deleting the session row itself.

        The session keeps its id, agent_id and metadata (name, daily flag) so
        the same conversation slot continues with a fresh, empty context.
        Override in subclasses for efficient backend-level deletion. Default:
        no-op (subclasses that persist data must override).
        """
        return None

    async def get_child_sessions(self, parent_session_id: str) -> list[AgentSession]:
        """Get all sessions spawned by a parent session."""
        return []

    async def purge_old_sessions(
        self,
        max_age_hours: int = 168,
        statuses: list[str] | None = None,
    ) -> int:
        """
        Delete sessions older than max_age_hours. Override in subclasses for
        efficient DB-level deletion. Default: no-op returning 0.
        """
        return 0

    # ══════════════════════════════════════════════════════════════════
    # Messages
    # ══════════════════════════════════════════════════════════════════

    @abstractmethod
    async def add_message(self, message: AgentMessage) -> AgentMessage: ...

    @abstractmethod
    async def get_messages(
        self,
        session_id: str,
        include_summarized: bool = False,
    ) -> list[AgentMessage]: ...

    async def get_message(self, session_id: str, message_id: str) -> AgentMessage | None:
        """Read one original, including summarized history, within this session.

        Compatibility fallback for third-party repositories. Shipped backends
        override this with a targeted read; callers must still authorize the
        session's owner and tenant before accessing its history.
        """
        return next((m for m in await self.get_messages(session_id, include_summarized=True)
                     if m.id == message_id), None)

    async def iter_messages(
        self, session_id: str, *, include_summarized: bool = True, batch_size: int = 64,
    ) -> AsyncIterator[AgentMessage]:
        """Scan originals for archive recovery, not model-context assembly.

        Shipped durable backends hydrate at most batch_size rows at a time.
        This compatibility fallback retains third-party repository behavior.
        Archive tie ordering may differ from live-context ordering; do not use
        this API to reconstruct a model's assistant/tool exchange sequence.
        """
        validate_message_batch_size(batch_size)
        for message in await self.get_messages(session_id, include_summarized=include_summarized):
            yield message

    @abstractmethod
    async def mark_messages_summarized(
        self,
        session_id: str,
        message_ids: list[str],
    ) -> None: ...

    # ══════════════════════════════════════════════════════════════════
    # Tool Executions
    # ══════════════════════════════════════════════════════════════════

    @abstractmethod
    async def log_tool_execution(
        self,
        session_id: str,
        message_id: str | None,
        tool_call_id: str,
        tool_name: str,
        input_data: dict,
        output_data: Any,
        status: str,
        duration_ms: float,
        error: str | None = None,
    ) -> None: ...

    # ══════════════════════════════════════════════════════════════════
    # Artifacts
    # ══════════════════════════════════════════════════════════════════

    @abstractmethod
    async def save_artifact(
        self,
        session_id: str,
        artifact_type: str,
        name: str,
        content: str | None = None,
        content_json: dict | None = None,
        metadata: dict | None = None,
    ) -> str: ...

    @abstractmethod
    async def get_artifacts(
        self,
        session_id: str,
        artifact_type: str | None = None,
    ) -> list[dict]: ...

    # ══════════════════════════════════════════════════════════════════
    # Audit Log
    # ══════════════════════════════════════════════════════════════════

    @abstractmethod
    async def log_audit(
        self,
        session_id: str,
        event_type: str,
        action: str,
        actor: str = "",
        detail: dict | None = None,
        llm_model: str | None = None,
        llm_input_tokens: int | None = None,
        llm_output_tokens: int | None = None,
        llm_latency_ms: float | None = None,
    ) -> None: ...

    @abstractmethod
    async def get_audit_log(self, session_id: str) -> list[dict]: ...

    # ══════════════════════════════════════════════════════════════════
    # Memory Store
    # ══════════════════════════════════════════════════════════════════

    @abstractmethod
    async def get_memory(self, agent_id: str, key: str) -> Any | None: ...

    @abstractmethod
    async def set_memory(
        self,
        agent_id: str,
        key: str,
        value: Any,
        memory_type: str = "long_term",
        session_id: str | None = None,
    ) -> None: ...

    @abstractmethod
    async def list_memories(
        self,
        agent_id: str,
        memory_type: str | None = None,
    ) -> list[dict]: ...
