"""
Fareground Agent Framework — Persistence Layer

Backend support: memory (testing), sqlite (dev), postgres (production).
"""

from fg_agents.persistence.base import BaseRepository
from fg_agents.persistence.factory import create_repository
from fg_agents.persistence.memory import InMemoryRepository

# The SQLAlchemy models (and PostgresRepository below) need the 'postgres'
# extra; a core install works without them.
try:
    from fg_agents.persistence.models import (
        AgentArtifactModel,
        AgentAuditLogModel,
        AgentMemoryModel,
        AgentMessageModel,
        AgentSessionModel,
        AgentToolExecutionModel,
        Base,
    )
except ImportError:
    Base = None  # type: ignore[assignment,misc]
    AgentArtifactModel = None  # type: ignore[assignment,misc]
    AgentAuditLogModel = None  # type: ignore[assignment,misc]
    AgentMemoryModel = None  # type: ignore[assignment,misc]
    AgentMessageModel = None  # type: ignore[assignment,misc]
    AgentSessionModel = None  # type: ignore[assignment,misc]
    AgentToolExecutionModel = None  # type: ignore[assignment,misc]

# PostgresRepository requires asyncpg + sqlalchemy
try:
    from fg_agents.persistence.repository import PostgresRepository, Repository
except ImportError:

    def _postgres_not_installed(*args, **kwargs):
        raise ImportError(
            "PostgresRepository requires SQLAlchemy and asyncpg. "
            "Install with: pip install 'fg-agents[postgres]'"
        )

    PostgresRepository = _postgres_not_installed  # type: ignore[assignment,misc]
    Repository = _postgres_not_installed  # type: ignore[assignment,misc]

# SQLiteRepository requires aiosqlite
try:
    from fg_agents.persistence.sqlite import SQLiteRepository
except ImportError:

    def _sqlite_not_installed(*args, **kwargs):
        raise ImportError(
            "SQLiteRepository requires aiosqlite. Install with: pip install 'fg-agents[sqlite]'"
        )

    SQLiteRepository = _sqlite_not_installed  # type: ignore[assignment,misc]

__all__ = [
    "BaseRepository",
    "create_repository",
    "InMemoryRepository",
    "PostgresRepository",
    "Repository",
    "SQLiteRepository",
    "Base",
    "AgentSessionModel",
    "AgentMessageModel",
    "AgentToolExecutionModel",
    "AgentArtifactModel",
    "AgentAuditLogModel",
    "AgentMemoryModel",
]
