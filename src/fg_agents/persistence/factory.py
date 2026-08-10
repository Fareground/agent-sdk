"""
Fareground Agent Framework — Repository Factory

Create a persistence backend by name.

Usage:
    from fg_agents import create_repository

    # Testing (no deps needed)
    repo = create_repository("memory")

    # Local dev
    repo = create_repository("sqlite", db_path="agents.db")

    # Production
    repo = create_repository("postgres", db_url="postgresql+asyncpg://...")
"""

from importlib.util import find_spec

from fg_agents.persistence.base import BaseRepository


def _module_available(module: str) -> bool:
    """True when a module can be found. A raising finder counts as missing."""
    try:
        return find_spec(module) is not None
    except ImportError:
        return False


def create_repository(backend: str = "memory", **kwargs) -> BaseRepository:
    """
    Factory for persistence backends.

    Args:
        backend: One of "memory", "sqlite", "postgres".
        **kwargs: Backend-specific arguments:
            - memory: (no args)
            - sqlite: db_path (str, default "fg_agents.db")
            - postgres: db_url (str), echo (bool)

    Returns:
        A BaseRepository instance (call .initialize() before use).
    """
    backend = backend.lower().strip()

    if backend == "memory":
        from fg_agents.persistence.memory import InMemoryRepository

        return InMemoryRepository()

    elif backend == "sqlite":
        if not _module_available("aiosqlite"):
            raise ImportError(
                "The sqlite backend requires the 'aiosqlite' package — "
                "pip install 'fg-agents[sqlite]'"
            )
        from fg_agents.persistence.sqlite import SQLiteRepository

        return SQLiteRepository(**kwargs)

    elif backend in ("postgres", "postgresql"):
        if not _module_available("sqlalchemy") or not _module_available("asyncpg"):
            raise ImportError(
                "The postgres backend requires SQLAlchemy and asyncpg — "
                "pip install 'fg-agents[postgres]'"
            )
        from fg_agents.persistence.repository import PostgresRepository

        return PostgresRepository(**kwargs)

    else:
        raise ValueError(f"Unknown backend: '{backend}'. Choose from: memory, sqlite, postgres")
