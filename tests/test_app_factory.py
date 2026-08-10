"""
Tests for the create_app() factory function.

Verifies that the factory returns a fully wired FastAPI app
with the expected routes.
"""
import pytest
from fastapi import FastAPI

from fg_agents.app import create_app
from fg_agents.core.types import AgentDefinition


def test_create_app_returns_fastapi():
    """create_app() returns a FastAPI instance with routes."""
    app = create_app(
        agents={"test": AgentDefinition(name="test", model="mock:test")},
    )
    assert isinstance(app, FastAPI)
    assert len(app.routes) > 0


def test_create_app_has_agent_routes():
    """create_app() registers /sessions and /health routes.

    Asserted against the OpenAPI schema rather than ``app.routes``: since
    FastAPI 0.141 an included router stays a single ``_IncludedRouter`` entry
    instead of being flattened into ``app.routes``, so walking that list finds
    nothing. The schema is the public contract and is stable across versions.
    """
    app = create_app(
        agents={"test": AgentDefinition(name="test", model="mock:test")},
    )
    paths = list(app.openapi()["paths"])
    assert any("/sessions" in p for p in paths), f"No /sessions route found in {paths}"
    assert any("/health" in p for p in paths), f"No /health route found in {paths}"


def test_create_app_missing_aiosqlite_fails_at_call_time(monkeypatch):
    """Default (SQLite) db_url without aiosqlite raises immediately with both fixes."""
    import fg_agents.app as app_module

    monkeypatch.setattr(app_module, "find_spec", lambda name: None)

    with pytest.raises(ImportError, match=r"fg-agents\[sqlite\].*memory"):
        create_app(agents={"test": AgentDefinition(name="test", model="mock:test")})


def test_create_app_memory_db_url_needs_no_sqlite(monkeypatch):
    """db_url='memory' works even when aiosqlite is unavailable."""
    import fg_agents.app as app_module

    monkeypatch.setattr(app_module, "find_spec", lambda name: None)

    app = create_app(
        agents={"test": AgentDefinition(name="test", model="mock:test")},
        db_url="memory",
    )
    assert isinstance(app, FastAPI)
