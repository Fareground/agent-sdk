"""
Tests for the Agent facade — the zero-config entry point.

Uses the shared MockLLM helpers so no network or DB is required.
"""
import pytest

from fg_agents import Agent, AgentRunResult
from fg_agents.core.engine import AgentEngine
from fg_agents.core.llm import AgentLLM
from fg_agents.core.types import EventType, RegisteredTool
from fg_agents.persistence.base import BaseRepository
from fg_agents.persistence.memory import InMemoryRepository
from fg_agents.persistence.sqlite import SQLiteRepository
from fg_agents.tools.decorators import tool
from fg_agents.tools.registry import ToolRegistry
from tests.helpers import MockLLM, make_text_response, make_tool_call_response

# ══════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════

def plain_greet(name: str) -> str:
    """Greet someone by name."""
    return f"Hello, {name}!"


@tool(description="Add two numbers")
def add(a: int, b: int) -> int:
    return a + b


# ══════════════════════════════════════════════════════════════════════
# Construction
# ══════════════════════════════════════════════════════════════════════

def test_construct_with_plain_function_tools():
    agent = Agent(model="mock:test", tools=[plain_greet, add])
    assert agent.tools.has("plain_greet")
    assert agent.tools.has("add")
    assert set(agent.definition.tools) == {"plain_greet", "add"}


def test_construct_with_registered_tool():
    registered: RegisteredTool = add.tool_definition
    agent = Agent(model="mock:test", tools=[registered])
    assert agent.tools.has("add")


def test_construct_rejects_non_callable_tool():
    with pytest.raises(TypeError):
        Agent(model="mock:test", tools=[42])


def test_agent_kwargs_pass_through_to_definition():
    agent = Agent(model="mock:test", system_prompt="Be terse.", max_turns=7, temperature=0.5)
    assert agent.definition.system_prompt == "Be terse."
    assert agent.definition.max_turns == 7
    assert agent.definition.temperature == 0.5


# ══════════════════════════════════════════════════════════════════════
# Memory / persistence options
# ══════════════════════════════════════════════════════════════════════

def test_default_memory_backend():
    agent = Agent(model="mock:test")
    assert isinstance(agent.repository, InMemoryRepository)


def test_sqlite_memory_option(tmp_path):
    db_path = str(tmp_path / "agents.db")
    agent = Agent(model="mock:test", memory=f"sqlite:{db_path}")
    assert isinstance(agent.repository, SQLiteRepository)


def test_repository_instance_accepted():
    repo = InMemoryRepository()
    agent = Agent(model="mock:test", memory=repo)
    assert agent.repository is repo


def test_unknown_memory_backend_rejected():
    with pytest.raises(ValueError):
        Agent(model="mock:test", memory="redis")


def test_postgres_memory_requires_url():
    with pytest.raises(ValueError):
        Agent(model="mock:test", memory="postgres")


# ══════════════════════════════════════════════════════════════════════
# Run / stream round-trips (lazy init — no explicit initialize call)
# ══════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_run_round_trip_with_mock_llm():
    llm = MockLLM([make_text_response("Hi there!")])
    agent = Agent(model="mock:test", llm=llm)

    result = await agent.run("hello")

    assert isinstance(result, AgentRunResult)
    assert result.text == "Hi there!"
    assert str(result) == "Hi there!"
    assert result.session_id == agent.session_id
    assert any(e.type == EventType.SESSION_COMPLETED for e in result.events)


@pytest.mark.asyncio
async def test_run_with_tool_call():
    llm = MockLLM(
        [
            make_tool_call_response("add", {"a": 2, "b": 3}),
            make_text_response("The answer is 5."),
        ]
    )
    agent = Agent(model="mock:test", llm=llm, tools=[add])

    result = await agent.run("what is 2+3?")
    assert result.text == "The answer is 5."


@pytest.mark.asyncio
async def test_stream_yields_events():
    llm = MockLLM([make_text_response("streamed")])
    agent = Agent(model="mock:test", llm=llm)

    events = [e async for e in agent.stream("hello")]
    types = [e.type for e in events]
    assert EventType.SESSION_STARTED in types
    assert EventType.SESSION_COMPLETED in types


@pytest.mark.asyncio
async def test_lazy_init_is_idempotent():
    llm = MockLLM([make_text_response("one"), make_text_response("two")])
    agent = Agent(model="mock:test", llm=llm)

    await agent.run("first")
    await agent.run("second")  # second run must not re-initialize or fail

    messages = await agent.repository.get_messages(agent.session_id)
    assert len(messages) == 4  # 2 user + 2 assistant — same conversation


@pytest.mark.asyncio
async def test_per_call_session_id_overrides_default():
    llm = MockLLM([make_text_response("ok")])
    agent = Agent(model="mock:test", llm=llm)

    result = await agent.run("hello", session_id="custom-session")
    assert result.session_id == "custom-session"
    assert await agent.repository.get_session("custom-session") is not None


@pytest.mark.asyncio
async def test_close_shuts_down_repository():
    llm = MockLLM([make_text_response("bye")])
    agent = Agent(model="mock:test", llm=llm)
    await agent.run("hello")
    await agent.close()  # must not raise


# ══════════════════════════════════════════════════════════════════════
# Graduation properties
# ══════════════════════════════════════════════════════════════════════

def test_properties_expose_internals():
    agent = Agent(model="mock:test", tools=[add])
    assert isinstance(agent.engine, AgentEngine)
    assert isinstance(agent.llm, AgentLLM)
    assert isinstance(agent.tools, ToolRegistry)
    assert isinstance(agent.repository, BaseRepository)
    assert agent.definition.model == "mock:test"
    assert agent.session_id
