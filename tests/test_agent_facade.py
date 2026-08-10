"""
Tests for the Agent facade — the zero-config entry point.

Uses the shared MockLLM helpers so no network or DB is required.
"""
import pytest

from fg_agents import Agent, AgentRunError, AgentRunResult
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


def test_duplicate_tool_names_rejected():
    def add(a: int, b: int) -> int:  # same name as the module-level @tool add
        return a + b

    with pytest.raises(ValueError, match="Duplicate tool name 'add'"):
        Agent(model="mock:test", tools=[globals()["add"], add])


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


def test_bogus_memory_reported_before_model_detection(monkeypatch):
    # With no API keys and a bad memory backend, the error must be about the
    # memory backend — not a ModelDetectionError about missing providers.
    for var in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GOOGLE_API_KEY", "GEMINI_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    with pytest.raises(ValueError, match="memory backend"):
        Agent(memory="redis")


def test_missing_provider_package_fails_at_construction():
    # The dev environment deliberately has no provider SDKs installed, so a
    # provider-qualified model must fail eagerly with the install hint.
    with pytest.raises(ImportError, match=r"fg-agents\[anthropic\]"):
        Agent(model="anthropic:claude-sonnet-4-6")


def test_bare_model_string_rejected_at_construction():
    with pytest.raises(ValueError, match="provider:model"):
        Agent(model="gpt-5")


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
    await agent.close()  # idempotent


@pytest.mark.asyncio
async def test_run_after_close_raises():
    agent = Agent(model="mock:test", llm=MockLLM([make_text_response("x")]))
    await agent.close()
    with pytest.raises(RuntimeError, match="Agent is closed"):
        await agent.run("hello")


@pytest.mark.asyncio
async def test_failure_surfaces_as_agent_run_error():
    class ExplodingLLM:
        async def stream_with_tools(self, **kwargs):
            raise RuntimeError("provider exploded")
            yield  # pragma: no cover — makes this an async generator

        async def count_tokens(self, text: str, model: str = "") -> int:
            return len(text.split())

    agent = Agent(model="mock:test", llm=ExplodingLLM())

    with pytest.raises(AgentRunError) as exc_info:
        await agent.run("hello")

    err = exc_info.value
    assert err.session_id == agent.session_id
    assert err.events  # partial events collected before the failure
    assert any(e.type == EventType.SESSION_STARTED for e in err.events)


@pytest.mark.asyncio
async def test_concurrent_runs_initialize_repository_once():
    import asyncio

    repo = InMemoryRepository()
    init_calls = 0
    original_initialize = repo.initialize

    async def counting_initialize():
        nonlocal init_calls
        init_calls += 1
        await asyncio.sleep(0)  # widen the race window
        await original_initialize()

    repo.initialize = counting_initialize
    llm = MockLLM([make_text_response("ok")])
    agent = Agent(model="mock:test", llm=llm, memory=repo)

    await asyncio.gather(
        agent.run("one", session_id="s-a"),
        agent.run("two", session_id="s-b"),
    )
    assert init_calls == 1


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
