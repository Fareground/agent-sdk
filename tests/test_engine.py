"""
Integration tests for the AgentEngine ReAct loop.

Uses InMemoryRepository and a mock LLM to test the full engine flow
without any external dependencies.
"""
import pytest

from fg_agents.core.engine import AgentEngine
from fg_agents.core.types import (
    AgentDefinition,
    EventType,
    ExecutionContext,
    MessageRole,
    SessionStatus,
)
from fg_agents.persistence.memory import InMemoryRepository
from fg_agents.skills.manager import SkillsManager
from fg_agents.tools.decorators import tool
from fg_agents.tools.registry import ToolRegistry
from tests.helpers import MockLLM, make_text_response, make_tool_call_response

# ══════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════

async def build_engine(
    mock_llm: MockLLM,
    tools: list | None = None,
) -> tuple[AgentEngine, InMemoryRepository, ToolRegistry]:
    """Build an engine with mock LLM and in-memory persistence."""
    repo = InMemoryRepository()
    await repo.initialize()

    tool_registry = ToolRegistry()
    if tools:
        for t in tools:
            tool_registry.register_function(t)

    engine = AgentEngine(
        llm=mock_llm,
        tool_registry=tool_registry,
        repository=repo,
        skills_manager=SkillsManager(),
        middleware=[],
    )
    return engine, repo, tool_registry


# ══════════════════════════════════════════════════════════════════════
# Tests
# ══════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_simple_text_response():
    """Engine returns a single text response with no tool calls."""
    llm = MockLLM([make_text_response("Hello! I'm your assistant.")])
    engine, repo, _ = await build_engine(llm)

    agent = AgentDefinition(name="test", model="mock:test")

    events = []
    async for event in engine.run("s1", "Hi", agent):
        events.append(event)

    # Should have: session_started, turn_started, turn_completed, session_completed
    event_types = [e.type for e in events]
    assert EventType.SESSION_STARTED in event_types
    assert EventType.TURN_STARTED in event_types
    assert EventType.TURN_COMPLETED in event_types
    assert EventType.SESSION_COMPLETED in event_types

    # Session should be completed in DB
    session = await repo.get_session("s1")
    assert session.status == SessionStatus.COMPLETED

    # Messages: user + assistant
    messages = await repo.get_messages("s1")
    assert len(messages) == 2
    assert messages[0].role == MessageRole.USER
    assert messages[0].content == "Hi"
    assert messages[1].role == MessageRole.ASSISTANT
    assert "Hello" in messages[1].content


@pytest.mark.asyncio
async def test_tool_call_then_text():
    """Engine calls a tool, gets result, then LLM responds with text."""
    @tool(name="lookup", description="Look up info")
    async def lookup(query: str) -> dict:
        return {"answer": f"Result for: {query}"}

    llm = MockLLM([
        make_tool_call_response("lookup", {"query": "weather"}),
        make_text_response("The weather is sunny."),
    ])
    engine, repo, _ = await build_engine(llm, tools=[lookup])

    agent = AgentDefinition(name="test", model="mock:test", tools=["lookup"])

    events = []
    async for event in engine.run("s1", "What's the weather?", agent):
        events.append(event)

    event_types = [e.type for e in events]
    assert EventType.TOOL_EXECUTING in event_types
    assert EventType.TOOL_RESULT in event_types
    assert EventType.SESSION_COMPLETED in event_types

    # Messages: user, assistant(tool_call), tool_result, assistant(text)
    messages = await repo.get_messages("s1")
    assert len(messages) == 4
    assert messages[0].role == MessageRole.USER
    assert messages[1].role == MessageRole.ASSISTANT
    assert messages[1].tool_calls is not None
    assert messages[2].role == MessageRole.TOOL_RESULT
    assert "Result for: weather" in messages[2].content
    assert messages[3].role == MessageRole.ASSISTANT
    assert "sunny" in messages[3].content


@pytest.mark.asyncio
async def test_max_turns_stops_loop():
    """Engine stops after max_turns even if LLM keeps requesting tools."""
    @tool(name="infinite", description="Never stops")
    async def infinite(x: str) -> dict:
        return {"status": "ok"}

    # LLM always requests tool call
    llm = MockLLM([
        make_tool_call_response("infinite", {"x": "1"}, "tc-1"),
        make_tool_call_response("infinite", {"x": "2"}, "tc-2"),
        make_tool_call_response("infinite", {"x": "3"}, "tc-3"),
    ])
    engine, repo, _ = await build_engine(llm, tools=[infinite])

    agent = AgentDefinition(name="test", model="mock:test", tools=["infinite"], max_turns=2)

    events = []
    async for event in engine.run("s1", "Go", agent):
        events.append(event)

    # Should still complete (not crash) — just stops at max_turns
    event_types = [e.type for e in events]
    assert EventType.SESSION_COMPLETED in event_types

    session = await repo.get_session("s1")
    assert session.status == SessionStatus.COMPLETED
    assert session.turn_count == 2


@pytest.mark.asyncio
async def test_tool_error_is_reported():
    """When a tool raises an exception, the error is captured gracefully."""
    @tool(name="fail_tool", description="Always fails")
    async def fail_tool(x: str) -> dict:
        raise ValueError("Something went wrong")

    llm = MockLLM([
        make_tool_call_response("fail_tool", {"x": "boom"}),
        make_text_response("The tool failed, but I can still respond."),
    ])
    engine, repo, _ = await build_engine(llm, tools=[fail_tool])

    agent = AgentDefinition(name="test", model="mock:test", tools=["fail_tool"])

    events = []
    async for event in engine.run("s1", "Try it", agent):
        events.append(event)

    # Tool result should show error
    tool_results = [e for e in events if e.type == EventType.TOOL_RESULT]
    assert len(tool_results) == 1
    assert tool_results[0].data["status"] == "error"

    # Session should still complete (engine doesn't crash on tool errors)
    assert EventType.SESSION_COMPLETED in [e.type for e in events]


@pytest.mark.asyncio
async def test_session_persistence_across_turns():
    """Verify tokens and turn count are persisted after completion."""
    llm = MockLLM([make_text_response("Done.")])
    engine, repo, _ = await build_engine(llm)

    agent = AgentDefinition(name="test", model="mock:test")
    _ = [e async for e in engine.run("s1", "Hello", agent)]

    session = await repo.get_session("s1")
    assert session.status == SessionStatus.COMPLETED
    assert session.turn_count == 1
    assert session.total_input_tokens == 50
    assert session.total_output_tokens == 20


@pytest.mark.asyncio
async def test_context_injection_for_tools():
    """Tools that accept ctx: ExecutionContext get it injected."""
    received_ctx = {}

    @tool(name="ctx_tool", description="Needs context")
    async def ctx_tool(query: str, ctx: ExecutionContext = None) -> dict:
        received_ctx["session_id"] = ctx.session_id if ctx else None
        received_ctx["turn"] = ctx.turn_number if ctx else None
        return {"ok": True}

    llm = MockLLM([
        make_tool_call_response("ctx_tool", {"query": "test"}),
        make_text_response("Got it."),
    ])
    engine, repo, _ = await build_engine(llm, tools=[ctx_tool])

    agent = AgentDefinition(name="test", model="mock:test", tools=["ctx_tool"])
    _ = [e async for e in engine.run("s1", "Go", agent)]

    assert received_ctx["session_id"] == "s1"
    assert received_ctx["turn"] == 1


@pytest.mark.asyncio
async def test_session_resumed():
    """Running the engine on an existing session appends to it."""
    llm = MockLLM([
        make_text_response("First answer."),
        make_text_response("Second answer."),
    ])
    engine, repo, _ = await build_engine(llm)
    agent = AgentDefinition(name="test", model="mock:test")

    # First message
    _ = [e async for e in engine.run("s1", "Hello", agent)]

    # Second message on same session
    _ = [e async for e in engine.run("s1", "Follow up", agent)]

    messages = await repo.get_messages("s1")
    # Should have: user1, assistant1, user2, assistant2
    assert len(messages) == 4
    assert messages[2].role == MessageRole.USER
    assert messages[2].content == "Follow up"


@pytest.mark.asyncio
async def test_all_events_have_session_id():
    """Every emitted event must carry the correct session_id."""
    llm = MockLLM([make_text_response("Ok.")])
    engine, repo, _ = await build_engine(llm)

    agent = AgentDefinition(name="test", model="mock:test")
    events = [e async for e in engine.run("s1", "Hi", agent)]

    for event in events:
        assert event.session_id == "s1", f"{event.type} missing session_id"


# ══════════════════════════════════════════════════════════════════════
# Cancel tests
# ══════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_cancel_sets_session_cancelled():
    """engine.cancel() sets the session status to CANCELLED."""
    llm = MockLLM([make_text_response("Done.")])
    engine, repo, _ = await build_engine(llm)

    agent = AgentDefinition(name="test", model="mock:test")

    # Create a session by running the engine once
    _ = [e async for e in engine.run("s1", "Hello", agent)]

    # Cancel the session
    await engine.cancel("s1")

    session = await repo.get_session("s1")
    assert session.status == SessionStatus.CANCELLED


@pytest.mark.asyncio
async def test_cancel_cascades_to_children():
    """cancel(cascade=True) cancels the parent and all child sessions."""
    llm = MockLLM([make_text_response("Done.")])
    engine, repo, _ = await build_engine(llm)

    # Create parent session
    from fg_agents.core.types import AgentSession
    parent = AgentSession(id="parent-1", agent_id="a1", config={})
    await repo.create_session(parent)

    # Create child session linked to parent
    child = AgentSession(
        id="child-1", agent_id="a2", config={},
        parent_session_id="parent-1",
    )
    await repo.create_session(child)

    count = await engine.cancel("parent-1", cascade=True)

    parent_session = await repo.get_session("parent-1")
    child_session = await repo.get_session("child-1")
    assert parent_session.status == SessionStatus.CANCELLED
    assert child_session.status == SessionStatus.CANCELLED
    assert count == 2


@pytest.mark.asyncio
async def test_cancel_no_cascade():
    """cancel(cascade=False) cancels only the parent, not child sessions."""
    llm = MockLLM([make_text_response("Done.")])
    engine, repo, _ = await build_engine(llm)

    from fg_agents.core.types import AgentSession
    parent = AgentSession(id="parent-2", agent_id="a1", config={})
    await repo.create_session(parent)

    child = AgentSession(
        id="child-2", agent_id="a2", config={},
        parent_session_id="parent-2",
    )
    await repo.create_session(child)

    count = await engine.cancel("parent-2", cascade=False)

    parent_session = await repo.get_session("parent-2")
    child_session = await repo.get_session("child-2")
    assert parent_session.status == SessionStatus.CANCELLED
    assert child_session.status != SessionStatus.CANCELLED
    assert count == 1


# ── Premature-stop guard: require_explicit_completion / min_turns ────────────

@pytest.mark.asyncio
async def test_require_explicit_completion_nudges_premature_stop():
    """With require_explicit_completion, a no-tool-call turn does NOT finalize;
    the agent is nudged to continue, bounded by the consecutive-nudge cap (3)."""
    llm = MockLLM([make_text_response("just narrating, not calling a tool")])
    engine, repo, _ = await build_engine(llm)
    agent = AgentDefinition(
        name="t", model="mock:test", require_explicit_completion=True, max_turns=10,
    )

    events = [e async for e in engine.run("s1", "Go", agent)]
    assert EventType.SESSION_COMPLETED in [e.type for e in events]

    msgs = await repo.get_messages("s1")
    assistant_turns = [m for m in msgs if m.role == MessageRole.ASSISTANT]
    nudges = [m for m in msgs if m.role == MessageRole.USER and "without calling a tool" in (m.content or "")]
    # 1 initial stop + 3 nudges = 4 assistant turns, then the cap finalizes it.
    assert len(assistant_turns) == 4, f"expected 4 turns, got {len(assistant_turns)}"
    assert len(nudges) == 3, f"expected 3 nudges, got {len(nudges)}"
    # Bounded — did NOT loop to max_turns.
    assert len(assistant_turns) < agent.max_turns


@pytest.mark.asyncio
async def test_default_still_stops_on_first_text_turn():
    """Backward compat: without the flag, a no-tool-call turn finalizes after 1
    turn (standard ReAct), unchanged."""
    llm = MockLLM([make_text_response("done")])
    engine, repo, _ = await build_engine(llm)
    agent = AgentDefinition(name="t", model="mock:test")  # defaults

    _ = [e async for e in engine.run("s1", "Go", agent)]
    assistant_turns = [m for m in await repo.get_messages("s1") if m.role == MessageRole.ASSISTANT]
    assert len(assistant_turns) == 1


@pytest.mark.asyncio
async def test_min_turns_prevents_early_stop():
    """min_turns (previously a dead field) now keeps the loop going until the
    minimum is met, even on no-tool-call turns."""
    llm = MockLLM([make_text_response("narrating")])
    engine, repo, _ = await build_engine(llm)
    agent = AgentDefinition(name="t", model="mock:test", min_turns=3, max_turns=10)

    _ = [e async for e in engine.run("s1", "Go", agent)]
    assistant_turns = [m for m in await repo.get_messages("s1") if m.role == MessageRole.ASSISTANT]
    assert len(assistant_turns) == 3, f"expected 3 turns, got {len(assistant_turns)}"


@pytest.mark.asyncio
async def test_resume_after_cancel_generates_again():
    """STOP is a LOCAL interrupt: after a session is cancelled, a NEW user
    message must run normally — not get instantly re-cancelled by a stale flag
    or a stuck CANCELLED status."""
    llm = MockLLM([make_text_response("first answer")])
    engine, repo, _ = await build_engine(llm)
    agent = AgentDefinition(name="t", model="mock:test")

    # First run completes.
    _ = [e async for e in engine.run("s1", "hello", agent)]

    # Simulate an aborted run that left a cancel (seq) + CANCELLED status behind
    # (what an immediate producer.cancel does).
    await engine.cancel("s1")
    assert engine._cancel_seq.get("s1", 0) > 0
    sess = await repo.get_session("s1")
    assert sess.status == SessionStatus.CANCELLED

    # New message must generate, not be swallowed or instantly re-cancelled: the
    # new run's baseline == the leftover cancel seq, so the cooperative check
    # (seq > baseline) is false and the run proceeds normally.
    llm._responses = [make_text_response("second answer")]
    llm._call_count = 0
    events = [e async for e in engine.run("s1", "are you there?", agent)]
    assert EventType.SESSION_COMPLETED in [e.type for e in events]
    msgs = await repo.get_messages("s1")
    assert any("second answer" in (m.content or "") for m in msgs if m.role == MessageRole.ASSISTANT)
