"""
Tests for the one-shot free functions (ask/stream) and model auto-detection.
"""
import pytest

import fg_agents.model_detection as model_detection
from fg_agents import (
    Agent,
    AgentRunResult,
    ModelDetectionError,
    ask,
    resolve_default_model,
    stream,
)
from fg_agents.core.types import EventType
from tests.helpers import MockLLM, make_text_response

_ENV_VARS = ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GOOGLE_API_KEY", "GEMINI_API_KEY")


@pytest.fixture
def clean_env(monkeypatch):
    for var in _ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(model_detection, "_probe_cached_at", None)
    monkeypatch.setattr(model_detection, "_probe_ollama_sync", lambda: False)

    async def no_async_probe():
        return False

    monkeypatch.setattr(model_detection, "_probe_ollama_async", no_async_probe)
    return monkeypatch


@pytest.fixture
def capture_agent(monkeypatch):
    """Spy on Agent construction so tests can inspect the ephemeral agent."""
    captured = {}
    original_init = Agent.__init__

    def spy_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        captured["agent"] = self

    monkeypatch.setattr(Agent, "__init__", spy_init)
    return captured


# ══════════════════════════════════════════════════════════════════════
# ask()
# ══════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_ask_happy_path():
    llm = MockLLM([make_text_response("4")])
    result = await ask("What's 2+2?", model="mock:test", llm=llm)
    assert isinstance(result, AgentRunResult)
    assert result.text == "4"
    assert str(result) == "4"  # print(await ask(...)) prints just the answer


@pytest.mark.asyncio
async def test_ask_cleans_up_repository(capture_agent):
    llm = MockLLM([make_text_response("done")])
    await ask("hi", model="mock:test", llm=llm)
    agent = capture_agent["agent"]
    assert agent._closed
    with pytest.raises(RuntimeError, match="closed"):
        await agent.run("again")


@pytest.mark.asyncio
async def test_ask_passes_kwargs_through():
    llm = MockLLM([make_text_response("terse")])
    result = await ask(
        "hi", model="mock:test", llm=llm, system_prompt="Be terse.", max_turns=3
    )
    assert result.text == "terse"


# ══════════════════════════════════════════════════════════════════════
# stream()
# ══════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_stream_free_function_yields_events():
    llm = MockLLM([make_text_response("hello there")])
    events = [e async for e in stream("hi", model="mock:test", llm=llm)]
    assert events, "expected at least one event"
    types = {e.type for e in events}
    assert EventType.SESSION_COMPLETED in types
    completed = next(e for e in events if e.type == EventType.SESSION_COMPLETED)
    assert completed.data.get("final_output") == "hello there"


@pytest.mark.asyncio
async def test_stream_exhaustion_closes_agent(capture_agent):
    llm = MockLLM([make_text_response("bye")])
    async for _ in stream("hi", model="mock:test", llm=llm):
        pass
    assert capture_agent["agent"]._closed


@pytest.mark.asyncio
async def test_stream_early_aclose_closes_agent(capture_agent):
    llm = MockLLM([make_text_response("long story")])
    gen = stream("hi", model="mock:test", llm=llm)
    first = await anext(gen)
    assert first is not None
    await gen.aclose()
    assert capture_agent["agent"]._closed


# ══════════════════════════════════════════════════════════════════════
# Agent as async context manager
# ══════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_agent_context_manager_enter_exit():
    llm = MockLLM([make_text_response("ok")])
    async with Agent(model="mock:test", llm=llm) as agent:
        assert isinstance(agent, Agent)
        result = await agent.run("hi")
        assert result.text == "ok"
    assert agent._closed
    with pytest.raises(RuntimeError, match="closed"):
        await agent.run("again")


@pytest.mark.asyncio
async def test_agent_context_manager_closes_on_exception():
    agent = Agent(model="mock:test", llm=MockLLM([make_text_response("x")]))
    with pytest.raises(ValueError):
        async with agent:
            raise ValueError("boom")
    assert agent._closed


# ══════════════════════════════════════════════════════════════════════
# Model auto-detection
# ══════════════════════════════════════════════════════════════════════

def test_detection_prefers_anthropic(clean_env):
    clean_env.setenv("ANTHROPIC_API_KEY", "sk-ant")
    clean_env.setenv("OPENAI_API_KEY", "sk-oai")
    clean_env.setenv("GOOGLE_API_KEY", "g-key")
    assert resolve_default_model() == "anthropic:claude-sonnet-4-6"


def test_detection_openai_second(clean_env):
    clean_env.setenv("OPENAI_API_KEY", "sk-oai")
    clean_env.setenv("GOOGLE_API_KEY", "g-key")
    assert resolve_default_model() == "openai:gpt-5.2"


def test_detection_google_third(clean_env):
    clean_env.setenv("GOOGLE_API_KEY", "g-key")
    assert resolve_default_model() == "google:gemini-2.5-flash"


def test_detection_gemini_alias(clean_env):
    clean_env.setenv("GEMINI_API_KEY", "g-key")
    assert resolve_default_model() == "google:gemini-2.5-flash"
    assert model_detection.env_api_key_overrides() == {"google": "g-key"}


def test_detection_ollama_fallback(clean_env):
    clean_env.setattr(model_detection, "_probe_ollama_sync", lambda: True)
    assert resolve_default_model() == "ollama:qwen3:8b"


@pytest.mark.asyncio
async def test_async_detection_never_calls_sync_probe(clean_env):
    def sync_probe_forbidden():
        raise AssertionError("sync socket probe must not run on the async path")

    async def async_probe():
        return True

    clean_env.setattr(model_detection, "_probe_ollama_sync", sync_probe_forbidden)
    clean_env.setattr(model_detection, "_probe_ollama_async", async_probe)
    assert await model_detection.resolve_default_model_async() == "ollama:qwen3:8b"


@pytest.mark.asyncio
async def test_ask_resolves_model_via_async_probe(clean_env, capture_agent):
    def sync_probe_forbidden():
        raise AssertionError("sync socket probe must not run inside ask()")

    async def async_probe():
        return True

    clean_env.setattr(model_detection, "_probe_ollama_sync", sync_probe_forbidden)
    clean_env.setattr(model_detection, "_probe_ollama_async", async_probe)
    llm = MockLLM([make_text_response("ok")])
    result = await ask("hi", llm=llm)
    assert result.text == "ok"
    assert capture_agent["agent"].definition.model == "ollama:qwen3:8b"


def test_probe_result_cached(clean_env):
    calls = {"n": 0}

    def counting_probe():
        calls["n"] += 1
        return True

    clean_env.setattr(model_detection, "_probe_ollama_sync", counting_probe)
    assert resolve_default_model() == "ollama:qwen3:8b"
    assert resolve_default_model() == "ollama:qwen3:8b"
    assert calls["n"] == 1  # second call served from the cache


def test_probe_cache_expires(clean_env):
    calls = {"n": 0}
    clock = {"now": 1000.0}

    def counting_probe():
        calls["n"] += 1
        return True

    clean_env.setattr(model_detection, "_probe_ollama_sync", counting_probe)
    clean_env.setattr(model_detection.time, "monotonic", lambda: clock["now"])
    resolve_default_model()
    clock["now"] += model_detection._PROBE_CACHE_TTL - 1
    resolve_default_model()
    assert calls["n"] == 1  # still within TTL
    clock["now"] += 2
    resolve_default_model()
    assert calls["n"] == 2  # TTL elapsed — re-probed


def test_detection_friendly_error(clean_env):
    with pytest.raises(ModelDetectionError) as exc:
        resolve_default_model()
    msg = str(exc.value)
    for needle in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GOOGLE_API_KEY", "Ollama"):
        assert needle in msg


def test_agent_uses_detected_model(clean_env):
    clean_env.setenv("ANTHROPIC_API_KEY", "sk-ant")
    agent = Agent()
    assert agent.definition.model == "anthropic:claude-sonnet-4-6"


def test_agent_explicit_model_wins(clean_env):
    clean_env.setenv("ANTHROPIC_API_KEY", "sk-ant")
    agent = Agent(model="mock:test")
    assert agent.definition.model == "mock:test"
