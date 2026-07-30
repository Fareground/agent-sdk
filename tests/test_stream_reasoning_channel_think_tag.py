"""Regression: spurious inline <think> must not swallow the answer when the
model reasons on the dedicated channel.

Qwen3.7-Plus on Together streams its chain-of-thought via delta.reasoning AND
prefixes the content channel with an unclosed "<think>\n" (chat-template
artifact). The old parser fed content through the _ThinkSplitter, entered
think-mode on the orphan tag, and routed the entire visible answer into the
thinking channel — the stored assistant message had no text. When the
reasoning channel is active, literal think tags in content are stripped and
content streams as text.
"""

import asyncio
from types import SimpleNamespace

from fg_agents.core.llm import AgentLLM


def _delta(content=None, reasoning=None):
    return SimpleNamespace(content=content, tool_calls=None,
                           reasoning_content=None, reasoning=reasoning)


def _chunk(delta=None, finish_reason=None, usage=None):
    choices = [] if delta is None and finish_reason is None else [
        SimpleNamespace(delta=delta or _delta(), finish_reason=finish_reason)
    ]
    return SimpleNamespace(choices=choices, usage=usage)


class _FakeStream:
    def __init__(self, chunks):
        self._chunks = chunks

    def __aiter__(self):
        async def gen():
            for c in self._chunks:
                yield c
        return gen()


class _FakeCompletions:
    def __init__(self, chunks):
        self._chunks = chunks

    async def create(self, **kwargs):
        return _FakeStream(self._chunks)


class _FakeClient:
    def __init__(self, chunks):
        self.chat = SimpleNamespace(completions=_FakeCompletions(chunks))


def _run_stream(chunks):
    llm = AgentLLM()

    async def _run():
        async def _fake_get_client(provider):
            return _FakeClient(chunks)
        llm._get_client = _fake_get_client  # type: ignore[assignment]
        out = []
        async for c in llm._stream_openai([], None, "Qwen/Qwen3.7-Plus", "", 0.0, 256, "together"):
            out.append(c)
        return out

    return asyncio.run(_run())


def test_reasoning_channel_strips_spurious_think_tag():
    chunks = [
        _chunk(_delta(reasoning="Thinking Process:\n1. Identify the question.")),
        _chunk(_delta(content="<think>\n")),
        _chunk(_delta(content="2 + 2 equals 4.")),
        _chunk(finish_reason="stop"),
    ]
    out = _run_stream(chunks)
    text = "".join(c.text or "" for c in out if c.type == "text_delta")
    thinking = "".join(c.text or "" for c in out if c.type == "thinking_delta")
    assert text == "\n2 + 2 equals 4."
    assert "<think>" not in text
    assert "Thinking Process" in thinking
    assert "equals 4" not in thinking


def test_inline_think_splitter_still_works_without_reasoning_channel():
    # gpt-oss style: no reasoning channel, real inline <think>...</think>.
    chunks = [
        _chunk(_delta(content="<think>plan the answer</think>")),
        _chunk(_delta(content="The answer is 4.")),
        _chunk(finish_reason="stop"),
    ]
    out = _run_stream(chunks)
    text = "".join(c.text or "" for c in out if c.type == "text_delta")
    thinking = "".join(c.text or "" for c in out if c.type == "thinking_delta")
    assert text == "The answer is 4."
    assert thinking == "plan the answer"
