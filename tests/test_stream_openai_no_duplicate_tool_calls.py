"""Regression: _stream_openai must emit each tool call exactly once.

Some OpenAI-compatible providers (notably via OpenRouter, and with
stream_options.include_usage) send more than one chunk carrying a
finish_reason plus a trailing usage-only chunk. The old parser flushed the
tool-call buffers inside the chunk loop on every finish_reason chunk, so each
buffered tool call was emitted repeatedly and the engine executed each tool
twice. This test drives the parser with such a stream and asserts a single
tool_call_end per call.
"""

import asyncio
from types import SimpleNamespace
import pytest

from fg_agents.core.llm import AgentLLM, _ThinkSplitter


def _delta(content=None, tool_calls=None):
    return SimpleNamespace(content=content, tool_calls=tool_calls,
                           reasoning_content=None, reasoning=None)


def _tc_delta(index, tid=None, name=None, args=None):
    fn = SimpleNamespace(name=name, arguments=args)
    return SimpleNamespace(index=index, id=tid, function=fn)


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


def test_stream_openai_emits_each_tool_call_once():
    # Two tool calls streamed, then a chunk with finish_reason=tool_calls,
    # then a SECOND finish_reason chunk (duplicate), then a trailing
    # usage-only chunk (choices=[]). The buggy parser emitted 4 tool_call_end
    # events (2 calls x 2 finish_reason chunks); the fixed parser emits 2.
    chunks = [
        _chunk(_delta(tool_calls=[_tc_delta(0, tid="call_a", name="manage_watch", args='{"x":1}')])),
        _chunk(_delta(tool_calls=[_tc_delta(1, tid="call_b", name="book_pnl", args='{}')])),
        _chunk(finish_reason="tool_calls"),
        _chunk(finish_reason="tool_calls"),  # duplicate terminal chunk
        _chunk(usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5,
                                     total_tokens=15, prompt_tokens_details=None)),
    ]

    llm = AgentLLM()

    async def _run():
        async def _fake_get_client(provider):
            return _FakeClient(chunks)
        llm._get_client = _fake_get_client  # type: ignore[assignment]
        out = []
        async for c in llm._stream_openai([], None, "gpt-oss", "", 0.0, 256, "cerebras"):
            out.append(c)
        return out

    events = asyncio.run(_run())

    ends = [e for e in events if e.type == "tool_call_end"]
    assert len(ends) == 2, f"expected 2 tool_call_end, got {len(ends)}"
    ids = sorted(tc.tool_call.id for tc in ends)
    assert ids == ["call_a", "call_b"], ids

    usage_events = [e for e in events if e.type == "usage"]
    assert len(usage_events) == 1, f"expected 1 usage event, got {len(usage_events)}"
    assert usage_events[0].usage.total_tokens == 15


@pytest.mark.parametrize('with_choice', [False, True])
@pytest.mark.parametrize('tool_call', [False, True])
@pytest.mark.parametrize('updated_trailer', [False, True])
def test_usage_snapshots_are_independent_of_choices(with_choice, tool_call, updated_trailer):
    usage = SimpleNamespace(prompt_tokens=28, completion_tokens=4, total_tokens=32,
                            prompt_tokens_details=SimpleNamespace(cached_tokens=6))
    content = _delta(tool_calls=[_tc_delta(0, tid='call_one', name='lookup', args='{}')]) if tool_call else _delta(content='OK')
    finish = 'tool_calls' if tool_call else 'stop'
    chunks = [_chunk(content, usage=usage if with_choice else None)]
    # Repeated finish/usage chunks must not duplicate either tool calls or
    # the bill. A later authoritative total replaces, rather than adds.
    chunks.extend([_chunk(finish_reason=finish if with_choice else None, usage=usage)] * 2)
    if updated_trailer:
        chunks.append(_chunk(usage=SimpleNamespace(prompt_tokens=30, completion_tokens=5,
            total_tokens=35, prompt_tokens_details=SimpleNamespace(cached_tokens=7))))
    async def run():
        llm = AgentLLM()
        async def client(provider):
            return _FakeClient(chunks)
        llm._get_client = client
        return [c async for c in llm._stream_openai([], None, 'model', '', 0.0, 64, 'openrouter')]
    events = asyncio.run(run())
    usages = [event.usage for event in events if event.type == 'usage']
    assert len(usages) == 1
    assert usages[0].model_dump() == dict(input_tokens=30 if updated_trailer else 28,
        output_tokens=5 if updated_trailer else 4, total_tokens=35 if updated_trailer else 32,
        cache_read_tokens=7 if updated_trailer else 6, cache_creation_tokens=0)
    assert len([event for event in events if event.type == 'tool_call_end']) == int(tool_call)
    assert ''.join(event.text or '' for event in events if event.type == 'text_delta') == ('' if tool_call else 'OK')


def _run_splitter(chunks):
    """Feed chunks through a splitter; return concatenated (thinking, text)."""
    sp = _ThinkSplitter()
    thinking, text = "", ""
    segs = []
    for c in chunks:
        segs.extend(sp.feed(c))
    segs.extend(sp.flush())
    for kind, seg in segs:
        if kind == "thinking":
            thinking += seg
        else:
            text += seg
    return thinking, text


def test_think_splitter_basic():
    thinking, text = _run_splitter(["<think>reasoning here</think>", "the answer"])
    assert thinking == "reasoning here"
    assert text == "the answer"


def test_think_splitter_tag_split_across_chunks():
    # Tags straddle chunk boundaries in every awkward place.
    thinking, text = _run_splitter(["<thi", "nk>hidden", " thoughts</thi", "nk>visible", " answer"])
    assert thinking == "hidden thoughts"
    assert text == "visible answer"


def test_think_splitter_no_tags_is_passthrough():
    thinking, text = _run_splitter(["plain ", "answer ", "text"])
    assert thinking == ""
    assert text == "plain answer text"


def test_think_splitter_lone_angle_bracket_not_held_forever():
    # A '<' that is not the start of a think tag must still be emitted.
    thinking, text = _run_splitter(["a < b < c"])
    assert thinking == ""
    assert text == "a < b < c"
