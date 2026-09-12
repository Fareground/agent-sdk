"""Incomplete model arguments must never invoke a tool with invented defaults."""
import asyncio
from types import SimpleNamespace

import pytest

from fg_agents.core.llm import AgentLLM
from fg_agents.core.types import (
    AgentDefinition,
    LLMStreamChunk,
    LLMUsage,
    MessageRole,
    StopReason,
    ToolCall,
)
from fg_agents.tools.decorators import tool
from tests.helpers import MockLLM, make_text_response, make_tool_call_response
from tests.test_engine import build_engine
from tests.test_stream_openai_no_duplicate_tool_calls import _chunk, _delta, _FakeClient, _tc_delta


@pytest.mark.parametrize('raw,finish,error', [
    ('{"payload":', 'length', 'cut off'),
    ('', 'length', 'cut off'),
    ('{"payload":', 'tool_calls', 'valid JSON'),
    ('{"payload":', None, 'valid JSON'),
    ('[]', 'tool_calls', 'valid JSON'),
    ('null', 'tool_calls', 'valid JSON'),
    ('{}', 'length', None),
    ('', 'tool_calls', None),
])
def test_stream_preserves_argument_failures_and_valid_parallel_calls(raw, finish, error):
    chunks = [
        _chunk(_delta(tool_calls=[_tc_delta(0, 'good', 'lookup', '{"x":1}')])),
        _chunk(_delta(tool_calls=[_tc_delta(1, 'candidate', 'publish', raw)])),
        _chunk(finish_reason=finish),
        _chunk(usage=SimpleNamespace(prompt_tokens=100, completion_tokens=4096,
                                    total_tokens=4196, prompt_tokens_details=None)),
    ]
    async def run():
        llm = AgentLLM()
        async def client(provider):
            return _FakeClient(chunks)
        llm._get_client = client
        return [c async for c in llm._stream_openai([], None, 'model', '', .2, 4096, 'openrouter')]
    events = asyncio.run(run())
    calls = [event.tool_call for event in events if event.type == 'tool_call_end']
    assert len(calls) == 2
    assert calls[0].arguments == {'x': 1} and calls[0].arguments_error is None
    assert calls[1].arguments == {}
    if error:
        assert error in calls[1].arguments_error
        diagnostic = calls[1].arguments_diagnostic
        assert diagnostic.characters == len(raw)
        assert diagnostic.fragments == int(bool(raw))
        assert diagnostic.stop_reason == (StopReason.MAX_TOKENS if finish == 'length' else StopReason.TOOL_USE if finish else StopReason.END_TURN)
    else:
        assert calls[1].arguments_error is None
    usage = [event for event in events if event.type == 'usage']
    assert len(usage) == 1 and usage[0].usage.total_tokens == 4196
    assert usage[0].stop_reason == (StopReason.MAX_TOKENS if finish == 'length' else StopReason.TOOL_USE if finish else StopReason.END_TURN)


def test_wire_fragments_reassemble_once_with_content_free_multiline_location():
    import json

    from fg_agents.core.tool_arguments import decode_tool_call
    from fg_agents.core.types import ToolCall

    raw = '{\n"secret":"PRIVATE_VALUE",\n"payload":}'
    try:
        json.loads(raw)
    except json.JSONDecodeError as error:
        expected = (error.lineno, error.colno, error.pos)
    # Every split preserves the same parser coordinates and call identity.
    for split in range(len(raw) + 1):
        call = decode_tool_call('one', 'publish', [raw[:split], raw[split:]], StopReason.TOOL_USE)
        diagnostic = call.arguments_diagnostic
        assert (diagnostic.line, diagnostic.column, diagnostic.position) == expected
        assert diagnostic.fragments == 2 and diagnostic.characters == len(raw)
        assert 'PRIVATE_VALUE' not in call.model_dump_json()
        assert len(call.arguments_error) < 300
        assert ToolCall.model_validate_json(call.model_dump_json()) == call


@pytest.mark.asyncio
async def test_engine_reports_truncation_without_executing_or_repeating_side_effects():
    effects = []
    @tool()
    async def lookup() -> str:
        effects.append('lookup')
        return 'Recorded once'
    @tool()
    async def publish(payload: dict | None = None) -> str:
        effects.append(payload)
        return 'Published'
    incomplete = [
        LLMStreamChunk(type='tool_call_end', tool_call=ToolCall(id='valid', tool_name='lookup')),
        LLMStreamChunk(type='tool_call_end', tool_call=ToolCall(id='invalid', tool_name='publish',
            arguments_error='Tool arguments were cut off by the output token limit. Re-issue a smaller complete JSON object.')),
        LLMStreamChunk(type='usage', usage=LLMUsage(input_tokens=100, output_tokens=4096, total_tokens=4196),
                       stop_reason=StopReason.MAX_TOKENS),
    ]
    llm = MockLLM([incomplete, make_tool_call_response('publish', {'payload': {'ready': True}}, 'fixed'),
                   make_text_response('Done')])
    engine, repo, _ = await build_engine(llm, tools=[lookup, publish])
    agent = AgentDefinition(name='test', model='mock:test', tools=['lookup', 'publish'], max_turns=4)
    _ = [event async for event in engine.run('s1', 'Inspect and publish', agent)]
    assert effects == ['lookup', {'ready': True}]
    messages = await repo.get_messages('s1')
    failures = [m for m in messages if m.role == MessageRole.TOOL_RESULT and m.tool_call_id == 'invalid']
    assert len(failures) == 1
    assert 'cut off' in failures[0].content and 'not executed' in failures[0].content
    assert 'must not be repeated' in failures[0].content
    saved_call = next(m.tool_calls[1] for m in messages if m.role == MessageRole.ASSISTANT and m.tool_calls and len(m.tool_calls) == 2)
    assert saved_call.arguments_error and 'cut off' in saved_call.arguments_error
    assert llm._call_count == 3
    session = await repo.get_session('s1')
    assert session.total_input_tokens == 210
    assert session.total_output_tokens == 4146


@pytest.mark.asyncio
async def test_repeated_invalid_arguments_remain_bounded_without_running_optional_tool():
    effects = []
    @tool()
    async def publish() -> str:
        effects.append('unsafe default')
        return 'Published'
    llm = MockLLM([[
        LLMStreamChunk(type='tool_call_end', tool_call=ToolCall(tool_name='publish', arguments_error='Incomplete JSON')),
        LLMStreamChunk(type='usage', usage=LLMUsage(input_tokens=100, output_tokens=20), stop_reason=StopReason.MAX_TOKENS),
    ]])
    engine, _, _ = await build_engine(llm, tools=[publish])
    agent = AgentDefinition(name='test', model='mock:test', tools=['publish'], max_turns=3)
    _ = [event async for event in engine.run('s1', 'Publish', agent)]
    assert effects == [] and llm._call_count == 3
