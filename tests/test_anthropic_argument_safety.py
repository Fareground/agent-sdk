"""Drive native provider wire events without credentials or network calls."""
from types import SimpleNamespace

import pytest

from fg_agents.core.llm import AgentLLM
from fg_agents.core.types import AgentDefinition, MessageRole, StopReason
from fg_agents.tools.decorators import tool
from tests.helpers import MockLLM, make_text_response
from tests.test_engine import build_engine


class NativeStream:
    def __init__(self, fragments, stop='tool_use', closed=True):
        self.fragments, self.stop, self.closed = fragments, stop, closed

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    def __aiter__(self):
        async def events():
            yield SimpleNamespace(type='content_block_start', content_block=SimpleNamespace(type='tool_use', id='call1', name='publish'))
            for fragment in self.fragments:
                yield SimpleNamespace(type='content_block_delta', delta=SimpleNamespace(type='input_json_delta', partial_json=fragment))
            if self.closed:
                yield SimpleNamespace(type='content_block_stop')
        return events()

    async def get_final_message(self):
        return SimpleNamespace(stop_reason=self.stop, usage=SimpleNamespace(input_tokens=10, output_tokens=20))


async def collect(fragments, stop='tool_use', closed=True):
    llm = AgentLLM()
    async def client(provider):
        assert provider == 'anthropic'
        return SimpleNamespace(messages=SimpleNamespace(stream=lambda **kwargs: NativeStream(fragments, stop, closed)))
    llm._get_client = client
    return [event async for event in llm._stream_anthropic([], None, 'model', '', 0, 32)]


@pytest.mark.asyncio
async def test_malformed_native_json_is_not_an_empty_default_action():
    # The native SDK deliberately allows partial input while streaming. Such
    # a partial object can reach block_stop; the adapter must still be strict.
    jiter = pytest.importorskip('jiter')
    assert jiter.from_json(b'{"secret":"PRIVATE_VALUE","payload":', partial_mode=True) == {'secret': 'PRIVATE_VALUE'}
    events = await collect(['{"secret":"PRIVATE_VALUE",', '"payload":'])
    call = next(event.tool_call for event in events if event.type == 'tool_call_end')
    assert call.arguments_error, 'Malformed native JSON must not execute as {}'
    assert call.arguments == {}
    assert 'PRIVATE_VALUE' not in call.model_dump_json()
    assert call.arguments_diagnostic.code == 'invalid_json'
    assert call.arguments_diagnostic.fragments == 2
    assert call.arguments_diagnostic.stop_reason == StopReason.TOOL_USE


@pytest.mark.parametrize('fragments,stop,closed,code', [
    (['{"payload":'], 'max_tokens', True, 'invalid_json'),
    ([], 'max_tokens', True, 'missing_arguments'),
    (['[]'], 'tool_use', True, 'non_object'),
    (['null'], 'tool_use', True, 'non_object'),
    (['{}'], 'tool_use', False, 'incomplete_stream'),
    (['{"é":', '"a\\', '"b"}'], 'tool_use', True, None),
    (['{}'], 'max_tokens', True, None),
    ([], 'tool_use', True, None),
])
@pytest.mark.asyncio
async def test_native_tool_contract_and_usage(fragments, stop, closed, code):
    events = await collect(fragments, stop, closed)
    calls = [event.tool_call for event in events if event.type == 'tool_call_end']
    assert len(calls) == 1 and calls[0].id == 'call1'
    call = calls[0]
    if code:
        assert call.arguments_error and call.arguments_diagnostic.code == code
        assert call.arguments == {}
        assert call.arguments_diagnostic.characters == len(''.join(fragments))
    else:
        assert call.arguments_error is None and call.arguments_diagnostic is None
        assert call.arguments == ({'é': 'a"b'} if len(fragments) == 3 else {})
    usage = [event for event in events if event.type == 'usage']
    assert len(usage) == 1 and usage[0].usage.total_tokens == 30
    assert usage[0].stop_reason == (StopReason.MAX_TOKENS if stop == 'max_tokens' else StopReason.TOOL_USE)


@pytest.mark.asyncio
async def test_native_failure_never_executes_default_and_persists_safe_diagnostics():
    effects = []
    @tool()
    async def publish(payload: dict | None = None) -> str:
        effects.append(payload)
        return 'published'
    broken = await collect(['{"secret":"PRIVATE_VALUE",', '"payload":'])
    llm = MockLLM([broken, make_text_response('Stopped')])
    engine, repo, _ = await build_engine(llm, tools=[publish])
    agent = AgentDefinition(name='test', model='mock:test', tools=['publish'], max_turns=2)
    _ = [event async for event in engine.run('native', 'Publish', agent)]
    assert effects == []
    messages = await repo.get_messages('native')
    saved = next(message for message in messages if message.role == MessageRole.ASSISTANT and message.tool_calls)
    assert saved.tool_calls[0].arguments_diagnostic.code == 'invalid_json'
    assert 'PRIVATE_VALUE' not in saved.model_dump_json()
    result = next(message for message in messages if message.role == MessageRole.TOOL_RESULT)
    assert 'not executed' in result.content and 'line 1' in result.content


@pytest.mark.parametrize('stage', ['iterate', 'final'])
@pytest.mark.parametrize('message', [
    'Unable to parse tool parameter JSON from model. JSON: PRIVATE_VALUE rate limit',
    'New SDK wording: PRIVATE_VALUE context length',
])
@pytest.mark.asyncio
async def test_native_sdk_parse_error_does_not_echo_private_tool_json(stage, message):
    from fg_agents.core.errors import LLMError

    class BrokenStream(NativeStream):
        def __aiter__(self):
            async def events():
                if stage == 'iterate':
                    raise ValueError(message)
                async for event in super(BrokenStream, self).__aiter__():
                    yield event
            return events()

        async def get_final_message(self):
            raise ValueError(message)

    llm = AgentLLM()
    async def client(provider):
        return SimpleNamespace(messages=SimpleNamespace(stream=lambda **kwargs: BrokenStream([])))
    llm._get_client = client
    with pytest.raises(LLMError, match='could not be decoded') as caught:
        _ = [event async for event in llm._stream_anthropic([], None, 'model', '', 0, 32)]
    assert 'PRIVATE_VALUE' not in str(caught.value)
    assert caught.value.retryable is False
    # No original exception is retained for downstream traceback/telemetry
    # serializers to accidentally expose, even if they ignore suppress_context.
    assert caught.value.__cause__ is None and caught.value.__context__ is None


@pytest.mark.asyncio
async def test_multiple_native_blocks_preserve_good_calls_and_reject_only_bad_call():
    class MultipleStream(NativeStream):
        def __aiter__(self):
            async def events():
                blocks = [['{"payload":', '{"id":1}}'], ['{"secret":"PRIVATE_VALUE","payload":'], ['{"payload":{"id":2}}']]
                for index, fragments in enumerate(blocks):
                    async for event in NativeStream(fragments):
                        if event.type == 'content_block_start':
                            event.content_block.id = f'call-{index}'
                        yield event
            return events()

    llm = AgentLLM()
    async def client(provider):
        return SimpleNamespace(messages=SimpleNamespace(stream=lambda **kwargs: MultipleStream([])))
    llm._get_client = client
    events = [event async for event in llm._stream_anthropic([], None, 'model', '', 0, 32)]
    calls = [event.tool_call for event in events if event.type == 'tool_call_end']
    assert [call.id for call in calls] == ['call-0', 'call-1', 'call-2']
    assert calls[0].arguments == {'payload': {'id': 1}}
    assert calls[1].arguments_error and calls[1].arguments == {}
    assert calls[2].arguments == {'payload': {'id': 2}}
    assert [event.usage.total_tokens for event in events if event.type == 'usage'] == [30]
    effects = []
    @tool()
    async def publish(payload: dict | None = None) -> str:
        effects.append(payload)
        return 'published'
    engine, _, _ = await build_engine(MockLLM([events, make_text_response('Done')]), tools=[publish])
    agent = AgentDefinition(name='test', model='mock:test', tools=['publish'], max_turns=2)
    _ = [event async for event in engine.run('multiple', 'Publish', agent)]
    assert effects == [{'id': 1}, {'id': 2}]
