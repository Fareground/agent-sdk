"""Project tool text in storage, never clip instructions or pretend it is full."""
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock

import pytest

from fg_agents import AgentMessage, MessageRole, ToolCall
from tests.test_archive_reads import archive as archive_fixture

archive = archive_fixture


async def test_content_windows_keep_unicode_order_and_scoped_originals(archive, monkeypatch):
    repo, own, foreign = archive
    when = datetime.now(UTC)
    text = '🎯préfix "quoted"\\line\n' * 1200 + 'tail 終点'
    messages = [
        AgentMessage(session_id=own, role=MessageRole.USER, content=text),
        AgentMessage(session_id=own, role=MessageRole.ASSISTANT, content=text,
                     tool_calls=[ToolCall(id='tool', tool_name='get', arguments={'full': text})]),
        AgentMessage(session_id=own, role=MessageRole.TOOL_RESULT, tool_call_id='tool', tool_name='get', content=text),
        AgentMessage(session_id=own, role=MessageRole.TOOL_RESULT, tool_name='load_skill', content=text),
        AgentMessage(session_id=own, role=MessageRole.TOOL_RESULT, tool_name='get', content='small'),
        AgentMessage(session_id=own, role=MessageRole.TOOL_RESULT, tool_name='get', content=[{'text': text}]),
    ]
    for i, message in enumerate(messages):
        message.created_at = when + timedelta(seconds=i)
        await repo.add_message(message)
    await repo.add_message(AgentMessage(session_id=own, role=MessageRole.USER, content='hidden', is_summarized=True))
    await repo.add_message(AgentMessage(session_id=foreign, role=MessageRole.USER, content='private'))
    original = await repo.get_messages(own)
    if type(repo).__name__ != 'InMemoryRepository':
        monkeypatch.setattr(repo, 'get_messages', AsyncMock(side_effect=AssertionError('All body read')))
    windows = await repo.get_context_windows(own, head_chars=500, tail_chars=40, exempt_tools=('load_skill',))
    assert [w.message.id for w in windows] == [m.id for m in original]
    for i, window in enumerate(windows):
        if i == 2:
            assert window.partial and window.message.content == ''
            assert window.total_characters == len(text)
            assert window.head == text[:500] and window.tail == text[-40:]
            assert (await repo.get_message(own, window.message.id)).content == text
            assert await repo.get_message(foreign, window.message.id) is None
        else:
            assert not window.partial and window.message.model_dump() == original[i].model_dump()


async def test_context_window_does_not_substitute_archive_uuid_tie_order(archive):
    repo, own, _ = archive
    when = datetime.now(UTC)
    for i, role in enumerate([MessageRole.USER, MessageRole.ASSISTANT, MessageRole.TOOL_RESULT]):
        await repo.add_message(AgentMessage(id=f'{3-i}{own[1:]}', session_id=own, role=role,
                                           content='x' * 10000, created_at=when))
    assert [w.message.id for w in await repo.get_context_windows(own)] == [m.id for m in await repo.get_messages(own)]


@pytest.mark.parametrize('head,tail', [(0, 0), (64001, 0), (100, -1), (100, 101), (True, 0), (12, 1.5)])
async def test_invalid_content_window_refuses_before_read(archive, head, tail):
    repo, own, _ = archive
    with pytest.raises(ValueError, match='content window'):
        await repo.get_context_windows(own, head_chars=head, tail_chars=tail)


async def test_zero_tail_is_empty_and_original_with_tool_calls_stays_whole(archive):
    repo, own, _ = archive
    message = AgentMessage(session_id=own, role=MessageRole.TOOL_RESULT, tool_name='get', content='x' * 1000)
    await repo.add_message(message)
    window = (await repo.get_context_windows(own, head_chars=100, tail_chars=0))[0]
    assert window.partial and window.tail == '' and len(window.head) == 100
    await repo.add_message(AgentMessage(session_id=own, role=MessageRole.TOOL_RESULT, content='y' * 1000,
        tool_calls=[ToolCall(id='unusual', tool_name='get', arguments={'preserve': True})]))
    assert not (await repo.get_context_windows(own, head_chars=100, tail_chars=0))[1].partial


async def test_durable_projection_never_hydrates_large_tool_bodies(archive, monkeypatch):
    repo, own, _ = archive
    if type(repo).__name__ == 'InMemoryRepository':
        pytest.skip('In-memory compatibility implementation already owns the full text')
    text = 'x' * 200_000
    for _ in range(30):
        await repo.add_message(AgentMessage(session_id=own, role=MessageRole.TOOL_RESULT, content=text, tool_name='get'))
    method = '_message_from_row' if hasattr(repo, '_message_from_row') else '_message_from_model'
    original = getattr(repo, method)
    def inspect(row):
        content = row['content'] if method == '_message_from_row' else row.content
        assert content == '', 'Database returned a full projected tool body'
        return original(row)
    monkeypatch.setattr(repo, method, inspect)
    monkeypatch.setattr(repo, 'get_messages', AsyncMock(side_effect=AssertionError('All body read')))
    windows = await repo.get_context_windows(own, head_chars=1500, tail_chars=400)
    assert len(windows) == 30 and all(w.partial and w.total_characters == 200_000 for w in windows)
    assert sum(len(w.head) + len(w.tail) for w in windows) == 57_000
