"""Archive recovery must not hydrate unrelated conversations or entire histories."""
import os
import uuid
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock

import pytest

from fg_agents import AgentMessage, AgentSession, MessageRole, ToolCall, create_repository


@pytest.fixture(params=['memory', 'sqlite', 'postgres'])
async def archive(request, tmp_path):
    backend = request.param
    if backend == 'postgres':
        url = os.environ.get('SDK_TEST_DATABASE_URL')
        if not url:
            pytest.skip('SDK_TEST_DATABASE_URL must name a disposable PostgreSQL database')
        repo = create_repository(backend, db_url=url)
    elif backend == 'sqlite':
        repo = create_repository(backend, db_path=str(tmp_path / 'archive.db'))
    else:
        repo = create_repository(backend)
    await repo.initialize()
    own, foreign = str(uuid.uuid4()), str(uuid.uuid4())
    for sid in (own, foreign):
        await repo.create_session(AgentSession(id=sid, agent_id='archive-test', user_id=sid))
    yield repo, own, foreign
    for sid in (own, foreign):
        await repo.delete_session(sid)
    await repo.close()


async def test_single_original_is_session_scoped_and_keeps_tool_arguments(archive, monkeypatch):
    repo, own, foreign = archive
    msg = AgentMessage(session_id=own, role=MessageRole.ASSISTANT,
        content='original' * 8000, is_summarized=True,
        tool_calls=[ToolCall(id='call', tool_name='build', arguments={'requirement': 'keep me'})])
    await repo.add_message(msg)
    monkeypatch.setattr(repo, 'get_messages', AsyncMock(side_effect=AssertionError('Hydrated the entire archive')))
    assert (await repo.get_message(own, msg.id)).model_dump() == msg.model_dump()
    assert await repo.get_message(foreign, msg.id) is None
    assert await repo.get_message(own, 'absent') is None


async def test_batch_scan_covers_timestamp_ties_and_summarized_originals(archive, monkeypatch):
    repo, own, foreign = archive
    when = datetime.now(UTC)
    expected = set()
    for index in range(137):
        msg = AgentMessage(session_id=own, role=MessageRole.USER,
            content=f'item {index}', created_at=when, is_summarized=index % 2 == 0)
        expected.add(msg.id)
        await repo.add_message(msg)
    await repo.add_message(AgentMessage(session_id=foreign, role=MessageRole.USER, content='private other owner'))
    monkeypatch.setattr(repo, 'get_messages', AsyncMock(side_effect=AssertionError('Hydrated the entire archive')))
    items = [m async for m in repo.iter_messages(own, batch_size=13)]
    assert len(items) == len(expected) == len({m.id for m in items})
    assert {m.id for m in items} == expected
    active = [m async for m in repo.iter_messages(own, batch_size=7, include_summarized=False)]
    assert len(active) == 68 and all(not m.is_summarized for m in active)


async def test_scan_high_water_does_not_follow_newly_appended_history(archive):
    repo, own, _ = archive
    when = datetime.now(UTC)
    originals = []
    for index in range(8):
        message = AgentMessage(session_id=own, role=MessageRole.USER, content=str(index),
                               created_at=when + timedelta(seconds=index))
        originals.append(message.id)
        await repo.add_message(message)
    iterator = repo.iter_messages(own, batch_size=2)
    first = await anext(iterator)
    await repo.add_message(AgentMessage(session_id=own, role=MessageRole.USER, content='new',
                                       created_at=when + timedelta(seconds=30)))
    assert [first.id, *[m.id async for m in iterator]] == originals


@pytest.mark.parametrize('batch_size', [0, -1, 257, True, 1.5])
async def test_invalid_batch_sizes_refuse_before_scan(archive, batch_size):
    repo, own, _ = archive
    with pytest.raises(ValueError, match='batch_size'):
        await anext(repo.iter_messages(own, batch_size=batch_size))


async def test_empty_archive_scan(archive):
    repo, own, _ = archive
    assert [m async for m in repo.iter_messages(own)] == []


async def test_sqlite_recovery_uses_bounded_keyset_queries(tmp_path):
    repo = create_repository('sqlite', db_path=str(tmp_path / 'large.db'))
    await repo.initialize()
    await repo.create_session(AgentSession(id='large', agent_id='test'))
    when = datetime.now(UTC)
    # A raw bulk insert makes this a read-path benchmark, not 1,000 fsyncs.
    db = repo._ensure_db()
    await db.executemany('INSERT INTO af_messages (id,session_id,role,content,created_at) VALUES (?,?,?,?,?)',
        [(f'm{i:05}', 'large', 'user', 'x' * 20_000, when.isoformat()) for i in range(1000)])
    await db.commit()
    queries = []
    await db.set_trace_callback(queries.append)
    assert len((await repo.get_message('large', 'm00500')).content) == 20_000
    assert len(queries) == 1 and "id = 'm00500'" in queries[0]
    queries.clear()
    count = 0
    async for message in repo.iter_messages('large', batch_size=32):
        count += 1
        assert len(message.content) == 20_000
    assert count == 1000
    reads = [q for q in queries if q.startswith('SELECT *')]
    assert len(reads) == 32 and all('LIMIT 32' in q and 'OFFSET' not in q for q in reads)
    await repo.close()
