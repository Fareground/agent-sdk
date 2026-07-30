"""
Tests for rate limit middleware.
"""
import pytest

from fg_agents.core.errors import RateLimitExceededError
from fg_agents.core.types import (
    AgentMessage,
    ExecutionContext,
    MessageRole,
)
from fg_agents.middleware.rate_limit import RateLimitMiddleware


def _ctx(session_id: str = "s1", user_id: str = "u1", tenant_id: str = "t1"):
    return ExecutionContext(
        session_id=session_id,
        agent_id="test",
        agent_name="test",
        metadata={"user_id": user_id, "tenant_id": tenant_id},
    )


@pytest.mark.asyncio
async def test_rate_limit_allows_under_limit():
    """Calls under the limit pass through."""
    mw = RateLimitMiddleware(max_calls=5, window_seconds=60)
    ctx = _ctx()
    msgs = [AgentMessage(role=MessageRole.USER, content="Hi")]

    for _ in range(4):
        result_msgs, _ = await mw.before_llm_call(msgs, [], ctx)
        # Should pass through without rate limit message
        assert len(result_msgs) == 1


@pytest.mark.asyncio
async def test_rate_limit_blocks_over_limit():
    """Calls over the limit get rate limit injection."""
    mw = RateLimitMiddleware(max_calls=3, window_seconds=60)
    ctx = _ctx()
    msgs = [AgentMessage(role=MessageRole.USER, content="Hi")]

    # Use up the limit
    for _ in range(3):
        await mw.before_llm_call(msgs, [], ctx)

    # 4th call is actually blocked (fail-closed), not silently proceeding.
    with pytest.raises(RateLimitExceededError):
        await mw.before_llm_call(msgs, [], ctx)


@pytest.mark.asyncio
async def test_rate_limit_per_user_isolation():
    """Different users have independent rate limits."""
    mw = RateLimitMiddleware(max_calls=2, window_seconds=60, scope="user")
    msgs = [AgentMessage(role=MessageRole.USER, content="Hi")]

    ctx1 = _ctx(user_id="alice")
    ctx2 = _ctx(user_id="bob")

    # Alice uses 2 calls
    await mw.before_llm_call(msgs, [], ctx1)
    await mw.before_llm_call(msgs, [], ctx1)

    # Alice is rate limited
    with pytest.raises(RateLimitExceededError):
        await mw.before_llm_call(msgs, [], ctx1)

    # Bob is NOT rate limited
    result, _ = await mw.before_llm_call(msgs, [], ctx2)
    assert len(result) == 1  # No rate limit message


@pytest.mark.asyncio
async def test_rate_limit_tenant_scope():
    """Tenant scope groups all users under same tenant."""
    mw = RateLimitMiddleware(max_calls=2, window_seconds=60, scope="tenant")
    msgs = [AgentMessage(role=MessageRole.USER, content="Hi")]

    ctx1 = _ctx(user_id="alice", tenant_id="acme")
    ctx2 = _ctx(user_id="bob", tenant_id="acme")

    await mw.before_llm_call(msgs, [], ctx1)
    await mw.before_llm_call(msgs, [], ctx2)

    # Both under same tenant — the shared limit is exhausted, so blocked.
    with pytest.raises(RateLimitExceededError):
        await mw.before_llm_call(msgs, [], ctx1)


@pytest.mark.asyncio
async def test_rate_limit_usage_report():
    """get_usage() returns current state."""
    mw = RateLimitMiddleware(max_calls=10, window_seconds=60)
    ctx = _ctx()
    msgs = [AgentMessage(role=MessageRole.USER, content="Hi")]

    await mw.before_llm_call(msgs, [], ctx)
    await mw.before_llm_call(msgs, [], ctx)

    usage = mw.get_usage(ctx)
    assert usage["calls_used"] == 2
    assert usage["calls_remaining"] == 8
    assert usage["max_calls"] == 10


def test_rate_limit_name():
    mw = RateLimitMiddleware()
    assert mw.name == "RateLimitMiddleware"


@pytest.mark.asyncio
async def test_rate_limit_buckets_are_lru_bounded():
    """Per-key buckets must not grow without bound on a long-lived server."""
    from fg_agents.middleware._bounded import BoundedLRU

    mw = RateLimitMiddleware(max_calls=100, window_seconds=60, scope="session")
    mw._buckets = BoundedLRU(max_entries=10)
    msgs = [AgentMessage(role=MessageRole.USER, content="Hi")]
    for i in range(50):
        await mw.before_llm_call(msgs, [], _ctx(session_id=f"s{i}"))
    assert len(mw._buckets) <= 10


def test_bounded_lru_get_refreshes_recency():
    """A key read via .get() must not be evicted ahead of an idle key."""
    from fg_agents.middleware._bounded import BoundedLRU

    b: BoundedLRU[str, int] = BoundedLRU(max_entries=3)
    b["a"], b["b"], b["c"] = 1, 2, 3
    for _ in range(5):
        assert b.get("a") == 1  # keep "a" hot via .get()
    b["d"] = 4  # forces one eviction — the idle "b", not the hot "a"
    assert "a" in b
    assert "b" not in b
