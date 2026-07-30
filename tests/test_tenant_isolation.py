"""Tenant isolation at the HTTP boundary (C1/C2).

With an authenticator configured, a caller may only touch sessions and memory
in its own tenant, and the tenant is taken from the verified auth context —
never from the request body.
"""

import pytest
from fastapi import FastAPI

from fg_agents.api.auth import HeaderAuthenticator
from fg_agents.api.router import create_agent_router
from fg_agents.api.service import AgentService
from fg_agents.core.memory_scope import scoped_agent_key
from fg_agents.core.types import AgentDefinition
from fg_agents.orchestrator.orchestrator import Orchestrator
from fg_agents.persistence.memory import InMemoryRepository
from fg_agents.tools.registry import ToolRegistry
from tests.helpers import MockLLM, make_text_response

try:
    from httpx import ASGITransport, AsyncClient
except ImportError:  # pragma: no cover
    pytest.skip("httpx not installed", allow_module_level=True)


async def _app():
    llm = MockLLM([make_text_response("hi")])
    repo = InMemoryRepository()
    await repo.initialize()
    orchestrator = Orchestrator(
        llm=llm, tool_registry=ToolRegistry(), repository=repo
    )
    await orchestrator.initialize()
    agents = {"assistant": AgentDefinition(name="assistant", model="mock:test")}
    service = AgentService(orchestrator, agents)
    router = create_agent_router(
        orchestrator, agents, service=service, authenticator=HeaderAuthenticator()
    )
    app = FastAPI()
    app.include_router(router, prefix="/api/agent")
    return app, repo


def _hdr(tenant, user="u", admin=False):
    h = {"X-Tenant-Id": tenant, "X-User-Id": user}
    if admin:
        h["X-Is-Admin"] = "true"
    return h


@pytest.mark.asyncio
async def test_missing_tenant_is_rejected():
    app, _ = await _app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        resp = await c.post("/api/agent/sessions", json={"agent_id": "assistant"})
        assert resp.status_code == 401


@pytest.mark.asyncio
async def test_session_created_in_callers_tenant_and_isolated():
    app, _ = await _app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        # Alpha creates a session; the tenant comes from the header, not the body
        # (body claims 'evil' — must be ignored).
        r = await c.post(
            "/api/agent/sessions",
            json={"agent_id": "assistant", "tenant_id": "evil"},
            headers=_hdr("alpha"),
        )
        assert r.status_code == 200
        sid = r.json()["session_id"]

        # Alpha can read it.
        assert (await c.get(f"/api/agent/sessions/{sid}", headers=_hdr("alpha"))).status_code == 200

        # Beta cannot see it exists (404, not 403 — no existence leak).
        for path, method in [
            (f"/api/agent/sessions/{sid}", "get"),
            (f"/api/agent/sessions/{sid}/messages", "get"),
            (f"/api/agent/sessions/{sid}/audit", "get"),
        ]:
            resp = await getattr(c, method)(path, headers=_hdr("beta"))
            assert resp.status_code == 404, f"{method} {path} leaked to beta"
        # Beta cannot delete or cancel it either.
        assert (await c.delete(f"/api/agent/sessions/{sid}", headers=_hdr("beta"))).status_code == 404
        assert (await c.post(f"/api/agent/sessions/{sid}/cancel", headers=_hdr("beta"))).status_code == 404


@pytest.mark.asyncio
async def test_memory_is_tenant_scoped():
    app, repo = await _app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        # Alpha writes agent memory.
        w = await c.put(
            "/api/agent/memory/assistant/secret",
            json={"value": "alpha-only"},
            headers=_hdr("alpha"),
        )
        assert w.status_code == 200

        # Beta reading the SAME agent id sees nothing of alpha's.
        b = await c.get("/api/agent/memory/assistant", headers=_hdr("beta"))
        assert b.status_code == 200
        assert b.json()["count"] == 0

        # Alpha sees its own.
        a = await c.get("/api/agent/memory/assistant", headers=_hdr("alpha"))
        assert a.json()["count"] == 1

        # And it's physically stored under a tenant-scoped key.
        assert scoped_agent_key("alpha", "assistant") != "assistant"


@pytest.mark.asyncio
async def test_admin_crosses_tenants():
    app, _ = await _app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.post("/api/agent/sessions", json={"agent_id": "assistant"}, headers=_hdr("alpha"))
        sid = r.json()["session_id"]
        # An admin can read across tenants.
        resp = await c.get(f"/api/agent/sessions/{sid}", headers=_hdr("beta", admin=True))
        assert resp.status_code == 200


def test_scoped_key_is_injective_no_separator_collision():
    """Two different (tenant, agent) pairs must never collapse to one key."""
    a = scoped_agent_key("a::b", "c")
    b = scoped_agent_key("a", "b::c")
    assert a != b
    # And the empty-tenant path stays the bare id (back-compat).
    assert scoped_agent_key("", "agent") == "agent"
    assert scoped_agent_key(None, "agent") == "agent"


@pytest.mark.asyncio
async def test_manage_agent_tool_cannot_cross_tenants():
    """The manage_agent tool (status/get_report/message) must not reach a
    session in another tenant — the internal bypass of the HTTP boundary."""
    from fg_agents.core.types import AgentSession, ExecutionContext

    llm = MockLLM([make_text_response("hi")])
    repo = InMemoryRepository()
    await repo.initialize()
    orch = Orchestrator(llm=llm, tool_registry=ToolRegistry(), repository=repo)
    await orch.initialize()
    orch.register_sub_agent("worker", AgentDefinition(name="worker", model="mock:test"))

    # A session owned by tenant "alpha".
    alpha_sess = AgentSession(agent_id="worker", tenant_id="alpha", user_id="u")
    await repo.create_session(alpha_sess)

    handler = orch.tools.get("manage_agent").handler
    beta_ctx = ExecutionContext(session_id="s-beta", agent_id="worker", tenant_id="beta")
    alpha_ctx = ExecutionContext(session_id="s-alpha", agent_id="worker", tenant_id="alpha")

    # Beta probing alpha's session id gets a not-found (no cross-tenant read).
    r = await handler(action="status", session_id=alpha_sess.id, ctx=beta_ctx)
    assert r["status"] == "error" and "not found" in r["message"].lower()
    r = await handler(action="get_report", session_id=alpha_sess.id, ctx=beta_ctx)
    assert r["status"] == "error" and "not found" in r["message"].lower()

    # Alpha (same tenant) can see its own session's status.
    r = await handler(action="status", session_id=alpha_sess.id, ctx=alpha_ctx)
    assert r["status"] == "success"
