"""Tenant-scoped memory keys.

Persistent agent memory (objectives, knowledge documents, skills) is keyed by
agent id. But a single registered agent definition is shared across tenants,
so keying by agent id ALONE means two tenants running the same agent share one
objective, one knowledge library, one skill set — one tenant's data leaks into
another's prompt. Scoping the storage key by ``(tenant_id, agent_id)`` keeps
each tenant's memory of the same agent separate.

The scope is a prefix on the storage "agent" key the repository sees, so no
repository/schema change is needed — an empty tenant (single-tenant or
unauthenticated deployments) maps to the bare agent id, preserving existing
data and behavior.
"""

from __future__ import annotations

from urllib.parse import quote

_SCOPE_SEP = "::"


def scoped_agent_key(tenant_id: str | None, agent_id: str) -> str:
    """The storage key namespacing this agent's memory to its tenant.

    Empty/absent tenant → the bare agent id (back-compatible with data written
    before tenant scoping and with single-tenant deployments).

    Both segments are percent-encoded (``safe=''``) before joining, so the
    separator can never appear *inside* a segment — otherwise ``tenant='a::b',
    agent='c'`` and ``tenant='a', agent='b::c'`` would collide onto one key and
    let a caller in one tenant read another's memory via a crafted ``agent_id``
    path parameter. Encoding both segments makes the concatenation injective."""
    tenant = (tenant_id or "").strip()
    if not tenant:
        return agent_id
    return f"t{_SCOPE_SEP}{quote(tenant, safe='')}{_SCOPE_SEP}{quote(agent_id, safe='')}"
