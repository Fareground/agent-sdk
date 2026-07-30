"""Authentication and tenant scoping for the HTTP boundary.

The framework's promise is per-tenant isolation, but the routes historically
took ``session_id``/``agent_id`` straight from the path with no authorization,
and sessions carried a *client-supplied* ``tenant_id`` that nothing verified.
This module closes that: an :class:`Authenticator` resolves each request to a
verified :class:`AuthContext`, and the router stamps and enforces the tenant
from that context — never from the client's request body.

Auth is pluggable. Bring your own authenticator (validate your JWT, session
cookie, mTLS identity, whatever your app already uses) and every route is
gated by it. For a genuinely single-tenant or local deployment, the default
:class:`UnauthenticatedAccess` keeps things working but logs a loud one-time
warning so the open boundary is never a silent surprise.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

import structlog
from fastapi import HTTPException, Request

log = structlog.get_logger("fg_agents.auth")


@dataclass(frozen=True)
class AuthContext:
    """The verified identity behind a request.

    ``tenant_id`` is the isolation boundary: a caller may only touch sessions
    and memory belonging to its own tenant (unless ``is_admin``). ``user_id``
    is the acting principal within the tenant. Both come from the
    authenticator, never from the request body."""

    tenant_id: str
    user_id: str = ""
    scopes: frozenset[str] = field(default_factory=frozenset)
    is_admin: bool = False

    def can_access_tenant(self, tenant_id: str) -> bool:
        """Admins cross tenants; everyone else is pinned to their own."""
        return self.is_admin or (tenant_id or "") == self.tenant_id


# An authenticator turns a request into a verified AuthContext, or raises
# fastapi.HTTPException(401) when the request is not authenticated.
Authenticator = Callable[[Request], Awaitable[AuthContext]]


class UnauthenticatedAccess:
    """Default authenticator for single-tenant / local deployments: every
    request is admin of a single default tenant. It logs ONCE, loudly, so an
    unauthenticated boundary can never be mistaken for a secured one — wire a
    real :class:`Authenticator` for any multi-tenant or exposed deployment."""

    def __init__(self, default_tenant: str = "default") -> None:
        self._tenant = default_tenant
        self._warned = False

    async def __call__(self, request: Request) -> AuthContext:
        if not self._warned:
            self._warned = True
            log.warning(
                "unauthenticated_api_boundary",
                detail=(
                    "No authenticator configured: the agent API is UNAUTHENTICATED "
                    "and every caller is treated as admin of one shared tenant. "
                    "Pass authenticator=... to create_agent_router() before exposing "
                    "this beyond a trusted single-tenant boundary."
                ),
            )
        return AuthContext(tenant_id=self._tenant, user_id="", is_admin=True)


class HeaderAuthenticator:
    """Trusted-proxy authenticator: reads the tenant/user from request headers
    a trusted upstream (API gateway, service mesh) has already authenticated
    and injected. Use ONLY when those headers cannot be set by the client —
    i.e. behind a proxy that strips and re-adds them. Rejects requests missing
    the tenant header."""

    def __init__(
        self,
        tenant_header: str = "X-Tenant-Id",
        user_header: str = "X-User-Id",
        admin_header: str = "X-Is-Admin",
    ) -> None:
        self._tenant_header = tenant_header
        self._user_header = user_header
        self._admin_header = admin_header

    async def __call__(self, request: Request) -> AuthContext:
        tenant = request.headers.get(self._tenant_header, "").strip()
        if not tenant:
            raise HTTPException(status_code=401, detail="missing tenant identity")
        return AuthContext(
            tenant_id=tenant,
            user_id=request.headers.get(self._user_header, "").strip(),
            is_admin=request.headers.get(self._admin_header, "").lower()
            in ("1", "true", "yes"),
        )
