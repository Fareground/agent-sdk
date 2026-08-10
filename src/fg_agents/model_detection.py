"""
Fareground Agent Framework — Default model detection

When no model is given (``Agent(model=None)`` or ``ask("...")``), the
framework picks a sensible default from the environment, in this order:

1. ``ANTHROPIC_API_KEY``  → ``anthropic:claude-sonnet-4-6``
2. ``OPENAI_API_KEY``     → ``openai:gpt-5.2``
3. ``GOOGLE_API_KEY`` or ``GEMINI_API_KEY`` → ``google:gemini-2.5-flash``
4. A local Ollama server on ``localhost:11434`` → ``ollama:qwen3:8b``

If none of those are available, :class:`ModelDetectionError` is raised
with instructions. An explicit ``model=`` always wins — detection only
runs when the caller passed nothing.
"""

import asyncio
import os
import socket
import time

from fg_agents.core.errors import AgentFrameworkError

ANTHROPIC_DEFAULT = "anthropic:claude-sonnet-4-6"
OPENAI_DEFAULT = "openai:gpt-5.2"
GOOGLE_DEFAULT = "google:gemini-2.5-flash"
OLLAMA_DEFAULT = "ollama:qwen3:8b"

_OLLAMA_HOST = "localhost"
_OLLAMA_PORT = 11434
_OLLAMA_PROBE_TIMEOUT = 0.25  # seconds


class ModelDetectionError(AgentFrameworkError):
    """Raised when no model was given and none could be detected."""


# The Ollama probe result is cached briefly so repeated constructions
# (e.g. many ask() calls in a row) don't each pay the connect timeout.
_PROBE_CACHE_TTL = 30.0  # seconds
_probe_cached_at: float | None = None
_probe_cached_result: bool = False


def _probe_cache_get() -> bool | None:
    """Return the cached probe result, or None when absent/expired."""
    if _probe_cached_at is None:
        return None
    if time.monotonic() - _probe_cached_at >= _PROBE_CACHE_TTL:
        return None
    return _probe_cached_result


def _probe_cache_put(result: bool) -> bool:
    global _probe_cached_at, _probe_cached_result
    _probe_cached_at = time.monotonic()
    _probe_cached_result = result
    return result


def _probe_ollama_sync(host: str = _OLLAMA_HOST, port: int = _OLLAMA_PORT) -> bool:
    """Blocking TCP probe: is a local Ollama server accepting connections?"""
    try:
        with socket.create_connection((host, port), timeout=_OLLAMA_PROBE_TIMEOUT):
            return True
    except OSError:
        return False


async def _probe_ollama_async(host: str = _OLLAMA_HOST, port: int = _OLLAMA_PORT) -> bool:
    """Non-blocking TCP probe: is a local Ollama server accepting connections?"""
    try:
        _, writer = await asyncio.wait_for(
            asyncio.open_connection(host, port), timeout=_OLLAMA_PROBE_TIMEOUT
        )
    except (TimeoutError, OSError):
        return False
    writer.close()
    try:
        await writer.wait_closed()
    except OSError:
        pass
    return True


def _ollama_running() -> bool:
    """Cached sync probe — safe for synchronous construction paths."""
    cached = _probe_cache_get()
    if cached is not None:
        return cached
    return _probe_cache_put(_probe_ollama_sync())


async def _ollama_running_async() -> bool:
    """Cached async probe — never blocks the event loop."""
    cached = _probe_cache_get()
    if cached is not None:
        return cached
    return _probe_cache_put(await _probe_ollama_async())


def _resolve_from_env() -> str | None:
    """The env-var part of detection, shared by the sync and async paths."""
    if os.environ.get("ANTHROPIC_API_KEY"):
        return ANTHROPIC_DEFAULT
    if os.environ.get("OPENAI_API_KEY"):
        return OPENAI_DEFAULT
    if os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY"):
        return GOOGLE_DEFAULT
    return None


def _nothing_detected_error() -> ModelDetectionError:
    return ModelDetectionError(
        "No model specified and no provider detected. Either set one of "
        "ANTHROPIC_API_KEY, OPENAI_API_KEY, or GOOGLE_API_KEY (checked in that "
        "order), run a local Ollama server (http://localhost:11434), or pass "
        "an explicit model, e.g. model='anthropic:claude-sonnet-4-6'."
    )


def resolve_default_model() -> str:
    """
    Pick a default provider-qualified model id from the environment.

    Detection order: ``ANTHROPIC_API_KEY`` → ``OPENAI_API_KEY`` →
    ``GOOGLE_API_KEY``/``GEMINI_API_KEY`` → local Ollama on port 11434.
    The Ollama probe blocks briefly; in async code prefer
    :func:`resolve_default_model_async`. The probe result is cached for a
    short interval.

    Raises:
        ModelDetectionError: nothing detected — set one of the env vars,
            run Ollama locally, or pass ``model=`` explicitly.
    """
    model = _resolve_from_env()
    if model:
        return model
    if _ollama_running():
        return OLLAMA_DEFAULT
    raise _nothing_detected_error()


async def resolve_default_model_async() -> str:
    """
    Async twin of :func:`resolve_default_model` — same detection order and
    error, but the Ollama probe uses a non-blocking connection so the event
    loop is never stalled. Used by ``ask()``/``stream()``.
    """
    model = _resolve_from_env()
    if model:
        return model
    if await _ollama_running_async():
        return OLLAMA_DEFAULT
    raise _nothing_detected_error()


def env_api_key_overrides() -> dict[str, str]:
    """
    API-key aliases the LLM layer doesn't resolve on its own.

    The LLM layer reads ``{PROVIDER}_API_KEY``; Google is commonly configured
    via ``GEMINI_API_KEY`` instead, so map it to the ``google`` provider when
    ``GOOGLE_API_KEY`` is absent.
    """
    if not os.environ.get("GOOGLE_API_KEY") and os.environ.get("GEMINI_API_KEY"):
        return {"google": os.environ["GEMINI_API_KEY"]}
    return {}
