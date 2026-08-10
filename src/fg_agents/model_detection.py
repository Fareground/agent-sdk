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

import os
import socket

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


def _ollama_running(host: str = _OLLAMA_HOST, port: int = _OLLAMA_PORT) -> bool:
    """Cheap TCP probe: is a local Ollama server accepting connections?"""
    try:
        with socket.create_connection((host, port), timeout=_OLLAMA_PROBE_TIMEOUT):
            return True
    except OSError:
        return False


def resolve_default_model() -> str:
    """
    Pick a default provider-qualified model id from the environment.

    Detection order: ``ANTHROPIC_API_KEY`` → ``OPENAI_API_KEY`` →
    ``GOOGLE_API_KEY``/``GEMINI_API_KEY`` → local Ollama on port 11434.

    Raises:
        ModelDetectionError: nothing detected — set one of the env vars,
            run Ollama locally, or pass ``model=`` explicitly.
    """
    if os.environ.get("ANTHROPIC_API_KEY"):
        return ANTHROPIC_DEFAULT
    if os.environ.get("OPENAI_API_KEY"):
        return OPENAI_DEFAULT
    if os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY"):
        return GOOGLE_DEFAULT
    if _ollama_running():
        return OLLAMA_DEFAULT
    raise ModelDetectionError(
        "No model specified and no provider detected. Either set one of "
        "ANTHROPIC_API_KEY, OPENAI_API_KEY, or GOOGLE_API_KEY (checked in that "
        "order), run a local Ollama server (http://localhost:11434), or pass "
        "an explicit model, e.g. model='anthropic:claude-sonnet-4-6'."
    )


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
