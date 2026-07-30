"""Truncation must never masquerade as a clean turn end."""
from fg_agents.core.llm import _map_stop
from fg_agents.core.types import StopReason


def test_provider_truncation_reasons_map_to_max_tokens():
    assert _map_stop("max_tokens") is StopReason.MAX_TOKENS   # anthropic
    assert _map_stop("length") is StopReason.MAX_TOKENS       # openai-compat


def test_normal_reasons_unchanged():
    assert _map_stop("tool_use") is StopReason.TOOL_USE
    assert _map_stop("tool_calls") is StopReason.TOOL_USE
    assert _map_stop("stop") is StopReason.END_TURN
    assert _map_stop(None, has_tool_calls=True) is StopReason.TOOL_USE
    assert _map_stop(None) is StopReason.END_TURN
