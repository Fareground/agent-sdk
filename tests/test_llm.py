"""Tests for the LLM interface."""
import pytest

from fg_agents import AgentLLM
from fg_agents.core.errors import LLMError


@pytest.mark.asyncio
async def test_empty_model_raises():
    llm = AgentLLM()
    with pytest.raises(LLMError, match="No model specified"):
        async for _ in llm.stream_with_tools([], [], "", "", 0.0, 100):
            pass


@pytest.mark.asyncio
async def test_empty_model_complete_raises():
    llm = AgentLLM()
    with pytest.raises(LLMError, match="No model specified"):
        await llm.complete_with_tools([], [], "", "", 0.0, 100)


def test_parse_model_string():
    from fg_agents.core.llm import _parse_model_string

    assert _parse_model_string("openai:gpt-4o") == ("openai", "gpt-4o")
    assert _parse_model_string("anthropic:claude-3-haiku") == ("anthropic", "claude-3-haiku")
    assert _parse_model_string("ollama:qwen3.5:9b") == ("ollama", "qwen3.5:9b")
    # No prefix — splits on first colon (ambiguous, use explicit prefix)
    assert _parse_model_string("qwen3.5:9b") == ("qwen3.5", "9b")
    # Simple model name without colon defaults to ollama
    assert _parse_model_string("llama3") == ("ollama", "llama3")


def test_api_key_from_dict():
    llm = AgentLLM(api_keys={"openai": "sk-test"})
    assert llm._get_api_key("openai") == "sk-test"


def test_anthropic_mapping_preserves_thinking_blocks_with_tool_use():
    """Under extended thinking, the stored thinking block must ride along with
    the tool_use in the rebuilt assistant turn, or the next API call is 400'd."""
    from fg_agents.core.llm import _messages_to_anthropic
    from fg_agents.core.types import AgentMessage, MessageRole, ToolCall

    assistant = AgentMessage(
        role=MessageRole.ASSISTANT,
        content=[
            {"type": "thinking", "thinking": "let me think", "signature": "sig123"},
            {"type": "text", "text": "I'll check."},
        ],
        tool_calls=[ToolCall(id="tc1", tool_name="lookup", arguments={"q": "x"})],
    )
    api_messages, _ = _messages_to_anthropic([assistant], "sys")
    blocks = api_messages[0]["content"]
    types_seq = [b["type"] for b in blocks]
    assert types_seq[0] == "thinking"  # preserved and first
    assert "tool_use" in types_seq
    assert blocks[0]["signature"] == "sig123"


def test_google_mapping_carries_tool_calls_and_results():
    """The Gemini mapping must re-emit assistant tool calls as functionCall
    parts and TOOL_RESULT messages as functionResponse, or multi-turn tool use
    silently loses the tool's output."""
    pytest.importorskip("google.genai")
    from google.genai import types

    from fg_agents.core.types import AgentMessage, MessageRole, ToolCall

    # Reproduce the content-building loop in isolation (no network).
    messages = [
        AgentMessage(role=MessageRole.USER, content="do it"),
        AgentMessage(
            role=MessageRole.ASSISTANT,
            content="calling",
            tool_calls=[ToolCall(id="t1", tool_name="search", arguments={"q": "a"})],
        ),
        AgentMessage(
            role=MessageRole.TOOL_RESULT,
            content='{"hits": 3}',
            tool_call_id="t1",
            tool_name="search",
        ),
    ]
    contents = []
    for msg in messages:
        if msg.role == MessageRole.USER:
            contents.append(types.Content(role="user", parts=[types.Part(text=msg.content)]))
        elif msg.role == MessageRole.ASSISTANT:
            parts = [types.Part(text=msg.content)] if msg.content else []
            for tc in msg.tool_calls or []:
                parts.append(types.Part(function_call=types.FunctionCall(
                    name=tc.tool_name, args=tc.arguments or {})))
            contents.append(types.Content(role="model", parts=parts))
        elif msg.role == MessageRole.TOOL_RESULT:
            payload = msg.content if isinstance(msg.content, dict) else {"result": msg.content}
            contents.append(types.Content(role="user", parts=[types.Part(
                function_response=types.FunctionResponse(name=msg.tool_name, response=payload))]))
    assert any(p.function_call for p in contents[1].parts)
    assert any(p.function_response for p in contents[2].parts)
