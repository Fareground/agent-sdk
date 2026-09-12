"""
Fareground Agent Framework — LLM Interface

Provider-agnostic LLM interface with streaming and tool-use support.

Pattern: "provider:model" string selects the backend.
  e.g. "ollama:qwen3:8b", "openai:gpt-5.4", "anthropic:claude-sonnet-4-6"

No default model — user must specify.
"""

import asyncio
import json
import time
from collections.abc import AsyncIterator
from typing import Any

import structlog

from fg_agents.core.errors import ContextOverflowError, LLMError, LLMRateLimitError
from fg_agents.core.tool_arguments import decode_tool_call
from fg_agents.core.types import (
    AgentMessage,
    LLMResponse,
    LLMStreamChunk,
    LLMUsage,
    MessageRole,
    StopReason,
    ToolCall,
    ToolSchema,
)

log = structlog.get_logger("fg_agents.llm")


def _parse_model_string(model: str) -> tuple[str, str]:
    """Parse 'provider:model_name' into (provider, model_name)."""
    if ":" in model:
        provider, model_name = model.split(":", 1)
        return provider.lower(), model_name
    return "ollama", model


def _messages_to_anthropic(
    messages: list[AgentMessage],
    system_prompt: str,
) -> tuple[list[dict], str | list[dict]]:
    """
    Convert framework messages to Anthropic API format.

    Anthropic requires strict user/assistant alternation.  Consecutive
    TOOL_RESULT messages (from multi-tool turns) are batched into a
    single 'user' message with multiple tool_result content blocks.
    """
    api_messages = []
    pending_tool_results: list[dict] = []

    def _flush_tool_results():
        nonlocal pending_tool_results
        if pending_tool_results:
            api_messages.append({"role": "user", "content": pending_tool_results})
            pending_tool_results = []

    for msg in messages:
        if msg.role == MessageRole.SYSTEM:
            continue
        elif msg.role == MessageRole.USER:
            _flush_tool_results()
            api_messages.append({"role": "user", "content": msg.content})
        elif msg.role == MessageRole.ASSISTANT:
            _flush_tool_results()
            content_blocks: list[dict] = []
            # Preserve any thinking blocks verbatim, signature included, and
            # BEFORE the text/tool_use blocks. Under extended thinking the API
            # requires the original thinking block to accompany a tool_use in
            # the same assistant turn; rebuilding from text() alone would drop
            # it and the next turn is rejected.
            if isinstance(msg.content, list):
                for block in msg.content:
                    if isinstance(block, dict) and block.get("type") in (
                        "thinking",
                        "redacted_thinking",
                    ):
                        content_blocks.append(block)
            if msg.content:
                text = msg.text() if isinstance(msg.content, list) else msg.content
                if text:
                    content_blocks.append({"type": "text", "text": text})
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    content_blocks.append(
                        {
                            "type": "tool_use",
                            "id": tc.id,
                            "name": tc.tool_name,
                            "input": tc.arguments,
                        }
                    )
            api_messages.append({"role": "assistant", "content": content_blocks or msg.content})
        elif msg.role == MessageRole.TOOL_RESULT:
            content = (
                msg.content
                if isinstance(msg.content, str)
                else json.dumps(msg.content, default=str)
            )
            pending_tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": msg.tool_call_id,
                    "content": content,
                }
            )

    _flush_tool_results()
    system_blocks = [
        {"type": "text", "text": system_prompt, "cache_control": {"type": "ephemeral"}}
    ]
    return api_messages, system_blocks


def _messages_to_openai(
    messages: list[AgentMessage],
    system_prompt: str,
) -> list[dict]:
    """Convert framework messages to OpenAI API format."""
    api_messages = [{"role": "system", "content": system_prompt}]
    for msg in messages:
        if msg.role == MessageRole.SYSTEM:
            continue
        elif msg.role == MessageRole.USER:
            api_messages.append(
                {
                    "role": "user",
                    "content": msg.content if isinstance(msg.content, str) else msg.text(),
                }
            )
        elif msg.role == MessageRole.ASSISTANT:
            m: dict[str, Any] = {"role": "assistant"}
            text = msg.text() if isinstance(msg.content, list) else msg.content
            m["content"] = text or ""
            if msg.tool_calls:
                m["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.tool_name,
                            "arguments": json.dumps(tc.arguments),
                        },
                    }
                    for tc in msg.tool_calls
                ]
            api_messages.append(m)
        elif msg.role == MessageRole.TOOL_RESULT:
            content = (
                msg.content
                if isinstance(msg.content, str)
                else json.dumps(msg.content, default=str)
            )
            api_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": msg.tool_call_id,
                    "content": content,
                }
            )
    return api_messages


def _tools_to_anthropic(tools: list[ToolSchema]) -> list[dict]:
    """Convert to Anthropic tool format."""
    result = []
    for t in tools:
        tool_def = {
            "name": t.name,
            "description": t.description,
            "input_schema": t.parameters,
        }
        result.append(tool_def)
    if result:
        result[-1]["cache_control"] = {"type": "ephemeral"}
    return result


def _openai_cached_tokens(usage: Any) -> int:
    """Cached-prefix tokens from an OpenAI-compatible usage object.

    Together/OpenAI report automatic prompt-cache hits in
    usage.prompt_tokens_details.cached_tokens; absent on providers
    that don't support it.
    """
    details = getattr(usage, "prompt_tokens_details", None)
    return (getattr(details, "cached_tokens", 0) or 0) if details else 0


def _tools_to_openai(tools: list[ToolSchema]) -> list[dict]:
    """Convert to OpenAI tool format."""
    return [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": t.parameters,
            },
        }
        for t in tools
    ]


# ══════════════════════════════════════════════════════════════════════
# OpenAI-compatible provider registry
# Any provider with an OpenAI-compatible API just needs a base URL.
# ══════════════════════════════════════════════════════════════════════

OPENAI_COMPATIBLE_PROVIDERS: dict[str, str] = {
    # Provider name → base URL (empty string = use OpenAI default)
    #
    # Cloud providers
    "openai": "",
    "deepseek": "https://api.deepseek.com/v1",
    "mistral": "https://api.mistral.ai/v1",
    "xai": "https://api.x.ai/v1",
    "perplexity": "https://api.perplexity.ai",
    "cohere": "https://api.cohere.com/compatibility/v1",
    # Fast inference
    "groq": "https://api.groq.com/openai/v1",
    "cerebras": "https://api.cerebras.ai/v1",
    "fireworks": "https://api.fireworks.ai/inference/v1",
    "together": "https://api.together.xyz/v1",
    # Local runners (OpenAI-compatible)
    "ollama": "http://localhost:11434/v1",
    "lmstudio": "http://localhost:1234/v1",
    "jan": "http://localhost:1337/v1",
    "llamacpp": "http://localhost:8080/v1",
    "localai": "http://localhost:8080/v1",
    "lemonade": "http://localhost:8000/api/v1",
    "jellybox": "http://localhost:39281/v1",
    "docker": "http://localhost:12434/engines/v1",
    "vllm": "http://localhost:8000/v1",
    # Cloud platforms (require base_url override via custom_providers)
    "azure": "",
    "bedrock": "",
}


class _ThinkSplitter:
    """Split inline ``<think>...</think>`` reasoning out of streamed content.

    gpt-oss style models (and some others served over the OpenAI-compatible
    API) emit their chain-of-thought inline in the *content* channel wrapped in
    ``<think></think>`` rather than in a separate ``reasoning`` field. Without
    this, the literal tags and the reasoning text leak into the answer.

    ``feed`` routes wrapped text to the ``thinking`` channel and the rest to
    ``text``, tolerating tags that straddle chunk boundaries by holding back a
    trailing partial-tag prefix until the next chunk resolves it.
    """

    OPEN = "<think>"
    CLOSE = "</think>"

    def __init__(self) -> None:
        self._buf = ""
        self._in_think = False

    def feed(self, text: str) -> list[tuple[str, str]]:
        self._buf += text
        out: list[tuple[str, str]] = []
        while True:
            tag = self.CLOSE if self._in_think else self.OPEN
            idx = self._buf.find(tag)
            if idx != -1:
                segment = self._buf[:idx]
                if segment:
                    out.append(("thinking" if self._in_think else "text", segment))
                self._buf = self._buf[idx + len(tag):]
                self._in_think = not self._in_think
                continue
            # No complete tag in the buffer. Emit everything except a trailing
            # suffix that could be the start of a tag split across chunks.
            keep = 0
            for candidate in (self.OPEN, self.CLOSE):
                limit = min(len(self._buf), len(candidate) - 1)
                for k in range(limit, 0, -1):
                    if self._buf.endswith(candidate[:k]):
                        keep = max(keep, k)
                        break
            emit_len = len(self._buf) - keep
            if emit_len > 0:
                out.append(
                    ("thinking" if self._in_think else "text", self._buf[:emit_len])
                )
                self._buf = self._buf[emit_len:]
            break
        return out

    def flush(self) -> list[tuple[str, str]]:
        if not self._buf:
            return []
        seg = self._buf
        self._buf = ""
        return [("thinking" if self._in_think else "text", seg)]


def _map_stop(provider_reason, has_tool_calls: bool = False) -> StopReason:
    """Provider finish/stop reason → StopReason.

    Truncation MUST map to MAX_TOKENS: collapsing it into END_TURN makes a
    token-capped, half-finished turn look like a deliberate clean end — the
    engine then stops as if the agent were done, silently abandoning
    half-built work (and a truncated tool-call JSON degrades into a
    missing-arguments error that never mentions truncation).
    Anthropic says "max_tokens"; OpenAI-compatible hosts say "length".
    """
    if provider_reason in ("max_tokens", "length"):
        return StopReason.MAX_TOKENS
    if provider_reason == "tool_use" or provider_reason == "tool_calls" or has_tool_calls:
        return StopReason.TOOL_USE
    return StopReason.END_TURN


class AgentLLM:
    """
    Provider-agnostic LLM interface with streaming and tool-use support.

    Supports 20+ providers out of the box via OpenAI-compatible API.
    Lazily initializes provider clients. Thread-safe via asyncio.Lock.
    """

    def __init__(
        self,
        api_keys: dict[str, str] | None = None,
        custom_providers: dict[str, str] | None = None,
    ):
        """
        Args:
            api_keys: Optional dict of provider → API key overrides.
                      Falls back to env vars (e.g., OPENAI_API_KEY).
            custom_providers: Optional dict of provider name → base URL
                              for providers not in the built-in registry.
        """
        self._api_keys = api_keys or {}
        self._custom_providers = custom_providers or {}
        self._clients: dict[str, Any] = {}
        self._lock = asyncio.Lock()

    def _get_api_key(self, provider: str) -> str:
        """
        Resolve API key for a provider.

        Priority: api_keys dict → {PROVIDER}_API_KEY env var → empty string.
        """
        if provider in self._api_keys:
            return self._api_keys[provider]
        import os

        return os.environ.get(f"{provider.upper()}_API_KEY", "")

    async def _get_client(self, provider: str) -> Any:
        async with self._lock:
            if provider in self._clients:
                return self._clients[provider]
            api_key = self._get_api_key(provider)
            if provider == "anthropic":
                import anthropic

                client = anthropic.AsyncAnthropic(api_key=api_key)
            elif provider == "google":
                from google import genai

                client = genai.Client(api_key=api_key)
            else:
                # OpenAI-compatible providers (OpenAI, Groq, DeepSeek, etc.)
                from openai import AsyncOpenAI

                base_url = (
                    OPENAI_COMPATIBLE_PROVIDERS.get(provider)
                    or self._custom_providers.get(provider)
                    or None  # None = use OpenAI default
                )
                client = AsyncOpenAI(
                    api_key=api_key,
                    base_url=base_url,
                )
            self._clients[provider] = client
            return client

    # ══════════════════════════════════════════════════════════════════
    # Streaming completion with tool support
    # ══════════════════════════════════════════════════════════════════

    async def stream_with_tools(
        self,
        messages: list[AgentMessage],
        tools: list[ToolSchema],
        model: str,
        system_prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 8192,
    ) -> AsyncIterator[LLMStreamChunk]:
        """
        Stream a completion with tool-use support.
        Yields LLMStreamChunk objects as they arrive.
        """
        if not model:
            raise LLMError(
                "No model specified. Set 'model' on AgentDefinition "
                "(e.g. 'openai:gpt-5.4', 'anthropic:claude-sonnet-4-6', 'ollama:qwen3:8b').",
                provider="",
                model="",
                retryable=False,
            )
        provider, model_name = _parse_model_string(model)

        if provider == "anthropic":
            async for chunk in self._stream_anthropic(
                messages, tools, model_name, system_prompt, temperature, max_tokens
            ):
                yield chunk
        elif provider == "google":
            response = await self._complete_google(
                messages, tools, model_name, system_prompt, temperature, max_tokens
            )
            yield LLMStreamChunk(
                type="complete",
                text=response.content,
                tool_call=None,
                usage=response.usage,
                stop_reason=response.stop_reason,
            )
        else:
            async for chunk in self._stream_openai(
                messages, tools, model_name, system_prompt, temperature, max_tokens, provider
            ):
                yield chunk

    async def complete_with_tools(
        self,
        messages: list[AgentMessage],
        tools: list[ToolSchema],
        model: str,
        system_prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 8192,
    ) -> LLMResponse:
        """Non-streaming completion — collects full response."""
        if not model:
            raise LLMError(
                "No model specified. Set 'model' on AgentDefinition "
                "(e.g. 'openai:gpt-5.4', 'anthropic:claude-sonnet-4-6', 'ollama:qwen3:8b').",
                provider="",
                model="",
                retryable=False,
            )
        provider, model_name = _parse_model_string(model)

        if provider == "anthropic":
            return await self._complete_anthropic(
                messages, tools, model_name, system_prompt, temperature, max_tokens
            )
        elif provider == "google":
            return await self._complete_google(
                messages, tools, model_name, system_prompt, temperature, max_tokens
            )
        else:
            return await self._complete_openai(
                messages, tools, model_name, system_prompt, temperature, max_tokens, provider
            )

    # ══════════════════════════════════════════════════════════════════
    # Anthropic
    # ══════════════════════════════════════════════════════════════════

    async def _stream_anthropic(
        self, messages, tools, model_name, system_prompt, temperature, max_tokens
    ) -> AsyncIterator[LLMStreamChunk]:
        client = await self._get_client("anthropic")
        api_messages, system_blocks = _messages_to_anthropic(messages, system_prompt)
        api_tools = _tools_to_anthropic(tools) if tools else []

        kwargs: dict[str, Any] = {
            "model": model_name,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "system": system_blocks,
            "messages": api_messages,
        }
        if api_tools:
            kwargs["tools"] = api_tools

        try:
            async with client.messages.stream(**kwargs) as stream:
                current_tool_id = None
                current_tool_name = None
                tool_buffers = []
                current_tool = None

                async for event in stream:
                    if event.type == "content_block_start":
                        block = event.content_block
                        if block.type == "tool_use":
                            current_tool_id = block.id
                            current_tool_name = block.name
                            current_tool = {'id': block.id, 'name': block.name, 'fragments': [], 'complete': False}
                            tool_buffers.append(current_tool)
                            yield LLMStreamChunk(
                                type="tool_call_start",
                                tool_call_id=current_tool_id,
                                tool_name=current_tool_name,
                            )
                    elif event.type == "content_block_delta":
                        delta = event.delta
                        if delta.type == "text_delta":
                            yield LLMStreamChunk(type="text_delta", text=delta.text)
                        elif delta.type == "thinking_delta":
                            yield LLMStreamChunk(type="thinking_delta", text=delta.thinking)
                        elif delta.type == "input_json_delta":
                            if current_tool is not None and delta.partial_json:
                                current_tool['fragments'].append(delta.partial_json)
                            yield LLMStreamChunk(
                                type="tool_call_delta",
                                tool_call_id=current_tool_id,
                                arguments_delta=delta.partial_json,
                            )
                    elif event.type == "content_block_stop":
                        if current_tool is not None:
                            current_tool['complete'] = True
                            current_tool_id = None
                            current_tool_name = None
                            current_tool = None

                final = await stream.get_final_message()
                usage = LLMUsage(
                    input_tokens=final.usage.input_tokens,
                    output_tokens=final.usage.output_tokens,
                    cache_creation_tokens=getattr(final.usage, "cache_creation_input_tokens", 0),
                    cache_read_tokens=getattr(final.usage, "cache_read_input_tokens", 0),
                    total_tokens=final.usage.input_tokens + final.usage.output_tokens,
                )
                stop = _map_stop(final.stop_reason)
                # Final stop metadata arrives after block_stop. Decode once,
                # with that metadata, before any consumer can execute a call.
                for buf in tool_buffers:
                    call = decode_tool_call(buf['id'], buf['name'], buf['fragments'], stop,
                                            complete=buf['complete'])
                    buf['fragments'].clear()
                    yield LLMStreamChunk(type='tool_call_end', tool_call=call)
                yield LLMStreamChunk(type="usage", usage=usage, stop_reason=stop)

        except Exception as e:
            # The native SDK's incremental parser includes the complete tool
            # JSON in this ValueError. Never forward that payload into logs,
            # chat errors or provider-error classification.
            if isinstance(e, ValueError) and str(e).startswith('Unable to parse tool parameter JSON from model.'):
                raise LLMError('Provider tool argument stream could not be decoded.',
                               provider='anthropic', model=model_name, retryable=False) from None
            self._handle_provider_error("anthropic", model_name, e)

    async def _complete_anthropic(
        self, messages, tools, model_name, system_prompt, temperature, max_tokens
    ) -> LLMResponse:
        client = await self._get_client("anthropic")
        api_messages, system_blocks = _messages_to_anthropic(messages, system_prompt)
        api_tools = _tools_to_anthropic(tools) if tools else []

        kwargs: dict[str, Any] = {
            "model": model_name,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "system": system_blocks,
            "messages": api_messages,
        }
        if api_tools:
            kwargs["tools"] = api_tools

        start = time.time()
        try:
            response = await client.messages.create(**kwargs)
        except Exception as e:
            self._handle_provider_error("anthropic", model_name, e)

        latency = (time.time() - start) * 1000

        text_parts = []
        tool_calls = []
        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append(
                    ToolCall(
                        id=block.id,
                        tool_name=block.name,
                        arguments=block.input,
                    )
                )

        stop = _map_stop(response.stop_reason)

        return LLMResponse(
            content="\n".join(text_parts),
            tool_calls=tool_calls,
            stop_reason=stop,
            model=f"anthropic:{model_name}",
            latency_ms=latency,
            usage=LLMUsage(
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
                cache_creation_tokens=getattr(response.usage, "cache_creation_input_tokens", 0),
                cache_read_tokens=getattr(response.usage, "cache_read_input_tokens", 0),
                total_tokens=response.usage.input_tokens + response.usage.output_tokens,
            ),
        )

    # ══════════════════════════════════════════════════════════════════
    # OpenAI-compatible (OpenAI, Cerebras, Together, etc.)
    # ══════════════════════════════════════════════════════════════════

    @staticmethod
    def _emit_content(splitter: "_ThinkSplitter", segments):
        for kind, seg in segments:
            if kind == "thinking":
                yield LLMStreamChunk(type="thinking_delta", text=seg)
            else:
                yield LLMStreamChunk(type="text_delta", text=seg)

    async def _stream_openai(
        self, messages, tools, model_name, system_prompt, temperature, max_tokens, provider
    ) -> AsyncIterator[LLMStreamChunk]:
        client = await self._get_client(provider)
        api_messages = _messages_to_openai(messages, system_prompt)
        api_tools = _tools_to_openai(tools) if tools else None

        kwargs: dict[str, Any] = {
            "model": model_name,
            "messages": api_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
            # Final stream chunk carries usage (incl. prompt cache hits)
            "stream_options": {"include_usage": True},
        }
        if api_tools:
            kwargs["tools"] = api_tools
            kwargs["tool_choice"] = "auto"

        # Reasoning support (Together AI — Qwen, DeepSeek, Kimi, etc.).
        # Keep effort low so thinking doesn't eat the max_tokens budget before
        # the model emits content or a tool call.
        model_lower = model_name.lower()
        if provider == "together" and (
            "qwen" in model_lower
            or "deepseek" in model_lower
            or "kimi" in model_lower
            or "moonshot" in model_lower
        ):
            kwargs["extra_body"] = {
                "reasoning": {"enabled": True},
                "reasoning_effort": "low",
            }

        try:
            log.debug(
                "llm_stream_request",
                provider=provider,
                model=model_name,
                tool_count=len(api_tools) if api_tools else 0,
                tool_choice=kwargs.get("tool_choice"),
            )
            stream = await client.chat.completions.create(**kwargs)

            tool_call_buffers: dict[int, dict[str, Any]] = {}
            usage_info: LLMUsage | None = None
            has_content = False
            has_tool_calls = False
            stop_reason = StopReason.END_TURN
            think_splitter = _ThinkSplitter()
            saw_reasoning_channel = False

            async for chunk in stream:
                # Usage is a cumulative provider snapshot, not a token
                # delta. OpenRouter can attach it to a normal choice;
                # native OpenAI commonly sends a choices=[] trailer.
                # Replace totals whenever supplied, then process content
                # independently so neither is dropped or counted twice.
                if chunk.usage is not None:
                    usage_info = LLMUsage(
                        input_tokens=chunk.usage.prompt_tokens or 0,
                        output_tokens=chunk.usage.completion_tokens or 0,
                        total_tokens=chunk.usage.total_tokens or 0,
                        cache_read_tokens=_openai_cached_tokens(chunk.usage),
                    )
                choice = chunk.choices[0] if chunk.choices else None
                if not choice:
                    continue

                delta = choice.delta

                # Thinking/reasoning tokens (Together, DeepSeek, Qwen, etc.)
                reasoning = getattr(delta, "reasoning_content", None) or getattr(delta, "reasoning", None)
                if reasoning:
                    saw_reasoning_channel = True
                    yield LLMStreamChunk(type="thinking_delta", text=reasoning)

                if delta and delta.content:
                    has_content = True
                    if saw_reasoning_channel:
                        # The model already delivers its chain-of-thought on the
                        # dedicated reasoning channel, so an inline <think> tag in
                        # the content channel is chat-template garbage, not real
                        # thought (Qwen3.7-Plus on Together emits a spurious
                        # unclosed "<think>\n" prefix that would otherwise swallow
                        # the entire answer into thinking). Strip literal tags and
                        # stream the content as text.
                        cleaned = delta.content.replace("<think>", "").replace("</think>", "")
                        if cleaned:
                            yield LLMStreamChunk(type="text_delta", text=cleaned)
                    else:
                        for out_chunk in self._emit_content(
                            think_splitter, think_splitter.feed(delta.content)
                        ):
                            yield out_chunk

                if delta and delta.tool_calls:
                    has_tool_calls = True
                    for tc_delta in delta.tool_calls:
                        idx = tc_delta.index
                        if idx not in tool_call_buffers:
                            tool_call_buffers[idx] = {
                                "id": tc_delta.id or "",
                                "name": tc_delta.function.name
                                if tc_delta.function and tc_delta.function.name
                                else "",
                                "fragments": [],
                            }
                            yield LLMStreamChunk(
                                type="tool_call_start",
                                tool_call_id=tool_call_buffers[idx]["id"],
                                tool_name=tool_call_buffers[idx]["name"],
                            )
                        if tc_delta.function and tc_delta.function.arguments:
                            tool_call_buffers[idx]["fragments"].append(tc_delta.function.arguments)
                        if tc_delta.id and not tool_call_buffers[idx]["id"]:
                            tool_call_buffers[idx]["id"] = tc_delta.id

                # Record the stop reason but do NOT flush here. Some
                # OpenAI-compatible providers (notably via OpenRouter, and with
                # include_usage) emit more than one chunk carrying a
                # finish_reason, plus a trailing usage-only chunk. Flushing
                # inside the loop re-emitted every buffered tool call on each
                # such chunk, so the engine executed each tool twice. Flush
                # exactly once, after the stream fully drains (below).
                if choice.finish_reason:
                    stop_reason = _map_stop(choice.finish_reason)

            # Flush any content held back for partial-tag matching.
            for out_chunk in self._emit_content(think_splitter, think_splitter.flush()):
                yield out_chunk

            # Single flush after the stream is exhausted — emits each buffered
            # tool call once and carries the final usage (captured from the
            # trailing include_usage chunk above).
            for idx, buf in sorted(tool_call_buffers.items()):
                call = decode_tool_call(buf['id'], buf['name'], buf['fragments'], stop_reason)
                buf['fragments'].clear()
                yield LLMStreamChunk(
                    type="tool_call_end",
                    tool_call=call,
                )
            yield LLMStreamChunk(
                type="usage",
                usage=usage_info or LLMUsage(),
                stop_reason=stop_reason,
            )

            log.info(
                "llm_stream_complete",
                provider=provider,
                model=model_name,
                has_content=has_content,
                has_tool_calls=has_tool_calls,
                tool_call_count=len(tool_call_buffers),
                input_tokens=usage_info.input_tokens if usage_info else 0,
                cache_read_tokens=usage_info.cache_read_tokens if usage_info else 0,
            )

        except Exception as e:
            self._handle_provider_error(provider, model_name, e)

    async def _complete_openai(
        self, messages, tools, model_name, system_prompt, temperature, max_tokens, provider
    ) -> LLMResponse:
        """Collect a full response by STREAMING and aggregating.

        The OpenAI-compatible surface has models that refuse a non-streamed
        request outright (Together's Qwen3.7-Plus answers one with HTTP 400
        `streaming_required`), and every provider here supports streaming — so
        the collected path rides the streamed one rather than keeping a
        per-model capability table that goes stale on the next model swap.
        Callers still get a single aggregated LLMResponse.
        """
        start = time.time()
        content_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        usage: LLMUsage | None = None
        stop_reason: StopReason | None = None

        async for chunk in self._stream_openai(
            messages, tools, model_name, system_prompt, temperature, max_tokens, provider
        ):
            if chunk.type == "text_delta" and chunk.text:
                content_parts.append(chunk.text)
            elif chunk.type == "tool_call_end" and chunk.tool_call:
                tool_calls.append(chunk.tool_call)
            elif chunk.type == "usage":
                usage = chunk.usage or usage
                stop_reason = chunk.stop_reason or stop_reason

        if stop_reason is None:
            stop_reason = StopReason.TOOL_USE if tool_calls else StopReason.END_TURN

        return LLMResponse(
            content="".join(content_parts),
            tool_calls=tool_calls,
            stop_reason=stop_reason,
            model=f"{provider}:{model_name}",
            latency_ms=(time.time() - start) * 1000,
            usage=usage or LLMUsage(),
        )

    # ══════════════════════════════════════════════════════════════════
    # Google Gemini (non-streaming for now)
    # ══════════════════════════════════════════════════════════════════

    async def _complete_google(
        self, messages, tools, model_name, system_prompt, temperature, max_tokens
    ) -> LLMResponse:
        """Google Gemini completion via google-genai SDK."""
        client = await self._get_client("google")
        from google.genai import types

        contents = []
        for msg in messages:
            if msg.role == MessageRole.USER:
                contents.append(
                    types.Content(
                        role="user",
                        parts=[
                            types.Part(
                                text=msg.text() if isinstance(msg.content, list) else msg.content
                            )
                        ],
                    )
                )
            elif msg.role == MessageRole.ASSISTANT:
                parts = []
                text = msg.text() if isinstance(msg.content, list) else msg.content
                if text:
                    parts.append(types.Part(text=text))
                # Re-emit the assistant's tool calls as functionCall parts, or
                # a later functionResponse has nothing to answer and the model
                # re-calls or hallucinates — multi-turn tool use is broken
                # without this.
                for tc in msg.tool_calls or []:
                    parts.append(
                        types.Part(
                            function_call=types.FunctionCall(
                                name=tc.tool_name, args=tc.arguments or {}
                            )
                        )
                    )
                contents.append(
                    types.Content(role="model", parts=parts or [types.Part(text="")])
                )
            elif msg.role == MessageRole.TOOL_RESULT:
                # A tool's output MUST reach the model as a functionResponse,
                # or the model never sees what its tool returned.
                response_payload = (
                    msg.content
                    if isinstance(msg.content, dict)
                    else {"result": msg.content}
                )
                contents.append(
                    types.Content(
                        role="user",
                        parts=[
                            types.Part(
                                function_response=types.FunctionResponse(
                                    name=msg.tool_name or msg.tool_call_id or "tool",
                                    response=response_payload,
                                )
                            )
                        ],
                    )
                )

        gemini_tools = []
        if tools:
            function_declarations = []
            for t in tools:
                function_declarations.append(
                    types.FunctionDeclaration(
                        name=t.name,
                        description=t.description,
                        parameters=t.parameters,
                    )
                )
            gemini_tools = [types.Tool(function_declarations=function_declarations)]

        start = time.time()
        try:
            response = await asyncio.to_thread(
                client.models.generate_content,
                model=model_name,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                    tools=gemini_tools if gemini_tools else None,
                ),
            )
        except Exception as e:
            self._handle_provider_error("google", model_name, e)

        latency = (time.time() - start) * 1000

        text_parts = []
        tool_calls = []
        if response.candidates:
            for part in response.candidates[0].content.parts:
                if part.text:
                    text_parts.append(part.text)
                elif part.function_call:
                    tool_calls.append(
                        ToolCall(
                            tool_name=part.function_call.name,
                            arguments=dict(part.function_call.args)
                            if part.function_call.args
                            else {},
                        )
                    )

        _gem_reason = None
        try:
            _cand = (response.candidates or [None])[0]
            _gem_reason = str(getattr(_cand, "finish_reason", "") or "").lower()
        except Exception:
            pass
        stop = (
            StopReason.MAX_TOKENS
            if _gem_reason and "max_tokens" in _gem_reason
            else _map_stop(None, has_tool_calls=bool(tool_calls))
        )
        usage_meta = response.usage_metadata
        return LLMResponse(
            content="\n".join(text_parts),
            tool_calls=tool_calls,
            stop_reason=stop,
            model=f"google:{model_name}",
            latency_ms=latency,
            usage=LLMUsage(
                input_tokens=getattr(usage_meta, "prompt_token_count", 0) or 0,
                output_tokens=getattr(usage_meta, "candidates_token_count", 0) or 0,
                total_tokens=getattr(usage_meta, "total_token_count", 0) or 0,
            ),
        )

    # ══════════════════════════════════════════════════════════════════
    # Error handling
    # ══════════════════════════════════════════════════════════════════

    def _handle_provider_error(self, provider: str, model: str, error: Exception) -> None:
        error_str = str(error).lower()

        if "rate" in error_str and "limit" in error_str:
            raise LLMRateLimitError(str(error), provider=provider, model=model) from error

        if "context" in error_str and (
            "length" in error_str or "window" in error_str or "overflow" in error_str
        ):
            raise ContextOverflowError(str(error), provider=provider, model=model) from error

        raise LLMError(
            str(error),
            provider=provider,
            model=model,
            retryable="timeout" in error_str or "connection" in error_str,
        ) from error
