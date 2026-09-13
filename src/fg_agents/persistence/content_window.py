"""Explicit content projections for consumers that already page tool output.

These are NOT messages ready for an LLM. A partial window has an empty message
body and separate original-character offsets; consumers must render their own
honest recovery envelope. Stored originals and normal get_messages are unchanged.
"""
from dataclasses import dataclass

from fg_agents.core.types import AgentMessage, MessageRole


def validate_content_window(head_chars: int, tail_chars: int, exempt_tools: tuple[str, ...]) -> None:
    if (type(head_chars) is not int or not 1 <= head_chars <= 64_000
            or type(tail_chars) is not int or not 0 <= tail_chars <= head_chars):
        raise ValueError("content window requires 1..64000 head characters and 0..head tail characters")
    if len(exempt_tools) > 64 or any(not isinstance(s, str) or len(s) > 256 for s in exempt_tools):
        raise ValueError("exempt_tools must contain at most 64 tool names of at most 256 characters")


@dataclass(frozen=True)
class MessageContentWindow:
    message: AgentMessage
    # None means complete content, including non-text/multimodal messages.
    total_characters: int | None = None
    head: str = ""
    tail: str = ""

    @property
    def partial(self) -> bool:
        return self.total_characters is not None

    @classmethod
    def from_message(cls, message: AgentMessage, *, head_chars: int, tail_chars: int,
                     exempt_tools: tuple[str, ...]):
        text = message.content
        if (message.role != MessageRole.TOOL_RESULT or message.tool_name in exempt_tools
                or message.tool_calls or not isinstance(text, str)
                or len(text) <= head_chars + tail_chars):
            return cls(message)
        return cls(message.model_copy(update={'content': '', 'token_count': 0}),
                   len(text), text[:head_chars], text[-tail_chars:] if tail_chars else '')
