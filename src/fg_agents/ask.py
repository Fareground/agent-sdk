"""
Fareground Agent Framework — One-shot free functions

The three-line hello world::

    from fg_agents import ask

    print(await ask("What's 2+2?"))

``ask`` (and its streaming sibling ``stream``) construct an ephemeral
:class:`~fg_agents.Agent`, run the prompt, and clean up — no client
object, no session management, no explicit model when an API key env
var is set (see :func:`~fg_agents.resolve_default_model` for the
detection order).

Need multi-turn conversations, shared tools, or a persistent backend?
Graduate to :class:`~fg_agents.Agent`::

    async with Agent(tools=[my_func]) as agent:
        first = await agent.run("hello")
        second = await agent.run("tell me more")  # same conversation
"""

from collections.abc import AsyncIterator, Callable

from fg_agents.agent import Agent, AgentRunResult
from fg_agents.core.types import RegisteredTool
from fg_agents.persistence.base import BaseRepository
from fg_agents.streaming.events import StreamEvent


async def ask(
    prompt: str,
    *,
    model: str | None = None,
    tools: list[Callable | RegisteredTool] | None = None,
    system_prompt: str = "",
    memory: str | BaseRepository = "memory",
    **kwargs,
) -> AgentRunResult:
    """
    Ask a one-shot question and get the final answer.

    Example::

        from fg_agents import ask

        print(await ask("What's 2+2?"))

    Args:
        prompt: The user message.
        model: Provider-qualified model id (``"anthropic:claude-sonnet-4-6"``).
            Omit to auto-detect from the environment.
        tools: Tools available to the agent (``@tool`` functions, plain
            callables, or ``RegisteredTool`` instances).
        system_prompt: System prompt for the run.
        memory: Persistence backend — ``"memory"`` (default), ``"sqlite[:path]"``,
            ``"postgres:<url>"``, or a ready repository.
        **kwargs: Everything else :class:`~fg_agents.Agent` accepts
            (``api_keys``, ``max_turns``, ``temperature``, ...).

    Returns:
        :class:`~fg_agents.AgentRunResult` — ``str()`` of it is the answer
        text, so ``print(await ask(...))`` prints just the answer.
    """
    async with Agent(
        model, tools=tools, system_prompt=system_prompt, memory=memory, **kwargs
    ) as agent:
        return await agent.run(prompt)


async def stream(
    prompt: str,
    *,
    model: str | None = None,
    tools: list[Callable | RegisteredTool] | None = None,
    system_prompt: str = "",
    memory: str | BaseRepository = "memory",
    **kwargs,
) -> AsyncIterator[StreamEvent]:
    """
    One-shot question, streamed: yields :class:`StreamEvent` s as they happen.

    Example::

        from fg_agents import stream

        async for event in stream("Tell me a story"):
            print(event.type, event.data)

    Accepts the same arguments as :func:`ask`. The ephemeral agent is
    closed when the stream is exhausted (or the iterator is closed).
    """
    async with Agent(
        model, tools=tools, system_prompt=system_prompt, memory=memory, **kwargs
    ) as agent:
        async for event in agent.stream(prompt):
            yield event
