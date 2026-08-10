"""
00 — Quickstart

The fastest way in: the Agent facade. One object, no wiring —
LLM, tools, persistence, and the engine are set up for you.
"""

import asyncio

from fg_agents import Agent


def greet(name: str) -> str:
    """Greet someone by name."""
    return f"Hello, {name}! Welcome to Fareground."


async def main():
    agent = Agent(
        model="anthropic:claude-sonnet-4-6",  # any provider:model you have a key for
        tools=[greet],
        system_prompt="You are a friendly greeter.",
    )

    result = await agent.run("Please greet Ada.")
    print(result.text)


if __name__ == "__main__":
    asyncio.run(main())
