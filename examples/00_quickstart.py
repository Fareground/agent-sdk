"""
00 — Quickstart

The fastest way in: the ask() one-shot. With an API key env var set
(ANTHROPIC_API_KEY, OPENAI_API_KEY, GOOGLE_API_KEY — or a local Ollama
server running), the model is detected for you.
"""

import asyncio

from fg_agents import Agent, ask


def greet(name: str) -> str:
    """Greet someone by name."""
    return f"Hello, {name}! Welcome to Fareground."


async def main():
    # One-shot: three lines is the whole program.
    print(await ask("What's 2+2?"))

    # Same thing with tools and a system prompt:
    print(await ask("Please greet Ada.", tools=[greet], system_prompt="You are a friendly greeter."))

    # Multi-turn: graduate to the Agent facade (still no wiring).
    async with Agent(tools=[greet], system_prompt="You are a friendly greeter.") as agent:
        print(await agent.run("Greet Grace."))
        print(await agent.run("Who did you just greet?"))  # same conversation


if __name__ == "__main__":
    asyncio.run(main())
