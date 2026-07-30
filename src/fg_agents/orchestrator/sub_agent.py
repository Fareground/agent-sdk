"""
Fareground Agent Framework — Sub-Agent Runner

Full lifecycle management for sub-agents:
- Spawn: create a new sub-agent with a task
- Message: send follow-up messages to an existing sub-agent
- Status: check if a sub-agent is still running or completed

Each sub-agent gets its own DB session, linked to the parent via
parent_session_id. Sessions persist — the orchestrator can return
to a sub-agent across multiple turns.
"""

import json
from collections.abc import Awaitable, Callable
from typing import Any

import structlog

from fg_agents.core.engine import AgentEngine
from fg_agents.core.types import (
    AgentDefinition,
    AgentSession,
    LLMUsage,
    MessageRole,
    SubAgentResult,
    ToolStatus,
    _uuid,
)
from fg_agents.streaming.events import StreamEvent, subagent_completed, subagent_started

log = structlog.get_logger("fg_agents.sub_agent")

EventCallback = Callable[[StreamEvent], Awaitable[None]]


class SubAgentRunner:
    """
    Spawn and manage sub-agents with isolated context.

    Design:
    - Each sub-agent creates a CHILD session in DB (linked via parent_session_id)
    - Sub-agent gets a fresh message history — only the task description + optional context
    - Runs its own ReAct loop via the shared AgentEngine
    - Returns ONLY the final result to the parent (full conversation persisted in child)
    - All intermediate reasoning stays isolated (auditable via child session)
    - Optional on_event callback streams events in real-time (e.g. to WebSocket)
    """

    def __init__(self, engine: AgentEngine):
        self._engine = engine

    async def run(
        self,
        task: str,
        agent_def: AgentDefinition,
        parent_session_id: str,
        context_data: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        on_event: EventCallback | None = None,
    ) -> SubAgentResult:
        """
        Run a sub-agent to completion and return its result.

        Args:
            task: Clear description of what the sub-agent should do
            agent_def: The sub-agent's definition (model, tools, skills, prompt)
            parent_session_id: Parent session for linking
            context_data: Optional data from parent to pass as context
            metadata: Additional metadata for the child session
            on_event: Optional async callback invoked for every stream event,
                      enabling real-time forwarding to WebSocket / UI.

        Returns:
            SubAgentResult with the final output text, data, and usage stats
        """
        child_session_id = _uuid()

        child_metadata = {
            "parent_session_id": parent_session_id,
            "task_preview": task[:200],
            **(metadata or {}),
        }

        user_message = self._build_task_message(task, context_data)

        # Auto-inject schema context from parent metadata if available
        schema_hint = child_metadata.get("schema_hint", "")
        if schema_hint and schema_hint not in user_message:
            user_message += f"\n\n## Available Schema\n\n{schema_hint}"

        log.info(
            "subagent_start",
            child_session_id=child_session_id,
            parent_session_id=parent_session_id,
            agent_name=agent_def.name,
            task_preview=task[:100],
        )

        # Strip non-serializable objects (like LLM instances) for DB persistence
        db_metadata = {
            k: v
            for k, v in child_metadata.items()
            if isinstance(v, (str, int, float, bool, list, dict, type(None)))
        }
        # Inherit user_id/tenant_id from parent session for proper isolation
        parent_user_id = ""
        parent_tenant_id = ""
        try:
            parent_session = await self._engine.repository.get_session(parent_session_id)
            if parent_session:
                parent_user_id = parent_session.user_id or ""
                parent_tenant_id = parent_session.tenant_id or ""
        except Exception:
            pass

        child_session = AgentSession(
            id=child_session_id,
            agent_id=agent_def.id,
            user_id=parent_user_id,
            tenant_id=parent_tenant_id,
            parent_session_id=parent_session_id,
            metadata=db_metadata,
        )
        await self._engine.repository.create_session(child_session)

        final_text = ""
        total_usage = LLMUsage()
        turn_count = 0
        error_msg: str | None = None

        # Emit subagent.started event
        if on_event:
            try:
                await on_event(
                    subagent_started(
                        parent_session_id,
                        child_session_id,
                        agent_def.name,
                        task[:200],
                    )
                )
            except Exception:
                pass

        import asyncio

        from fg_agents.core.types import AgentMessage, MessageRole

        # Wrap-up nudge + hard cancel, from the agent's definition
        # (defaults: nudge at 3 min, cancel at 10 min).
        nudge_after = agent_def.nudge_after_seconds
        hard_deadline = agent_def.hard_deadline_seconds

        completed = asyncio.Event()

        async def _run_child():
            nonlocal final_text, total_usage, turn_count, error_msg
            async for event in self._engine.run(
                session_id=child_session_id,
                user_message=user_message,
                agent_def=agent_def,
                metadata=child_metadata,
            ):
                if on_event:
                    try:
                        await on_event(event)
                    except Exception as cb_err:
                        log.warning("on_event_callback_error", error=str(cb_err))

                if event.type.value == "session.completed":
                    final_text = event.data.get("final_output", "")
                    total_usage.input_tokens = event.data.get("input_tokens", 0)
                    total_usage.output_tokens = event.data.get("output_tokens", 0)
                    turn_count = event.data.get("total_turns", 0)
                elif event.type.value == "error":
                    error_msg = event.data.get("message", "Unknown error")
            completed.set()

        async def _nudge_timer():
            """Inject a wrap-up message if the child hasn't completed after nudge_after."""
            try:
                await asyncio.wait_for(asyncio.shield(completed.wait()), timeout=nudge_after)
            except TimeoutError:
                if not completed.is_set():
                    log.warning(
                        "subagent_nudge",
                        child_session_id=child_session_id,
                        nudge_after=nudge_after,
                    )
                    nudge = AgentMessage(
                        session_id=child_session_id,
                        role=MessageRole.SYSTEM,
                        content=(
                            f"You have been running for {nudge_after // 60} minutes. "
                            "Stop what you are doing. Write your report NOW with whatever "
                            "findings you have so far, then call session_complete immediately."
                        ),
                    )
                    try:
                        await self._engine.repository.add_message(nudge)
                    except Exception as e:
                        log.warning("nudge_inject_failed", error=str(e))

        child_task = asyncio.ensure_future(_run_child())
        nudge_task = asyncio.ensure_future(_nudge_timer())

        try:
            await asyncio.wait_for(asyncio.shield(child_task), timeout=hard_deadline)
        except TimeoutError:
            log.warning(
                "subagent_hard_deadline",
                child_session_id=child_session_id,
                hard_deadline=hard_deadline,
            )
            await self._engine.cancel(child_session_id)
            # Give the child a moment to finish its current turn after cancel
            try:
                await asyncio.wait_for(child_task, timeout=30)
            except (TimeoutError, asyncio.CancelledError):
                pass
            error_msg = f"Task exceeded {hard_deadline}s — partial results captured"
        except Exception as e:
            log.error("subagent_error", child_session_id=child_session_id, error=str(e)[:500])
            error_msg = str(e)
        finally:
            nudge_task.cancel()

        # Robust result capture — 3-tier fallback.
        # NOTE: the session.completed event truncates final_output to 2000 chars
        # (streaming/events.py), so `final_text` captured from the event may be
        # cut off. Always recover the FULL report from the child's message
        # history and prefer it when it's longer, so the parent receives the
        # complete report on the first return (no separate get_report needed).
        messages = await self._engine.repository.get_messages(child_session_id)

        # Tier 1: last substantial assistant message — the agent's report.
        full_report = ""
        for m in reversed(messages):
            if m.role == MessageRole.ASSISTANT and m.content:
                text = m.text() if isinstance(m.content, list) else str(m.content)
                if text and text.strip():
                    full_report = text.strip()
                    break
        if len(full_report) > len(final_text or ""):
            final_text = full_report

        # Tier 2: compile from tool results (the actual work product)
        if not final_text:
            tool_outputs = []
            for m in reversed(messages):
                if m.role == MessageRole.TOOL_RESULT and m.content:
                    content = m.content if isinstance(m.content, str) else str(m.content)
                    if content.strip() and len(content.strip()) > 20:
                        label = m.tool_name or "tool"
                        tool_outputs.append(f"[{label}] {content.strip()[:3000]}")
                        if len(tool_outputs) >= 10:
                            break
            if tool_outputs:
                tool_outputs.reverse()
                final_text = "Sub-agent results:\n\n" + "\n\n".join(tool_outputs)

        # Tier 3: minimal summary so the orchestrator knows something happened
        if not final_text:
            final_text = (
                f"Sub-agent '{agent_def.name}' completed {turn_count} turns. "
                f"No text output was produced. Check child session {child_session_id} for details."
            )

        # Auto-close: if the sub-agent finished without calling session_complete,
        # synthesize the call so (a) the UI shows a clean completion card, and
        # (b) the parent receives `final_text` packaged the same way as an
        # explicit call. This eliminates the "agent stranded mid-message"
        # symptom when the model forgets the closing protocol.
        try:
            from fg_agents.core.types import ToolCall
            msgs_check = await self._engine.repository.get_messages(child_session_id)
            called = any(
                tc.tool_name == "session_complete"
                for m in msgs_check
                if m.tool_calls
                for tc in m.tool_calls
            )
            if not called and final_text and len(final_text.strip()) > 0:
                summary = final_text.strip()
                if len(summary) < 50:
                    summary = (summary + " ").ljust(50, ".")
                synth_call = ToolCall(tool_name="session_complete", arguments={"summary": summary})
                await self._engine.repository.add_message(AgentMessage(
                    session_id=child_session_id,
                    role=MessageRole.ASSISTANT,
                    content="",
                    tool_calls=[synth_call],
                ))
                await self._engine.repository.add_message(AgentMessage(
                    session_id=child_session_id,
                    role=MessageRole.TOOL_RESULT,
                    content="Session marked complete (auto-synthesized — agent did not call session_complete explicitly).",
                    tool_name="session_complete",
                    tool_call_id=synth_call.id,
                ))
                # (Working memory is per-run, not stored on the AgentSession —
                # the auto-synthesized TOOL_RESULT above is what records the
                # completion; there is no session-level flag to set here.)
                log.info(
                    "subagent_session_complete_synthesized",
                    child_session_id=child_session_id,
                    agent=agent_def.name,
                    summary_chars=len(summary),
                )
        except Exception as e:
            log.warning("subagent_auto_close_failed", error=str(e)[:300])

        log.info(
            "subagent_complete",
            child_session_id=child_session_id,
            turns=turn_count,
            tokens=total_usage.total_tokens,
            output_length=len(final_text),
            has_error=bool(error_msg),
        )

        # If we have useful output, treat as success even if there was an error
        # (the error details are appended to the output so the orchestrator can see them)
        has_useful_output = bool(final_text and len(final_text) > 50)
        if error_msg and has_useful_output:
            final_text += f"\n\n[Note: sub-agent encountered an error: {error_msg}]"

        # Emit subagent.completed event
        status_val = "success" if has_useful_output else ("error" if error_msg else "success")
        if on_event:
            try:
                await on_event(
                    subagent_completed(
                        parent_session_id,
                        child_session_id,
                        agent_def.name,
                        status_val,
                        turn_count,
                    )
                )
            except Exception:
                pass

        return SubAgentResult(
            child_session_id=child_session_id,
            output=final_text,
            data={"task": task},
            status=ToolStatus.SUCCESS
            if has_useful_output
            else (ToolStatus.ERROR if error_msg else ToolStatus.SUCCESS),
            error=error_msg if not has_useful_output else None,
            turns_used=turn_count,
            tokens_used=total_usage,
        )

    async def send_message(
        self,
        child_session_id: str,
        message: str,
        agent_def: AgentDefinition,
        metadata: dict[str, Any] | None = None,
        on_event: EventCallback | None = None,
    ) -> SubAgentResult:
        """
        Send a follow-up message to an existing sub-agent session.

        The sub-agent resumes its conversation with full history intact.
        Use this to refine, redirect, or ask follow-up questions.

        Args:
            child_session_id: The session ID of the sub-agent to message
            message: The follow-up instruction
            agent_def: The sub-agent's definition (for model/tools config)
            metadata: Additional metadata
            on_event: Optional event callback

        Returns:
            SubAgentResult with the new response
        """
        log.info(
            "subagent_followup", child_session_id=child_session_id, message_preview=message[:100]
        )

        final_text = ""
        total_usage = LLMUsage()
        turn_count = 0
        error_msg: str | None = None

        try:
            async for event in self._engine.run(
                session_id=child_session_id,
                user_message=message,
                agent_def=agent_def,
                metadata=metadata or {},
            ):
                if on_event:
                    try:
                        await on_event(event)
                    except Exception as cb_err:
                        log.warning("on_event_callback_error", error=str(cb_err))

                if event.type.value == "session.completed":
                    final_text = event.data.get("final_output", "")
                    total_usage.input_tokens = event.data.get("input_tokens", 0)
                    total_usage.output_tokens = event.data.get("output_tokens", 0)
                    turn_count = event.data.get("total_turns", 0)
                elif event.type.value == "error":
                    error_msg = event.data.get("message", "Unknown error")

        except Exception as e:
            log.error(
                "subagent_followup_error", child_session_id=child_session_id, error=str(e)[:500]
            )
            error_msg = str(e)

        # Recover the FULL response from message history — the session.completed
        # event truncates final_output to 2000 chars (streaming/events.py), so
        # prefer the complete last assistant message when it's longer.
        messages = await self._engine.repository.get_messages(child_session_id)
        full_report = ""
        for m in reversed(messages):
            if m.role == MessageRole.ASSISTANT and m.content:
                text = m.text() if isinstance(m.content, list) else str(m.content)
                if text and text.strip():
                    full_report = text.strip()
                    break
        if len(full_report) > len(final_text or ""):
            final_text = full_report

        if not final_text:
            final_text = f"Sub-agent follow-up completed {turn_count} turns with no text output."

        return SubAgentResult(
            child_session_id=child_session_id,
            output=final_text,
            data={"follow_up": message[:200]},
            status=ToolStatus.SUCCESS if final_text else ToolStatus.ERROR,
            error=error_msg,
            turns_used=turn_count,
            tokens_used=total_usage,
        )

    @staticmethod
    def _build_task_message(task: str, context_data: dict | None = None) -> str:
        """Build the initial message for the sub-agent."""
        parts = [task]
        if context_data:
            parts.append("\n## Context Data\n")
            for key, value in context_data.items():
                if isinstance(value, (dict, list)):
                    val_str = json.dumps(value, indent=2, default=str)
                    if len(val_str) > 3000:
                        val_str = val_str[:3000] + "\n... (truncated)"
                else:
                    val_str = str(value)
                parts.append(f"**{key}**:\n{val_str}\n")
        return "\n".join(parts)
