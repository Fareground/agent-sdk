"""
Fareground Agent Framework — ReAct Agent Engine

The core agent loop. Provider-agnostic. DB-backed. Streaming.

Flow:
  1. Load/create session from DB
  2. Build context: system_prompt + skills + memory + messages
  3. Call LLM with tools (streaming) — with retry on transient errors
  4. If tool_calls → execute tools → persist → loop back to 3
  5. If end_turn → persist → finalize
  6. Stream events throughout

Production-grade ReAct loop implementation.
"""

import asyncio
import json
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import Any

import structlog

from fg_agents.core.errors import (
    ContextOverflowError,
    LLMError,
    LLMRateLimitError,
    RateLimitExceededError,
)
from fg_agents.core.llm import AgentLLM
from fg_agents.core.memory_scope import scoped_agent_key
from fg_agents.core.types import (
    AgentDefinition,
    AgentMessage,
    AgentSession,
    ExecutionContext,
    LLMResponse,
    LLMUsage,
    MessageRole,
    SessionStatus,
    StopReason,
    ToolCall,
    ToolResult,
    ToolSchema,
    ToolStatus,
)
from fg_agents.memory.context import ContextManager
from fg_agents.memory.health import (
    CONTEXT_HARD_STOP_MESSAGE,
    CONTEXT_WARNING_MESSAGE,
    ContextHealthConfig,
    assess_health,
)
from fg_agents.memory.working import WorkingMemory
from fg_agents.middleware.base import BaseMiddleware
from fg_agents.persistence.base import BaseRepository
from fg_agents.prompts.templates import build_system_prompt
from fg_agents.skills.manager import SkillsManager
from fg_agents.streaming.events import (
    StreamEvent,
    error_event,
    session_completed,
    session_started,
    text_delta,
    thinking_delta,
    tool_call_event,
    tool_executing,
    tool_result_event,
    turn_completed,
    turn_started,
)
from fg_agents.tools.registry import ToolRegistry

log = structlog.get_logger("fg_agents.engine")


def _json_safe(obj: Any) -> Any:
    """Recursively strip non-JSON-serializable values from a dict/list tree."""
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            safe = _json_safe(v)
            if safe is not _SKIP:
                out[k] = safe
        return out
    if isinstance(obj, (list, tuple)):
        cleaned = (_json_safe(x) for x in obj)
        return [x for x in cleaned if x is not _SKIP]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    try:
        json.dumps(obj, default=str)
        return str(obj)
    except (TypeError, ValueError):
        return _SKIP


_SKIP = object()


class AgentEngine:
    """
    The ReAct agent loop engine.

    Orchestrates: LLM calls → tool execution → message persistence → streaming.
    Stateless between runs — all state lives in the repository.

    Session locking: Only one run() can execute per session_id at a time.
    Concurrent requests to the same session wait in line.
    """

    # Max CONSECUTIVE no-tool-call nudges before we give up and finalize a
    # premature-stopping agent (reset whenever a real tool call happens). Keeps
    # require_explicit_completion / min_turns from ever looping to max_turns on a
    # model that simply refuses to act.
    _MAX_CONTINUE_NUDGES = 3

    def __init__(
        self,
        llm: AgentLLM,
        tool_registry: ToolRegistry,
        repository: BaseRepository,
        skills_manager: SkillsManager | None = None,
        middleware: list[BaseMiddleware] | None = None,
    ):
        self._llm = llm
        self._tools = tool_registry
        self._repo = repository
        self._skills = skills_manager or SkillsManager()
        self._middleware = middleware or []
        self._context_manager = ContextManager(repository)
        self._session_locks: dict[str, asyncio.Lock] = {}
        # Cooperative cancellation scoped by a monotonic sequence number. A
        # cancel records the current counter into `_cancel_seq[session_id]`;
        # each run captures that value as its baseline at start and stops only
        # if a LATER cancel arrives (seq > baseline). This way a cancel only
        # ever stops the run that was in flight when it was issued — a leftover
        # cancel can no longer kill a freshly-started run at turn 0 (the
        # "Session cancelled by user" right after sending a new message bug).
        self._cancel_counter: int = 0
        self._cancel_seq: dict[str, int] = {}

    @property
    def repository(self) -> BaseRepository:
        """Public accessor for the persistence repository."""
        return self._repo

    async def cancel(self, session_id: str, cascade: bool = True) -> int:
        """
        Nuclear stop — cancel a running session and optionally all its children.

        Args:
            session_id: The session to cancel.
            cascade: If True (default), also cancel all child sessions
                     (sub-agents spawned by this session).

        Returns:
            Number of sessions cancelled (including children).
        """
        cancelled_count = 0

        # Mark for cancellation — the run loop compares this monotonic seq to the
        # baseline it captured at start, so only the in-flight run stops.
        self._cancel_counter += 1
        cancel_at = self._cancel_counter
        self._cancel_seq[session_id] = cancel_at
        await self._repo.update_session(
            session_id,
            status=SessionStatus.CANCELLED,
            error="Cancelled by user",
        )
        cancelled_count += 1
        log.info("session_cancelled", session_id=session_id, cascade=cascade)

        if cascade:
            # Find and cancel all child sessions
            children = await self._repo.get_child_sessions(session_id)
            for child in children:
                child_id = child.id if hasattr(child, "id") else child.get("id", "")
                if child_id:
                    self._cancel_seq[child_id] = cancel_at
                    await self._repo.update_session(
                        child_id,
                        status=SessionStatus.CANCELLED,
                        error="Cancelled (parent cancelled)",
                    )
                    cancelled_count += 1
                    log.info("child_session_cancelled", child_id=child_id, parent_id=session_id)

        return cancelled_count

    async def run(
        self,
        session_id: str,
        user_message: str,
        agent_def: AgentDefinition,
        metadata: dict | None = None,
        variables: dict | None = None,
    ) -> AsyncIterator[StreamEvent]:
        """
        Execute the agent loop for a session.

        Creates or resumes a session, adds the user message, and runs
        the ReAct loop until completion or max_turns.

        Yields StreamEvent objects for real-time progress tracking.
        """
        # Acquire per-session lock to prevent concurrent writes
        if session_id not in self._session_locks:
            self._session_locks[session_id] = asyncio.Lock()
        session_lock = self._session_locks[session_id]

        async with session_lock:
            async for event in self._run_locked(
                session_id,
                user_message,
                agent_def,
                metadata,
                variables,
            ):
                yield event

        # Clean up lock if no longer needed
        if session_id in self._session_locks and not session_lock.locked():
            self._session_locks.pop(session_id, None)

    async def _run_locked(
        self,
        session_id: str,
        user_message: str,
        agent_def: AgentDefinition,
        metadata: dict | None = None,
        variables: dict | None = None,
    ) -> AsyncIterator[StreamEvent]:
        """Internal: execute the agent loop under session lock."""
        session = await self._ensure_session(session_id, agent_def, metadata)

        # STOP is a LOCAL interrupt, not a permanent session kill. A new run
        # means the user wants to continue. Cancellation is sequence-scoped, so
        # there is no leftover flag to clear — this run captures its baseline
        # below and ignores any cancel issued before it started. We only lift a
        # CANCELLED status back to RUNNING so the session resumes.
        if session.status == SessionStatus.CANCELLED:
            await self._repo.update_session(session_id, status=SessionStatus.RUNNING)

        # Cancel baseline: the value of `_cancel_seq` at the moment this run
        # starts. Only a cancel issued AFTER this point (seq strictly greater)
        # stops this run; a leftover cancel from a previously-aborted run has
        # seq <= baseline and is correctly ignored. This is what prevents a
        # fresh message from being killed at turn 0 by a prior run's cancel.
        run_cancel_baseline = self._cancel_seq.get(session_id, 0)

        # Merge stored session metadata with runtime metadata.
        # Session metadata (from DB) serves as the base; runtime metadata
        # (from caller) takes precedence. This ensures tools always see
        # session-level config like connection_id even when the engine is
        # invoked directly (e.g. by SubAgentRunner) rather than through
        # AgentService.send_message.
        effective_metadata = dict(session.metadata) if session.metadata else {}
        if metadata:
            effective_metadata.update(metadata)

        working_memory = WorkingMemory(session_id=session_id, metadata=effective_metadata)

        context = ExecutionContext(
            session_id=session_id,
            agent_id=agent_def.id,
            agent_name=agent_def.name,
            parent_session_id=session.parent_session_id,
            # Identity comes ONLY from the persisted session (stamped from the
            # verified auth context at creation) — never from request-body
            # metadata, which a client controls. A metadata fallback here would
            # let a caller override the tenant of a session whose stored tenant
            # is empty.
            user_id=session.user_id,
            tenant_id=session.tenant_id,
            metadata=effective_metadata,
            working_memory=working_memory,
            repository=self._repo,
        )

        # Persist user message
        user_msg = AgentMessage(
            session_id=session_id,
            role=MessageRole.USER,
            content=user_message,
        )
        await self._repo.add_message(user_msg)
        await self._repo.update_session(session_id, status=SessionStatus.RUNNING)

        yield session_started(session_id, agent_def.id, agent_def.name)

        # Resolve skills (including runtime-injected) and build tool list
        skills = self._skills.resolve(agent_def.skills)
        if context.active_skills:
            skills = skills + list(context.active_skills)
        skill_tools = self._skills.get_required_tools(skills)
        all_tool_names = list(set(agent_def.tools + skill_tools))
        tool_schemas = self._tools.get_schemas(all_tool_names)

        turn = 0
        total_usage = LLMUsage()
        llm_response = LLMResponse()
        thinking_text = ""

        try:
            while turn < agent_def.max_turns:
                # Cooperative stop check — exit if a cancel was issued during
                # THIS run (seq strictly greater than the baseline captured at
                # start). A leftover cancel from a prior run has seq <= baseline
                # and is correctly ignored, so a freshly-sent message is never
                # killed at turn 0.
                if self._cancel_seq.get(session_id, 0) > run_cancel_baseline:
                    log.info("session_cancel_detected", session_id=session_id, turn=turn)
                    await self._repo.update_session(
                        session_id,
                        status=SessionStatus.CANCELLED,
                    )
                    # User-initiated stop is NOT an error — emit a clean terminal
                    # event so the UI ends quietly instead of flashing a red
                    # "Session cancelled by user" error box.
                    yield session_completed(
                        session_id, turn, total_usage.input_tokens,
                        total_usage.output_tokens, final_output="Stopped.",
                    )
                    return

                turn += 1
                context.turn_number = turn
                yield turn_started(session_id, turn)

                # Check if agent requested manual compaction on a prior turn
                force_compact = bool(working_memory.get("_compact_requested"))
                if force_compact:
                    working_memory.store("_compact_requested", False)
                    log.info("manual_compaction_triggered", session_id=session_id, turn=turn)

                # Load persistent data from DB for prompt visibility. Scoped by
                # tenant so two tenants running the same agent don't share (or
                # leak into each other's prompt) one memory.
                mem_key = scoped_agent_key(context.tenant_id, agent_def.id)
                knowledge_index = await self._load_knowledge_index(mem_key)
                skill_index = await self._load_skill_index(mem_key)
                objective = await self._load_objective(mem_key)
                plan = working_memory.get("_plan") or []

                # Build context (system prompt + messages, with compaction).
                # The system prompt is split into a byte-stable static part and
                # a per-turn volatile part (Notes/Plan): keeping the volatile
                # state OUT of the system message preserves provider prompt
                # caches (automatic prefix caching on Together/OpenAI-compatible
                # APIs, cache_control on Anthropic) across turns.
                system_prompt, volatile_context = build_system_prompt(
                    agent_def,
                    skills=skills,
                    working_memory_context=working_memory.get_context_for_llm(),
                    variables=variables or {},
                    registered_tools=self._tools.get_schemas(all_tool_names)
                    if all_tool_names
                    else None,
                    knowledge_index=knowledge_index,
                    skill_index=skill_index,
                    objective=objective,
                    plan=plan,
                    split_volatile=True,
                )
                messages = await self._context_manager.build_context(
                    session_id,
                    system_prompt,
                    agent_def.model,
                    llm=self._llm,
                    force_compact=force_compact,
                )
                if volatile_context:
                    # Trailing, NON-persisted message: rebuilt fresh each turn,
                    # never written to the repo, so history stays append-only
                    # and the cached prefix keeps hitting.
                    messages = messages + [
                        AgentMessage(
                            session_id=session_id,
                            role=MessageRole.USER,
                            content=(
                                "[Session state — auto-generated each turn; "
                                "not a user message]\n\n" + volatile_context
                            ),
                        )
                    ]

                # Run middleware: before_llm_call. A locally-configured rate
                # limit is a transient, self-inflicted block, so it takes the
                # turn-level recoverable path (session pauses at WAITING_INPUT
                # and can resume) rather than the engine-crash path that would
                # terminally FAIL the session.
                current_messages, current_tools = messages, tool_schemas
                try:
                    for mw in self._middleware:
                        current_messages, current_tools = await mw.before_llm_call(
                            current_messages, current_tools, context
                        )
                except RateLimitExceededError as e:
                    log.warning(
                        "turn_rate_limited", session_id=session_id, turn=turn,
                        retry_after_seconds=getattr(e, "retry_after_seconds", 0),
                    )
                    yield error_event(
                        session_id, "turn_error", str(e)[:1000], turn, recoverable=True
                    )
                    await self._repo.update_session(
                        session_id,
                        status=SessionStatus.WAITING_INPUT,
                        error=str(e)[:2000],
                    )
                    for mw in self._middleware:
                        await mw.on_error(e, context)
                    break

                # Call LLM with retry — text/thinking deltas are streamed
                # in real-time via an async queue, yielded as they arrive.
                _stream_q: asyncio.Queue[StreamEvent | None] = asyncio.Queue()
                _llm_error: list[Exception] = []  # Capture errors from the LLM task

                async def _on_stream(evt: StreamEvent):
                    await _stream_q.put(evt)

                async def _llm_task():
                    try:
                        return await self._call_llm_with_retry(
                            session_id=session_id,
                            messages=current_messages,
                            tools=current_tools,
                            agent_def=agent_def,
                            system_prompt=system_prompt,
                            turn=turn,
                            on_stream_event=_on_stream,
                        )
                    except Exception as e:
                        _llm_error.append(e)
                        raise
                    finally:
                        await _stream_q.put(None)  # Sentinel

                llm_future = asyncio.ensure_future(_llm_task())

                # Yield streaming events as they arrive. If this generator is
                # cancelled mid-stream (nuclear stop → the driving task is
                # cancelled), the finally aborts the in-flight LLM call instead
                # of leaving the HTTP request running detached — so STOP is
                # immediate, not "after the current turn finishes".
                try:
                    while True:
                        evt = await _stream_q.get()
                        if evt is None:
                            break
                        yield evt
                except (asyncio.CancelledError, GeneratorExit):
                    # Nuclear stop: abort the in-flight LLM call now rather than
                    # leaving the HTTP request running detached. Only on
                    # cancellation — the normal path falls through to read
                    # llm_future's result below.
                    if not llm_future.done():
                        llm_future.cancel()
                    raise

                # Re-raise any LLM error
                try:
                    full_text, tool_calls, usage, stop_reason, thinking_text = await llm_future
                except (LLMError, ContextOverflowError) as e:
                    # Turn-level recovery: don't fail the session, allow resume
                    log.error(
                        "turn_llm_failed", session_id=session_id, turn=turn, error=str(e)[:500]
                    )
                    yield error_event(
                        session_id, "turn_error", str(e)[:1000], turn, recoverable=True
                    )
                    await self._repo.update_session(
                        session_id,
                        status=SessionStatus.WAITING_INPUT,
                        error=str(e)[:2000],
                    )
                    for mw in self._middleware:
                        await mw.on_error(e, context)
                    break

                # Accumulate usage
                total_usage.input_tokens += usage.input_tokens
                total_usage.output_tokens += usage.output_tokens
                total_usage.total_tokens += usage.total_tokens
                total_usage.cache_creation_tokens += usage.cache_creation_tokens
                total_usage.cache_read_tokens += usage.cache_read_tokens

                # Run middleware: after_llm_call
                llm_response = LLMResponse(
                    content=full_text,
                    tool_calls=tool_calls,
                    stop_reason=stop_reason,
                    usage=usage,
                    model=agent_def.model,
                )
                for mw in self._middleware:
                    llm_response = await mw.after_llm_call(llm_response, context)

                # Persist assistant message — always preserve thinking when present.
                # Cases:
                #   thinking + text       → [thinking, text]
                #   thinking + no text    → [thinking]   (e.g. turn produced only tool_calls)
                #   no thinking + text    → text (string)
                #   no thinking + no text → "" (string)
                persist_content: str | list
                if thinking_text:
                    blocks: list = [{"type": "thinking", "thinking": thinking_text}]
                    if llm_response.content:
                        blocks.append({"type": "text", "text": llm_response.content})
                    persist_content = blocks
                else:
                    persist_content = llm_response.content

                assistant_msg = AgentMessage(
                    session_id=session_id,
                    role=MessageRole.ASSISTANT,
                    content=persist_content,
                    tool_calls=llm_response.tool_calls if llm_response.tool_calls else None,
                    token_count=usage.output_tokens,
                    model=agent_def.model,
                )
                # Reset thinking for next turn
                thinking_text = ""
                await self._repo.add_message(assistant_msg)

                # Truncation is NOT completion. A MAX_TOKENS stop with no
                # tool calls is a half-finished turn — ending here silently
                # abandons whatever the agent was mid-way through building.
                # Same bounded-nudge machinery as the premature-stop path.
                if (not llm_response.tool_calls
                        and stop_reason == StopReason.MAX_TOKENS):
                    _t_nudges = int(working_memory.get("_truncation_nudges") or 0)
                    if _t_nudges < self._MAX_CONTINUE_NUDGES:
                        working_memory.store("_truncation_nudges", _t_nudges + 1)
                        log.info(
                            "truncation_nudge",
                            session_id=session_id,
                            turn=turn,
                            nudge=_t_nudges + 1,
                        )
                        await self._repo.add_message(
                            AgentMessage(
                                session_id=session_id,
                                role=MessageRole.USER,
                                content=(
                                    "Your last output was CUT OFF by the token "
                                    "limit — it did not finish. Continue exactly "
                                    "where you stopped. If you were composing a "
                                    "large tool call, re-issue it SMALLER (split "
                                    "the payload across several calls)."
                                ),
                            )
                        )
                        yield turn_completed(session_id, turn, "truncated_continue")
                        for mw in self._middleware:
                            await mw.on_turn_complete(turn, context)
                        continue

                # No tool calls → standard ReAct treats the agent as done.
                # But for agents that must finish via an explicit terminal tool
                # (require_explicit_completion) or that haven't met min_turns, a
                # bare narration turn is a PREMATURE stop — the agent emitted
                # text without acting and without signaling completion. Nudge it
                # to continue instead of finalizing. Bounded by a consecutive-
                # nudge cap (reset whenever a real tool call happens) and by
                # max_turns, so this can never loop forever.
                if not llm_response.tool_calls:
                    _completed = bool(
                        working_memory.get("_session_complete")
                        or working_memory.get("_task_complete")
                    )
                    _below_min = turn < agent_def.min_turns
                    _needs_explicit = (
                        getattr(agent_def, "require_explicit_completion", False)
                        and not _completed
                    )
                    _nudges = int(working_memory.get("_continue_nudges") or 0)
                    if (_below_min or _needs_explicit) and _nudges < self._MAX_CONTINUE_NUDGES:
                        working_memory.store("_continue_nudges", _nudges + 1)
                        log.info(
                            "premature_stop_nudge",
                            session_id=session_id,
                            turn=turn,
                            below_min=_below_min,
                            needs_explicit=_needs_explicit,
                            nudge=_nudges + 1,
                        )
                        await self._repo.add_message(
                            AgentMessage(
                                session_id=session_id,
                                role=MessageRole.USER,
                                content=(
                                    "You ended a turn without calling a tool and "
                                    "without finishing. If your task is genuinely "
                                    "complete, call session_complete now. Otherwise "
                                    "take the next concrete action by calling a tool "
                                    "— do not just narrate."
                                ),
                            )
                        )
                        yield turn_completed(session_id, turn, "continue")
                        for mw in self._middleware:
                            await mw.on_turn_complete(turn, context)
                        continue
                    # Natural stop (standard ReAct termination).
                    yield turn_completed(session_id, turn, stop_reason.value)
                    for mw in self._middleware:
                        await mw.on_turn_complete(turn, context)
                    break

                # A real tool call means the agent is acting — reset the
                # consecutive premature-stop nudge budget so it refreshes for
                # any future stall later in the session.
                working_memory.store("_continue_nudges", 0)

                # Execute tool calls
                for tc in llm_response.tool_calls:
                    yield tool_executing(session_id, tc.tool_name, tc.id, turn)

                    # Middleware: before_tool_call
                    current_tc: ToolCall | None = tc
                    for mw in self._middleware:
                        if current_tc is None or tc.arguments_error:
                            break
                        current_tc = await mw.before_tool_call(current_tc, context)

                    if tc.arguments_error:
                        result = ToolResult(
                            tool_call_id=tc.id,
                            tool_name=tc.tool_name,
                            status=ToolStatus.ERROR,
                            error=tc.arguments_error + " This call was not executed. Other successful calls in this turn must not be repeated.",
                        )
                    elif current_tc is None:
                        result = ToolResult(
                            tool_call_id=tc.id,
                            tool_name=tc.tool_name,
                            status=ToolStatus.DENIED,
                            error="Blocked by middleware",
                        )
                    else:
                        result = await self._tools.execute(current_tc, context)

                    # Middleware: after_tool_call
                    for mw in self._middleware:
                        result = await mw.after_tool_call(tc, result, context)

                    # Store tool result in working memory
                    if result.status == ToolStatus.SUCCESS and result.output:
                        working_memory.store(f"tool:{tc.tool_name}:{tc.id[:8]}", result.output)

                    # Persist tool execution
                    await self._repo.log_tool_execution(
                        session_id=session_id,
                        message_id=assistant_msg.id,
                        tool_call_id=tc.id,
                        tool_name=tc.tool_name,
                        input_data=tc.arguments,
                        output_data=result.output,
                        status=result.status.value,
                        duration_ms=result.duration_ms,
                        error=result.error,
                    )

                    # Persist tool result as a message
                    output_content = result.output
                    if isinstance(output_content, (dict, list)):
                        output_content = json.dumps(output_content, default=str)
                    elif output_content is None:
                        output_content = result.error or "No output"
                    # The model reads ONLY this content — never the ToolResult
                    # status. So a failed call must be UNMISTAKABLE in the text,
                    # or the model can mistake it for success and repeat the same
                    # bad arguments. Lead with the failure and the reason.
                    if result.status in (ToolStatus.ERROR, ToolStatus.TIMEOUT):
                        reason = result.error or output_content
                        output_content = (
                            f"TOOL CALL FAILED ({tc.tool_name}): {reason}\n"
                            "This call did NOT take effect. Read the reason, change "
                            "your arguments accordingly, and try a DIFFERENT call — "
                            "do not repeat the same arguments.\n"
                            f"Raw result: {output_content}"
                        )
                    tool_msg = AgentMessage(
                        session_id=session_id,
                        role=MessageRole.TOOL_RESULT,
                        content=str(output_content),
                        tool_call_id=tc.id,
                        tool_name=tc.tool_name,
                    )
                    await self._repo.add_message(tool_msg)

                    output_preview = str(output_content) if output_content else ""
                    # Extract card_data for frontend tool card rendering.
                    # Tools return ToolOutput with card_data dict containing
                    # structured results (query, code, results, etc.)
                    _card_data = None
                    _result_dict = None
                    if isinstance(result.output, dict):
                        _card_data = result.output.get("card_data")
                        _result_dict = result.output
                    yield tool_result_event(
                        session_id,
                        tc.tool_name,
                        tc.id,
                        result.status.value,
                        result.duration_ms,
                        output_preview,
                        turn,
                        card_data=_card_data,
                        result=_result_dict,
                    )

                yield turn_completed(session_id, turn, "tool_use")
                for mw in self._middleware:
                    await mw.on_turn_complete(turn, context)

                # ── Working memory stop signals ────────────────────────
                # Tools like task_complete / session_complete set flags in
                # working memory to signal "I'm done — stop the loop."
                # Without this, the engine runs another turn and the LLM
                # re-generates its report text, causing duplication.
                if working_memory.get("_task_complete") or working_memory.get("_session_complete"):
                    log.info(
                        "stop_signal_detected",
                        session_id=session_id,
                        turn=turn,
                        task_complete=bool(working_memory.get("_task_complete")),
                        session_complete=bool(working_memory.get("_session_complete")),
                    )
                    break  # Exit loop → finalize session

                # ── External termination callback ──────────────────────
                # Callers can provide a should_continue(context) callback in
                # metadata for custom termination logic (cost limits, time
                # budgets, external signals, etc.)
                _should_continue = (metadata or {}).get("should_continue")
                if callable(_should_continue):
                    try:
                        if not _should_continue(context):
                            log.info("should_continue_false", session_id=session_id, turn=turn)
                            break
                    except Exception as _sc_err:
                        log.warning(
                            "should_continue_error", session_id=session_id, error=str(_sc_err)[:200]
                        )

                # ── Context health monitoring ──────────────────────────
                health_config = agent_def.metadata.get("context_health")
                if isinstance(health_config, ContextHealthConfig):
                    messages_count = len(await self._repo.get_messages(session_id))
                    health = assess_health(
                        total_usage.input_tokens,
                        messages_count,
                        health_config,
                        already_warned=context.metadata.get("_health_warned", False),
                    )

                    if health.should_hard_stop:
                        # Force one final LLM call with no tools
                        log.warning(
                            "context_hard_stop", session_id=session_id, score=health.health_score
                        )
                        stop_msg = AgentMessage(
                            session_id=session_id,
                            role=MessageRole.USER,
                            content=CONTEXT_HARD_STOP_MESSAGE,
                        )
                        await self._repo.add_message(stop_msg)
                        break  # Exit loop → finalize

                    if health.should_compact and not context.metadata.get("_compacted", False):
                        # Trigger LLM-based summarization of old messages
                        context.metadata["_compacted"] = True
                        log.info(
                            "context_compaction_triggered",
                            session_id=session_id,
                            score=health.health_score,
                        )
                        system_prompt = build_system_prompt(
                            agent_def,
                            skills=skills,
                            working_memory_context=working_memory.get_context_for_llm(),
                            variables=variables or {},
                            objective=objective,
                            plan=working_memory.get("_plan") or [],
                        )
                        await self._context_manager.build_context(
                            session_id,
                            system_prompt,
                            agent_def.model,
                            llm=self._llm,
                        )

                    if health.should_warn_agent:
                        context.metadata["_health_warned"] = True
                        warn_msg = AgentMessage(
                            session_id=session_id,
                            role=MessageRole.USER,
                            content=CONTEXT_WARNING_MESSAGE,
                        )
                        await self._repo.add_message(warn_msg)

            # Finalize session — whether we broke out naturally or hit max_turns
            # Always treat as COMPLETED. The agent did useful work either way.
            # Use _task_report (set by session_complete) or last LLM response as fallback.
            final_output = working_memory.get("_task_report") or llm_response.content or ""
            if not final_output:
                # Fall back to tool results
                if not final_output:
                    tool_outputs = []
                    # `msgs` was undefined here (NameError crashed finalize and
                    # surfaced as error="name 'msgs' is not defined"). Fetch the
                    # session's messages from the repo for the fallback scan.
                    _fallback_msgs = await self._repo.get_messages(session_id)
                    for m in reversed(_fallback_msgs):
                        if m.role == MessageRole.TOOL_RESULT and m.content:
                            c = m.content if isinstance(m.content, str) else str(m.content)
                            if c.strip() and len(c.strip()) > 20:
                                tool_outputs.append(c.strip()[:1000])
                                if len(tool_outputs) >= 3:
                                    break
                    if tool_outputs:
                        tool_outputs.reverse()
                        final_output = "\n\n".join(tool_outputs)

            at_max = turn >= agent_def.max_turns
            if at_max:
                log.info(
                    "max_turns_reached",
                    session_id=session_id,
                    turns=turn,
                    max_turns=agent_def.max_turns,
                )

            await self._repo.update_session(
                session_id,
                status=SessionStatus.COMPLETED,
                completed_at=datetime.now(UTC),
                turn_count=turn,
                total_input_tokens=total_usage.input_tokens,
                total_output_tokens=total_usage.output_tokens,
            )

            yield session_completed(
                session_id,
                turn,
                total_usage.input_tokens,
                total_usage.output_tokens,
                final_output,
            )

        except Exception as e:
            log.exception("engine_error", session_id=session_id)
            await self._repo.update_session(
                session_id,
                status=SessionStatus.FAILED,
                error=str(e)[:2000],
            )
            yield error_event(session_id, "engine_error", str(e)[:1000], turn)
            for mw in self._middleware:
                await mw.on_error(e, context)

    # ══════════════════════════════════════════════════════════════════
    # Internal helpers
    # ══════════════════════════════════════════════════════════════════

    async def _call_llm_with_retry(
        self,
        session_id: str,
        messages: list[AgentMessage],
        tools: list[ToolSchema],
        agent_def: AgentDefinition,
        system_prompt: str,
        turn: int,
        on_stream_event: Any = None,
    ) -> tuple[str, list[ToolCall], LLMUsage, StopReason, str]:
        """
        Call LLM with automatic retry on transient errors + provider fallback.

        Retries on:
        - Retryable LLMError (timeout, connection): exponential backoff
        - LLMRateLimitError: wait retry_after_seconds
        - ContextOverflowError: compact context, retry once

        Fallback chain: if all retries on primary model fail and
        fallback_models is configured, tries each fallback in order.

        Returns (full_text, tool_calls, usage, stop_reason, thinking_text).
        Raises the original error if all retries AND fallbacks exhausted.
        """
        # Build model chain: primary + fallbacks
        models_to_try = [agent_def.model] + list(agent_def.fallback_models or [])
        last_error: Exception | None = None

        for model_idx, current_model in enumerate(models_to_try):
            if model_idx > 0:
                log.warning(
                    "llm_fallback",
                    session_id=session_id,
                    failed_model=models_to_try[model_idx - 1],
                    fallback_model=current_model,
                    attempt=model_idx,
                )

            try:
                return await self._call_single_model(
                    session_id,
                    messages,
                    tools,
                    agent_def,
                    system_prompt,
                    turn,
                    current_model,
                    on_stream_event=on_stream_event,
                )
            except LLMError as e:
                last_error = e
                if model_idx < len(models_to_try) - 1:
                    continue  # Try next fallback
                raise  # No more fallbacks

        # Should not reach here
        raise last_error or LLMError("All models exhausted", provider="", model=agent_def.model)

    async def _call_single_model(
        self,
        session_id: str,
        messages: list[AgentMessage],
        tools: list[ToolSchema],
        agent_def: AgentDefinition,
        system_prompt: str,
        turn: int,
        model: str,
        on_stream_event: Any = None,
    ) -> tuple[str, list[ToolCall], LLMUsage, StopReason]:
        """Call a single model with retry logic.

        Args:
            on_stream_event: Optional async callback(StreamEvent) invoked for
                each text_delta / thinking chunk during streaming. This enables
                real-time token streaming to SSE clients while the engine still
                accumulates the full response for persistence.
        """
        max_retries = agent_def.max_llm_retries
        context_compacted = False

        timeout = agent_def.llm_timeout_seconds

        for attempt in range(max_retries + 1):
            try:
                response = self._llm.stream_with_tools(
                    messages=messages,
                    tools=tools,
                    model=model,
                    system_prompt=system_prompt,
                    temperature=agent_def.temperature,
                    max_tokens=agent_def.max_tokens_per_turn,
                )

                # Collect streamed response with timeout
                full_text = ""
                thinking_text = ""
                tool_calls: list[ToolCall] = []
                usage = LLMUsage()
                stop_reason = StopReason.END_TURN

                async def _collect_stream():
                    nonlocal full_text, thinking_text, usage, stop_reason
                    async for chunk in response:
                        if chunk.type == "text_delta" and chunk.text:
                            full_text += chunk.text
                            # Emit real-time text streaming event
                            if on_stream_event:
                                await on_stream_event(text_delta(session_id, chunk.text, turn))
                        elif chunk.type == "thinking_delta" and chunk.text:
                            thinking_text += chunk.text
                            # Emit real-time thinking streaming event
                            if on_stream_event:
                                await on_stream_event(thinking_delta(session_id, chunk.text, turn))
                        elif chunk.type == "tool_call_end" and chunk.tool_call:
                            tool_calls.append(chunk.tool_call)
                            # Emit tool_call event for SSE buffering
                            if on_stream_event and chunk.tool_call:
                                await on_stream_event(
                                    tool_call_event(
                                        session_id,
                                        chunk.tool_call.tool_name,
                                        chunk.tool_call.id,
                                        chunk.tool_call.arguments,
                                        turn,
                                    )
                                )
                        elif chunk.type == "usage":
                            if chunk.usage:
                                usage = chunk.usage
                            if chunk.stop_reason:
                                stop_reason = chunk.stop_reason
                        elif chunk.type == "complete":
                            if chunk.text:
                                full_text = chunk.text
                            if chunk.usage:
                                usage = chunk.usage
                            if chunk.stop_reason:
                                stop_reason = chunk.stop_reason

                try:
                    await asyncio.wait_for(_collect_stream(), timeout=timeout)
                except TimeoutError:
                    raise LLMError(
                        f"LLM call timed out after {timeout}s",
                        provider="",
                        model=agent_def.model,
                        retryable=True,
                    )

                # Reasoning-only turn rescue: some models (Qwen3.7 on Together,
                # via an unclosed <think> tag or reasoning-channel streaming)
                # deliver their entire answer through the thinking channel. A
                # final turn with thinking but no text and no tool calls would
                # store an empty visible message — promote the thinking to text
                # instead (mirrors the non-streaming path's fallback).
                if not full_text.strip() and not tool_calls and thinking_text.strip():
                    log.warning(
                        "reasoning_only_turn_promoted",
                        model=agent_def.model,
                        thinking_chars=len(thinking_text),
                    )
                    full_text = thinking_text.strip()
                    thinking_text = ""

                return full_text, tool_calls, usage, stop_reason, thinking_text

            except LLMRateLimitError as e:
                if attempt >= max_retries:
                    raise
                wait = e.retry_after_seconds or min(2**attempt, 30)
                log.warning(
                    "llm_rate_limit_retry",
                    session_id=session_id,
                    attempt=attempt + 1,
                    max_retries=max_retries,
                    wait_seconds=wait,
                )
                await asyncio.sleep(wait)

            except ContextOverflowError:
                if context_compacted or attempt >= max_retries:
                    raise
                log.warning("context_overflow_retry", session_id=session_id, attempt=attempt + 1)
                # Compact context and retry once
                messages = await self._context_manager.build_context(
                    session_id,
                    system_prompt,
                    agent_def.model,
                    llm=self._llm,
                )
                context_compacted = True

            except LLMError as e:
                if not e.retryable or attempt >= max_retries:
                    raise
                wait = min(2**attempt, 30)
                log.warning(
                    "llm_error_retry",
                    session_id=session_id,
                    attempt=attempt + 1,
                    max_retries=max_retries,
                    wait_seconds=wait,
                    error=str(e)[:200],
                )
                await asyncio.sleep(wait)

        # Should not reach here, but just in case
        raise LLMError("All retry attempts exhausted", provider="", model=model)

    async def _load_knowledge_index(self, agent_id: str) -> list[dict]:
        """Load the knowledge document listing for prompt injection."""
        try:
            all_memories = await self._repo.list_memories(agent_id, memory_type="document")
            result = []
            for m in all_memories:
                if not m.get("key", "").startswith("doc:"):
                    continue
                name = m["key"].removeprefix("doc:")
                value = str(m.get("value", ""))
                # First non-empty line as summary
                summary = ""
                for line in value.splitlines():
                    stripped = line.strip().lstrip("#").strip()
                    if stripped:
                        summary = stripped[:80]
                        break
                result.append({"name": name, "description": summary})
            return result
        except Exception:
            return []

    async def _load_skill_index(self, agent_id: str) -> list[dict]:
        """Load the skill listing for prompt injection."""
        try:
            all_memories = await self._repo.list_memories(agent_id, memory_type="skill")
            result = []
            for m in all_memories:
                if m.get("key", "").startswith("skill:"):
                    val = m.get("value", {})
                    desc = val.get("description", "") if isinstance(val, dict) else ""
                    result.append(
                        {
                            "name": m["key"].removeprefix("skill:"),
                            "description": desc,
                        }
                    )
            return result
        except Exception:
            return []

    async def _load_objective(self, agent_id: str) -> str | None:
        """Load the agent's objective (full content) for prompt injection."""
        try:
            value = await self._repo.get_memory(agent_id, "objective")
            if value is None:
                return None
            return value if isinstance(value, str) else str(value)
        except Exception:
            return None

    async def _ensure_session(
        self,
        session_id: str,
        agent_def: AgentDefinition,
        metadata: dict | None,
    ) -> AgentSession:
        """Load existing session or create new one."""
        session = await self._repo.get_session(session_id)
        if session:
            return session

        safe_metadata = _json_safe(metadata) if metadata else {}

        session = AgentSession(
            id=session_id,
            agent_id=agent_def.id,
            config=agent_def.model_dump(exclude={"sub_agents"}),
            metadata=safe_metadata,
        )
        return await self._repo.create_session(session)
