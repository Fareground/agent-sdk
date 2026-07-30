"""
Fareground Agent Framework — LoopGuard Middleware

Detects and breaks infinite loops in the agent ReAct loop.
Three detection mechanisms:

1. Identical tool calls (same tool + args hash) repeated N times
2. Tool-only turns without substantive text output
3. Management-only turns (agent shuffling notes without real work)

On soft detection: injects a system message nudging the agent to wrap up.
On hard detection: sets _session_complete in working memory → engine stops.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import structlog

from fg_agents.core.types import (
    AgentMessage,
    ExecutionContext,
    LLMResponse,
    MessageRole,
    ToolCall,
    ToolResult,
    ToolSchema,
    ToolStatus,
)
from fg_agents.middleware.base import BaseMiddleware

from ._bounded import BoundedLRU

log = structlog.get_logger("fg_agents.loop_guard")


@dataclass
class LoopGuardConfig:
    """Configuration for loop detection thresholds."""

    # Identical call detection. Two regimes, split by whether the calls are
    # failing: repeating a call that keeps ERRORING is a death spiral (the
    # model isn't reading the error), while repeating a call that SUCCEEDS is
    # usually legitimate polling (order status, sub-agent report, inbox
    # check) and deserves a much longer leash plus a warning before any stop.
    max_identical_calls: int = 3            # identical + erroring → loop
    max_identical_success_calls: int = 10   # identical + succeeding → polling budget

    # Tool-only stagnation: N consecutive turns with tools but no text → intervene
    max_toolonly_turns: int = 4

    # Management-only: N turns with ONLY these tools (no substantive work)
    max_management_only_turns: int = 5
    management_tools: frozenset[str] = frozenset(
        {
            "manage_plan",
            "manage_notes",
            "manage_knowledge",
            "manage_objective",
            "manage_context",
            "manage_skill",
        }
    )

    # Give one soft warning before hard stopping
    soft_warning_before_hard_stop: bool = True


class LoopGuardMiddleware(BaseMiddleware):
    """
    Detects and breaks infinite loops in the agent ReAct loop.

    Usage::

        from fg_agents.middleware.loop_guard import LoopGuardMiddleware

        engine = AgentEngine(
            llm=llm, tool_registry=registry, repository=repo,
            middleware=[LoopGuardMiddleware()],
        )

    With custom config::

        config = LoopGuardConfig(max_identical_calls=5, max_toolonly_turns=12)
        middleware = [LoopGuardMiddleware(config=config)]
    """

    def __init__(self, config: LoopGuardConfig | None = None):
        self._config = config or LoopGuardConfig()
        # Per-session tracking (keyed by session_id). LRU-bounded so a
        # long-lived server that has seen many sessions cannot grow these
        # without limit; evicting a long-idle session's best-effort loop
        # counters is harmless (they just reset if it is ever touched again).
        # Per call: (hash of tool+args, whether the call errored)
        self._call_hashes: BoundedLRU[str, list[tuple[str, bool]]] = BoundedLRU()
        self._toolonly_turns: BoundedLRU[str, int] = BoundedLRU()
        self._mgmt_only_turns: BoundedLRU[str, int] = BoundedLRU()
        self._warned: BoundedLRU[str, bool] = BoundedLRU()

    @property
    def name(self) -> str:
        return "LoopGuard"

    # ── Helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _hash_call(tc: ToolCall) -> str:
        """Stable hash of tool_name + arguments for dedup detection."""
        raw = f"{tc.tool_name}:{json.dumps(tc.arguments, sort_keys=True, default=str)}"
        return hashlib.md5(raw.encode()).hexdigest()[:16]

    def _soft_warn(self, context: ExecutionContext, reason: str) -> None:
        """Store a warning in working memory. Injected on next LLM call."""
        log.info("loop_guard_soft_warning", session_id=context.session_id, reason=reason)
        if context.working_memory:
            context.working_memory.store(
                "_loop_guard_warning",
                f"[LOOP GUARD] {reason}. "
                "You MUST wrap up NOW. Deliver your final response and call "
                "session_complete(summary='...'). Do NOT call more tools.",
            )

    def _force_stop(self, context: ExecutionContext, reason: str) -> None:
        """Force session termination via working memory flag."""
        log.warning("loop_guard_force_stop", session_id=context.session_id, reason=reason)
        if context.working_memory:
            context.working_memory.store("_session_complete", True)
            context.working_memory.store(
                "_session_summary",
                f"Session terminated by LoopGuard: {reason}",
            )

    def _intervene(self, context: ExecutionContext, reason: str) -> None:
        """Soft warn first, then hard stop on second violation."""
        sid = context.session_id
        if self._config.soft_warning_before_hard_stop and not self._warned.get(sid):
            self._soft_warn(context, reason)
            self._warned[sid] = True
        else:
            self._force_stop(context, reason)

    def _cleanup(self, sid: str) -> None:
        """Remove tracking state for a completed session."""
        self._call_hashes.pop(sid, None)
        self._toolonly_turns.pop(sid, None)
        self._mgmt_only_turns.pop(sid, None)
        self._warned.pop(sid, None)

    # ── Middleware hooks ────────────────────────────────────────────

    async def before_llm_call(
        self,
        messages: list[AgentMessage],
        tools: list[ToolSchema],
        context: ExecutionContext,
    ) -> tuple[list[AgentMessage], list[ToolSchema]]:
        """Inject loop guard warning into messages if one is pending."""
        wm = context.working_memory
        if wm:
            warning = wm.get("_loop_guard_warning")
            if warning:
                wm.remove("_loop_guard_warning")
                warning_msg = AgentMessage(
                    session_id=context.session_id,
                    role=MessageRole.USER,
                    content=f"[SYSTEM] {warning}",
                )
                messages = list(messages) + [warning_msg]
        return messages, tools

    async def after_llm_call(
        self,
        response: LLMResponse,
        context: ExecutionContext,
    ) -> LLMResponse:
        """Track whether LLM produced text and management-only tool calls."""
        wm = context.working_memory
        sid = context.session_id
        if not wm:
            return response

        has_text = bool(response.content and response.content.strip())

        # Track management-only turns (no text + only management tools)
        if response.tool_calls:
            all_mgmt = all(
                tc.tool_name in self._config.management_tools for tc in response.tool_calls
            )
            if all_mgmt and not has_text:
                self._mgmt_only_turns[sid] = self._mgmt_only_turns.get(sid, 0) + 1
            else:
                self._mgmt_only_turns[sid] = 0

            if self._mgmt_only_turns.get(sid, 0) >= self._config.max_management_only_turns:
                self._intervene(
                    context,
                    f"Only management tools for {self._config.max_management_only_turns} "
                    f"consecutive turns with no substantive work",
                )

        # Store text flag for on_turn_complete stagnation check
        wm.store("_loop_guard_last_had_text", has_text)

        return response

    async def after_tool_call(
        self,
        tool_call: ToolCall,
        result: ToolResult,
        context: ExecutionContext,
    ) -> ToolResult:
        """Track tool call signatures for identical-call detection.

        A repeated identical call that keeps ERRORING means the model is not
        reading the error — stop it fast. A repeated identical call that
        SUCCEEDS is usually deliberate polling (order fills, sub-agent
        reports), so it gets a larger budget and a soft "wrap up" warning
        before any hard stop.
        """
        sid = context.session_id
        if sid not in self._call_hashes:
            self._call_hashes[sid] = []

        errored = result.status != ToolStatus.SUCCESS or result.error is not None
        self._call_hashes[sid].append((self._hash_call(tool_call), errored))

        history = self._call_hashes[sid]

        def _identical_streak(n: int) -> list[bool] | None:
            """Error flags of the last n calls if they share one hash, else None."""
            recent = history[-n:]
            if len(recent) == n and len({h for h, _ in recent}) == 1:
                return [e for _, e in recent]
            return None

        n_err = self._config.max_identical_calls
        n_ok = self._config.max_identical_success_calls

        err_streak = _identical_streak(n_err)
        if err_streak is not None and all(err_streak):
            log.warning(
                "loop_guard_identical_calls",
                session_id=sid, tool=tool_call.tool_name, count=n_err, erroring=True,
            )
            history.clear()
            self._intervene(
                context,
                f"Identical tool call detected: {tool_call.tool_name} "
                f"called {n_err} times with same arguments, erroring every time — "
                f"read the error and change the arguments or the approach",
            )
            return result

        ok_streak = _identical_streak(n_ok)
        if ok_streak is not None:
            log.warning(
                "loop_guard_identical_calls",
                session_id=sid, tool=tool_call.tool_name, count=n_ok, erroring=False,
            )
            history.clear()
            self._intervene(
                context,
                f"{tool_call.tool_name} called {n_ok} times with the same arguments — "
                f"if you are waiting on something, set a watch/timer and end the "
                f"session instead of polling",
            )

        return result

    async def on_turn_complete(
        self,
        turn_number: int,
        context: ExecutionContext,
    ) -> None:
        """Track tool-only turns for stagnation detection."""
        sid = context.session_id
        wm = context.working_memory
        if not wm:
            return

        last_had_text = wm.get("_loop_guard_last_had_text")

        # Only count as tool-only if the flag was explicitly False
        # (None means text-only turn which is fine)
        if last_had_text is False:
            self._toolonly_turns[sid] = self._toolonly_turns.get(sid, 0) + 1
        else:
            self._toolonly_turns[sid] = 0

        if self._toolonly_turns.get(sid, 0) >= self._config.max_toolonly_turns:
            self._intervene(
                context,
                f"No text output for {self._config.max_toolonly_turns} "
                f"consecutive turns — agent is looping without communicating",
            )
