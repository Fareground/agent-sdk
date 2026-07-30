"""
Fareground Agent Framework — Tool Registry

Central tool store. Register tools by function, config, or RegisteredTool.
Resolves tool schemas for LLM calls, executes tools with timeout/retry/audit.

Inspired by Sentinel tool_registry.py + CrewAI @tool + Deep Agents StructuredTool.
"""

import asyncio
import time
from typing import Any

import structlog

from fg_agents.core.errors import (
    ToolExecutionError,
    ToolTimeoutError,
)
from fg_agents.core.types import (
    ExecutionContext,
    RegisteredTool,
    ToolCall,
    ToolResult,
    ToolSchema,
    ToolStatus,
    ToolType,
)

log = structlog.get_logger("fg_agents.tools")


class ToolRegistry:
    """
    Central tool store with execution capabilities.

    Tools can be registered via:
    1. @tool decorated functions (have .tool_definition attribute)
    2. RegisteredTool instances directly
    3. Dictionaries with tool config

    Execution handles timeout, retry, and error wrapping.
    """

    def __init__(self, vault_resolver=None):
        self._tools: dict[str, RegisteredTool] = {}
        self._vault_resolver = vault_resolver

    @property
    def tools(self) -> dict[str, RegisteredTool]:
        return dict(self._tools)

    def register(self, tool: RegisteredTool) -> None:
        """Register a tool definition."""
        if tool.name in self._tools:
            log.warning("tool_overwrite", name=tool.name)
        self._tools[tool.name] = tool
        log.debug("tool_registered", name=tool.name, type=tool.tool_type.value)

    def register_function(self, fn) -> None:
        """Register a @tool-decorated function."""
        if hasattr(fn, "tool_definition"):
            self.register(fn.tool_definition)
        else:
            raise ValueError(
                f"Function {fn.__name__} is not decorated with @tool. "
                "Use the @tool decorator or register a RegisteredTool directly."
            )

    def register_many(self, tools: list) -> None:
        """Register multiple tools (functions or RegisteredTool instances)."""
        for t in tools:
            if isinstance(t, RegisteredTool):
                self.register(t)
            elif callable(t) and hasattr(t, "tool_definition"):
                self.register_function(t)
            else:
                raise ValueError(f"Cannot register {t} — must be RegisteredTool or @tool function")

    def unregister(self, name: str) -> None:
        self._tools.pop(name, None)

    def get(self, name: str) -> RegisteredTool | None:
        return self._tools.get(name)

    def has(self, name: str) -> bool:
        return name in self._tools

    def get_schemas(self, tool_names: list[str] | None = None) -> list[ToolSchema]:
        """
        Get LLM-facing tool schemas.
        If tool_names is provided, only return schemas for those tools.
        """
        if tool_names is not None:
            tools = [self._tools[n] for n in tool_names if n in self._tools]
        else:
            tools = list(self._tools.values())
        return [t.to_tool_schema() for t in tools]

    def get_tools_for_agent(self, allowed_tools: list[str]) -> list[ToolSchema]:
        """Get schemas for tools an agent is allowed to use."""
        return self.get_schemas(allowed_tools)

    async def execute(
        self,
        tool_call: ToolCall,
        context: ExecutionContext,
    ) -> ToolResult:
        """
        Execute a tool call with timeout, retry, and error handling.

        Returns ToolResult with status, output, duration, and any error info.
        """
        tool_def = self._tools.get(tool_call.tool_name)
        if not tool_def:
            return ToolResult(
                tool_call_id=tool_call.id,
                tool_name=tool_call.tool_name,
                status=ToolStatus.ERROR,
                error=f"Tool '{tool_call.tool_name}' not found in registry",
            )

        # For non-FUNCTION tools without a handler, dispatch to type-based handlers
        if not tool_def.handler and tool_def.tool_type != ToolType.FUNCTION:
            from fg_agents.tools.handlers import dispatch_tool

            return await dispatch_tool(
                tool_def,
                tool_call,
                context,
                vault_resolver=self._vault_resolver,
            )

        if not tool_def.handler:
            return ToolResult(
                tool_call_id=tool_call.id,
                tool_name=tool_call.tool_name,
                status=ToolStatus.ERROR,
                error=f"Tool '{tool_call.tool_name}' has no handler",
            )

        attempts = 1 + tool_def.retry_max
        last_error: Exception | None = None

        for attempt in range(attempts):
            start = time.time()
            try:
                result = await asyncio.wait_for(
                    self._invoke_handler(tool_def, tool_call, context),
                    timeout=tool_def.timeout_seconds,
                )
                duration = (time.time() - start) * 1000

                output = self._normalize_output(result)

                # A tool can signal failure WITHOUT raising — by returning an
                # error-shaped payload. Historically the engine marked ANY
                # return SUCCESS, so a failed validation reached the model as a
                # green "success" whose payload merely happened to contain an
                # error. Models (smaller ones especially) then repeat the same
                # bad call instead of correcting, because nothing framed it as a
                # failure. Detect the two conventions in use — {"error": msg}
                # and {"status": "error", "message": msg} — and surface them as
                # a real ERROR so the reason reaches the model as a failure.
                err_msg = self._detect_returned_error(output)
                if err_msg is not None:
                    return ToolResult(
                        tool_call_id=tool_call.id,
                        tool_name=tool_call.tool_name,
                        output=output,
                        status=ToolStatus.ERROR,
                        error=err_msg,
                        duration_ms=duration,
                    )

                return ToolResult(
                    tool_call_id=tool_call.id,
                    tool_name=tool_call.tool_name,
                    output=output,
                    status=ToolStatus.SUCCESS,
                    duration_ms=duration,
                )

            except TimeoutError:
                duration = (time.time() - start) * 1000
                last_error = ToolTimeoutError(
                    f"Tool '{tool_call.tool_name}' timed out after {tool_def.timeout_seconds}s",
                    tool_name=tool_call.tool_name,
                    tool_call_id=tool_call.id,
                )
                log.warning(
                    "tool_timeout",
                    tool=tool_call.tool_name,
                    attempt=attempt + 1,
                    timeout=tool_def.timeout_seconds,
                )

            except Exception as e:
                duration = (time.time() - start) * 1000
                last_error = e
                log.warning(
                    "tool_error", tool=tool_call.tool_name, attempt=attempt + 1, error=str(e)[:200]
                )

            if attempt < attempts - 1:
                await asyncio.sleep(min(2**attempt, 10))

        return ToolResult(
            tool_call_id=tool_call.id,
            tool_name=tool_call.tool_name,
            status=ToolStatus.TIMEOUT
            if isinstance(last_error, ToolTimeoutError)
            else ToolStatus.ERROR,
            duration_ms=(time.time() - start) * 1000,
            error=str(last_error)[:2000] if last_error else "Unknown error",
        )

    async def _invoke_handler(
        self,
        tool_def: RegisteredTool,
        tool_call: ToolCall,
        context: ExecutionContext,
    ) -> Any:
        """Invoke the tool handler, injecting context if the function accepts it."""
        import inspect

        handler = tool_def.handler
        sig = inspect.signature(handler)
        params = sig.parameters
        accepts_var_kw = any(
            p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()
        )
        injected = {"ctx", "context"}

        # Validate the model's arguments against the handler signature at this
        # boundary, so malformed model output produces a clear, catchable
        # validation error instead of an opaque TypeError deep in the handler.
        raw_args = tool_call.arguments if isinstance(tool_call.arguments, dict) else {}
        if not accepts_var_kw:
            # Drop keys the handler does not accept (a stray key the model
            # emitted must not collide with — or be silently forwarded to — an
            # unrelated parameter).
            unexpected = [k for k in raw_args if k not in params or k in injected]
            kwargs = {k: v for k, v in raw_args.items() if k in params and k not in injected}
            if unexpected:
                log.warning(
                    "tool_args_dropped_unknown",
                    tool=tool_call.tool_name,
                    dropped=unexpected,
                )
        else:
            kwargs = {k: v for k, v in raw_args.items() if k not in injected}

        for param_name, param in params.items():
            if param_name in injected:
                hints = {}
                try:
                    hints = inspect.get_annotations(handler)
                except Exception:
                    pass
                if hints.get(param_name) is ExecutionContext or param_name == "ctx":
                    kwargs[param_name] = context

        missing = [
            name
            for name, p in params.items()
            if name not in injected
            and name not in kwargs
            and p.default is inspect.Parameter.empty
            and p.kind
            in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
        ]
        if missing:
            raise ToolExecutionError(
                f"tool {tool_call.tool_name!r} called without required "
                f"argument(s): {missing}"
            )

        return await handler(**kwargs)

    @staticmethod
    def _detect_returned_error(output: Any) -> str | None:
        """Return an error message if a tool signalled failure via its RETURN
        value (no exception raised), else None.

        Recognizes the two conventions in the codebase:
          - ``{"error": <truthy str>}``            (operator-style tools)
          - ``{"status": "error", "message": ...}`` (framework builtins)

        Only the TOP-LEVEL shape counts — an aggregate success that merely
        contains nested per-item errors (e.g. spawn_parallel's results list)
        is left as SUCCESS. Requires a truthy message so ``{"error": ""}`` or
        ``{"error": None}`` on success is not misclassified.
        """
        if not isinstance(output, dict):
            return None
        status = output.get("status")
        if isinstance(status, str) and status.strip().lower() == "error":
            msg = output.get("message") or output.get("error")
            return str(msg) if msg else "tool reported status=error"
        err = output.get("error")
        if isinstance(err, str) and err.strip():
            return err
        return None

    @staticmethod
    def _normalize_output(result: Any) -> Any:
        """
        Normalize tool output to a JSON-serializable value.

        Validates that the output is actually serializable to prevent
        surprise errors when the framework tries to persist or stream it.
        """
        import json

        if result is None:
            return {"result": "completed"}
        if isinstance(result, str):
            return result
        if isinstance(result, (int, float, bool)):
            return result
        if isinstance(result, (dict, list)):
            # Validate JSON-serializable
            try:
                json.dumps(result, default=str)
                return result
            except (TypeError, ValueError):
                return json.loads(json.dumps(result, default=str))
        try:
            if hasattr(result, "model_dump"):
                return result.model_dump()
            if hasattr(result, "__dict__"):
                return {k: v for k, v in result.__dict__.items() if not k.startswith("_")}
            return str(result)
        except Exception:
            return str(result)
