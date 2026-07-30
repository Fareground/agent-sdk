"""Tests for the tool system."""
import pytest

from fg_agents import ExecutionContext, ToolRegistry, tool
from fg_agents.core.types import ToolCall, ToolStatus


@tool(description="Add two numbers")
def add(a: float, b: float) -> float:
    return a + b


@tool(description="Greet someone")
async def greet(name: str) -> str:
    return f"Hello, {name}!"


@tool(description="Uses context")
def with_context(ctx: ExecutionContext, key: str) -> str:
    ctx.working_memory.store(key, "stored")
    return f"stored {key}"


def test_tool_decorator_creates_definition():
    assert hasattr(add, "tool_definition")
    assert add.tool_definition.name == "add"
    assert add.tool_definition.description == "Add two numbers"


def test_tool_json_schema():
    schema = add.tool_definition.to_tool_schema()
    assert schema.name == "add"
    props = schema.parameters.get("properties", {})
    assert "a" in props
    assert "b" in props
    assert props["a"]["type"] == "number"


def test_tool_context_param_excluded_from_schema():
    schema = with_context.tool_definition.to_tool_schema()
    props = schema.parameters.get("properties", {})
    assert "ctx" not in props
    assert "key" in props


def test_registry_register_and_get():
    reg = ToolRegistry()
    reg.register_function(add)
    assert reg.has("add")
    assert reg.get("add") is not None


def test_registry_register_many():
    reg = ToolRegistry()
    reg.register_many([add, greet])
    assert reg.has("add")
    assert reg.has("greet")


def test_registry_get_schemas():
    reg = ToolRegistry()
    reg.register_many([add, greet])
    schemas = reg.get_schemas(["add", "greet"])
    assert len(schemas) == 2
    names = {s.name for s in schemas}
    assert names == {"add", "greet"}


@pytest.mark.asyncio
async def test_registry_execute_sync_tool():
    reg = ToolRegistry()
    reg.register_function(add)
    ctx = ExecutionContext(session_id="test")
    result = await reg.execute(
        ToolCall(tool_name="add", arguments={"a": 3, "b": 4}), ctx
    )
    assert result.status == ToolStatus.SUCCESS
    assert result.output == 7.0


@pytest.mark.asyncio
async def test_registry_execute_async_tool():
    reg = ToolRegistry()
    reg.register_function(greet)
    ctx = ExecutionContext(session_id="test")
    result = await reg.execute(
        ToolCall(tool_name="greet", arguments={"name": "Alice"}), ctx
    )
    assert result.status == ToolStatus.SUCCESS
    assert "Alice" in str(result.output)


# ── Returned-error detection ────────────────────────────────────────
# A tool that RETURNS an error-shaped payload (without raising) must be
# surfaced as ToolStatus.ERROR so the reason reaches the model as a failure,
# not a green "success" it repeats. Two conventions are recognized.


@tool(name="fails_operator_style")
def fails_operator_style() -> dict:
    return {"error": "timer requires a note"}


@tool(name="fails_framework_style")
def fails_framework_style() -> dict:
    return {"status": "error", "message": "Provide 'agent_name' to spawn."}


@tool(name="ok_with_nested_errors")
def ok_with_nested_errors() -> dict:
    # Aggregate SUCCESS that merely contains nested per-item errors — the
    # top-level shape is success, so it must NOT be reclassified.
    return {"status": "completed", "results": [{"status": "error", "error": "x"}]}


@tool(name="ok_empty_error_field")
def ok_empty_error_field() -> dict:
    # Falsy error field on an otherwise successful result → still SUCCESS.
    return {"created": True, "error": ""}


@pytest.mark.asyncio
async def test_returned_error_operator_style_is_error():
    reg = ToolRegistry()
    reg.register_function(fails_operator_style)
    result = await reg.execute(
        ToolCall(tool_name="fails_operator_style", arguments={}),
        ExecutionContext(session_id="t"),
    )
    assert result.status == ToolStatus.ERROR
    assert result.error == "timer requires a note"
    # Output payload is preserved for the UI / tool card.
    assert result.output == {"error": "timer requires a note"}


@pytest.mark.asyncio
async def test_returned_error_framework_style_is_error():
    reg = ToolRegistry()
    reg.register_function(fails_framework_style)
    result = await reg.execute(
        ToolCall(tool_name="fails_framework_style", arguments={}),
        ExecutionContext(session_id="t"),
    )
    assert result.status == ToolStatus.ERROR
    assert "agent_name" in result.error


@pytest.mark.asyncio
async def test_nested_errors_stay_success():
    reg = ToolRegistry()
    reg.register_function(ok_with_nested_errors)
    result = await reg.execute(
        ToolCall(tool_name="ok_with_nested_errors", arguments={}),
        ExecutionContext(session_id="t"),
    )
    assert result.status == ToolStatus.SUCCESS


@pytest.mark.asyncio
async def test_falsy_error_field_stays_success():
    reg = ToolRegistry()
    reg.register_function(ok_empty_error_field)
    result = await reg.execute(
        ToolCall(tool_name="ok_empty_error_field", arguments={}),
        ExecutionContext(session_id="t"),
    )
    assert result.status == ToolStatus.SUCCESS


@pytest.mark.asyncio
async def test_registry_drops_unexpected_args():
    """A stray key the model emitted must not reach a handler that can't take
    it (would be an opaque TypeError); it is dropped and the call still runs."""
    reg = ToolRegistry()
    reg.register_function(add)
    ctx = ExecutionContext(session_id="test")
    result = await reg.execute(
        ToolCall(tool_name="add", arguments={"a": 1, "b": 2, "bogus": "x"}), ctx
    )
    assert result.status == ToolStatus.SUCCESS
    assert result.output == 3.0


@pytest.mark.asyncio
async def test_registry_missing_required_arg_is_clean_error():
    """A missing required argument surfaces as a clean tool error, not a raw
    TypeError from deep in the handler."""
    reg = ToolRegistry()
    reg.register_function(greet)
    ctx = ExecutionContext(session_id="test")
    result = await reg.execute(ToolCall(tool_name="greet", arguments={}), ctx)
    assert result.status == ToolStatus.ERROR
    assert "required" in str(result.output).lower() or "required" in str(result.error).lower()
