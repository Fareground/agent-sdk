"""Tests for tool handlers (code execution, dispatch routing)."""
import os

import pytest

from fg_agents.core.types import (
    ExecutionContext,
    RegisteredTool,
    ToolCall,
    ToolStatus,
    ToolType,
)
from fg_agents.tools.handlers import (
    TOOL_TYPE_HANDLERS,
    dispatch_tool,
    execute_api_tool,
    execute_code_tool,
    execute_db_query_tool,
    execute_workflow_tool,
)

# ══════════════════════════════════════════════════════════════════════
# Code tool — safety gate
# ══════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_code_tool_blocked_without_env():
    """Code execution must be rejected when FG_ALLOW_CODE_EXECUTION is not set."""
    # Ensure the env var is unset
    os.environ.pop("FG_ALLOW_CODE_EXECUTION", None)

    tool_def = RegisteredTool(
        name="run_code",
        description="Execute Python",
        tool_type=ToolType.CODE,
    )
    context = ExecutionContext(session_id="s1")

    result = await execute_code_tool(tool_def, args={"code": "print('hi')"}, context=context)

    assert result["status"] == "error"
    assert "FG_ALLOW_CODE_EXECUTION" in result["message"]


@pytest.mark.asyncio
async def test_code_tool_allowed_with_env():
    """Code execution should succeed when the env var is set."""
    tool_def = RegisteredTool(
        name="run_code",
        description="Execute Python",
        tool_type=ToolType.CODE,
    )
    context = ExecutionContext(session_id="s1")

    os.environ["FG_ALLOW_CODE_EXECUTION"] = "true"
    try:
        result = await execute_code_tool(
            tool_def,
            args={"code": "import json; print(json.dumps({'result': 42}))"},
            context=context,
        )
        assert result["status"] == "success"
        assert result["result"]["result"] == 42
    finally:
        os.environ.pop("FG_ALLOW_CODE_EXECUTION", None)


# ══════════════════════════════════════════════════════════════════════
# Handler dispatch routing
# ══════════════════════════════════════════════════════════════════════

def test_dispatch_tool_routing():
    """TOOL_TYPE_HANDLERS should map all non-function tool types to the correct handlers."""
    assert ToolType.API in TOOL_TYPE_HANDLERS
    assert ToolType.DB_QUERY in TOOL_TYPE_HANDLERS
    assert ToolType.WORKFLOW in TOOL_TYPE_HANDLERS
    assert ToolType.CODE in TOOL_TYPE_HANDLERS

    assert TOOL_TYPE_HANDLERS[ToolType.API] is execute_api_tool
    assert TOOL_TYPE_HANDLERS[ToolType.DB_QUERY] is execute_db_query_tool
    assert TOOL_TYPE_HANDLERS[ToolType.WORKFLOW] is execute_workflow_tool
    assert TOOL_TYPE_HANDLERS[ToolType.CODE] is execute_code_tool


@pytest.mark.asyncio
async def test_dispatch_tool_unknown_type_returns_error():
    """dispatch_tool should return an error ToolResult for unregistered tool types (e.g. FUNCTION)."""
    tool_def = RegisteredTool(
        name="my_func",
        description="A plain function tool",
        tool_type=ToolType.FUNCTION,
    )
    tool_call = ToolCall(tool_name="my_func", arguments={})
    context = ExecutionContext(session_id="s1")

    result = await dispatch_tool(tool_def, tool_call, context)

    assert result.status == ToolStatus.ERROR
    assert "No handler" in result.error


@pytest.mark.asyncio
async def test_api_tool_rejects_offhost_and_bad_scheme():
    """The SSRF guard: a URL that resolves to a different host than the
    configured base, or a non-http(s) scheme, is refused before any request."""
    from fg_agents.core.errors import ToolExecutionError
    from fg_agents.core.types import ExecutionContext

    ctx = ExecutionContext(session_id="t")

    # Absolute endpoint pointing off the configured host is rejected (host-pin).
    off_host = RegisteredTool(
        id="x", name="api", description="", tool_type=ToolType.API,
        parameters=[],
        config={"base_url": "https://api.internal.example", "endpoint": "http://evil.com/x"},
    )
    with pytest.raises(ToolExecutionError):
        await execute_api_tool(off_host, {}, ctx)

    # Non-http(s) scheme is rejected.
    bad_scheme = RegisteredTool(
        id="y", name="api", description="", tool_type=ToolType.API,
        parameters=[],
        config={"base_url": "file:///etc", "endpoint": "passwd"},
    )
    with pytest.raises(ToolExecutionError):
        await execute_api_tool(bad_scheme, {}, ctx)


def test_read_only_sql_guard_rejects_writes_on_any_backend():
    """read_only must mean read-only regardless of backend, not only Postgres."""
    from fg_agents.core.errors import ToolExecutionError
    from fg_agents.tools.handlers import _assert_read_only

    _assert_read_only("SELECT * FROM t WHERE x = :x")  # ok
    _assert_read_only("  with cte as (select 1) select * from cte")  # ok
    # Legit reads with a semicolon inside a string literal must still pass.
    _assert_read_only("SELECT * FROM t WHERE note = 'a;b'")
    for bad in [
        "UPDATE t SET x = 1",
        "DELETE FROM t",
        "DROP TABLE t",
        "INSERT INTO t VALUES (1)",
        "SELECT 1; DROP TABLE t",                          # piggybacked write
        "select 1 -- \n; delete from t",
        "WITH x AS (DELETE FROM users RETURNING *) SELECT * FROM x",  # writable CTE
        "EXPLAIN ANALYZE DELETE FROM users",               # ANALYZE runs the DML
        "SELECT * INTO new_table FROM users",              # SELECT INTO writes
        "select pg_sleep(1); truncate t",
    ]:
        with pytest.raises(ToolExecutionError):
            _assert_read_only(bad)
