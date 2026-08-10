"""
Core-install behavior: `pip install fg-agents` ships no web stack and no
SQLAlchemy — the engine, tools, ask/Agent facade, and in-memory persistence
must work with the optional dependencies absent, and the optional surfaces
must fail with actionable install hints.

The optional packages ARE installed in the dev environment, so absence is
simulated in a subprocess whose meta_path blocks their import.
"""

import subprocess
import sys

BLOCKED = ("fastapi", "starlette", "uvicorn", "sqlalchemy", "asyncpg", "aiosqlite")

_SCRIPT = r"""
import asyncio
import importlib.abc
import sys

BLOCKED = {blocked!r}


class _Blocker(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        root = fullname.split(".")[0]
        if root in BLOCKED:
            raise ImportError(f"blocked for core-only test: {{fullname}}")
        return None


sys.meta_path.insert(0, _Blocker())
for mod in list(sys.modules):
    if mod.split(".")[0] in BLOCKED:
        del sys.modules[mod]

# 1. The package imports cleanly without any optional dependency.
import fg_agents

# 2. Agent construction + a full run on the in-memory repository works.
from tests.helpers import MockLLM, make_text_response

agent = fg_agents.Agent(
    model="mock:test", llm=MockLLM([make_text_response("4")])
)
result = asyncio.run(agent.run("What's 2+2?"))
assert result.text == "4", result.text

# 3. SQLAlchemy's Base is not leaked on the public core surface.
assert "Base" not in fg_agents.__all__
assert not hasattr(fg_agents, "Base")

# 4. Web surface fails with the actionable extras hint.
try:
    fg_agents.create_app(agents={{}})
except ImportError as e:
    assert "fg-agents[web]" in str(e), str(e)
else:
    raise AssertionError("create_app should raise ImportError without fastapi")

# 5. Postgres/sqlite backends fail with the actionable extras hint.
for backend, extra in (("postgres", "fg-agents[postgres]"), ("sqlite", "fg-agents[sqlite]")):
    try:
        fg_agents.create_repository(backend, **({{"db_url": "x"}} if backend == "postgres" else {{}}))
    except ImportError as e:
        assert extra in str(e), str(e)
    else:
        raise AssertionError(f"{{backend}} backend should raise ImportError")

print("core-only-ok")
"""


def test_core_only_import_and_run():
    script = _SCRIPT.format(blocked=BLOCKED)
    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    assert "core-only-ok" in proc.stdout
