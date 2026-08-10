# Persistence

Sessions and message history are stored through one async interface,
`BaseRepository`, with three interchangeable backends. Pick by environment:

| Backend | Class | Use for | Extra dependency |
|---|---|---|---|
| `memory` | `InMemoryRepository` | Tests, throwaway scripts. Lost on process exit. | none |
| `sqlite` | `SQLiteRepository` | Local dev, single-process apps. One file on disk. | `pip install "fg-agents[sqlite]"` (aiosqlite) |
| `postgres` | `PostgresRepository` | Production, multi-process. | none — SQLAlchemy + asyncpg are core deps |

## Choosing a backend

### Agent facade (`memory=` argument)

```python
Agent(model=..., memory="memory")                                # default
Agent(model=..., memory="sqlite")                                # ./fg_agents.db
Agent(model=..., memory="sqlite:./my_agent.db")                  # explicit path
Agent(model=..., memory="postgres:postgresql+asyncpg://user:pass@host/db")
Agent(model=..., memory=my_repository)                           # any BaseRepository instance
```

The postgres form **requires** the URL after the colon; a bare `"postgres"`
raises `ValueError`.

### Low-level (`create_repository`)

```python
from fg_agents import create_repository

repo = create_repository("memory")
repo = create_repository("sqlite", db_path="agents.db")     # default path: "fg_agents.db"
repo = create_repository("postgres", db_url="postgresql+asyncpg://user:pass@host/db", echo=False)
```

### Web app (`create_app(db_url=...)`)

`create_app` picks the backend from the URL string:

- `db_url=None` → SQLite file `fg_agents.db` (dev default — needs
  `fg-agents[sqlite]` installed)
- URL containing `postgres`/`postgresql` → PostgreSQL
- URL containing `sqlite` → SQLite at the path after `///`
- anything else → in-memory (no error — check your URL)

## Initialization semantics

- **Agent facade — lazy.** The repository is initialized automatically on the
  first `run()`/`stream()` call (exactly once, concurrency-safe). Forgetting
  to initialize is never an error. `await agent.close()` closes it.
- **Low-level — explicit.** You must `await repo.initialize()` before handing
  it to `AgentEngine`, and `await repo.close()` on shutdown. Skipping
  `initialize()` is a bug (tables/connections won't exist).
- **`create_app` — managed.** Initialization and shutdown are wired into the
  FastAPI lifespan; you do nothing.

`initialize()` creates schema (tables) on SQL backends, so first-run against a
fresh database needs no separate migration step. PostgreSQL URLs must use the
async driver: `postgresql+asyncpg://...`.

## Notes

- Importing `SQLiteRepository` without `aiosqlite` installed gives a stub that
  raises `ImportError` with the install hint when called.
- All backends implement the same async CRUD surface for sessions and
  messages (`fg_agents/persistence/base.py`), so swapping backends is a
  one-line change.
