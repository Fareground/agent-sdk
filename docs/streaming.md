# Streaming: the SSE wire format

This is the contract between the framework and your frontend. Every event the
engine emits is a `StreamEvent` (see `src/fg_agents/streaming/events.py`),
serialized as one Server-Sent Event frame.

## The wire frame

```
id: <event uuid>
event: <event type, e.g. llm.text_delta>
data: <full event as JSON>

```

The `data:` JSON always has the same envelope:

```json
{
  "id": "6f6c…",
  "type": "llm.text_delta",
  "session_id": "session-1",
  "data": { "text": "Hel" },
  "timestamp": "2026-08-10T17:02:11.412Z",
  "turn_number": 1
}
```

- `id` — unique per event; doubles as the SSE `id:` for `Last-Event-ID`
  reconnection (`GET /sessions/{id}/stream` replays and skips past it).
- `type` — one of the event types below; also the SSE `event:` name, so
  `EventSource.addEventListener("llm.text_delta", …)` works directly.
- `data` — the per-type payload (tables below).
- `turn_number` — the ReAct turn, `null` for session-level events.

The stream also carries periodic heartbeat comments to keep proxies from
closing idle connections; SSE clients ignore them automatically.

## Event types

Listed in the order a typical run emits them.

| Event type | Payload (`data`) fields | When emitted |
|---|---|---|
| `session.started` | `agent_id`, `agent_name` | Once, when a run begins on a session. |
| `turn.started` | `turn` | At the start of each ReAct turn. |
| `llm.thinking` | `text` | Streaming chunk of model reasoning/thinking (models that expose it). |
| `llm.text_delta` | `text` | Streaming chunk of assistant text. Concatenate these to render the reply. |
| `llm.tool_call` | `tool_name`, `tool_call_id`, `arguments` | The model requested a tool call. |
| `tool.executing` | `tool_name`, `tool_call_id` | Tool execution is starting. |
| `tool.result` | `tool_name`, `tool_call_id`, `status`, `duration_ms`, `output_preview`; optional `card_data`, `result` | Tool execution finished (success or error — see `status`). |
| `subagent.started` | `child_session_id`, `agent_name`, `task` (truncated to 200 chars) | Orchestrator delegated work to a sub-agent. |
| `subagent.completed` | `child_session_id`, `agent_name`, `status` | A sub-agent finished. |
| `context.compacting` | `messages_before`, `messages_after` | The conversation history was compressed to fit the context window. |
| `turn.completed` | `turn`, `stop_reason` | End of a ReAct turn. `stop_reason`: `end_turn`, `tool_use`, `max_turns`, `max_tokens`, `error`, `cancelled`. |
| `session.completed` | `total_turns`, `input_tokens`, `output_tokens`, `final_output` | The run finished successfully. `final_output` is the complete assistant reply. |
| `error` | `error_type`, `message` (truncated to 1000 chars), `recoverable` | Something failed. If `recoverable` is true the session can continue. |

## A minimal frontend loop

```js
// POST /api/agent/sessions            -> { session_id }
// POST /api/agent/sessions/{id}/messages  (SSE response)
let reply = "";
for await (const event of sseEvents(response)) {   // your SSE parser
  const e = JSON.parse(event.data);
  switch (e.type) {
    case "llm.text_delta":    reply += e.data.text; render(reply); break;
    case "tool.executing":    showSpinner(e.data.tool_name); break;
    case "session.completed": render(e.data.final_output); break;
    case "error":             showError(e.data.message); break;
  }
}
```

Ready-made clients live in [`examples/frontend/`](../examples/frontend):
`sse-client.js` (vanilla) and `use-agent.ts` (React hook).

## Consuming events in Python

The same `StreamEvent` objects come out of `Agent.stream(...)` and
`AgentEngine.run(...)` directly — no SSE involved:

```python
async for event in agent.stream("hello"):
    if event.type == EventType.LLM_TEXT_DELTA:
        print(event.data["text"], end="")
```

`event.to_sse()` produces the wire frame above; `event.to_dict()` the JSON
envelope as a dict.
