# Cambrian Agent SDK

Build [Cambrian](https://github.com/cambrian-sh/core) agents in Python with zero boilerplate.

An agent is a small Python class. The SDK gives it a ReAct loop, tool calling, working memory,
and a gRPC server that speaks the Cambrian agent-plane contract — you write `run()` (or just
declare `@tool` methods) and the kernel does the orchestration.

```bash
pip install cambrian-agent-sdk      # or: uv pip install cambrian-agent-sdk
```

## Quickstart

```python
from cambrian_agent_sdk import CognitiveAgent

class AnalystAgent(CognitiveAgent):
    role = "You are a rigorous analytical reasoning engine."
    constraints = ("Ground every claim; say so when the context is insufficient.",)

agent = AnalystAgent(agent_id="analyst_agent", description="Analyses and compares information.")

if __name__ == "__main__":
    agent.serve()   # boots the gRPC agent server; the kernel connects and dispatches tasks
```

The kernel spawns your agent, hands it tasks over the agent plane, and your `run()`/`think()`
loop can call **memory** (`memory_query` for facts, `recall_precedents` for prior transitions
with their outcomes), **tools** (`tool_call` — kernel-owned or your own
`@tool` methods), and **delegate** (`yield_subgoal`) to the planner.

## Agent shapes

- **`CognitiveAgent`** — LLM reasoning: `run()` → `think()` ReAct loop, memory, and `@tool` actions.
- **`DeterministicAgent`** — *deprecated.* A scripted cell: typed `run()` only, no reasoning.
  Subclassing it emits a `DeprecationWarning`: the kernel no longer selects agents by static
  bid (default dispatch matches the capabilities your manifest declares) and has deprecated
  trait `"tool"`. Subclass `CognitiveAgent` and override `run()` with your scripted logic
  instead; deterministic kernel functions are kernel-owned tools (`ExecuteTool`). See
  `CHANGELOG.md` in the repository.
- **`DaemonAgent`** — a persistent background producer on the `SignalStream` contract.

Declare tools with the `@tool` decorator (auto-injected into the agent's action menu) and
capabilities with `@capability`. See `example-agents/` in the repo.

## Memory and event time

`MemoryClient.remember()` stores text; `recall()` retrieves it. Both understand **event
time**, when something happened as opposed to when it was stored:

```python
from datetime import datetime, timezone

mem.remember("We moved the launch to Friday.",
             occurred_at=datetime(2024, 5, 2, 14, 5, tzinfo=timezone.utc),
             time_precision="minute")
hits = mem.recall("when is the launch?",
                  reference_time=datetime(2024, 5, 10, tzinfo=timezone.utc))
hits[0]["event_time"], hits[0]["time_source"]   # aware UTC datetime, "explicit"
```

Times must be timezone-aware; a naive `datetime` raises `ValueError`. Unstated times are
labelled `ingest_default`, never shown as real. `parts=[(text, time), ...]` ingests a chat
with a time per turn, and `event_after` / `event_before` ask for a time window.

## Connecting to a kernel

The SDK dials the Substrate through one builder (`cambrian_agent_sdk.channel.build_channel`),
configured from the environment:

| Variable | Meaning |
| --- | --- |
| `CAMBRIAN_AGENT_TOKEN` | The agent's own credential (the kernel's named secret `agent:<id>:token`), sent as `authorization: Bearer` on every RPC. The kernel sets it for the agents it spawns. |
| `CAMBRIAN_AGENT_ID` | The agent the token was issued to; sent as `x-agent-id` when a call does not set one. |
| `CAMBRIAN_TLS_CA` | PEM root the kernel's certificate chains to. Set ⇒ TLS. Unset ⇒ plaintext, warned once. |
| `CAMBRIAN_TLS_SERVER_NAME` | Optional hostname override for certificate verification. |

A loopback kernel started with `server.insecure_localhost=true` needs none of these.

## Versioning

The SDK's minor version tracks the agent-plane proto contract it targets. Pin it like any
dependency; the vendored gRPC/protobuf gencode is shipped inside the wheel, so you don't run
`protoc` yourself. The runtime requires `protobuf>=6.31,<7`.

## License

Apache-2.0. See [LICENSE](LICENSE).
