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

## Versioning

The SDK's minor version tracks the agent-plane proto contract it targets. Pin it like any
dependency; the vendored gRPC/protobuf gencode is shipped inside the wheel, so you don't run
`protoc` yourself. The runtime requires `protobuf>=6.31,<7`.

## License

Apache-2.0. See [LICENSE](LICENSE).
