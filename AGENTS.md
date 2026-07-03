Sub-repo: sdk
Language: Python
Top-level context: pyproject.toml + ../cambrian-core/docs/adr/0036-trait-aligned-cognitive-agent-sdk.md
Authoritative ADR: ../cambrian-core/docs/adr/0036-trait-aligned-cognitive-agent-sdk.md

# sdk, AGENTS.md

The Python agent SDK. Authors subclass one of three trait-aligned bases, decorate
`@tool` methods, and call `agent.serve()`. The kernel auto-discovers the agent
file by name and reads the `AGENT_DESCRIPTION` / `AGENT_MANIFEST` module-level
literals before the Python process boots.

## What this is

The SDK builds Cambrian agents in Python with zero boilerplate. Authors import
`CognitiveAgent`, `DeterministicAgent`, or `DaemonAgent` from
`cambrian_agent_sdk`, override `run()` (or `think()` for the cognitive trait),
and call `agent.serve()`. On PyPI, the package is `cambrian-agent-sdk` on its
own version line (currently `0.1.0`) under the Business Source License 1.1.
The gRPC proto is the only contract with the kernel: the SDK ships its own
vendored stubs under `cambrian_agent_sdk/_proto/` and never imports the Go
kernel.

## Hard rules

- The SDK only speaks the pinned proto. No second transport, no REST fallback,
  no direct database access.
- The Go kernel API stays unstable through v0.x. The proto surface and the
  `pyproject.toml` config schema are the only held-stable contracts.
- Local skills (`agent.local_skills`) shadow same-name system skills by
  lexical scoping (ADR-0046 D10). A `local_skills["foo"]` entry wins over a
  kernel-served skill with the same name in the agent's [skills] menu.
- No kernel bypass. Author code never reaches into Go internals, shared
  files, or the agent's host process. Everything crosses the gRPC boundary
  through `SubstrateClient`.

## Layout

| Path | Role |
| --- | --- |
| `cambrian_agent_sdk/` | The SDK package; load-bearing files: |
| | `base.py` (agent loop bases: `Agent` abstract + `CognitiveAgent` / `DeterministicAgent` / `DaemonAgent` traits; the ReAct pattern wires through `react.run_think`, ADR-0036 D1) |
| | `runtime.py` (gRPC server base for `AgentService` and `SignalStream`) |
| | `tools.py` (in-process `@tool` registration and `ToolRegistry`) |
| | `recurrence.py` (Local Recurrent Workspace gate, ADR-0041 D4) |
| | `reflection.py` (verbal reflection on the gate, ADR-0052) |
| `examples/` | One minimal agent: `benchmark_agent.py` |
| `example-agents/` | Production-style references (9 agents): `analyst_agent.py`, `calculator_agent.py`, `code_executor_agent.py`, `code_generator_agent.py`, `example_daemon_agent.py`, `pulse_agent.py`, `research_agent.py`, `summariser_agent.py`, `terminal_agent.py` |
| `tests/` | pytest suite |

## Build & verify

```sh
uv sync            # or: pip install -e .
pytest
ruff check
mypy
```

## Change discipline

- **Keep the sub-repo's `CONTEXT.md` in sync.** When you add, remove, or change code, architecture, status, modules, domain terms, or known gaps, update the sub-repo's `CONTEXT.md` to reflect the change. The context is the source of truth for AI agents (and humans) navigating the sub-repo; a stale context is worse than no context. Update the relevant section: `Module Breakdown` (paths), `Implementation Status` (areas with ADR + status), `Terminology Glossary` (new terms), `Known Gaps` (new deferred work), `Core Philosophy` (principle changes).

## Cross-repo pointer

- [`../CONTEXT.md`](../CONTEXT.md): the monorepo map; this sub-repo's place in the topology.
- [`../AGENTS.md`](../AGENTS.md): the four cross-repo invariants, including "gRPC proto is the only SDK contract".
- [`../cambrian-core/docs/adr/0036-trait-aligned-cognitive-agent-sdk.md`](../cambrian-core/docs/adr/0036-trait-aligned-cognitive-agent-sdk.md): the SDK's main ADR (trait taxonomy, structural enforcement).
- [`../cambrian-core/docs/adr/0041-local-recurrent-workspace.md`](../cambrian-core/docs/adr/0041-local-recurrent-workspace.md): the LRW gate implemented in `recurrence.py`.
- [`../cambrian-core/docs/adr/0046-agent-skills.md`](../cambrian-core/docs/adr/0046-agent-skills.md): skill model and the local-vs-system lexical scoping rule.
- [`../cambrian-core/docs/adr/0052-verbal-self-reflection.md`](../cambrian-core/docs/adr/0052-verbal-self-reflection.md): verbal reflection on the recurrence gate, in `reflection.py`.
