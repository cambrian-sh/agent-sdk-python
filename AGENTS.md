Sub-repo: sdk
Language: Python
Top-level context: pyproject.toml + ../cambrian-knowledge/adrs/0036-trait-aligned-cognitive-agent-sdk-v2.md
Authoritative ADR: ../cambrian-knowledge/adrs/0036-trait-aligned-cognitive-agent-sdk-v2.md

# sdk, AGENTS.md

## Knowledge substrate & tickets (MANDATORY — every task)

The full protocol is root [`../AGENTS.md` §0](../AGENTS.md) — read it. **Live status lives in HQ**
on a cut-over machine (`../cambrian-knowledge/.hq-cutover` exists — the owner's machine): use the
`hq` MCP tools (`ready`, `create_ticket`, `update_ticket`, `list_tickets`, `brief`, `record_finding`,
`verify_card`, `list_decisions` / `set_decision_status`, the benchmark tools) or
`C:/Users/afsin/Dev/hq/bin/hq.exe … --config ../hq.yaml`. There the old `bin/ticket` / `bin/bench`
scripts and the codebase-memory-mcp ticket tools **refuse writes**. Without the marker, the file
workflow of root §0.4/§0.5 applies. In short, for every change in this repo, however small:

1. **Before editing:** `ready` / `list_tickets` (`repo: "sdk"`) → `create_ticket` if the work is
   not ticketed → `update_ticket` action `claim`. `brief <path>` (or the card that loads when you
   open a file) for the cards, ADRs, decisions and open tickets that govern the code you will
   touch; a drifted card or stale ADR is a claim to re-verify against code before relying on it.
2. **While working:** `log` findings, dead ends and `decision:` lines on the ticket. A bug or
   follow-up you will not fix now gets its own `create_ticket` (`triage=true`), immediately.
3. **Decisions:** architectural → `cambrian-knowledge/adrs/NNNN-slug.md` (proposed, per
   `cambrian-knowledge/meta/SCHEMA.md`; its lifecycle status is set in HQ, not the frontmatter);
   operating rule → `cambrian-knowledge/decisions/`; tradeoff → the ADR's `tradeoffs:`. **Never
   write ADRs to `docs/adr/`** — no sub-repo has that directory; ADRs live in `cambrian-knowledge/adrs/`.
4. **Before reporting done:** benchmark runs → a run record each (`record_benchmark_run`; the
   procedure is root `../AGENTS.md` §0.5); other numbers you produced → the ADR's `measurements:`;
   gotchas → `observations:`; re-verified ADRs → `last_verified` + `verified_at_rev`; cards you
   re-read against code → `verify_card` (the Stop hook blocks while a card your edit drifted is
   unverified); `CONTEXT.md` if architecture or gaps changed (never edit between the
   `<!-- hq:cards:start -->` / `<!-- hq:cards:end -->` markers — `hq context` generates that block).
   Then `close` the ticket with a resolution (files changed, ADRs updated, tests run and result) —
   or, if unfinished, `log` where you stopped and `release`.

Pass `by: "agent:<name>"` on ticket calls. Never hand-edit `cambrian-knowledge/tickets/`; tickets
never close from commit messages.

**Default execution lane (ADR-0150, owner decision 2026-11-13).** A request runs in the
**single-agent lane** by default: one agent holds the person's tools and answers itself, fanning
out read-only work through the kernel-run `run_subtasks` tool (`execution.lane.default:
single_agent`, `cambrian-core/internal/config/lane.go`). The ADR-0145 **planner lane** (planner →
Gatekeeper → Dispatcher → DAG, coverage, verifier) runs only under `execution.lane.default:
planner` and is kept for research — so plan views, dispatch events and Gatekeeper scoring appear
only in that lane. See [`decisions/default-lane-single-agent-with-fan-out.md`](../cambrian-knowledge/decisions/default-lane-single-agent-with-fan-out.md).

The Python agent SDK. Authors subclass one of three trait-aligned bases, decorate
`@tool` methods, and call `agent.serve()`. The kernel auto-discovers the agent
file by name and reads the `AGENT_DESCRIPTION` / `AGENT_MANIFEST` module-level
literals before the Python process boots.

## What this is

The SDK builds Cambrian agents in Python with zero boilerplate. Authors import
`CognitiveAgent` or `DaemonAgent` from `cambrian_agent_sdk` (`DeterministicAgent`
still exists but is deprecated since 2026-10-31 and warns on subclassing; new agents
use `CognitiveAgent` with an overridden `run()`), override `run()` (or `think()` for
the cognitive trait), and call `agent.serve()`. On PyPI, the package is `cambrian-agent-sdk` on its
own version line (currently `0.1.5`) under the Apache License 2.0 (ADR-0057 D10,
amended 2026-10-31).
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
| | `base.py` (agent loop bases: `Agent` abstract + `CognitiveAgent` / `DeterministicAgent` (deprecated) / `DaemonAgent` traits; the ReAct pattern wires through `react.run_think`, ADR-0036 D1) |
| | `runtime.py` (trait-agent `AgentService` servicer and `serve()` boot) · `server.py` (gRPC server lifecycle) · `daemon.py` (`DaemonAgent` runtime: the `SignalStream` side) |
| | `channel.py` (the one way the SDK dials the kernel) · `react.py` (the `think()` ReAct loop) · `context_engine.py` (what one run puts in front of the model, ADR-0139 D3) · `working_memory.py` (LRW buffers, ADR-0041 D1) |
| | `tools.py` (in-process `@tool` registration and `ToolRegistry`) |
| | `recurrence.py` (Local Recurrent Workspace gate, ADR-0041 D4) |
| | `reflection.py` (verbal reflection on the gate, ADR-0052) |
| | `profiles.py` (declarative agent profiles: loader, strict validator, tool allowlist, `ProfileScopedSubstrate`; format in `cambrian-core/agents/profiles/README.md`) |
| `examples/` | The `@tool` tutorial: `tool_tutorial_agent.py` (a tutorial, not a shipped agent) |
| `example-agents/` | Reference agents: `pulse_agent.py` (daemon), `example_ingress_agent.py` (ingress) |
| `tests/` | pytest suite |

## Build & verify

```sh
uv sync            # or: pip install -e .
pytest             # the only configured gate (pyproject dev extras: pytest, grpcio-tools)
```

`ruff` and `mypy` are **not** configured here: neither is a dev dependency and `pyproject.toml` has
no `[tool.ruff]` / `[tool.mypy]` section, so nothing runs them and an ad-hoc run reports real errors
(F821 in `server.py`, ~70 mypy errors) despite the package shipping `py.typed`. Adding them (or
deciding not to) is `T-20261113-sdk-agents-md-prescribes-ruff-check-and-mypy`; until then, do not
claim a lint or type gate passed.

## Change discipline

- **Keep the sub-repo's `CONTEXT.md` in sync.** When you add, remove, or change code, architecture, status, modules, domain terms, or known gaps, update the sub-repo's `CONTEXT.md` to reflect the change. The context is the source of truth for AI agents (and humans) navigating the sub-repo; a stale context is worse than no context. Update the relevant section: `Module Breakdown` (paths), `Implementation Status` (areas with ADR + status), `Terminology Glossary` (new terms), `Known Gaps` (new deferred work), `Core Philosophy` (principle changes).

## Cross-repo pointer

- [`../CONTEXT.md`](../CONTEXT.md): the monorepo map; this sub-repo's place in the topology.
- [`../AGENTS.md`](../AGENTS.md): the cross-repo invariants (§5), including "gRPC proto is the only SDK contract".
- [`../cambrian-knowledge/adrs/0036-trait-aligned-cognitive-agent-sdk-v2.md`](../cambrian-knowledge/adrs/0036-trait-aligned-cognitive-agent-sdk-v2.md): the SDK's main ADR (trait taxonomy, structural enforcement).
- [`../cambrian-knowledge/adrs/0041-local-recurrent-workspace-lrw-for-the-agent-loop.md`](../cambrian-knowledge/adrs/0041-local-recurrent-workspace-lrw-for-the-agent-loop.md): the LRW gate implemented in `recurrence.py`.
- [`../cambrian-knowledge/adrs/0046-agent-skills-authored-procedural-capabilities-system-and-age.md`](../cambrian-knowledge/adrs/0046-agent-skills-authored-procedural-capabilities-system-and-age.md): skill model and the local-vs-system lexical scoping rule.
- [`../cambrian-knowledge/adrs/0052-verbal-self-reflection-on-the-recurrence-gate.md`](../cambrian-knowledge/adrs/0052-verbal-self-reflection-on-the-recurrence-gate.md): verbal reflection on the recurrence gate, in `reflection.py`.
