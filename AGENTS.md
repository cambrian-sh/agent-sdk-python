Sub-repo: sdk
Language: Python
Top-level context: pyproject.toml + HQ document `adrs/0036-trait-aligned-cognitive-agent-sdk-v2.md`
Authoritative ADR: HQ document `adrs/0036-trait-aligned-cognitive-agent-sdk-v2.md`

# sdk, AGENTS.md

## Knowledge & tickets in HQ (MANDATORY — every task)

The full protocol is root [`../AGENTS.md` §0](../AGENTS.md) — read it. **All project knowledge and
all live status live in HQ**: ADRs, decisions, investigations, research and knowledge cards are
versioned HQ documents (the `cambrian-knowledge/` repo is read-only until it is archived — never edit
it), and tickets, ADR status, card freshness and benchmark runs are HQ records. Use the `hq` MCP
tools (`ready`, `create_ticket`, `update_ticket`, `list_tickets`, `brief`, `read_doc`, `search_docs`,
`write_doc`, `record_finding`, `verify_card`, `list_decisions` / `set_decision_status`, the benchmark
tools) or `C:/Users/afsin/Dev/hq/bin/hq.exe … --config ../hq.yaml`. Every call carries your HQ token
(MCP header; the CLI and hooks read `~/.hq/token-cambrian`) — never print it. In short, for every
change in this repo, however small:

1. **Before editing:** `ready` / `list_tickets` (`repo: "sdk"`) → `create_ticket` if the work is
   not ticketed → `update_ticket` action `claim`. `brief <path>` (or the card that loads when you
   open a file) for the cards, ADRs, decisions and open tickets that govern the code you will
   touch; a drifted card or stale ADR is a claim to re-verify against code before relying on it.
2. **While working:** `log` findings, dead ends and `decision:` lines on the ticket. A bug or
   follow-up you will not fix now gets its own `create_ticket` (`triage=true`), immediately.
3. **Decisions:** architectural → document `adrs/NNNN-slug.md` via `write_doc` (`status: proposed`,
   template `meta/SCHEMA.md`; its lifecycle status is set in HQ, not the frontmatter); operating rule
   → document `decisions/<slug>.md`; tradeoff → the ADR's `tradeoffs:`. ADR and decision writes are
   **drafts** only the owner approves — never approve one yourself. **Never write ADRs to
   `docs/adr/`** — no sub-repo has that directory.
4. **Before reporting done:** benchmark runs → a run record each (`record_benchmark_run`; the
   procedure is root `../AGENTS.md` §0.5); other numbers you produced → the ADR's `measurements:`;
   gotchas → `observations:` (ADR drafts) or `record_finding` on the card; re-verified ADRs →
   `last_verified` + `verified_at_rev`; cards you re-read against code → `verify_card` (the Stop hook
   blocks while a card your edit drifted is unverified); architecture, behaviour or gaps changed →
   the area's card via `read_doc` → `write_doc` (with `base_rev` and a reason; never an empty or
   truncated text). `CONTEXT.md` here is a short stub — never add facts to it. Then `close` the
   ticket with a resolution (files changed, documents written with their revisions, drafts left for
   the owner, tests run and result) — or, if unfinished, `log` where you stopped and `release`.

Pass `by: "agent:<name>"` on HQ calls (you are recorded as `agent:<name>@<person>`). Tickets never
close from commit messages.

**Default execution lane (ADR-0150, owner decision 2026-11-13).** A request runs in the
**single-agent lane** by default: one agent holds the person's tools and answers itself, fanning
out read-only work through the kernel-run `run_subtasks` tool (`execution.lane.default:
single_agent`, `cambrian-core/internal/config/lane.go`). The ADR-0145 **multi-agent research lane** (planner →
Gatekeeper → Dispatcher → DAG, coverage, verifier) runs only under `execution.lane.default:
planner` and is kept for research — so plan views, dispatch events and Gatekeeper scoring appear
only in that lane. See document `decisions/default-lane-single-agent-with-fan-out.md`.

The Python agent SDK. Authors subclass one of three trait-aligned bases, decorate
`@tool` methods, and call `agent.serve()`. The kernel auto-discovers the agent
file by name and reads the `AGENT_DESCRIPTION` / `AGENT_MANIFEST` module-level
literals before the Python process boots.

## What this is

The SDK builds Cambrian agents in Python with zero boilerplate. Authors import
`CognitiveAgent` or `DaemonAgent` from `cambrian_agent_sdk` (`DeterministicAgent`
was deleted 2026-11-14; a scripted capability is a `CognitiveAgent` with an overridden
`run()`), override `run()` (or `think()` for
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
| | `base.py` (agent loop bases: `Agent` abstract + `CognitiveAgent` / `DaemonAgent` traits (`DeterministicAgent` deleted 2026-11-14); the ReAct pattern wires through `react.run_think`, ADR-0036 D1) |
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
uv sync --extra dev   # or: pip install -e ".[dev]"
pytest                # tests
ruff check            # lint: E, F, B, I (E501 ignored; _proto/ excluded) — [tool.ruff] in pyproject.toml
mypy                  # types: the cambrian_agent_sdk package, _proto/ excluded — [tool.mypy]
```

All three pass on a clean tree (ENG-1450, 2026-11-14). No CI workflow runs them yet, so run them
yourself before claiming a change is clean.

## Change discipline

- **Keep the knowledge cards true.** When you add, remove, or change code, architecture, behaviour, modules, domain terms, or known gaps, update the card(s) for that area in HQ (`brief <path>` names them; `read_doc` → edit → `write_doc` with a reason and `base_rev`, then `verify_card`). A new area gets a new card (`read_doc cards/README.md` for the format); new terms go into `card.concept.glossary`; known gaps are tickets, not prose. The cards are the source of truth for agents and humans navigating the repo; a stale card is worse than none. `CONTEXT.md` is a short stub and holds no facts.

## Cross-repo pointer

- [`../CONTEXT.md`](../CONTEXT.md): the monorepo map; this sub-repo's place in the topology.
- [`../AGENTS.md`](../AGENTS.md): the cross-repo invariants (§5), including "gRPC proto is the only SDK contract".
- document `adrs/0036-trait-aligned-cognitive-agent-sdk-v2.md`: the SDK's main ADR (trait taxonomy, structural enforcement).
- document `adrs/0041-local-recurrent-workspace-lrw-for-the-agent-loop.md`: the LRW gate implemented in `recurrence.py`.
- document `adrs/0046-agent-skills-authored-procedural-capabilities-system-and-age.md`: skill model and the local-vs-system lexical scoping rule.
- document `adrs/0052-verbal-self-reflection-on-the-recurrence-gate.md`: verbal reflection on the recurrence gate, in `reflection.py`.
