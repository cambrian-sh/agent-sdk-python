# sdk, CONTEXT.md

The manual for the `cambrian-agent-sdk` Python package. It complements
`sdk/AGENTS.md` (the agent guide, with the hard rules) by covering
what's in the package: the trait taxonomy, the local-recurrent loop,
the agent skill model, and the gRPC contract surface.

## Knowledge & work tracking

This file is the *current shape* of the SDK. It is not where decisions or work live:

- **Why it is this shape** — ADRs and decisions in `../cambrian-knowledge/`. Query the
  graph (`search_graph`, `query_graph`) rather than citing from memory; check an ADR's
  freshness before building on it.
- **What is being done, and what is broken** — tickets: `list_tickets` (`repo: "sdk"`), or the
  knowledge-board Tickets view. Every change is ticketed, claimed, and closed with a
  resolution — the protocol is root [`AGENTS.md`](../AGENTS.md) §0.
- **Every gap listed in this file should have a ticket.** Put its id next to the gap
  (`— T-20261031-…`); when the ticket closes, update or remove the gap here in the same change.

## Ingress agents (ADR-0090)

`cambrian_agent_sdk.IngressAgent` is the base class for an entry point into Cambrian —
Telegram, a webhook receiver, a websocket listener, an inbound API. Chat is one payload type
riding through one.

It exists because neither other base class can express it: `DaemonAgent` produces signals but
is never called (no delivery path), `CognitiveAgent` is called but produces none (no inbound
path). `IngressAgent` does both — it serves the gRPC endpoint the kernel delivers to on the
daemon's existing UDS socket, and runs the signal stream on a background thread. The server
owns the process lifetime, so an ingress that can no longer poll can still deliver.

Two methods, one per direction, and neither waits for the other:

- `listen()` — your inbound loop; call `receive(external_id, text)` per message. Supervised,
  so a dropped connection is restarted with backoff rather than ending the ingress.
- `on_deliver(recipient, text, conversation_id)` — send one outbound message. No return value,
  because nothing is correlated with it. Raise `PermanentDeliveryError` for something a retry
  can never fix (blocked, deleted); raise anything else for something that might clear, which
  the kernel treats as transient.

**There is no way to declare your own surface and no way to choose a recipient.** The surface
comes from the registration an operator made out of band; the recipient is resolved by the
kernel from the conversation. A daemon is a black box, and a black box that asserts its own
privilege level is not a security boundary (INV-5) — so the API simply cannot express it.

Reference implementation: `example-agents/example_ingress_agent.py` (a polled file stands in
for the network, so it runs with no credentials).


## Implementation Status

| Area | Source | Status |
| --- | --- | --- |
| Trait taxonomy (`Agent` abstract + `CognitiveAgent` / `DeterministicAgent` / `DaemonAgent`) | `base.py`; ADR-0036 D1 | Implemented |
| ReAct pattern: `run_think` driven by the cognitive trait's `think()` | `react.py` | Implemented |
| Local Recurrent Workspace gate (recurrent reconciliation) | `recurrence.py`; ADR-0041 D4 | Implemented |
| Verbal self-reflection on the gate (pinned working-memory lesson) | `reflection.py`; ADR-0052 | Implemented (residual) |
| In-process `@tool` registration, schema auto-derivation, `ToolRegistry` | `tools.py`; ADR-0036 D4 | Implemented |
| Typed working memory (`ToolCard`, `TextEntry`, `Step`, bounded assembly) | `working_memory.py`, `types.py`; ADR-0041 D1-D3 | Implemented |
| gRPC clients (`SubstrateClient`, `MemoryClient`, `ArtifactManager`) | `__init__.py` (`SubstrateClient`), `clients.py` (memory, artifacts) | Implemented |
| gRPC server base for `AgentService` and health | `runtime.py`, `server.py`; ADR-0036 D2 | Implemented |
| Daemon-agent path: `SignalStream` runtime + supervised `daemon_loop` | `daemon.py`; ADR-0033, ADR-0036 issue 0036-06 | Implemented. **Fixed 2026-08-08: one loop per daemon, not one per reconnect** — see Known Gaps. |
| Agent skills: lexical scoping of local vs system skills | ADR-0046 | Proposed |
| Typed errors (`BudgetExceededError`, `InvalidTagError`, `ArtifactNotFound`, `SelfDelegationError`) | `errors.py`, `clients.py` | Implemented |
| Structured logging (`SlogHandler` writing slog-compatible JSON to stdout) | `_logging.py` | Implemented |
| Vendored proto stubs (the pinned gRPC contract) | `_proto/cambrian_pb2.py`, `_proto/cambrian_pb2_grpc.py` | Implemented |
| Public surface, `assemble_context` helper, top-level `configure_logging` | `__init__.py` | Implemented |
| Cache-friendly v2 prompt builder, code-block extraction, default constraints | `helpers.py` | Implemented |

## Core Philosophy

The SDK is a trait-aligned cognitive agent library for Python. Authors
subclass one of three trait bases (`CognitiveAgent`, `DeterministicAgent`,
`DaemonAgent`) and the kernel auto-discovers the agent file by name,
reading the `AGENT_DESCRIPTION` and `AGENT_MANIFEST` module-level
literals before the Python process boots. The gRPC proto is the only
contract with the Go kernel: the SDK ships its own vendored stubs under
`cambrian_agent_sdk/_proto/` and never imports the Go runtime. The
trait taxonomy maps 1:1 to the class structure, so a `DeterministicAgent`
literally has no `think()` and a `DaemonAgent` serves a different
contract (`SignalStream`) than the task-responders.

Two cognitive-loop refinements shape the surface. ADR-0041 adds a
**structural** recurrent gate that counts how many times a proposed
tool call near-duplicates a prior failed card, escalating a soft nudge
into a hard veto. ADR-0052 adds a **verbal** layer on the first hard
veto: one small LLM call extracts why the action failed and what to
change, and the lesson is pinned into working memory so every later
attempt reads it. The structural gate counts; the verbal reflection
learns. On the skills plane, ADR-0046 splits system skills
(kernel-owned, scope-gated) from agent-local skills (always present in
the SDK). A local skill with the same name as a system skill shadows
it by lexical scoping, with no central ranking involved.

## Module Breakdown

| Path | Role |
| --- | --- |
| `cambrian_agent_sdk/__init__.py` | Public surface, `assemble_context`, top-level `configure_logging`, the `SubstrateClient` definition |
| `cambrian_agent_sdk/base.py` | `Agent` abstract + the three trait classes; wires `run_think` into the cognitive trait |
| `cambrian_agent_sdk/clients.py` | `MemoryClient` (recall lanes + remember — `recall` facts, `recall_actions`, `recall_precedents` prior transitions; `recall_entity` is **deprecated**: the kernel removed the entity lane on 2026-08-14 and serves `x-lane=entity` from the fact lane), `ArtifactManager` (save/get/list_from_step), read-only `WorkingMemory` view, plus `InvalidTagError`, `ArtifactNotFound`, `SelfDelegationError` |
| `cambrian_agent_sdk/daemon.py` | `start_daemon` + `run_supervised`: the `SignalStream` runtime with crash supervision (capped exponential backoff, 30s cap). The supervised loop is started ONCE, **outside** the reconnect loop — that placement is load-bearing, see Known Gaps |
| `cambrian_agent_sdk/errors.py` | `BudgetExceededError`, the typed signal that a task refused on budget grounds |
| `cambrian_agent_sdk/helpers.py` | `build_prompt` (v2 cache-friendly layout), `extract_code_block`, `find_step_ref`, `DEFAULT_CONSTRAINTS`, `DEFAULT_ANTI_PATTERNS`, XML escape |
| `cambrian_agent_sdk/host.py` | `host_facts` / `host_block` / `goos`: the host the agent is running ON (`os`, `home`, `desktop`, `cwd`, `path_separator`) plus the path rules that make those values usable, appended once to the ReAct prompt's loop-invariant prefix. **This is the executing half of a split with the kernel's planner prompt:** the planner is told only the OS and told explicitly that paths are not its job, because it cannot see the host and because its plans are replayed as `<PlanLTM>` precedents (an absolute path with a username in it is wrong under another account). Three rules to preserve: `goos()` uses **Go's** `runtime.GOOS` spelling so plan and agent name the machine identically; a fact that cannot be determined is **omitted rather than guessed** (the model treats what it is given as authoritative); and `host_facts` is cached so the prompt prefix stays byte-identical across turns. Desktop resolution is redirection-aware — Windows reads Explorer's Shell Folders registry value (OneDrive Known-Folder redirection means `home/Desktop` can exist as a shadow folder while the real desktop is under `OneDrive`), Linux reads `XDG_DESKTOP_DIR` (a localised desktop is `~/Escritorio`). Origin: ADR-0100 P3 residual #3 — the Scout carried the only "never write `~/Desktop`" rule and took it with it when retired |
| `cambrian_agent_sdk/react.py` | `run_think` (the ReAct loop), `ReActLoopError`, prompt/action-protocol composition, all per-action handlers. The retrieval vocabulary is two actions on ONE budget: `memory_query` (facts) and `recall_precedents` (ADR-0049 D11 world-model pull lane — prior transitions + outcomes, failures first; empty means "no precedent") |
| `cambrian_agent_sdk/recurrence.py` | `count_failed_duplicates` and `count_successful_duplicates`: the local recurrent reconciliation gate |
| `cambrian_agent_sdk/reflection.py` | `build_reflection_prompt`: pure prompt builder for the verbal self-reflection (ADR-0052) |
| `cambrian_agent_sdk/runtime.py` | `TraitServicer`, `start_agent_server`, `_task_from_handoff`, `_coerce_agent_result`, the single-threaded gRPC runtime (ADR-0036 D2) |
| `cambrian_agent_sdk/server.py` | gRPC server lifecycle, `_parse_listen_address`, `_parse_substrate_addr`, `is_daemon_mode`, `_wire_health` (grpc.health.v1) |
| `cambrian_agent_sdk/tools.py` | `@tool` and `@capability` decorators, `ToolSpec`, `ToolRegistry`, `derive_schema`, `validate_args` (no `exec`/`eval`) |
| `cambrian_agent_sdk/types.py` | `AgentResult`, `AgentTask`, `Capability`, `Payload`, `ContextRef`, `ContextNode`, `ExecuteRequest`/`Response`, `ProposalRequest`/`Response`, `VerifyRequest`/`Response`, `SubGoal`, `yield_subgoal`, `ScopeConfig` |
| `cambrian_agent_sdk/working_memory.py` | `WorkingMemory` (bounded assembly, mandatory pins, relevance ranking), `ToolCard`, `TextEntry`, `Step`, `render_step_xml`, `resolve_content` (CID drill-down) |
| `cambrian_agent_sdk/_logging.py` | `SlogHandler` (slog-compatible JSON to stdout), `configure_logging`, `set_task_context` |
| `cambrian_agent_sdk/_proto/` | Vendored gRPC stubs: `cambrian_pb2.py`, `cambrian_pb2_grpc.py`, `__init__.py` (the pinned proto contract) |

## Examples

`examples/benchmark_agent.py` is the one curated smoke test: a
generalist `CognitiveAgent` that proxies every task to the Substrate's
LLM gateway. It validates the Python SDK and the gRPC boundary; it is
not a production agent.

The `example-agents/` directory holds nine production-style references,
one per trait flavor:

- `analyst_agent.py`: `CognitiveAgent` for structured chain-of-thought analysis, comparison, and evaluation.
- `calculator_agent.py`: `CognitiveAgent` demonstrating the `@tool` registry with `add` / `subtract` / `multiply` / `divide`.
- `code_executor_agent.py`: `CognitiveAgent` that marshals Python code into the kernel's `execute_python` system tool (ADR-0039/0040).
- `code_generator_agent.py`: `CognitiveAgent` with a `generate_python_code` `@tool` for clean Python 3.
- `example_daemon_agent.py`: `DaemonAgent` emitting periodic heartbeat signals on a `SignalStream`.
- `pulse_agent.py`: `DaemonAgent` emitting an incrementing heartbeat with an ISO timestamp.
- `research_agent.py`: `CognitiveAgent` combining web search and read/write tools from the kernel's tool registry.
- `summariser_agent.py`: `CognitiveAgent` condensing long text into bullet-point summaries.
- `terminal_agent.py`: `CognitiveAgent` calling the kernel's `execute_command` system tool for shell work.

## Build & verify

```sh
uv sync            # or: pip install -e .
pytest
ruff check
mypy
```

`cambrian-agent-sdk` on PyPI, at `0.1.0`, under the Business Source
License 1.1 (matching the kernel license per ADR-0057 D6/D10). Python
`>=3.10` is required; runtime dependencies are `grpcio>=1.60.0`,
`grpcio-health-checking>=1.60.0`, and `protobuf>=4.25.0`.

## Known Gaps

**A reconnecting SignalStream used to start a SECOND daemon_loop — FIXED 2026-08-08.**
`start_daemon` created its supervised `daemon_loop` thread *inside* the reconnect loop, so
every `SignalStream` disconnect spawned another one while the previous thread kept running:
`should_continue` only flips at shutdown, so nothing ever stopped the old one. A flapping
stream therefore left N concurrent copies of the author's loop running for the life of the
process.

How it presented, and why it went unnoticed: the telegram ingress reconnected 6 times and
then logged 133 × `409 conflict — another getUpdates poller is running on this token`. The
"other poller" was itself, five times over. Telegram refuses concurrent `getUpdates` on one
token, so the 409 made the duplication VISIBLE — **a daemon without that server-side
interlock would instead have silently done every unit of work N times**, which is the worse
version of this bug and the reason it is fixed here rather than in the telegram agent. Any
daemon whose stream flapped was affected.

Fixed by hoisting the thread start above the reconnect loop. Safe because `daemon_loop`
never touches the channel — it hands work to `agent._signal_queue`, which `_request_iter`
drains on whichever stream is currently open, so signals queue across a disconnect and flush
on reconnect. Pinned by `tests/test_daemon.py::test_reconnecting_the_stream_does_not_start_a_second_daemon_loop`,
which is mutation-checked: with the start moved back inside the loop it fails `3 == 1` and
names `Thread-2/3/4 (run_supervised)`. Live-verified over an 8-minute window — 5 stream
reconnects per agent, **0** conflicts and **0** loop crashes, against 133 of each before.


- **Go API is unstable in v0.x.** Only the gRPC proto surface and the `pyproject.toml` config schema are held stable; the Go package API can change at any minor release (per ADR-0057 D8). The SDK pins to the vendored proto in `_proto/`, so a kernel-side proto bump needs re-vendoring and a new SDK release, not just a `pip install --upgrade`.
- **BSL 1.1, not OSI-open.** The Business Source License restricts production use; the four-years-after-change date is the eventual Apache 2.0 conversion. Treat the package as source-available.
- **Cross-run `LessonsLearned` write-back is open.** ADR-0052 ships the per-run reflection, but it lives in working memory only. Writing reflections back to LTM (the FORGE / `LessonsLearned` lane per ADR-0049) is the natural follow-up, and is a kernel change, not an SDK one.
- **The A2A server in `server.py` is minimal.** The current daemon runtime handles a `SignalStream` round-trip, but is not a full Agent2Agent protocol implementation. A canonical A2A stack would add an AgentCard, server-side skills index, and JSON-RPC translation. Deferred.
- **Proto is vendored, not generated.** Re-vendoring requires running `grpcio-tools` against the kernel's `.proto` and overwriting `_proto/`. There is no automatic check that the vendored stubs match the kernel's current proto; treat any drift as a build break.

## Terminology Glossary

- **Agent**: the abstract base class in `base.py`. Cannot be instantiated directly; the structural enforcement of ADR-0036 D1 is that authors must pick a trait.
- **CognitiveAgent**: an LLM-reasoning trait with `run()`, `think()`, memory, and `@tool` actions. The ReAct loop is wired through `react.run_think`; the default `run()` catches `ReActLoopError` and returns a typed `type="error"` result.
- **DeterministicAgent**: a scripted cell with a typed `run()` and no `think()`. It answers `RequestProposal` with a static 1.0-confidence / 5 ms proposal, which is what made a selection mechanism treat it as a tool rather than a reasoning capability. **As of 2026-08-07 the kernel never asks.** ADR-0100 P3 deleted the auction and its bid round, so nothing in the kernel calls `RequestProposal` any more — `runtime.proposal_bid` and `TraitServicer.RequestProposal` are still served because the RPC is still in the proto (deliberately unchanged, so the operator contract stays `0093`), but on a current kernel they are unreachable. What distinguishes a DeterministicAgent to the Dispatcher now is its declared capabilities in its manifest, not a bid.
- **DaemonAgent**: a long-running signal producer on `SignalStream` (not `AgentService`). The author overrides `daemon_loop()` and calls `send_signal()`; the runtime supervises crashes with exponential backoff capped at 30 seconds.
- **React**: shorthand for the reasoning pattern itself, not a class. The pattern lives in `react.run_think` and is invoked from `CognitiveAgent.think()`. The class hierarchy uses `CognitiveAgent`, not `React`, as the trait name.
- **ReAct**: the original Yao et al. 2022 paper ("ReAct: Synergizing Reasoning and Acting in Language Models"). Cambrian's loop generalises ReAct with the LRW gate, the v2 prompt layout, and verbal reflection.
- **LRW**: the Local Recurrent Workspace, the typed working memory plus recurrent reconciliation gate from ADR-0041. The LLM is a storage-less Central Executive; the workspace builds the buffers it directs.
- **RunGrantOverlay**: a kernel-side concept (the run-scoped grant stack from ADR-0046 D6). The SDK encounters it only as the authorisation outcome of `use_skill` for a system skill; the overlay is held by the kernel and never visible in the agent process.
- **local_skills**: SDK-local skill registration. Per ADR-0046 D5, an agent's local skills are always listed first, and a local skill with the same name as a system skill shadows it by lexical scoping. The kernel never sees local skills; the structural prioritisation is the SDK's job, with no central ranking.
- **SubstrateClient**: the SDK's gRPC client for Substrate-side RPCs: `generate` / `generate_stream` (managed LLM proxy), `get_context_node` / `put_context_node` (ContentStore), `execute_tool` (kernel system tools), `list_tools` / `list_skills`, `ask` (LTM), and `execute` (sub-goal delegation, refused for self).
- **ToolRegistry**: the bound `@tool` menu for one agent instance, built lazily by scanning the class MRO for `_cambrian_tool` markers. Inputs are validated against the auto-derived JSON Schema before the call, and a bad call returns a structured `{"error": ..., "tool": ...}` dict rather than raising.
- **ContentStore**: the kernel's CID-addressed content store. Used for heavy result offload (ADR-0041 D3) and large working-memory text blocks (R7). The agent addresses entries by CID; the kernel reads are session-gated.

## Cross-repo pointer

- [`../CONTEXT.md`](../CONTEXT.md): the monorepo map.
- [`../AGENTS.md`](../AGENTS.md): the four cross-repo invariants, including "gRPC proto is the only SDK contract".
- [`../cambrian-core/docs/adr/0036-trait-aligned-cognitive-agent-sdk.md`](../cambrian-core/docs/adr/0036-trait-aligned-cognitive-agent-sdk.md): the SDK's main ADR (trait taxonomy, single-threaded server, protocol invisibility, the `@tool` vs `@capability` split).
- [`../cambrian-core/docs/adr/0041-local-recurrent-workspace.md`](../cambrian-core/docs/adr/0041-local-recurrent-workspace.md): the LRW gate in `recurrence.py` and the typed working memory in `working_memory.py`.
- [`../cambrian-core/docs/adr/0046-agent-skills.md`](../cambrian-core/docs/adr/0046-agent-skills.md): the system vs agent-local skills split and the lexical-scoping rule.
- [`../cambrian-core/docs/adr/0052-verbal-self-reflection.md`](../cambrian-core/docs/adr/0052-verbal-self-reflection.md): the verbal self-reflection in `reflection.py`, on the first hard veto.
- [`./pyproject.toml`](./pyproject.toml): the package name (`cambrian-agent-sdk`), version (`0.1.0`), Python floor (`>=3.10`), and runtime dependencies.
- [`./AGENTS.md`](./AGENTS.md): the companion agent guide (hard rules, layout, build & verify).
