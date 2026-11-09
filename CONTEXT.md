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
- **What has been measured** — benchmark runs in `../cambrian-knowledge/benchmarks/`
  (`list_benchmarks` by `tag`/`adr`, or the knowledge-board Benchmarks page). Every benchmark
  run is recorded there, with a verdict — the procedure is root [`AGENTS.md`](../AGENTS.md) §0.5.
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

**`HTTPChatIngress` is deprecated (ADR-0142 K-5, 2026-11-06).** It is not the conversation
lane: a person talks to Cambrian through the operator plane's `OpenConversation` / `SendTurn`
on `OperatorConsole`, authenticated per principal through the kernel Authenticator. The class
has no authentication and sits outside the kernel's listener policy, so it now warns
(`DeprecationWarning`) on construction and refuses to bind a non-loopback `addr`
(`ValueError` from `listen()`, before anything binds; `localhost`, `127.0.0.0/8`, `::1` and an
empty host are accepted). It stays exported for the local airline benchmark harness
(`cambrian-premium/agents/airline_chat_ingress.py`) and is to be removed with it.


## Implementation Status

| Area | Source | Status |
| --- | --- | --- |
| Trait taxonomy (`Agent` abstract + `CognitiveAgent` / `DeterministicAgent` / `DaemonAgent`) | `base.py`; ADR-0036 D1 | Implemented. `DeterministicAgent` **deprecated 2026-10-31** (owner decision; `DeprecationWarning` on subclassing, unreleased — see `CHANGELOG.md`, T-20261031-sdk-deterministicagent-still-advertises-trait) |
| ReAct pattern: `run_think` driven by the cognitive trait's `think()` | `react.py` | Implemented |
| ADR-0145 D3 evidence re-prompt (2026-11-08, default off: `run_think(evidence_reprompt=)`, else `$CAMBRIAN_EVIDENCE_REPROMPT`, which the kernel sets from `execution.workers.evidence_reprompt`) | `react.py` `_no_evidence_note`, `prompts/react-nudges/v4.md` | Implemented. A plain-text reply in a run with no tool call, no memory hit and nothing seeded (upstream, workspace, seed recall, yield result; or the caller's `turn_evidence=True`) is re-prompted toward `delegation_tool` (arg or agent attribute, only when the agent can call it) or the next tool, not toward final_answer. Structural, never by reading the reply. Chat agent wiring: `T-20261108-chat-agent-pass-delegation-tool-and-turn`. Tests `tests/test_evidence_reprompt.py`. |
| Deterministic skill menu order (2026-11-08) | `profiles.py` `ProfileScopedSubstrate.list_skills` | The profile menu is sorted by name (kernel `InMemorySkillRegistry.All` too), so the system prompt is byte-stable across a run's steps (`T-20261108-system-prompt-skill-list-order-is-not-stable`). |
| Agent-pushed live step output (ADR-0139 amendment 2026-11-06): `SubstrateClient.step_output(lease)` (context manager, one background thread per writer) and `emit_step_output(lease, chunks, final=)` over the client-streaming `EmitStepOutput` RPC; per-lease seq (dedupe-safe re-send), failures recorded never raised; profile field `streams_output` (validator + manifest, parity fixture) so the kernel stops forwarding gateway deltas. SDK ReAct agents need nothing: the kernel forwards their gateway calls (native tool turns included) itself | `step_output.py`, `__init__.py`, `profiles.py`; README "Live step output"; `tests/test_step_output.py` | Implemented 2026-11-06 |
| Local Recurrent Workspace gate (recurrent reconciliation) | `recurrence.py`; ADR-0041 D4 | Implemented |
| Verbal self-reflection on the gate (pinned working-memory lesson) | `reflection.py`; ADR-0052 | Implemented (residual) |
| Prompt library (versioned prompt files, slot rendering, sent-text hash, shared lint; fragment sets) | `prompt_library.py`, `prompts/`; ADR-0139 C4/D2 | Implemented. Migrated: `reflection`, and (wave 1d, 2026-11-05) the ReAct prompts as fragment sets `react-constraints`, `react-anti-patterns`, `react-action-protocol`, `react-turn`, `react-nudges` (renders byte-identical to the pre-move text except two L7 lint fixes; `tests/test_react_prompt_golden.py`). `react-turn` v4 (2026-11-07, T-20261107-delegated-steps-name-the-conversation-owner-in) adds only the `acting-for` section: `react.with_acting_for` appends it after the step objective in the Task region when the kernel stamps `_acting_for` / `_acting_for_name` on the step handoff (the run principal the step's tools authenticate as); recall and tool ranking keep the bare objective. `react-turn` v5 (2026-11-08, T-20261107-planner-and-chat-agent-are-not-told-today-s) adds only `current-date`: `react.with_current_date` appends the kernel's date block (`_current_date`, cambrian-core `domain.Calendar`) after the objective in the Task region; the SDK never reads its own clock for it |
| Declarative agent profiles (agent stack v2: id, description, capabilities, skills, tool allowlist globs, model preference, trust tier, limits, role, optional example tasks) | `profiles.py`; decisions/agent-stack-v2-mcp-tools-and-profiles.md, ADR-0139 C2/C3 | Implemented 2026-11-05 (parts A and B, T-20261105-agent-stack-v2-worker-agent-runtime-plus): strict loader/validator (YAML subset shared with the prompt library), `ToolAllowlist`, `ProfileScopedSubstrate` (client-side menu filter). Run by `cambrian-core/agents/worker_agent.py`. The kernel registers one agent per profile and enforces `tools` as grants plus a ceiling (also under `tools_unrestricted`); its validator is a Go port held to this one by `tests/test_profile_parity.py` (fixture written by cambrian-core `internal/agentprofile`). `CAPABILITY_VOCABULARY` is now the starter profiles' tags. Optional `examples` (up to 5, 300 chars; 2026-11-05, ADR-0139 D4) go on the manifest and the kernel's signed A2A card |
| C1 managed generation: `system` blocks, `messages`, `response_schema`, `cache_ttl`, options with field presence (`temperature=0` honoured), `stop`, `strict`; `GenerateResult` (text, tool_calls, stop_reason, usage with cache read/write/known/estimated, ignored, model; unpacks as the old 3-tuple); typed status errors | `__init__.py` (`SubstrateClient.generate_request` / `generate` / `generate_stream` / `generate_with_tools`), `llm.py`, `errors.py`; ADR-0139 C1, wave 1d | Implemented 2026-11-05. Pre-C1 kernels: the stream also carries the flattened prompt in the deprecated `prompt` field; a reply without `usage` sets `legacy_kernel` (and, on GenerateWithTools, `ignored += ["system"]`) |
| ReAct prompt split: cached stable system tier (role, constraints, anti-patterns, action protocol + menus, per-turn contract) + volatile user turn (task, trajectory marked as data, per-turn notes); `cache_ttl` short by default (`CognitiveAgent.cache_ttl`, `CAMBRIAN_AGENT_CACHE_TTL`); degrade to the legacy single prompt on `ignored: system` / `InvalidRequest` / a pre-C1 substrate; debug logs of prompt ids, versions, static hashes and the system sha256 per call | `helpers.py` (`build_prompt_parts`, `PromptParts`), `react.py` (`run_think`, `_generate_text_turn`); ADR-0139 D2, wave 1d | Implemented 2026-11-05 |
| In-process `@tool` registration, schema auto-derivation, `ToolRegistry` | `tools.py`; ADR-0036 D4 | Implemented |
| Typed working memory (`ToolCard`, `TextEntry`, `Step`, bounded assembly) | `working_memory.py`, `types.py`; ADR-0041 D1-D3 | Superseded as the default by the D3 context engine (2026-11-05); reachable for one release behind `CAMBRIAN_CONTEXT_ENGINE=legacy` (`LegacyWorkingMemoryEngine`) |
| Context engine: pluggable `ContextEngine` + `DefaultContextEngine`. Token-budgeted (chars/4) regions in the user turn: D14 memory-notes snapshot, pinned Instructions (skills, grant notes, reflections; own budget, overflow refused), Evidence (`<documents>`: workspace, seed recall, delegated), Trajectory (ONE step per tool call, output in `<evidence>`), Task last. Two-phase compaction on both paths (phase 1 placeholder + cid, uncapped re-open; phase 2 structured summary via `react-compaction`, loud fallback to phase 1). Every native control verb is a tool (`memory_query`, `use_skill`, `find_tools`, `find_skills`, `resolve_cid`, `yield_subgoal`, `final_answer`); plain native text is re-prompted once, then accepted and logged. Read-only system tools of one native turn run in parallel (max 4). `ContextConfig` budgets: `run_think(context_config=)` > `agent.context_config` > `CAMBRIAN_CONTEXT_*` | `context_engine.py`, `react.py`, prompts `react-* v2` (constraints/turn v3), `react-tools`, `react-compaction`; ADR-0139 D3/D14 | Implemented 2026-11-05 (T-20261105-adr-0139-d3-token-budgeted-context-engine-with). Not measured on a live model yet |
| gRPC clients (`SubstrateClient`, `MemoryClient`, `ArtifactManager`) | `__init__.py` (`SubstrateClient`), `clients.py` (memory, artifacts) | Implemented |
| Structured Execute result (ADR-0145 D8): `SubstrateClient.execute()` returns `ExecuteResult`, a `str` subclass that IS the report text (every existing caller unchanged; `.text` is the plain text) and carries the kernel's `_execute_result` metadata as attributes: `run_id`, `session_id`, `actions` (plan action ledger), `steps` (every step of the final plan: status, ledger `ref`, preview, bytes, fan_out_item), `step_refs`, `partial`, `timed_out`, `timeout_cause`, `failed_step`, `error`, `ledger`, `structured`, `metadata`, `as_dict()`. An older kernel without the key still yields `actions` / `partial` / `timed_out` from `_plan_actions` / `_partial_plan` / `_plan_timed_out` | `execute_result.py`, `__init__.py`; `tests/test_execute_result.py` | Implemented 2026-11-08 (T-20261108-execute-returns-a-structured-result-the-sdk) |
| Fan-out item object (ADR-0145 D6.3): `react.with_fan_out_item` appends the `react-turn` v6 `fan-out-item` block (the kernel's `_fan_out_item`, the whole source element a fan-out child runs for) after the objective in the Task region | `react.py`, `prompts/react-turn/v6.md` | Implemented 2026-11-08 |
| Answer envelope (ADR-0145 D6, phase 2): when the kernel stamps `_answer_items` on a step (`execution.plan.checklist_enabled`), `react.with_answer_items` renders the `react-turn` v7 `answer-items` block (and `step-inputs` for `_step_inputs`, `envelope-repair` for the one repair turn's `_envelope_violations`) in the Task region; the native menu offers `final_answer` with `item_results` / `evidence_refs` (`_FINAL_ANSWER_ENVELOPE_TOOL_DEF`, only then); the deliverable is `build_answer_envelope` = `{answer_text, item_results: [{item, status, values: [{name, type, value, evidence_ref}], reason}], evidence_refs}`, validated by the kernel against the ledger. A kernel result carrying both `result_json` and `result_cid` is an inline result plus its ledger reference (`_ledger_ref`), so values can be cited. `ExecuteResult.requirements` carries the kernel-decided checklist coverage. Without the keys nothing changes | `react.py`, `execute_result.py`, `prompts/react-turn/v7.md`; `tests/test_answer_envelope.py` | Implemented 2026-11-08 (T-20261108-adr-0145-phase-2-requirement-checklist-with); not yet run on a live kernel |
| Repair turn and value types (ADR-0145 D6/D14.6, `react-turn` v8, 2026-11-09): `react.with_answer_items` also renders the kernel's `_repair_prior_calls` (`REPAIR_PRIOR_CALLS_KEY`, section `repair-prior-calls`: the step's earlier tool results by ledger reference and its executed writes); `envelope-repair` says correct the envelope, do not redo the work; `answer-items` says a `ref` is a ledger id only, a system's record identifier is an `id`, and a document item's full text is the answer (nothing saved outside unless a write item asks). Recalled memory: `working_memory.memory_provenance_attrs` adds `document=`, `section=`, `ledger=` when the kernel sends `_source_document`, `_section_path`, `_ledger_ref` (every recall is a kernel ledger entry) | `react.py`, `working_memory.py`, `prompts/react-turn/v8.md`; `tests/test_answer_envelope.py`, `tests/test_seed_recall_rendering.py` | Implemented 2026-11-09 (T-20261109-d6-envelope-repair-turn-re-runs-the-step-s, T-20261109-d6-envelope-keeps-items-open-when-the-model, T-20261109-chat-agent-answers-a-published-rule); not yet run live |
| Run-ledger tools (ADR-0145 D1): `ledger_get` / `ledger_find` are kernel-served tools on the `ListTools` menu when the kernel's `execution.ledger.enabled` is on; the SDK calls them like any system tool through `ExecuteTool`, and `resolve_cid` reads `led_…` ledger references through `GetContextNode` unchanged. No SDK code is ledger-specific | kernel `internal/ledger` | Implemented 2026-11-08 (kernel side) |
| gRPC server base for `AgentService` and health | `runtime.py`, `server.py`; ADR-0036 D2 | Implemented |
| Daemon-agent path: `SignalStream` runtime + supervised `daemon_loop` | `daemon.py`; ADR-0033, ADR-0036 issue 0036-06 | Implemented. **Fixed 2026-08-08: one loop per daemon, not one per reconnect** — see Known Gaps. |
| Agent skills: lexical scoping of local vs system skills | ADR-0046 | Proposed |
| Loaded skills pinned in full (never offloaded/evicted; outside the 10-entry cap), 5-load + 32k-char skill budget (overflow refused, never truncated), native `use_skill`/`find_skills` tools answered in their tool turn | `working_memory.py` (`WorkingMemory.add_skill`), `react.py`; ADR-0139 Phase 0 | Implemented 2026-11-04 |
| A loaded system skill's granted tools join the callable menu (native defs append-only + prose menu, re-merged after a latch); local skill menu lines use `derive_summary` (port of the kernel deriver, parity-tested) | `react.py` (`_skill_granted_tools`, `derive_summary`); ADR-0046 D6, ADR-0139 Phase 0 | Implemented 2026-11-04 (overlay-only tools lack a schema until T-20261104-kernel-listtools-names-ignores-run-scoped-skill) |
| Mode-specific prompt text (native vs text protocol) and constraints generated from the enforced budgets; `CognitiveAgent.max_tokens=None` defers to run_think's 4096 | `helpers.py` (`default_constraints`, `default_anti_patterns`), `react.py`, `base.py`; ADR-0139 Phase 0 | Implemented 2026-11-04 |
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
(deprecated 2026-10-31; new agents use `CognitiveAgent` with an overridden `run()`)
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
| `cambrian_agent_sdk/run_trace.py` | ADR-0145 D10 agent-side run trace: when the kernel sets `CAMBRIAN_TRACE_DIR` (`execution.trace.enabled`), `react.py` records every model call (native, text and legacy paths: system, messages, tools offered, response, usage) as a `chat` span and every tool call (local and kernel, single and parallel: arguments, result, call id) as an `execute_tool` span, OTel GenAI shape, one JSONL file per agent process, bounded by `CAMBRIAN_TRACE_MAX_CHARS`. Joins the kernel's file on `cambrian.lease.id` / `cambrian.step.task_id`. No-op without the env var; never raises into the loop |
| `cambrian_agent_sdk/execute_result.py` | `ExecuteResult` (ADR-0145 D8): the delegated Execute's report text as a `str`, plus the structured result decoded from the `_execute_result` response metadata (run id, action ledger, step ledger refs, partial / timeout flags) |
| `cambrian_agent_sdk/clients.py` | `MemoryClient` (recall lanes + remember, both with ADR-0134 event time; `TIME_PRECISIONS` / `TIME_SOURCES` vocabulary — `recall` facts, `recall_actions`; two **deprecated** shims: `recall_entity` (the kernel removed the entity lane on 2026-08-14 and serves `x-lane=entity` from the fact lane) and `recall_precedents` (precedents were removed 2026-11-05, `decisions/scenes-cancelled.md`; it warns and returns `[]` without calling the kernel)), `ArtifactManager` (save/get/list_from_step), read-only `WorkingMemory` view, plus `InvalidTagError`, `ArtifactNotFound`, `SelfDelegationError` |
| `cambrian_agent_sdk/daemon.py` | `start_daemon` + `run_supervised`: the `SignalStream` runtime with crash supervision (capped exponential backoff, 30s cap). The supervised loop is started ONCE, **outside** the reconnect loop — that placement is load-bearing, see Known Gaps |
| `cambrian_agent_sdk/errors.py` | `BudgetExceededError`, the typed signal that a task refused on budget grounds; `ToolCallingUnsupported` (FAILED_PRECONDITION / UNIMPLEMENTED on GenerateWithTools: fall back to the text protocol); the C1 generation errors `GenerateError` > `InvalidRequest` (INVALID_ARGUMENT), `SchemaValidationFailed` (ABORTED), `GenerationResourceExhausted` (RESOURCE_EXHAUSTED), `ModelUnavailable` (UNAVAILABLE), all also `grpc.RpcError` |
| `cambrian_agent_sdk/llm.py` | C1 request/result data types (`SystemBlock`, `ResponseSchema`, `GenerateUsage`, `GenerateResult`, `CACHE_TTL_SHORT`/`LONG`) and the shaping helpers (`coerce_*`, `flatten_prompt`); no gRPC import |
| `cambrian_agent_sdk/helpers.py` | `build_prompt` (v2 cache-friendly single string, kept for callers), `build_prompt_parts` / `PromptParts` (the C1 split: cached system tier + user turn, `legacy_prompt()` flattening), `prompt_versions`, `extract_code_block`, `find_step_ref`, `default_constraints` / `default_anti_patterns` (generated from the loop's effective budgets, per protocol; `DEFAULT_CONSTRAINTS` / `DEFAULT_ANTI_PATTERNS` remain as computed text-protocol aliases), XML escape |
| `cambrian_agent_sdk/host.py` | `host_facts` / `host_block` / `goos`: the host the agent is running ON (`os`, `home`, `desktop`, `cwd`, `path_separator`) plus the path rules that make those values usable, appended once to the ReAct prompt's loop-invariant prefix. **This is the executing half of a split with the kernel's planner prompt:** the planner is told only the OS and told explicitly that paths are not its job, because it cannot see the host and because its plans are replayed as `<PlanLTM>` examples for later plans (an absolute path with a username in it is wrong under another account). Three rules to preserve: `goos()` uses **Go's** `runtime.GOOS` spelling so plan and agent name the machine identically; a fact that cannot be determined is **omitted rather than guessed** (the model treats what it is given as authoritative); and `host_facts` is cached so the prompt prefix stays byte-identical across turns. Desktop resolution is redirection-aware — Windows reads Explorer's Shell Folders registry value (OneDrive Known-Folder redirection means `home/Desktop` can exist as a shadow folder while the real desktop is under `OneDrive`), Linux reads `XDG_DESKTOP_DIR` (a localised desktop is `~/Escritorio`). Origin: ADR-0100 P3 residual #3 — the Scout carried the only "never write `~/Desktop`" rule and took it with it when retired |
| `cambrian_agent_sdk/react.py` | `run_think` (the ReAct loop), `ReActLoopError`, prompt/action-protocol composition, all per-action handlers. The retrieval vocabulary is one action, `memory_query` (facts), on the memory budget. `recall_precedents` was removed from the vocabulary on 2026-11-05 with the precedent lane (`decisions/scenes-cancelled.md`) |
| `cambrian_agent_sdk/recurrence.py` | `count_failed_duplicates` and `count_successful_duplicates`: the local recurrent reconciliation gate |
| `cambrian_agent_sdk/reflection.py` | `build_reflection_prompt`: fills the slots of the library prompt `reflection` for the verbal self-reflection (ADR-0052) |
| `cambrian_agent_sdk/prompt_library.py` + `cambrian_agent_sdk/prompts/<id>/v<N>.md` | SDK prompt library (ADR-0139 C4/D2, owner Q6): packaged prompts read with `importlib.resources`; `get` / `get_version` / `Prompt.render` with the same rules, YAML-subset frontmatter and sent-text `static_hash` as the kernel's `cambrian-core/prompts` (spec: `cambrian-core/prompts/SPEC.md`). Held to the Go loader by `tests/test_prompt_hash_parity.py`; lint runner `tests/test_prompt_lint.py`. Migrated: `reflection` and the ReAct prompts (`react-*`, fragment sets: one file, `<!-- section: name -->` delimiters, `fragment(id, section).render(values)` fills only that section's slots; one static hash per file) |
| `cambrian_agent_sdk/composition.py` | ADR-0145 D12 (2026-11-08): `Composition` (the kernel's `_composition` handoff value: runtime, professions, skills, tools, effects, brief, ledger refs) and `ComposedScopedSubstrate`, the composed runtime's substrate view: presents the step lease on menu / skill calls (the kernel resolves and enforces the composition from it), keeps the skill menu to the composition in name order, refuses locally a call outside the composition and the kernel-served menu, offers nothing when no composition is bound. Used by `cambrian-core/agents/worker_agent.py --composed`. |
| `cambrian_agent_sdk/profiles.py` | `Profile`, `parse_profile` / `load_profile` / `load_profiles`, `skill_index`, `ToolAllowlist`, `ProfileScopedSubstrate`, `CAPABILITY_VOCABULARY`; tests `tests/test_profiles.py` |
| `cambrian_agent_sdk/runtime.py` | `TraitServicer`, `start_agent_server`, `_task_from_handoff`, `_coerce_agent_result`, the single-threaded gRPC runtime (ADR-0036 D2) |
| `cambrian_agent_sdk/server.py` | gRPC server lifecycle, `_parse_listen_address`, `_parse_substrate_addr`, `is_daemon_mode`, `_wire_health` (grpc.health.v1) |
| `cambrian_agent_sdk/tools.py` | `@tool` and `@capability` decorators, `ToolSpec`, `ToolRegistry`, `derive_schema`, `validate_args` (no `exec`/`eval`) |
| `cambrian_agent_sdk/types.py` | `AgentResult`, `AgentTask`, `Capability`, `Payload`, `ContextRef`, `ContextNode`, `ExecuteRequest`/`Response`, `ProposalRequest`/`Response`, `VerifyRequest`/`Response`, `SubGoal`, `yield_subgoal`, `ScopeConfig` |
| `cambrian_agent_sdk/working_memory.py` | `WorkingMemory` (bounded assembly, mandatory pins, relevance ranking; the legacy engine), `ToolCard`, `TextEntry`, `Step`, `render_step_xml`, `resolve_content` (CID drill-down) |
| `cambrian_agent_sdk/__init__.py` (`SubstrateClient`) kernel verbs | ADR-0139 D14/D13 (2026-11-05): `get_memory_notes` (GetMemoryNotes; documents personal first), `propose_memory_note` (ProposeMemoryNote), `propose_skill` (ProposeSkill). Lease in `lease_id` and `x-lease-id`; UNIMPLEMENTED → None; other RPC errors → `{error, code}`; refusals (`accepted=False`) are data. Loop verbs in `react.py` (`_offered_kernel_verbs`, `_run_kernel_verb`), texts in prompt `react-control-verbs` v2 (v2 = v1 without the spawn-subagent sections); tests `tests/test_kernel_verbs.py`. `spawn_subagent` was removed 2026-11-09 (ADR-0145 D14) |
| `cambrian_agent_sdk/step_dependencies.py` | Renders a dependent plan step's upstream results (kernel `domain.StepDependency` manifest in `task.context['_step_dependencies']`: step, objective, status, bounded content, cid) as one `<upstream_results kind='data'>` Evidence region; `react.run_think` drops the step_result refs it covers from the `<workspace>` block (T-20261105-a-planner-thought-step-that-depends-on-two); entries with `prior_plan` (results carried over a replan) render as `prior_step_N` from `previous_plan_N`, a `failed` entry carries its `error` (T-20261106-an-injected-injectcorrection-step-cannot-see); tests `tests/test_step_dependencies.py` |
| `cambrian_agent_sdk/context_engine.py` | ADR-0139 D3: `ContextEngine` (protocol), `DefaultContextEngine` (regions, compaction, local cid store, `events`, `archive`), `LegacyWorkingMemoryEngine`, `ContextConfig`, `estimate_tokens`, `make_engine`, `memory_notes_text`; tests `tests/test_context_engine.py` |
| `cambrian_agent_sdk/_logging.py` | `SlogHandler` (slog-compatible JSON to stdout), `configure_logging`, `set_task_context` |
| `cambrian_agent_sdk/_proto/` | Vendored gRPC stubs: `cambrian_pb2.py`, `cambrian_pb2_grpc.py`, `__init__.py` (the pinned proto contract) |

## Examples

`examples/tool_tutorial_agent.py` is the `@tool` tutorial: a four-tool calculator
`CognitiveAgent` showing the `@tool` registry and the `think()` loop. It is a tutorial,
not a shipped agent.

The `example-agents/` directory holds two references:

- `pulse_agent.py`: `DaemonAgent` emitting an incrementing heartbeat with an ISO timestamp.
- `example_ingress_agent.py`: `IngressAgent` reference (see "Ingress" above).

Removed 2026-11-05 (agent stack v2, decisions/agent-stack-v2-mcp-tools-and-profiles.md):
the seven thin prompt-only agents (analyst, calculator, code_executor, code_generator,
research, summariser, terminal), the duplicate `example_daemon_agent.py`, and
`examples/benchmark_agent.py` (nothing referenced it). Specialists are now declarative
profiles (`cambrian_agent_sdk/profiles.py`) run by `cambrian-core/agents/worker_agent.py`.

## Build & verify

```sh
uv sync            # or: pip install -e .
pytest
ruff check
mypy
```

`cambrian-agent-sdk` on PyPI, at `0.1.5` (`pyproject.toml`), under the
Apache License 2.0 (`license = "Apache-2.0"`, `LICENSE`), per ADR-0057 D10 as
amended 2026-10-31. The kernel is Elastic License 2.0; the SDK is deliberately the
permissive one so third-party agents can depend on it freely. Python
`>=3.10` is required; runtime dependencies are `grpcio>=1.60.0`,
`grpcio-health-checking>=1.60.0`, and `protobuf>=4.25.0`.

## Known Gaps

- **Kernel-surface verbs (ADR-0139 D14/D13, 2026-11-05).** `propose_memory_note` and `propose_skill` (and `schedule_task` / `manage_schedules` / `propose_knowledge`) are native + text verbs, offered per agent attribute (`allow_memory_note_writes`, `allow_skill_proposals`, ...; off on `CognitiveAgent`) and only when the substrate serves them (notes: only after GetMemoryNotes answered). After UNIMPLEMENTED a verb answers "not available" for the rest of the run (its tool definition stays, for cache stability). Refusals come back to the model as data (`<observation verb=... kind='data'>`), never as success.
- **spawn_subagent removed (ADR-0145 D14, owner decision 2026-11-09).** The kernel no longer serves SpawnSubagent (UNIMPLEMENTED; the proto messages stay for wire compatibility, so the vendored stubs still carry them), sets no `_subagent*` handoff keys and has no child leases. The SDK deleted the verb, `SubstrateClient.spawn_subagent`, `allow_subagents` / `run_think(allow_spawn_subagent=)`, the subagent-child shaping in `run_think` (toolset menu filter, tool-round cap, fork summary / refs / output schema from `_subagent_*` keys, child-only skip of seed recall and notes), `ContextConfig.subagent_seed_recall` / `subagent_memory_notes` and the engines' `fork_summary`. A model that still emits `spawn_subagent` gets the unknown-action handling. A worker that needs more work done hands it to the planner through its own local `@tool delegate_to_planner` (cambrian-core/agents/worker_agent.py), not an SDK kernel verb: `_CONTROL_ACTION_NAMES` would shadow the chat agent's local tool of the same name.
- **Step lease on every menu fetch (ADR-0145 D14).** `_list_system_tools` (run-start menu, latch refetch, `find_tools`) and `_describe_tool` pass `session_token_id=task.session_token_id` to `substrate.list_tools`, so the kernel resolves the principal the step runs for and the menu matches what ExecuteTool allows; a substrate without the kwarg falls back (TypeError). Tests `tests/test_step_lease_and_denial.py`.
- **Repeated identical denial stops the loop (ADR-0145 D14.5).** `_repeated_denial` counts kernel refusals (`denied: <reason>`) per (tool, canonical JSON args, reason), with the kernel's " (This identical call was already refused in this step" suffix stripped; the second identical refusal ends the run with an `error` AgentResult carrying the reason (`react_repeated_denial_stop`), without another model round. Also since 2026-11-09: a run ended because the model kept querying memory past its budget is an `error` (with what was gathered), no longer a `text` result read as the finished answer.

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
- **Licence: Apache-2.0** (ADR-0057 D10, amended 2026-10-31; the original BSL 1.1 plan was dropped). OSI-open, unlike the ELv2 kernel.
- **Cross-run `LessonsLearned` write-back is open.** ADR-0052 ships the per-run reflection, but it lives in working memory only. Writing reflections back to LTM (the FORGE / `LessonsLearned` lane per ADR-0049) is the natural follow-up, and is a kernel change, not an SDK one.
- **The A2A server in `server.py` is minimal.** The current daemon runtime handles a `SignalStream` round-trip, but is not a full Agent2Agent protocol implementation. A canonical A2A stack would add an AgentCard, server-side skills index, and JSON-RPC translation. Deferred.
- **`DeterministicAgent` is deprecated but not removed**, and three cambrian-core system agents (`agents/system/{docling,kg_extractor,reranker}_agent`) still subclass it, so they print the `DeprecationWarning` at boot once they run on an SDK containing it. Migrating them flips their trait from `"tool"` to `"cognitive"` (interview / verifier-pool eligibility), so it is an owner decision — T-20261031-core-system-agents-docling-kg-extractor.
- **Proto is vendored, not generated.** Re-vendoring requires running `grpcio-tools` against the kernel's `.proto` and overwriting `_proto/`. There is no automatic check that the vendored stubs match the kernel's current proto; treat any drift as a build break. Last re-vendored 2026-11-01 at kernel contract 0099 with grpcio-tools 1.81.1 (`python -m grpc_tools.protoc -I ../cambrian-core/api/proto --python_out=_proto --grpc_python_out=_proto cambrian.proto`, then `import cambrian_pb2` → `from . import cambrian_pb2` in the grpc file): the retired watch-CRUD RPCs are gone, and `IngestMemoryRequest` / `MemoryRequest` / `MemoryResult` carry the ADR-0134 event-time fields. `MemoryClient.remember()` exposes them as keyword-only `occurred_at` / `parts=[(text, occurred_at)]` / `time_precision`, `recall()` as `reference_time` / `event_after` / `event_before` (timezone-aware datetimes only; a naive one raises `ValueError`, owner decision Q3), and every recall result dict carries `event_time` / `event_time_end` (aware UTC or None), `time_source`, `time_precision`. `CambrianAgent.ask()` still returns the old three-key dicts. Re-vendored again 2026-11-03 (same command and toolchain, LF) for ADR-0136 Amendment A3: `IngestMemoryRequest.source_kind`/`title`/`author` (fields 9-11) and `MemoryPart.speaker` (3), exposed by `remember()` as keyword-only `source_kind` / `title` / `author` and an optional third part element `(text, occurred_at, speaker)`. Descriptive only: the kernel keeps the authenticated agent as writer of record and never lets a speaker change the stored text. Re-vendored again 2026-11-05 (same command and toolchain) for ADR-0139 wave 1c: `GenerateStreamRequest`/`GenerateWithToolsRequest` carry `system` (`SystemBlock`), `response_schema` (`ResponseSchema`), `cache_ttl` (`CacheTTL`), and `GenerateStreamRequest.messages`; `GenerateOptions.max_tokens`/`temperature` have field presence plus `strict`; the final `GenerateChunk` and `GenerateWithToolsResponse` carry `ignored`, `usage` (`GenerateUsage`), `stop_reason`, `model`. The SDK sends and reads them since wave 1d (2026-11-05). Re-vendored again 2026-11-06 (same command and toolchain) for the ADR-0139 amendment (live step-output stream): the client-streaming `EmitStepOutput` RPC with `StepOutputChunk` / `StepOutputKind` / `EmitStepOutputAck`. Re-vendored again 2026-11-06 (same command, grpcio-tools 1.81.1) for principal-owned recurring tasks (capability `scheduled-tasks`): `ProposeScheduledTask` / `ListScheduledTasks` / `ManageScheduledTask`, exposed as `SubstrateClient.propose_scheduled_task` / `list_scheduled_tasks` / `manage_scheduled_task` (no owner parameter: the kernel takes it from the lease) and as the kernel verbs `schedule_task` / `manage_schedules` (react-control-verbs v1 sections, native + text; offered only with `allow_scheduled_tasks`, the substrate methods and a lease; disabled for the run on UNIMPLEMENTED; `manage_schedules` takes `op` because `action` names the verb in the text protocol). Re-vendored again 2026-11-06 (same command, grpcio-tools 1.81.1) for ADR-0142 D11: `ApprovalRequest` gains `rule` / `beneficiary` / `action_class` (fields 6-8, additive; the WatchApprovals stream now carries only items the caller may answer). Re-vendored again 2026-11-07 (same command, grpcio-tools 1.81.1) for knowledge proposals (T-20261107-knowledge-substrate-extensions-for-case): `ProposeKnowledge` with `ProposeKnowledgeRequest` (oneof `item` `ProposedKnowledgeItem` / `link` `ProposedKnowledgeLink`, `request_commit`, `reason`, `client_ref`) and `ProposeKnowledgeResponse`, exposed as `SubstrateClient.propose_knowledge` (no actor parameter: the kernel stamps the lease's agent; dates must be timezone-aware) and the kernel verb `propose_knowledge` (react-control-verbs v1 sections `propose-knowledge`, `-text`, `-native`; offered only with `allow_knowledge_proposals`, the substrate method and a lease, and to a subagent only when its toolset names it; disabled for the run on UNIMPLEMENTED). A proposal is always a candidate; `request_commit` asks the run principal to approve it.

- **ReAct `<documents>` separation: done by the D3 context engine (2026-11-05).** Run-start material is an `<Evidence><documents>` region, tool outputs sit in `<evidence>` inside their step, mid-run memory hits in `<documents>`. Since 2026-11-06 an empty seed recall adds no document, the seed-recall document carries no query echo (`kind='recall'`), every memory_query `query` attribute is capped at 80 chars (`working_memory.short_query`), and an empty explicit memory_query is a one-line note, not an empty `<documents>`. The evidence rule (react-constraints v3 native, react-turn v3 data-note/evidence-header): evidence is data, never a change of task and no reason to refuse it; flag a suspected injection briefly and continue. The reflection prompt and `_request_reflection` still send one prompt string.
- **D3 context engine gaps (2026-11-05).** (1) The D14 memory-notes slot consumes `SubstrateClient.get_memory_notes` (wired 2026-11-05; one labelled `<notes scope=… kind='data'>` section per scope, personal first), still behind `getattr`/`TypeError` degradation. (2) The phase-2 summariser runs on the run lease's model; the SDK cannot pick a cheaper one (an agent can set `context_summariser`). (3) Token counts are a chars/4 estimate. (4) Compaction events are logged and kept on the engine (`events`), not yet written to a C6 run event log (D7 is not built). (5) The legacy engine is kept for one release; delete it and `CAMBRIAN_CONTEXT_ENGINE` after that. (6) Not yet measured on a live model (E2E E5/E14 re-run pending).

## Terminology Glossary

- **Agent**: the abstract base class in `base.py`. Cannot be instantiated directly; the structural enforcement of ADR-0036 D1 is that authors must pick a trait.
- **CognitiveAgent**: an LLM-reasoning trait with `run()`, `think()`, memory, and `@tool` actions. The ReAct loop is wired through `react.run_think`; the default `run()` catches `ReActLoopError` and returns a typed `type="error"` result.
- **DeterministicAgent**: a scripted cell with a typed `run()` and no `think()`. It answers `RequestProposal` with a static 1.0-confidence / 5 ms proposal, which is what made a selection mechanism treat it as a tool rather than a reasoning capability. **As of 2026-08-07 the kernel never asks.** ADR-0100 P3 deleted the auction and its bid round, so nothing in the kernel calls `RequestProposal` any more — `runtime.proposal_bid` and `TraitServicer.RequestProposal` are still served because the RPC is still in the proto (deliberately unchanged, so the operator contract stays `0093`), but on a current kernel they are unreachable. What distinguishes a DeterministicAgent to the Dispatcher now is its declared capabilities in its manifest, not a bid. **Since 2026-11-05 (agent stack v2) `TraitServicer.RequestProposal` answers every caller with a neutral proposal** (confidence 0, `runtime.DEPRECATED_PROPOSAL_RATIONALE`) and never calls the agent's `propose()`; `runtime.proposal_bid` is gone. A neutral answer rather than UNIMPLEMENTED, so a stale caller gets a well-formed proposal that attracts nothing and author code cannot crash the servicer. **Deprecated 2026-10-31** (owner decision, T-20261031-sdk-deterministicagent-still-advertises-trait): defining a subclass emits a `DeprecationWarning` once per subclass (`DeterministicAgent.__init_subclass__`, `stacklevel=3` so it is attributed to the author's module — an agent run as `python agent.py` sees it under the default filter) naming the replacement: `CognitiveAgent` with an overridden `run()`, or a kernel-owned tool via `ExecuteTool` (ADR-0039 A1.3). Behaviour is otherwise unchanged (`trait = "tool"`; `propose()` still returns the static bid when called directly, but the RPC no longer calls it); the class goes when the kernel drops `TraitTool`. Three cambrian-core system agents (docling, kg_extractor, reranker) still subclass it; migrating them changes their trait and is an owner decision — T-20261031-core-system-agents-docling-kg-extractor.
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
