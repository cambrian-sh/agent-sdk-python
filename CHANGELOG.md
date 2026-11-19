# Changelog

All notable changes to `cambrian-agent-sdk`. Versions before this file existed are
described by their git history.

## Unreleased

### Changed

- **Vendored `cambrian_pb2` stubs match kernel contract 0111** (ENG-2143): additive. A large
  `IngestMemory` is accepted at once and embedded in the background; the response gains
  `document_id` and `status` (`IngestStatus`: state `accepted` | `embedding` | `searchable` |
  `failed`, chunks embedded of total, error), and the new `GetIngestStatus` RPC reads a
  document's progress (writer only) for a caller that needs read-your-writes. No SDK API
  change: `remember()` returns the entity id as before.

## 0.1.8 (2026-11-18)

Ships heartbeat liveness (ADR-0070 D3): a kernel-spawned daemon or pool worker that stays
alive but can serve nobody is now detected and replaced by the kernel. Needs a kernel that
passes `--heartbeat-interval-ms` (cambrian-core with ENG-1348); against an older kernel
nothing changes. Also fixes the skill menu breaking the prompt cache across chat turns
(ENG-126), a pool worker whose class declares a different `agent_id` being refused by the
kernel (ENG-2129), and a worker's gateway failure reaching the kernel as an opaque INTERNAL
instead of the gateway's status code (ENG-2125); and the reasoning loop now recovers once
from a context overflow by shrinking the newest oversized tool result (ENG-2128). The vendored
`cambrian_pb2` stubs match kernel contract 0110 (ENG-2147).

### Added

- **Heartbeat liveness** (ENG-1348, ADR-0070 D3). New `cambrian_agent_sdk.liveness`. When the
  kernel launches a long-lived process with `--heartbeat-interval-ms N`, the SDK sends a
  heartbeat on the process's SignalStream every N ms with no author code: a `DaemonAgent` or
  `IngressAgent` interleaves it on its open signal stream, and an AgentService worker (a chat
  pool worker) opens a heartbeat-only stream beside its server. A beat is WITHHELD while an
  `Execute` its caller abandoned (cancelled or past its deadline) is still running: the server
  is single-threaded, so such a worker can serve nobody, and the kernel kills it after N missed
  beats and restarts it through its restart policy. A long call whose caller is still waiting
  keeps beating (`busy`). Heartbeats are signals with `_signal_type: _heartbeat`; the kernel
  consumes them before routing, so they never reach a pipeline. `start_agent_server` takes
  `heartbeat_stream=False` for a process whose own signal stream carries the beats.

- **The reasoning loop recovers once from a context overflow** (ENG-2128). When a model call
  is refused `OUT_OF_RANGE` (`ContextOverflow`: the kernel's pre-send window check or the
  provider's own "prompt is too long"), `think()` replaces the newest oversized tool result
  in the conversation with the placeholder compaction already uses (its run-ledger id,
  readable with `ledger_get`, or else its cid) and retries that call once. If the retry
  overflows too, or there is no tool result to shrink, the step fails as before. The system
  tier, the user's turn and pinned skill turns are never shrunk. Each shrink is logged
  (`react_context_overflow_shrunk_retrying`), recorded as an `overflow_shrink` context-engine
  event and written as a `context.overflow_shrink` run-trace span. New
  `DefaultContextEngine.shrink_for_overflow`.

### Changed

- **Token estimate is content-aware** (ENG-1050). `context_engine.estimate_tokens` counts
  high-entropy runs (base64, hex: at least 64 characters mixing two of lower case, upper case
  and digits) at 1.3 characters a token instead of 4, the same rule as the kernel's pre-send
  context-window check. len/4 under-counted base64 2-3x, so compaction could report a prompt
  under budget that overflowed the model's window. Ordinary text is unchanged.

- **Re-vendored `cambrian_pb2` for kernel contract 0110** (ENG-2147, ENG-2154): the
  agent-plane `ErasureSelector.memory_note_ids` (name entries of the run principal's own memory
  notes) and `ErasureReceipt.redacted_messages` (the chat replies an erase redacted). Additive;
  no SDK API changes.

### Fixed

- **The skill menu no longer breaks the prompt cache across turns** (ENG-126). An agent on
  the full tool menu (`seed_tools_full`, e.g. the single-agent lane's chat agent and the
  profile workers) now also gets the full permitted skill menu (`list_skills` with no
  query; the kernel serves it sorted by name) instead of the top-3 skills ranked by the
  task text. The menu is rendered into the cached system tier, and a chat turn is a fresh
  run, so a per-message top-k changed that tier on every turn of a conversation and the
  provider re-read the whole system prompt at full price. Agents without
  `seed_tools_full` keep the task-relevant push.
- **A pool worker whose class declares a different `agent_id` is no longer refused by the
  kernel** (ENG-2129). `CognitiveAgent.serve()` now adopts `CAMBRIAN_AGENT_ID`, the registry
  id the kernel launched the process as and minted its per-instance token for, over the id
  the class declares, the way a daemon adopts `--agent-id`. Before, the declared id went out
  as `x-agent-id` and every memory, LLM gateway and tool call was refused UNAUTHENTICATED.
  A mismatch is logged once at WARN naming both ids. With `CAMBRIAN_AGENT_ID` unset (an agent
  started by hand) the declared id is kept. New helper `channel.kernel_agent_id`.
- **A worker whose `run()` ends on a gateway failure now answers with the gateway's code, not
  INTERNAL** (ENG-2125). When the exception chain holds a `GenerateError` coded
  `UNAUTHENTICATED`, `PERMISSION_DENIED`, `RESOURCE_EXHAUSTED` or `UNAVAILABLE`, the worker's
  Execute status carries that code (as it already did `OUT_OF_RANGE` for a context overflow),
  so the kernel's self-healer and the chat lane see a refused credential, a rate limit or a
  provider outage instead of an opaque INTERNAL. The provider's retry delay travels too:
  `GenerateError.retry_delay` (seconds, or `None`) is read from the gateway's
  `google.rpc.RetryInfo` detail and written back on the worker's status. Any other failure,
  including an author bug, stays INTERNAL.

## 0.1.7 (2026-11-18)

Releases the lease fixes: on 0.1.6 the reasoning loop's skill menu and the artifact calls sent
no lease, so an authenticated kernel refused the menu `no_lease` on every chat turn.

### Fixed

- **Skill and tool menus present the running task's lease** (ENG-2087). `SubstrateClient.
  list_skills`, `list_tools` and `find_tools` called without `session_token_id` inside a task
  now send the task's lease as `x-lease-id`, as memory reads have since 0.1.6. The reasoning
  loop's skill menu sent no lease at all, so a kernel on an authenticated deployment refused
  it `no_lease` on every chat turn and composed step. A use_skill load still sends its run
  key as `x-session-token` too.

- **Artifact calls present the running task's lease** (ENG-2086). `ArtifactManager.save`, `get`
  and `list_from_step` take an optional `session_token_id` and otherwise send the task's lease
  as `x-lease-id`, so an upload is filed in the run, session and step it was made in and an
  artifact read resolves the caller's conversation owner. `save` returns the artifact reference
  (a ledger id), as it has since ADR-0157 S4.

## 0.1.6 (2026-11-18)

The first release that ships `cambrian_agent_sdk.channel`, the unix kernel-socket path a
sandboxed agent uses to reach the kernel (ADR-0157 S2/S5). 0.1.5 on PyPI lacks it, so a
release-built agent environment could not boot a sandboxed agent (ENG-2033). Everything below
was unreleased until now.

### Changed

- **Memory reads present the running task's lease by default** (ENG-518, owner decision
  2026-11-14). `MemoryClient.recall` / `recall_actions` / `recall_entity` called without
  `session_token_id` inside a task now send the lease the kernel dispatched that task under
  (bound by the runtime around `run()`). A kernel on an authenticated deployment refuses a
  lease-less agent-plane memory read (`no_lease`); an explicit `session_token_id` still wins,
  and outside a task nothing is sent.

### Added

- **Vendored stubs for `PreviewForgetResponse.truncated`** (ENG-1953; kernel contract 0107,
  additive). The kernel now floors a query-form forget preview on each hit's cosine to the
  query and caps it at `execution.erasure.preview_max_candidates`; `truncated` is set when the
  cap cut the list (narrow the query to reach the rest). No client method is added.

- **Vendored stubs for the query form of `PreviewForget`** (ADR-0137 D2, ENG-1948; kernel
  contract 0106, additive). `PreviewForgetRequest.query` (free text, exclusive with
  `selector`) and `ErasureCandidate.snippet` (a clipped excerpt of the caller's own text, set
  on query previews only). The kernel resolves the query with the caller's own scoped recall;
  every hit becomes its whole document and only records the caller wrote are listed. No client
  method or ReAct action is added (owner decision 2026-11-17); agents reach it through the
  kernel's `forget_memory` tool, which gained a `query` argument.

- **Vendored stubs for the memory-erasure RPCs** (ADR-0137 S6, ENG-411 stage C; kernel
  contract 0105). `cambrian_agent_sdk/_proto` now carries `PreviewForget`, `Forget` and
  `ListMyErasures` with `ErasureSelector` / `ErasureCandidate` / `ErasureReceipt`. No client
  method or ReAct action is added: agents erase through the kernel's `forget_memory` tool on
  their tool menu (offered only when the kernel enables agent erasure).

- **Structured managed generation** (ADR-0139 C1, wave 1d). `SubstrateClient.generate_request(
  lease, *, system, messages, response_schema, cache_ttl, max_tokens, temperature, stop,
  strict)` returns a `GenerateResult` with `text`, `tool_calls`, `stop_reason`, `usage`
  (`GenerateUsage`: input/output, cache read/write, `cache_write_known`, `estimated`),
  `ignored` (request fields the model did not honour) and `model`. `generate`,
  `generate_stream` and `generate_with_tools` keep their signatures and accept the same
  keywords; `generate_with_tools` now returns a `GenerateResult` that still unpacks as
  `(text, tool_calls, stop_reason)`. `max_tokens`/`temperature` use field presence
  (`None` = model default, `0.0` is sent). System text goes only in `system`: a
  `"system"`-role message raises `InvalidRequest` before any RPC. Status codes map to
  typed errors, all also `grpc.RpcError`: `InvalidRequest` (INVALID_ARGUMENT),
  `SchemaValidationFailed` (ABORTED), `GenerationResourceExhausted` (RESOURCE_EXHAUSTED),
  `ModelUnavailable` (UNAVAILABLE), `GenerateError` (other); FAILED_PRECONDITION /
  UNIMPLEMENTED on `generate_with_tools` stay `ToolCallingUnsupported`. Against a kernel
  that predates C1 the stream also carries the flattened prompt, and a reply without
  `usage` sets `result.legacy_kernel`.
- **The ReAct loop sends a cached system tier plus messages** (ADR-0139 D2). Role,
  constraints, anti-patterns, the action protocol (tool policy, skill index) and the
  per-turn contract are one `system` block with `cache=True`, byte-identical across a
  run's turns; the task, the trajectory and per-turn notices (for example "Memory budget
  is exhausted", formerly a constraint) are the user turn. `cache_ttl` defaults to
  `"short"`; set `CognitiveAgent.cache_ttl = "long"` or `CAMBRIAN_AGENT_CACHE_TTL`. The
  loop degrades to the legacy single prompt when the kernel reports `system` as ignored,
  refuses the structured call, or the substrate predates it. `helpers.build_prompt_parts`
  / `PromptParts` expose the split; `build_prompt` is unchanged.
- **ReAct prompt texts in the prompt library** (ADR-0139 C4): `prompts/react-constraints`,
  `react-anti-patterns`, `react-action-protocol`, `react-turn`, `react-nudges` (v1,
  fragment sets read with `prompt_library.fragment(id, section)`). Renders are identical
  to the previous inline text except two lint fixes: "Output EXACTLY ONE JSON object" ->
  "Output exactly ONE JSON object", "TOOL BUDGET EXHAUSTED" -> "Tool budget exhausted".
  Each call logs (debug) the prompt ids, versions and static hashes and the system text's
  sha256.

- **Source metadata on memory** (ADR-0136 Amendment A3). `MemoryClient.remember()` takes
  keyword-only `source_kind`, `title` and `author`, and a part may carry a speaker as a
  third element, `parts=[(text, occurred_at, speaker), ...]`. All are descriptive: the
  kernel never lets `author` replace the authenticated writer, and speakers never enter
  the stored text. Vendored stubs re-generated from core's `cambrian.proto`
  (`IngestMemoryRequest` fields 9-11, `MemoryPart.speaker` = 3; grpcio-tools 1.81.1).
- **Event time on memory** (ADR-0134, kernel contract 0099). `MemoryClient.remember()`
  takes keyword-only `occurred_at` (when the content happened), `parts=[(text,
  occurred_at), ...]` for multi-message content such as a chat's turns (replaces `text`),
  and `time_precision` (`year` … `second`, see `TIME_PRECISIONS`). `recall()` takes
  `reference_time` (the asker's "now"), `event_after` (inclusive) and `event_before`
  (exclusive). Every time must be timezone-aware: a naive `datetime` raises `ValueError`
  before any RPC, because the SDK will not guess a zone. Every recall result dict
  (`recall`, `recall_actions`, `recall_precedents` (removed 2026-11-05; now returns `[]`), `recall_entity`) gains `event_time` /
  `event_time_end` (aware UTC `datetime` or `None`), `time_source` (`TIME_SOURCES`) and
  `time_precision`. A kernel without the event window wired answers `UNIMPLEMENTED` for
  `event_after`/`event_before` rather than ignoring them.
- **One channel builder, with a credential and a TLS path** (ADR-0130 D1/D4,
  `cambrian_agent_sdk/channel.py`). Every dial to the kernel — `Agent.serve`, the daemon
  `SignalStream`, the lazily-dialed `SubstrateClient` — now goes through `build_channel`.
  It reads `CAMBRIAN_AGENT_TOKEN` (the agent's own secret, presented as
  `authorization: Bearer` on every RPC), `CAMBRIAN_AGENT_ID` (sent as `x-agent-id` where
  the call did not set one) and `CAMBRIAN_TLS_CA` (a PEM root; set ⇒ TLS channel, with
  `CAMBRIAN_TLS_SERVER_NAME` to override the verified hostname). The kernel injects the
  token and id into the agents it spawns. Without a root the channel is plaintext and the
  SDK logs one WARN per process, which is the expected shape on a loopback kernel running
  with `server.insecure_localhost=true` — nothing changes for that deployment.

### Deprecated

- **`DeterministicAgent`** (owner decision 2026-10-31). Defining a subclass now emits a
  `DeprecationWarning`, once per subclass, attributed to the module that defines it, so an
  agent run as `python agent.py` sees it under Python's default warning filter. Why: the
  kernel marks trait `"tool"` deprecated (superseded by kernel-owned tools, ADR-0039 A1.3),
  removed the static-bid bypass (ADR-0023), and its default capability-typed dispatch
  (ADR-0100) solicits no bids, so the class's defining feature, a static
  Confidence=1.0 / 5 ms bid, no longer reaches any selection path.

  Migrate by subclassing `CognitiveAgent` and overriding `run()` with the scripted logic;
  agents are dispatched by the capabilities their manifest declares. Note the manifest
  trait changes from `"tool"` to `"cognitive"`, so the kernel will interview the agent and
  may draft it as a verifier (kernel system agents are interview-exempt either way). A
  deterministic kernel function belongs in the kernel tool registry, invoked via
  `ExecuteTool`.

  Nothing else changes: `trait` stays `"tool"`, `propose()` still returns the static bid,
  and the class will be removed only after the kernel drops `TraitTool`.

- **`MemoryClient.recall_precedents`** (owner decision 2026-11-05,
  `cambrian-knowledge/decisions/scenes-cancelled.md`). Precedents and scenes were
  removed from the kernel, including the `x-lane=precedents` lane. The method stays as a
  shim: it emits a `DeprecationWarning` and returns `[]` **without calling the kernel**,
  because a current kernel would answer the lane from the fact lane and hand back facts
  labelled as prior transitions. Use `recall()` for facts.

### Removed

- **`HTTPChatIngress`** (owner decision 2026-11-14, ENG-1449; deprecated since ADR-0142
  K-5). It was an unauthenticated loopback HTTP chat endpoint: anyone who could reach its
  port spoke as every conversation it served. Its only consumer, the airline benchmark
  driver, now drives conversations through the authenticated operator plane
  (`OpenConversation` / `SendTurn`). A person or harness talks to Cambrian through that
  plane; a transceiver entry point subclasses `IngressAgent`.

- **`DeterministicAgent`** and the agent trait **`"tool"`** (owner decision 2026-11-14,
  ENG-1351; deprecated since 2026-10-31). No agent in any repository used them. A scripted
  capability subclasses `CognitiveAgent` and overrides `run()`, or becomes a kernel-owned
  tool invoked via `ExecuteTool`. The kernel now refuses a manifest declaring
  `"trait": "tool"` (the agent is not registered) and migrates registry records stored
  with it.

- **The `recall_precedents` ReAct action** (same decision). The action protocol
  (`prompts/react-action-protocol/v1.md`, text and native variants) no longer advertises
  it, the loop no longer handles it, it is no longer a native control verb, and the
  `recall_precedents` trajectory step and its `<precedent>` rendering are gone from
  `working_memory`. `tests/golden/react_prompts_v1.json` was updated deliberately: the
  only change in every action-protocol render is the removed `recall_precedents` bullet.
