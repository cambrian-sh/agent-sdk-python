# Changelog

All notable changes to `cambrian-agent-sdk`. Versions before this file existed are
described by their git history.

## Unreleased

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
