# Changelog

All notable changes to `cambrian-agent-sdk`. Versions before this file existed are
described by their git history.

## Unreleased

### Added

- **Event time on memory** (ADR-0134, kernel contract 0099). `MemoryClient.remember()`
  takes keyword-only `occurred_at` (when the content happened), `parts=[(text,
  occurred_at), ...]` for multi-message content such as a chat's turns (replaces `text`),
  and `time_precision` (`year` … `second`, see `TIME_PRECISIONS`). `recall()` takes
  `reference_time` (the asker's "now"), `event_after` (inclusive) and `event_before`
  (exclusive). Every time must be timezone-aware: a naive `datetime` raises `ValueError`
  before any RPC, because the SDK will not guess a zone. Every recall result dict
  (`recall`, `recall_actions`, `recall_precedents`, `recall_entity`) gains `event_time` /
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
