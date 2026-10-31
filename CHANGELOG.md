# Changelog

All notable changes to `cambrian-agent-sdk`. Versions before this file existed are
described by their git history.

## Unreleased

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
