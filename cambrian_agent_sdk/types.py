"""Pure Python domain types mirroring the Cambrian proto contract.

Agent developers interact only with these classes; proto and gRPC are never
exposed beyond the SDK boundary.
"""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional


def _check(condition: bool, msg: str) -> None:
    if not condition:
        raise ValueError(msg)


@dataclass
class ScopeConfig:
    """Advisory three-set access-scope hint parsed from Handoff.Context (ADR-0034 D13).

    PHASE 1 WARNING: these caller-supplied tags carry **no security weight**. The
    Substrate enforces only the non-forgeable agent_scope (set by the operator at
    registration) and ignores these keys entirely. They exist for SDK-side
    construction convenience until Session.CallerScope (Phase 2) lands. Do not rely
    on them for isolation.
    """

    required_tags: List[str] = field(default_factory=list)
    any_of_tags: List[str] = field(default_factory=list)
    forbidden_tags: List[str] = field(default_factory=list)

    @classmethod
    def from_context(cls, context: Dict[str, str]) -> "ScopeConfig":
        """Parse the _required_tags / _any_of_tags / _forbidden_tags JSON arrays."""

        def parse(key: str) -> List[str]:
            raw = context.get(key, "")
            if not raw:
                return []
            try:
                val = json.loads(raw)
            except (ValueError, TypeError):
                return []
            return [str(x) for x in val] if isinstance(val, list) else []

        return cls(
            required_tags=parse("_required_tags"),
            any_of_tags=parse("_any_of_tags"),
            forbidden_tags=parse("_forbidden_tags"),
        )


@dataclass
class Payload:
    """Mirrors proto Object — the data body of a Handoff."""
    id: str = ""
    type: str = "text"
    data: bytes = b""
    metadata: Dict[str, str] = field(default_factory=dict)

    @property
    def text(self) -> str:
        """Decode data as UTF-8 text (convenience property)."""
        return self.data.decode("utf-8", errors="replace")

    @staticmethod
    def from_text(text: str, type: str = "text") -> "Payload":
        return Payload(type=type, data=text.encode("utf-8"))


@dataclass
class ContextRef:
    """A lightweight handle to a piece of content in the ContentStore.

    ADR-0022 Phase 3 — the unit of currency in the Global Workspace.

    Precision semantics:
      - precision >= 0 : cosine similarity to the step query (pgvector seeds have this)
      - precision == -1.0 : sentinel — not yet computed (BFS-discovered nodes)

    Call ``assemble_context(request.working_memory, fetch_fn=...)`` to convert
    these refs into a prompt-ready context string.
    """
    cid: str = ""
    type: str = ""
    labels: List[str] = field(default_factory=list)
    activation: float = 0.0
    precision: float = -1.0  # sentinel default; pgvector seeds receive a real score
    snippet: str = ""


@dataclass
class ContextNode:
    """A fully-resolved content node from the ContentStore or LTM.

    Returned by ``SubstrateClient.get_context_node(cid)`` and used as the
    resolved payload inside ``assemble_context(fetch_fn=...)``. ADR-0022.
    """
    cid: str = ""
    type: str = ""
    data: bytes = b""
    labels: List[str] = field(default_factory=list)


@dataclass
class ExecuteRequest:
    """Typed wrapper for an inbound Execute(Handoff) RPC call.

    ``deadline_remaining_ms`` carries the Go caller's context timeout. Pass it
    to ``agent.substrate.generate(timeout_ms=request.deadline_remaining_ms)``
    to propagate the step budget into the LLM call. 0 means no deadline.

    ``working_memory`` is populated in Phase 3 (use_global_workspace=true).
    Use ``assemble_context(request.working_memory, ...)`` to build a context string.
    ``context`` holds Phase 0/1/2 key-value context (mutually exclusive with working_memory).
    """
    handoff_id: str = ""
    from_agent: str = ""
    to_agent: str = ""
    payload: Payload = field(default_factory=Payload)
    confidence: float = 0.0
    uncertainties: List[str] = field(default_factory=list)
    context: Dict[str, str] = field(default_factory=dict)
    step_index: int = 0
    plan_id: str = ""
    session_token_id: str = ""
    deadline_remaining_ms: int = 0
    working_memory: List[ContextRef] = field(default_factory=list)

    def caller_scope(self) -> "ScopeConfig":
        """Advisory caller scope parsed from context (ADR-0034 D13).

        Phase 1: informational only — the Substrate enforces agent_scope and
        ignores these tags. Useful for an agent that wants to construct an initial
        ScopeConfig object, not for making access decisions.
        """
        return ScopeConfig.from_context(self.context)


@dataclass
class ExecuteResponse:
    """Return value from an agent's capability handler.

    ``context`` holds any *additive* keys the agent wants to propagate to
    downstream DAG steps. The Substrate namespaces them automatically under
    ``step_{N}_{key}`` — the agent must not include the prefix itself.
    """
    payload: Payload = field(default_factory=Payload)
    confidence: float = 1.0
    uncertainties: List[str] = field(default_factory=list)
    context: Dict[str, str] = field(default_factory=dict)


@dataclass
class ProposalRequest:
    """Typed wrapper for an inbound RequestProposal RPC call."""
    task_id: str = ""
    description: str = ""
    context: str = ""
    confidence_hint: float = 0.0


@dataclass
class ProposalResponse:
    """Return value from an agent's proposal handler."""
    confidence: float = 0.0
    rationale: str = ""
    requirements: List[str] = field(default_factory=list)
    estimated_latency_ms: int = 100
    metadata: Dict[str, str] = field(default_factory=dict)


@dataclass
class VerifyRequest:
    """Typed wrapper for a Verifier Pool judicial scoring request."""
    task_id: str = ""
    original_query: str = ""
    winner_output: str = ""
    winner_agent_id: str = ""
    bid_confidence: float = 0.0


@dataclass
class VerifyResponse:
    """Return value from an agent's verify handler."""
    quality_score: float = 0.5
    critique: str = ""


@dataclass
class AgentTask:
    """A protocol-free inbound task delivered to ``run()`` (ADR-0036 D3).

    The author never sees ``Handoff`` / ``Payload`` / proto types — the SDK
    translates an inbound ``Execute(Handoff)`` into this at the boundary.

    ``text`` is the convenience decoding of the inbound payload; ``data`` holds
    the raw bytes when the payload is binary. ``working_memory`` is the read-only
    Global Workspace set (ADR-0022); ``context`` carries Phase-0/1/2 key-values.
    """

    text: str = ""
    type: str = "text"
    data: bytes = b""
    metadata: Dict[str, str] = field(default_factory=dict)
    context: Dict[str, str] = field(default_factory=dict)
    working_memory: List[ContextRef] = field(default_factory=list)
    step_index: int = 0
    plan_id: str = ""
    task_id: str = ""  # ADR-0049 D3: per-step correlation key (step-{index}-{planID})
    session_token_id: str = ""
    deadline_remaining_ms: int = 0


@dataclass
class SubGoal:
    """A sub-goal an agent yields to the Central Executive (ADR-0037 D10).

    Expressed in **capability-space**: ``intent`` is a task description, NEVER an
    agent ID — the agent is blind to the resource population and the CE is the
    sole binder. ``capability_hint`` is an optional advisory soft prior.
    ``continuation_state`` is opaque, agent-owned bytes the CE stores and returns
    verbatim so the agent can resume statelessly. No agent IDs cross this boundary.
    """

    intent: str = ""
    capability_hint: Optional[str] = None
    payload: Optional[Payload] = None
    continuation_state: bytes = b""


# YIELD_RESULT_TYPE marks an AgentResult as a yield (D10). The kernel routes it
# to the YieldCoordinator instead of treating it as a final step result.
YIELD_RESULT_TYPE = "yield_subgoal"


@dataclass
class AgentResult:
    """A typed return value from ``run()`` (ADR-0036 D3).

    ``type`` is routing-significant and preserved through to the kernel
    (``code`` → executor, ``budget_signal`` → circuit breaker, ``image/png`` →
    binary artifact, ``yield_subgoal`` → YieldCoordinator). Authors may also
    return a plain ``dict`` / ``str`` / ``bytes`` — the SDK coerces those into an
    ``AgentResult`` (issue 0036-02).
    """

    data: bytes = b""
    type: str = "text"
    confidence: float = 1.0
    uncertainties: List[str] = field(default_factory=list)
    context: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, str] = field(default_factory=dict)
    subgoal: Optional[SubGoal] = None

    @staticmethod
    def from_text(text: str, type: str = "text") -> "AgentResult":
        return AgentResult(data=text.encode("utf-8"), type=type)

    @property
    def text(self) -> str:
        return self.data.decode("utf-8", errors="replace")

    @property
    def is_yield(self) -> bool:
        """True when this result is a yielded sub-goal (ADR-0037 D10)."""
        return self.subgoal is not None


def yield_subgoal(
    intent: str,
    capability_hint: Optional[str] = None,
    payload: Optional[Payload] = None,
    continuation_state: bytes = b"",
) -> AgentResult:
    """Yield a sub-goal to the Central Executive instead of blocking (ADR-0037 D10).

    Replaces the blocking ``SubstrateClient.execute(target=agent_id)``: the agent
    returns this from ``run()`` and is later re-dispatched with the sub-result and
    its ``continuation_state``. The single worker is freed in the meantime
    (``max_workers=1``). The sub-goal is encoded into the result context so it
    reaches the kernel without exposing proto to the author. **No agent ID is
    accepted** — only an intent (+ optional capability hint).
    """
    if not intent or not intent.strip():
        raise ValueError("yield_subgoal: intent must be a non-empty capability-space description")

    sg = SubGoal(
        intent=intent,
        capability_hint=capability_hint,
        payload=payload,
        continuation_state=continuation_state,
    )
    ctx: Dict[str, str] = {"_yield": "true", "_yield_intent": intent}
    if capability_hint:
        ctx["_yield_capability_hint"] = capability_hint
    if continuation_state:
        ctx["_yield_continuation_state"] = base64.b64encode(continuation_state).decode("ascii")
    return AgentResult(type=YIELD_RESULT_TYPE, context=ctx, subgoal=sg)


@dataclass
class Capability:
    """Declares a named capability an agent can service.

    Used to populate AGENT_MANIFEST and for default proposal scoring.
    """
    name: str = ""
    description: str = ""
    input_schema: Dict = field(default_factory=dict)
    output_schema: Dict = field(default_factory=dict)
    latency_p50_ms: int = 100

    def __post_init__(self) -> None:
        _check(bool(self.name and self.name.strip()), "Capability.name must be a non-empty string")
        _check(
            self.latency_p50_ms >= 0,
            f"Capability.latency_p50_ms must be >= 0, got {self.latency_p50_ms}",
        )
