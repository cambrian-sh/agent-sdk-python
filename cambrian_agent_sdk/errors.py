"""Typed exceptions for Cambrian agent authors — ADR-0023 Issue #0023-06."""

from __future__ import annotations

from typing import TYPE_CHECKING


class BudgetExceededError(Exception):
    """Raised by an agent's ``run()`` when a task's estimated cost exceeds budget.

    ``runtime.dispatch_run`` catches this at the dispatch boundary and returns a sentinel
    response with ``payload.type = "budget_signal"`` so the DAGExecutor can
    distinguish a principled budget refusal from genuine incapability
    (``confidence=0.0`` alone is ambiguous).

    The VerificationWorker skips responses where ``payload.type == "budget_signal"``
    — preventing TrustScore penalisation for agents that refuse overbudget tasks.

    Usage::

        from cambrian_agent_sdk import AgentTask, BudgetExceededError, CognitiveAgent

        class CodeAgent(CognitiveAgent):
            def run(self, task: AgentTask):
                estimated = estimate_tokens(task.text) * COST_PER_TOKEN
                if estimated > MAX_BUDGET:
                    raise BudgetExceededError(
                        f"estimated ${estimated:.4f} exceeds ${MAX_BUDGET:.4f}",
                        estimated_cost=estimated,
                    )
                ...
    """

    def __init__(self, reason: str, estimated_cost: float = 0.0) -> None:
        super().__init__(reason)
        self.reason = reason
        self.estimated_cost = estimated_cost


class ToolCallingUnsupported(Exception):
    """Native tool-calling is not available for this call (ADR-0097 Phase B).

    Raised by ``substrate.generate_with_tools`` when the kernel answers
    ``UNIMPLEMENTED`` (a build predating the RPC) or ``FAILED_PRECONDITION`` (the RPC
    exists, but the model allocated to this step cannot do tool-calling).

    It is a CAPABILITY answer, not a failure. The ReAct loop catches it and falls back
    to the prompt-encoded action protocol for the rest of the run. It is a distinct
    exception precisely so that a real outage — a timeout, an auth error, a degraded
    model — cannot be mistaken for "no tool support" and silently downgrade every
    subsequent turn.
    """


# ── Managed-generation errors (ADR-0139 C1, wave 1d) ─────────────────────────
#
# The kernel's agent plane answers a managed LLM call with a small, documented set
# of status codes (cambrian.proto, GenerateWithToolsRequest / ResponseSchema). The
# SDK maps each to a typed exception so a caller can tell "my request was wrong"
# from "the model is down" from "this deployment has no native tools" without
# parsing gRPC details. Each also subclasses grpc.RpcError (when grpc is importable,
# which it always is in a real install) so code written against the raw RpcError
# keeps catching them.
if TYPE_CHECKING:
    from grpc import RpcError as _RpcBase
else:
    try:  # pragma: no cover - grpc is a hard dependency; the fallback is for stripped envs
        import grpc as _grpc
        _RpcBase = _grpc.RpcError
    except Exception:  # noqa: BLE001
        _RpcBase = Exception


class GenerateError(_RpcBase):
    """A managed generation call failed. ``code`` is the gRPC status name
    (e.g. ``"UNAVAILABLE"``); ``details`` the kernel's message. The ``code()`` and
    ``details()`` methods mirror ``grpc.RpcError`` for code written against it.

    ``retry_delay`` is the seconds the provider asked to wait before trying again
    (the kernel's ``google.rpc.RetryInfo`` detail on a ``RESOURCE_EXHAUSTED`` or
    ``UNAVAILABLE``), or None when the kernel sent none."""

    def __init__(self, details: str = "", code: str = "UNKNOWN",
                 retry_delay: "float | None" = None) -> None:
        super().__init__(details)
        self._code = code
        self._details = details
        self.retry_delay = retry_delay

    def code(self):  # grpc.RpcError compatibility
        try:
            return getattr(_grpc.StatusCode, self._code)
        except Exception:  # noqa: BLE001
            return self._code

    def details(self) -> str:  # grpc.RpcError compatibility
        return self._details

    @property
    def status(self) -> str:
        return self._code

    def __str__(self) -> str:
        return f"{self._code}: {self._details}" if self._details else self._code


class InvalidRequest(GenerateError):
    """``INVALID_ARGUMENT``: the request itself is wrong. A ``"system"``-role message,
    an unknown role, a response schema that is not valid JSON, or a field marked
    ``strict`` that the allocated model cannot honour. Retrying unchanged cannot help."""


class SchemaValidationFailed(GenerateError):
    """``ABORTED``: the reply still failed its ``response_schema`` after the kernel's
    one repair attempt. Never retried on another provider by the kernel."""


class GenerationResourceExhausted(GenerateError):
    """``RESOURCE_EXHAUSTED``: a budget, quota or the gateway's admission limit."""


class ModelUnavailable(GenerateError):
    """``UNAVAILABLE``: every model on the step's fallback chain failed to serve."""


class ContextOverflow(GenerateError):
    """``OUT_OF_RANGE``: the prompt is larger than the serving model's context window
    (the provider's "prompt is too long"). Deterministic for an unchanged request: the
    kernel never retries it on another provider, and an agent that cannot shrink its
    prompt ends the step with this status so the self-healer does not re-run it
    (T-20261112-worker-resolve-cid-inline-has-no-size-cap-a, owner decision A4)."""


def _chain(exc: BaseException):
    """``exc`` and its cause/context chain, outermost first, each once."""
    seen = set()
    cur: "BaseException | None" = exc
    while cur is not None and id(cur) not in seen:
        yield cur
        seen.add(id(cur))
        cur = cur.__cause__ or cur.__context__


def is_context_overflow(exc: BaseException) -> bool:
    """True when ``exc`` or anything in its cause/context chain is a
    :class:`ContextOverflow`."""
    return any(isinstance(e, ContextOverflow) for e in _chain(exc))


#: Gateway failure codes a worker answers its Execute with when its ``run()`` ended
#: on one, instead of INTERNAL (ENG-2125, owner decision 2026-11-18). Each says
#: something the kernel acts on: the self-healer never retries OUT_OF_RANGE or
#: UNAUTHENTICATED / PERMISSION_DENIED (deterministic) and backs off on
#: RESOURCE_EXHAUSTED / UNAVAILABLE (transient); the chat lane tells the person
#: which (cambrian-core internal/chat/failure.go). Any other code — INTERNAL from
#: the gateway, INVALID_ARGUMENT, ABORTED, an author bug — stays INTERNAL: it is a
#: fault of this worker's step, not a gateway verdict the kernel can act on.
RELAYED_GATEWAY_CODES = frozenset({
    "OUT_OF_RANGE", "UNAUTHENTICATED", "PERMISSION_DENIED", "RESOURCE_EXHAUSTED", "UNAVAILABLE",
})


def relayed_gateway_error(exc: BaseException) -> "GenerateError | None":
    """The :class:`GenerateError` in ``exc``'s cause/context chain whose code is one
    of :data:`RELAYED_GATEWAY_CODES` (the outermost such), or None."""
    for e in _chain(exc):
        if isinstance(e, GenerateError) and e.status in RELAYED_GATEWAY_CODES:
            return e
    return None
