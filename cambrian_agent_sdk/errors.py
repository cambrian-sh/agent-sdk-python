"""Typed exceptions for Cambrian agent authors — ADR-0023 Issue #0023-06."""

from __future__ import annotations


class BudgetExceededError(Exception):
    """Raised by an agent handler when a task's estimated cost exceeds budget.

    ``server.py`` catches this at the dispatch boundary and returns a sentinel
    response with ``payload.type = "budget_signal"`` so the DAGExecutor can
    distinguish a principled budget refusal from genuine incapability
    (``confidence=0.0`` alone is ambiguous).

    The VerificationWorker skips responses where ``payload.type == "budget_signal"``
    — preventing TrustScore penalisation for agents that refuse overbudget tasks.

    Usage::

        from cambrian_agent_sdk import BudgetExceededError

        @agent.capability("code_generation")
        def generate(request):
            estimated = estimate_tokens(request.payload.text) * COST_PER_TOKEN
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
try:  # pragma: no cover - grpc is a hard dependency; the fallback is for stripped envs
    import grpc as _grpc
    _RpcBase = _grpc.RpcError
except Exception:  # noqa: BLE001
    _RpcBase = Exception


class GenerateError(_RpcBase):
    """A managed generation call failed. ``code`` is the gRPC status name
    (e.g. ``"UNAVAILABLE"``); ``details`` the kernel's message. The ``code()`` and
    ``details()`` methods mirror ``grpc.RpcError`` for code written against it."""

    def __init__(self, details: str = "", code: str = "UNKNOWN") -> None:
        super().__init__(details)
        self._code = code
        self._details = details

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
