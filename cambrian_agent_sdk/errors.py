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
