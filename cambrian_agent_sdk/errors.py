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
