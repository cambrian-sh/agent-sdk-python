"""Tests for BudgetExceededError dispatch path — ADR-0023 Issue #0023-06."""
import pytest

from cambrian_agent_sdk import CognitiveAgent, Capability, AgentTask, AgentResult
from cambrian_agent_sdk.runtime import dispatch_run
from cambrian_agent_sdk.errors import BudgetExceededError


# ── BudgetExceededError construction ─────────────────────────────────────────


def test_budget_exceeded_error_has_reason():
    err = BudgetExceededError("cost too high")
    assert err.reason == "cost too high"


def test_budget_exceeded_error_optional_cost():
    err = BudgetExceededError("too expensive", estimated_cost=0.12)
    assert err.estimated_cost == 0.12


def test_budget_exceeded_error_default_cost_is_zero():
    err = BudgetExceededError("too expensive")
    assert err.estimated_cost == 0.0


def test_budget_exceeded_error_is_exception():
    with pytest.raises(BudgetExceededError):
        raise BudgetExceededError("test")


# ── dispatch_run: BudgetExceededError → sentinel payload ────────────────


def test_dispatch_budget_error_returns_budget_signal_type():
    class _Expensive(CognitiveAgent):
        role = "expensive"

        def run(self, task):
            raise BudgetExceededError("too costly")

    agent = _Expensive(agent_id="test-agent")
    req = AgentTask(text="do something", data=b"do something")
    resp = dispatch_run(agent, req)
    assert resp is not None
    assert resp.type == "budget_signal"


def test_dispatch_budget_error_payload_contains_reason():
    class _Expensive(CognitiveAgent):
        role = "expensive"

        def run(self, task):
            raise BudgetExceededError("estimated $0.12 exceeds $0.05")

    agent = _Expensive(agent_id="test-agent")
    req = AgentTask(text="task", data=b"task")
    resp = dispatch_run(agent, req)
    assert b"BUDGET_EXCEEDED" in resp.data
    assert b"estimated $0.12 exceeds $0.05" in resp.data


def test_dispatch_budget_error_confidence_is_zero():
    class _Expensive(CognitiveAgent):
        role = "expensive"

        def run(self, task):
            raise BudgetExceededError("over budget")

    agent = _Expensive(agent_id="test-agent")
    req = AgentTask(text="task", data=b"task")
    resp = dispatch_run(agent, req)
    assert resp.confidence == 0.0


def test_dispatch_non_budget_error_propagates():
    class _Crash(CognitiveAgent):
        role = "crash"

        def run(self, task):
            raise ValueError("unrelated error")

    agent = _Crash(agent_id="test-agent")
    req = AgentTask(text="task", data=b"task")
    with pytest.raises(ValueError, match="unrelated error"):
        dispatch_run(agent, req)


def test_budget_exceeded_exported_from_sdk():
    from cambrian_agent_sdk import BudgetExceededError as BEE
    assert BEE is BudgetExceededError
