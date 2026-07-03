"""Yield delegation SDK surface (ADR-0037 D10, issue 0037-07)."""

import base64

import pytest

from cambrian_agent_sdk import AgentResult, SubGoal, yield_subgoal


def test_yield_subgoal_builds_yield_result():
    state = b'{"step": 2}'
    r = yield_subgoal("fetch the exchange rate", capability_hint="currency", continuation_state=state)

    assert isinstance(r, AgentResult)
    assert r.is_yield
    assert r.type == "yield_subgoal"
    assert r.subgoal.intent == "fetch the exchange rate"
    assert r.subgoal.capability_hint == "currency"
    assert r.subgoal.continuation_state == state


def test_yield_encodes_subgoal_into_context_for_the_kernel():
    state = b"opaque-bytes"
    r = yield_subgoal("summarize the doc", continuation_state=state)

    # The kernel reads these context keys (no proto change); continuation_state
    # is base64 so binary survives the string-typed context map.
    assert r.context["_yield"] == "true"
    assert r.context["_yield_intent"] == "summarize the doc"
    assert base64.b64decode(r.context["_yield_continuation_state"]) == state


def test_subgoal_carries_no_agent_id():
    # Structural guarantee: agents stay blind to the resource population (D10).
    assert not hasattr(SubGoal(), "agent_id")
    assert not hasattr(SubGoal(), "target")


def test_plain_result_is_not_a_yield():
    assert not AgentResult.from_text("done").is_yield


def test_yield_rejects_empty_intent():
    with pytest.raises(ValueError):
        yield_subgoal("")
