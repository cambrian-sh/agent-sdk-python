"""ADR-0036 issue 0036-07: the DeterministicAgent — a scripted cell.

Typed run(request) -> AgentResult, no reasoning/memory/tool surface, and automatic
static bidding (Confidence=1.0, Latency=5ms) so it behaves in the auction as the
kernel expects without the author overriding proposal logic.

DeterministicAgent is deprecated (owner decision 2026-10-31, ticket
T-20261031-sdk-deterministicagent-still-advertises-trait): subclassing it warns,
and these tests pin that the deprecated behaviour is otherwise unchanged.
"""

import warnings

import pytest

from cambrian_agent_sdk import AgentResult, AgentTask, CognitiveAgent, DeterministicAgent

with pytest.deprecated_call():

    class Echo(DeterministicAgent):
        def run(self, task):
            return AgentResult.from_text("echo:" + task.text)


def test_subclassing_warns_deprecation_naming_the_replacement():
    with pytest.warns(DeprecationWarning, match="CognitiveAgent") as rec:

        class Scripted(DeterministicAgent):
            def run(self, task):
                return "x"

    assert len(rec) == 1
    msg = str(rec[0].message)
    assert "Scripted subclasses DeterministicAgent" in msg
    assert "ADR-0100" in msg and "ExecuteTool" in msg
    # stacklevel=3 attributes the warning to the module defining the class,
    # which is what Python's default __main__ filter keys on.
    assert rec[0].filename == __file__


def test_warns_once_per_subclass_not_per_instance():
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        Echo(agent_id="a")
        Echo(agent_id="b")
    assert [w for w in rec if issubclass(w.category, DeprecationWarning)] == []

    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")

        class Child(Echo):
            pass

    assert len([w for w in rec if issubclass(w.category, DeprecationWarning)]) == 1


def test_cognitive_agent_does_not_warn():
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)

        class Scripted(CognitiveAgent):
            def run(self, task):
                return "x"

    assert Scripted(agent_id="s").run(None) == "x"


def test_deprecated_trait_and_manifest_unchanged():
    import json

    e = Echo(agent_id="echo")
    assert e.trait == "tool"
    assert json.loads(e.manifest_json())["trait"] == "tool"


def test_static_bid_without_author_override():
    bid = Echo(agent_id="echo").propose(AgentTask(text="anything"))
    assert bid.confidence == 1.0
    assert bid.estimated_latency_ms == 5


def test_run_is_typed_and_deterministic():
    out = Echo(agent_id="echo").run(AgentTask(text="hi"))
    assert isinstance(out, AgentResult)
    assert out.text == "echo:hi"
    # deterministic: same input ⇒ same output, no reasoning path
    assert Echo(agent_id="echo").run(AgentTask(text="hi")).text == "echo:hi"


def test_no_reasoning_memory_or_tool_surface():
    assert not hasattr(DeterministicAgent, "think")
    assert not hasattr(DeterministicAgent, "tools")
    e = Echo(agent_id="echo")
    assert not hasattr(e, "think")


def test_deprecated_request_proposal_rpc_is_neutral():
    """The RPC is deprecated (no kernel caller): it answers a neutral proposal and never
    runs the agent's propose(), so it cannot attract work or crash the server."""
    from cambrian_agent_sdk._proto import cambrian_pb2
    from cambrian_agent_sdk.runtime import DEPRECATED_PROPOSAL_RATIONALE, TraitServicer

    class Boom(Echo):
        def propose(self, task=None):
            raise RuntimeError("must not be called")

    for agent in (Echo(agent_id="echo"), Boom(agent_id="boom")):
        resp = TraitServicer(agent).RequestProposal(
            cambrian_pb2.ProposalRequest(task_id="t", description="anything", confidence_hint=0.2),
            None,
        )
        assert resp.confidence == 0.0
        assert resp.rationale == DEPRECATED_PROPOSAL_RATIONALE
