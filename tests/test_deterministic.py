"""ADR-0036 issue 0036-07: the DeterministicAgent — a scripted cell.

Typed run(request) -> AgentResult, no reasoning/memory/tool surface, and automatic
static bidding (Confidence=1.0, Latency=5ms) so it behaves in the auction as the
kernel expects without the author overriding proposal logic.
"""

from cambrian_agent_sdk import AgentResult, AgentTask, DeterministicAgent


class Echo(DeterministicAgent):
    def run(self, task):
        return AgentResult.from_text("echo:" + task.text)


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


def test_static_bid_served_over_request_proposal_rpc():
    from cambrian_agent_sdk._proto import cambrian_pb2
    from cambrian_agent_sdk.runtime import TraitServicer

    servicer = TraitServicer(Echo(agent_id="echo"))
    resp = servicer.RequestProposal(
        cambrian_pb2.ProposalRequest(task_id="t", description="anything", confidence_hint=0.2),
        None,
    )
    assert resp.confidence == 1.0
    assert resp.estimated_latency_ms == 5
