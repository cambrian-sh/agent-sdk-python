"""Tests for the migrated summariser_agent (CognitiveAgent, ADR-0036)."""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'agents'))

from cambrian_agent_sdk import AgentResult, AgentTask
from cambrian_agent_sdk.types import ProposalRequest


class _FakeSubstrate:
    def __init__(self, response="- point one\n- point two"):
        self.response = response

    def generate(self, session_token_id=None, prompt="", **kw):
        return self.response

    def get_context_node(self, cid):
        return None


def test_run_returns_summary_typed_result():
    import summariser_agent as sm

    agent = sm.SummariserAgent(agent_id="summariser_agent")
    agent.substrate = _FakeSubstrate()
    res = agent.run(AgentTask(text="summarise this", session_token_id="t"))
    assert isinstance(res, AgentResult)
    assert res.type == "summary"


def test_propose_high_for_summary_keywords():
    import summariser_agent as sm

    bid = sm.SummariserAgent(agent_id="s").propose(ProposalRequest(description="give me a tldr summary of this"))
    assert bid.confidence >= 0.85


def test_propose_low_for_analysis_keywords():
    import summariser_agent as sm

    bid = sm.SummariserAgent(agent_id="s").propose(ProposalRequest(description="compare and evaluate these options"))
    assert bid.confidence <= 0.2  # not a summarisation fit


def test_manifest_trait_cognitive():
    import summariser_agent as sm

    m = json.loads(sm.AGENT_MANIFEST)
    assert m["trait"] == "cognitive"
    assert "summarisation" in m["tools"]
