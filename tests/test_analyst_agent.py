"""Tests for the migrated analyst_agent (CognitiveAgent, ADR-0036)."""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'agents'))

from cambrian_agent_sdk import AgentResult, AgentTask
from cambrian_agent_sdk.types import ContextRef, ProposalRequest


class _FakeSubstrate:
    def __init__(self, response="result"):
        self.response = response
        self.last_prompt = None

    def generate(self, session_token_id=None, prompt="", **kw):
        self.last_prompt = prompt
        return self.response

    def get_context_node(self, cid):
        return None


def _task(text, working_memory=None):
    return AgentTask(text=text, working_memory=working_memory or [], session_token_id="tok-1")


def test_run_returns_analysis_result():
    import analyst_agent as aa

    agent = aa.AnalystAgent(agent_id="analyst_agent")
    agent.substrate = _FakeSubstrate("Observations: x\nReasoning: y\nConclusion: z")
    res = agent.run(_task("Compare REST and GraphQL"))
    assert isinstance(res, AgentResult)
    assert res.type == "analysis"
    assert b"Conclusion" in res.data


def test_low_precision_refs_excluded_from_prompt():
    import analyst_agent as aa

    agent = aa.AnalystAgent(agent_id="analyst_agent")
    agent.substrate = _FakeSubstrate("ok")
    low = ContextRef(cid="low", type="ltm_doc", activation=0.9, precision=0.2, snippet="IRRELEVANT_NOISE")
    high = ContextRef(cid="high", type="ltm_doc", activation=0.9, precision=0.9, snippet="RELEVANT_FACT")
    agent.run(_task("Analyse", working_memory=[low, high]))
    assert "IRRELEVANT_NOISE" not in agent.substrate.last_prompt


def test_propose_high_for_analysis_keywords():
    import analyst_agent as aa

    bid = aa.AnalystAgent(agent_id="a").propose(ProposalRequest(description="compare and evaluate the trade-offs"))
    assert bid.confidence >= 0.8


def test_manifest_trait_cognitive():
    import analyst_agent as aa

    m = json.loads(aa.AGENT_MANIFEST)
    assert m["trait"] == "cognitive"
    assert "analysis" in m["tools"]


def test_description_mentions_analysis():
    import analyst_agent as aa

    assert any(kw in aa.AGENT_DESCRIPTION.lower() for kw in ("analy", "reason", "evaluat", "compar"))
