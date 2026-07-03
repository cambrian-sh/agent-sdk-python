"""Tests for the migrated code_generator_agent (CognitiveAgent, ADR-0036)."""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'agents'))

from cambrian_agent_sdk import AgentResult, AgentTask
from cambrian_agent_sdk.types import ProposalRequest


class _FakeSubstrate:
    def __init__(self, response="```python\nprint('hi')\n```"):
        self.response = response

    def generate(self, session_token_id=None, prompt="", **kw):
        return self.response

    def get_context_node(self, cid):
        return None


def test_run_returns_code_typed_result():
    import code_generator_agent as cg

    agent = cg.CodeGeneratorAgent(agent_id="code_generator_agent")
    agent.substrate = _FakeSubstrate()
    res = agent.run(AgentTask(text="write a hello world", session_token_id="t"))
    assert isinstance(res, AgentResult)
    # result_type="code" forces routing to the executor regardless of the LLM's type.
    assert res.type == "code"
    assert b"print" in res.data


def test_propose_high_for_code_keywords():
    import code_generator_agent as cg

    bid = cg.CodeGeneratorAgent(agent_id="c").propose(ProposalRequest(description="write a function to sort a list"))
    assert bid.confidence >= 0.85


def test_manifest_trait_cognitive():
    import code_generator_agent as cg

    m = json.loads(cg.AGENT_MANIFEST)
    assert m["trait"] == "cognitive"
    assert "code_generation" in m["tools"]
