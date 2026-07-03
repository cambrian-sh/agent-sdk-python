"""Tests for code_executor_agent — ADR-0040 registry model.

The agent no longer executes code itself: execution + isolation + env-scrub moved
to the kernel's confined code-exec tool (tested in Go at internal/tool/proc). Here
we verify the agent is a CognitiveAgent that marshals an execute_python tool_call
to the kernel via ExecuteTool (ADR-0039), so the code code_generator produced can
actually be run.
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "agents"))

from cambrian_agent_sdk import AgentTask

import code_executor_agent as cea


class _FakeSub:
    def __init__(self, responses):
        self._r = list(responses)
        self.tool_calls = []

    def generate(self, *a, **k):
        return self._r.pop(0)

    def execute_tool(self, tool_name, args_json="", session_token_id="", step_index=0, timeout_ms=0, task_id=""):
        self.tool_calls.append((tool_name, json.loads(args_json or "{}")))
        return {"result_json": json.dumps({"stdout": "42\n", "exit_code": 0}),
                "result_cid": "", "denied": False, "deny_reason": "", "error": "",
                "arg_hash": "h", "result_hash": "h"}


def test_agent_description_is_nonempty():
    assert cea.AGENT_DESCRIPTION.strip() != ""


def test_manifest_is_cognitive_not_tool():
    m = json.loads(cea.AGENT_MANIFEST)
    assert m["trait"] == "cognitive"


def test_routes_execute_python_to_kernel():
    bot = cea.agent
    bot.substrate = _FakeSub([
        json.dumps({"action": "tool_call", "tool": "execute_python", "args": {"code": "print(6*7)"}}),
        json.dumps({"action": "final_answer", "answer": "the answer is 42"}),
    ])
    res = bot.run(AgentTask(text="run print(6*7)"))
    assert "42" in res.text
    assert bot.substrate.tool_calls == [("execute_python", {"code": "print(6*7)"})]


def test_denied_execution_degrades_not_crashes():
    bot = cea.agent

    class _DenySub(_FakeSub):
        def execute_tool(self, tool_name, args_json="", session_token_id="", step_index=0, timeout_ms=0, task_id=""):
            self.tool_calls.append((tool_name, json.loads(args_json or "{}")))
            return {"result_json": "", "result_cid": "", "denied": True,
                    "deny_reason": "approval required but unavailable", "error": "",
                    "arg_hash": "h", "result_hash": ""}

    bot.substrate = _DenySub([
        json.dumps({"action": "tool_call", "tool": "execute_python", "args": {"code": "print(1)"}}),
        json.dumps({"action": "final_answer", "answer": "execution was not permitted"}),
    ])
    res = bot.run(AgentTask(text="run code"))
    # A denial is data the loop reasons about, not a crash.
    assert res.text == "execution was not permitted"
