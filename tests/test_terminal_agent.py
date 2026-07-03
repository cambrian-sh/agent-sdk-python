"""Tests for terminal_agent — ADR-0040 registry model.

The agent no longer contains an allowlist or executes commands itself: command
authorization (allowlist / pipe-redirect blocklist) and execution moved to the
kernel's ToolResourcePolicy + the confined terminal tool (tested in Go at
internal/domain/tool_policy_test.go and internal/tool/proc). Here we verify the
agent is a CognitiveAgent that marshals an execute_command tool_call to the
kernel via ExecuteTool (ADR-0039).
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "agents"))

from cambrian_agent_sdk import AgentTask

import terminal_agent as ta


class _FakeSub:
    def __init__(self, responses):
        self._r = list(responses)
        self.tool_calls = []

    def generate(self, *a, **k):
        return self._r.pop(0)

    def execute_tool(self, tool_name, args_json="", session_token_id="", step_index=0, timeout_ms=0, task_id=""):
        self.tool_calls.append((tool_name, json.loads(args_json or "{}")))
        return {"result_json": json.dumps({"stdout": "hello\n", "exit_code": 0}),
                "result_cid": "", "denied": False, "deny_reason": "", "error": "",
                "arg_hash": "h", "result_hash": "h"}


def test_agent_description_is_nonempty():
    assert ta.AGENT_DESCRIPTION.strip() != ""


def test_manifest_is_cognitive_not_tool():
    # TraitTool / DeterministicAgent is superseded by the kernel tool registry
    # (ADR-0039 A1.3): the terminal agent is now a CognitiveAgent.
    m = json.loads(ta.AGENT_MANIFEST)
    assert m["trait"] == "cognitive"


def test_routes_execute_command_to_kernel():
    bot = ta.agent
    bot.substrate = _FakeSub([
        json.dumps({"action": "tool_call", "tool": "execute_command", "args": {"command": "echo hello"}}),
        json.dumps({"action": "final_answer", "answer": "done"}),
    ])
    res = bot.run(AgentTask(text="run echo hello"))
    assert res.text == "done"
    assert bot.substrate.tool_calls == [("execute_command", {"command": "echo hello"})]


def test_agent_does_not_self_validate_commands():
    # The agent must NOT contain its own allowlist/blocklist anymore — that is the
    # kernel's ToolResourcePolicy. A "dangerous" command is just routed; the kernel
    # denies it (returned as a structured denial, not a crash here).
    bot = ta.agent
    bot.substrate = _FakeSub([
        json.dumps({"action": "tool_call", "tool": "execute_command", "args": {"command": "rm -rf /"}}),
        json.dumps({"action": "final_answer", "answer": "could not run"}),
    ])
    res = bot.run(AgentTask(text="delete everything"))
    # routed to the kernel (which would deny); the agent did not block it itself
    assert bot.substrate.tool_calls[0][0] == "execute_command"
    assert res.text == "could not run"
