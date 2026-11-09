"""ADR-0145 D3: the evidence re-prompt.

A plain-text reply in a run that has gathered NO evidence (no tool ran, no memory hit,
nothing seeded) is re-prompted toward delegation or the next tool, not toward
final_answer, so a draft that promises to look something up ("I can find this for
you") does not become an answer from general knowledge. The decision is structural:
the tests vary what the loop saw, never the words of the reply.
"""
import json

import pytest

from cambrian_agent_sdk import AgentTask, CognitiveAgent, tool
from cambrian_agent_sdk.react import EVIDENCE_REPROMPT_ENV, run_think


class _ScriptedLLM:
    def __init__(self, responses, recall=None):
        self._responses = list(responses)
        self.prompts = []
        self.tool_calls = 0
        self._recall = recall or []

    def generate(self, session_token_id=None, prompt="", **kw):
        self.prompts.append(prompt)
        return self._responses.pop(0)

    def execute_tool(self, *a, **kw):
        self.tool_calls += 1
        return {"result_json": json.dumps({"ok": True}), "result_cid": "", "denied": False,
                "deny_reason": "", "error": "", "arg_hash": "", "result_hash": ""}


class FrontDesk(CognitiveAgent):
    """A chat-like agent whose only tool hands work to the kernel."""
    role = "a test front desk"

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.delegated = []

    @tool
    def delegate_to_planner(self, task_description: str) -> str:
        """Hand the task to the planner."""
        self.delegated.append(task_description)
        return "CUS-004 owes 1200 EUR (ledger led_abc)"

    def run(self, task):
        return self.think(task)


_PROMISE = "I can find this for you. Let me check the overdue invoices."
_DELEGATE = json.dumps({"action": "tool_call", "tool": "delegate_to_planner",
                        "args": {"task_description": "List overdue invoices"}})
_FINAL = json.dumps({"action": "final_answer", "answer": "CUS-004 owes 1200 EUR", "type": "text"})


def _run(responses, **kw):
    sub = _ScriptedLLM(responses)
    bot = FrontDesk(agent_id="chat")
    bot.substrate = sub
    res = run_think(bot, AgentTask(text="Which customers are overdue?"),
                    seed_recall=False, seed_system_tools=False, allow_yield_subgoal=False, **kw)
    return res, bot, sub


def test_promise_with_no_evidence_is_turned_into_delegation():
    res, bot, sub = _run([_PROMISE, _DELEGATE, _FINAL],
                         evidence_reprompt=True, delegation_tool="delegate_to_planner")
    nudge = sub.prompts[1]
    assert "Nothing has been looked up" in nudge
    assert "call delegate_to_planner now" in nudge
    assert "general knowledge" in nudge
    assert bot.delegated == ["List overdue invoices"], "the re-prompt must lead to a real delegation"
    assert res.data.decode() == "CUS-004 owes 1200 EUR"


def test_decision_is_structural_not_textual():
    """Any wording (here: another language) gets the same re-prompt when no evidence exists."""
    _, _, sub = _run(["Bunu sizin için bulabilirim.", _DELEGATE, _FINAL],
                     evidence_reprompt=True, delegation_tool="delegate_to_planner")
    assert "call delegate_to_planner now" in sub.prompts[1]


def test_delegation_tool_from_agent_attribute():
    sub = _ScriptedLLM([_PROMISE, _DELEGATE, _FINAL])
    bot = FrontDesk(agent_id="chat")
    bot.delegation_tool = "delegate_to_planner"
    bot.substrate = sub
    run_think(bot, AgentTask(text="q"), seed_recall=False, seed_system_tools=False,
              allow_yield_subgoal=False, evidence_reprompt=True)
    assert "call delegate_to_planner now" in sub.prompts[1]


def test_turn_evidence_keeps_the_ordinary_reprompt():
    """The caller seeded evidence (the chat agent's recalled memory): no delegation push."""
    _, _, sub = _run([_PROMISE, _FINAL], evidence_reprompt=True, turn_evidence=True,
                     delegation_tool="delegate_to_planner")
    assert "Nothing has been looked up" not in sub.prompts[1]
    assert "not a valid action" in sub.prompts[1]


def test_flag_off_keeps_the_ordinary_reprompt(monkeypatch):
    monkeypatch.delenv(EVIDENCE_REPROMPT_ENV, raising=False)
    _, bot, sub = _run([_PROMISE, _FINAL], delegation_tool="delegate_to_planner")
    assert "Nothing has been looked up" not in sub.prompts[1]
    assert bot.delegated == []


def test_flag_from_environment(monkeypatch):
    monkeypatch.setenv(EVIDENCE_REPROMPT_ENV, "1")
    _, _, sub = _run([_PROMISE, _DELEGATE, _FINAL], delegation_tool="delegate_to_planner")
    assert "call delegate_to_planner now" in sub.prompts[1]


def test_after_a_tool_ran_the_ordinary_reprompt_applies():
    """Evidence exists once a tool returned: the draft is pushed to final_answer as before."""
    _, bot, sub = _run([_DELEGATE, _PROMISE, _FINAL],
                       evidence_reprompt=True, delegation_tool="delegate_to_planner")
    assert len(bot.delegated) == 1
    assert "Nothing has been looked up" not in sub.prompts[2]


def test_unknown_delegation_tool_is_never_named():
    """A delegation tool the agent cannot call is not offered; the next-tool note is."""
    _, _, sub = _run([_PROMISE, _FINAL], evidence_reprompt=True, delegation_tool="no_such_tool")
    assert "no_such_tool" not in sub.prompts[1]
    assert "call the tool for your next step" in sub.prompts[1]


def test_agent_without_any_tool_keeps_the_ordinary_reprompt():
    class Bare(CognitiveAgent):
        role = "bare"

        def run(self, task):
            return self.think(task)

    sub = _ScriptedLLM([_PROMISE, _FINAL])
    bot = Bare(agent_id="bare")
    bot.substrate = sub
    run_think(bot, AgentTask(text="q"), seed_recall=False, seed_system_tools=False,
              allow_yield_subgoal=False, evidence_reprompt=True)
    assert "Nothing has been looked up" not in sub.prompts[1]
