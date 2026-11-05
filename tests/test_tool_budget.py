"""Running out of tool rounds produces an ANSWER, not an exception.

A research agent keeps searching until something stops it, because it cannot see
the budget it is spending. When the only thing that stopped it was
`ReActLoopError`, a whole run's gathered evidence was thrown away and the caller
could report nothing but a failure — which is what happened live: twenty-odd
consecutive web searches, then the session died with the deterministic findings
already in hand and nothing built on them.

The memory budget already degrades this way: nudge once, terminate if ignored.
This is the tool path made symmetric.
"""
import json

import pytest

from cambrian_agent_sdk import AgentTask, CognitiveAgent, ReActLoopError, tool


class Searcher(CognitiveAgent):
    role = "a researcher"

    def run(self, task):
        return self.think(task)

    @tool
    def search(self, q: str) -> str:
        """Search."""
        return "results for " + q


class Spinner:
    """A model that searches forever unless told to stop."""

    def __init__(self, obey_the_nudge=True):
        self.obey = obey_the_nudge
        self.calls = 0
        self.saw_notice = False

    def generate_with_tools(self, session_token_id=None, messages=None, tools=None, **kw):
        blob = json.dumps(messages or [])
        if "tool budget exhausted" in blob.lower():
            self.saw_notice = True
            if self.obey:
                return (json.dumps({"action": "final_answer",
                                    "answer": {"action": "abstain", "reason": "ran out of rounds"},
                                    "type": "text"}), [], "stop")
        self.calls += 1
        return ("", [{"id": f"call_{self.calls}", "name": "search",
                      "arguments": json.dumps({"q": f"query {self.calls}"})}], "tool_use")

    def execute_tool(self, *a, **kw):
        return {"result_json": json.dumps({"ok": True}), "result_cid": "", "denied": False,
                "deny_reason": "", "error": "", "arg_hash": "", "result_hash": ""}


def test_exhaustion_asks_for_an_answer_instead_of_raising():
    bot = Searcher("bot")
    llm = Spinner(obey_the_nudge=True)
    bot.substrate = llm

    res = bot.think(AgentTask(text="research it"), max_tool_rounds=3)

    assert llm.saw_notice, "the model was never told it had run out"
    # The run produced its domain answer rather than dying.
    assert json.loads(res.data)["action"] == "abstain", res.data


def test_the_runaway_guard_still_fires():
    # The nudge is ONE chance. A model that spends it on another tool call is a
    # runaway, and the guard that exists to stop it must still stop it.
    bot = Searcher("bot")
    bot.substrate = Spinner(obey_the_nudge=False)

    with pytest.raises(ReActLoopError):
        bot.think(AgentTask(text="research it"), max_tool_rounds=3)


def test_the_budget_is_still_enforced():
    # Graceful does not mean unlimited: the tool must stop running at the cap.
    bot = Searcher("bot")
    llm = Spinner(obey_the_nudge=True)
    bot.substrate = llm
    bot.think(AgentTask(text="research it"), max_tool_rounds=3)
    # 3 executed, and the 4th is what triggered the notice rather than running.
    assert llm.calls <= 4, f"{llm.calls} tool calls ran against a budget of 3"
