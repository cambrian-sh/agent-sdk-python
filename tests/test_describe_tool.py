"""ADR-0045: the describe_tool Tier-2 fetch in the ReAct loop.

The menu lists tools tersely (Tier-1: summary + arg names). Before calling a
tool the agent issues describe_tool(name) to fetch the full description + arg
schema (Tier-2), which is injected into working memory for the next turn.
"""

import json

from cambrian_agent_sdk import AgentTask, CognitiveAgent


class _Bot(CognitiveAgent):
    role = "a tool-using assistant"

    def run(self, task):
        return self.think(task)


class _FakeSubstrateDescribe:
    """Records list_tools calls; serves Tier-2 specs only for a names+full fetch
    (the menu push, query-only, returns nothing here)."""

    def __init__(self, responses, specs):
        self._responses = list(responses)
        self.prompts = []
        self.list_tools_calls = []
        self._specs = specs

    def generate(self, session_token_id=None, prompt="", **kw):
        self.prompts.append(prompt)
        return self._responses.pop(0)

    def list_tools(self, query="", k=0, names=None, full=False):
        self.list_tools_calls.append({"query": query, "k": k, "names": names, "full": full})
        if names and full:
            return [s for s in self._specs if s["name"] in names]
        return []


def test_describe_tool_fetches_and_injects_full_spec():
    specs = [{
        "name": "scrape",
        "description": "Scrape a URL and return content.",
        "schema_json": '{"properties":{"url":{"type":"string"}}}',
        "dangerous": False,
    }]
    bot = _Bot(agent_id="b")
    bot.substrate = _FakeSubstrateDescribe([
        json.dumps({"action": "describe_tool", "tool": "scrape"}),
        json.dumps({"action": "final_answer", "answer": "done"}),
    ], specs)

    res = bot.think(AgentTask(text="scrape a site"))
    assert res.text == "done"

    # A Tier-2 fetch (names + full) was issued for exactly the named tool.
    fetches = [c for c in bot.substrate.list_tools_calls if c["full"] and c["names"] == ["scrape"]]
    assert fetches, bot.substrate.list_tools_calls

    # The fetched full spec reached the next prompt (description + arg schema).
    last = bot.substrate.prompts[-1]
    assert "tool_spec" in last
    assert "Scrape a URL and return content" in last
    assert "url" in last


def test_describe_tool_unavailable_yields_note_not_crash():
    bot = _Bot(agent_id="b")
    bot.substrate = _FakeSubstrateDescribe([
        json.dumps({"action": "describe_tool", "tool": "forbidden"}),
        json.dumps({"action": "final_answer", "answer": "ok"}),
    ], [])  # no specs ⇒ unavailable / ungranted

    res = bot.think(AgentTask(text="x"))
    assert res.text == "ok"
    assert "not available" in bot.substrate.prompts[-1]


def test_describe_tool_missing_name_yields_note():
    bot = _Bot(agent_id="b")
    bot.substrate = _FakeSubstrateDescribe([
        json.dumps({"action": "describe_tool"}),  # no tool name
        json.dumps({"action": "final_answer", "answer": "ok"}),
    ], [])

    res = bot.think(AgentTask(text="x"))
    assert res.text == "ok"
    assert "describe_tool needs a 'tool' name" in bot.substrate.prompts[-1]
