"""The loop must accept its own protocol when it arrives as a native tool call.

<ActionProtocol> advertises `final_answer`, `memory_query`, `use_skill` and the
rest as actions. A model given native tools expresses an action the only way that
interface offers — as a tool call. Only `find_tools` was recognised, so every
other verb was forwarded to the kernel's tool registry and refused:

    ⊘ refused final_answer: unknown tool
    ⊘ refused memory_query: unknown tool

`final_answer` made that fatal. The model tried to FINISH, was told the tool does
not exist, tried again, was refused again, and eventually summarised in prose —
which the controller then reported as "the planner answered in prose". The loop
was refusing its own protocol and blaming the model for the consequence.
"""
import json

from cambrian_agent_sdk import AgentTask, CognitiveAgent, tool
from cambrian_agent_sdk.react import action_from_native_turn


def _call(name, args):
    return [{"id": "c1", "name": name, "arguments": json.dumps(args)}]


def test_final_answer_as_a_native_call_is_an_answer_not_a_tool():
    act = action_from_native_turn("", _call("final_answer", {"answer": "done", "type": "text"}), "tool_use", {})
    assert act["action"] == "final_answer", act
    assert act["answer"] == "done", act


def test_a_structured_answer_survives():
    # The discovery planner's answer is an OBJECT. Losing it here would strand the
    # controller with nothing to execute.
    payload = {"action": "abstain", "reason": "no cadence stated"}
    act = action_from_native_turn("", _call("final_answer", {"answer": payload, "type": "text"}), "tool_use", {})
    assert act["answer"] == payload, act


def test_every_advertised_verb_is_recognised():
    # Whatever the protocol advertises, the loop must accept. A verb described in
    # the prompt and refused at dispatch is the exact shape of this bug.
    for verb in ("memory_query", "recall_precedents", "find_tools", "describe_tool",
                 "use_skill", "find_skills", "yield_subgoal", "resolve_cid"):
        act = action_from_native_turn("", _call(verb, {}), "tool_use", {})
        assert act["action"] == verb, f"{verb} was not recognised: {act}"


def test_a_real_tool_is_still_dispatched():
    # The fix must not swallow ordinary tool calls.
    act = action_from_native_turn(
        "", _call("mcp_firecrawl_firecrawl_search", {"query": "x"}), "tool_use",
        {"mcp_firecrawl_firecrawl_search": "mcp:firecrawl/firecrawl_search"})
    assert act["action"] == "tool_call", act
    assert act["tool"] == "mcp:firecrawl/firecrawl_search", act


class Finisher(CognitiveAgent):
    role = "a test assistant"
    result_type = "discovery_action"

    def run(self, task):
        return self.think(task)

    @tool
    def search(self, q: str) -> str:
        """Search."""
        return "results"


class NativeFinisher:
    """A model that finishes by CALLING final_answer, as native tool-calling invites."""

    def list_tools(self, query="", k=None, full=False):
        return []

    def generate_with_tools(self, session_token_id=None, messages=None, tools=None, **kw):
        return ("", [{"id": "c1", "name": "final_answer", "arguments": json.dumps(
            {"answer": {"action": "abstain", "reason": "done"}, "type": "text"})}], "tool_use")

    def execute_tool(self, *a, **kw):
        raise AssertionError("final_answer must never reach the tool plane")


def test_the_run_ends_instead_of_refusing_the_agent_its_ending():
    bot = Finisher("bot")
    bot.substrate = NativeFinisher()
    res = bot.think(AgentTask(text="do it"), max_tool_rounds=4)
    assert json.loads(res.data) == {"action": "abstain", "reason": "done"}, res.data
