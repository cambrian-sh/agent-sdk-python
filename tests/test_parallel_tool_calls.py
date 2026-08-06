"""Every tool_call_id the model emits gets a tool message back.

A model may emit SEVERAL tool calls in one turn. The loop runs one action per
turn, so only the first is executed — but the assistant turn is echoed back
verbatim carrying all of them, and every OpenAI-compatible provider requires one
tool message per tool_call_id:

    An assistant message with 'tool_calls' must be followed by tool messages
    responding to each 'tool_call_id'. (insufficient tool messages following
    tool_calls message)

Answering only the first therefore 400s the NEXT request, and the kernel surfaces
that as `model_unavailable: all 1 candidates degraded` — which reads as a broken
provider rather than a conversation we built wrong. It took down live discovery
sessions the moment the planner was given a toolset worth batching.
"""
import json

from cambrian_agent_sdk import AgentTask, CognitiveAgent, tool


class Bot(CognitiveAgent):
    role = "a test assistant"

    def run(self, task):
        return self.think(task)


class ToolBot(Bot):
    @tool
    def alpha(self, x: str) -> str:
        """First tool."""
        return "alpha:" + x

    @tool
    def beta(self, x: str) -> str:
        """Second tool."""
        return "beta:" + x


class ParallelLLM:
    """A substrate that emits two tool calls in one turn, then answers."""

    def __init__(self):
        self.sent = []
        self.turn = 0

    def generate_with_tools(self, session_token_id=None, messages=None, tools=None, **kw):
        self.sent.append([dict(m) for m in (messages or [])])
        self.turn += 1
        if self.turn == 1:
            return ("", [
                {"id": "call_1", "name": "alpha", "arguments": json.dumps({"x": "1"})},
                {"id": "call_2", "name": "beta", "arguments": json.dumps({"x": "2"})},
            ], "tool_use")
        return (json.dumps({"action": "final_answer", "answer": "done", "type": "text"}), [], "stop")

    def execute_tool(self, *a, **kw):
        return {"result_json": json.dumps({"ok": True}), "result_cid": "", "denied": False,
                "deny_reason": "", "error": "", "arg_hash": "", "result_hash": ""}


def _conversation_is_valid(messages):
    """Every tool_calls id in an assistant turn has a following tool message."""
    answered = {m.get("tool_call_id") for m in messages if m.get("role") == "tool"}
    for m in messages:
        if m.get("role") != "assistant":
            continue
        for c in m.get("tool_calls") or []:
            if c.get("id") not in answered:
                return False, c.get("id")
    return True, None


def test_parallel_tool_calls_all_get_a_reply():
    bot = ToolBot("bot")
    llm = ParallelLLM()
    bot.substrate = llm
    bot.think(AgentTask(text="do it"), max_tool_rounds=4)

    # The SECOND request is the one the provider rejected: it carries the assistant
    # turn with two tool_calls.
    assert len(llm.sent) >= 2, "the loop never made a second request"
    ok, missing = _conversation_is_valid(llm.sent[-1])
    assert ok, f"tool_call {missing!r} was never answered — this is the 400"


def test_the_unexecuted_call_is_told_why():
    # Not a silent empty reply: the model should learn its second call did not run
    # so it re-issues deliberately rather than assuming the tool is broken.
    bot = ToolBot("bot")
    llm = ParallelLLM()
    bot.substrate = llm
    bot.think(AgentTask(text="do it"), max_tool_rounds=4)

    replies = [m for m in llm.sent[-1] if m.get("role") == "tool"]
    joined = " ".join(str(m.get("content", "")) for m in replies)
    assert "one action per turn" in joined, f"no explanation was given: {replies}"


def test_a_single_tool_call_is_unaffected():
    # The common case must not gain an extra message.
    class SingleLLM(ParallelLLM):
        def generate_with_tools(self, session_token_id=None, messages=None, tools=None, **kw):
            self.sent.append([dict(m) for m in (messages or [])])
            self.turn += 1
            if self.turn == 1:
                return ("", [{"id": "call_1", "name": "alpha",
                              "arguments": json.dumps({"x": "1"})}], "tool_use")
            return (json.dumps({"action": "final_answer", "answer": "done", "type": "text"}), [], "stop")

    bot = ToolBot("bot")
    llm = SingleLLM()
    bot.substrate = llm
    bot.think(AgentTask(text="do it"), max_tool_rounds=4)

    ok, missing = _conversation_is_valid(llm.sent[-1])
    assert ok, f"tool_call {missing!r} unanswered"
    tool_msgs = [m for m in llm.sent[-1] if m.get("role") == "tool"]
    assert len(tool_msgs) == 1, f"an extra tool message appeared: {tool_msgs}"
