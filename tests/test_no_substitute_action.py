"""An agent that lacks the tool its step needs yields the step (or says so); it never
imitates the action with a destructive substitute.

T-20261106-planner-routes-a-file-move-step-to-an-agent: a "rename and move each
receipt" step reached an agent whose tools could write a file but not move one. It
asked to write ``.../Documents/Receipts/2026-10-11 Apex Courier Shipping Receipt.pdf``
with ``{placeholder for binary content ...}`` as the content — "moving" the receipt by
replacing it with a text stub. react-anti-patterns v2 tells a native agent not to do
that and to hand the step back with ``yield_subgoal`` instead; these tests pin that
the rule reaches the model's system prompt and that a model which follows it ends
the run as a yield, with no tool call made.
"""
import json

from cambrian_agent_sdk import AgentTask, CognitiveAgent

WRITE_FILE = {"name": "mcp:filesystem/write_file", "description": "Write a file (overwrites).",
              "schema_json": json.dumps({"type": "object", "properties": {
                  "path": {"type": "string"}, "content": {"type": "string"}}})}


class FileAgent(CognitiveAgent):
    role = "a careful assistant"

    def run(self, task):
        return self.think(task)


class NativeYielder:
    """A native tool-calling model with write_file but no move tool, which follows
    the no-substitute rule: it yields the move to an agent that can do it."""

    def __init__(self):
        self.systems = []
        self.tool_names = []
        self.executed = []

    def list_tools(self, query="", k=None, full=False):
        return [WRITE_FILE]

    def generate_with_tools(self, session_token_id=None, messages=None, tools=None, **kw):
        sys_blocks = kw.get("system") or []
        self.systems.append("\n".join(b.get("text", "") if isinstance(b, dict) else str(b)
                                      for b in sys_blocks))
        self.tool_names.append([t.get("name") for t in (tools or [])])
        return ("", [{"id": "c1", "name": "yield_subgoal", "arguments": json.dumps({
            "intent": "Move Downloads/receipt(2).pdf to Documents/Receipts/2026-10-11 Apex "
                      "Courier Shipping Receipt.pdf",
            "capability_hint": "file_organisation"})}], "tool_use")

    def execute_tool(self, *a, **kw):
        self.executed.append((a, kw))
        raise AssertionError("an agent without a move tool must not call a tool to fake the move")


def test_the_model_is_told_not_to_substitute_and_to_yield():
    bot = FileAgent(agent_id="assistant")
    bot.substrate = NativeYielder()
    bot.think(AgentTask(text="Rename receipt(2).pdf by its date and title and move it to "
                             "Documents/Receipts"), max_tool_rounds=4)
    system = bot.substrate.systems[0]
    assert "Don't stand in a different action for one you have no tool for." in system
    assert "placeholder content over a document destroys it" in system
    assert "yield_subgoal" in system
    # yield is actually offered as a tool, so the rule names something callable.
    assert "yield_subgoal" in bot.substrate.tool_names[0]


def test_an_agent_without_the_tool_yields_the_step_instead_of_writing():
    bot = FileAgent(agent_id="assistant")
    bot.substrate = NativeYielder()
    res = bot.think(AgentTask(text="Rename receipt(2).pdf by its date and title and move it "
                                   "to Documents/Receipts"), max_tool_rounds=4)
    assert res.is_yield, res
    assert res.subgoal.capability_hint == "file_organisation"
    assert "Move Downloads/receipt(2).pdf" in res.subgoal.intent
    assert bot.substrate.executed == []
