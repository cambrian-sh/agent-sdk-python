"""Regression tests for the inverted final-answer default.

The loop used to treat ANY response it could not parse as an action as a final
answer and return immediately. So a model narrating its next step — "I need to write
a one-line summary to the output file" — or dumping its working-memory markup ended
the task REPORTING SUCCESS with the work undone. Silent, and strictly worse than the
loud ReActLoopError it displaced as the dominant failure.

The contract now: "finished" must be DECLARED, not inferred from a parse failure.
A non-action response is re-prompted once, then accepted so the loop still always
terminates. This mirrors LangChain's handle_parsing_errors / RetryWithErrorOutputParser
(feed the failure back, bounded) and Anthropic's "when stop_reason is NOT end_turn,
treat the response as incomplete".
"""
import json

from cambrian_agent_sdk import AgentTask, CognitiveAgent
from cambrian_agent_sdk.react import run_think


class Bot(CognitiveAgent):
    role = "a careful test assistant"

    def run(self, task):
        return self.think(task)


class _ScriptedLLM:
    """Returns canned generate() responses in order and records the prompts."""

    def __init__(self, responses, tool_result=None):
        self._responses = list(responses)
        self._tool_result = tool_result if tool_result is not None else {"ok": True}
        self.prompts = []
        self.tool_calls = 0

    def generate(self, session_token_id=None, prompt="", **kw):
        self.prompts.append(prompt)
        return self._responses.pop(0)

    def execute_tool(self, *a, **kw):
        self.tool_calls += 1
        return {"result_json": json.dumps(self._tool_result),
                "result_cid": "", "denied": False, "deny_reason": "",
                "error": "", "arg_hash": "", "result_hash": ""}


def _run(responses, tool_result=None):
    sub = _ScriptedLLM(responses, tool_result)
    bot = Bot(agent_id="b")
    bot.substrate = sub
    return run_think(bot, AgentTask(text="do the task")), sub


def test_narration_is_reprompted_not_accepted():
    """The exact production failure: the model announces the work instead of doing it."""
    res, sub = _run([
        "The file content is already known. I need to write a one-line summary to the "
        "output file.",
        json.dumps({"action": "tool_call", "tool": "mcp:filesystem/write_file",
                    "args": {"path": "out.md", "content": "summary"}}),
        json.dumps({"action": "final_answer", "answer": "written", "type": "text"}),
    ])

    assert sub.tool_calls == 1, "the re-prompt must give the model a chance to act"
    assert res.data.decode() == "written"
    # The nudge reached the model rather than the narration being returned as-is.
    assert "not a valid action" in sub.prompts[1]


def test_working_memory_markup_is_reprompted():
    """The other observed shape: internal markup leaking out as the 'answer'."""
    _, sub = _run([
        '<workspace>\n<fact precision="1.00">the third source document</fact>\n</workspace>',
        json.dumps({"action": "final_answer", "answer": "real answer", "type": "text"}),
    ])
    assert "not a valid action" in sub.prompts[1]


def test_declared_final_answer_returns_immediately():
    """A DECLARED answer is untouched — no extra round, no nudge."""
    res, sub = _run([
        json.dumps({"action": "final_answer", "answer": "42", "type": "text"}),
    ])
    assert res.data.decode() == "42"
    assert len(sub.prompts) == 1, "a declared final answer must not cost an extra round"


def test_inferred_answer_is_accepted_on_second_occurrence():
    """The loop must still always terminate: nudge once, then take the prose.

    This is the bounded half of the contract — without it, a model that only ever
    speaks prose would spin until the tool-round cap.
    """
    res, sub = _run([
        "I think the answer is probably 42.",
        "I still think the answer is 42.",
    ])
    assert res.data.decode() == "I still think the answer is 42."
    assert len(sub.prompts) == 2, "exactly one re-prompt before accepting"
