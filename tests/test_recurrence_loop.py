"""Regression tests for the recurrence gate blocking identical successful reads.

The field observed the model call the same read 4 times in a row. Two
contributing causes:

  1. The body was truncated to 400 chars (the model couldn't see the full
     content, so it kept re-reading). Fixed by the v2 tool_call Step
     renderer (full body inlined for results <= 4000 chars).
  2. The symmetric success-dedup guard either didn't fire or was masked
     by the truncation issue. These tests assert the guard IS firing.

Both are tested end-to-end through ``run_think`` with a substrate that
returns the same body on every ``execute_tool`` call.
"""
import json

from cambrian_agent_sdk import AgentResult, AgentTask, CognitiveAgent
from cambrian_agent_sdk.react import run_think


class Bot(CognitiveAgent):
    role = "a careful test assistant"

    def run(self, task):
        return self.think(task)


class _AlwaysOkLLM:
    """A substrate whose generate() returns the next canned response and whose
    execute_tool() always returns the same dict — the field's exact-args
    repeat case."""

    def __init__(self, responses, tool_result):
        self._responses = list(responses)
        self._tool_result = tool_result
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


def test_run_think_blocks_second_identical_successful_read():
    """The field saw the model call the same read 4 times. The symmetric
    success-dedup should block the 2nd identical call. With the v2 body-
    inlining fix, the model also SEES the full content on the 1st call —
    so it doesn't NEED to re-read — but the guard is the safety net.
    """
    args = {"path": "C:\\Users\\afsin\\Dev\\cambrian\\cambrian-runtime\\pong.html"}
    tool_action = {"action": "tool_call",
                    "tool": "mcp:filesystem/read_text_file",
                    "args": args}
    responses = [
        json.dumps(tool_action),  # 1st read — runs
        # 2nd read: the model ATTEMPTS the same call again (the field's
        # exact-args repeat case). The gate adds an "ALREADY DONE" note +
        # continues, and the 3rd response is the model's answer.
        json.dumps(tool_action),
        json.dumps({"action": "final_answer", "answer": "done", "type": "text"}),
    ]
    sub = _AlwaysOkLLM(
        responses,
        tool_result={"content": "<!DOCTYPE html>... " * 200},  # ~4000 chars
    )
    bot = Bot(agent_id="b")
    bot.substrate = sub
    res = run_think(bot, AgentTask(text="read the file"))
    assert res.type == "text"
    # The kernel saw exactly ONE tool call (the 2nd was blocked by the gate).
    assert sub.tool_calls == 1, (
        f"recurrence gate should block the 2nd identical read; got "
        f"{sub.tool_calls} tool calls"
    )
    # The 2nd prompt (after the gate fires) carries the "ALREADY DONE" note.
    assert "ALREADY DONE" in sub.prompts[-1]


def test_run_think_blocks_third_identical_successful_read_and_finalizes():
    """Three identical successful reads: 1st runs, 2nd blocked (note +
    continue), 3rd blocked and run_think finalizes (best-effort).
    """
    args = {"path": "C:\\Users\\afsin\\Dev\\cambrian\\cambrian-runtime\\pong.html"}
    tool_action = {"action": "tool_call",
                    "tool": "mcp:filesystem/read_text_file",
                    "args": args}
    responses = [
        json.dumps(tool_action),  # 1st — runs
        json.dumps(tool_action),  # 2nd — blocked (note + continue)
        json.dumps(tool_action),  # 3rd — gated, run_think finalizes
    ]
    sub = _AlwaysOkLLM(
        responses,
        tool_result={"content": "<!DOCTYPE html>... " * 200},
    )
    bot = Bot(agent_id="b")
    bot.substrate = sub
    res = run_think(bot, AgentTask(text="read the file"))
    assert res.type == "text"
    # The kernel saw only ONE tool call (the 2nd and 3rd were blocked).
    assert sub.tool_calls == 1
    # The "ALREADY DONE" note reached the model (in the final prompt).
    assert "ALREADY DONE" in sub.prompts[-1]


def test_run_think_distinguishes_different_tool_names_with_same_args():
    """The gate's dedup keys on (tool, args), not on args alone. read_text_file
    and read_file are DIFFERENT tools; the 2nd call (different tool, same
    args) is NOT blocked. This prevents the gate from over-blocking when the
    model legitimately switches between equivalent tools."""
    args = {"path": "C:\\Users\\afsin\\Dev\\cambrian\\cambrian-runtime\\pong.html"}
    responses = [
        json.dumps({"action": "tool_call",
                    "tool": "mcp:filesystem/read_text_file", "args": args}),
        json.dumps({"action": "tool_call",
                    "tool": "mcp:filesystem/read_file", "args": args}),
        json.dumps({"action": "final_answer", "answer": "done", "type": "text"}),
    ]
    sub = _AlwaysOkLLM(responses, tool_result={"content": "x" * 200})
    bot = Bot(agent_id="b")
    bot.substrate = sub
    res = run_think(bot, AgentTask(text="x"))
    # Both calls run (they are different tools).
    assert sub.tool_calls == 2
    # But neither is blocked by ALREADY DONE (different tools).
    # The 2nd prompt should NOT contain "ALREADY DONE" for the 1st call.
    # (This is a weak check — the symmetric guard does fire within a single
    # tool; we're asserting the cross-tool case is NOT over-blocked.)
    assert "ALREADY DONE: 'mcp:filesystem/read_text_file'" not in sub.prompts[1]
