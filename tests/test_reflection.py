"""ADR-0052: verbal self-reflection (Reflexion) layered on the recurrence gate.

The structural gate (ADR-0041 D4) counts repeated FAILED actions and hard-vetoes
them. This adds the *verbal* layer the literature flags as the natural next step:
on the first hard-veto, the loop asks the LLM *why* it failed + *what* to change,
and pins that lesson into working memory so every later attempt reads it.
"""

import json

from cambrian_agent_sdk import AgentResult, AgentTask, CognitiveAgent
from cambrian_agent_sdk.react import run_think
from cambrian_agent_sdk.reflection import build_reflection_prompt
from cambrian_agent_sdk.working_memory import WorkingMemory


class Bot(CognitiveAgent):
    role = "a careful test assistant"

    def run(self, task):
        return self.think(task)


_FAIL_CMD = json.dumps({"action": "tool_call", "tool": "execute_command",
                        "args": {"command": "find . -size +1M"}})
_REFLECTION = "It failed because the path filter is invalid; I will list the directory first."


class _ReflectingFailSubstrate:
    """A system tool that ALWAYS fails. ``generate`` returns the same failing action
    UNLESS the prompt is a reflection request (ADR-0052), for which it returns a plain
    prose lesson — the way a real model answers the two different prompts."""

    def __init__(self):
        self.prompts = []
        self.reflections_requested = 0
        self.tool_call_count = 0

    def generate(self, session_token_id=None, prompt="", **kw):
        self.prompts.append(prompt)
        if "do differently next" in prompt:   # the reflection prompt's distinctive phrase
            self.reflections_requested += 1
            return _REFLECTION
        return _FAIL_CMD

    def execute_tool(self, tool_name, args_json="", **kw):
        self.tool_call_count += 1
        return {"result_json": "", "result_cid": "", "denied": False, "deny_reason": "",
                "error": "DEADLINE_EXCEEDED", "arg_hash": "", "result_hash": ""}


# ── the pure prompt builder ────────────────────────────────────────────────────


def test_reflection_prompt_names_failure_and_forbids_actions():
    p = build_reflection_prompt(
        role="a coder", task_text="find big files",
        tool="execute_command", args={"command": "find . -size +1M"},
        failure_summary="DEADLINE_EXCEEDED", prior_reflections=[],
    )
    assert "find . -size +1M" in p          # the specific failing action
    assert "DEADLINE_EXCEEDED" in p          # the concrete failure
    assert "do differently next" in p        # asks why + what to change
    assert "Do NOT output JSON" in p         # forbids emitting another action


def test_reflection_prompt_includes_prior_reflections():
    p = build_reflection_prompt(
        role="r", task_text="t", tool="x", args={}, failure_summary="boom",
        prior_reflections=["earlier lesson about timeouts"],
    )
    assert "earlier lesson about timeouts" in p  # so the model does not repeat itself


def test_reflection_prompt_condenses_huge_args():
    """A multi-KB arg value must not bloat the reflection call (mirrors the card)."""
    p = build_reflection_prompt(
        role="r", task_text="t", tool="write_file",
        args={"path": "/a", "content": "X" * 5000}, failure_summary="boom",
        prior_reflections=[],
    )
    assert "X" * 5000 not in p                # the heavy value is NOT inlined
    assert "5000 chars" in p                  # it is condensed to a marker
    assert "/a" in p                          # small values stay verbatim


# ── pinned working-memory channel ──────────────────────────────────────────────


def test_reflection_entry_is_pinned_and_survives_bounding():
    """A reflection is kept by assembly even when the buffer overflows its cap —
    an ordinary note would be bounded out; the lesson must persist across attempts.

    v2: the reflection renders as a <reflection n="..."> block (the new format)."""
    wm = WorkingMemory(cap=3)
    wm.add_reflection("the lesson: try a different tool")
    for i in range(10):  # flood well past the cap with ordinary notes
        wm.add_text(f"filler note {i}")
    assembled = wm.assemble()
    # The lesson text is preserved in the reflection block; the exact form is the
    # v2 typed form (``<reflection n="N">lesson</reflection>``).
    assert "the lesson: try a different tool" in assembled
    assert "<reflection" in assembled
    assert "</reflection>" in assembled


def test_empty_reflection_is_ignored():
    wm = WorkingMemory()
    wm.add_reflection("   ")
    assert len(wm) == 0


# ── loop integration ───────────────────────────────────────────────────────────


def test_hard_veto_extracts_and_pins_a_verbal_reflection():
    """On the first hard-veto the loop requests a reflection and pins it so the next
    attempt's prompt carries the lesson (the verbal layer over the structural gate)."""
    bot = Bot(agent_id="b")
    bot.substrate = _ReflectingFailSubstrate()
    res = run_think(bot, AgentTask(text="find big files"))

    assert res.type == "error"                       # still escalates (gate intact)
    assert bot.substrate.reflections_requested == 1  # exactly one reflection extracted
    # the lesson reached a later action prompt, tagged as a reflection
    assert any(_REFLECTION in p and "<reflection>" in p for p in bot.substrate.prompts)


def test_reflection_can_be_disabled():
    """reflect_enabled=False ⇒ the structural gate still works, but no reflection
    LLM call is made (the escalation path is unchanged)."""
    bot = Bot(agent_id="b")
    bot.substrate = _ReflectingFailSubstrate()
    res = run_think(bot, AgentTask(text="x"), reflect_enabled=False)

    assert res.type == "error"
    assert bot.substrate.reflections_requested == 0  # opt-out respected


def test_reflection_does_not_change_recurrence_escalation_outcome():
    """Reflection is additive: the tool still runs exactly twice (novel + one soft-
    nudge retry) and the run still escalates — the lesson does not relax the gate."""
    bot = Bot(agent_id="b")
    bot.substrate = _ReflectingFailSubstrate()
    res = run_think(bot, AgentTask(text="find big files"))
    assert bot.substrate.tool_call_count == 2
    assert isinstance(res, AgentResult) and res.type == "error"
