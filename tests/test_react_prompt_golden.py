"""The ReAct prompt texts moved into the SDK prompt library (ADR-0139 C4, wave 1d)
render exactly what the loop sent before the move.

``golden/react_prompts_v1.json`` was captured from the pre-move code (every
action-protocol variant: agent with/without a local tool, text/native, memory
on/off, yield on/off, skills, system tools, a domain format; and the constraint and
anti-pattern grids). It differs from the pre-move text in exactly one place, a
prompt-lint fix (L7, a run of capitals): "Output EXACTLY ONE JSON object" became
"Output exactly ONE JSON object". The loop nudges are checked against their
pre-move literals below; "TOOL BUDGET EXHAUSTED" became "Tool budget exhausted" for
the same lint rule.

A deliberate wording change is a new prompt VERSION (SPEC.md §1) plus a regenerated
golden, never an edit that makes this test pass.
"""
import itertools
import json
from pathlib import Path

import pytest

from cambrian_agent_sdk import CognitiveAgent, tool
from cambrian_agent_sdk import helpers, react
from cambrian_agent_sdk import prompt_library as pl

GOLDEN = json.loads((Path(__file__).parent / "golden" / "react_prompts_v1.json")
                    .read_text(encoding="utf-8"))


class Plain(CognitiveAgent):
    role = "r"

    def run(self, task):
        return self.think(task)


class WithTool(Plain):
    @tool
    def echo(self, text: str) -> str:
        """Echo."""
        return text


SYS_TOOLS = [{"name": "mcp:filesystem/write_file", "description": "Write a file.",
              "schema_json": json.dumps({"type": "object", "properties": {
                  "path": {"type": "string"}, "content": {"type": "string"}}})}]
SKILLS = [{"name": "audit", "description": "Audit a codebase."}]


@pytest.mark.parametrize("key", sorted(GOLDEN["protocol"]))
def test_action_protocol_renders_match_golden(key):
    kind, native, mem, yld, skills, st, dom = key.split("|")
    agent = (WithTool if kind == "tool" else Plain)(agent_id="a")
    got = react._compose_action_protocol(
        agent, dom, SYS_TOOLS if st == "1" else [], SKILLS if skills == "1" else [],
        allow_yield_subgoal=yld == "True", native_tools=native == "True",
        allow_memory=mem == "True")
    assert got == GOLDEN["protocol"][key]


def test_constraint_and_anti_pattern_grids_match_golden():
    for native, mem, tools in itertools.product([False, True], [3, 0], [12, 5]):
        assert helpers.default_constraints(
            native_tools=native, max_memory_queries=mem, max_tool_rounds=tools,
        ) == GOLDEN["constraints"][f"{native}|{mem}|{tools}"]
    for native, lw in itertools.product([False, True], [None, "fast_large_write_file"]):
        assert helpers.default_anti_patterns(native_tools=native, large_write_tool=lw) \
            == GOLDEN["anti_patterns"][f"{native}|{lw}"]


def test_turn_contracts_match_golden():
    assert react._PER_TURN_OUTPUT_CONTRACT == GOLDEN["turn"]["text"]
    assert react._PER_TURN_OUTPUT_CONTRACT_NATIVE == GOLDEN["turn"]["native"]
    assert react.INFERRED_FINAL_ANSWER_NOTE == GOLDEN["nudges"]["inferred_answer"]


# The pre-move f-strings, verbatim (with the one lint fix on tool-budget).
_PRE_MOVE = {
    ("truncated", (("how", "HOW."),)): (
        "<note>Your last action was TRUNCATED at the output-token limit before it "
        "finished — it was NOT executed. Do NOT emit a large payload in one action. "
        "HOW. Otherwise emit a smaller action.</note>"),
    ("truncated-how-tool", (("tool", "fast_large_write_file"),)): (
        "To write a large file, call fast_large_write_file in SECTIONS: the first "
        "call with append=false (create/overwrite), then append=true for each "
        "subsequent section."),
    ("truncated-how-none", ()): (
        "To write a large file, write it in SECTIONS: create it with the first "
        "part, then append each further part in its own call (or split the "
        "work across several smaller calls)."),
    ("vetoed", (("tool", "'x'"), ("count", 2))): (
        "<note>VETOED: 'x' already failed 2 times with similar "
        "arguments. Do NOT retry it — change your approach or give a final answer.</note>"),
    ("retry-once", (("tool", "'x'"),)): (
        "<note>'x' already failed once. Retrying once more — if it fails "
        "again, change your approach.</note>"),
    ("already-done", (("tool", "'x'"),)): (
        "<note>ALREADY DONE: 'x' already completed successfully this run "
        "with these arguments — its result is above. Do NOT repeat it; continue "
        "to the next step, or if the task is now complete, emit final_answer.</note>"),
    ("tool-budget", (("max_tool_rounds", 3),)): (
        "<note>Tool budget exhausted after 3 calls. No further "
        "tool call will run. Answer NOW, in your required output format, using "
        "what you have already gathered. If what you have is not enough to "
        "conclude, say so in that format rather than searching again.</note>"),
    ("memory-budget", ()): "<note>memory query budget exhausted; answer best-effort</note>",
    ("memory-exhausted", ()): ("<note>Memory budget is exhausted — answer best-effort "
                               "from what you have.</note>"),
}


@pytest.mark.parametrize("section,values", list(_PRE_MOVE), ids=lambda v: str(v)[:30])
def test_nudges_match_pre_move_text(section, values):
    assert react._nudge(section, **dict(values)) == _PRE_MOVE[(section, values)]


def test_legacy_build_prompt_trajectory_header_unchanged():
    out = helpers.build_prompt(role="r", task="t", context_str="ctx", step_no=4)
    assert ("Your ReAct loop so far (round 4). Each <step> is one round (oldest first). "
            "Use the trajectory to avoid repeating work; then choose your next single action. "
            "If a <reflection> block is present, it was extracted from a previous failure — "
            "read it before retrying.") in out


def test_fragment_sections_render_only_their_own_slots():
    frag = pl.fragment("react-constraints", "memory-text")
    assert frag.inputs == ["max_memory_queries"]
    with pytest.raises(pl.SlotMismatch):
        frag.render({})
    with pytest.raises(pl.SlotMismatch):
        frag.render({"max_memory_queries": 1, "max_tool_rounds": 2})
    with pytest.raises(pl.PromptNotFound):
        pl.fragment("react-constraints", "no-such-section")
    assert "<!-- section" not in pl.fragment("react-turn", "contract-text").body


def test_split_parts_keep_volatile_text_out_of_the_system_tier():
    kw = dict(role="r", task="the task", action_protocol="AP", output_schema="OS")
    a = helpers.build_prompt_parts(context_str="step one", step_no=1, **kw)
    b = helpers.build_prompt_parts(context_str="step one\nstep two", step_no=2,
                                   notes=["<note>budget gone</note>"], **kw)
    assert a.system == b.system and a.system_sha256 == b.system_sha256
    assert a.system[0]["cache"] is True
    assert "(round 2)" in b.messages[0]["content"] and "(round" not in b.system_text
    assert b.messages[0]["content"].endswith("<note>budget gone</note>")
    assert "budget gone" not in b.system_text
    assert b.legacy_prompt().startswith("<System>")
    # TaskType carries no step number in the split layout (it would vary per turn).
    c = helpers.build_prompt_parts(task_type="code", step_no=7, **kw)
    assert "This is a code task.\n" in c.system_text and "Step 7" not in c.system_text
