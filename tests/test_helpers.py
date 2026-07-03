"""Tests for cambrian_agent_sdk.helpers — ADR-0023 Issue #0023-05."""
import pytest
from cambrian_agent_sdk.helpers import extract_code_block, find_step_ref, build_prompt
from cambrian_agent_sdk.types import ContextRef


# ── extract_code_block ────────────────────────────────────────────────────────

def test_extract_code_block_python_fence():
    text = "Here is the code:\n```python\nprint('hello')\n```\nDone."
    assert extract_code_block(text) == "print('hello')"


def test_extract_code_block_generic_fence():
    text = "```\nx = 1 + 2\n```"
    assert extract_code_block(text) == "x = 1 + 2"


def test_extract_code_block_no_fence_returns_full_text():
    text = "x = 1 + 2"
    assert extract_code_block(text) == "x = 1 + 2"


def test_extract_code_block_strips_whitespace():
    text = "```python\n\n  def foo():\n      pass\n\n```"
    assert extract_code_block(text) == "def foo():\n      pass"


def test_extract_code_block_empty_input():
    assert extract_code_block("") == ""


def test_extract_code_block_multiline_code():
    code = "def sieve(n):\n    primes = []\n    for i in range(2, n):\n        primes.append(i)\n    return primes"
    text = f"```python\n{code}\n```"
    assert extract_code_block(text) == code


def test_extract_code_block_takes_first_fence_only():
    text = "```python\nfirst()\n```\n\n```python\nsecond()\n```"
    assert extract_code_block(text) == "first()"


# ── find_step_ref ─────────────────────────────────────────────────────────────

def _make_ref(cid: str, step_index: int, ref_type: str = "step_result") -> ContextRef:
    return ContextRef(
        cid=cid,
        type=ref_type,
        labels=[f"step_{step_index}", "result"],
        activation=0.8,
        precision=0.9,
    )


def test_find_step_ref_returns_matching_ref():
    refs = [_make_ref("cid-0", 0), _make_ref("cid-1", 1)]
    found = find_step_ref(refs, 0)
    assert found is not None
    assert found.cid == "cid-0"


def test_find_step_ref_correct_step_index():
    refs = [_make_ref("cid-0", 0), _make_ref("cid-1", 1), _make_ref("cid-2", 2)]
    assert find_step_ref(refs, 1).cid == "cid-1"
    assert find_step_ref(refs, 2).cid == "cid-2"


def test_find_step_ref_returns_none_when_not_found():
    refs = [_make_ref("cid-0", 0)]
    assert find_step_ref(refs, 5) is None


def test_find_step_ref_empty_list():
    assert find_step_ref([], 0) is None


def test_find_step_ref_wrong_type_not_returned():
    ref = ContextRef(cid="cid-ltm", type="ltm_doc", labels=["step_0", "result"],
                     activation=0.8, precision=0.9)
    assert find_step_ref([ref], 0) is None


def test_find_step_ref_ignores_refs_without_step_label():
    ref = ContextRef(cid="cid-x", type="step_result", labels=["result"],
                     activation=0.8, precision=0.9)
    assert find_step_ref([ref], 0) is None


# ── build_prompt ──────────────────────────────────────────────────────────────

def test_build_prompt_contains_system_and_task():
    result = build_prompt("You are a coder.", "Write hello world.")
    assert "<System>" in result
    assert "You are a coder." in result
    assert "<Task>" in result
    assert "Write hello world." in result


def test_build_prompt_no_context_omits_context_block():
    result = build_prompt("system", "task")
    assert "<Trajectory" not in result


def test_build_prompt_with_context_includes_context_block():
    result = build_prompt("system", "task", context_str="some prior knowledge")
    assert '<Trajectory' in result
    assert "some prior knowledge" in result


def test_build_prompt_trajectory_stamps_step_number():
    """v2: the step number goes in the BODY (header) so the <Trajectory> tag is
    byte-identical across turns — the prefix cache covers the tag.

    (In v1, step_no was on the tag, which broke the cache when it changed.)
    """
    result = build_prompt("system", "task", context_str="did stuff", step_no=4)
    # The tag is the byte-identical header (no step= attribute).
    assert "<Trajectory>\n" in result
    # The step number surfaces in the body header.
    assert "(round 4)" in result
    assert "your next single action" in result.lower()


def test_build_prompt_v2_cache_friendly_order():
    """v2: the stable prefix (System + Task + ActionProtocol + OutputSchema) is the
    cacheable region; only the Trajectory breaks the cache."""
    ap = "ACTION PROTOCOL (composed once)"
    os_ = "OUTPUT SCHEMA (loop-invariant)"
    p1 = build_prompt(role="r", task="T", action_protocol=ap, output_schema=os_, step_no=1)
    p2 = build_prompt(
        role="r", task="T", action_protocol=ap, output_schema=os_, step_no=2,
        context_str='<step n="1" type="memory_query" query="X" status="empty">\n  <note>no relevant memory</note>\n</step>',
    )
    # Both prompts must contain the same 4 cacheable sections in the same order.
    # The fifth section (Trajectory) is the variable part — p1 has none, p2 has one.
    for section in ("<System>", "<Task>", "<ActionProtocol>", "<OutputSchema>"):
        i1 = p1.index(section)
        i2 = p2.index(section)
        assert i1 == i2, f"{section} position differs between turns: p1={i1}, p2={i2}"
    # The bytes from start through the end of </OutputSchema> are byte-identical.
    prefix1 = p1[: p1.index("</OutputSchema>") + len("</OutputSchema>")]
    prefix2 = p2[: p2.index("</OutputSchema>") + len("</OutputSchema>")]
    assert prefix1 == prefix2, "Stable prefix (System+Task+ActionProtocol+OutputSchema) must be byte-identical across turns"


def test_build_prompt_v2_includes_constraints_and_anti_patterns():
    """The 4 Rules from the SUMMARY must be visible at the top of <System>."""
    p = build_prompt("r", "t", task_type="research", session_context="Plan=1; budget=3/5")
    assert "<Constraints>" in p
    assert "<AntiPatterns>" in p
    assert "<TaskType>" in p
    assert "<SessionContext>" in p
    for required in [
        "ONE JSON action per turn",
        "memory_query is read-only",
        "recurrence gate VETOES",
        "ground it: at least one memory_query",
    ]:
        assert required in p, f"Default constraint missing: {required!r}"
    # The cid-handoff anti-pattern must be present (regression: see issue
    # in the field where the model did memory_query for content that already
    # had a cid, instead of passing the cid as {"$cid": "..."} to a tool_call).
    p_lower = p.lower()
    assert "use the cid as" in p_lower, (
        "cid-handoff anti-pattern missing from <AntiPatterns>"
    )


def test_build_prompt_whitespace_only_context_omitted():
    result = build_prompt("system", "task", context_str="   ")
    assert "<context>" not in result


def test_build_prompt_section_order():
    """v2: System, Task, Trajectory — the cacheable prefix (1-4) is byte-identical
    across turns; only the Trajectory (5, last) breaks the cache."""
    result = build_prompt("SYS", "TASK", context_str="CTX")
    # <System> must come before <Task>; <Task> must come before <Trajectory>.
    assert result.index("<System>") < result.index("<Task>")
    assert result.index("<Task>") < result.index("<Trajectory>")
    # The Trajectory must be the LAST section: its </Trajectory> close is the
    # final characters of the prompt (no content after it).
    assert result.rstrip().endswith("</Trajectory>"), (
        f"Trajectory must be the last section; got tail: {result[-80:]!r}"
    )


def test_build_prompt_strips_internal_whitespace():
    result = build_prompt("  system  ", "  task  ")
    assert "  system  " not in result
    assert "system" in result
