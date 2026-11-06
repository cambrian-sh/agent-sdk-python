"""T-20261105-after-context-compaction-re-reading-a-compacted: a re-issued call
whose output compaction placeholdered is served from the engine (not a dedup),
and the degraded answer never renders the instructions region."""

from cambrian_agent_sdk.context_engine import ContextConfig, DefaultContextEngine


def _engine():
    return DefaultContextEngine(ContextConfig(budget_tokens=200, tool_output_compact_chars=50,
                                              protect_tail=1, phase2_enabled=False))


def test_native_compacted_output_is_served_once():
    eng = _engine()
    big = "KEY-1: walnut " + "x" * 4000
    conv = [{"role": "user", "content": "task"}]
    for i in range(3):
        tid = f"c{i}"
        conv.append({"role": "assistant", "content": "",
                     "tool_calls": [{"id": tid, "name": "read", "arguments": "{}"}]})
        eng.note_tool_call(tid, "mcp:filesystem/read_text_file", {"path": f"part-{i}.txt"})
        conv.append({"role": "tool", "tool_call_id": tid, "content": big})
    assert eng.compact_conversation(conv, fixed_tokens=0)
    args = {"path": "part-0.txt"}
    assert eng.take_compacted_output("mcp:filesystem/read_text_file", args) == big
    # taken once: the body is back in context, a further identical call is a real duplicate
    assert eng.take_compacted_output("mcp:filesystem/read_text_file", args) is None
    # a call never compacted is not served
    assert eng.take_compacted_output("mcp:filesystem/read_text_file", {"path": "zzz"}) is None


def test_best_effort_excludes_pinned_skill():
    eng = _engine()
    eng.add_skill("bulk-digest", "SKILL BODY THAT IS NOT AN ANSWER")
    eng.add_text("observation: 1=walnut")
    out = eng.best_effort_text()
    assert "SKILL BODY" not in out and "<Instructions>" not in out
    assert "walnut" in out
