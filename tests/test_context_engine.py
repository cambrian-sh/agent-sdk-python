"""ADR-0139 D3: the context engine.

Covers the defects it exists to remove (investigations/2026-11-04-agent-context-
interview-prompts.md §1-2) and the behaviours it adds:

- a loaded skill is pinned for the run, however much tool output follows;
- each tool call reaches the prompt ONCE (the old working memory rendered a
  ToolCard and a Step for every call);
- phase-1 compaction turns old tool outputs into placeholder + cid, re-openable by
  resolve_cid with no per-run cap;
- phase-2 compaction summarises the middle turns with a library prompt, and fails
  LOUDLY (phase 1 only, nothing dropped) when the summariser cannot;
- native final_answer ends the run with exactly its argument (no narration);
- read-only system tools of one native turn run in parallel; a dangerous one does not;
- the D14 memory-notes slot is a no-op without the RPC and frozen when present;
- the legacy engine stays reachable behind the flag.
"""
import json
import logging
import types

import pytest

from cambrian_agent_sdk import AgentTask, CognitiveAgent, tool
from cambrian_agent_sdk import context_engine as ce
from cambrian_agent_sdk.react import run_think

SKILL_BODY = ("# Audit procedure\n" + "\n".join(f"{i}. step {i} of the audit." for i in range(1, 60))
              + "\nFINAL-RULE-9c1e: the first line of the answer is AUDIT-ID.")


class Bot(CognitiveAgent):
    role = "a test assistant"
    local_skills = [{"name": "audit", "description": "Audit a codebase.",
                     "instructions": SKILL_BODY, "tool_grants": []}]

    def run(self, task):
        return self.think(task)

    @tool
    def fetch(self, key: str) -> str:
        """Fetch a document by key."""
        return f"DOC-{key}-START " + ("x" * 3000) + f" DOC-{key}-END"


def _act(**kw):
    return json.dumps(kw)


class TextSub:
    """Text-protocol substrate: scripted actions, every prompt recorded. Phase-2
    summariser calls (system text starts with the compaction prompt) are answered
    from ``summaries`` and recorded separately."""

    def __init__(self, actions, summaries=None):
        self.actions = list(actions)
        self.summaries = list(summaries or [])
        self.prompts = []
        self.summary_requests = []

    def generate_request(self, token, *, system=None, messages=None, **kw):
        sys_text = "\n\n".join(b["text"] for b in system or [])
        if sys_text.startswith("You compact the working history"):
            self.summary_requests.append(messages[0]["content"])
            out = self.summaries.pop(0) if self.summaries else ""
            if isinstance(out, Exception):
                raise out
            return types.SimpleNamespace(text=out, ignored=[], model="m", usage=None)
        self.prompts.append(messages[0]["content"])
        return types.SimpleNamespace(text=self.actions.pop(0), ignored=[], model="m", usage=None)

    def generate(self, token, prompt, **kw):  # pragma: no cover - structured path is used
        self.prompts.append(prompt)
        return self.actions.pop(0)


def _fetches(n):
    return [_act(action="tool_call", tool="fetch", args={"key": str(i)}) for i in range(n)]


# ── one representation per tool call ─────────────────────────────────────────

def test_each_tool_call_is_rendered_once():
    bot = Bot(agent_id="b")
    bot.substrate = TextSub(_fetches(2) + [_act(action="final_answer", answer="ok")])
    run_think(bot, AgentTask(text="t"))
    last = bot.substrate.prompts[-1]
    for key in ("0", "1"):
        assert last.count(f"DOC-{key}-START") == 1, "a tool output reached the prompt twice"
    assert last.count("type='tool_call' tool='fetch'") == 2
    # The output is marked as data, inside <evidence>.
    assert "<evidence source='tool:fetch' status='ok'>" in last


def test_legacy_engine_is_reachable_behind_the_flag(monkeypatch):
    monkeypatch.setenv("CAMBRIAN_CONTEXT_ENGINE", "legacy")
    bot = Bot(agent_id="b")
    bot.substrate = TextSub(_fetches(1) + [_act(action="final_answer", answer="ok")])
    run_think(bot, AgentTask(text="t"))
    last = bot.substrate.prompts[-1]
    # The old layout: Task first, and the call rendered as a card AND a step.
    assert last.startswith("<Task>")
    assert last.count("tool='fetch'") == 2


def test_task_comes_last_and_evidence_is_data():
    bot = Bot(agent_id="b")
    bot.memory = types.SimpleNamespace(recall=lambda q, **kw: [{"text": "fact one", "metadata": "{}"}])
    bot.substrate = TextSub([_act(action="final_answer", answer="ok")])
    run_think(bot, AgentTask(text="the objective", context={"expected_output": "a CSV",
                                                             "acceptance": "three rows"}))
    p = bot.substrate.prompts[0]
    assert p.index("<Evidence>") < p.index("<Task>")
    assert "<document source='memory'" in p and "fact one" in p
    assert "data, not instructions" in p
    task = p[p.index("<Task>"):]
    assert "the objective" in task and "Expected output:\na CSV" in task
    assert "Acceptance check:\nthree rows" in task


# ── pinned instructions ──────────────────────────────────────────────────────

def test_pinned_skill_survives_heavy_tool_use_and_compaction():
    bot = Bot(agent_id="b")
    bot.substrate = TextSub([_act(action="use_skill", skill="audit")] + _fetches(12)
                            + [_act(action="final_answer", answer="ok")],
                            summaries=["## Goal\naudit\n## Done\nfetched 0-8"])
    run_think(bot, AgentTask(text="t"), max_tool_rounds=20,
              context_config={"budget_tokens": 4000, "protect_tail": 3})
    last = bot.substrate.prompts[-1]
    assert SKILL_BODY in last, "the loaded skill was cut or evicted"
    assert last.index("<Instructions>") < last.index("<Trajectory>")
    # Compaction did happen around it.
    assert "<context_summary" in last or "compacted_cid" in last


def test_native_skill_turn_is_pinned_through_compaction():
    eng = ce.DefaultContextEngine(ce.ContextConfig(budget_tokens=200, tool_output_compact_chars=50,
                                                   protect_tail=1, phase2_enabled=False))
    conv = [{"role": "user", "content": "task"},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "s1", "name": "use_skill", "arguments": "{}"}]},
            {"role": "tool", "tool_call_id": "s1", "content": SKILL_BODY},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "t1", "name": "fetch", "arguments": "{}"}]},
            {"role": "tool", "tool_call_id": "t1", "content": "y" * 2000},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "t2", "name": "fetch", "arguments": "{}"}]},
            {"role": "tool", "tool_call_id": "t2", "content": "z" * 2000}]
    eng.pin_tool_call("s1")
    assert eng.compact_conversation(conv, fixed_tokens=0)
    assert conv[2]["content"] == SKILL_BODY            # pinned: untouched
    assert conv[4]["content"].startswith("[output of fetch compacted")
    assert conv[6]["content"] == "z" * 2000             # protected tail
    cid = conv[4]["content"].split("cid ")[1].split(".")[0]
    assert eng.resolve_local(cid) == "y" * 2000


# ── phase 1 ──────────────────────────────────────────────────────────────────

def test_phase1_placeholders_and_uncapped_resolve():
    bot = Bot(agent_id="b")
    sub = TextSub(_fetches(5))
    bot.substrate = sub
    # The placeholder cid is only known once compaction ran, so the script reads it
    # from the latest prompt.
    original = sub.generate_request

    def scripted(token, **kw):
        res = original(token, **kw)
        if not sub.actions and len(sub.prompts) == 5:
            p = sub.prompts[-1]
            cid = p.split("compacted_cid='")[1].split("'")[0]
            sub.actions.extend([_act(action="resolve_cid", cid=cid, **{"as": "inline"}),
                                _act(action="resolve_cid", cid=cid, **{"as": "inline"}),
                                _act(action="final_answer", answer="ok")])
        return res

    sub.generate_request = scripted
    res = run_think(bot, AgentTask(text="t"), max_inline_resolves=0,
                    context_config={"budget_tokens": 3500, "protect_tail": 2, "phase2_enabled": False})
    assert res.data == b"ok"
    p5 = sub.prompts[4]
    assert "compacted_cid=" in p5 and "Re-open it with resolve_cid" in p5
    assert p5.count("DOC-0-START") == 0, "an old output was not compacted"
    last = sub.prompts[-1]
    # Re-opened twice although the inline cap is 0: compacted cids are uncapped.
    assert last.count("DOC-0-START") >= 2
    assert "inline budget exhausted" not in last


def test_oversize_single_output_is_compacted_with_a_preview():
    eng = ce.DefaultContextEngine(ce.ContextConfig(tool_output_max_chars=1000,
                                                   tool_output_preview_chars=100))
    card = ce.ToolCard(tool="fetch", args={}, status="ok", summary="s")
    eng.record_tool_call(card, "H" * 100 + "T" * 5000)
    out = eng.render_history()
    assert "H" * 100 in out and "T" * 200 not in out
    assert "too large to show in full" in out
    assert len(eng.cards) == 1


# ── phase 2 ──────────────────────────────────────────────────────────────────

def test_phase2_summary_replaces_middle_turns_when_still_over_budget():
    bot = Bot(agent_id="b")
    sub = TextSub(_fetches(8) + [_act(action="final_answer", answer="ok")],
                  summaries=["## Goal\nfetch docs\n## Done\nSUMMARY-MARK-1",
                             "## Goal\nfetch docs\n## Done\nSUMMARY-MARK-2"])
    bot.substrate = sub
    run_think(bot, AgentTask(text="t"),
              context_config={"budget_tokens": 2100, "protect_tail": 2,
                              "tool_output_compact_chars": 100000})  # force phase 2
    assert sub.summary_requests, "phase 2 never ran"
    first = sub.summary_requests[0]
    assert "<turns>" in first and "<previous_summary>\n(none)" in first
    last = sub.prompts[-1]
    assert "<context_summary" in last and "SUMMARY-MARK" in last
    if len(sub.summary_requests) > 1:  # iterative: the previous summary is passed back
        assert "SUMMARY-MARK-1" in sub.summary_requests[1]


def test_phase2_failure_is_loud_and_drops_nothing(caplog):
    eng = ce.DefaultContextEngine(ce.ContextConfig(budget_tokens=100, protect_tail=1,
                                                   tool_output_compact_chars=100000),
                                  summariser=lambda prev, turns: (_ for _ in ()).throw(RuntimeError("boom")))
    for i in range(4):
        eng.record_tool_call(ce.ToolCard(tool="fetch", args={"k": i}, status="ok", summary="s"),
                             f"BODY-{i} " + "q" * 400)
    with caplog.at_level(logging.ERROR, logger="cambrian.context"):
        eng.compact_prompt(5000)
    assert any(r.message == "context_compaction_failed" for r in caplog.records)
    assert any(e["event"] == "compaction_failed" and "boom" in e["reason"] for e in eng.events)
    hist = eng.render_history()
    # Phase 1 is not applicable (threshold), phase 2 failed: every turn is still there.
    for i in range(4):
        assert f"BODY-{i}" in hist
    assert not eng.archive


def test_phase2_guard_refuses_content_larger_than_the_summariser_window():
    called = []
    eng = ce.DefaultContextEngine(ce.ContextConfig(budget_tokens=100, protect_tail=1,
                                                   tool_output_compact_chars=100000,
                                                   summariser_input_max_tokens=50),
                                  summariser=lambda p, t: called.append(1) or "s")
    for i in range(3):
        eng.record_tool_call(ce.ToolCard(tool="fetch", args={"k": i}, status="ok", summary="s"),
                             "w" * 1000)
    eng.compact_prompt(5000)
    assert not called
    assert any("summariser input budget" in e.get("reason", "") for e in eng.events)


def test_native_phase2_keeps_tool_call_groups_intact_and_folds_into_head():
    eng = ce.DefaultContextEngine(ce.ContextConfig(budget_tokens=50, protect_tail=1),
                                  summariser=lambda prev, turns: "## Goal\nNATIVE-SUMMARY")
    eng.add_skill("audit", "SKILL-BODY", "Loaded skill 'audit'.")
    conv = [{"role": "user", "content": "HEAD"}]
    for i in range(4):
        conv.append({"role": "assistant", "content": "",
                     "tool_calls": [{"id": f"c{i}", "name": "fetch", "arguments": "{}"},
                                    {"id": f"d{i}", "name": "fetch", "arguments": "{}"}]})
        conv.append({"role": "tool", "tool_call_id": f"c{i}", "content": "r" * 300})
        conv.append({"role": "tool", "tool_call_id": f"d{i}", "content": "s" * 300})
    assert eng.compact_conversation(conv, fixed_tokens=0)
    assert conv[0]["content"].startswith("HEAD") and "NATIVE-SUMMARY" in conv[0]["content"]
    assert "SKILL-BODY" in conv[0]["content"]
    # The tail group is whole: an assistant turn followed by both of its replies.
    assert [m["role"] for m in conv] == ["user", "assistant", "tool", "tool"]
    ids = {c["id"] for c in conv[1]["tool_calls"]}
    assert ids == {m["tool_call_id"] for m in conv[2:]}
    assert len(eng.archive) == 9


# ── native control verbs ─────────────────────────────────────────────────────

class NativeSub:
    def __init__(self, turns, tools=None, tool_results=None):
        self.turns = list(turns)
        self.sent = []
        self.tools_offered = []
        self.executed = []
        self._tools = tools or []
        self._results = tool_results or {}

    def generate_with_tools(self, token, messages, tools, **kw):
        self.sent.append([dict(m) for m in messages])
        self.tools_offered.append([t["name"] for t in tools])
        return self.turns.pop(0)

    def list_tools(self, query="", k=0, names=None, full=False, **kw):
        return list(self._tools)

    def execute_tool(self, name, args_json="", **kw):
        self.executed.append(name)
        return {"result_json": json.dumps(self._results.get(name, {"ok": name})), "result_cid": "",
                "denied": False, "deny_reason": "", "error": ""}


def _call(cid, name, **args):
    return {"id": cid, "name": name, "arguments": json.dumps(args)}


def test_native_final_answer_returns_exactly_its_argument():
    sub = NativeSub([(
        "Perfect! I now have all the information needed. Let me compile the results:",
        [_call("f1", "final_answer", answer="AUDIT-ID: 7\nCHECKSUM: 23")], "tool_use")])
    bot = Bot(agent_id="b")
    bot.substrate = sub
    res = run_think(bot, AgentTask(text="audit"), seed_system_tools=False)
    assert res.data.decode() == "AUDIT-ID: 7\nCHECKSUM: 23"
    offered = sub.tools_offered[0]
    for verb in ("final_answer", "memory_query", "resolve_cid", "yield_subgoal", "use_skill"):
        assert verb in offered, f"{verb} is not a native tool"


def test_native_plain_text_is_reprompted_once_then_accepted(caplog):
    sub = NativeSub([("first prose", [], "end_turn"), ("second prose", [], "end_turn")])
    bot = Bot(agent_id="b")
    bot.substrate = sub
    with caplog.at_level(logging.WARNING, logger="cambrian.react"):
        res = run_think(bot, AgentTask(text="t"), seed_system_tools=False)
    assert res.data.decode() == "second prose"
    assert "call final_answer" in sub.sent[1][-1]["content"]
    rec = [r for r in caplog.records if r.message == "react_inferred_answer_accepted"]
    assert rec and rec[0].answer_source == "assistant_text" and rec[0].path == "native"


def test_native_memory_query_is_answered_in_its_tool_turn():
    sub = NativeSub([("", [_call("m1", "memory_query", query="airline")], "tool_use"),
                     ("", [_call("f1", "final_answer", answer="done")], "tool_use")])
    bot = Bot(agent_id="b")
    bot.memory = types.SimpleNamespace(recall=lambda q, **kw: [{"text": "HIT-" + q, "metadata": "{}"}])
    bot.substrate = sub
    res = run_think(bot, AgentTask(text="t"), seed_system_tools=False, seed_recall=False)
    assert res.data == b"done"
    reply = [m for m in sub.sent[1] if m.get("tool_call_id") == "m1"]
    assert reply and "HIT-airline" in reply[0]["content"] and "<documents" in reply[0]["content"]


READ_TOOLS = [
    {"name": "read_a", "description": "Read A.", "schema_json": "{}", "dangerous": False},
    {"name": "read_b", "description": "Read B.", "schema_json": "{}", "dangerous": False},
    {"name": "write_c", "description": "Write C.", "schema_json": "{}", "dangerous": True},
]


def _answered(conv):
    answered = {m.get("tool_call_id") for m in conv if m.get("role") == "tool"}
    return all(c["id"] in answered for m in conv if m.get("role") == "assistant"
               for c in m.get("tool_calls") or [])


def test_read_only_calls_run_in_parallel_and_every_id_is_answered():
    sub = NativeSub([("", [_call("a", "read_a"), _call("b", "read_b")], "tool_use"),
                     ("", [_call("f", "final_answer", answer="done")], "tool_use")],
                    tools=READ_TOOLS, tool_results={"read_a": {"v": "A-RESULT"}, "read_b": {"v": "B-RESULT"}})
    bot = Bot(agent_id="b")
    bot.substrate = sub
    res = run_think(bot, AgentTask(text="t"), seed_recall=False)
    assert res.data == b"done"
    assert sorted(sub.executed) == ["read_a", "read_b"]
    conv = sub.sent[1]
    assert _answered(conv)
    replies = {m["tool_call_id"]: m["content"] for m in conv if m.get("role") == "tool"}
    assert "A-RESULT" in replies["a"] and "B-RESULT" in replies["b"]


def test_a_dangerous_call_is_not_parallelised():
    sub = NativeSub([("", [_call("a", "read_a"), _call("c", "write_c")], "tool_use"),
                     ("", [_call("f", "final_answer", answer="done")], "tool_use")],
                    tools=READ_TOOLS)
    bot = Bot(agent_id="b")
    bot.substrate = sub
    run_think(bot, AgentTask(text="t"), seed_recall=False)
    assert sub.executed == ["read_a"], "the dangerous call ran in the same turn"
    conv = sub.sent[1]
    assert _answered(conv)
    other = [m for m in conv if m.get("tool_call_id") == "c"][0]
    assert "not executed" in other["content"]


def test_missing_dangerous_field_is_not_read_only():
    from cambrian_agent_sdk.react import _is_read_only_tool
    bot = Bot(agent_id="b")
    assert not _is_read_only_tool(bot, [{"name": "x"}], "x")
    assert not _is_read_only_tool(bot, [{"name": "x", "dangerous": False, "effects": ["write"]}], "x")
    assert _is_read_only_tool(bot, [{"name": "x", "dangerous": False, "effects": ["read"]}], "x")
    assert not _is_read_only_tool(bot, [], "fetch")  # local @tool: no descriptor


# ── D14 memory notes ─────────────────────────────────────────────────────────

def test_memory_notes_slot_is_a_noop_without_the_rpc():
    bot = Bot(agent_id="b")
    bot.substrate = TextSub([_act(action="final_answer", answer="ok")])
    assert not hasattr(bot.substrate, "get_memory_notes")
    run_think(bot, AgentTask(text="t"))
    assert "<MemoryNotes" not in bot.substrate.prompts[0]


def test_memory_notes_are_read_once_and_frozen():
    calls = []

    class NotesSub(TextSub):
        def get_memory_notes(self, session_token_id=""):
            calls.append(session_token_id)
            return types.SimpleNamespace(text=f"Prefer metric units. (read {len(calls)})")

    bot = Bot(agent_id="b")
    bot.substrate = NotesSub(_fetches(2) + [_act(action="final_answer", answer="ok")])
    run_think(bot, AgentTask(text="t", session_token_id="lease-1"))
    assert calls == ["lease-1"]
    blocks = [p[p.index("<MemoryNotes"):p.index("</MemoryNotes>")] for p in bot.substrate.prompts]
    assert len(set(blocks)) == 1 and "Prefer metric units. (read 1)" in blocks[0]
    assert bot.substrate.prompts[0].index("<MemoryNotes") == 0


def test_memory_notes_degrade_on_type_error_and_rpc_error():
    class OldShape(TextSub):
        def get_memory_notes(self):  # no lease kwarg
            return "old-shape notes"

    class Broken(TextSub):
        def get_memory_notes(self, **kw):
            raise RuntimeError("Unimplemented")

    a = Bot(agent_id="a")
    a.substrate = OldShape([_act(action="final_answer", answer="ok")])
    run_think(a, AgentTask(text="t"))
    assert "old-shape notes" in a.substrate.prompts[0]
    b = Bot(agent_id="b")
    b.substrate = Broken([_act(action="final_answer", answer="ok")])
    assert run_think(b, AgentTask(text="t")).data == b"ok"
    assert "<MemoryNotes" not in b.substrate.prompts[0]


# ── configuration ────────────────────────────────────────────────────────────

def test_config_precedence_explicit_over_agent_over_env(monkeypatch):
    monkeypatch.setenv("CAMBRIAN_CONTEXT_BUDGET_TOKENS", "111")
    monkeypatch.setenv("CAMBRIAN_CONTEXT_PROTECT_TAIL", "7")
    assert ce.ContextConfig.resolve().budget_tokens == 111
    agent = types.SimpleNamespace(context_config={"budget_tokens": 222})
    cfg = ce.ContextConfig.resolve(agent)
    assert cfg.budget_tokens == 222 and cfg.protect_tail == 7
    assert ce.ContextConfig.resolve(agent, {"budget_tokens": 333}).budget_tokens == 333


def test_estimate_tokens_is_chars_over_four():
    assert ce.estimate_tokens("") == 0
    assert ce.estimate_tokens("abcd") == 1
    assert ce.estimate_tokens("abcde") == 2


@pytest.mark.parametrize("value,want", [
    ("plain", "plain"), ({"text": "d"}, "d"), (["a", {"notes": "b"}], "a\nb"),
    (types.SimpleNamespace(content="c"), "c"), (None, ""), (object(), ""),
])
def test_memory_notes_text_shapes(value, want):
    assert ce.memory_notes_text(value) == want
