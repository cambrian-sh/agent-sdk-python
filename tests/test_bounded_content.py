"""Bounded content in the agent loop (2026-11-12 owner decisions;
T-20261112-worker-resolve-cid-inline-has-no-size-cap-a and
T-20261112-summarise-a-document-step-gets-only-the-first).

A3  resolve_cid returns ONE window (a page, or the matches of a search) with a note;
    a led_ id is read through the kernel's ledger_get, never GetContextNode.
A2  an offloaded tool result renders with its preview, size, type and shape.
A5  the window size comes from ContextConfig.resolve_page_chars (kernel-served env).
B2  a shortened upstream result renders head + omission marker + tail + outline.
B3  final_answer may hand on one of the step's own tool results by answer_ref.
"""

import json

import pytest

from cambrian_agent_sdk import AgentTask, CognitiveAgent, react
from cambrian_agent_sdk import context_engine as ce
from cambrian_agent_sdk.content_window import read_window
from cambrian_agent_sdk.react import run_think
from cambrian_agent_sdk.step_dependencies import render_step_dependencies


class Bot(CognitiveAgent):
    role = "a test assistant"

    def run(self, task):
        return self.think(task)


def _ok(result_json="{}", result_cid=""):
    return {"result_json": result_json, "result_cid": result_cid, "denied": False,
            "deny_reason": "", "error": "", "arg_hash": "", "result_hash": ""}


class _LedgerKernel:
    """A fake kernel ledger: ledger_get pages a body in page_bytes pages and searches
    it, in the shape cambrian-core internal/ledger/tools.go answers."""

    def __init__(self, entries, page=16000):
        self.entries, self.page, self.calls = dict(entries), page, []

    def ledger_get(self, args):
        self.calls.append(dict(args))
        body = self.entries.get(args["id"])
        if body is None:
            return {"found": False, "id": args["id"]}
        info = read_window(args["id"], body, self.page, args.get("page"), args.get("find", ""))
        info.pop("cid")
        info["note"] = info["note"].replace("resolve_cid cid=", "ledger_get id=")
        return {"found": True, "id": args["id"], **info}


class NativeSub:
    """Native tool-calling substrate: scripted call batches; every request's messages
    are recorded."""

    def __init__(self, turns, ledger=None, tools=None):
        self.turns = list(turns)
        self.requests = []
        self.ledger = ledger
        self.tools = dict(tools or {})
        self.executed = []
        self.context_node_calls = []

    def list_tools(self, query="", k=None, full=False, **kw):
        return [{"name": n, "description": n, "schema_json": "{}", "dangerous": False}
                for n in self.tools]

    def generate_with_tools(self, session_token_id=None, messages=None, tools=None, **kw):
        self.requests.append([dict(m) for m in messages])
        calls = self.turns.pop(0)
        if callable(calls):
            calls = calls(self)
        return ("", [{"id": f"c{len(self.requests)}-{i}", "name": n, "arguments": json.dumps(a)}
                     for i, (n, a) in enumerate(calls)], "tool_use")

    def execute_tool(self, name, args_json="{}", **kw):
        args = json.loads(args_json or "{}")
        self.executed.append((name, args))
        if name == "ledger_get" and self.ledger is not None:
            return _ok(json.dumps(self.ledger.ledger_get(args)))
        out = self.tools.get(name)
        return out(args) if callable(out) else (out or _ok())

    def get_context_node(self, cid, session_token_id=""):
        self.context_node_calls.append(cid)
        return None


def _tool_turns(req):
    return [m for m in req if m.get("role") == "tool"]


# ── A3: resolve_cid is windowed ───────────────────────────────────────────────

def test_resolve_cid_inline_is_windowed_for_a_ledger_id():
    body = "".join(f"line {i:06d} " for i in range(30000))  # ~360k chars
    led = _LedgerKernel({"led_big": body})
    sub = NativeSub([[("resolve_cid", {"cid": "led_big", "as": "inline"})],
                     [("final_answer", {"answer": "ok"})]], ledger=led)
    bot = Bot(agent_id="b")
    bot.substrate = sub
    res = run_think(bot, AgentTask(text="t"), seed_recall=False)
    assert res.data == b"ok"
    reply = _tool_turns(sub.requests[-1])[-1]["content"]
    assert len(reply) <= 16000 + 1500, len(reply)
    assert f"{len(body)} chars total, showing 1–16000 (page 1 of" in reply
    assert "ledger_get id='led_big' page=2" in reply
    # One read pathway: ledger_get, never GetContextNode, for a led_ id.
    assert led.calls == [{"id": "led_big"}] and sub.context_node_calls == []


def test_resolve_cid_inline_is_windowed_for_a_content_store_cid():
    body = "B" * 300_000

    class Store(NativeSub):
        def get_context_node(self, cid, session_token_id=""):
            self.context_node_calls.append(cid)
            return type("N", (), {"data": body.encode()})()

    sub = Store([[("resolve_cid", {"cid": "abc123", "as": "inline", "page": 2})],
                 [("final_answer", {"answer": "ok"})]])
    bot = Bot(agent_id="b")
    bot.substrate = sub
    run_think(bot, AgentTask(text="t"), seed_recall=False,
              context_config={"resolve_page_chars": 5000})
    reply = _tool_turns(sub.requests[-1])[-1]["content"]
    assert reply.count("B") <= 5000 + 50
    assert "300000 chars total, showing 5001–10000 (page 2 of 60)" in reply
    assert "resolve_cid cid='abc123' page=3" in reply


def test_resolve_cid_find_returns_snippets_with_pages():
    body = ("filler " * 3000) + "Leipzig is the finding " + ("tail " * 3000)
    info = read_window("cid1", body, 4000, query="leipzig")
    assert info["match_count"] == 1
    m = info["matches"][0]
    assert m["page"] == body.lower().index("leipzig") // 4000 + 1
    assert "Leipzig" in m["snippet"]
    assert "page=<n>" in info["note"]
    # Through the loop, for a led_ id: find reaches ledger_get.
    led = _LedgerKernel({"led_doc": body}, page=4000)
    sub = NativeSub([[("resolve_cid", {"cid": "led_doc", "find": "Leipzig"})],
                     [("final_answer", {"answer": "ok"})]], ledger=led)
    bot = Bot(agent_id="b")
    bot.substrate = sub
    run_think(bot, AgentTask(text="t"), seed_recall=False)
    assert led.calls == [{"id": "led_doc", "find": "Leipzig"}]
    reply = _tool_turns(sub.requests[-1])[-1]["content"]
    assert f"[page {m['page']}," in reply and "Leipzig" in reply


def test_resolve_offload_of_a_ledger_id_is_a_note_not_a_copy():
    led = _LedgerKernel({"led_x": "x" * 50})
    sub = NativeSub([[("resolve_cid", {"cid": "led_x", "as": "offload"})],
                     [("final_answer", {"answer": "ok"})]], ledger=led)
    bot = Bot(agent_id="b")
    bot.substrate = sub
    run_think(bot, AgentTask(text="t"), seed_recall=False)
    reply = _tool_turns(sub.requests[-1])[-1]["content"]
    assert "already a reference" in reply and led.calls == []


def test_resolve_window_stays_under_the_oversize_cap():
    cfg = ce.ContextConfig(resolve_page_chars=50000, tool_output_max_chars=24000)
    assert react._resolve_page_chars(cfg) < cfg.tool_output_max_chars


# ── A3: no compaction/resolve ping-pong (p12) ─────────────────────────────────

def test_resolve_compacted_cid_no_pingpong():
    """p12: a 37,345-char output was compacted, re-opened inline, compacted again
    under a NEW cid, re-opened again (~5 times). Now a re-opened window is compacted
    back to the ORIGINAL cid and page, and the same body always keeps one cid."""
    doc = "".join(f"para {i:05d} " for i in range(3395))  # 37,345 chars
    assert len(doc) == 37345
    eng = ce.DefaultContextEngine(ce.ContextConfig(budget_tokens=3000, tool_output_compact_chars=500,
                                                   protect_tail=1, phase2_enabled=False))
    conv = [{"role": "user", "content": "task"},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "t1", "name": "read_pdf", "arguments": "{}"}]},
            {"role": "tool", "tool_call_id": "t1", "content": doc},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "t2", "name": "noop", "arguments": "{}"}]},
            {"role": "tool", "tool_call_id": "t2", "content": "ok"}]
    assert eng.compact_conversation(conv, fixed_tokens=0)
    cid = conv[2]["content"].split("cid ")[1].split(".")[0]
    assert eng.resolve_local(cid) == doc
    # The model re-opens page 1: one window, noted as such.
    info = read_window(cid, eng.resolve_local(cid), 16000)
    conv += [{"role": "assistant", "content": "", "tool_calls": [{"id": "r1", "name": "resolve_cid", "arguments": "{}"}]},
             {"role": "tool", "tool_call_id": "r1", "content": info["text"]}]
    eng.note_resolve_window("r1", cid, info["page"], info["pages"], info["total_chars"])
    conv += [{"role": "assistant", "content": "", "tool_calls": [{"id": "t3", "name": "noop", "arguments": "{}"}]},
             {"role": "tool", "tool_call_id": "t3", "content": "ok"}]
    eng.compact_conversation(conv, fixed_tokens=0)
    placeholder = conv[6]["content"]
    assert placeholder.startswith("[page 1 of 3 of cid " + cid)
    assert f'resolve_cid (cid "{cid}", page 1)' in placeholder
    # No new cid was minted for the window: nothing new to re-open.
    assert set(eng._local) == {cid}
    # And storing the same body again returns the same cid.
    assert eng._store(doc) == cid


# ── A2: an offloaded result is described ──────────────────────────────────────

def test_offloaded_result_renders_preview_and_size():
    desc = {"_offloaded": {"ref": "led_960e", "tool": "mcp:code-sandbox/execute_code",
                           "size_bytes": 334201, "content_type": "application/json",
                           "shape": "MCP content array: 3 blocks (text, text, text)",
                           "preview": "Chart created successfully!", "pages": 21,
                           "read": "ledger_get id='led_960e' page=1..21"}}

    class Sub:
        def execute_tool(self, *a, **kw):
            return _ok(json.dumps(desc), "led_960e")

    agent = type("A", (), {"substrate": Sub()})()
    out = react._call_system_tool(agent, "mcp:code-sandbox/execute_code", {})
    assert out["offloaded"] is True and out["cid"] == "led_960e"
    assert out["size_bytes"] == 334201 and out["preview"] == "Chart created successfully!"
    assert out["shape"].startswith("MCP content array") and out["content_type"] == "application/json"
    assert "resolve_cid cid='led_960e' page=1" in out["note"]
    assert set(out) != {"cid", "tool"}
    # The text path's card carries the preview too (not just "offloaded").
    card = ce.ToolCard.from_result("mcp:code-sandbox/execute_code", {}, out)
    assert "Chart created successfully!" in card.summary and "334201 bytes" in card.summary
    # Its ledger id counts as one of the run's own results (answer_ref).
    assert react._ledger_refs_of(out) == ["led_960e"]


# ── A5: kernel-served limits ──────────────────────────────────────────────────

def test_context_config_resolve_page_chars_env(monkeypatch):
    assert ce.ContextConfig().resolve_page_chars == 16000
    monkeypatch.setenv("CAMBRIAN_CONTEXT_RESOLVE_PAGE_CHARS", "8000")
    monkeypatch.setenv("CAMBRIAN_CONTEXT_TOOL_OUTPUT_MAX_CHARS", "30000")
    monkeypatch.setenv("CAMBRIAN_CONTEXT_TOOL_OUTPUT_PREVIEW_CHARS", "700")
    monkeypatch.setenv("CAMBRIAN_CONTEXT_BUDGET_TOKENS", "40000")
    cfg = ce.ContextConfig.resolve()
    assert (cfg.resolve_page_chars, cfg.tool_output_max_chars, cfg.tool_output_preview_chars,
            cfg.budget_tokens) == (8000, 30000, 700, 40000)


# ENG-595 (owner decision 2026-11-14): execution.lane.context reaches the agent as
# kernel-stamped ``_context_*`` handoff keys, which win over every other layer.
def test_context_config_handoff_keys_win(monkeypatch):
    monkeypatch.setenv("CAMBRIAN_CONTEXT_BUDGET_TOKENS", "40000")
    monkeypatch.setenv("CAMBRIAN_CONTEXT_PROTECT_TAIL", "2")
    task = AgentTask(text="t", context={"_context_budget_tokens": "48000",
                                        "_context_protect_tail": "5",
                                        "_context_tool_output_compact_chars": "nope"})
    agent = type("A", (), {"context_config": {"budget_tokens": 1000, "keep_recent_tool_results": 7}})()
    cfg = ce.ContextConfig.resolve(agent, {"protect_tail": 9}, task=task)
    assert cfg.budget_tokens == 48000           # handoff > agent config > env
    assert cfg.protect_tail == 5                # handoff > explicit > env
    assert cfg.keep_recent_tool_results == 7    # not stamped: the lower layers stand
    assert cfg.tool_output_compact_chars == 1200  # a bad value is skipped, default stands
    # Without a task (or without the keys) nothing changes.
    assert ce.ContextConfig.resolve(task=AgentTask(text="t")).budget_tokens == 40000


def test_run_think_builds_its_engine_from_the_handoff_context():
    seen = {}

    def factory(config, embed_fn, offload_fn, summariser):
        seen["cfg"] = config
        return ce.DefaultContextEngine(config, offload_fn=offload_fn, summariser=summariser,
                                       embed_fn=embed_fn)

    sub = NativeSub([[("final_answer", {"answer": "ok"})]])
    bot = Bot(agent_id="b")
    bot.substrate = sub
    res = run_think(bot, AgentTask(text="t", context={"_context_budget_tokens": "12345",
                                                      "_context_keep_recent_tool_results": "4"}),
                    seed_recall=False, context_engine=factory)
    assert res.data == b"ok"
    assert seen["cfg"].budget_tokens == 12345
    assert seen["cfg"].keep_recent_tool_results == 4


# ── B2: head + omission marker + tail + outline ───────────────────────────────

def test_shortened_dependency_renders_head_marker_tail_outline():
    deps = [{"step": 0, "objective": "extract the report", "status": "completed",
             "cid": "led_49e3", "bytes": 37444, "shortened": True,
             "content": "HEAD of the report", "tail": "END: Leipzig finding",
             "omitted_from": 9000, "omitted_to": 33000, "page_bytes": 16000,
             "outline": [{"heading": "# Findings", "page": 1}, {"heading": "## Leipzig", "page": 3}],
             "read": "ledger_get id='led_49e3' page=1"}]
    out = render_step_dependencies(deps)
    head, marker, tail = out.index("HEAD of the report"), out.index("[OMITTED"), out.index("END: Leipzig")
    assert head < marker < tail
    assert "Pages 1–3 of this result are NOT shown" in out
    assert "ledger_get id='led_49e3' page=1" in out and "resolve_cid cid='led_49e3' page=1" in out
    assert "(bytes 9000–33000 of 37444)" in out
    assert "- ## Leipzig (page 3)" in out and out.index("Outline") > tail


def test_shortened_dependency_from_an_old_kernel_keeps_the_old_rendering():
    deps = [{"step": 0, "objective": "o", "status": "completed", "cid": "abc",
             "bytes": 100, "shortened": True, "content": "HEAD"}]
    out = render_step_dependencies(deps)
    assert "[shortened: 4 of 100 characters shown; re-open it with resolve_cid" in out
    assert "OMITTED" not in out


# ── B3: answer_ref ────────────────────────────────────────────────────────────

def _reader_tools():
    return {"read_pdf": lambda args: _ok(json.dumps({"text": "report body"}), "led_own")}


def test_final_answer_answer_ref_for_an_own_result_sets_the_handoff_key():
    sub = NativeSub([[("read_pdf", {"path": "r.pdf"})],
                     [("final_answer", {"answer": "the report text", "answer_ref": "led_own"})]],
                    tools=_reader_tools())
    bot = Bot(agent_id="b")
    bot.substrate = sub
    res = run_think(bot, AgentTask(text="extract the report"), seed_recall=False)
    assert res.context[react.ANSWER_REF_KEY] == "led_own"
    assert res.data == b"the report text"


def test_final_answer_answer_ref_unknown_is_refused_and_the_run_continues():
    sub = NativeSub([[("read_pdf", {"path": "r.pdf"})],
                     [("final_answer", {"answer": "x", "answer_ref": "led_someone_else"})],
                     [("final_answer", {"answer": "typed out"})]],
                    tools=_reader_tools())
    bot = Bot(agent_id="b")
    bot.substrate = sub
    res = run_think(bot, AgentTask(text="extract the report"), seed_recall=False)
    assert res.data == b"typed out" and react.ANSWER_REF_KEY not in res.context
    refusal = _tool_turns(sub.requests[-1])[-1]["content"]
    assert "not one of your own tool results" in refusal and "led_own" in refusal


def test_final_answer_tool_offers_answer_ref_only_without_an_envelope():
    defs, _ = react.build_tool_definitions(Bot(agent_id="b"), [], include_final=True)
    fa = next(d for d in defs if d["name"] == "final_answer")
    assert "answer_ref" in fa["parameters"]["properties"]
    defs, _ = react.build_tool_definitions(Bot(agent_id="b"), [], include_final=True,
                                           final_envelope=True)
    fa = next(d for d in defs if d["name"] == "final_answer")
    assert "answer_ref" not in json.dumps(fa["parameters"])


@pytest.mark.parametrize("out,refs", [
    ({"x": 1, "_ledger_ref": "led_a"}, ["led_a"]),
    ({"cid": "ctx-123"}, []),
    ("text", []),
])
def test_ledger_refs_of(out, refs):
    assert react._ledger_refs_of(out) == refs
