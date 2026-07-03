"""ADR-0041 D1/D3: the Local Recurrent Workspace typed buffers + invocation cards."""

import json
from datetime import datetime, timezone

from cambrian_agent_sdk.working_memory import (
    ToolCard,
    WorkingMemory,
    _condense_dict,
    _status_of,
    _summarize,
    memory_provenance_attrs,
    resolve_content,
)


def _meta(**kw):
    """A recall result whose metadata is a JSON string (the kernel wire form)."""
    return {"text": "f", "metadata": json.dumps(kw)}


# ── ADR-0048 A1 D9: author attribution ──────────────────────────────────────

def test_provenance_system_author_renders_as_system():
    # A System source_agent (the agent's own auto-recorded step echo) renders as
    # author='system' so the agent can discount it vs. a real grounded fact.
    attrs = memory_provenance_attrs(_meta(source_agent="System", session_id="s1"))
    assert "author='system'" in attrs
    assert "session='s1'" in attrs
    assert attrs.startswith("source='LTM'")


def test_provenance_real_agent_author_preserved():
    attrs = memory_provenance_attrs(_meta(source_agent="planner_x"))
    assert "author='planner_x'" in attrs


def test_provenance_absent_fields_omitted():
    # No empty author='' / session='' — absent keys produce no attribute at all.
    attrs = memory_provenance_attrs({"text": "f"})
    assert attrs == "source='LTM'"


# ── ADR-0048 A1 D10: freshness signal ───────────────────────────────────────

def test_freshness_low_activation_is_stale():
    # Low stored activation ⇒ stale regardless of age.
    attrs = memory_provenance_attrs(_meta(_activation_strength=0.02))
    assert "freshness='stale'" in attrs


def test_freshness_old_but_active_is_aging():
    old = "2000-01-01T00:00:00Z"  # definitively > 30 days old
    attrs = memory_provenance_attrs(_meta(_activation_strength=0.9, _last_accessed_at=old))
    assert "freshness='aging'" in attrs


def test_freshness_recent_and_active_is_fresh():
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    attrs = memory_provenance_attrs(_meta(_activation_strength=0.9, _last_accessed_at=now))
    assert "freshness='fresh'" in attrs
    assert "age='0d'" in attrs


def test_written_date_rendered_date_only():
    attrs = memory_provenance_attrs(_meta(_created_at="2026-05-07T12:34:56Z", _activation_strength=0.5))
    assert "written='2026-05-07'" in attrs  # date only, time stripped


def test_no_temporal_fields_no_freshness_attr():
    # When the kernel sent no temporal/activation fields, no freshness is asserted.
    attrs = memory_provenance_attrs(_meta(source_agent="x"))
    assert "freshness=" not in attrs
    assert "age=" not in attrs


class _FakeNode:
    def __init__(self, data):
        self.data = data


class _FakeSubstrateCAS:
    """Serves ContentStore nodes by CID via get_context_node (ADR-0022)."""

    def __init__(self, store):
        self._store = store  # cid -> bytes

    def get_context_node(self, cid):
        d = self._store.get(cid)
        return _FakeNode(d) if d is not None else None


def test_tool_card_render_carries_provenance():
    c = ToolCard.from_result("read_file", {"path": "/x"}, {"content": "hi"})
    r = c.render()
    assert "read_file" in r           # tool name
    assert '"path": "/x"' in r        # args — what it was called with
    assert "status='ok'" in r         # outcome status
    assert "hi" in r                  # result summary


def test_render_condenses_large_args_and_leads_with_result():
    """#2: a write_file card must NOT re-render its multi-KB content every turn.
    The big `content` collapses to a marker; `path` stays; the result leads; and the
    block is framed as an executed <step>, not a proposed <tool> call."""
    big_content = "**LLMs**\n" + ("Large language models are transformer networks. " * 80)
    c = ToolCard.from_result(
        "mcp:filesystem/write_file",
        {"path": "cambrian1.txt", "content": big_content},
        {"content": "Successfully wrote to cambrian1.txt"},
    )
    r = c.render()

    assert big_content not in r                       # the 3 KB payload is NOT carried
    assert f"<{len(big_content)} chars>" in r          # collapsed to a size marker
    assert '"path": "cambrian1.txt"' in r              # small arg kept verbatim
    assert "Successfully wrote to cambrian1.txt" in r  # result preserved
    assert "<step action=" in r and "</step>" in r     # framed as an executed step
    # result is highlighted: it appears BEFORE the args block
    assert r.index("<result>") < r.index("<args>")


def test_status_derivation():
    assert _status_of({"error": "boom"}) == "error"
    assert _status_of({"denied": True}) == "denied"
    assert _status_of({"content": "ok"}) == "ok"
    assert _status_of(5) == "ok"


def test_summary_truncates_large_results_without_llm():
    big = {"data": "x" * 2000}
    s = _summarize(big, max_chars=100)
    assert len(s) < 200
    assert "chars total" in s          # length marker on truncation


def test_assemble_joins_text_and_cards():
    wm = WorkingMemory()
    wm.add_text("<memory>fact</memory>")
    wm.add_tool_card(ToolCard.from_result("add", {"a": 1}, 2))
    out = wm.assemble()
    assert "fact" in out
    assert "add" in out and "status='ok'" in out
    assert len(wm.cards) == 1          # episodic buffer exposes the cards


def test_non_dict_args_are_wrapped():
    c = ToolCard.from_result("t", "rawstring", {})
    assert c.args == {"value": "rawstring"}


def test_failure_card_keeps_args():
    """A failed call's card carries status='error' AND the args — the provenance
    the recurrence gate (0041-04) will read to avoid re-issuing the same call."""
    c = ToolCard.from_result("execute_command", {"command": "find . -size +1M"},
                             {"error": "DEADLINE_EXCEEDED", "tool": "execute_command"})
    r = c.render()
    assert "status='error'" in r
    assert "find . -size +1M" in r


# ── ADR-0041 D3: offload + drill-down + opt-in summary ──────────────────────────

def test_offloaded_result_keeps_cid_and_marker_not_payload():
    """A kernel-offloaded system-tool result (dict with a cid) keeps {marker + cid},
    not the (absent) payload."""
    card = ToolCard.from_result("read_file", {"path": "/big"}, {"cid": "Qm1", "tool": "read_file"})
    assert card.cid == "Qm1"
    assert "offloaded" in card.summary.lower()
    assert card.status == "ok"
    assert "Qm1" in card.render()


def test_resolve_content_drills_down_by_cid():
    card = ToolCard.from_result("read_file", {"path": "/big"}, {"cid": "Qm1"})
    sub = _FakeSubstrateCAS({"Qm1": b"the full big content"})
    assert resolve_content(card, sub) == "the full big content"


def test_resolve_content_none_for_inline_card():
    card = ToolCard.from_result("add", {"a": 1}, 2)  # inline result, no cid
    assert card.cid is None
    assert resolve_content(card, _FakeSubstrateCAS({})) is None


def test_opt_in_summarizer_used_only_for_heavy_results():
    calls = []

    def fake_sum(s):
        calls.append(s)
        return "GIST"

    # Small result: heuristic, summarizer NOT called (zero LLM cost).
    small = ToolCard.from_result("t", {}, "small", summarizer=fake_sum)
    assert small.summary == "small"
    assert calls == []

    # Heavy result: opt-in summarizer invoked once.
    big = ToolCard.from_result("t", {}, "x" * 5000, summarizer=fake_sum)
    assert big.summary == "GIST"
    assert len(calls) == 1


def test_heavy_result_without_summarizer_truncates_no_llm():
    card = ToolCard.from_result("t", {}, "x" * 5000)  # no summarizer
    assert "chars total" in card.summary           # heuristic truncation marker
    assert len(card.summary) < 700                 # bounded, not the full 5000


# ── ADR-0041 D2: bounded, relevance-ranked + pinned assembly ────────────────────

def _kw_embed(text):
    """Deterministic 2-D 'embedding': [has 'alpha', has 'beta']."""
    return [1.0 if "alpha" in text else 0.0, 1.0 if "beta" in text else 0.0]


def test_under_cap_keeps_all_entries_in_order():
    wm = WorkingMemory(embed_fn=None, cap=10)
    wm.add_text("a")
    wm.add_text("b")
    assert wm.assemble() == "a\nb"          # behavior-preserving below cap


def test_assemble_bounds_and_ranks_by_relevance():
    wm = WorkingMemory(embed_fn=_kw_embed, cap=2)
    wm.add_tool_card(ToolCard.from_result("beta0", {}, "beta result 0"))
    wm.add_tool_card(ToolCard.from_result("alpha0", {}, "alpha result 0"))
    wm.add_tool_card(ToolCard.from_result("alpha1", {}, "alpha result 1"))  # last → pinned
    out = wm.assemble(intent_vec=[1.0, 0.0])  # intent ~ alpha
    assert out.count("<step") == 2            # bounded to cap
    assert "alpha1" in out                    # the pinned most-recent
    assert "alpha0" in out                    # the relevant one kept over beta
    assert "beta" not in out                  # the irrelevant one evicted


def test_assemble_numbered_renders_ordered_trajectory():
    # v2: numbered mode prefixes kept entries with ordinals (oldest first) so the
    # assembly reads as a sequence, not a flat pile. Default mode stays unnumbered
    # (each <step> carries its own ``n`` attribute in the new rich XML).
    wm = WorkingMemory(embed_fn=None, cap=10)
    wm.add_text("<memory>recalled fact</memory>")
    wm.add_tool_card(ToolCard.from_result("write_file", {"path": "a.txt"}, {"written": "a.txt"}))

    plain = wm.assemble()
    assert not plain.lstrip().startswith("1. ")     # default unchanged

    numbered = wm.assemble(numbered=True)
    assert numbered.startswith("1. ")                # oldest entry is item 1
    # The action is item 2; the v2 rich XML uses ``<step n=0 type="tool_call" tool='write_file' ...>``.
    assert "2. <step" in numbered
    assert "type=\"tool_call\"" in numbered
    assert "tool='write_file'" in numbered


def test_pinned_failure_retained_even_at_low_relevance():
    wm = WorkingMemory(embed_fn=_kw_embed, cap=2)
    # A FAILED card irrelevant to the 'alpha' intent — must survive as a pin so the
    # recurrence gate (0041-04) keeps its evidence.
    wm.add_tool_card(ToolCard.from_result("execute_command", {"command": "gamma"},
                                          {"error": "boom"}))
    wm.add_tool_card(ToolCard.from_result("alpha0", {}, "alpha 0"))
    wm.add_tool_card(ToolCard.from_result("alpha1", {}, "alpha 1"))
    out = wm.assemble(intent_vec=[1.0, 0.0])
    assert "status='error'" in out            # pinned despite ~zero relevance to alpha


def test_recency_fallback_without_embedder():
    wm = WorkingMemory(embed_fn=None, cap=2)
    for i in range(5):
        wm.add_text(f"entry{i}")
    out = wm.assemble()                        # no embedder, no intent
    assert "entry4" in out and "entry3" in out  # most recent kept
    assert "entry0" not in out
    assert out.count("entry") == 2


def test_token_bound_many_cards_stay_capped():
    """Regression for O(N²): assembled prompt stays ≤ cap regardless of N rounds."""
    wm = WorkingMemory(embed_fn=None, cap=10)
    for i in range(50):
        wm.add_tool_card(ToolCard.from_result(f"t{i}", {}, f"r{i}"))
    out = wm.assemble()
    assert out.count("<step") <= 10


# ── #2: content-aware ToolCard summarization (errors, listings) ──────────────────

def test_condense_error_keeps_only_message_not_full_blob():
    big_err = {"error": "ENOENT: no such file or directory, open " + "x" * 4000, "tool": "read_file"}
    s = _summarize(big_err)
    assert s.startswith("error: ENOENT")
    assert "tool read_file" in s
    assert len(s) < 400              # the 4000-char blob is condensed, not truncated-at-400
    assert "chars total" not in s    # it's a semantic condensation, not a head-truncation


def test_condense_listing_gives_count_and_head_not_whole_list():
    listing = {"matches": [f"/path/file{i}.go" for i in range(200)]}
    s = _condense_dict(listing)
    assert s is not None
    assert s.startswith("200 matches:")
    assert "(+195 more)" in s
    assert len(s) < 350              # 200 paths are NOT dumped


def test_condense_returns_none_for_unrecognized_shape():
    # A write_file-style result has no recognized list/error key → falls back to JSON.
    assert _condense_dict({"written": "x.txt", "bytes": 10}) is None
    card = ToolCard.from_result("write_file", {"path": "x.txt"}, {"written": "x.txt", "bytes": 10})
    assert "x.txt" in card.summary   # informative content preserved


# ── #3: assemble() dedups the seed/recall/step-result triplication ───────────────

def test_assemble_dedups_near_identical_text_blocks():
    payload = (
        "Observations: the task requires a comprehensive explanation of LLMs, agents "
        "and tools, and their interconnection in a modern AI stack. Reasoning: define "
        "the core technology, the orchestration paradigm, and the interface layer. "
        "Conclusion: LLMs are transformer networks, agents loop over them, tools extend them."
    )
    wm = WorkingMemory(embed_fn=None, cap=10)
    wm.add_text(f"<workspace>{payload}</workspace>")                  # seed A
    wm.add_text(f"<memory>step_0: {payload} | Step 0 result: {payload}</memory>")  # recall (reordered)
    wm.add_text(f"<memory>Step 0 result: {payload}</memory>")        # later memory_query (same)
    wm.add_text("<note>distinct short note that must survive</note>")  # distinct → kept

    out = wm.assemble()
    # The big payload appears once, not three times.
    assert out.count("comprehensive explanation of LLMs") == 1
    # The distinct note is never deduped.
    assert "distinct short note that must survive" in out


def test_assemble_does_not_dedup_distinct_small_notes():
    wm = WorkingMemory(embed_fn=None, cap=10)
    wm.add_text("<note>first finding</note>")
    wm.add_text("<note>second finding</note>")
    out = wm.assemble()
    assert "first finding" in out and "second finding" in out


# ── R7: content-offload for large text entries (pointer + drill-down) ─────────────

def test_large_text_offloaded_to_pointer_when_offloader_wired():
    store = {}

    def offload(text):
        store["txtcid"] = text.encode()
        return "txtcid"

    wm = WorkingMemory(embed_fn=None, offload_fn=offload, text_inline_cap=50)
    big = "Observations: " + ("LLMs are transformer networks. " * 50)
    wm.add_text(big)

    out = wm.assemble()
    assert "<offloaded" in out and "cid='txtcid'" in out
    assert big not in out                 # the full payload is NOT carried in the prompt
    assert "Observations" in out          # but the gist locates it for the agent
    # Drill-down recovers the full text by cid (the agent can still fetch it).
    assert resolve_content(wm._entries[0], _FakeSubstrateCAS(store)) == big


def test_large_text_verbatim_without_offloader_is_safe_default():
    # No offloader (no ContentStore write path) ⇒ never collapse, so a later step
    # that needs the full text (e.g. write_file) is never starved.
    wm = WorkingMemory(embed_fn=None, offload_fn=None, text_inline_cap=50)
    big = "x" * 500
    wm.add_text(big)
    assert big in wm.assemble()


def test_small_text_never_offloaded():
    wm = WorkingMemory(embed_fn=None, offload_fn=lambda t: "c", text_inline_cap=1000)
    wm.add_text("<note>tiny note</note>")
    out = wm.assemble()
    assert "tiny note" in out and "<offloaded" not in out


def test_offloaded_tool_call_carries_cid_hint():
    """v2: the offloaded <result offloaded_cid=...> block must include a
    <cid_hint> child telling the model to use the cid as {"$cid": "..."} in the
    next tool_call (issue #3 in the trajectory critique: "offloaded content is
    unknown"). The hint is IN the block, not in a separate instruction.
    """
    from cambrian_agent_sdk.working_memory import Step, render_step_xml
    s = Step(
        kind="tool_call", n=1, tool="mcp:filesystem/write_file",
        args={"path": "pong.html"}, status="ok",
        summary="5 pages, abstract mentions 91% accuracy on GAIA",
        body="", cid="abc123def", chars=7281,
    )
    xml = render_step_xml(s)
    # The cid is in the <result> attribute (single-quoted, per Python repr).
    assert "offloaded_cid='abc123def'" in xml
    # The summary is preserved (the one-liner the model can read).
    assert "5 pages, abstract" in xml
    # The cid_hint child is right inside the result block — actionable.
    assert "<cid_hint>" in xml
    assert '{"$cid": "abc123def"}' in xml
    # The hint tells the model the kernel resolves it (so it knows the
    # pattern is safe; it does NOT have to be in its working memory).
    assert "kernel resolves" in xml
    # The hint also tells the model NOT to re-emit the body (the common
    # failure mode the field caught).
    assert "do NOT re-emit" in xml


def test_offloaded_text_entry_carries_cid_hint():
    """Symmetry: the offloaded TextEntry (workspace seed, large note) also
    carries the cid_hint. Same pattern, same words.
    """
    from cambrian_agent_sdk.working_memory import TextEntry, render_entry_xml
    entry = TextEntry(
        content="<!DOCTYPE html>... [truncated] ...",
        cid="xyz789",
        summary="Here is a complete HTML Pong game.",
    )
    xml = render_entry_xml(entry)
    assert "cid='xyz789'" in xml
    assert "<cid_hint>" in xml
    assert '{"$cid": "xyz789"}' in xml
    assert "do NOT re-emit" in xml


# ──────────────────────────────────────────────────────────────────────
# v2 tool_call body inlining — the fix for the "model re-reads 4 times"
# field issue (the model couldn't see the full content because the result
# was truncated to 400 chars). Small results inline; large results show a
# preview + an offload hint; offloaded (kernel-cid) results use the cid branch.
# ──────────────────────────────────────────────────────────────────────

def test_tool_call_step_inlines_small_body_in_full():
    """v2: tool results ≤ 4000 chars inline in the trajectory so the model
    doesn't need to re-read. This is the path that closed the field's
    'model re-reads 4 times' gap (a 3475-char HTML was being truncated
    to 400 chars; the model kept re-reading)."""
    import html as _html
    from cambrian_agent_sdk.working_memory import Step, render_step_xml
    body = "<!DOCTYPE html>... 3475 chars ...</html>"
    s = Step(
        kind="tool_call", n=1, tool="mcp:filesystem/read_file",
        args={"path": "pong.html"}, status="ok",
        summary="read_file result", body=body, cid="", chars=len(body),
    )
    xml = render_step_xml(s)
    # The FULL body is in the prompt (XML-escaped) — not a 400-char preview.
    assert _html.escape(body) in xml
    # The status is visible.
    assert "status='ok'" in xml
    # There's no offload hint (the body is inlined, so no need).
    assert "<body_preview>" not in xml
    assert "<offload_hint>" not in xml


def test_tool_call_step_offloads_large_body_with_preview():
    """v2: tool results > 4000 chars show a 200-char preview + an offload
    hint (the model can either reason from the preview or act on the body
    via a tool_call). The hint tells the model NOT to re-read the same path."""
    from cambrian_agent_sdk.working_memory import Step, render_step_xml
    long_body = ("Large tool result content. " * 300).strip()  # ~7500 chars
    s = Step(
        kind="tool_call", n=1, tool="mcp:filesystem/read_file",
        args={"path": "big.txt"}, status="ok",
        summary="read_file result (truncated)", body=long_body, cid="",
        chars=len(long_body),
    )
    xml = render_step_xml(s)
    # The full body is NOT inlined.
    assert long_body not in xml
    # A 200-char preview IS shown.
    assert "<body_preview>" in xml
    # The offload hint tells the model to NOT re-read the same path.
    assert "<note>" in xml
    assert "recurrence gate blocks re-reading" in xml


def test_tool_call_step_kernel_offloaded_branch_unchanged():
    """v2: when the kernel offloads the body (cid set), the trajectory shows
    the cid_hint + summary (the model passes the cid to the next tool_call)."""
    from cambrian_agent_sdk.working_memory import Step, render_step_xml
    s = Step(
        kind="tool_call", n=1, tool="mcp:filesystem/read_file",
        args={"path": "big.txt"}, status="ok",
        summary="read_file result (offloaded)", body="",
        cid="abc123", chars=7281,
    )
    xml = render_step_xml(s)
    assert "offloaded_cid='abc123'" in xml
    assert "<cid_hint>" in xml
    assert '{"$cid": "abc123"}' in xml
    # The body is NOT inlined (it's offloaded).
    assert "<body_preview>" not in xml


def test_offload_fn_failure_degrades_to_verbatim():
    def boom(_):
        raise RuntimeError("offload down")

    wm = WorkingMemory(embed_fn=None, offload_fn=boom, text_inline_cap=10)
    big = "y" * 200
    wm.add_text(big)
    assert big in wm.assemble()           # degraded to verbatim, never crashed


# ── D7: supersession-collapse (consumed specs, superseded failures) ───────────────

def test_collapse_consumed_tool_spec_immediately():
    wm = WorkingMemory(embed_fn=None, cap=10)
    wm.add_text("<tool_spec name='firecrawl_agent'>\n  <description>long async polling prose...</description>\n</tool_spec>")
    wm.add_tool_card(ToolCard.from_result("firecrawl_agent", {"prompt": "x"}, {"job": "123"}))
    out = wm.assemble()
    assert "spec for 'firecrawl_agent' consumed" in out
    assert "async polling prose" not in out          # the full spec is collapsed


def test_collapse_superseded_failure_after_hysteresis():
    wm = WorkingMemory(embed_fn=None, cap=10)
    wm.add_tool_card(ToolCard.from_result("write_file", {"path": "a.txt"},
                                          {"error": "ENOENT", "tool": "write_file"}))   # fail
    wm.add_tool_card(ToolCard.from_result("write_file", {"path": "a.txt"}, {"written": "a.txt"}))  # success
    wm.add_text("<note>a later note</note>")          # success is no longer most-recent
    out = wm.assemble()
    assert "'write_file' failed, then succeeded" in out
    assert "status='error'" not in out               # the failed card collapsed to the note


def test_failure_kept_while_success_is_most_recent():
    wm = WorkingMemory(embed_fn=None, cap=10)
    wm.add_tool_card(ToolCard.from_result("write_file", {"path": "a.txt"},
                                          {"error": "ENOENT", "tool": "write_file"}))
    wm.add_tool_card(ToolCard.from_result("write_file", {"path": "a.txt"}, {"written": "a.txt"}))  # most-recent
    out = wm.assemble()
    assert "failed, then succeeded" not in out        # hysteresis: kept this turn
    assert "status='error'" in out                    # failed card still rendered


def test_collapse_is_non_destructive():
    wm = WorkingMemory(embed_fn=None, cap=10)
    wm.add_text("<tool_spec name='x'>spec</tool_spec>")
    wm.add_tool_card(ToolCard.from_result("x", {}, {"ok": 1}))
    wm.add_text("<note>note</note>")
    _ = wm.assemble()
    assert len(wm.cards) == 1                          # episodic buffer intact
    assert len(wm._entries) == 3                       # nothing removed from the buffer
