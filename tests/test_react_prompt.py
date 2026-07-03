"""The ReAct action menu must explain each action and steer the agent to ground
its answer in retrieved memory before answering from its own knowledge.

Regression for the observed failure: agents jumped straight to final_answer and
never issued memory_query because the schema listed the actions without
explaining what they were for or when to use them.
"""

from cambrian_agent_sdk.react import build_output_schema, _render_recall


class _FakeRegistry:
    def specs(self):
        return []


class _FakeAgent:
    def __init__(self):
        self.tools = _FakeRegistry()


def test_action_menu_explains_each_action():
    schema = build_output_schema(_FakeAgent()).lower()
    # Each action is named AND described, not just listed.
    assert "memory_query" in schema
    assert "tool_call" in schema
    assert "final_answer" in schema
    # Explanations present (purpose of each), not just the JSON shape.
    assert "retriev" in schema or "recall" in schema or "knowledge base" in schema
    assert "tool" in schema


def test_action_menu_instructs_memory_before_answering():
    schema = build_output_schema(_FakeAgent()).lower()
    # The agent must be told to ground claims via memory_query before final_answer,
    # rather than trusting its own training.
    assert "memory_query" in schema
    assert "before" in schema  # "...before you answer / before final_answer"
    # Discourages answering from the model's own knowledge unverified.
    assert "own knowledge" in schema or "your own knowledge" in schema or "training" in schema


def test_action_menu_steers_find_tools_on_capability_gap():
    """Regression: small models declared tasks impossible when a needed tool was
    absent from the (task-relevant SUBSET) menu, instead of pulling more via
    find_tools. The schema must steer find_tools on a capability gap and forbid
    concluding impossibility before trying it."""
    schema = build_output_schema(_FakeAgent()).lower()
    assert "capability-gap rule" in schema
    assert "find_tools" in schema
    # The menu is framed as a subset, and find_tools is the remedy for a gap.
    assert "subset" in schema
    # Concluding "impossible / missing tool" is explicitly gated behind find_tools.
    assert "impossible" in schema or "missing" in schema


def test_action_menu_still_lists_json_shapes():
    schema = build_output_schema(_FakeAgent())
    assert '"action": "memory_query"' in schema
    assert '"action": "tool_call"' in schema
    assert '"action": "final_answer"' in schema


def test_menu_lists_granted_system_tools():
    """ADR-0039: kernel system tools must appear on the closed menu so the LLM
    selects from them instead of hallucinating a name out of its Role prose."""
    system_tools = [
        {
            "name": "execute_command",
            "description": "Run a shell command",
            "schema_json": '{"properties": {"command": {"type": "string"}}}',
            "dangerous": True,
        }
    ]
    schema = build_output_schema(_FakeAgent(), system_tools)
    assert "execute_command" in schema
    assert "(system tool)" in schema
    assert "Run a shell command" in schema
    assert "command" in schema  # the arg schema was surfaced
    # And the empty-menu sentinel is gone now that a tool is present.
    assert "(no tools registered)" not in schema


def test_menu_tolerates_system_tool_with_bad_schema():
    """A malformed manifest schema degrades to a name-only entry, never an error."""
    system_tools = [{"name": "weird_tool", "description": "", "schema_json": "not json"}]
    schema = build_output_schema(_FakeAgent(), system_tools)
    assert "weird_tool" in schema


# ── D8: action protocol moved into <ActionProtocol>, OutputSchema is the closer ───

def test_action_protocol_section_holds_menu_not_output_schema():
    from cambrian_agent_sdk.helpers import build_prompt
    from cambrian_agent_sdk.react import _PER_TURN_OUTPUT_CONTRACT, _compose_action_protocol

    ap = _compose_action_protocol(_FakeAgent(), "")
    prompt = build_prompt(role="r", task="t", action_protocol=ap,
                          output_schema=_PER_TURN_OUTPUT_CONTRACT)

    assert "<ActionProtocol>" in prompt and "<OutputSchema>" in prompt
    # the action menu lives in <ActionProtocol>
    ap_body = prompt.split("<ActionProtocol>")[1].split("</ActionProtocol>")[0]
    assert "memory_query" in ap_body and "tool_call" in ap_body
    # <OutputSchema> is the short per-turn contract — NOT the full menu
    os_body = prompt.split("<OutputSchema>")[1]
    assert "EXACTLY ONE JSON object" in os_body
    assert "memory_query" not in os_body
    # ActionProtocol precedes OutputSchema (the recency-anchored closer)
    assert prompt.index("<ActionProtocol>") < prompt.index("<OutputSchema>")


def test_render_recall_empty_is_explicit_no_relevant_memory():
    # ADR-0048 #1: when the kernel's relevance floor drops everything, the agent
    # must be TOLD recall was empty (and may answer from own knowledge) — not handed
    # a silent/blank block that prior-task junk used to fill.
    #
    # v2: backward-compat: ``_render_recall(results)`` (single-arg) still produces
    # the legacy ``<memory status='empty'>...</memory>`` block.
    block = _render_recall([])
    assert "status='empty'" in block
    assert "no relevant memory found" in block
    assert "own" in block and "knowledge" in block


def test_render_recall_nonempty_lists_results():
    # v2 backward-compat: the single-arg call still produces the legacy block.
    block = _render_recall([{"text": "fact-A"}, {"text": "fact-B"}])
    assert "fact-A" in block and "fact-B" in block
    assert "status='empty'" not in block


def test_render_recall_v2_includes_query_attribute():
    """v2: the new ``_render_recall(query, results)`` form surfaces the query as
    an XML attribute, so the agent can see WHAT was asked, not just what was
    returned (issue #1 in the trajectory critique)."""
    block = _render_recall("Q4 2024 revenue", [{"text": "$4.2B"}])
    assert 'query="Q4 2024 revenue"' in block
    assert "status=" in block
    assert "$4.2B" in block


def test_render_recall_v2_empty_has_query_attribute():
    """v2: even on an empty result, the query attribute is visible (so the model
    sees what was searched, even when nothing came back)."""
    block = _render_recall("Q4 2024 revenue", [])
    assert 'query="Q4 2024 revenue"' in block
    assert "status=" in block
    assert "empty" in block


def test_action_menu_documents_cid_handoff():
    """ADR-0048 #1: the tool_call action must surface the cid-handoff pattern
    ({"$cid": "<cid>"} in args). Regression: the v1 prompt had this; the first
    v2 refactor dropped it; the field caught the issue (model did memory_query
    for an offloaded workspace content instead of passing the cid to a tool).
    """
    schema = build_output_schema(_FakeAgent())
    # The hint mentions both the OLD-style recalled-fact cid AND the v2
    # offloaded_cid attribute so the model can match either form to the pattern.
    assert "cid" in schema
    assert "$cid" in schema
    # The hint must appear in the Tool actions section, not the Memory section.
    ap_t = schema.lower()
    assert "tool_call" in ap_t
    assert "memory_query" in ap_t
    # The example for tool_call must mention the cid handoff is a way to avoid
    # re-emitting the body (so the model understands WHY to use it).
    assert "kernel resolves" in schema.lower() or "do not" in schema.lower()


def test_recall_surfaces_summary_and_content_cid():
    # ADR-0048 #1: recall serves the SUMMARY plus the cid of the full body, so the
    # agent reads the gist and can drill down rather than ingesting the whole fact.
    import json as _json
    from cambrian_agent_sdk.react import _format_memory
    results = [{
        "text": "Web search results listing the world's longest rivers.",
        "score": 0.8,
        "metadata": _json.dumps({"content_cid": "cid-xyz"}),
    }]
    out = _format_memory(results)
    assert "longest rivers" in out          # the summary is the surface
    assert "cid:cid-xyz" in out              # the full-body pointer is exposed


def test_recall_no_cid_when_metadata_absent():
    from cambrian_agent_sdk.react import _format_memory
    assert "[full content cid" not in _format_memory([{"text": "a short fact"}])


# ── #2: truncated action is detected, not laundered into a final answer ──────────

def test_parse_action_valid_action_parses():
    from cambrian_agent_sdk.react import parse_action
    a = parse_action('{"action": "tool_call", "tool": "write_file", "args": {"path": "x"}}')
    assert a["action"] == "tool_call" and a["tool"] == "write_file"


def test_parse_action_prose_is_final_answer():
    from cambrian_agent_sdk.react import parse_action
    a = parse_action("Here is my plain prose answer with no JSON at all.")
    assert a["action"] == "final_answer"


def test_parse_action_truncated_tool_call_is_flagged_not_finalized():
    # A write_file action cut off mid-content: started the envelope, never closed it.
    # Must NOT be laundered into a final_answer (which would silently end the loop and
    # skip the tool_call) — it's flagged _truncated for recoverable re-prompting.
    from cambrian_agent_sdk.react import parse_action
    truncated = '{"action": "tool_call", "tool": "fast_large_write_file", "args": {"path": "a.md", "content": "# Title\n\nLong body that gets cut off right here'
    a = parse_action(truncated)
    assert a["action"] == "_truncated"


def test_parse_action_truncated_behind_code_fence():
    from cambrian_agent_sdk.react import parse_action
    a = parse_action('```json\n{"action": "tool_call", "tool": "x", "args": {"content": "unterminated')
    assert a["action"] == "_truncated"
