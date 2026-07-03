"""Tests for the resolve_cid action (ADR-0048 residual)."""
import json

from cambrian_agent_sdk import AgentResult, AgentTask, CognitiveAgent
from cambrian_agent_sdk.react import run_think
from cambrian_agent_sdk.working_memory import (
    Step,
    TextEntry,
    WorkingMemory,
    render_entry_xml,
    render_step_xml,
)


class Bot(CognitiveAgent):
    role = "a careful test assistant"

    def run(self, task):
        return self.think(task)


class _Store:
    """An in-memory ContentStore backing the resolve_cid path."""

    def __init__(self):
        self._nodes: dict = {}  # cid -> bytes

    def put(self, text: str) -> str:
        import hashlib
        cid = hashlib.sha256(text.encode()).hexdigest()[:32]
        self._nodes[cid] = text.encode()
        return cid

    # substrate interface (ADR-0022)
    def get_context_node(self, cid, session_token_id=""):
        data = self._nodes.get(cid)
        if data is None:
            return None
        class _Node:
            def __init__(self, data):
                self.data = data
        return _Node(data)

    def put_context_node(self, text, session_token_id=""):
        return self.put(text)


class _ResolveSubstrate:
    """A substrate that knows how to fetch + put cids (for resolve_cid tests)."""

    def __init__(self, store: _Store, responses):
        self._store = store
        self._responses = list(responses)
        self.prompts = []
        self.resolve_cid_calls = 0
        self.put_calls = 0

    def generate(self, session_token_id=None, prompt="", **kw):
        self.prompts.append(prompt)
        return self._responses.pop(0)

    def execute_tool(self, *a, **kw):  # not used here
        return {"result_json": "{}", "result_cid": "", "denied": False, "deny_reason": "",
                "error": "", "arg_hash": "", "result_hash": ""}

    def get_context_node(self, cid, session_token_id=""):
        self.resolve_cid_calls += 1
        return self._store.get_context_node(cid, session_token_id=session_token_id)

    def put_context_node(self, text, session_token_id=""):
        self.put_calls += 1
        return self._store.put_context_node(text, session_token_id=session_token_id)


# ──────────────────────────────────────────────────────────────────────
# Prompt mentions: the action is documented in <ActionProtocol> and the
# AntiPattern is in <AntiPatterns>.
# ──────────────────────────────────────────────────────────────────────

def test_action_protocol_documents_resolve_cid():
    from cambrian_agent_sdk.react import build_output_schema
    class _FakeTools:
        def specs(self): return []
    class _FakeAgent:
        tools = _FakeTools()
    schema = build_output_schema(_FakeAgent())
    # The action is in the prompt.
    assert "resolve_cid" in schema
    assert "offload" in schema
    assert "inline" in schema
    # The hint says "Default: prefer passing the cid as `{\"$cid\": ...}` to a
    # tool_call" — so the model knows the offload-pass is the default.
    assert "default" in schema.lower()
    # The inline cap is mentioned.
    assert "3" in schema  # the per-run cap


def test_anti_patterns_warn_against_unnecessary_resolve_cid():
    from cambrian_agent_sdk.helpers import build_prompt, DEFAULT_ANTI_PATTERNS
    p = build_prompt("r", "t")
    # The new anti-pattern is in <AntiPatterns>.
    assert "resolve_cid" in p.lower()
    # The offload-mode-is-free note is there.
    assert "offload mode is free" in p.lower() or "offload" in p.lower()


# ──────────────────────────────────────────────────────────────────────
# Step / renderer: the resolve_cid kind renders correctly for both modes.
# ──────────────────────────────────────────────────────────────────────

def test_render_step_xml_resolve_cid_offload():
    s = Step(
        kind="resolve_cid", n=3, cid="new_cid_abc",
        resolved_from="orig_cid_xyz", mode="offload", chars=3339,
        summary="re-offloaded as new cid (3339 chars; body is NOT in your context).",
    )
    xml = render_step_xml(s)
    # The new cid is the one to use next.
    assert "cid='new_cid_abc'" in xml
    # The original cid is preserved as ``resolved_from`` so the model can audit.
    assert "resolved_from='orig_cid_xyz'" in xml
    # The mode is visible.
    assert "mode='offload'" in xml
    # The cid_hint tells the model what to do next.
    assert "<cid_hint>" in xml
    assert '{"$cid": "new_cid_abc"}' in xml
    # The body is NOT in the prompt (it's offloaded).
    assert "<body>" not in xml


def test_render_step_xml_resolve_cid_inline():
    body = "Full body of the resolved content. " * 50
    s = Step(
        kind="resolve_cid", n=3, cid="orig_cid_xyz",
        resolved_from="orig_cid_xyz", mode="inline", body=body,
        summary="2550 chars inlined; full body in your context.",
    )
    xml = render_step_xml(s)
    # The original cid is preserved.
    assert "cid='orig_cid_xyz'" in xml
    assert "resolved_from='orig_cid_xyz'" in xml
    # The mode is visible.
    assert "mode='inline'" in xml
    # The body IS in the prompt (so the model can read it).
    assert "<body>" in xml
    assert "Full body of the resolved content." in xml


# ──────────────────────────────────────────────────────────────────────
# Loop integration: the resolve_cid action handler works end-to-end.
# ──────────────────────────────────────────────────────────────────────

def test_resolve_cid_offload_creates_new_cid_and_emits_step():
    """End-to-end: agent does resolve_cid as=offload; the kernel creates a new
    cid; the trajectory has a resolve_cid Step; the body is NOT in the prompt
    (offload mode)."""
    store = _Store()
    orig_cid = store.put("This is the original body of the workspace content.")
    sub = _ResolveSubstrate(
        store,
        responses=[
            # The agent's only action: resolve_cid as=offload.
            json.dumps({"action": "resolve_cid", "cid": orig_cid, "as": "offload"}),
            # Then a final answer.
            json.dumps({"action": "final_answer", "answer": "resolved", "type": "text"}),
        ],
    )
    bot = Bot(agent_id="b")
    bot.substrate = sub
    res = run_think(bot, AgentTask(text="x"))
    assert res.type != "error"
    # The kernel saw a fetch + a put (the offload re-store).
    assert sub.resolve_cid_calls == 1
    assert sub.put_calls == 1
    # The trajectory has the resolve_cid Step (offload mode).
    last_prompt = sub.prompts[-1]
    assert "type='resolve_cid'" in last_prompt
    assert "mode='offload'" in last_prompt
    # The new cid is the one to use next.
    assert '{"$cid": "' in last_prompt
    # The body is NOT inlined (offload mode).
    assert "This is the original body of the workspace content." not in last_prompt


def test_resolve_cid_inline_puts_body_in_context():
    """End-to-end: agent does resolve_cid as=inline; the body IS in the prompt
    so the model can read it."""
    store = _Store()
    orig_cid = store.put("BODY_MARKER: this is what was offloaded")
    sub = _ResolveSubstrate(
        store,
        responses=[
            json.dumps({"action": "resolve_cid", "cid": orig_cid, "as": "inline"}),
            json.dumps({"action": "final_answer", "answer": "saw the body", "type": "text"}),
        ],
    )
    bot = Bot(agent_id="b")
    bot.substrate = sub
    res = run_think(bot, AgentTask(text="x"))
    assert res.type != "error"
    # The body IS now in the prompt (inline mode).
    last_prompt = sub.prompts[-1]
    assert "BODY_MARKER: this is what was offloaded" in last_prompt
    assert "mode='inline'" in last_prompt


def test_resolve_cid_inline_budget_caps_at_three():
    """The per-run cap on inline resolves is enforced. The 4th inline resolve
    is rejected with a note; the agent must fall back to offload or answer."""
    store = _Store()
    cids = [store.put(f"body {i}") for i in range(5)]
    # The agent does 4 inline resolves in a row. The 4th must be capped.
    responses = [
        json.dumps({"action": "resolve_cid", "cid": cids[0], "as": "inline"}),
        json.dumps({"action": "resolve_cid", "cid": cids[1], "as": "inline"}),
        json.dumps({"action": "resolve_cid", "cid": cids[2], "as": "inline"}),
        json.dumps({"action": "resolve_cid", "cid": cids[3], "as": "inline"}),  # capped
        json.dumps({"action": "final_answer", "answer": "done", "type": "text"}),
    ]
    sub = _ResolveSubstrate(store, responses)
    bot = Bot(agent_id="b")
    bot.substrate = sub
    res = run_think(bot, AgentTask(text="x"))
    assert res.type != "error"
    last_prompt = sub.prompts[-1]
    # The 4th inline resolve was rejected (the budget-exhausted note is in the
    # trajectory, NOT a 4th inline body).
    assert "inline budget exhausted" in last_prompt
    # Only 3 of the 4 cids' bodies were inlined.
    assert "body 0" in last_prompt
    assert "body 1" in last_prompt
    assert "body 2" in last_prompt
    assert "body 3" not in last_prompt


def test_resolve_cid_missing_cid_emits_note():
    """resolve_cid without a cid is a malformed action; the agent gets a note
    and the loop continues (no crash)."""
    sub = _ResolveSubstrate(
        _Store(),
        responses=[
            json.dumps({"action": "resolve_cid"}),  # no cid
            json.dumps({"action": "final_answer", "answer": "ok", "type": "text"}),
        ],
    )
    bot = Bot(agent_id="b")
    bot.substrate = sub
    res = run_think(bot, AgentTask(text="x"))
    assert res.type == "text"
    assert "needs a 'cid'" in sub.prompts[-1]


def test_resolve_cid_unknown_cid_emits_note():
    """resolve_cid with a cid that doesn't exist in the ContentStore gets a
    note, not a crash."""
    sub = _ResolveSubstrate(
        _Store(),
        responses=[
            json.dumps({"action": "resolve_cid", "cid": "no-such-cid"}),
            json.dumps({"action": "final_answer", "answer": "ok", "type": "text"}),
        ],
    )
    bot = Bot(agent_id="b")
    bot.substrate = sub
    res = run_think(bot, AgentTask(text="x"))
    assert res.type == "text"
    assert "not found in ContentStore" in sub.prompts[-1]


def test_resolve_cid_invalid_as_value_emits_note():
    """resolve_cid with as=foo (not 'offload' or 'inline') gets a note."""
    sub = _ResolveSubstrate(
        _Store(),
        responses=[
            json.dumps({"action": "resolve_cid", "cid": "abc", "as": "garbage"}),
            json.dumps({"action": "final_answer", "answer": "ok", "type": "text"}),
        ],
    )
    bot = Bot(agent_id="b")
    bot.substrate = sub
    res = run_think(bot, AgentTask(text="x"))
    assert res.type == "text"
    assert "must be 'offload' or 'inline'" in sub.prompts[-1]
