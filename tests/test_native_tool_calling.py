"""ADR-0097 Phase B — the loop uses the provider's structured signal when available.

The point of the whole exercise: stop inferring what the model meant from whether its
text parses. These tests pin the mapping from a native turn onto the loop's internal
action, and the two fallback paths that must never become silent.
"""
import json

import pytest

from cambrian_agent_sdk import AgentTask, CognitiveAgent, ToolCallingUnsupported, tool
from cambrian_agent_sdk.react import (
    action_from_native_turn,
    build_tool_definitions,
    run_think,
)


class Bot(CognitiveAgent):
    role = "a careful test assistant"

    def run(self, task):
        return self.think(task)


class ToolBot(Bot):
    @tool
    def echo(self, text: str) -> str:
        """Echo the text back."""
        return text


class _NativeLLM:
    """A substrate whose generate_with_tools returns canned turns."""

    def __init__(self, turns, text_responses=None):
        self._turns = list(turns)
        self._text = list(text_responses or [])
        self.native_calls = 0
        self.text_calls = 0
        self.last_tools = None
        self.sent_conversations = []

    def generate_with_tools(self, session_token_id=None, messages=None, tools=None, **kw):
        self.native_calls += 1
        self.last_tools = tools
        # Snapshot: the loop mutates the same list in place, so keeping a reference
        # would show only the final state and hide what each turn actually sent.
        self.sent_conversations.append([dict(m) for m in (messages or [])])
        return self._turns.pop(0)

    def generate(self, session_token_id=None, prompt="", **kw):
        self.text_calls += 1
        return self._text.pop(0)

    def execute_tool(self, *a, **kw):
        return {"result_json": json.dumps({"ok": True}), "result_cid": "", "denied": False,
                "deny_reason": "", "error": "", "arg_hash": "", "result_hash": ""}


class _UnsupportedLLM(_NativeLLM):
    def generate_with_tools(self, *a, **kw):
        self.native_calls += 1
        raise ToolCallingUnsupported("model cannot do tools")


# ── the mapping ──────────────────────────────────────────────────────────────

def test_tool_call_wins_over_stop_reason():
    """opencode #14972: providers report "stop" while returning calls. The ACTION
    outranks the narration — otherwise the loop runs one tool and halts."""
    act = action_from_native_turn(
        "I'll write it", [{"id": "c1", "name": "write_file", "arguments": '{"path":"x"}'}], "end_turn")
    assert act["action"] == "tool_call"
    assert act["tool"] == "write_file"
    assert act["args"] == {"path": "x"}


def test_end_turn_without_calls_is_a_declared_answer():
    act = action_from_native_turn("the answer is 42", [], "end_turn")
    assert act["action"] == "final_answer"
    assert not act.get("_inferred"), "an explicit end_turn is DECLARED, not inferred"


@pytest.mark.parametrize("stop", ["max_tokens", "refusal", "unknown", ""])
def test_non_end_turn_without_calls_is_incomplete(stop):
    """Anthropic's rule: anything that is not end_turn means the response is
    incomplete. Marked _inferred so the existing re-prompt handles it."""
    act = action_from_native_turn("half a thought", [], stop)
    assert act["action"] == "final_answer"
    assert act["_inferred"] is True


def test_malformed_arguments_are_not_a_final_answer():
    """A tool call the provider mangled is a failed ACTION. Treating it as an answer
    would finish the task on a parse error — the exact bug this ADR removes."""
    act = action_from_native_turn("", [{"id": "c1", "name": "w", "arguments": "{not json"}], "tool_use")
    assert act["action"] == "_truncated"


# ── tool definitions ─────────────────────────────────────────────────────────

def test_build_tool_definitions_includes_system_tools():
    bot = Bot(agent_id="b")
    defs, name_map = build_tool_definitions(bot, [
        {"name": "mcp:filesystem/write_file", "description": "write",
         "schema_json": '{"type":"object","properties":{"path":{"type":"string"}}}'},
        {"name": "", "description": "nameless — dropped"},
    ])
    names = [d["name"] for d in defs]
    # Sent SANITIZED: the raw name has ':' and '/', which providers reject with an
    # opaque 400 that names nothing (measured 2026-07-28).
    assert "mcp_filesystem_write_file" in names
    assert "mcp:filesystem/write_file" not in names
    assert name_map["mcp_filesystem_write_file"] == "mcp:filesystem/write_file"
    assert "" not in names, "a nameless tool must not reach the provider"
    schema = next(d for d in defs if d["name"].endswith("write_file"))["parameters"]
    assert schema["properties"]["path"]["type"] == "string"


def test_build_tool_definitions_survives_bad_schema_json():
    bot = Bot(agent_id="b")
    defs, _ = build_tool_definitions(bot, [{"name": "t", "schema_json": "{broken"}])
    got = next(d for d in defs if d["name"] == "t")
    assert got["parameters"] == {"type": "object", "properties": {}}


# ── loop integration ─────────────────────────────────────────────────────────

def test_loop_uses_native_path_and_offers_tools():
    """The whole point: with a native substrate the loop never parses prose, and the
    agent's tool menu reaches the provider as schemas."""
    sub = _NativeLLM(turns=[
        ("", [{"id": "c1", "name": "echo", "arguments": '{"text":"hi"}'}], "tool_use"),
        ("done", [], "end_turn"),
    ])
    bot = ToolBot(agent_id="b")
    bot.substrate = sub

    res = run_think(bot, AgentTask(text="do it"), seed_system_tools=False)

    assert res.data.decode() == "done"
    assert sub.native_calls == 2, "both turns must go through the native path"
    assert sub.text_calls == 0, "the text path must not be used when native works"
    offered = [t["name"] for t in sub.last_tools]
    assert "echo" in offered, "the local @tool must be offered as a native definition"


def test_unsupported_latches_off_and_falls_back():
    """The capability answer must switch the loop to the text path ONCE, not retry an
    RPC that cannot start succeeding mid-run."""
    sub = _UnsupportedLLM(turns=[], text_responses=[
        json.dumps({"action": "tool_call", "tool": "echo", "args": {"text": "hi"}}),
        json.dumps({"action": "final_answer", "answer": "fallback answer", "type": "text"}),
    ])
    # ToolBot, so tool_defs is non-empty and the native path is genuinely attempted —
    # with a tool-less agent this test would pass without ever reaching the latch.
    bot = ToolBot(agent_id="b")
    bot.substrate = sub

    res = run_think(bot, AgentTask(text="do it"), seed_system_tools=False)

    assert res.data.decode() == "fallback answer"
    assert sub.native_calls == 1, "the native RPC must be attempted exactly once, then latched off"
    assert sub.text_calls == 2, "every subsequent turn must use the text path"


def test_missing_method_falls_back_silently():
    """An older substrate without the method must take the fallback, not AttributeError."""
    class _OldSubstrate:
        def __init__(self):
            self.calls = 0

        def generate(self, session_token_id=None, prompt="", **kw):
            self.calls += 1
            return json.dumps({"action": "final_answer", "answer": "old path", "type": "text"})

    bot = ToolBot(agent_id="b")
    bot.substrate = _OldSubstrate()
    res = run_think(bot, AgentTask(text="do it"), seed_system_tools=False)
    assert res.data.decode() == "old path"
    assert bot.substrate.calls == 1


def test_sanitized_call_maps_back_to_the_real_tool_name():
    """The provider echoes the name IT was given. Dispatching that would look up a
    tool the kernel's registry has never heard of."""
    act = action_from_native_turn(
        "", [{"id": "c1", "name": "mcp_filesystem_write_file", "arguments": "{}"}], "tool_use",
        {"mcp_filesystem_write_file": "mcp:filesystem/write_file"})
    assert act["tool"] == "mcp:filesystem/write_file"


def test_sanitized_names_do_not_collide():
    """Two tools that sanitize to the same string must stay distinguishable, and
    deterministically so — two runs offering different names would desync a
    provider-side cache and our own logs."""
    bot = Bot(agent_id="b")
    defs, name_map = build_tool_definitions(bot, [
        {"name": "mcp:fs/read"}, {"name": "mcp_fs_read"}, {"name": "mcp/fs:read"},
    ])
    names = [d["name"] for d in defs]
    assert len(set(names)) == len(names), f"collision: {names}"
    for safe, original in name_map.items():
        assert safe in names
    # The three colliding system tools stay distinguishable alongside the built-in
    # discovery tools.
    assert {"mcp:fs/read", "mcp_fs_read", "mcp/fs:read"} <= set(name_map.values())


def test_parameters_get_a_top_level_object_type():
    """The kernel stores MCP schemas as `{"properties": {...}}` with no top-level
    "type". Providers reject that with an opaque 400 naming no field — measured
    2026-07-28: properties-only 400, + type:object 200."""
    from cambrian_agent_sdk.react import normalize_tool_parameters

    got = normalize_tool_parameters({"properties": {"path": {}, "content": {}}})
    assert got["type"] == "object"
    assert set(got["properties"]) == {"path", "content"}
    # Empty properties mean "any" and are legal once the top-level type is present —
    # inventing "string" would be a lie the provider then enforces.
    assert got["properties"]["path"] == {}

    assert normalize_tool_parameters(None) == {"type": "object", "properties": {}}
    assert normalize_tool_parameters({}) == {"type": "object", "properties": {}}
    # A schema that already declares itself is left alone.
    arr = {"type": "array", "items": {"type": "string"}}
    assert normalize_tool_parameters(arr) == arr


def test_definitions_always_carry_a_valid_schema():
    bot = Bot(agent_id="b")
    defs, _ = build_tool_definitions(bot, [
        {"name": "mcp:filesystem/write_file", "schema_json": '{"properties":{"path":{}}}'},
    ])
    assert defs[0]["parameters"]["type"] == "object"


def test_conversation_accumulates_assistant_and_tool_turns():
    """ADR-0097 D8 — the core fix.

    Native tool-calling is conversational: the model must be sent its OWN assistant turn
    back, plus a tool turn carrying the result under the provider's call id. The first
    cut sent one user message per round, so every round was a fresh conversation in which
    the model had never called anything — it re-explored forever and never completed.
    """
    sub = _NativeLLM(turns=[
        ("", [{"id": "call_1", "name": "echo", "arguments": '{"text":"hi"}'}], "tool_use"),
        ("all done", [], "end_turn"),
    ])
    bot = ToolBot(agent_id="b")
    bot.substrate = sub

    res = run_think(bot, AgentTask(text="do it"), seed_system_tools=False)
    assert res.data.decode() == "all done"
    assert len(sub.sent_conversations) == 2

    first, second = sub.sent_conversations
    assert [m["role"] for m in first] == ["user"], "turn 1 is just the seeded prompt"

    roles = [m["role"] for m in second]
    assert roles == ["user", "assistant", "tool"], f"turn 2 must carry the exchange, got {roles}"

    # The assistant turn goes back verbatim, tool_calls included — that is what makes
    # the call the model's own rather than a narration about someone else.
    assert second[1]["tool_calls"][0]["id"] == "call_1"
    # The tool turn is correlated by the PROVIDER's id; a synthesized one is rejected.
    assert second[2]["tool_call_id"] == "call_1"
    assert "ok" in second[2]["content"] or second[2]["content"]


def test_notes_reach_the_model_as_turns_under_native():
    """With the prompt built once, anything the loop writes to working memory would be
    invisible unless it becomes a real turn — hence the conversation mirror."""
    sub = _NativeLLM(turns=[
        ("thinking out loud", [], "max_tokens"),   # not finished -> re-prompt note
        ("done", [], "end_turn"),
    ])
    bot = ToolBot(agent_id="b")
    bot.substrate = sub
    run_think(bot, AgentTask(text="do it"), seed_system_tools=False)

    second = sub.sent_conversations[1]
    assert any(m["role"] == "user" and "not a valid action" in (m.get("content") or "")
               for m in second), f"the nudge must reach the model as a turn: {second}"


def test_latching_off_discards_the_conversation():
    """On fallback the prompt is rebuilt from working memory each turn, so a half-built
    conversation is dead weight — and the mirror would keep duplicating into it."""
    sub = _UnsupportedLLM(turns=[], text_responses=[
        json.dumps({"action": "final_answer", "answer": "fallback", "type": "text"}),
    ])
    bot = ToolBot(agent_id="b")
    bot.substrate = sub
    res = run_think(bot, AgentTask(text="do it"), seed_system_tools=False)
    assert res.data.decode() == "fallback"
    assert sub.native_calls == 1


def test_tool_turn_immediately_follows_its_assistant_turn():
    """A `role:"tool"` message MUST directly follow the assistant turn that requested
    it. Measured against the live endpoint, everything else equal:
        user, assistant(call), tool       -> 200
        user, assistant(call), NOTE, tool -> 400
        user, assistant(call), tool, NOTE -> 200
    The loop writes notes between requesting a tool and running it, so mirroring them
    immediately would land them in exactly that gap.
    """
    sub = _NativeLLM(turns=[
        ("", [{"id": "call_1", "name": "echo", "arguments": '{"text":"hi"}'}], "tool_use"),
        ("", [{"id": "call_2", "name": "echo", "arguments": '{"text":"hi"}'}], "tool_use"),
        ("done", [], "end_turn"),
    ])
    bot = ToolBot(agent_id="b")
    bot.substrate = sub
    run_think(bot, AgentTask(text="do it"), seed_system_tools=False)

    for convo in sub.sent_conversations:
        roles = [m["role"] for m in convo]
        for i, m in enumerate(convo):
            if m["role"] == "assistant" and m.get("tool_calls"):
                assert i + 1 < len(roles) and roles[i + 1] == "tool", (
                    f"assistant call not immediately followed by its tool turn: {roles}")


def test_unexecuted_call_is_still_answered():
    """A call the loop declines to run (recurrence veto, budget, ungranted tool) must
    still get a tool turn, or the assistant turn dangles unanswered."""
    # The same call twice: the second is blocked by the success-dedup guard.
    call = {"id": "call_1", "name": "echo", "arguments": '{"text":"hi"}'}
    sub = _NativeLLM(turns=[
        ("", [call], "tool_use"),
        ("", [dict(call, id="call_2")], "tool_use"),
        ("", [dict(call, id="call_3")], "tool_use"),
        ("done", [], "end_turn"),
    ])
    bot = ToolBot(agent_id="b")
    bot.substrate = sub
    run_think(bot, AgentTask(text="do it"), seed_system_tools=False)

    final = sub.sent_conversations[-1]
    requested = [c["id"] for m in final if m["role"] == "assistant" for c in (m.get("tool_calls") or [])]
    answered = [m["tool_call_id"] for m in final if m["role"] == "tool"]
    for cid in requested:
        assert cid in answered, f"call {cid} was never answered: {[m['role'] for m in final]}"


def test_discovery_tools_are_offered_natively():
    """D7.3 withdrew the JSON tool_call action — correctly — but took find_tools and
    describe_tool with it. Those are not invocation; they are how the model widens a
    menu that is only ever a top-k guess. Without them a bad seed is unrecoverable."""
    bot = ToolBot(agent_id="b")
    defs, name_map = build_tool_definitions(bot, [])
    names = [d["name"] for d in defs]
    assert "find_tools" in names
    # describe_tool is NOT offered natively: it fetches the ADR-0045 Tier-2 spec, which
    # a native tool definition already carries. Offering it would buy a wasted round.
    assert "describe_tool" not in names
    # They must survive the sanitize/reverse-map round trip under their own names.
    assert name_map["find_tools"] == "find_tools"


def test_native_discovery_call_becomes_a_loop_action():
    """find_tools is handled by the loop, not dispatched to the tool plane — otherwise
    it would be looked up in the kernel's tool registry and fail."""
    act = action_from_native_turn(
        "", [{"id": "c1", "name": "find_tools", "arguments": '{"need":"write a file"}'}],
        "tool_use", {"find_tools": "find_tools"})
    assert act["action"] == "find_tools"
    assert act["need"] == "write a file"




def test_discovered_tools_become_offerable():
    """A discovery that does not rebuild tool_defs teaches the model names it still
    cannot call."""
    class _DiscoveringLLM(_NativeLLM):
        def list_tools(self, query="", k=3, names=None, full=False):
            # The seed misses write_file; discovery finds it.
            if "write" in (query or ""):
                return [{"name": "mcp:filesystem/write_file", "schema_json": '{"properties":{"path":{}}}'}]
            return [{"name": "mcp:filesystem/read_file", "schema_json": '{"properties":{"path":{}}}'}]

    sub = _DiscoveringLLM(turns=[
        ("", [{"id": "c1", "name": "find_tools", "arguments": '{"need":"write a file"}'}], "tool_use"),
        ("done", [], "end_turn"),
    ])
    bot = ToolBot(agent_id="b")
    bot.substrate = sub
    run_think(bot, AgentTask(text="create a file"))

    offered_last = [t["name"] for t in sub.last_tools]
    assert "mcp_filesystem_write_file" in offered_last, (
        f"a discovered tool must become callable, got {offered_last}")


def test_tool_array_prefix_is_stable_across_discovery():
    """`tools` is a per-request parameter — it is resent every turn and cannot be
    registered once. The only lever is prefix stability, which is what lets a provider
    cache it. Discovery must therefore APPEND; a reorder would silently cost cache hits
    with nothing failing.
    """
    bot = ToolBot(agent_id="b")
    before, _ = build_tool_definitions(bot, [{"name": "mcp:fs/read"}])
    after, _ = build_tool_definitions(bot, [{"name": "mcp:fs/read"}, {"name": "mcp:fs/write"}])

    names_before = [d["name"] for d in before]
    names_after = [d["name"] for d in after]
    assert names_after[: len(names_before)] == names_before, (
        f"discovery must append, not reorder: {names_before} -> {names_after}")
    assert names_after[len(names_before):] == ["mcp_fs_write"]


def test_native_definitions_request_tier_two_schemas():
    """A provider VALIDATES the schema it is given, so a native definition needs the
    real one. Tier-1 is arg-names-only — right for the prose menu, wrong here."""
    seen = {}

    class _TierRecordingSub(_NativeLLM):
        def list_tools(self, query="", k=3, names=None, full=False):
            seen["full"] = full
            return [{"name": "mcp:fs/write_file",
                     "schema_json": '{"type":"object","properties":{"path":{"type":"string"}}}'}]

    sub = _TierRecordingSub(turns=[("done", [], "end_turn")])
    bot = ToolBot(agent_id="b")
    bot.substrate = sub
    run_think(bot, AgentTask(text="write a file"))
    assert seen.get("full") is True, "native definitions must be seeded at Tier-2"


def test_text_encoded_control_envelope_is_honoured_not_answered():
    """A control envelope written as TEXT must dispatch, not become the answer.

    With native tool-calling the model sometimes writes `{"action": "memory_query", ...}`
    as prose instead of emitting a tool call. Wrapping that as a final answer did two bad
    things at once: the user was shown raw JSON, and the memory query they asked for never
    ran — the agent said it was searching and then didn't.
    """
    from cambrian_agent_sdk.react import action_from_native_turn

    # Bare envelope.
    act = action_from_native_turn(
        '{"action": "memory_query", "query": "airline"}', [], "end_turn", {}
    )
    assert act["action"] == "memory_query", act
    assert act["query"] == "airline", act

    # The shape actually seen: a sentence of narration, THEN the envelope. Requiring a
    # leading brace let precisely this through.
    act = action_from_native_turn(
        "I'll query my memory for airline-related content in detail.\n\n"
        '{"action": "memory_query", "query": "airline aviation airport flight"}',
        [], "end_turn", {},
    )
    assert act["action"] == "memory_query", act
    assert "airline" in act["query"], act


def test_ordinary_prose_is_still_a_final_answer():
    """The guard must not hijack normal replies — including ones that mention JSON."""
    from cambrian_agent_sdk.react import action_from_native_turn

    for text in [
        "The Little Prince lives on asteroid B-612.",
        "Here is the JSON you asked for: {\"name\": \"afsin\"}",
        "",
    ]:
        act = action_from_native_turn(text, [], "end_turn", {})
        assert act["action"] == "final_answer", (text, act)
        assert act["answer"] == text, (text, act)
