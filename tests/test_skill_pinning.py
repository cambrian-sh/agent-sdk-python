"""ADR-0139 Phase 0 (D1/D2/D8/D9): a loaded skill reaches the model IN FULL and stays.

The defect these pin down: on the text path ``use_skill`` went through
``WorkingMemory.add_text``, whose offloader replaced any body over 1000 chars with a
240-char head plus a cid (the grant note at the end was cut too); and the entry was
an ordinary rankable one, so a few tool calls later the 10-entry cap evicted it. On
the native path ``use_skill`` was never offered as a tool, and a native call to it was
answered "call was not executed". The old tests used a 13-char body and no
offloader, so none of this was visible to CI.
"""
import json

from cambrian_agent_sdk import AgentTask, CognitiveAgent, react, tool
from cambrian_agent_sdk.helpers import default_anti_patterns, default_constraints
from cambrian_agent_sdk.reflection import build_reflection_prompt

# A real-sized procedure: ~3k chars, with a sentinel at the END so a head cut is caught.
_STEPS = "".join(
    f"Step {i}: inspect module {i}, record its public functions, and note any TODO markers. "
    for i in range(1, 36)
)
_BIG_BODY = _STEPS + "FINAL STEP: report the sentinel SKILL-TAIL-7f3a in the answer."
assert len(_BIG_BODY) > 2500


class _SkillBot(CognitiveAgent):
    role = "a skill-using assistant"

    def run(self, task):
        return self.think(task)

    @tool
    def echo(self, text: str) -> str:
        """Echo the text back."""
        return "echoed " + text


def _bot(skills):
    bot = _SkillBot(agent_id="b")
    bot.local_skills = skills
    return bot


class _TextSub:
    """Text-path substrate WITH a content offloader, like the real SDK client."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.prompts = []
        self.offloaded = []

    def generate(self, session_token_id=None, prompt="", **kw):
        self.prompts.append(prompt)
        return self._responses.pop(0)

    def put_context_node(self, text, session_token_id=""):
        self.offloaded.append(text)
        return f"cid-{len(self.offloaded)}"


def _echo_calls(n):
    return [json.dumps({"action": "tool_call", "tool": "echo", "args": {"text": f"call {i}"}})
            for i in range(n)]


def test_big_skill_body_survives_offloader_and_many_tool_calls():
    bot = _bot([{"name": "audit", "description": "Audit a codebase.",
                 "instructions": _BIG_BODY, "tool_grants": ["echo"]}])
    bot.substrate = _TextSub(
        [json.dumps({"action": "use_skill", "skill": "audit"})]
        + _echo_calls(7)
        + [json.dumps({"action": "final_answer", "answer": "done"})]
    )

    res = bot.think(AgentTask(text="audit the repo"))
    assert res.text == "done"
    last = bot.substrate.prompts[-1]
    # The WHOLE body is in the prompt after 7 tool calls (14 trajectory entries,
    # well past the 10-entry cap) — including its last sentence.
    assert _BIG_BODY in last
    assert "SKILL-TAIL-7f3a" in last
    # The grant note survives as its own entry.
    assert "Its tools (echo) are available" in last
    # The skill body never went to the offloader.
    assert not any("SKILL-TAIL-7f3a" in t for t in bot.substrate.offloaded)




def test_overflowing_skill_is_refused_whole_never_truncated(monkeypatch):
    monkeypatch.setattr(react, "_DEFAULT_MAX_SKILL_CHARS", 4000)
    bot = _bot([
        {"name": "first", "description": "a", "instructions": _BIG_BODY, "tool_grants": []},
        {"name": "second", "description": "b", "instructions": "SECOND-" + _BIG_BODY, "tool_grants": []},
    ])
    bot.substrate = _TextSub([
        json.dumps({"action": "use_skill", "skill": "first"}),
        json.dumps({"action": "use_skill", "skill": "second"}),
        json.dumps({"action": "final_answer", "answer": "ok"}),
    ])
    bot.think(AgentTask(text="x"))
    last = bot.substrate.prompts[-1]
    assert _BIG_BODY in last  # the first fits and is in full
    assert "'second' was NOT loaded" in last
    assert "SECOND-" not in last  # not even a head of it


def test_sixth_skill_load_is_refused():
    skills = [{"name": f"s{i}", "description": "d", "instructions": f"body {i}", "tool_grants": []}
              for i in range(6)]
    bot = _bot(skills)
    bot.substrate = _TextSub(
        [json.dumps({"action": "use_skill", "skill": f"s{i}"}) for i in range(6)]
        + [json.dumps({"action": "final_answer", "answer": "ok"})]
    )
    bot.think(AgentTask(text="x"))
    last = bot.substrate.prompts[-1]
    assert "body 4" in last
    assert "body 5" not in last
    assert "skill-load budget exhausted" in last


SEP = chr(10) * 2


class _NativeSub:
    """Native tool-calling substrate: a use_skill CALL, then an end_turn answer."""

    def __init__(self, turns):
        self._turns = list(turns)
        self.sent = []
        self.systems = []
        self.tools = None

    def generate_with_tools(self, session_token_id=None, messages=None, tools=None, **kw):
        self.tools = tools
        self.sent.append([dict(m) for m in (messages or [])])
        self.systems.append(SEP.join(b["text"] for b in kw.get("system") or []))
        return self._turns.pop(0)

    def generate(self, *a, **kw):
        raise AssertionError("text path must not be taken")

    def put_context_node(self, text, session_token_id=""):
        return "cid-n"


def test_native_use_skill_is_offered_and_answered_with_the_full_body():
    bot = _bot([{"name": "audit", "description": "Audit a codebase.",
                 "instructions": _BIG_BODY, "tool_grants": ["echo"]}])
    bot.substrate = _NativeSub([
        ("", [{"id": "c1", "name": "use_skill", "arguments": json.dumps({"skill": "audit"})}], "tool_use"),
        ("", [{"id": "fa", "name": "final_answer", "arguments": '{"answer": "done"}'}], "tool_use"),
    ])
    res = bot.think(AgentTask(text="audit the repo"))
    assert res.text == "done"

    names = {t["name"] for t in bot.substrate.tools}
    assert {"use_skill", "find_skills"} <= names

    second = bot.substrate.sent[1]
    tool_turns = [m for m in second if m.get("role") == "tool"]
    assert len(tool_turns) == 1 and tool_turns[0]["tool_call_id"] == "c1"
    assert _BIG_BODY in tool_turns[0]["content"]
    assert "Its tools (echo) are available" in tool_turns[0]["content"]
    assert not any("not executed" in (m.get("content") or "") for m in second)
    # The body is delivered once, as the tool result — not repeated as a prose turn.
    assert sum(_BIG_BODY in (m.get("content") or "") for m in second) == 1

    # One protocol per prompt: the native prompt never prescribes the text protocol.
    # The action menu is in the cached system tier (ADR-0139 D2), not the first turn.
    prompt = bot.substrate.systems[1] + SEP + second[0]["content"]
    assert "EXACTLY ONE JSON object" not in prompt
    assert "tool_call" not in prompt
    assert "Call the `use_skill` tool" in prompt


def test_no_skill_tools_without_a_skill_menu():
    defs, _ = react.build_tool_definitions(_SkillBot(agent_id="b"), [], include_skills=False)
    assert not {"use_skill", "find_skills"} & {d["name"] for d in defs}


def test_constraints_state_the_real_budgets():
    text = " ".join(default_constraints())
    assert f"capped at {react._DEFAULT_MAX_MEMORY_QUERIES} per run" in text
    assert f"capped at {react._DEFAULT_MAX_TOOL_ROUNDS} per run" in text
    assert "uncapped" not in text
    custom = " ".join(default_constraints(max_memory_queries=0, max_tool_rounds=7))
    assert "no memory access" in custom and "capped at 7 per run" in custom
    native = " ".join(default_constraints(native_tools=True) + default_anti_patterns(native_tools=True))
    assert "tool_call" not in native and "ONE JSON action" not in native


def test_large_write_tool_named_only_when_in_menu():
    assert "fast_large_write_file" not in " ".join(default_anti_patterns())
    assert "fast_large_write_file" in " ".join(
        default_anti_patterns(large_write_tool="fast_large_write_file"))
    bot = _SkillBot(agent_id="b")
    assert not react._menu_has_tool(bot, [{"name": "write_file"}], "fast_large_write_file")
    assert react._menu_has_tool(bot, [{"name": "mcp:fs/fast_large_write_file"}], "fast_large_write_file")


def test_reflection_role_is_not_doubled():
    for role in ("You are a precise summarisation engine.", "a precise summarisation engine"):
        p = build_reflection_prompt(role, "t", "echo", {}, "err", [])
        assert p.startswith("You are a precise summarisation engine.\n"), p[:80]
        assert "You are You are" not in p


def test_native_find_tools_is_answered_in_its_tool_turn():
    # Same D8 class as use_skill: the discovery result must BE the tool result.
    class _Disc(_NativeSub):
        def list_tools(self, query="", k=3, names=None, full=False):
            if "write" in (query or ""):
                return [{"name": "mcp:fs/write_file", "schema_json": '{"properties":{"path":{}}}'}]
            return []

    bot = _SkillBot(agent_id="b")
    bot.substrate = _Disc([
        ("", [{"id": "c1", "name": "find_tools", "arguments": json.dumps({"need": "write a file"})}], "tool_use"),
        ("", [{"id": "fa", "name": "final_answer", "arguments": '{"answer": "done"}'}], "tool_use"),
    ])
    bot.think(AgentTask(text="create a file"))
    tool_turns = [m for m in bot.substrate.sent[1] if m.get("role") == "tool"]
    assert len(tool_turns) == 1 and "mcp:fs/write_file" in tool_turns[0]["content"]
    assert "Call one." in tool_turns[0]["content"]
