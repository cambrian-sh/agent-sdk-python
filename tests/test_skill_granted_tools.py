"""ADR-0046 D6: a loaded system skill's tool grants become CALLABLE.

The defect (T-20261104-a-loaded-system-skill-s-tool-grants-never-enter): loading a
system skill made the kernel confer its grants run-scoped (ListSkills ->
ConferSkillGrants) and the note told the model "Its tools (X) are available", but X
was never added to the native tool definitions nor to the prose menu, so the model
could not call it.

The fake substrate below behaves like the kernel: ``list_tools(names=...)`` serves
STANDING grants only (AvailableToolsNamed does not read the run overlay), so a tool
the skill conferred and the agent did not already hold comes back absent — the SDK
must still offer it, because ListSkills returned it as granted.
"""
import json

from cambrian_agent_sdk import AgentTask, CognitiveAgent
from cambrian_agent_sdk import react

_WRITE = "mcp:fs/write_file"          # standing grant: describable via list_tools(names)
_OVERLAY = "mcp:fs/overlay_only"       # conferred by the skill only: not describable
_BODY = "Gather, outline, draft, verify, then write the file with write_file."


class _Bot(CognitiveAgent):
    role = "a report writer"

    def run(self, task):
        return self.think(task)


class _KernelLike:
    """ListSkills + ListTools + ExecuteTool shaped like the kernel's."""

    def __init__(self):
        self.executed = []
        self.list_tools_calls = []
        self.skill_fetches = []

    # -- skills --
    def list_skills(self, query="", k=0, names=None, full=False, session_token_id=""):
        if names and full:
            self.skill_fetches.append(session_token_id)
            return [{"name": "file-report", "description": "Write a report.",
                     "instructions": _BODY, "tool_grants": [_WRITE, _OVERLAY]}]
        return [{"name": "file-report", "description": "Write a report."}]

    # -- tools --
    def list_tools(self, query="", k=0, names=None, full=False):
        self.list_tools_calls.append({"query": query, "names": names, "full": full})
        if names:
            schema = ('{"type":"object","properties":{"path":{"type":"string"},'
                      '"content":{"type":"string"}},"required":["path","content"]}'
                      if full else '{"properties":{"path":{},"content":{}}}')
            return [{"name": _WRITE, "description": "Write a file.", "schema_json": schema}
                    for n in names if n == _WRITE]
        return []  # the seeded menu is empty: the tools arrive only with the skill

    def execute_tool(self, name, args_json="", session_token_id="", timeout_ms=0, task_id=""):
        self.executed.append((name, json.loads(args_json or "{}"), session_token_id))
        return {"result_json": json.dumps({"ok": True})}

    def put_context_node(self, text, session_token_id=""):
        return "cid-x"


class _Native(_KernelLike):
    def __init__(self):
        super().__init__()
        self.tools_per_call = []
        self.n = 0

    def generate_with_tools(self, session_token_id=None, messages=None, tools=None, **kw):
        self.tools_per_call.append([dict(t) for t in tools])
        self.n += 1
        if self.n == 1:
            return ("", [{"id": "c1", "name": "use_skill",
                          "arguments": json.dumps({"skill": "file-report"})}], "tool_use")
        if self.n == 2:
            # Call the granted tool under the name the PROVIDER was given.
            safe = next(t["name"] for t in tools if "write_file" in t["name"])
            return ("", [{"id": "c2", "name": safe, "arguments": json.dumps(
                {"path": "report.md", "content": "# Report"})}], "tool_use")
        if self.n == 3:
            safe = next(t["name"] for t in tools if "overlay_only" in t["name"])
            return ("", [{"id": "c3", "name": safe, "arguments": "{}"}], "tool_use")
        return ("done", [], "end_turn")

    def generate(self, *a, **kw):
        raise AssertionError("text path must not be taken")


def test_native_granted_tools_become_definitions_and_execute():
    bot = _Bot(agent_id="b")
    bot.substrate = sub = _Native()
    res = bot.think(AgentTask(text="write a report on the files", session_token_id="sess-1"))
    assert res.text == "done"

    before = {t["name"] for t in sub.tools_per_call[0]}
    after = sub.tools_per_call[1]
    assert not any("write_file" in n or "overlay_only" in n for n in before)
    by_name = {t["name"]: t for t in after}
    write = next(t for n, t in by_name.items() if "write_file" in n)
    overlay = next(t for n, t in by_name.items() if "overlay_only" in n)
    # The describable tool carries its REAL Tier-2 schema (the provider validates it).
    assert write["parameters"]["required"] == ["path", "content"]
    # The overlay-only tool is offered with a valid permissive object schema.
    assert overlay["parameters"]["type"] == "object"
    assert "file-report" in overlay["description"]
    # Append-only: the definitions sent before the load are an unchanged prefix.
    assert after[:len(sub.tools_per_call[0])] == sub.tools_per_call[0]

    # Both calls were dispatched to the kernel under their ORIGINAL names, with the
    # run's session token (the overlay key the kernel granted them under).
    assert sub.skill_fetches == ["sess-1"]
    assert [(n, s) for n, _, s in sub.executed] == [(_WRITE, "sess-1"), (_OVERLAY, "sess-1")]
    assert sub.executed[0][1] == {"path": "report.md", "content": "# Report"}
    # Schemas were fetched through the disclosure path, Tier-2 for native.
    assert {"query": "", "names": [_WRITE, _OVERLAY], "full": True} in sub.list_tools_calls


class _Text(_KernelLike):
    def __init__(self, responses):
        super().__init__()
        self._responses = list(responses)
        self.prompts = []

    def generate(self, session_token_id=None, prompt="", **kw):
        self.prompts.append(prompt)
        return self._responses.pop(0)


def test_text_path_granted_tools_enter_the_prose_menu_and_execute():
    bot = _Bot(agent_id="b")
    bot.substrate = sub = _Text([
        json.dumps({"action": "use_skill", "skill": "file-report"}),
        json.dumps({"action": "tool_call", "tool": _WRITE,
                    "args": {"path": "r.md", "content": "x"}}),
        json.dumps({"action": "final_answer", "answer": "done"}),
    ])
    # No native tool-calling on this substrate: the loop latches to the text path.
    res = bot.think(AgentTask(text="write a report", session_token_id="sess-2"))
    assert res.text == "done"
    menu_before = sub.prompts[0].split("## Tools (tool_call may name only these)")[1]
    menu_after = sub.prompts[1].split("## Tools (tool_call may name only these)")[1]
    assert _WRITE not in menu_before.split("## Skill")[0]
    tools_after = menu_after.split("## Skill")[0]
    assert f"{_WRITE} (system tool)" in tools_after
    assert f"{_OVERLAY} (system tool)" in tools_after
    assert sub.executed == [(_WRITE, {"path": "r.md", "content": "x"}, "sess-2")]


def test_local_skill_grants_are_not_invented():
    # A local skill's grants are not kernel-confirmed: a name nothing describes is
    # not offered (it would be a tool the kernel has never heard of).
    bot = _Bot(agent_id="b")
    out = react._skill_granted_tools(bot, "s", ["mcp:nope/x"], [], full=True,
                                     kernel_confirmed=False)
    assert out == []
    # ...whereas a confirmed grant is offered even when undescribed.
    out = react._skill_granted_tools(bot, "s", ["mcp:nope/x"], [], full=True,
                                     kernel_confirmed=True)
    assert [t["name"] for t in out] == ["mcp:nope/x"]
    # Already in the menu -> nothing added.
    assert react._skill_granted_tools(bot, "s", ["a"], [{"name": "a"}], full=True,
                                      kernel_confirmed=True) == []


class _LeaseAwareNative(_Native):
    """A kernel that describes overlay-conferred tools when the run's lease rides
    ListTools (T-20261104-kernel-listtools-names-ignores-run-scoped-skill)."""

    def list_tools(self, query="", k=0, names=None, full=False, session_token_id=""):
        self.list_tools_calls.append({"query": query, "names": names, "full": full,
                                      "lease": session_token_id})
        out = super().list_tools(query=query, k=k, names=names, full=full)
        if names and session_token_id == "sess-1" and _OVERLAY in names:
            out.append({"name": _OVERLAY, "description": "Overlay tool.",
                        "schema_json": '{"type":"object","properties":{"q":{"type":"string"}},'
                                       '"required":["q"]}'})
        return out


def test_overlay_granted_tool_gets_its_real_schema_when_the_lease_rides_list_tools():
    bot = _Bot(agent_id="b")
    bot.substrate = sub = _LeaseAwareNative()
    bot.think(AgentTask(text="write a report on the files", session_token_id="sess-1"))

    # The describe call carried the run's lease (the overlay key) ...
    described = [c for c in sub.list_tools_calls if c.get("names") and "lease" in c]
    assert described and all(c["lease"] == "sess-1" for c in described)
    # ... so the overlay-only tool is offered with its REAL schema, not the fallback.
    after = {t["name"]: t for t in sub.tools_per_call[1]}
    overlay = next(t for n, t in after.items() if "overlay_only" in n)
    assert overlay["parameters"]["required"] == ["q"]
