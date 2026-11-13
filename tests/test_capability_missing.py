"""The capability_missing outcome (T-20261112-yield-sub-goal-on-a-composed-lease-runs-under).

A worker whose step needs a tool it does not have ENDS the step as capability_missing
instead of yielding a sub-goal that would run with the same tools. The kernel half
(cambrian-core domain/step_recovery.go) reads the payload type and the
``_capability_*`` handoff keys; find_tools learns from the ListTools response header
``x-tool-outside-step`` which matching tools exist outside the step.
"""
import json

import pytest

from cambrian_agent_sdk import (
    CAPABILITY_MISSING_RESULT_TYPE,
    AgentTask,
    CognitiveAgent,
    SubstrateClient,
    capability_missing,
)
from cambrian_agent_sdk import react
from cambrian_agent_sdk.composition import Composition, ComposedScopedSubstrate
from cambrian_agent_sdk.runtime import _handoff_from_result


# ── types.capability_missing ────────────────────────────────────────────────

def test_capability_missing_carries_the_kernel_wire_keys():
    r = capability_missing("  run a PowerShell command ", tools=["mcp:shell/run", " ", "mcp:shell/run",
                                                                "mcp:ps/exec"],
                           tried="x" * 900)
    assert CAPABILITY_MISSING_RESULT_TYPE == "capability_missing"
    assert r.type == "capability_missing"
    assert r.text == "run a PowerShell command"
    assert r.context["_capability_missing"] == "true"
    assert r.context["_capability_needed"] == "run a PowerShell command"
    assert r.context["_capability_tool"] == "mcp:shell/run,mcp:ps/exec"
    assert len(r.context["_capability_tried"]) == 500
    assert not r.is_yield


def test_capability_missing_omits_optional_keys_and_refuses_an_empty_need():
    r = capability_missing("fetch a web page")
    assert set(r.context) == {"_capability_missing", "_capability_needed"}
    with pytest.raises(ValueError):
        capability_missing("  ")


def test_capability_missing_reaches_the_handoff_metadata():
    class Req:
        id, from_agent = "t1", "kernel"

    h = _handoff_from_result(capability_missing("fetch a web page", tools=["mcp:fetch/get"]),
                             Req(), "composed_worker")
    assert h.payload.type == "capability_missing"
    assert h.metadata["_capability_missing"] == "true"
    assert h.metadata["_capability_tool"] == "mcp:fetch/get"


# ── SubstrateClient.find_tools reads x-tool-outside-step ────────────────────

class _Tool:
    def __init__(self, name):
        self.name, self.description, self.schema_json, self.dangerous = name, "d", "{}", False


class _Resp:
    def __init__(self, names):
        self.tools = [_Tool(n) for n in names]


class _Call:
    def __init__(self, md):
        self._md = md

    def initial_metadata(self):
        return self._md


class _ListTools:
    def __init__(self, names, md, fail=False):
        self.names, self.md, self.fail, self.seen = names, md, fail, None

    def with_call(self, req, metadata=None):
        self.seen = list(metadata or [])
        if self.fail:
            raise RuntimeError("unavailable")
        return _Resp(self.names), _Call(self.md)


class _Stub:
    def __init__(self, lt):
        self.ListTools = lt


def test_substrate_find_tools_returns_the_outside_step_header():
    lt = _ListTools(["mcp:fs/read_file"], [("x-tool-outside-step", "mcp:shell/run, mcp:ps/exec")])
    client = SubstrateClient(stub=_Stub(lt), agent_id="composed_worker")
    tools, outside = client.find_tools("run powershell", k=5, full=True, session_token_id="lease-1")
    assert [t["name"] for t in tools] == ["mcp:fs/read_file"]
    assert outside == ["mcp:shell/run", "mcp:ps/exec"]
    md = dict(lt.seen)
    assert md["x-tool-query"] == "run powershell" and md["x-lease-id"] == "lease-1"
    assert md["x-tool-k"] == "5" and md["x-tool-full"] == "true"


def test_substrate_find_tools_without_header_or_on_error_degrades():
    client = SubstrateClient(stub=_Stub(_ListTools(["a"], [])), agent_id="w")
    assert client.find_tools("q")[1] == []
    client = SubstrateClient(stub=_Stub(_ListTools(["a"], [], fail=True)), agent_id="w")
    assert client.find_tools("q") == ([], [])


# ── ComposedScopedSubstrate.find_tools passes the query ─────────────────────

class _Inner:
    def __init__(self):
        self.find_calls, self.executed = [], []

    def list_tools(self, query="", k=0, names=None, full=False, session_token_id=""):
        return [{"name": "mcp:fs/read_file"}]

    def find_tools(self, query, k=0, full=False, session_token_id=""):
        self.find_calls.append((query, session_token_id))
        return [{"name": "mcp:fs/read_file"}], ["mcp:shell/run", "mcp:fs/read_file"]

    def execute_tool(self, name, *a, **kw):
        self.executed.append(name)
        return {"ok": True}


def test_composed_find_tools_sends_the_query_and_never_serves_outside_tools():
    inner = _Inner()
    sub = ComposedScopedSubstrate(inner)
    sub.bind(Composition(runtime="composed_worker", tools=("mcp:fs/read_file",)), "lease-9")
    tools, outside = sub.find_tools("run powershell", k=3)
    assert inner.find_calls == [("run powershell", "lease-9")]
    assert [t["name"] for t in tools] == ["mcp:fs/read_file"]
    assert outside == ["mcp:shell/run"]  # a served name is never reported as outside
    assert sub.execute_tool("mcp:shell/run").get("denied") is True
    assert inner.executed == []


def test_composed_find_tools_unbound_offers_nothing():
    assert ComposedScopedSubstrate(_Inner()).find_tools("x") == ([], [])


def test_profile_scoped_find_tools_keeps_the_allowlist():
    from cambrian_agent_sdk.profiles import ProfileScopedSubstrate

    class _Allow:
        def allows(self, name):
            return name == "mcp:fs/read_file"

    class _Profile:
        name, allowlist = "p", _Allow()

    class _Inner2(_Inner):
        def find_tools(self, query, k=0, full=False, session_token_id=""):
            return [{"name": "mcp:fs/read_file"}, {"name": "mcp:web/search"}], ["mcp:shell/run"]

    tools, outside = ProfileScopedSubstrate(_Inner2(), _Profile()).find_tools("q")
    assert [t["name"] for t in tools] == ["mcp:fs/read_file"]
    assert outside == ["mcp:shell/run"]


# ── the loop ────────────────────────────────────────────────────────────────

class Worker(CognitiveAgent):
    role = "a worker"

    def run(self, task):
        return self.think(task)


class _Scripted:
    """A native model that plays a script of tool calls; find_tools answers with an
    outside-step tool."""

    def __init__(self, script, outside=("mcp:shell/run",)):
        self.script, self.outside = list(script), list(outside)
        self.tools_seen, self.messages = [], []

    def list_tools(self, query="", k=None, full=False, session_token_id=""):
        return []

    def find_tools(self, query, k=0, full=False, session_token_id=""):
        return [], list(self.outside)

    def generate_with_tools(self, session_token_id=None, messages=None, tools=None, **kw):
        self.tools_seen.append([t["name"] for t in tools or []])
        self.messages = list(messages or [])
        name, args = self.script.pop(0)
        return ("", [{"id": f"c{len(self.script)}", "name": name,
                      "arguments": json.dumps(args)}], "tool_use")

    def execute_tool(self, *a, **kw):
        raise AssertionError("no tool should run")


def _tool_turns(sub):
    return [m.get("content", "") for m in sub.messages if m.get("role") == "tool"]


def test_find_tools_names_outside_tools_and_report_ends_the_step():
    sub = _Scripted([("find_tools", {"need": "run a PowerShell command"}),
                     ("report_capability_missing", {"needed": "run a PowerShell command",
                                                    "tried": "searched for a shell tool"})])
    bot = Worker("w")
    bot.substrate = sub
    res = bot.think(AgentTask(text="write the CSV with PowerShell"), max_tool_rounds=4)
    assert res.type == "capability_missing"
    assert res.context["_capability_needed"] == "run a PowerShell command"
    # find_tools's outside-step name is carried even though the model did not repeat it
    assert res.context["_capability_tool"] == "mcp:shell/run"
    assert res.context["_capability_tried"] == "searched for a shell tool"
    # the verb is offered, and the outside tool never reached the menu
    assert "report_capability_missing" in sub.tools_seen[0]
    assert all("mcp:shell/run" not in names and "mcp_shell_run" not in names
               for names in sub.tools_seen)
    note = _tool_turns(sub)[0]
    assert "mcp:shell/run exists within the person's access but not in this step" in note
    assert "report_capability_missing" in note and "yield_subgoal" not in note


def test_find_tools_miss_without_outside_tools_names_the_report_not_a_yield():
    note = react._find_tools_miss_note("fetch a page", [], True)
    assert "report_capability_missing" in note and "yield_subgoal" not in note
    note = react._find_tools_miss_note("fetch a page", [], False)
    assert "report_capability_missing" not in note and "yield_subgoal" not in note


def test_an_agent_that_may_not_yield_is_not_offered_the_verb():
    sub = _Scripted([("report_capability_missing", {"needed": "x"}),
                     ("final_answer", {"answer": "done"})])
    bot = Worker("w")
    bot.substrate = sub
    from cambrian_agent_sdk.react import run_think
    res = run_think(bot, AgentTask(text="t"), allow_yield_subgoal=False, max_tool_rounds=4)
    assert res.type != "capability_missing" and res.text == "done"
    assert "report_capability_missing" not in sub.tools_seen[0]
    assert "not available to you" in _tool_turns(sub)[0]


def test_an_empty_need_is_refused_and_the_loop_continues():
    sub = _Scripted([("report_capability_missing", {"needed": " "}),
                     ("final_answer", {"answer": "done"})])
    bot = Worker("w")
    bot.substrate = sub
    res = bot.think(AgentTask(text="t"), max_tool_rounds=4)
    assert res.text == "done"
    assert "needs 'needed'" in _tool_turns(sub)[0]


def test_agent_attribute_overrides_the_yield_default():
    class NoReport(Worker):
        allow_capability_missing = False

    sub = _Scripted([("final_answer", {"answer": "done"})])
    bot = NoReport("w")
    bot.substrate = sub
    bot.think(AgentTask(text="t"), max_tool_rounds=2)
    assert "report_capability_missing" not in sub.tools_seen[0]
    assert "yield_subgoal" in sub.tools_seen[0]


def test_native_verb_is_recognised_as_a_control_action():
    act = react.action_from_native_turn(
        "", [{"id": "c", "name": "report_capability_missing",
              "arguments": json.dumps({"needed": "x"})}], "tool_use", {})
    assert act["action"] == "report_capability_missing"


# ── T-20261112-worker-final-answer-envelope-value-has-no-json ───────────────

def test_envelope_value_has_a_json_type_and_a_list_is_an_array():
    defs, _ = react.build_tool_definitions(Worker("w"), [], include_final=True, final_envelope=True)
    fa = next(d for d in defs if d["name"] == "final_answer")
    value = (fa["parameters"]["properties"]["item_results"]["items"]["properties"]["values"]
             ["items"]["properties"]["value"])
    assert set(value["type"]) == {"string", "number", "boolean", "array", "object"}
    assert "list = a JSON array (never a string holding one)" in value["description"]
    # the declared kinds the description maps are exactly the kernel's vocabulary
    kinds = (fa["parameters"]["properties"]["item_results"]["items"]["properties"]["values"]
             ["items"]["properties"]["type"]["enum"])
    for kind in kinds:
        assert kind in value["description"], kind
    json.dumps(fa)  # serialisable as a provider tool schema
