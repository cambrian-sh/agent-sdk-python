"""ADR-0145 D12: the composed runtime's substrate view (cambrian_agent_sdk.composition)."""
import json

from cambrian_agent_sdk.composition import COMPOSITION_KEY, ComposedScopedSubstrate, Composition, composition_from_task
from cambrian_agent_sdk.types import AgentTask


class _Inner:
    def __init__(self):
        self.calls = []

    def list_tools(self, query="", k=0, names=None, full=False, session_token_id=""):
        self.calls.append(("tools", session_token_id, names))
        return [{"name": "mcp:a/read"}, {"name": "ledger_find"}]

    def list_skills(self, query="", k=0, names=None, full=False, session_token_id=""):
        self.calls.append(("skills", session_token_id, names))
        return [{"name": n, "description": n, "instructions": "B" if full else ""}
                for n in (names or ["zeta", "alpha", "outside"])]

    def execute_tool(self, name, *a, **kw):
        self.calls.append(("exec", name))
        return {"result_json": "{}"}

    def embed(self, text):
        return [1.0]


_COMP = {"runtime": "rt", "skills": ["zeta", "alpha"], "tools": ["mcp:a/read"], "step": 3}


def test_parse_and_from_task():
    c = Composition.from_json(json.dumps(_COMP))
    assert c.skills == ("alpha", "zeta") and c.step == 3 and c.allows_tool("mcp:a/read")
    assert Composition.from_json("{}") is None and Composition.from_json("not json") is None
    t = AgentTask(text="x", context={COMPOSITION_KEY: json.dumps(_COMP)})
    assert composition_from_task(t).runtime == "rt"
    assert composition_from_task(AgentTask(text="x")) is None


def test_unbound_proxy_offers_nothing():
    sub = ComposedScopedSubstrate(_Inner())
    assert sub.list_tools() == [] and sub.list_skills() == []
    assert sub.execute_tool("mcp:a/read")["denied"]


def test_bound_proxy_presents_the_lease_and_keeps_the_composition():
    inner = _Inner()
    sub = ComposedScopedSubstrate(inner)
    sub.bind(Composition.from_json(json.dumps(_COMP)), "L9")
    assert [t["name"] for t in sub.list_tools(query="ignored", k=3)] == ["mcp:a/read", "ledger_find"]
    assert inner.calls[-1] == ("tools", "L9", None)
    menu = sub.list_skills(query="q")
    assert [s["name"] for s in menu] == ["alpha", "zeta"]  # composition only, sorted
    out = sub.list_skills(names=["outside"], full=True)
    assert out[0]["access"] == "denied" and out[0]["instructions"] == ""
    assert sub.execute_tool("mcp:b/write")["denied"]
    assert not sub.execute_tool("ledger_find").get("denied")  # served by the kernel's menu
    assert sub.embed("x") == [1.0]
    sub.bind(None, "")
    assert sub.execute_tool("mcp:a/read")["denied"]
