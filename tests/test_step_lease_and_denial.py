"""ADR-0145 D14 changes to the ReAct loop.

- The step's lease rides every tool-menu fetch: the run-start menu, a find_tools
  pull and a describe_tool fetch pass ``session_token_id=task.session_token_id`` to
  ``substrate.list_tools``, so the kernel resolves the principal the step runs for
  and the menu is what ExecuteTool will allow. A substrate (or fake) without the
  kwarg still works (TypeError fallback).
- The same kernel denial twice (same tool, same canonical args, same reason, the
  kernel's "already refused" suffix stripped) ends the loop with an ``error``
  result carrying the reason, without another model round.
- A run that ends because the model kept querying memory past its budget is an
  ``error``, never a "text" result a caller would read as the finished answer.
"""
import json

from cambrian_agent_sdk import AgentTask, CognitiveAgent
from cambrian_agent_sdk.react import _repeated_denial, run_think


class _Bot(CognitiveAgent):
    role = "a tool-using assistant"

    def run(self, task):
        return self.think(task)


def _act(**a):
    return json.dumps(a)


class _LeasedSub:
    """A text-path substrate whose list_tools takes the lease kwarg."""

    def __init__(self, replies, deny_reasons=None):
        self.replies, self.prompts = list(replies), []
        self.list_calls, self.exec_calls = [], []
        self.deny_reasons = list(deny_reasons or [])

    def generate(self, session_token_id=None, prompt="", **kw):
        self.prompts.append(prompt)
        return self.replies.pop(0)

    def list_tools(self, query="", k=0, names=None, full=False, session_token_id=""):
        self.list_calls.append({"query": query, "names": names, "full": full,
                                "session_token_id": session_token_id})
        spec = {"name": "crm_write", "description": "write a CRM record",
                "schema_json": '{"type":"object","properties":{"id":{"type":"string"}}}'}
        if names:
            return [spec] if "crm_write" in names else []
        return [spec]

    def execute_tool(self, name, args_json="", session_token_id="", timeout_ms=0, task_id=""):
        self.exec_calls.append((name, json.loads(args_json or "{}"), session_token_id))
        if self.deny_reasons:
            return {"result_json": "", "result_cid": "", "denied": True,
                    "deny_reason": self.deny_reasons.pop(0), "error": ""}
        return {"result_json": json.dumps({"ok": True}), "result_cid": "", "denied": False,
                "deny_reason": "", "error": ""}


class _OldSub(_LeasedSub):
    """A substrate without the lease kwarg (an older client or a test fake)."""

    def list_tools(self, query="", k=0, names=None, full=False):  # noqa: D401 — no lease kwarg
        return super().list_tools(query=query, k=k, names=names, full=full)


# ── the step lease on every menu fetch ───────────────────────────────


def test_menu_fetch_find_tools_and_describe_pass_the_step_lease():
    sub = _LeasedSub([
        _act(action="find_tools", query="crm"),
        _act(action="describe_tool", tool="crm_write"),
        _act(action="final_answer", answer="done", type="text"),
    ])
    bot = _Bot(agent_id="w")
    bot.substrate = sub
    res = run_think(bot, AgentTask(text="update the crm", session_token_id="step-lease-7"),
                    seed_recall=False)
    assert res.text == "done"
    assert sub.list_calls, "the menu was never fetched"
    assert all(c["session_token_id"] == "step-lease-7" for c in sub.list_calls), sub.list_calls
    assert any(c["names"] == ["crm_write"] and c["full"] for c in sub.list_calls)  # describe
    assert any(c["query"] == "crm" for c in sub.list_calls)  # find_tools
    assert any(c["query"] == "update the crm" and not c["names"] for c in sub.list_calls)  # menu


def test_a_substrate_without_the_lease_kwarg_still_gets_a_menu_and_a_describe():
    sub = _OldSub([
        _act(action="describe_tool", tool="crm_write"),
        _act(action="final_answer", answer="ok", type="text"),
    ])
    bot = _Bot(agent_id="w")
    bot.substrate = sub
    res = run_think(bot, AgentTask(text="t", session_token_id="lease"), seed_recall=False)
    assert res.text == "ok"
    assert "crm_write" in sub.prompts[0]  # the run-start menu
    assert "write a CRM record" in sub.prompts[-1]  # the describe_tool spec


# ── the same denial twice stops the loop ─────────────────────────────


_REFUSED = "approval denied by the owner"
_REPEAT_SUFFIX = (" (This identical call was already refused in this step; do not retry it "
                  "unchanged.)")


def test_the_same_denial_twice_ends_the_loop_as_an_error_without_another_round():
    call = _act(action="tool_call", tool="crm_write", args={"id": "42"})
    sub = _LeasedSub([call, call, _act(action="final_answer", answer="all done", type="text")],
                     deny_reasons=[_REFUSED, _REFUSED + _REPEAT_SUFFIX])
    bot = _Bot(agent_id="w")
    bot.substrate = sub
    res = run_think(bot, AgentTask(text="update record 42", session_token_id="lease"),
                    seed_recall=False)
    assert res.type == "error"
    assert _REFUSED in res.text and "already refused" not in res.text
    assert len(sub.exec_calls) == 2
    assert len(sub.prompts) == 2  # no model round after the second refusal
    assert sub.replies  # the scripted "all done" answer was never asked for


def test_different_args_or_reasons_are_not_the_same_denial():
    counts = {}
    deny = {"error": "denied: no grant", "tool": "t"}
    assert _repeated_denial(counts, "t", {"a": 1, "b": 2}, deny) == ""
    assert _repeated_denial(counts, "t", {"a": 2}, deny) == ""  # other args
    assert _repeated_denial(counts, "t", {"a": 1, "b": 2}, {"error": "denied: other"}) == ""
    assert _repeated_denial(counts, "u", {"a": 1, "b": 2}, deny) == ""  # other tool
    # A non-denial error and a success never count.
    assert _repeated_denial(counts, "t", {"a": 1, "b": 2}, {"error": "timeout"}) == ""
    assert _repeated_denial(counts, "t", {"a": 1, "b": 2}, {"ok": True}) == ""
    # Same tool, canonically equal args (key order), same reason after the suffix strip.
    again = {"error": "denied: no grant" + _REPEAT_SUFFIX, "tool": "t"}
    assert _repeated_denial(counts, "t", {"b": 2, "a": 1}, again) == "no grant"


def test_one_denial_then_success_does_not_stop():
    call = _act(action="tool_call", tool="crm_write", args={"id": "42"})
    sub = _LeasedSub([call, call, _act(action="final_answer", answer="written", type="text")],
                     deny_reasons=[_REFUSED])
    bot = _Bot(agent_id="w")
    bot.substrate = sub
    res = run_think(bot, AgentTask(text="update record 42", session_token_id="lease"),
                    seed_recall=False)
    assert res.type == "text" and res.text == "written"


# ── a forced stop is not a success ───────────────────────────────────


def test_memory_budget_exhaustion_is_an_error_not_an_answer():
    q = _act(action="memory_query", query="anything")
    sub = _LeasedSub([q, q, q])
    bot = _Bot(agent_id="w")
    bot.substrate = sub
    res = run_think(bot, AgentTask(text="t", session_token_id="lease"), seed_recall=False,
                    max_memory_queries=0)
    assert res.type == "error"
    assert "memory-query budget" in res.text
