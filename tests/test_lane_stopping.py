"""ADR-0150 D4/D5/D7: the single-agent lane's stopping rule, top-level ask-back,
and the context engine's age trigger with ledger-id placeholders."""
import json

from cambrian_agent_sdk import AgentTask, CognitiveAgent, tool
from cambrian_agent_sdk.context_engine import ContextConfig, DefaultContextEngine, ledger_ref_of
from cambrian_agent_sdk.react import run_think
from cambrian_agent_sdk import stopping
from cambrian_agent_sdk.stopping import LaneClock, LaneLimits, StuckDetector


class Bot(CognitiveAgent):
    role = "a careful test assistant"

    def run(self, task):
        return self.think(task)


class ToolBot(Bot):
    calls = 0

    @tool
    def lookup(self, key: str) -> str:
        """Look a key up."""
        ToolBot.calls += 1
        return "same answer"

    @tool
    def counter(self, n: int) -> str:
        """Count."""
        return f"value {n}"


class _LLM:
    def __init__(self, turns, text=None):
        self._turns = list(turns)
        self._text = list(text or [])
        self.prompts = []
        self.last_tools = []

    def generate_with_tools(self, session_token_id=None, messages=None, tools=None, **kw):
        self.last_tools = [t["name"] for t in (tools or [])]
        return self._turns.pop(0)

    def generate(self, session_token_id=None, prompt="", **kw):
        self.prompts.append(prompt)
        return self._text.pop(0) if self._text else "summary of what I have"


def _call(i, name, args):
    return ("", [{"id": f"c{i}", "name": name, "arguments": json.dumps(args)}], "tool_use")


def _final(text="done"):
    return ("", [{"id": "f", "name": "final_answer", "arguments": json.dumps({"answer": text})}], "tool_use")


def _lane_task(rounds=500, wall=3600, same=4, fail=3, **extra):
    ctx = {"_lane": "single_agent", "_lane_max_tool_rounds": str(rounds),
           "_lane_max_wall_seconds": str(wall), "_lane_stuck_same_result": str(same),
           "_lane_stuck_same_failure": str(fail)}
    ctx.update(extra)
    return AgentTask(text="find it", context=ctx)


def _bot(sub):
    bot = ToolBot(agent_id="b")
    bot.substrate = sub
    return bot


def test_lane_limits_read_from_the_kernel_keys():
    assert LaneLimits.from_task(AgentTask(text="x")) is None
    lim = LaneLimits.from_task(_lane_task(rounds=7, wall=60))
    assert (lim.max_tool_rounds, lim.max_wall_seconds, lim.same_result_repeats,
            lim.same_failure_repeats) == (7, 60, 4, 3)


def test_cap_ends_in_a_forced_summary_marked_partial():
    """D4.3: at the tool-round cap, one tool-less summary call; partial, not an error."""
    sub = _LLM([_call(i, "counter", {"n": i}) for i in range(10)],
               text=["Found values 0-2; the rest is missing."])
    res = run_think(_bot(sub), _lane_task(rounds=3), seed_system_tools=False, recurrence_enabled=False)
    assert res.metadata.get("partial") == "true" and res.metadata.get("exit_reason") == "cap"
    assert res.data.decode() == "Found values 0-2; the rest is missing."
    assert len(sub.prompts) == 1 and "No tool will run" in sub.prompts[0]


def test_wall_cap_ends_partial(monkeypatch):
    t = [1000.0]
    monkeypatch.setattr(stopping.time, "monotonic", lambda: t[0])
    clock = LaneClock(max_wall_seconds=1, now=lambda: t[0], started=t[0])
    assert not clock.expired()
    t[0] += 5
    assert clock.expired()
    assert not LaneClock(max_wall_seconds=0, now=lambda: t[0], started=0.0).expired()

    # In the loop: the clock is already past the cap when the first round starts.
    class _Clock(LaneClock):
        def expired(self):
            return True

    monkeypatch.setattr("cambrian_agent_sdk.react.LaneClock", _Clock)
    sub = _LLM([_call(0, "counter", {"n": 0})], text=["partial"])
    res = run_think(_bot(sub), _lane_task(wall=1), seed_system_tools=False)
    assert res.metadata == {"partial": "true", "exit_reason": "wall"}
    assert res.data.decode() == "partial"


def test_stuck_same_result_corrects_once_then_ends_partial():
    """D4.4: the same action with the same result 4x -> one corrective note; repeated
    anyway -> partial with exit_reason stuck."""
    sub = _LLM([_call(i, "lookup", {"key": "a"}) for i in range(8)])
    ToolBot.calls = 0
    res = run_think(_bot(sub), _lane_task(), seed_system_tools=False, recurrence_enabled=False)
    assert res.metadata.get("exit_reason") == "stuck"
    assert ToolBot.calls == 5  # 4 to trip the detector, 1 more after the correction


def test_stuck_detector_thresholds():
    d = StuckDetector(same_result_repeats=4, same_failure_repeats=3)
    err = {"error": "boom"}
    assert [d.observe("t", {"a": 1}, err) for _ in range(3)] == [None, None, "same_failure"]
    assert d.observe("t", {"a": 1}, err) == "stuck"
    d2 = StuckDetector(4, 3)
    assert all(d2.observe("t", {}, f"r{i}") is None for i in range(10)), "different results are progress"
    d3 = StuckDetector(1, 1)
    assert all(d3.observe("t", {}, "same") is None for _ in range(10)), "1 disables it"


def test_outside_the_lane_nothing_changes():
    sub = _LLM([_call(i, "counter", {"n": i}) for i in range(5)] + [_final()])
    res = run_think(_bot(sub), AgentTask(text="x"), seed_system_tools=False, recurrence_enabled=False)
    assert res.data.decode() == "done" and not res.metadata
    assert "ask_person" not in sub.last_tools


def test_top_level_ask_back_returns_input_required():
    """D5: the lane's main agent may stop to ask; the result is input_required."""
    sub = _LLM([("", [{"id": "a", "name": "ask_person",
                       "arguments": json.dumps({"questions": ["Which ACME?"], "reason": "two match"})}],
                 "tool_use")])
    res = run_think(_bot(sub), _lane_task(), seed_system_tools=False)
    assert "ask_person" in sub.last_tools
    assert res.type == "input_required"
    body = json.loads(res.data)
    assert body["questions"][0]["question"] == "Which ACME?" and body["reason"] == "two match"


def test_subtasks_and_chat_turns_never_get_ask_person():
    sub = _LLM([_final("ok")])
    run_think(_bot(sub), _lane_task(_subtask_id="st-1"), seed_system_tools=False)
    assert "ask_person" not in sub.last_tools
    sub2 = _LLM([_final("ok")])
    run_think(_bot(sub2), _lane_task(), seed_system_tools=False, allow_ask_person=False)
    assert "ask_person" not in sub2.last_tools


# ── D7: age trigger and ledger-id placeholders ───────────────────────────────

def _conversation(n, body_of):
    conv = [{"role": "user", "content": "task"}]
    for i in range(n):
        conv.append({"role": "assistant", "content": "",
                     "tool_calls": [{"id": f"t{i}", "name": "fetch", "arguments": "{}"}]})
        conv.append({"role": "tool", "tool_call_id": f"t{i}", "content": body_of(i)})
    return conv


def test_age_trigger_clears_old_tool_results_under_budget_with_ledger_ids():
    eng = DefaultContextEngine(ContextConfig(budget_tokens=10**9, keep_recent_tool_results=2))
    conv = _conversation(5, lambda i: json.dumps({"_ledger_ref": f"led_{i:032x}", "rows": "x" * 3000}))
    assert eng.compact_conversation(conv, 0)
    tools = [m for m in conv if m["role"] == "tool"]
    for m in tools[:3]:
        assert "ledger_get" in m["content"] and "led_" in m["content"], m["content"][:120]
    for m in tools[3:]:
        assert m["content"].startswith("{"), "the newest results are kept"


def test_age_trigger_off_by_default_and_sdk_cid_without_ledger():
    eng = DefaultContextEngine(ContextConfig(budget_tokens=10**9))
    assert not eng.compact_conversation(_conversation(5, lambda i: "y" * 3000), 0)
    eng2 = DefaultContextEngine(ContextConfig(budget_tokens=10**9, keep_recent_tool_results=1))
    conv2 = _conversation(3, lambda i: "y" * 3000)
    assert eng2.compact_conversation(conv2, 0)
    assert "resolve_cid" in conv2[2]["content"]


def test_keep_recent_tool_results_from_env(monkeypatch):
    monkeypatch.setenv("CAMBRIAN_CONTEXT_KEEP_RECENT_TOOL_RESULTS", "6")
    assert ContextConfig.from_env().keep_recent_tool_results == 6


def test_ledger_ref_of():
    assert ledger_ref_of('{"cid": "led_ab12", "x": 1}') == "led_ab12"
    assert ledger_ref_of('{"cid": "ctx-1"}') == ""
