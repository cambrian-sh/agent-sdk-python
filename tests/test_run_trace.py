"""ADR-0145 D10: the ReAct loop writes an OTel-GenAI-shaped trace of every model
call (what was sent, what came back) and every tool call (arguments, result) when
the kernel sets CAMBRIAN_TRACE_DIR; without it nothing is written."""

import json

import pytest

from cambrian_agent_sdk import AgentTask, CognitiveAgent, run_trace, tool


class ToolBot(CognitiveAgent):
    role = "a test assistant"

    def run(self, task):
        return self.think(task)

    @tool
    def lookup(self, invoice: str) -> str:
        """Look up an invoice."""
        return "INV-" + invoice + " total 120.00"


class OneToolLLM:
    def __init__(self):
        self.turn = 0

    def generate_with_tools(self, session_token_id=None, messages=None, tools=None, **kw):
        self.turn += 1
        if self.turn == 1:
            return ("", [{"id": "call_1", "name": "lookup", "arguments": json.dumps({"invoice": "7"})}], "tool_use")
        return (json.dumps({"action": "final_answer", "answer": "total is 120.00", "type": "text"}), [], "stop")


@pytest.fixture
def trace_dir(tmp_path, monkeypatch):
    monkeypatch.setenv(run_trace.ENV_DIR, str(tmp_path))
    monkeypatch.setenv(run_trace.ENV_MAX_CHARS, "50")
    run_trace._reset_for_tests()
    yield tmp_path
    run_trace._reset_for_tests()


def _spans(d):
    files = list(d.glob("*.jsonl"))
    assert len(files) == 1, files
    return [json.loads(line) for line in files[0].read_text(encoding="utf-8").splitlines()]


def test_model_and_tool_calls_are_traced(trace_dir):
    bot = ToolBot("bot-1")
    bot.substrate = OneToolLLM()
    bot.think(AgentTask(text="what is the total of invoice 7?", session_token_id="lease-9"), max_tool_rounds=3)
    spans = _spans(trace_dir)
    done = [s for s in spans if s["attributes"].get(run_trace.PHASE_ATTR) != run_trace.PHASE_START]
    chats = [s for s in done if s["attributes"]["gen_ai.operation.name"] == "chat"]
    tools = [s for s in done if s["attributes"]["gen_ai.operation.name"] == "execute_tool"]
    assert len(chats) >= 2 and len(tools) == 1
    first = chats[0]["attributes"]
    assert first["cambrian.lease.id"] == "lease-9"
    assert first["gen_ai.agent.name"] == "bot-1"
    assert "lookup" in first["cambrian.tools.offered"]
    assert first["gen_ai.input.messages"][0]["role"] == "user"
    out = first["gen_ai.output.messages"][0]["parts"]
    assert out[0]["type"] == "tool_call" and out[0]["name"] == "lookup"
    t = tools[0]
    assert t["name"] == "execute_tool lookup"
    assert json.loads(t["attributes"]["gen_ai.tool.call.arguments"]) == {"invoice": "7"}
    assert "120.00" in t["attributes"]["gen_ai.tool.call.result"]
    assert t["attributes"]["gen_ai.tool.call.id"] == "call_1"
    # the tool menu is written in full once (on the first call's start event), then by hash
    full = [s for s in spans if "gen_ai.tool.definitions" in s["attributes"]]
    assert len(full) == 1 and full[0]["attributes"][run_trace.PHASE_ATTR] == run_trace.PHASE_START
    assert "gen_ai.tool.definitions" not in first
    assert chats[1]["attributes"]["gen_ai.tool.definitions_sha256"] == first["gen_ai.tool.definitions_sha256"]
    # one lease, one trace id
    assert len({s["trace_id"] for s in spans}) == 1


def test_content_is_bounded(trace_dir):
    w = run_trace.writer("x")
    w.span("chat m", {"gen_ai.input.messages": [{"role": "user", "parts": [{"type": "text", "content": "y" * 500}]}]})
    s = _spans(trace_dir)[0]
    content = s["attributes"]["gen_ai.input.messages"][0]["parts"][0]["content"]
    assert content.startswith("y" * 50) and "truncated 450 of 500" in content


def test_off_without_env(tmp_path, monkeypatch):
    monkeypatch.delenv(run_trace.ENV_DIR, raising=False)
    run_trace._reset_for_tests()
    bot = ToolBot("bot-1")
    bot.substrate = OneToolLLM()
    bot.think(AgentTask(text="x"), max_tool_rounds=3)
    assert run_trace.writer() is None
    run_trace._reset_for_tests()


class SlowLLM(OneToolLLM):
    """Snapshots the trace file while "the model" is answering."""

    def __init__(self, trace_dir):
        super().__init__()
        self.trace_dir = trace_dir
        self.seen_during_call = []

    def generate_with_tools(self, session_token_id=None, messages=None, tools=None, **kw):
        self.seen_during_call.append(_spans(self.trace_dir))
        return super().generate_with_tools(session_token_id, messages, tools, **kw)


def test_start_events_precede_their_spans(trace_dir):
    """Live view (T-20261109-live-run-viewer): every model and tool call writes a
    start event with the request before it runs; the finished span shares its id."""
    bot = ToolBot("bot-1")
    llm = SlowLLM(trace_dir)
    bot.substrate = llm
    bot.think(AgentTask(text="what is the total of invoice 7?", session_token_id="lease-9",
                        context={"_composition": json.dumps({"runtime": "composed_worker",
                                                             "skills": ["ar-collections"], "step": 2})}),
              max_tool_rounds=3)
    # while the first model call ran, its start event (and nothing else) was on file
    during = llm.seen_during_call[0]
    assert len(during) == 1
    st = during[0]
    a = st["attributes"]
    assert a[run_trace.PHASE_ATTR] == run_trace.PHASE_START and st["end_time_unix_nano"] == 0
    assert a["gen_ai.input.messages"][0]["role"] == "user"
    assert "lookup" in a["cambrian.tools.offered"]
    assert "gen_ai.output.messages" not in a
    assert a["cambrian.composition.skills"] == ["ar-collections"] and a["cambrian.step.index"] == 2
    spans = _spans(trace_dir)
    starts = {s["span_id"]: s for s in spans if s["attributes"].get(run_trace.PHASE_ATTR) == run_trace.PHASE_START}
    ends = [s for s in spans if s["attributes"].get(run_trace.PHASE_ATTR) is None]
    assert len(starts) == len(ends) and len(ends) >= 3
    for e in ends:
        assert e["span_id"] in starts, e["name"]
        s = starts[e["span_id"]]
        assert spans.index(s) < spans.index(e)
        assert s["start_time_unix_nano"] == e["start_time_unix_nano"]
    tool_start = next(s for s in starts.values() if s["attributes"]["gen_ai.operation.name"] == "execute_tool")
    assert json.loads(tool_start["attributes"]["gen_ai.tool.call.arguments"]) == {"invoice": "7"}
    assert "gen_ai.tool.call.result" not in tool_start["attributes"]
    assert tool_start["attributes"]["cambrian.tool.executed_by"] == "agent"  # a local @tool


def test_start_helpers_are_noops_without_env(tmp_path, monkeypatch):
    monkeypatch.delenv(run_trace.ENV_DIR, raising=False)
    run_trace._reset_for_tests()
    assert run_trace.model_call_start(object(), AgentTask(), path="native", round_no=1) == ""
    assert run_trace.tool_call_start(object(), AgentTask(), name="x", args={}) == ""
    assert list(tmp_path.iterdir()) == []
    run_trace._reset_for_tests()
