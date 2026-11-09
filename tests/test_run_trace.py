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
    chats = [s for s in spans if s["attributes"]["gen_ai.operation.name"] == "chat"]
    tools = [s for s in spans if s["attributes"]["gen_ai.operation.name"] == "execute_tool"]
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
    # the tool menu is written in full once, then by hash
    assert "gen_ai.tool.definitions" in first
    assert "gen_ai.tool.definitions" not in chats[1]["attributes"]
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
