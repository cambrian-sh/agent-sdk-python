"""ENG-2128 (owner decision 2026-11-18): reactive recovery from a context overflow.

When a model call is refused OUT_OF_RANGE (the kernel's pre-send window check from
ENG-1050, or the provider's own "prompt is too long"), the reasoning loop replaces
the NEWEST oversized tool result with its placeholder / ledger reference — the same
placeholder compaction uses — and retries the call ONCE. A second overflow, or no
tool result to shrink, fails the step as before (ContextOverflow, which the worker
reports as OUT_OF_RANGE). The system tier and the user's turn are never shrunk.

Unit half: DefaultContextEngine.shrink_for_overflow and run_think with scripted
substrates (native and text paths). End-to-end half: a real worker process booted
through ``serve()`` against an in-process kernel stand-in whose GenerateWithTools
answers OUT_OF_RANGE for an over-limit prompt; the turn completes and the request
the model finally saw carries the ledger reference instead of the tool output.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
from concurrent import futures
from pathlib import Path

import grpc
import pytest

from cambrian_agent_sdk import AgentTask, CognitiveAgent, ContextOverflow, tool
from cambrian_agent_sdk._proto import cambrian_pb2, cambrian_pb2_grpc
from cambrian_agent_sdk.context_engine import ContextConfig, DefaultContextEngine
from cambrian_agent_sdk.react import run_think

SDK_ROOT = Path(__file__).resolve().parents[1]
LEDGER_REF = "led_0123abcd"
BIG = 10_000          # over tool_output_compact_chars (1200), under tool_output_max_chars
LIMIT = 6_000         # the fake gateway's "context window", in chars of any one message


def _big_body(ref: str = LEDGER_REF, fill: str = "x") -> str:
    return json.dumps({"_ledger_ref": ref, "rows": fill * BIG})


def _conv():
    return [
        {"role": "user", "content": "the task " + "u" * 3000},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "name": "fetch", "arguments": "{}"}]},
        {"role": "tool", "tool_call_id": "c1", "content": _big_body("led_00000001", "a")},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "c2", "name": "fetch", "arguments": "{}"}]},
        {"role": "tool", "tool_call_id": "c2", "content": _big_body("led_00000002", "b")},
    ]


# ── unit: the engine ─────────────────────────────────────────────────────────


def test_native_shrink_replaces_only_the_newest_oversized_tool_result():
    eng = DefaultContextEngine(ContextConfig())
    conv = _conv()
    user, older = conv[0]["content"], conv[2]["content"]
    ev = eng.shrink_for_overflow(conv)
    assert ev["event"] == "overflow_shrink" and ev["path"] == "native"
    assert ev["tool"] == "fetch" and ev["ref"] == "led_00000002" and ev["tokens_saved"] > 0
    assert "led_00000002" in conv[4]["content"] and "ledger_get" in conv[4]["content"]
    assert len(conv[4]["content"]) < 500
    assert conv[2]["content"] == older, "only the newest result is shrunk"
    assert conv[0]["content"] == user, "the user's turn is never shrunk"
    assert eng.events[-1] == ev
    # The next overflow shrinks the next-newest; then there is nothing left.
    assert eng.shrink_for_overflow(conv)["ref"] == "led_00000001"
    assert eng.shrink_for_overflow(conv) is None
    assert conv[0]["content"] == user


def test_native_shrink_without_ledger_ref_uses_the_cid_placeholder():
    eng = DefaultContextEngine(ContextConfig())
    conv = _conv()
    conv[4]["content"] = "plain output " + "z" * BIG
    ev = eng.shrink_for_overflow(conv)
    assert ev["ref"] and not ev["ref"].startswith("led_")
    assert ev["ref"] in conv[4]["content"]
    assert eng.resolve_local(ev["ref"]) == "plain output " + "z" * BIG, "the raw body stays retrievable"


def test_native_shrink_skips_pinned_skill_turns_and_small_results():
    eng = DefaultContextEngine(ContextConfig())
    conv = _conv()
    eng.pin_tool_call("c2")
    pinned = conv[4]["content"]
    assert eng.shrink_for_overflow(conv)["ref"] == "led_00000001"
    assert conv[4]["content"] == pinned
    small = [{"role": "user", "content": "q"},
             {"role": "assistant", "content": "", "tool_calls": [{"id": "s", "name": "t", "arguments": "{}"}]},
             {"role": "tool", "tool_call_id": "s", "content": "short"}]
    assert eng.shrink_for_overflow(small) is None


def test_native_shrink_with_no_tool_result_is_none():
    eng = DefaultContextEngine(ContextConfig())
    assert eng.shrink_for_overflow([{"role": "user", "content": "u" * 50_000}]) is None


def test_text_shrink_replaces_the_newest_tool_step():
    eng = DefaultContextEngine(ContextConfig())
    eng.add_step(kind="tool_call", tool="fetch", args={}, status="ok", body="A" * BIG)
    eng.add_step(kind="tool_call", tool="fetch", args={"p": 2}, status="ok", body="B" * BIG)
    ev = eng.shrink_for_overflow()
    assert ev["path"] == "text" and ev["tool"] == "fetch" and ev["chars"] == BIG
    turn = eng.render_user_turn(task_text="the task", step_no=3, notes=[])
    assert "B" * 200 not in turn and "A" * 200 in turn
    assert ev["ref"] in turn and "the task" in turn
    assert eng.shrink_for_overflow()["path"] == "text"
    assert eng.shrink_for_overflow() is None


# ── unit: the loop ───────────────────────────────────────────────────────────


class FetchBot(CognitiveAgent):
    role = "a careful test assistant"

    @tool
    def fetch(self) -> str:
        """Fetch the rows."""
        return _big_body()

    def run(self, task):
        return self.think(task)


def _over(messages) -> bool:
    return any(len(m.get("content") or "") > LIMIT for m in messages)


class _NativeGateway:
    """generate_with_tools: call fetch first; OUT_OF_RANGE while any message is over
    LIMIT (``always`` keeps refusing); else a final answer."""

    def __init__(self, always: bool = False, call_tool: bool = True):
        self.always, self.call_tool = always, call_tool
        self.sent, self.systems = [], []

    def generate_with_tools(self, session_token_id=None, messages=None, tools=None, **kw):
        self.sent.append([dict(m) for m in messages])
        self.systems.append(kw.get("system"))
        if self.call_tool and not any(m.get("role") == "tool" for m in messages):
            if self.always and len(self.sent) > 1:
                raise ContextOverflow("prompt is too long", code="OUT_OF_RANGE")
            return ("", [{"id": "c1", "name": "fetch", "arguments": "{}"}], "tool_use")
        if self.always or _over(messages):
            raise ContextOverflow("prompt is too long: ~9000 tokens (limit 6000)", code="OUT_OF_RANGE")
        return ("", [{"id": "fa", "name": "final_answer", "arguments": '{"answer": "done"}'}], "tool_use")

    def execute_tool(self, *a, **kw):  # pragma: no cover - no system tools here
        raise AssertionError("no system tool expected")


def test_loop_shrinks_the_newest_tool_result_and_retries_once():
    sub = _NativeGateway()
    bot = FetchBot(agent_id="b")
    bot.substrate = sub
    res = run_think(bot, AgentTask(text="get the rows"), seed_system_tools=False)
    assert res.data == b"done"
    assert len(sub.sent) == 3, "call, refused call, one retry"
    refused, retried = sub.sent[1], sub.sent[2]
    tool_refused = next(m for m in refused if m["role"] == "tool")
    tool_retried = next(m for m in retried if m["role"] == "tool")
    assert len(tool_refused["content"]) > LIMIT
    # A local @tool's output is not in the run ledger: the cid placeholder, whose
    # body stays retrievable with resolve_cid.
    assert "resolve_cid" in tool_retried["content"] and len(tool_retried["content"]) < 500
    cid = tool_retried["content"].split('cid "')[1].split('"')[0]
    assert cid.startswith("ctx-")
    # The user's turn and the system tier are byte-identical across the retry.
    assert retried[0] == refused[0] == sub.sent[0][0]
    assert sub.systems[2] == sub.systems[1] == sub.systems[0]


def test_loop_second_overflow_fails_as_before():
    sub = _NativeGateway(always=True)
    bot = FetchBot(agent_id="b")
    bot.substrate = sub
    with pytest.raises(ContextOverflow):
        run_think(bot, AgentTask(text="get the rows"), seed_system_tools=False)
    assert len(sub.sent) == 3, "one retry, never a loop"


def test_loop_with_no_tool_result_fails_without_retry():
    class Refusing(_NativeGateway):
        def generate_with_tools(self, session_token_id=None, messages=None, tools=None, **kw):
            self.sent.append(list(messages))
            raise ContextOverflow("prompt is too long", code="OUT_OF_RANGE")

    sub = Refusing()
    bot = FetchBot(agent_id="b")
    bot.substrate = sub
    with pytest.raises(ContextOverflow):
        run_think(bot, AgentTask(text="get the rows"), seed_system_tools=False)
    assert len(sub.sent) == 1


def test_loop_other_errors_are_not_retried():
    class Broken(_NativeGateway):
        def generate_with_tools(self, session_token_id=None, messages=None, tools=None, **kw):
            self.sent.append(list(messages))
            if len(self.sent) == 1:
                return ("", [{"id": "c1", "name": "fetch", "arguments": "{}"}], "tool_use")
            raise RuntimeError("boom")

    sub = Broken()
    bot = FetchBot(agent_id="b")
    bot.substrate = sub
    with pytest.raises(RuntimeError):
        run_think(bot, AgentTask(text="get the rows"), seed_system_tools=False)
    assert len(sub.sent) == 2


class _TextGateway:
    """The text protocol (no native tools): one tool call, then OUT_OF_RANGE while
    the prompt carries the tool output."""

    def __init__(self):
        self.prompts = []

    def generate(self, session_token_id, prompt, **kw):
        self.prompts.append(prompt)
        if len(self.prompts) == 1:
            return json.dumps({"action": "tool_call", "tool": "fetch", "args": {}})
        if "x" * 2000 in prompt:
            raise ContextOverflow("prompt is too long", code="OUT_OF_RANGE")
        return json.dumps({"action": "final_answer", "answer": "done"})


def test_text_loop_shrinks_and_retries_once():
    sub = _TextGateway()
    bot = FetchBot(agent_id="b")
    bot.substrate = sub
    res = run_think(bot, AgentTask(text="get the rows"), seed_system_tools=False)
    assert res.data == b"done"
    assert len(sub.prompts) == 3
    assert "x" * 2000 in sub.prompts[1] and "x" * 2000 not in sub.prompts[2]
    assert "get the rows" in sub.prompts[2]


def test_shrink_is_a_trace_span(tmp_path, monkeypatch):
    from cambrian_agent_sdk import run_trace

    monkeypatch.setenv(run_trace.ENV_DIR, str(tmp_path))
    run_trace._reset_for_tests()
    try:
        bot = FetchBot(agent_id="b")
        bot.substrate = _NativeGateway()
        run_think(bot, AgentTask(text="get the rows"), seed_system_tools=False)
        spans = [json.loads(line) for f in tmp_path.glob("*.jsonl")
                 for line in f.read_text(encoding="utf-8").splitlines()]
    finally:
        run_trace._reset_for_tests()
    shrink = [s for s in spans if s["name"] == "context.overflow_shrink"]
    assert len(shrink) == 1
    attrs = shrink[0]["attributes"]
    assert attrs["cambrian.context.ref"].startswith("ctx-")
    assert attrs["cambrian.context.tool"] == "fetch"
    assert attrs["cambrian.context.path"] == "native"


# ── e2e: a real worker process against a kernel gateway stand-in ─────────────


class _Kernel(cambrian_pb2_grpc.OrchestratorServicer):
    """A kernel stand-in. GenerateWithTools behaves like the gateway since ENG-1050:
    OUT_OF_RANGE for a prompt over the window, else the scripted model (call the
    system tool fetch_rows, then answer). fetch_rows is a kernel system tool whose
    result carries the run-ledger reference, as the kernel's ExecuteTool does."""

    def __init__(self):
        self.requests = []
        self.tool_calls = 0

    def ListTools(self, request, context):
        return cambrian_pb2.ListToolsResponse(tools=[cambrian_pb2.ToolDescriptor(
            name="fetch_rows", description="Fetch the rows.",
            schema_json='{"type":"object","properties":{}}')])

    def ExecuteTool(self, request, context):
        self.tool_calls += 1
        return cambrian_pb2.ExecuteToolResponse(result_json=_big_body())

    def GenerateWithTools(self, request, context):
        self.requests.append(request)
        msgs = list(request.messages)
        if not any(m.role == "tool" for m in msgs):
            resp = cambrian_pb2.GenerateWithToolsResponse(
                stop_reason="tool_use",
                tool_calls=[cambrian_pb2.ModelToolCallProto(id="c1", name="fetch_rows", arguments_json="{}")])
        elif sum(len(m.content) for m in msgs) > LIMIT:
            # The whole conversation is over the window (the system tier aside).
            context.abort(grpc.StatusCode.OUT_OF_RANGE,
                          "prompt is too long: ~9000 tokens (limit 6000)")
        else:
            resp = cambrian_pb2.GenerateWithToolsResponse(
                stop_reason="tool_use",
                tool_calls=[cambrian_pb2.ModelToolCallProto(
                    id="fa", name="final_answer", arguments_json='{"answer": "rows fetched"}')])
        resp.usage.input_tokens = 1  # a C1 kernel: the system tier was honoured
        return resp

    def SignalStream(self, request_iterator, context):
        for _ in request_iterator:
            pass
        return iter(())


_WORKER = """
import sys
sys.path.insert(0, {sdk!r})
from cambrian_agent_sdk import CognitiveAgent

class Worker(CognitiveAgent):
    role = "a worker"

    def run(self, task):
        return self.think(task)

if __name__ == "__main__":
    Worker(agent_id="overflow_worker").serve()
"""


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_e2e_worker_recovers_from_out_of_range(tmp_path):
    kernel = _Kernel()
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=8))
    cambrian_pb2_grpc.add_OrchestratorServicer_to_server(kernel, server)
    kport = server.add_insecure_port("127.0.0.1:0")
    server.start()
    script = tmp_path / "overflow_worker.py"
    script.write_text(_WORKER.format(sdk=str(SDK_ROOT)), encoding="utf-8")
    aport = _free_port()
    env = {k: v for k, v in os.environ.items() if not k.startswith("CAMBRIAN_")}
    log = open(tmp_path / "worker.log", "wb")
    proc = subprocess.Popen(
        [sys.executable, str(script), "--substrate-addr", f"127.0.0.1:{kport}", "--port", str(aport)],
        env=env, stdout=log, stderr=subprocess.STDOUT,
    )
    chan = grpc.insecure_channel(f"localhost:{aport}")
    try:
        try:
            grpc.channel_ready_future(chan).result(timeout=30)
        except grpc.FutureTimeoutError:
            pytest.fail("worker never served:\n" + (tmp_path / "worker.log").read_text(errors="replace"))
        stub = cambrian_pb2_grpc.AgentServiceStub(chan)
        try:
            resp = stub.Execute(cambrian_pb2.Handoff(
                id="h-overflow", from_agent="kernel",
                payload=cambrian_pb2.Object(type="text", data=b"get the rows"),
                metadata={"_session_token_id": "lease-overflow"},
            ), timeout=60)
        except grpc.RpcError as exc:
            sizes = [[(m.role, len(m.content)) for m in r.messages] for r in kernel.requests]
            pytest.fail(f"Execute failed: {exc.code()} {exc.details()}; message sizes={sizes}")
    finally:
        chan.close()
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=10)
        log.close()
        server.stop(grace=None)
    worker_log = (tmp_path / "worker.log").read_text(errors="replace")
    assert resp.payload.data.decode() == "rows fetched", worker_log
    sizes = [[(m.role, len(m.content)) for m in r.messages] for r in kernel.requests]
    assert kernel.tool_calls == 1, (sizes, worker_log)
    assert len(kernel.requests) == 3, (sizes, worker_log)
    refused, served = list(kernel.requests[1].messages), list(kernel.requests[2].messages)
    assert any(m.role == "tool" and len(m.content) > LIMIT for m in refused)
    tool_turn = next(m for m in served if m.role == "tool")
    assert LEDGER_REF in tool_turn.content and "ledger_get" in tool_turn.content
    assert served[0].content == refused[0].content, "the user's turn is untouched"
    assert list(kernel.requests[2].system), "the system tier is sent"
    assert list(kernel.requests[2].system) == list(kernel.requests[1].system), "and never shrunk"
    assert "react_context_overflow_shrunk_retrying" in worker_log
