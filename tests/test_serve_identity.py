"""ENG-2129: ``CognitiveAgent.serve()`` asserts the registry id the kernel launched it as.

The kernel mints a pool worker's per-instance token for its REGISTRY id (derived from
the filename) and passes that id in ``CAMBRIAN_AGENT_ID``. A class that declares another
``agent_id`` used to assert the declared one as ``x-agent-id``, and the kernel's
``authz.AgentCredentialVerifier`` refused every agent->kernel call UNAUTHENTICATED
(token's AgentID != asserted). Owner decision 2026-11-18: the registry id wins, the
way ``daemon.py`` adopts ``--agent-id``; a mismatch warns once; unset keeps the
declared id.

Unit half: :func:`channel.kernel_agent_id` and the adoption in ``serve()``.

End-to-end half: a real agent process booted through ``serve()`` (declares ``hb_probe``,
launched as ``hb_worker`` with a token minted for ``hb_worker`` — the ENG-1348 e2e's
shape) against an in-process kernel stand-in whose server interceptor enforces the
same rule as the kernel's verifier: the bearer token names an agent, and the asserted
``x-agent-id`` must equal it. The agent's ``run()`` makes the memory, LLM gateway and
tool calls; all must be accepted, and the reply's ``from_agent`` is the registry id.
"""

from __future__ import annotations

import logging
import os
import socket
import subprocess
import sys
import threading
import time
from concurrent import futures
from pathlib import Path

import grpc
import pytest

from cambrian_agent_sdk import CognitiveAgent
from cambrian_agent_sdk import channel as ch
from cambrian_agent_sdk._proto import cambrian_pb2, cambrian_pb2_grpc

SDK_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in (ch.TOKEN_ENV, ch.AGENT_ID_ENV, ch.TLS_CA_ENV, ch.TLS_SERVER_NAME_ENV):
        monkeypatch.delenv(name, raising=False)
    ch._reset_identity_warning_for_tests()
    yield
    ch._reset_identity_warning_for_tests()


# ── unit: kernel_agent_id ────────────────────────────────────────────────────


def test_env_wins_over_the_declared_id(monkeypatch):
    monkeypatch.setenv(ch.AGENT_ID_ENV, "hb_worker")
    assert ch.kernel_agent_id("hb_probe") == "hb_worker"


def test_env_unset_keeps_the_declared_id():
    assert ch.kernel_agent_id("hb_probe") == "hb_probe"


def test_blank_env_keeps_the_declared_id(monkeypatch):
    monkeypatch.setenv(ch.AGENT_ID_ENV, "   ")
    assert ch.kernel_agent_id("hb_probe") == "hb_probe"


def test_mismatch_warns_once_naming_both(monkeypatch, caplog):
    monkeypatch.setenv(ch.AGENT_ID_ENV, "hb_worker")
    with caplog.at_level(logging.WARNING, logger="cambrian.channel"):
        ch.kernel_agent_id("hb_probe")
        ch.kernel_agent_id("hb_probe")
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    msg = warnings[0].getMessage()
    assert "hb_probe" in msg and "hb_worker" in msg


def test_matching_ids_do_not_warn(monkeypatch, caplog):
    monkeypatch.setenv(ch.AGENT_ID_ENV, "chat_agent")
    with caplog.at_level(logging.WARNING, logger="cambrian.channel"):
        assert ch.kernel_agent_id("chat_agent") == "chat_agent"
    assert not [r for r in caplog.records if r.levelno == logging.WARNING]


# ── unit: serve() adopts it before binding the clients ───────────────────────


class _Probe(CognitiveAgent):
    def run(self, task):
        return "ok"


def _serve_capturing(monkeypatch, agent):
    """Run serve() with the channel and server stubbed; return (channel agent_id, bound agent_id)."""
    seen = {}

    def fake_build_channel(addr, options=None, agent_id=""):
        seen["channel"] = agent_id
        return grpc.insecure_channel("127.0.0.1:1")

    monkeypatch.setattr(ch, "build_channel", fake_build_channel)
    import cambrian_agent_sdk.runtime as rt

    monkeypatch.setattr(rt, "start_agent_server", lambda a, addr, **kw: seen.setdefault("served", a.agent_id))
    agent.serve("127.0.0.1:0")
    return seen["channel"], agent.memory._agent_id, agent.substrate._agent_id, seen["served"]


def test_serve_binds_the_registry_id(monkeypatch):
    monkeypatch.setenv(ch.AGENT_ID_ENV, "hb_worker")
    agent = _Probe(agent_id="hb_probe")
    assert _serve_capturing(monkeypatch, agent) == ("hb_worker",) * 4
    assert agent.agent_id == "hb_worker"


def test_serve_outside_the_kernel_keeps_the_declared_id(monkeypatch):
    agent = _Probe(agent_id="hb_probe")
    assert _serve_capturing(monkeypatch, agent) == ("hb_probe",) * 4


# ── e2e: a real agent process against a verifying kernel stand-in ────────────

_TOKEN = "tok-hb-worker-instance-1"
_TOKENS = {_TOKEN: "hb_worker"}  # what the kernel minted: token -> registry id


class _Verifier(grpc.ServerInterceptor):
    """The kernel's AgentCredentialVerifier rule: the bearer names an agent and the
    asserted x-agent-id must be that agent; anything else is UNAUTHENTICATED."""

    def __init__(self):
        self.accepted = []  # (method, agent)
        self.refused = []  # (method, asserted)
        self._lock = threading.Lock()

    def intercept_service(self, continuation, details):
        md = {k.lower(): v for k, v in (details.invocation_metadata or ())}
        token = md.get("authorization", "").removeprefix("Bearer ").strip()
        asserted = md.get("x-agent-id", "")
        owner = _TOKENS.get(token)
        method = details.method.rsplit("/", 1)[-1]
        with self._lock:
            if owner is None or asserted != owner:
                self.refused.append((method, asserted))
                refuse = True
            else:
                self.accepted.append((method, owner))
                refuse = False
        if not refuse:
            return continuation(details)
        handler = continuation(details)

        def deny_unary(req, ctx):
            ctx.abort(grpc.StatusCode.UNAUTHENTICATED, f"token is not {asserted!r}'s")

        def deny_stream(req_or_iter, ctx):
            ctx.abort(grpc.StatusCode.UNAUTHENTICATED, f"token is not {asserted!r}'s")
            yield  # pragma: no cover

        if handler is None:
            return None
        if handler.unary_unary:
            return grpc.unary_unary_rpc_method_handler(deny_unary)
        if handler.unary_stream:
            return grpc.unary_stream_rpc_method_handler(deny_stream)
        if handler.stream_stream:
            return grpc.stream_stream_rpc_method_handler(deny_stream)
        return grpc.stream_unary_rpc_method_handler(deny_unary)


class _Kernel(cambrian_pb2_grpc.OrchestratorServicer):
    def QueryMemory(self, request, context):
        return cambrian_pb2.MemoryResponse()

    def IngestMemory(self, request, context):
        return cambrian_pb2.IngestMemoryResponse(doc_id="doc-1")

    def GenerateViaModelStream(self, request, context):
        yield cambrian_pb2.GenerateChunk(text="generated", is_final=True)

    def ExecuteTool(self, request, context):
        return cambrian_pb2.ExecuteToolResponse(result_json='{"ok":true}')

    def ListTools(self, request, context):
        return cambrian_pb2.ListToolsResponse()

    def SignalStream(self, request_iterator, context):
        for _ in request_iterator:
            pass
        return iter(())


_PROBE = """
import sys
sys.path.insert(0, {sdk!r})
from cambrian_agent_sdk import CognitiveAgent

class Probe(CognitiveAgent):
    def run(self, task):
        out = []
        out.append("recall=%d" % len(self.memory.recall("q")))
        out.append("remember=" + self.memory.remember("a fact"))
        out.append("gen=" + self.substrate.generate(task.session_token_id, prompt="hi"))
        r = self.substrate.execute_tool("echo", "{{}}", session_token_id=task.session_token_id)
        out.append("tool=" + (r["result_json"] or r["error"]))
        out.append("tools=%d" % len(self.substrate.list_tools()))
        return ";".join(out)

if __name__ == "__main__":
    Probe(agent_id="hb_probe").serve()
"""


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _boot(tmp_path, env_agent_id):
    verifier = _Verifier()
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=8), interceptors=[verifier])
    cambrian_pb2_grpc.add_OrchestratorServicer_to_server(_Kernel(), server)
    kport = server.add_insecure_port("127.0.0.1:0")
    server.start()

    script = tmp_path / "hb_worker.py"
    script.write_text(_PROBE.format(sdk=str(SDK_ROOT)), encoding="utf-8")
    aport = _free_port()
    env = {k: v for k, v in os.environ.items() if not k.startswith("CAMBRIAN_")}
    env[ch.TOKEN_ENV] = _TOKEN
    if env_agent_id:
        env[ch.AGENT_ID_ENV] = env_agent_id
    proc = subprocess.Popen(
        [sys.executable, str(script), "--substrate-addr", f"127.0.0.1:{kport}", "--port", str(aport),
         "--heartbeat-interval-ms", "200", "--stream-id", "chat-hb_worker-0"],
        env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    return verifier, server, proc, aport, kport


def _execute(aport, proc):
    chan = grpc.insecure_channel(f"localhost:{aport}")
    try:
        grpc.channel_ready_future(chan).result(timeout=30)
    except grpc.FutureTimeoutError:
        proc.kill()
        out = proc.communicate(timeout=10)[0].decode(errors="replace")
        pytest.fail(f"agent never served:\n{out}")
    stub = cambrian_pb2_grpc.AgentServiceStub(chan)
    resp = stub.Execute(cambrian_pb2.Handoff(
        id="h1", from_agent="kernel",
        payload=cambrian_pb2.Object(type="text", data=b"go"),
        metadata={"_session_token_id": "lease-1"},
    ), timeout=30)
    chan.close()
    return resp


def _stop(server, proc):
    proc.terminate()
    try:
        proc.communicate(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.communicate(timeout=10)
    server.stop(grace=None)


def test_e2e_pool_worker_with_a_different_declared_id_is_accepted_on_every_call(tmp_path):
    verifier, server, proc, aport, kport = _boot(tmp_path, env_agent_id="hb_worker")
    try:
        resp = _execute(aport, proc)
        text = resp.payload.data.decode()
        assert text == "recall=0;remember=doc-1;gen=generated;tool={\"ok\":true};tools=0", text
        assert resp.from_agent == "hb_worker"
        deadline = time.time() + 10  # the heartbeat stream rides the same credential
        while time.time() < deadline and not any(m == "SignalStream" for m, _ in verifier.accepted):
            time.sleep(0.05)
    finally:
        _stop(server, proc)

    assert verifier.refused == [], verifier.refused
    methods = {m for m, _ in verifier.accepted}
    assert {"QueryMemory", "IngestMemory", "GenerateViaModelStream", "ExecuteTool", "ListTools",
            "SignalStream"} <= methods, methods
    assert {a for _, a in verifier.accepted} == {"hb_worker"}

    # Control: the stand-in really enforces the pairing — the declared id with the
    # registry id's token is what the kernel refused before ENG-2129.
    server2 = grpc.server(futures.ThreadPoolExecutor(max_workers=2), interceptors=[_Verifier()])
    cambrian_pb2_grpc.add_OrchestratorServicer_to_server(_Kernel(), server2)
    p2 = server2.add_insecure_port("127.0.0.1:0")
    server2.start()
    try:
        stub = cambrian_pb2_grpc.OrchestratorStub(grpc.insecure_channel(f"127.0.0.1:{p2}"))
        with pytest.raises(grpc.RpcError) as exc:
            stub.QueryMemory(cambrian_pb2.MemoryRequest(query="q"), timeout=10,
                             metadata=[("authorization", "Bearer " + _TOKEN), ("x-agent-id", "hb_probe")])
        assert exc.value.code() == grpc.StatusCode.UNAUTHENTICATED
    finally:
        server2.stop(grace=None)
