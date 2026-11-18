"""ENG-2125 (owner decision 2026-11-18): a worker whose ``run()`` ended on a gateway
verdict answers its Execute with that verdict's code, not INTERNAL.

The kernel's LLM gateway answers a refused provider credential UNAUTHENTICATED, a
lease refused to the caller PERMISSION_DENIED, a 429 that outlived the backoff
RESOURCE_EXHAUSTED (+ ``google.rpc.RetryInfo``), a 5xx / unanswered endpoint
UNAVAILABLE, and a prompt over the window OUT_OF_RANGE. In the single-agent lane the
agent is an SDK worker, so the kernel only sees what the worker's Execute says; as
INTERNAL the person read ``StatusCode.INTERNAL: UNAUTHENTICATED: openai: HTTP 401``.

Unit half: the chain lookup, the RetryInfo codec and TraitServicer with a context
double. End-to-end half: a real worker process booted through ``serve()`` against an
in-process kernel stand-in whose ``GenerateViaModelStream`` answers each code (the
lease id picks it); the worker's Execute status must carry the same code, the
RESOURCE_EXHAUSTED one the same RetryInfo, and anything else stays INTERNAL.
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
from concurrent import futures
from pathlib import Path

import grpc
import pytest

from cambrian_agent_sdk import CognitiveAgent, GenerateError, _map_generate_error
from cambrian_agent_sdk._proto import cambrian_pb2, cambrian_pb2_grpc
from cambrian_agent_sdk._rpc_status import (
    STATUS_DETAILS_KEY,
    _messages,
    retry_delay_from_rpc,
    status_details_bin,
)
from cambrian_agent_sdk.errors import RELAYED_GATEWAY_CODES, relayed_gateway_error
from cambrian_agent_sdk.runtime import TraitServicer

SDK_ROOT = Path(__file__).resolve().parents[1]
RELAYED = ["UNAUTHENTICATED", "PERMISSION_DENIED", "RESOURCE_EXHAUSTED", "UNAVAILABLE", "OUT_OF_RANGE"]


class _Rpc(grpc.RpcError):
    def __init__(self, code, details, trailers=()):
        self._c, self._d, self._t = code, details, trailers

    def code(self):
        return self._c

    def details(self):
        return self._d

    def trailing_metadata(self):
        return self._t


class _Ctx:
    def __init__(self):
        self.code, self.details, self.trailers = None, "", ()

    def time_remaining(self):
        return None

    def set_code(self, code):
        self.code = code

    def set_details(self, details):
        self.details = details

    def set_trailing_metadata(self, md):
        self.trailers = md


def _execute_raising(exc):
    class Agent(CognitiveAgent):
        def run(self, task):
            raise exc

    ctx = _Ctx()
    TraitServicer(Agent(agent_id="w")).Execute(cambrian_pb2.Handoff(id="h"), ctx)
    return ctx


# ── unit ─────────────────────────────────────────────────────────────────────


def test_relayed_codes_are_the_owner_decided_set():
    assert RELAYED_GATEWAY_CODES == set(RELAYED)


@pytest.mark.parametrize("name", RELAYED)
def test_execute_relays_each_gateway_code(name):
    code = getattr(grpc.StatusCode, name)
    gen = _map_generate_error(_Rpc(code, "openai: HTTP 401"), tools=False)
    ctx = _execute_raising(gen)
    assert ctx.code == code
    assert ctx.details == f"{name}: openai: HTTP 401"
    assert ctx.trailers == ()  # no RetryInfo was sent, none is invented


@pytest.mark.parametrize("name", RELAYED)
def test_execute_relays_through_the_cause_chain(name):
    gen = GenerateError("down", code=name)
    try:
        try:
            raise gen
        except GenerateError as inner:
            raise RuntimeError("step failed") from inner
    except RuntimeError as outer:
        wrapped = outer
    assert relayed_gateway_error(wrapped) is gen
    assert _execute_raising(wrapped).code == getattr(grpc.StatusCode, name)


@pytest.mark.parametrize("name", ["INTERNAL", "INVALID_ARGUMENT", "ABORTED", "UNKNOWN", "NOT_FOUND"])
def test_other_gateway_codes_stay_internal(name):
    gen = _map_generate_error(_Rpc(getattr(grpc.StatusCode, name), "x"), tools=False)
    assert relayed_gateway_error(gen) is None
    assert _execute_raising(gen).code == grpc.StatusCode.INTERNAL


def test_author_bug_stays_internal():
    ctx = _execute_raising(RuntimeError("author bug"))
    assert ctx.code == grpc.StatusCode.INTERNAL and ctx.details == "author bug"


def test_retry_info_round_trips():
    raw = status_details_bin(grpc.StatusCode.RESOURCE_EXHAUSTED.value[0], "slow down", 2.5)
    Status = _messages()[0]
    st = Status.FromString(raw)
    assert st.code == 8 and st.message == "slow down" and len(st.details) == 1
    assert st.details[0].type_url == "type.googleapis.com/google.rpc.RetryInfo"
    rpc = _Rpc(grpc.StatusCode.RESOURCE_EXHAUSTED, "slow down", ((STATUS_DETAILS_KEY, raw),))
    assert retry_delay_from_rpc(rpc) == pytest.approx(2.5)


def test_retry_delay_absent_or_unreadable_is_none():
    assert retry_delay_from_rpc(_Rpc(grpc.StatusCode.RESOURCE_EXHAUSTED, "x")) is None
    assert retry_delay_from_rpc(_Rpc(grpc.StatusCode.RESOURCE_EXHAUSTED, "x",
                                     ((STATUS_DETAILS_KEY, b"\xff\xff garbage"),))) is None

    class Bare(grpc.RpcError):
        pass

    assert retry_delay_from_rpc(Bare()) is None


def test_resource_exhausted_carries_retry_info_through_execute():
    raw = status_details_bin(8, "429", 7.0)
    gen = _map_generate_error(_Rpc(grpc.StatusCode.RESOURCE_EXHAUSTED, "openai: HTTP 429",
                                   ((STATUS_DETAILS_KEY, raw),)), tools=True)
    assert gen.retry_delay == pytest.approx(7.0)
    ctx = _execute_raising(gen)
    assert ctx.code == grpc.StatusCode.RESOURCE_EXHAUSTED
    (key, value), = ctx.trailers
    assert key == STATUS_DETAILS_KEY
    st = _messages()[0].FromString(value)
    # grpc-go rejects a details trailer whose code disagrees with the status.
    assert st.code == 8 and st.message == ctx.details
    assert retry_delay_from_rpc(_Rpc(ctx.code, ctx.details, ctx.trailers)) == pytest.approx(7.0)


# ── e2e: a real worker process against a kernel gateway stand-in ─────────────


class _Kernel(cambrian_pb2_grpc.OrchestratorServicer):
    """The gateway answers the code the lease id names ("lease-<CODE>")."""

    def GenerateViaModelStream(self, request, context):
        name = request.lease_id.removeprefix("lease-")
        if name == "OK":
            yield cambrian_pb2.GenerateChunk(text="fine", is_final=True)
            return
        code = getattr(grpc.StatusCode, name)
        msg = f"openai: provider says {name}"
        if code == grpc.StatusCode.RESOURCE_EXHAUSTED:
            context.set_trailing_metadata(((STATUS_DETAILS_KEY, status_details_bin(8, msg, 3.0)),))
        context.abort(code, msg)

    def SignalStream(self, request_iterator, context):
        for _ in request_iterator:
            pass
        return iter(())


_WORKER = """
import sys
sys.path.insert(0, {sdk!r})
from cambrian_agent_sdk import CognitiveAgent

class Worker(CognitiveAgent):
    def run(self, task):
        return self.substrate.generate(task.session_token_id, prompt="hi")

if __name__ == "__main__":
    Worker(agent_id="relay_worker").serve()
"""


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def worker(tmp_path_factory):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=8))
    cambrian_pb2_grpc.add_OrchestratorServicer_to_server(_Kernel(), server)
    kport = server.add_insecure_port("127.0.0.1:0")
    server.start()
    tmp = tmp_path_factory.mktemp("relay")
    script = tmp / "relay_worker.py"
    script.write_text(_WORKER.format(sdk=str(SDK_ROOT)), encoding="utf-8")
    aport = _free_port()
    env = {k: v for k, v in os.environ.items() if not k.startswith("CAMBRIAN_")}
    # The worker logs a traceback per failed run(); into a file, not an unread pipe,
    # whose few-KB buffer would block the worker's next write.
    log = open(tmp / "worker.log", "wb")
    proc = subprocess.Popen(
        [sys.executable, str(script), "--substrate-addr", f"127.0.0.1:{kport}", "--port", str(aport)],
        env=env, stdout=log, stderr=subprocess.STDOUT,
    )
    chan = grpc.insecure_channel(f"localhost:{aport}")
    try:
        grpc.channel_ready_future(chan).result(timeout=30)
    except grpc.FutureTimeoutError:
        proc.kill()
        proc.wait(timeout=10)
        log.close()
        server.stop(grace=None)
        pytest.fail("worker never served:\n" + (tmp / "worker.log").read_text(errors="replace"))
    try:
        yield cambrian_pb2_grpc.AgentServiceStub(chan)
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


def _call(stub, name):
    return stub.Execute(cambrian_pb2.Handoff(
        id="h-" + name, from_agent="kernel",
        payload=cambrian_pb2.Object(type="text", data=b"go"),
        metadata={"_session_token_id": "lease-" + name},
    ), timeout=30)


def test_e2e_success_still_answers(worker):
    assert _call(worker, "OK").payload.data.decode() == "fine"


@pytest.mark.parametrize("name", RELAYED)
def test_e2e_worker_status_matches_the_gateway(worker, name):
    with pytest.raises(grpc.RpcError) as ei:
        _call(worker, name)
    err = ei.value
    assert err.code() == getattr(grpc.StatusCode, name), err
    assert f"openai: provider says {name}" in err.details()
    delay = retry_delay_from_rpc(err)
    if name == "RESOURCE_EXHAUSTED":
        assert delay == pytest.approx(3.0)
    else:
        assert delay is None


@pytest.mark.parametrize("name", ["INTERNAL", "INVALID_ARGUMENT", "NOT_FOUND"])
def test_e2e_unrelayed_gateway_codes_stay_internal(worker, name):
    with pytest.raises(grpc.RpcError) as ei:
        _call(worker, name)
    assert ei.value.code() == grpc.StatusCode.INTERNAL
