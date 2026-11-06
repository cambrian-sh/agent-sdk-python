"""Agent-pushed live step output (ADR-0139 amendment 2026-11-06): the SDK helpers
around the client-streaming ``EmitStepOutput`` RPC."""
import threading

import pytest

from cambrian_agent_sdk import SubstrateClient
from cambrian_agent_sdk._proto import cambrian_pb2 as pb


class _Ack:
    def __init__(self, accepted, last_seq):
        self.accepted, self.duplicates, self.over_cap, self.after_final = accepted, 0, 0, 0
        self.last_seq = last_seq


class _Stub:
    """Consumes the request iterator like grpc does, on the calling thread."""

    def __init__(self, error=None):
        self.calls = []
        self.error = error
        self.lock = threading.Lock()

    def EmitStepOutput(self, it, timeout=None, metadata=None):
        chunks = list(it)
        with self.lock:
            self.calls.append((chunks, metadata))
        if self.error is not None:
            raise self.error
        return _Ack(len(chunks), max((c.seq for c in chunks), default=0))


class _RpcError(Exception):
    def __init__(self, name):
        self._name = name

    def code(self):
        return type("C", (), {"name": self._name})()

    def details(self):
        return f"{self._name} detail"


def _client(stub):
    return SubstrateClient(stub=stub, agent_id="ext-agent")


def test_writer_streams_chunks_with_increasing_seq_and_a_final_marker():
    stub = _Stub()
    c = _client(stub)
    with c.step_output("L1") as out:
        out.progress("calling the model")
        out.text("Hel")
        out.text("")  # empty text is not sent
        out.text("lo")
    chunks, md = stub.calls[0]
    assert [(ch.seq, ch.data, ch.final) for ch in chunks] == [
        (1, "calling the model", False), (2, "Hel", False), (3, "lo", False), (4, "", True)]
    assert chunks[0].kind == pb.STEP_OUTPUT_KIND_PROGRESS and chunks[1].kind == pb.STEP_OUTPUT_KIND_TEXT
    assert all(ch.lease_id == "L1" for ch in chunks)
    assert ("x-agent-id", "ext-agent") in md and ("x-lease-id", "L1") in md
    assert out.ack["accepted"] == 4 and out.error is None


def test_seq_continues_across_pushes_for_one_lease_and_restarts_per_lease():
    stub = _Stub()
    c = _client(stub)
    c.emit_step_output("L1", ["a", "b"])
    c.emit_step_output("L1", [{"data": "half", "kind": "progress"}], final=True)
    c.emit_step_output("L2", ["x"])
    seqs = [[ch.seq for ch in chunks] for chunks, _ in stub.calls]
    assert seqs == [[1, 2], [3, 4], [1]]
    assert stub.calls[1][0][-1].final is True


def test_explicit_seq_is_kept_for_a_safe_resend():
    stub = _Stub()
    c = _client(stub)
    c.emit_step_output("L1", [{"data": "a", "seq": 7}])
    c.emit_step_output("L1", ["b"])
    assert [ch.seq for ch in stub.calls[0][0]] == [7]
    assert [ch.seq for ch in stub.calls[1][0]] == [8]


def test_failures_are_recorded_not_raised():
    c = _client(_Stub(error=_RpcError("PERMISSION_DENIED")))
    with c.step_output("dead-lease") as out:
        out.text("x")
    assert out.ack is None and out.error["code"] == "PERMISSION_DENIED"
    assert c.emit_step_output("dead-lease", ["x"])["code"] == "PERMISSION_DENIED"


def test_unimplemented_kernel_reads_as_none():
    c = _client(_Stub(error=_RpcError("UNIMPLEMENTED")))
    assert c.emit_step_output("L1", ["x"]) is None
    with c.step_output("L1") as out:
        out.text("x")
    assert out.ack is None and out.error is None


def test_bad_kind_and_missing_lease_are_caller_errors():
    c = _client(_Stub())
    with pytest.raises(ValueError):
        c.emit_step_output("L1", [{"data": "x", "kind": "token"}])
    with pytest.raises(ValueError):
        c.step_output("")
    with pytest.raises(ValueError):
        c.emit_step_output("", ["x"])


def test_writer_refuses_writes_after_close():
    c = _client(_Stub())
    out = c.step_output("L1").open()
    out.close()
    with pytest.raises(RuntimeError):
        out.text("late")
