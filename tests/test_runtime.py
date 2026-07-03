"""ADR-0036 issue 0036-02: single-threaded server (D2) + return coercion (D3).

The author returns plain Python from run(); the SDK coerces it into an AgentResult
at the boundary, preserving the routing-significant payload `type`. run() is never
invoked concurrently on the same instance (max_workers=1).
"""

import threading
import time
from concurrent import futures

import pytest

from cambrian_agent_sdk import AgentResult, AgentTask, CognitiveAgent
from cambrian_agent_sdk._proto import cambrian_pb2
from cambrian_agent_sdk.runtime import (
    MAX_WORKERS,
    TraitServicer,
    _coerce_agent_result,
    dispatch_run,
)


class _FakeCtx:
    """Minimal stand-in for a gRPC servicer context."""

    def __init__(self, remaining=None):
        self._remaining = remaining
        self.code = None

    def time_remaining(self):
        return self._remaining

    def set_code(self, code):
        self.code = code


def test_coerce_dict_preserves_type():
    """A dict return with type='code' must route as code (→ executor)."""
    r = _coerce_agent_result({"data": b"print(1)", "type": "code"})
    assert isinstance(r, AgentResult)
    assert r.data == b"print(1)"
    assert r.type == "code"


def test_coerce_str_defaults_to_text():
    r = _coerce_agent_result("hello")
    assert r.data == b"hello"
    assert r.type == "text"


def test_coerce_bytes_and_passthrough_and_image_type():
    assert _coerce_agent_result(b"\x89PNG").data == b"\x89PNG"
    ar = AgentResult(data=b"x", type="image/png")
    assert _coerce_agent_result(ar) is ar  # passthrough, no copy
    assert _coerce_agent_result({"data": b"x", "type": "image/png"}).type == "image/png"


def test_coerce_unsupported_raises():
    with pytest.raises(TypeError):
        _coerce_agent_result(12345)


# ── budget_signal preservation via dispatch_run ──────────────────────────────


def test_budget_refusal_becomes_budget_signal():
    from cambrian_agent_sdk import BudgetExceededError

    class Tight(CognitiveAgent):
        def run(self, task):
            raise BudgetExceededError("too costly")

    r = dispatch_run(Tight(agent_id="t"), AgentTask())
    assert r.type == "budget_signal"
    assert r.confidence == 0.0
    assert b"too costly" in r.data


# ── D3: run() receives a protocol-free AgentTask ─────────────────────────────


def test_execute_delivers_protocol_free_task_and_preserves_type():
    seen = {}

    class Capture(CognitiveAgent):
        def run(self, task):
            seen["task"] = task
            assert isinstance(task, AgentTask)  # not a Handoff/proto
            return {"data": b"print(1)", "type": "code"}

    servicer = TraitServicer(Capture(agent_id="cap"))
    req = cambrian_pb2.Handoff(
        id="h1",
        from_agent="caller",
        payload=cambrian_pb2.Object(type="text", data=b"hello"),
        metadata={"_step_index": "2", "_session_token_id": "tok"},
    )
    resp = servicer.Execute(req, _FakeCtx(remaining=3.0))

    task = seen["task"]
    assert task.text == "hello"
    assert task.step_index == 2
    assert task.session_token_id == "tok"
    assert task.deadline_remaining_ms == 3000
    # the routing-significant type survives the round-trip to the proto response
    assert resp.payload.type == "code"
    assert resp.payload.data == b"print(1)"


def test_execute_author_exception_becomes_internal_not_crash():
    import grpc

    class Boom(CognitiveAgent):
        def run(self, task):
            raise RuntimeError("author bug")

    ctx = _FakeCtx()
    resp = TraitServicer(Boom(agent_id="b")).Execute(cambrian_pb2.Handoff(id="h"), ctx)
    assert ctx.code == grpc.StatusCode.INTERNAL


# ── D2: single-threaded contract ─────────────────────────────────────────────


def test_max_workers_is_one():
    """The keystone — one request at a time per process."""
    assert MAX_WORKERS == 1


def test_overlapping_calls_serialize_under_single_worker():
    """Two submissions through the single-worker executor must NOT interleave."""
    events = []
    lock = threading.Lock()

    class Slow(CognitiveAgent):
        def run(self, task):
            with lock:
                events.append("start")
            time.sleep(0.05)
            with lock:
                events.append("end")
            return "ok"

    agent = Slow(agent_id="slow")
    task = AgentTask()
    with futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        f1 = ex.submit(dispatch_run, agent, task)
        f2 = ex.submit(dispatch_run, agent, task)
        f1.result()
        f2.result()
    # serialized: start,end,start,end — never start,start,...
    assert events == ["start", "end", "start", "end"]


def test_per_turn_self_state_is_safe_across_sequential_turns():
    """Authors may keep per-turn state on self; sequential turns stay consistent."""

    class Counter(CognitiveAgent):
        def __init__(self, **kw):
            super().__init__(**kw)
            self.count = 0

        def run(self, task):
            self.count += 1
            return str(self.count)

    agent = Counter(agent_id="c")
    results = [dispatch_run(agent, AgentTask()).text for _ in range(5)]
    assert results == ["1", "2", "3", "4", "5"]
    assert agent.count == 5
