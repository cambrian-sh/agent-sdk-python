"""Owner decision A4 (T-20261112-worker-resolve-cid-inline-has-no-size-cap-a): a prompt
over the model's context window is a deterministic, typed failure.

The kernel gateway answers it with OUT_OF_RANGE; the SDK maps that to
ContextOverflow, and an agent whose run() ends on one reports OUT_OF_RANGE (not
INTERNAL) so the kernel's self-healer never re-runs the step into the same overflow.
"""

import grpc

from cambrian_agent_sdk import CognitiveAgent, ContextOverflow, GenerateError, _map_generate_error
from cambrian_agent_sdk._proto import cambrian_pb2
from cambrian_agent_sdk.errors import is_context_overflow
from cambrian_agent_sdk.runtime import TraitServicer


class _Rpc(grpc.RpcError):
    def __init__(self, code, details):
        self._c, self._d = code, details

    def code(self):
        return self._c

    def details(self):
        return self._d


class _Ctx:
    def __init__(self):
        self.code, self.details = None, ""

    def time_remaining(self):
        return None

    def set_code(self, code):
        self.code = code

    def set_details(self, details):
        self.details = details


def test_out_of_range_maps_to_context_overflow():
    exc = _map_generate_error(_Rpc(grpc.StatusCode.OUT_OF_RANGE,
                                   "prompt is too long: ~205074 tokens (limit 200000)"),
                              tools=True)
    assert isinstance(exc, ContextOverflow) and isinstance(exc, GenerateError)
    assert exc.status == "OUT_OF_RANGE"
    other = _map_generate_error(_Rpc(grpc.StatusCode.INTERNAL, "boom"), tools=True)
    assert not isinstance(other, ContextOverflow)


def test_is_context_overflow_follows_the_cause_chain():
    try:
        try:
            raise ContextOverflow("too long", code="OUT_OF_RANGE")
        except ContextOverflow as inner:
            raise RuntimeError("step failed") from inner
    except RuntimeError as outer:
        assert is_context_overflow(outer)
    assert not is_context_overflow(RuntimeError("x"))


def test_execute_context_overflow_is_out_of_range_with_details():
    class Over(CognitiveAgent):
        def run(self, task):
            raise ContextOverflow("Prompt is too long", code="OUT_OF_RANGE")

    ctx = _Ctx()
    TraitServicer(Over(agent_id="o")).Execute(cambrian_pb2.Handoff(id="h"), ctx)
    assert ctx.code == grpc.StatusCode.OUT_OF_RANGE
    assert "Prompt is too long" in ctx.details


def test_execute_other_failure_stays_internal_and_carries_details():
    class Boom(CognitiveAgent):
        def run(self, task):
            raise RuntimeError("author bug")

    ctx = _Ctx()
    TraitServicer(Boom(agent_id="b")).Execute(cambrian_pb2.Handoff(id="h"), ctx)
    assert ctx.code == grpc.StatusCode.INTERNAL
    assert ctx.details == "author bug"
