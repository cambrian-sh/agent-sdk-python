"""ADR-0070 D3: heartbeat liveness (cambrian_agent_sdk/liveness.py).

A kernel-spawned worker beats every interval while it can serve; it withholds the beat
while a call its caller abandoned is still running, so a single-threaded worker stuck on
an orphaned call is reclaimed by the kernel. Beats resume once the handler returns.
"""

import json

from cambrian_agent_sdk.liveness import (
    HEARTBEAT_SIGNAL_TYPE,
    Liveness,
    heartbeat_handoff,
    heartbeat_requests,
    parse_heartbeat_interval,
)
from cambrian_agent_sdk.runtime import TraitServicer


class _Clock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


class _Ctx:
    """A gRPC ServicerContext double: is_active() flips when the caller goes away."""

    def __init__(self, active=True):
        self.active = active

    def is_active(self):
        return self.active

    def time_remaining(self):
        return None

    def invocation_metadata(self):
        return ()


def test_interval_flag_parsing():
    assert parse_heartbeat_interval(["agent.py"]) == 0.0
    assert parse_heartbeat_interval(["a", "--heartbeat-interval-ms", "2500"]) == 2.5
    assert parse_heartbeat_interval(["a", "--heartbeat-interval-ms", "0"]) == 0.0
    assert parse_heartbeat_interval(["a", "--heartbeat-interval-ms", "x"]) == 0.0
    assert parse_heartbeat_interval(["a", "--heartbeat-interval-ms"]) == 0.0


def test_disabled_never_beats():
    lv = Liveness(0.0)
    assert lv.next_beat() is None


def test_beats_once_per_interval():
    clock = _Clock()
    lv = Liveness(1.0, clock=clock)
    assert lv.next_beat() == {"busy": False}  # first beat at once: arms the kernel
    assert lv.next_beat() is None
    clock.t += 0.5
    assert lv.next_beat() is None
    clock.t += 0.5
    assert lv.next_beat() == {"busy": False}


def test_busy_call_with_a_waiting_caller_keeps_beating():
    clock = _Clock()
    lv = Liveness(1.0, clock=clock)
    tok = lv.begin(_Ctx(active=True))
    assert lv.next_beat() == {"busy": True}
    clock.t += 1
    assert lv.next_beat() == {"busy": True}
    lv.end(tok)
    clock.t += 1
    assert lv.next_beat() == {"busy": False}


def test_abandoned_call_withholds_beats_until_it_returns():
    clock = _Clock()
    lv = Liveness(1.0, clock=clock)
    assert lv.next_beat() is not None
    ctx = _Ctx(active=True)
    tok = lv.begin(ctx)
    ctx.active = False  # the kernel gave up on the turn; the handler runs on
    for _ in range(5):
        clock.t += 1
        assert lv.next_beat() is None, "a worker stuck on an orphaned call must go silent"
    lv.end(tok)
    # Withheld beats did not advance the clock: the first poll after return beats.
    assert lv.next_beat() == {"busy": False}


def test_context_without_is_active_counts_as_active():
    lv = Liveness(1.0, clock=_Clock())
    lv.begin(object())
    assert lv.next_beat() == {"busy": True}


def test_heartbeat_handoff_shape():
    h = heartbeat_handoff("chat_agent", "chat-chat_agent-0", {"busy": True})
    assert h.from_agent == "chat_agent"
    assert h.metadata["_signal_type"] == HEARTBEAT_SIGNAL_TYPE
    assert h.metadata["_stream_id"] == "chat-chat_agent-0"
    assert h.metadata["_heartbeat_busy"] == "true"
    assert json.loads(h.payload.data) == {"busy": True}


def test_heartbeat_requests_yields_due_beats_only():
    clock = _Clock()
    lv = Liveness(1.0, clock=clock)
    polls = {"n": 0}

    def should_continue():
        polls["n"] += 1
        clock.t += 0.5  # two polls per interval
        return polls["n"] <= 6

    beats = list(heartbeat_requests("a", "s", should_continue, lv, poll=0))
    assert 2 <= len(beats) <= 4
    assert all(b.metadata["_signal_type"] == HEARTBEAT_SIGNAL_TYPE for b in beats)


class _Agent:
    agent_id = "probe"

    def __init__(self, lv, ctx):
        self.lv, self.ctx = lv, ctx
        self.seen_busy = None

    def run(self, task):
        # While run() executes, the process-wide tracker holds this call.
        from cambrian_agent_sdk.liveness import LIVENESS

        self.seen_busy = LIVENESS.busy()
        return "ok"


def test_execute_registers_the_call_for_its_whole_duration():
    from cambrian_agent_sdk._proto import cambrian_pb2
    from cambrian_agent_sdk.liveness import LIVENESS

    agent = _Agent(None, None)
    TraitServicer(agent).Execute(
        cambrian_pb2.Handoff(payload=cambrian_pb2.Object(type="text", data=b"hi")), _Ctx())
    assert agent.seen_busy is True
    assert LIVENESS.busy() is False  # released on return


def test_execute_releases_the_call_when_run_raises():
    from cambrian_agent_sdk._proto import cambrian_pb2
    from cambrian_agent_sdk.liveness import LIVENESS

    class Boom(_Agent):
        def run(self, task):
            raise RuntimeError("boom")

    class Ctx(_Ctx):
        def set_code(self, c):
            pass

        def set_details(self, d):
            pass

    TraitServicer(Boom(None, None)).Execute(
        cambrian_pb2.Handoff(payload=cambrian_pb2.Object(type="text", data=b"hi")), Ctx())
    assert LIVENESS.busy() is False
