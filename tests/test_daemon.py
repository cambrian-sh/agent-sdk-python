"""ADR-0036 issue 0036-06: the DaemonAgent runtime.

A persistent background signal producer on the SignalStream contract (not
AgentService). A crashing daemon_loop() restarts with exponential backoff capped at
30s — never silent death. One process per stream_id (ADR-0033) ⇒ no shared-self bleed.
"""

import contextlib
import time

from cambrian_agent_sdk import DaemonAgent
from cambrian_agent_sdk.daemon import run_supervised


class Beat(DaemonAgent):
    def daemon_loop(self):
        self.send_signal({"tick": 1}, "heartbeat")


def test_send_signal_enqueues_onto_the_instance_queue():
    d = Beat(agent_id="beat")
    d.send_signal({"x": 1}, "hello")
    payload, raw = d._signal_queue.get_nowait()
    assert payload == {"x": 1}
    assert raw == "hello"


def test_crashing_loop_restarts_with_exponential_backoff_capped_at_30():
    sleeps = []
    state = {"n": 0}

    def always_crash():
        raise RuntimeError("boom")

    def should_continue():
        return state["n"] < 6  # allow 6 restarts then stop

    def fake_sleep(secs):
        sleeps.append(secs)
        state["n"] += 1

    restarts = run_supervised(always_crash, should_continue=should_continue, sleep_fn=fake_sleep)
    # 1 → 2 → 4 → 8 → 16 → 30 (32 capped to 30) — never silent death
    assert sleeps == [1, 2, 4, 8, 16, 30]
    assert restarts == 6


def test_crash_is_surfaced_not_swallowed_silently():
    crashes = []

    def crash_once():
        crashes.append("crashed")
        raise ValueError("oops")

    # stop after the first crash is observed
    run_supervised(
        crash_once,
        should_continue=lambda: len(crashes) < 1,
        sleep_fn=lambda s: None,
        on_crash=lambda exc: crashes.append(("seen", str(exc))),
    )
    assert ("seen", "oops") in crashes


def test_clean_loop_return_ends_supervision_without_restart():
    calls = {"n": 0}

    def finite_loop():
        calls["n"] += 1  # returns cleanly

    restarts = run_supervised(finite_loop, should_continue=lambda: True, sleep_fn=lambda s: None)
    assert restarts == 0
    assert calls["n"] == 1  # ran once, not restarted


def test_daemon_cli_parses_stream_id_and_mode(monkeypatch):
    import sys

    from cambrian_agent_sdk.server import _parse_daemon_cli, is_daemon_mode

    monkeypatch.setattr(
        sys, "argv",
        ["prog", "--daemon-mode", "--stream-id", "conv:acme", "--substrate-socket", "/tmp/s.sock"],
    )
    assert is_daemon_mode()
    stream_id, addr, params = _parse_daemon_cli()
    assert stream_id == "conv:acme"


def test_two_daemons_have_independent_state_no_bleed():
    """ADR-0033: one process per stream_id — instances never share self/history."""
    acme = Beat(agent_id="conv:acme")
    globex = Beat(agent_id="conv:globex")
    acme.send_signal({"x": 1})

    assert acme._signal_queue is not globex._signal_queue
    assert acme._signal_queue.qsize() == 1
    assert globex._signal_queue.qsize() == 0  # no shared queue → no bleed


def test_daemon_uses_signalstream_not_agentservice():
    """Structural: the daemon runtime serves SignalStream, not AgentService."""
    import inspect

    import cambrian_agent_sdk.daemon as daemon_mod

    src = inspect.getsource(daemon_mod)
    assert "SignalStream" in src
    assert "add_AgentServiceServicer_to_server" not in src
    # and the trait class has no task-responder surface
    assert not hasattr(DaemonAgent, "run")
    assert DaemonAgent.trait == "daemon"


def test_start_daemon_off_main_thread_does_not_raise_on_signals(monkeypatch):
    """Regression: an ingress runs the daemon loop on a background thread.

    `signal.signal` only works on the main thread, so installing handlers
    unconditionally raised ValueError there and killed the inbound half on
    startup while the outbound half kept serving — a half-dead ingress that
    accepts deliveries and never sends anything inbound. That is why the SDK
    ingresses were "built but not measured".
    """
    import threading

    from cambrian_agent_sdk import daemon as daemon_mod

    # ENG-555: end the daemon instead of leaving it behind. The dial to
    # 127.0.0.1:1 fails, and the reconnect loop's backoff sleep then raises, so
    # start_daemon returns and this thread ends inside the test; the supervised
    # author loop is not under test and is not started. Before, both threads
    # outlived the test (reconnecting forever), and one of them later raised into
    # a different test as PytestUnhandledThreadExceptionWarning.
    class _StopDaemon(Exception):
        pass

    def _stop_on_backoff(_seconds):
        raise _StopDaemon()

    monkeypatch.setattr(daemon_mod.time, "sleep", _stop_on_backoff)
    monkeypatch.setattr(daemon_mod, "run_supervised", lambda *a, **k: None)

    outcome = {}

    def run():
        try:
            # Fails later on the gRPC dial, which is fine — we only care that it
            # got PAST signal registration without raising ValueError.
            daemon_mod.start_daemon(_StubAgent(), substrate_addr="127.0.0.1:1")
        except ValueError as exc:  # the bug
            outcome["err"] = f"ValueError: {exc}"
        except Exception:  # noqa: BLE001 — anything else means signals were fine
            outcome["ok"] = True
        else:
            outcome["ok"] = True

    t = threading.Thread(target=run, daemon=True)
    t.start()
    t.join(timeout=15)

    assert not t.is_alive(), "the daemon thread outlived the test"
    assert "err" not in outcome, outcome["err"]


def test_reconnecting_the_stream_does_not_start_a_second_daemon_loop(monkeypatch):
    """Regression: N stream reconnects must not leave N copies of daemon_loop running.

    `start_daemon` started the supervised loop INSIDE its reconnect loop, so every
    SignalStream disconnect spawned another one while the previous thread kept going
    (`should_continue` only flips at shutdown). A flapping stream left N concurrent
    copies of the author's loop.

    Observed live: the telegram ingress reconnected 6 times and then logged 133
    `409 conflict — another getUpdates poller is running on this token`. The "other
    poller" was itself. The 409 only made it VISIBLE — Telegram refuses concurrent
    getUpdates on one token. A daemon without that server-side interlock would have
    silently done every unit of work N times instead, which is why this is asserted
    at the SDK and not in the telegram agent.
    """
    import threading

    from cambrian_agent_sdk import daemon as daemon_mod

    starts = []
    shutdown_after = 3  # let the reconnect loop go round three times

    class _FlappingAgent:
        agent_id = "flapping"

        def __init__(self):
            import queue

            self._signal_queue = queue.Queue()
            self.stream_id = ""
            self.params = {}

        def daemon_loop(self):
            starts.append(threading.current_thread().name)
            # Block, the way a real long-poll does. A loop that returned here would
            # end supervision and mask a second start.
            while True:
                time.sleep(0.05)

    reconnects = {"n": 0}

    class _Err(Exception):
        pass

    def _fake_channel(_addr, options=None):  # noqa: ARG001
        class _Ch:
            def close(self):
                pass

        return _Ch()

    class _FakeStub:
        def __init__(self, _ch):
            pass

        def SignalStream(self, _it):  # noqa: N802 — mirrors the generated stub
            reconnects["n"] += 1
            if reconnects["n"] >= shutdown_after:
                # Stop the outer while-loop the same way SIGTERM would.
                raise KeyboardInterrupt
            raise _rpc_error()

    def _rpc_error():
        import grpc

        class _E(grpc.RpcError):
            def code(self):
                return "UNAVAILABLE"

        return _E()

    monkeypatch.setattr(daemon_mod.time, "sleep", lambda _s: None)
    monkeypatch.setattr(
        daemon_mod, "_parse_daemon_cli", lambda: ("s1", "127.0.0.1:1", {}), raising=False
    )

    import grpc as _grpc

    monkeypatch.setattr(_grpc, "insecure_channel", _fake_channel)
    from cambrian_agent_sdk._proto import cambrian_pb2_grpc

    monkeypatch.setattr(cambrian_pb2_grpc, "OrchestratorStub", _FakeStub)

    with contextlib.suppress(KeyboardInterrupt):
        daemon_mod.start_daemon(_FlappingAgent(), substrate_addr="127.0.0.1:1")

    assert reconnects["n"] >= 2, "the test did not actually exercise a reconnect"
    assert len(starts) == 1, (
        f"daemon_loop was started {len(starts)} times across "
        f"{reconnects['n']} stream connections; it must be started exactly once"
    )


class _StubAgent:
    agent_id = "stub_ingress"

    def __init__(self):
        import queue

        self._signal_queue = queue.Queue()
        self.stream_id = ""
        self.params = {}

    def daemon_loop(self):
        raise RuntimeError("stop")
