"""ADR-0036 issue 0036-06: the DaemonAgent runtime.

A persistent background signal producer on the SignalStream contract (not
AgentService). A crashing daemon_loop() restarts with exponential backoff capped at
30s — never silent death. One process per stream_id (ADR-0033) ⇒ no shared-self bleed.
"""

import pytest

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


def test_start_daemon_off_main_thread_does_not_raise_on_signals():
    """Regression: an ingress runs the daemon loop on a background thread.

    `signal.signal` only works on the main thread, so installing handlers
    unconditionally raised ValueError there and killed the inbound half on
    startup while the outbound half kept serving — a half-dead ingress that
    accepts deliveries and never sends anything inbound. That is why the SDK
    ingresses were "built but not measured".
    """
    import threading

    from cambrian_agent_sdk import daemon as daemon_mod

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

    assert "err" not in outcome, outcome["err"]


class _StubAgent:
    agent_id = "stub_ingress"

    def __init__(self):
        import queue

        self._signal_queue = queue.Queue()
        self.stream_id = ""
        self.params = {}

    def daemon_loop(self):
        raise RuntimeError("stop")
