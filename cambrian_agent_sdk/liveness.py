"""Heartbeat liveness for kernel-spawned long-lived agents (ADR-0070 D3).

A daemon or pool worker that stays alive but stops responding used to be reported
``running`` forever: only a process exit reached the kernel's restart policy. The kernel
now passes ``--heartbeat-interval-ms N`` when it spawns a long-lived process, and this
module emits a heartbeat signal every interval on the process's SignalStream. The kernel
tracks the last beat per instance and, after N missed beats, kills the process; the
existing restart policy brings up a fresh one. No author code is involved.

**What a beat certifies.** Not merely "the process exists" — a background thread would
keep beating through any hang that releases the GIL (``time.sleep``, a blocked socket
read), and that is exactly the case that took the chat lane down: a pool worker stuck in
``Execute`` while the kernel had long given up on the turn. The server is single-threaded
(ADR-0036 D2), so a worker whose one handler is still running a call its caller has
abandoned can serve nobody. A beat is therefore WITHHELD while an in-flight call's caller
is gone (its gRPC context is no longer active: cancelled or past its deadline). If the
handler returns before the kernel's patience runs out, beats resume and nothing is
killed. A wedged interpreter (a C extension holding the GIL, a deadlock) stops the beats
by itself.

A long call whose caller is still waiting keeps beating (``busy=true``): slowness is the
caller's deadline to judge, not liveness.

**Old SDKs.** A process on an SDK without this module sends no beats; the kernel arms
tracking for an instance only on its first beat, so it is never killed for not sending
what it cannot send. A kernel that does not pass the flag gets no beats from this SDK.
"""

from __future__ import annotations

import json
import logging
import sys
import threading
import time
from typing import Any, Callable, Optional

logger = logging.getLogger("cambrian.liveness")

#: The ``_signal_type`` metadata value of a heartbeat. The kernel consumes these on the
#: SignalStream before any signal routing: a heartbeat never reaches a pipeline, the
#: planner or an ingress.
HEARTBEAT_SIGNAL_TYPE = "_heartbeat"

#: argv flag the kernel passes (milliseconds; absent or 0 = heartbeats off).
HEARTBEAT_FLAG = "--heartbeat-interval-ms"


def parse_heartbeat_interval(argv: Optional[list] = None) -> float:
    """Return the heartbeat interval in seconds from argv, 0.0 when off or malformed."""
    argv = sys.argv if argv is None else argv
    if HEARTBEAT_FLAG not in argv:
        return 0.0
    idx = argv.index(HEARTBEAT_FLAG)
    if idx + 1 >= len(argv):
        return 0.0
    try:
        ms = int(argv[idx + 1])
    except ValueError:
        logger.warning("%s: %r is not an integer; heartbeats off", HEARTBEAT_FLAG, argv[idx + 1])
        return 0.0
    return ms / 1000.0 if ms > 0 else 0.0


def _context_active(context: Any) -> bool:
    """Whether the caller of an in-flight call is still waiting for it.

    A context without ``is_active`` (a test double, an in-process call) counts as
    active: withholding a beat kills the process, so only a positive "the caller is
    gone" may do it.
    """
    probe = getattr(context, "is_active", None)
    if probe is None:
        return True
    try:
        return bool(probe())
    except Exception:  # noqa: BLE001 — a broken probe must not kill the worker
        return True


class Liveness:
    """Process-wide in-flight tracker and heartbeat clock.

    The runtime registers every inbound ``Execute`` with :meth:`begin`/:meth:`end`;
    request iterators on a SignalStream call :meth:`next_beat` on every poll. The beat
    clock is shared, so a process with two streams (an ingress: delivery server plus
    inbound stream) still emits one beat per interval.
    """

    def __init__(self, interval: float = 0.0, clock: Callable[[], float] = time.monotonic) -> None:
        self.interval = interval
        self._clock = clock
        self._lock = threading.Lock()
        self._inflight: dict = {}
        self._next_id = 0
        self._last_beat: Optional[float] = None
        self._withheld_logged = False

    @property
    def enabled(self) -> bool:
        return self.interval > 0

    def begin(self, context: Any) -> int:
        """Register an in-flight call; returns the token :meth:`end` takes."""
        with self._lock:
            self._next_id += 1
            token = self._next_id
            self._inflight[token] = context
            return token

    def end(self, token: int) -> None:
        with self._lock:
            self._inflight.pop(token, None)

    def busy(self) -> bool:
        with self._lock:
            return bool(self._inflight)

    def responsive(self) -> bool:
        """False while any in-flight call's caller has gone away (see module docstring)."""
        with self._lock:
            contexts = list(self._inflight.values())
        return all(_context_active(c) for c in contexts)

    def next_beat(self) -> Optional[dict]:
        """Return a heartbeat payload when one is due and may be sent, else None.

        Due: no beat yet, or ``interval`` elapsed since the last one. A beat that is due
        but withheld (an abandoned call is still running) does not advance the clock, so
        beats resume on the first poll after the handler returns.
        """
        if not self.enabled:
            return None
        now = self._clock()
        # One lock for check-and-advance, so two streams polling at once emit one beat.
        with self._lock:
            if self._last_beat is not None and now - self._last_beat < self.interval:
                return None
            if not all(_context_active(c) for c in self._inflight.values()):
                if not self._withheld_logged:
                    logger.warning(
                        "heartbeat withheld: a call its caller abandoned is still running "
                        "(this single-threaded worker can serve nobody until it returns)")
                    self._withheld_logged = True
                return None
            self._last_beat = now
            self._withheld_logged = False
            busy = bool(self._inflight)
        return {"busy": busy}


#: The process's tracker. Configured from argv once, on first use by a server or stream.
LIVENESS = Liveness()
_configured = False
_configure_lock = threading.Lock()


def configure_from_argv(argv: Optional[list] = None) -> Liveness:
    """Set :data:`LIVENESS`'s interval from argv (idempotent) and return it."""
    global _configured
    with _configure_lock:
        if not _configured:
            LIVENESS.interval = parse_heartbeat_interval(argv)
            _configured = True
    return LIVENESS


def heartbeat_handoff(agent_id: str, stream_id: str, beat: dict):
    """The SignalStream message carrying one beat."""
    from ._proto import cambrian_pb2

    return cambrian_pb2.Handoff(
        from_agent=agent_id,
        metadata={"_stream_id": stream_id, "_signal_type": HEARTBEAT_SIGNAL_TYPE,
                  "_heartbeat_busy": "true" if beat.get("busy") else "false"},
        payload=cambrian_pb2.Object(type="heartbeat", data=json.dumps(beat).encode()),
    )


def heartbeat_requests(agent_id: str, stream_id: str, should_continue: Callable[[], bool],
                       liveness: Optional[Liveness] = None, poll: float = 0.1):
    """Request iterator for a heartbeat-only SignalStream (a worker with no other stream)."""
    lv = liveness or LIVENESS
    while should_continue():
        beat = lv.next_beat()
        if beat is not None:
            yield heartbeat_handoff(agent_id, stream_id, beat)
        time.sleep(poll)


def start_heartbeat_stream(agent_id: str, substrate_addr: str, stream_id: str = "") -> Optional[threading.Thread]:
    """Open a heartbeat-only SignalStream on a daemon thread, reconnecting with backoff.

    For a kernel-spawned AgentService worker (a chat pool worker) that has no signal
    stream of its own. Returns None when the kernel did not ask for heartbeats.
    """
    lv = configure_from_argv()
    if not lv.enabled:
        return None
    # The identity the kernel launched this process as (CAMBRIAN_AGENT_ID, beside the
    # per-instance token) — not the id the class declares. An explicit agent_id would
    # override the env in build_channel, and a declared id that differs from the
    # registry id is refused UNAUTHENTICATED: the token names the registry id.
    import os

    from .channel import AGENT_ID_ENV

    agent_id = (os.environ.get(AGENT_ID_ENV) or "").strip() or agent_id

    def _run() -> None:
        import grpc

        from ._proto import cambrian_pb2_grpc
        from .channel import build_channel

        backoff = 1.0
        while True:
            channel = None
            try:
                channel = build_channel(substrate_addr, agent_id=agent_id)
                stub = cambrian_pb2_grpc.OrchestratorStub(channel)
                logger.info("agent '%s' opening heartbeat stream (every %.1fs)", agent_id, lv.interval)
                for _ in stub.SignalStream(heartbeat_requests(agent_id, stream_id, lambda: True, lv)):
                    pass
                backoff = 1.0
            except grpc.RpcError as exc:
                logger.warning("heartbeat stream disconnected (%s) — retry in %.1fs", exc.code(), backoff)
            except Exception as exc:  # noqa: BLE001 — the beat thread must not die silently
                logger.warning("heartbeat stream failed (%s) — retry in %.1fs", exc, backoff)
            finally:
                if channel is not None:
                    try:
                        channel.close()
                    except Exception:  # noqa: BLE001
                        pass
            time.sleep(backoff)
            backoff = min(backoff * 2, 30.0)

    t = threading.Thread(target=_run, name=f"{agent_id}-heartbeat", daemon=True)
    t.start()
    return t
