"""DaemonAgent runtime (ADR-0036 issue 0036-06).

A daemon serves a *different* gRPC contract than task-responders: it opens a
persistent ``SignalStream`` and produces signals, rather than answering ``Execute``
calls on ``AgentService``. The author overrides ``daemon_loop()`` and calls
``send_signal()``; this runtime supervises the loop, restarting a crash with
**exponential backoff capped at 30s** (an unhandled exception must not silently kill
the stream). One process per ``stream_id`` (ADR-0033) gives chatbots cross-conversation
OS-level isolation for free.
"""

from __future__ import annotations

import json
import logging
import signal
import sys
import time
from typing import Callable, Optional

logger = logging.getLogger("cambrian.daemon")

_BACKOFF_CAP_SECONDS = 30.0


def run_supervised(
    loop_fn: Callable[[], None],
    *,
    should_continue: Callable[[], bool],
    sleep_fn: Callable[[float], None] = time.sleep,
    backoff_cap: float = _BACKOFF_CAP_SECONDS,
    on_crash: Optional[Callable[[BaseException], None]] = None,
) -> int:
    """Run ``loop_fn`` under crash supervision; return the number of restarts.

    A clean return from ``loop_fn`` ends supervision (the daemon is done). An
    exception triggers a restart after a backoff that doubles 1→2→4→…, capped at
    ``backoff_cap`` (30s). ``should_continue`` is re-checked before every (re)start
    and before sleeping, so a shutdown signal stops the loop promptly without delay.
    """
    backoff = 1.0
    restarts = 0
    while should_continue():
        try:
            loop_fn()
            return restarts  # clean completion — nothing to restart
        except BaseException as exc:  # noqa: BLE001 — supervise *any* crash, never die silently
            if on_crash is not None:
                on_crash(exc)
            logger.warning("daemon_loop crashed (%s) — restart in %.1fs", exc, backoff)
            if not should_continue():
                return restarts
            sleep_fn(backoff)
            backoff = min(backoff * 2, backoff_cap)
            restarts += 1
    return restarts


def start_daemon(agent, substrate_addr: Optional[str] = None) -> None:
    """Open the SignalStream and run the supervised daemon loop until SIGTERM.

    Parses ``--stream-id`` / ``--substrate-socket`` / ``--daemon-params`` from argv,
    binds the agent's queue-backed ``send_signal`` onto the open stream, and supervises
    ``daemon_loop``.
    """
    import threading

    import grpc

    from .server import _parse_daemon_cli
    from ._proto import cambrian_pb2, cambrian_pb2_grpc

    stream_id, parsed_addr, params = _parse_daemon_cli()
    agent.stream_id = stream_id
    agent.params = params
    addr = substrate_addr or parsed_addr

    shutdown = {"flag": False}

    def _handle_sigterm(*_):
        logger.info("SIGTERM received — stopping daemon '%s'", agent.agent_id)
        shutdown["flag"] = True

    signal.signal(signal.SIGTERM, _handle_sigterm)
    if sys.platform != "win32":
        signal.signal(signal.SIGHUP, _handle_sigterm)

    def _should_continue():
        return not shutdown["flag"]

    def _request_iter():
        import queue as _queue

        while not shutdown["flag"]:
            try:
                payload_dict, raw_text = agent._signal_queue.get(timeout=0.1)
            except _queue.Empty:
                continue
            yield cambrian_pb2.Handoff(
                from_agent=agent.agent_id,
                metadata={"_stream_id": stream_id, "_signal_type": stream_id},
                payload=cambrian_pb2.Object(type="signal", data=json.dumps(payload_dict).encode()),
            )

    backoff = 1.0
    while _should_continue():
        try:
            channel = grpc.insecure_channel(
                addr,
                options=[
                    ("grpc.max_send_message_length", 50 * 1024 * 1024),
                    ("grpc.max_receive_message_length", 50 * 1024 * 1024),
                ],
            )
            stub = cambrian_pb2_grpc.OrchestratorStub(channel)
            logger.info("Daemon '%s' opening SignalStream (stream_id=%s)", agent.agent_id, stream_id)

            # Supervise the author's loop in a background thread so the stream drains
            # concurrently; a crash restarts with backoff (capped 30s), never silent.
            t = threading.Thread(
                target=run_supervised,
                args=(agent.daemon_loop,),
                kwargs={"should_continue": _should_continue},
                daemon=True,
            )
            t.start()

            for _response in stub.SignalStream(_request_iter()):
                pass  # plan telemetry from the Substrate is fire-and-forget
            backoff = 1.0
        except grpc.RpcError as exc:
            if shutdown["flag"]:
                break
            logger.warning("SignalStream disconnected (%s) — retry in %.1fs", exc.code(), backoff)
            time.sleep(backoff)
            backoff = min(backoff * 2, _BACKOFF_CAP_SECONDS)
        finally:
            try:
                channel.close()
            except Exception:
                pass

    logger.info("Daemon '%s' stopped", agent.agent_id)
