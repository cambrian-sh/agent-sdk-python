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

    from .channel import build_channel
    from .server import _parse_daemon_cli, agent_id_override
    from ._proto import cambrian_pb2, cambrian_pb2_grpc

    stream_id, parsed_addr, params = _parse_daemon_cli()
    agent.stream_id = stream_id
    agent.params = params

    # Adopt the identity the kernel launched us as.
    #
    # Discovery derives an agent's REGISTRY id from its filename, while a signal's
    # sender is the id the process declared for itself. Nothing made the two
    # agree, and when they diverge a reply is generated and then has nowhere to
    # go — the divergence is silent on the way in and only shows up as a missing
    # answer. Adopting the kernel's id removes it by construction.
    #
    # It is also what lets ONE script run as several distinct principals: N
    # ingress bots, N registrations, N surfaces, one file. Absent on an older
    # kernel, in which case the agent keeps the id it chose.
    launched_as = agent_id_override()
    if launched_as and launched_as != agent.agent_id:
        logger.info(
            "daemon '%s' adopting the registry id it was launched as: %s",
            agent.agent_id, launched_as,
        )
        agent.agent_id = launched_as
    addr = substrate_addr or parsed_addr

    shutdown = {"flag": False}

    def _handle_sigterm(*_):
        logger.info("SIGTERM received — stopping daemon '%s'", agent.agent_id)
        shutdown["flag"] = True

    # Signal handlers may only be installed from the main thread, and an ingress
    # runs this loop on a background one (IngressAgent.serve keeps the main thread
    # for the delivery server). Installing unconditionally raised ValueError there
    # and killed the inbound half on startup while the outbound half kept serving —
    # a half-dead ingress that accepts deliveries and never sends anything inbound.
    #
    # Off the main thread the process owns shutdown anyway: the server that holds
    # the main thread stops, and this loop's own `_should_continue` falls away with
    # the process. So skipping registration here loses nothing.
    if threading.current_thread() is threading.main_thread():
        signal.signal(signal.SIGTERM, _handle_sigterm)
        if sys.platform != "win32":
            signal.signal(signal.SIGHUP, _handle_sigterm)
    else:
        logger.debug(
            "daemon '%s': not the main thread, skipping signal handlers", agent.agent_id
        )

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

    # Supervise the author's loop in a background thread so the stream drains
    # concurrently; a crash restarts with backoff (capped 30s), never silent.
    #
    # Started ONCE, outside the reconnect loop, and this placement is load-bearing.
    # It used to be started inside the loop, so every SignalStream reconnect spawned
    # ANOTHER supervised daemon_loop while the previous thread kept running —
    # `_should_continue` only flips at shutdown, so nothing ever stopped the old one.
    # A flapping stream therefore left N copies of the author's loop running forever.
    #
    # How it presented: the telegram ingress reconnected 6 times and then logged 133
    # `409 conflict — another getUpdates poller is running on this token`. The "other
    # poller" was itself, five times over. The 409 made it visible because Telegram
    # refuses concurrent getUpdates on one token; a daemon with no such server-side
    # interlock would instead have silently processed every signal N times, which is
    # the worse version of this bug and the reason to fix it here rather than in the
    # telegram agent.
    #
    # Safe to hoist: daemon_loop does not touch the channel. It hands work to
    # `agent._signal_queue`, which `_request_iter` drains on whichever stream is
    # currently open — so signals queue across a disconnect and flush on reconnect.
    loop_thread = threading.Thread(
        target=run_supervised,
        args=(agent.daemon_loop,),
        kwargs={"should_continue": _should_continue},
        daemon=True,
    )
    loop_thread.start()

    backoff = 1.0
    while _should_continue():
        try:
            # ADR-0130 D4: credential + transport from the environment (channel.py).
            channel = build_channel(
                addr,
                options=[
                    ("grpc.max_send_message_length", 50 * 1024 * 1024),
                    ("grpc.max_receive_message_length", 50 * 1024 * 1024),
                ],
                agent_id=agent.agent_id,
            )
            stub = cambrian_pb2_grpc.OrchestratorStub(channel)
            logger.info("Daemon '%s' opening SignalStream (stream_id=%s)", agent.agent_id, stream_id)

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
