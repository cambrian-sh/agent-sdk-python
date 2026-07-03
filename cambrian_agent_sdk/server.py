"""gRPC server lifecycle for Cambrian agents.

Implements ``AgentService`` from the proto — the three RPCs every agent must
serve: ``Execute``, ``RequestProposal``, and ``VerifyOutput``.

The Orchestrator-side RPCs (``GenerateViaModelStream``, ``QueryMemory``) are
stubs invoked by the Agent to call *back* to the Substrate.
"""

from __future__ import annotations

import json
import logging
import signal
import sys
import time

import grpc

logger = logging.getLogger("cambrian.server")


def _parse_listen_address() -> str:
    """Resolve listen address from CLI args injected by AgentManager.

    Priority: ``--socket`` (UDS path), ``--port`` (TCP port).
    Default: ``localhost:50052``.
    """
    argv = sys.argv
    if "--socket" in argv:
        idx = argv.index("--socket")
        if idx + 1 < len(argv):
            path = argv[idx + 1]
            if not path.startswith("localhost") and not path.startswith("["):
                return f"unix:{path}"
            return path
    if "--port" in argv:
        idx = argv.index("--port")
        if idx + 1 < len(argv):
            return f"localhost:{argv[idx + 1]}"
    return "localhost:50052"


def _parse_substrate_addr() -> str:
    """Extract ``--substrate-addr`` from CLI args injected by AgentManager."""
    argv = sys.argv
    if "--substrate-addr" in argv:
        idx = argv.index("--substrate-addr")
        if idx + 1 < len(argv):
            return argv[idx + 1]
    return "localhost:50051"


def is_daemon_mode() -> bool:
    """Return True if ``--daemon-mode`` was passed as a CLI arg. ADR-0033."""
    return "--daemon-mode" in sys.argv


def _parse_daemon_cli() -> tuple[str, str, dict]:
    """Parse ``--stream-id``, ``--substrate-addr``, ``--daemon-params`` from argv.

    Returns (stream_id, substrate_addr, params_dict).
    """
    argv = sys.argv
    stream_id = ""
    if "--stream-id" in argv:
        idx = argv.index("--stream-id")
        if idx + 1 < len(argv):
            stream_id = argv[idx + 1]

    substrate_addr = "localhost:50051"
    if "--substrate-addr" in argv:
        idx = argv.index("--substrate-addr")
        if idx + 1 < len(argv):
            substrate_addr = argv[idx + 1]

    params: dict = {}
    if "--daemon-params" in argv:
        idx = argv.index("--daemon-params")
        if idx + 1 < len(argv):
            try:
                params = json.loads(argv[idx + 1])
            except json.JSONDecodeError:
                logger.warning("--daemon-params: invalid JSON, using empty dict")
    return stream_id, substrate_addr, params


def start_daemon(agent: "Agent") -> None:
    """Start the agent in daemon mode — opens a persistent SignalStream instead
    of an AgentService gRPC server. ADR-0033.

    The agent emits signals by calling ``agent.send_signal(payload, raw_text)``.
    This function blocks until SIGTERM or max retries exhausted.
    """
    import time as _time

    stream_id, substrate_addr, params = _parse_daemon_cli()
    agent.stream_id = stream_id
    agent.params = params

    def _build_channel():
        return grpc.insecure_channel(
            substrate_addr,
            options=[
                ("grpc.max_send_message_length", 50 * 1024 * 1024),
                ("grpc.max_receive_message_length", 50 * 1024 * 1024),
            ],
        )

    from ._proto import cambrian_pb2, cambrian_pb2_grpc

    _shutdown = False

    def _handle_sigterm(*_):
        nonlocal _shutdown
        logger.info("SIGTERM received — stopping daemon '%s'", agent.agent_id)
        _shutdown = True

    signal.signal(signal.SIGTERM, _handle_sigterm)
    if sys.platform != "win32":
        signal.signal(signal.SIGHUP, _handle_sigterm)

    # Signal-queue for send_signal calls from the daemon loop.
    import queue
    _signal_queue: queue.Queue = queue.Queue()

    def _send_signal(payload: dict, raw_text: str = "") -> None:
        """Enqueue a signal for emission on the open SignalStream."""
        _signal_queue.put((payload, raw_text))

    agent.send_signal = _send_signal

    max_retries = 5
    retry = 0
    backoff = 1.0

    while not _shutdown and retry <= max_retries:
        try:
            channel = _build_channel()
            stub = cambrian_pb2_grpc.OrchestratorStub(channel)
            logger.info("Daemon '%s' opening SignalStream (stream_id=%s)", agent.agent_id, stream_id)

            def _request_iter():
                while not _shutdown:
                    try:
                        payload_dict, raw_text = _signal_queue.get(timeout=0.1)
                        hs = cambrian_pb2.Handoff(
                            from_agent=agent.agent_id,
                            metadata={
                                "_stream_id": stream_id,
                                "_signal_type": stream_id,
                            },
                            payload=cambrian_pb2.Object(
                                type="signal",
                                data=json.dumps(payload_dict).encode(),
                            ),
                        )
                        yield hs
                    except queue.Empty:
                        continue

            # Call daemon_loop if the agent defines it (non-blocking thread).
            if hasattr(agent, "daemon_loop"):
                import threading
                t = threading.Thread(target=agent.daemon_loop, daemon=True)
                t.start()

            for _response in stub.SignalStream(_request_iter()):
                pass  # responses from Substrate (plan telemetry) are fire-and-forget

            retry = 0
            backoff = 1.0
        except grpc.RpcError as exc:
            if _shutdown:
                break
            logger.warning("SignalStream disconnected (%s) — retry %d/%d in %.1fs",
                           exc.code(), retry, max_retries, backoff)
            _time.sleep(backoff)
            backoff = min(backoff * 2, 30.0)
            retry += 1
        finally:
            try:
                channel.close()
            except Exception:
                pass

    logger.info("Daemon '%s' stopped", agent.agent_id)


def _wire_health(server, agent_id: str) -> None:
    """Add grpc.health.v1.Health servicer, reporting SERVING for the agent service."""
    try:
        from grpc_health.v1 import health, health_pb2, health_pb2_grpc

        servicer = health.HealthServicer()
        servicer.set("", health_pb2.HealthCheckResponse.SERVING)
        servicer.set(agent_id, health_pb2.HealthCheckResponse.SERVING)
        health_pb2_grpc.add_HealthServicer_to_server(servicer, server)
    except ImportError:
        logger.warning(
            "grpcio-health-checking not installed — health check endpoint unavailable. "
            "Install with: pip install grpcio-health-checking"
        )
