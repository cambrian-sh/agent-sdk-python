"""gRPC server lifecycle for Cambrian agents.

Implements ``AgentService`` from the proto — the three RPCs every agent must
serve: ``Execute``, ``RequestProposal``, and ``VerifyOutput``.

The Orchestrator-side RPCs (``GenerateViaModelStream``, ``QueryMemory``) are
stubs invoked by the Agent to call *back* to the Substrate.
"""

from __future__ import annotations

import json
import logging
import sys

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


def agent_id_override() -> str:
    """The registry id the kernel launched this process as (``--agent-id``).

    Empty when the kernel did not pass one, which is every non-daemon launch and
    any kernel older than this flag.

    Why it exists: discovery derives an agent's REGISTRY id from its filename,
    while a signal's sender is the id the process declares for itself. Nothing
    made the two agree, and when they diverge a reply is generated and then has
    nowhere to go. It is also what lets one script run as several distinct
    principals — N ingress bots, N registrations, N surfaces, one file.
    """
    argv = sys.argv
    if "--agent-id" in argv:
        idx = argv.index("--agent-id")
        if idx + 1 < len(argv):
            return argv[idx + 1]
    return ""


def stream_id_arg() -> str:
    """``--stream-id`` from argv, "" when absent (the stream a kernel-spawned daemon or
    pool worker was launched for; carried on its heartbeats, ADR-0070 D3)."""
    argv = sys.argv
    if "--stream-id" in argv:
        idx = argv.index("--stream-id")
        if idx + 1 < len(argv):
            return argv[idx + 1]
    return ""


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
