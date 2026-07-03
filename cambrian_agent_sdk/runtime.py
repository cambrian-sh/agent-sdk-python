"""Single-threaded gRPC runtime for trait agents (ADR-0036 D2 + D3).

Two load-bearing decisions live here:

- **D2 (single-threaded):** the server runs ``ThreadPoolExecutor(max_workers=1)`` so
  ``run()`` is *never invoked concurrently on the same instance*. Authors may keep
  per-turn state on ``self`` without locks. Throughput is recovered at the **process**
  level (JIT/pool, and ADR-0033 daemon-per-``stream_id``), not by intra-process threads.
- **D3 (protocol invisibility):** the inbound ``Execute(Handoff)`` is translated into a
  protocol-free :class:`AgentTask`; the author's plain return is coerced into an
  :class:`AgentResult` (preserving the routing-significant payload ``type``) and wrapped
  back into a ``Handoff``. The author never imports ``Handoff`` / ``Payload``.
"""

from __future__ import annotations

import logging
import signal
import sys
import time
from concurrent import futures
from typing import TYPE_CHECKING, Union

from .types import AgentResult, AgentTask, ContextRef
from .errors import BudgetExceededError

if TYPE_CHECKING:
    from .base import Agent

logger = logging.getLogger("cambrian.runtime")

#: D2 keystone — one request at a time per process. Do not raise this; throughput is
#: a process-count axis, not a thread-count one. The invariant survives a future async
#: model (one event loop per process, one ``await run()`` at a time).
MAX_WORKERS = 1


def _coerce_agent_result(raw: Union[AgentResult, dict, str, bytes, None]) -> AgentResult:
    """Coerce an author's plain return into an :class:`AgentResult` (D3).

    Preserves the routing-significant ``type`` (``code`` → executor,
    ``budget_signal`` → circuit breaker, ``image/png`` → binary artifact). A ``dict``
    return may carry ``data`` / ``type`` / ``confidence`` / ``metadata`` / ``context`` /
    ``uncertainties``; a ``str`` / ``bytes`` becomes a ``text`` result.
    """
    if isinstance(raw, AgentResult):
        return raw
    if raw is None:
        return AgentResult()
    if isinstance(raw, dict):
        data = raw.get("data", b"")
        if isinstance(data, str):
            data = data.encode("utf-8")
        return AgentResult(
            data=data,
            type=raw.get("type", "text"),
            confidence=raw.get("confidence", 1.0),
            uncertainties=raw.get("uncertainties", []),
            context=raw.get("context", {}),
            metadata=raw.get("metadata", {}),
        )
    if isinstance(raw, bytes):
        return AgentResult(data=raw)
    if isinstance(raw, str):
        return AgentResult(data=raw.encode("utf-8"))
    raise TypeError(f"run() returned unsupported type {type(raw)}")


def _task_from_handoff(request, context) -> AgentTask:
    """Translate an inbound proto ``Handoff`` into a protocol-free :class:`AgentTask`."""
    payload = request.payload
    data = bytes(payload.data) if payload else b""
    ptype = payload.type if payload else "text"
    meta = dict(payload.metadata) if payload else {}

    ctx = dict(request.metadata)
    step_index = int(ctx.get("_step_index", "0"))
    session_token_id = ctx.get("_session_token_id", "")

    working_mem = [
        ContextRef(
            cid=ref.cid,
            type=ref.type,
            labels=list(ref.labels),
            activation=ref.activation,
            precision=ref.precision,
            snippet=ref.snippet,
        )
        for ref in request.working_memory
    ]

    remaining = context.time_remaining() if context is not None else None
    deadline_remaining_ms = int(remaining * 1000) if remaining else 0

    return AgentTask(
        text=data.decode("utf-8", errors="replace"),
        type=ptype,
        data=data,
        metadata=meta,
        context=ctx,
        working_memory=working_mem,
        step_index=step_index,
        plan_id=request.id,
        task_id=ctx.get("_task_id", ""),  # ADR-0049 D3: per-step correlation key
        session_token_id=session_token_id,
        deadline_remaining_ms=deadline_remaining_ms,
    )


def _handoff_from_result(result: AgentResult, request, agent_id: str):
    """Wrap an :class:`AgentResult` back into a proto ``Handoff`` response."""
    from ._proto import cambrian_pb2

    return cambrian_pb2.Handoff(
        id=request.id,
        from_agent=agent_id,
        to_agent=request.from_agent,
        payload=cambrian_pb2.Object(
            type=result.type,
            data=result.data,
            metadata=result.metadata,
        ),
        confidence=result.confidence,
        uncertainties=result.uncertainties,
        metadata=result.context,
    )


def dispatch_run(agent: "Agent", task: AgentTask) -> AgentResult:
    """Call the agent's run() and coerce, mapping a principled budget refusal to a
    ``budget_signal`` result (the DAGExecutor distinguishes it from incapability)."""
    try:
        return _coerce_agent_result(agent.run(task))
    except BudgetExceededError as e:
        return AgentResult(data=f"BUDGET_EXCEEDED:{e.reason}".encode(), type="budget_signal", confidence=0.0)


def proposal_bid(agent, request):
    """Resolve an agent's auction bid. Uses ``agent.propose()`` when present (the
    DeterministicAgent static bid); otherwise a generic best-effort bid."""
    from .types import ProposalResponse

    propose = getattr(agent, "propose", None)
    if callable(propose):
        return propose(request)
    return ProposalResponse(confidence=0.5, rationale="general-purpose agent", estimated_latency_ms=1000)


class TraitServicer:
    """Routes inbound AgentService RPCs to a trait agent's run() over the D3 boundary."""

    def __init__(self, agent: "Agent") -> None:
        self._agent = agent

    def RequestProposal(self, request, context):
        """Serve the agent's bid. A DeterministicAgent returns its static 1.0/5ms bid."""
        from ._proto import cambrian_pb2

        bid = proposal_bid(self._agent, request)
        return cambrian_pb2.ProposalResponse(
            confidence=bid.confidence,
            rationale=bid.rationale,
            estimated_latency_ms=bid.estimated_latency_ms,
            requirements=bid.requirements,
            metadata=bid.metadata,
        )

    def Execute(self, request, context):
        from ._proto import cambrian_pb2

        task = _task_from_handoff(request, context)
        try:
            result = dispatch_run(self._agent, task)
        except Exception as exc:  # author bug — surface as INTERNAL, never crash the server
            logger.error("run() raised: %s", exc, exc_info=True)
            import grpc

            context.set_code(grpc.StatusCode.INTERNAL)
            return cambrian_pb2.Handoff()
        return _handoff_from_result(result, request, self._agent.agent_id)


def start_agent_server(agent: "Agent", address: str) -> None:
    """Boot a single-threaded AgentService server for a trait agent and block (D2)."""
    import grpc

    from ._proto import cambrian_pb2_grpc

    servicer = TraitServicer(agent)
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=MAX_WORKERS),
        options=[
            ("grpc.max_send_message_length", 50 * 1024 * 1024),
            ("grpc.max_receive_message_length", 50 * 1024 * 1024),
        ],
    )

    class _Wired(cambrian_pb2_grpc.AgentServiceServicer):
        def Execute(self, request, context):
            return servicer.Execute(request, context)

        def RequestProposal(self, request, context):
            return servicer.RequestProposal(request, context)

    cambrian_pb2_grpc.add_AgentServiceServicer_to_server(_Wired(), server)

    # Wire grpc.health.v1 so the Go InstanceManager can distinguish
    # "socket inode exists" from "gRPC stack is SERVING". Without this,
    # waitHealthy() times out and the agent is treated as unhealthy.
    from .server import _wire_health
    _wire_health(server, agent.agent_id)

    server.add_insecure_port(address)
    server.start()
    logger.info("Agent '%s' serving on %s (max_workers=%d)", agent.agent_id, address, MAX_WORKERS)

    def _handle_sigterm(*_):
        server.stop(grace=5)

    signal.signal(signal.SIGTERM, _handle_sigterm)
    if sys.platform != "win32":
        signal.signal(signal.SIGHUP, _handle_sigterm)
    server.wait_for_termination()
