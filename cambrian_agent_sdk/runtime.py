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
from concurrent import futures
from typing import Any, Protocol, Union

from .errors import BudgetExceededError
from .types import AgentResult, AgentTask, ContextRef

logger = logging.getLogger("cambrian.runtime")


class ServableAgent(Protocol):
    """What the AgentService boundary needs from an agent: an id and ``run``.

    ``base.Agent`` itself declares no ``run`` (a DaemonAgent has none; it is served
    over SignalStream, never here), so the AgentService path is typed by this
    protocol: CognitiveAgent and IngressAgent satisfy it.
    """

    agent_id: str

    def run(self, task: AgentTask) -> Any: ...

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


def dispatch_run(agent: ServableAgent, task: AgentTask) -> AgentResult:
    """Call the agent's run() and coerce, mapping a principled budget refusal to a
    ``budget_signal`` result (the DAGExecutor distinguishes it from incapability)."""
    from .clients import task_lease

    try:
        # ENG-518: memory reads made inside run() present this task's lease by default.
        with task_lease(getattr(task, "session_token_id", "") or ""):
            return _coerce_agent_result(agent.run(task))
    except BudgetExceededError as e:
        return AgentResult(data=f"BUDGET_EXCEEDED:{e.reason}".encode(), type="budget_signal", confidence=0.0)


#: The answer to the deprecated RequestProposal RPC (agent stack v2). The kernel's
#: capability-typed dispatch (ADR-0100) solicits no bids, and the LLM interview that
#: was the RPC's last caller is gone; the RPC stays in the proto (deprecated, never
#: removed in v0.x). A neutral, well-formed proposal rather than UNIMPLEMENTED, so a
#: stale caller gets an answer that attracts nothing instead of an error status, and
#: the servicer never runs author code (``propose()``) that could raise.
DEPRECATED_PROPOSAL_RATIONALE = "deprecated: capability-typed dispatch solicits no bids (ADR-0100)"


def _set_retry_info(context, code, details: str, retry_delay: float) -> None:
    """Attach ``retry_delay`` to the Execute status as a google.rpc.RetryInfo detail, the
    way the kernel gateway sent it (ENG-2125). Advice only: a failure here leaves the
    code and details as set."""
    from ._rpc_status import STATUS_DETAILS_KEY, status_details_bin

    try:
        raw = status_details_bin(code.value[0], details, retry_delay)
        context.set_trailing_metadata(((STATUS_DETAILS_KEY, raw),))
    except Exception as e:  # noqa: BLE001
        logger.debug("could not attach RetryInfo to the Execute status: %s", e)


class TraitServicer:
    """Routes inbound AgentService RPCs to a trait agent's run() over the D3 boundary."""

    def __init__(self, agent: ServableAgent) -> None:
        self._agent = agent

    def RequestProposal(self, request, context):
        """Deprecated RPC: always a neutral proposal (confidence 0), never the agent's
        ``propose()``. See :data:`DEPRECATED_PROPOSAL_RATIONALE`."""
        from ._proto import cambrian_pb2

        return cambrian_pb2.ProposalResponse(
            confidence=0.0,
            rationale=DEPRECATED_PROPOSAL_RATIONALE,
            estimated_latency_ms=0,
        )

    def Execute(self, request, context):
        # ADR-0070 D3: the heartbeat is withheld while this call runs on after its caller
        # has gone (liveness.py) — registered for the whole handler, released on any exit.
        from .liveness import LIVENESS

        token = LIVENESS.begin(context)
        try:
            return self._execute(request, context)
        finally:
            LIVENESS.end(token)

    def _execute(self, request, context):
        from ._proto import cambrian_pb2

        task = _task_from_handoff(request, context)
        try:
            result = dispatch_run(self._agent, task)
        except Exception as exc:  # never crash the server: answer with a status
            import grpc

            from .errors import is_context_overflow, relayed_gateway_error

            code, retry_delay = grpc.StatusCode.INTERNAL, None
            relayed = relayed_gateway_error(exc)
            # A prompt over the model's context window is deterministic for this
            # step: OUT_OF_RANGE, which the kernel's self-healer never retries.
            # Reported as INTERNAL it was retried as "unclassified" and the whole
            # 12-round step re-ran into the same overflow (owner decision A4,
            # T-20261112-worker-resolve-cid-inline-has-no-size-cap-a).
            if is_context_overflow(exc):
                logger.error("run() ended on a context overflow: %s", exc)
                code = grpc.StatusCode.OUT_OF_RANGE
            elif relayed is not None:
                # A gateway verdict (refused credential, rate limit, provider down) is
                # passed through as the kernel's own code, with its RetryInfo, so the
                # self-healer and the chat lane read the same thing they would from the
                # gateway itself. As INTERNAL the person saw "StatusCode.INTERNAL:
                # UNAUTHENTICATED: ..." (ENG-2125, owner decision 2026-11-18).
                logger.error("run() ended on a gateway %s: %s", relayed.status, exc)
                code, retry_delay = getattr(grpc.StatusCode, relayed.status), relayed.retry_delay
            else:
                logger.error("run() raised: %s", exc, exc_info=True)
            context.set_code(code)
            details = str(exc)[:2000]
            try:
                context.set_details(details)
            except Exception:  # noqa: BLE001 — a context double without set_details
                pass
            if retry_delay:
                _set_retry_info(context, code, details, retry_delay)
            return cambrian_pb2.Handoff()
        return _handoff_from_result(result, request, self._agent.agent_id)


def start_agent_server(agent: ServableAgent, address: str, *, heartbeat_stream: bool = True) -> None:
    """Boot a single-threaded AgentService server for a trait agent and block (D2).

    When the kernel spawned this process as a long-lived worker and asked for heartbeats
    (``--heartbeat-interval-ms``, ADR-0070 D3), a heartbeat-only SignalStream is opened
    beside the server. ``heartbeat_stream=False`` is for a process whose own signal stream
    already carries the beats (an ingress: ``daemon.start_daemon`` emits them).
    """
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

    if heartbeat_stream:
        from .liveness import start_heartbeat_stream
        from .server import _parse_substrate_addr, stream_id_arg

        start_heartbeat_stream(agent.agent_id, _parse_substrate_addr(), stream_id_arg())

    def _handle_sigterm(*_):
        server.stop(grace=5)

    signal.signal(signal.SIGTERM, _handle_sigterm)
    if sys.platform != "win32":
        signal.signal(signal.SIGHUP, _handle_sigterm)
    server.wait_for_termination()
