"""Cambrian Agent SDK — build Cambrian agents in Python with zero boilerplate.

Minimal usage — subclass a trait class and override ``run(task)``::

    from cambrian_agent_sdk import AgentTask, CognitiveAgent

    AGENT_DESCRIPTION = "Echoes the task text back."
    AGENT_MANIFEST = '''
    {"version": "1.0.0", "trait": "cognitive", "capabilities": ["echo"]}
    '''

    class EchoAgent(CognitiveAgent):
        def run(self, task: AgentTask) -> str:
            return f"Echo: {task.text}"

    agent = EchoAgent(agent_id="echo_agent", version="1.0.0", description=AGENT_DESCRIPTION)

    if __name__ == "__main__":
        agent.serve()

``CognitiveAgent``'s default ``run()`` drives the ``think()`` ReAct loop over the
agent's ``@tool`` methods, so an LLM agent usually overrides nothing but ``role``
(see ``examples/tool_tutorial_agent.py``). ``run()`` may return an
:class:`AgentResult`, a ``dict``, ``str`` or ``bytes``.

The agent file must be named ``*agent.py`` and live in the agents directory the
kernel is configured with; the kernel discovers it on startup.

AGENT_DESCRIPTION and AGENT_MANIFEST must be module-level literals — the kernel
parses them with a regex before the Python process boots, so write them by hand
(there is no generator command).
"""

from __future__ import annotations

import json
import logging
import threading
from typing import Any, Callable, Dict, List, Optional, Tuple

from .base import Agent, CognitiveAgent, DaemonAgent
from .clients import (
    TIME_PRECISIONS,
    TIME_SOURCES,
    ArtifactManager,
    INGEST_STATES,
    ArtifactNotFound,
    IngestFailedError,
    IngestStatus,
    IngestStatusNotFound,
    InvalidTagError,
    MemoryClient,
    RememberResult,
    SelfDelegationError,
    WorkingMemory,
    _lease_metadata,
    _lease_or_current,
)
from .context_engine import ContextConfig, ContextEngine, DefaultContextEngine
from .errors import (
    BudgetExceededError,
    ContextOverflow,
    GenerateError,
    GenerationResourceExhausted,
    InvalidRequest,
    ModelUnavailable,
    SchemaValidationFailed,
    ToolCallingUnsupported,
)
from .execute_result import ExecuteResult
from .helpers import PromptParts, build_prompt, build_prompt_parts, extract_code_block, find_step_ref
from .ingress import IngressAgent, PermanentDeliveryError, RateLimitedError
from .llm import (
    CACHE_TTL_LONG,
    CACHE_TTL_SHORT,
    GenerateResult,
    GenerateUsage,
    ResponseSchema,
    SystemBlock,
)
from .react import ReActLoopError
from .tools import ToolRegistry, ToolSpec, capability, tool
from .types import (
    CAPABILITY_MISSING_RESULT_TYPE,
    AgentResult,
    AgentTask,
    Capability,
    ContextNode,
    ContextRef,
    ExecuteRequest,
    ExecuteResponse,
    Payload,
    ProposalRequest,
    ProposalResponse,
    ScopeConfig,
    SubGoal,
    VerifyRequest,
    VerifyResponse,
    capability_missing,
    yield_subgoal,
)

__all__ = [
    # ADR-0036 v2 trait-aligned surface
    "Agent",  # abstract base
    "CognitiveAgent",
    "DaemonAgent",
    "IngressAgent",
    "PermanentDeliveryError",
    "RateLimitedError",
    "AgentTask",
    "AgentResult",
    "SubGoal",
    "yield_subgoal",
    "capability_missing",
    "CAPABILITY_MISSING_RESULT_TYPE",
    "tool",
    "capability",
    "ToolRegistry",
    "ToolSpec",
    "ReActLoopError",
    "ContextConfig",
    "ContextEngine",
    "DefaultContextEngine",
    "MemoryClient",
    "ArtifactManager",
    "WorkingMemory",
    "InvalidTagError",
    "ArtifactNotFound",
    "IngestStatus",
    "IngestFailedError",
    "IngestStatusNotFound",
    "RememberResult",
    "INGEST_STATES",
    "SelfDelegationError",
    "TIME_PRECISIONS",
    "TIME_SOURCES",
    # shared vocabulary + helpers
    "assemble_context",
    "build_prompt",
    "BudgetExceededError",
    "ToolCallingUnsupported",
    # ADR-0139 C1 managed-generation request/result and typed errors
    "GenerateError",
    "InvalidRequest",
    "SchemaValidationFailed",
    "GenerationResourceExhausted",
    "ModelUnavailable",
    "ContextOverflow",
    "GenerateResult",
    "GenerateUsage",
    "ResponseSchema",
    "SystemBlock",
    "CACHE_TTL_SHORT",
    "CACHE_TTL_LONG",
    "build_prompt_parts",
    "PromptParts",
    "Capability",
    "ContextNode",
    "ContextRef",
    "ExecuteRequest",
    "ExecuteResponse",
    "ExecuteResult",
    "extract_code_block",
    "find_step_ref",
    "Payload",
    "ProposalRequest",
    "ProposalResponse",
    "ScopeConfig",
    "SubstrateClient",
    "VerifyRequest",
    "VerifyResponse",
    "configure_logging",
]

logger = logging.getLogger("cambrian.agent")

# gRPC ASCII metadata values must be a single line of printable ASCII (0x20-0x7E).
# A raw query used as a header — e.g. a multi-line user message, or one with non-ASCII
# (Turkish, accents) — makes the whole call fail with "Illegal header value", which
# surfaced as the chat agent silently returning an empty reply. Collapse whitespace
# (incl. newlines/tabs) to single spaces, drop anything outside printable ASCII, and cap
# the length so the retrieval/skill hint stays a valid header on any input.
_MAX_HEADER_QUERY = 512

def _header_safe(value: str) -> str:
    """Make an arbitrary string safe to send as a gRPC ASCII metadata value."""
    if not value:
        return ""
    cleaned = "".join(c if 0x20 <= ord(c) <= 0x7E else " " for c in value)
    cleaned = " ".join(cleaned.split())  # collapse runs of whitespace, strip ends
    return cleaned[:_MAX_HEADER_QUERY]

def configure_logging(level: int = logging.INFO) -> None:
    """Install SlogHandler for Substrate-compatible structured JSON logs."""
    from ._logging import configure_logging as _cfg
    _cfg(level=level)


# ── C1 request/response conversion (ADR-0139, wave 1d) ──────────────────────
# Plain-data <-> vendored protobuf. Kept as functions, not methods, so tests can
# exercise the mapping without a channel.

def _generate_options(pb, max_tokens, temperature, stop, strict):
    """GenerateOptions with field presence: ``None`` leaves max_tokens/temperature
    unset (the model's default); an explicit ``0``/``0.0`` is sent and honoured."""
    opts = pb.GenerateOptions()
    if max_tokens is not None:
        opts.max_tokens = int(max_tokens)
    if temperature is not None:
        opts.temperature = float(temperature)
    if stop:
        opts.stop_sequences.extend([stop] if isinstance(stop, str) else list(stop))
    if strict:
        opts.strict = True
    return opts


def _pb_system(pb, blocks):
    return [pb.SystemBlock(text=b.text, cache=b.cache) for b in blocks]


def _pb_messages(pb, messages):
    out = []
    for m in messages:
        calls = [
            pb.ModelToolCallProto(
                id=c.get("id", ""),
                name=c.get("name", ""),
                # The provider's own argument string, forwarded unchanged: it
                # matches this against its record of the call it made.
                arguments_json=c.get("arguments", "") or "",
            )
            for c in (m.get("tool_calls") or [])
        ]
        out.append(pb.ModelMessageProto(
            role=m.get("role", "user") or "user",
            content=m.get("content", "") or "",
            tool_calls=calls,
            tool_call_id=m.get("tool_call_id", "") or "",
        ))
    return out


def _pb_schema(pb, schema):
    return pb.ResponseSchema(name=schema.name, schema_json=schema.schema_json, strict=schema.strict)


def _pb_cache_ttl(pb, ttl: str):
    from .llm import CACHE_TTL_LONG, CACHE_TTL_SHORT
    if ttl == CACHE_TTL_SHORT:
        return pb.CACHE_TTL_SHORT
    if ttl == CACHE_TTL_LONG:
        return pb.CACHE_TTL_LONG
    return pb.CACHE_TTL_UNSPECIFIED


_legacy_kernel_logged = False


def _fill_result_from(out, msg, *, structured: bool, tools: bool) -> None:
    """Copy usage/ignored/stop_reason/model from a final GenerateChunk or a
    GenerateWithToolsResponse into ``out`` (a GenerateResult).

    A C1 kernel always sets ``usage`` on a normally completed call. Its absence
    after a structured request means the kernel predates C1: on the stream the
    flattened prompt was read instead (nothing lost); on GenerateWithTools the
    system blocks were dropped, which is reported as ``ignored: ["system"]`` so a
    caller can re-send the system text inside the conversation."""
    from .llm import GenerateUsage
    global _legacy_kernel_logged
    out.usage_tokens = int(getattr(msg, "usage_tokens", 0) or 0)
    if msg.HasField("usage"):
        u = msg.usage
        out.usage = GenerateUsage(
            input_tokens=u.input_tokens, output_tokens=u.output_tokens,
            cache_read_tokens=u.cache_read_tokens, cache_write_tokens=u.cache_write_tokens,
            cache_write_known=u.cache_write_known, estimated=u.estimated)
    out.ignored = list(msg.ignored)
    if msg.stop_reason:
        out.stop_reason = msg.stop_reason
    out.model = msg.model
    if structured and not msg.HasField("usage"):
        out.legacy_kernel = True
        if tools and "system" not in out.ignored:
            out.ignored.append("system")
        if not _legacy_kernel_logged:
            _legacy_kernel_logged = True
            logger.warning("llm_legacy_kernel", extra={
                "rpc": "GenerateWithTools" if tools else "GenerateViaModelStream",
                "effect": ("system blocks dropped; reported as ignored" if tools
                           else "served from the flattened prompt; no cache breakpoint")})
    if out.ignored:
        logger.info("llm_fields_ignored", extra={"ignored": list(out.ignored), "model": out.model})


def _map_generate_error(exc, *, tools: bool):
    """A gRPC error from a managed generation call -> the typed SDK exception.

    FAILED_PRECONDITION (and UNIMPLEMENTED, a kernel predating the RPC) on
    GenerateWithTools is ``ToolCallingUnsupported``: the documented signal to fall
    back to the prompt-encoded action protocol, deliberately distinct from every
    real failure so an outage never silently downgrades a run."""
    import grpc

    from .errors import (
        ContextOverflow,
        GenerateError,
        GenerationResourceExhausted,
        InvalidRequest,
        ModelUnavailable,
        SchemaValidationFailed,
    )
    try:
        code = exc.code()
        details = exc.details() or ""
    except Exception:  # noqa: BLE001 — a bare RpcError without status
        code, details = grpc.StatusCode.UNKNOWN, str(exc)
    if tools and code in (grpc.StatusCode.UNIMPLEMENTED, grpc.StatusCode.FAILED_PRECONDITION):
        return ToolCallingUnsupported(str(details or code))
    name = getattr(code, "name", str(code))
    cls = {
        grpc.StatusCode.INVALID_ARGUMENT: InvalidRequest,
        grpc.StatusCode.ABORTED: SchemaValidationFailed,
        grpc.StatusCode.RESOURCE_EXHAUSTED: GenerationResourceExhausted,
        grpc.StatusCode.UNAVAILABLE: ModelUnavailable,
        # The prompt is over the model's context window (owner decision A4).
        grpc.StatusCode.OUT_OF_RANGE: ContextOverflow,
    }.get(code, GenerateError)
    # The provider's Retry-After, when the gateway attached it as RetryInfo (ENG-2125).
    from ._rpc_status import retry_delay_from_rpc
    return cls(details, code=name, retry_delay=retry_delay_from_rpc(exc))


def _log_request(rpc: str, blocks, messages, schema, ttl: str) -> None:
    """Debug-level request metadata: block count, cache marks, the sha256 of the
    system text (byte-identical across turns of a run when the caller follows the
    stable-tier rule), message count, schema name and cache lifetime."""
    if not logger.isEnabledFor(logging.DEBUG):
        return
    import hashlib
    sys_text = "\n\n".join(b.text for b in blocks)
    logger.debug("llm_request", extra={
        "rpc": rpc,
        "system_blocks": len(blocks),
        "system_cache_marks": sum(1 for b in blocks if b.cache),
        "system_sha256": hashlib.sha256(sys_text.encode("utf-8")).hexdigest() if blocks else "",
        "messages": len(messages),
        "response_schema": schema.name if schema is not None else "",
        "cache_ttl": ttl,
    })


class SubstrateClient:
    """Backchannel to the Substrate for delegation calls and LTM queries.

    Injected as ``self.substrate`` on every ExecuteRequest so agents can
    sub-delegate tasks and query shared memory without gRPC boilerplate.
    """

    def __init__(self, substrate_addr: str = "localhost:50051", stub=None, agent_id: str = "") -> None:
        self._addr = substrate_addr
        self._channel: Optional[Any] = None  # grpc.Channel once dialled
        self._stub = stub  # injectable for tests; otherwise lazily dialed
        self._agent_id = agent_id
        self._lock = threading.Lock()

    def _ensure(self) -> None:
        # Double-checked locking: avoids the lock cost on the hot path after
        # first connection while still being safe for parallel DAG step dispatch.
        if self._stub is not None:
            return
        with self._lock:
            if self._stub is not None:
                return
            from ._proto import cambrian_pb2_grpc
            from .channel import build_channel

            # ADR-0130 D4: credential + transport from the environment (channel.py).
            self._channel = build_channel(self._addr, agent_id=self._agent_id)
            self._stub = cambrian_pb2_grpc.OrchestratorStub(self._channel)

    # ── Managed generation (ADR-0018 proxy; ADR-0139 C1 request shape) ──────────
    #
    # Four entry points, one wire path per RPC:
    #   generate(...)            -> str              (legacy signature, kept)
    #   generate_stream(...)     -> Iterator[str]    (legacy signature, kept)
    #   generate_request(...)    -> GenerateResult   (text + usage, ignored, model)
    #   generate_with_tools(...) -> GenerateResult   (unpacks as the old 3-tuple)
    # All accept the C1 fields as keywords: system, messages, response_schema,
    # cache_ttl, stop, strict. See cambrian_agent_sdk/llm.py for the shapes.

    def generate(
        self,
        session_token_id: str,
        prompt: str = "",
        max_tokens: Optional[int] = 1024,
        temperature: Optional[float] = 0.7,
        timeout_ms: int = 0,
        *,
        system=None,
        messages=None,
        response_schema=None,
        cache_ttl=None,
        stop=None,
        strict: bool = False,
    ) -> str:
        """Call the Substrate's managed LLM proxy and return the full text.

        Pass ``timeout_ms=request.deadline_remaining_ms`` to propagate the Go
        step deadline into the gRPC call. Without this, a slow LLM call will
        block the thread after the Go side has already moved on with
        DEADLINE_EXCEEDED, leaking a ThreadPoolExecutor slot per hung call.

        ``prompt`` alone is the original call shape and is sent as one user turn.
        The C1 keywords (``system``, ``messages``, ...) are described on
        :meth:`generate_request`, which also returns usage and ``ignored``.
        """
        return self.generate_request(
            session_token_id, prompt=prompt, system=system, messages=messages,
            response_schema=response_schema, cache_ttl=cache_ttl,
            max_tokens=max_tokens, temperature=temperature, stop=stop, strict=strict,
            timeout_ms=timeout_ms,
        ).text

    def generate_stream(
        self,
        session_token_id: str,
        prompt: str = "",
        max_tokens: Optional[int] = 1024,
        temperature: Optional[float] = 0.7,
        timeout_ms: int = 0,
        *,
        system=None,
        messages=None,
        response_schema=None,
        cache_ttl=None,
        stop=None,
        strict: bool = False,
        result=None,
    ):
        """Stream text chunks from the Substrate's managed LLM proxy.

        Yields each text fragment as it arrives from ``GenerateViaModelStream``.
        Use this for large outputs to avoid buffering the entire response before
        returning — prevents ThreadPoolExecutor saturation on 8k-token replies.

        Pass ``result=GenerateResult()`` to receive the final chunk's usage,
        ``ignored``, ``stop_reason`` and ``model`` once the stream is exhausted.

        Example::

            for chunk in agent.substrate.generate_stream(token, prompt, timeout_ms=5000):
                partial_output += chunk
        """
        self._ensure()
        import grpc

        from ._proto import cambrian_pb2
        from .llm import GenerateResult as _Result

        req, structured = self._build_stream_request(
            cambrian_pb2, session_token_id, prompt=prompt, system=system, messages=messages,
            response_schema=response_schema, cache_ttl=cache_ttl, max_tokens=max_tokens,
            temperature=temperature, stop=stop, strict=strict)
        out = result if result is not None else _Result()
        timeout_secs = (timeout_ms / 1000.0) if timeout_ms > 0 else None
        md = [("x-agent-id", self._agent_id)] if self._agent_id else []
        texts: List[str] = []
        try:
            for chunk in self._stub.GenerateViaModelStream(req, timeout=timeout_secs, metadata=md):
                if chunk.text:
                    texts.append(chunk.text)
                    yield chunk.text
                if chunk.is_final:
                    _fill_result_from(out, chunk, structured=structured, tools=False)
        except grpc.RpcError as exc:
            raise _map_generate_error(exc, tools=False) from exc
        out.text = "".join(texts)

    def generate_request(
        self,
        session_token_id: str,
        *,
        system=None,
        messages=None,
        prompt: str = "",
        response_schema=None,
        cache_ttl=None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stop=None,
        strict: bool = False,
        timeout_ms: int = 0,
    ):
        """One managed generation in the ADR-0139 C1 shape; returns a ``GenerateResult``.

        - ``system``: ``[{"text": ..., "cache": True}, ...]`` (or ``SystemBlock``s, or a
          plain string). Stable text first; mark the last stable block ``cache=True``.
          Volatile text (budget notices, step state) belongs in ``messages``.
        - ``messages``: ``[{"role": "user"|"assistant"|"tool", "content": ...}]``. A
          ``"system"`` role raises ``InvalidRequest``. ``prompt`` alone is sent as one
          user turn.
        - ``response_schema``: ``{"name", "schema": dict|str, "strict": bool}``.
        - ``cache_ttl``: ``"short"`` (5 min, agent runs) | ``"long"`` (1 h) | ``None``.
        - ``max_tokens`` / ``temperature``: ``None`` = the model's default; ``0.0`` is
          sent as a real temperature (field presence).
        - ``stop``: stop sequences. ``strict=True`` turns an option the model cannot
          honour into ``InvalidRequest`` instead of an entry in ``result.ignored``.

        Raises ``InvalidRequest`` (INVALID_ARGUMENT), ``SchemaValidationFailed``
        (ABORTED), ``GenerationResourceExhausted`` (RESOURCE_EXHAUSTED),
        ``ModelUnavailable`` (UNAVAILABLE), or ``GenerateError`` for anything else.
        """
        from .llm import GenerateResult as _Result
        out = _Result()
        out.text = "".join(self.generate_stream(
            session_token_id, prompt=prompt, max_tokens=max_tokens,
            temperature=temperature, timeout_ms=timeout_ms, system=system,
            messages=messages, response_schema=response_schema, cache_ttl=cache_ttl,
            stop=stop, strict=strict, result=out))
        return out

    def _build_stream_request(self, pb, session_token_id, *, prompt, system, messages,
                              response_schema, cache_ttl, max_tokens, temperature, stop, strict):
        """The GenerateStreamRequest, plus whether it used the structured C1 fields.

        Compatibility with a kernel that predates C1: such a kernel drops the unknown
        ``system``/``messages`` fields and reads ``prompt``. So whenever the request
        is structured, the flattened prompt rides along in the deprecated ``prompt``
        field. A C1 kernel ignores ``prompt`` when ``messages`` is non-empty, so it
        is never read twice; an older kernel still gets the whole text (without the
        cache breakpoint).
        """
        from .llm import coerce_cache_ttl, coerce_messages, coerce_response_schema, coerce_system, flatten_prompt
        blocks = coerce_system(system)
        msgs = coerce_messages(messages)
        schema = coerce_response_schema(response_schema)
        ttl = coerce_cache_ttl(cache_ttl)
        structured = bool(blocks or msgs or schema)
        if structured and not msgs and prompt:
            msgs = [{"role": "user", "content": prompt}]
        legacy_prompt = flatten_prompt(blocks, msgs) if structured else prompt
        req = pb.GenerateStreamRequest(
            # Phase 1: lease_id is the honest name; session_token_id is sent too so a
            # kernel predating the field still authenticates the call. Same value.
            lease_id=session_token_id,
            session_token_id=session_token_id,
            prompt=legacy_prompt,
            options=_generate_options(pb, max_tokens, temperature, stop, strict),
        )
        if structured:
            req.system.extend(_pb_system(pb, blocks))
            req.messages.extend(_pb_messages(pb, msgs))
            if schema is not None:
                req.response_schema.CopyFrom(_pb_schema(pb, schema))
        if ttl:
            req.cache_ttl = _pb_cache_ttl(pb, ttl)
        _log_request("GenerateViaModelStream", blocks, msgs, schema, ttl)
        return req, structured

    def generate_with_tools(
        self,
        session_token_id: str,
        messages: list,
        tools: list,
        max_tokens: Optional[int] = 1024,
        temperature: Optional[float] = 0.7,
        timeout_ms: int = 0,
        *,
        system=None,
        response_schema=None,
        cache_ttl=None,
        stop=None,
        strict: bool = False,
    ):
        """One generation turn with NATIVE tool-calling (ADR-0097 Phase B).

        ``messages`` is the CONVERSATION so far — a list of
        ``{"role", "content", "tool_calls"?, "tool_call_id"?}`` dicts. Native
        tool-calling is conversational: an assistant turn carrying ``tool_calls`` and a
        ``role="tool"`` turn carrying the result under the same id are what let the model
        see that its own call happened. Sending a single user turn per round — the first
        cut of this method — makes every round a fresh conversation.

        ``system`` / ``response_schema`` / ``cache_ttl`` / ``stop`` / ``strict`` are the
        C1 fields, as on :meth:`generate_request`. System text goes only in ``system``;
        a ``"system"`` message raises ``InvalidRequest``.

        Returns a ``GenerateResult``. It unpacks as the old ``(text, tool_calls,
        stop_reason)`` tuple, where ``tool_calls`` is a list of ``{"id", "name",
        "arguments"}`` dicts and ``stop_reason`` is the kernel's normalized value —
        only ``"end_turn"`` means finished — and also carries ``usage``, ``ignored``
        and ``model``.

        ``tools`` is a list of ``{"name", "description", "parameters"}`` dicts, where
        ``parameters`` is a JSON Schema object (dict or pre-encoded string).

        Raises ``ToolCallingUnsupported`` when this deployment or the model allocated
        to the step cannot do native tool-calling. That is a CAPABILITY answer, not a
        failure: the caller is expected to fall back to the prompt-encoded action
        protocol, which is why it is a distinct exception rather than a generic error.
        Other failures raise the typed errors listed on :meth:`generate_request`.
        """
        self._ensure()
        import grpc

        from ._proto import cambrian_pb2
        from .llm import GenerateResult as _Result
        from .llm import coerce_cache_ttl, coerce_messages, coerce_response_schema, coerce_system

        blocks = coerce_system(system)
        msgs = coerce_messages(messages)
        schema = coerce_response_schema(response_schema)
        ttl = coerce_cache_ttl(cache_ttl)

        pb_tools = []
        for t in tools or []:
            params = t.get("parameters", "")
            if not isinstance(params, str):
                params = json.dumps(params)
            pb_tools.append(cambrian_pb2.ToolDefinitionProto(
                name=t.get("name", ""),
                description=t.get("description", "") or "",
                parameters_json=params,
            ))

        req = cambrian_pb2.GenerateWithToolsRequest(
            lease_id=session_token_id,
            options=_generate_options(cambrian_pb2, max_tokens, temperature, stop, strict),
            tools=pb_tools,
            messages=_pb_messages(cambrian_pb2, msgs),
            system=_pb_system(cambrian_pb2, blocks),
        )
        if schema is not None:
            req.response_schema.CopyFrom(_pb_schema(cambrian_pb2, schema))
        if ttl:
            req.cache_ttl = _pb_cache_ttl(cambrian_pb2, ttl)
        _log_request("GenerateWithTools", blocks, msgs, schema, ttl)
        timeout_secs = (timeout_ms / 1000.0) if timeout_ms > 0 else None
        md = [("x-agent-id", self._agent_id)] if self._agent_id else []
        try:
            resp = self._stub.GenerateWithTools(req, timeout=timeout_secs, metadata=md)
        except grpc.RpcError as exc:
            raise _map_generate_error(exc, tools=True) from exc

        out = _Result(text=resp.text, stop_reason=resp.stop_reason, tool_calls=[
            {"id": tc.id, "name": tc.name, "arguments": tc.arguments_json}
            for tc in resp.tool_calls
        ])
        _fill_result_from(out, resp, structured=bool(blocks or schema), tools=True)
        return out

    def get_context_node(self, cid: str, session_token_id: str = "") -> Optional["ContextNode"]:
        """Resolve a CID from the ContentStore. Drill-down for offloaded content.

        Returns a :class:`ContextNode` with ``.data`` (bytes) when found, or
        ``None`` when the CID is unknown OR the node is owned by a different
        session (ADR-0048 D4 read-gate). Pass ``session_token_id`` to read an
        agent's own offloaded node (R7); ownerless nodes (tool/step results) read
        without it. Pass as ``fetch_fn`` to :func:`assemble_context`.
        """
        self._ensure()
        from ._proto import cambrian_pb2
        req = cambrian_pb2.ContextNodeRequest(cid=cid)
        md = _lease_metadata(session_token_id) or None
        resp = self._stub.GetContextNode(req, metadata=md)
        if not resp.data:
            return None
        return ContextNode(cid=resp.cid, type=resp.type, data=bytes(resp.data), labels=list(resp.labels))

    def put_context_node(self, data: str, session_token_id: str = "") -> Optional[str]:
        """Offload a text block to the ephemeral ContentStore, returning its CID
        (ADR-0048 D4/R7). The kernel owner-stamps the blob with ``session_token_id``
        so reads are gated to the same session (:meth:`get_context_node`). The blob
        is plan-scoped (reclaimed at plan end). Returns ``None`` on failure so the
        working-memory offloader degrades to rendering the block verbatim.
        """
        self._ensure()
        from ._proto import cambrian_pb2
        payload = data.encode("utf-8") if isinstance(data, str) else bytes(data)
        req = cambrian_pb2.PutContextNodeRequest(data=payload, node_type="agent_offload")
        md = _lease_metadata(session_token_id) or None
        try:
            resp = self._stub.PutContextNode(req, metadata=md)
            return resp.cid or None
        except Exception:  # noqa: BLE001 — degrade to verbatim, never crash the loop
            return None

    def ask(self, query: str, top_k: int = 5) -> List[Dict]:
        """Query the Substrate's shared LTM (pgvector documents table).

        Returns a list of ``{"text": str, "score": float, "metadata": str}`` dicts.
        Results are filtered by the Substrate's ACL — agents can only read
        documents they are permitted to access.
        """
        self._ensure()
        from ._proto import cambrian_pb2
        req = cambrian_pb2.MemoryRequest(query=query, top_k=top_k)
        resp = self._stub.QueryMemory(req)
        return [{"text": r.text, "score": r.score, "metadata": r.metadata} for r in resp.results]

    def execute_tool(
        self,
        tool_name: str,
        args_json: str = "",
        session_token_id: str = "",
        step_index: int = 0,
        timeout_ms: int = 0,
        task_id: str = "",
    ) -> Dict:
        """Invoke a kernel-owned system tool (ADR-0039). The agent marshals the
        args (``args_json``) in its think() loop; the kernel authorizes (grant +
        resource policy + scope + approval) and runs the tool in a confined
        process. Returns the raw response as a dict — a denial or error is data,
        never an exception — so the reasoning loop degrades instead of crashing.
        """
        self._ensure()
        from ._proto import cambrian_pb2

        req = cambrian_pb2.ExecuteToolRequest(
            tool_name=tool_name,
            args_json=args_json or "{}",
            # Phase 1: see generate() — both fields carry the same per-step lease.
            lease_id=session_token_id,
            session_token_id=session_token_id,
            step_index=step_index,
        )
        timeout_secs = (timeout_ms / 1000.0) if timeout_ms > 0 else None
        md = [("x-agent-id", self._agent_id)] if self._agent_id else []
        if task_id:
            md.append(("x-task-id", task_id))  # ADR-0049 D3: per-step correlation key
        resp = self._stub.ExecuteTool(req, timeout=timeout_secs, metadata=md)
        return {
            "result_json": resp.result_json,
            "result_cid": resp.result_cid,
            "denied": resp.denied,
            "deny_reason": resp.deny_reason,
            "error": resp.error,
            "arg_hash": resp.arg_hash,
            "result_hash": resp.result_hash,
            # ADR-0139 D11 hook context, carried beside the result (contract 0104);
            # an older kernel sends none.
            "annotations": list(getattr(resp, "annotations", []) or []),
        }

    def embed(self, text: str) -> List[float]:
        """Embed text into a vector via the kernel embedder (ADR-0041) — used by the
        agent's Local Recurrent Workspace to relevance-rank its episodic buffer.

        Read-only; the single minimal kernel surface LRW adds. Degrades to an empty
        list on any failure so ranking falls back to recency rather than crashing
        the loop.
        """
        self._ensure()
        from ._proto import cambrian_pb2

        try:
            resp = self._stub.Embed(cambrian_pb2.EmbedRequest(text=text))
        except Exception:  # noqa: BLE001 — degrade to recency ranking, never crash
            return []
        return list(resp.vector)

    def list_tools(
        self, query: str = "", k: int = 0,
        names: Optional[List[str]] = None, full: bool = False,
        session_token_id: str = "",
    ) -> List[Dict]:
        """Return the system tools this agent may invoke (ADR-0039), for the ReAct
        prompt menu. The kernel resolves the agent's grants from the x-agent-id
        metadata and honors the unrestricted bypass; the list is **advisory** —
        ``execute_tool`` still authorizes each call kernel-side. Degrades to an
        empty list on any RPC failure (e.g. a kernel without a tool registry), so
        the reasoning loop never crashes for lack of a tool plane.

        ADR-0044 semantic retrieval: pass ``query`` (and optional ``k``) to get
        only the task-relevant tools (kernel-ranked) instead of the full registry —
        the query rides x-tool-query / x-tool-k metadata. No query ⇒ the full menu.

        ADR-0045 two-tier disclosure: by default the menu is Tier-1 (a terse
        one-line summary + arg names). Pass ``names`` + ``full=True`` (the
        ``describe_tool`` path) to fetch Tier-2 (full description + full arg
        schema) for the named granted tools only — ungranted names are absent.
        These ride x-tool-names / x-tool-full metadata.

        ``session_token_id`` (the run's lease) lets the kernel also describe tools a
        system skill conferred for THIS run (ADR-0046 D6); without it only standing
        grants are described. Unset, the running task's lease is presented
        (:func:`~cambrian_agent_sdk.clients.task_lease`), as memory reads do.
        """
        self._ensure()
        from ._proto import cambrian_pb2

        md = [("x-agent-id", self._agent_id)] if self._agent_id else []
        if query:
            md.append(("x-tool-query", _header_safe(query)))
        if k > 0:
            md.append(("x-tool-k", str(k)))
        if names:
            md.append(("x-tool-names", ",".join(names)))
        if full:
            md.append(("x-tool-full", "true"))
        lease = _lease_or_current(session_token_id)
        if lease:
            md.append(("x-lease-id", lease))
        try:
            resp = self._stub.ListTools(cambrian_pb2.ListToolsRequest(), metadata=md)
        except Exception:  # noqa: BLE001 — degrade to an empty menu, never crash
            return []
        return [
            {
                "name": t.name,
                "description": t.description,
                "schema_json": t.schema_json,
                "dangerous": t.dangerous,
            }
            for t in resp.tools
        ]

    def find_tools(
        self, query: str, k: int = 0, full: bool = False, session_token_id: str = "",
    ) -> Tuple[List[Dict], List[str]]:
        """The find_tools pull: ``list_tools(query=...)`` plus the tools that match the
        query within the person's access but are NOT in this step.

        Under a composed step lease the kernel names those in the ListTools response
        header ``x-tool-outside-step`` (comma-separated, ranked; cambrian-core
        network.ToolOutsideStepHeader). They are never in the response body: the
        worker cannot load or call them, only report them in its capability_missing
        outcome (T-20261112-yield-sub-goal-on-a-composed-lease-runs-under). Returns
        ``(tools, outside)``; degrades to ``([], [])`` on any RPC failure.
        """
        self._ensure()
        from ._proto import cambrian_pb2

        md = [("x-agent-id", self._agent_id)] if self._agent_id else []
        if query:
            md.append(("x-tool-query", _header_safe(query)))
        if k > 0:
            md.append(("x-tool-k", str(k)))
        if full:
            md.append(("x-tool-full", "true"))
        lease = _lease_or_current(session_token_id)
        if lease:
            md.append(("x-lease-id", lease))
        try:
            resp, call = self._stub.ListTools.with_call(cambrian_pb2.ListToolsRequest(), metadata=md)
        except Exception:  # noqa: BLE001 — degrade to an empty menu, never crash
            return [], []
        tools = [
            {
                "name": t.name,
                "description": t.description,
                "schema_json": t.schema_json,
                "dangerous": t.dangerous,
            }
            for t in resp.tools
        ]
        outside: List[str] = []
        try:
            for key, value in call.initial_metadata() or ():
                if key == "x-tool-outside-step":
                    outside.extend(n.strip() for n in str(value).split(",") if n.strip())
        except Exception:  # noqa: BLE001 — the header is advisory
            outside = []
        return tools, list(dict.fromkeys(outside))

    def list_skills(
        self, query: str = "", k: int = 0,
        names: Optional[List[str]] = None, full: bool = False,
        session_token_id: str = "", rationale: str = "", discover: bool = False,
    ) -> List[Dict]:
        """Return the system skills this agent may load (ADR-0046), for the agent's
        [skills] menu section. The kernel resolves grants/scope from x-agent-id and
        gates visibility by the agent's effective scope. Degrades to an empty list
        on any RPC failure, so the loop never crashes for lack of a skill plane.

        Tier-1 (summary) by default; pass ``names`` + ``full=True`` (the use_skill
        path) to fetch Tier-2 (full instructions + bundled tool grants) for the
        named skills. query/k/names/full ride x-skill-* metadata (mirroring tools).
        Agent-local skills are NOT served here — they live in the SDK.

        Per-agent skill access (decisions/skill-access-grants.md): each dict carries
        ``access`` — "profile"/"granted" (in the agent's set), "requestable" (found,
        outside it), "pending"/"denied" (a use_skill request an operator did not grant
        in time / refused; no body) or "" (the kernel does not bound this agent) — and
        ``access_detail``. A full load of a requestable skill makes the kernel ask an
        operator and wait (up to execution.skills.access_request_timeout_seconds);
        ``rationale`` rides x-skill-rationale to that operator.
        ``discover`` is accepted for the profile proxy's signature; the kernel's query
        path is already discovery, so it changes nothing here.
        """
        self._ensure()
        from ._proto import cambrian_pb2

        md = [("x-agent-id", self._agent_id)] if self._agent_id else []
        if query:
            md.append(("x-skill-query", _header_safe(query)))
        if k > 0:
            md.append(("x-skill-k", str(k)))
        if names:
            md.append(("x-skill-names", ",".join(names)))
        if full:
            md.append(("x-skill-full", "true"))
        # The lease the kernel dispatched this task under (explicit, else the task's
        # lease the runtime set around run()), as x-lease-id: the kernel resolves the
        # caller's session / conversation owner from it, and a credential-authenticated
        # menu call without one is refused ``no_lease`` (ENG-518;
        # T-20261118-chat-agent-and-composed-worker-memory-reads-are).
        lease = _lease_or_current(session_token_id)
        if lease:
            md.append(("x-lease-id", lease))
        if session_token_id:
            md.append(("x-session-token", session_token_id))  # ADR-0046 D6: run key for grant activation
        if rationale:
            md.append(("x-skill-rationale", _header_safe(rationale)))
        try:
            resp = self._stub.ListSkills(cambrian_pb2.ListSkillsRequest(), metadata=md)
        except Exception:  # noqa: BLE001 — degrade to an empty menu, never crash
            return []
        return [
            {
                "name": sk.name,
                "description": sk.description,
                "instructions": sk.instructions,
                "tool_grants": list(sk.tool_grants),
                "access": sk.access,
                "access_detail": sk.access_detail,
            }
            for sk in resp.skills
        ]

    def execute(self, description: str, session_token_id: str = "", target: Optional[str] = None, timeout_ms: int = 0) -> "ExecuteResult":
        """Delegate a natural-language task to the Planner (ADR-0036 user story 11).

        Self-delegation is refused (recursion guard). Sub-plan tokens are charged to
        the **parent step's** session token by propagating ``session_token_id`` into
        the delegated handoff — delegation cannot bypass the budget (ADR-0018).

        Returns the whole result (ADR-0145 D8): an :class:`ExecuteResult`, which IS the
        report text (a ``str``, so callers that read text are unchanged) and carries the
        run id, the action ledger, every step with its ledger reference, and the
        partial / timeout flags as attributes (``.text`` is the plain text).
        """
        from .clients import SelfDelegationError

        if target is not None and target == self._agent_id and self._agent_id:
            raise SelfDelegationError(self._agent_id)
        self._ensure()
        from ._proto import cambrian_pb2

        req = cambrian_pb2.Handoff(
            from_agent=self._agent_id,
            to_agent=target or "",
            payload=cambrian_pb2.Object(type="text", data=description.encode("utf-8")),
            metadata={"_session_token_id": session_token_id, "_delegated": "true"},
        )
        timeout = (timeout_ms / 1000.0) if timeout_ms and timeout_ms > 0 else None
        resp = self._stub.Execute(req, timeout=timeout)
        return ExecuteResult(
            resp.payload.data.decode("utf-8", errors="replace"),
            metadata=dict(resp.metadata),
            payload_type=resp.payload.type,
        )

    # ── ADR-0139 D12 / D14 agent-plane verbs ────────────────────────────────────
    #
    # All three put the lease in the request's lease_id AND in x-lease-id metadata,
    # and the caller identity in x-agent-id. UNIMPLEMENTED (a kernel without the
    # surface) returns None: "this deployment has no such surface", never a failed
    # call to retry (cambrian.proto). Any other RPC error is returned as data
    # ({"error": ..., "code": ...}) so the reasoning loop can show it to the model.

    def _lease_md(self, lease: str):
        md = [("x-agent-id", self._agent_id)] if self._agent_id else []
        if lease:
            md.append(("x-lease-id", lease))
        return md

    @staticmethod
    def _rpc_failure(exc) -> Optional[Dict]:
        """None for UNIMPLEMENTED; an error dict for any other gRPC failure."""
        code = getattr(exc, "code", None)
        code = code() if callable(code) else None
        name = getattr(code, "name", "") or ""
        if name == "UNIMPLEMENTED":
            return None
        details = getattr(exc, "details", None)
        detail = details() if callable(details) else str(exc)
        return {"error": detail or str(exc), "code": name or "UNKNOWN"}

    def get_memory_notes(self, session_token_id: str = "") -> Optional[Dict]:
        """The run principal's FROZEN run-start memory notes (ADR-0139 D14 / D3).

        Returns ``{"version", "snapshot_at", "documents": [...]}`` with each document
        ``{scope, owner_id, content, size_chars, budget_chars, etag, truncated,
        updated_at}`` (``scope`` is ``"personal"`` / ``"org"``), PERSONAL first. None
        when the kernel does not serve notes (UNIMPLEMENTED); an error dict for any
        other refusal (e.g. PERMISSION_DENIED for a foreign lease).
        """
        self._ensure()
        from ._proto import cambrian_pb2

        try:
            snap = self._stub.GetMemoryNotes(
                cambrian_pb2.GetMemoryNotesRequest(lease_id=session_token_id),
                metadata=self._lease_md(session_token_id))
        except Exception as exc:  # noqa: BLE001 — notes are optional
            return self._rpc_failure(exc)

        def _ts(t) -> str:
            try:
                return t.ToDatetime().isoformat() + "Z" if (t.seconds or t.nanos) else ""
            except Exception:  # noqa: BLE001
                return ""

        docs = []
        for d in snap.documents:
            name = cambrian_pb2.MemoryNotesScope.Name(d.scope)
            docs.append({
                "scope": name.replace("MEMORY_NOTES_SCOPE_", "").lower(),
                "owner_id": d.owner_id,
                "content": d.content,
                "size_chars": d.size_chars,
                "budget_chars": d.budget_chars,
                "etag": d.etag,
                "truncated": d.truncated,
                "updated_at": _ts(d.updated_at),
            })
        docs.sort(key=lambda d: 0 if d["scope"] == "personal" else 1)
        return {"version": snap.version, "snapshot_at": _ts(snap.snapshot_at), "documents": docs}

    def propose_memory_note(
        self,
        session_token_id: str,
        *,
        scope: str = "personal",
        op: str = "append",
        section: str = "",
        content: str = "",
        reason: str = "",
        if_match: str = "",
    ) -> Optional[Dict]:
        """Propose one write to the principal's memory notes (ADR-0139 D14).

        ``scope`` is ``personal`` | ``org``; ``op`` is ``append`` | ``replace_section``
        | ``remove_section``. A refused write is an OK reply with ``accepted=False``,
        a ``rejection`` (``over_budget``, ``injection_flagged``, ``scope_denied``,
        ``conflict``, ``section_not_found``) and a ``detail``: data for the model,
        not an exception. The write is durable at once but reaches the prompt only
        next run (the snapshot is frozen). None when the kernel does not serve notes.
        """
        self._ensure()
        from ._proto import cambrian_pb2

        scope_v = {"personal": cambrian_pb2.MEMORY_NOTES_SCOPE_PERSONAL,
                   "org": cambrian_pb2.MEMORY_NOTES_SCOPE_ORG}.get(
            str(scope).lower(), cambrian_pb2.MEMORY_NOTES_SCOPE_UNSPECIFIED)
        op_v = {"append": cambrian_pb2.MEMORY_NOTE_OP_APPEND,
                "replace_section": cambrian_pb2.MEMORY_NOTE_OP_REPLACE_SECTION,
                "remove_section": cambrian_pb2.MEMORY_NOTE_OP_REMOVE_SECTION}.get(
            str(op).lower(), cambrian_pb2.MEMORY_NOTE_OP_UNSPECIFIED)
        req = cambrian_pb2.ProposeMemoryNoteRequest(
            lease_id=session_token_id, scope=scope_v, op=op_v, section=section or "",
            content=content or "", if_match=if_match or "", reason=reason or "")
        try:
            resp = self._stub.ProposeMemoryNote(req, metadata=self._lease_md(session_token_id))
        except Exception as exc:  # noqa: BLE001 — a refusal is data for the loop
            return self._rpc_failure(exc)
        return self._note_reply(cambrian_pb2, resp)

    def propose_skill(
        self,
        session_token_id: str,
        *,
        name: str,
        skill_md: str,
        rationale: str = "",
        evidence_cids: Optional[List[str]] = None,
        base_version: str = "",
    ) -> Optional[Dict]:
        """Propose a new skill, or a new version of one (ADR-0139 D13).

        ``name`` must equal the SKILL.md frontmatter ``name``; ``base_version`` is ""
        for a new skill, else the version being revised. The proposal enters the
        operator's staged approval queue and is NEVER active until approved. A
        refusal is an OK reply ``{"accepted": False, "rejection", "detail"}``
        (``invalid_format``, ``name_taken``, ``locked``, ``scan_dangerous``,
        ``quota``, ``denied``), data for the model; None when the kernel does not
        serve skill proposals (UNIMPLEMENTED).
        """
        self._ensure()
        from ._proto import cambrian_pb2

        req = cambrian_pb2.ProposeSkillRequest(
            lease_id=session_token_id, name=name or "", skill_md=skill_md or "",
            rationale=rationale or "", evidence_cids=[str(c) for c in (evidence_cids or [])],
            base_version=base_version or "")
        try:
            resp = self._stub.ProposeSkill(req, metadata=self._lease_md(session_token_id))
        except Exception as exc:  # noqa: BLE001 — a refusal is data for the loop
            return self._rpc_failure(exc)
        rej = cambrian_pb2.SkillProposalRejection.Name(resp.rejection)
        return {
            "accepted": resp.accepted,
            "proposal_id": resp.proposal_id,
            "rejection": ("" if resp.accepted or rej.endswith("UNSPECIFIED")
                          else rej.replace("SKILL_PROPOSAL_REJECTION_", "").lower()),
            "detail": resp.detail,
        }

    # ── Principal-owned recurring tasks (capability "scheduled-tasks") ───────
    #
    # The OWNER is never a parameter: the kernel resolves it from the lease (the
    # person the run is for). A proposal BLOCKS while that person is asked through
    # the approval channel (the console Inbox), up to the kernel's
    # execution.schedules.approval_timeout_seconds, and creates nothing unless they
    # approve. UNIMPLEMENTED -> None, any other failure -> an error dict (data).

    @staticmethod
    def _scheduled_task_dict(t) -> Dict:
        return {"id": t.id, "owner_id": t.owner_id, "title": t.title, "goal": t.goal,
                "cron": t.cron, "timezone": t.timezone, "agent_id": t.agent_id,
                "delivery": t.delivery, "active": t.active,
                "conversation_id": t.conversation_id, "created_at": t.created_at}

    def propose_scheduled_task(
        self,
        session_token_id: str,
        *,
        goal: str,
        cron: str = "",
        rrule: str = "",
        timezone: str = "",
        title: str = "",
        agent_id: str = "",
        delivery: str = "",
        timeout_s: Optional[float] = None,
    ) -> Optional[Dict]:
        """Propose a recurring task for the person this run is for.

        Give exactly one of ``cron`` (5 fields or @daily/@weekly/...) or ``rrule``
        (FREQ=HOURLY|DAILY|WEEKLY|MONTHLY with BYHOUR/BYMINUTE/BYDAY/BYMONTHDAY).
        ``delivery`` is ``conversation`` (default), ``notify`` or ``email``.
        ``agent_id`` is DEPRECATED: leave it empty — the task always runs as its
        owner on the chat lane, and the kernel refuses a non-empty value as
        ``invalid``. Returns
        ``{"created": True, "task": {...}}`` once approved, or ``{"created": False,
        "rejection": "invalid"|"limit"|"too_frequent"|"denied"|"no_decision"|
        "no_principal", "detail": ...}``; None when the kernel has no scheduled tasks.
        """
        self._ensure()
        from ._proto import cambrian_pb2

        req = cambrian_pb2.ProposeScheduledTaskRequest(
            lease_id=session_token_id, goal=goal or "", cron=cron or "", rrule=rrule or "",
            timezone=timezone or "", title=title or "", agent_id=agent_id or "",
            delivery=delivery or "")
        try:
            resp = self._stub.ProposeScheduledTask(req, metadata=self._lease_md(session_token_id),
                                                   timeout=timeout_s)
        except Exception as exc:  # noqa: BLE001 — a refusal is data for the loop
            return self._rpc_failure(exc)
        if resp.created:
            return {"created": True, "task": self._scheduled_task_dict(resp.task)}
        rej = cambrian_pb2.ScheduledTaskRejection.Name(resp.rejection)
        return {"created": False,
                "rejection": "" if rej.endswith("UNSPECIFIED") else rej.replace("SCHEDULED_TASK_REJECTION_", "").lower(),
                "detail": resp.detail}

    def propose_knowledge(
        self,
        session_token_id: str,
        *,
        item: Optional[Dict] = None,
        link: Optional[Dict] = None,
        request_commit: bool = False,
        reason: str = "",
        client_ref: str = "",
    ) -> Optional[Dict]:
        """Propose a knowledge item or a candidate link (Orchestrator/ProposeKnowledge).

        ``item`` is ``{"kind", "entity_id", "values": [{"predicate", "type", "value"}],
        "valid_from"?, "valid_to"?, "negation"?, "evidence_id"?, "classification"?}`` —
        ``type`` is ``text`` | ``number`` | ``date`` (an aware datetime or an ISO-8601
        string) | ``entity``. It is stored as a CANDIDATE asserted by this agent (the
        kernel takes the identity from the lease); ``request_commit`` also asks the
        person this run is for to approve it, and the approver becomes the committed
        value's asserter. ``link`` is ``{"relation", "from_ref", "to_ref", "evidence_id",
        "family"?, "confidence"?}`` and is only ever a candidate. Returns ``{"accepted",
        "detail", "candidate_id", "created", "approval_id", "approval_state"}``; a
        refusal is ``accepted=False`` with the named reason. None when the kernel does
        not serve knowledge proposals.
        """
        self._ensure()
        from datetime import datetime

        from google.protobuf import timestamp_pb2

        from ._proto import cambrian_pb2

        def ts(v):
            if v is None or v == "":
                return None
            if isinstance(v, str):
                v = datetime.fromisoformat(v.replace("Z", "+00:00"))
            if v.tzinfo is None:
                raise ValueError("propose_knowledge: datetimes must be timezone-aware")
            out = timestamp_pb2.Timestamp()
            out.FromDatetime(v)
            return out

        req = cambrian_pb2.ProposeKnowledgeRequest(
            lease_id=session_token_id, request_commit=bool(request_commit),
            reason=reason or "", client_ref=client_ref or "")
        if item is not None:
            pi = req.item
            pi.kind = str(item.get("kind") or "")
            pi.entity_id = str(item.get("entity_id") or "")
            pi.negation = bool(item.get("negation", False))
            pi.evidence_id = str(item.get("evidence_id") or "")
            pi.classification.extend([str(t) for t in (item.get("classification") or [])])
            for field in ("valid_from", "valid_to"):
                t = ts(item.get(field))
                if t is not None:
                    getattr(pi, field).CopyFrom(t)
            for v in item.get("values") or []:
                kv = pi.values.add(predicate=str(v.get("predicate") or ""), type=str(v.get("type") or "text"))
                val = v.get("value")
                if kv.type == "number":
                    kv.number = float(val or 0)
                elif kv.type == "date":
                    t = ts(val)
                    if t is not None:
                        kv.date.CopyFrom(t)
                elif kv.type == "entity":
                    kv.entity_ref = str(val or "")
                else:
                    kv.text = "" if val is None else str(val)
        elif link is not None:
            req.link.CopyFrom(cambrian_pb2.ProposedKnowledgeLink(
                family=str(link.get("family") or ""), from_ref=str(link.get("from_ref") or ""),
                to_ref=str(link.get("to_ref") or ""), relation=str(link.get("relation") or ""),
                evidence_id=str(link.get("evidence_id") or ""),
                confidence=float(link.get("confidence") or 0)))
        else:
            return {"accepted": False, "detail": "propose_knowledge needs an item or a link"}
        try:
            resp = self._stub.ProposeKnowledge(req, metadata=self._lease_md(session_token_id))
        except Exception as exc:  # noqa: BLE001 — a refusal is data for the loop
            return self._rpc_failure(exc)
        return {"accepted": resp.accepted, "detail": resp.detail, "candidate_id": resp.candidate_id,
                "created": resp.created, "approval_id": resp.approval_id,
                "approval_state": resp.approval_state}

    def propose_ontology_change(
        self,
        session_token_id: str,
        *,
        change: object,
        namespace: str = "",
        reason: str = "",
        client_ref: str = "",
    ) -> Optional[Dict]:
        """Draft a change to a namespace's kind/relation declarations
        (Orchestrator/ProposeOntologyChange, ADR-0149 D2).

        ``change`` is ``{"kinds": [KindSpec...], "relations": [RelationSpec...]}`` (a dict
        or its JSON string): each spec is added, or replaces the namespace's spec of the
        same name. The agent only DRAFTS: the kernel classifies it (a breaking, invalid
        or empty change is refused with the reason), records it as drafted by this agent
        (from the lease) and asks a person to approve it; the approver applies it.
        Returns ``{"accepted", "detail", "draft_id", "created", "base_version", "class",
        "changes", "state", "approval_id", "approval_state"}``; a refusal is
        ``accepted=False`` with the reason. None when the kernel does not serve drafts.
        """
        self._ensure()
        from ._proto import cambrian_pb2

        change_json = change if isinstance(change, str) else json.dumps(change or {})
        req = cambrian_pb2.ProposeOntologyChangeRequest(
            lease_id=session_token_id, namespace=namespace or "", change_json=change_json,
            reason=reason or "", client_ref=client_ref or "")
        try:
            resp = self._stub.ProposeOntologyChange(req, metadata=self._lease_md(session_token_id))
        except Exception as exc:  # noqa: BLE001 — a refusal is data for the loop
            return self._rpc_failure(exc)
        return {"accepted": resp.accepted, "detail": resp.detail, "draft_id": resp.draft_id,
                "created": resp.created, "base_version": resp.base_version, "class": getattr(resp, "class"),
                "changes": [{"class": getattr(c, "class"), "subject": c.subject, "detail": c.detail}
                            for c in resp.changes],
                "state": resp.state, "approval_id": resp.approval_id, "approval_state": resp.approval_state}

    def list_scheduled_tasks(self, session_token_id: str) -> Optional[Dict]:
        """The scheduled tasks of the person this run is for: ``{"tasks": [...]}``."""
        self._ensure()
        from ._proto import cambrian_pb2

        try:
            resp = self._stub.ListScheduledTasks(
                cambrian_pb2.ListScheduledTasksRequest(lease_id=session_token_id),
                metadata=self._lease_md(session_token_id))
        except Exception as exc:  # noqa: BLE001
            return self._rpc_failure(exc)
        return {"tasks": [self._scheduled_task_dict(t) for t in resp.tasks]}

    def manage_scheduled_task(self, session_token_id: str, task_id: str, action: str) -> Optional[Dict]:
        """Pause, resume or delete one of the person's scheduled tasks.
        ``action`` is ``pause`` | ``resume`` | ``delete``; ``{"ok": bool, "detail"}``."""
        self._ensure()
        from ._proto import cambrian_pb2

        act = {"pause": cambrian_pb2.SCHEDULED_TASK_ACTION_PAUSE,
               "resume": cambrian_pb2.SCHEDULED_TASK_ACTION_RESUME,
               "delete": cambrian_pb2.SCHEDULED_TASK_ACTION_DELETE}.get(
            str(action).lower(), cambrian_pb2.SCHEDULED_TASK_ACTION_UNSPECIFIED)
        try:
            resp = self._stub.ManageScheduledTask(
                cambrian_pb2.ManageScheduledTaskRequest(lease_id=session_token_id, id=task_id or "", action=act),
                metadata=self._lease_md(session_token_id))
        except Exception as exc:  # noqa: BLE001
            return self._rpc_failure(exc)
        return {"ok": resp.ok, "detail": resp.detail}

    # ── Live step output (ADR-0139 amendment 2026-11-06) ─────────────────────
    #
    # For an agent that calls its OWN model (an external adapter: Claude Code / the
    # Claude Agent SDK as a worker, LangGraph, a custom agent). A model call through
    # the gateway (generate_with_tools / generate_stream) is forwarded by the kernel
    # itself; such an agent must not also push. See step_output.py for the rules and
    # an example adapter.

    def _step_output_seqs(self):
        with self._lock:
            seqs = getattr(self, "_seqs", None)
            if seqs is None:
                from .step_output import _SeqCounter
                seqs = self._seqs = _SeqCounter()
            return seqs

    def step_output(self, session_token_id: str, timeout_ms: int = 0):
        """A :class:`~cambrian_agent_sdk.step_output.StepOutputWriter` for the step:
        ``with substrate.step_output(lease) as out: out.text(token)``. Leaving the
        block sends the final marker. Failures are recorded on the writer
        (``.error``), never raised — live output never fails the step."""
        self._ensure()
        from .step_output import StepOutputWriter
        return StepOutputWriter(self._stub, self._lease_md(session_token_id), session_token_id,
                                self._step_output_seqs(), timeout_ms=timeout_ms)

    def emit_step_output(self, session_token_id: str, chunks, *, final: bool = False,
                         timeout_ms: int = 0) -> Optional[Dict]:
        """Push live output for the step as one stream. ``chunks`` are strings (text)
        or dicts ``{"data", "kind": "text"|"progress"|"status", "final", "seq"}``;
        ``final=True`` appends the final marker. Returns the kernel's ack
        (``accepted``, ``duplicates``, ``over_cap``, ``after_final``, ``last_seq``),
        None when the kernel does not serve the RPC, or an error dict."""
        self._ensure()
        from .step_output import emit_step_output
        return emit_step_output(self._stub, self._lease_md(session_token_id),
                                self._step_output_seqs(), session_token_id, chunks,
                                final=final, timeout_ms=timeout_ms)

    @staticmethod
    def _note_reply(cambrian_pb2, resp) -> Dict:
        rej = cambrian_pb2.MemoryNoteRejection.Name(resp.rejection)
        return {
            "accepted": resp.accepted,
            "rejection": ("" if resp.accepted or rej.endswith("UNSPECIFIED")
                          else rej.replace("MEMORY_NOTE_REJECTION_", "").lower()),
            "detail": resp.detail,
            "etag": resp.etag,
            "size_chars": resp.size_chars,
            "budget_chars": resp.budget_chars,
        }

# ── assemble_context ─────────────────────────────────────────────────────────


def assemble_context(
    refs: List["ContextRef"],
    min_precision: float = 0.5,
    max_tokens: int = 800,
    fetch_fn: Optional[Callable] = None,
    fetch_threshold: float = 0.7,
) -> str:
    """Build a prompt-ready context string from a Global Workspace working set.

    ADR-0022 Phase 3 SDK helper. Encapsulates ref ranking, precision filtering,
    ContentStore fetching, and token budgeting so agent authors write one line:

        context_str = assemble_context(
            request.working_memory,
            min_precision=0.5,
            max_tokens=800,
            fetch_fn=agent.substrate.get_context_node,
        )

    Sort order: ``activation × precision`` descending (structural × semantic).
    BFS nodes with ``precision == -1.0`` are treated as unknown — they require
    ``fetch_fn`` to resolve; without it they are skipped.

    Args:
        refs: The ``working_memory`` list from an ``ExecuteRequest``.
        min_precision: Skip refs whose precision is below this threshold.
                       Ignored for refs with precision==-1.0 and fetch_fn provided.
        max_tokens: Approximate token budget (chars ÷ 4). Stops adding chunks
                    when budget is exhausted.
        fetch_fn: Optional callable ``(cid: str) -> node`` where ``node.data``
                  is bytes. When None, snippet-only degraded mode is used.
                  fetch_fn errors fall back to snippet rather than raising.
        fetch_threshold: Minimum precision to trigger a full fetch. Refs between
                         min_precision and fetch_threshold use their snippet.

    Returns:
        A multi-line string of ``[type] content`` blocks, or ``""`` when empty.
    """
    if not refs:
        return ""

    def _score(r: "ContextRef") -> float:
        if r.precision < 0:
            return r.activation * 0.0  # unknown precision → sort to end
        return r.activation * r.precision

    ranked = sorted(refs, key=_score, reverse=True)

    parts: List[str] = []
    tokens_used = 0

    for r in ranked:
        # Skip refs below precision threshold (unknown precision requires fetch_fn).
        if r.precision >= 0 and r.precision < min_precision:
            continue
        if r.precision < 0 and fetch_fn is None:
            continue  # BFS node, no fetch — cannot determine precision

        # Resolve content: full fetch or snippet.
        text = r.snippet
        if fetch_fn is not None and (r.precision < 0 or r.precision >= fetch_threshold):
            try:
                node = fetch_fn(r.cid)
                if node is not None and hasattr(node, "data"):
                    fetched = node.data
                    if isinstance(fetched, bytes):
                        fetched = fetched.decode("utf-8", errors="replace")
                    text = fetched
            except Exception:
                pass  # degraded: use snippet

        if not text:
            continue

        budget = (max_tokens - tokens_used) * 4
        chunk = text[:budget] if budget > 0 else ""
        if not chunk:
            break

        # REQ-AGENT-CTX-2: structured XML fact blocks with precision attribute.
        # When the ref is offloaded (has a cid), the rendered chunk is only the head of
        # the full body — surface the cid so the agent can pass it BY REFERENCE
        # (`{"$cid": …}`) into a tool arg (e.g. write_file content) and the kernel
        # resolves the WHOLE content server-side. Without this the agent only ever sees
        # the truncated head of a prior step's output and is forced to regenerate it.
        marker = f" [full content cid:{r.cid}]" if getattr(r, "cid", "") else ""
        parts.append(f'<fact precision="{r.precision:.2f}">{chunk}{marker}</fact>')
        tokens_used += len(chunk) // 4
        if tokens_used >= max_tokens:
            break

    result = "\n".join(parts)
    logger.info(
        "assemble_context_done",
        extra={
            "cid_count": len(refs),
            "tokens_used": tokens_used,
            "parts_count": len(parts),
            "degraded_mode": fetch_fn is None,
        },
    )
    return result



