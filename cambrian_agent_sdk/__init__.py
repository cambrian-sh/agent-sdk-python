"""Cambrian Agent SDK — build Cambrian agents in Python with zero boilerplate.

Minimal usage::

    from cambrian_agent_sdk import Agent, Capability

    agent = Agent(
        agent_id="my-agent",
        capabilities=[Capability(name="text_generation", latency_p50_ms=3000)],
    )

    @agent.capability("text_generation")
    def generate(request):
        text = request.payload.text
        return {"data": f"Echo: {text}".encode()}

    if __name__ == "__main__":
        agent.serve()

The agent file must be named ``*agent.py`` and live in the ``agents/`` directory
configured in ``config.json``. The Substrate auto-discovers it on startup.

AGENT_DESCRIPTION and AGENT_MANIFEST must be module-level literals — the
Substrate parses them with regex before the Python process boots. Use
``cambrian manifest ./my_agent.py`` to generate or update the manifest block.
"""

from __future__ import annotations

import json
import logging
import sys
import threading
from typing import Callable, Dict, List, Optional, TYPE_CHECKING

from .types import (
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
    yield_subgoal,
)
from .base import Agent, CognitiveAgent, DeterministicAgent, DaemonAgent
from .tools import tool, capability, ToolRegistry, ToolSpec
from .react import ReActLoopError
from .clients import (
    _lease_metadata,
    ArtifactManager,
    ArtifactNotFound,
    InvalidTagError,
    MemoryClient,
    SelfDelegationError,
    WorkingMemory,
)
from .helpers import extract_code_block, find_step_ref, build_prompt
from .errors import BudgetExceededError

__all__ = [
    # ADR-0036 v2 trait-aligned surface
    "Agent",  # abstract base
    "CognitiveAgent",
    "DeterministicAgent",
    "DaemonAgent",
    "AgentTask",
    "AgentResult",
    "SubGoal",
    "yield_subgoal",
    "tool",
    "capability",
    "ToolRegistry",
    "ToolSpec",
    "ReActLoopError",
    "MemoryClient",
    "ArtifactManager",
    "WorkingMemory",
    "InvalidTagError",
    "ArtifactNotFound",
    "SelfDelegationError",
    # shared vocabulary + helpers
    "assemble_context",
    "build_prompt",
    "BudgetExceededError",
    "Capability",
    "ContextNode",
    "ContextRef",
    "ExecuteRequest",
    "ExecuteResponse",
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


class SubstrateClient:
    """Backchannel to the Substrate for delegation calls and LTM queries.

    Injected as ``self.substrate`` on every ExecuteRequest so agents can
    sub-delegate tasks and query shared memory without gRPC boilerplate.
    """

    def __init__(self, substrate_addr: str = "localhost:50051", stub=None, agent_id: str = "") -> None:
        self._addr = substrate_addr
        self._channel = None
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
            import grpc
            from ._proto import cambrian_pb2_grpc
            self._channel = grpc.insecure_channel(self._addr)
            self._stub = cambrian_pb2_grpc.OrchestratorStub(self._channel)

    def generate(
        self,
        session_token_id: str,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        timeout_ms: int = 0,
    ) -> str:
        """Call the Substrate's managed LLM proxy and return the full text.

        Pass ``timeout_ms=request.deadline_remaining_ms`` to propagate the Go
        step deadline into the gRPC call. Without this, a slow LLM call will
        block the thread after the Go side has already moved on with
        DEADLINE_EXCEEDED, leaking a ThreadPoolExecutor slot per hung call.

        Internally delegates to :meth:`generate_stream` — see that method for
        the streaming variant (useful for > 4k-token outputs).
        """
        return "".join(self.generate_stream(
            session_token_id=session_token_id,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout_ms=timeout_ms,
        ))

    def generate_stream(
        self,
        session_token_id: str,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        timeout_ms: int = 0,
    ):
        """Stream text chunks from the Substrate's managed LLM proxy.

        Yields each text fragment as it arrives from ``GenerateViaModelStream``.
        Use this for large outputs to avoid buffering the entire response before
        returning — prevents ThreadPoolExecutor saturation on 8k-token replies.

        Example::

            for chunk in agent.substrate.generate_stream(token, prompt, timeout_ms=5000):
                partial_output += chunk
        """
        self._ensure()
        from ._proto import cambrian_pb2
        req = cambrian_pb2.GenerateStreamRequest(
            # Phase 1: lease_id is the honest name; session_token_id is sent too so a
            # kernel predating the field still authenticates the call. Same value.
            lease_id=session_token_id,
            session_token_id=session_token_id,
            prompt=prompt,
            options=cambrian_pb2.GenerateOptions(
                max_tokens=max_tokens,
                temperature=temperature,
            ),
        )
        timeout_secs = (timeout_ms / 1000.0) if timeout_ms > 0 else None
        md = [("x-agent-id", self._agent_id)] if self._agent_id else []
        for chunk in self._stub.GenerateViaModelStream(req, timeout=timeout_secs, metadata=md):
            if chunk.text:
                yield chunk.text

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

    def list_skills(
        self, query: str = "", k: int = 0,
        names: Optional[List[str]] = None, full: bool = False,
        session_token_id: str = "",
    ) -> List[Dict]:
        """Return the system skills this agent may load (ADR-0046), for the agent's
        [skills] menu section. The kernel resolves grants/scope from x-agent-id and
        gates visibility by the agent's effective scope. Degrades to an empty list
        on any RPC failure, so the loop never crashes for lack of a skill plane.

        Tier-1 (summary) by default; pass ``names`` + ``full=True`` (the use_skill
        path) to fetch Tier-2 (full instructions + bundled tool grants) for the
        named skills. query/k/names/full ride x-skill-* metadata (mirroring tools).
        Agent-local skills are NOT served here — they live in the SDK.
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
        if session_token_id:
            md.append(("x-session-token", session_token_id))  # ADR-0046 D6: run key for grant activation
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
            }
            for sk in resp.skills
        ]

    def execute(self, description: str, session_token_id: str = "", target: Optional[str] = None, timeout_ms: int = 0) -> str:
        """Delegate a natural-language task to the Planner (ADR-0036 user story 11).

        Self-delegation is refused (recursion guard). Sub-plan tokens are charged to
        the **parent step's** session token by propagating ``session_token_id`` into
        the delegated handoff — delegation cannot bypass the budget (ADR-0018).
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
        return resp.payload.data.decode("utf-8", errors="replace")

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

        label = r.type or "context"
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



