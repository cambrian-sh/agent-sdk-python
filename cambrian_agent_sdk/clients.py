"""Memory / artifact / substrate client surfaces (ADR-0036 issue 0036-05).

Thin Python wrappers over already-shipped gRPC RPCs. They carry **no client-side
scope or classification authority** (D5; honors ADR-0034/0035):

- **reads** (``recall`` / ``get`` / ``list_from_step``) send no scope params — scope is
  server-derived from the authenticated agent;
- **writes** (``remember`` / ``save``) carry only a **narrow-only** ``hint`` — the kernel
  derives the authoritative classification from operator-configured ``DefaultWriteTags``.

Error surfaces: a coined hint → :class:`InvalidTagError`; a scope-denied read →
:class:`ArtifactNotFound` (absence, never an existence leak).
"""

from __future__ import annotations

from typing import List, Optional


class InvalidTagError(Exception):
    """A narrow-only hint named a tag outside the controlled vocabulary (coinage)."""


class ArtifactNotFound(Exception):
    """The artifact does not exist or is not readable under the agent's scope."""


class SelfDelegationError(Exception):
    """An agent attempted to delegate a task back to itself (recursion guard)."""


def _is_invalid_argument(exc) -> bool:
    import grpc

    return isinstance(exc, grpc.RpcError) and exc.code() == grpc.StatusCode.INVALID_ARGUMENT


class MemoryClient:
    """``memory`` — recall (read) + remember (write, narrow-only hint)."""

    def __init__(self, stub=None, agent_id: str = "", channel=None) -> None:
        self._stub = stub
        self._agent_id = agent_id
        self._channel = channel

    def recall(self, query: str, top_k: int = 5, timeout_ms: int = 0,
               session_token_id: str = "") -> List[dict]:
        """Retrieve from LTM. Results are already scope-filtered server-side; this
        request carries **no** scope params (D5/ADR-0034).

        ``session_token_id`` is threaded as ``x-session-id`` so the kernel can apply
        the same-session step-record filter (ADR-0048 D1): without it the server sees
        an empty session and D1 silently no-ops, so the agent's OWN step output is
        recalled straight back into the same run — the context feedback loop D1 exists
        to break."""
        from ._proto import cambrian_pb2

        req = cambrian_pb2.MemoryRequest(query=query, top_k=top_k)
        md = self._identity_metadata()
        if session_token_id:
            md = md + [("x-session-id", session_token_id)]
        resp = self._stub.QueryMemory(req, timeout=_secs(timeout_ms), metadata=md)
        return [{"text": r.text, "score": r.score, "metadata": r.metadata} for r in resp.results]

    def recall_actions(self, query: str, top_k: int = 5, timeout_ms: int = 0,
                       session_token_id: str = "") -> List[dict]:
        """Retrieve from the ACTIONS lane (ADR-0049 D4) — "what did I do" (action
        records), a SEPARATE intent from :meth:`recall` ("what do I know") so action
        breadcrumbs never re-bloat fact grounding. Routed by the ``x-lane=actions``
        gRPC metadata; same server-side scope/relevance gating as fact recall."""
        from ._proto import cambrian_pb2

        req = cambrian_pb2.MemoryRequest(query=query, top_k=top_k)
        md = self._identity_metadata() + [("x-lane", "actions")]
        if session_token_id:
            md = md + [("x-session-id", session_token_id)]
        resp = self._stub.QueryMemory(req, timeout=_secs(timeout_ms), metadata=md)
        return [{"text": r.text, "score": r.score, "metadata": r.metadata} for r in resp.results]

    def recall_entity(self, kind_id: str, timeout_ms: int = 0,
                      session_token_id: str = "") -> List[dict]:
        """Resolve ONE entity's current reconstructed state by its canonical ``kind:id``
        (ADR-0049 Issue 012) — e.g. ``file:c:/repo/a.md`` or ``api:https://x.com``. This
        is an EXACT lookup ("what is true of that thing now?"), not a semantic search:
        the returned text is the materialized field-LWW view (a deleted file reads
        ``exists=false``) plus the link to its most-recent engaging scene. Routed by
        ``x-lane=entity``. Empty list when the entity is unknown."""
        from ._proto import cambrian_pb2

        req = cambrian_pb2.MemoryRequest(query=kind_id, top_k=1)
        md = self._identity_metadata() + [("x-lane", "entity")]
        if session_token_id:
            md = md + [("x-session-id", session_token_id)]
        resp = self._stub.QueryMemory(req, timeout=_secs(timeout_ms), metadata=md)
        return [{"text": r.text, "score": r.score, "metadata": r.metadata} for r in resp.results]

    def recall_precedents(self, query: str, top_k: int = 5, timeout_ms: int = 0,
                          session_token_id: str = "") -> List[dict]:
        """Retrieve from the PRECEDENT lane (ADR-0049 D11/Issue 014) — prior TRANSITIONS
        for the current sub-situation ("situations like this → what followed"), so the
        agent can anticipate the consequence of its next action. Failure-weighted
        (negative precedents first) and similarity-gated (empty == "no precedent", never
        a fabricated analogy). A DISTINCT lane from facts/actions, routed by
        ``x-lane=precedents``; fed by the live pull path (``PrimeForStep`` is not
        revived). The agent's LLM reasons over the returned transitions."""
        from ._proto import cambrian_pb2

        req = cambrian_pb2.MemoryRequest(query=query, top_k=top_k)
        md = self._identity_metadata() + [("x-lane", "precedents")]
        if session_token_id:
            md = md + [("x-session-id", session_token_id)]
        resp = self._stub.QueryMemory(req, timeout=_secs(timeout_ms), metadata=md)
        return [{"text": r.text, "score": r.score, "metadata": r.metadata} for r in resp.results]

    def remember(
        self,
        text: str,
        hint: Optional[List[str]] = None,
        source: str = "",
        session_id: str = "",
        importance: float = 0.0,
    ) -> str:
        """Persist a synthesized insight. ``hint`` is a **narrow-only** classification
        hint (never authoritative tags); the kernel derives the real classification.
        Returns the new doc ID. A coined hint raises :class:`InvalidTagError`."""
        from ._proto import cambrian_pb2

        req = cambrian_pb2.IngestMemoryRequest(
            text=text,
            tags=list(hint or []),  # narrow-only hint
            importance=importance,
            source=source,
            session_id=session_id,
        )
        try:
            resp = self._stub.IngestMemory(req, metadata=self._identity_metadata())
        except Exception as exc:
            if _is_invalid_argument(exc):
                raise InvalidTagError(str(exc)) from exc
            raise
        return resp.doc_id

    def _identity_metadata(self):
        """gRPC metadata carrying the agent principal. The server derives the
        scope principal from x-agent-id and fail-closes on an empty one, so a
        memory call without it is denied (ADR-0034)."""
        return [("x-agent-id", self._agent_id)] if self._agent_id else []


class ArtifactManager:
    """``artifacts`` — save (narrow-only hint) / get / list_from_step. No scope tags."""

    def __init__(self, stub=None, channel=None) -> None:
        self._stub = stub
        self._channel = channel

    def save(
        self,
        content: bytes,
        hint: Optional[List[str]] = None,
        content_type: str = "",
        session_id: str = "",
        step_index: int = 0,
        semantic_summary: str = "",
    ) -> str:
        """Store bytes + metadata. ``hint`` is a **narrow-only** classification hint.
        Returns the content hash. A coined hint raises :class:`InvalidTagError`."""
        from ._proto import cambrian_pb2

        req = cambrian_pb2.UploadArtifactRequest(
            content=content,
            content_type=content_type,
            session_id=session_id,
            step_index=step_index,
            tags=list(hint or []),  # narrow-only hint
            semantic_summary=semantic_summary,
        )
        try:
            resp = self._stub.UploadArtifact(req)
        except Exception as exc:
            if _is_invalid_argument(exc):
                raise InvalidTagError(str(exc)) from exc
            raise
        return resp.hash

    def get(self, artifact_hash: str) -> bytes:
        """Fetch artifact bytes. Carries no scope tags — a scope-denied or missing
        artifact raises :class:`ArtifactNotFound` (absence, no existence leak)."""
        from ._proto import cambrian_pb2

        resp = self._stub.GetArtifact(cambrian_pb2.GetArtifactRequest(hash=artifact_hash))
        if not resp.found:
            raise ArtifactNotFound(artifact_hash)
        return bytes(resp.content)

    def list_from_step(self, session_id: str, step_index: int) -> list:
        """List prior-step artifacts readable under the agent's scope. No scope tags sent."""
        from ._proto import cambrian_pb2

        resp = self._stub.ListStepArtifacts(
            cambrian_pb2.ListStepArtifactsRequest(session_id=session_id, step_index=step_index)
        )
        return list(resp.artifacts)


class WorkingMemory:
    """Read-only Global Workspace view (ADR-0022), distinct from ``memory.recall()``.

    ``working_memory`` is *given* (assembled by the Substrate and delivered on the
    task); ``recall()`` is *initiated* by the agent. This class exposes only
    ``assemble()`` — there is deliberately no write method."""

    def __init__(self, refs, fetch_fn=None) -> None:
        self._refs = list(refs or [])
        self._fetch_fn = fetch_fn

    def assemble(self, min_precision: float = 0.5, max_tokens: int = 800) -> str:
        from . import assemble_context

        return assemble_context(
            self._refs,
            min_precision=min_precision,
            max_tokens=max_tokens,
            fetch_fn=self._fetch_fn,
        )


def _secs(timeout_ms: int):
    return (timeout_ms / 1000.0) if timeout_ms and timeout_ms > 0 else None
