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

import warnings
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Sequence, Tuple

#: ADR-0134 time vocabulary (kernel ``domain/event_time.go``, contract 0099). How
#: precise a stated time is; a date-only source is ``"day"``, never an exact instant.
TIME_PRECISIONS = ("year", "month", "day", "hour", "minute", "second")
#: Where a hit's ``event_time`` came from. ``ingest_default`` means nothing was stated:
#: the time is when the kernel recorded it. ``clamped_future`` means the stated time
#: was later than the kernel's clock (plus tolerance) and was clamped to the ingest time.
TIME_SOURCES = ("explicit", "source_metadata", "inferred", "ingest_default", "clamped_future")

_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)


def _to_timestamp(name: str, value: datetime):
    """An aware ``datetime`` as a protobuf ``Timestamp`` (an absolute UTC instant).

    A naive datetime is refused (ADR-0134 owner decision Q3): it has no zone, so the
    SDK cannot know which instant it means, and guessing would silently shift every
    event by the caller's UTC offset.
    """
    from google.protobuf.timestamp_pb2 import Timestamp

    if not isinstance(value, datetime):
        raise TypeError(f"{name} must be a datetime, got {type(value).__name__}")
    if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
        raise ValueError(
            f"{name} is a naive datetime (no timezone). Pass a timezone-aware datetime, "
            f"e.g. datetime(2024, 5, 1, 9, 30, tzinfo=timezone.utc) or "
            f"dt.replace(tzinfo=ZoneInfo('Europe/Istanbul')); the SDK does not guess a zone."
        )
    delta = value.astimezone(timezone.utc) - _EPOCH
    return Timestamp(seconds=delta.days * 86400 + delta.seconds, nanos=delta.microseconds * 1000)


def _from_timestamp(msg, field: str) -> Optional[datetime]:
    """A set ``Timestamp`` field as an aware UTC ``datetime``; ``None`` when unset."""
    if not msg.HasField(field):
        return None
    ts = getattr(msg, field)
    return _EPOCH + timedelta(seconds=ts.seconds, microseconds=ts.nanos // 1000)


def _check_precision(time_precision: str) -> None:
    if time_precision and time_precision not in TIME_PRECISIONS:
        raise ValueError(f"time_precision must be one of {TIME_PRECISIONS}, got {time_precision!r}")


def _result_dict(r) -> dict:
    """One ``MemoryResult`` as the dict every recall returns.

    ``event_time`` / ``event_time_end`` are aware UTC datetimes (``None`` when the
    kernel sent none); ``time_source`` and ``time_precision`` are strings from
    :data:`TIME_SOURCES` / :data:`TIME_PRECISIONS` (``""`` from an older kernel).
    """
    return {
        "text": r.text,
        "score": r.score,
        "metadata": r.metadata,
        "event_time": _from_timestamp(r, "event_time"),
        "event_time_end": _from_timestamp(r, "event_time_end"),
        "time_source": r.time_source,
        "time_precision": r.time_precision,
    }


class InvalidTagError(Exception):
    """A narrow-only hint named a tag outside the controlled vocabulary (coinage)."""


class ArtifactNotFound(Exception):
    """The artifact does not exist or is not readable under the agent's scope."""


class SelfDelegationError(Exception):
    """An agent attempted to delegate a task back to itself (recursion guard)."""


def _lease_metadata(lease_id: str) -> list:
    """gRPC identity headers for a call made under a per-step BudgetLease.

    ``x-lease-id`` is the header a current kernel resolves. It looks the lease up in its
    own registry and derives the task session SERVER-SIDE, so an agent never names a
    session it was not dispatched under — the agent only ever presents the opaque
    credential it was handed.

    ``x-session-id`` is still sent, with the same value, for one release: an OLDER kernel
    reads only that header, and a CURRENT kernel tries lease resolution on it before
    falling back to treating it as a session ID. Sending both is therefore correct against
    either side. It is deprecated and goes away once the floor kernel understands
    ``x-lease-id``.

    Historically this value was sent ONLY as ``x-session-id``, where the kernel read a
    per-step lease as if it were a durable task session — the conflation that silently
    disabled caller-scope narrowing, session-scoped node ownership, and the same-session
    step-record filter.
    """
    if not lease_id:
        return []
    return [("x-lease-id", lease_id), ("x-session-id", lease_id)]


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
               session_token_id: str = "", *,
               reference_time: Optional[datetime] = None,
               event_after: Optional[datetime] = None,
               event_before: Optional[datetime] = None) -> List[dict]:
        """Retrieve from LTM. Results are already scope-filtered server-side; this
        request carries **no** scope params (D5/ADR-0034).

        ``session_token_id`` is the per-step BudgetLease, sent via
        :func:`_lease_metadata`. The kernel resolves it to this run's task session and
        applies the same-session step-record filter (ADR-0048 D1): without it the server
        sees no session and D1 silently no-ops, so the agent's OWN step output is recalled
        straight back into the same run — the context feedback loop D1 exists to break.

        Event time (ADR-0134 D4, contract 0099), all optional and timezone-aware only
        (a naive datetime raises :class:`ValueError`):

        - ``reference_time`` is the asker's "now" ("last month" is relative to it).
          Unset means the kernel's clock. Use it to replay history as of a past date.
        - ``event_after`` (inclusive) / ``event_before`` (exclusive) restrict hits to
          those whose event span overlaps ``[event_after, event_before)``. A kernel that
          has not wired the window yet answers ``UNIMPLEMENTED`` rather than ignoring it.

        Each result dict carries ``text``, ``score``, ``metadata`` and the hit's
        ``event_time`` / ``event_time_end`` (aware UTC datetimes or ``None``),
        ``time_source`` (:data:`TIME_SOURCES`) and ``time_precision``
        (:data:`TIME_PRECISIONS`)."""
        from ._proto import cambrian_pb2

        req = cambrian_pb2.MemoryRequest(query=query, top_k=top_k)
        if reference_time is not None:
            req.reference_time.CopyFrom(_to_timestamp("reference_time", reference_time))
        if event_after is not None:
            req.event_after.CopyFrom(_to_timestamp("event_after", event_after))
        if event_before is not None:
            req.event_before.CopyFrom(_to_timestamp("event_before", event_before))
        if event_after is not None and event_before is not None and event_after >= event_before:
            raise ValueError("event_after must be earlier than event_before (the window is [after, before))")
        md = self._identity_metadata()
        md = md + _lease_metadata(session_token_id)
        resp = self._stub.QueryMemory(req, timeout=_secs(timeout_ms), metadata=md)
        return [_result_dict(r) for r in resp.results]

    def recall_actions(self, query: str, top_k: int = 5, timeout_ms: int = 0,
                       session_token_id: str = "") -> List[dict]:
        """Retrieve from the ACTIONS lane (ADR-0049 D4) — "what did I do" (action
        records), a SEPARATE intent from :meth:`recall` ("what do I know") so action
        breadcrumbs never re-bloat fact grounding. Routed by the ``x-lane=actions``
        gRPC metadata; same server-side scope/relevance gating as fact recall."""
        from ._proto import cambrian_pb2

        req = cambrian_pb2.MemoryRequest(query=query, top_k=top_k)
        md = self._identity_metadata() + [("x-lane", "actions")]
        md = md + _lease_metadata(session_token_id)
        resp = self._stub.QueryMemory(req, timeout=_secs(timeout_ms), metadata=md)
        return [_result_dict(r) for r in resp.results]

    def recall_entity(self, kind_id: str, timeout_ms: int = 0,
                      session_token_id: str = "") -> List[dict]:
        """**Deprecated.** The kernel REMOVED the entity lane on 2026-08-14 (the
        world-model entity records are gone), and now routes ``x-lane=entity`` to the
        default FACT lane. This call therefore no longer resolves an entity's
        reconstructed state: it is a semantic fact search whose query happens to be a
        ``kind:id`` string, which almost never retrieves anything useful. Use
        :meth:`recall` for facts or :meth:`recall_precedents` for prior transitions.

        Kept (it is published API surface) with unchanged wire behaviour — it still
        sends ``x-lane=entity``, because an OLDER kernel that still serves the lane must
        keep answering it — and emits a :class:`DeprecationWarning` on call.

        Historically (ADR-0049 Issue 012): an EXACT lookup by canonical ``kind:id`` —
        e.g. ``file:c:/repo/a.md`` — returning the materialized field-LWW view (a
        deleted file read ``exists=false``) plus the link to its most-recent engaging
        scene."""
        from ._proto import cambrian_pb2

        warnings.warn(
            "MemoryClient.recall_entity is deprecated: the kernel removed the entity "
            "lane on 2026-08-14 and now serves x-lane=entity from the default fact "
            "lane. Use recall() for facts or recall_precedents() for prior transitions.",
            DeprecationWarning,
            stacklevel=2,
        )
        req = cambrian_pb2.MemoryRequest(query=kind_id, top_k=1)
        md = self._identity_metadata() + [("x-lane", "entity")]
        md = md + _lease_metadata(session_token_id)
        resp = self._stub.QueryMemory(req, timeout=_secs(timeout_ms), metadata=md)
        return [_result_dict(r) for r in resp.results]

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
        md = md + _lease_metadata(session_token_id)
        resp = self._stub.QueryMemory(req, timeout=_secs(timeout_ms), metadata=md)
        return [_result_dict(r) for r in resp.results]

    def remember(
        self,
        text: str,
        hint: Optional[List[str]] = None,
        source: str = "",
        session_id: str = "",
        importance: float = 0.0,
        *,
        occurred_at: Optional[datetime] = None,
        parts: Optional[Sequence[Tuple]] = None,
        time_precision: str = "",
        source_kind: str = "",
        title: str = "",
        author: str = "",
    ) -> str:
        """Persist a synthesized insight. ``hint`` is a **narrow-only** classification
        hint (never authoritative tags); the kernel derives the real classification.
        Returns the new doc ID. A coined hint raises :class:`InvalidTagError`.

        Event time (ADR-0134 D1, contract 0099), all optional:

        - ``occurred_at`` is when the content HAPPENED, as opposed to when it is
          stored. It must be timezone-aware; a naive datetime raises
          :class:`ValueError`. Unset means unknown: the kernel dates it by its ingest
          clock and labels it ``ingest_default``, never as a real time. A time beyond
          the kernel's clock (plus a tolerance, default 5 minutes) is clamped to the
          ingest time and labelled ``clamped_future``.
        - ``parts`` is multi-message content, e.g. a chat's turns, as
          ``[(text, occurred_at_or_None), ...]``. It replaces ``text`` (pass
          ``text=""``); the kernel joins the parts with a blank line. The document's
          time is the earliest part time unless ``occurred_at`` is given.
        - ``time_precision`` says how precise those times are, one of
          :data:`TIME_PRECISIONS` (``"day"`` for a date-only source). Empty means
          ``"second"``.

        Source metadata (ADR-0136 A3), all optional and DESCRIPTIVE only:

        - ``source_kind`` is free text describing the source (``"conversation"``,
          ``"email"``, ``"report"``, ...). It is shown to the kernel's fact extractor as
          data; nothing branches on it, and it never changes how the kernel parses.
        - ``title`` is the source's own title (unset: the body's first line).
        - ``author`` is the source's author as the source names it. It never changes
          who the writer is: the kernel stamps the authenticated agent as the writer.
        - A part may carry a speaker as a third element:
          ``[(text, occurred_at_or_None, speaker), ...]``. The speaker never changes the
          stored text; each chunk records the speaker at its start.

        The kernel trims each value, cuts it to a configured length, and rejects one
        containing a control character (raised as :class:`InvalidTagError`, the SDK's
        INVALID_ARGUMENT mapping)."""
        from ._proto import cambrian_pb2

        _check_precision(time_precision)
        if parts and text:
            raise ValueError("text and parts are mutually exclusive: pass text='' with parts")
        req = cambrian_pb2.IngestMemoryRequest(
            text=text,
            tags=list(hint or []),  # narrow-only hint
            importance=importance,
            source=source,
            session_id=session_id,
            time_precision=time_precision,
            source_kind=source_kind,
            title=title,
            author=author,
        )
        if occurred_at is not None:
            req.occurred_at.CopyFrom(_to_timestamp("occurred_at", occurred_at))
        for i, p in enumerate(parts or ()):
            if len(p) not in (2, 3):
                raise ValueError(f"parts[{i}] must be (text, occurred_at) or (text, occurred_at, speaker)")
            part_text, part_time = p[0], p[1]
            speaker = p[2] if len(p) == 3 else ""
            part = req.parts.add(text=part_text, speaker=speaker or "")
            if part_time is not None:
                part.occurred_at.CopyFrom(_to_timestamp(f"parts[{i}].occurred_at", part_time))
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
