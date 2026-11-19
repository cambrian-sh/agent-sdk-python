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

import time
import warnings
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
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


#: The ingest states the kernel reports (contract 0111, ENG-2143).
INGEST_STATES = ("accepted", "embedding", "searchable", "failed")


@dataclass(frozen=True)
class IngestStatus:
    """Where an ingested document stands (contract 0111, ENG-2143).

    A large document (at least the kernel's
    ``execution.ingestion.background_embed_min_chunks`` chunks, default 256) is
    ACCEPTED at once and embedded in the background: ``accepted`` →
    ``embedding`` (``chunks_embedded`` of ``chunks_total``) → ``searchable``, or
    ``failed`` with ``error``. A smaller document is ``searchable`` when
    ``remember()`` returns."""

    document_id: str
    state: str
    chunks_total: int = 0
    chunks_embedded: int = 0
    error: str = ""

    @property
    def searchable(self) -> bool:
        return self.state == "searchable"

    @property
    def failed(self) -> bool:
        return self.state == "failed"

    @property
    def terminal(self) -> bool:
        """True when the state no longer changes on its own."""
        return self.searchable or self.failed

    @classmethod
    def _from_pb(cls, st) -> "IngestStatus":
        return cls(document_id=st.document_id, state=st.state, chunks_total=st.chunks_total,
                   chunks_embedded=st.chunks_embedded, error=st.error)


class RememberResult(str):
    """What ``remember()`` returns: the source-doc entity id, as before (it IS that
    ``str``, so existing callers keep working), plus the stored document's
    ``document_id`` and its ingest ``status`` (:class:`IngestStatus`). Against a
    kernel older than contract 0111 ``document_id`` is ``""`` and ``status`` is
    ``None``."""

    document_id: str
    status: Optional[IngestStatus]

    def __new__(cls, doc_id: str, document_id: str = "", status: Optional[IngestStatus] = None):
        obj = super().__new__(cls, doc_id)
        obj.document_id = document_id
        obj.status = status
        return obj

    @property
    def doc_id(self) -> str:
        return str(self)


class IngestFailedError(Exception):
    """The kernel gave up embedding the document (state ``failed``); ``status``
    carries the reason in ``status.error``."""

    def __init__(self, status: IngestStatus) -> None:
        super().__init__(f"ingest of {status.document_id} failed: {status.error or 'no reason given'}")
        self.status = status


class IngestStatusNotFound(LookupError):
    """No document of this agent carries the id (the kernel answers NOT_FOUND for an
    unknown id and for another writer's document alike)."""


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


#: The lease of the task this thread/context is running (ADR-0130 / ENG-518). Set by
#: the runtime around ``agent.run(task)`` (:func:`task_lease`), read by the memory
#: reads when the caller passed no ``session_token_id``. A kernel on an authenticated
#: deployment refuses a lease-less agent-plane memory read (``no_lease``, owner
#: decision 2026-11-14), and an author's ``self.memory.recall(q)`` passes none — so
#: the SDK presents the lease the kernel dispatched this task under. Never a value an
#: author or the model chooses: it is the lease the kernel handed in with the task.
_current_lease: "ContextVar[str]" = ContextVar("cambrian_current_lease", default="")


@contextmanager
def task_lease(lease_id: str):
    """Run a block with ``lease_id`` as the current task's lease (see _current_lease)."""
    token = _current_lease.set(lease_id or "")
    try:
        yield
    finally:
        _current_lease.reset(token)


def _lease_or_current(session_token_id: str) -> str:
    return session_token_id or _current_lease.get()


def _lease_kwargs(session_token_id: str) -> dict:
    """``{"metadata": <lease headers>}`` for a call made under the task's lease, or
    ``{}`` when there is none (so a stub that takes no metadata keeps working)."""
    md = _lease_metadata(_lease_or_current(session_token_id))
    return {"metadata": md} if md else {}


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
        md = md + _lease_metadata(_lease_or_current(session_token_id))
        resp = self._stub.QueryMemory(req, timeout=_secs(timeout_ms), metadata=md)
        return [_result_dict(r) for r in resp.results]

    def recall_actions(self, query: str, top_k: int = 5, timeout_ms: int = 0,
                       session_token_id: str = "") -> List[dict]:
        """**Deprecated; always returns an empty list.** The kernel removed the
        ADR-0049 D4 actions lane (ENG-1645): the action records it read are no longer
        written since the owner decision of 2026-11-14 (ENG-1595), pending the redesign
        of agent-experience memory. Like :meth:`recall_precedents`, this shim does NOT
        call the kernel — a current kernel serves ``x-lane=actions`` from the fact lane,
        which would hand the caller facts mislabelled as its own actions. Use
        :meth:`recall` for facts.

        Kept (it is published API surface) so existing callers keep working, and emits
        a :class:`DeprecationWarning` on call. The arguments are accepted and ignored."""
        warnings.warn(
            "MemoryClient.recall_actions is deprecated: the kernel removed the actions "
            "lane (ENG-1645). It returns [] without calling the kernel. Use recall() "
            "for facts.",
            DeprecationWarning,
            stacklevel=2,
        )
        return []

    def recall_entity(self, kind_id: str, timeout_ms: int = 0,
                      session_token_id: str = "") -> List[dict]:
        """**Deprecated.** The kernel REMOVED the entity lane on 2026-08-14 (the
        world-model entity records are gone), and now routes ``x-lane=entity`` to the
        default FACT lane. This call therefore no longer resolves an entity's
        reconstructed state: it is a semantic fact search whose query happens to be a
        ``kind:id`` string, which almost never retrieves anything useful. Use
        :meth:`recall` for facts.

        Kept (it is published API surface) with unchanged wire behaviour — it still
        sends ``x-lane=entity``, because an OLDER kernel that still serves the lane must
        keep answering it — and emits a :class:`DeprecationWarning` on call.

        Historically (ADR-0049 Issue 012): an EXACT lookup by canonical ``kind:id`` —
        e.g. ``file:c:/repo/a.md`` — returning the materialized field-LWW view (a
        deleted file read ``exists=false``)."""
        from ._proto import cambrian_pb2

        warnings.warn(
            "MemoryClient.recall_entity is deprecated: the kernel removed the entity "
            "lane on 2026-08-14 and now serves x-lane=entity from the default fact "
            "lane. Use recall() for facts.",
            DeprecationWarning,
            stacklevel=2,
        )
        req = cambrian_pb2.MemoryRequest(query=kind_id, top_k=1)
        md = self._identity_metadata() + [("x-lane", "entity")]
        md = md + _lease_metadata(_lease_or_current(session_token_id))
        resp = self._stub.QueryMemory(req, timeout=_secs(timeout_ms), metadata=md)
        return [_result_dict(r) for r in resp.results]

    def recall_precedents(self, query: str, top_k: int = 5, timeout_ms: int = 0,
                          session_token_id: str = "") -> List[dict]:
        """**Deprecated; always returns an empty list.** Precedents were removed on
        2026-11-05 (owner decision ``decisions/scenes-cancelled.md``): the kernel no
        longer has a precedent lane. Unlike :meth:`recall_entity`, this shim does NOT
        call the kernel — a current kernel would serve the request from the fact lane,
        handing the caller facts mislabelled as prior transitions. Use :meth:`recall`
        for facts.

        Kept (it is published API surface) so existing callers keep working, and
        emits a :class:`DeprecationWarning` on call. The arguments are accepted and
        ignored."""
        warnings.warn(
            "MemoryClient.recall_precedents is deprecated: precedents were removed "
            "2026-11-05; see decisions/scenes-cancelled.md. It returns [] without "
            "calling the kernel. Use recall() for facts.",
            DeprecationWarning,
            stacklevel=2,
        )
        return []

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
    ) -> "RememberResult":
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
        INVALID_ARGUMENT mapping).

        Returns a :class:`RememberResult`: the source-doc entity id (a ``str``, as
        before) with ``.document_id`` and ``.status`` (:class:`IngestStatus`). A large
        document comes back ``accepted`` and is embedded in the background; a caller
        that must read its own write waits with :meth:`wait_until_searchable`
        (contract 0111, ENG-2143)."""
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
        status = IngestStatus._from_pb(resp.status) if resp.HasField("status") else None
        return RememberResult(resp.doc_id, document_id=resp.document_id, status=status)

    def ingest_status(self, document_id) -> IngestStatus:
        """Read where an ingested document stands (``GetIngestStatus``, contract 0111).
        ``document_id`` is :attr:`RememberResult.document_id` (a :class:`RememberResult`
        itself is accepted). Only the document's writer may read it: another agent's
        document, like an unknown id, raises :class:`IngestStatusNotFound`."""
        from ._proto import cambrian_pb2

        doc = _ingest_document_id(document_id)
        try:
            st = self._stub.GetIngestStatus(cambrian_pb2.GetIngestStatusRequest(document_id=doc),
                                            metadata=self._identity_metadata())
        except Exception as exc:
            if _is_not_found(exc):
                raise IngestStatusNotFound(doc) from exc
            raise
        return IngestStatus._from_pb(st)

    def wait_until_searchable(self, document_id, timeout: float = 120.0,
                              poll_interval: float = 0.5) -> IngestStatus:
        """Block until the document is searchable — read-your-writes after a large
        ``remember()`` that came back ``accepted`` (contract 0111, ENG-2143).

        Polls :meth:`ingest_status` every ``poll_interval`` seconds. Returns the
        searchable status at once for a document that already is (every small
        ingest). Raises :class:`IngestFailedError` when the kernel reports
        ``failed``, :class:`TimeoutError` (carrying the last status) when ``timeout``
        seconds pass first, and :class:`IngestStatusNotFound` for an unknown id."""
        doc = _ingest_document_id(document_id)
        if timeout <= 0:
            raise ValueError("timeout must be > 0 seconds")
        deadline = _monotonic() + timeout
        while True:
            st = self.ingest_status(doc)
            if st.searchable:
                return st
            if st.failed:
                raise IngestFailedError(st)
            left = deadline - _monotonic()
            if left <= 0:
                err = TimeoutError(
                    f"{doc} not searchable after {timeout:g}s: {st.state} "
                    f"({st.chunks_embedded}/{st.chunks_total} chunks embedded)")
                err.status = st  # type: ignore[attr-defined]
                raise err
            _sleep(min(max(poll_interval, 0.01), left))

    def _identity_metadata(self):
        """gRPC metadata carrying the agent principal. The server derives the
        scope principal from x-agent-id and fail-closes on an empty one, so a
        memory call without it is denied (ADR-0034)."""
        return [("x-agent-id", self._agent_id)] if self._agent_id else []


class ArtifactManager:
    """``artifacts`` — save (narrow-only hint) / get / list_from_step. No scope tags.

    Every call presents the running task's lease (``x-lease-id``; an explicit
    ``session_token_id`` wins, outside a task nothing is sent), as memory reads do:
    the kernel files an upload in that lease's run, session and step, and resolves
    the caller's conversation owner for reads from it
    (T-20261118-agent-artifacts-from-planner-lane-steps-carry). The request's own
    ``session_id`` / ``step_index`` are never trusted for provenance."""

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
        session_token_id: str = "",
    ) -> str:
        """Store bytes + metadata. ``hint`` is a **narrow-only** classification hint.
        Returns the artifact reference (a ledger id). A coined hint raises
        :class:`InvalidTagError`."""
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
            resp = self._stub.UploadArtifact(req, **_lease_kwargs(session_token_id))
        except Exception as exc:
            if _is_invalid_argument(exc):
                raise InvalidTagError(str(exc)) from exc
            raise
        return resp.hash

    def get(self, artifact_hash: str, session_token_id: str = "") -> bytes:
        """Fetch artifact bytes. Carries no scope tags — a scope-denied or missing
        artifact raises :class:`ArtifactNotFound` (absence, no existence leak)."""
        from ._proto import cambrian_pb2

        resp = self._stub.GetArtifact(cambrian_pb2.GetArtifactRequest(hash=artifact_hash),
                                      **_lease_kwargs(session_token_id))
        if not resp.found:
            raise ArtifactNotFound(artifact_hash)
        return bytes(resp.content)

    def list_from_step(self, session_id: str, step_index: int, session_token_id: str = "") -> list:
        """List prior-step artifacts readable under the agent's scope. No scope tags sent."""
        from ._proto import cambrian_pb2

        resp = self._stub.ListStepArtifacts(
            cambrian_pb2.ListStepArtifactsRequest(session_id=session_id, step_index=step_index),
            **_lease_kwargs(session_token_id),
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


# Indirections so tests drive wait_until_searchable without real sleeps.
_monotonic = time.monotonic
_sleep = time.sleep


def _ingest_document_id(document_id) -> str:
    """The stored document id to read a status for: a RememberResult's document_id
    (its str value is the entity id, never the document id) or the id itself."""
    if isinstance(document_id, RememberResult):
        doc = document_id.document_id
        if not doc:
            raise ValueError("this remember() result carries no document_id "
                             "(the kernel predates contract 0111: no ingest status)")
        return doc
    if not document_id:
        raise ValueError("document_id is required")
    return str(document_id)


def _is_not_found(exc) -> bool:
    import grpc

    return isinstance(exc, grpc.RpcError) and exc.code() == grpc.StatusCode.NOT_FOUND
