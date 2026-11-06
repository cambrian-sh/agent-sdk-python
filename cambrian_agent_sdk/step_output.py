"""Agent-pushed live step output (ADR-0139 amendment 2026-11-06).

The kernel keeps ONE live output stream per step. When the agent's model call goes
through the kernel's gateway (the SDK ReAct loop: ``generate_with_tools`` /
``generate_stream``), the kernel forwards it itself and the agent does nothing. An
agent that calls its OWN model — Claude Code or the Claude Agent SDK running as a
worker, LangGraph, a hand-written agent — pushes its output instead, over the
agent-plane ``EmitStepOutput`` RPC. The kernel delivers it to the operator feed and to
MCP callers waiting on the task (``notifications/progress``).

Such an agent should declare ``streams_output: true`` (profile) or
``"streams_output": true`` (``AGENT_MANIFEST``), so the kernel stops forwarding
gateway deltas for its steps and one step never streams twice.

Rules the kernel enforces: every chunk names the step's lease, and the lease must be
live and issued to this agent (else ``PERMISSION_DENIED``); ``seq`` is strictly
increasing per lease, and a chunk at or below the highest accepted seq is dropped as a
duplicate, so re-sending after a broken stream is safe; chunks after ``final`` are
dropped; sizes are capped per chunk and per step.

Live output is best-effort and never fails the step: these helpers log and record a
failure (``StepOutputWriter.error``), they do not raise it.

Example — an external agent adapter streaming its own model's tokens::

    from cambrian_agent_sdk import SubstrateClient

    def run_step(substrate: SubstrateClient, lease_id: str, prompt: str) -> str:
        parts = []
        with substrate.step_output(lease_id) as out:      # one client stream
            out.progress("calling the model")
            for token in my_model.stream(prompt):        # the agent's own model
                parts.append(token)
                out.text(token)                          # live, to the operator/MCP
        return "".join(parts)                            # the step's final output

    # Or, with everything already in hand:
    substrate.emit_step_output(lease_id, ["partial ", "answer"], final=True)
"""

from __future__ import annotations

import logging
import queue
import threading
from typing import Any, Dict, Iterable, Iterator, Optional, Union

log = logging.getLogger("cambrian_agent_sdk.step_output")

#: The chunk kinds (cambrian.proto StepOutputKind).
KINDS = ("text", "progress", "status")

_KIND_ENUM = {
    "text": "STEP_OUTPUT_KIND_TEXT",
    "progress": "STEP_OUTPUT_KIND_PROGRESS",
    "status": "STEP_OUTPUT_KIND_STATUS",
}

ChunkLike = Union[str, Dict[str, Any]]


class _SeqCounter:
    """Per-lease sequence numbers, shared by every writer of one client, so two
    pushes for one lease never reuse a seq (the kernel would drop the second)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._last: Dict[str, int] = {}

    def next(self, lease_id: str) -> int:
        with self._lock:
            n = self._last.get(lease_id, 0) + 1
            self._last[lease_id] = n
            return n

    def observe(self, lease_id: str, seq: int) -> None:
        with self._lock:
            if seq > self._last.get(lease_id, 0):
                self._last[lease_id] = seq


def _pb_chunk(pb, lease_id: str, seq: int, kind: str, data: str, final: bool):
    if kind not in _KIND_ENUM:
        raise ValueError(f"step output kind must be one of {KINDS}, got {kind!r}")
    return pb.StepOutputChunk(lease_id=lease_id, seq=seq,
                              kind=getattr(pb, _KIND_ENUM[kind]), data=data or "", final=final)


def _ack_dict(ack) -> Dict[str, int]:
    return {"accepted": ack.accepted, "duplicates": ack.duplicates, "over_cap": ack.over_cap,
            "after_final": ack.after_final, "last_seq": ack.last_seq}


def _failure(exc: BaseException) -> Optional[Dict[str, str]]:
    """None for UNIMPLEMENTED (a kernel without the surface); else an error dict."""
    code = getattr(exc, "code", None)
    code = code() if callable(code) else None
    name = getattr(code, "name", "") or ""
    if name == "UNIMPLEMENTED":
        return None
    details = getattr(exc, "details", None)
    detail = details() if callable(details) else str(exc)
    return {"error": detail or str(exc), "code": name or "UNKNOWN"}


class StepOutputWriter:
    """A context manager holding one ``EmitStepOutput`` client stream open.

    ``text()`` / ``progress()`` / ``status()`` queue a chunk and return at once; one
    background thread (per writer, not per chunk) feeds the stream. Leaving the
    ``with`` block sends the final marker, closes the stream and stores the kernel's
    acknowledgement in :attr:`ack` (``None`` when the kernel does not serve the RPC) or
    the failure in :attr:`error`. Never raises for an RPC failure."""

    _CLOSE = object()

    def __init__(self, stub, md, lease_id: str, seqs: _SeqCounter, timeout_ms: int = 0) -> None:
        if not lease_id:
            raise ValueError("step output needs the step's lease id")
        from ._proto import cambrian_pb2
        self._pb = cambrian_pb2
        self._stub = stub
        self._md = md
        self._lease = lease_id
        self._seqs = seqs
        self._timeout = (timeout_ms / 1000.0) if timeout_ms > 0 else None
        self._q: "queue.Queue[Any]" = queue.Queue()
        self._thread: Optional[threading.Thread] = None
        self._closed = False
        self._drained = False  # the stream consumed the close marker
        self.ack: Optional[Dict[str, int]] = None
        self.error: Optional[Dict[str, str]] = None

    # ── producer side ────────────────────────────────────────────────────────
    def text(self, data: str) -> None:
        self._put("text", data, False)

    def progress(self, data: str) -> None:
        self._put("progress", data, False)

    def status(self, data: str) -> None:
        self._put("status", data, False)

    def _put(self, kind: str, data: str, final: bool) -> None:
        if self._closed:
            raise RuntimeError("step output writer is closed")
        if not data and not final:
            return
        self._q.put(_pb_chunk(self._pb, self._lease, self._seqs.next(self._lease), kind, data, final))

    # ── lifecycle ────────────────────────────────────────────────────────────
    def open(self) -> "StepOutputWriter":
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, name="cambrian-step-output", daemon=True)
            self._thread.start()
        return self

    def close(self, final: bool = True) -> Optional[Dict[str, int]]:
        if self._closed:
            return self.ack
        if final:
            self._put("text", "", True)
        self._closed = True
        self._q.put(self._CLOSE)
        if self._thread is not None:
            self._thread.join()
        return self.ack

    def __enter__(self) -> "StepOutputWriter":
        return self.open()

    def __exit__(self, *exc: Any) -> None:
        self.close(final=True)

    def _chunks(self) -> Iterator[Any]:
        while True:
            item = self._q.get()
            if item is self._CLOSE:
                self._drained = True
                return
            yield item

    def _run(self) -> None:
        try:
            ack = self._stub.EmitStepOutput(self._chunks(), timeout=self._timeout, metadata=self._md)
            self.ack = _ack_dict(ack)
            self._seqs.observe(self._lease, ack.last_seq)
        except Exception as exc:  # noqa: BLE001 — live output never fails the step
            self.error = _failure(exc)
            if self.error is not None:
                log.warning("step output push failed (the step's final output is unaffected): %s",
                            self.error)
            # Consume up to the close marker (unless the stream already did), so
            # close() returns after a failure too.
            while not self._drained and self._q.get() is not self._CLOSE:
                pass


def emit_step_output(stub, md, seqs: _SeqCounter, lease_id: str, chunks: Iterable[ChunkLike], *,
                     final: bool = False, timeout_ms: int = 0) -> Optional[Dict[str, Any]]:
    """Push ``chunks`` as one stream. A chunk is a string (text) or a dict with
    ``data`` and optional ``kind`` / ``final`` / ``seq``. ``final=True`` appends the
    final marker. Returns the ack dict, None for a kernel without the RPC, or an
    error dict."""
    if not lease_id:
        raise ValueError("step output needs the step's lease id")
    from ._proto import cambrian_pb2 as pb
    msgs = []
    for c in chunks:
        if isinstance(c, str):
            c = {"data": c}
        seq = int(c.get("seq") or 0) or seqs.next(lease_id)
        seqs.observe(lease_id, seq)
        msgs.append(_pb_chunk(pb, lease_id, seq, c.get("kind", "text"), c.get("data", ""),
                              bool(c.get("final", False))))
    if final:
        msgs.append(_pb_chunk(pb, lease_id, seqs.next(lease_id), "text", "", True))
    if not msgs:
        return {"accepted": 0, "duplicates": 0, "over_cap": 0, "after_final": 0, "last_seq": 0}
    try:
        ack = stub.EmitStepOutput(iter(msgs), timeout=(timeout_ms / 1000.0) if timeout_ms > 0 else None,
                                  metadata=md)
    except Exception as exc:  # noqa: BLE001 — live output never fails the step
        err = _failure(exc)
        if err is not None:
            log.warning("step output push failed (the step's final output is unaffected): %s", err)
        return err
    return _ack_dict(ack)
