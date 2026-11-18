"""Agent-side run trace (ADR-0145 D10, the ADR-0139 C6 run event log).

When the kernel runs with ``execution.trace.enabled`` it spawns agents with
``CAMBRIAN_TRACE_DIR`` (and ``CAMBRIAN_TRACE_MAX_CHARS``). The ReAct loop then
records, as OpenTelemetry-GenAI-shaped spans in one JSONL file per agent process:

- every model call: the system instructions, the messages and the tools offered
  exactly as sent, and the response (text, tool calls, stop reason, usage);
- every tool call: the arguments and the result the loop received.

Each call is also announced when it BEGINS by a start event: a line with the
same ``span_id`` as the finished span, the request (system instructions,
messages, tools offered; tool name and arguments), ``"cambrian.phase": "start"``
and ``end_time_unix_nano`` 0. The file is appended and closed per line, so a
follower (cambrian-benchmarks' ``usecases.watch``) sees a call while it runs. A
reader that wants finished spans only skips the start events.

The kernel writes its own file to the same directory (model calls as the
gateway saw them, kernel-executed tools, step outputs). The two join on
``cambrian.lease.id`` / ``cambrian.step.task_id``. Names follow
github.com/open-telemetry/semantic-conventions-genai (client-inference.md,
execute-tool.md; checked 2026-11-08).

Tracing is best-effort: it never raises into the loop. Without the env var
every function here is a no-op.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import secrets
import threading
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

ENV_DIR = "CAMBRIAN_TRACE_DIR"
ENV_MAX_CHARS = "CAMBRIAN_TRACE_MAX_CHARS"
_DEFAULT_MAX_CHARS = 20000
# Large, mostly repeated inputs: written in full once per file, then by hash.
_DEDUPED = ("gen_ai.system_instructions", "gen_ai.tool.definitions")


#: The start event of a call carries this attribute with value PHASE_START
#: (cambrian-core domain.TraceAttrPhase / TracePhaseStart).
PHASE_ATTR = "cambrian.phase"
PHASE_START = "start"


def now_ns() -> int:
    return time.time_ns()


def new_span_id() -> str:
    return secrets.token_hex(8)


class RunTraceWriter:
    """Appends spans to ``<dir>/agent-<agent>-<stamp>-<pid>.jsonl``."""

    def __init__(self, directory: str, max_chars: int = _DEFAULT_MAX_CHARS, service: str = "cambrian-agent"):
        os.makedirs(directory, exist_ok=True)
        stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in service)
        self.path = os.path.join(directory, f"{safe}-{stamp}-{os.getpid()}.jsonl")
        self.max_chars = max_chars
        self.service = service
        self._seen: set = set()
        self._lock = threading.Lock()
        self._broken = False

    def span(self, name: str, attributes: Dict[str, Any], *, kind: str = "INTERNAL",
             start_ns: Optional[int] = None, end_ns: Optional[int] = None,
             error: Optional[str] = None, trace_key: str = "",
             span_id: str = "", phase: str = "") -> None:
        """Append one span, or with ``phase="start"`` the start event of one (end
        time 0). ``span_id`` joins a start event to its finished span; "" mints one."""
        end = end_ns or now_ns()
        attrs = dict(attributes)
        if phase:
            attrs[PHASE_ATTR] = phase
        with self._lock:
            if self._broken:
                return
            for key in _DEDUPED:
                if key not in attrs:
                    continue
                body = json.dumps(attrs[key], sort_keys=True, default=str)
                digest = hashlib.sha256(body.encode("utf-8")).hexdigest()
                attrs[key + "_sha256"] = digest
                if (key, digest) in self._seen:
                    del attrs[key]
                else:
                    self._seen.add((key, digest))
            rec = {
                "trace_id": _trace_id(trace_key),
                "span_id": span_id or new_span_id(),
                "name": name,
                "kind": kind,
                "start_time_unix_nano": start_ns or end,
                "end_time_unix_nano": 0 if phase == PHASE_START else end,
                "attributes": _bound(attrs, self.max_chars),
                "status": {"code": "ERROR", "message": error} if error else {"code": "OK"},
                "resource": {"service.name": self.service},
                "schema": "otel-semconv-genai",
            }
            try:
                with open(self.path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
            except OSError as exc:
                self._broken = True
                logger.warning("run_trace_write_failed", extra={"path": self.path, "err": str(exc)})


def _trace_id(key: str) -> str:
    if not key:
        return secrets.token_hex(16)
    return hashlib.sha256(("cambrian-lease:" + key).encode("utf-8")).hexdigest()[:32]


def _bound(v: Any, max_chars: int) -> Any:
    if max_chars <= 0:
        return v
    if isinstance(v, str):
        if len(v) <= max_chars:
            return v
        return v[:max_chars] + f"…[truncated {len(v) - max_chars} of {len(v)} chars]"
    if isinstance(v, dict):
        return {k: _bound(x, max_chars) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_bound(x, max_chars) for x in v]
    return v


_writer: Optional[RunTraceWriter] = None
_writer_lock = threading.Lock()
_resolved = False


def writer(agent_id: str = "") -> Optional[RunTraceWriter]:
    """The process's writer, opened on first use from the environment; None when off."""
    global _writer, _resolved
    if _resolved:
        return _writer
    with _writer_lock:
        if _resolved:
            return _writer
        directory = os.getenv(ENV_DIR, "")
        if directory:
            try:
                max_chars = int(os.getenv(ENV_MAX_CHARS, "") or _DEFAULT_MAX_CHARS)
            except ValueError:
                max_chars = _DEFAULT_MAX_CHARS
            try:
                _writer = RunTraceWriter(directory, max_chars, "agent-" + (agent_id or "unknown"))
            except OSError as exc:
                logger.warning("run_trace_open_failed", extra={"dir": directory, "err": str(exc)})
                _writer = None
        _resolved = True
        return _writer


def _reset_for_tests() -> None:
    global _writer, _resolved
    with _writer_lock:
        _writer, _resolved = None, False


def _task_attrs(agent, task) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    agent_id = getattr(agent, "agent_id", "") or ""
    if agent_id:
        out["gen_ai.agent.name"] = agent_id
    lease = getattr(task, "session_token_id", "") or ""
    if lease:
        out["cambrian.lease.id"] = lease
    task_id = getattr(task, "task_id", "") or ""
    if task_id:
        out["cambrian.step.task_id"] = task_id
    return out


def _composition_attrs(task) -> Dict[str, Any]:
    """The composed worker's composition (ADR-0145 D12), for start events."""
    try:
        from .composition import composition_from_task
        comp = composition_from_task(task)
    except Exception:  # noqa: BLE001 - tracing must never break the loop
        return {}
    if comp is None:
        return {}
    out: Dict[str, Any] = {"cambrian.composition.runtime": comp.runtime}
    if comp.skills:
        out["cambrian.composition.skills"] = list(comp.skills)
    if comp.professions:
        out["cambrian.composition.professions"] = list(comp.professions)
    if comp.step >= 0:
        out["cambrian.step.index"] = comp.step
    return out


def _messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """OpenAI-style chat messages → GenAI {role, parts} messages."""
    out = []
    for m in messages or []:
        role = m.get("role", "")
        parts: List[Dict[str, Any]] = []
        content = m.get("content")
        if role == "tool":
            parts.append({"type": "tool_call_response", "id": m.get("tool_call_id", ""), "response": content})
        elif content:
            parts.append({"type": "text", "content": content if isinstance(content, str) else json.dumps(content, default=str)})
        for c in m.get("tool_calls") or []:
            parts.append(_tool_call_part(c))
        out.append({"role": role, "parts": parts})
    return out


def _tool_call_part(c: Any) -> Dict[str, Any]:
    if not isinstance(c, dict):
        return {"type": "tool_call", "content": str(c)}
    fn_raw = c.get("function")
    fn: Dict[str, Any] = fn_raw if isinstance(fn_raw, dict) else {}
    return {"type": "tool_call", "id": c.get("id", ""), "name": c.get("name") or fn.get("name", ""),
            "arguments": c.get("arguments", fn.get("arguments", ""))}


def _model_request_attrs(agent, task, *, path: str, round_no: int,
                         system: Optional[List[Dict[str, Any]]], messages: Optional[List[Dict[str, Any]]],
                         prompt: Optional[str], tools: Optional[List[Dict[str, Any]]]) -> Dict[str, Any]:
    attrs: Dict[str, Any] = {
        "gen_ai.operation.name": "chat",
        "cambrian.react.path": path,
        "cambrian.react.round": round_no,
        **_task_attrs(agent, task),
    }
    if system:
        attrs["gen_ai.system_instructions"] = [{"type": "text", "content": b.get("text", "")} for b in system]
    if messages is not None:
        attrs["gen_ai.input.messages"] = _messages(messages)
    elif prompt is not None:
        attrs["gen_ai.input.messages"] = [{"role": "user", "parts": [{"type": "text", "content": prompt}]}]
    if tools:
        attrs["gen_ai.tool.definitions"] = tools
        attrs["cambrian.tools.offered"] = [
            (t.get("function") or {}).get("name") or t.get("name", "") for t in tools if isinstance(t, dict)]
    return attrs


def model_call_start(agent, task, *, path: str, round_no: int, system: Optional[List[Dict[str, Any]]] = None,
                     messages: Optional[List[Dict[str, Any]]] = None, prompt: Optional[str] = None,
                     tools: Optional[List[Dict[str, Any]]] = None, start_ns: Optional[int] = None) -> str:
    """Write the start event of a model call (the request, before the model answers).

    Returns the span id :func:`model_call` completes it with; "" when tracing is off.
    Never raises."""
    try:
        w = writer(getattr(agent, "agent_id", ""))
        if w is None:
            return ""
        sid = new_span_id()
        attrs = _model_request_attrs(agent, task, path=path, round_no=round_no, system=system,
                                     messages=messages, prompt=prompt, tools=tools)
        attrs.update(_composition_attrs(task))
        w.span("chat model", attrs, kind="CLIENT", start_ns=start_ns or now_ns(),
               trace_key=getattr(task, "session_token_id", "") or "", span_id=sid, phase=PHASE_START)
        return sid
    except Exception as exc:  # noqa: BLE001 - tracing must never break the loop
        logger.debug("run_trace_model_call_start_failed", extra={"err": str(exc)})
        return ""


def model_call(agent, task, *, path: str, round_no: int, system: Optional[List[Dict[str, Any]]] = None,
               messages: Optional[List[Dict[str, Any]]] = None, prompt: Optional[str] = None,
               tools: Optional[List[Dict[str, Any]]] = None, text: str = "",
               tool_calls: Optional[List[Any]] = None, stop: str = "", res: Any = None,
               start_ns: Optional[int] = None, error: Optional[str] = None, span_id: str = "") -> None:
    """Record one model call as a ``chat`` span (completing the start event
    ``span_id`` names, when given). Never raises."""
    try:
        w = writer(getattr(agent, "agent_id", ""))
        if w is None:
            return
        model = getattr(res, "model", "") or ""
        attrs = _model_request_attrs(agent, task, path=path, round_no=round_no, system=system,
                                     messages=messages, prompt=prompt, tools=tools)
        parts: List[Dict[str, Any]] = []
        if text:
            parts.append({"type": "text", "content": text})
        for c in tool_calls or []:
            parts.append(_tool_call_part(c))
        attrs["gen_ai.output.messages"] = [{"role": "assistant", "parts": parts, "finish_reason": stop or ""}]
        if stop:
            attrs["gen_ai.response.finish_reasons"] = [stop]
        if model:
            attrs["gen_ai.response.model"] = model
        usage = getattr(res, "usage", None)
        if usage is not None:
            attrs["gen_ai.usage.input_tokens"] = getattr(usage, "input_tokens", 0)
            attrs["gen_ai.usage.output_tokens"] = getattr(usage, "output_tokens", 0)
            attrs["gen_ai.usage.cache_read.input_tokens"] = getattr(usage, "cache_read_tokens", 0)
        w.span("chat " + (model or "model"), attrs, kind="CLIENT", start_ns=start_ns, error=error,
               trace_key=getattr(task, "session_token_id", "") or "", span_id=span_id)
    except Exception as exc:  # noqa: BLE001 - tracing must never break the loop
        logger.debug("run_trace_model_call_failed", extra={"err": str(exc)})


def context_event(agent, task, name: str, attributes: Dict[str, Any]) -> None:
    """Record one context-engine event (e.g. ``context.overflow_shrink``, ENG-2128)
    as an INTERNAL span with the step's attributes. Never raises."""
    try:
        w = writer(getattr(agent, "agent_id", ""))
        if w is None:
            return
        attrs = _task_attrs(agent, task)
        attrs.update({"cambrian.context." + k: v for k, v in attributes.items()})
        w.span(name, attrs, trace_key=getattr(task, "session_token_id", "") or "")
    except Exception as exc:  # noqa: BLE001 - tracing must never break the loop
        logger.debug("run_trace_context_event_failed", extra={"err": str(exc)})


def _executed_by(agent, name: str) -> str:
    """"agent" for a tool the agent runs in-process, else "kernel" (ExecuteTool, which
    the kernel traces too: a reader can show one side of such a call)."""
    try:
        return "agent" if name in (getattr(agent, "tools", None) or {}) else "kernel"
    except Exception:  # noqa: BLE001 - a custom tool registry without __contains__
        return "kernel"


def _tool_request_attrs(agent, task, *, name: str, args: Any, call_id: str, round_no: int,
                        executed_by: str = "") -> Dict[str, Any]:
    attrs: Dict[str, Any] = {
        "cambrian.tool.executed_by": executed_by or _executed_by(agent, name),
        "gen_ai.operation.name": "execute_tool",
        "gen_ai.tool.name": name,
        "gen_ai.tool.call.arguments": args if isinstance(args, str) else json.dumps(args, default=str, ensure_ascii=False),
        "cambrian.react.round": round_no,
        **_task_attrs(agent, task),
    }
    if call_id:
        attrs["gen_ai.tool.call.id"] = call_id
    return attrs


def tool_call_start(agent, task, *, name: str, args: Any, call_id: str = "", round_no: int = 0,
                    start_ns: Optional[int] = None, executed_by: str = "") -> str:
    """Write the start event of a tool call (name and arguments, before it runs).

    Returns the span id :func:`tool_call` completes it with; "" when tracing is off.
    Never raises."""
    try:
        w = writer(getattr(agent, "agent_id", ""))
        if w is None:
            return ""
        sid = new_span_id()
        w.span("execute_tool " + name,
               _tool_request_attrs(agent, task, name=name, args=args, call_id=call_id, round_no=round_no,
                                   executed_by=executed_by),
               start_ns=start_ns or now_ns(), trace_key=getattr(task, "session_token_id", "") or "",
               span_id=sid, phase=PHASE_START)
        return sid
    except Exception as exc:  # noqa: BLE001 - tracing must never break the loop
        logger.debug("run_trace_tool_call_start_failed", extra={"err": str(exc)})
        return ""


def tool_call(agent, task, *, name: str, args: Any, result: Any, call_id: str = "",
              round_no: int = 0, start_ns: Optional[int] = None, error: Optional[str] = None,
              span_id: str = "", executed_by: str = "") -> None:
    """Record one tool call as an ``execute_tool`` span (completing the start event
    ``span_id`` names, when given). Never raises."""
    try:
        w = writer(getattr(agent, "agent_id", ""))
        if w is None:
            return
        if not isinstance(result, str):
            result = json.dumps(result, default=str, ensure_ascii=False)
        attrs = _tool_request_attrs(agent, task, name=name, args=args, call_id=call_id, round_no=round_no,
                                    executed_by=executed_by)
        attrs["gen_ai.tool.call.result"] = result
        w.span("execute_tool " + name, attrs, start_ns=start_ns, error=error,
               trace_key=getattr(task, "session_token_id", "") or "", span_id=span_id)
    except Exception as exc:  # noqa: BLE001 - tracing must never break the loop
        logger.debug("run_trace_tool_call_failed", extra={"err": str(exc)})
