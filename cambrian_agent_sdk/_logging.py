"""Structured logging for Cambrian agents.

``configure_logging()`` installs a handler that writes slog-compatible JSON to
stdout. The Substrate reads agent stdout/stderr and forwards JSON lines to its
own slog stream, giving a unified cross-language log timeline correlated by
``task_id``.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from contextvars import ContextVar

_task_id_var: ContextVar[str] = ContextVar("task_id", default="")
_agent_id_var: ContextVar[str] = ContextVar("agent_id", default="")

# Attributes every LogRecord carries; anything else on a record came from
# ``extra={...}`` and is forwarded as a structured field. Without this the SDK's
# structured lines (react_llm_request, react_llm_usage, ...) reached the kernel log
# with their data silently dropped (T-20261105-sdk-react-llm-request-react-llm-usage-evidence).
# "taskName" is listed explicitly: LogRecord carries it only from Python 3.12, and
# without it the reserved set (and so what safe_extra renames) differed between the
# 3.10 floor pyproject declares and newer Pythons.
_STD_ATTRS = frozenset(vars(logging.LogRecord("", 0, "", 0, "", None, None))) | {"message", "asctime", "taskName"}
_RESERVED = frozenset({"time", "level", "msg", "logger", "task_id", "agent_id"})


# LogRecord's own attribute names. ``logger.info(..., extra={"created": ...})``
# raises KeyError("Attempt to overwrite 'created' in LogRecord") the moment the
# record is built, i.e. only when the level is enabled: a chat turn failed that way
# right after schedule_task succeeded (T-20261109-chat-turn-crashes-after-schedule-task-succeeds).
LOG_RECORD_RESERVED = _STD_ATTRS


def safe_extra(fields) -> dict:
    """Return ``fields`` as a logging ``extra`` mapping that can never collide with a
    LogRecord attribute: a reserved key (``created``, ``name``, ``msg``, ``module``...)
    is renamed ``x_<key>``, the same convention SlogHandler uses for its own keys.
    Every ``extra=`` built from a non-literal mapping goes through this; literal keys
    are checked by tests/test_logging_extras.py."""
    return {("x_" + str(k) if k in LOG_RECORD_RESERVED else str(k)): v for k, v in dict(fields).items()}


def _json_safe(v, depth: int = 0):
    if v is None or isinstance(v, (bool, int, float, str)):
        return v
    if depth >= 4:
        return str(v)[:500]
    if isinstance(v, dict):
        return {str(k): _json_safe(x, depth + 1) for k, x in list(v.items())[:100]}
    if isinstance(v, (list, tuple, set, frozenset)):
        return [_json_safe(x, depth + 1) for x in list(v)[:100]]
    return str(v)[:500]


_LEVEL_MAP = {
    logging.DEBUG: "debug",
    logging.INFO: "info",
    logging.WARNING: "warn",
    logging.ERROR: "error",
    logging.CRITICAL: "error",
}


class SlogHandler(logging.Handler):
    """Formats Python log records as slog-compatible JSON for Substrate pipe reading."""

    def emit(self, record: logging.LogRecord) -> None:
        entry = {
            "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(record.created)),
            "level": _LEVEL_MAP.get(record.levelno, "info"),
            "msg": self.format(record),
            "logger": record.name,
        }
        task_id = _task_id_var.get()
        agent_id = _agent_id_var.get()
        if task_id:
            entry["task_id"] = task_id
        if agent_id:
            entry["agent_id"] = agent_id
        for k, v in record.__dict__.items():
            if k in _STD_ATTRS or k.startswith("_"):
                continue
            entry["x_" + k if k in _RESERVED else k] = _json_safe(v)
        try:
            line = json.dumps(entry)
        except (TypeError, ValueError):  # never let a log line crash the agent
            line = json.dumps({key: entry[key] for key in ("time", "level", "msg", "logger") if key in entry})
        sys.stdout.write(line + "\n")
        sys.stdout.flush()


def configure_logging(level: int = logging.INFO, agent_id: str = "") -> None:
    """Install SlogHandler on the root logger.

    Call this once at agent startup before ``agent.serve()``.
    """
    if agent_id:
        _agent_id_var.set(agent_id)
    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()
    root.addHandler(SlogHandler())


def set_task_context(task_id: str, agent_id: str = "") -> None:
    """Set per-task context vars. Called by the SDK at on_execute entry."""
    _task_id_var.set(task_id)
    if agent_id:
        _agent_id_var.set(agent_id)
