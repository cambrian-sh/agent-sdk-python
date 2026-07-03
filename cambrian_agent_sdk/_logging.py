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
        sys.stdout.write(json.dumps(entry) + "\n")
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
