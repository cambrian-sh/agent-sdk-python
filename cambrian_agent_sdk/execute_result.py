"""The structured result of a delegated Execute (ADR-0145 D8).

The kernel's Execute answers with the report text as its payload (unchanged, for
callers that read text) and the structured result as JSON under the response
metadata key ``_execute_result``: the run id, the plan's action ledger, every step
of the final plan with its status and — when the run ledger is on — the ledger
reference of its full output, and the partial / timeout flags.

``ExecuteResult`` delivers all of it. It IS the report text (a ``str`` subclass),
so every existing caller that treated ``execute()``'s return as a string keeps
working unchanged; the structured parts are attributes.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Mapping, Optional

EXECUTE_RESULT_KEY = "_execute_result"
_PLAN_ACTIONS_KEY = "_plan_actions"
_PARTIAL_KEY = "_partial_plan"
_TIMED_OUT_KEY = "_plan_timed_out"
_TIMEOUT_CAUSE_KEY = "_plan_timeout_cause"


class ExecuteResult(str):
    """The report text of an Execute, carrying the structured result.

    Attributes:
        text: the report text (the same string ``self`` is).
        run_id: the run the kernel executed ("" when the request did not run a plan).
        session_id: the task session the run belongs to.
        actions: the plan's action ledger — side-effecting tool calls with status and ids.
        steps: every step of the final plan: ``{step, query, status, ref, preview,
            bytes, fan_out_item, thought}``; ``ref`` is the step output's ledger id when
            the kernel's run ledger is on (readable with the ``ledger_get`` tool).
        partial: the plan stopped before every step completed.
        timed_out / timeout_cause: a plan limit ended it, and which.
        failed_step: the step that failed, -1 when none.
        error: the failure, when there was one.
        ledger: whether ``steps[].ref`` are ledger references.
        requirements: the request's requirement checklist and its kernel-decided
            coverage (ADR-0145 D5): ``{checklist, items: [{item, status, reason,
            values: [{name, type, value, evidence_ref, step, member}], evidence}],
            stages, settled, stalled}``; {} when the checklist is off.
        structured: the decoded ``_execute_result`` object ({} when absent).
        metadata: the whole response metadata.
        payload_type: the response payload's type.
    """

    text: str
    run_id: str
    session_id: str
    actions: List[Dict[str, Any]]
    steps: List[Dict[str, Any]]
    partial: bool
    timed_out: bool
    timeout_cause: str
    failed_step: int
    error: str
    ledger: bool
    requirements: Dict[str, Any]
    structured: Dict[str, Any]
    metadata: Dict[str, str]
    payload_type: str

    def __new__(cls, text: str, metadata: Optional[Mapping[str, str]] = None, payload_type: str = ""):
        obj = super().__new__(cls, text)
        meta = dict(metadata or {})
        structured: Dict[str, Any] = {}
        raw = meta.get(EXECUTE_RESULT_KEY, "")
        if raw:
            try:
                decoded = json.loads(raw)
                if isinstance(decoded, dict):
                    structured = decoded
            except ValueError:
                structured = {}
        obj.text = str(text)
        obj.metadata = meta
        obj.payload_type = payload_type or ""
        obj.structured = structured
        obj.run_id = str(structured.get("run_id") or "")
        obj.session_id = str(structured.get("session_id") or meta.get("_session_id", "") or "")
        obj.actions = _list_of_dicts(structured.get("actions"))
        req = structured.get("requirements")
        obj.requirements = req if isinstance(req, dict) else {}
        if not obj.actions and meta.get(_PLAN_ACTIONS_KEY):
            # An older kernel: the action ledger JSON only.
            try:
                obj.actions = _list_of_dicts(json.loads(meta[_PLAN_ACTIONS_KEY]))
            except ValueError:
                obj.actions = []
        obj.steps = _list_of_dicts(structured.get("steps"))
        obj.partial = bool(structured.get("partial")) or meta.get(_PARTIAL_KEY) == "true"
        obj.timed_out = bool(structured.get("timed_out")) or meta.get(_TIMED_OUT_KEY) == "true"
        obj.timeout_cause = str(structured.get("timeout_cause") or meta.get(_TIMEOUT_CAUSE_KEY, "") or "")
        fs = structured.get("failed_step", -1)
        obj.failed_step = fs if isinstance(fs, int) else -1
        obj.error = str(structured.get("error") or "")
        obj.ledger = bool(structured.get("ledger"))
        return obj

    @property
    def step_refs(self) -> List[str]:
        """The ledger references of every completed step's output, in step order."""
        return [s["ref"] for s in self.steps if s.get("ref")]

    def as_dict(self) -> Dict[str, Any]:
        """The whole result as a plain dict (text plus every structured part)."""
        return {
            "text": self.text,
            "run_id": self.run_id,
            "session_id": self.session_id,
            "actions": self.actions,
            "steps": self.steps,
            "partial": self.partial,
            "timed_out": self.timed_out,
            "timeout_cause": self.timeout_cause,
            "failed_step": self.failed_step,
            "error": self.error,
            "ledger": self.ledger,
        }


def _list_of_dicts(v: Any) -> List[Dict[str, Any]]:
    if not isinstance(v, list):
        return []
    return [x for x in v if isinstance(x, dict)]
