"""T-20261105-sdk-react-llm-request-react-llm-usage-evidence: SlogHandler must
forward ``extra={...}`` fields; they used to be dropped silently."""
import io
import json
import logging
import sys

from cambrian_agent_sdk._logging import SlogHandler


def _emit(extra):
    rec = logging.LogRecord("cambrian.react", logging.INFO, __file__, 1, "react_llm_request", None, None)
    for k, v in extra.items():
        setattr(rec, k, v)
    buf, old = io.StringIO(), sys.stdout
    sys.stdout = buf
    try:
        SlogHandler().emit(rec)
    finally:
        sys.stdout = old
    return json.loads(buf.getvalue())


def test_extra_fields_are_forwarded():
    out = _emit({"system_sha256": "abc", "prompts": [{"id": "react-turn", "version": 1}], "round": 2})
    assert out["msg"] == "react_llm_request"
    assert out["system_sha256"] == "abc"
    assert out["prompts"] == [{"id": "react-turn", "version": 1}]
    assert out["round"] == 2


def test_reserved_names_and_unserialisable_values_are_safe():
    out = _emit({"msg_extra": 1, "level": "spoof", "obj": object()})
    assert out["level"] == "info"            # a reserved key is never overwritten
    assert out["x_level"] == "spoof"
    assert isinstance(out["obj"], str)        # non-JSON values are stringified


# ---- T-20261109-chat-turn-crashes-after-schedule-task-succeeds -----------------
# logger.info("react_schedule_task", extra={"created": ...}) raised
# KeyError("Attempt to overwrite 'created' in LogRecord") once INFO was enabled (it is
# in a running agent, not under pytest's default WARNING), failing the chat turn
# right after the kernel had created the task.

import ast  # noqa: E402
import pathlib  # noqa: E402

from cambrian_agent_sdk._logging import LOG_RECORD_RESERVED, safe_extra  # noqa: E402

_SDK = pathlib.Path(__file__).resolve().parents[1] / "cambrian_agent_sdk"
#: the kernel-shipped Python agents (chat_agent, worker_agent, system agents), when the
#: workspace sibling is present
_CORE_AGENTS = pathlib.Path(__file__).resolve().parents[2] / "cambrian-core" / "agents"


def test_safe_extra_renames_logrecord_attributes():
    out = safe_extra({"created": True, "name": "n", "round": 1})
    assert out == {"x_created": True, "x_name": "n", "round": 1}
    rec = logging.makeLogRecord({})
    logging.getLogger("t").makeRecord("t", logging.INFO, __file__, 1, "m", None, None, extra=out)
    assert "created" in LOG_RECORD_RESERVED and "taskName" in LOG_RECORD_RESERVED and rec


def test_schedule_task_verb_logs_at_info_without_crashing(caplog):
    from cambrian_agent_sdk import AgentTask, CognitiveAgent
    from cambrian_agent_sdk.react import _run_kernel_verb

    class _Sub:
        def propose_scheduled_task(self, lease, **kw):
            return {"created": True, "task": {"id": "sched-1"}}

    class Assistant(CognitiveAgent):
        role = "r"
        allow_scheduled_tasks = True

        def run(self, task):
            return self.think(task)

    agent = Assistant(agent_id="a")
    agent.substrate = _Sub()
    caplog.set_level(logging.INFO)
    out = _run_kernel_verb(agent, AgentTask(text="t", session_token_id="lease-1"), "schedule_task",
                           {"goal": "Summarise my week", "cron": "0 8 * * 1"}, 1)
    assert "verb='schedule_task'" in out
    rec = [r for r in caplog.records if r.getMessage() == "react_schedule_task"][0]
    assert rec.task_created is True


def _extra_violations(root: pathlib.Path) -> list[str]:
    bad = []
    for path in sorted(root.rglob("*.py")):
        if "_pb2" in path.name or ".venv" in path.parts or "node_modules" in path.parts:
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            for kw in node.keywords:
                if kw.arg != "extra":
                    continue
                where = f"{path.name}:{node.lineno}"
                v = kw.value
                if isinstance(v, ast.Dict):
                    for k in v.keys:
                        if k is None:
                            bad.append(f"{where}: **spread in extra (use safe_extra)")
                        elif isinstance(k, ast.Constant) and k.value in LOG_RECORD_RESERVED:
                            bad.append(f"{where}: reserved key {k.value!r}")
                        elif not isinstance(k, ast.Constant):
                            bad.append(f"{where}: computed key (use safe_extra)")
                elif not (isinstance(v, ast.Call) and getattr(v.func, "id", getattr(v.func, "attr", "")) == "safe_extra"):
                    # a non-literal mapping can carry any key: it must go through safe_extra
                    if _is_logging_call(node):
                        bad.append(f"{where}: non-literal extra (use safe_extra)")
    return bad


def _is_logging_call(node: ast.Call) -> bool:
    f = node.func
    return isinstance(f, ast.Attribute) and f.attr in {
        "debug", "info", "warning", "warn", "error", "exception", "critical", "log"}


def test_no_logging_extra_can_overwrite_a_logrecord_attribute():
    assert _extra_violations(_SDK) == []
    if _CORE_AGENTS.is_dir():
        assert _extra_violations(_CORE_AGENTS) == []
