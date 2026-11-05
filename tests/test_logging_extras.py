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
