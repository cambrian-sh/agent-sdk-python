"""Unit tests for Cambrian Agent SDK — no gRPC required."""

import pytest

from cambrian_agent_sdk import (
    CognitiveAgent,
    DeterministicAgent,
    Capability,
    AgentResult,
    AgentTask,
    SubstrateClient,
    configure_logging,
)
from cambrian_agent_sdk.types import Capability
from cambrian_agent_sdk.runtime import _coerce_agent_result, dispatch_run
from cambrian_agent_sdk.errors import BudgetExceededError


# ── Types ─────────────────────────────────────────────────────────────────────


def test_payload_text_property():
    from cambrian_agent_sdk.types import Payload
    p = Payload(data=b"hello world")
    assert p.text == "hello world"


def test_payload_from_text():
    from cambrian_agent_sdk.types import Payload
    p = Payload.from_text("hello", type="json")
    assert p.data == b"hello"
    assert p.type == "json"


# ── Coercion ──────────────────────────────────────────────────────────────────


def test_coerce_agent_result_dict_bytes():
    raw = {"data": b"result"}
    resp = _coerce_agent_result(raw)
    assert isinstance(resp, AgentResult)
    assert resp.data == b"result"


def test_coerce_agent_result_dict_str():
    raw = {"data": "hello"}
    resp = _coerce_agent_result(raw)
    assert resp.data == b"hello"


def test_coerce_agent_result_bytes():
    resp = _coerce_agent_result(b"raw bytes")
    assert resp.data == b"raw bytes"


def test_coerce_agent_result_str():
    resp = _coerce_agent_result("plain text")
    assert resp.data == b"plain text"


def test_coerce_agent_result_passthrough():
    ar = AgentResult(data=b"x", confidence=0.9)
    assert _coerce_agent_result(ar) is ar


def test_coerce_agent_result_unsupported_type():
    with pytest.raises(TypeError):
        _coerce_agent_result(12345)


# ── Agent base validation ───────────────────────────────────────────────────


def test_agent_rejects_empty_id():
    with pytest.raises(ValueError, match="agent_id"):
        CognitiveAgent(agent_id="")


def test_capability_rejects_empty_name():
    with pytest.raises(ValueError, match="non-empty"):
        Capability(name="")


def test_capability_rejects_whitespace_name():
    with pytest.raises(ValueError, match="non-empty"):
        Capability(name="   ")


def test_capability_rejects_negative_latency():
    with pytest.raises(ValueError, match="latency_p50_ms"):
        Capability(name="valid", latency_p50_ms=-1)


def test_capability_accepts_zero_latency():
    c = Capability(name="fast", latency_p50_ms=0)
    assert c.latency_p50_ms == 0


# ── CognitiveAgent ────────────────────────────────────────────────────────────


class _EchoCognitive(CognitiveAgent):
    role = "echo bot"

    def run(self, task):
        return AgentResult(data=task.text.encode("utf-8"))


def test_cognitive_agent_run_returns_agent_result():
    agent = _EchoCognitive(agent_id="echo")
    task = AgentTask(text="ping", data=b"ping")
    result = agent.run(task)
    assert isinstance(result, AgentResult)
    assert result.data == b"ping"


def test_cognitive_agent_react_loop_error_caught():
    class _Broken(CognitiveAgent):
        role = "broken"

        def think(self, task, **kwargs):
            from cambrian_agent_sdk.react import ReActLoopError
            raise ReActLoopError("loop exceeded")

    agent = _Broken(agent_id="broken")
    task = AgentTask(text="x", data=b"x")
    result = agent.run(task)
    assert result.type == "error"
    assert result.confidence == 0.0


# ── DeterministicAgent ──────────────────────────────────────────────────────


class _AddTool(DeterministicAgent):
    def run(self, task):
        return AgentResult(data=str(int(task.text) + 1).encode())


def test_deterministic_static_bid():
    agent = _AddTool(agent_id="add")
    bid = agent.propose(None)
    assert bid.confidence == 1.0
    assert bid.estimated_latency_ms == 5


def test_deterministic_run():
    agent = _AddTool(agent_id="add")
    task = AgentTask(text="5", data=b"5")
    result = agent.run(task)
    assert result.data == b"6"


# ── dispatch_run budget handling ─────────────────────────────────────────────


def test_dispatch_run_catches_budget_exceeded():
    class _Expensive(CognitiveAgent):
        role = "expensive"

        def run(self, task):
            raise BudgetExceededError("too costly")

    agent = _Expensive(agent_id="expensive")
    task = AgentTask(text="x", data=b"x")
    result = dispatch_run(agent, task)
    assert result.type == "budget_signal"
    assert b"BUDGET_EXCEEDED" in result.data


def test_dispatch_run_non_budget_error_propagates():
    class _Crash(CognitiveAgent):
        role = "crash"

        def run(self, task):
            raise ValueError("boom")

    agent = _Crash(agent_id="crash")
    task = AgentTask(text="x", data=b"x")
    with pytest.raises(ValueError, match="boom"):
        dispatch_run(agent, task)


# ── Manifest JSON ─────────────────────────────────────────────────────────────


def test_manifest_json_valid():
    agent = _EchoCognitive(agent_id="my_agent", version="1.2.3")
    manifest = agent.manifest_json()
    import json
    parsed = json.loads(manifest)
    assert parsed["version"] == "1.2.3"
    assert parsed["trait"] == "cognitive"


def test_deterministic_manifest_trait():
    agent = _AddTool(agent_id="tool", version="0.0.1")
    import json
    parsed = json.loads(agent.manifest_json())
    assert parsed["trait"] == "tool"


# ── P3: generate_stream ───────────────────────────────────────────────────────


def test_substrate_client_generate_delegates_to_stream():
    client = SubstrateClient("localhost:99999")
    chunks_seen = []

    def _fake_stream(*args, **kwargs):
        yield from ["hello ", "world"]

    client.generate_stream = _fake_stream
    result = client.generate("tok", "prompt")
    assert result == "hello world"


# ── P1: Deadline propagation ──────────────────────────────────────────────────


def test_execute_request_deadline_field_exists():
    from cambrian_agent_sdk.types import ExecuteRequest
    req = ExecuteRequest(deadline_remaining_ms=5000)
    assert req.deadline_remaining_ms == 5000


def test_execute_request_deadline_defaults_to_zero():
    from cambrian_agent_sdk.types import ExecuteRequest
    req = ExecuteRequest()
    assert req.deadline_remaining_ms == 0
