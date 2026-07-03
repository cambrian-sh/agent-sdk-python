"""ADR-0036 issue 0036-01: trait-aligned base classes.

Trait contracts are enforced by class STRUCTURE, not documentation:
a DeterministicAgent literally has no think(); a DaemonAgent is a sibling
(different gRPC contract), not a mixin.
"""

import pytest

from cambrian_agent_sdk import Agent, CognitiveAgent, DeterministicAgent, DaemonAgent


def test_abstract_agent_cannot_be_instantiated():
    """The shared base is abstract — authors subclass a trait, never Agent directly."""
    with pytest.raises(TypeError):
        Agent(agent_id="x")


def test_three_trait_classes_share_the_abstract_base():
    for cls in (CognitiveAgent, DeterministicAgent, DaemonAgent):
        assert issubclass(cls, Agent)


def test_deterministic_agent_has_no_think():
    """Structural trait enforcement: a scripted cell literally cannot reason."""
    assert not hasattr(DeterministicAgent, "think")


def test_cognitive_agent_has_think():
    assert hasattr(CognitiveAgent, "think")


def test_daemon_is_a_sibling_not_a_mixin():
    """DaemonAgent extends the abstract base directly — it is NOT a CognitiveAgent,
    and has no task-responder run()/think() (it serves a different gRPC contract)."""
    assert not issubclass(DaemonAgent, CognitiveAgent)
    assert not issubclass(DaemonAgent, DeterministicAgent)
    assert not hasattr(DaemonAgent, "run")
    assert not hasattr(DaemonAgent, "think")


def test_daemon_exposes_loop_and_send_signal():
    assert hasattr(DaemonAgent, "daemon_loop")
    assert hasattr(DaemonAgent, "send_signal")


def test_subclassing_requires_no_proto_import():
    """An author writes business logic returning plain Python — no Handoff/Payload."""

    class Echo(CognitiveAgent):
        def run(self, task):
            return "ok"

    e = Echo(agent_id="echo")
    assert e.run(None) == "ok"
    assert e.trait == "cognitive"


def test_deterministic_subclass_runs_without_reasoning_surface():
    class Doubler(DeterministicAgent):
        def run(self, task):
            return "42"

    d = Doubler(agent_id="doubler")
    assert d.run(None) == "42"
    assert d.trait == "tool"
    assert not hasattr(d, "think")


def test_manifest_carries_the_trait():
    import json

    class Bot(CognitiveAgent):
        def run(self, task):
            return "x"

    manifest = json.loads(Bot(agent_id="bot", version="2.0.0").manifest_json())
    assert manifest["trait"] == "cognitive"
    assert manifest["version"] == "2.0.0"
