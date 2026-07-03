"""ADR-0036 issue 0036-08: wiring + migration.

The CognitiveAgent binds its memory/artifact/substrate clients over the Substrate
stub at serve() time; the real agents are migrated onto the trait-aligned base
classes and import no Handoff / Payload / proto type.
"""

import os

from cambrian_agent_sdk import (
    AgentTask,
    CognitiveAgent,
    DaemonAgent,
    MemoryClient,
)
from cambrian_agent_sdk import SubstrateClient
from cambrian_agent_sdk.clients import ArtifactManager
from cambrian_agent_sdk._proto import cambrian_pb2 as pb


class _FakeStub:
    def __init__(self):
        self.calls = {}

    def QueryMemory(self, req, timeout=None, metadata=None):
        self.calls["QueryMemory"] = req
        return pb.MemoryResponse(results=[pb.MemoryResult(text="bound", score=1.0, metadata="")])


class _Bot(CognitiveAgent):
    role = "wiring bot"

    def run(self, task):
        return "x"


def test_bind_clients_wires_memory_artifacts_substrate():
    bot = _Bot(agent_id="bot")
    stub = _FakeStub()
    bot._bind_clients(stub)

    assert isinstance(bot.memory, MemoryClient)
    assert isinstance(bot.artifacts, ArtifactManager)
    assert isinstance(bot.substrate, SubstrateClient)
    # the bound memory client actually talks to the stub
    assert bot.memory.recall("q")[0]["text"] == "bound"


_AGENTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "agents")
_MIGRATED = [
    "analyst_agent.py",
    "code_generator_agent.py",
    "summariser_agent.py",
    "example_daemon_agent.py",
]


def test_migrated_agents_import_no_proto_types():
    """D3: a migrated agent never imports Handoff / Payload / proto."""
    for fname in _MIGRATED:
        src = open(os.path.join(_AGENTS_DIR, fname), encoding="utf-8").read()
        assert "Handoff" not in src, fname
        assert "Payload" not in src, fname
        assert "_proto" not in src and "cambrian_pb2" not in src, fname
        assert "LegacyAgent" not in src, fname  # fully migrated off the legacy class


def test_cognitive_agents_are_cognitive_and_daemon_is_daemon():
    import sys

    sys.path.insert(0, os.path.abspath(_AGENTS_DIR))
    import analyst_agent, code_generator_agent, summariser_agent, example_daemon_agent

    assert isinstance(analyst_agent.agent, CognitiveAgent)
    assert isinstance(code_generator_agent.agent, CognitiveAgent)
    assert isinstance(summariser_agent.agent, CognitiveAgent)
    assert isinstance(example_daemon_agent.agent, DaemonAgent)
