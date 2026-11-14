"""ADR-0036 issue 0036-08: wiring + migration.

The CognitiveAgent binds its memory/artifact/substrate clients over the Substrate
stub at serve() time; the real agents are migrated onto the trait-aligned base
classes and import no Handoff / Payload / proto type.
"""

import os

from cambrian_agent_sdk import (
    CognitiveAgent,
    DaemonAgent,
    MemoryClient,
    SubstrateClient,
)
from cambrian_agent_sdk._proto import cambrian_pb2 as pb
from cambrian_agent_sdk.clients import ArtifactManager


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


# The example agents ship inside this repo: example-agents/ (the reference daemon and
# ingress agents) and examples/ (the @tool tutorial). The thin prompt-only agents that
# used to live here were deleted with agent stack v2
# (decisions/agent-stack-v2-mcp-tools-and-profiles.md): specialists are now declarative
# profiles over cambrian-core/agents/worker_agent.py.
_SDK_DIR = os.path.join(os.path.dirname(__file__), "..")
_EXAMPLES = [
    os.path.join("example-agents", "pulse_agent.py"),
    os.path.join("example-agents", "example_ingress_agent.py"),
    os.path.join("examples", "tool_tutorial_agent.py"),
]


def test_example_agents_import_no_proto_types():
    """D3: an example agent never imports Handoff / Payload / proto."""
    for rel in _EXAMPLES:
        src = open(os.path.join(_SDK_DIR, rel), encoding="utf-8").read()
        assert "Handoff" not in src, rel
        assert "Payload" not in src, rel
        assert "_proto" not in src and "cambrian_pb2" not in src, rel
        assert "LegacyAgent" not in src, rel  # fully migrated off the legacy class


def test_tutorial_is_cognitive_and_daemon_is_daemon():
    import sys

    sys.path.insert(0, os.path.abspath(os.path.join(_SDK_DIR, "examples")))
    import pulse_agent
    import tool_tutorial_agent

    assert isinstance(tool_tutorial_agent.agent, CognitiveAgent)
    assert sorted(tool_tutorial_agent.agent.tools.names()) == ["add", "divide", "multiply", "subtract"]
    assert isinstance(pulse_agent.agent, DaemonAgent)
