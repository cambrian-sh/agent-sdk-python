"""Trait-aligned agent base classes (ADR-0036 D1).

Three sibling classes mapped 1:1 to the ADR-0001/0033 trait taxonomy, sharing an
abstract :class:`Agent`. Trait contracts are enforced by **class structure**:

- :class:`CognitiveAgent` — LLM reasoning (``think()``), memory, ``@tool`` actions.
- :class:`DeterministicAgent` — a scripted cell; **no** ``think()`` / memory surface.
- :class:`DaemonAgent` — a persistent signal producer on a *different* gRPC contract
  (``SignalStream``, not ``AgentService``); it is a **sibling, not a mixin**, because it
  cannot be both a task-responder and a signal-producer in one lifecycle.

Authors subclass a trait and override ``run()`` (cognitive/deterministic) or
``daemon_loop()`` (daemon) — never importing ``Handoff`` / ``Payload`` / any proto type.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Optional, Union

from .types import AgentResult, AgentTask, Capability  # noqa: F401  (re-exported vocabulary)


class Agent(ABC):
    """Abstract base for every Cambrian agent.

    Holds only what all traits share — identity and the manifest — plus the
    abstract ``serve()`` *contract bootstrap* each trait implements differently
    (AgentService vs SignalStream). The abstract ``serve()`` is what makes a bare
    ``Agent`` non-instantiable: authors must pick a trait.
    """

    #: The ADR-0001/0033 trait tag stamped into the manifest. Set by each subclass.
    trait: str = ""

    def __init__(
        self,
        agent_id: str,
        *,
        version: str = "0.1.0",
        description: Optional[str] = None,
    ) -> None:
        if not agent_id or not agent_id.strip():
            raise ValueError("Agent.agent_id must be a non-empty string")
        self.agent_id = agent_id
        self.version = version
        self.description = description or f"Agent {agent_id}"

    @abstractmethod
    def serve(self) -> None:
        """Boot the trait's gRPC contract and block until shutdown.

        Implemented per-trait: cognitive/deterministic serve ``AgentService``;
        the daemon opens a ``SignalStream``. Abstract here so a bare ``Agent``
        cannot be instantiated — the structural enforcement of D1.
        """
        raise NotImplementedError

    def manifest_json(self) -> str:
        """Return the AGENT_MANIFEST JSON block carrying this agent's trait."""
        manifest = {
            "version": self.version,
            "trait": self.trait,
            "supported_formats": ["text"],
            "release_notes": f"Agent {self.agent_id} v{self.version}",
            "dependencies": [],
        }
        return json.dumps(manifest, indent=2)


class CognitiveAgent(Agent):
    """An LLM-reasoning agent: ``run()`` + ``think()`` + memory + ``@tool`` actions."""

    trait = "cognitive"

    #: One-sentence persona for the prompt's <Role>. Override on the subclass.
    role: str = ""

    #: The domain output contract the LLM's final answer must follow (goes into the
    #: prompt's <OutputSchema>). Override on the subclass.
    output_schema: str = ""

    #: Extra <Constraints> for the prompt. Override on the subclass.
    constraints: tuple = ()

    #: Forces the final AgentResult.type (e.g. "code" → executor, "summary"). When
    #: None, the LLM's declared final-answer type is used.
    result_type = None

    #: Generation parameters passed to substrate.generate() inside the ReAct loop.
    #: Override on the subclass to control creativity and length.
    max_tokens: int = 1024
    temperature: float = 0.7

    #: Whether think() performs a mandatory initial LTM recall each run (the
    #: agent-initiated retrieval loop). On by default for every cognitive agent.
    seed_recall: bool = True

    #: Opt-in heavy-result summarizer (ADR-0041 D3). A callable ``str -> str``
    #: applied to a tool result that exceeds the inline budget; ``None`` (default)
    #: keeps the no-LLM heuristic truncation. Set it to e.g. an LLM-backed callable
    #: to get semantic gists for big results — the per-agent/size-trigger opt-in.
    tool_summarizer = None

    #: Memory client + LLM gateway are bound by the runtime (issue 0036-05); set to
    #: a fake in unit tests of think().
    memory = None
    substrate = None

    def run(self, task: "AgentTask") -> Union["AgentResult", dict, str, bytes]:
        """Default handler: drive the ReAct loop. Override for custom orchestration.

        A runaway tool loop (``ReActLoopError``) is caught and returned as a typed
        ``type="error"`` result rather than crashing the process.
        """
        from .react import ReActLoopError

        try:
            return self.think(task)
        except ReActLoopError as exc:
            return AgentResult(data=str(exc).encode("utf-8"), type="error", confidence=0.0)

    def think(
        self,
        task: "AgentTask",
        *,
        max_memory_queries: int = 3,
        max_tool_rounds: int = 5,
    ) -> "AgentResult":
        """ReAct-style reason/retrieve/act loop (issue 0036-04)."""
        from .react import run_think

        return run_think(
            self,
            task,
            role=self.role or None,
            output_schema=self.output_schema,
            constraints=list(self.constraints) if self.constraints else None,
            result_type=self.result_type,
            seed_recall=self.seed_recall,
            max_memory_queries=max_memory_queries,
            max_tool_rounds=max_tool_rounds,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )

    def _bind_clients(self, stub) -> None:
        """Bind the memory / artifact / substrate clients over a connected stub.

        Called by ``serve()`` (and directly in tests with a fake stub). After this,
        ``self.memory`` / ``self.artifacts`` / ``self.substrate`` are live — the
        clients carry no client-side scope authority (ADR-0034/0035)."""
        from . import SubstrateClient
        from .clients import ArtifactManager, MemoryClient

        self.substrate = SubstrateClient(stub=stub, agent_id=self.agent_id)
        self.memory = MemoryClient(stub=stub, agent_id=self.agent_id)
        self.artifacts = ArtifactManager(stub=stub)

    def serve(self, address: Optional[str] = None) -> None:
        """Bind clients to the Substrate, then boot the single-threaded server (D2)."""
        import grpc

        from .runtime import start_agent_server
        from .server import _parse_listen_address, _parse_substrate_addr
        from ._proto import cambrian_pb2_grpc

        channel = grpc.insecure_channel(_parse_substrate_addr())
        self._bind_clients(cambrian_pb2_grpc.OrchestratorStub(channel))
        start_agent_server(self, address or _parse_listen_address())

    @property
    def tools(self):
        """The agent's intra-agent ``@tool`` registry (D4), built lazily and bound."""
        reg = self.__dict__.get("_tools")
        if reg is None:
            from .tools import ToolRegistry

            reg = ToolRegistry(self)
            self._tools = reg
        return reg

    @property
    def capability_names(self) -> list:
        """Names of ``@capability``-marked methods — the *separate* inter-agent registry."""
        from .tools import _CAPABILITY_ATTR

        out, seen = [], set()
        for klass in type(self).__mro__:
            for attr_name, member in vars(klass).items():
                if attr_name in seen:
                    continue
                cap = getattr(member, _CAPABILITY_ATTR, None)
                if cap is not None:
                    seen.add(attr_name)
                    out.append(cap)
        return sorted(out)


class DeterministicAgent(Agent):
    """A scripted cell: typed ``run()`` only — no reasoning, memory, or tools.

    Has no ``think()`` by construction (structural trait enforcement). Bids
    statically in the auction (``Confidence=1.0``, ``Latency=5ms``) — issue 0036-07.
    """

    trait = "tool"

    #: A scripted cell bids statically — it always can, and fast. The auction treats
    #: it as a deterministic tool rather than scoring a reasoning capability.
    STATIC_CONFIDENCE = 1.0
    STATIC_LATENCY_MS = 5

    @abstractmethod
    def run(self, task: "AgentTask") -> Union["AgentResult", dict, str, bytes]:
        """Handle one task deterministically. Return a plain result."""
        raise NotImplementedError

    def propose(self, task: "AgentTask" = None):
        """The automatic static bid (Confidence=1.0, Latency=5ms) — no override needed."""
        from .types import ProposalResponse

        return ProposalResponse(
            confidence=self.STATIC_CONFIDENCE,
            rationale="deterministic tool — static bid",
            estimated_latency_ms=self.STATIC_LATENCY_MS,
        )

    def serve(self, address: Optional[str] = None) -> None:
        """Boot the single-threaded AgentService server and block (D2)."""
        from .runtime import start_agent_server
        from .server import _parse_listen_address

        start_agent_server(self, address or _parse_listen_address())


class DaemonAgent(Agent):
    """A persistent background signal producer on the ``SignalStream`` contract.

    A sibling of the task-responders, not a mixin: it has no ``run()`` / ``think()``.
    The author overrides ``daemon_loop()`` and calls ``send_signal()``. The supervising
    runtime (issue 0036-06) restarts a crashing loop with exponential backoff.
    """

    trait = "daemon"

    def __init__(self, agent_id: str, **kwargs) -> None:
        super().__init__(agent_id, **kwargs)
        self.stream_id: str = ""
        self.params: dict = {}
        import queue

        # Per-instance signal queue — each daemon process owns its own (ADR-0033:
        # one process per stream_id ⇒ no shared self/history bleed across conversations).
        self._signal_queue: "queue.Queue" = queue.Queue()

    @abstractmethod
    def daemon_loop(self) -> None:
        """The author's persistent producer loop — emits via ``send_signal()``."""
        raise NotImplementedError

    def send_signal(self, payload: dict, raw_text: str = "") -> None:
        """Enqueue a signal for emission on the open SignalStream."""
        self._signal_queue.put((payload, raw_text))

    def serve(self, address: Optional[str] = None) -> None:
        """Open the SignalStream and run the supervised daemon loop (issue 0036-06)."""
        from .daemon import start_daemon

        start_daemon(self, substrate_addr=address)
