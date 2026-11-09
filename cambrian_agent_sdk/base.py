"""Trait-aligned agent base classes (ADR-0036 D1).

Three sibling classes mapped 1:1 to the ADR-0001/0033 trait taxonomy, sharing an
abstract :class:`Agent`. Trait contracts are enforced by **class structure**:

- :class:`CognitiveAgent` — LLM reasoning (``think()``), memory, ``@tool`` actions.
- :class:`DeterministicAgent` — a scripted cell; **no** ``think()`` / memory surface.
  *Deprecated* (2026-10-31): subclassing it emits a ``DeprecationWarning``; see the class.
- :class:`DaemonAgent` — a persistent signal producer on a *different* gRPC contract
  (``SignalStream``, not ``AgentService``); it is a **sibling, not a mixin**, because it
  cannot be both a task-responder and a signal-producer in one lifecycle.

Authors subclass a trait and override ``run()`` (cognitive/deterministic) or
``daemon_loop()`` (daemon) — never importing ``Handoff`` / ``Payload`` / any proto type.
"""

from __future__ import annotations

import json
import warnings
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
    #:
    #: ``max_tokens = None`` means "use run_think's default" (4096). It was a literal
    #: 1024 here, which SHADOWED that default for every agent that goes through
    #: think(): run_think raised 1024 -> 4096 because 1024 cut substantive outputs
    #: (a tool call whose content arg is a multi-KB file) mid-emit, and that fix never
    #: reached any CognitiveAgent. One default, one place (ADR-0139 Phase 0). Set an
    #: int on the subclass to cap a deliberately short-answer agent.
    max_tokens: Optional[int] = None
    temperature: float = 0.7

    #: Lifetime of the cached system prompt in think() (ADR-0139 D1): ``"short"``
    #: (5 min) suits machine-paced runs and is the default; ``"long"`` (1 h) suits
    #: an agent serving a human-paced session. ``None`` defers to the
    #: ``CAMBRIAN_AGENT_CACHE_TTL`` environment variable, then ``"short"``.
    cache_ttl: Optional[str] = None

    #: Whether think() performs a mandatory initial LTM recall each run (the
    #: agent-initiated retrieval loop). On by default for every cognitive agent.
    seed_recall: bool = True

    #: ADR-0139 D14 / D13 kernel-surface verbs, off for a plain CognitiveAgent
    #: (an agent opts in): ``propose_memory_note`` (durable principal notes) and
    #: ``propose_skill`` (a skill proposal for operator approval). Each is offered only
    #: when the substrate serves it; the kernel enforces every limit whatever is
    #: offered. ``run_think``'s ``allow_*`` keywords override these per call.
    #: (ADR-0145 D14 removed ``spawn_subagent`` and its ``allow_subagents`` switch.)
    allow_memory_note_writes: bool = False
    allow_skill_proposals: bool = False
    #: Principal-owned recurring tasks (kernel capability "scheduled-tasks"):
    #: ``schedule_task`` proposes one for the person the run is for (created only
    #: after that person approves it) and ``manage_schedules`` lists, pauses,
    #: resumes or deletes theirs. Off unless an agent opts in; offered only when
    #: the substrate serves it.
    allow_scheduled_tasks: bool = False
    #: Knowledge proposals (Orchestrator/ProposeKnowledge): ``propose_knowledge``
    #: writes a CANDIDATE knowledge item or link asserted by this agent, optionally
    #: asking the person the run is for to commit it. Never a committed value. Off
    #: unless an agent opts in; offered only when the substrate serves it.
    allow_knowledge_proposals: bool = False

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
        max_memory_queries: int | None = None,
        max_tool_rounds: int | None = None,
    ) -> "AgentResult":
        """ReAct-style reason/retrieve/act loop (issue 0036-04).

        The budgets default to None and are resolved from react.py rather than
        restated here. They used to be literal 3/5 in this signature, which SHADOWED
        react.py's constants — `think()` passed its own 5 on every call, so raising
        `_DEFAULT_MAX_TOOL_ROUNDS` there had no effect on any agent that goes through
        this method (i.e. all of them). One default, one place.
        """
        from .react import (
            _DEFAULT_MAX_MEMORY_QUERIES,
            _DEFAULT_MAX_TOOL_ROUNDS,
            run_think,
        )

        if max_memory_queries is None:
            max_memory_queries = _DEFAULT_MAX_MEMORY_QUERIES
        if max_tool_rounds is None:
            max_tool_rounds = _DEFAULT_MAX_TOOL_ROUNDS
        # max_tokens=None: omit it so run_think's own default applies (see the
        # class attribute's comment).
        gen_kwargs = {} if self.max_tokens is None else {"max_tokens": self.max_tokens}

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
            temperature=self.temperature,
            **gen_kwargs,
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
        from .channel import build_channel
        from .runtime import start_agent_server
        from .server import _parse_listen_address, _parse_substrate_addr
        from ._proto import cambrian_pb2_grpc

        # ADR-0130 D4: the one channel builder — TLS root and agent credential
        # from the environment, plaintext (warned once) on a loopback kernel.
        channel = build_channel(_parse_substrate_addr(), agent_id=self.agent_id)
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


#: Message for the DeterministicAgent deprecation. Module-level so tests can match it.
_DETERMINISTIC_AGENT_DEPRECATION = (
    "{name} subclasses DeterministicAgent, which is deprecated and will be removed once "
    "the kernel drops trait \"tool\" (deprecated there by ADR-0039 A1.3). The kernel no "
    "longer selects agents by static bid: default dispatch matches the capabilities an "
    "agent's manifest declares (ADR-0100) and solicits no bids. Subclass CognitiveAgent "
    "and override run() with the scripted logic, or implement the capability as a "
    "kernel-owned tool invoked via ExecuteTool."
)


class DeterministicAgent(Agent):
    """A scripted cell: typed ``run()`` only — no reasoning, memory, or tools.

    Has no ``think()`` by construction (structural trait enforcement).

    What ``trait = "tool"`` means to the kernel today: the agent skips the LLM
    capability interview (its ``AGENT_DESCRIPTION`` is embedded directly) and is
    never drafted as a verifier. It does NOT win the auction by bidding statically:
    the kernel's Confidence=1.0 static-bid bypass was removed (ADR-0023), the kernel
    marks ``"tool"`` deprecated in favour of kernel-owned tools invoked via
    ExecuteTool (ADR-0039 A1.3), and default capability-typed dispatch solicits no
    bids at all (ADR-0100). :meth:`propose` only answers if a caller still asks.
    For a deterministic capability, prefer a kernel tool.

    .. deprecated:: unreleased (after 0.1.5)
        Owner decision 2026-10-31 (ticket
        T-20261031-sdk-deterministicagent-still-advertises-trait). Defining a
        subclass emits a :class:`DeprecationWarning` once per subclass (from
        :meth:`__init_subclass__`, so instantiating does not repeat it). Use instead:

        - :class:`CognitiveAgent` with ``run()`` overridden to hold the scripted
          logic — the kernel's default dispatch picks agents by the capabilities their
          manifest declares (ADR-0100), not by trait or bid. Note the trait change
          does matter in two places: a ``"cognitive"`` agent goes through the kernel's
          LLM capability interview and can be drafted as a verifier, which a
          ``"tool"`` agent is not (kernel system agents are interview-exempt either
          way). This is how ``example-agents/terminal_agent.py`` was migrated.
        - a kernel-owned tool, invoked via ``ExecuteTool`` (ADR-0039 A1.3), when the
          capability is a deterministic kernel function rather than an agent.

        Behaviour is unchanged while deprecated: ``trait`` stays ``"tool"`` (the kernel
        still parses it, skips the interview and excludes it from verifier pools) and
        :meth:`propose` still returns the static bid. The class will be removed once the
        kernel drops ``TraitTool`` (cambrian-core ``domain/agent.go``).
    """

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        # stacklevel=3 attributes the warning to the module running the ``class``
        # statement: frame 1 is this hook, frame 2 is ``ABCMeta.__new__`` (Agent is an
        # ABC, and abc's metaclass __new__ is a Python frame), frame 3 is the author's
        # module. That module is what Python's default filter checks: an agent run as
        # ``python agent.py`` defines its class in __main__ and sees the warning;
        # library imports do not unless warnings are enabled (pytest, -W default).
        # tests/test_deterministic.py pins the attributed filename.
        warnings.warn(
            _DETERMINISTIC_AGENT_DEPRECATION.format(name=cls.__qualname__),
            DeprecationWarning,
            stacklevel=3,
        )

    trait = "tool"

    #: The bid :meth:`propose` returns if a bid is solicited (issue 0036-07). The
    #: kernel gives it no special weight; see the class docstring.
    STATIC_CONFIDENCE = 1.0
    STATIC_LATENCY_MS = 5

    @abstractmethod
    def run(self, task: "AgentTask") -> Union["AgentResult", dict, str, bytes]:
        """Handle one task deterministically. Return a plain result."""
        raise NotImplementedError

    def propose(self, task: "AgentTask" = None):
        """The fixed bid (Confidence=1.0, Latency=5ms) if one is solicited; the default
        capability-typed dispatch (ADR-0100) does not solicit bids."""
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
