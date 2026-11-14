"""ENG-518 (owner decision 2026-11-14): a kernel on an authenticated deployment
refuses a lease-less agent-plane memory read (``no_lease``). An author's
``self.memory.recall(q)`` passes no lease, so the SDK presents the lease the kernel
dispatched the running task under; an explicit lease still wins, and outside a
task nothing is invented."""

from cambrian_agent_sdk import AgentTask, CognitiveAgent
from cambrian_agent_sdk.clients import MemoryClient, task_lease
from cambrian_agent_sdk.runtime import dispatch_run


class _Resp:
    results = []


class _Stub:
    def __init__(self):
        self.metadata = []

    def QueryMemory(self, req, timeout=None, metadata=None):
        self.metadata.append(dict(metadata or []))
        return _Resp()


def _lease(md):
    return md.get("x-lease-id", "")


def test_recall_defaults_to_the_current_task_lease():
    stub = _Stub()
    mem = MemoryClient(stub=stub, agent_id="a")
    with task_lease("lease-7"):
        mem.recall("q")
        mem.recall_actions("q")
        mem.recall("q", session_token_id="explicit")
    mem.recall("q")  # outside a task: none
    leases = [_lease(m) for m in stub.metadata]
    assert leases == ["lease-7", "lease-7", "explicit", ""]


def test_dispatch_run_binds_the_task_lease_for_author_recall():
    stub = _Stub()

    class Author(CognitiveAgent):
        def run(self, task):
            self.memory.recall("what do we know")  # no lease passed by the author
            return "ok"

    agent = Author(agent_id="author")
    agent.memory = MemoryClient(stub=stub, agent_id="author")
    dispatch_run(agent, AgentTask(text="x", data=b"x", session_token_id="lease-from-kernel"))
    assert _lease(stub.metadata[0]) == "lease-from-kernel"
