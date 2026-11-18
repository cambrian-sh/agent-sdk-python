"""T-20261118-chat-agent-and-composed-worker-memory-reads-are: the skill and tool
menus the reasoning loop fetches inside a task present the task's lease as
``x-lease-id``. A credential-authenticated kernel refuses a lease-less agent-plane
read ``no_lease`` (ENG-518); the menu call (``react._list_system_skills``) passed
no lease, so every chat turn and composed step logged a denial."""

from cambrian_agent_sdk import AgentTask, CognitiveAgent, SubstrateClient
from cambrian_agent_sdk.clients import task_lease
from cambrian_agent_sdk.react import _list_system_skills
from cambrian_agent_sdk.runtime import dispatch_run


class _Resp:
    skills = []
    tools = []


class _Call:
    def initial_metadata(self):
        return ()


class _ListTools:
    def __init__(self, sink):
        self._sink = sink

    def __call__(self, req, metadata=None):
        self._sink.append(("ListTools", dict(metadata or [])))
        return _Resp()

    def with_call(self, req, metadata=None):
        self._sink.append(("ListTools", dict(metadata or [])))
        return _Resp(), _Call()


class _Stub:
    def __init__(self):
        self.calls = []
        self.ListTools = _ListTools(self.calls)

    def ListSkills(self, req, metadata=None):
        self.calls.append(("ListSkills", dict(metadata or [])))
        return _Resp()


def test_menus_default_to_the_current_task_lease():
    stub = _Stub()
    sub = SubstrateClient(stub=stub, agent_id="chat_agent")
    with task_lease("lease-7"):
        sub.list_skills(query="q", k=3)
        sub.list_tools(query="q")
        sub.find_tools("q")
    leases = [md.get("x-lease-id", "") for _, md in stub.calls]
    assert leases == ["lease-7", "lease-7", "lease-7"]
    # The menu is not a skill load: no x-session-token grant-activation key.
    assert "x-session-token" not in stub.calls[0][1]


def test_explicit_lease_wins_and_skill_load_keeps_its_run_key():
    stub = _Stub()
    sub = SubstrateClient(stub=stub, agent_id="a")
    with task_lease("lease-7"):
        sub.list_skills(names=["s"], full=True, session_token_id="explicit")
    md = stub.calls[0][1]
    assert md["x-lease-id"] == "explicit"
    assert md["x-session-token"] == "explicit"


def test_outside_a_task_no_lease_is_invented():
    stub = _Stub()
    sub = SubstrateClient(stub=stub, agent_id="a")
    sub.list_skills()
    sub.list_tools()
    assert all("x-lease-id" not in md for _, md in stub.calls)


def test_a_turns_skill_menu_carries_the_dispatched_lease():
    """Through dispatch_run (what the kernel's Execute reaches): the loop's own
    skill-menu helper presents the lease the kernel handed in with the task."""
    stub = _Stub()

    class Turn(CognitiveAgent):
        def run(self, task):
            _list_system_skills(self, query=task.text)
            return "ok"

    agent = Turn(agent_id="chat_agent")
    agent.substrate = SubstrateClient(stub=stub, agent_id="chat_agent")
    dispatch_run(agent, AgentTask(text="hi", data=b"hi", session_token_id="lease-from-kernel"))
    assert stub.calls and stub.calls[0][0] == "ListSkills"
    assert stub.calls[0][1].get("x-lease-id") == "lease-from-kernel"
