"""Principal-owned recurring tasks in the SDK (kernel capability "scheduled-tasks",
T-20261106-the-assistant-cannot-set-up-a-recurring-task): the SubstrateClient
mapping, when schedule_task / manage_schedules are offered, and the loop calling
them as kernel verbs (native and text)."""
import json

import grpc

from cambrian_agent_sdk import AgentTask, CognitiveAgent, SubstrateClient
from cambrian_agent_sdk._proto import cambrian_pb2
from cambrian_agent_sdk.react import (
    _offered_kernel_verbs,
    _run_kernel_verb,
    build_output_schema,
    build_tool_definitions,
)


class _RpcError(grpc.RpcError):
    def __init__(self, code):
        self._code = code

    def code(self):
        return self._code

    def details(self):
        return "nope"


class _Stub:
    def __init__(self, reply=None, raise_code=None):
        self.reply, self.raise_code, self.calls = reply, raise_code, []

    def _answer(self, name, req, metadata, **kw):
        self.calls.append((name, req, dict(metadata or []), kw))
        if self.raise_code is not None:
            raise _RpcError(self.raise_code)
        return self.reply

    def ProposeScheduledTask(self, req, metadata=None, timeout=None):
        return self._answer("ProposeScheduledTask", req, metadata, timeout=timeout)

    def ListScheduledTasks(self, req, metadata=None):
        return self._answer("ListScheduledTasks", req, metadata)

    def ManageScheduledTask(self, req, metadata=None):
        return self._answer("ManageScheduledTask", req, metadata)


def test_propose_maps_the_request_the_task_and_refusals():
    task = cambrian_pb2.ScheduledTask(id="sched-1", owner_id="lena", goal="g", cron="0 8 * * 1", active=True,
                                      conversation_id="sched:sched-1")
    stub = _Stub(cambrian_pb2.ProposeScheduledTaskResponse(created=True, task=task))
    res = SubstrateClient(stub=stub, agent_id="chat_agent").propose_scheduled_task(
        "lease-1", goal="g", rrule="FREQ=WEEKLY;BYDAY=MO", timezone="Europe/Berlin", delivery="email")
    name, req, md, _ = stub.calls[0]
    assert name == "ProposeScheduledTask"
    assert req.lease_id == "lease-1" and md["x-lease-id"] == "lease-1" and md["x-agent-id"] == "chat_agent"
    assert req.rrule == "FREQ=WEEKLY;BYDAY=MO" and req.timezone == "Europe/Berlin" and req.delivery == "email"
    # No owner field exists to send: the kernel takes the owner from the lease.
    assert "owner" not in [f.name for f in cambrian_pb2.ProposeScheduledTaskRequest.DESCRIPTOR.fields]
    assert res["created"] is True and res["task"]["owner_id"] == "lena" and res["task"]["id"] == "sched-1"

    stub = _Stub(cambrian_pb2.ProposeScheduledTaskResponse(
        created=False, rejection=cambrian_pb2.SCHEDULED_TASK_REJECTION_DENIED, detail="declined"))
    assert SubstrateClient(stub=stub).propose_scheduled_task("l", goal="g", cron="@daily") == {
        "created": False, "rejection": "denied", "detail": "declined"}
    assert SubstrateClient(stub=_Stub(raise_code=grpc.StatusCode.UNIMPLEMENTED)).propose_scheduled_task(
        "l", goal="g", cron="@daily") is None


def test_list_and_manage():
    stub = _Stub(cambrian_pb2.ListScheduledTasksResponse(tasks=[cambrian_pb2.ScheduledTask(id="a", owner_id="o")]))
    assert SubstrateClient(stub=stub).list_scheduled_tasks("l")["tasks"][0]["id"] == "a"
    stub = _Stub(cambrian_pb2.ManageScheduledTaskResponse(ok=True))
    assert SubstrateClient(stub=stub).manage_scheduled_task("l", "a", "pause") == {"ok": True, "detail": ""}
    assert stub.calls[0][1].action == cambrian_pb2.SCHEDULED_TASK_ACTION_PAUSE


class _Sub:
    def __init__(self):
        self.proposed, self.managed = [], []

    def propose_scheduled_task(self, lease, **kw):
        self.proposed.append((lease, kw))
        return {"created": True, "task": {"id": "sched-1", "owner_id": "lena"}}

    def list_scheduled_tasks(self, lease):
        return {"tasks": [{"id": "sched-1"}]}

    def manage_scheduled_task(self, lease, task_id, action):
        self.managed.append((task_id, action))
        return {"ok": True, "detail": ""}


class Plain(CognitiveAgent):
    role = "r"

    def run(self, task):
        return self.think(task)


class Assistant(Plain):
    allow_scheduled_tasks = True


def _offered(agent, lease="lease"):
    agent.substrate = _Sub()
    return _offered_kernel_verbs(agent, AgentTask(text="t", session_token_id=lease),
                                 notes_served=False, notes=None, skills=None)


def test_offered_only_when_allowed_served_and_leased():
    assert _offered(Plain(agent_id="p")) == []
    assert _offered(Assistant(agent_id="a")) == ["schedule_task", "manage_schedules"]
    assert _offered(Assistant(agent_id="a"), lease="") == []
    a = Assistant(agent_id="a")
    a.substrate = object()  # a substrate without the methods: an older SDK/kernel pairing
    assert _offered_kernel_verbs(a, AgentTask(text="t", session_token_id="l"), notes_served=False,
                                 notes=None, skills=None) == []


def test_native_and_text_surfaces_follow_the_offer():
    defs, _ = build_tool_definitions(Plain(agent_id="a"), [], include_final=True,
                                     kernel_verbs=["schedule_task", "manage_schedules"])
    by = {d["name"]: d for d in defs}
    assert "approve" in by["schedule_task"]["description"]
    assert by["manage_schedules"]["parameters"]["required"] == ["op"]
    text = build_output_schema(Plain(agent_id="a"), kernel_verbs=["schedule_task", "manage_schedules"])
    assert '"action": "schedule_task"' in text and '"action": "manage_schedules", "op"' in text
    assert "schedule_task" not in build_output_schema(Plain(agent_id="a"))


def test_the_loop_calls_the_kernel_and_renders_data():
    agent = Assistant(agent_id="a")
    agent.substrate = _Sub()
    task = AgentTask(text="every monday summarise", session_token_id="lease-7")
    rendered = _run_kernel_verb(agent, task, "schedule_task",
                                {"goal": "Summarise my notes", "cron": "0 8 * * 1"}, 1)
    assert "verb='schedule_task'" in rendered and json.loads(rendered.split("\n")[1])["created"]
    lease, kw = agent.substrate.proposed[0]
    assert lease == "lease-7" and kw["goal"] == "Summarise my notes" and kw["cron"] == "0 8 * * 1"
    assert "needs a goal" in _run_kernel_verb(agent, task, "schedule_task", {}, 1)
    _run_kernel_verb(agent, task, "manage_schedules", {"op": "delete", "id": "sched-1"}, 2)
    assert agent.substrate.managed == [("sched-1", "delete")]
    assert "needs the task id" in _run_kernel_verb(agent, task, "manage_schedules", {"op": "pause"}, 3)
    listed = _run_kernel_verb(agent, task, "manage_schedules", {"op": "list"}, 4)
    assert "sched-1" in listed
