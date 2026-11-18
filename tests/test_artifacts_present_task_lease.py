"""T-20261118-agent-artifacts-from-planner-lane-steps-carry: artifacts.save / get /
list_from_step present the running task's lease (x-lease-id), so the kernel files
an upload in that lease's run, session and step and resolves the reader's
conversation owner. An explicit lease wins; outside a task no metadata is sent."""

from cambrian_agent_sdk import AgentTask, CognitiveAgent
from cambrian_agent_sdk._proto import cambrian_pb2 as pb
from cambrian_agent_sdk.clients import ArtifactManager, task_lease
from cambrian_agent_sdk.runtime import dispatch_run


class _Stub:
    def __init__(self):
        self.calls = []

    def UploadArtifact(self, req, **kw):
        self.calls.append(("UploadArtifact", dict(kw.get("metadata") or [])))
        return pb.UploadArtifactResponse(hash="led_1")

    def GetArtifact(self, req, **kw):
        self.calls.append(("GetArtifact", dict(kw.get("metadata") or [])))
        return pb.GetArtifactResponse(content=b"x", found=True)

    def ListStepArtifacts(self, req, **kw):
        self.calls.append(("ListStepArtifacts", dict(kw.get("metadata") or [])))
        return pb.ListStepArtifactsResponse()


def test_artifact_calls_default_to_the_current_task_lease():
    stub = _Stub()
    art = ArtifactManager(stub=stub)
    with task_lease("lease-7"):
        art.save(b"bytes", content_type="text/plain")
        art.get("led_1")
        art.list_from_step("sess", 0)
        art.save(b"bytes", session_token_id="explicit")
    assert [md.get("x-lease-id") for _, md in stub.calls] == ["lease-7", "lease-7", "lease-7", "explicit"]


def test_outside_a_task_no_lease_metadata_is_sent():
    stub = _Stub()
    ArtifactManager(stub=stub).save(b"bytes")
    assert stub.calls == [("UploadArtifact", {})]


def test_an_authors_save_inside_run_carries_the_dispatched_lease():
    stub = _Stub()

    class Author(CognitiveAgent):
        def run(self, task):
            self.artifacts.save(b"report", content_type="text/csv")  # no lease passed by the author
            return "ok"

    agent = Author(agent_id="author")
    agent.artifacts = ArtifactManager(stub=stub)
    dispatch_run(agent, AgentTask(text="x", data=b"x", session_token_id="lease-from-kernel"))
    assert stub.calls[0] == ("UploadArtifact", {"x-lease-id": "lease-from-kernel", "x-session-id": "lease-from-kernel"})
