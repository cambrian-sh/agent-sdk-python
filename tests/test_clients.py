"""ADR-0036 issue 0036-05: memory / artifact / substrate clients.

These wrap already-shipped server RPCs and carry NO client-side scope/classification
authority (D5; honors ADR-0034/0035): reads send no scope params; writes carry only a
narrow-only hint. A coined hint surfaces InvalidTagError; a scope-denied read surfaces
ArtifactNotFound.
"""

import pytest

from cambrian_agent_sdk._proto import cambrian_pb2 as pb
from cambrian_agent_sdk.clients import (
    ArtifactManager,
    ArtifactNotFound,
    InvalidTagError,
    MemoryClient,
    WorkingMemory,
)


import grpc


class _FakeRpcError(grpc.RpcError):
    def __init__(self, code):
        self._code = code

    def code(self):
        return self._code


class _FakeStub:
    """Records the last request per RPC and returns a scripted response.

    Pass ``raise_invalid={"IngestMemory", ...}`` to simulate an InvalidArgument
    (coined hint) on those RPCs; ``found=False`` makes GetArtifact report absence.
    Pass ``execute_tool_error="..."" to simulate a tool-call error response.
    """

    def __init__(self, raise_invalid=None, found=True, execute_tool_error=""):
        self.calls = {}
        self._raise_invalid = raise_invalid or set()
        self._found = found
        self._execute_tool_error = execute_tool_error

    def _maybe_raise(self, name):
        if name in self._raise_invalid:
            raise _FakeRpcError(grpc.StatusCode.INVALID_ARGUMENT)

    def QueryMemory(self, req, timeout=None, metadata=None):
        self.calls["QueryMemory"] = req
        self.calls["QueryMemory_metadata"] = metadata
        return pb.MemoryResponse(results=[pb.MemoryResult(text="paris", score=0.9, metadata="m")])

    def IngestMemory(self, req, timeout=None, metadata=None):
        self.calls["IngestMemory"] = req
        self.calls["IngestMemory_metadata"] = metadata
        self._maybe_raise("IngestMemory")
        return pb.IngestMemoryResponse(doc_id="doc-7")

    def UploadArtifact(self, req, timeout=None):
        self.calls["UploadArtifact"] = req
        self._maybe_raise("UploadArtifact")
        return pb.UploadArtifactResponse(hash="h9", tags=list(req.tags))

    def GetArtifact(self, req, timeout=None):
        self.calls["GetArtifact"] = req
        return pb.GetArtifactResponse(content=b"bytes", content_type="text", found=self._found)

    def ListStepArtifacts(self, req, timeout=None):
        self.calls["ListStepArtifacts"] = req
        return pb.ListStepArtifactsResponse(artifacts=[])

    def GenerateViaModelStream(self, req, timeout=None, metadata=None):
        self.calls["GenerateViaModelStream"] = req
        self.calls["GenerateViaModelStream_metadata"] = metadata
        yield pb.GenerateChunk(text="hi ")
        yield pb.GenerateChunk(text="there")

    def Execute(self, req, timeout=None):
        self.calls["Execute"] = req
        return pb.Handoff(payload=pb.Object(type="text", data=b"delegated-result"))

    def ExecuteTool(self, req, timeout=None, metadata=None):
        self.calls["ExecuteTool"] = req
        self.calls["ExecuteTool_metadata"] = metadata or []
        return pb.ExecuteToolResponse(
            result_json="" if self._execute_tool_error else '{"echo":"hello","kernel_side":true}',
            arg_hash="abc123",
            result_hash="def456",
            error=self._execute_tool_error,
        )

    def Embed(self, req, timeout=None, metadata=None):
        self.calls["Embed"] = req
        return pb.EmbedResponse(vector=[0.5, 0.25, -0.5])


def test_embed_returns_vector_from_kernel_embedder():
    """ADR-0041: substrate.embed proxies the kernel Embed RPC and returns the vector."""
    from cambrian_agent_sdk import SubstrateClient

    stub = _FakeStub()
    sub = SubstrateClient(stub=stub, agent_id="a")
    vec = sub.embed("some text")
    assert vec == [0.5, 0.25, -0.5]
    assert stub.calls["Embed"].text == "some text"


def test_recall_sends_agent_identity_metadata():
    """The server derives the scope principal from x-agent-id gRPC metadata
    (fail-closed on empty). Without it, agent memory queries are denied —
    regression for 'memory query: denied unknown principal'."""
    stub = _FakeStub()
    mem = MemoryClient(stub=stub, agent_id="analyst")

    mem.recall("capital of france")

    md = dict(stub.calls["QueryMemory_metadata"] or [])
    assert md.get("x-agent-id") == "analyst", f"expected x-agent-id metadata, got {md}"


def test_recall_threads_session_id_for_d1_filter():
    """ADR-0048 D1: recall MUST send x-session-id, else the kernel sees an empty
    session and the same-session step-record filter silently no-ops — the agent's
    own step output is recalled straight back. Regression for that no-op."""
    stub = _FakeStub()
    mem = MemoryClient(stub=stub, agent_id="analyst")

    mem.recall("write the analysis to a file", session_token_id="sess-42")

    md = dict(stub.calls["QueryMemory_metadata"] or [])
    assert md.get("x-session-id") == "sess-42", f"expected x-session-id, got {md}"
    assert md.get("x-agent-id") == "analyst"


def test_recall_actions_routes_actions_lane():
    """ADR-0049 D4: recall_actions is a separate intent — it sends x-lane=actions so
    the kernel queries action records, not facts."""
    stub = _FakeStub()
    mem = MemoryClient(stub=stub, agent_id="analyst")

    mem.recall_actions("what did I write", session_token_id="sess-1")

    md = dict(stub.calls["QueryMemory_metadata"] or [])
    assert md.get("x-lane") == "actions"
    assert md.get("x-agent-id") == "analyst"
    assert md.get("x-session-id") == "sess-1"


def test_recall_does_not_send_actions_lane():
    stub = _FakeStub()
    mem = MemoryClient(stub=stub, agent_id="analyst")
    mem.recall("what do I know")
    md = dict(stub.calls["QueryMemory_metadata"] or [])
    assert "x-lane" not in md  # fact recall is the default lane


def test_recall_entity_routes_entity_lane():
    """ADR-0049 Issue 012: recall_entity is an exact lookup — it sends x-lane=entity and
    the canonical kind:id as the query."""
    stub = _FakeStub()
    mem = MemoryClient(stub=stub, agent_id="analyst")

    mem.recall_entity("file:c:/repo/a.md", session_token_id="sess-1")

    md = dict(stub.calls["QueryMemory_metadata"] or [])
    assert md.get("x-lane") == "entity"
    assert stub.calls["QueryMemory"].query == "file:c:/repo/a.md"


def test_recall_precedents_routes_precedent_lane():
    """ADR-0049 Issue 014: recall_precedents is the world-model pull lane — it sends
    x-lane=precedents so the kernel returns transitions, not facts."""
    stub = _FakeStub()
    mem = MemoryClient(stub=stub, agent_id="analyst")

    mem.recall_precedents("about to deploy the service", session_token_id="sess-1")

    md = dict(stub.calls["QueryMemory_metadata"] or [])
    assert md.get("x-lane") == "precedents"
    assert md.get("x-agent-id") == "analyst"
    assert md.get("x-session-id") == "sess-1"


def test_recall_omits_session_id_when_absent():
    stub = _FakeStub()
    mem = MemoryClient(stub=stub, agent_id="analyst")

    mem.recall("q")

    md = dict(stub.calls["QueryMemory_metadata"] or [])
    assert "x-session-id" not in md


def test_recall_calls_querymemory_with_no_scope_params():
    stub = _FakeStub()
    mem = MemoryClient(stub=stub, agent_id="analyst")

    results = mem.recall("capital of france", top_k=3)

    req = stub.calls["QueryMemory"]
    assert req.query == "capital of france"
    assert req.top_k == 3
    # D5: the request type has no scope fields at all — nothing to leak client-side
    field_names = {f.name for f in req.DESCRIPTOR.fields}
    assert field_names == {"query", "top_k"}
    assert results[0]["text"] == "paris"


def test_remember_passes_only_a_narrow_only_hint():
    stub = _FakeStub()
    mem = MemoryClient(stub=stub, agent_id="analyst")

    doc_id = mem.remember("an insight", hint=["analytics"], source="analyst", session_id="s1")

    req = stub.calls["IngestMemory"]
    assert list(req.tags) == ["analytics"]  # the hint rides on `tags` — narrow-only
    assert req.text == "an insight"
    assert doc_id == "doc-7"


def test_remember_coined_hint_raises_invalid_tag_error():
    stub = _FakeStub(raise_invalid={"IngestMemory"})
    mem = MemoryClient(stub=stub, agent_id="a")
    with pytest.raises(InvalidTagError):
        mem.remember("x", hint=["invented"])


def test_artifact_save_passes_narrow_only_hint():
    stub = _FakeStub()
    art = ArtifactManager(stub=stub)
    h = art.save(b"png-bytes", hint=["public_kb"], content_type="image/png", session_id="s", step_index=1)
    assert h == "h9"
    assert list(stub.calls["UploadArtifact"].tags) == ["public_kb"]


def test_artifact_save_coined_hint_raises_invalid_tag_error():
    stub = _FakeStub(raise_invalid={"UploadArtifact"})
    with pytest.raises(InvalidTagError):
        ArtifactManager(stub=stub).save(b"x", hint=["invented"])


def test_artifact_get_returns_bytes_and_sends_no_scope_tags():
    stub = _FakeStub(found=True)
    data = ArtifactManager(stub=stub).get("h9")
    assert data == b"bytes"
    # GetArtifactRequest carries only the hash — no scope tags to leak
    assert {f.name for f in stub.calls["GetArtifact"].DESCRIPTOR.fields} == {"hash"}


def test_artifact_get_scope_denied_or_missing_raises_not_found():
    stub = _FakeStub(found=False)  # scope-denied reads collapse to found=false
    with pytest.raises(ArtifactNotFound):
        ArtifactManager(stub=stub).get("secret-hash")


def test_list_from_step_sends_no_scope_tags():
    stub = _FakeStub()
    ArtifactManager(stub=stub).list_from_step("s", 2)
    assert {f.name for f in stub.calls["ListStepArtifacts"].DESCRIPTOR.fields} == {"session_id", "step_index"}


def test_working_memory_is_read_only_and_distinct_from_recall():
    from cambrian_agent_sdk.types import ContextRef

    refs = [ContextRef(cid="c1", type="fact", activation=1.0, precision=0.9, snippet="a known fact")]
    wm = WorkingMemory(refs)
    out = wm.assemble()
    assert "a known fact" in out  # assembled from the GIVEN refs, no RPC
    # read-only: no write/remember surface on working memory
    assert not hasattr(wm, "remember")
    assert not hasattr(wm, "save")


# ── SubstrateClient: generate (token-accounted) + execute (self-delegation) ───


def test_generate_is_token_accounted():
    from cambrian_agent_sdk import SubstrateClient

    stub = _FakeStub()
    sub = SubstrateClient(stub=stub, agent_id="a")
    text = sub.generate(session_token_id="tok-42", prompt="hello")
    assert text == "hi there"
    # the managed-gateway request carries the session token (ADR-0018 accounting)
    assert stub.calls["GenerateViaModelStream"].session_token_id == "tok-42"


def test_execute_charges_subplan_tokens_to_parent_step():
    from cambrian_agent_sdk import SubstrateClient

    stub = _FakeStub()
    sub = SubstrateClient(stub=stub, agent_id="parent")
    out = sub.execute("summarise the doc", session_token_id="parent-tok")
    assert out == "delegated-result"
    # sub-plan tokens charged to the parent step's session token
    assert stub.calls["Execute"].metadata["_session_token_id"] == "parent-tok"
    assert stub.calls["Execute"].from_agent == "parent"


def test_execute_refuses_self_delegation():
    from cambrian_agent_sdk import SubstrateClient, SelfDelegationError

    sub = SubstrateClient(stub=_FakeStub(), agent_id="me")
    with pytest.raises(SelfDelegationError):
        sub.execute("do it myself", target="me")


def test_execute_tool_calls_execute_tool_rpc():
    from cambrian_agent_sdk import SubstrateClient

    stub = _FakeStub()
    sub = SubstrateClient(stub=stub, agent_id="tool-user")
    resp = sub.execute_tool(
        tool_name="tracer",
        args_json='{"message":"hello"}',
        session_token_id="tok-1",
        step_index=2,
    )
    assert resp["result_json"] == '{"echo":"hello","kernel_side":true}'
    assert resp["arg_hash"] != ""
    assert resp["result_hash"] != ""
    assert resp["error"] == ""
    assert stub.calls["ExecuteTool"].tool_name == "tracer"
    assert stub.calls["ExecuteTool"].args_json == '{"message":"hello"}'


def test_execute_tool_passes_agent_identity():
    from cambrian_agent_sdk import SubstrateClient

    stub = _FakeStub()
    sub = SubstrateClient(stub=stub, agent_id="my-agent")
    sub.execute_tool(tool_name="tracer", args_json="{}")
    md = stub.calls.get("ExecuteTool_metadata", [])
    assert any(k == "x-agent-id" and v == "my-agent" for k, v in md)


def test_execute_tool_error_non_empty():
    from cambrian_agent_sdk import SubstrateClient

    stub = _FakeStub(execute_tool_error="not allowed")
    sub = SubstrateClient(stub=stub, agent_id="a")
    resp = sub.execute_tool(tool_name="blocked", args_json="{}")
    assert resp["error"] == "not allowed"
    assert resp["result_json"] == ""
