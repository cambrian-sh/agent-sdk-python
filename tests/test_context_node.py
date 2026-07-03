"""ADR-0048 D4/R7: put_context_node offload + session-gated get_context_node."""

from cambrian_agent_sdk import SubstrateClient
from cambrian_agent_sdk._proto import cambrian_pb2


class _MockStub:
    def __init__(self):
        self.put_args = None
        self.get_md = None

    def PutContextNode(self, req, metadata=None):
        self.put_args = (bytes(req.data), req.node_type, metadata)
        return cambrian_pb2.PutContextNodeResponse(cid="cid-123")

    def GetContextNode(self, req, metadata=None):
        self.get_md = metadata
        return cambrian_pb2.ContextNodeResponse(cid=req.cid, data=b"full text")


def test_put_context_node_offloads_and_threads_session():
    stub = _MockStub()
    sub = SubstrateClient(stub=stub)

    cid = sub.put_context_node("big block", session_token_id="sess-7")

    assert cid == "cid-123"
    data, ntype, md = stub.put_args
    assert data == b"big block"
    assert ntype == "agent_offload"
    assert ("x-session-id", "sess-7") in md  # session threaded so the kernel owner-stamps


def test_get_context_node_threads_session_for_owned_read():
    stub = _MockStub()
    sub = SubstrateClient(stub=stub)

    node = sub.get_context_node("cid-123", session_token_id="sess-7")

    assert node is not None and node.data == b"full text"
    assert ("x-session-id", "sess-7") in stub.get_md


def test_put_context_node_degrades_to_none_on_error():
    class _BoomStub:
        def PutContextNode(self, req, metadata=None):
            raise RuntimeError("rpc down")

    sub = SubstrateClient(stub=_BoomStub())
    # Must not raise — the working-memory offloader treats None as "render verbatim".
    assert sub.put_context_node("x", session_token_id="s") is None
