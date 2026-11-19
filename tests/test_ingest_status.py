"""ENG-2176 (contract 0111, ENG-2143): remember() exposes the ingest status and
wait_until_searchable() polls GetIngestStatus until the document is searchable."""

import grpc
import pytest

import cambrian_agent_sdk.clients as clients
from cambrian_agent_sdk import (
    IngestFailedError,
    IngestStatus,
    IngestStatusNotFound,
    MemoryClient,
    RememberResult,
)
from cambrian_agent_sdk._proto import cambrian_pb2 as pb


class _RpcError(grpc.RpcError):
    def __init__(self, code):
        self._code = code

    def code(self):
        return self._code


class _Stub:
    """IngestMemory returns ``ingest``; GetIngestStatus returns the scripted
    ``statuses`` in order (the last one repeats) or raises ``status_error``."""

    def __init__(self, ingest=None, statuses=(), status_error=None):
        self._ingest = ingest or pb.IngestMemoryResponse(doc_id="source_doc:x")
        self._statuses = list(statuses)
        self._status_error = status_error
        self.status_calls = []

    def IngestMemory(self, req, timeout=None, metadata=None):
        return self._ingest

    def GetIngestStatus(self, req, timeout=None, metadata=None):
        self.status_calls.append((req.document_id, metadata))
        if self._status_error is not None:
            raise _RpcError(self._status_error)
        st = self._statuses.pop(0) if len(self._statuses) > 1 else self._statuses[0]
        return st


def _st(state, total=0, embedded=0, error="", doc="doc-1"):
    return pb.IngestStatus(document_id=doc, state=state, chunks_total=total,
                           chunks_embedded=embedded, error=error)


@pytest.fixture
def fake_clock(monkeypatch):
    now = [0.0]
    monkeypatch.setattr(clients, "_monotonic", lambda: now[0])

    def sleep(s):
        now[0] += s

    monkeypatch.setattr(clients, "_sleep", sleep)
    return now


def test_remember_returns_the_entity_id_as_str_plus_status():
    stub = _Stub(ingest=pb.IngestMemoryResponse(doc_id="source_doc:big", document_id="doc-big",
                                                 status=_st("accepted", total=900, doc="doc-big")))
    got = MemoryClient(stub=stub, agent_id="analyst").remember("a large body")
    # Existing callers: it IS the entity id string.
    assert got == "source_doc:big" and isinstance(got, str) and got.upper() == "SOURCE_DOC:BIG"
    assert isinstance(got, RememberResult) and got.doc_id == "source_doc:big"
    assert got.document_id == "doc-big"
    assert got.status == IngestStatus(document_id="doc-big", state="accepted", chunks_total=900)
    assert not got.status.terminal


def test_remember_against_an_old_kernel_has_no_status():
    got = MemoryClient(stub=_Stub(), agent_id="analyst").remember("x")
    assert got == "source_doc:x" and got.document_id == "" and got.status is None
    with pytest.raises(ValueError, match="contract 0111"):
        MemoryClient(stub=_Stub(), agent_id="analyst").wait_until_searchable(got)


def test_ingest_status_sends_identity_and_maps_not_found():
    stub = _Stub(statuses=[_st("embedding", 10, 4)])
    mc = MemoryClient(stub=stub, agent_id="analyst")
    st = mc.ingest_status("doc-1")
    assert st.state == "embedding" and st.chunks_embedded == 4 and st.chunks_total == 10
    assert stub.status_calls == [("doc-1", [("x-agent-id", "analyst")])]
    with pytest.raises(IngestStatusNotFound):
        MemoryClient(stub=_Stub(status_error=grpc.StatusCode.NOT_FOUND), agent_id="a").ingest_status("nope")
    with pytest.raises(grpc.RpcError):
        MemoryClient(stub=_Stub(status_error=grpc.StatusCode.UNIMPLEMENTED), agent_id="a").ingest_status("d")


def test_wait_polls_until_searchable(fake_clock):
    stub = _Stub(statuses=[_st("accepted", 10), _st("embedding", 10, 5), _st("searchable")])
    st = MemoryClient(stub=stub, agent_id="a").wait_until_searchable("doc-1", timeout=10, poll_interval=0.5)
    assert st.searchable and len(stub.status_calls) == 3 and fake_clock[0] == pytest.approx(1.0)


def test_wait_accepts_a_remember_result(fake_clock):
    stub = _Stub(ingest=pb.IngestMemoryResponse(doc_id="source_doc:big", document_id="doc-big",
                                                 status=_st("accepted", doc="doc-big")),
                 statuses=[_st("searchable", doc="doc-big")])
    mc = MemoryClient(stub=stub, agent_id="a")
    st = mc.wait_until_searchable(mc.remember("big"))
    assert st.searchable and stub.status_calls[0][0] == "doc-big"


def test_wait_raises_on_failed(fake_clock):
    stub = _Stub(statuses=[_st("embedding", 10, 2), _st("failed", 10, 2, error="embedding returned no vector")])
    with pytest.raises(IngestFailedError, match="no vector") as e:
        MemoryClient(stub=stub, agent_id="a").wait_until_searchable("doc-1", timeout=10)
    assert e.value.status.failed


def test_wait_times_out_with_the_last_status(fake_clock):
    stub = _Stub(statuses=[_st("embedding", 10, 3)])
    with pytest.raises(TimeoutError, match="3/10") as e:
        MemoryClient(stub=stub, agent_id="a").wait_until_searchable("doc-1", timeout=2, poll_interval=0.5)
    assert e.value.status.state == "embedding"
    assert fake_clock[0] == pytest.approx(2.0)


def test_wait_returns_at_once_for_a_searchable_document(fake_clock):
    stub = _Stub(statuses=[_st("searchable")])
    MemoryClient(stub=stub, agent_id="a").wait_until_searchable("doc-1")
    assert len(stub.status_calls) == 1 and fake_clock[0] == 0.0


def test_wait_validates_arguments():
    mc = MemoryClient(stub=_Stub(statuses=[_st("searchable")]), agent_id="a")
    with pytest.raises(ValueError):
        mc.wait_until_searchable("", timeout=1)
    with pytest.raises(ValueError):
        mc.wait_until_searchable("doc-1", timeout=0)
