"""propose_knowledge (Orchestrator/ProposeKnowledge): the SDK client mapping, when the
verb is offered, and how a reply reaches the model.

The kernel stamps the asserter from the lease and caps the write at candidate; the
SDK only has to map the request faithfully, never send an actor, and treat a refusal
as data.
"""
from datetime import datetime, timezone

import grpc
import pytest

from cambrian_agent_sdk import AgentTask, CognitiveAgent, SubstrateClient
from cambrian_agent_sdk._proto import cambrian_pb2
from cambrian_agent_sdk.react import _offered_kernel_verbs, _run_kernel_verb, build_tool_definitions


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

    def ProposeKnowledge(self, req, metadata=None):
        self.calls.append((req, dict(metadata or [])))
        if self.raise_code is not None:
            raise _RpcError(self.raise_code)
        return self.reply


def test_item_proposal_maps_typed_values_and_carries_no_actor():
    stub = _Stub(cambrian_pb2.ProposeKnowledgeResponse(
        accepted=True, candidate_id="ki_1", created=True, approval_id="ap1", approval_state="pending"))
    when = datetime(2026, 11, 3, tzinfo=timezone.utc)
    res = SubstrateClient(stub=stub, agent_id="w").propose_knowledge(
        "lease-1", item={"kind": "decision", "entity_id": "decision/d1", "values": [
            {"predicate": "status", "type": "text", "value": "decided"},
            {"predicate": "budget", "type": "number", "value": 12.5},
            {"predicate": "due", "type": "date", "value": "2026-12-01T00:00:00Z"},
            {"predicate": "owner", "type": "entity", "value": "person/alice"}],
            "valid_from": when, "evidence_id": "ev1"},
        request_commit=True, reason="agreed", client_ref="k1")
    req, md = stub.calls[0]
    assert req.lease_id == "lease-1" and md["x-lease-id"] == "lease-1"
    assert req.WhichOneof("proposal") == "item" and req.request_commit and req.reason == "agreed"
    assert req.client_ref == "k1" and req.item.kind == "decision" and req.item.evidence_id == "ev1"
    by = {v.predicate: v for v in req.item.values}
    assert by["status"].text == "decided" and by["budget"].number == 12.5
    assert by["due"].date.ToDatetime(tzinfo=timezone.utc) == datetime(2026, 12, 1, tzinfo=timezone.utc)
    assert by["owner"].entity_ref == "person/alice"
    assert req.item.valid_from.ToDatetime(tzinfo=timezone.utc) == when
    # The request type has no field that could name an asserter.
    assert not any("assert" in f.name or "actor" in f.name for f in cambrian_pb2.ProposeKnowledgeRequest.DESCRIPTOR.fields)
    assert res == {"accepted": True, "detail": "", "candidate_id": "ki_1", "created": True,
                   "approval_id": "ap1", "approval_state": "pending"}


def test_link_proposal_refusal_and_unimplemented():
    stub = _Stub(cambrian_pb2.ProposeKnowledgeResponse(accepted=False, detail="no evidence"))
    res = SubstrateClient(stub=stub).propose_knowledge(
        "l", link={"relation": "about", "from_ref": "entity:a", "to_ref": "entity:b"})
    req, _ = stub.calls[0]
    assert req.WhichOneof("proposal") == "link" and req.link.relation == "about"
    assert res["accepted"] is False and res["detail"] == "no evidence"
    assert SubstrateClient(stub=_Stub(raise_code=grpc.StatusCode.UNIMPLEMENTED)).propose_knowledge(
        "l", item={"kind": "k", "entity_id": "e"}) is None
    with pytest.raises(ValueError):
        SubstrateClient(stub=_Stub()).propose_knowledge(
            "l", item={"kind": "k", "entity_id": "e",
                       "values": [{"predicate": "d", "type": "date", "value": datetime(2026, 1, 1)}]})


class _Sub:
    def __init__(self):
        self.calls = []

    def propose_knowledge(self, lease, **kw):
        self.calls.append((lease, kw))
        return {"accepted": False, "detail": "transition status \"open\" -> \"closed\" is not declared"}


class Plain(CognitiveAgent):
    role = "r"

    def run(self, task):
        return self.think(task)


class Proposer(Plain):
    allow_knowledge_proposals = True


def _offered(agent):
    agent.substrate = _Sub()
    return _offered_kernel_verbs(agent, AgentTask(text="t", session_token_id="lease"),
                                 notes_served=False, notes=None, skills=None)


def test_offered_only_on_opt_in():
    assert "propose_knowledge" not in _offered(Plain(agent_id="p"))
    assert _offered(Proposer(agent_id="w")) == ["propose_knowledge"]
    defs, _ = build_tool_definitions(Plain(agent_id="a"), [], include_final=True, kernel_verbs=["propose_knowledge"])
    d = next(d for d in defs if d["name"] == "propose_knowledge")
    assert "never as the committed record" in d["description"]


def test_a_refusal_reaches_the_model_as_data():
    agent = Proposer(agent_id="w")
    agent.substrate = _Sub()
    obs = _run_kernel_verb(agent, AgentTask(text="t", session_token_id="lease"), "propose_knowledge",
                           {"kind": "decision", "entity_id": "d1", "request_commit": True,
                            "values": [{"predicate": "status", "type": "text", "value": "closed"}]},
                           1)
    lease, kw = agent.substrate.calls[0]
    assert lease == "lease" and kw["request_commit"] is True and kw["link"] is None
    assert kw["item"]["values"][0]["value"] == "closed"
    assert "verb='propose_knowledge'" in obs and "is not declared" in obs
