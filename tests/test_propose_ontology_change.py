"""propose_ontology_change (Orchestrator/ProposeOntologyChange, ADR-0149 D2): the SDK
client mapping, when the verb is offered (opt-in only), and how a reply reaches the
model.

The kernel takes the drafter from the lease and only ever records a DRAFT; a person
applies it by approving. The SDK maps the request faithfully, never sends an actor,
and treats a refusal as data.
"""
import json

import grpc

from cambrian_agent_sdk import AgentTask, CognitiveAgent, SubstrateClient
from cambrian_agent_sdk._proto import cambrian_pb2
from cambrian_agent_sdk.react import (_offered_kernel_verbs, _run_kernel_verb, build_output_schema,
                                      build_tool_definitions)


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

    def ProposeOntologyChange(self, req, metadata=None):
        self.calls.append((req, dict(metadata or [])))
        if self.raise_code is not None:
            raise _RpcError(self.raise_code)
        return self.reply


CHANGE = {"kinds": [{"kind": "shipment", "predicates": {"carrier": {"type": "text"}}}]}


def test_draft_maps_the_change_and_carries_no_actor():
    reply = cambrian_pb2.ProposeOntologyChangeResponse(
        accepted=True, draft_id="od_1", created=True, base_version=3, state="pending",
        approval_id="apr-1", approval_state="pending",
        changes=[cambrian_pb2.OntologyDraftChange(subject="kind shipment", detail="new kind")])
    # "class" is a Python keyword: the field is reachable only by name.
    setattr(reply, "class", "additive")
    setattr(reply.changes[0], "class", "additive")
    stub = _Stub(reply)
    res = SubstrateClient(stub=stub, agent_id="w").propose_ontology_change(
        "lease-1", change=CHANGE, namespace="tenant-a", reason="carriers report", client_ref="k1")
    req, md = stub.calls[0]
    assert req.lease_id == "lease-1" and md["x-lease-id"] == "lease-1"
    assert json.loads(req.change_json) == CHANGE and req.namespace == "tenant-a"
    assert req.reason == "carriers report" and req.client_ref == "k1"
    # The request type has no field that could name a drafter or an approver.
    assert not any(w in f.name for f in cambrian_pb2.ProposeOntologyChangeRequest.DESCRIPTOR.fields
                   for w in ("actor", "author", "approver", "agent"))
    assert res == {"accepted": True, "detail": "", "draft_id": "od_1", "created": True, "base_version": 3,
                   "class": "additive", "changes": [{"class": "additive", "subject": "kind shipment", "detail": "new kind"}],
                   "state": "pending", "approval_id": "apr-1", "approval_state": "pending"}
    # A JSON string passes through unchanged.
    SubstrateClient(stub=stub).propose_ontology_change("l", change='{"relations":[]}')
    assert stub.calls[1][0].change_json == '{"relations":[]}'


def test_refusal_and_unimplemented():
    stub = _Stub(cambrian_pb2.ProposeOntologyChangeResponse(accepted=False, detail="breaking changes need a migration"))
    res = SubstrateClient(stub=stub).propose_ontology_change("l", change=CHANGE)
    assert res["accepted"] is False and "breaking" in res["detail"]
    assert SubstrateClient(stub=_Stub(raise_code=grpc.StatusCode.UNIMPLEMENTED)).propose_ontology_change(
        "l", change=CHANGE) is None


class _Sub:
    def __init__(self):
        self.calls = []

    def propose_ontology_change(self, lease, **kw):
        self.calls.append((lease, kw))
        return {"accepted": False, "detail": "ontology change refused: breaking"}


class Plain(CognitiveAgent):
    role = "r"

    def run(self, task):
        return self.think(task)


class Drafter(Plain):
    allow_ontology_proposals = True


def _offered(agent, lease="lease"):
    agent.substrate = _Sub()
    return _offered_kernel_verbs(agent, AgentTask(text="t", session_token_id=lease),
                                 notes_served=False, notes=None, skills=None)


def test_offered_only_to_an_opted_in_agent_with_a_lease():
    assert "propose_ontology_change" not in _offered(Plain(agent_id="p"))
    assert _offered(Drafter(agent_id="w")) == ["propose_ontology_change"]
    assert _offered(Drafter(agent_id="w"), lease="") == []
    # Opting in to knowledge proposals does not offer ontology drafts.
    class KnowledgeOnly(Plain):
        allow_knowledge_proposals = True
    assert "propose_ontology_change" not in _offered(KnowledgeOnly(agent_id="k"))
    defs, _ = build_tool_definitions(Plain(agent_id="a"), [], include_final=True,
                                     kernel_verbs=["propose_ontology_change"])
    d = next(d for d in defs if d["name"] == "propose_ontology_change")
    assert "You only draft it" in d["description"] and d["parameters"]["required"] == ["change"]


def test_protocol_sections_render_only_when_offered():
    with_verb = build_output_schema(Plain(agent_id="a"), kernel_verbs=["propose_ontology_change"])
    native = build_output_schema(Plain(agent_id="a"), native_tools=True, kernel_verbs=["propose_ontology_change"])
    without = build_output_schema(Plain(agent_id="a"), kernel_verbs=[])
    assert '"action": "propose_ontology_change"' in with_verb and "## Ontology drafts" in native
    assert "Ontology drafts" not in without


def test_a_refusal_reaches_the_model_as_data():
    agent = Drafter(agent_id="w")
    agent.substrate = _Sub()
    task = AgentTask(text="t", session_token_id="lease")
    obs = _run_kernel_verb(agent, task, "propose_ontology_change",
                           {"change": CHANGE, "namespace": "ns", "reason": "why"}, 1)
    lease, kw = agent.substrate.calls[0]
    assert lease == "lease" and kw == {"change": CHANGE, "namespace": "ns", "reason": "why"}
    assert "verb='propose_ontology_change'" in obs and "breaking" in obs
    # No change: a note, not a call.
    assert "needs a change" in _run_kernel_verb(agent, task, "propose_ontology_change", {}, 2)
    assert len(agent.substrate.calls) == 1


def test_the_kernel_refusing_an_agent_without_the_grant_is_data():
    # The kernel enforces the opt-in (T-20261113-agent-proposal-rpcs-proposeknowledge):
    # an agent without a grant of the verb gets PERMISSION_DENIED, which reaches the
    # loop as an error dict, never an exception.
    res = SubstrateClient(stub=_Stub(raise_code=grpc.StatusCode.PERMISSION_DENIED)).propose_ontology_change(
        "l", change=CHANGE)
    assert res == {"error": "nope", "code": "PERMISSION_DENIED"}
