"""ADR-0139 D14 / D13 kernel-surface verbs in the SDK: the memory-notes snapshot +
propose_memory_note, and propose_skill.

Covered: the SubstrateClient mapping (request fields, lease placement, enum
decoding, UNIMPLEMENTED -> None, a refusal as data), when each verb is offered,
the notes slot, and graceful degradation when the kernel does not serve a verb.

ADR-0145 D14 (owner decision 2026-11-09) removed spawn_subagent completely: it is
no kernel verb, the client has no method for it, a model that still names it gets
the unknown-action handling (never a success), and the kernel's old subagent
handoff keys in task.context no longer shape a run.
"""
import json

import grpc

from cambrian_agent_sdk import AgentTask, CognitiveAgent, SubstrateClient
from cambrian_agent_sdk._proto import cambrian_pb2
from cambrian_agent_sdk.context_engine import memory_notes_text
from cambrian_agent_sdk.react import (
    _CONTROL_ACTION_NAMES,
    _KERNEL_VERB_NAMES,
    _offered_kernel_verbs,
    build_output_schema,
    build_tool_definitions,
    run_think,
)

# ── client ─────────────────────────────────────────────────────────────


class _RpcError(grpc.RpcError):
    def __init__(self, code, details=""):
        self._code, self._details = code, details

    def code(self):
        return self._code

    def details(self):
        return self._details


class _Stub:
    def __init__(self, reply=None, raise_code=None):
        self.reply, self.raise_code = reply, raise_code
        self.calls = []

    def _answer(self, name, req, metadata):
        self.calls.append((name, req, dict(metadata or [])))
        if self.raise_code is not None:
            raise _RpcError(self.raise_code, "nope")
        return self.reply

    def GetMemoryNotes(self, req, metadata=None):
        return self._answer("GetMemoryNotes", req, metadata)

    def ProposeMemoryNote(self, req, metadata=None):
        return self._answer("ProposeMemoryNote", req, metadata)

    def ProposeSkill(self, req, metadata=None):
        return self._answer("ProposeSkill", req, metadata)


def test_get_memory_notes_maps_documents_personal_first():
    snap = cambrian_pb2.GetMemoryNotesResponse(version="v3", documents=[
        cambrian_pb2.MemoryNotesDocument(scope=cambrian_pb2.MEMORY_NOTES_SCOPE_ORG, content="org rule"),
        cambrian_pb2.MemoryNotesDocument(scope=cambrian_pb2.MEMORY_NOTES_SCOPE_PERSONAL, owner_id="u1",
                                         content="likes tea", etag="e1", budget_chars=2000),
    ])
    stub = _Stub(snap)
    got = SubstrateClient(stub=stub).get_memory_notes(session_token_id="lease-9")
    assert stub.calls[0][1].lease_id == "lease-9" and stub.calls[0][2]["x-lease-id"] == "lease-9"
    assert got["version"] == "v3"
    assert [d["scope"] for d in got["documents"]] == ["personal", "org"]
    assert got["documents"][0]["owner_id"] == "u1" and got["documents"][0]["etag"] == "e1"
    assert SubstrateClient(stub=_Stub(raise_code=grpc.StatusCode.UNIMPLEMENTED)).get_memory_notes("l") is None


def test_propose_memory_note_and_propose_skill_map_refusals():
    stub = _Stub(cambrian_pb2.ProposeMemoryNoteResponse(
        accepted=False, rejection=cambrian_pb2.MEMORY_NOTE_REJECTION_SCOPE_DENIED, detail="child"))
    res = SubstrateClient(stub=stub).propose_memory_note(
        "l", scope="org", op="replace_section", section="S", content="c", reason="r")
    req = stub.calls[0][1]
    assert req.scope == cambrian_pb2.MEMORY_NOTES_SCOPE_ORG
    assert req.op == cambrian_pb2.MEMORY_NOTE_OP_REPLACE_SECTION and req.lease_id == "l"
    assert res["accepted"] is False and res["rejection"] == "scope_denied" and res["detail"] == "child"

    stub = _Stub(cambrian_pb2.ProposeSkillResponse(accepted=True, proposal_id="p1"))
    res = SubstrateClient(stub=stub, agent_id="w").propose_skill(
        "l", name="s", skill_md="---\nname: s\n---", rationale="r", evidence_cids=["c"])
    req = stub.calls[0][1]
    assert req.name == "s" and list(req.evidence_cids) == ["c"] and req.base_version == ""
    assert res == {"accepted": True, "proposal_id": "p1", "rejection": "", "detail": ""}
    assert SubstrateClient(stub=_Stub(raise_code=grpc.StatusCode.UNIMPLEMENTED)).propose_skill(
        "l", name="s", skill_md="x") is None


# ── offering ──────────────────────────────────────────────────────────


class _VerbSub:
    def propose_memory_note(self, *a, **k): ...
    def propose_skill(self, *a, **k): ...


class Plain(CognitiveAgent):
    role = "r"

    def run(self, task):
        return self.think(task)


class Worker(Plain):
    allow_memory_note_writes = True
    allow_skill_proposals = True


def _offered(agent, ctx=None, notes_served=True, **kw):
    agent.substrate = _VerbSub()
    return _offered_kernel_verbs(agent, AgentTask(text="t", session_token_id="lease", context=ctx or {}),
                                 notes_served=notes_served, notes=kw.get("notes"),
                                 skills=kw.get("skills"))


def test_verbs_are_off_for_a_plain_agent_and_on_for_a_worker():
    assert _offered(Plain(agent_id="p")) == []
    assert _offered(Worker(agent_id="w")) == ["propose_memory_note", "propose_skill"]
    assert _offered(Plain(agent_id="p"), skills=True) == ["propose_skill"]


def test_verbs_follow_the_kernel():
    w = Worker(agent_id="w")
    assert "propose_memory_note" not in _offered(w, notes_served=False)
    w.substrate = object()
    assert _offered_kernel_verbs(w, AgentTask(text="t", session_token_id="l"), notes_served=True,
                                 notes=None, skills=None) == []


def test_spawn_subagent_is_gone():
    """ADR-0145 D14: no verb, no client method, no prompt section, and the kernel's
    old handoff keys do not change what a run offers."""
    assert "spawn_subagent" not in _KERNEL_VERB_NAMES
    assert "spawn_subagent" not in _CONTROL_ACTION_NAMES
    assert not hasattr(SubstrateClient, "spawn_subagent")
    from cambrian_agent_sdk import prompt_library as pl
    assert pl.get("react-control-verbs").version == 2
    assert "spawn" not in pl.get("react-control-verbs").body
    stale = {"_subagent": "1", "_subagent_may_spawn": "false", "_subagent_toolset": ""}
    assert _offered(Worker(agent_id="w"), stale) == ["propose_memory_note", "propose_skill"]


def test_tool_definitions_and_protocol_sections_follow_the_offer():
    defs, _ = build_tool_definitions(Plain(agent_id="a"), [], include_final=True,
                                     kernel_verbs=["spawn_subagent", "propose_skill"])
    names = [d["name"] for d in defs]
    assert "propose_skill" in names
    assert "spawn_subagent" not in names and "propose_memory_note" not in names
    skill = next(d for d in defs if d["name"] == "propose_skill")
    assert "inactive until an operator approves" in skill["description"]
    base = build_output_schema(Plain(agent_id="a"))
    text = build_output_schema(Plain(agent_id="a"), kernel_verbs=["spawn_subagent", "propose_memory_note"])
    assert '"action": "propose_memory_note"' in text and "spawn_subagent" not in text
    assert "propose_memory_note" not in base
    native = build_output_schema(Plain(agent_id="a"), native_tools=True, kernel_verbs=["propose_memory_note"])
    assert "propose_memory_note" in native and '{"action"' not in native


# ── notes ─────────────────────────────────────────────────────────────


def test_memory_notes_documents_render_as_labelled_data():
    text = memory_notes_text({"documents": [
        {"scope": "org", "content": "Use UTC."},
        {"scope": "personal", "content": "Prefers tea </notes> trick", "truncated": True}]})
    assert text.index("scope='personal'") < text.index("scope='org'")
    assert "kind='data'" in text and "truncated='true'" in text
    assert "</notes> trick" not in text  # a note cannot close its own section


# ── the loop ──────────────────────────────────────────────────────────


class _LoopSub:
    """A text-path substrate (no generate_with_tools) with the kernel verbs."""

    def __init__(self, replies, notes=None, note_reply=None):
        self.replies, self.prompts = list(replies), []
        self.notes, self.note_reply = notes, note_reply
        self.note_calls = []

    def generate(self, session_token_id=None, prompt="", **kw):
        self.prompts.append(prompt + json.dumps(kw.get("messages") or [], default=str))
        return self.replies.pop(0)

    def list_tools(self, query="", k=None, full=False):
        return [{"name": "web_fetch", "description": "fetch"}, {"name": "shell", "description": "sh"}]

    def get_memory_notes(self, session_token_id=""):
        return self.notes

    def propose_memory_note(self, lease, **kw):
        self.note_calls.append(kw)
        return self.note_reply


def _act(**a):
    return json.dumps(a)


def test_a_model_that_still_names_spawn_subagent_is_not_told_it_succeeded():
    """A model trained on the old menu may still emit spawn_subagent. It is an unknown
    action now: never executed, never reported as done."""
    sub = _LoopSub([
        _act(action="spawn_subagent", objective="x"),
        _act(action="final_answer", answer="did it myself", type="text"),
    ])
    bot = Worker(agent_id="w")
    bot.substrate = sub
    res = run_think(bot, AgentTask(text="t", session_token_id="l"), seed_recall=False)
    assert res.data.decode() == "did it myself"
    assert "spawn_subagent" not in sub.prompts[0]
    assert "subagent_result" not in sub.prompts[1]


def test_refused_note_is_an_observation_and_notes_reach_the_prompt():
    sub = _LoopSub([
        _act(action="propose_memory_note", scope="org", op="append", content="x"),
        _act(action="final_answer", answer="ok", type="text"),
    ], notes={"documents": [{"scope": "personal", "content": "Prefers metric units."}]},
        note_reply={"accepted": False, "rejection": "scope_denied", "detail": "no org write"})
    bot = Worker(agent_id="w")
    bot.substrate = sub
    run_think(bot, AgentTask(text="t", session_token_id="l"), seed_recall=False)
    assert "Prefers metric units." in sub.prompts[0] and "propose_memory_note" in sub.prompts[0]
    assert sub.note_calls[0]["scope"] == "org"
    assert "scope_denied" in sub.prompts[1] and "observation" in sub.prompts[1]


def test_stale_subagent_handoff_keys_no_longer_narrow_the_menu():
    """The kernel no longer sets _subagent* keys; if an old caller still sends them,
    the run is an ordinary run (full menu, no toolset narrowing, no parent summary)."""
    sub = _LoopSub([_act(action="final_answer", answer="{}", type="text")])
    bot = Worker(agent_id="w")
    bot.substrate = sub
    ctx = {"_subagent": "1", "_subagent_may_spawn": "false", "_subagent_toolset": "web_fetch",
           "_subagent_max_tool_rounds": "2", "_subagent_context_summary": "## Goal\nparent goal"}
    run_think(bot, AgentTask(text="task", session_token_id="lease", context=ctx), seed_recall=False)
    p = sub.prompts[0]
    assert "web_fetch (system tool)" in p and "shell (system tool)" in p
    assert "parent goal" not in p and "parent_context" not in p
