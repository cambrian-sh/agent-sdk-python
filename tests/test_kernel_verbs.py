"""ADR-0139 D12 / D14 / D13 kernel-surface verbs in the SDK: spawn_subagent,
the memory-notes snapshot + propose_memory_note, and propose_skill.

Covered: the SubstrateClient mapping (request fields, lease placement, enum
decoding, UNIMPLEMENTED -> None, a refusal as data), when each verb is offered,
the result rendering (structured, in Evidence, incomplete flagged), the FORK
summary, a subagent run shaped by its handoff, the notes slot, and graceful
degradation when the kernel does not serve a verb.
"""
import json

import grpc
import pytest

from cambrian_agent_sdk import AgentTask, CognitiveAgent, SubstrateClient
from cambrian_agent_sdk._proto import cambrian_pb2
from cambrian_agent_sdk.context_engine import DefaultContextEngine, memory_notes_text
from cambrian_agent_sdk.react import (
    _offered_kernel_verbs,
    _render_subagent_result,
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

    def SpawnSubagent(self, req, metadata=None):
        return self._answer("SpawnSubagent", req, metadata)

    def GetMemoryNotes(self, req, metadata=None):
        return self._answer("GetMemoryNotes", req, metadata)

    def ProposeMemoryNote(self, req, metadata=None):
        return self._answer("ProposeMemoryNote", req, metadata)

    def ProposeSkill(self, req, metadata=None):
        return self._answer("ProposeSkill", req, metadata)


def test_spawn_subagent_maps_the_request_and_the_result():
    reply = cambrian_pb2.SpawnSubagentResponse(
        status=cambrian_pb2.SUBAGENT_STATUS_STOPPED,
        exit_reason=cambrian_pb2.SUBAGENT_EXIT_REASON_BUDGET,
        text="partial", truncated=True, artifacts=["cid1"], child_run_id="subagent-1",
        tool_rounds=3, usage=cambrian_pb2.GenerateUsage(input_tokens=10, output_tokens=5))
    stub = _Stub(reply)
    c = SubstrateClient(stub=stub, agent_id="worker")
    res = c.spawn_subagent("lease-1", "do it", mode="fork", toolset=["web_*"],
                           output_schema={"type": "object"}, context_summary="## Goal",
                           context_refs=[{"cid": "abc"}], client_spawn_id="k")
    name, req, md = stub.calls[0]
    assert name == "SpawnSubagent"
    assert req.lease_id == "lease-1" and md["x-lease-id"] == "lease-1" and md["x-agent-id"] == "worker"
    assert req.mode == cambrian_pb2.SUBAGENT_MODE_FORK and list(req.toolset) == ["web_*"]
    assert json.loads(req.output_schema.schema_json) == {"type": "object"}
    assert req.context_summary == "## Goal" and req.context_refs[0].cid == "abc"
    assert req.client_spawn_id == "k"
    assert res["status"] == "stopped" and res["exit_reason"] == "budget"
    assert res["truncated"] and res["artifacts"] == ["cid1"] and res["usage"]["input_tokens"] == 10


def test_spawn_subagent_unimplemented_is_none_and_a_refusal_is_data():
    assert SubstrateClient(stub=_Stub(raise_code=grpc.StatusCode.UNIMPLEMENTED)).spawn_subagent("l", "x") is None
    res = SubstrateClient(stub=_Stub(raise_code=grpc.StatusCode.INVALID_ARGUMENT)).spawn_subagent("l", "x")
    assert res == {"error": "nope", "code": "INVALID_ARGUMENT"}


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
    def spawn_subagent(self, *a, **k): ...
    def propose_memory_note(self, *a, **k): ...
    def propose_skill(self, *a, **k): ...


class Plain(CognitiveAgent):
    role = "r"

    def run(self, task):
        return self.think(task)


class Worker(Plain):
    allow_subagents = True
    allow_memory_note_writes = True
    allow_skill_proposals = True


def _offered(agent, ctx=None, notes_served=True, toolset=None, **kw):
    agent.substrate = _VerbSub()
    ctx = ctx or {}
    return _offered_kernel_verbs(agent, AgentTask(text="t", session_token_id="lease", context=ctx),
                                 is_subagent=ctx.get("_subagent") == "1", sub_ctx=ctx,
                                 sub_toolset=toolset, notes_served=notes_served,
                                 spawn=kw.get("spawn"), notes=kw.get("notes"), skills=kw.get("skills"))


def test_verbs_are_off_for_a_plain_agent_and_on_for_a_worker():
    assert _offered(Plain(agent_id="p")) == []
    assert _offered(Worker(agent_id="w")) == ["spawn_subagent", "propose_memory_note", "propose_skill"]
    assert _offered(Plain(agent_id="p"), spawn=True) == ["spawn_subagent"]


def test_verbs_follow_the_kernel_and_the_child_toolset():
    w = Worker(agent_id="w")
    assert "propose_memory_note" not in _offered(w, notes_served=False)
    child = {"_subagent": "1", "_subagent_may_spawn": "false"}
    assert _offered(w, child, toolset=set()) == []
    assert _offered(w, {"_subagent": "1"}, toolset={"propose_memory_note"}) == [
        "spawn_subagent", "propose_memory_note"]
    w.substrate = object()
    assert _offered_kernel_verbs(w, AgentTask(text="t", session_token_id="l"), is_subagent=False,
                                 sub_ctx={}, sub_toolset=None, notes_served=True,
                                 spawn=None, notes=None, skills=None) == []


def test_tool_definitions_and_protocol_sections_follow_the_offer():
    defs, _ = build_tool_definitions(Plain(agent_id="a"), [], include_final=True,
                                     kernel_verbs=["spawn_subagent", "propose_skill"])
    names = [d["name"] for d in defs]
    assert "spawn_subagent" in names and "propose_skill" in names and "propose_memory_note" not in names
    skill = next(d for d in defs if d["name"] == "propose_skill")
    assert "inactive until an operator approves" in skill["description"]
    base = build_output_schema(Plain(agent_id="a"))
    text = build_output_schema(Plain(agent_id="a"), kernel_verbs=["spawn_subagent", "propose_memory_note"])
    assert '"action": "spawn_subagent"' in text and '"action": "propose_memory_note"' in text
    assert "spawn_subagent" not in base
    native = build_output_schema(Plain(agent_id="a"), native_tools=True, kernel_verbs=["spawn_subagent"])
    assert "spawn_subagent" in native and '{"action"' not in native


# ── rendering, fork, notes ────────────────────────────────────────────


def test_subagent_result_is_structured_and_flags_incomplete():
    rendered, evidence = _render_subagent_result(
        {"status": "stopped", "exit_reason": "timeout", "text": "half", "truncated": True})
    assert json.loads(evidence)["exit_reason"] == "timeout"
    assert "<subagent_result kind='data'>" in rendered and "incomplete" in rendered
    done, _ = _render_subagent_result({"status": "done", "exit_reason": "completed", "text": "x"})
    assert "incomplete" not in done
    err, ev = _render_subagent_result({"error": "bad lease", "code": "PERMISSION_DENIED"})
    assert json.loads(ev)["status"] == "failed" and "incomplete" in err


def test_fork_summary_is_built_from_the_engine():
    from cambrian_agent_sdk.working_memory import ToolCard
    e = DefaultContextEngine()
    e.add_step(kind="memory_query", query="q", status="ok", results=[{"text": "t"}])
    card = ToolCard.from_result("web_fetch", {"url": "u"}, {"ok": True})
    e.record_tool_call(card, json.dumps({"ok": True}))
    e._summary = "## Decisions\n- chose X"
    text, refs = e.fork_summary("find the thing")
    assert text.startswith("## Goal\nfind the thing")
    assert "## Summary of earlier turns" in text and "chose X" in text
    assert "web_fetch" in text and "memory_query 'q'" in text
    assert isinstance(refs, list)


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

    def __init__(self, replies, spawn=None, notes=None, note_reply=None):
        self.replies, self.prompts = list(replies), []
        self.spawn_reply, self.notes, self.note_reply = spawn, notes, note_reply
        self.spawn_calls, self.note_calls = [], []

    def generate(self, session_token_id=None, prompt="", **kw):
        self.prompts.append(prompt + json.dumps(kw.get("messages") or [], default=str))
        return self.replies.pop(0)

    def list_tools(self, query="", k=None, full=False):
        return [{"name": "web_fetch", "description": "fetch"}, {"name": "shell", "description": "sh"}]

    def spawn_subagent(self, lease, objective, **kw):
        self.spawn_calls.append((lease, objective, kw))
        return self.spawn_reply

    def get_memory_notes(self, session_token_id=""):
        return self.notes

    def propose_memory_note(self, lease, **kw):
        self.note_calls.append(kw)
        return self.note_reply


def _act(**a):
    return json.dumps(a)


def test_loop_forks_a_subagent_and_places_its_result_in_evidence():
    sub = _LoopSub([
        _act(action="spawn_subagent", objective="count files", mode="fork", toolset=["web_fetch"]),
        _act(action="final_answer", answer="there are 3", type="text"),
    ], spawn={"status": "done", "exit_reason": "completed", "text": "3 files", "truncated": False})
    bot = Worker(agent_id="w")
    bot.substrate = sub
    res = run_think(bot, AgentTask(text="how many files?", session_token_id="lease-1", task_id="t1"),
                    seed_recall=False)
    assert res.data.decode() == "there are 3"
    lease, objective, kw = sub.spawn_calls[0]
    assert lease == "lease-1" and objective == "count files" and kw["mode"] == "fork"
    assert kw["toolset"] == ["web_fetch"] and kw["context_summary"].startswith("## Goal\nhow many files?")
    assert kw["client_spawn_id"]
    # The second turn saw the structured result as data.
    assert "3 files" in sub.prompts[1] and "subagent_result" in sub.prompts[1]


def test_loop_degrades_when_the_kernel_has_no_subagents():
    sub = _LoopSub([
        _act(action="spawn_subagent", objective="x"),
        _act(action="spawn_subagent", objective="x again"),
        _act(action="final_answer", answer="did it myself", type="text"),
    ], spawn=None)
    bot = Worker(agent_id="w")
    bot.substrate = sub
    res = run_think(bot, AgentTask(text="t", session_token_id="l"), seed_recall=False)
    assert res.data.decode() == "did it myself"
    assert len(sub.spawn_calls) == 1  # after UNIMPLEMENTED the verb is not called again
    assert "spawn_subagent is not available" in sub.prompts[1]


def test_plain_agent_is_not_offered_spawn():
    sub = _LoopSub([_act(action="final_answer", answer="ok", type="text")])
    bot = Plain(agent_id="p")
    bot.substrate = sub
    run_think(bot, AgentTask(text="t", session_token_id="l"), seed_recall=False)
    assert "spawn_subagent" not in sub.prompts[0]


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


def test_a_subagent_run_is_shaped_by_its_handoff():
    sub = _LoopSub([_act(action="final_answer", answer="{}", type="text")])
    bot = Worker(agent_id="w")
    bot.substrate = sub
    ctx = {"_subagent": "1", "_subagent_may_spawn": "false", "_subagent_toolset": "web_fetch",
           "_subagent_max_tool_rounds": "2", "_subagent_context_summary": "## Goal\nparent goal",
           "_subagent_output_schema": '{"type":"object"}'}
    run_think(bot, AgentTask(text="child task", session_token_id="child-lease", context=ctx),
              seed_recall=False)
    p = sub.prompts[0]
    assert "parent goal" in p and "parent_context" in p
    assert "web_fetch (system tool)" in p and "shell (system tool)" not in p
    assert '{"type":"object"}' in p
    assert '"action": "spawn_subagent"' not in p  # depth exhausted: not offered
