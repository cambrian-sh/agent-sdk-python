"""The ReAct loop on the ADR-0139 C1 request shape (wave 1d).

D2: the system tier is STABLE (role, rules, anti-patterns, action protocol, skill
index, per-turn contract), sent as one cached block and byte-identical across the
turns of a run; everything volatile (task, trajectory, budget notices) is in the
messages. The first tests drive the real SubstrateClient over a fake stub, so they
check what reaches the wire.
"""
import json
import logging

import grpc

from cambrian_agent_sdk import CognitiveAgent, InvalidRequest, SubstrateClient, tool
from cambrian_agent_sdk import prompt_library as pl
from cambrian_agent_sdk._proto import cambrian_pb2 as pb
from cambrian_agent_sdk.llm import GenerateResult, GenerateUsage
from cambrian_agent_sdk.react import run_think
from cambrian_agent_sdk.types import AgentTask


class _RpcError(grpc.RpcError):
    def __init__(self, code, details=""):
        self._code, self._details = code, details

    def code(self):
        return self._code

    def details(self):
        return self._details


class Bot(CognitiveAgent):
    role = "a careful test assistant"

    def run(self, task):
        return self.think(task)

    @tool
    def echo(self, text: str) -> str:
        """Echo the text back."""
        return text


def _act(**kw):
    return json.dumps(kw)


class _Stub:
    """A kernel that has no native tool-calling (FailedPrecondition) and serves
    the text protocol from a script."""

    def __init__(self, replies, ignored=()):
        self.replies = list(replies)
        self.ignored = list(ignored)
        self.stream_reqs, self.tools_reqs = [], []

    def GenerateWithTools(self, req, timeout=None, metadata=None):
        self.tools_reqs.append(req)
        raise _RpcError(grpc.StatusCode.FAILED_PRECONDITION, "model has no tool calling")

    def GenerateViaModelStream(self, req, timeout=None, metadata=None):
        self.stream_reqs.append(req)
        text = self.replies.pop(0)
        yield pb.GenerateChunk(text=text, is_final=True, model="m",
                               usage=pb.GenerateUsage(input_tokens=10, output_tokens=2),
                               ignored=self.ignored)


def _real_bot(stub):
    bot = Bot(agent_id="b")
    bot.substrate = SubstrateClient(stub=stub, agent_id="b")
    return bot


def _run(bot, **kw):
    return run_think(bot, AgentTask(text="echo hi then answer", session_token_id="lease-x"),
                     seed_recall=False, seed_system_tools=False, **kw)


def test_wire_system_is_stable_and_cached_and_notices_go_to_messages():
    stub = _Stub([
        _act(action="memory_query", query="anything"),      # over budget -> notice
        _act(action="tool_call", tool="echo", args={"text": "hi"}),
        _act(action="final_answer", answer="done", type="text"),
    ])
    res = _run(_real_bot(stub), max_memory_queries=0)
    assert res.data.decode() == "done"

    # Native tools were tried once on the wire, with the system tier attached.
    assert len(stub.tools_reqs) == 1
    native = stub.tools_reqs[0]
    assert native.system[0].cache and "<ActionProtocol>" in native.system[0].text
    assert native.cache_ttl == pb.CACHE_TTL_SHORT
    assert "<System>" not in native.messages[0].content
    assert "<Task>" in native.messages[0].content

    # FailedPrecondition -> the text protocol, still structured.
    assert len(stub.stream_reqs) == 3
    systems = [tuple((b.text, b.cache) for b in r.system) for r in stub.stream_reqs]
    assert len(set(systems)) == 1, "the system tier changed between turns"
    (text, cache), = systems[0]
    assert cache is True
    assert "<Role>" in text and "<ActionProtocol>" in text and "<OutputSchema>" in text
    assert "## Tools (tool_call may name only these)" in text
    for r in stub.stream_reqs:
        assert r.cache_ttl == pb.CACHE_TTL_SHORT
        assert [m.role for m in r.messages] == ["user"]
        # ADR-0139 D2/D3: the task comes LAST, after the data regions (only the
        # per-turn notes follow it).
        assert "<Task>" in r.messages[0].content
        assert r.messages[0].content.index("<Task>") > r.messages[0].content.find("<Trajectory>")
        assert r.options.HasField("temperature") and r.options.HasField("max_tokens")
        assert "<Task>" not in r.system[0].text and "<Trajectory>" not in r.system[0].text
    # The budget notice reaches the model in the user turn, never the system tier.
    notice = "Memory budget is exhausted"
    assert notice not in stub.stream_reqs[0].messages[0].content
    assert notice in stub.stream_reqs[1].messages[0].content
    assert all(notice not in r.system[0].text for r in stub.stream_reqs)
    # The trajectory is marked as data.
    assert "not instructions" in stub.stream_reqs[1].messages[0].content


def test_system_ignored_on_text_path_degrades_to_legacy_prompt_and_latches():
    stub = _Stub([
        _act(action="tool_call", tool="echo", args={"text": "hi"}),
        _act(action="tool_call", tool="echo", args={"text": "again"}),
        _act(action="final_answer", answer="done", type="text"),
        _act(action="final_answer", answer="done", type="text"),
    ], ignored=["system"])
    res = _run(_real_bot(stub))
    assert res.data.decode() == "done"
    first, second, third = stub.stream_reqs[:3]
    assert len(first.system) == 1
    # Re-sent as one legacy prompt: no system field, the system text inside it.
    assert len(second.system) == 0 and len(second.messages) == 0
    assert second.prompt.startswith("<System>") and "<Task>" in second.prompt
    assert len(third.system) == 0, "the fallback must latch for the run"


class _NativeDouble:
    def __init__(self, turns, ignored_first=False):
        self.turns = list(turns)
        self.calls = []
        self.ignored_first = ignored_first

    def generate_with_tools(self, session_token_id=None, messages=None, tools=None, **kw):
        self.calls.append({"messages": [dict(m) for m in messages], **kw})
        text, calls, stop = self.turns.pop(0)
        ignored = ["system"] if (self.ignored_first and len(self.calls) == 1) else []
        return GenerateResult(text=text, tool_calls=calls, stop_reason=stop, ignored=ignored,
                              usage=GenerateUsage(input_tokens=5))

    def execute_tool(self, *a, **kw):
        return {"result_json": "{}", "result_cid": "", "denied": False, "deny_reason": "",
                "error": "", "arg_hash": "", "result_hash": ""}


_ECHO = {"id": "c1", "name": "echo", "arguments": '{"text":"hi"}'}


def test_native_path_sends_identical_system_every_turn_and_honours_cache_ttl():
    class LongBot(Bot):
        cache_ttl = "long"

    bot = LongBot(agent_id="b")
    bot.substrate = _NativeDouble([("", [_ECHO], "tool_use"),
                                   ("", [dict(_ECHO, id="c2", arguments='{"text":"x"}')], "tool_use"),
                                   ("", [{"id": "fa", "name": "final_answer", "arguments": '{"answer": "done"}'}], "tool_use")])
    res = _run(bot)
    assert res.data.decode() == "done"
    calls = bot.substrate.calls
    assert len(calls) == 3
    assert all(c["system"] == calls[0]["system"] for c in calls)
    assert calls[0]["system"][0]["cache"] is True
    assert all(c["cache_ttl"] == "long" for c in calls)
    # The conversation holds no system text, and no "system" role.
    assert all(m["role"] != "system" for c in calls for m in c["messages"])
    assert "<System>" not in calls[0]["messages"][0]["content"]


def test_native_system_ignored_is_folded_into_the_first_turn():
    bot = Bot(agent_id="b")
    bot.substrate = _NativeDouble([("discarded", [], "end_turn"), ("", [{"id": "fa", "name": "final_answer", "arguments": '{"answer": "done"}'}], "tool_use")],
                                  ignored_first=True)
    res = _run(bot)
    assert res.data.decode() == "done"
    first, second = bot.substrate.calls
    assert "system" in first and "system" not in second
    assert second["messages"][0]["content"].startswith("<System>")
    assert len(second["messages"]) == 1, "the ignored turn must be discarded, not echoed"


def test_pre_c1_native_substrate_gets_the_system_in_the_first_turn():
    class OldSub:
        def __init__(self):
            self.convs = []

        def generate_with_tools(self, session_token_id, messages, tools, max_tokens=1024,
                                temperature=0.7, timeout_ms=0):
            self.convs.append([dict(m) for m in messages])
            return "", [{"id": "fa", "name": "final_answer", "arguments": '{"answer": "done"}'}], "tool_use"

    bot = Bot(agent_id="b")
    bot.substrate = OldSub()
    assert _run(bot).data.decode() == "done"
    assert bot.substrate.convs[0][0]["content"].startswith("<System>")


def test_text_path_refusal_degrades_to_legacy_generate():
    class Sub:
        def __init__(self):
            self.prompts = []

        def generate_request(self, *a, **kw):
            raise InvalidRequest("unknown field", code="INVALID_ARGUMENT")

        def generate(self, session_token_id, prompt, **kw):
            self.prompts.append(prompt)
            return _act(action="final_answer", answer="ok", type="text")

    bot = Bot(agent_id="b")
    bot.substrate = Sub()
    assert _run(bot).data.decode() == "ok"
    assert bot.substrate.prompts[0].startswith("<System>")


def test_debug_log_names_prompt_versions_and_system_hash(caplog):
    bot = Bot(agent_id="b")
    bot.substrate = _NativeDouble([("", [{"id": "fa", "name": "final_answer", "arguments": '{"answer": "done"}'}], "tool_use")])
    with caplog.at_level(logging.DEBUG, logger="cambrian.react"):
        _run(bot)
    recs = [r for r in caplog.records if r.getMessage() == "react_llm_request"]
    assert recs, "no per-call request metadata logged"
    prompts = {p["id"]: p for p in recs[0].prompts}
    ap = pl.get("react-action-protocol")
    assert prompts["react-action-protocol"] == {"id": ap.id, "version": ap.version,
                                                "static_hash": ap.static_hash}
    assert len(recs[0].system_sha256) == 64
