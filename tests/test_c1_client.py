"""SubstrateClient on the ADR-0139 C1 agent-plane fields (wave 1d).

A fake stub records the real protobuf requests the client builds, so these tests
assert the WIRE shape (HasField, repeated fields, enums), not a Python double's
kwargs. Responses are real protobufs too, so the usage/ignored mapping is exercised
on the same types the kernel sends.
"""
import grpc
import pytest

from cambrian_agent_sdk import (
    GenerateError,
    GenerateResult,
    GenerationResourceExhausted,
    InvalidRequest,
    ModelUnavailable,
    SchemaValidationFailed,
    SubstrateClient,
    SystemBlock,
    ToolCallingUnsupported,
)
from cambrian_agent_sdk._proto import cambrian_pb2 as pb


class _RpcError(grpc.RpcError):
    def __init__(self, code, details=""):
        self._code, self._details = code, details

    def code(self):
        return self._code

    def details(self):
        return self._details


def _usage(**kw):
    base = dict(input_tokens=100, output_tokens=7, cache_read_tokens=80,
                cache_write_tokens=0, cache_write_known=False, estimated=False)
    base.update(kw)
    return pb.GenerateUsage(**base)


class _Stub:
    def __init__(self, chunks=None, tools_resp=None, error=None, error_after=None):
        self.stream_reqs, self.tools_reqs = [], []
        self._chunks = chunks if chunks is not None else [
            pb.GenerateChunk(text="hel"),
            pb.GenerateChunk(text="lo", is_final=True, usage_tokens=107, usage=_usage(),
                             ignored=["temperature"], stop_reason="end_turn", model="m-1"),
        ]
        self._tools_resp = tools_resp
        self._error = error
        self._error_after = error_after

    def GenerateViaModelStream(self, req, timeout=None, metadata=None):
        self.stream_reqs.append(req)
        if self._error is not None:
            raise self._error
        for i, c in enumerate(self._chunks):
            if self._error_after is not None and i == self._error_after[0]:
                raise self._error_after[1]
            yield c

    def GenerateWithTools(self, req, timeout=None, metadata=None):
        self.tools_reqs.append(req)
        if self._error is not None:
            raise self._error
        return self._tools_resp or pb.GenerateWithToolsResponse(
            text="", stop_reason="tool_use", usage_tokens=50, usage=_usage(cache_write_tokens=9,
                                                                           cache_write_known=True),
            model="m-2",
            tool_calls=[pb.ModelToolCallProto(id="c1", name="echo", arguments_json='{"a":1}')])


def _client(stub):
    return SubstrateClient(stub=stub, agent_id="ag")


# ── request shape ────────────────────────────────────────────────────────────

def test_structured_stream_request_carries_system_messages_schema_options():
    stub = _Stub()
    c = _client(stub)
    res = c.generate_request(
        "lease-1",
        system=[{"text": "STABLE RULES", "cache": True}, SystemBlock("ctx")],
        messages=[{"role": "user", "content": "the task"}],
        response_schema={"name": "answer", "schema": {"type": "object"}, "strict": True},
        cache_ttl="short", max_tokens=256, temperature=0.0, stop=["</done>"], strict=True)
    req = stub.stream_reqs[0]
    assert [(b.text, b.cache) for b in req.system] == [("STABLE RULES", True), ("ctx", False)]
    assert [(m.role, m.content) for m in req.messages] == [("user", "the task")]
    assert req.HasField("response_schema")
    assert req.response_schema.name == "answer" and req.response_schema.strict
    assert req.response_schema.schema_json == '{"type":"object"}'
    assert req.cache_ttl == pb.CACHE_TTL_SHORT
    # Field presence: an explicit 0 temperature is SENT, not dropped as the default.
    assert req.options.HasField("temperature") and req.options.temperature == 0.0
    assert req.options.HasField("max_tokens") and req.options.max_tokens == 256
    assert list(req.options.stop_sequences) == ["</done>"] and req.options.strict
    assert req.lease_id == "lease-1"
    # Pre-C1 kernel compatibility: the flattened text rides in the deprecated field.
    assert req.prompt == "STABLE RULES\n\nctx\n\nthe task"
    assert res.text == "hello"


def test_unset_options_are_absent_not_zero():
    stub = _Stub()
    _client(stub).generate_request("l", messages=[{"role": "user", "content": "x"}])
    opts = stub.stream_reqs[0].options
    assert not opts.HasField("temperature") and not opts.HasField("max_tokens")
    assert stub.stream_reqs[0].cache_ttl == pb.CACHE_TTL_UNSPECIFIED


def test_legacy_generate_prompt_is_unchanged_on_the_wire():
    stub = _Stub()
    text = _client(stub).generate("tok", "just a prompt")
    req = stub.stream_reqs[0]
    assert text == "hello"
    assert req.prompt == "just a prompt"
    assert len(req.system) == 0 and len(req.messages) == 0
    assert not req.HasField("response_schema")
    assert req.options.max_tokens == 1024 and req.options.HasField("temperature")
    assert abs(req.options.temperature - 0.7) < 1e-6
    assert req.session_token_id == "tok" and req.lease_id == "tok"


def test_prompt_with_system_becomes_one_user_turn():
    stub = _Stub()
    _client(stub).generate("tok", "do it", system="be brief")
    req = stub.stream_reqs[0]
    assert [(m.role, m.content) for m in req.messages] == [("user", "do it")]
    assert req.system[0].text == "be brief"


def test_system_role_message_is_refused_before_the_rpc():
    stub = _Stub()
    with pytest.raises(InvalidRequest):
        _client(stub).generate_request("l", messages=[{"role": "system", "content": "x"}])
    with pytest.raises(InvalidRequest):
        _client(stub).generate_with_tools("l", [{"role": "system", "content": "x"}], [])
    assert not stub.stream_reqs and not stub.tools_reqs


def test_bad_cache_ttl_is_refused():
    with pytest.raises(InvalidRequest):
        _client(_Stub()).generate_request("l", prompt="x", cache_ttl="forever")


def test_tools_request_carries_system_and_cache_ttl():
    stub = _Stub()
    _client(stub).generate_with_tools(
        "l", [{"role": "user", "content": "go"}],
        [{"name": "echo", "description": "d", "parameters": {"type": "object"}}],
        system=[{"text": "S", "cache": True}], cache_ttl="long", temperature=None)
    req = stub.tools_reqs[0]
    assert [(b.text, b.cache) for b in req.system] == [("S", True)]
    assert req.cache_ttl == pb.CACHE_TTL_LONG
    assert not req.options.HasField("temperature")
    assert req.tools[0].parameters_json == '{"type": "object"}'


# ── response mapping ─────────────────────────────────────────────────────────

def test_final_chunk_maps_usage_ignored_model_stop_reason():
    res = _client(_Stub()).generate_request("l", messages=[{"role": "user", "content": "x"}])
    assert res.usage.input_tokens == 100 and res.usage.cache_read == 80
    assert res.usage.known is False and res.usage.estimated is False
    assert res.ignored == ["temperature"] and not res.honoured("temperature")
    assert res.model == "m-1" and res.stop_reason == "end_turn" and res.usage_tokens == 107
    assert res.legacy_kernel is False


def test_generate_stream_reports_metadata_through_result():
    out = GenerateResult()
    chunks = list(_client(_Stub()).generate_stream("l", "p", result=out))
    assert chunks == ["hel", "lo"] and out.text == "hello" and out.model == "m-1"


def test_tools_result_is_tuple_compatible_and_carries_usage():
    res = _client(_Stub()).generate_with_tools("l", [{"role": "user", "content": "go"}], [],
                                              system="S")
    text, calls, stop = res
    assert calls == [{"id": "c1", "name": "echo", "arguments": '{"a":1}'}]
    assert stop == "tool_use" and res["model"] == "m-2"
    assert res.usage.cache_write == 9 and res.usage.known
    assert res.ignored == []


def test_legacy_kernel_without_usage_is_detected():
    # A kernel that predates C1 sets no usage block. On the stream nothing was lost
    # (the flattened prompt was read), so only the flag is set.
    old_chunks = [pb.GenerateChunk(text="ok", is_final=True, usage_tokens=3)]
    res = _client(_Stub(chunks=old_chunks)).generate_request(
        "l", system="S", messages=[{"role": "user", "content": "x"}])
    assert res.legacy_kernel and "system" not in res.ignored
    # On GenerateWithTools the system blocks WERE dropped: reported as ignored.
    old = pb.GenerateWithToolsResponse(text="hi", stop_reason="end_turn", usage_tokens=3)
    res = _client(_Stub(tools_resp=old)).generate_with_tools(
        "l", [{"role": "user", "content": "x"}], [], system="S")
    assert res.legacy_kernel and "system" in res.ignored


def test_plain_prompt_on_old_kernel_is_not_flagged():
    old_chunks = [pb.GenerateChunk(text="ok", is_final=True)]
    res = _client(_Stub(chunks=old_chunks)).generate_request("l", prompt="x")
    assert not res.legacy_kernel


# ── status codes ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("code,exc", [
    (grpc.StatusCode.INVALID_ARGUMENT, InvalidRequest),
    (grpc.StatusCode.ABORTED, SchemaValidationFailed),
    (grpc.StatusCode.RESOURCE_EXHAUSTED, GenerationResourceExhausted),
    (grpc.StatusCode.UNAVAILABLE, ModelUnavailable),
    (grpc.StatusCode.PERMISSION_DENIED, GenerateError),
])
def test_status_codes_map_to_typed_errors(code, exc):
    for call in (
        lambda c: c.generate_request("l", prompt="x"),
        lambda c: c.generate_with_tools("l", [{"role": "user", "content": "x"}], []),
    ):
        with pytest.raises(exc) as info:
            call(_client(_Stub(error=_RpcError(code, "why"))))
        err = info.value
        assert isinstance(err, grpc.RpcError), "callers catching RpcError keep working"
        assert err.code() == code and err.details() == "why"


@pytest.mark.parametrize("code", [grpc.StatusCode.FAILED_PRECONDITION,
                                  grpc.StatusCode.UNIMPLEMENTED])
def test_no_native_tools_is_the_fallback_signal(code):
    with pytest.raises(ToolCallingUnsupported):
        _client(_Stub(error=_RpcError(code))).generate_with_tools(
            "l", [{"role": "user", "content": "x"}], [])


def test_error_mid_stream_is_mapped():
    stub = _Stub(error_after=(1, _RpcError(grpc.StatusCode.UNAVAILABLE, "gone")))
    with pytest.raises(ModelUnavailable):
        _client(stub).generate("l", "p")
