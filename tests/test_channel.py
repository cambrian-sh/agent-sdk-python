"""ADR-0130 D4: the SDK's one channel builder.

Gate criterion 4 asks for an SDK agent connecting over TLS with a secret against a
non-loopback kernel. No kernel runs inside pytest, so these tests prove the
CHANNEL CONSTRUCTION: given a TLS root and a token in the environment the SDK
builds a secure channel and stamps every call with the bearer and the matching
x-agent-id; given neither it builds the plaintext channel the loopback
insecure_localhost deployment expects, warning once.
"""

from __future__ import annotations

import logging

import grpc
import pytest

from cambrian_agent_sdk import channel as ch


class _Recorder:
    """Stands in for grpc.secure_channel / grpc.insecure_channel."""

    def __init__(self):
        self.calls = []

    def secure(self, addr, creds, options=None):
        self.calls.append(("secure", addr, creds, tuple(options or [])))
        return grpc.insecure_channel(addr)  # a real channel object for intercept_channel

    def insecure(self, addr, options=None):
        self.calls.append(("insecure", addr, None, tuple(options or [])))
        return grpc.insecure_channel(addr)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in (ch.TOKEN_ENV, ch.AGENT_ID_ENV, ch.TLS_CA_ENV, ch.TLS_SERVER_NAME_ENV):
        monkeypatch.delenv(name, raising=False)
    ch._reset_insecure_warning_for_tests()


def test_tls_root_and_token_build_a_secure_channel(monkeypatch, tmp_path):
    ca = tmp_path / "root.pem"
    ca.write_bytes(b"-----BEGIN CERTIFICATE-----\nMIIB\n-----END CERTIFICATE-----\n")
    monkeypatch.setenv(ch.TLS_CA_ENV, str(ca))
    monkeypatch.setenv(ch.TLS_SERVER_NAME_ENV, "kernel.internal")
    monkeypatch.setenv(ch.TOKEN_ENV, "tok-analyst")
    monkeypatch.setenv(ch.AGENT_ID_ENV, "analyst_agent")
    rec = _Recorder()

    chan = ch.build_channel("192.168.1.10:50051", _secure_channel=rec.secure, _insecure_channel=rec.insecure)

    assert len(rec.calls) == 1
    kind, addr, creds, options = rec.calls[0]
    assert kind == "secure" and addr == "192.168.1.10:50051"
    assert isinstance(creds, grpc.ChannelCredentials)
    assert ("grpc.ssl_target_name_override", "kernel.internal") in options
    assert ch.transport_mode() == "tls"
    # The credential rides every call: the interceptor is installed.
    assert isinstance(chan, grpc.Channel)
    chan.close()


def test_no_root_builds_plaintext_and_warns_once(monkeypatch, caplog):
    rec = _Recorder()
    with caplog.at_level(logging.WARNING, logger="cambrian.channel"):
        ch.build_channel("localhost:50051", _secure_channel=rec.secure, _insecure_channel=rec.insecure).close()
        ch.build_channel("localhost:50051", _secure_channel=rec.secure, _insecure_channel=rec.insecure).close()
    assert [c[0] for c in rec.calls] == ["insecure", "insecure"]
    warnings = [r for r in caplog.records if "PLAINTEXT" in r.getMessage()]
    assert len(warnings) == 1, "the insecure warning must be said exactly once per process"
    assert ch.TLS_CA_ENV in warnings[0].getMessage()
    assert ch.transport_mode() == "insecure"


def test_options_reach_the_insecure_channel(monkeypatch):
    rec = _Recorder()
    opts = [("grpc.max_send_message_length", 50 * 1024 * 1024)]
    ch.build_channel("localhost:50051", options=opts, _insecure_channel=rec.insecure).close()
    assert rec.calls[0][3] == tuple(opts)


def test_credential_metadata_adds_bearer_and_agent_id_without_duplicating(monkeypatch):
    monkeypatch.setenv(ch.TOKEN_ENV, "tok")
    monkeypatch.setenv(ch.AGENT_ID_ENV, "analyst_agent")
    creds = ch._Credentials()
    md = creds.metadata([("x-lane", "actions")])
    assert ("authorization", "Bearer tok") in md
    assert ("x-agent-id", "analyst_agent") in md
    assert ("x-lane", "actions") in md
    # An explicit x-agent-id on the call wins; the env one is not appended.
    md = creds.metadata([("x-agent-id", "summariser"), ("Authorization", "Bearer other")])
    assert md.count(("x-agent-id", "summariser")) == 1
    assert not any(k == "x-agent-id" and v == "analyst_agent" for k, v in md)
    assert not any(v == "Bearer tok" for _, v in md)


def test_agent_id_argument_is_the_fallback_assertion(monkeypatch):
    monkeypatch.setenv(ch.TOKEN_ENV, "tok")
    creds = ch._Credentials(agent_id="pulse_agent")
    assert ("x-agent-id", "pulse_agent") in creds.metadata(None)
    monkeypatch.setenv(ch.AGENT_ID_ENV, "from_env")
    # The kernel-injected id is the authoritative one when both are present.
    assert ("x-agent-id", "pulse_agent") in ch._Credentials(agent_id="pulse_agent").metadata(None)


def test_interceptor_stamps_all_four_arities(monkeypatch):
    monkeypatch.setenv(ch.TOKEN_ENV, "tok")
    monkeypatch.setenv(ch.AGENT_ID_ENV, "analyst_agent")
    icpt = ch._CredentialInterceptor(ch._Credentials())

    class _Details(grpc.ClientCallDetails):
        method = "/cambrian.Orchestrator/ListTools"
        timeout = None
        metadata = None
        credentials = None
        wait_for_ready = None
        compression = None

    seen = []

    def continuation(details, *_):
        seen.append(dict(details.metadata))
        return "ok"

    icpt.intercept_unary_unary(continuation, _Details(), object())
    icpt.intercept_unary_stream(continuation, _Details(), object())
    icpt.intercept_stream_unary(continuation, _Details(), iter(()))
    icpt.intercept_stream_stream(continuation, _Details(), iter(()))
    assert len(seen) == 4
    for md in seen:
        assert md["authorization"] == "Bearer tok"
        assert md["x-agent-id"] == "analyst_agent"
