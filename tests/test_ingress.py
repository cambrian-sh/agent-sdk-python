"""Tests for IngressAgent (ADR-0090) — the two one-way flows.

The acknowledgement shapes here are a CONTRACT with the kernel's delivery path
(`internal/ingress/transport.go: interpret`). If these change, that changes too.
"""

import json

import pytest

from cambrian_agent_sdk import IngressAgent, PermanentDeliveryError
from cambrian_agent_sdk.types import AgentTask


class FakeIngress(IngressAgent):
    def __init__(self, **kwargs):
        super().__init__(agent_id="telegram_ingress", **kwargs)
        self.sent = []
        self.raise_permanent = False
        self.raise_transient = False

    def listen(self):  # pragma: no cover — never run in these tests
        raise AssertionError("listen must not be called by the delivery path")

    def on_deliver(self, recipient, text, conversation_id):
        if self.raise_permanent:
            raise PermanentDeliveryError("bot was blocked by the user")
        if self.raise_transient:
            raise RuntimeError("429 too many requests")
        self.sent.append((recipient, text, conversation_id))


def delivery(**over):
    body = {
        "kind": "ingress.deliver",
        "conversation_id": "conv-7",
        "recipient": "tg:12345",
        "text": "Which dates?",
        "txn_id": "msg-1",
    }
    body.update(over)
    return AgentTask(data=json.dumps(body).encode())


def ack_of(result):
    return json.loads(result.data)


# ── outbound ───────────────────────────────────────────────────────────────

def test_delivery_reaches_on_deliver():
    ing = FakeIngress()
    ack = ack_of(ing.run(delivery()))

    assert ack["status"] == "sent"
    assert ing.sent == [("tg:12345", "Which dates?", "conv-7")]


def test_permanent_failure_is_labelled_so_the_kernel_stops_retrying():
    ing = FakeIngress()
    ing.raise_permanent = True
    ack = ack_of(ing.run(delivery()))

    assert ack["status"] == "failed"
    assert ack["permanent"] is True
    assert "blocked" in ack["error"]


def test_unlabelled_failure_is_transient():
    """Anything that is not PermanentDeliveryError may be retried — wrongly
    retrying costs one duplicate attempt, wrongly giving up loses the message."""
    ing = FakeIngress()
    ing.raise_transient = True
    ack = ack_of(ing.run(delivery()))

    assert ack["status"] == "failed"
    assert ack["permanent"] is False


def test_malformed_delivery_is_permanent():
    """A retry would send exactly the same broken payload, so it is terminal."""
    ing = FakeIngress()

    for bad in (delivery(recipient=""), delivery(text=""), delivery(kind="something.else")):
        ack = ack_of(ing.run(bad))
        assert ack["status"] == "failed"
        assert ack["permanent"] is True
    assert ing.sent == []


def test_undecodable_payload_does_not_crash_the_ingress():
    ing = FakeIngress()
    ack = ack_of(ing.run(AgentTask(data=b"not json")))
    assert ack["status"] == "failed"


# ── inbound ────────────────────────────────────────────────────────────────

def test_receive_queues_a_signal_and_returns_immediately():
    ing = FakeIngress()
    ing.receive("tg:12345", "book me a flight")

    payload, raw = ing._signal_queue.get_nowait()
    assert payload["external_id"] == "tg:12345"
    assert payload["text"] == "book me a flight"
    assert raw == "book me a flight"


def test_receive_carries_extra_fields():
    ing = FakeIngress()
    ing.receive("tg:1", "hi", username="afsin")
    payload, _ = ing._signal_queue.get_nowait()
    assert payload["username"] == "afsin"


def test_receive_refuses_a_message_from_nobody():
    """An unaddressed message could open a conversation nothing can ever reply to."""
    ing = FakeIngress()
    with pytest.raises(ValueError):
        ing.receive("", "hello")
    with pytest.raises(ValueError):
        ing.receive("tg:1", "")


# ── the shape itself ───────────────────────────────────────────────────────

def test_an_ingress_cannot_declare_its_own_surface_or_recipient():
    """INV-5, enforced by omission rather than by documentation: there is no API
    here for a daemon to name its own privilege level or choose who reads a
    message. The surface comes from the operator's registration; the recipient is
    resolved by the kernel from the conversation."""
    surface_like = [n for n in dir(IngressAgent) if "surface" in n.lower()]
    assert surface_like == []
    assert not hasattr(IngressAgent, "deliver_to")


def test_daemon_loop_delegates_to_listen():
    """The runtime supervises daemon_loop; authors write listen()."""
    calls = []

    class L(IngressAgent):
        def listen(self):
            calls.append(1)

        def on_deliver(self, recipient, text, conversation_id):  # pragma: no cover
            pass

    L(agent_id="x").daemon_loop()
    assert calls == [1]


def test_trait_is_daemon_so_the_kernel_supervises_it():
    assert FakeIngress().trait == "daemon"
