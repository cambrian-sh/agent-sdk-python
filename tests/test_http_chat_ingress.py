"""Tests for HTTPChatIngress — a synchronous protocol over a fire-and-forget core.

The contract exercised here is the one the airline benchmark driver already
speaks (`airline_driver_chat.py`): POST /open, /turn, /close, with the reply in
the /turn response body. If these break, that driver breaks.
"""

import json
import threading
import time
import urllib.request

import pytest

from cambrian_agent_sdk import HTTPChatIngress
from cambrian_agent_sdk.http_chat_ingress import EXTERNAL_PREFIX


class Harness(HTTPChatIngress):
    """Captures what would have gone to the kernel, and lets a test reply."""

    def __init__(self, **kw):
        super().__init__(agent_id="test_chat_ingress", **kw)
        self.inbound = []

    def receive(self, external_id, text, **extra):
        self.inbound.append((external_id, text, extra))

    def reply(self, conv, text):
        """Stand in for the kernel delivering an agent's reply."""
        self.on_deliver(EXTERNAL_PREFIX + conv, text, conv)


@pytest.fixture()
def served():
    ing = Harness()
    ing.params = {"addr": "127.0.0.1:8899", "turn_timeout": 3}
    t = threading.Thread(target=ing.listen, daemon=True)
    t.start()
    for _ in range(100):  # wait for the listener rather than sleeping a fixed time
        try:
            post("/open", {"conversation_id": "warmup"})
            break
        except Exception:
            time.sleep(0.02)
    yield ing
    ing.stop()


def post(path, payload, timeout=6.0):
    req = urllib.request.Request(
        "http://127.0.0.1:8899" + path,
        data=json.dumps(payload).encode(),
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read() or b"{}")


# ── the driver's contract ──────────────────────────────────────────────────

def test_turn_blocks_until_the_reply_arrives(served):
    """The whole reason correlation lives in the ingress: the caller wants the
    reply in the HTTP response, while the core is fire-and-forget."""
    post("/open", {"conversation_id": "airline-0", "policy": "be helpful"})

    result = {}

    def do_turn():
        result.update(post("/turn", {"conversation_id": "airline-0", "message": "I need a flight"}))

    t = threading.Thread(target=do_turn)
    t.start()

    # The request is in flight and unanswered until something delivers.
    time.sleep(0.2)
    assert result == {}, "the turn must not answer before a reply exists"

    served.reply("airline-0", "Which dates?")
    t.join(timeout=5)

    assert result["reply"] == "Which dates?"
    assert result["error"] == ""


def test_policy_rides_the_message_and_the_kernel_applies_it_once(served):
    post("/open", {"conversation_id": "airline-1", "policy": "AIRLINE POLICY"})

    t = threading.Thread(target=lambda: post("/turn", {"conversation_id": "airline-1", "message": "hi"}))
    t.start()
    time.sleep(0.2)
    served.reply("airline-1", "hello")
    t.join(timeout=5)

    external_id, text, extra = served.inbound[-1]
    assert external_id == "chat:airline-1"
    assert text == "hi"
    # Sent every turn; the KERNEL applies it only at open, so a later turn cannot
    # rewrite the standing instructions mid-transcript.
    assert extra["policy"] == "AIRLINE POLICY"


def test_a_turn_with_no_reply_times_out_with_an_error_not_a_hang(served):
    post("/open", {"conversation_id": "airline-2"})
    r = post("/turn", {"conversation_id": "airline-2", "message": "hello"}, timeout=10)

    assert r["reply"] == ""
    assert "no reply" in r["error"]


def test_second_reply_is_queued_for_the_next_turn(served):
    """An agent may speak twice. HTTP can only carry one reply per request, so the
    extra is kept for the next one rather than dropped — the limitation is the
    protocol's, and losing the message would be ours."""
    post("/open", {"conversation_id": "airline-3"})

    t = threading.Thread(target=lambda: post("/turn", {"conversation_id": "airline-3", "message": "q"}))
    t.start()
    time.sleep(0.2)
    served.reply("airline-3", "Checking now...")
    served.reply("airline-3", "3 options, from 89 EUR")
    t.join(timeout=5)

    second = post("/turn", {"conversation_id": "airline-3", "message": "and?"})
    assert second["reply"] == "3 options, from 89 EUR"


def test_conversations_do_not_cross(served):
    post("/open", {"conversation_id": "airline-a"})
    post("/open", {"conversation_id": "airline-b"})

    out = {}
    ta = threading.Thread(target=lambda: out.update(a=post("/turn", {"conversation_id": "airline-a", "message": "qa"})))
    ta.start()
    time.sleep(0.2)
    # A reply for B must not satisfy A's pending request.
    served.reply("airline-b", "for B")
    time.sleep(0.2)
    assert "a" not in out, "a reply for another conversation must not answer this one"

    served.reply("airline-a", "for A")
    ta.join(timeout=5)
    assert out["a"]["reply"] == "for A"


def test_missing_fields_are_rejected(served):
    r = post("/turn", {"conversation_id": "airline-4", "message": ""})
    assert r["error"]
    with pytest.raises(Exception):
        post("/turn", {"message": "no conversation id"})


def test_close_is_accepted_and_forgets_the_conversation(served):
    post("/open", {"conversation_id": "airline-5", "policy": "p"})
    assert post("/close", {"conversation_id": "airline-5"})["ok"] is True
    assert "airline-5" not in served._policies


def test_external_ids_are_namespaced(served):
    """Every id this ingress claims must fall inside the namespace it registered
    with; the prefix is what makes that check meaningful."""
    post("/open", {"conversation_id": "airline-6"})
    t = threading.Thread(target=lambda: post("/turn", {"conversation_id": "airline-6", "message": "x"}))
    t.start()
    time.sleep(0.2)
    served.reply("airline-6", "y")
    t.join(timeout=5)

    assert served.inbound[-1][0].startswith(EXTERNAL_PREFIX)
