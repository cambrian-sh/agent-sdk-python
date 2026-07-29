"""Ingress agents — the points where the outside world enters Cambrian (ADR-0090).

Telegram, a webhook receiver, a websocket listener, an inbound REST API. Chat is
one payload type riding through one; the concept is the entry point, not the
payload.

An ingress is a **transceiver**, not a request/response handler. Two independent
fire-and-forget flows:

  inbound   the outside sends something -> ``receive()`` -> the kernel
  outbound  something inside speaks     -> ``on_deliver()`` -> the outside

Nothing correlates them. One inbound message may produce no replies, one, or
five, seconds or hours apart, and the kernel may speak with no inbound message at
all. That is not an implementation detail: the Telegram Bot API acknowledges an
update and then sends replies through a *separate* outbound call, so a
request/response ingress cannot be built against it correctly.

Neither existing base class fits, which is why this one exists:

  * ``DaemonAgent`` produces signals but is never called — no delivery path.
  * ``CognitiveAgent`` is called but produces no signals — no inbound path.

``IngressAgent`` does both: it serves the gRPC endpoint the kernel delivers to
AND holds the signal stream it pushes inbound traffic onto.

What it deliberately does NOT have is any way to declare its own surface or to
name a delivery recipient. The surface comes from an out-of-band registration the
operator made, and the recipient is resolved by the kernel from the conversation.
A daemon is a black box, and a black box that asserts its own privilege level is
not a security boundary (INV-5). Leaving those out of the API is a stronger
guarantee than documenting that they must not be used.
"""

from __future__ import annotations

import json
import logging
import threading
from typing import Optional

from .base import Agent
from .types import AgentResult, AgentTask

logger = logging.getLogger(__name__)

#: The payload kind the kernel sends for an outbound message.
DELIVER_KIND = "ingress.deliver"

#: The payload kind the kernel sends for a supersedable progress snapshot (ADR-0098).
#: An ingress that does not override ``on_progress`` silently ignores these, which is the
#: correct degradation: silence, not a wall of status lines.
PROGRESS_KIND = "ingress.progress"


class PermanentDeliveryError(Exception):
    """Raised by ``on_deliver`` when a message will NEVER be deliverable.

    The user blocked the bot, the account is gone, the chat was deleted. The
    kernel dead-letters these and stops retrying.

    Raise it only when you are sure. Everything else — a timeout, a 429, an
    outage — should be an ordinary exception, which the kernel treats as
    transient and may retry. Retrying a permanent failure forever is how an
    integration gets rate-limited and then banned; giving up on a transient one
    silently loses a message that would have gone through a minute later.
    """


class IngressAgent(Agent):
    """Base class for an entry point into Cambrian.

    Subclasses implement two methods, one per direction::

        class TelegramIngress(IngressAgent):
            def listen(self):
                for update in telegram.poll():          # your loop, your library
                    self.receive(f"tg:{update.chat_id}", update.text)

            def on_deliver(self, recipient, text, conversation_id):
                telegram.send_message(chat_id=recipient.removeprefix("tg:"), text=text)

    ``listen`` runs supervised in the background and never returns. ``on_deliver``
    is called by the kernel and should send and return; it has no return value
    because there is nothing to correlate it with.
    """

    #: Spawned and supervised as a daemon by the kernel (ADR-0033/0070).
    trait = "daemon"

    def __init__(self, agent_id: str, **kwargs) -> None:
        super().__init__(agent_id, **kwargs)
        self.stream_id: str = ""
        self.params: dict = {}
        import queue

        self._signal_queue: "queue.Queue" = queue.Queue()

    # ── inbound ────────────────────────────────────────────────────────────

    def listen(self) -> None:
        """Your inbound loop: poll or serve the outside world, call ``receive``.

        Runs supervised — if it raises, the runtime restarts it with backoff, so
        a dropped connection does not take the ingress down permanently.
        """
        raise NotImplementedError("an ingress must implement listen()")

    def receive(self, external_id: str, text: str, **extra) -> None:
        """Hand one inbound message to the kernel and return immediately.

        ``external_id`` identifies the sender on your side — a Telegram chat id, a
        websocket connection key. It must fall inside the namespace this ingress
        was registered with, or the kernel refuses it; that is what stops one
        ingress speaking for another's users.

        Fire-and-forget by design. There is no reply here, and waiting for one is
        exactly the coupling this shape removes: acknowledge the sender now, and
        let ``on_deliver`` carry whatever the kernel eventually says.
        """
        if not external_id:
            raise ValueError("receive(): external_id is required — a message from nobody cannot be replied to")
        if not text:
            raise ValueError("receive(): text is required")
        payload = {"external_id": external_id, "text": text}
        payload.update(extra)
        self._signal_queue.put((payload, text))

    # ── outbound ───────────────────────────────────────────────────────────

    def on_deliver(self, recipient: str, text: str, conversation_id: str) -> None:
        """Send one outbound message. Called by the kernel; no return value.

        ``recipient`` is the address the kernel resolved from the conversation —
        never something an agent chose. Raise :class:`PermanentDeliveryError` if
        it can never be delivered; raise anything else if a retry might work.
        """
        raise NotImplementedError("an ingress must implement on_deliver()")

    # ── progress (ADR-0098) ────────────────────────────────────────────────

    def on_progress(self, recipient: str, text: str, conversation_id: str,
                    final: bool = False) -> None:
        """Render one supersedable progress snapshot. Optional.

        A snapshot REPLACES the previous one rather than joining it: a turn that takes
        minutes should occupy one evolving line, not a transcript of its own internals.
        Where the transport can edit in place — Telegram ``editMessageText``, Slack
        ``chat.update`` — that is the natural implementation.

        The default does nothing, and that is a deliberate default rather than an
        oversight. An ingress that has not thought about supersession is better off
        silent than appending a status line per update, and older ingresses keep working
        against a newer kernel without change.

        ``final`` marks the last update of a turn. Render it, then STOP tracking the line,
        so the next turn opens a fresh one instead of overwriting this one — which matters
        when the closing line is an error the user should keep seeing.

        An EMPTY ``text`` is the CLEAR signal: the turn is over, take the status line
        down. It arrives on every ending, successful or not — which matters, because a
        turn that fails before replying sends nothing else, and a status line left up
        reads to the user as a hang.

        Best-effort by contract: raising here is tolerated but pointless — the kernel does
        not retry progress, because by the time a retry landed the snapshot would already
        be stale and superseded.
        """
        return None

    def run(self, task: AgentTask):
        """Translate an inbound kernel call into ``on_deliver``.

        This is the served side of the daemon. It is not part of the author's
        surface — authors implement ``on_deliver`` — but it has to exist because
        the kernel reaches an ingress the same way it reaches any other agent.
        """
        body = _decode(task)
        kind = body.get("kind")

        if kind == PROGRESS_KIND:
            # Progress is fire-and-forget and must never fail the turn it describes, so a
            # rendering error is logged and acknowledged rather than reported back.
            try:
                self.on_progress(body.get("recipient", ""), body.get("text", ""),
                                 body.get("conversation_id", ""), bool(body.get("final", False)))
            except Exception as exc:  # noqa: BLE001 — progress must not propagate
                logger.debug("ingress %s: progress render failed: %s", self.agent_id, exc)
            return _ack(status="sent")

        if kind != DELIVER_KIND:
            return _ack(status="failed", permanent=True,
                        error=f"unsupported payload kind {kind!r}")

        recipient = body.get("recipient", "")
        text = body.get("text", "")
        if not recipient or not text:
            # Nothing a retry can fix — the kernel would send the same again.
            return _ack(status="failed", permanent=True, error="delivery is missing a recipient or text")

        try:
            self.on_deliver(recipient, text, body.get("conversation_id", ""))
        except PermanentDeliveryError as exc:
            logger.warning("ingress %s: permanently undeliverable to %s: %s", self.agent_id, recipient, exc)
            return _ack(status="failed", permanent=True, error=str(exc))
        except Exception as exc:  # noqa: BLE001 — anything unlabelled is retryable
            logger.warning("ingress %s: delivery to %s failed, may retry: %s", self.agent_id, recipient, exc)
            return _ack(status="failed", permanent=False, error=str(exc))
        return _ack(status="sent")

    # ── lifecycle ──────────────────────────────────────────────────────────

    def serve(self, address: Optional[str] = None) -> None:
        """Run both directions: the delivery server, and the inbound stream.

        The signal stream runs on a background thread and the gRPC server blocks
        the main one. That order matters — the server is what the kernel dials to
        deliver, so it owns the process lifetime; if the inbound loop is what
        ends, the ingress should still be reachable for outbound messages.
        """
        from .runtime import start_agent_server
        from .server import _parse_listen_address

        stream = threading.Thread(target=self._run_inbound, name=f"{self.agent_id}-inbound", daemon=True)
        stream.start()
        start_agent_server(self, address or _parse_listen_address())

    def _run_inbound(self) -> None:
        """Open the signal stream and run the supervised ``listen`` loop."""
        from .daemon import start_daemon

        try:
            start_daemon(self)
        except Exception:  # noqa: BLE001
            # The outbound half must survive an inbound failure: a bot that cannot
            # poll should still be able to deliver a queued reply.
            logger.exception("ingress %s: inbound stream stopped; outbound still served", self.agent_id)

    def daemon_loop(self) -> None:
        """Adapter for the daemon runtime, which supervises ``daemon_loop``.

        Authors override :meth:`listen`; this exists so the two names do not have
        to be the same, and so ``listen`` reads as what it is.
        """
        self.listen()

    def send_signal(self, payload: dict, raw_text: str = "") -> None:
        """Low-level enqueue. Prefer :meth:`receive`, which names the sender."""
        self._signal_queue.put((payload, raw_text))


def _decode(task: AgentTask) -> dict:
    """Pull the delivery payload out of a task, whichever way it arrived."""
    raw = task.data if task.data else (task.text or "").encode()
    if not raw:
        return {}
    try:
        body = json.loads(raw)
    except (ValueError, TypeError):
        return {}
    return body if isinstance(body, dict) else {}


def _ack(status: str, permanent: bool = False, error: str = "") -> AgentResult:
    """Build the acknowledgement the kernel's delivery path expects."""
    body = {"status": status}
    if status != "sent":
        body["permanent"] = permanent
        if error:
            body["error"] = error
    return AgentResult(data=json.dumps(body).encode(), type="text")
