"""HTTP chat ingress — a synchronous protocol on top of the fire-and-forget core.

This is the SDK replacement for the Go chat ingress (ADR-0080 D1, renamed by
ADR-0090). It serves the same three endpoints::

    POST /open   {"conversation_id": "...", "policy": "..."}  -> {"ok": true}
    POST /turn   {"conversation_id": "...", "message": "..."}  -> {"reply": "...", "error": ""}
    POST /close  {"conversation_id": "..."}                    -> {"ok": true}

**Correlation lives here, and only here.** The caller blocks on ``/turn`` waiting
for a reply in the response body, while Cambrian's core is two independent
fire-and-forget flows. Something has to bridge those shapes, and the ingress is
the right place: request/response is a property of *this* external protocol, not
of Cambrian. Putting the waiting in the kernel would drag every ingress back into
a round trip for the sake of one that happens to need it.

The bridge is small: ``/turn`` sends the message inbound, then waits on a future
keyed by conversation. ``on_deliver`` completes it. If the agent speaks more than
once, the first reply satisfies the waiting request and the rest are queued for
subsequent turns, because an HTTP response can only carry one — a limitation of
the protocol, not of the model, and worth knowing about rather than hiding.
"""

from __future__ import annotations

import json
import logging
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Dict, Optional

from .ingress import IngressAgent

logger = logging.getLogger(__name__)

#: Namespace prefix for conversation ids arriving over HTTP. Every external id an
#: ingress claims must fall inside the namespace it was registered with; this is
#: what stops one ingress speaking for another's users.
EXTERNAL_PREFIX = "chat:"

#: How long ``/turn`` waits for a reply before giving up. Generous because a turn
#: may involve tool calls and an LLM; the caller's own timeout should be longer.
DEFAULT_TURN_TIMEOUT = 240.0


class _Waiter:
    """One conversation's pending reply, plus anything said after it."""

    def __init__(self) -> None:
        self.event = threading.Event()
        self.replies: list[str] = []
        self.lock = threading.Lock()

    def deliver(self, text: str) -> None:
        with self.lock:
            self.replies.append(text)
        self.event.set()

    def take(self, timeout: float) -> Optional[str]:
        if not self.event.wait(timeout):
            return None
        with self.lock:
            reply = self.replies.pop(0) if self.replies else None
            if not self.replies:
                self.event.clear()
        return reply


class HTTPChatIngress(IngressAgent):
    """Serves an HTTP chat surface and relays it to Cambrian.

    Daemon params::

        {"addr": "127.0.0.1:8890", "turn_timeout": 240}
    """

    def __init__(self, agent_id: str = "http_chat_ingress", **kwargs) -> None:
        super().__init__(agent_id=agent_id, **kwargs)
        self._waiters: Dict[str, _Waiter] = {}
        self._waiters_lock = threading.Lock()
        self._policies: Dict[str, str] = {}
        self._httpd: Optional[ThreadingHTTPServer] = None

    # ── inbound: the HTTP server IS the listen loop ────────────────────────

    def listen(self) -> None:
        addr = self.params.get("addr", "127.0.0.1:8890")
        host, _, port = addr.rpartition(":")
        ingress = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, fmt, *args):  # quieter than the default
                logger.debug("http chat ingress: " + fmt, *args)

            def _read(self) -> dict:
                length = int(self.headers.get("Content-Length") or 0)
                if not length:
                    return {}
                try:
                    return json.loads(self.rfile.read(length) or b"{}")
                except ValueError:
                    return {}

            def _send(self, body: dict, code: int = 200) -> None:
                raw = json.dumps(body).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def do_POST(self):  # noqa: N802 — BaseHTTPRequestHandler's contract
                body = self._read()
                conv = (body.get("conversation_id") or "").strip()
                if not conv:
                    return self._send({"error": "conversation_id is required"}, 400)

                if self.path == "/open":
                    ingress._open(conv, body.get("policy", ""))
                    return self._send({"ok": True})
                if self.path == "/turn":
                    reply, err = ingress._turn(conv, (body.get("message") or "").strip())
                    return self._send({"reply": reply, "error": err})
                if self.path == "/close":
                    ingress._close(conv)
                    return self._send({"ok": True})
                return self._send({"error": "not found"}, 404)

        # daemon_threads so an in-flight request cannot hold the process open past
        # shutdown; allow_reuse_address so a restart is not blocked by TIME_WAIT.
        class _Server(ThreadingHTTPServer):
            daemon_threads = True
            allow_reuse_address = True

        self._httpd = _Server((host or "127.0.0.1", int(port)), Handler)
        logger.info("http chat ingress listening on %s", addr)
        self._httpd.serve_forever()

    # ── the three endpoints ────────────────────────────────────────────────

    def _open(self, conv: str, policy: str) -> None:
        """Remember the policy. The conversation itself is opened by the kernel on
        first message — it owns conversation identity, and inventing one here would
        mean two sources of truth for the same thing."""
        if policy:
            self._policies[conv] = policy
        self._waiter(conv)  # ready before any reply can arrive

    def _turn(self, conv: str, message: str) -> tuple[str, str]:
        if not message:
            return "", "message is required"

        waiter = self._waiter(conv)
        timeout = float(self.params.get("turn_timeout", DEFAULT_TURN_TIMEOUT))
        try:
            # The policy rides the FIRST message; the kernel applies it only when it
            # opens the conversation and ignores it thereafter, so a later turn
            # cannot rewrite the standing instructions mid-transcript.
            self.receive(EXTERNAL_PREFIX + conv, message, policy=self._policies.get(conv, ""))
        except Exception as exc:  # noqa: BLE001
            return "", f"send failed: {exc}"

        reply = waiter.take(timeout)
        if reply is None:
            return "", f"no reply within {timeout:g}s"
        return reply, ""

    def _close(self, conv: str) -> None:
        with self._waiters_lock:
            self._waiters.pop(conv, None)
        self._policies.pop(conv, None)

    # ── outbound ───────────────────────────────────────────────────────────

    def on_deliver(self, recipient: str, text: str, conversation_id: str) -> None:
        """Complete whatever ``/turn`` is waiting on this conversation.

        A reply that arrives with nobody waiting is queued rather than dropped: an
        agent may legitimately speak twice, and the second reply belongs to the
        next request rather than to nothing.
        """
        conv = recipient[len(EXTERNAL_PREFIX):] if recipient.startswith(EXTERNAL_PREFIX) else recipient
        self._waiter(conv).deliver(text)

    # ── internals ──────────────────────────────────────────────────────────

    def _waiter(self, conv: str) -> _Waiter:
        with self._waiters_lock:
            w = self._waiters.get(conv)
            if w is None:
                w = _Waiter()
                self._waiters[conv] = w
            return w

    def stop(self) -> None:
        """Stop the HTTP server and release its port.

        Both calls matter: ``shutdown`` ends the serve loop, ``server_close``
        releases the socket. Without the second the port stays bound after the
        loop has stopped, so a restarted ingress fails to bind and looks like a
        port conflict with something else.
        """
        httpd, self._httpd = self._httpd, None
        if httpd is not None:
            httpd.shutdown()
            httpd.server_close()
