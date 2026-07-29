"""Example ingress agent (IngressAgent) — ADR-0090.

An *ingress* is any point where the outside world enters Cambrian: Telegram, a
webhook receiver, a websocket listener, an inbound API. This example uses a
polled file as its "network" so it runs anywhere with no credentials, but the
shape is exactly what a real Telegram bridge looks like — the only difference is
which library ``listen`` and ``on_deliver`` call.

The two directions are independent and neither waits for the other:

    inbound    a line appears in the inbox file -> receive() -> the kernel
    outbound   the kernel delivers             -> on_deliver() -> the outbox file

One inbound line may produce no replies, one, or several, and the kernel may
speak with no inbound line at all. That asymmetry is the point: the Telegram Bot
API acknowledges an update and then sends replies through a separate outbound
call, so anything shaped as request/response cannot be built against it correctly.

Note what this class does NOT let you do. There is no way to declare your own
surface and no way to choose a recipient. The surface comes from the registration
an operator made out of band, and the recipient is resolved by the kernel from
the conversation — a daemon is a black box, and a black box that asserts its own
privilege level is not a security boundary (INV-5).

Run by Cambrian with:
  python example_ingress_agent.py \
      --daemon-mode \
      --stream-id example_ingress \
      --substrate-socket <path> \
      --daemon-params '{"inbox": "/tmp/in.txt", "outbox": "/tmp/out.txt", "poll_seconds": 1}'

Register it before it can act as an entry point (ADR-0090 D2), e.g.:
  ingress agent_id=example_ingress surface=chat:example namespace=["ex:"]
"""

from __future__ import annotations

import logging
import os
import time

from cambrian_agent_sdk import IngressAgent, PermanentDeliveryError

AGENT_DESCRIPTION = "Example ingress: relays messages between a file and Cambrian"
AGENT_MANIFEST = '''
{
    "trait": "daemon",
    "runtime": "python",
    "version": "0.1.0",
    "capabilities": ["ingress", "chat"],
    "input_schema": {
        "properties": {
            "inbox": {"type": "string", "description": "file polled for inbound lines"},
            "outbox": {"type": "string", "description": "file appended to on delivery"},
            "poll_seconds": {"type": "number", "description": "how often to poll the inbox"}
        }
    }
}
'''

logger = logging.getLogger(__name__)

#: Prefix every sender id carries. It must fall inside the namespace this ingress
#: was registered with, or the kernel refuses the message — that bound is what
#: stops one ingress speaking for another's users.
SENDER_PREFIX = "ex:"


class ExampleIngress(IngressAgent):
    """Relays between a polled file and Cambrian."""

    # ── inbound ────────────────────────────────────────────────────────────

    def listen(self) -> None:
        """Poll the inbox and hand each new line to the kernel.

        Supervised: if this raises, the runtime restarts it with backoff, so a
        dropped connection does not take the ingress down for good. That is why
        the loop does not need its own try/except around the whole body.
        """
        inbox = self.params.get("inbox", "")
        interval = float(self.params.get("poll_seconds", 1))
        if not inbox:
            raise ValueError("example ingress: --daemon-params needs an 'inbox' path")

        offset = 0
        logger.info("example ingress: polling %s every %ss", inbox, interval)
        while True:
            if os.path.exists(inbox):
                with open(inbox, "r", encoding="utf-8") as fh:
                    fh.seek(offset)
                    for line in fh:
                        line = line.strip()
                        if not line:
                            continue
                        # "<sender>|<text>" — a stand-in for a chat id and a message.
                        sender, _, text = line.partition("|")
                        if not text:
                            logger.warning("example ingress: skipping unaddressed line %r", line)
                            continue
                        # Acknowledged and forgotten. Nothing here waits for a reply;
                        # whatever the kernel eventually says arrives via on_deliver.
                        self.receive(SENDER_PREFIX + sender.strip(), text.strip())
                    offset = fh.tell()
            time.sleep(interval)

    # ── outbound ───────────────────────────────────────────────────────────

    def on_deliver(self, recipient: str, text: str, conversation_id: str) -> None:
        """Send one outbound message. No return value — nothing is correlated.

        ``recipient`` is the address the kernel resolved from the conversation,
        never something an agent picked.
        """
        outbox = self.params.get("outbox", "")
        if not outbox:
            # Misconfiguration, not a transient fault: retrying writes the same
            # message to the same missing path forever.
            raise PermanentDeliveryError("no 'outbox' configured for this ingress")

        try:
            with open(outbox, "a", encoding="utf-8") as fh:
                fh.write(f"{recipient}|{text}\n")
        except OSError as exc:
            # A full or unmounted disk may well clear, so let the kernel retry.
            raise RuntimeError(f"could not write to {outbox}: {exc}") from exc


agent = ExampleIngress(
    agent_id="example_ingress",
    version="0.1.0",
    description=AGENT_DESCRIPTION,
)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    agent.serve()
