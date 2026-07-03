"""Example daemon agent (DaemonAgent) — ADR-0033, migrated to SDK v2 (ADR-0036).

Reads ``--daemon-params`` (expects ``interval_seconds: int``), loops calling
``self.send_signal`` at the configured interval, and exits on SIGTERM. Imports no
kernel wire-protocol types.

Run by Cambrian with:
  python example_daemon_agent.py \
      --daemon-mode \
      --stream-id example_daemon \
      --substrate-socket <path> \
      --daemon-params '{"interval_seconds": 5}'
"""

from __future__ import annotations

import logging
import time

from cambrian_agent_sdk import DaemonAgent

AGENT_DESCRIPTION = "Example daemon agent that emits periodic heartbeat signals"
AGENT_MANIFEST = '''
{
    "trait": "daemon",
    "runtime": "python",
    "version": "0.1.0",
    "capabilities": ["heartbeat", "monitoring"],
    "input_schema": {
        "properties": {
            "interval_seconds": {"type": "integer", "default": 60}
        }
    },
    "output_schema": {
        "properties": {
            "tick": {"type": "integer"},
            "message": {"type": "string"}
        }
    }
}
'''

logger = logging.getLogger("example_daemon")


class ExampleDaemon(DaemonAgent):
    """Emits a heartbeat signal every ``interval_seconds`` (from --daemon-params)."""

    def daemon_loop(self) -> None:
        tick = 0
        interval = int(self.params.get("interval_seconds", 60))
        while True:
            tick += 1
            self.send_signal(
                {"tick": tick, "message": f"heartbeat #{tick}"},
                raw_text=f"heartbeat signal {tick}",
            )
            logger.info("sent signal tick=%d", tick)
            time.sleep(interval)


agent = ExampleDaemon(agent_id="example_daemon", description=AGENT_DESCRIPTION)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    agent.serve()
