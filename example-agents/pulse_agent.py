"""Pulse agent (DaemonAgent) — SDK v2 demo.

A persistent background signal producer. Every ``interval_seconds`` it emits a
heartbeat signal carrying an incrementing tick and an ISO timestamp. Demonstrates a
DaemonAgent: override daemon_loop(), call self.send_signal(); the runtime opens a
SignalStream and restarts the loop with capped exponential backoff if it crashes.

Run by Cambrian with:
  python pulse_agent.py --daemon-mode --stream-id pulse \\
      --substrate-socket <path> --daemon-params '{"interval_seconds": 10}'
"""

from __future__ import annotations

import datetime
import logging
import time

from cambrian_agent_sdk import DaemonAgent

AGENT_DESCRIPTION = "Emits a periodic heartbeat pulse signal with a tick counter and timestamp."
AGENT_MANIFEST = '''
{
  "trait": "daemon",
  "runtime": "python",
  "version": "1.0.0",
  "capabilities": ["heartbeat", "pulse", "monitoring"],
  "input_schema": {
    "properties": {
      "interval_seconds": {"type": "integer", "default": 10}
    }
  },
  "output_schema": {
    "properties": {
      "tick": {"type": "integer"},
      "timestamp": {"type": "string"}
    }
  }
}
'''

logger = logging.getLogger("pulse_agent")


class PulseAgent(DaemonAgent):
    def daemon_loop(self) -> None:
        tick = 0
        interval = int(self.params.get("interval_seconds", 10))
        while True:
            tick += 1
            self.send_signal(
                {"tick": tick, "timestamp": datetime.datetime.now().isoformat() + "Z"},
                raw_text=f"pulse #{tick}",
            )
            logger.info("pulse tick=%d", tick)
            time.sleep(interval)


agent = PulseAgent(agent_id="pulse_agent", description=AGENT_DESCRIPTION)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    agent.serve()
