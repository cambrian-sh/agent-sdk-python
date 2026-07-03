"""Cambrian Benchmark Generalist Agent — Phase 1 reference implementation.

This is the first real Python agent in the Cambrian ecosystem. It demonstrates
 the full SDK surface: subclassing :class:`CognitiveAgent`, Substrate-proxied
LLM calls via ``self.substrate.generate()``, and the AGENT_MANIFEST block format.

Discovery contract:
    File name must end in ``agent.py``. The Substrate's BBoltAdapter scans
    ``agents_dir`` at startup, extracting AGENT_DESCRIPTION and AGENT_MANIFEST
    with regex before this process boots.

Usage (manual):
    python benchmark_agent.py --port 50052 --substrate-addr localhost:50051

Usage (via Substrate):
    Place this file in the ``agents/`` directory. The Substrate will
    auto-discover, interview, and dispatch tasks to it.
"""

import logging

from cambrian_agent_sdk import CognitiveAgent, configure_logging

# ── Discovery metadata (parsed by BBoltAdapter regex before process boots) ────

AGENT_DESCRIPTION = "General-purpose cognitive agent for text generation and reasoning tasks"

AGENT_MANIFEST = '''
{
  "version": "0.1.0",
  "trait": "cognitive",
  "supported_formats": ["text", "json"],
  "tools": ["text_generation", "summarise", "qa"],
  "release_notes": "Initial benchmark agent — validates Python SDK and Substrate gRPC boundary",
  "dependencies": []
}
'''

# ── Agent definition ──────────────────────────────────────────────────────────

configure_logging(logging.INFO)


class BenchmarkAgent(CognitiveAgent):
    """General-purpose cognitive agent that proxies tasks to the Substrate LLM."""

    role = "generalist text-generation agent"
    output_schema = "text"
    max_tokens = 1024
    temperature = 0.7

    def run(self, task):
        """Handle any text generation task by proxying to the Substrate LLM gateway."""
        log = logging.getLogger("benchmark_agent")
        prompt = task.text

        # Unconditionally inject whatever prior step results the executor passed.
        # The Substrate's filterSnapshotForStep already ensured that task.context
        # contains ONLY step_N_result keys for N declared in this step's DependsOn —
        # no irrelevant results are present. Guarding on English anaphora patterns
        # here would be wrong: it could suppress context the planner explicitly declared.
        prior = {
            k: v for k, v in task.context.items()
            if k.startswith("step_") and not k.endswith("_checkpoint")
        }
        if prior:
            context_block = "\n".join(
                f"  {k}: {v[:300]}" for k, v in sorted(prior.items())
            )
            prompt = f"Prior steps:\n{context_block}\n\nCurrent task:\n{prompt}"
            log.info("Injecting %d prior step keys for step %d",
                     len(prior), task.step_index)

        log.info("Generating response for step %d", task.step_index)

        # ``self.substrate`` is a SubstrateClient wired at serve() time.
        # ``timeout_ms`` propagates the Go step deadline into the gRPC LLM call.
        # Without it, a slow Ollama response outlives the Go context and leaks a
        # ThreadPoolExecutor thread for every timed-out step.
        response_text = self.substrate.generate(
            session_token_id=task.session_token_id,
            prompt=prompt,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            timeout_ms=task.deadline_remaining_ms,
        )

        log.info("Generated %d chars", len(response_text))
        return {"data": response_text.encode("utf-8"), "type": "text"}


agent = BenchmarkAgent(
    agent_id="benchmark_agent",
    version="0.1.0",
    description=AGENT_DESCRIPTION,
)

# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    agent.serve()
