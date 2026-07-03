"""Analyst agent (CognitiveAgent) — ADR-0023, migrated to SDK v2 (ADR-0036).

Reasons, compares, and evaluates using a structured chain-of-thought prompt. The
most general cognitive agent: handles inference, comparison, and evaluation when
neither code_generator nor summariser is a better fit.

Uses the default ``CognitiveAgent.run()`` → ``think()`` ReAct loop (memory +
@tool rounds) with class-level configuration.
"""

from cambrian_agent_sdk import CognitiveAgent, AgentResult, AgentTask
from cambrian_agent_sdk.types import ProposalResponse
from cambrian_agent_sdk._logging import configure_logging


AGENT_DESCRIPTION = (
    "Analyses, compares, and evaluates information using structured reasoning. "
    "Produces observations, reasoning chains, and conclusions. "
    "Handles trade-off analysis and explanation tasks."
)

AGENT_MANIFEST = '''
{
  "version": "1.0.0",
  "trait": "cognitive",
  "supported_formats": ["text"],
  "tools": ["analysis", "comparison", "evaluation", "reasoning"],
  "release_notes": "LLM-powered chain-of-thought analyst via Substrate managed gateway.",
  "dependencies": []
}
'''

_ANALYSIS_KEYWORDS = (
    "compare", "analyse", "analyze", "evaluate", "explain", "trade-off",
    "tradeoff", "which is better", "assess", "reason about", "implications",
    "why", "pros and cons", "justify", "recommendation", "recommend",
)


class AnalystAgent(CognitiveAgent):
    role = "You are a rigorous analytical reasoning engine."
    output_schema = (
        "Structure your answer with three sections:\n"
        "Observations: the relevant facts and constraints\n"
        "Reasoning: step-by-step inference connecting observations to a judgement\n"
        "Conclusion: the final answer or recommendation"
    )
    constraints = (
        "Analyse the task using structured chain-of-thought.",
        "Be precise and avoid unsupported claims. If the provided context is insufficient, say so rather than inventing facts.",
    )
    result_type = "analysis"
    max_tokens = 1024
    temperature = 0.4

    def propose(self, request=None) -> ProposalResponse:
        desc = (getattr(request, "description", "") or getattr(request, "text", "") or "").lower()
        if any(k in desc for k in _ANALYSIS_KEYWORDS):
            return ProposalResponse(
                confidence=0.85,
                rationale=f"Analysis task detected: {desc[:80]}",
                estimated_latency_ms=6000,
            )
        return ProposalResponse(
            confidence=0.3,
            rationale="General-purpose fallback for analysis.",
            estimated_latency_ms=6000,
        )


agent = AnalystAgent(
    agent_id="analyst_agent",
    version="1.0.0",
    description=AGENT_DESCRIPTION,
)


if __name__ == "__main__":
    configure_logging(agent_id="analyst_agent")
    agent.serve()
