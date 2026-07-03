"""Summariser agent (CognitiveAgent) — ADR-0023, migrated to SDK v2 (ADR-0036).

Condenses long text or multiple prior step results into concise bullet-point
summaries. Distinct from analyst_agent: the summariser *condenses* existing content,
it does not reason toward new conclusions.

Uses the default ``CognitiveAgent.run()`` → ``think()`` ReAct loop with
class-level configuration.
"""

from cambrian_agent_sdk import CognitiveAgent
from cambrian_agent_sdk.types import ProposalResponse
from cambrian_agent_sdk._logging import configure_logging

AGENT_DESCRIPTION = (
    "Condenses long text and multiple inputs into concise bullet-point summaries. "
    "Synthesises and shortens existing content without adding new analysis or conclusions."
)

AGENT_MANIFEST = '''
{
  "version": "1.0.0",
  "trait": "cognitive",
  "supported_formats": ["text"],
  "tools": ["summarisation", "text_summary", "synthesis"],
  "release_notes": "LLM-powered bullet-point summariser via Substrate managed gateway.",
  "dependencies": []
}
'''

_SUMMARY_KEYWORDS = (
    "summarise", "summarize", "tldr", "tl;dr", "concise", "overview",
    "condense", "summary of", "key points", "in brief",
)

_ANALYSIS_KEYWORDS = (
    "analyse", "analyze", "compare", "evaluate", "assess",
    "justify", "recommend", "trade-off", "which is better",
)


class SummariserAgent(CognitiveAgent):
    role = "You are a precise summarisation engine."
    output_schema = (
        "- Output ONLY bullet points (start each line with '- ')\n"
        "- Capture the key facts, omit filler\n"
        "- Keep it short: at most 6 bullets"
    )
    constraints = (
        "Condense the provided content into a concise bullet-point summary.",
        "Do NOT add new analysis, opinions, or conclusions not present in the source.",
        "If no source content is provided, generate a concise summary based on your internal knowledge.",
    )
    result_type = "summary"
    max_tokens = 512
    temperature = 0.3

    def propose(self, request=None) -> ProposalResponse:
        desc = (getattr(request, "description", "") or getattr(request, "text", "") or "").lower()
        if any(k in desc for k in _ANALYSIS_KEYWORDS):
            return ProposalResponse(
                confidence=0.1,
                rationale="Task requires analysis/comparison — not a summarisation fit.",
                estimated_latency_ms=4000,
            )
        if any(k in desc for k in _SUMMARY_KEYWORDS):
            return ProposalResponse(
                confidence=0.9,
                rationale=f"Summarisation task detected: {desc[:80]}",
                estimated_latency_ms=4000,
            )
        return ProposalResponse(
            confidence=0.3,
            rationale="General-purpose fallback for summarisation.",
            estimated_latency_ms=4000,
        )


agent = SummariserAgent(
    agent_id="summariser_agent",
    version="1.0.0",
    description=AGENT_DESCRIPTION,
)


if __name__ == "__main__":
    configure_logging(agent_id="summariser_agent")
    agent.serve()
