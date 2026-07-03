"""Research agent (CognitiveAgent) — ADR-0040 tool-using reference agent.

Combines web search, web extraction, and file read/write behind the ADR-0039
kernel-owned tool registry. Answers research questions by searching the web,
reading sources, and writing findings to files.

Uses the default ``CognitiveAgent.run()`` → ``think()`` ReAct loop.
"""

from cambrian_agent_sdk import CognitiveAgent
from cambrian_agent_sdk.types import ProposalResponse
from cambrian_agent_sdk._logging import configure_logging

AGENT_DESCRIPTION = (
    "Researches topics using web search, reads source pages, and synthesises "
    "findings. Writes structured research notes to files."
)

AGENT_MANIFEST = '''
{
  "version": "1.0.0",
  "trait": "cognitive",
  "supported_formats": ["text"],
  "tools": ["web_research", "research", "information_retrieval"],
  "release_notes": "Reference tool-using research agent (ADR-0040).",
  "dependencies": []
}
'''

_RESEARCH_KEYWORDS = (
    "research", "find information", "look up", "search for", "what is",
    "who is", "when did", "how does", "tell me about", "explain about",
)


class ResearchAgent(CognitiveAgent):
    role = "You are a thorough research analyst. Find, read, and synthesise information."
    output_schema = (
        "Structure your answer with three sections:\n"
        "Sources: the URLs or file paths you consulted\n"
        "Findings: the key facts and insights gathered\n"
        "Recommendations: actionable conclusions or next steps"
    )
    constraints = (
        "Use web_search to find information, web_extract to read pages.",
        "Use read_file to review local documents, write_file to save notes.",
        "Always cite your sources.",
        "If information is unavailable, state so clearly.",
    )
    result_type = "research"
    max_tokens = 2048
    temperature = 0.4

    def propose(self, request=None) -> ProposalResponse:
        desc = (getattr(request, "description", "") or getattr(request, "text", "") or "").lower()
        if any(k in desc for k in _RESEARCH_KEYWORDS):
            return ProposalResponse(
                confidence=0.9,
                rationale=f"Research task detected: {desc[:80]}",
                estimated_latency_ms=15000,
            )
        return ProposalResponse(
            confidence=0.3,
            rationale="General-purpose research agent.",
            estimated_latency_ms=15000,
        )


agent = ResearchAgent(
    agent_id="research_agent",
    version="1.0.0",
    description=AGENT_DESCRIPTION,
)


if __name__ == "__main__":
    configure_logging(agent_id="research_agent")
    agent.serve()
