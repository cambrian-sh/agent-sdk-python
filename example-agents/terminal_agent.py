"""Terminal agent (CognitiveAgent) — ADR-0040 tracer bullet.

Restores system command execution behind the ADR-0039 kernel-owned tool registry.
The agent receives the ``execute_command`` tool schema and marshals arguments in
its think() loop; calls route to the kernel via ``ExecuteTool`` and execute in a
capped, isolated child process. Approval is required (dangerous tool).

Uses the default ``CognitiveAgent.run()`` → ``think()`` ReAct loop with
class-level configuration.
"""

from cambrian_agent_sdk import CognitiveAgent
from cambrian_agent_sdk.types import ProposalResponse
from cambrian_agent_sdk._logging import configure_logging

AGENT_DESCRIPTION = (
    "Executes shell commands in a sandboxed, kernel-supervised child process. "
    "Only allowlisted commands run; dangerous verbs require operator approval."
)

AGENT_MANIFEST = '''
{
  "version": "1.0.0",
  "trait": "cognitive",
  "supported_formats": ["text"],
  "tools": ["shell_command", "terminal", "command_execution"],
  "release_notes": "Shell command execution behind kernel-owned tool registry (ADR-0039/0040).",
  "dependencies": []
}
'''

_SHELL_KEYWORDS = (
    "run", "execute", "command", "shell", "terminal", "bash", "cmd",
    "list files", "ls", "dir", "mkdir", "cp", "mv", "rm",
)

_SHELL_NEGATIVE_KEYWORDS = (
    "summarise", "analyze", "compare", "evaluate", "review",
)


class TerminalAgent(CognitiveAgent):
    role = "You are a precise command executor. Translate natural-language instructions into shell commands"
    output_schema = (
        "Respond with the command execution result.\n"
        "- If successful, show the command output.\n"
        "- If the command failed, explain the error and suggest alternatives.\n"
        "- Never modify or invent file content unless explicitly asked."
    )
    constraints = (
        "Use the execute_command system tool for every shell operation.",
        "Report the exact command you ran and its exit status.",
        "Do not run destructive commands (rm, drop, delete) unless explicitly asked.",
    )
    result_type = "command_output"
    max_tokens = 1024
    temperature = 0.3

    def propose(self, request=None) -> ProposalResponse:
        desc = (getattr(request, "description", "") or getattr(request, "text", "") or "").lower()
        if any(k in desc for k in _SHELL_NEGATIVE_KEYWORDS):
            return ProposalResponse(
                confidence=0.1,
                rationale="Task requires reasoning, not command execution.",
                estimated_latency_ms=4000,
            )
        if any(k in desc for k in _SHELL_KEYWORDS):
            return ProposalResponse(
                confidence=0.85,
                rationale=f"Shell task detected: {desc[:80]}",
                estimated_latency_ms=3000,
            )
        return ProposalResponse(
            confidence=0.3,
            rationale="General-purpose fallback for command execution.",
            estimated_latency_ms=3000,
        )


agent = TerminalAgent(
    agent_id="terminal_agent",
    version="1.0.0",
    description=AGENT_DESCRIPTION,
)


if __name__ == "__main__":
    configure_logging(agent_id="terminal_agent")
    agent.serve()
