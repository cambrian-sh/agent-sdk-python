"""Code executor agent (CognitiveAgent) — ADR-0040 tracer bullet.

Restores Python code execution behind the ADR-0039 kernel-owned tool registry.
The agent receives the ``execute_python`` tool schema and marshals arguments in
its think() loop; calls route to the kernel via ``ExecuteTool`` and execute in a
capped, isolated child process. Approval is required (dangerous tool).

Uses the default ``CognitiveAgent.run()`` → ``think()`` ReAct loop with
class-level configuration.
"""

from cambrian_agent_sdk import CognitiveAgent
from cambrian_agent_sdk.types import ProposalResponse
from cambrian_agent_sdk._logging import configure_logging

AGENT_DESCRIPTION = (
    "Executes Python code snippets in a sandboxed, kernel-supervised child process. "
    "Returns stdout, stderr, and exit code. Code execution requires operator approval."
)

AGENT_MANIFEST = '''
{
  "version": "1.0.0",
  "trait": "cognitive",
  "supported_formats": ["text", "code"],
  "tools": ["code_execution", "python_execution", "execute_code"],
  "release_notes": "Python code execution behind kernel-owned tool registry (ADR-0039/0040).",
  "dependencies": []
}
'''

_CODE_EXEC_KEYWORDS = (
    "run this code", "execute", "run script", "import", "def", "class",
    "python", ".py", "compute", "calculate",
)

_CODE_EXEC_NEGATIVE_KEYWORDS = (
    "summarise", "analyze article", "compare text",
)


class CodeExecutorAgent(CognitiveAgent):
    role = "You are a safe Python code executor. Run the provided code and return the output."
    output_schema = (
        "Return the code execution result with three sections:\n"
        "Exit Code: the numeric exit code\n"
        "Stdout: the standard output of the code\n"
        "Stderr: any errors or warnings (or 'none' if empty)"
    )
    constraints = (
        "Use the execute_python system tool to run code.",
        "Never modify the code before execution unless fixing an obvious syntax error.",
        "Report the exact exit code and all output.",
        "If the code fails, explain the error in plain language.",
    )
    result_type = "code_output"
    max_tokens = 1024
    temperature = 0.2

    def propose(self, request=None) -> ProposalResponse:
        desc = (getattr(request, "description", "") or getattr(request, "text", "") or "").lower()
        if any(k in desc for k in _CODE_EXEC_NEGATIVE_KEYWORDS):
            return ProposalResponse(
                confidence=0.1,
                rationale="Task requires reasoning, not code execution.",
                estimated_latency_ms=4000,
            )
        if any(k in desc for k in _CODE_EXEC_KEYWORDS):
            return ProposalResponse(
                confidence=0.85,
                rationale=f"Code execution task detected: {desc[:80]}",
                estimated_latency_ms=5000,
            )
        return ProposalResponse(
            confidence=0.2,
            rationale="Not obviously a code execution task.",
            estimated_latency_ms=5000,
        )


agent = CodeExecutorAgent(
    agent_id="code_executor_agent",
    version="1.0.0",
    description=AGENT_DESCRIPTION,
)


if __name__ == "__main__":
    configure_logging(agent_id="code_executor_agent")
    agent.serve()
