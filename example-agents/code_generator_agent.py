"""Python code generator agent — ADR-0023, migrated to SDK v2 (ADR-0036).

A general cognitive agent with a specialised ``@tool`` for producing clean,
executable Python 3 code.  When the task is purely conversational or analytical
it answers in natural text; when code is required it calls the
``generate_python_code`` tool and includes the result in its final answer.

Uses the default ``CognitiveAgent.run()`` → ``think()`` ReAct loop with
class-level configuration.
"""

import json

from cambrian_agent_sdk import CognitiveAgent, tool
from cambrian_agent_sdk.types import ProposalResponse
from cambrian_agent_sdk._logging import configure_logging

AGENT_DESCRIPTION = (
    "An expert Python 3 software engineer that generates clean, executable code "
    "from natural language descriptions.  Can also explain, analyse, and discuss "
    "programming topics in plain text when code is not required."
)

AGENT_MANIFEST = '''
{
  "version": "1.0.0",
  "trait": "cognitive",
  "supported_formats": ["text", "code"],
  "tools": ["code_generation", "python_generation", "generate_python_code"],
  "release_notes": "LLM-powered Python expert with a structured code-generation tool.",
  "dependencies": []
}
'''

_CODE_KEYWORDS = (
    "write", "implement", "create a function", "create a class",
    "python code", "code that", "function to", "program", "script",
    "algorithm", "sort", "search", "parse", "generate code",
)


class CodeGeneratorAgent(CognitiveAgent):
    role = "You are an expert Python 3 software engineer."
    output_schema = (
        "Respond naturally in text.  When code is required, explain briefly what "
        "the code does, then include a fenced Python block (```python ... ```) "
        "containing the clean, executable implementation.  Do not put explanatory "
        "text inside the fenced block."
    )
    constraints = (
        "Use the @tool generate_python_code when the user explicitly asks for code.",
        "Code must be immediately executable Python 3 with type hints where helpful.",
        "Include minimal inline comments only where the logic is non-obvious.",
    )
    result_type = "code"
    max_tokens = 1024
    temperature = 0.2

    @tool
    def generate_python_code(self, description: str, function_name: str = "") -> str:
        """Generate clean, executable Python 3 code from a natural language description.

        Args:
            description: A detailed explanation of what the code should do.
            function_name: Optional desired function or class name.

        Returns:
            The generated Python code as a string (already fenced if markdown).
        """
        # The ReAct loop will actually drive the LLM to produce the code.
        # This tool exists so the agent can *intentionally* decide to generate code
        # rather than having every response forced into a code block.
        prompt = (
            f"Write clean, executable Python 3 code for the following task:\n\n"
            f"{description}\n\n"
        )
        if function_name:
            prompt += f"Name the main function/class: {function_name}\n"
        prompt += (
            "Return ONLY the code inside a ```python ... ``` fenced block. "
            "No explanatory text outside the block."
        )
        code = self.substrate.generate(
            session_token_id="",
            prompt=prompt,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        return code

    def propose(self, request=None) -> ProposalResponse:
        desc = (getattr(request, "description", "") or getattr(request, "text", "") or "").lower()
        if any(kw in desc for kw in _CODE_KEYWORDS):
            return ProposalResponse(
                confidence=0.9,
                rationale=f"Code generation task detected in: {desc[:80]}",
                estimated_latency_ms=5000,
            )
        return ProposalResponse(
            confidence=0.3,
            rationale="General-purpose fallback bid for code generation.",
            estimated_latency_ms=5000,
        )


agent = CodeGeneratorAgent(
    agent_id="code_generator_agent",
    version="1.0.0",
    description=AGENT_DESCRIPTION,
)


if __name__ == "__main__":
    configure_logging(agent_id="code_generator_agent")
    agent.serve()
