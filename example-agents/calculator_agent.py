"""Calculator agent (CognitiveAgent + @tool) — SDK v2 demo.

Solves arithmetic and math word problems. Demonstrates the two headline SDK v2
features together:

  * the ``@tool`` intra-agent registry — a closed, schema-validated menu of local
    functions (add/subtract/multiply/divide) the agent's own LLM may call;
  * the ``think()`` ReAct loop — the default ``run()`` reasons, picks a tool by name,
    and the SDK validates the args and invokes the bound method directly (no eval).

Try it with tasks like "what is 47 times 89 plus 12?".
"""

from cambrian_agent_sdk import CognitiveAgent, tool
from cambrian_agent_sdk.types import ProposalResponse
from cambrian_agent_sdk._logging import configure_logging

AGENT_DESCRIPTION = "Solves arithmetic and mathematical word problems using calculator tools (add, subtract, multiply, divide). Use for any computation, math, or numeric reasoning task."

AGENT_MANIFEST = '''
{
  "version": "1.0.0",
  "trait": "cognitive",
  "supported_formats": ["text"],
  "tools": ["calculation", "arithmetic", "math"],
  "release_notes": "SDK v2 demo: @tool registry + think() ReAct loop.",
  "dependencies": []
}
'''

_MATH_KEYWORDS = (
    "calculate", "compute", "math", "arithmetic", "sum", "add", "subtract",
    "multiply", "divide", "product", "plus", "minus", "times", "how much",
    "how many", "total", "average",
)


class CalculatorAgent(CognitiveAgent):
    # The think() loop uses this persona in the <Role> section of the prompt.
    role = "a precise calculator. Use the provided tools for every arithmetic step rather than computing in your head."

    @tool
    def add(self, a: float, b: float) -> float:
        """Return the sum a + b."""
        return a + b

    @tool
    def subtract(self, a: float, b: float) -> float:
        """Return the difference a - b."""
        return a - b

    @tool
    def multiply(self, a: float, b: float) -> float:
        """Return the product a * b."""
        return a * b

    @tool
    def divide(self, a: float, b: float) -> float:
        """Return the quotient a / b (errors on division by zero)."""
        if b == 0:
            raise ValueError("division by zero")
        return a / b

    # No run() override needed — CognitiveAgent's default run() drives think(),
    # which injects the @tool menu into the prompt and executes tool calls.

    def propose(self, request=None) -> ProposalResponse:
        desc = (getattr(request, "description", "") or getattr(request, "text", "") or "").lower()
        if any(k in desc for k in _MATH_KEYWORDS):
            return ProposalResponse(confidence=0.9, rationale="Arithmetic/math task detected", estimated_latency_ms=4000)
        return ProposalResponse(confidence=0.2, rationale="Not obviously a math task", estimated_latency_ms=4000)


agent = CalculatorAgent(agent_id="calculator_agent", version="1.0.0", description=AGENT_DESCRIPTION)


if __name__ == "__main__":
    configure_logging(agent_id="calculator_agent")
    agent.serve()
