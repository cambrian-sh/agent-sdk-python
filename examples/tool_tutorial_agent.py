"""SDK ``@tool`` tutorial: a tiny calculator agent.

THIS IS A TUTORIAL, NOT A SHIPPED AGENT. Nothing deploys it, and the kernel's curated
agents do not include a calculator (owner directive
decisions/agent-stack-v2-mcp-tools-and-profiles.md: specialists are declarative
profiles over ``worker_agent``, and tools come from MCP servers). It exists to show,
in one screen, the two pieces of the SDK an agent author meets first:

  * the ``@tool`` intra-agent registry: a closed, schema-validated menu of local
    functions (add/subtract/multiply/divide) the agent's own LLM may call. The schema
    is derived from the type hints, and the SDK validates arguments and invokes the
    bound method directly (no eval);
  * the ``think()`` ReAct loop: ``CognitiveAgent``'s default ``run()`` reasons, picks a
    tool by name, and feeds the result back until it can answer.

Run it against a local kernel with::

    python tool_tutorial_agent.py --port 50070 --substrate-addr localhost:50051

and send it a task such as "what is 47 times 89 plus 12?". To have the kernel discover
it, a file must end in ``agent.py`` and carry the AGENT_DESCRIPTION / AGENT_MANIFEST
literals below (the kernel reads them by regex before the process starts).
"""

from cambrian_agent_sdk import CognitiveAgent, tool
from cambrian_agent_sdk._logging import configure_logging

AGENT_DESCRIPTION = "Tutorial agent that solves arithmetic with four local calculator tools."

AGENT_MANIFEST = '''
{
  "version": "1.0.0",
  "trait": "cognitive",
  "capabilities": ["calculation"],
  "supported_formats": ["text"],
  "release_notes": "SDK @tool tutorial: @tool registry + think() ReAct loop.",
  "dependencies": []
}
'''


class ToolTutorialAgent(CognitiveAgent):
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

    # No run() override needed: CognitiveAgent's default run() drives think(), which
    # puts the @tool menu in front of the model and executes the calls it makes.


agent = ToolTutorialAgent(agent_id="tool_tutorial_agent", version="1.0.0", description=AGENT_DESCRIPTION)


if __name__ == "__main__":
    configure_logging(agent_id="tool_tutorial_agent")
    agent.serve()
