"""ADR-0036 issue 0036-03: the @tool intra-agent registry (D4).

A closed, schema-validated menu of local Python functions the agent's own LLM may
call — distinct from, and orthogonal to, @capability (inter-agent auction routing).
"""

from typing import Optional

import pytest

from cambrian_agent_sdk import CognitiveAgent, tool


class Bot(CognitiveAgent):
    @tool
    def search(self, query: str, limit: int = 10) -> str:
        return f"{query}:{limit}"

    def run(self, task):
        return "x"


from cambrian_agent_sdk import capability


def test_tool_auto_derives_schema_from_type_hints():
    schema = Bot(agent_id="b").tools.schema("search")
    assert schema["type"] == "object"
    assert schema["properties"]["query"] == {"type": "string"}
    assert schema["properties"]["limit"] == {"type": "integer"}
    assert schema["required"] == ["query"]  # limit has a default ⇒ optional


def test_optional_hint_is_unwrapped_and_not_required():
    class A(CognitiveAgent):
        @tool
        def f(self, name: str, note: Optional[str] = None) -> str:
            return name

        def run(self, task):
            return "x"

    schema = A(agent_id="a").tools.schema("f")
    assert schema["properties"]["note"] == {"type": "string"}
    assert schema["required"] == ["name"]  # Optional ⇒ not required


def test_explicit_schema_overrides_derivation():
    custom = {"type": "object", "properties": {"x": {"type": "number"}}, "required": ["x"]}

    class A(CognitiveAgent):
        @tool(name="calc", schema=custom)
        def _calc(self, x):
            return x * 2

        def run(self, task):
            return "x"

    a = A(agent_id="a")
    assert "calc" in a.tools
    assert a.tools.schema("calc") is custom


def test_valid_call_invokes_the_bound_method_directly():
    a = Bot(agent_id="b")
    assert a.tools.call("search", query="cats", limit=3) == "cats:3"


def test_missing_required_argument_returns_structured_error():
    a = Bot(agent_id="b")
    out = a.tools.call("search")  # query missing
    assert isinstance(out, dict)
    assert out["tool"] == "search"
    assert "query" in out["error"]


def test_wrong_type_returns_structured_error():
    a = Bot(agent_id="b")
    out = a.tools.call("search", query=123)  # query must be string
    assert out["tool"] == "search" and "string" in out["error"]


def test_unknown_tool_returns_structured_error():
    a = Bot(agent_id="b")
    out = a.tools.call("nonexistent")
    assert out == {"error": "unknown tool 'nonexistent'", "tool": "nonexistent"}


def test_in_tool_exception_degrades_to_structured_error():
    class A(CognitiveAgent):
        @tool
        def boom(self, x: int) -> int:
            raise ValueError("kaboom")

        def run(self, task):
            return "x"

    out = A(agent_id="a").tools.call("boom", x=1)
    assert out["tool"] == "boom" and "kaboom" in out["error"]


def test_no_exec_or_eval_in_the_tool_path():
    """Closed-menu invocation must never route author/LLM input through exec/eval."""
    import cambrian_agent_sdk.tools as tools_mod
    import inspect as _inspect

    src = _inspect.getsource(tools_mod)
    assert "exec(" not in src
    assert "eval(" not in src


def test_tool_and_capability_are_independent_registries():
    class Mixed(CognitiveAgent):
        @tool
        def lookup(self, q: str) -> str:
            return q

        @capability
        def answer_questions(self, task):
            return "answered"

        def run(self, task):
            return "x"

    m = Mixed(agent_id="m")
    # the @tool is in the tool registry, NOT among capabilities
    assert "lookup" in m.tools
    assert "lookup" not in m.capability_names
    # the @capability is in the capability registry, NOT among tools
    assert "answer_questions" in m.capability_names
    assert "answer_questions" not in m.tools
