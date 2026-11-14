"""The package docstring's "Minimal usage" example runs as written.

It once showed an ``Agent(capabilities=...)`` / ``@agent.capability`` API and a
``cambrian manifest`` command that do not exist
(T-20261113-sdk-package-docstring-and-budgetexceedederror).
"""
from __future__ import annotations

import json
import re
import textwrap

import cambrian_agent_sdk
from cambrian_agent_sdk import AgentTask


def _example() -> str:
    doc = cambrian_agent_sdk.__doc__ or ""
    m = re.search(r"Minimal usage[^\n]*::\n\n((?:    .*\n|\n)+)", doc)
    assert m, "package docstring lost its Minimal usage block"
    return textwrap.dedent(m.group(1))


def test_minimal_usage_example_runs() -> None:
    ns: dict[str, object] = {"__name__": "example_agent"}
    exec(compile(_example(), "<docstring example>", "exec"), ns)  # noqa: S102
    agent = ns["agent"]
    assert agent.run(AgentTask(text="hi")) == "Echo: hi"  # type: ignore[attr-defined]
    manifest = json.loads(ns["AGENT_MANIFEST"])  # type: ignore[arg-type]
    assert manifest["trait"] == "cognitive"
