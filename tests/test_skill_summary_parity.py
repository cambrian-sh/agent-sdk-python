"""One deriver for skill menu lines (ADR-0045 D3 / ADR-0046 D5).

The kernel caps a system skill's served description with ``deriveSummary`` (first
sentence or line, 120-char word-boundary cap, humanized-name fallback); agent-local
skills used to be rendered uncapped. ``react.derive_summary`` is the Python port, held
here to the fixture the Go test writes from the real deriver
(cambrian-core/internal/skill/discovery/testdata/summary_cases.json, pinned by
TestSummaryCasesFixture).
"""
import json
from pathlib import Path

import pytest

from cambrian_agent_sdk import CognitiveAgent
from cambrian_agent_sdk import react

_FIXTURE = (Path(__file__).resolve().parents[2] / "cambrian-core" / "internal" / "skill"
            / "discovery" / "testdata" / "summary_cases.json")


def _cases():
    if not _FIXTURE.exists():
        return []
    return json.loads(_FIXTURE.read_text(encoding="utf-8"))


@pytest.mark.skipif(not _FIXTURE.exists(), reason="cambrian-core checkout not alongside sdk/")
@pytest.mark.parametrize("case", _cases(), ids=lambda c: c["name"])
def test_derive_summary_matches_go(case):
    assert react.derive_summary(case["name"], case["description"]) == case["want"]


class _Bot(CognitiveAgent):
    def run(self, task):
        return self.think(task)


def test_local_skill_menu_line_is_derived_like_a_system_skill():
    bot = _Bot(agent_id="b")
    long_desc = ("Investigate a Python codebase module by module and record what each "
                 "public function does, its callers, and any TODOs.\nSecond paragraph.")
    bot.local_skills = [{"name": "investigate", "description": long_desc,
                         "instructions": "body", "tool_grants": []}]
    (sk,) = react._agent_local_skills(bot)
    assert sk["description"] == react.derive_summary("investigate", long_desc)
    assert len(sk["description"]) <= react.SUMMARY_MAX_CHARS + 1  # + the ellipsis
    assert "Second paragraph" not in sk["description"]
    # The body is untouched: only the menu line is derived.
    assert sk["instructions"] == "body"
