"""The SDK prompt loader is held to the kernel's Go loader (cambrian-core/prompts).

cambrian-core/prompts/testdata/hash_cases.json is written by the Go loader
(TestHashCasesFixture, regenerate with ``go test ./prompts -run
TestHashCasesFixture -update-hash-cases``). It carries rendering cases (template,
slots -> text, static hash), how Go read every real core prompt file, and the lint
constants. This file checks the Python loader and lint agree on all three.
"""
import json
from pathlib import Path

import pytest
import test_prompt_lint as lint

from cambrian_agent_sdk import prompt_library as pl

_CORE_PROMPTS = Path(__file__).resolve().parents[2] / "cambrian-core" / "prompts"
_FIXTURE = _CORE_PROMPTS / "testdata" / "hash_cases.json"
_needs_core = pytest.mark.skipif(not _FIXTURE.exists(),
                                 reason="cambrian-core checkout not alongside sdk/")


def _fixture():
    return json.loads(_FIXTURE.read_text(encoding="utf-8")) if _FIXTURE.exists() else {}


@_needs_core
def test_placeholder_matches_go():
    assert _fixture()["placeholder"] == pl.PLACEHOLDER


@_needs_core
@pytest.mark.parametrize("case", _fixture().get("cases", []), ids=lambda c: c["name"])
def test_render_and_hash_match_go(case):
    p = pl.Prompt(id=case["name"], version=1, tier="stable", inputs=tuple(case["inputs"]),
                  description="d", body=pl.normalise_body(case["template"]))
    r = p.render(case["slots"])
    assert r.text == case["want_text"]
    assert r.static_hash == case["want_hash"]
    assert pl.static_hash(case["template"]) == case["want_hash"]


@_needs_core
def test_python_reads_core_prompt_files_like_go():
    """The YAML-subset parser reads every real core prompt as yaml.v3 does."""
    lib = pl.PromptLibrary.load(_CORE_PROMPTS)
    got = [{"id": p.id, "version": p.version, "tier": p.tier, "inputs": list(p.inputs),
            "description": p.description, "has_schema": p.response_schema is not None,
            "static_hash": p.static_hash} for p in lib]
    assert got == _fixture()["prompts"]


@_needs_core
def test_lint_constants_match_go():
    fx = _fixture()["lint"]
    assert set(fx["acronyms"]) == lint.ACRONYMS
    assert fx["max_caps_run"] == lint.MAX_CAPS_RUN
    assert fx["min_shout_words"] == lint.MIN_SHOUT_WORDS
    for body, fail in fx["bodies_fail"].items():
        assert bool(lint.lint_body(body)[0]) == fail, body
