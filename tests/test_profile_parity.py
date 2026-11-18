"""The SDK profile validator is held to the kernel's Go port (agent stack v2 part B).

cambrian-core/internal/agentprofile/testdata/profile_cases.json is written by the Go
validator (TestProfileCasesFixture; regenerate with ``go test ./internal/agentprofile
-run TestProfileCasesFixture -update-profile-cases``). The kernel registers a profile as
an agent only if its port accepts it, and worker_agent.py refuses at boot what this
module refuses, so the two must agree on every case: a profile the kernel registered but
the runtime refused would be an agent that is routed to and never answers. Error text is
not compared, only the verdict and the parsed fields.
"""
import fnmatch
import json
from pathlib import Path

import pytest

from cambrian_agent_sdk import profiles as pr

_FIXTURE = (Path(__file__).resolve().parents[2] / "cambrian-core" / "internal"
            / "agentprofile" / "testdata" / "profile_cases.json")
_needs_core = pytest.mark.skipif(not _FIXTURE.exists(),
                                 reason="cambrian-core checkout not alongside sdk/")


def _fixture():
    return json.loads(_FIXTURE.read_text(encoding="utf-8")) if _FIXTURE.exists() else {}


def _skills_dir(tmp_path: Path, skills):
    if skills is None:
        return None
    root = tmp_path / "skills"
    root.mkdir()
    for name, tools in skills.items():
        d = root / name
        d.mkdir()
        lines = ["---", f"name: {name}", "description: A fixture skill."]
        if tools:
            lines.append("tools:")
            lines += [f"  - {t}" for t in tools]
        lines += ["---", "Body."]
        (d / "SKILL.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return root


@_needs_core
@pytest.mark.parametrize("case", _fixture().get("cases", []), ids=lambda c: c["name"])
def test_profile_verdict_matches_go(case, tmp_path):
    sdir = _skills_dir(tmp_path, case["skills"])
    try:
        got = pr.parse_profile(case["text"], skills_dir=sdir, expected_name=case["expected_name"] or None)
    except pr.InvalidProfile:
        got = None
    assert (got is not None) == case["valid"], case["text"]
    if got is None:
        return
    want = case["profile"]
    assert got.name == want["name"]
    assert got.description == want["description"]
    assert got.role == want["role"]
    assert list(got.capabilities) == want["capabilities"]
    assert list(got.skills) == want["skills"]
    assert list(got.tools) == want["tools"]
    assert got.trust_tier == want["trust_tier"]
    assert got.model == want["model"]
    assert got.max_tool_rounds == want["max_tool_rounds"]
    assert got.max_tokens == want["max_tokens"]
    assert got.version == want["version"]
    assert list(got.examples) == (want.get("examples") or [])
    assert got.streams_output == bool(want.get("streams_output"))
    # ADR-0157 D1.2 (ENG-1960): the declared paths, in the manifest's shape.
    assert [p.manifest() for p in got.sandbox_paths] == (want.get("sandbox_paths") or [])
    # ADR-0157 D7.6 (ENG-2009): the declared packages, in the manifest's shape.
    assert [p.manifest() for p in got.python_packages] == (want.get("python_packages") or [])
    assert len(got.warnings) == case["warnings"]


@_needs_core
def test_constants_match_go():
    c = _fixture()["constants"]
    assert set(c["capability_vocabulary"]) == pr.CAPABILITY_VOCABULARY
    assert set(c["reserved_capabilities"]) == pr.RESERVED_CAPABILITIES
    assert tuple(c["trust_tiers"]) == pr.TRUST_TIERS
    assert tuple(c["model_roles"]) == pr.MODEL_ROLES
    assert set(c["fields"]) == pr._FIELDS
    assert tuple(c["required"]) == pr._REQUIRED
    assert tuple(c["max_tool_rounds"]) == pr._MAX_TOOL_ROUNDS
    assert tuple(c["max_tokens"]) == pr._MAX_TOKENS
    assert c["max_examples"] == pr._MAX_EXAMPLES
    assert c["max_example_runes"] == pr._MAX_EXAMPLE_CHARS


@_needs_core
@pytest.mark.parametrize("g", _fixture().get("globs", []), ids=lambda g: f"{g['pattern']}~{g['name']}")
def test_glob_matching_matches_go(g):
    """The kernel's grant/ceiling matcher (domain.MatchToolGlob) and the client-side
    filter (ToolAllowlist.allows, fnmatchcase) agree on every name."""
    assert fnmatch.fnmatchcase(g["name"], g["pattern"]) == g["want"]
