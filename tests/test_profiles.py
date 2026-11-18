"""Declarative agent profiles (agent stack v2): the loader, the validator and the
client-side scope (cambrian_agent_sdk/profiles.py; format in
cambrian-core/agents/profiles/README.md)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cambrian_agent_sdk.profiles import (
    InvalidProfile,
    ProfileScopedSubstrate,
    PythonPackage,
    SandboxPath,
    ToolAllowlist,
    load_profile,
    load_profiles,
    parse_profile,
    skill_index,
)

_VALID = """\
# a comment line is allowed
name: analyst
version: 1.2.0
description: Analyses data in a sandbox and reports findings with caveats.
role: "You are a careful analyst."
capabilities:
  - analysis
  - data_analysis
skills:
  - report
tools:
  - mcp:code-sandbox/*
  - mcp:filesystem/read_*
  - mcp:filesystem/write_file
model: role:planner
trust_tier: first-party
max_tool_rounds: 12
max_tokens: 4096
"""

_SKILL = """\
---
name: report
description: Write a report.
version: 1.0.0
scope: []
tools:
  - mcp:filesystem/read_text_file
  - mcp:filesystem/write_file
---

# Report

Body.
"""


@pytest.fixture
def skills(tmp_path: Path) -> Path:
    d = tmp_path / "skills"
    (d / "report").mkdir(parents=True)
    (d / "report" / "SKILL.md").write_text(_SKILL, encoding="utf-8")
    return d


def _profile(tmp_path: Path, text: str, name: str = "analyst") -> Path:
    d = tmp_path / "profiles"
    d.mkdir(exist_ok=True)
    p = d / f"{name}.yaml"
    p.write_text(text, encoding="utf-8")
    return p


# ── the valid profile ────────────────────────────────────────────────────────


def test_valid_profile_loads_every_field(tmp_path, skills):
    _profile(tmp_path, _VALID)
    p = load_profile("analyst", profiles_dir=tmp_path / "profiles", skills_dir=skills)
    assert p.name == "analyst"
    assert p.version == "1.2.0"
    assert p.role == "You are a careful analyst."
    assert p.capabilities == ("analysis", "data_analysis")
    assert p.skills == ("report",)
    assert p.tools == ("mcp:code-sandbox/*", "mcp:filesystem/read_*", "mcp:filesystem/write_file")
    assert p.model == "role:planner"
    assert p.trust_tier == "first-party"
    assert (p.max_tool_rounds, p.max_tokens) == (12, 4096)
    assert p.warnings == ()


def test_crlf_profile_loads(tmp_path, skills):
    _profile(tmp_path, _VALID.replace("\n", "\r\n"))
    assert load_profile("analyst", profiles_dir=tmp_path / "profiles", skills_dir=skills).name == "analyst"


def test_optional_fields_default(skills):
    text = "\n".join(line for line in _VALID.splitlines()
                     if not line.startswith(("model", "max_", "version")))
    p = parse_profile(text, skills_dir=skills)
    assert (p.model, p.max_tool_rounds, p.max_tokens, p.version) == ("", None, None, "1.0.0")


def test_manifest_keeps_allowlist_out_of_legacy_tools_key(skills):
    m = parse_profile(_VALID, skills_dir=skills).manifest()
    # The kernel reads manifest "tools" as capability tags (dispatch _capability).
    assert "tools" not in m
    assert m["tool_allowlist"] == ["mcp:code-sandbox/*", "mcp:filesystem/read_*", "mcp:filesystem/write_file"]
    assert m["capabilities"] == ["analysis", "data_analysis"]
    assert m["trait"] == "cognitive" and m["profile"] == "analyst" and m["skills"] == ["report"]
    json.dumps(m)


def test_streams_output_reaches_the_manifest(skills):
    """ADR-0139 amendment 2026-11-06: an agent that pushes its own live output says
    so, and the kernel then forwards no gateway deltas for it."""
    assert parse_profile(_VALID, skills_dir=skills).streams_output is False
    assert "streams_output" not in parse_profile(_VALID, skills_dir=skills).manifest()
    p = parse_profile(_VALID + "streams_output: true\n", skills_dir=skills)
    assert p.streams_output is True and p.manifest()["streams_output"] is True
    with pytest.raises(InvalidProfile):
        parse_profile(_VALID + "streams_output: yes\n", skills_dir=skills)


def test_sandbox_paths_reach_the_manifest(skills):
    """ADR-0157 D1.2 (ENG-1960): a profile declares the paths its agent needs when
    confined, in the manifest's shape; the kernel admits them at boot against
    allowlist_cap like an AGENT_MANIFEST's."""
    assert parse_profile(_VALID, skills_dir=skills).sandbox_paths == ()
    assert "sandbox_paths" not in parse_profile(_VALID, skills_dir=skills).manifest()
    text = _VALID + 'sandbox_paths: [{"path": "/srv/models"}, {"path": "~/cache/hf", "write": true}]\n'
    p = parse_profile(text, skills_dir=skills)
    assert p.sandbox_paths == (SandboxPath("/srv/models"), SandboxPath("~/cache/hf", write=True))
    m = p.manifest()
    assert m["sandbox_paths"] == [{"path": "/srv/models"}, {"path": "~/cache/hf", "write": True}]
    json.dumps(m)


@pytest.mark.parametrize("value", [
    '[{"path": "models/cache"}]',          # relative
    '[{"path": "/srv/a", "mode": "rw"}]',  # unknown key
    '[{"path": "/srv/a", "write": "true"}]',
    '[{"path": "/srv/a"}, {"path": "/srv/a"}]',
    '[/srv/models]',                       # a flow list of strings is not the shape
    '[{"path": "/srv/a",}]',               # not JSON
])
def test_sandbox_paths_refused(skills, value):
    with pytest.raises(InvalidProfile):
        parse_profile(_VALID + f"sandbox_paths: {value}\n", skills_dir=skills)


def test_python_packages_reach_the_manifest(skills):
    """ADR-0157 D7.6 (ENG-2009): a profile declares the packages its agent's own layer
    needs, exact versions, in the manifest's shape; the kernel resolves, card-approves and
    installs them before it starts the agent."""
    assert parse_profile(_VALID, skills_dir=skills).python_packages == ()
    assert "python_packages" not in parse_profile(_VALID, skills_dir=skills).manifest()
    text = _VALID + ('python_packages: [{"name": "polars", "version": "1.9.0"}, '
                     '{"name": "httpx", "version": "0.27.2", "extras": ["http2"]}]\n')
    p = parse_profile(text, skills_dir=skills)
    assert p.python_packages == (PythonPackage("polars", "1.9.0"),
                                 PythonPackage("httpx", "0.27.2", ("http2",)))
    m = p.manifest()
    assert m["python_packages"] == [{"name": "polars", "version": "1.9.0"},
                                    {"name": "httpx", "version": "0.27.2", "extras": ["http2"]}]
    json.dumps(m)


@pytest.mark.parametrize("value", [
    '[{"name": "polars", "version": ">=1.9"}]',               # a range
    '[{"name": "polars", "version": "==1.9.0"}]',             # an operator
    '[{"name": "polars", "version": "1.9.0\\n"}]',           # trailing newline (fullmatch)
    '[{"name": "polars", "version": 1.9}]',                   # not a string
    '[{"name": "polars", "version": "1.9.0", "url": "x"}]',   # unknown key
    '[{"name": "Foo_Bar", "version": "1"}, {"name": "foo-bar", "version": "2"}]',
    '[{"name": "httpx", "version": "1", "extras": ["a", "A"]}]',
    '[polars]',
])
def test_python_packages_refused(skills, value):
    with pytest.raises(InvalidProfile):
        parse_profile(_VALID + f"python_packages: {value}\n", skills_dir=skills)


# ── strictness ───────────────────────────────────────────────────────────────


def test_unknown_field_is_an_error(skills):
    with pytest.raises(InvalidProfile, match="unknown field"):
        parse_profile(_VALID + "temperature: 0.2\n", skills_dir=skills)


@pytest.mark.parametrize("field", ["name", "description", "capabilities", "skills", "tools",
                                   "trust_tier", "role"])
def test_missing_required_field_is_an_error(field, skills):
    lines, skip = [], False
    for line in _VALID.splitlines():
        if line.startswith(field + ":"):
            skip = True
            continue
        if skip and line.startswith("  "):
            continue
        skip = False
        lines.append(line)
    with pytest.raises(InvalidProfile, match="missing required"):
        parse_profile("\n".join(lines) + "\n", skills_dir=skills)


def test_missing_skill_is_an_error(skills):
    with pytest.raises(InvalidProfile, match=r"skill\(s\) \['nope'\] not found"):
        parse_profile(_VALID.replace("  - report\n", "  - report\n  - nope\n"), skills_dir=skills)


def test_skill_tools_outside_the_allowlist_are_an_error(skills):
    text = _VALID.replace("  - mcp:filesystem/write_file\n", "")
    with pytest.raises(InvalidProfile, match="grants .*write_file"):
        parse_profile(text, skills_dir=skills)


@pytest.mark.parametrize("glob", [
    "mcp:*/read_*",            # wildcard server id
    "mcp:filesystem",          # no tool part
    "mcp:file system/x",       # whitespace
    "mcp:filesystem/[ab]*",    # character class
    "mcp:filesystem/**",       # double star
    "*",                       # everything
    "ftp:server/x",            # unknown namespace
    "'  '",                    # blank
])
def test_bad_glob_is_an_error(glob, skills):
    text = _VALID.replace("  - mcp:code-sandbox/*\n", f"  - {glob}\n")
    with pytest.raises(InvalidProfile):
        parse_profile(text, skills_dir=skills)


@pytest.mark.parametrize("old,new,match", [
    ("trust_tier: first-party", "trust_tier: trusted", "trust_tier"),
    ("  - analysis\n", "  - Analysis\n", "lower_snake_case"),
    ("  - analysis\n", "  - chat\n", "reserved"),
    ("max_tool_rounds: 12", "max_tool_rounds: 0", "max_tool_rounds"),
    ("max_tokens: 4096", "max_tokens: lots", "max_tokens"),
    ("model: role:planner", "model: role:interview", "role must be one of"),
    ("name: analyst", "name: Analyst", "name"),
    ("version: 1.2.0", "version: one", "version"),
])
def test_field_value_errors(old, new, match, skills):
    with pytest.raises(InvalidProfile, match=match):
        parse_profile(_VALID.replace(old, new), skills_dir=skills)


def test_name_must_equal_file_name(tmp_path, skills):
    _profile(tmp_path, _VALID, name="other")
    with pytest.raises(InvalidProfile, match="file name"):
        load_profile("other", profiles_dir=tmp_path / "profiles", skills_dir=skills)


def test_yaml_outside_the_subset_is_an_error(skills):
    with pytest.raises(InvalidProfile, match="subset"):
        parse_profile(_VALID.replace('role: "You are a careful analyst."',
                                     "role: |\n  multi\n  line"), skills_dir=skills)


def test_description_with_quotes_is_an_error(skills):
    with pytest.raises(InvalidProfile, match="quote"):
        parse_profile(_VALID.replace("with caveats.", "with 'caveats'."), skills_dir=skills)


def test_new_capability_is_a_warning_not_an_error(skills):
    p = parse_profile(_VALID.replace("  - data_analysis\n", "  - network_analysis\n"), skills_dir=skills)
    assert "network_analysis" in p.capabilities
    assert any("network_analysis" in w for w in p.warnings)


def test_skills_dir_none_skips_the_skill_checks():
    assert parse_profile(_VALID.replace("  - report\n", "  - nope\n"), skills_dir=None).skills == ("nope",)


def test_skill_index_falls_back_for_full_yaml_frontmatter(tmp_path):
    d = tmp_path / "skills" / "x"
    d.mkdir(parents=True)
    (d / "SKILL.md").write_text(
        "---\nname: fancy\ndescription: >\n  folded text\ntools: [mcp:a/b, mcp:a/c]\n---\nbody\n",
        encoding="utf-8")
    idx = skill_index(tmp_path / "skills")
    assert idx["fancy"].tools == ("mcp:a/b", "mcp:a/c")


# ── the allowlist and the scoped substrate ───────────────────────────────────


def test_allowlist_globs():
    a = ToolAllowlist(["mcp:filesystem/read_*", "mcp:fetch/*", "calculate"])
    assert a.allows("mcp:filesystem/read_text_file")
    assert not a.allows("mcp:filesystem/write_file")
    assert a.allows("mcp:fetch/fetch")
    assert not a.allows("mcp:fetcher/fetch")
    assert a.allows("calculate") and not a.allows("calc")
    assert not a.allows("")


class _Inner:
    def __init__(self):
        self.executed = []
        self.skill_calls = []

    def list_tools(self, query="", k=0, names=None, full=False):
        all_ = [{"name": n} for n in ("mcp:code-sandbox/execute_code", "mcp:filesystem/read_file",
                                      "mcp:filesystem/delete_all", "mcp:shell/shell_execute")]
        return [t for t in all_ if not names or t["name"] in names]

    def list_skills(self, query="", k=0, names=None, full=False, session_token_id=""):
        self.skill_calls.append({"query": query, "names": names, "full": full})
        known = {"report": ["mcp:filesystem/read_text_file", "mcp:shell/shell_execute"],
                 "other": []}
        return [{"name": n, "description": n, "instructions": "b" if full else "",
                 "tool_grants": known[n] if full else []}
                for n in (names or known) if n in known]

    def execute_tool(self, name, args_json="", **kw):
        self.executed.append(name)
        return {"result_json": "{}"}

    def embed(self, text):
        return [0.0]


def test_scoped_substrate(skills):
    prof = parse_profile(_VALID, skills_dir=skills)
    inner = _Inner()
    sub = ProfileScopedSubstrate(inner, prof)

    assert [t["name"] for t in sub.list_tools(query="x")] == [
        "mcp:code-sandbox/execute_code", "mcp:filesystem/read_file"]
    assert sub.list_tools(names=["mcp:shell/shell_execute"]) == []

    # The menu is the kernel's full listing (no query, no names), and against a kernel
    # that marks no access (an older one) it is bounded to the profile's skills.
    menu = sub.list_skills(query="anything", k=5)
    assert [s["name"] for s in menu] == ["report"]
    assert inner.skill_calls[-1] == {"query": "", "names": None, "full": False}
    # An unmarked load of a skill outside the profile is dropped; a load clips the grants.
    assert sub.list_skills(names=["other"], full=True) == []
    loaded = sub.list_skills(names=["report"], full=True, session_token_id="s")
    assert loaded[0]["tool_grants"] == ["mcp:filesystem/read_text_file"]  # shell clipped

    denied = sub.execute_tool("mcp:shell/shell_execute", args_json="{}")
    assert denied["denied"] and "allowlist" in denied["deny_reason"]
    sub.execute_tool("mcp:filesystem/read_file", args_json="{}")
    assert inner.executed == ["mcp:filesystem/read_file"]

    assert sub.embed("x") == [0.0]  # everything else passes through


# ── the shipped starter profiles ─────────────────────────────────────────────

_CORE = Path(__file__).resolve().parents[2] / "cambrian-core"


@pytest.mark.skipif(not (_CORE / "agents" / "profiles").is_dir(),
                    reason="cambrian-core is not checked out beside sdk/")
def test_shipped_starter_profiles_validate():
    profs = load_profiles(_CORE / "agents" / "profiles", skills_dir=_CORE / "skills")
    assert set(profs) >= {"assistant", "researcher", "data-analyst"}
    for p in profs.values():
        assert p.warnings == (), (p.name, p.warnings)


# ── per-agent skill access (decisions/skill-access-grants.md) ─────────────────


class _GrantingKernel(_Inner):
    """A kernel that bounds the agent: profile skill "report", granted skill "granted-x",
    "other" outside the set (requestable on discovery, pending on a load)."""

    def __init__(self, load_answer="pending"):
        super().__init__()
        self.load_answer = load_answer
        self.rationales = []

    def list_skills(self, query="", k=0, names=None, full=False, session_token_id="",
                    rationale=""):
        self.skill_calls.append({"query": query, "names": names, "full": full})
        self.rationales.append(rationale)
        sets = {"report": "profile", "granted-x": "granted"}
        if names:
            out = []
            for n in names:
                acc = sets.get(n) or (self.load_answer if full else "requestable")
                body = full and acc in ("profile", "granted")
                out.append({"name": n, "description": n, "instructions": "BODY" if body else "",
                            "tool_grants": [], "access": acc, "access_detail": "" if body else "why"})
            return out
        if query:
            return [{"name": n, "description": n, "instructions": "", "tool_grants": [],
                     "access": sets.get(n, "requestable")} for n in ("other", "report")]
        return [{"name": n, "description": n, "instructions": "", "tool_grants": [], "access": a}
                for n, a in sets.items()]


def test_scoped_substrate_effective_set_from_kernel(skills):
    prof = parse_profile(_VALID, skills_dir=skills)
    kern = _GrantingKernel()
    sub = ProfileScopedSubstrate(kern, prof)
    # Menu = profile + grants, as the kernel marked them.
    assert {s["name"]: s["access"] for s in sub.list_skills()} == {"report": "profile", "granted-x": "granted"}
    # A granted skill loads with its body.
    assert sub.list_skills(names=["granted-x"], full=True)[0]["instructions"] == "BODY"
    # Discovery keeps requestable skills; the plain (menu) query does not.
    assert {s["name"] for s in sub.list_skills(query="need", k=3, discover=True)} == {"other", "report"}
    assert "other" not in {s["name"] for s in sub.list_skills(query="need", k=3)}
    # A load outside the set passes to the kernel with the rationale; no body comes back.
    pend = sub.list_skills(names=["other"], full=True, rationale="for the report")
    assert pend[0]["access"] == "pending" and not pend[0]["instructions"]
    assert kern.rationales[-1] == "for the report"


def test_scoped_skill_menu_order_is_deterministic():
    """T-20261108-system-prompt-skill-list-order-is-not-stable: whatever order the kernel
    serves the menu in, the profile menu (rendered into the system prompt) is sorted."""

    class _Shuffled:
        def __init__(self, order):
            self.order = order

        def list_skills(self, **kw):
            return [{"name": n, "description": n, "access": "profile"} for n in self.order]

    class _Prof:
        name = "p"
        skills = ("summarise", "file-report", "organise-files")
        allowlist = None

    a = ProfileScopedSubstrate(_Shuffled(["summarise", "file-report", "organise-files"]), _Prof())
    b = ProfileScopedSubstrate(_Shuffled(["organise-files", "summarise", "file-report"]), _Prof())
    want = ["file-report", "organise-files", "summarise"]
    assert [s["name"] for s in a.list_skills(query="step one")] == want
    assert [s["name"] for s in b.list_skills(query="step two")] == want
