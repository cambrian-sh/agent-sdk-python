"""Declarative agent profiles (agent stack v2; ADR-0139 C2/C3).

A **profile** is one specialist: a name, routing text, the capabilities it declares, the
skills it loads, the tools it may reach, an optional model preference and a trust tier.
One runtime (``cambrian-core/agents/worker_agent.py``) runs any profile, so a customer
authors a data analyst or a network analyst by writing a file, not an agent. The format
is specified in ``cambrian-core/agents/profiles/README.md``; this module is its parser
and validator, and the client-side half of the tool allowlist.

**YAML.** PyYAML is not an SDK dependency, and a profile is flat (scalars and lists of
scalars), so a profile file is the same strict YAML subset as a prompt's frontmatter
(``cambrian-core/prompts/SPEC.md`` §2.2) and is parsed by the same parser
(:func:`cambrian_agent_sdk.prompt_library.parse_frontmatter`). Every file in the subset is
also valid YAML that a full parser reads the same way; the kernel ports this subset
(internal/agentprofile) rather than using one, so both refuse the same files. Anything
outside it — multi-line strings, anchors, nested maps, an unquoted value containing
``": "`` — is an error, not a guess.

**Validation is strict.** An unknown field, a missing required field, a skill that does
not exist, a malformed tool glob, or a skill whose tools fall outside the profile's
allowlist all raise :class:`InvalidProfile`. Capabilities outside the fleet's current
vocabulary are accepted (the kernel's vocabulary is the union of what agents declare,
ADR-0100) but reported in :attr:`Profile.warnings`, because a new tag routes nothing
until a planner step asks for it.

**Enforcement.** :class:`ProfileScopedSubstrate` restricts the menu and refuses calls
outside the allowlist on the client, which keeps the menu small. The kernel is the
enforcing control (decisions/agent-stack-v2-mcp-tools-and-profiles.md D1/D3): it
registers each profile as an agent, grants it exactly its ``tools`` globs' expansion over
the tool registry (re-expanded when MCP servers connect), and holds the same globs as a
hard ceiling that ``tools_unrestricted`` does not lift (cambrian-core
app/profile_agents.go, domain/tool_ceiling.go). The kernel's validator is a Go port of
this module, held to it by tests/test_profile_parity.py.
"""

from __future__ import annotations

import fnmatch
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

from .prompt_library import InvalidPrompt, parse_frontmatter, split_frontmatter

__all__ = [
    "Profile",
    "InvalidProfile",
    "ToolAllowlist",
    "ProfileScopedSubstrate",
    "load_profile",
    "load_profiles",
    "parse_profile",
    "skill_index",
    "CAPABILITY_VOCABULARY",
    "RESERVED_CAPABILITIES",
    "TRUST_TIERS",
]

logger = logging.getLogger("cambrian_agent_sdk.profiles")

#: ADR-0139 D8 containment tiers.
TRUST_TIERS = ("system", "first-party", "third-party")

#: The capability tags the fleet declares today, in the spelling it declares them
#: (the L1 gate compares verbatim unless ``execution.canonical_vocab`` is on, so the
#: spelling matters). The kernel has no closed list: its vocabulary is the union of the
#: registered agents' manifests (domain/capability_resolve.go). This list is that union
#: for the shipped dispatchable fleet, which since agent stack v2 part B is the starter
#: profiles (cambrian-core/agents/profiles/*.yaml); the tags only the deleted thin
#: agents declared (planning, calculation, shell_execution, ...) left with them. A
#: profile may declare a tag outside it; it is reported as a warning, not refused. The
#: kernel's port (internal/agentprofile.CapabilityVocabulary) is held to this set by
#: tests/test_profile_parity.py.
CAPABILITY_VOCABULARY = frozenset({
    "general_purpose",
    "analysis", "data_analysis", "code_execution",
    "research", "web_research",
    "file_read", "file_write", "file_organisation",
    "summarisation",
})

#: Tags a profile may not declare: they belong to the chat front desk, and the audit
#: (§7 Routing) drops them from the dispatchable vocabulary.
RESERVED_CAPABILITIES = frozenset({"conversation", "chat"})

#: Generator roles a ``model: role:<name>`` preference may name (operator.proto
#: RoleAssignmentOp; the retired "interview" role is not accepted).
MODEL_ROLES = ("planner", "verifier", "router", "memory", "facts")

_FIELDS = {
    "name", "description", "capabilities", "skills", "tools", "model", "trust_tier",
    "max_tool_rounds", "max_tokens", "role", "version", "examples", "streams_output",
}
_REQUIRED = ("name", "description", "capabilities", "skills", "tools", "trust_tier", "role")

_NAME = re.compile(r"^[a-z][a-z0-9_-]{0,62}$")
_CAPABILITY = re.compile(r"^[a-z][a-z0-9]*(?:_[a-z0-9]+)*$")
_SKILL_NAME = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
# Example tasks (ADR-0139 C2: "3-5 example tasks"): optional; when set, 1.._MAX_EXAMPLES
# unique non-blank strings of at most _MAX_EXAMPLE_CHARS characters. The kernel
# publishes them on the agent's A2A card (cambrian-core internal/agentcard).
_MAX_EXAMPLES = 5
_MAX_EXAMPLE_CHARS = 300
_VERSION = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+$")
_GENERATOR_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]*$")
# mcp:<server id>/<tool glob>. The server id is literal: an allowlist names the servers
# it reaches, so a wildcard there (mcp:*/read_*) is refused — it would silently widen to
# every server a deployment adds later.
_MCP_GLOB = re.compile(r"^mcp:([a-z0-9][a-z0-9._-]*)/([A-Za-z0-9_.*?-]+)$")
# A kernel-native tool name (tools/ process tools): no namespace, optional glob chars.
_NATIVE_GLOB = re.compile(r"^[a-z][a-z0-9_.*?-]*$")

_MAX_TOOL_ROUNDS = (1, 64)
_MAX_TOKENS = (256, 131072)


class InvalidProfile(ValueError):
    """A profile file violates the profile format (agents/profiles/README.md)."""


# ── the allowlist ────────────────────────────────────────────────────────────


def _check_glob(pattern: Any) -> str:
    if not isinstance(pattern, str) or not pattern.strip():
        raise InvalidProfile(f"tool pattern {pattern!r} must be a non-empty string")
    if pattern != pattern.strip() or any(c.isspace() for c in pattern):
        raise InvalidProfile(f"tool pattern {pattern!r} contains whitespace")
    if "[" in pattern or "]" in pattern or "**" in pattern:
        raise InvalidProfile(
            f"tool pattern {pattern!r}: only '*' and '?' are supported (no [...] classes, no **)")
    if pattern.startswith("mcp:"):
        m = _MCP_GLOB.match(pattern)
        if not m:
            raise InvalidProfile(
                f"tool pattern {pattern!r} must be mcp:<server id>/<tool glob> with a literal "
                "server id (no wildcard before the '/')")
        return pattern  # mcp:<server>/* (every tool of one named server) is allowed
    if ":" in pattern or "/" in pattern:
        raise InvalidProfile(
            f"tool pattern {pattern!r}: unknown namespace (use mcp:<server>/<tool> or a native tool name)")
    if not _NATIVE_GLOB.match(pattern) or set(pattern) <= {"*", "?"}:
        raise InvalidProfile(
            f"tool pattern {pattern!r} is not a native tool name; a bare wildcard would allow every tool")
    return pattern


class ToolAllowlist:
    """Glob allowlist over kernel tool names (``fnmatch`` case-sensitive semantics,
    ``*`` and ``?`` only). An empty allowlist allows nothing."""

    def __init__(self, patterns: Iterable[str]):
        self.patterns: Tuple[str, ...] = tuple(_check_glob(p) for p in patterns)

    def allows(self, tool_name: str) -> bool:
        if not tool_name:
            return False
        return any(fnmatch.fnmatchcase(tool_name, p) for p in self.patterns)

    def filter(self, names: Iterable[str]) -> List[str]:
        return [n for n in names if self.allows(n)]

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"ToolAllowlist({list(self.patterns)!r})"


# ── skills on disk ───────────────────────────────────────────────────────────


@dataclass(frozen=True)
class SkillRef:
    """What a profile needs to know about a skill: its name and the tools it grants."""

    name: str
    tools: Tuple[str, ...]
    path: Path


def _skill_frontmatter(text: str) -> Dict[str, Any]:
    """Read a SKILL.md frontmatter. The shipped skills are in the strict subset; a
    customer skill written in fuller YAML falls back to a line scan for the two keys a
    profile needs (``name`` and the ``tools`` list), so it is not refused here — the
    kernel's YAML loader is the authority on the rest of the file."""
    front, _ = split_frontmatter(text)
    try:
        return parse_frontmatter(front)
    except InvalidPrompt:
        out: Dict[str, Any] = {}
        in_tools = False
        for line in front.split("\n"):
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            if line.startswith((" ", "\t")):
                item = line.strip()
                if in_tools and item.startswith("- "):
                    out.setdefault("tools", []).append(item[2:].strip().strip("'\""))
                continue
            in_tools = False
            key, _, value = line.partition(":")
            key, value = key.strip(), value.strip()
            if key == "name":
                out["name"] = value.strip("'\"")
            elif key == "tools":
                if value.startswith("[") and value.endswith("]"):
                    out["tools"] = [v.strip().strip("'\"") for v in value[1:-1].split(",") if v.strip()]
                else:
                    in_tools = True
                    out["tools"] = []
        return out


def skill_index(skills_dir: Union[str, Path]) -> Dict[str, SkillRef]:
    """Index ``<skills_dir>/<dir>/SKILL.md`` by the skill's frontmatter ``name`` — the
    identity the kernel registers it under (internal/skill/discovery)."""
    root = Path(skills_dir)
    out: Dict[str, SkillRef] = {}
    if not root.is_dir():
        return out
    for md in sorted(root.glob("*/SKILL.md")):
        try:
            fm = _skill_frontmatter(md.read_text(encoding="utf-8"))
        except (OSError, InvalidPrompt) as exc:
            logger.warning("skill_unreadable", extra={"path": str(md), "error": str(exc)})
            continue
        name = fm.get("name") or md.parent.name
        tools = tuple(t for t in (fm.get("tools") or []) if isinstance(t, str) and t)
        out[str(name)] = SkillRef(name=str(name), tools=tools, path=md)
    return out


# ── the profile ──────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Profile:
    """One validated profile. Construct with :func:`parse_profile` / :func:`load_profile`."""

    name: str
    description: str
    capabilities: Tuple[str, ...]
    skills: Tuple[str, ...]
    tools: Tuple[str, ...]
    trust_tier: str
    role: str
    model: str = ""
    max_tool_rounds: Optional[int] = None
    max_tokens: Optional[int] = None
    version: str = "1.0.0"
    examples: Tuple[str, ...] = ()
    #: ADR-0139 amendment 2026-11-06: the agent pushes its own live step output
    #: (``SubstrateClient.emit_step_output``), so the kernel forwards no gateway
    #: deltas for its steps — one step never streams twice.
    streams_output: bool = False
    path: Optional[Path] = None
    warnings: Tuple[str, ...] = field(default=(), compare=False)

    @property
    def allowlist(self) -> ToolAllowlist:
        return ToolAllowlist(self.tools)

    def manifest(self) -> Dict[str, Any]:
        """The AGENT_MANIFEST this profile stands for (the kernel's ManifestRecord shape,
        internal/storage/dto.go).

        The allowlist goes under ``tool_allowlist``, NOT the manifest's ``tools`` key: the
        kernel still reads ``tools`` as legacy capability tags (dispatch.go injects
        ``tools[0]`` as ``_capability``; profile_seeder falls back to it), so a glob there
        would surface as a capability. The kernel builds the same manifest from the
        profile file itself (internal/storage/profile_discovery.go); it reads
        ``tool_allowlist`` and ``skills`` (ManifestRecord) and keeps ``trust_tier`` and
        ``model`` on the agent definition."""
        m: Dict[str, Any] = {
            "version": self.version,
            "trait": "cognitive",
            "capabilities": list(self.capabilities),
            "supported_formats": ["text"],
            "tool_allowlist": list(self.tools),
            "skills": list(self.skills),
            "profile": self.name,
            "trust_tier": self.trust_tier,
            "release_notes": f"worker_agent profile {self.name} (agent stack v2).",
            "dependencies": [],
        }
        if self.model:
            m["model"] = self.model
        if self.examples:
            m["examples"] = list(self.examples)
        if self.streams_output:
            m["streams_output"] = True
        return m


def _str_field(data: Mapping[str, Any], key: str, where: str) -> str:
    v = data.get(key)
    if not isinstance(v, str) or not v.strip():
        raise InvalidProfile(f"{where}: {key!r} must be a non-empty string")
    return v.strip()


def _list_field(data: Mapping[str, Any], key: str, where: str) -> List[Any]:
    v = data.get(key)
    if not isinstance(v, list) or not v:
        raise InvalidProfile(f"{where}: {key!r} must be a non-empty list")
    if len(set(map(str, v))) != len(v):
        raise InvalidProfile(f"{where}: {key!r} has duplicate entries")
    return v


def _int_field(data: Mapping[str, Any], key: str, bounds: Tuple[int, int], where: str) -> Optional[int]:
    if key not in data or data[key] is None:
        return None
    v = data[key]
    if not isinstance(v, int) or isinstance(v, bool) or not bounds[0] <= v <= bounds[1]:
        raise InvalidProfile(f"{where}: {key!r} must be an integer in [{bounds[0]}, {bounds[1]}]")
    return v


def _bool_field(data: Mapping[str, Any], key: str, where: str) -> bool:
    """An optional boolean. The frontmatter subset has no boolean scalar (an unquoted
    ``true`` is the string "true"), so exactly "true" / "false"; absent, null or blank
    is False. The Go port is agentprofile.boolField."""
    v = data.get(key)
    if v is None or v == "":
        return False
    if isinstance(v, str) and v.strip() in ("true", "false"):
        return v.strip() == "true"
    raise InvalidProfile(f"{where}: {key!r} must be true or false (got {v!r})")


def parse_profile(
    text: str,
    *,
    skills_dir: Optional[Union[str, Path]] = None,
    expected_name: Optional[str] = None,
    where: str = "<profile>",
    path: Optional[Path] = None,
) -> Profile:
    """Parse and validate one profile's text.

    ``skills_dir`` is the directory the kernel serves skills from (``cambrian-core/skills``).
    When given, every listed skill must exist there and every tool a listed skill grants
    must fall inside the profile's allowlist. ``None`` skips those two checks (the runtime
    does this only when no skills directory is reachable, and logs it)."""
    try:
        data = parse_frontmatter(text.replace("\r\n", "\n"))
    except InvalidPrompt as exc:
        raise InvalidProfile(f"{where}: {exc} (profiles use the strict YAML subset of "
                             "prompts/SPEC.md §2.2)") from None

    unknown = sorted(set(data) - _FIELDS)
    if unknown:
        raise InvalidProfile(f"{where}: unknown field(s) {unknown}; allowed: {sorted(_FIELDS)}")
    missing = [k for k in _REQUIRED if data.get(k) in (None, "", [])]
    if missing:
        raise InvalidProfile(f"{where}: missing required field(s) {missing}")

    warnings: List[str] = []

    name = _str_field(data, "name", where)
    if not _NAME.match(name):
        raise InvalidProfile(f"{where}: name {name!r} must match {_NAME.pattern}")
    if expected_name is not None and name != expected_name:
        raise InvalidProfile(f"{where}: name {name!r} must equal the file name {expected_name!r}")

    description = _str_field(data, "description", where)
    # The kernel reads AGENT_DESCRIPTION with a regex that stops at a quote
    # (internal/storage/bbolt_adapter.go descRegex); keep the routing text quote-free so
    # the profile and any generated manifest carry the same string.
    if '"' in description or "'" in description:
        raise InvalidProfile(f"{where}: description must not contain quote characters")

    role = _str_field(data, "role", where)

    capabilities = []
    for c in _list_field(data, "capabilities", where):
        if not isinstance(c, str) or not _CAPABILITY.match(c):
            raise InvalidProfile(
                f"{where}: capability {c!r} must be lower_snake_case (the fleet's declared spelling)")
        if c in RESERVED_CAPABILITIES:
            raise InvalidProfile(f"{where}: capability {c!r} is reserved for the chat front desk")
        if c not in CAPABILITY_VOCABULARY:
            warnings.append(f"capability {c!r} is new to the fleet vocabulary; "
                            "it routes only when a planner step requires it")
        capabilities.append(c)

    skills = []
    for s in _list_field(data, "skills", where):
        if not isinstance(s, str) or not _SKILL_NAME.match(s):
            raise InvalidProfile(f"{where}: skill name {s!r} is malformed")
        skills.append(s)

    tools = [_check_glob(t) for t in _list_field(data, "tools", where)]
    allow = ToolAllowlist(tools)

    trust_tier = _str_field(data, "trust_tier", where)
    if trust_tier not in TRUST_TIERS:
        raise InvalidProfile(f"{where}: trust_tier {trust_tier!r} must be one of {list(TRUST_TIERS)}")

    model = ""
    if data.get("model") not in (None, ""):
        model = _str_field(data, "model", where)
        if model.startswith("role:"):
            if model[len("role:"):] not in MODEL_ROLES:
                raise InvalidProfile(f"{where}: model {model!r}: role must be one of {list(MODEL_ROLES)}")
        elif not _GENERATOR_ID.match(model):
            raise InvalidProfile(f"{where}: model {model!r} must be a generator id or role:<role>")

    version = "1.0.0"
    if data.get("version") not in (None, ""):
        version = _str_field(data, "version", where)
        if not _VERSION.match(version):
            raise InvalidProfile(f"{where}: version {version!r} must be MAJOR.MINOR.PATCH")

    examples: List[str] = []
    if data.get("examples") not in (None, "", []):
        raw_examples = _list_field(data, "examples", where)
        if len(raw_examples) > _MAX_EXAMPLES:
            raise InvalidProfile(f"{where}: 'examples' has {len(raw_examples)} entries; "
                                 f"at most {_MAX_EXAMPLES}")
        for e in raw_examples:
            if not isinstance(e, str) or not e.strip():
                raise InvalidProfile(f"{where}: example {e!r} must be a non-empty string")
            if len(e.strip()) > _MAX_EXAMPLE_CHARS:
                raise InvalidProfile(f"{where}: example {e.strip()!r} is longer than "
                                     f"{_MAX_EXAMPLE_CHARS} characters")
            examples.append(e.strip())

    streams_output = _bool_field(data, "streams_output", where)

    max_tool_rounds = _int_field(data, "max_tool_rounds", _MAX_TOOL_ROUNDS, where)
    max_tokens = _int_field(data, "max_tokens", _MAX_TOKENS, where)

    if skills_dir is not None:
        index = skill_index(skills_dir)
        absent = [s for s in skills if s not in index]
        if absent:
            raise InvalidProfile(f"{where}: skill(s) {absent} not found under {skills_dir}")
        for s in skills:
            outside = [t for t in index[s].tools if not allow.allows(t)]
            if outside:
                raise InvalidProfile(
                    f"{where}: skill {s!r} grants {outside}, which the profile's tools do not "
                    "allow; add them to tools or drop the skill (the kernel would clip them "
                    "and the skill's procedure would fail at that step)")

    return Profile(
        name=name, description=description, capabilities=tuple(capabilities),
        skills=tuple(skills), tools=tuple(tools), trust_tier=trust_tier, role=role,
        model=model, max_tool_rounds=max_tool_rounds, max_tokens=max_tokens,
        version=version, examples=tuple(examples), streams_output=streams_output,
        path=path, warnings=tuple(warnings),
    )


def load_profile(
    name_or_path: Union[str, Path],
    *,
    profiles_dir: Optional[Union[str, Path]] = None,
    skills_dir: Optional[Union[str, Path]] = None,
) -> Profile:
    """Load ``<profiles_dir>/<name>.yaml``, or a path to a profile file. The file's stem
    must equal the profile's ``name`` (one file, one agent id)."""
    p = Path(name_or_path)
    if p.suffix not in (".yaml", ".yml"):
        if profiles_dir is None:
            raise InvalidProfile(f"profile {name_or_path!r}: give a path or a profiles_dir")
        if not _NAME.match(str(name_or_path)):
            raise InvalidProfile(f"profile name {name_or_path!r} must match {_NAME.pattern}")
        p = Path(profiles_dir) / f"{name_or_path}.yaml"
    try:
        text = p.read_text(encoding="utf-8")
    except OSError as exc:
        raise InvalidProfile(f"profile {str(p)!r} unreadable: {exc}") from None
    return parse_profile(text, skills_dir=skills_dir, expected_name=p.stem, where=str(p), path=p)


def load_profiles(
    profiles_dir: Union[str, Path], *, skills_dir: Optional[Union[str, Path]] = None,
) -> Dict[str, Profile]:
    """Load every ``*.yaml`` profile in a directory. Any invalid file raises."""
    out: Dict[str, Profile] = {}
    for p in sorted(Path(profiles_dir).glob("*.yaml")):
        prof = load_profile(p, skills_dir=skills_dir)
        out[prof.name] = prof
    return out


# ── the client-side scope ────────────────────────────────────────────────────


class ProfileScopedSubstrate:
    """A substrate proxy that scopes an agent's kernel surface to its profile.

    - ``list_tools``: only allowlisted tools come back (menu, ``find_tools``,
      ``describe_tool`` and the skill-grant lookup all read through it).
    - ``list_skills``: the menu is the agent's EFFECTIVE skill set — the profile's
      skills plus the skills an operator granted this agent (decisions/
      skill-access-grants.md). The kernel enforces that set and marks every skill it
      serves with ``access``; this proxy keeps what the kernel marked, and against an
      older kernel that marks nothing it falls back to the profile's skills alone. A
      ``use_skill`` load of a skill outside the set is passed to the kernel, which asks
      an operator and answers granted (body), pending or denied (no body).
      ``discover=True`` (find_skills) also keeps skills the kernel marks
      ``requestable``. A loaded skill's ``tool_grants`` are clipped to the allowlist so
      the loop never offers a tool the profile does not allow.
    - ``execute_tool``: a call outside the allowlist is refused locally with a
      ``denied`` response the loop already understands.

    This runs in the agent's own process, so on its own it bounds a well-behaved agent,
    not a hostile one; the kernel enforces the same ``tools`` list as grants plus a
    ceiling, so a call this filter missed is still refused kernel-side.
    Every other attribute passes through to the wrapped substrate.
    """

    def __init__(self, inner: Any, profile: Profile):
        object.__setattr__(self, "_inner", inner)
        object.__setattr__(self, "_profile", profile)
        object.__setattr__(self, "_allow", profile.allowlist)

    @property
    def inner(self) -> Any:
        return self._inner

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def _keep(self, tools: Optional[Sequence[Dict]]) -> List[Dict]:
        kept, dropped = [], []
        for t in tools or []:
            (kept if self._allow.allows(t.get("name", "")) else dropped).append(t)
        if dropped:
            logger.debug("profile_tools_filtered",
                         extra={"profile": self._profile.name,
                                "dropped": [t.get("name") for t in dropped]})
        return kept

    def list_tools(self, *args: Any, **kwargs: Any) -> List[Dict]:
        if not hasattr(self._inner, "list_tools"):
            return []
        names = kwargs.get("names")
        if names:
            kwargs["names"] = self._allow.filter(names)
            if not kwargs["names"]:
                return []
        return self._keep(self._inner.list_tools(*args, **kwargs))

    def find_tools(self, query: str, k: int = 0, full: bool = False,
                   session_token_id: str = "") -> Tuple[List[Dict], List[str]]:
        """The find_tools pull, kept to the allowlist like :meth:`list_tools` (without
        this, ``__getattr__`` would hand the loop the unfiltered inner method). The
        ``outside`` names are passed through: names only, never callable."""
        inner_find = getattr(self._inner, "find_tools", None)
        if inner_find is None:
            return self.list_tools(query=query, k=k, full=full,
                                   session_token_id=session_token_id), []
        tools, outside = inner_find(query, k=k, full=full, session_token_id=session_token_id)
        return self._keep(tools), list(outside or [])

    #: Kernel access values (cambrian.proto SkillDescriptor.access). In the set:
    #: served with a body on use_skill. The rest carry no body and are passed through
    #: so the loop can tell the model why.
    _IN_SET = ("profile", "granted")
    _OUTSIDE = ("requestable", "pending", "denied")

    def list_skills(self, query: str = "", k: int = 0, names: Optional[List[str]] = None,
                    full: bool = False, session_token_id: str = "", rationale: str = "",
                    discover: bool = False) -> List[Dict]:
        if not hasattr(self._inner, "list_skills"):
            return []
        own = list(self._profile.skills)
        kwargs: Dict[str, Any] = {"full": full}
        if names:
            kwargs["names"] = list(names)
        elif discover and query:
            kwargs["query"] = query
            if k:
                kwargs["k"] = k
        # else: the menu — the kernel's full listing, which a bounding kernel limits to
        # the effective set (and an older one does not, so the filter below keeps own).
        if session_token_id:
            kwargs["session_token_id"] = session_token_id
        if rationale:
            kwargs["rationale"] = rationale
        specs = None
        while True:
            try:
                specs = self._inner.list_skills(**kwargs)
                break
            except TypeError:  # a substrate (or fake) without an optional kwarg
                for opt in ("rationale", "session_token_id"):
                    if opt in kwargs:
                        kwargs.pop(opt)
                        break
                else:
                    raise
        keep = self._IN_SET + (self._OUTSIDE if (names or discover) else ())
        out = []
        for s in specs or []:
            access = s.get("access") or ""
            if access:
                if access not in keep:
                    continue
            elif s.get("name") not in own:
                continue  # an unmarked skill (older kernel): the profile's list bounds it
            if s.get("tool_grants"):
                s = dict(s, tool_grants=self._allow.filter(s["tool_grants"]))
            out.append(s)
        if not names and not discover:
            # The menu is rendered into the system prompt: a deterministic order keeps
            # that prompt byte-stable across the steps of a run (provider caching,
            # replay; T-20261108-system-prompt-skill-list-order-is-not-stable).
            out.sort(key=lambda s: str(s.get("name") or ""))
            served = {s.get("name") for s in out}
            unserved = [n for n in own if n not in served]
            if unserved:
                logger.warning("profile_skills_not_served",
                               extra={"profile": self._profile.name, "skills": unserved})
        return out

    def execute_tool(self, name: str, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        if not self._allow.allows(name):
            return {"denied": True,
                    "deny_reason": f"tool {name!r} is outside profile {self._profile.name!r}'s "
                                   "tool allowlist (client-side check; the kernel enforces the same allowlist)"}
        return self._inner.execute_tool(name, *args, **kwargs)


def manifest_json(profile: Profile) -> str:
    """The profile's manifest as the JSON text an AGENT_MANIFEST literal carries."""
    return json.dumps(profile.manifest(), indent=2)
