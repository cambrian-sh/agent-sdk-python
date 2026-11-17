"""The SDK prompt library (ADR-0139 C4 / D2, owner decision Q6).

Every model-facing prompt the SDK sends lives as a versioned file in the package
data directory ``cambrian_agent_sdk/prompts/<id>/v<N>.md``. The format is
specified once, in ``cambrian-core/prompts/SPEC.md``; this module implements the
same rules as the kernel's Go loader (``github.com/cambrian-sh/core/prompts``)
and is held to it by ``tests/test_prompt_hash_parity.py``, which reads a fixture
the Go tests write.

Third-party agents never depend on core files: the SDK prompts ship inside the
wheel and are read with ``importlib.resources``.

The frontmatter is a strict YAML subset (SPEC.md §2.2) parsed here without a YAML
dependency. The static hash is sha256 over the body with every slot replaced by
``{{}}`` -- the static text actually sent, identical across renders.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple, Union

__all__ = [
    "PLACEHOLDER",
    "TIERS",
    "Prompt",
    "Rendered",
    "PromptLibrary",
    "PromptError",
    "PromptNotFound",
    "SlotMismatch",
    "InvalidPrompt",
    "default_library",
    "Fragment",
    "fragment",
    "sections",
    "get",
    "get_version",
    "normalise_body",
    "static_hash",
    "slots",
    "split_frontmatter",
]

PLACEHOLDER = "{{}}"
TIERS = ("stable", "context", "volatile")

_ID = re.compile(r"^[a-z][a-z0-9_-]*$")
_VERSION_FILE = re.compile(r"^v([1-9][0-9]*)\.md$")
_SLOT = re.compile(r"\{\{([a-z][a-z0-9_]*)\}\}")
_SLOT_NAME = re.compile(r"^[a-z][a-z0-9_]*$")
_INT = re.compile(r"^-?[0-9]+$")
_KNOWN = {"id", "version", "tier", "description", "inputs", "response_schema", "model_notes"}
_REQUIRED = ("id", "version", "tier", "description", "inputs")
_BOM = chr(0xFEFF)


class PromptError(Exception):
    """Base class for prompt-library errors."""


class PromptNotFound(PromptError, LookupError):
    """No prompt with that id, or no such version."""


class SlotMismatch(PromptError, ValueError):
    """``render`` was given a missing or an extra slot value."""


class InvalidPrompt(PromptError, ValueError):
    """A prompt file violates SPEC.md."""


def normalise_body(body: str) -> str:
    """SPEC.md §3: CRLF to LF, leading newlines and trailing space/tab/newline removed."""
    body = body.replace("\r\n", "\n").lstrip("\n")
    return body.rstrip(" \t\r\n")


def static_hash(body: str) -> str:
    """SPEC.md §4: sha256 hex of the normalised body with every slot as ``{{}}``."""
    static = _SLOT.sub(lambda _m: PLACEHOLDER, normalise_body(body))
    return hashlib.sha256(static.encode("utf-8")).hexdigest()


def slots(body: str) -> List[str]:
    """The distinct slot names in a body, in first-use order."""
    out: List[str] = []
    for name in _SLOT.findall(body):
        if name not in out:
            out.append(name)
    return out


@dataclass(frozen=True)
class Rendered:
    text: str
    static_hash: str
    id: str
    version: int
    response_schema: Optional[bytes] = None


@dataclass(frozen=True)
class Prompt:
    id: str
    version: int
    tier: str
    inputs: Tuple[str, ...]
    description: str
    body: str
    model_notes: str = ""
    response_schema: Optional[bytes] = field(default=None, repr=False)

    @property
    def static_hash(self) -> str:
        return static_hash(self.body)

    def render(self, values: Mapping[str, str]) -> Rendered:
        """Fill the slots. Every input needs a value (``""`` counts); no value may
        name an undeclared slot. Single pass: a value containing ``{{x}}`` is
        inserted literally."""
        missing = [i for i in self.inputs if i not in values]
        extra = sorted(k for k in values if k not in self.inputs)
        if missing or extra:
            raise SlotMismatch(
                f"{self.id} v{self.version}: missing {missing}, extra {extra}")
        text = _SLOT.sub(lambda m: str(values[m.group(1)]), self.body)
        return Rendered(text=text, static_hash=self.static_hash, id=self.id,
                        version=self.version, response_schema=self.response_schema)


# ── frontmatter ──────────────────────────────────────────────────────────────


def split_frontmatter(text: str) -> Tuple[str, str]:
    """Separate a version file into (frontmatter, body) after normalising line
    endings -- the same split as the Go loader's ``SplitFrontmatter``."""
    text = text.replace("\r\n", "\n")
    if text.startswith(_BOM):
        text = text[1:]
    if not text.startswith("---\n"):
        raise InvalidPrompt("file must start with a --- frontmatter line")
    rest = text[len("---\n"):]
    if rest.startswith("---\n"):
        return "", rest[len("---\n"):]
    end = rest.find("\n---\n")
    if end >= 0:
        return rest[:end + 1], rest[end + len("\n---\n"):]
    if rest.endswith("\n---"):
        return rest[:-len("---")], ""
    raise InvalidPrompt("frontmatter has no closing --- line")


def _scalar(raw: str) -> Any:
    raw = raw.strip()
    if raw.startswith('"'):
        try:
            return json.loads(raw)
        except ValueError as e:
            raise InvalidPrompt(f"bad double-quoted scalar {raw!r}: {e}") from None
    if raw.startswith("'"):
        if len(raw) < 2 or not raw.endswith("'"):
            raise InvalidPrompt(f"bad single-quoted scalar {raw!r}")
        return raw[1:-1].replace("''", "'")
    if _INT.match(raw):
        return int(raw)
    if raw in ("", "~", "null"):
        return None
    if raw[0] in "&*!|>%@`" or ": " in raw or " #" in raw:
        raise InvalidPrompt(f"scalar {raw!r} is outside the frontmatter subset (quote it)")
    return raw


def _flow_list(raw: str) -> List[Any]:
    inner = raw.strip()[1:-1].strip()
    if not inner:
        return []
    return [_scalar(part) for part in inner.split(",")]


def parse_frontmatter(front: str) -> Dict[str, Any]:
    """Parse the SPEC.md §2.2 YAML subset."""
    out: Dict[str, Any] = {}
    block_key: Optional[str] = None
    with_items = set()
    block_keys = set()
    for n, line in enumerate(front.split("\n"), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if line.startswith((" ", "\t")):
            item = line.strip()
            if block_key is None or not item.startswith("- "):
                raise InvalidPrompt(f"frontmatter line {n}: unexpected indentation")
            out[block_key].append(_scalar(item[2:]))
            with_items.add(block_key)
            continue
        key, sep, value = line.partition(":")
        key = key.strip()
        if not sep or not key:
            raise InvalidPrompt(f"frontmatter line {n}: expected 'key: value'")
        if key in out:
            raise InvalidPrompt(f"frontmatter key {key!r} appears twice")
        value = value.strip()
        block_key = None
        if value == "":
            out[key] = []
            block_key = key
            block_keys.add(key)
        elif value.startswith("["):
            if not value.endswith("]"):
                raise InvalidPrompt(f"frontmatter line {n}: unterminated flow list")
            if value[1:].strip().startswith("{"):
                # An inline list of objects is one-line JSON, as an inline object is
                # (a profile's sandbox_paths, ADR-0157 D1.2). JSON is YAML flow syntax,
                # so a full YAML parser reads it the same way. Go port:
                # cambrian-core internal/agentprofile parseSubset.
                try:
                    out[key] = json.loads(value)
                except ValueError as e:
                    raise InvalidPrompt(
                        f"frontmatter line {n}: inline list of objects is not JSON: {e}") from None
                continue
            out[key] = _flow_list(value)
        elif value.startswith("{"):
            try:
                out[key] = json.loads(value)
            except ValueError as e:
                raise InvalidPrompt(f"frontmatter line {n}: inline object is not JSON: {e}") from None
        else:
            out[key] = _scalar(value)
    # A bare "key:" with no block items is null, as YAML reads it (so a bare
    # "inputs:" is a missing field, exactly as in the Go loader).
    for k in block_keys - with_items:
        out[k] = None
    return out


# ── loading ──────────────────────────────────────────────────────────────────

_Root = Union[Path, Any]  # a pathlib.Path or an importlib.resources Traversable


def _parse_file(root: _Root, prompt_id: str, name: str) -> Prompt:
    where = f"{prompt_id}/{name}"

    def fail(msg: str) -> InvalidPrompt:
        return InvalidPrompt(f"{where}: {msg}")

    data = (root / prompt_id / name).read_bytes().decode("utf-8")
    try:
        front, body = split_frontmatter(data)
        fm = parse_frontmatter(front)
    except InvalidPrompt as e:
        raise fail(str(e)) from None
    unknown = sorted(set(fm) - _KNOWN)
    if unknown:
        raise fail(f"unknown frontmatter field(s) {unknown}")
    for key in _REQUIRED:
        if fm.get(key) is None:
            raise fail(f"frontmatter field {key!r} is required")
    if fm["id"] != prompt_id:
        raise fail(f"id {fm['id']!r} does not match its directory {prompt_id!r}")
    version_match = _VERSION_FILE.match(name)
    if version_match is None:
        raise fail("file name is not v<N>.md")
    file_version = int(version_match.group(1))
    if not isinstance(fm["version"], int) or fm["version"] != file_version:
        raise fail(f"version {fm['version']!r} does not match the file name")
    if fm["tier"] not in TIERS:
        raise fail(f"tier {fm['tier']!r} is not stable, context or volatile")
    if not isinstance(fm["description"], str) or not fm["description"].strip():
        raise fail("description is empty")
    inputs = fm["inputs"]
    if not isinstance(inputs, list) or not all(isinstance(i, str) for i in inputs):
        raise fail("inputs must be a list of slot names")
    body = normalise_body(body)
    try:
        _check_slots(inputs, body)
        schema = _load_schema(root, prompt_id, fm.get("response_schema"))
    except InvalidPrompt as e:
        raise fail(str(e)) from None
    notes = fm.get("model_notes")
    return Prompt(id=prompt_id, version=file_version, tier=fm["tier"], inputs=tuple(inputs),
                  description=fm["description"].strip(), body=body,
                  model_notes=str(notes).strip() if notes is not None else "",
                  response_schema=schema)


def _check_slots(inputs: List[str], body: str) -> None:
    seen = set()
    for i in inputs:
        if not _SLOT_NAME.match(i):
            raise InvalidPrompt(f"input {i!r} is not a valid slot name")
        if i in seen:
            raise InvalidPrompt(f"input {i!r} is listed twice")
        seen.add(i)
    used = set(slots(body))
    for s in used - seen:
        raise InvalidPrompt(f"body uses slot {{{{{s}}}}} that inputs does not declare")
    for i in inputs:
        if i not in used:
            raise InvalidPrompt(f"input {i!r} is declared but the body never uses it")
    if "{{" in _SLOT.sub("", body):
        raise InvalidPrompt('body has a malformed "{{" (slots are {{name}} with name [a-z][a-z0-9_]*)')


def _load_schema(root: _Root, prompt_id: str, value: Any) -> Optional[bytes]:
    if value is None or value == "":
        return None
    if isinstance(value, dict):
        return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    if not isinstance(value, str):
        raise InvalidPrompt("response_schema must be a file name or an inline JSON object")
    if "/" in value or "\\" in value or not value.endswith(".schema.json"):
        raise InvalidPrompt(f"response_schema {value!r} must be a sibling file named *.schema.json")
    f = root / prompt_id / value
    if not f.is_file():
        raise InvalidPrompt(f"response_schema {value!r} does not exist")
    data = f.read_bytes().replace(b"\r\n", b"\n")
    try:
        obj = json.loads(data)
    except ValueError as e:
        raise InvalidPrompt(f"response_schema {value!r} is not JSON: {e}") from None
    if not isinstance(obj, dict):
        raise InvalidPrompt(f"response_schema {value!r} is not a JSON object")
    return data


class PromptLibrary:
    """A loaded prompt tree. ``PromptLibrary.load(root)`` validates every file
    (SPEC.md L1-L5) and fails on the first violation."""

    def __init__(self, by_id: Dict[str, List[Prompt]]):
        self._by_id = by_id

    @classmethod
    def load(cls, root: _Root) -> "PromptLibrary":
        by_id: Dict[str, List[Prompt]] = {}
        for d in sorted(root.iterdir(), key=lambda p: p.name):
            if not d.is_dir() or d.name in ("testdata", "__pycache__"):
                continue
            versions = [_parse_file(root, d.name, f.name)
                        for f in d.iterdir() if f.is_file() and _VERSION_FILE.match(f.name)]
            if not versions:
                continue
            if not _ID.match(d.name):
                raise InvalidPrompt(f"{d.name}: id must match {_ID.pattern}")
            versions.sort(key=lambda p: p.version)
            for i, p in enumerate(versions, 1):
                if p.version != i:
                    raise InvalidPrompt(
                        f"{d.name}: versions must be contiguous from 1, found v{p.version} at position {i}")
            by_id[d.name] = versions
        return cls(by_id)

    def ids(self) -> List[str]:
        return sorted(self._by_id)

    def versions(self, prompt_id: str) -> List[Prompt]:
        return list(self._by_id.get(prompt_id, ()))

    def get(self, prompt_id: str) -> Prompt:
        vs = self._by_id.get(prompt_id)
        if not vs:
            raise PromptNotFound(f"prompt {prompt_id!r} not found")
        return vs[-1]

    def get_version(self, prompt_id: str, n: int) -> Prompt:
        vs = self._by_id.get(prompt_id, [])
        if n < 1 or n > len(vs):
            raise PromptNotFound(f"prompt {prompt_id!r} v{n} not found")
        return vs[n - 1]

    def __iter__(self) -> Iterable[Prompt]:
        for pid in self.ids():
            yield from self._by_id[pid]


# ── fragment sets ────────────────────────────────────────────────────────────
#
# Some prompts are not sent whole: the ReAct loop assembles its system prompt from
# pieces chosen by protocol (text or native tool-calling), budget and menu. Such a
# prompt file is a FRAGMENT SET: its body is a sequence of named sections, each
# introduced by a line of exactly
#
#     <!-- section: <name> -->
#
# The file is still one prompt under SPEC.md: one id, one version, one inputs list
# (the union of every section's slots), one static hash over the whole body, and
# the same lint. A section is rendered on its own: only the slots it uses are
# required, and the delimiter line is never sent. Its text is the lines between
# its delimiter and the next, with leading and trailing blank lines dropped.

_SECTION_LINE = re.compile(r"^<!-- section: ([a-z][a-z0-9_-]*) -->$")


@dataclass(frozen=True)
class Fragment:
    """One named section of a fragment-set prompt."""

    prompt_id: str
    version: int
    name: str
    body: str
    file_hash: str

    @property
    def inputs(self) -> List[str]:
        return slots(self.body)

    @property
    def static_hash(self) -> str:
        return static_hash(self.body)

    def render(self, values: Optional[Mapping[str, Any]] = None) -> str:
        """Fill this section's slots (exactly its own; missing or extra is an error)."""
        values = dict(values or {})
        want = self.inputs
        missing = [i for i in want if i not in values]
        extra = sorted(k for k in values if k not in want)
        if missing or extra:
            raise SlotMismatch(
                f"{self.prompt_id} v{self.version} section {self.name!r}: "
                f"missing {missing}, extra {extra}")
        return _SLOT.sub(lambda m: str(values[m.group(1)]), self.body)


def sections(prompt: Prompt) -> Dict[str, Fragment]:
    """The named sections of a fragment-set prompt, in file order."""
    out: Dict[str, Fragment] = {}
    name: Optional[str] = None
    buf: List[str] = []

    def flush() -> None:
        if name is not None:
            out[name] = Fragment(prompt.id, prompt.version, name,
                                 "\n".join(buf).strip("\n"), prompt.static_hash)

    for line in prompt.body.split("\n"):
        m = _SECTION_LINE.match(line)
        if m:
            flush()
            name = m.group(1)
            if name in out:
                raise InvalidPrompt(f"{prompt.id} v{prompt.version}: section {name!r} appears twice")
            buf = []
            continue
        if name is None:
            if line.strip():
                raise InvalidPrompt(
                    f"{prompt.id} v{prompt.version}: text before the first section delimiter")
            continue
        buf.append(line)
    flush()
    if not out:
        raise InvalidPrompt(f"{prompt.id} v{prompt.version}: not a fragment set (no sections)")
    return out


_FRAGMENTS: Dict[Tuple[str, int], Dict[str, Fragment]] = {}


def fragment(prompt_id: str, name: str, version: Optional[int] = None) -> Fragment:
    """One section of a packaged fragment-set prompt (latest version by default)."""
    p = get(prompt_id) if version is None else get_version(prompt_id, version)
    key = (p.id, p.version)
    secs = _FRAGMENTS.get(key)
    if secs is None:
        secs = _FRAGMENTS[key] = sections(p)
    try:
        return secs[name]
    except KeyError:
        raise PromptNotFound(f"prompt {prompt_id!r} v{p.version} has no section {name!r}") from None


_DEFAULT: Optional[PromptLibrary] = None


def package_root():
    """The packaged prompt tree (``cambrian_agent_sdk/prompts``)."""
    return resources.files("cambrian_agent_sdk") / "prompts"


def default_library() -> PromptLibrary:
    """The library over the SDK's packaged prompts, loaded once."""
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = PromptLibrary.load(package_root())
    return _DEFAULT


def get(prompt_id: str) -> Prompt:
    """The latest version of a packaged SDK prompt."""
    return default_library().get(prompt_id)


def get_version(prompt_id: str, n: int) -> Prompt:
    """One version of a packaged SDK prompt."""
    return default_library().get_version(prompt_id, n)
