"""The structured managed-generation request and result (ADR-0139 C1, wave 1d).

``SubstrateClient.generate_request`` / ``generate_with_tools`` speak the C1 shape
the kernel's agent plane accepts on ``GenerateViaModelStream`` and
``GenerateWithTools``:

- ``system``: the system prompt as blocks, stable tier first. A block with
  ``cache=True`` marks the END of a prefix the provider may cache. System text has
  one way in; a message with role ``"system"`` is refused (``InvalidRequest``).
- ``messages``: the conversation, ``"user" | "assistant" | "tool"`` turns.
- ``response_schema``: ask for a JSON reply conforming to a JSON Schema.
- ``cache_ttl``: lifetime of the cache breakpoints (``"short"`` = 5 minutes, for
  machine-paced agent runs; ``"long"`` = 1 hour, for human-paced sessions).
- options with field presence: ``max_tokens``, ``temperature`` (``None`` = the
  provider's default, ``0`` is a real temperature), ``stop`` and ``strict``.

The result reports what the kernel says actually happened: ``usage`` (with the
cache breakdown), ``ignored`` (request fields the model did not honour, so a
degraded call is visible, never silent), ``stop_reason`` and ``model``.

This module holds plain data types and the request-shaping helpers. It has no
gRPC import; ``SubstrateClient`` converts to and from the vendored protobufs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Union

from .errors import InvalidRequest

__all__ = [
    "CACHE_TTL_SHORT",
    "CACHE_TTL_LONG",
    "CACHE_TTL_DEFAULT",
    "SystemBlock",
    "ResponseSchema",
    "GenerateUsage",
    "GenerateResult",
    "coerce_system",
    "coerce_messages",
    "coerce_response_schema",
    "coerce_cache_ttl",
    "flatten_prompt",
]

#: Cache-breakpoint lifetimes (cambrian.proto ``CacheTTL``). ``""`` leaves it to
#: the provider's default (Anthropic: 5 minutes).
CACHE_TTL_DEFAULT = ""
CACHE_TTL_SHORT = "short"
CACHE_TTL_LONG = "long"

_TTL_ALIASES = {
    None: CACHE_TTL_DEFAULT, "": CACHE_TTL_DEFAULT, "default": CACHE_TTL_DEFAULT,
    "unspecified": CACHE_TTL_DEFAULT,
    "short": CACHE_TTL_SHORT, "5m": CACHE_TTL_SHORT,
    "long": CACHE_TTL_LONG, "1h": CACHE_TTL_LONG,
}

_ROLES = ("user", "assistant", "tool")


@dataclass(frozen=True)
class SystemBlock:
    """One block of the system prompt. ``cache=True`` ends a cacheable prefix."""

    text: str
    cache: bool = False


@dataclass(frozen=True)
class ResponseSchema:
    """A JSON Schema the reply must satisfy. ``schema`` is a dict or a JSON string.
    ``strict`` demands constrained decoding: a model that can only best-effort it
    answers ``InvalidRequest`` instead of a validated-and-repaired reply."""

    name: str
    schema: Union[Dict[str, Any], str]
    strict: bool = False

    @property
    def schema_json(self) -> str:
        if isinstance(self.schema, str):
            return self.schema
        return json.dumps(self.schema, sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True)
class GenerateUsage:
    """Provider-reported usage for one call. ``input_tokens`` counts every input
    token, cached or not; the cache fields are breakdowns of it, never additions.

    ``cache_write_known`` is false when the provider does not report cache writes
    (then ``cache_write_tokens == 0`` means "unknown", not "none"). ``estimated`` is
    true when the numbers are the kernel's length/4 estimate."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    cache_write_known: bool = False
    estimated: bool = False

    # Short aliases used in logs and by callers that think in cache terms.
    @property
    def cache_read(self) -> int:
        return self.cache_read_tokens

    @property
    def cache_write(self) -> int:
        return self.cache_write_tokens

    @property
    def known(self) -> bool:
        """True when the cache-write figure is a provider report."""
        return self.cache_write_known

    def to_dict(self) -> Dict[str, Any]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_read_tokens": self.cache_read_tokens,
            "cache_write_tokens": self.cache_write_tokens,
            "cache_write_known": self.cache_write_known,
            "estimated": self.estimated,
        }


@dataclass
class GenerateResult:
    """The outcome of one managed generation call.

    Backward compatible with the old ``generate_with_tools`` return value: it
    unpacks as ``text, tool_calls, stop_reason = result``. It also reads like a dict
    (``result["usage"]``) for callers that prefer one.

    ``legacy_kernel`` is true when the kernel predates the structured fields (its
    reply carried no usage block). On ``GenerateViaModelStream`` the SDK also sent
    the flattened prompt, so nothing was lost; on ``GenerateWithTools`` the system
    blocks were dropped and ``"system"`` is added to ``ignored``.
    """

    text: str = ""
    tool_calls: List[Dict[str, str]] = field(default_factory=list)
    stop_reason: str = ""
    usage: Optional[GenerateUsage] = None
    ignored: List[str] = field(default_factory=list)
    model: str = ""
    usage_tokens: int = 0
    legacy_kernel: bool = False

    def __iter__(self) -> Iterator[Any]:
        # The pre-C1 3-tuple (text, tool_calls, stop_reason).
        return iter((self.text, self.tool_calls, self.stop_reason))

    def __len__(self) -> int:
        return 3

    def __getitem__(self, key):
        if isinstance(key, int):
            return (self.text, self.tool_calls, self.stop_reason)[key]
        if key in self.__dataclass_fields__:
            return getattr(self, key)
        raise KeyError(key)

    def honoured(self, field_name: str) -> bool:
        """False when the kernel reported it did not honour ``field_name``."""
        return field_name not in self.ignored

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "tool_calls": list(self.tool_calls),
            "stop_reason": self.stop_reason,
            "usage": self.usage.to_dict() if self.usage else None,
            "ignored": list(self.ignored),
            "model": self.model,
            "usage_tokens": self.usage_tokens,
            "legacy_kernel": self.legacy_kernel,
        }


SystemLike = Union[None, str, SystemBlock, Mapping[str, Any], Sequence[Any]]


def coerce_system(system: SystemLike) -> List[SystemBlock]:
    """Normalise the accepted ``system`` shapes to a list of ``SystemBlock``.

    Accepts ``None``, a string (one uncached block), a ``SystemBlock``, a
    ``{"text", "cache"}`` mapping, or a list of any of those."""
    if system is None:
        return []
    if isinstance(system, (str, SystemBlock, Mapping)):
        system = [system]
    out: List[SystemBlock] = []
    for b in system:
        if isinstance(b, SystemBlock):
            blk = b
        elif isinstance(b, str):
            blk = SystemBlock(text=b)
        elif isinstance(b, Mapping):
            blk = SystemBlock(text=str(b.get("text", "") or ""), cache=bool(b.get("cache", False)))
        else:
            raise InvalidRequest(f"system block must be a str, dict or SystemBlock, got {type(b).__name__}",
                                 code="INVALID_ARGUMENT")
        if blk.text:
            out.append(blk)
    return out


def coerce_messages(messages: Optional[Iterable[Mapping[str, Any]]]) -> List[Dict[str, Any]]:
    """Validate a conversation: roles are user/assistant/tool. A ``"system"`` turn
    is refused here, as the kernel would refuse it: system text goes in ``system``."""
    out: List[Dict[str, Any]] = []
    for m in messages or []:
        role = (m.get("role") or "user")
        if role == "system":
            raise InvalidRequest(
                'a message with role "system" is not allowed: pass system text as system=[...]',
                code="INVALID_ARGUMENT")
        if role not in _ROLES:
            raise InvalidRequest(f"unknown message role {role!r}", code="INVALID_ARGUMENT")
        out.append(dict(m))
    return out


def coerce_response_schema(schema: Any) -> Optional[ResponseSchema]:
    """Accept ``None``, a ``ResponseSchema`` or a ``{"name", "schema", "strict"}``
    mapping (``schema`` may also be spelled ``schema_json``)."""
    if schema is None:
        return None
    if isinstance(schema, ResponseSchema):
        return schema
    if isinstance(schema, Mapping):
        body = schema.get("schema", schema.get("schema_json"))
        if body is None:
            raise InvalidRequest("response_schema needs a 'schema'", code="INVALID_ARGUMENT")
        if isinstance(body, (bytes, bytearray)):
            body = bytes(body).decode("utf-8")
        return ResponseSchema(name=str(schema.get("name", "") or "response"),
                              schema=body, strict=bool(schema.get("strict", False)))
    raise InvalidRequest("response_schema must be a dict or ResponseSchema", code="INVALID_ARGUMENT")


def coerce_cache_ttl(ttl: Any) -> str:
    """``None``/``""``/``"short"``/``"long"`` (also ``"5m"``/``"1h"``) to a canonical value."""
    key = ttl.strip().lower() if isinstance(ttl, str) else ttl
    if key not in _TTL_ALIASES:
        raise InvalidRequest(f"cache_ttl {ttl!r} is not one of short, long or unset",
                             code="INVALID_ARGUMENT")
    return _TTL_ALIASES[key]


def flatten_prompt(system: Sequence[SystemBlock], messages: Sequence[Mapping[str, Any]]) -> str:
    """One prompt string for a kernel that predates ``system`` and ``messages``.

    System text first, then each turn. A lone user turn is rendered bare (the
    common case: one assembled prompt); other turns are tagged with their role so
    a multi-turn history stays legible as one string."""
    parts = [b.text for b in system if b.text]
    for m in messages:
        content = m.get("content") or ""
        role = m.get("role") or "user"
        if role == "user":
            parts.append(content)
        else:
            parts.append(f"<{role}>\n{content}\n</{role}>")
    return "\n\n".join(p for p in parts if p)
