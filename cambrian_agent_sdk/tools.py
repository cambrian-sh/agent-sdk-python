"""The ``@tool`` intra-agent registry (ADR-0036 D4).

A *closed menu* of local Python functions the agent's own LLM may call. This is
**not** ``@capability`` (inter-agent auction routing) — it answers "which of my
functions does my reasoning loop call", a different question with different security
properties:

- the schema is **auto-derived** from type hints (or supplied explicitly);
- inputs are **validated** against the schema *before* the call;
- invocation is a **direct bound-method call** — never ``exec`` / ``eval``;
- a bad call returns a structured ``{"error": ..., "tool": ...}`` dict (graceful
  degradation the LLM can read), not an exception.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, get_args, get_origin, get_type_hints
import typing


@dataclass
class ToolSpec:
    """The registered shape of one ``@tool`` method."""

    name: str
    schema: Dict[str, Any]
    description: str = ""


# Marker attributes — kept distinct from the @capability marker so the two
# registries never cross-contaminate (D4: independent registries).
_TOOL_ATTR = "_cambrian_tool"
_CAPABILITY_ATTR = "_cambrian_capability"


def tool(fn: Optional[Callable] = None, *, name: Optional[str] = None, schema: Optional[Dict] = None):
    """Mark a method as an intra-agent tool.

    Usage::

        @tool
        def search(self, query: str, limit: int = 10) -> str: ...

        @tool(name="web_search", schema={...})
        def search(self, query): ...

    When ``schema`` is omitted it is auto-derived from the function's type hints.
    """

    def wrap(f: Callable) -> Callable:
        tool_name = name or f.__name__
        tool_schema = schema if schema is not None else derive_schema(f)
        setattr(f, _TOOL_ATTR, ToolSpec(name=tool_name, schema=tool_schema, description=(f.__doc__ or "").strip()))
        return f

    return wrap(fn) if fn is not None else wrap


def capability(fn: Optional[Callable] = None, *, name: Optional[str] = None):
    """Mark a method as an inter-agent capability (auction routing).

    Deliberately a *separate* marker from :func:`tool` — the two are independent
    registries (D4). Present here so the distinction is structural, not documentary.
    """

    def wrap(f: Callable) -> Callable:
        setattr(f, _CAPABILITY_ATTR, name or f.__name__)
        return f

    return wrap(fn) if fn is not None else wrap


_PY_TO_JSON = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    dict: "object",
    list: "array",
}


def _json_type(hint: Any) -> Optional[str]:
    """Map a Python type hint to a JSON-Schema primitive type (None if unknown)."""
    origin = get_origin(hint)
    if origin is typing.Union:  # Optional[X] == Union[X, None]
        non_none = [a for a in get_args(hint) if a is not type(None)]
        if len(non_none) == 1:
            return _json_type(non_none[0])
        return None
    return _PY_TO_JSON.get(hint)


def derive_schema(fn: Callable) -> Dict[str, Any]:
    """Build a JSON-Schema object from ``fn``'s signature + type hints.

    Parameters with no default are ``required``; ``Optional[...]`` unwraps to the
    inner type. ``self`` is skipped. Unannotated params get an empty (any) schema.
    """
    sig = inspect.signature(fn)
    try:
        hints = get_type_hints(fn)
    except Exception:
        hints = {}

    properties: Dict[str, Any] = {}
    required: List[str] = []
    for pname, param in sig.parameters.items():
        if pname == "self" or param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            continue
        hint = hints.get(pname)
        jtype = _json_type(hint) if hint is not None else None
        properties[pname] = {"type": jtype} if jtype else {}
        is_optional = get_origin(hint) is typing.Union and type(None) in get_args(hint)
        if param.default is inspect.Parameter.empty and not is_optional:
            required.append(pname)

    return {"type": "object", "properties": properties, "required": required}


def validate_args(schema: Dict[str, Any], args: Dict[str, Any]) -> Optional[str]:
    """Validate ``args`` against ``schema``. Return an error string, or None if valid.

    A minimal validator (no external ``jsonschema`` dependency): checks required
    keys are present and that provided values match their declared JSON type.
    """
    for req in schema.get("required", []):
        if req not in args:
            return f"missing required argument '{req}'"
    props = schema.get("properties", {})
    for key, value in args.items():
        if key not in props:
            return f"unexpected argument '{key}'"
        expected = props[key].get("type")
        if expected and not _matches(expected, value):
            return f"argument '{key}' must be {expected}, got {type(value).__name__}"
    return None


def _matches(json_type: str, value: Any) -> bool:
    if json_type == "string":
        return isinstance(value, str)
    if json_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if json_type == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if json_type == "boolean":
        return isinstance(value, bool)
    if json_type == "object":
        return isinstance(value, dict)
    if json_type == "array":
        return isinstance(value, (list, tuple))
    return True


class ToolRegistry:
    """The bound ``@tool`` menu for one agent instance.

    Discovers ``@tool``-marked methods on the agent, binds them, and validates +
    invokes by direct call. Returns a structured ``{"error", "tool"}`` dict on any
    failure (unknown tool, schema violation, or in-tool exception) — never raises
    into the reasoning loop.
    """

    def __init__(self, agent: Any) -> None:
        self._specs: Dict[str, ToolSpec] = {}
        self._bound: Dict[str, Callable] = {}
        # Scan the class MRO (raw functions carry the @tool marker). We deliberately
        # avoid inspect.getmembers(agent), which would evaluate instance properties
        # (e.g. ``.tools`` itself) and recurse infinitely.
        seen: set = set()
        for klass in type(agent).__mro__:
            for attr_name, member in vars(klass).items():
                if attr_name in seen:
                    continue
                spec = getattr(member, _TOOL_ATTR, None)
                if spec is not None:
                    seen.add(attr_name)
                    self._specs[spec.name] = spec
                    self._bound[spec.name] = getattr(agent, attr_name)  # bound to the instance

    def names(self) -> List[str]:
        return sorted(self._specs)

    def __contains__(self, name: str) -> bool:
        return name in self._specs

    def schema(self, name: str) -> Dict[str, Any]:
        return self._specs[name].schema

    def specs(self) -> List[ToolSpec]:
        return [self._specs[n] for n in self.names()]

    def call(self, name: str, **kwargs) -> Any:
        """Validate + invoke a tool by name. Structured error dict on any failure."""
        if name not in self._specs:
            return {"error": f"unknown tool '{name}'", "tool": name}
        err = validate_args(self._specs[name].schema, kwargs)
        if err is not None:
            return {"error": err, "tool": name}
        try:
            return self._bound[name](**kwargs)  # direct call — no exec/eval
        except Exception as exc:  # tool failure degrades gracefully for the LLM
            return {"error": str(exc), "tool": name}
