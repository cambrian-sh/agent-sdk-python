"""Workers composed by the kernel (ADR-0145 D12), the runtime side.

The kernel composes the worker for a plan step — one ReAct runtime, the skills of the
professions that cover the step, the tools those professions may use that the step's
effects and the person's scope allow, the brief and the ledger references of its
inputs — binds that composition to the step's lease, and sends it with the task under
the handoff key ``_composition`` (kernel-authored DATA, never instructions).

The kernel is the enforcing control: a composed lease may call only its composition's
tools (a per-lease ceiling held over every bypass), and its skill index is the
composition (a skill outside it is never served and never requested from an operator).
This module only keeps the runtime's own view consistent with it:

- :class:`Composition` parses the handoff value.
- :class:`ComposedScopedSubstrate` presents the step's lease on the menu, skill-menu
  and skill-load calls (the kernel resolves the composition from it), keeps the skill
  menu to the composition in a stable order, and refuses locally a tool call that is
  neither in the composition nor on the menu the kernel served for the step (the
  kernel's own tools, such as the ledger reads, are on that menu without being part of
  the composition).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger("cambrian_agent_sdk.composition")

#: The handoff context key the kernel sends the composition under
#: (cambrian-core domain.HandoffKeyComposition).
COMPOSITION_KEY = "_composition"

@dataclass(frozen=True)
class Composition:
    runtime: str
    professions: Tuple[str, ...] = ()
    skills: Tuple[str, ...] = ()
    tools: Tuple[str, ...] = ()
    effects: Tuple[str, ...] = ()
    effects_declared: bool = False
    brief: str = ""
    ledger_refs: Tuple[str, ...] = ()
    model: str = ""
    system_prompt: str = ""
    run_id: str = ""
    step: int = -1
    raw: Dict[str, Any] = field(default_factory=dict, compare=False, repr=False)

    @classmethod
    def from_json(cls, text: str) -> Optional["Composition"]:
        try:
            d = json.loads(text or "")
        except (TypeError, ValueError):
            return None
        if not isinstance(d, dict) or not d.get("runtime"):
            return None

        def strs(key: str) -> Tuple[str, ...]:
            v = d.get(key) or []
            return tuple(sorted({str(x) for x in v if x})) if isinstance(v, list) else ()

        step = d.get("step", -1)
        return cls(
            runtime=str(d["runtime"]), professions=strs("professions"), skills=strs("skills"),
            tools=strs("tools"), effects=strs("effects"),
            effects_declared=bool(d.get("effects_declared")), brief=str(d.get("brief") or ""),
            ledger_refs=strs("ledger_refs"), model=str(d.get("model") or ""),
            system_prompt=str(d.get("system_prompt") or ""), run_id=str(d.get("run_id") or ""),
            step=step if isinstance(step, int) else -1, raw=d,
        )

    def allows_tool(self, name: str) -> bool:
        return name in self.tools

    def allows_skill(self, name: str) -> bool:
        return name in self.skills


def composition_from_task(task: Any) -> Optional[Composition]:
    """The task's composition, or None when the kernel sent none."""
    ctx = getattr(task, "context", None) or {}
    raw = ctx.get(COMPOSITION_KEY) if isinstance(ctx, dict) else None
    return Composition.from_json(raw) if raw else None


class ComposedScopedSubstrate:
    """A substrate proxy for the composed runtime: the current step's composition and
    lease are bound with :meth:`bind` at the start of each task.

    Without a bound composition it offers nothing (no tools, no skills) — the kernel
    does the same for the runtime without a composed lease. Every other attribute
    passes through to the wrapped substrate.
    """

    def __init__(self, inner: Any):
        object.__setattr__(self, "_inner", inner)
        object.__setattr__(self, "_comp", None)
        object.__setattr__(self, "_lease", "")
        object.__setattr__(self, "_served", set())

    @property
    def inner(self) -> Any:
        return self._inner

    @property
    def composition(self) -> Optional[Composition]:
        return self._comp

    def bind(self, comp: Optional[Composition], lease: str) -> None:
        object.__setattr__(self, "_comp", comp)
        object.__setattr__(self, "_lease", lease or "")
        object.__setattr__(self, "_served", set())

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def _call(self, fn, kwargs: Dict[str, Any], optional: Sequence[str]):
        while True:
            try:
                return fn(**kwargs)
            except TypeError:  # a substrate (or fake) without an optional kwarg
                for opt in optional:
                    if opt in kwargs:
                        kwargs.pop(opt)
                        break
                else:
                    raise

    def list_tools(self, query: str = "", k: int = 0, names: Optional[List[str]] = None,
                   full: bool = False, session_token_id: str = "") -> List[Dict]:
        comp = self._comp
        if comp is None or not hasattr(self._inner, "list_tools"):
            return []
        # The kernel serves the composed lease's menu (the composition plus its own
        # in-process tools) whole: the full listing, never a top-k guess.
        kwargs: Dict[str, Any] = {"full": full, "session_token_id": session_token_id or self._lease}
        if names:
            kwargs["names"] = list(names)
        tools = self._call(self._inner.list_tools, kwargs, ("session_token_id",)) or []
        self._served.update(t.get("name", "") for t in tools)
        return tools

    def list_skills(self, query: str = "", k: int = 0, names: Optional[List[str]] = None,
                    full: bool = False, session_token_id: str = "", rationale: str = "",
                    discover: bool = False) -> List[Dict]:
        comp = self._comp
        if comp is None or not hasattr(self._inner, "list_skills"):
            return []
        kwargs: Dict[str, Any] = {"full": full, "session_token_id": session_token_id or self._lease}
        if names:
            kwargs["names"] = list(names)
        specs = self._call(self._inner.list_skills, kwargs, ("session_token_id",)) or []
        out = []
        for s in specs:
            name = s.get("name") or ""
            if comp.allows_skill(name):
                out.append(s)
            elif names:
                # A named load outside the composition: passed through without a body
                # so the loop tells the model it is not available (never widened).
                out.append({"name": name, "description": s.get("description", ""),
                            "instructions": "", "tool_grants": [], "access": "denied",
                            "access_detail": s.get("access_detail")
                            or "outside the skills the kernel composed for this step"})
        if not names:
            # Rendered into the system prompt: one order for every step of a run.
            out.sort(key=lambda s: str(s.get("name") or ""))
        return out

    def execute_tool(self, name: str, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        comp = self._comp
        if comp is None or not (comp.allows_tool(name) or name in self._served):
            return {"denied": True,
                    "deny_reason": f"tool {name!r} is outside the tools composed for this step "
                                   "(client-side check; the kernel enforces the same composition)"}
        return self._inner.execute_tool(name, *args, **kwargs)
