"""Shared helpers for Cambrian agent authors — ADR-0023 Issue #0023-05.

All functions are pure and stateless — no gRPC, no external dependencies.
"""

from __future__ import annotations

import hashlib
import html
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from .prompt_library import fragment

if TYPE_CHECKING:
    from .types import ContextRef

_FENCE_RE = re.compile(r"```(?:python)?\n([\s\S]*?)```", re.MULTILINE)


def extract_code_block(text: str) -> str:
    """Extract the first fenced code block from LLM output.

    LLMs wrap generated code in ```python ... ``` or ``` ... ``` markdown
    fences. Passing raw LLM output to subprocess causes SyntaxError on the
    fence line. This helper strips the fences and returns the raw code.

    Falls back to the full text (stripped) if no fence is found — so agents
    work correctly whether or not the LLM chooses to use a fence.
    """
    if not text:
        return ""
    match = _FENCE_RE.search(text)
    if match:
        return match.group(1).strip()
    return text.strip()


def find_step_ref(working_memory: "List[ContextRef]", step_index: int) -> "Optional[ContextRef]":
    """Return the first step_result ContextRef for the given step index.

    Iterates working_memory looking for a ref with type="step_result" and
    a label of the form "step_{step_index}". Returns None if not found.

    Used by tool agents (e.g. code_executor) to locate the CID of a prior
    step's output without reimplementing label-scanning logic.
    """
    label = f"step_{step_index}"
    for ref in working_memory:
        if ref.type == "step_result" and label in ref.labels:
            return ref
    return None


# ──────────────────────────────────────────────────────────────────────
# v2 prompt defaults — see docs/research/agent-prompting/SUMMARY.md
# ──────────────────────────────────────────────────────────────────────

# These four are loop-invariant: composed once per task and pinned in <System>
# at the top of the prompt. The loop can append runtime-injected constraints
# (memory budget exhausted, truncation retry, veto note) on top of these.
# They mirror the 4 Rules in <ActionProtocol> for the prompt-mid attention
# boost — the copy at the top is the recency-priority one.
#
# GENERATED, not literal (ADR-0139 Phase 0, prompt/code drift). The literal list
# told the model "memory_query is ... uncapped; tool_call has a per-run cap (5 by
# default)" while the loop enforced 3 memory queries and 12 tool rounds, so the
# model planned against budgets that did not exist. The text is now rendered from
# the EFFECTIVE limits the loop enforces, in one of two protocol variants: under
# native tool-calling there is no JSON tool_call action, and a prompt that says
# "emit one JSON object" beside attached tool schemas gives the model two ways to
# act, of which it picks the one that does nothing (measured 2026-07-28, react.py).
def default_constraints(*, native_tools: bool = False,
                        max_memory_queries: Optional[int] = None,
                        max_tool_rounds: Optional[int] = None) -> List[str]:
    """The <Constraints> defaults for one run, from the limits the loop enforces.

    ``None`` budgets resolve to the loop's own defaults in ``react.py`` (one
    default, one place), imported lazily because ``react`` imports this module.
    """
    if max_memory_queries is None or max_tool_rounds is None:
        from .react import _DEFAULT_MAX_MEMORY_QUERIES, _DEFAULT_MAX_TOOL_ROUNDS
        if max_memory_queries is None:
            max_memory_queries = _DEFAULT_MAX_MEMORY_QUERIES
        if max_tool_rounds is None:
            max_tool_rounds = _DEFAULT_MAX_TOOL_ROUNDS
    # The text lives in the prompt library (prompts/react-constraints/v1.md, ADR-0139
    # C4); this function only picks the protocol variant and fills the budgets.
    variant = "native" if native_tools else "text"
    if max_memory_queries > 0:
        mem = fragment("react-constraints", f"memory-{variant}").render(
            {"max_memory_queries": max_memory_queries})
    else:
        mem = fragment("react-constraints", "memory-none").render()
    body = fragment("react-constraints", variant).render(
        {"memory_clause": mem, "max_tool_rounds": max_tool_rounds})
    return body.splitlines()


# AntiPatterns — the "do not" list, lifted from observed failure modes
# (recurrence gate vetoes, truncation retries, premature final_answer).
# These are NOT loop-injected; they live in the prompt for the entire run.
# Mode-specific for the same reason as default_constraints (ADR-0139 Phase 0).
def default_anti_patterns(*, native_tools: bool = False,
                          large_write_tool: Optional[str] = None) -> List[str]:
    """The <AntiPatterns> defaults. ``large_write_tool`` names a sectioned-write tool
    only when it is actually in this agent's menu: naming a tool the model cannot
    call sends it looking for one it will never find."""
    # Text: prompts/react-anti-patterns/v1.md (ADR-0139 C4).
    if large_write_tool:
        chunk = fragment("react-anti-patterns", "chunk-tool").render(
            {"large_write_tool": large_write_tool})
    else:
        chunk = fragment("react-anti-patterns", "chunk-none").render()
    return fragment("react-anti-patterns", "native" if native_tools else "text").render(
        {"chunk": chunk}).splitlines()


def __getattr__(name: str):
    # Backward compatibility: DEFAULT_CONSTRAINTS / DEFAULT_ANTI_PATTERNS were module
    # constants. They are now the TEXT-protocol defaults rendered from the real loop
    # budgets, computed on access so they cannot drift from react.py again.
    if name == "DEFAULT_CONSTRAINTS":
        return default_constraints()
    if name == "DEFAULT_ANTI_PATTERNS":
        return default_anti_patterns()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _esc(s: str) -> str:
    """XML-escape user/tool content so a query with < or & doesn't break the tags.

    Single quotes in the output preserve backward compat with existing
    trajectory-rendering tests (which assert `status='ok'` etc).
    """
    return html.escape(str(s or ""), quote=False).replace('"', '&quot;')


def build_prompt(
    role: str,
    task: str,
    context_str: str = "",
    output_schema: str = "",
    constraints: "list[str] | None" = None,
    action_protocol: str = "",
    step_no: int = 0,
    task_type: "Optional[str]" = None,
    session_context: "Optional[str]" = None,
    extra_anti_patterns: "Optional[List[str]]" = None,
    base_constraints: "Optional[List[str]]" = None,
    base_anti_patterns: "Optional[List[str]]" = None,
) -> str:
    """Assemble a canonical LLM prompt (v2 — max-cache layout).

    Section order (positions 1-4 are the cacheable prefix; position 5 is
    the variable suffix that breaks the cache as intended):

        1. <System>            Role + Constraints + TaskType + SessionContext + AntiPatterns
        2. <Task>              loop-invariant; high attention at front
        3. <ActionProtocol>    composed once per task (loop-invariant)
        4. <OutputSchema>      per-turn contract; recency-adjacent
        5. <Trajectory>        VARIABLE; must be last for prefix caching

    The ActionProtocol, OutputSchema, Role, Constraints, Task, TaskType,
    SessionContext, and AntiPatterns are all loop-invariant — composed
    ONCE per task — so the entire 1-4 region is byte-identical across
    turns. The provider's KV cache covers all of it; only the Trajectory
    is recomputed per turn.

    The v1 order (System → Trajectory → Task → ActionProtocol → OutputSchema)
    is intentionally NOT used here. The v1 order placed the variable
    Trajectory in position 2, which broke the cache for the entire
    ActionProtocol — the longest loop-invariant section — on every turn.

    Args:
        role: Multi-sentence persona for <Role>.
        task: The current task instruction for <Task>.
        context_str: The agent's assembled working memory (trajectory body).
        output_schema: Per-turn output contract for <OutputSchema>. Omitted when empty.
        constraints: Optional runtime-injected constraints; appended to DEFAULT_CONSTRAINTS.
        action_protocol: Optional agent-loop action menu + rules for <ActionProtocol>.
        step_no: Current loop step (1-based); surfaced on the <Trajectory> tag. 0 omits it.
        task_type: Optional classification ("research" | "code" | "data" | "chat") for <TaskType>.
        session_context: Optional "you are agent X in plan Y, budget Z" string for <SessionContext>.
        extra_anti_patterns: Optional runtime-injected anti-patterns; appended to DEFAULT_ANTI_PATTERNS.
        base_constraints: Replaces the text-protocol DEFAULT_CONSTRAINTS. The ReAct loop
            passes the variant for its protocol and effective budgets (ADR-0139).
        base_anti_patterns: Replaces DEFAULT_ANTI_PATTERNS likewise.
    """
    system_block = _system_section(
        role, constraints=constraints, task_type=task_type,
        task_type_step=step_no, session_context=session_context,
        extra_anti_patterns=extra_anti_patterns, base_constraints=base_constraints,
        base_anti_patterns=base_anti_patterns)
    parts = [system_block]

    # 2. <Task> — loop-invariant within a run; high attention at front.
    parts.append(f"<Task>\n{task.strip()}\n</Task>")

    # 3. <ActionProtocol> — composed once per task (loop-invariant).
    # 4. <OutputSchema> — recency-adjacent (right before Trajectory).
    parts.extend(_protocol_sections(action_protocol, output_schema))

    # 5. <Trajectory> — VARIABLE; must be the last section for prefix caching.
    # The step number goes in the BODY (not the tag) so it does NOT invalidate
    # the cache prefix when the step increments turn-over-turn. The tag is
    # therefore byte-identical for every turn of the same run, and the
    # provider's prefix cache covers it.
    if context_str and context_str.strip():
        parts.append(f"<Trajectory>\n{_trajectory_header(step_no)}\n\n{context_str.strip()}\n</Trajectory>")

    return "\n\n".join(parts)


def _trajectory_header(step_no: int) -> str:
    if step_no > 0:
        return fragment("react-turn", "trajectory-header").render({"round": step_no})
    return fragment("react-turn", "trajectory-header-unnumbered").render()


def _system_section(
    role: str,
    *,
    constraints: "Optional[List[str]]" = None,
    task_type: "Optional[str]" = None,
    task_type_step: "Optional[int]" = None,
    session_context: "Optional[str]" = None,
    extra_anti_patterns: "Optional[List[str]]" = None,
    base_constraints: "Optional[List[str]]" = None,
    base_anti_patterns: "Optional[List[str]]" = None,
) -> str:
    """The ``<System>`` block: Role, Constraints, TaskType, SessionContext, AntiPatterns.

    ``task_type_step`` keeps the legacy "Step N." in <TaskType> for ``build_prompt``;
    the split layout passes ``None`` because a step number is volatile and must not
    reach the system tier."""
    system_parts = [f"<Role>\n{role.strip()}\n</Role>"]
    base_c = default_constraints() if base_constraints is None else base_constraints
    all_constraints = list(base_c) + list(constraints or [])
    if all_constraints:
        system_parts.append(
            "<Constraints>\n"
            + "\n".join(f"- {c}" for c in all_constraints)
            + "\n</Constraints>"
        )
    if task_type:
        step = f" Step {task_type_step}." if task_type_step is not None else ""
        system_parts.append(
            f"<TaskType>\nThis is a {task_type} task.{step}\n</TaskType>"
        )
    if session_context:
        system_parts.append(
            f"<SessionContext>\n{session_context.strip()}\n</SessionContext>"
        )
    base_a = default_anti_patterns() if base_anti_patterns is None else base_anti_patterns
    all_anti = list(base_a) + list(extra_anti_patterns or [])
    if all_anti:
        system_parts.append(
            "<AntiPatterns>\n"
            + "\n".join(f"- {p}" for p in all_anti)
            + "\n</AntiPatterns>"
        )
    return "<System>\n" + "\n".join(system_parts) + "\n</System>"


def _protocol_sections(action_protocol: str, output_schema: str) -> List[str]:
    out = []
    if action_protocol and action_protocol.strip():
        out.append(f"<ActionProtocol>\n{action_protocol.strip()}\n</ActionProtocol>")
    if output_schema and output_schema.strip():
        out.append(f"<OutputSchema>\n{output_schema.strip()}\n</OutputSchema>")
    return out


# ──────────────────────────────────────────────────────────────────────
# The split layout (ADR-0139 D2, wave 1d): a cached system tier + messages
# ──────────────────────────────────────────────────────────────────────

# The SDK prompt-library ids the ReAct prompt is built from. Logged with their
# version and static hash on every call (debug level), so an end-to-end test can
# assert which prompt version a run actually sent.
REACT_PROMPT_IDS = ("react-constraints", "react-anti-patterns", "react-action-protocol",
                    "react-turn", "react-nudges", "react-tools", "react-compaction")


@dataclass
class PromptParts:
    """An assembled prompt in the C1 shape: ``system`` blocks plus ``messages``.

    - ``system`` is the STABLE tier: role, constraints, anti-patterns, the action
      protocol (tool policy, skill index) and the per-turn output contract. One
      block, marked ``cache=True``. Nothing volatile is allowed in it, so it is
      byte-identical across the turns of a run and a provider can cache it (an
      unchanged system prompt over the provider's minimum caches; one that changes
      every turn never does).
    - ``messages`` is the volatile tier: one user turn holding the task, the
      trajectory (memory and tool results, marked as data) and the per-turn notes
      (budget notices and the like).

    ``legacy_prompt()`` flattens both into the single string an older kernel, or a
    substrate without the structured call, expects.
    """

    system: List[Dict[str, Any]] = field(default_factory=list)
    messages: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def system_text(self) -> str:
        return "\n\n".join(b["text"] for b in self.system if b.get("text"))

    @property
    def system_sha256(self) -> str:
        return hashlib.sha256(self.system_text.encode("utf-8")).hexdigest()

    @property
    def user_text(self) -> str:
        return "\n\n".join(m.get("content") or "" for m in self.messages)

    def legacy_prompt(self) -> str:
        return "\n\n".join(p for p in (self.system_text, self.user_text) if p)


def build_prompt_parts(
    role: str,
    task: str,
    context_str: str = "",
    output_schema: str = "",
    constraints: "list[str] | None" = None,
    action_protocol: str = "",
    step_no: int = 0,
    task_type: "Optional[str]" = None,
    session_context: "Optional[str]" = None,
    extra_anti_patterns: "Optional[List[str]]" = None,
    base_constraints: "Optional[List[str]]" = None,
    base_anti_patterns: "Optional[List[str]]" = None,
    notes: "Optional[List[str]]" = None,
    cache: bool = True,
    user_content: "Optional[str]" = None,
) -> PromptParts:
    """Assemble the ReAct prompt as ``system`` + ``messages`` (ADR-0139 D2).

    Same inputs as :func:`build_prompt` plus ``notes``, the per-turn notices
    (memory budget exhausted, and the like) that the old layout appended to
    <Constraints>. They go at the end of the user turn instead: a notice in the
    system tier changes the system text mid-run and voids the provider's cache.

    System block (stable, cached)::

        <System> Role, Constraints, TaskType, SessionContext, AntiPatterns </System>
        <ActionProtocol> ... </ActionProtocol>
        <OutputSchema> ... </OutputSchema>

    User turn (volatile)::

        <Task> ... </Task>
        <Trajectory> header + data note + history </Trajectory>
        <note>...</note>   (per-turn notes)

    ``constraints`` must be stable for the run (caller rules, host facts); a
    step-dependent value belongs in ``notes``.

    ``user_content`` (ADR-0139 D3) replaces the whole user turn: the context engine
    renders its own regions (memory notes, instructions, evidence, history, the task
    last, then the notes), so ``task``, ``context_str`` and ``notes`` are then unused
    for the user turn. The system tier is built the same way either way.
    """
    system_text = "\n\n".join([
        _system_section(role, constraints=constraints, task_type=task_type,
                        task_type_step=None, session_context=session_context,
                        extra_anti_patterns=extra_anti_patterns,
                        base_constraints=base_constraints,
                        base_anti_patterns=base_anti_patterns),
        *_protocol_sections(action_protocol, output_schema),
    ])
    if user_content is not None:
        return PromptParts(
            system=[{"text": system_text, "cache": bool(cache)}],
            messages=[{"role": "user", "content": user_content}],
        )
    user = [f"<Task>\n{task.strip()}\n</Task>"]
    if context_str and context_str.strip():
        user.append(
            f"<Trajectory>\n{_trajectory_header(step_no)}\n"
            f"{fragment('react-turn', 'data-note').render()}\n\n"
            f"{context_str.strip()}\n</Trajectory>")
    for n in notes or []:
        if n and n.strip():
            user.append(n.strip())
    return PromptParts(
        system=[{"text": system_text, "cache": bool(cache)}],
        messages=[{"role": "user", "content": "\n\n".join(user)}],
    )


def prompt_versions() -> List[Dict[str, Any]]:
    """``[{"id", "version", "static_hash"}]`` for the ReAct prompt-library files."""
    from . import prompt_library
    out = []
    for pid in REACT_PROMPT_IDS:
        p = prompt_library.get(pid)
        out.append({"id": p.id, "version": p.version, "static_hash": p.static_hash})
    return out
