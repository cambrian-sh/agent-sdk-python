"""Shared helpers for Cambrian agent authors — ADR-0023 Issue #0023-05.

All functions are pure and stateless — no gRPC, no external dependencies.
"""

from __future__ import annotations

import html
import re
from typing import List, Optional, TYPE_CHECKING

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
DEFAULT_CONSTRAINTS: List[str] = [
    "Issue at most ONE JSON action per turn. Do NOT emit multiple actions in one response.",
    "memory_query is read-only and uncapped; tool_call has a per-run cap (5 by default). When the tool budget is exhausted, answer from what you have.",
    "After 2 failed attempts of the same action (cosine > 0.85), the kernel's recurrence gate VETOES further attempts. Change your approach or answer.",
    "When emitting final_answer, ground it: at least one memory_query or tool_call must have preceded it in the trajectory.",
]

# AntiPatterns — the "do not" list, lifted from observed failure modes
# (recurrence gate vetoes, truncation retries, premature final_answer).
# These are NOT loop-injected; they live in the prompt for the entire run.
DEFAULT_ANTI_PATTERNS: List[str] = [
    "[NO] Don't re-run a successful tool_call — the kernel's success-dedup will block; see the trajectory.",
    "[NO] Don't claim a tool is 'missing' without first trying find_tools (and find_skills for a procedure).",
    "[NO] Don't emit final_answer after a single empty memory_query — the result may not have been relevant; try a different query or find_tools.",
    "[NO] Don't truncate a large tool_call payload at the token limit — chunk with append (e.g. fast_large_write_file in sections).",
    "[NO] Don't ignore a <reflection> block in the trajectory — it was extracted from a previous failure for a reason.",
    "[NO] Don't memory_query for content that already has a cid (an offloaded block in the trajectory, a recalled fact with `[full content cid:…]`, or a workspace seed). Use the cid as {\"$cid\": \"<cid>\"} in the next tool_call — the kernel resolves it.",
    "[NO] Don't resolve_cid when you can pass the cid directly to the next tool_call. resolve_cid is for the rare case where you need the body in YOUR context (to quote, summarize, or reason about it). Offload mode is free; inline mode costs N tokens and is capped at 3 per run.",
]


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
    """
    # 1. <System> — frame-setting, constraints, context, anti-patterns.
    system_parts = [f"<Role>\n{role.strip()}\n</Role>"]
    all_constraints = list(DEFAULT_CONSTRAINTS) + list(constraints or [])
    if all_constraints:
        system_parts.append(
            "<Constraints>\n"
            + "\n".join(f"- {c}" for c in all_constraints)
            + "\n</Constraints>"
        )
    if task_type:
        system_parts.append(
            f"<TaskType>\nThis is a {task_type} task. Step {step_no}.\n</TaskType>"
        )
    if session_context:
        system_parts.append(
            f"<SessionContext>\n{session_context.strip()}\n</SessionContext>"
        )
    all_anti = list(DEFAULT_ANTI_PATTERNS) + list(extra_anti_patterns or [])
    if all_anti:
        system_parts.append(
            "<AntiPatterns>\n"
            + "\n".join(f"- {p}" for p in all_anti)
            + "\n</AntiPatterns>"
        )
    parts = ["<System>\n" + "\n".join(system_parts) + "\n</System>"]

    # 2. <Task> — loop-invariant within a run; high attention at front.
    parts.append(f"<Task>\n{task.strip()}\n</Task>")

    # 3. <ActionProtocol> — composed once per task (loop-invariant).
    if action_protocol and action_protocol.strip():
        parts.append(f"<ActionProtocol>\n{action_protocol.strip()}\n</ActionProtocol>")

    # 4. <OutputSchema> — recency-adjacent (right before Trajectory).
    if output_schema and output_schema.strip():
        parts.append(f"<OutputSchema>\n{output_schema.strip()}\n</OutputSchema>")

    # 5. <Trajectory> — VARIABLE; must be the last section for prefix caching.
    # The step number goes in the BODY (not the tag) so it does NOT invalidate
    # the cache prefix when the step increments turn-over-turn. The tag is
    # therefore byte-identical for every turn of the same run, and the
    # provider's prefix cache covers it.
    if context_str and context_str.strip():
        step_note = f" (round {step_no})" if step_no > 0 else ""
        header = (
            f"Your ReAct loop so far{step_note}. Each <step> is one round (oldest first). "
            "Use the trajectory to avoid repeating work; then choose your next single action. "
            "If a <reflection> block is present, it was extracted from a previous failure — read it before retrying."
        )
        parts.append(f"<Trajectory>\n{header}\n\n{context_str.strip()}\n</Trajectory>")

    return "\n\n".join(parts)
