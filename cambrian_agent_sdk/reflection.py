"""ADR-0052: verbal self-reflection (Reflexion, Shinn et al. 2023).

The recurrence gate (``recurrence.py``) is a *structural* reflection — it counts
how many times an action near-duplicates a prior **failed** card and re-prompts
with a terse veto note. It never asks the model *why* the action failed or *what*
to change. Reflexion's finding is that a one-sentence **verbal** reflection,
carried into the next attempt, lets an agent learn from failure *within a single
run* — the gap ``docs/research/agent-loops/reflexion-2023.md`` flags as the
"natural next step" after the gate.

This module is the verbal counterpart to the structural gate: it builds the focused
prompt that asks the LLM to articulate the lesson. The orchestration (when to ask,
storing the answer as a PINNED working-memory entry) lives in ``react.run_think`` —
exactly as ``recurrence.py`` is pure detection and ``react.py`` owns the loop. The
reflection is kept verbal+local (no extra subsystem): structural gate for the *fast*
veto, verbal reflection for *slow* learning, both bounded by the same veto depth.
"""

from __future__ import annotations

from typing import List

from .working_memory import _condense_args, action_text

# Keep a reflection terse: it is pinned into every later prompt, so it must be a
# lesson, not an essay. The orchestrator caps generation tokens to match.
DEFAULT_MAX_REFLECTION_TOKENS = 200


def build_reflection_prompt(
    role: str,
    task_text: str,
    tool: str,
    args,
    failure_summary: str,
    prior_reflections: List[str],
) -> str:
    """Build the prompt that asks the LLM for a one/two-sentence verbal reflection.

    Focused and closed: it names the role, the task, the **specific** repeatedly
    failing action (with condensed args, so a multi-KB payload does not bloat the
    reflection call), the most recent failure message, and any reflections already
    gathered this run (so the model does not repeat a lesson). It explicitly forbids
    emitting JSON/an action — the answer is plain prose the loop stores as a
    ``<reflection>`` block, NOT another action to execute.
    """
    sig = action_text(tool, _condense_args(args if isinstance(args, dict) else {"value": args}))
    prior = "\n".join(f"- {r}" for r in prior_reflections) if prior_reflections else "(none yet)"
    return (
        f"You are {role}.\n"
        f"Task: {task_text}\n\n"
        "An action you keep issuing has FAILED repeatedly and was vetoed to stop a "
        "retry loop:\n"
        f"  action: {sig}\n"
        f"  most recent failure: {failure_summary or '(no detail)'}\n\n"
        f"Reflections you already recorded this run:\n{prior}\n\n"
        "In ONE or TWO plain sentences, reflect: WHY did this action fail, and WHAT "
        "will you do differently next — a different tool, different arguments, or a "
        "different approach. Be concrete and specific to THIS failure.\n"
        "Output only the reflection sentence(s). Do NOT output JSON, and do NOT "
        "emit an action."
    )
