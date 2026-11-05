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

from . import prompt_library
from .working_memory import _condense_args, action_text

# The library prompt this module renders (cambrian_agent_sdk/prompts/reflection/
# v<N>.md, ADR-0139 C4). The wording lives there; this module fills the slots.
REFLECTION_PROMPT_ID = "reflection"

# Keep a reflection terse: it is pinned into every later prompt, so it must be a
# lesson, not an essay. The orchestrator caps generation tokens to match.
DEFAULT_MAX_REFLECTION_TOKENS = 200


def _role_sentence(role: str) -> str:
    """Render the role as one "You are ..." sentence, idempotently (ADR-0139 Phase 0).

    Roles are authored both ways: as a noun phrase ("a careful test assistant") and
    as a full sentence ("You are a precise summarisation engine."). Prefixing the
    second with "You are" produced "You are You are a precise summarisation
    engine.." in every reflection prompt."""
    text = (role or "").strip()
    if not text:
        return "You are an agent."
    if text.lower().startswith(("you are ", "you're ")):
        return text if text.endswith((".", "!", "?")) else text + "."
    return f"You are {text.rstrip('.')}."


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
    return prompt_library.get(REFLECTION_PROMPT_ID).render({
        "role_sentence": _role_sentence(role),
        "task": task_text,
        "action": sig,
        "failure": failure_summary or "(no detail)",
        "prior_reflections": prior,
    }).text
