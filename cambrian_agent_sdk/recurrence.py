"""ADR-0041 D4: the recurrent reconciliation gate.

The flat ReAct loop is a pure *feedforward sweep* (Recurrent Processing Theory):
it proposes an action and runs it, with no reentrant check against what already
happened — so it re-issues a failing call with cosmetic variations (the canonical
``find … -size +1048576`` then ``find … -size +1M``, both timing out).

This module is the *reentrant* pass: before a ``tool_call`` is committed, it
reconciles the proposed action against the episodic buffer and counts how many
prior **failed** cards it near-duplicates. The loop (``react.py``) turns that count
into a local escalation ladder (soft nudge → hard veto → escalate), mirroring the
kernel's FailureEscalationLadder (ADR-0037 D3) at Local-RP scale. It is *local
active inference*: act to reduce expected surprise rather than repeat it.

Detection combines an **exact** arg match (identical retries) with a **semantic**
cosine over the cached action embeddings (cosmetic-variation retries) — the latter
is why exact hashing alone is insufficient.
"""

from __future__ import annotations

from typing import List, Optional

from .working_memory import ToolCard, _cosine, action_text

# Cosine at/above which two same-tool actions are "the same action" (a deferred
# estimator — tuned in practice, not a calibrated constant).
DEFAULT_THRESHOLD = 0.85


def count_failed_duplicates(
    tool: str,
    args,
    action_vec: Optional[List[float]],
    cards: List[ToolCard],
    threshold: float = DEFAULT_THRESHOLD,
) -> int:
    """Number of prior **failed** cards that near-duplicate this action.

    A prior card counts when it is a failure (``status`` error/denied), names the
    **same tool**, and matches the proposed action either **exactly** (identical
    canonical args) or **semantically** (``cosine(action embeddings) >= threshold``).
    ``0`` means a novel action — the gate lets it through.
    """
    sig = action_text(tool, args)
    n = 0
    for c in cards:
        if c.status not in ("error", "denied") or c.tool != tool:
            continue
        if action_text(c.tool, c.args) == sig:
            n += 1
        elif action_vec and c.action_vec and _cosine(action_vec, c.action_vec) >= threshold:
            n += 1
    return n


def count_successful_duplicates(
    tool: str,
    args,
    action_vec: Optional[List[float]],
    cards: List[ToolCard],
    threshold: float = DEFAULT_THRESHOLD,
) -> int:
    """Number of prior **successful** cards that near-duplicate this action.

    The symmetric twin of :func:`count_failed_duplicates`: where that gate stops
    the loop re-issuing a *failing* call, this one detects re-issuing an action
    that **already succeeded** — an idempotent no-op spin (e.g. writing the same
    file with the same content turn after turn).

    A prior card counts when it succeeded (``status`` ``"ok"``), names the **same
    tool**, and matches the proposed action either **exactly** (identical canonical
    args) or **semantically** (``cosine >= threshold``). It is **content-keyed, not
    count-keyed**: a multi-step plan of *distinct* successful calls never trips it —
    only re-running the *same* call does. ``0`` means a genuinely new action.
    """
    sig = action_text(tool, args)
    n = 0
    for c in cards:
        if c.status != "ok" or c.tool != tool:
            continue
        if action_text(c.tool, c.args) == sig:
            n += 1
        elif action_vec and c.action_vec and _cosine(action_vec, c.action_vec) >= threshold:
            n += 1
    return n
