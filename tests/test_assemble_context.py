"""Tests for assemble_context() and ContextRef (ADR-0022 Phase 3 Python SDK)."""

import pytest

from cambrian_agent_sdk import assemble_context
from cambrian_agent_sdk.types import ContextRef, ExecuteRequest


# ── Fixtures ──────────────────────────────────────────────────────────────────


def ref(cid: str, activation: float, precision: float, snippet: str = "") -> ContextRef:
    """Build a ContextRef with given activation and precision."""
    return ContextRef(cid=cid, activation=activation, precision=precision, snippet=snippet)


# ── Tracer bullet: empty refs returns empty string ────────────────────────────


def test_assemble_context_empty_refs():
    assert assemble_context([]) == ""


def test_assemble_context_surfaces_cid_for_by_reference_passing():
    """A high-precision prior-step ref renders its head PLUS the cid marker, so the agent
    can pass the FULL body by reference (`{"$cid": …}`) instead of only seeing the
    truncated snippet (the bug that made a 'write the previous step's output' task
    regenerate a different summary)."""
    refs = [ref("cid-helo-9k", activation=1.0, precision=1.0, snippet="# Helicopter summary head…")]
    result = assemble_context(refs, min_precision=0.5)  # snippet-only (no fetch_fn)
    assert "[full content cid:cid-helo-9k]" in result, result
    assert "Helicopter summary head" in result


# ── Sort order: activation × precision descending ────────────────────────────


def test_assemble_context_sort_activation_times_precision():
    refs = [
        ref("a", activation=0.5, precision=0.5, snippet="A"),  # score=0.25
        ref("b", activation=0.9, precision=0.9, snippet="B"),  # score=0.81
        ref("c", activation=0.7, precision=0.6, snippet="C"),  # score=0.42
    ]
    # All above min_precision=0.4 default; snippet-only mode (fetch_fn=None)
    result = assemble_context(refs, min_precision=0.4)
    # B should appear first (0.81), then C (0.42), then A (0.25)
    b_pos = result.index("[step_result]") if "[step_result]" in result else result.find("B")
    assert result.index("B") < result.index("C") < result.index("A"), \
        f"Expected B > C > A order, got: {result!r}"


# ── min_precision filter skips low-precision refs ────────────────────────────


def test_assemble_context_skips_below_min_precision():
    refs = [
        ref("high", activation=0.8, precision=0.9, snippet="high precision"),
        ref("low",  activation=0.9, precision=0.3, snippet="low precision"),
    ]
    result = assemble_context(refs, min_precision=0.5)
    assert "high precision" in result
    assert "low precision" not in result


# ── Precision sentinel -1.0 + no fetch_fn → skip ─────────────────────────────


def test_assemble_context_sentinel_without_fetch_fn_skipped():
    """BFS nodes (precision=-1.0) are skipped when fetch_fn is None."""
    refs = [
        ref("bfs", activation=0.9, precision=-1.0, snippet="bfs snippet"),
        ref("seed", activation=0.7, precision=0.8, snippet="seed snippet"),
    ]
    result = assemble_context(refs, fetch_fn=None)
    assert "seed snippet" in result
    assert "bfs snippet" not in result


# ── fetch_fn called only above fetch_threshold ───────────────────────────────


def test_assemble_context_fetch_fn_called_above_threshold():
    fetched = []

    def mock_fetch(cid):
        fetched.append(cid)

        class _Node:
            data = f"full content of {cid}".encode()
        return _Node()

    refs = [
        ref("above", activation=0.9, precision=0.8, snippet="snippet-above"),
        ref("below", activation=0.7, precision=0.5, snippet="snippet-below"),
    ]
    result = assemble_context(refs, fetch_fn=mock_fetch, fetch_threshold=0.7, min_precision=0.4)

    assert "above" in fetched         # precision=0.8 ≥ fetch_threshold=0.7 → fetched
    assert "below" not in fetched     # precision=0.5 < fetch_threshold=0.7 → uses snippet
    assert "full content of above" in result
    assert "snippet-below" in result


# ── Snippet fallback when fetch_fn is None ────────────────────────────────────


def test_assemble_context_snippet_fallback_no_fetch():
    refs = [ref("r", activation=0.8, precision=0.7, snippet="fallback content")]
    result = assemble_context(refs, fetch_fn=None)
    assert "fallback content" in result


# ── Token budget stops accumulation ──────────────────────────────────────────


def test_assemble_context_token_budget_enforced():
    """Result must stay within max_tokens (rough chars/4 estimate)."""
    refs = [ref(str(i), activation=0.9, precision=0.9, snippet="x" * 100) for i in range(20)]
    result = assemble_context(refs, max_tokens=10, fetch_fn=None)
    # max_tokens=10 → ~40 chars budget
    assert len(result) <= 200, f"Result too long: {len(result)} chars"


# ── Returns empty string when all filtered ───────────────────────────────────


def test_assemble_context_all_filtered_returns_empty():
    refs = [ref("r", activation=0.8, precision=0.2, snippet="data")]
    result = assemble_context(refs, min_precision=0.5)
    assert result == ""


# ── Never raises ──────────────────────────────────────────────────────────────


def test_assemble_context_never_raises_on_bad_fetch():
    def bad_fetch(cid):
        raise RuntimeError("connection error")

    refs = [ref("r", activation=0.8, precision=0.9, snippet="fallback")]
    # Should fall back to snippet, not raise
    result = assemble_context(refs, fetch_fn=bad_fetch, fetch_threshold=0.7)
    assert "fallback" in result


# ── ContextRef type ───────────────────────────────────────────────────────────


def test_context_ref_defaults():
    r = ContextRef(cid="abc123")
    assert r.cid == "abc123"
    assert r.activation == 0.0
    assert r.precision == -1.0  # sentinel default
    assert r.snippet == ""
    assert r.type == ""
    assert r.labels == []


# ── ExecuteRequest has working_memory and plan_id ────────────────────────────


def test_execute_request_has_working_memory():
    r = ContextRef(cid="doc-1", activation=0.9, precision=0.8)
    req = ExecuteRequest(working_memory=[r], plan_id="plan-abc")
    assert len(req.working_memory) == 1
    assert req.working_memory[0].cid == "doc-1"
    assert req.plan_id == "plan-abc"


def test_execute_request_working_memory_defaults_empty():
    req = ExecuteRequest()
    assert req.working_memory == []
    assert req.plan_id == ""
