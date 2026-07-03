"""ADR-0041 D4: the recurrent reconciliation gate detection logic."""

from cambrian_agent_sdk.recurrence import (
    count_failed_duplicates,
    count_successful_duplicates,
)
from cambrian_agent_sdk.working_memory import ToolCard


def _failed(tool, args, action_vec=None):
    c = ToolCard.from_result(tool, args, {"error": "DEADLINE_EXCEEDED"})
    c.action_vec = action_vec
    return c


def _ok(tool, args, action_vec=None):
    c = ToolCard.from_result(tool, args, {"content": "ok"})  # status ok
    c.action_vec = action_vec
    return c


def test_exact_arg_duplicate_of_failure_counts():
    cards = [_failed("execute_command", {"command": "find . -size +1M"})]
    assert count_failed_duplicates("execute_command", {"command": "find . -size +1M"}, None, cards) == 1


def test_success_is_not_a_duplicate():
    ok = ToolCard.from_result("execute_command", {"command": "ls"}, {"out": "files"})  # status ok
    assert count_failed_duplicates("execute_command", {"command": "ls"}, None, [ok]) == 0


def test_different_tool_is_not_a_duplicate():
    cards = [_failed("read_file", {"path": "/x"})]
    assert count_failed_duplicates("write_file", {"path": "/x"}, None, cards) == 0


def test_novel_action_is_zero():
    cards = [_failed("execute_command", {"command": "find . -size +1M"})]
    assert count_failed_duplicates("execute_command", {"command": "ls -la"}, None, cards) == 0


def test_semantic_near_duplicate_detected_not_just_hash():
    """`+1048576` vs `+1M`: different arg strings (exact-hash MISSES) but a
    near-identical action — caught semantically via the cached action embeddings."""
    prior = _failed("execute_command", {"command": "find . -size +1048576"}, action_vec=[1.0, 0.0])
    proposed_vec = [0.99, 0.01]  # ~same direction
    n = count_failed_duplicates(
        "execute_command", {"command": "find . -size +1M"}, proposed_vec, [prior], threshold=0.85,
    )
    assert n == 1


def test_semantic_below_threshold_not_a_duplicate():
    prior = _failed("execute_command", {"command": "find . -size +1M"}, action_vec=[1.0, 0.0])
    orthogonal = [0.0, 1.0]  # cosine 0 < threshold
    assert count_failed_duplicates("execute_command", {"command": "grep x"}, orthogonal, [prior]) == 0


# ── count_successful_duplicates: the symmetric guard ───────────────────────────


def test_repeated_successful_action_is_a_duplicate():
    """The loop bug: same tool + same args already succeeded → idempotent no-op."""
    cards = [_ok("mcp:filesystem/write_file", {"path": "hello.txt", "content": "hi"})]
    assert count_successful_duplicates(
        "mcp:filesystem/write_file", {"path": "hello.txt", "content": "hi"}, None, cards
    ) == 1


def test_distinct_successful_calls_are_not_duplicates():
    """A real multi-step plan: writing a DIFFERENT file must never trip the guard."""
    cards = [_ok("mcp:filesystem/write_file", {"path": "a.txt", "content": "hi"})]
    assert count_successful_duplicates(
        "mcp:filesystem/write_file", {"path": "b.txt", "content": "hi"}, None, cards
    ) == 0


def test_failure_is_not_a_successful_duplicate():
    """A prior FAILURE of the same action is the other gate's business, not this one."""
    cards = [_failed("write_file", {"path": "x"})]
    assert count_successful_duplicates("write_file", {"path": "x"}, None, cards) == 0


def test_successful_semantic_near_duplicate_detected():
    prior = _ok("write_file", {"path": "hello.txt", "content": "hi"}, action_vec=[1.0, 0.0])
    proposed_vec = [0.99, 0.01]
    assert count_successful_duplicates(
        "write_file", {"path": "hello.txt", "content": "hi "}, proposed_vec, [prior], threshold=0.85
    ) == 1
