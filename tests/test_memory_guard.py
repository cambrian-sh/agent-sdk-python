"""The session memory guard in ``tests/conftest.py`` turns a runaway allocation into a MemoryError.

The runaway runs in a child process so the limit under test never touches this pytest
process (ENG-1606).
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

CONFTEST = Path(__file__).resolve().parent / "conftest.py"


def _guard() -> ModuleType:
    spec = importlib.util.spec_from_file_location("sdk_tests_memory_guard", CONFTEST)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_CHILD = """
import importlib.util, sys
spec = importlib.util.spec_from_file_location("guard", {conftest!r})
guard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(guard)
print(guard.apply_memory_limit(256), flush=True)
small = bytearray(32 * 1024 * 1024)  # well under the cap: must succeed
chunks = []
try:
    while True:  # a runaway: allocates without bound until something stops it
        chunks.append(bytearray(64 * 1024 * 1024))
        if len(chunks) > 64:  # 4 GB: the guard did not fire
            break
except MemoryError:
    print("REFUSED after", len(chunks), "chunks", flush=True)
    sys.exit(0)
print("ALLOCATED", flush=True)
sys.exit(1)
"""


def test_a_runaway_allocation_is_refused_and_one_under_the_limit_succeeds() -> None:
    proc = subprocess.run(
        [sys.executable, "-c", _CHILD.format(conftest=str(CONFTEST))],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert "REFUSED" in proc.stdout, (proc.stdout, proc.stderr)
    assert proc.returncode == 0


def test_limit_env_parsing(monkeypatch: pytest.MonkeyPatch) -> None:
    guard = _guard()
    monkeypatch.delenv(guard.MEMORY_ENV, raising=False)
    assert guard.memory_limit_mb() == guard.DEFAULT_MEMORY_MB
    monkeypatch.setenv(guard.MEMORY_ENV, "0")
    assert guard.memory_limit_mb() == 0
    assert guard.apply_memory_limit(0) == "off"
    monkeypatch.setenv(guard.MEMORY_ENV, "-1")
    with pytest.raises(ValueError, match=guard.MEMORY_ENV):
        guard.memory_limit_mb()


def test_this_session_runs_under_the_guard() -> None:
    # The suite's own process: the guard applied at pytest_configure (unless switched off).
    if os.environ.get("CAMBRIAN_PYTEST_MEMORY_MB", "").strip() == "0":
        pytest.skip("guard switched off for this run")
    how = sys.modules["conftest"]._guard_state.get("how", "") if "conftest" in sys.modules else ""
    assert how and not how.startswith("NOT APPLIED"), how
