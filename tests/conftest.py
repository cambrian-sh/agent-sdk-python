"""Session-wide guard: the test process may not commit more than a fixed amount of memory.

Why (ENG-1606, T-20261114-no-per-test-memory-limit-in-the-python-test): a test that
allocates without bound — a self-calling helper, a fake pagination loop that never ends, a
huge fixture — can commit tens of GB before any timeout fires; on 2026-11-14 one such run
made Windows shut the WSL VM (Docker, HQ, Plane) down. Until now the only guard was an
external capped runner living in one session's temp folder. This guard ships with the repo:
the OS refuses the allocation, so the offending test fails with ``MemoryError`` under its own
name and the machine stays responsive — also under a plain ``pytest``.

Mechanism (the same as cambrian-benchmarks/tests/conftest.py): on Windows the process joins a
job object with ``JOB_OBJECT_LIMIT_PROCESS_MEMORY`` (committed private memory; child processes
inherit the per-process limit). Elsewhere ``RLIMIT_DATA`` caps the data segment (Linux >= 4.7
counts private writable mappings, i.e. the heap and malloc arenas, not reserved address space,
so it does not trip on address-space reservations the way ``RLIMIT_AS`` would). The limit is
``CAMBRIAN_PYTEST_MEMORY_MB`` (default 1536, one knob for both Python repos); ``0`` turns it
off. The terminal summary prints the peak so headroom is visible.
"""
from __future__ import annotations

import ctypes
import os
import sys
from typing import Any

import pytest

MEMORY_ENV = "CAMBRIAN_PYTEST_MEMORY_MB"
DEFAULT_MEMORY_MB = 1536

_JOB_OBJECT_EXTENDED_LIMIT_INFORMATION = 9
_JOB_OBJECT_LIMIT_PROCESS_MEMORY = 0x100

_job_handle: int | None = None


def memory_limit_mb() -> int:
    raw = os.environ.get(MEMORY_ENV, "").strip()
    if not raw:
        return DEFAULT_MEMORY_MB
    value = int(raw)
    if value < 0:
        raise ValueError(f"{MEMORY_ENV} must be >= 0 (0 = off), got {raw!r}")
    return value


if sys.platform == "win32":
    from ctypes import wintypes

    class _BasicLimits(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", ctypes.c_int64),
            ("PerJobUserTimeLimit", ctypes.c_int64),
            ("LimitFlags", wintypes.DWORD),
            ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t),
            ("ActiveProcessLimit", wintypes.DWORD),
            ("Affinity", ctypes.c_size_t),
            ("PriorityClass", wintypes.DWORD),
            ("SchedulingClass", wintypes.DWORD),
        ]

    class _IoCounters(ctypes.Structure):
        _fields_ = [(n, ctypes.c_uint64) for n in (
            "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
            "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]

    class _ExtendedLimits(ctypes.Structure):
        _fields_ = [
            ("BasicLimitInformation", _BasicLimits),
            ("IoInfo", _IoCounters),
            ("ProcessMemoryLimit", ctypes.c_size_t),
            ("JobMemoryLimit", ctypes.c_size_t),
            ("PeakProcessMemoryUsed", ctypes.c_size_t),
            ("PeakJobMemoryUsed", ctypes.c_size_t),
        ]


def apply_memory_limit(limit_mb: int) -> str:
    """Cap this process's committed memory at ``limit_mb``; returns how (for the report)."""
    global _job_handle
    if limit_mb <= 0:
        return "off"
    limit = limit_mb * 1024 * 1024
    if sys.platform == "win32":
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.CreateJobObjectW.restype = wintypes.HANDLE
        k32.GetCurrentProcess.restype = wintypes.HANDLE
        k32.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
        k32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        job = k32.CreateJobObjectW(None, None)
        if not job:
            raise OSError(ctypes.get_last_error(), "CreateJobObjectW failed")
        info = _ExtendedLimits()
        info.BasicLimitInformation.LimitFlags = _JOB_OBJECT_LIMIT_PROCESS_MEMORY
        info.ProcessMemoryLimit = limit
        if not k32.SetInformationJobObject(job, _JOB_OBJECT_EXTENDED_LIMIT_INFORMATION,
                                           ctypes.byref(info), ctypes.sizeof(info)):
            raise OSError(ctypes.get_last_error(), "SetInformationJobObject failed")
        if not k32.AssignProcessToJobObject(job, k32.GetCurrentProcess()):
            raise OSError(ctypes.get_last_error(), "AssignProcessToJobObject failed")
        _job_handle = job  # keep the job alive for the life of the process
        return f"job object, {limit_mb} MB per process"
    import resource

    soft, hard = resource.getrlimit(resource.RLIMIT_DATA)
    if hard != resource.RLIM_INFINITY:
        limit = min(limit, hard)
    if soft == resource.RLIM_INFINITY or soft > limit:
        resource.setrlimit(resource.RLIMIT_DATA, (limit, hard))
    return f"RLIMIT_DATA, {limit // (1024 * 1024)} MB"


def peak_memory_mb() -> int | None:
    """Peak committed memory of this process (Windows job accounting), else None."""
    if sys.platform != "win32" or _job_handle is None:
        return None
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.QueryInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p,
                                              wintypes.DWORD, ctypes.c_void_p]
    info = _ExtendedLimits()
    if not k32.QueryInformationJobObject(_job_handle, _JOB_OBJECT_EXTENDED_LIMIT_INFORMATION,
                                         ctypes.byref(info), ctypes.sizeof(info), None):
        return None
    return int(info.PeakProcessMemoryUsed // (1024 * 1024))


_guard_state: dict[str, Any] = {}


def pytest_configure(config: pytest.Config) -> None:
    try:
        _guard_state["how"] = apply_memory_limit(memory_limit_mb())
    except (OSError, ValueError) as exc:  # never block a run on the guard itself; say so loudly
        _guard_state["how"] = f"NOT APPLIED ({exc})"


@pytest.hookimpl(wrapper=True)
def pytest_make_collect_report(collector: pytest.Collector) -> Any:
    """Module imports count too: a module-level fixture can cost more than any test."""
    before = peak_memory_mb()
    try:
        return (yield)
    finally:
        after = peak_memory_mb()
        if isinstance(collector, pytest.Module) and before is not None and after is not None and after > before:
            _guard_state.setdefault("raised", []).append((after - before, after, f"{collector.nodeid} (import)"))


@pytest.hookimpl(wrapper=True)
def pytest_runtest_protocol(item: pytest.Item, nextitem: pytest.Item | None) -> Any:
    """Record which tests raised the process peak, so the summary can name the heavy ones."""
    before = peak_memory_mb()
    try:
        return (yield)
    finally:
        after = peak_memory_mb()
        if before is not None and after is not None and after > before:
            _guard_state.setdefault("raised", []).append((after - before, after, item.nodeid))


def pytest_terminal_summary(terminalreporter: Any) -> None:
    peak = peak_memory_mb()
    line = f"memory guard: {_guard_state.get('how', 'not configured')}"
    if peak is not None:
        line += f"; peak committed {peak} MB"
    terminalreporter.write_line(line)
    for grew, at, nodeid in sorted(_guard_state.get("raised", []), reverse=True)[:5]:
        if grew >= 50:
            terminalreporter.write_line(f"  peak +{grew} MB (to {at} MB): {nodeid}")
