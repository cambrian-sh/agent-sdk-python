#!/usr/bin/env bash
# Run pytest under a hard memory cap and a wall-clock timeout (Linux/macOS).
#
# The external capped runner AGENTS.md section 4.7 requires for Python tests
# (ENG-2053, owner decision 2026-11-18; the in-repo guard in tests/conftest.py
# is the other half, ENG-1606). CPython 3.13.1 skips the recursion limit for
# calls that rely on default arguments, so one self-calling test helper once
# committed 94 GB and took the host's Docker VM down (ENG-1604).
#
# Usage: scripts/pytest_capped.sh [--memory-mb N] [--timeout-sec N] [--] [pytest args...]
#   defaults: 2048 MB, 120 s; env PYTEST_CAPPED_MEMORY_MB / PYTEST_CAPPED_TIMEOUT_SEC
#   python:   $PYTEST_CAPPED_PYTHON, else <repo>/.venv/bin/python, else python3
#
# The cap: a systemd scope with MemoryMax (cgroup v2: the whole tree is killed at
# the cap) when `systemd-run --user --scope` works here, else an address-space
# limit (ulimit -v) on pytest and everything it starts. Exit codes: pytest's own;
# 124 = timeout; 137 = memory cap.
set -u

mem_mb="${PYTEST_CAPPED_MEMORY_MB:-2048}"
timeout_sec="${PYTEST_CAPPED_TIMEOUT_SEC:-120}"
while [ $# -gt 0 ]; do
  case "$1" in
    --memory-mb) mem_mb="$2"; shift 2 ;;
    --timeout-sec) timeout_sec="$2"; shift 2 ;;
    --) shift; break ;;
    *) break ;;
  esac
done

repo="$(cd "$(dirname "$0")/.." && pwd)"
python="${PYTEST_CAPPED_PYTHON:-}"
if [ -z "$python" ]; then
  if [ -x "$repo/.venv/bin/python" ]; then python="$repo/.venv/bin/python"; else python="python3"; fi
fi

if ! command -v timeout >/dev/null 2>&1; then
  echo "pytest_capped: 'timeout' (coreutils) is required" >&2
  exit 2
fi

echo "pytest_capped: $python -m pytest $* (memory cap ${mem_mb} MB, timeout ${timeout_sec} s)"

mode=ulimit
if command -v systemd-run >/dev/null 2>&1 && systemd-run --user --scope --quiet -p MemoryMax=64M true >/dev/null 2>&1; then
  mode=systemd
fi

start=$(date +%s)
log="$(mktemp)"
trap 'rm -f "$log"' EXIT
if [ "$mode" = systemd ]; then
  timeout --kill-after=5 "$timeout_sec" \
    systemd-run --user --scope --quiet -p MemoryMax="${mem_mb}M" -p MemorySwapMax=0 \
    "$python" -m pytest "$@" 2>&1 | tee "$log"
  code=${PIPESTATUS[0]}
else
  # ulimit -v caps each process's address space: allocations past it fail
  # (MemoryError), so a runaway test dies at the cap instead of the host.
  timeout --kill-after=5 "$timeout_sec" \
    bash -c 'ulimit -v $(( $1 * 1024 )) && shift && exec "$@"' _ "$mem_mb" "$python" -m pytest "$@" 2>&1 | tee "$log"
  code=${PIPESTATUS[0]}
fi

elapsed=$(( $(date +%s) - start ))
# timeout(1) exits 124, or 137 when the tree ignored SIGTERM and needed SIGKILL.
if [ "$code" -eq 124 ] || { [ "$code" -eq 137 ] && [ "$elapsed" -ge "$timeout_sec" ]; }; then
  echo "pytest_capped: KILLED at the timeout (${timeout_sec} s)" >&2
  exit 124
fi
# A tree killed by the cgroup (systemd) exits 137. Under ulimit the cap shows as a
# MemoryError inside pytest, or as a crash (SIGSEGV 139 / SIGABRT 134) when the
# interpreter itself cannot allocate: the CPython 3.13.1 defaulted-arg recursion
# (ENG-1604) ends with 139 at the cap instead of eating the host.
crashed=no
if [ "$mode" = ulimit ] && { [ "$code" -eq 139 ] || [ "$code" -eq 134 ]; }; then crashed=yes; fi
if [ "$code" -eq 137 ] || [ "$crashed" = yes ] || { [ "$code" -ne 0 ] && grep -qE "MemoryError|Cannot allocate memory|std::bad_alloc|out of memory" "$log"; }; then
  echo "pytest_capped: KILLED at the memory cap (${mem_mb} MB, ${mode})" >&2
  exit 137
fi
echo "pytest_capped: pytest exited $code"
exit "$code"
