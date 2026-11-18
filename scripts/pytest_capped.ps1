<#
.SYNOPSIS
  Run pytest under a hard memory cap and a wall-clock timeout (Windows).

.DESCRIPTION
  The external capped runner AGENTS.md section 4.7 requires for Python tests (ENG-2053,
  owner decision 2026-11-18; the in-repo guard in tests/conftest.py is the other
  half, ENG-1606). CPython 3.13.1 skips the recursion limit for calls that rely
  on default arguments, so one self-calling test helper once committed 94 GB and
  took the host's Docker VM down (ENG-1604).

  pytest runs inside a Windows Job Object: the job's committed memory is capped
  (allocations past it fail), and the whole process tree is killed when the job
  reaches the cap or the timeout. Children (xdist workers, subprocesses) are in
  the same job. Exit codes: pytest's own; 124 = timeout; 137 = memory cap.

.EXAMPLE
  powershell -NoProfile -File scripts/pytest_capped.ps1 tests/test_tools.py -q
  powershell -NoProfile -File scripts/pytest_capped.ps1 --memory-mb 1024 --timeout-sec 60 -k slow

.NOTES
  Limits: --memory-mb N / --timeout-sec N, or $env:PYTEST_CAPPED_MEMORY_MB /
  $env:PYTEST_CAPPED_TIMEOUT_SEC. Python: $env:PYTEST_CAPPED_PYTHON, else <repo>/.venv/Scripts/python.exe, else
  `python` on PATH. Works on Windows PowerShell 5.1 and PowerShell 7.
#>
# No param block on purpose: PowerShell binds abbreviated parameter names, so a
# declared -MemoryMB would swallow pytest's -m. Every argument lands in $args;
# only --memory-mb N and --timeout-sec N (as in pytest_capped.sh) are ours, the
# rest goes to pytest unchanged.
$MemoryMB = 2048
$TimeoutSec = 120
if ($env:PYTEST_CAPPED_MEMORY_MB) { $MemoryMB = [int]$env:PYTEST_CAPPED_MEMORY_MB }
if ($env:PYTEST_CAPPED_TIMEOUT_SEC) { $TimeoutSec = [int]$env:PYTEST_CAPPED_TIMEOUT_SEC }
$PytestArgs = @()
$rest = @($args)
for ($i = 0; $i -lt $rest.Count; $i++) {
    $a = [string]$rest[$i]
    if ($a -eq '--memory-mb' -and $i + 1 -lt $rest.Count) { $MemoryMB = [int]$rest[$i + 1]; $i++; continue }
    if ($a -eq '--timeout-sec' -and $i + 1 -lt $rest.Count) { $TimeoutSec = [int]$rest[$i + 1]; $i++; continue }
    $PytestArgs += $a
}

$ErrorActionPreference = 'Stop'

if (-not ('PytestCapped.Job' -as [type])) {
    Add-Type -TypeDefinition @'
using System;
using System.Runtime.InteropServices;

namespace PytestCapped {
    public static class Job {
        [StructLayout(LayoutKind.Sequential)]
        struct BASIC_LIMIT {
            public long PerProcessUserTimeLimit;
            public long PerJobUserTimeLimit;
            public uint LimitFlags;
            public UIntPtr MinimumWorkingSetSize;
            public UIntPtr MaximumWorkingSetSize;
            public uint ActiveProcessLimit;
            public UIntPtr Affinity;
            public uint PriorityClass;
            public uint SchedulingClass;
        }
        [StructLayout(LayoutKind.Sequential)]
        struct IO_COUNTERS {
            public ulong ReadOperationCount, WriteOperationCount, OtherOperationCount;
            public ulong ReadTransferCount, WriteTransferCount, OtherTransferCount;
        }
        [StructLayout(LayoutKind.Sequential)]
        struct EXTENDED_LIMIT {
            public BASIC_LIMIT BasicLimitInformation;
            public IO_COUNTERS IoInfo;
            public UIntPtr ProcessMemoryLimit;
            public UIntPtr JobMemoryLimit;
            public UIntPtr PeakProcessMemoryUsed;
            public UIntPtr PeakJobMemoryUsed;
        }
        const int ExtendedLimitInformation = 9;
        const uint LIMIT_JOB_MEMORY = 0x200;
        const uint LIMIT_KILL_ON_JOB_CLOSE = 0x2000;

        [DllImport("kernel32.dll", SetLastError = true, CharSet = CharSet.Unicode)]
        static extern IntPtr CreateJobObject(IntPtr attrs, string name);
        [DllImport("kernel32.dll", SetLastError = true)]
        static extern bool SetInformationJobObject(IntPtr job, int cls, ref EXTENDED_LIMIT info, uint len);
        [DllImport("kernel32.dll", SetLastError = true)]
        static extern bool QueryInformationJobObject(IntPtr job, int cls, out EXTENDED_LIMIT info, uint len, IntPtr ret);
        [DllImport("kernel32.dll", SetLastError = true)]
        static extern bool AssignProcessToJobObject(IntPtr job, IntPtr process);
        [DllImport("kernel32.dll", SetLastError = true)]
        static extern bool TerminateJobObject(IntPtr job, uint code);

        public static IntPtr Create(long limitBytes) {
            IntPtr job = CreateJobObject(IntPtr.Zero, null);
            if (job == IntPtr.Zero) throw new System.ComponentModel.Win32Exception();
            var info = new EXTENDED_LIMIT();
            info.BasicLimitInformation.LimitFlags = LIMIT_JOB_MEMORY | LIMIT_KILL_ON_JOB_CLOSE;
            info.JobMemoryLimit = new UIntPtr((ulong)limitBytes);
            if (!SetInformationJobObject(job, ExtendedLimitInformation, ref info, (uint)Marshal.SizeOf(typeof(EXTENDED_LIMIT))))
                throw new System.ComponentModel.Win32Exception();
            return job;
        }
        public static void Assign(IntPtr job, IntPtr process) {
            if (!AssignProcessToJobObject(job, process)) throw new System.ComponentModel.Win32Exception();
        }
        public static long Peak(IntPtr job) {
            EXTENDED_LIMIT info;
            if (!QueryInformationJobObject(job, ExtendedLimitInformation, out info, (uint)Marshal.SizeOf(typeof(EXTENDED_LIMIT)), IntPtr.Zero))
                return -1;
            return (long)info.PeakJobMemoryUsed.ToUInt64();
        }
        public static void Kill(IntPtr job, uint code) { TerminateJobObject(job, code); }
    }
}
'@
}

$repo = Split-Path -Parent $PSScriptRoot
$python = $env:PYTEST_CAPPED_PYTHON
if (-not $python) {
    $venv = Join-Path $repo '.venv\Scripts\python.exe'
    if (Test-Path $venv) { $python = $venv } else { $python = 'python' }
}

$pyArgs = @('-m', 'pytest') + @($PytestArgs | Where-Object { $_ -ne '--' })
$quoted = ($pyArgs | ForEach-Object {
    if ($_ -match '[\s"]') { '"' + ($_ -replace '"', '\"') + '"' } else { $_ }
}) -join ' '

$limit = [long]$MemoryMB * 1MB
$job = [PytestCapped.Job]::Create($limit)

$psi = New-Object System.Diagnostics.ProcessStartInfo
$psi.FileName = $python
$psi.Arguments = $quoted
$psi.UseShellExecute = $false
$psi.WorkingDirectory = (Get-Location).Path
$proc = [System.Diagnostics.Process]::Start($psi)
[PytestCapped.Job]::Assign($job, $proc.Handle)

Write-Host "pytest_capped: $python $quoted  (memory cap ${MemoryMB} MB, timeout ${TimeoutSec} s)"

$deadline = (Get-Date).AddSeconds($TimeoutSec)
$verdict = ''
while (-not $proc.HasExited) {
    if ([PytestCapped.Job]::Peak($job) -ge ($limit - 16MB)) { $verdict = 'memory'; break }
    if ((Get-Date) -ge $deadline) { $verdict = 'timeout'; break }
    Start-Sleep -Milliseconds 100
}
if (-not $verdict -and [PytestCapped.Job]::Peak($job) -ge ($limit - 16MB)) { $verdict = 'memory' }

$peakMB = [math]::Round([PytestCapped.Job]::Peak($job) / 1MB)
switch ($verdict) {
    'memory' {
        [PytestCapped.Job]::Kill($job, 137)
        $proc.WaitForExit()
        Write-Host "pytest_capped: KILLED at the memory cap (${MemoryMB} MB; peak ${peakMB} MB)" -ForegroundColor Red
        exit 137
    }
    'timeout' {
        [PytestCapped.Job]::Kill($job, 124)
        $proc.WaitForExit()
        Write-Host "pytest_capped: KILLED at the timeout (${TimeoutSec} s; peak ${peakMB} MB)" -ForegroundColor Red
        exit 124
    }
}
$proc.WaitForExit()
$code = $proc.ExitCode
Write-Host "pytest_capped: pytest exited $code (peak ${peakMB} MB)"
exit $code
