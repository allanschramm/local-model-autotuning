"""Process Guard: OS-native parent lifecycle bind + graceful-to-force teardown.

Cross-platform zombie-process prevention for harness subprocesses (ADR 0010,
CONTEXT.md Glossary: Process Guard). Binds spawned children so an abrupt parent
exit cannot orphan them, and tears them down explicitly:

- Windows: children are assigned to a Job Object with
  ``JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE`` — the kernel kills every job member
  the moment the owning handle dies with the parent Python process.
- Linux: ``PR_SET_PDEATHSIG`` via ctypes so the kernel SIGKILLs the child when
  the parent dies.
- macOS/POSIX: children spawn in a detached process group
  (``start_new_session``) and are group-signalled on teardown / atexit.

Teardown policy: ``terminate()`` (SIGTERM / TerminateProcess), wait a grace
period (default 2.0 s, matching ADR 0010), then force ``kill()`` (SIGKILL /
TerminateProcess).

Pre-flight orphan sweep (issue #37, ADR 0010 decision 2): before a Trial server
starts, ``cleanup_leftover_processes()`` force-kills leftover processes that
BOTH listen on a harness port AND match a target process name. LM Studio,
Ollama, and off-port user servers stay outside the name+port filter.

Wired into the runners (LlamaServerRunner / SGLangServerRunner / ServiceManager)
and the single-load gate (single_load.py, issue #41): the pre-flight sweep runs
before a Trial server binds, and spawned servers are guarded.

Example
-------
    guard = ProcessGuard()
    proc = guard.spawn(["llama-server", "--port", "18080"])
    ...
    guard.teardown()

    killed = cleanup_leftover_processes()   # pre-flight orphan sweep
"""

from __future__ import annotations

import atexit
import csv
import os
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Sequence
from typing import Any

IS_WINDOWS = os.name == "nt"
IS_LINUX = sys.platform == "linux"

# SIGKILL is not defined on Windows; keep the POSIX value (9) as the canonical
# "force kill" signal so teardown logic stays platform-independent.
SIGKILL = getattr(signal, "SIGKILL", 9)


def spawn_kwargs(**extra: Any) -> dict[str, Any]:
    """Popen/run kwargs that keep every harness-spawned child headless.

    Windows allocates a console window for each child unless told otherwise, so
    a coding Trial that runs 40 test subprocesses (or the Claw mock services)
    pops 40 console windows and steals focus on a desktop rig. Every spawn in
    this repo must go through here or pass ``CREATE_NO_WINDOW`` itself.
    """
    kwargs = dict(extra)
    if IS_WINDOWS:
        kwargs.setdefault("creationflags", subprocess.CREATE_NO_WINDOW)
    return kwargs


# Default grace period between graceful terminate() and force kill().
TEARDOWN_GRACE_SECONDS = 2.0

# Linux prctl PR_SET_PDEATHSIG.
_PR_SET_PDEATHSIG = 1

# Win32 Job Object flags / access rights.
_JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
_JOB_OBJECT_EXTENDED_LIMIT_INFORMATION = 9
_PROCESS_SET_QUOTA = 0x0100
_PROCESS_TERMINATE = 0x0001

if IS_WINDOWS:
    import ctypes
    from ctypes import wintypes

    _kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

    class _JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", wintypes.LARGE_INTEGER),
            ("PerJobUserTimeLimit", wintypes.LARGE_INTEGER),
            ("LimitFlags", wintypes.DWORD),
            ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t),
            ("ActiveProcessLimit", wintypes.DWORD),
            ("Affinity", ctypes.c_size_t),
            ("PriorityClass", wintypes.DWORD),
            ("SchedulingClass", wintypes.DWORD),
        ]

    class _IO_COUNTERS(ctypes.Structure):
        _fields_ = [
            ("ReadOperationCount", ctypes.c_ulonglong),
            ("WriteOperationCount", ctypes.c_ulonglong),
            ("OtherOperationCount", ctypes.c_ulonglong),
            ("ReadTransferCount", ctypes.c_ulonglong),
            ("WriteTransferCount", ctypes.c_ulonglong),
            ("OtherTransferCount", ctypes.c_ulonglong),
        ]

    class _JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
        _fields_ = [
            ("BasicLimitInformation", _JOBOBJECT_BASIC_LIMIT_INFORMATION),
            ("IoInfo", _IO_COUNTERS),
            ("ProcessMemoryLimit", ctypes.c_size_t),
            ("JobMemoryLimit", ctypes.c_size_t),
            ("PeakProcessMemoryUsed", ctypes.c_size_t),
            ("PeakJobMemoryUsed", ctypes.c_size_t),
        ]

    _kernel32.CreateJobObjectW.restype = wintypes.HANDLE
    _kernel32.CreateJobObjectW.argtypes = [wintypes.LPCWSTR, wintypes.LPCWSTR]
    _kernel32.CloseHandle.restype = wintypes.BOOL
    _kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    _kernel32.SetInformationJobObject.restype = wintypes.BOOL
    _kernel32.SetInformationJobObject.argtypes = [
        wintypes.HANDLE,
        ctypes.c_int,
        wintypes.LPVOID,
        wintypes.DWORD,
    ]
    _kernel32.OpenProcess.restype = wintypes.HANDLE
    _kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    _kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
    _kernel32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]

if IS_LINUX:
    import ctypes as _ctypes

    _libc = _ctypes.CDLL(None, use_errno=True)
else:
    _libc = None


def _create_job_object() -> int:
    """Create a Job Object that kills its members when its handle closes."""
    job = _kernel32.CreateJobObjectW(None, None)
    if not job:
        raise ctypes.WinError(ctypes.get_last_error())
    info = _JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
    info.BasicLimitInformation.LimitFlags = _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    if not _kernel32.SetInformationJobObject(
        job,
        _JOB_OBJECT_EXTENDED_LIMIT_INFORMATION,
        ctypes.byref(info),
        ctypes.sizeof(info),
    ):
        err = ctypes.get_last_error()
        _kernel32.CloseHandle(job)
        raise ctypes.WinError(err)
    return job


def _assign_to_job(job: int, pid: int) -> bool:
    """Best-effort bind of a live process into a Job Object (never raises).

    Returns True when the process was bound. Assignment can fail legitimately —
    e.g. the child already belongs to a parent job — so failures are ignored:
    the OS bind is best-effort and the explicit teardown API still applies.
    """
    handle = _kernel32.OpenProcess(_PROCESS_SET_QUOTA | _PROCESS_TERMINATE, False, pid)
    if not handle:
        return False
    try:
        ok = _kernel32.AssignProcessToJobObject(job, handle)
    finally:
        _kernel32.CloseHandle(handle)
    return bool(ok)


def _close_job_object(job: int) -> None:
    """Close the Job Object handle; with KILL_ON_JOB_CLOSE this kills members."""
    _kernel32.CloseHandle(job)


def _linux_child_setup() -> None:
    """Child-side preexec: SIGKILL this process when the parent dies.

    Also closes the parent-death race window: if the parent died between
    ``fork`` and ``prctl``, ``getppid()`` has already collapsed to the reaper
    (1) — kill ourselves instead of outliving the parent.
    """
    if not IS_LINUX or _libc is None:
        return
    _libc.prctl(_PR_SET_PDEATHSIG, SIGKILL)
    if os.getppid() == 1:
        os.kill(os.getpid(), SIGKILL)


def _os_spawn_kwargs(given: dict[str, Any]) -> dict[str, Any]:
    """OS bind Popen kwargs, unless the caller already chose a session mode."""
    if IS_WINDOWS:
        return {}
    if {"process_group", "start_new_session"} & given.keys():
        return {}
    if IS_LINUX:
        caller = given.get("preexec_fn")
        if caller is None:
            return {"preexec_fn": _linux_child_setup}

        # Compose PDEATHSIG with any caller-supplied preexec (e.g. SGLang's
        # os.setsid) so no spawn path loses the parent-death bind.
        def _composed() -> None:
            _linux_child_setup()
            caller()

        return {"preexec_fn": _composed}

    # Non-Linux POSIX (macOS): no PDEATHSIG. Respect an explicit caller preexec
    # (e.g. os.setsid) — pairing it with start_new_session would double-setsid.
    if "preexec_fn" in given:
        return {}
    return {"start_new_session": True}


class ProcessGuard:
    """Binds subprocesses to the parent lifecycle and tears them down.

    The OS-native bind (Job Object / PDEATHSIG / detached process group)
    protects against an abrupt parent exit; ``teardown()`` plus an atexit hook
    covers the graceful path with terminate -> grace -> force kill.
    """

    def __init__(self, grace_seconds: float = TEARDOWN_GRACE_SECONDS) -> None:
        self.grace_seconds = grace_seconds
        self._procs: list[subprocess.Popen] = []
        self._lock = threading.Lock()
        self._job: int | None = _create_job_object() if IS_WINDOWS else None
        atexit.register(self.teardown)

    # -- public API -------------------------------------------------------

    def spawn(self, command: Sequence[str], **popen_kwargs: Any) -> subprocess.Popen:
        """Launch a guarded subprocess (OS bind kwargs merged in)."""
        kwargs = spawn_kwargs(**popen_kwargs)
        kwargs.update(_os_spawn_kwargs(kwargs))
        proc = subprocess.Popen(command, **kwargs)
        return self.attach(proc)

    def attach(self, proc: subprocess.Popen) -> subprocess.Popen:
        """Register an already-running subprocess with the guard."""
        with self._lock:
            # Completed children no longer need lifecycle tracking. Prune them
            # here so long-lived benchmark guards do not retain every Trial.
            self._procs[:] = [child for child in self._procs if child.poll() is None]
            self._procs.append(proc)
            if self._job:
                _assign_to_job(self._job, proc.pid)
        return proc

    def terminate(self) -> None:
        """Send a graceful stop signal to every guarded process."""
        self._signal_all(signal.SIGTERM)

    def kill(self) -> None:
        """Force-kill every guarded process."""
        self._signal_all(SIGKILL)

    def teardown(self) -> None:
        """terminate() -> grace wait -> force kill, then release OS binds.

        Idempotent: safe to call explicitly and again from atexit.
        """
        for proc in self._snapshot():
            self._teardown_proc(proc)
        if self._job:
            # Keep teardown safe when platform flags are mocked in tests or
            # when a guard instance crosses a platform-specific code path.
            if IS_WINDOWS:
                _close_job_object(self._job)
            self._job = None
        with self._lock:
            self._procs.clear()
        # Do not retain one atexit callback per completed Trial.
        atexit.unregister(self.teardown)

    # -- internals --------------------------------------------------------

    def _snapshot(self) -> list[subprocess.Popen]:
        with self._lock:
            # Keep lifecycle scans proportional to live children only.
            self._procs[:] = [proc for proc in self._procs if proc.poll() is None]
            return list(self._procs)

    def _signal_all(self, sig: int) -> None:
        for proc in self._snapshot():
            if proc.poll() is None:
                self._signal(proc, sig)

    def _teardown_proc(self, proc: subprocess.Popen) -> None:
        if proc.poll() is not None:
            return
        self._signal(proc, signal.SIGTERM)
        try:
            proc.wait(timeout=self.grace_seconds)
        except subprocess.TimeoutExpired:
            self._signal(proc, SIGKILL)
            proc.wait()

    def _signal(self, proc: subprocess.Popen, sig: int) -> None:
        if IS_WINDOWS:
            if sig == SIGKILL:
                proc.kill()
            else:
                proc.terminate()
            return
        # POSIX: group-signal only when the child detached into its own process
        # group — never our own group, which would kill us too.
        try:
            pgid = os.getpgid(proc.pid)
        except ProcessLookupError:
            return
        if pgid != os.getpgrp():
            try:
                os.killpg(pgid, sig)
                return
            except ProcessLookupError:
                return
        try:
            proc.send_signal(sig)
        except ProcessLookupError:
            pass


# ---------------------------------------------------------------------------
# Pre-flight orphan sweep (issue #37, ADR 0010 decision 2)
# ---------------------------------------------------------------------------

# TCP ports the harness uses for its servers; an orphan holding one of these
# would otherwise surface as EADDRINUSE on the next Trial.
HARNESS_PORTS: tuple[int, ...] = (18080, 28080, *range(9100, 9115))

# Process name targets for the sweep. Substring match, so ".exe" variants,
# "sglang.launch_server", etc. are covered; "ollama"/"LM Studio" never match.
TARGET_PROCESS_NAMES: tuple[str, ...] = (
    "llama-server",
    "llama-cli",
    "llama-bench",
    "llama-perplexity",
    "sglang",
)


def _run_capture(command: Sequence[str]) -> str:
    """Run a process and capture stdout; returns "" when it cannot run."""
    kwargs: dict[str, Any] = {"capture_output": True, "text": True, "check": False}
    if IS_WINDOWS:
        kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
    try:
        completed = subprocess.run(list(command), **kwargs)
    except OSError:
        return ""
    return completed.stdout or ""


def listeners_on_ports(ports: Sequence[int]) -> set[int]:
    """PIDs of processes with an active TCP listener on any of ``ports``."""
    if IS_WINDOWS:
        return _listeners_windows(ports)
    return _listeners_posix(ports)


def _listeners_windows(ports: Sequence[int]) -> set[int]:
    port_set = set(ports)
    out = _run_capture(["netstat", "-ano", "-p", "tcp"])
    pids: set[int] = set()
    for line in out.splitlines():
        tokens = line.split()
        if len(tokens) < 5 or tokens[0] != "TCP" or tokens[3] != "LISTENING":
            continue
        port = tokens[1].rsplit(":", 1)[-1]
        if port.isdigit() and int(port) in port_set:
            pid = tokens[4]
            if pid.isdigit() and int(pid) > 0:
                pids.add(int(pid))
    return pids


def _listeners_posix(ports: Sequence[int]) -> set[int]:
    if not ports:
        return set()
    lsof_args: list[str] = [f"-iTCP:{port}" for port in ports]
    out = _run_capture(["lsof", "-nP", "-sTCP:LISTEN", "-t", *lsof_args])
    pids: set[int] = set()
    for line in out.splitlines():
        line = line.strip()
        if line.isdigit() and int(line) > 0:
            pids.add(int(line))
    return pids


def processes_by_name(names: Sequence[str]) -> set[int]:
    """PIDs of running processes whose name matches any of ``names``."""
    if IS_WINDOWS:
        return _processes_windows(names)
    return _processes_posix(names)


def _normalize_name(name: str) -> str:
    base = os.path.basename(name)
    if IS_WINDOWS:
        base = base[:-4] if base.lower().endswith(".exe") else base
        return base.lower()
    return base


def _name_matches(name: str, targets: Sequence[str]) -> bool:
    normalized = _normalize_name(name)
    return any(target in normalized for target in targets)


def _processes_windows(names: Sequence[str]) -> set[int]:
    out = _run_capture(["tasklist", "/FO", "CSV", "/NH"])
    pids: set[int] = set()
    for row in csv.reader(out.splitlines()):
        if len(row) < 2:
            continue
        image, pid = row[0], row[1]
        if pid.isdigit() and _name_matches(image, names):
            pids.add(int(pid))
    return pids


def _processes_posix(names: Sequence[str]) -> set[int]:
    out = _run_capture(["ps", "-eo", "pid=,comm="])
    pids: set[int] = set()
    for line in out.splitlines():
        parts = line.strip().split(None, 1)
        if len(parts) != 2 or not parts[0].isdigit():
            continue
        pid, comm = int(parts[0]), parts[1]
        if pid > 0 and _name_matches(comm, names):
            pids.add(pid)
    return pids


def _pid_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _terminate_pid(pid: int, grace_seconds: float) -> None:
    """terminate -> grace -> force kill an unowned PID (best-effort).

    POSIX: SIGTERM, wait up to ``grace_seconds``, then SIGKILL if still alive.
    Windows: ``os.kill(pid, SIGTERM)`` is TerminateProcess (immediate force),
    so there is no graceful window for a raw PID there.
    """
    if IS_WINDOWS:
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            pass
        return
    if not _pid_exists(pid):
        return
    try:
        os.kill(pid, signal.SIGTERM)
    except OSError:
        return
    deadline = time.monotonic() + grace_seconds
    while time.monotonic() < deadline:
        if not _pid_exists(pid):
            return
        time.sleep(0.05)
    try:
        os.kill(pid, SIGKILL)
    except OSError:
        pass


def cleanup_leftover_processes(
    ports: Sequence[int] = HARNESS_PORTS,
    process_names: Sequence[str] = TARGET_PROCESS_NAMES,
    grace_seconds: float = TEARDOWN_GRACE_SECONDS,
) -> list[int]:
    """Force-kill leftover harness processes bound to a harness port.

    Before a Trial server binds a harness port, orphans from an earlier run may
    still hold the port / VRAM. Only processes that BOTH listen on a harness
    port AND match a target process name are killed; anything else — LM Studio,
    Ollama, or a user's llama server on a non-harness port — is left alone.

    Returns the PIDs killed (empty when nothing matched).
    """
    targets = sorted(listeners_on_ports(ports) & processes_by_name(process_names))
    for pid in targets:
        _terminate_pid(pid, grace_seconds)
    return targets
