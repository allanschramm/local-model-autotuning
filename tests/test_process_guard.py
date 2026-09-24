"""Unit + integration tests for the Process Guard (issue #36, ADR 0010).

Dummy children only (sleeping Python processes, no GPU). OS-specific branches
(Linux PDEATHSIG, POSIX process groups) are exercised via mocked module state so
the suite passes on Windows; the real Windows Job Object path is tested for real
when the suite runs on Windows.
"""

import os
import signal
import subprocess
import sys
import time
from unittest.mock import MagicMock

import pytest

import autoresearch.core.process_guard as process_guard

IS_REAL_WINDOWS = os.name == "nt"


class _DummyProc:
    """Fake Popen with scriptable wait()/poll() for teardown-path tests."""

    def __init__(self, pid: int = 4242, alive: bool = True) -> None:
        self.pid = pid
        self._alive = alive
        self.sigs: list[int] = []
        self.waits: list[float | None] = []
        self.kills: list[int] = []

    def poll(self) -> int | None:
        return None if self._alive else 0

    def wait(self, timeout: float | None = None) -> int:
        self.waits.append(timeout)
        if self._alive and len(self.waits) == 1:
            raise subprocess.TimeoutExpired(["cmd"], self.pid)
        self._alive = False
        return 0

    def terminate(self) -> None:
        self.sigs.append(signal.SIGTERM)

    def kill(self) -> None:
        self.sigs.append(process_guard.SIGKILL)
        self._alive = False

    def send_signal(self, sig: int) -> None:
        self.sigs.append(sig)


def _fake_popen(captured: dict):
    def _factory(command, **kwargs) -> _DummyProc:
        captured["command"] = command
        captured["kwargs"] = kwargs
        return _DummyProc()

    return _factory


# ---------------------------------------------------------------------------
# Windows Job Object (real, when running on Windows)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not IS_REAL_WINDOWS, reason="Windows Job Object only exists on Windows")
def test_windows_guard_creates_kill_on_close_job_object():
    guard = process_guard.ProcessGuard()
    try:
        assert guard._job, "expected a Job Object handle"
        assert guard._job != 0
    finally:
        guard.teardown()


@pytest.mark.skipif(not IS_REAL_WINDOWS, reason="Windows Job Object only exists on Windows")
def test_windows_job_object_kills_members_on_handle_close():
    proc = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        creationflags=subprocess.CREATE_NO_WINDOW,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        job = process_guard._create_job_object()
        assigned = process_guard._assign_to_job(job, proc.pid)
        process_guard._close_job_object(job)
        if not assigned:
            pytest.skip("child could not be bound to a job object (parent job constraints)")
        deadline = time.monotonic() + 5.0
        while proc.poll() is None and time.monotonic() < deadline:
            time.sleep(0.05)
        assert proc.poll() is not None, "child survived job object handle close"
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()


@pytest.mark.skipif(not IS_REAL_WINDOWS, reason="Windows Job Object only exists on Windows")
def test_windows_spawn_attach_and_teardown_reaps_child():
    guard = process_guard.ProcessGuard(grace_seconds=0.5)
    proc = guard.spawn(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        creationflags=subprocess.CREATE_NO_WINDOW,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        assert proc in guard._procs
        assert proc.poll() is None
        guard.teardown()
        assert proc.poll() is not None
        assert guard._procs == []
        assert guard._job is None
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()


# ---------------------------------------------------------------------------
# Linux: PR_SET_PDEATHSIG (mocked)
# ---------------------------------------------------------------------------


def test_linux_spawn_merges_pdeathsig_preexec(monkeypatch):
    monkeypatch.setattr(process_guard, "IS_WINDOWS", False)
    monkeypatch.setattr(process_guard, "IS_LINUX", True)
    captured: dict = {}
    monkeypatch.setattr(process_guard.subprocess, "Popen", _fake_popen(captured))
    guard = process_guard.ProcessGuard(grace_seconds=0.1)
    guard.spawn(["llama-server", "--port", "18080"])
    assert captured["command"] == ["llama-server", "--port", "18080"]
    assert captured["kwargs"]["preexec_fn"] is process_guard._linux_child_setup


def test_linux_spawn_respects_caller_session_kwargs(monkeypatch):
    monkeypatch.setattr(process_guard, "IS_WINDOWS", False)
    monkeypatch.setattr(process_guard, "IS_LINUX", True)
    captured: dict = {}
    monkeypatch.setattr(process_guard.subprocess, "Popen", _fake_popen(captured))
    guard = process_guard.ProcessGuard(grace_seconds=0.1)
    guard.spawn(["llama-server"], start_new_session=True)
    assert "preexec_fn" not in captured["kwargs"]
    assert captured["kwargs"]["start_new_session"] is True


def test_linux_spawn_composes_caller_preexec(monkeypatch):
    monkeypatch.setattr(process_guard, "IS_WINDOWS", False)
    monkeypatch.setattr(process_guard, "IS_LINUX", True)
    captured: dict = {}
    monkeypatch.setattr(process_guard.subprocess, "Popen", _fake_popen(captured))
    guard = process_guard.ProcessGuard(grace_seconds=0.1)
    calls: list[str] = []
    monkeypatch.setattr(process_guard, "_linux_child_setup", lambda: calls.append("pdeathsig"))
    guard.spawn(["llama-server"], preexec_fn=lambda: calls.append("caller"))
    preexec = captured["kwargs"]["preexec_fn"]
    assert preexec is not None
    preexec()
    assert calls == ["pdeathsig", "caller"]


def test_linux_child_setup_arms_pdeathsig(monkeypatch):
    monkeypatch.setattr(process_guard, "IS_LINUX", True)
    libc = MagicMock()
    monkeypatch.setattr(process_guard, "_libc", libc)
    monkeypatch.setattr(os, "getppid", lambda: 100)
    kill_calls: list = []
    monkeypatch.setattr(os, "kill", lambda pid, sig: kill_calls.append((pid, sig)))
    process_guard._linux_child_setup()
    libc.prctl.assert_called_once_with(1, process_guard.SIGKILL)
    assert kill_calls == []


def test_linux_child_setup_kills_self_when_parent_already_dead(monkeypatch):
    monkeypatch.setattr(process_guard, "IS_LINUX", True)
    libc = MagicMock()
    monkeypatch.setattr(process_guard, "_libc", libc)
    monkeypatch.setattr(os, "getppid", lambda: 1)
    kill_calls: list = []
    monkeypatch.setattr(os, "kill", lambda pid, sig: kill_calls.append((pid, sig)))
    process_guard._linux_child_setup()
    libc.prctl.assert_called_once_with(1, process_guard.SIGKILL)
    assert kill_calls == [(os.getpid(), process_guard.SIGKILL)]


def test_linux_child_setup_is_noop_off_linux(monkeypatch):
    monkeypatch.setattr(process_guard, "IS_LINUX", False)
    libc = MagicMock()
    monkeypatch.setattr(process_guard, "_libc", libc)
    monkeypatch.setattr(os, "getppid", lambda: 1)
    kill_calls: list = []
    monkeypatch.setattr(os, "kill", lambda pid, sig: kill_calls.append((pid, sig)))
    process_guard._linux_child_setup()
    libc.prctl.assert_not_called()
    assert kill_calls == []


# ---------------------------------------------------------------------------
# macOS / POSIX: process group + killpg (mocked)
# ---------------------------------------------------------------------------


def test_posix_spawn_uses_start_new_session(monkeypatch):
    monkeypatch.setattr(process_guard, "IS_WINDOWS", False)
    monkeypatch.setattr(process_guard, "IS_LINUX", False)
    captured: dict = {}
    monkeypatch.setattr(process_guard.subprocess, "Popen", _fake_popen(captured))
    guard = process_guard.ProcessGuard(grace_seconds=0.1)
    guard.spawn(["sglang.launch_server"])
    assert captured["kwargs"]["start_new_session"] is True


def test_posix_signal_group_signals_when_detached(monkeypatch):
    monkeypatch.setattr(process_guard, "IS_WINDOWS", False)
    monkeypatch.setattr(process_guard, "IS_LINUX", False)
    monkeypatch.setattr(process_guard.os, "getpgid", lambda pid: 999, raising=False)
    monkeypatch.setattr(process_guard.os, "getpgrp", lambda: 123, raising=False)
    killpg_calls: list = []
    monkeypatch.setattr(
        process_guard.os,
        "killpg",
        lambda pgid, sig: killpg_calls.append((pgid, sig)),
        raising=False,
    )
    guard = process_guard.ProcessGuard(grace_seconds=0.1)
    proc = _DummyProc()
    guard._signal(proc, signal.SIGTERM)
    assert killpg_calls == [(999, signal.SIGTERM)]
    assert proc.sigs == []


def test_posix_signal_falls_back_when_shares_parent_group(monkeypatch):
    monkeypatch.setattr(process_guard, "IS_WINDOWS", False)
    monkeypatch.setattr(process_guard, "IS_LINUX", False)
    monkeypatch.setattr(process_guard.os, "getpgid", lambda pid: 123, raising=False)
    monkeypatch.setattr(process_guard.os, "getpgrp", lambda: 123, raising=False)
    killpg_calls: list = []
    monkeypatch.setattr(
        process_guard.os,
        "killpg",
        lambda pgid, sig: killpg_calls.append((pgid, sig)),
        raising=False,
    )
    guard = process_guard.ProcessGuard(grace_seconds=0.1)
    proc = _DummyProc()
    guard._signal(proc, process_guard.SIGKILL)
    assert killpg_calls == []
    assert proc.sigs == [process_guard.SIGKILL]


# ---------------------------------------------------------------------------
# Teardown policy: terminate -> grace -> force kill
# ---------------------------------------------------------------------------


def test_teardown_terminates_then_waits_then_force_kills(monkeypatch):
    monkeypatch.setattr(process_guard, "IS_WINDOWS", False)
    monkeypatch.setattr(process_guard, "IS_LINUX", False)
    guard = process_guard.ProcessGuard(grace_seconds=0.5)
    proc = _DummyProc()
    monkeypatch.setattr(guard, "_signal", lambda child, sig: child.sigs.append(sig))
    guard.attach(proc)
    guard.teardown()
    assert proc.sigs == [signal.SIGTERM, process_guard.SIGKILL]
    assert proc.waits == [0.5, None]
    assert guard._procs == []


def test_teardown_closes_job_object_on_windows(monkeypatch):
    monkeypatch.setattr(process_guard, "IS_WINDOWS", True)
    monkeypatch.setattr(process_guard, "IS_LINUX", False)
    monkeypatch.setattr(process_guard, "_create_job_object", lambda: 777)
    close_calls: list = []
    monkeypatch.setattr(process_guard, "_close_job_object", lambda job: close_calls.append(job))
    guard = process_guard.ProcessGuard(grace_seconds=0.1)
    guard.teardown()
    assert close_calls == [777]
    assert guard._job is None


def test_terminate_and_kill_signal_every_live_proc(monkeypatch):
    monkeypatch.setattr(process_guard, "IS_WINDOWS", False)
    monkeypatch.setattr(process_guard, "IS_LINUX", False)
    guard = process_guard.ProcessGuard(grace_seconds=0.1)
    a, b = _DummyProc(), _DummyProc()
    monkeypatch.setattr(guard, "_signal", lambda child, sig: child.sigs.append(sig))
    guard.attach(a)
    guard.attach(b)
    guard.terminate()
    guard.kill()
    assert a.sigs == [signal.SIGTERM, process_guard.SIGKILL]
    assert b.sigs == [signal.SIGTERM, process_guard.SIGKILL]


# ---------------------------------------------------------------------------
# Attach / spawn bookkeeping
# ---------------------------------------------------------------------------


def test_attach_registers_and_assigns_to_job(monkeypatch):
    monkeypatch.setattr(process_guard, "IS_WINDOWS", True)
    monkeypatch.setattr(process_guard, "IS_LINUX", False)
    monkeypatch.setattr(process_guard, "_create_job_object", lambda: 555)
    assign_calls: list = []
    monkeypatch.setattr(
        process_guard, "_assign_to_job", lambda job, pid: assign_calls.append((job, pid))
    )
    guard = process_guard.ProcessGuard(grace_seconds=0.1)
    proc = _DummyProc()
    guard.attach(proc)
    assert proc in guard._procs
    assert assign_calls == [(555, proc.pid)]


# ---------------------------------------------------------------------------
# Pre-flight orphan sweep (issue #37, ADR 0010 decision 2)
# ---------------------------------------------------------------------------


NETSTAT_SAMPLE = """Active Connections

  Proto  Local Address          Foreign Address        State           PID
  TCP    0.0.0.0:18080          0.0.0.0:0              LISTENING       1234
  TCP    127.0.0.1:28080        0.0.0.0:0              LISTENING       5678
  TCP    0.0.0.0:9100           0.0.0.0:0              LISTENING       1111
  TCP    0.0.0.0:9101           0.0.0.0:0              LISTENING       1111
  TCP    0.0.0.0:9999           0.0.0.0:0              LISTENING       2222
  TCP    0.0.0.0:18080          0.0.0.0:0              TIME_WAIT       3333
  UDP    0.0.0.0:18080          *:*                                   4444
"""


def test_windows_listeners_parse_only_listening_harness_ports(monkeypatch):
    monkeypatch.setattr(process_guard, "IS_WINDOWS", True)
    monkeypatch.setattr(process_guard, "_run_capture", lambda cmd: NETSTAT_SAMPLE)
    assert process_guard._listeners_windows(process_guard.HARNESS_PORTS) == {
        1234,
        5678,
        1111,
    }


def test_posix_listeners_parse(monkeypatch):
    monkeypatch.setattr(process_guard, "IS_WINDOWS", False)
    monkeypatch.setattr(process_guard, "_run_capture", lambda cmd: "1234\n5678\nbad\n0\n")
    assert process_guard._listeners_posix(process_guard.HARNESS_PORTS) == {1234, 5678}


TASKLIST_SAMPLE = (
    '"llama-server.exe","1234","Console","1","200,000 K"\n'
    '"ollama.exe","4321","Console","1","100,000 K"\n'
    '"sglang.launch_server.exe","7777","Console","1","80,000 K"\n'
    '"LM Studio.exe","8888","Console","0","1,000,000 K"\n'
)


def test_windows_processes_match_target_names_only(monkeypatch):
    monkeypatch.setattr(process_guard, "IS_WINDOWS", True)
    monkeypatch.setattr(process_guard, "_run_capture", lambda cmd: TASKLIST_SAMPLE)
    assert process_guard._processes_windows(process_guard.TARGET_PROCESS_NAMES) == {
        1234,
        7777,
    }


def test_posix_processes_match_target_names_only(monkeypatch):
    monkeypatch.setattr(process_guard, "IS_WINDOWS", False)
    ps_sample = "1234 llama-server\n5678 sglang.launch_server\n4321 ollama\n9999 python3\n"
    monkeypatch.setattr(process_guard, "_run_capture", lambda cmd: ps_sample)
    assert process_guard._processes_posix(process_guard.TARGET_PROCESS_NAMES) == {
        1234,
        5678,
    }


def test_name_matching_windows_strips_exe_case_insensitive(monkeypatch):
    monkeypatch.setattr(process_guard, "IS_WINDOWS", True)
    assert process_guard._name_matches("LLAMA-SERVER.EXE", ["llama-server"])
    assert not process_guard._name_matches("LM Studio.exe", ["llama-server"])


def test_name_matching_never_matches_ollama(monkeypatch):
    monkeypatch.setattr(process_guard, "IS_WINDOWS", True)
    assert not process_guard._name_matches("ollama.exe", process_guard.TARGET_PROCESS_NAMES)


def test_pid_exists_true_when_kill0_succeeds(monkeypatch):
    monkeypatch.setattr(process_guard.os, "kill", lambda pid, sig: None)
    assert process_guard._pid_exists(123) is True


def test_pid_exists_false_when_process_gone(monkeypatch):
    def boom(pid, sig):
        raise ProcessLookupError()

    monkeypatch.setattr(process_guard.os, "kill", boom)
    assert process_guard._pid_exists(123) is False


def test_terminate_pid_windows_force_kills_immediately(monkeypatch):
    monkeypatch.setattr(process_guard, "IS_WINDOWS", True)
    kills: list = []
    monkeypatch.setattr(process_guard.os, "kill", lambda pid, sig: kills.append((pid, sig)))
    process_guard._terminate_pid(123, grace_seconds=0.5)
    assert kills == [(123, signal.SIGTERM)]


def test_terminate_pid_posix_graceful_then_force(monkeypatch):
    monkeypatch.setattr(process_guard, "IS_WINDOWS", False)
    monkeypatch.setattr(process_guard, "_pid_exists", lambda pid: True)
    kills: list = []
    monkeypatch.setattr(process_guard.os, "kill", lambda pid, sig: kills.append((pid, sig)))
    times = iter([0.0, 0.1, 1.1])
    monkeypatch.setattr(process_guard.time, "monotonic", lambda: next(times))
    process_guard._terminate_pid(99, grace_seconds=1.0)
    assert kills == [(99, signal.SIGTERM), (99, process_guard.SIGKILL)]


def test_terminate_pid_posix_no_force_after_graceful_exit(monkeypatch):
    monkeypatch.setattr(process_guard, "IS_WINDOWS", False)
    alive = iter([True, True, False])
    monkeypatch.setattr(process_guard, "_pid_exists", lambda pid: next(alive))
    kills: list = []
    monkeypatch.setattr(process_guard.os, "kill", lambda pid, sig: kills.append((pid, sig)))
    process_guard._terminate_pid(7, grace_seconds=0.2)
    assert kills == [(7, signal.SIGTERM)]


def test_cleanup_kills_only_name_plus_port_intersection(monkeypatch):
    monkeypatch.setattr(process_guard, "IS_WINDOWS", True)
    monkeypatch.setattr(process_guard, "listeners_on_ports", lambda ports: {101, 202, 303})
    monkeypatch.setattr(process_guard, "processes_by_name", lambda names: {202, 404})
    killed: list = []
    monkeypatch.setattr(process_guard, "_terminate_pid", lambda pid, g: killed.append(pid))
    assert process_guard.cleanup_leftover_processes() == [202]
    assert killed == [202]


def test_cleanup_leaves_off_filter_processes_untouched(monkeypatch):
    monkeypatch.setattr(process_guard, "IS_WINDOWS", True)
    # 700 is on a harness port but matches no target name (e.g. LM Studio);
    # 800 matches a target name but listens nowhere on a harness port.
    monkeypatch.setattr(process_guard, "listeners_on_ports", lambda ports: {700})
    monkeypatch.setattr(process_guard, "processes_by_name", lambda names: {800})
    killed: list = []
    monkeypatch.setattr(process_guard, "_terminate_pid", lambda pid, g: killed.append(pid))
    assert process_guard.cleanup_leftover_processes() == []
    assert killed == []


def test_cleanup_end_to_end_with_fake_commands(monkeypatch):
    monkeypatch.setattr(process_guard, "IS_WINDOWS", True)
    netstat = (
        "  TCP    0.0.0.0:18080   0.0.0.0:0   LISTENING   1234\n"
        "  TCP    0.0.0.0:28080   0.0.0.0:0   LISTENING   5678\n"
    )
    tasklist = (
        '"llama-server.exe","1234","Console","1","200,000 K"\n'
        '"ollama.exe","9999","Console","1","100,000 K"\n'
    )
    monkeypatch.setattr(
        process_guard,
        "_run_capture",
        lambda cmd: netstat if cmd[0] == "netstat" else tasklist,
    )
    killed: list = []
    monkeypatch.setattr(process_guard, "_terminate_pid", lambda pid, g: killed.append(pid))
    result = process_guard.cleanup_leftover_processes([18080, 28080], ["llama-server"])
    assert result == [1234]
    assert killed == [1234]
