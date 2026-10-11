# tests/scripts/test_restart_exonit_server.py
"""Tests for scripts/restart_exonit_server.py (incident 2026-08-15).

Everything injected — no real subprocess/os.kill/network calls. Mirrors the
injection style already used by scripts/run_health_check.py callers and
autonomy/tests/test_supervisor_armed.py (run_cmd/urlopen fakes).
"""
from __future__ import annotations

import io
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

from scripts.restart_exonit_server import _capture_forensics, restart_server


def _proc(stdout="", rc=0):
    return SimpleNamespace(returncode=rc, stdout=stdout, stderr="")


class _HealthResp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _write_cgroup_procs(tmp_path: Path, pids: list[int]) -> Path:
    p = tmp_path / "cgroup.procs"
    p.write_text("\n".join(str(pid) for pid in pids) + ("\n" if pids else ""))
    return p


def _missing_pyspy(tmp_path: Path) -> str:
    """A `pyspy_bin` path that does not exist -- forensics capture short-circuits
    on the very first check, so existing kill/health behaviour tests below stay
    untouched by the new forensics step (and don't write into the real repo's
    `data/diagnostics/` while running)."""
    return str(tmp_path / "no-py-spy-here")


def _fake_pyspy(tmp_path: Path) -> str:
    """A `pyspy_bin` path that does exist, to exercise the forensics capture path."""
    p = tmp_path / "fake-py-spy"
    p.write_text("#!/bin/sh\n")
    return str(p)


def test_restart_kills_all_cgroup_pids_including_main_pid(tmp_path):
    """MainPID (from systemctl show) + every pid in cgroup.procs get SIGKILL,
    even if they don't overlap — the incident's whole point was that killing
    only MainPID left hung children behind."""
    cgroup_path = _write_cgroup_procs(tmp_path, [111, 222])
    run_cmd = MagicMock(return_value=_proc(stdout="999\n"))
    kill_fn = MagicMock()
    urlopen = MagicMock(return_value=_HealthResp(b"ok"))
    sleep_fn = MagicMock()
    monotonic = MagicMock(side_effect=[0.0, 0.0])

    rc = restart_server(
        run_cmd=run_cmd,
        sleep_fn=sleep_fn,
        monotonic=monotonic,
        urlopen=urlopen,
        kill_fn=kill_fn,
        cgroup_procs_path=cgroup_path,
        pyspy_bin=_missing_pyspy(tmp_path),
        timeout_s=240,
        print_fn=lambda *_: None,
    )

    assert rc == 0
    killed_pids = {c.args[0] for c in kill_fn.call_args_list}
    assert killed_pids == {111, 222, 999}
    sleep_fn.assert_not_called()  # first health poll succeeded, no waiting needed


def test_restart_kill_ignores_missing_and_permission_denied_pids(tmp_path):
    """A pid that already exited (ProcessLookupError) or one we can't kill
    (PermissionError) must not stop the rest from being killed."""
    cgroup_path = _write_cgroup_procs(tmp_path, [111, 222, 333])
    run_cmd = MagicMock(return_value=_proc(stdout=""))  # no MainPID readable

    def kill_fn(pid, sig):
        if pid == 111:
            raise ProcessLookupError()
        if pid == 222:
            raise PermissionError()

    urlopen = MagicMock(return_value=_HealthResp(b"ok"))

    rc = restart_server(
        run_cmd=run_cmd,
        sleep_fn=MagicMock(),
        monotonic=MagicMock(side_effect=[0.0, 0.0]),
        urlopen=urlopen,
        kill_fn=kill_fn,
        cgroup_procs_path=cgroup_path,
        pyspy_bin=_missing_pyspy(tmp_path),
        print_fn=lambda *_: None,
    )

    assert rc == 0  # health responded despite the two problem pids


def test_restart_returns_1_when_health_never_responds(tmp_path):
    cgroup_path = _write_cgroup_procs(tmp_path, [111])
    run_cmd = MagicMock(return_value=_proc(stdout="111\n"))
    kill_fn = MagicMock()
    urlopen = MagicMock(side_effect=OSError("connection refused"))
    sleep_fn = MagicMock()

    # monotonic: deadline computed at t=0 (timeout_s=10) -> deadline=10.
    # Loop checks: 0 (start, <10, sleeps), 12 (>=10, exit loop).
    monotonic = MagicMock(side_effect=[0.0, 12.0])

    rc = restart_server(
        run_cmd=run_cmd,
        sleep_fn=sleep_fn,
        monotonic=monotonic,
        urlopen=urlopen,
        kill_fn=kill_fn,
        cgroup_procs_path=cgroup_path,
        pyspy_bin=_missing_pyspy(tmp_path),
        timeout_s=10,
        print_fn=lambda *_: None,
    )

    assert rc == 1
    urlopen.assert_called()


def test_dry_run_does_not_kill_anything_and_returns_0(tmp_path):
    cgroup_path = _write_cgroup_procs(tmp_path, [111, 222])
    run_cmd = MagicMock(return_value=_proc(stdout="999\n"))
    kill_fn = MagicMock()
    urlopen = MagicMock()

    messages = []
    rc = restart_server(
        run_cmd=run_cmd,
        kill_fn=kill_fn,
        urlopen=urlopen,
        cgroup_procs_path=cgroup_path,
        dry_run=True,
        print_fn=messages.append,
    )

    assert rc == 0
    kill_fn.assert_not_called()
    urlopen.assert_not_called()
    assert any("999" in m and "111" in m and "222" in m for m in messages)


def test_missing_cgroup_procs_file_falls_back_to_main_pid_only(tmp_path):
    """cgroup.procs absent (service already gone / path unavailable) is
    non-fatal — falls back to just MainPID, or empty if that's also gone."""
    missing_path = tmp_path / "does-not-exist" / "cgroup.procs"
    run_cmd = MagicMock(return_value=_proc(stdout="42\n"))
    kill_fn = MagicMock()
    urlopen = MagicMock(return_value=_HealthResp(b"ok"))

    rc = restart_server(
        run_cmd=run_cmd,
        sleep_fn=MagicMock(),
        monotonic=MagicMock(side_effect=[0.0, 0.0]),
        urlopen=urlopen,
        kill_fn=kill_fn,
        cgroup_procs_path=missing_path,
        pyspy_bin=_missing_pyspy(tmp_path),
        print_fn=lambda *_: None,
    )

    assert rc == 0
    killed_pids = {c.args[0] for c in kill_fn.call_args_list}
    assert killed_pids == {42}


def test_dry_run_never_calls_pyspy_capture(tmp_path):
    """--dry-run must not touch py-spy at all, even if the binary exists."""
    cgroup_path = _write_cgroup_procs(tmp_path, [111])
    run_cmd = MagicMock(return_value=_proc(stdout="999\n"))

    rc = restart_server(
        run_cmd=run_cmd,
        kill_fn=MagicMock(),
        urlopen=MagicMock(),
        cgroup_procs_path=cgroup_path,
        pyspy_bin=_fake_pyspy(tmp_path),
        diagnostics_root=tmp_path / "diagnostics",
        dry_run=True,
        print_fn=lambda *_: None,
    )

    assert rc == 0
    # Only the systemctl show call for MainPID -- no py-spy dump invocation.
    for call in run_cmd.call_args_list:
        assert "py-spy" not in call.args[0][0]
    assert not (tmp_path / "diagnostics").exists()


def test_skip_forensics_flag_never_calls_pyspy_capture(tmp_path):
    """--skip-forensics must not capture, even if py-spy exists and pids exist."""
    cgroup_path = _write_cgroup_procs(tmp_path, [111])
    run_cmd = MagicMock(return_value=_proc(stdout=""))
    urlopen = MagicMock(return_value=_HealthResp(b"ok"))

    rc = restart_server(
        run_cmd=run_cmd,
        sleep_fn=MagicMock(),
        monotonic=MagicMock(side_effect=[0.0, 0.0]),
        urlopen=urlopen,
        kill_fn=MagicMock(),
        cgroup_procs_path=cgroup_path,
        pyspy_bin=_fake_pyspy(tmp_path),
        diagnostics_root=tmp_path / "diagnostics",
        skip_forensics=True,
        print_fn=lambda *_: None,
    )

    assert rc == 0
    for call in run_cmd.call_args_list:
        assert "py-spy" not in call.args[0][0]
    assert not (tmp_path / "diagnostics").exists()


def test_capture_forensics_dumps_each_pid_via_injected_run_cmd(tmp_path):
    """The forensics step calls the injected run_cmd with `py-spy dump --pid
    <pid>` for every pid and writes stdout+stderr to a per-pid file under the
    timestamped diagnostics dir."""
    pyspy_bin = _fake_pyspy(tmp_path)
    diagnostics_root = tmp_path / "diagnostics"

    def run_cmd(cmd, **kwargs):
        pid = cmd[cmd.index("--pid") + 1]
        return _proc(stdout=f"stack for {pid}\n")

    messages = []
    _capture_forensics(
        [111, 222],
        run_cmd=run_cmd,
        monotonic=MagicMock(side_effect=[0.0, 0.0, 0.0]),
        print_fn=messages.append,
        pyspy_bin=pyspy_bin,
        diagnostics_root=diagnostics_root,
    )

    out_dirs = list(diagnostics_root.glob("serverhang_*"))
    assert len(out_dirs) == 1
    out_dir = out_dirs[0]
    assert (out_dir / "pid111.txt").read_text() == "stack for 111\n"
    assert (out_dir / "pid222.txt").read_text() == "stack for 222\n"


def test_capture_forensics_respects_global_60s_deadline(tmp_path):
    """Once the fake monotonic clock crosses the 60s deadline, remaining pids
    are skipped -- the forensics step must never blow the caller's time
    budget."""
    pyspy_bin = _fake_pyspy(tmp_path)
    diagnostics_root = tmp_path / "diagnostics"
    run_cmd = MagicMock(return_value=_proc(stdout="stack\n"))

    # First call: deadline = 0 + 60 = 60. Then per-pid checks: 0 (<60, dump
    # pid 111), 61 (>=60, stop before pid 222).
    monotonic = MagicMock(side_effect=[0.0, 0.0, 61.0])

    _capture_forensics(
        [111, 222, 333],
        run_cmd=run_cmd,
        monotonic=monotonic,
        print_fn=lambda *_: None,
        pyspy_bin=pyspy_bin,
        diagnostics_root=diagnostics_root,
    )

    assert run_cmd.call_count == 1
    out_dir = next(diagnostics_root.glob("serverhang_*"))
    assert (out_dir / "pid111.txt").exists()
    assert not (out_dir / "pid222.txt").exists()
    assert not (out_dir / "pid333.txt").exists()


def test_capture_forensics_exception_does_not_prevent_kills(tmp_path):
    """If the forensics step raises (e.g. py-spy crashes), restart_server must
    still proceed to kill every pid and continue to wait for health."""
    cgroup_path = _write_cgroup_procs(tmp_path, [111, 222])

    def run_cmd(cmd, **kwargs):
        if "py-spy" in cmd[0]:
            raise OSError("py-spy blew up")
        return _proc(stdout="")

    kill_fn = MagicMock()
    urlopen = MagicMock(return_value=_HealthResp(b"ok"))

    rc = restart_server(
        run_cmd=run_cmd,
        sleep_fn=MagicMock(),
        # 3 calls for forensics (deadline + 2 pid checks) + 1 for
        # _wait_for_health's own deadline calc (health succeeds first try).
        monotonic=MagicMock(side_effect=[0.0, 0.0, 0.0, 0.0]),
        urlopen=urlopen,
        kill_fn=kill_fn,
        cgroup_procs_path=cgroup_path,
        pyspy_bin=_fake_pyspy(tmp_path),
        diagnostics_root=tmp_path / "diagnostics",
        print_fn=lambda *_: None,
    )

    assert rc == 0
    killed_pids = {c.args[0] for c in kill_fn.call_args_list}
    assert killed_pids == {111, 222}


def test_capture_forensics_missing_pyspy_binary_is_a_noop(tmp_path):
    diagnostics_root = tmp_path / "diagnostics"
    run_cmd = MagicMock()
    messages = []

    _capture_forensics(
        [111],
        run_cmd=run_cmd,
        monotonic=MagicMock(),
        print_fn=messages.append,
        pyspy_bin=_missing_pyspy(tmp_path),
        diagnostics_root=diagnostics_root,
    )

    run_cmd.assert_not_called()
    assert not diagnostics_root.exists()
    assert any("py-spy not available" in m for m in messages)


# --------------------------------------------------------------------- #
# Review findings (reviewer report, CHANGES_REQUESTED):
#  1. forensics call at the restart_server() call site must be wrapped in
#     its own try/except -- a mutation that moves the capture AFTER the
#     kills must be caught by a test (finding 2 below).
#  2. an exception outside _capture_forensics' internal try/except blocks
#     (e.g. the deadline calculation itself) must not prevent the kills.
#  3. worst-case time budget must stay <= 290s.
# --------------------------------------------------------------------- #

def test_forensics_capture_happens_before_kills(tmp_path):
    """Mutation-test regression (reviewer report finding 2): a reviewer moved
    the `_capture_forensics(...)` call to AFTER `_kill_pids(...)` in
    `restart_server()` and no test caught it. Both the injected run_cmd
    (py-spy dump) and kill_fn append to a shared, ordered event log so this
    test fails if that ordering mutation is reintroduced."""
    cgroup_path = _write_cgroup_procs(tmp_path, [111, 222])
    events: list = []

    def run_cmd(cmd, **kwargs):
        if "py-spy" in str(cmd[0]):
            pid = cmd[cmd.index("--pid") + 1]
            events.append(("dump", pid))
            return _proc(stdout=f"stack {pid}\n")
        return _proc(stdout="")  # systemctl show -p MainPID

    def kill_fn(pid, sig):
        events.append(("kill", pid))

    urlopen = MagicMock(return_value=_HealthResp(b"ok"))

    rc = restart_server(
        run_cmd=run_cmd,
        sleep_fn=MagicMock(),
        # 3 calls for forensics (deadline + 2 pid checks) + 1 for
        # _wait_for_health's own deadline calc (health succeeds first try).
        monotonic=MagicMock(side_effect=[0.0, 0.0, 0.0, 0.0]),
        urlopen=urlopen,
        kill_fn=kill_fn,
        cgroup_procs_path=cgroup_path,
        pyspy_bin=_fake_pyspy(tmp_path),
        diagnostics_root=tmp_path / "diagnostics",
        print_fn=lambda *_: None,
    )

    assert rc == 0
    dump_indices = [i for i, e in enumerate(events) if e[0] == "dump"]
    kill_indices = [i for i, e in enumerate(events) if e[0] == "kill"]
    assert dump_indices, "expected at least one py-spy dump event"
    assert kill_indices, "expected at least one kill event"
    assert max(dump_indices) < min(kill_indices), (
        f"forensics dumps must all happen before any kill, got order: {events}"
    )


def test_deadline_calculation_exception_does_not_prevent_kills(tmp_path):
    """Reviewer-reproduced scenario: an exception raised by `monotonic()`
    itself (simulating the deadline calc, or equally a `PermissionError`
    escaping `Path(pyspy_bin).exists()`, which pathlib does NOT swallow) must
    not prevent restart_server() from reaching _kill_pids(). This exercises a
    failure point OUTSIDE _capture_forensics' own internal try/except blocks
    — only the try/except now wrapping the whole call site in
    restart_server() can catch it."""
    cgroup_path = _write_cgroup_procs(tmp_path, [111, 222])
    run_cmd = MagicMock(return_value=_proc(stdout=""))
    kill_fn = MagicMock()
    urlopen = MagicMock(return_value=_HealthResp(b"ok"))

    calls = {"n": 0}

    def flaky_monotonic():
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("clock exploded")
        return 0.0

    rc = restart_server(
        run_cmd=run_cmd,
        sleep_fn=MagicMock(),
        monotonic=flaky_monotonic,
        urlopen=urlopen,
        kill_fn=kill_fn,
        cgroup_procs_path=cgroup_path,
        pyspy_bin=_fake_pyspy(tmp_path),
        diagnostics_root=tmp_path / "diagnostics",
        print_fn=lambda *_: None,
    )

    assert rc == 0
    killed_pids = {c.args[0] for c in kill_fn.call_args_list}
    assert killed_pids == {111, 222}


def test_pyspy_exists_check_permission_error_does_not_prevent_kills(tmp_path, monkeypatch):
    """The other half of the reviewer's exact repro: `Path.exists()` on
    CPython only swallows ENOENT/ENOTDIR/EBADF/ELOOP -- a PermissionError
    (EACCES) propagates. Simulated by monkeypatching Path.exists globally;
    must not prevent the kills."""
    import scripts.restart_exonit_server as mod

    cgroup_path = _write_cgroup_procs(tmp_path, [111])
    run_cmd = MagicMock(return_value=_proc(stdout=""))
    kill_fn = MagicMock()
    urlopen = MagicMock(return_value=_HealthResp(b"ok"))

    def raising_exists(self):
        raise PermissionError("EACCES")

    monkeypatch.setattr(mod.Path, "exists", raising_exists)

    rc = restart_server(
        run_cmd=run_cmd,
        sleep_fn=MagicMock(),
        monotonic=MagicMock(side_effect=[0.0, 0.0]),
        urlopen=urlopen,
        kill_fn=kill_fn,
        cgroup_procs_path=cgroup_path,
        pyspy_bin=_fake_pyspy(tmp_path),
        diagnostics_root=tmp_path / "diagnostics",
        print_fn=lambda *_: None,
    )

    assert rc == 0
    killed_pids = {c.args[0] for c in kill_fn.call_args_list}
    assert killed_pids == {111}


def test_worst_case_time_budget_stays_under_290s():
    """systemctl show timeout (_MAIN_PID_TIMEOUT_S) + forensics hard deadline
    (_FORENSICS_DEADLINE_S, now a real hard cap since the per-pid py-spy
    timeout is clamped to the time remaining until the deadline) + the
    restart_server() default --timeout-s health poll must sum to <= 290s,
    leaving margin under the Executor playbook's 300s subprocess timeout for
    the (near-instantaneous) kill step and interpreter/systemd overhead."""
    import inspect

    import scripts.restart_exonit_server as mod

    default_timeout_s = inspect.signature(mod.restart_server).parameters["timeout_s"].default
    worst_case_s = mod._MAIN_PID_TIMEOUT_S + mod._FORENSICS_DEADLINE_S + default_timeout_s

    assert worst_case_s <= 290, f"worst-case budget {worst_case_s}s exceeds the 290s ceiling"
