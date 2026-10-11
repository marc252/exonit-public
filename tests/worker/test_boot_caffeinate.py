"""Regression tests for the macOS caffeinate boot helper.

Bug observed 2026-04-21: worker-<host> (MacBook Air) stopped heartbeating
mid-run because the system went to sleep. The worker binary does not create
any power assertion, so without an external ``caffeinate`` wrapper the
laptop suspends. These tests lock in that ``_boot_start_caffeinate_if_mac``
in ``worker.py``:

  * runs only on macOS (Linux / Windows are a no-op),
  * invokes ``caffeinate -i -s -w <pid>`` bound to the worker's own PID,
  * swallows ``FileNotFoundError`` / ``OSError`` so a missing binary or a
    launchd sandbox does not crash the worker at startup,
  * is invoked once at module load.
"""
from __future__ import annotations

import ast
from pathlib import Path
from unittest import mock

import pytest


_WORKER_SRC = Path(__file__).resolve().parents[2] / "worker" / "worker.py"


def _load_boot_fn():
    """Return the ``_boot_start_caffeinate_if_mac`` function object,
    extracted by parsing ``worker.py`` without executing the rest of
    the module (which would start heavy imports and register a worker).
    """
    tree = ast.parse(_WORKER_SRC.read_text())
    fn_node = next(
        (n for n in ast.walk(tree)
         if isinstance(n, ast.FunctionDef)
         and n.name == "_boot_start_caffeinate_if_mac"),
        None,
    )
    assert fn_node is not None, \
        "worker.py must define _boot_start_caffeinate_if_mac"
    # Wrap the function in a standalone module so we can exec it in an
    # isolated namespace that mirrors the worker's boot-time imports.
    module = ast.Module(body=[fn_node], type_ignores=[])
    code = compile(module, str(_WORKER_SRC), "exec")
    import os as _os_boot
    import sys as _sys_boot
    ns: dict = {"_os_boot": _os_boot, "_sys_boot": _sys_boot}
    exec(code, ns)
    return ns["_boot_start_caffeinate_if_mac"]


def test_no_op_on_linux(monkeypatch):
    fn = _load_boot_fn()
    monkeypatch.setattr("sys.platform", "linux")
    with mock.patch("subprocess.Popen") as popen:
        fn()
    popen.assert_not_called()


def test_no_op_on_windows(monkeypatch):
    fn = _load_boot_fn()
    monkeypatch.setattr("sys.platform", "win32")
    with mock.patch("subprocess.Popen") as popen:
        fn()
    popen.assert_not_called()


def test_spawns_caffeinate_on_darwin(monkeypatch):
    fn = _load_boot_fn()
    monkeypatch.setattr("sys.platform", "darwin")
    monkeypatch.setattr("os.getpid", lambda: 4242)
    with mock.patch("subprocess.Popen") as popen:
        fn()
    popen.assert_called_once()
    args, kwargs = popen.call_args
    cmd = args[0]
    assert cmd[0] == "caffeinate"
    assert "-i" in cmd and "-s" in cmd, \
        "must prevent both idle and system sleep"
    # PID bound so the child dies with the worker
    assert "-w" in cmd
    w_idx = cmd.index("-w")
    assert cmd[w_idx + 1] == "4242"
    # Detached session so a terminal SIGHUP doesn't kill caffeinate
    assert kwargs.get("start_new_session") is True


def test_missing_caffeinate_binary_is_tolerated(monkeypatch):
    fn = _load_boot_fn()
    monkeypatch.setattr("sys.platform", "darwin")
    with mock.patch("subprocess.Popen",
                    side_effect=FileNotFoundError("caffeinate")):
        # Must not raise — the worker boots just fine without a sleep
        # guard if the binary is unavailable; the loss is only that
        # MacBook Airs can sleep, which is no worse than today.
        fn()


def test_launchd_sandbox_oserror_is_tolerated(monkeypatch):
    """Some launchd environments restrict subprocess spawn and raise
    generic ``OSError`` (not ``FileNotFoundError``). The worker must
    still boot rather than crash on startup."""
    fn = _load_boot_fn()
    monkeypatch.setattr("sys.platform", "darwin")
    with mock.patch("subprocess.Popen",
                    side_effect=OSError("sandbox denied")):
        fn()


def test_boot_block_invokes_caffeinate_once():
    """The module-load boot block must call the caffeinate helper
    exactly once, alongside the CPU-cap and rusticl helpers."""
    src = _WORKER_SRC.read_text()
    # Count bare function invocations at module level. The call must
    # appear outside the def itself.
    occurrences = [
        line for line in src.splitlines()
        if line.strip() == "_boot_start_caffeinate_if_mac()"
    ]
    assert len(occurrences) == 1, (
        f"worker.py boot block should invoke "
        f"_boot_start_caffeinate_if_mac() exactly once, "
        f"found {len(occurrences)}")
