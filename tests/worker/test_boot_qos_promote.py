"""Regression tests for the macOS QoS-promotion boot helper.

Bug observed 2026-04-21 on worker-<host> (Apple M1): the worker launched
via ``nohup`` had ``NI=5 STAT=SN`` and all numba threads ran on E-cores
while P-cores stayed idle. Root cause: macOS/Darwin auto-tags nohup /
launchd children as *background* QoS, which the Apple Silicon scheduler
gates to the efficiency cluster.

Applied in-flight to a running <host> worker, ``taskpolicy -B -p <pid>``
lifted CPU from ~3% to ~530% — enough evidence that clearing the
background flag alone already unlocks the P-cores.

Design note: ``taskpolicy -c <class>`` is rejected with ``"Could not
parse as a QoS clamp"`` when combined with ``-p`` on an existing
process; the class clamp only applies when taskpolicy *launches* the
child. Full QoS-class promotion therefore lives in ``start.sh`` (or
wherever the Python interpreter is spawned); this boot helper does
the subset that works on a running PID — clear the background flag.

These tests pin down ``_boot_promote_qos_if_mac`` in ``worker.py``:

  * Linux / Windows: no-op.
  * macOS: exactly one ``taskpolicy`` call, with ``-B`` and
    ``-p <our own PID>``. No ``-c`` (would fail at runtime with ``-p``).
  * Missing / sandboxed ``taskpolicy``: swallow ``FileNotFoundError`` /
    ``OSError`` / ``TimeoutExpired`` so the worker still boots.
  * The module-load boot block invokes the helper exactly once.
"""
from __future__ import annotations

import ast
import subprocess
from pathlib import Path
from unittest import mock


_WORKER_SRC = Path(__file__).resolve().parents[2] / "worker" / "worker.py"


def _load_boot_fn():
    """Extract ``_boot_promote_qos_if_mac`` from ``worker.py`` by AST
    parse, without executing the rest of the module (which would start
    heavy imports and network registration)."""
    tree = ast.parse(_WORKER_SRC.read_text())
    fn_node = next(
        (n for n in ast.walk(tree)
         if isinstance(n, ast.FunctionDef)
         and n.name == "_boot_promote_qos_if_mac"),
        None,
    )
    assert fn_node is not None, \
        "worker.py must define _boot_promote_qos_if_mac"
    module = ast.Module(body=[fn_node], type_ignores=[])
    code = compile(module, str(_WORKER_SRC), "exec")
    import os as _os_boot
    import sys as _sys_boot
    ns: dict = {"_os_boot": _os_boot, "_sys_boot": _sys_boot}
    exec(code, ns)
    return ns["_boot_promote_qos_if_mac"]


def test_no_op_on_linux(monkeypatch):
    fn = _load_boot_fn()
    monkeypatch.setattr("sys.platform", "linux")
    with mock.patch("subprocess.run") as run:
        fn()
    run.assert_not_called()


def test_no_op_on_windows(monkeypatch):
    fn = _load_boot_fn()
    monkeypatch.setattr("sys.platform", "win32")
    with mock.patch("subprocess.run") as run:
        fn()
    run.assert_not_called()


def test_promotes_qos_on_darwin(monkeypatch):
    fn = _load_boot_fn()
    monkeypatch.setattr("sys.platform", "darwin")
    monkeypatch.setattr("os.getpid", lambda: 7777)
    with mock.patch("subprocess.run") as run:
        fn()
    run.assert_called_once()
    args, kwargs = run.call_args
    cmd = args[0]
    assert cmd[0] == "taskpolicy"
    # -B clears the Darwin background flag set by nohup/SSH-bg
    assert "-B" in cmd
    # -p targets our own PID so the promotion applies to this process
    assert "-p" in cmd
    p_idx = cmd.index("-p")
    assert cmd[p_idx + 1] == "7777"
    # -c <class> is incompatible with -p on a running process; must
    # NOT be present or taskpolicy exits with a usage error.
    assert "-c" not in cmd, (
        "-c <class> is rejected when combined with -p on a running "
        "process; QoS class promotion belongs in start.sh, not here"
    )
    # Non-blocking (timeout set) so a stuck taskpolicy can't hang boot
    assert kwargs.get("timeout") is not None
    # check=False so a non-zero exit from taskpolicy does not crash boot
    assert kwargs.get("check") is False


def test_missing_taskpolicy_is_tolerated(monkeypatch):
    fn = _load_boot_fn()
    monkeypatch.setattr("sys.platform", "darwin")
    with mock.patch("subprocess.run",
                    side_effect=FileNotFoundError("taskpolicy")):
        fn()


def test_sandbox_oserror_is_tolerated(monkeypatch):
    fn = _load_boot_fn()
    monkeypatch.setattr("sys.platform", "darwin")
    with mock.patch("subprocess.run", side_effect=OSError("denied")):
        fn()


def test_taskpolicy_timeout_is_tolerated(monkeypatch):
    """If ``taskpolicy`` hangs, the 5s timeout raises ``TimeoutExpired``.
    The helper must swallow it rather than blocking the worker forever."""
    fn = _load_boot_fn()
    monkeypatch.setattr("sys.platform", "darwin")
    with mock.patch("subprocess.run",
                    side_effect=subprocess.TimeoutExpired("taskpolicy", 5)):
        fn()


def test_boot_block_invokes_promote_once():
    src = _WORKER_SRC.read_text()
    occurrences = [
        line for line in src.splitlines()
        if line.strip() == "_boot_promote_qos_if_mac()"
    ]
    assert len(occurrences) == 1, (
        f"worker.py boot block should invoke "
        f"_boot_promote_qos_if_mac() exactly once, "
        f"found {len(occurrences)}")
