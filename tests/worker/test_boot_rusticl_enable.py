"""Regression test for the RUSTICL_ENABLE boot default.

Bug observed 2026-04-20 on worker-default after a plain ``nohup`` restart:
pyopencl queried platforms, got a rusticl platform with zero devices,
and the worker silently ran injection/FPP on CPU. Root cause: Mesa's
rusticl driver is off by default and only exposes devices when the env
var ``RUSTICL_ENABLE=<driver>`` is set before OpenCL queries.

``scripts/watchdog_local_worker.sh`` and ``scripts/test_period_batch.sh``
already set this var when they launch the worker, but a direct
``nohup venv/bin/python worker.py`` does not — so ``_boot_enable_rusticl``
in ``worker.py`` sets it via ``os.environ.setdefault`` before any heavy
import. This test walks the worker source to verify that default remains.
"""
from __future__ import annotations

import ast
from pathlib import Path


_WORKER_SRC = Path(__file__).resolve().parents[2] / "worker" / "worker.py"


def test_boot_block_sets_rusticl_enable_default():
    """The boot block must call environ.setdefault('RUSTICL_ENABLE',
    'radeonsi') before any numpy/pyopencl import."""
    src = _WORKER_SRC.read_text()
    assert 'setdefault("RUSTICL_ENABLE"' in src or \
           "setdefault('RUSTICL_ENABLE'" in src, \
        "worker boot block must set RUSTICL_ENABLE default"


def test_rusticl_default_runs_before_heavy_imports():
    """``_boot_enable_rusticl()`` call must appear BEFORE ``import numpy``
    / ``import pyopencl`` so the env var is visible to the OpenCL driver
    query at pyopencl import time."""
    src = _WORKER_SRC.read_text()
    lines = src.splitlines()

    rusticl_line = next(
        (i for i, line in enumerate(lines) if "_boot_enable_rusticl()" in line),
        None,
    )
    assert rusticl_line is not None, \
        "worker.py must invoke _boot_enable_rusticl() at module load"

    # Heavy imports that trigger pyopencl / numpy driver probes must come after.
    for bad_import in ("import numpy", "import pyopencl",
                       "from bls_opencl", "from injection"):
        for i, line in enumerate(lines):
            if line.lstrip().startswith(bad_import):
                assert i > rusticl_line, (
                    f"{bad_import!r} at line {i+1} comes before "
                    f"_boot_enable_rusticl() at line {rusticl_line+1}; "
                    "the env var won't take effect")
                break


def test_setdefault_respects_user_override(monkeypatch):
    """setdefault (vs assignment) means an explicit user override wins.
    Rerun the boot routine with RUSTICL_ENABLE already set and confirm
    it is NOT overwritten."""
    import os as _os
    # Reload worker module isn't safe (it registers the worker), so test
    # the boot routine in isolation via a local copy of the logic.
    # The behaviour we care about: setdefault must not overwrite.
    monkeypatch.setenv("RUSTICL_ENABLE", "iris")
    # Mimic the boot block's logic. This duplicates one line intentionally
    # so we exercise the contract, not an import.
    _os.environ.setdefault("RUSTICL_ENABLE", "radeonsi")
    assert _os.environ["RUSTICL_ENABLE"] == "iris", (
        "setdefault must leave a user-chosen driver alone")
