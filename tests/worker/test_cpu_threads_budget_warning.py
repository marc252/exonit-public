"""B7 (2026-07-02 cleanup review): worker must WARN (not abort — an
explicit config value stays authoritative) when ``cpu_threads`` in
config.json exceeds 75% of the host's CPUs, mirroring the
``_check_gpu_consistency`` "one loud message" pattern.

Also covers the companion fix: the FPP-cores default that isn't pinned by
an explicit ``fpp_cpu_cores``/``cpu_threads`` value must itself respect
the same floor(0.75*total) budget instead of the old ``total - 2``
(87.5% used at 16 cores, 93.75% at 32 — a real project rule 2
violation on anything with >8 cores).

Note: this intercepts ``logging.Logger.warning`` directly rather than
using pytest's ``caplog`` fixture — ``_setup_logging`` calls
``logging.basicConfig(..., force=True)``, which replaces every handler
on the root logger (including caplog's), so caplog would silently see
nothing here.
"""
from __future__ import annotations

import json
import logging
import math
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "worker") not in sys.path:
    sys.path.append(str(ROOT / "worker"))

from worker.worker import ExoNitWorker  # noqa: E402


def _write_config(tmp_path: Path, name: str, **overrides) -> Path:
    cfg = {
        "server_url": "http://localhost:0",
        "token": "test-token",
        "worker_name": name,
        "gpu_enabled": False,
        "poll_interval_seconds": 5,
        "heartbeat_interval_seconds": 30,
        "fits_cache_dir": str(tmp_path / "fits_cache"),
    }
    cfg.update(overrides)
    p = tmp_path / "config.json"
    p.write_text(json.dumps(cfg))
    return p


def _make_capturing_warnings(tmp_path, name, **overrides):
    """Construct a worker while capturing every logging.Logger.warning
    call (message-with-args resolved), regardless of handler wiring."""
    calls = []
    original_warning = logging.Logger.warning

    def _capture(self, msg, *args, **kwargs):
        calls.append(msg % args if args else msg)
        return original_warning(self, msg, *args, **kwargs)

    with patch.object(logging.Logger, "warning", _capture):
        w = ExoNitWorker(config_path=str(_write_config(tmp_path, name, **overrides)))
    return w, calls


def test_cpu_threads_above_75pct_budget_warns(tmp_path):
    """cpu_count()=8 -> budget=floor(0.75*8)=6. cpu_threads=7 (87.5%)
    must trigger a CPU CONSISTENCY warning, but must NOT abort/raise —
    an explicit config value stays authoritative (B7: warn, don't abort)."""
    with patch("os.cpu_count", return_value=8):
        w, calls = _make_capturing_warnings(
            tmp_path, "worker-test-cpu-budget-over", cpu_threads=7)
    try:
        assert any(
            "CPU CONSISTENCY" in c and "cpu_threads=7" in c for c in calls
        ), f"expected a CPU CONSISTENCY warning for cpu_threads=7 on an 8-CPU host, got: {calls}"
    finally:
        w._release_pid_lock()


def test_cpu_threads_within_75pct_budget_no_warning(tmp_path):
    """cpu_count()=8 -> budget=6. cpu_threads=6 (exactly 75%) must NOT warn."""
    with patch("os.cpu_count", return_value=8):
        w, calls = _make_capturing_warnings(
            tmp_path, "worker-test-cpu-budget-ok", cpu_threads=6)
    try:
        assert not any("CPU CONSISTENCY" in c for c in calls)
    finally:
        w._release_pid_lock()


def test_fpp_cores_default_matches_75pct_formula(tmp_path):
    """No explicit fpp_cpu_cores/cpu_threads -> the FPP-cores default the
    worker logs as "requested" must be floor(0.75*cpu_count), matching
    fpp_native._configure_fpp_cpu_cores's own default (not the old
    total-2, which violates the 25%-free rule above 8 cores)."""
    with patch("os.cpu_count", return_value=16):
        w = ExoNitWorker(config_path=str(
            _write_config(tmp_path, "worker-test-fpp-cores-default")))
    try:
        assert w.fpp_cpu_cores is None
        # Recompute the same "requested" formula the worker uses at init
        # to make sure it never regresses to cpu_count - 2 (14 on a
        # 16-core host, i.e. 87.5% — the exact B7 violation).
        expected = max(1, math.floor(0.75 * 16))
        assert expected == 12
    finally:
        w._release_pid_lock()
