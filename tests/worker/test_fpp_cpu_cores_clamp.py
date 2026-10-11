"""Regression test for _configure_fpp_cpu_cores vs cpu_threads interaction.

Bug observed 2026-04-20 on worker-default: config had ``cpu_threads=4``
and ``fpp_cpu_cores=6``. At worker boot, ``NUMBA_NUM_THREADS=4`` is set
as an env var (so numba's pool maxes out at 4). Then the FPP config
step called ``numba.set_num_threads(6)`` which raises
``ValueError: The number of threads must be between 1 and 4``. The
exception was caught and logged as "FPP CPU core config failed", which
is accurate but doesn't explain the cpu_threads interaction.

Fix: ``_configure_fpp_cpu_cores`` now clamps silently to
``numba.config.NUMBA_NUM_THREADS`` and returns the effective value so
the worker can log "FPP CPU cores: 4 (requested 6, capped by cpu_threads=4)".
"""
from __future__ import annotations

import pytest


def test_configure_fpp_returns_none_without_numba(monkeypatch):
    import fpp_native
    monkeypatch.setattr(fpp_native, "HAS_NUMBA", False)
    assert fpp_native._configure_fpp_cpu_cores(6) is None
    assert fpp_native._configure_fpp_cpu_cores() is None


def test_configure_fpp_clamps_to_numba_cap(monkeypatch):
    """Requesting more threads than NUMBA_NUM_THREADS silently clamps.
    Before the fix, numba.set_num_threads(6) would raise when the cap
    was 4."""
    import fpp_native
    if not fpp_native.HAS_NUMBA:
        pytest.skip("numba unavailable")
    import numba

    original_threads = numba.get_num_threads()
    monkeypatch.setattr(numba.config, "NUMBA_NUM_THREADS", 4, raising=False)
    try:
        effective = fpp_native._configure_fpp_cpu_cores(6)
        assert effective == 4, (
            f"expected clamp to NUMBA_NUM_THREADS (4), got {effective}")
    finally:
        # Restore to whatever numba had before — don't leak into other tests
        numba.set_num_threads(min(original_threads,
                                   getattr(numba.config, "NUMBA_NUM_THREADS",
                                           original_threads)))


def test_configure_fpp_honors_request_below_cap(monkeypatch):
    """When request ≤ cap, use the request verbatim."""
    import fpp_native
    if not fpp_native.HAS_NUMBA:
        pytest.skip("numba unavailable")
    import numba

    original_threads = numba.get_num_threads()
    monkeypatch.setattr(numba.config, "NUMBA_NUM_THREADS", 8, raising=False)
    try:
        effective = fpp_native._configure_fpp_cpu_cores(3)
        assert effective == 3
    finally:
        numba.set_num_threads(min(original_threads,
                                   getattr(numba.config, "NUMBA_NUM_THREADS",
                                           original_threads)))


def test_configure_fpp_default_leaves_system_room(monkeypatch):
    """None → floor(0.75 * cpu_count) (project rule 2: >=25% free),
    still clamped by cap.

    Was ``max(1, cpu_count - 2)`` until the 2026-07-02 cleanup review
    (B7): that only reserves a fixed 2 cores regardless of host size,
    which violates the 25%-free rule on anything with >8 cores (87.5%
    used at 16 cores, 93.75% at 32). Now matches the same
    floor(0.75*total) formula agent.py:207-223 already used.
    """
    import fpp_native
    if not fpp_native.HAS_NUMBA:
        pytest.skip("numba unavailable")
    import numba
    import math as _math
    import os as _os

    original_threads = numba.get_num_threads()
    monkeypatch.setattr(numba.config, "NUMBA_NUM_THREADS", 128, raising=False)
    try:
        effective = fpp_native._configure_fpp_cpu_cores(None)
        expected = max(1, _math.floor(0.75 * (_os.cpu_count() or 4)))
        assert effective == expected
    finally:
        numba.set_num_threads(min(original_threads,
                                   getattr(numba.config, "NUMBA_NUM_THREADS",
                                           original_threads)))


def test_configure_fpp_no_longer_raises_on_request_above_cap(monkeypatch):
    """The pre-fix regression: calling set_num_threads above the cap
    raised ValueError. The clamp prevents that path entirely."""
    import fpp_native
    if not fpp_native.HAS_NUMBA:
        pytest.skip("numba unavailable")
    import numba

    original_threads = numba.get_num_threads()
    monkeypatch.setattr(numba.config, "NUMBA_NUM_THREADS", 2, raising=False)
    try:
        # Before the fix this raised ValueError; now it must clamp silently.
        result = fpp_native._configure_fpp_cpu_cores(32)
        assert result == 2
    finally:
        numba.set_num_threads(min(original_threads,
                                   getattr(numba.config, "NUMBA_NUM_THREADS",
                                           original_threads)))
