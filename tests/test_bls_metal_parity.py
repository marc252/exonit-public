"""BLS-Metal parity tests for the fused injection kernel (worker/bls_metal_batch.py).

Spec: docs/superpowers/specs/2026-07-11-gpu-bls-f32-unified-kernel.md.
No committed Metal parity test existed before this file (gap flagged in the
2026-05-30 audit) — this is the regression gate for both:

1. The algorithm/semantics of the fused kernel's f32 arithmetic, checked via
   ``kernel_reference_numpy`` (a pure-numpy simulator of the exact kernel
   math) against ``bls_fast.bls_fast_batch`` (the numba CPU reference).
   Platform-independent — runs on Linux CI / <host>, no MLX required.
2. The SoC allowlist + ``EXONIT_INJECTION_METAL`` kill-switch gating in
   ``bls_metal_batch.is_available()``/``is_supported_soc()`` — exercised
   with a fake ``mx`` stand-in, so it also runs without real MLX/hardware.
3. The GPU-glitch -> CPU-retry robustness path in
   ``injection._run_batched_gpu`` (stub ``batch_fn`` that raises, no MLX
   needed either), including the GPU-failure circuit breaker
   (``injection._record_gpu_batch_failure`` / ``gpu_circuit_is_open``).
4. ``injection.is_fused_metal_active()`` — the boolean routing predicate
   used by worker.py instead of a ``get_backend_name() == "..."`` string
   comparison.
5. The Metal threadgroup-memory budget guard in ``bls_gpu_batch_v2``
   (``n_bins`` too large for one threadgroup must raise a clear error,
   not fail obscurely in the kernel).
6. The real Metal kernel (``bls_gpu_batch_v2``) against the CPU reference,
   plus the decisive end-to-end check: a completeness map computed by
   ``injection._run_batched_gpu`` must be BIT-IDENTICAL (max|Delta| == 0.0)
   whether ``batch_fn`` is the CPU or the Metal batch function. Skipped
   cleanly when MLX/Metal is unavailable (matches the existing
   test_bls_cuda_parity.py / test_bls_opencl_batch.py pattern).
7. The two invariants the kernel's docstrings claim most prominently and
   that (2) above never actually exercises: the phase-wrap window branch
   (``_wrap_boundary_batch``, a fixture whose winning window provably
   requires ``e = j + w > n_bins``) and the strict-``>`` tie-break
   (``_tie_break_batch``, an exact-integer-arithmetic fixture engineered so
   two distinct (width, phase) candidates tie bit-for-bit). 2026-07-12
   re-review, Finding 2 — see the fault-injection evidence in
   ``progress/impl-metal-hardening.md``.
8. The combined ``EXONIT_INJECTION_METAL=0`` fallback chain in
   ``injection.py``'s module-level auto-detect (not just
   ``bls_metal_batch.is_available()`` in isolation) — 2026-07-12
   re-review, Finding 1.
"""

import importlib
import logging

import numpy as np
import pytest

from bls_fast import bls_fast_batch
import bls_metal
import injection

try:
    import bls_metal_batch
    _METAL_MODULE_OK = True
except Exception:
    _METAL_MODULE_OK = False

_METAL_V2 = False
if _METAL_MODULE_OK:
    try:
        _METAL_V2 = bls_metal_batch.is_available()
    except Exception:
        _METAL_V2 = False


@pytest.fixture(autouse=True)
def _reset_gpu_circuit_breaker():
    """The GPU-failure circuit breaker is process-global state by design
    (it must survive across stars). Isolate every test in this file from
    it regardless of order/failure, so one test's flaky batch_fn can't
    trip the breaker for another."""
    injection.reset_gpu_circuit_breaker()
    yield
    injection.reset_gpu_circuit_breaker()


class _FakeMx:
    """Trivial mx.core stand-in so is_available()'s smoke-test
    (``mx.array`` -> ``mx.eval(mx.sum(...))``) succeeds without real MLX —
    used to exercise the SoC/kill-switch branch logic on any host."""
    float32 = "float32"

    @staticmethod
    def array(*a, **k):
        return object()

    @staticmethod
    def sum(*a, **k):
        return object()

    @staticmethod
    def eval(*a, **k):
        return None


# ============================================================
# Synthetic injection-like batch (small, fast — semantics gate, not perf)
# ============================================================

def _injection_like_batch(n_points=150, n_flux=4, period_min=2.0,
                          period_max=10.0, seed=0):
    """A handful of fluxes over a shared time grid, each with its own
    injected transit — mirrors one period-cell of
    ``injection._run_batched_gpu`` (many fluxes sharing one BLS call)."""
    rng = np.random.default_rng(seed)
    time = np.linspace(0.0, 30.0, n_points).astype(np.float64)
    periods = np.linspace(period_min + 0.3, period_max - 0.3, n_flux)
    flux_list = []
    for i, period in enumerate(periods):
        flux = 1.0 + rng.normal(0, 3e-4, n_points)
        t0 = 0.7 + 0.1 * i
        depth = 4e-3
        duration = 0.15 * period
        phase = ((time - t0) % period) / period
        half_dur = (duration / period) / 2
        flux[(phase < half_dur) | (phase > 1.0 - half_dur)] -= depth
        flux_list.append(flux)
    return time, flux_list, float(period_min), float(period_max)


def _wrap_boundary_batch(n_points=150, seed=1, n_bins=40):
    """A single flux whose injected transit lands within one bin-width of
    the RECOVERY's phase 0/1 boundary, forcing the winning (width, phase)
    window through the wrap branch (``e = j + w > n_bins``) instead of the
    straight ``e <= n_bins`` one.

    ``_injection_like_batch`` above never does this: its ``t0`` is an
    absolute-time offset subtracted inside the injection's OWN phase
    definition (``(time - t0) % period``), which always centers the
    injected dip at phase 0 *relative to t0* — but the recovery phase-folds
    with NO t0 offset (``time / period % 1``, see ``bls_metal._phase_fold_cpu``
    / ``bls_fast``'s inlined fold). Relative to that unshifted phase, the
    dip actually sits at ``(t0 % period) / period``, which for
    ``_injection_like_batch``'s t0 values (0.7-1.0) and periods (2.3-9.7 d)
    lands solidly mid-range (~0.10-0.30) — nowhere near the 0/1 seam. Here
    ``t0`` is chosen so that ratio is deliberately close to 1.0 instead
    (2026-07-12 re-review, Finding 2).
    """
    rng = np.random.default_rng(seed)
    time = np.linspace(0.0, 30.0, n_points).astype(np.float64)
    period = 5.0  # inside the fixture's [period_min, period_max] below
    depth = 6e-3
    duration = 0.12 * period
    t0 = 0.995 * period  # recovery-phase ~= 0.995 -- within 1 bin (1/40) of the 0/1 seam
    flux = 1.0 + rng.normal(0, 2e-4, n_points)
    phase = ((time - t0) % period) / period
    half_dur = (duration / period) / 2
    flux[(phase < half_dur) | (phase > 1.0 - half_dur)] -= depth
    return time, [flux], 2.0, 10.0


def _recovered_bin_window(result, n_bins):
    """Recover the integer (j, w) bin window from a kernel_reference_numpy /
    bls_fast_batch result dict (both report ``t0``/``duration`` in physical
    units, derived from the winning bin indices as
    ``t0 = j/n_bins*period + t0_ref`` / ``duration = w/n_bins*period`` — see
    ``kernel_reference_numpy``). Used only to confirm a fixture actually
    exercises the wrap branch (``j + w > n_bins``); not a kernel_reference_numpy
    API, so this must recompute from the public dict fields, not internals."""
    period = result["period"]
    w = int(round(result["duration"] / period * n_bins))
    j = int(round(result["t0"] / period * n_bins)) % n_bins
    return j, w


def _tie_break_batch():
    """An engineered exact SR tie between two (width, phase) candidates.

    All flux values and their sums are exact small integers (well within
    float32's 24-bit exact-integer range), so the window sums -- computed
    via prefix-sum differences, the same technique the Metal kernel and
    ``kernel_reference_numpy`` use -- are bit-for-bit reproducible
    regardless of summation order. That matters: a naive tie construction
    using *noisy* data that merely looks symmetric does NOT tie exactly
    once ws is computed via ``cumsum[e] - cumsum[j]`` (verified by hand —
    the accumulated rounding differs at the last bit between two
    "symmetric" prefix-sum differences taken at different offsets).

    n_bins=12, one flux sample per bin (single period, single duration
    step of width 3 bins), values:
        [-48,-48,-48, 0,0,0, -48,-48,-48, 0,0,0]
    Centered, bins 0-2 and 6-8 are each [-24,-24,-24] (identical), so the
    width-3 windows starting at j=0 and j=6 have identical ws=-72 -> the
    SAME sr, exactly, and are the global maximum (every other window mixes
    -24s and +24s, giving strictly smaller depth). Under the real strict
    ``>`` tie-break, the FIRST one encountered in iteration order (j=0,
    smaller phase) wins; the second occurrence (j=6) has ``sr > bs`` false
    since they're equal, so it's never applied.
    """
    n_bins = 12
    period = 1.0
    # time_rel[0] is always 0 (kernel_reference_numpy/bls_fast_batch both
    # subtract time[0]) -- that lands bin 0 safely at phase exactly 0.0 (no
    # rounding ambiguity). The remaining 11 points sit at (i+0.5)/n_bins,
    # comfortably inside bin i, away from any edge-rounding risk.
    time = np.concatenate(([0.0], (np.arange(1, n_bins) + 0.5) / n_bins * period))
    flux = np.array([-48., -48., -48., 0., 0., 0.,
                     -48., -48., -48., 0., 0., 0.])
    return time, [flux], period, period  # period_min == period_max


# ============================================================
# 1. Platform-independent: exact-f32 numpy simulator vs bls_fast_batch
# ============================================================

@pytest.mark.skipif(not _METAL_MODULE_OK, reason="bls_metal_batch module unavailable")
class TestKernelReferenceNumpyParity:
    """kernel_reference_numpy must reproduce bls_fast_batch's period
    selection and be numerically close (pure f32-rounding drift only —
    same tolerances as the spec's PoC validation: rel Delta power ~1e-6,
    Delta depth/snr small). Pure numpy — no MLX/hardware required."""

    def test_period_selection_matches_bls_fast_batch(self):
        time, flux_list, pmin, pmax = _injection_like_batch()
        kwargs = dict(period_min=pmin, period_max=pmax, n_periods=200,
                      n_bins=40, n_dur_steps=5)
        cpu = bls_fast_batch(time, flux_list, **kwargs)
        sim = bls_metal_batch.kernel_reference_numpy(time, flux_list, **kwargs)
        assert len(sim) == len(cpu) == len(flux_list)
        for i, (c, s) in enumerate(zip(cpu, sim)):
            assert s["period"] == pytest.approx(c["period"], rel=1e-9), (
                f"flux #{i}: period selection differs -- cpu={c['period']:.6f} "
                f"sim={s['period']:.6f}")

    def test_value_tolerances(self):
        time, flux_list, pmin, pmax = _injection_like_batch(seed=7)
        kwargs = dict(period_min=pmin, period_max=pmax, n_periods=200,
                      n_bins=40, n_dur_steps=5)
        cpu = bls_fast_batch(time, flux_list, **kwargs)
        sim = bls_metal_batch.kernel_reference_numpy(time, flux_list, **kwargs)
        for i, (c, s) in enumerate(zip(cpu, sim)):
            rel_power = abs(c["power"] - s["power"]) / max(abs(c["power"]), 1e-9)
            assert rel_power < 1e-4, f"flux #{i}: rel power drift {rel_power:.2e}"
            rel_depth = abs(c["depth"] - s["depth"]) / max(abs(c["depth"]), 1e-9)
            assert rel_depth < 1e-3, f"flux #{i}: rel depth drift {rel_depth:.2e}"
            rel_snr = abs(c["snr"] - s["snr"]) / max(abs(c["snr"]), 1e-9)
            assert rel_snr < 5e-2, f"flux #{i}: rel snr drift {rel_snr:.2e}"

    def test_wrap_branch_matches_bls_fast_batch(self):
        """2026-07-12 re-review, Finding 2: the wrap branch (``e > n_bins``)
        has ~zero regression coverage from the other fixtures in this file
        -- ``_injection_like_batch`` never places the recovered window near
        the phase seam (see its docstring), so a dropped/broken wrap branch
        passes every other test here. This fixture forces it."""
        time, flux_list, pmin, pmax = _wrap_boundary_batch()
        kwargs = dict(period_min=pmin, period_max=pmax, n_periods=200,
                      n_bins=40, n_dur_steps=5)
        cpu = bls_fast_batch(time, flux_list, **kwargs)
        sim = bls_metal_batch.kernel_reference_numpy(time, flux_list, **kwargs)

        # Sanity check the fixture itself: the winning window must actually
        # require the wrap branch, on BOTH the numba reference and the
        # simulator, or this test would be vacuous (see acceptance-criterion
        # evidence in progress/impl-metal-hardening.md).
        j_cpu, w_cpu = _recovered_bin_window(cpu[0], n_bins=40)
        j_sim, w_sim = _recovered_bin_window(sim[0], n_bins=40)
        assert j_cpu + w_cpu > 40, (
            f"fixture bug: cpu winning window (j={j_cpu}, w={w_cpu}) does "
            "not cross the phase-wrap boundary")
        assert j_sim + w_sim > 40, (
            f"fixture bug: sim winning window (j={j_sim}, w={w_sim}) does "
            "not cross the phase-wrap boundary")

        c, s = cpu[0], sim[0]
        assert s["period"] == pytest.approx(c["period"], rel=1e-9)
        rel_power = abs(c["power"] - s["power"]) / max(abs(c["power"]), 1e-9)
        assert rel_power < 1e-4, f"rel power drift {rel_power:.2e}"
        rel_depth = abs(c["depth"] - s["depth"]) / max(abs(c["depth"]), 1e-9)
        assert rel_depth < 1e-3, f"rel depth drift {rel_depth:.2e}"

    def test_tie_break_matches_numba_first_in_order_winner(self):
        """2026-07-12 re-review, Finding 2: an exact SR tie is unfalsifiable
        by any noise-based fixture (float32 values essentially never tie on
        real data), so a flipped tie-break rule (``>`` -> ``>=``) passes
        every other test here. This fixture engineers a genuine bit-exact
        tie and asserts the strict-``>`` (first-in-iteration-order) winner,
        matching numba's own tie-break exactly."""
        time, flux_list, pmin, pmax = _tie_break_batch()
        kwargs = dict(period_min=pmin, period_max=pmax, n_periods=1,
                      n_bins=12, n_dur_steps=1,
                      duration_min_frac=0.25, duration_max_frac=0.25)
        cpu = bls_fast_batch(time, flux_list, **kwargs)[0]
        sim = bls_metal_batch.kernel_reference_numpy(time, flux_list, **kwargs)[0]

        j_cpu, w_cpu = _recovered_bin_window(cpu, n_bins=12)
        j_sim, w_sim = _recovered_bin_window(sim, n_bins=12)
        # j=0 and j=6 (width 3) are engineered to tie exactly -- the winner
        # must be j=0, the one first encountered in ascending-j iteration
        # order, not j=6 (which a `>=` tie-break would pick instead).
        assert (j_cpu, w_cpu) == (0, 3), (
            f"fixture bug or tie-break regression: numba picked (j={j_cpu}, "
            f"w={w_cpu}), expected the first-in-order tie winner (0, 3)")
        assert (j_sim, w_sim) == (j_cpu, w_cpu), (
            f"kernel_reference_numpy picked (j={j_sim}, w={w_sim}), numba "
            f"picked (j={j_cpu}, w={w_cpu}) -- tie-break disagreement")


# ============================================================
# 2. SoC allowlist + kill-switch gating (mocked mx — no real MLX needed)
# ============================================================

@pytest.mark.skipif(not _METAL_MODULE_OK, reason="bls_metal_batch module unavailable")
class TestSocGateAndKillSwitch:
    """is_available() gating logic. Requires only that the module import
    itself succeeds (bls_fast + bls_metal, no MLX) — MLX_AVAILABLE and `mx`
    are monkeypatched with a fake so the eval smoke-test passes anywhere."""

    def test_kill_switch_hard_disables_regardless_of_soc(self, monkeypatch):
        monkeypatch.setenv("EXONIT_INJECTION_METAL", "0")
        monkeypatch.setattr(bls_metal_batch, "MLX_AVAILABLE", True)
        monkeypatch.setattr(bls_metal_batch, "mx", _FakeMx(), raising=False)
        monkeypatch.setattr(bls_metal_batch, "is_supported_soc", lambda: True)
        assert bls_metal_batch.is_available() is False

    def test_no_mlx_is_never_available(self, monkeypatch):
        monkeypatch.delenv("EXONIT_INJECTION_METAL", raising=False)
        monkeypatch.setattr(bls_metal_batch, "MLX_AVAILABLE", False)
        assert bls_metal_batch.is_available() is False

    def test_unsupported_soc_without_override_is_unavailable(self, monkeypatch):
        monkeypatch.delenv("EXONIT_INJECTION_METAL", raising=False)
        monkeypatch.setattr(bls_metal_batch, "MLX_AVAILABLE", True)
        monkeypatch.setattr(bls_metal_batch, "mx", _FakeMx(), raising=False)
        monkeypatch.setattr(bls_metal_batch, "is_supported_soc", lambda: False)
        assert bls_metal_batch.is_available() is False

    def test_force_enable_bypasses_soc_allowlist(self, monkeypatch):
        monkeypatch.setenv("EXONIT_INJECTION_METAL", "1")
        monkeypatch.setattr(bls_metal_batch, "MLX_AVAILABLE", True)
        monkeypatch.setattr(bls_metal_batch, "mx", _FakeMx(), raising=False)
        monkeypatch.setattr(bls_metal_batch, "is_supported_soc", lambda: False)
        assert bls_metal_batch.is_available() is True

    @pytest.mark.parametrize("chip,expected", [
        ("Apple M1", True),
        ("Apple M1 Pro", True),
        ("Apple M3 Max", True),
        ("Apple M4", True),
        ("Apple M5", True),
        ("Intel(R) Core(TM) i7-9700K", False),
        ("AMD Ryzen 9 5900X", False),
        ("", False),
    ])
    def test_is_supported_soc_matches_apple_m_series_only(self, monkeypatch,
                                                          chip, expected):
        class _FakeResult:
            returncode = 0
            stdout = chip

        monkeypatch.setattr("subprocess.run", lambda *a, **k: _FakeResult())
        assert bls_metal_batch.is_supported_soc() is expected

    def test_legacy_bls_metal_is_metal_available_honors_kill_switch(self, monkeypatch):
        """2026-07-12 re-review, Finding 1: bls_metal.is_metal_available()
        (the legacy per-flux path) used to not read EXONIT_INJECTION_METAL
        at all -- confirmed here with no MLX mocking needed at all, since
        the kill-switch check must short-circuit before the MLX_AVAILABLE
        check."""
        monkeypatch.setenv("EXONIT_INJECTION_METAL", "0")
        assert bls_metal.is_metal_available() is False

    def test_legacy_bls_metal_kill_switch_is_case_insensitive_like_v2(self, monkeypatch):
        monkeypatch.setenv("EXONIT_INJECTION_METAL", "False")
        assert bls_metal.is_metal_available() is False


# ============================================================
# 2b. Integrated fallback chain: injection.py's module-level auto-detect
# must never resolve to Metal when EXONIT_INJECTION_METAL=0, independent
# of what either per-module is_available() check does on its own.
# ============================================================

class TestIntegratedKillSwitchChain:
    """2026-07-12 re-review, Finding 1: no committed test exercised the
    COMBINED chain before this -- only bls_metal_batch.is_available() in
    isolation. Reproduces the reviewer's own empirical falsification (a
    mocked-working legacy Metal + EXONIT_INJECTION_METAL=0 used to still
    resolve to 'Metal/MLX') and shows the central injection.py-level check
    now blocks it independent of bls_metal.is_metal_available()'s own
    (now-fixed) kill-switch check."""

    def test_kill_switch_blocks_legacy_metal_even_if_its_own_check_would_not(
            self, monkeypatch):
        # Simulate the exact bug the review found: the legacy per-module
        # check reports "available" (as it incorrectly did pre-fix,
        # ignoring the kill-switch) -- injection.py's own central check
        # must still block the chain from ever trying it.
        monkeypatch.setattr(bls_metal, "is_metal_available", lambda: True)
        monkeypatch.setenv("EXONIT_INJECTION_METAL", "0")
        # This test's original comment ("no CUDA/OpenCL/real-MLX in this
        # sandbox") predates the OpenCL backend having any host with real
        # hardware behind it (spec 2026-07-24: <host>'s Intel Arc iGPU).
        # On such a host the chain would otherwise legitimately resolve to
        # "OpenCL" once Metal is blocked -- that is a correct, intentional
        # consequence of that spec, not a regression here. This test's own
        # concern is narrowly the METAL kill-switch (see the class
        # docstring / Finding 1), so pin OpenCL off too via its own
        # kill-switch to keep the assertion "falls through all the way to
        # CPU" meaningful on every host, Arc-equipped or not.
        monkeypatch.setenv("EXONIT_INJECTION_OPENCL", "0")
        try:
            importlib.reload(injection)
            resolved = injection.get_backend_name()
            assert resolved not in ("Metal/MLX", "Metal/MLX-fused"), (
                f"kill-switch failed to stop the integrated chain -- "
                f"resolved backend: {resolved!r}")
            assert resolved == "CPU", (
                "expected CPU with Metal and OpenCL both kill-switched off "
                f"(no CUDA in this sandbox); got {resolved!r}")
        finally:
            # Revert the mock/env BEFORE the final reload, so the module
            # re-detects the REAL backend for this host -- injection.py's
            # auto-detect runs once at import time, so a mocked reload
            # would otherwise leak into every other test in this session.
            monkeypatch.undo()
            importlib.reload(injection)


# ============================================================
# 3. GPU-glitch -> CPU-retry robustness (no MLX needed — stub batch_fn)
# ============================================================

class TestGpuFailureCpuRetry:
    """injection._run_batched_gpu must retry a failing GPU batch_fn once on
    bls_fast_batch (CPU), log a WARNING (never silent), and keep producing
    the SAME completeness map as a clean CPU-only run (deterministic seeds
    -> identical injections regardless of which backend serves the cell)."""

    def _reference_completeness(self, time, flux, radius_grid, period_grid,
                                time_span):
        return injection._run_batched_gpu(
            time, flux, 1.0, None, radius_grid, period_grid,
            n_injections_per_cell=2, snr_threshold=7.0, time_span=time_span,
            batch_fn=bls_fast_batch, backend_label="CPU (reference)",
            bls_n_periods=200, bls_n_bins=40, bls_n_dur_steps=5)

    def test_retries_on_cpu_and_matches_clean_cpu_run(self, caplog):
        rng = np.random.default_rng(11)
        n = 200
        time = np.linspace(0.0, 40.0, n).astype(np.float64)
        flux = 1.0 + rng.normal(0, 3e-4, n)
        radius_grid = np.array([4.0, 8.0])
        period_grid = np.array([3.0, 6.0])
        time_span = time[-1] - time[0]

        reference = self._reference_completeness(
            time, flux, radius_grid, period_grid, time_span)

        def _flaky_gpu_batch(time_, flux_list, **kwargs):
            raise RuntimeError("simulated Metal command-buffer failure")

        with caplog.at_level(logging.WARNING, logger="exonit"):
            result = injection._run_batched_gpu(
                time, flux, 1.0, None, radius_grid, period_grid,
                n_injections_per_cell=2, snr_threshold=7.0, time_span=time_span,
                batch_fn=_flaky_gpu_batch, backend_label="FakeGPU",
                bls_n_periods=200, bls_n_bins=40, bls_n_dur_steps=5)

        assert result["completeness_map"] == reference["completeness_map"], (
            "GPU-failure CPU-retry must reproduce the clean CPU completeness "
            "map exactly (deterministic seeds)")
        warnings = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        assert any("retrying this cell on CPU" in m for m in warnings), (
            "GPU batch_fn failure must be logged as a WARNING, never silent "
            f"(saw: {warnings})")

    def test_cpu_backend_failure_propagates_not_swallowed(self, monkeypatch):
        """When batch_fn IS the CPU reference itself (no GPU to fall back
        to), a failure must propagate rather than being silently retried
        into the exact same code path."""
        def _broken_bls_fast_batch(time_, flux_list, **kwargs):
            raise RuntimeError("simulated CPU BLS failure")

        monkeypatch.setattr(injection, "bls_fast_batch", _broken_bls_fast_batch)

        rng = np.random.default_rng(12)
        n = 100
        time = np.linspace(0.0, 20.0, n).astype(np.float64)
        flux = 1.0 + rng.normal(0, 3e-4, n)
        radius_grid = np.array([4.0])
        period_grid = np.array([3.0])
        time_span = time[-1] - time[0]

        with pytest.raises(RuntimeError, match="simulated CPU BLS failure"):
            injection._run_batched_gpu(
                time, flux, 1.0, None, radius_grid, period_grid,
                n_injections_per_cell=2, snr_threshold=7.0, time_span=time_span,
                batch_fn=injection.bls_fast_batch, backend_label="CPU",
                bls_n_periods=100, bls_n_bins=30, bls_n_dur_steps=4)


# ============================================================
# 4. is_fused_metal_active() — boolean routing predicate for worker.py
# ============================================================

class TestIsFusedMetalActive:
    """worker.py must route on this predicate, not on a
    get_backend_name() == "Metal/MLX-fused" string comparison (review
    item 4) — a single source of truth inside injection.py."""

    def test_true_when_backend_is_fused_metal(self, monkeypatch):
        monkeypatch.setattr(injection, "_gpu_backend", "Metal/MLX-fused")
        assert injection.is_fused_metal_active() is True

    @pytest.mark.parametrize("label", [
        "Metal/MLX", "CUDA", "OpenCL", "CPU", "CPU (disabled by config)",
    ])
    def test_false_for_every_other_backend_label(self, monkeypatch, label):
        monkeypatch.setattr(injection, "_gpu_backend", label)
        assert injection.is_fused_metal_active() is False


# ============================================================
# 5. Metal threadgroup-memory budget guard on bls_gpu_batch_v2
# ============================================================

@pytest.mark.skipif(not _METAL_MODULE_OK, reason="bls_metal_batch module unavailable")
class TestThreadgroupMemoryBudgetGuard:
    """n_bins drives threadgroup memory (TG_FLUX*(n_bins+1)*4 +
    (n_bins+1)*4 bytes); a host config raising injection_n_bins past the
    ~32 KB Metal limit must fail loudly and specifically, not degrade
    every star on that host to the CPU-retry path with no top-level
    signal. Checked before MLX_AVAILABLE, so it's testable on any host."""

    def test_threadgroup_bytes_formula_matches_docstring(self):
        # From the module docstring / spec: n_bins=200 -> 26,532 B.
        assert bls_metal_batch._threadgroup_bytes(200) == 26532
        # Production default (150) must stay comfortably under budget.
        assert (bls_metal_batch._threadgroup_bytes(150)
                < bls_metal_batch._METAL_THREADGROUP_LIMIT_BYTES)

    def test_oversized_n_bins_raises_before_touching_mlx(self):
        # n_bins=300 -> 4*(301)*(33) = 39,732 B > 32,768 B limit. Must raise
        # regardless of whether MLX is installed on this host (the n_bins
        # check runs before the MLX_AVAILABLE check).
        with pytest.raises(ValueError, match="threadgroup memory"):
            bls_metal_batch.bls_gpu_batch_v2(
                np.array([0.0, 1.0, 2.0]),
                [np.array([1.0, 1.0, 1.0])],
                n_bins=300)

    def test_error_message_names_budget_limit_and_n_bins(self):
        with pytest.raises(ValueError) as exc_info:
            bls_metal_batch.bls_gpu_batch_v2(
                np.array([0.0, 1.0, 2.0]),
                [np.array([1.0, 1.0, 1.0])],
                n_bins=300)
        msg = str(exc_info.value)
        assert "300" in msg
        assert str(bls_metal_batch._METAL_THREADGROUP_LIMIT_BYTES) in msg
        assert str(bls_metal_batch._threadgroup_bytes(300)) in msg


# ============================================================
# 6. GPU-failure circuit breaker (platform-independent — stub batch_fn)
# ============================================================

class TestGpuCircuitBreaker:
    """After _GPU_FAILURE_THRESHOLD consecutive GPU batch_fn failures,
    injection must log ONE ERROR and permanently route straight to the
    CPU batch path for the rest of the process — no more per-cell GPU
    attempts, retries, or WARNINGs. A single success resets the streak."""

    @staticmethod
    def _always_fail(time_, flux_list, **kwargs):
        raise RuntimeError("simulated structurally-broken GPU kernel")

    def _run(self, batch_fn, n_periods_cells=3, backend_label="FakeGPU",
            caplog=None):
        rng = np.random.default_rng(5)
        n = 120
        time = np.linspace(0.0, 30.0, n).astype(np.float64)
        flux = 1.0 + rng.normal(0, 3e-4, n)
        radius_grid = np.array([4.0])
        period_grid = np.linspace(2.0, 8.0, n_periods_cells)
        time_span = time[-1] - time[0]
        return injection._run_batched_gpu(
            time, flux, 1.0, None, radius_grid, period_grid,
            n_injections_per_cell=1, snr_threshold=7.0, time_span=time_span,
            batch_fn=batch_fn, backend_label=backend_label,
            bls_n_periods=100, bls_n_bins=30, bls_n_dur_steps=4)

    def test_trips_after_threshold_consecutive_failures(self, caplog):
        assert injection._GPU_FAILURE_THRESHOLD == 3, (
            "test assumes the documented default threshold of 3")
        with caplog.at_level(logging.WARNING, logger="exonit"):
            self._run(self._always_fail, n_periods_cells=4)

        assert injection.gpu_circuit_is_open() is True
        errors = [r.message for r in caplog.records if r.levelno == logging.ERROR]
        assert len(errors) == 1, f"expected exactly one ERROR, saw: {errors}"
        assert "disabling it for the remainder of this process" in errors[0]
        assert "FakeGPU" in errors[0]

        # 4 period-cells, breaker trips on the 3rd -> the 4th cell must NOT
        # attempt the GPU path again (no 4th WARNING).
        warnings = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 3, (
            f"expected exactly 3 retry warnings (trip on the 3rd failure, "
            f"4th cell skips the GPU attempt entirely), saw: {warnings}")

    def test_single_success_resets_the_streak(self, caplog):
        calls = {"i": 0}
        # fail, fail, SUCCEED (resets streak), fail, fail -> never reaches
        # 3 consecutive failures, breaker must stay closed.
        fail_pattern = [True, True, False, True, True]

        def _intermittent(time_, flux_list, **kwargs):
            i = calls["i"]
            calls["i"] += 1
            if fail_pattern[i % len(fail_pattern)]:
                raise RuntimeError("simulated transient GPU glitch")
            return bls_fast_batch(time_, flux_list, **kwargs)

        with caplog.at_level(logging.WARNING, logger="exonit"):
            self._run(_intermittent, n_periods_cells=len(fail_pattern))

        assert injection.gpu_circuit_is_open() is False
        errors = [r.message for r in caplog.records if r.levelno == logging.ERROR]
        assert errors == [], f"breaker must not trip: saw ERROR(s) {errors}"

    def test_open_circuit_skips_gpu_attempt_on_a_later_call(self, caplog):
        call_count = {"n": 0}

        def _counting_fail(time_, flux_list, **kwargs):
            call_count["n"] += 1
            raise RuntimeError("simulated structurally-broken GPU kernel")

        with caplog.at_level(logging.WARNING, logger="exonit"):
            self._run(_counting_fail, n_periods_cells=3)
        assert call_count["n"] == 3  # attempted the GPU on every cell pre-trip
        assert injection.gpu_circuit_is_open() is True

        call_count["n"] = 0
        caplog.clear()
        # "next star" in the same process, same (still-broken) GPU backend.
        with caplog.at_level(logging.WARNING, logger="exonit"):
            result = self._run(_counting_fail, n_periods_cells=3)

        assert call_count["n"] == 0, (
            "circuit breaker must skip the GPU batch_fn entirely once open "
            "-- it must not even be called")
        assert not any(r.levelno in (logging.WARNING, logging.ERROR)
                       for r in caplog.records), (
            "no further per-cell warnings/errors once the breaker is open")
        # Still produces a correct completeness map via the CPU path.
        assert "completeness_map" in result


# ============================================================
# 7. Real Metal kernel parity (skipped when MLX/Metal unavailable)
# ============================================================

_metal_only = pytest.mark.skipif(
    not _METAL_V2, reason="MLX/Metal fused kernel unavailable on this host")


class TestBlsGpuBatchV2Parity:
    @_metal_only
    def test_period_selection_matches_bls_fast_batch(self):
        time, flux_list, pmin, pmax = _injection_like_batch(n_points=600, seed=21)
        kwargs = dict(period_min=pmin, period_max=pmax, n_periods=2000,
                      n_bins=150, n_dur_steps=10)
        cpu = bls_fast_batch(time, flux_list, **kwargs)
        gpu = bls_metal_batch.bls_gpu_batch_v2(time, flux_list, **kwargs)
        for i, (c, g) in enumerate(zip(cpu, gpu)):
            rel = abs(c["period"] - g["period"]) / c["period"]
            assert rel < 0.05, (
                f"flux #{i}: period mismatch cpu={c['period']:.4f} "
                f"gpu={g['period']:.4f} (rel {rel:.3f})")

    @_metal_only
    def test_e2e_completeness_map_bit_identical(self):
        """The decisive gate from the spec's phase-2 bench: an end-to-end
        _run_batched_gpu completeness map must be BIT-IDENTICAL between the
        CPU and Metal fused-kernel batch_fn (max|Delta| == 0.0), not just
        close."""
        rng = np.random.default_rng(99)
        n = 600
        time = np.linspace(0.0, 27.0, n).astype(np.float64)
        flux = 1.0 + rng.normal(0, 4e-4, n)
        radius_grid = np.array([2.0, 4.0, 8.0])
        period_grid = np.array([2.0, 5.0, 10.0])
        time_span = time[-1] - time[0]

        cpu_result = injection._run_batched_gpu(
            time, flux, 1.0, None, radius_grid, period_grid,
            n_injections_per_cell=3, snr_threshold=7.0, time_span=time_span,
            batch_fn=bls_fast_batch, backend_label="CPU",
            bls_n_periods=2000, bls_n_bins=150, bls_n_dur_steps=10)
        gpu_result = injection._run_batched_gpu(
            time, flux, 1.0, None, radius_grid, period_grid,
            n_injections_per_cell=3, snr_threshold=7.0, time_span=time_span,
            batch_fn=bls_metal_batch.bls_gpu_batch_v2, backend_label="Metal/MLX-fused",
            bls_n_periods=2000, bls_n_bins=150, bls_n_dur_steps=10)

        cpu_map = np.array(cpu_result["completeness_map"])
        gpu_map = np.array(gpu_result["completeness_map"])
        max_delta = float(np.max(np.abs(cpu_map - gpu_map)))
        assert max_delta == 0.0, (
            f"completeness map max|Delta|={max_delta} -- expected 0.0 "
            "(bit-identical, per the 2026-07-11 e2e bench gate)")
