"""BLS-CUDA parity tests — verify bls_gpu / bls_gpu_two_pass match bls_fast,
plus the f32 fused injection-batch kernel (spec
``docs/superpowers/specs/2026-07-11-gpu-bls-f32-unified-kernel.md``, Phase 2).

Two kinds of tests live here:

1. CPU-runnable (no CuPy/CUDA needed — MUST pass on any host, including CI
   and <host> which has no GPU): the phase-fold/bincount host helpers
   (bit-parity vs the numba fold), the f32 kernel's exact arithmetic via
   ``bls_metal_batch.kernel_reference_numpy`` (same kernel design, reused
   rather than duplicated — see that module's docstring point 8 in the
   implementation briefing) against ``bls_fast_batch`` (f64 ground truth),
   and the ``EXONIT_BLS_CUDA_F64`` / shared-memory-budget routing logic in
   ``bls_gpu_batch`` (monkeypatched, no real GPU).
2. GPU-gated (``skipif not is_cuda_available()``): the real CUDA kernel
   against the reference simulator, and the original SDE/grid-fix guards
   from before this file existed (the linear-period grid bug that collapsed
   the noise floor and crushed the SDE peak-to-noise ratio — see commit
   history). These only run on <host> / CUDA hosts.
"""

import numpy as np
import pytest

# sys.path is configured by tests/conftest.py: repo root is prepended and
# worker/ is appended for legacy flat imports like bls_fast/bls_cuda.
from bls_fast import bls_fast, bls_fast_batch, bls_two_pass, _frequency_period_grid

import bls_cuda
import injection
from bls_cuda import bls_gpu, bls_gpu_batch, bls_gpu_two_pass, is_cuda_available

try:
    _CUDA = is_cuda_available()
except Exception:
    _CUDA = False

# The Metal fused-kernel reference module is pure numpy + guarded MLX import
# (see its docstring / module header) and imports cleanly on Linux without
# MLX — reused here per the spec's "one algorithm, two backends" design so
# the f32 arithmetic simulator is not duplicated between the Metal and CUDA
# test suites (single source of truth, same as test_bls_metal_parity.py).
try:
    import bls_metal_batch
    _METAL_BATCH_MODULE_OK = True
except Exception:
    _METAL_BATCH_MODULE_OK = False

_cuda_only = pytest.mark.skipif(not _CUDA, reason="no CUDA GPU available")


# ============================================================
# Synthetic light curve fixture
# ============================================================

@pytest.fixture
def synthetic_lc():
    """30-day TESS-like LC at 2-min cadence with one ~5d transit."""
    rng = np.random.default_rng(20260514)
    cadence_min = 2.0
    duration_days = 30.0
    n = int(duration_days * 1440 / cadence_min)  # ~21600 points
    t = np.linspace(0, duration_days, n)
    f = np.ones_like(t)
    # Inject a transit: P=5.123 d, depth=2e-3, T14=4 hours
    period = 5.123
    t0 = 1.7
    depth = 2.0e-3
    duration = 4.0 / 24.0  # 4 hours in days
    phase = ((t - t0 + 0.5 * period) % period) - 0.5 * period
    in_transit = np.abs(phase) < 0.5 * duration
    f[in_transit] -= depth
    # Add white noise comparable to TESS PDCSAP scatter
    f += rng.normal(0, 5e-4, size=n)
    return t, f, period


# ============================================================
# Sanity: CUDA available + module imports
# ============================================================

@_cuda_only
class TestCUDAAvailable:
    def test_bls_cuda_imports(self):
        assert callable(bls_gpu)
        assert callable(bls_gpu_two_pass)
        assert is_cuda_available()


# ============================================================
# Grid uniformity (the bug)
# ============================================================

@_cuda_only
class TestFrequencyGrid:
    """The SDE collapse came from a linear-in-P grid in bls_gpu. Verify
    that the period grid is now uniform-in-frequency (matches bls_fast)."""

    def test_bls_gpu_period_grid_is_uniform_in_frequency(self, synthetic_lc):
        t, f, _ = synthetic_lc
        r = bls_gpu(t, f, period_min=0.5, period_max=50.0, n_periods=2000)
        periods = np.asarray(r["periods"], dtype=np.float64)
        # Uniform-in-frequency: 1/P should be linearly spaced
        freqs = 1.0 / periods
        diffs = np.diff(freqs)
        # Relative spread of df: should be <1e-10 for a clean linspace in f
        assert np.max(np.abs(diffs - diffs[0]) / np.abs(diffs[0])) < 1e-9, (
            "bls_gpu period grid is NOT uniform-in-frequency — regression of "
            "the SDE collapse bug. See commit history for context."
        )


# ============================================================
# SDE doesn't collapse: peak/median ratio
# ============================================================

@_cuda_only
class TestSDENotCollapsed:
    """Direct guard against the original symptom — a peak that was real
    in the CPU spectrum had SNR ~5 on the buggy GPU vs ~17 on CPU
    because the median of the GPU spectrum was inflated 4-5×."""

    def test_bls_gpu_snr_matches_bls_fast_within_15pct(self, synthetic_lc):
        t, f, _ = synthetic_lc
        r_cpu = bls_fast(t, f, period_min=0.5, period_max=50.0, n_periods=3000)
        r_gpu = bls_gpu(t, f, period_min=0.5, period_max=50.0, n_periods=3000)
        # Same SDE formula, same grid → SNR should be very close (~1%).
        # Allow 15% slack for the synthetic-LC scatter realisation, but
        # the buggy version was ~70% lower so this guard catches regressions.
        rel = abs(r_cpu["snr"] - r_gpu["snr"]) / max(r_cpu["snr"], 1e-9)
        assert rel < 0.15, (
            f"bls_gpu SNR {r_gpu['snr']:.2f} vs bls_fast SNR {r_cpu['snr']:.2f} "
            f"(rel {rel:.3f}) — the buggy grid would put rel ~0.7"
        )


# ============================================================
# Two-pass parity — the production switch
# ============================================================

@_cuda_only
class TestTwoPassParity:
    """bls_gpu_two_pass should match bls_two_pass to the decimals the
    worker prints (period, snr, depth, t0, duration). Validated against
    a synthetic LC in CI; the 50/50 bit-exact A/B over real TESS LCs
    is documented in worker/_perf_poc/."""

    def test_period_matches(self, synthetic_lc):
        t, f, expected_period = synthetic_lc
        r_cpu = bls_two_pass(t, f, period_min=0.5, period_max=50.0, n_periods=20000)
        r_gpu = bls_gpu_two_pass(t, f, period_min=0.5, period_max=50.0, n_periods=20000)
        # Both should find the injected period within the grid resolution
        rel = abs(r_cpu["period"] - r_gpu["period"]) / r_cpu["period"]
        assert rel < 1e-4, (
            f"period mismatch: cpu={r_cpu['period']:.6f} gpu={r_gpu['period']:.6f}"
        )
        # And both should be close to the injected period
        assert abs(r_cpu["period"] - expected_period) / expected_period < 5e-3
        assert abs(r_gpu["period"] - expected_period) / expected_period < 5e-3

    def test_snr_matches(self, synthetic_lc):
        t, f, _ = synthetic_lc
        r_cpu = bls_two_pass(t, f, period_min=0.5, period_max=50.0, n_periods=20000)
        r_gpu = bls_gpu_two_pass(t, f, period_min=0.5, period_max=50.0, n_periods=20000)
        rel = abs(r_cpu["snr"] - r_gpu["snr"]) / max(r_cpu["snr"], 1e-9)
        assert rel < 5e-3, (
            f"SNR drift: cpu={r_cpu['snr']:.4f} gpu={r_gpu['snr']:.4f}"
        )

    def test_depth_matches(self, synthetic_lc):
        t, f, _ = synthetic_lc
        r_cpu = bls_two_pass(t, f, period_min=0.5, period_max=50.0, n_periods=20000)
        r_gpu = bls_gpu_two_pass(t, f, period_min=0.5, period_max=50.0, n_periods=20000)
        rel = abs(r_cpu["depth_ppm"] - r_gpu["depth_ppm"]) / max(r_cpu["depth_ppm"], 1.0)
        assert rel < 5e-3, (
            f"depth drift: cpu={r_cpu['depth_ppm']:.0f}ppm gpu={r_gpu['depth_ppm']:.0f}ppm"
        )


# ============================================================================
# f32 fused injection-batch kernel — spec Phase 2 (bls_gpu_batch)
# ============================================================================
#
# Everything below tests the fused injection-recovery batch path
# (bls_gpu_batch / _bls_gpu_batch_f32), NOT bls_gpu/bls_gpu_two_pass above
# (primary BLS, untouched -- spec Phase 3, later).


def _injection_like_batch(n_points=150, n_flux=4, period_min=2.0,
                          period_max=10.0, seed=0):
    """A handful of fluxes over a shared time grid, each with its own
    injected transit — mirrors test_bls_metal_parity.py's workload builder
    (same design, duplicated rather than imported so this test file stays
    self-contained, matching the existing convention of each backend's test
    module owning its own fixtures)."""
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


# ============================================================
# 1. Fold + bincount host-side helpers: bit-parity vs the numba fold
#    (CPU-only, no CuPy/CUDA needed)
# ============================================================

class TestPhaseFoldAndCountsParity:
    """bls_cuda._phase_fold_cpu / _counts_per_period must reproduce,
    bit-for-bit, the fold+bin numba does inside
    bls_fast._bls_batch_all_periods (ph=(t/P)%1, b=int(ph*n_bins), clipped
    to [0, n_bins)). Uses the standard injection workload dimensions (90
    fluxes, N=600, 5000 periods) — fold/counts only depend on time/periods/
    n_bins, not on the flux values themselves, but the workload is built the
    same way as the e2e/GPU tests below for consistency."""

    def _standard_grid(self, n_bins=150, n_periods=5000, seed=3):
        time, flux_list, pmin, pmax = _injection_like_batch(
            n_points=600, n_flux=90, seed=seed)
        periods = _frequency_period_grid(pmin, pmax, n_periods)
        time_rel = time - time[0]
        return time_rel, periods, n_bins

    def test_phase_fold_matches_numba_reference(self):
        time_rel, periods, n_bins = self._standard_grid()
        bin_idx = bls_cuda._phase_fold_cpu(time_rel, periods, n_bins)

        # Reference: the exact per-sample fold numba does in
        # bls_fast._bls_batch_all_periods (ph=(t/P)%1, b=int(ph*n_bins),
        # clipped) — vectorised per-period here for speed, identical
        # arithmetic order to the scalar numba loop.
        ref = np.empty_like(bin_idx)
        for pi, period in enumerate(periods):
            ph = (time_rel / period) % 1.0
            b = (ph * n_bins).astype(np.int64)
            np.clip(b, 0, n_bins - 1, out=b)
            ref[pi] = b

        assert np.array_equal(bin_idx, ref), (
            "_phase_fold_cpu bin indices differ from the numba fold formula "
            "on the standard 90-flux/N=600/5000-period workload")

    def test_counts_per_period_matches_bincount_reference(self):
        time_rel, periods, n_bins = self._standard_grid()
        n_periods = len(periods)
        bin_idx = bls_cuda._phase_fold_cpu(
            time_rel, periods, n_bins).astype(np.int16)
        counts = bls_cuda._counts_per_period(bin_idx, n_bins)

        ref_counts = np.stack([
            np.bincount(bin_idx[pi].astype(np.int64), minlength=n_bins)
            for pi in range(n_periods)
        ]).astype(np.float32)

        assert np.array_equal(counts, ref_counts), (
            "_counts_per_period bin counts differ from the direct bincount "
            "reference on the standard workload")


# ============================================================
# 2. f32 kernel exact arithmetic vs bls_fast_batch (f64 ground truth)
#    (CPU-only — reuses bls_metal_batch.kernel_reference_numpy, the SAME
#    kernel design per the spec's "one algorithm, two backends", instead of
#    duplicating a second f32 numpy simulator)
# ============================================================

@pytest.mark.skipif(
    not _METAL_BATCH_MODULE_OK,
    reason="bls_metal_batch (home of the shared f32 kernel reference "
           "simulator) unavailable")
class TestF32KernelReferenceParity:
    """CUDA's f32 kernel implements the exact same arithmetic as the Metal
    kernel (spec 'Design — one algorithm, two backends'): fold once in f64,
    exact integer bin counts, per-thread private bin sums in f32, prefix
    sum, then the (width asc, phase asc) loop with strict '>'. Reusing
    bls_metal_batch.kernel_reference_numpy here is therefore not a
    convenience shortcut but the correct regression gate — if a future edit
    made the two kernels diverge, a CUDA-specific simulator would need to
    be split out of this shared one instead."""

    def test_period_selection_matches_bls_fast_batch(self):
        time, flux_list, pmin, pmax = _injection_like_batch(seed=17)
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
        time, flux_list, pmin, pmax = _injection_like_batch(seed=23)
        kwargs = dict(period_min=pmin, period_max=pmax, n_periods=200,
                      n_bins=40, n_dur_steps=5)
        cpu = bls_fast_batch(time, flux_list, **kwargs)
        sim = bls_metal_batch.kernel_reference_numpy(time, flux_list, **kwargs)
        for i, (c, s) in enumerate(zip(cpu, sim)):
            rel_power = abs(c["power"] - s["power"]) / max(abs(c["power"]), 1e-9)
            assert rel_power <= 1e-5, f"flux #{i}: rel power drift {rel_power:.2e}"


# ============================================================
# 3. Shared-memory budget guard + EXONIT_BLS_CUDA_F64 routing
#    (CPU-only — monkeypatched, no real GPU/CuPy needed)
# ============================================================

class TestSharedMemBudgetGuard:
    """_shared_mem_bytes uses the same formula as
    bls_metal_batch._threadgroup_bytes (same kernel design); only the
    per-block byte budget differs (CUDA 48 KB default vs Metal 32 KB)."""

    def test_shared_mem_bytes_matches_spec_values(self):
        # Same numbers the spec/Metal test document for this exact formula
        # (TG_FLUX*(n_bins+1)*4 + (n_bins+1)*4): n_bins=150 -> 19,932 B,
        # n_bins=200 -> 26,532 B.
        assert bls_cuda._shared_mem_bytes(150) == 19932
        assert bls_cuda._shared_mem_bytes(200) == 26532
        assert (bls_cuda._shared_mem_bytes(150)
                < bls_cuda._CUDA_SHARED_MEM_BUDGET_BYTES)
        assert (bls_cuda._shared_mem_bytes(200)
                < bls_cuda._CUDA_SHARED_MEM_BUDGET_BYTES)

    def test_oversized_n_bins_exceeds_budget(self):
        # n_bins=400 -> 132*401 = 52,932 B > 49,152 B (48 KB) default budget.
        assert (bls_cuda._shared_mem_bytes(400)
                > bls_cuda._CUDA_SHARED_MEM_BUDGET_BYTES)


class TestBlsGpuBatchRouting:
    """bls_gpu_batch's f64/f32 routing (EXONIT_BLS_CUDA_F64 env var + the
    shared-memory budget guard + the f32 CPU-fold size ceiling), exercised
    by monkeypatching CUDA_AVAILABLE and stubbing both internal batch
    functions — no CuPy/real GPU needed to test the routing decision
    itself."""

    @pytest.fixture(autouse=True)
    def _reset_size_route_log_once(self):
        """_size_route_logged is a module-level "log once per process" flag
        (see bls_cuda._log_size_route_once) -- reset it around every test in
        this class so one test tripping it doesn't silence the INFO log a
        later test asserts on."""
        bls_cuda._size_route_logged = False
        yield
        bls_cuda._size_route_logged = False

    @staticmethod
    def _patch_stubs(monkeypatch):
        calls = {"f64": 0, "f32": 0}

        def _fake_f64(*a, **k):
            calls["f64"] += 1
            return [{"stub": "f64"}]

        def _fake_f32(*a, **k):
            calls["f32"] += 1
            return [{"stub": "f32"}]

        monkeypatch.setattr(bls_cuda, "CUDA_AVAILABLE", True)
        monkeypatch.setattr(bls_cuda, "_bls_gpu_batch_f64", _fake_f64)
        monkeypatch.setattr(bls_cuda, "_bls_gpu_batch_f32", _fake_f32)
        return calls

    def test_default_routes_to_f32(self, monkeypatch):
        monkeypatch.delenv("EXONIT_BLS_CUDA_F64", raising=False)
        calls = self._patch_stubs(monkeypatch)
        result = bls_cuda.bls_gpu_batch(
            np.array([0.0, 1.0]), [np.array([1.0, 1.0])], n_bins=150)
        assert calls == {"f64": 0, "f32": 1}
        assert result == [{"stub": "f32"}]

    @pytest.mark.parametrize("flag", ["1", "true", "True", "TRUE"])
    def test_env_flag_truthy_variants_route_to_f64(self, monkeypatch, flag):
        monkeypatch.setenv("EXONIT_BLS_CUDA_F64", flag)
        calls = self._patch_stubs(monkeypatch)
        bls_cuda.bls_gpu_batch(
            np.array([0.0, 1.0]), [np.array([1.0, 1.0])], n_bins=150)
        assert calls == {"f64": 1, "f32": 0}

    @pytest.mark.parametrize("flag", ["0", "false", "", "garbage"])
    def test_env_flag_falsy_variants_route_to_f32(self, monkeypatch, flag):
        monkeypatch.setenv("EXONIT_BLS_CUDA_F64", flag)
        calls = self._patch_stubs(monkeypatch)
        bls_cuda.bls_gpu_batch(
            np.array([0.0, 1.0]), [np.array([1.0, 1.0])], n_bins=150)
        assert calls == {"f64": 0, "f32": 1}

    def test_oversized_n_bins_falls_back_to_f64_with_warning(
            self, monkeypatch, caplog):
        import logging
        monkeypatch.delenv("EXONIT_BLS_CUDA_F64", raising=False)
        calls = self._patch_stubs(monkeypatch)
        with caplog.at_level(logging.WARNING):
            bls_cuda.bls_gpu_batch(
                np.array([0.0, 1.0]), [np.array([1.0, 1.0])], n_bins=400)
        assert calls == {"f64": 1, "f32": 0}
        warnings = [r.message for r in caplog.records
                    if r.levelno == logging.WARNING]
        assert any("shared memory" in m for m in warnings), (
            f"expected a WARNING naming the shared-memory budget, saw: "
            f"{warnings}")
        assert any("400" in m for m in warnings)

    def test_env_flag_takes_priority_over_undersized_n_bins(self, monkeypatch):
        """Even when n_bins comfortably fits the f32 budget, the escape
        hatch must still win — it is an explicit override, not just a
        fallback for oversized configs."""
        monkeypatch.setenv("EXONIT_BLS_CUDA_F64", "1")
        calls = self._patch_stubs(monkeypatch)
        bls_cuda.bls_gpu_batch(
            np.array([0.0, 1.0]), [np.array([1.0, 1.0])], n_bins=150)
        assert calls == {"f64": 1, "f32": 0}

    def test_raises_importerror_when_cuda_unavailable(self, monkeypatch):
        monkeypatch.setattr(bls_cuda, "CUDA_AVAILABLE", False)
        with pytest.raises(ImportError):
            bls_cuda.bls_gpu_batch(
                np.array([0.0, 1.0]), [np.array([1.0, 1.0])])

    # -- f32 CPU-fold size ceiling (2026-07-12 <host> A/B regression) --

    def test_just_below_size_threshold_routes_to_f32(self, monkeypatch):
        monkeypatch.delenv("EXONIT_BLS_CUDA_F64", raising=False)
        monkeypatch.delenv("EXONIT_BLS_CUDA_F32_MAX_PTS", raising=False)
        calls = self._patch_stubs(monkeypatch)
        n = bls_cuda._F32_MAX_POINTS_DEFAULT - 1
        bls_cuda.bls_gpu_batch(
            np.zeros(n), [np.zeros(n)], n_bins=150)
        assert calls == {"f64": 0, "f32": 1}

    def test_exactly_at_size_threshold_routes_to_f32(self, monkeypatch):
        """N == _F32_MAX_POINTS_DEFAULT exactly (not +/-1) must still route
        to f32 -- the routing check is a strict '>' (only points STRICTLY
        past the threshold fall back to f64), matching the spec's stated
        intent that 20000 "still has a slight f32 edge" (measured 1.09x).
        The +/-1 tests above cannot catch a '>' -> '>=' regression at this
        exact boundary, since both land on the same side of the threshold
        either way -- this test is the one that would."""
        monkeypatch.delenv("EXONIT_BLS_CUDA_F64", raising=False)
        monkeypatch.delenv("EXONIT_BLS_CUDA_F32_MAX_PTS", raising=False)
        calls = self._patch_stubs(monkeypatch)
        n = bls_cuda._F32_MAX_POINTS_DEFAULT
        bls_cuda.bls_gpu_batch(
            np.zeros(n), [np.zeros(n)], n_bins=150)
        assert calls == {"f64": 0, "f32": 1}

    def test_just_above_size_threshold_routes_to_f64(self, monkeypatch, caplog):
        import logging
        monkeypatch.delenv("EXONIT_BLS_CUDA_F64", raising=False)
        monkeypatch.delenv("EXONIT_BLS_CUDA_F32_MAX_PTS", raising=False)
        calls = self._patch_stubs(monkeypatch)
        n = bls_cuda._F32_MAX_POINTS_DEFAULT + 1
        with caplog.at_level(logging.INFO):
            bls_cuda.bls_gpu_batch(
                np.zeros(n), [np.zeros(n)], n_bins=150)
        assert calls == {"f64": 1, "f32": 0}
        infos = [r.message for r in caplog.records
                 if r.levelno == logging.INFO]
        assert any(str(n) in m and "f64" in m for m in infos), (
            f"expected an INFO log naming the point count and the f64 "
            f"routing decision, saw: {infos}")

    def test_env_max_pts_override_routes_small_workload_to_f64(
            self, monkeypatch):
        """EXONIT_BLS_CUDA_F32_MAX_PTS=50 must make even a tiny (90-point)
        workload route to f64 -- the ceiling is fully env-controlled, not
        just a floor under the hardcoded default."""
        monkeypatch.delenv("EXONIT_BLS_CUDA_F64", raising=False)
        monkeypatch.setenv("EXONIT_BLS_CUDA_F32_MAX_PTS", "50")
        calls = self._patch_stubs(monkeypatch)
        bls_cuda.bls_gpu_batch(
            np.zeros(90), [np.zeros(90)], n_bins=150)
        assert calls == {"f64": 1, "f32": 0}

    def test_invalid_env_max_pts_falls_back_to_default(self, monkeypatch):
        """A malformed EXONIT_BLS_CUDA_F32_MAX_PTS must not crash routing --
        _f32_max_points() falls back to the default, so a small workload
        still gets the f32 speedup rather than silently losing it (or
        raising) on a config typo."""
        monkeypatch.setenv("EXONIT_BLS_CUDA_F32_MAX_PTS", "not-an-int")
        assert bls_cuda._f32_max_points() == bls_cuda._F32_MAX_POINTS_DEFAULT

        monkeypatch.delenv("EXONIT_BLS_CUDA_F64", raising=False)
        calls = self._patch_stubs(monkeypatch)
        bls_cuda.bls_gpu_batch(
            np.zeros(100), [np.zeros(100)], n_bins=150)
        assert calls == {"f64": 0, "f32": 1}

    def test_env_f64_flag_wins_regardless_of_size(self, monkeypatch):
        """EXONIT_BLS_CUDA_F64=1 must win even for a workload far past the
        f32 size ceiling too -- same override, checked first regardless of
        which later check would also have picked f64."""
        monkeypatch.setenv("EXONIT_BLS_CUDA_F64", "1")
        calls = self._patch_stubs(monkeypatch)
        n = bls_cuda._F32_MAX_POINTS_DEFAULT * 10
        bls_cuda.bls_gpu_batch(
            np.zeros(n), [np.zeros(n)], n_bins=150)
        assert calls == {"f64": 1, "f32": 0}


# ============================================================
# 4. Real CUDA kernel parity (skipped when CUDA is unavailable — exercised
#    on <host>, not here)
# ============================================================

@_cuda_only
class TestF32KernelGpuParity:
    """The real CUDA f32 kernel (_bls_gpu_batch_f32 / bls_gpu_batch) against
    the reference simulator and bls_fast_batch, on the standard injection
    workload (N=600, n_bins=150 -- the production defaults)."""

    def test_kernel_matches_reference_simulator(self):
        assert _METAL_BATCH_MODULE_OK, (
            "reference simulator module unavailable even on a CUDA host")
        time, flux_list, pmin, pmax = _injection_like_batch(
            n_points=600, n_flux=48, seed=31)
        kwargs = dict(period_min=pmin, period_max=pmax, n_periods=2000,
                      n_bins=150, n_dur_steps=10)
        sim = bls_metal_batch.kernel_reference_numpy(time, flux_list, **kwargs)
        gpu = bls_cuda._bls_gpu_batch_f32(time, flux_list, **kwargs)
        for i, (s, g) in enumerate(zip(sim, gpu)):
            # Near-tie grid flips between f32 kernel executions are expected
            # and documented (spec: "period match 86/90" on the Metal bench,
            # inherent f32<->f64 rounding at near-tied SR values) -- allow
            # the same 5% period tolerance the Metal GPU parity test uses.
            rel = abs(s["period"] - g["period"]) / s["period"]
            assert rel < 0.05, (
                f"flux #{i}: period mismatch sim={s['period']:.4f} "
                f"gpu={g['period']:.4f} (rel {rel:.3f})")

    def test_e2e_matches_bls_fast_batch(self):
        time, flux_list, pmin, pmax = _injection_like_batch(
            n_points=600, n_flux=48, seed=41)
        kwargs = dict(period_min=pmin, period_max=pmax, n_periods=2000,
                      n_bins=150, n_dur_steps=10)
        cpu = bls_fast_batch(time, flux_list, **kwargs)
        gpu = bls_gpu_batch(time, flux_list, **kwargs)
        for i, (c, g) in enumerate(zip(cpu, gpu)):
            rel_period = abs(c["period"] - g["period"]) / c["period"]
            assert rel_period < 0.05, (
                f"flux #{i}: period mismatch cpu={c['period']:.4f} "
                f"gpu={g['period']:.4f} (rel {rel_period:.3f})")
            rel_depth = abs(c["depth"] - g["depth"]) / max(abs(c["depth"]), 1e-9)
            assert rel_depth < 0.05, (
                f"flux #{i}: depth drift {rel_depth:.3f} "
                f"(cpu={c['depth']:.6e} gpu={g['depth']:.6e})")

    def test_e2e_completeness_map_bit_identical(self):
        """The decisive gate from the spec's Gates section (non-negotiable
        per CLAUDE.md #3): an end-to-end ``injection._run_batched_gpu``
        completeness map must be BIT-IDENTICAL between the CPU and CUDA f32
        fused-kernel batch_fn (max|Delta| == 0.0), not just close. Mirrors
        ``test_bls_metal_parity.py``'s
        ``test_e2e_completeness_map_bit_identical`` line-for-line (same
        workload, same tolerances, same assertion) — the CUDA analogue of
        that test, gated on real CUDA hardware exactly like the Metal one is
        gated on real Metal hardware."""
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
            batch_fn=bls_cuda.bls_gpu_batch, backend_label="CUDA/f32-fused",
            bls_n_periods=2000, bls_n_bins=150, bls_n_dur_steps=10)

        cpu_map = np.array(cpu_result["completeness_map"])
        gpu_map = np.array(gpu_result["completeness_map"])
        max_delta = float(np.max(np.abs(cpu_map - gpu_map)))
        assert max_delta == 0.0, (
            f"completeness map max|Delta|={max_delta} -- expected 0.0 "
            "(bit-identical, per the spec's Gates section)")
