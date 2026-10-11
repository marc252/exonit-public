"""BLS-OpenCL parity tests for the fused injection kernel (worker/bls_opencl.py).

Spec: docs/superpowers/specs/2026-07-24-opencl-injection-backend-arc.md.
Calqued from tests/test_bls_metal_parity.py / tests/test_bls_cuda_parity.py --
same "one algorithm, three backends" design (bls_metal_batch.py / bls_cuda.py
/ bls_opencl.py all implement byte-for-byte the same fused f32 kernel), so
this file reuses the SAME shared reference simulator
(``bls_metal_batch.kernel_reference_numpy``) rather than duplicating a
second f32 numpy simulator, per that module's own docstring rationale.

Tests here fall into three tiers:

1. CPU-only (no OpenCL/hardware needed -- MUST pass anywhere, including a
   host with no GPU at all): the phase-fold/bincount host helpers
   (bit-parity vs the numba fold), the shared f32 kernel simulator
   (``kernel_reference_numpy``) against ``bls_fast_batch`` (f64 ground
   truth) -- including the wrap-boundary and exact-tie fixtures that
   ``test_bls_metal_parity.py``'s 2026-07-12 re-review added specifically
   because noise-based fixtures never exercise them -- and the
   local-memory/work-group budget guard + ``EXONIT_INJECTION_OPENCL``
   kill-switch routing logic (monkeypatched, no real device required).
2. Real-hardware-gated (``skipif not bls_opencl.is_opencl_available()``):
   the actual Arc kernel against the shared simulator and against
   ``bls_fast_batch``, at the standard injection workload (90 fluxes,
   N=600, 5000 periods, n_bins=150 -- spec §5's validation protocol step 2),
   PLUS the wrap-boundary and tie-break fixtures run through the real
   kernel (not just the simulator) -- those are exactly the code paths most
   likely to carry an OpenCL-C-vs-Metal/CUDA-C porting bug, so validating
   them only against the simulator would leave the real kernel unchecked
   for the one thing a translation typo is most likely to break.
3. The decisive e2e gate (spec §5 / Annex, "Gate decisiu"): an end-to-end
   ``injection._run_batched_gpu`` completeness map must be BIT-IDENTICAL
   (max|Delta| == 0.0) between the CPU and OpenCL fused-kernel batch_fn.
"""

import importlib
import threading

import numpy as np
import pytest

from bls_fast import bls_fast_batch, _frequency_period_grid
import injection
import opencl_serial

try:
    import bls_metal_batch
    _METAL_BATCH_MODULE_OK = True
except Exception:
    _METAL_BATCH_MODULE_OK = False

try:
    import bls_opencl
    _OPENCL_MODULE_OK = True
except Exception:
    _OPENCL_MODULE_OK = False

_OPENCL_HW = False
if _OPENCL_MODULE_OK:
    try:
        _OPENCL_HW = bls_opencl.is_opencl_available()
    except Exception:
        _OPENCL_HW = False

# PyOpenCL itself importable inside bls_opencl (bls_opencl.cl bound) --
# narrower than _OPENCL_MODULE_OK, which succeeds even without PyOpenCL
# installed (bls_opencl.py guards its own `import pyopencl` with try/
# except). Tests that monkeypatch ``bls_opencl.cl`` directly need this.
_OPENCL_PY_BOUND = _OPENCL_MODULE_OK and getattr(bls_opencl, "OPENCL_AVAILABLE", False)

_opencl_only = pytest.mark.skipif(
    not _OPENCL_HW, reason="OpenCL device unavailable on this host")


@pytest.fixture(autouse=True)
def _reset_gpu_circuit_breaker():
    """The GPU-failure circuit breaker is process-global state by design
    (it must survive across stars). Isolate every test in this file from
    it regardless of order/failure, so one test's flaky batch_fn can't
    trip the breaker for another. Same fixture as test_bls_metal_parity.py."""
    injection.reset_gpu_circuit_breaker()
    yield
    injection.reset_gpu_circuit_breaker()


# ============================================================
# Synthetic injection-like batch (small, fast — semantics gate, not perf)
# Duplicated from test_bls_metal_parity.py / test_bls_cuda_parity.py by
# convention (each backend's test module owns its own fixtures rather than
# importing them cross-module).
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
    straight ``e <= n_bins`` one. See test_bls_metal_parity.py's docstring
    for the full derivation (identical here — same fixture, same math)."""
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
    """Recover the integer (j, w) bin window from a result dict's physical
    t0/duration fields (see test_bls_metal_parity.py's docstring)."""
    period = result["period"]
    w = int(round(result["duration"] / period * n_bins))
    j = int(round(result["t0"] / period * n_bins)) % n_bins
    return j, w


def _tie_break_batch():
    """An engineered exact SR tie between two (width, phase) candidates --
    see test_bls_metal_parity.py's docstring for the full construction
    (identical here: same integer flux values, same exact-arithmetic
    argument)."""
    n_bins = 12
    period = 1.0
    time = np.concatenate(([0.0], (np.arange(1, n_bins) + 0.5) / n_bins * period))
    flux = np.array([-48., -48., -48., 0., 0., 0.,
                     -48., -48., -48., 0., 0., 0.])
    return time, [flux], period, period  # period_min == period_max


# ============================================================
# 1a. Platform-independent: exact-f32 numpy simulator vs bls_fast_batch
#     (reused shared simulator -- bls_metal_batch.kernel_reference_numpy --
#     same as test_bls_cuda_parity.py's TestF32KernelReferenceParity)
# ============================================================

@pytest.mark.skipif(not _METAL_BATCH_MODULE_OK,
                    reason="bls_metal_batch (shared f32 kernel reference "
                           "simulator) unavailable")
class TestKernelReferenceNumpyParity:
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
        time, flux_list, pmin, pmax = _wrap_boundary_batch()
        kwargs = dict(period_min=pmin, period_max=pmax, n_periods=200,
                      n_bins=40, n_dur_steps=5)
        cpu = bls_fast_batch(time, flux_list, **kwargs)
        sim = bls_metal_batch.kernel_reference_numpy(time, flux_list, **kwargs)

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
        time, flux_list, pmin, pmax = _tie_break_batch()
        kwargs = dict(period_min=pmin, period_max=pmax, n_periods=1,
                      n_bins=12, n_dur_steps=1,
                      duration_min_frac=0.25, duration_max_frac=0.25)
        cpu = bls_fast_batch(time, flux_list, **kwargs)[0]
        sim = bls_metal_batch.kernel_reference_numpy(time, flux_list, **kwargs)[0]

        j_cpu, w_cpu = _recovered_bin_window(cpu, n_bins=12)
        j_sim, w_sim = _recovered_bin_window(sim, n_bins=12)
        assert (j_cpu, w_cpu) == (0, 3), (
            f"fixture bug or tie-break regression: numba picked (j={j_cpu}, "
            f"w={w_cpu}), expected the first-in-order tie winner (0, 3)")
        assert (j_sim, w_sim) == (j_cpu, w_cpu), (
            f"kernel_reference_numpy picked (j={j_sim}, w={w_sim}), numba "
            f"picked (j={j_cpu}, w={w_cpu}) -- tie-break disagreement")


# ============================================================
# 1b. Phase-fold + bincount host helpers: bit-parity vs the numba fold
#     (CPU-only, no OpenCL needed)
# ============================================================

@pytest.mark.skipif(not _OPENCL_MODULE_OK, reason="bls_opencl module unavailable")
class TestPhaseFoldAndCountsParity:
    """bls_opencl._phase_fold_cpu / _counts_per_period must reproduce,
    bit-for-bit, the fold+bin numba does inside
    bls_fast._bls_batch_all_periods. Uses the standard injection workload
    dimensions (90 fluxes, N=600, 5000 periods) -- same convention as
    test_bls_cuda_parity.py's TestPhaseFoldAndCountsParity."""

    def _standard_grid(self, n_bins=150, n_periods=5000, seed=3):
        time, flux_list, pmin, pmax = _injection_like_batch(
            n_points=600, n_flux=90, seed=seed)
        periods = _frequency_period_grid(pmin, pmax, n_periods)
        time_rel = time - time[0]
        return time_rel, periods, n_bins

    def test_phase_fold_matches_numba_reference(self):
        time_rel, periods, n_bins = self._standard_grid()
        bin_idx = bls_opencl._phase_fold_cpu(time_rel, periods, n_bins)

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
        bin_idx = bls_opencl._phase_fold_cpu(
            time_rel, periods, n_bins).astype(np.int16)
        counts = bls_opencl._counts_per_period(bin_idx, n_bins)

        ref_counts = np.stack([
            np.bincount(bin_idx[pi].astype(np.int64), minlength=n_bins)
            for pi in range(n_periods)
        ]).astype(np.float32)

        assert np.array_equal(counts, ref_counts), (
            "_counts_per_period bin counts differ from the direct bincount "
            "reference on the standard workload")


# ============================================================
# 2. Local-memory / work-group budget guard (CPU-only — monkeypatched
#    device queries, no real OpenCL device needed)
# ============================================================

@pytest.mark.skipif(not _OPENCL_MODULE_OK, reason="bls_opencl module unavailable")
class TestLocalMemBudgetGuard:
    """_local_mem_bytes uses the same formula as
    bls_metal_batch._threadgroup_bytes / bls_cuda._shared_mem_bytes (same
    kernel design); unlike those two (fixed 32 KB / 48 KB budgets), OpenCL
    devices vary, so the actual per-device limit is queried at runtime
    (spec §1) -- monkeypatched here so this guard is testable without a
    real device."""

    def test_local_mem_bytes_matches_metal_cuda_formula(self):
        # Same numbers bls_metal_batch/bls_cuda document for this formula.
        assert bls_opencl._local_mem_bytes(150) == 19932
        assert bls_opencl._local_mem_bytes(200) == 26532

    def test_oversized_n_bins_raises_before_touching_buffers(self, monkeypatch):
        monkeypatch.setattr(bls_opencl, "_query_local_mem_size", lambda: 20000)
        monkeypatch.setattr(bls_opencl, "_query_max_work_group_size", lambda: 1024)
        # n_bins=300 -> 39,732 B > fake 20,000 B budget.
        with pytest.raises(ValueError, match="local memory"):
            bls_opencl._check_kernel_limits(300)

    def test_error_message_names_budget_and_n_bins(self, monkeypatch):
        monkeypatch.setattr(bls_opencl, "_query_local_mem_size", lambda: 20000)
        monkeypatch.setattr(bls_opencl, "_query_max_work_group_size", lambda: 1024)
        with pytest.raises(ValueError) as exc_info:
            bls_opencl._check_kernel_limits(300)
        msg = str(exc_info.value)
        assert "300" in msg
        assert "20000" in msg

    def test_within_budget_does_not_raise(self, monkeypatch):
        # Measured 2026-07-24 on <host>'s Arc iGPU: 65536 B / 1024 max WG.
        monkeypatch.setattr(bls_opencl, "_query_local_mem_size", lambda: 65536)
        monkeypatch.setattr(bls_opencl, "_query_max_work_group_size", lambda: 1024)
        bls_opencl._check_kernel_limits(150)  # must not raise

    def test_tg_flux_exceeding_max_work_group_size_raises(self, monkeypatch):
        monkeypatch.setattr(bls_opencl, "_query_local_mem_size", lambda: 65536)
        monkeypatch.setattr(bls_opencl, "_query_max_work_group_size", lambda: 16)
        with pytest.raises(ValueError, match="MAX_WORK_GROUP_SIZE"):
            bls_opencl._check_kernel_limits(150)


# ============================================================
# 2b. f32/f64 throughput crossover constant (spec §5 point 4, validation
#     session 2026-07-24 -- CPU-only, no real device needed: this is pure
#     env-resolution + log-once bookkeeping, mirroring
#     bls_cuda._f32_max_points()'s tests, adapted since bls_opencl has no
#     f64 kernel to route to (see _F32_MAX_POINTS_DEFAULT's docstring
#     comment: purely informational here, not a routing decision).
# ============================================================

@pytest.mark.skipif(not _OPENCL_MODULE_OK, reason="bls_opencl module unavailable")
class TestF32MaxPointsResolution:
    def test_default_matches_measured_crossover(self):
        # Measured 2026-07-24 on <host>'s Arc iGPU (see the block comment
        # above _F32_MAX_POINTS_DEFAULT in worker/bls_opencl.py) -- pins the
        # constant against silent drift, same as the CUDA equivalent test.
        assert bls_opencl._F32_MAX_POINTS_DEFAULT == 600

    def test_no_env_returns_default(self, monkeypatch):
        monkeypatch.delenv("EXONIT_BLS_OPENCL_F32_MAX_PTS", raising=False)
        assert bls_opencl._f32_max_points() == bls_opencl._F32_MAX_POINTS_DEFAULT

    def test_env_override_wins(self, monkeypatch):
        monkeypatch.setenv("EXONIT_BLS_OPENCL_F32_MAX_PTS", "1234")
        assert bls_opencl._f32_max_points() == 1234

    def test_invalid_env_falls_back_to_default(self, monkeypatch):
        """A malformed EXONIT_BLS_OPENCL_F32_MAX_PTS must not crash --
        _f32_max_points() falls back to the default rather than raising
        (same contract as bls_cuda._f32_max_points())."""
        monkeypatch.setenv("EXONIT_BLS_OPENCL_F32_MAX_PTS", "not-an-int")
        assert bls_opencl._f32_max_points() == bls_opencl._F32_MAX_POINTS_DEFAULT

    def test_empty_env_falls_back_to_default(self, monkeypatch):
        monkeypatch.setenv("EXONIT_BLS_OPENCL_F32_MAX_PTS", "   ")
        assert bls_opencl._f32_max_points() == bls_opencl._F32_MAX_POINTS_DEFAULT


@pytest.mark.skipif(not _OPENCL_MODULE_OK, reason="bls_opencl module unavailable")
class TestOversizedLogOnce:
    """_log_oversized_once is purely informational (no CPU-fallback routing
    happens inside bls_opencl.py -- see the design-decision note in its
    docstring): it must fire exactly once per process, at INFO, naming both
    the point count and the threshold."""

    @pytest.fixture(autouse=True)
    def _reset_log_once_flag(self):
        bls_opencl._size_note_logged = False
        yield
        bls_opencl._size_note_logged = False

    def test_fires_once_at_info_with_n_and_threshold(self, caplog):
        import logging
        with caplog.at_level(logging.INFO):
            bls_opencl._log_oversized_once(10000, 600)
        infos = [r.message for r in caplog.records
                 if r.levelno == logging.INFO]
        assert any("10000" in m and "600" in m for m in infos), (
            f"expected an INFO log naming N and the threshold, saw: {infos}")

    def test_does_not_fire_twice(self, caplog):
        import logging
        with caplog.at_level(logging.INFO):
            bls_opencl._log_oversized_once(10000, 600)
            bls_opencl._log_oversized_once(20000, 600)
        infos = [r.message for r in caplog.records
                 if r.levelno == logging.INFO]
        assert len(infos) == 1, (
            f"expected exactly one INFO log across two calls in the same "
            f"process, saw {len(infos)}: {infos}")


# ============================================================
# 3. opencl_serial.py: guard object + EXONIT_INJECTION_OPENCL kill-switch
#    (CPU-only, no OpenCL needed)
# ============================================================

class TestOpenclSerialGuard:
    def test_guard_returns_a_reentrant_lock(self):
        guard = opencl_serial.opencl_guard()
        assert isinstance(guard, type(threading.RLock()))
        # Reentrant: acquiring twice from the same thread must not deadlock.
        with guard:
            with guard:
                pass

    def test_guard_is_the_same_object_every_call(self):
        assert opencl_serial.opencl_guard() is opencl_serial.opencl_guard()

    @pytest.mark.parametrize("flag", ["0", "false", "False", "FALSE", " 0 "])
    def test_disabled_by_env_truthy_variants(self, monkeypatch, flag):
        monkeypatch.setenv("EXONIT_INJECTION_OPENCL", flag)
        assert opencl_serial.injection_opencl_disabled_by_env() is True

    @pytest.mark.parametrize("flag", ["1", "true", "", "garbage"])
    def test_not_disabled_by_other_values(self, monkeypatch, flag):
        monkeypatch.setenv("EXONIT_INJECTION_OPENCL", flag)
        assert opencl_serial.injection_opencl_disabled_by_env() is False

    def test_unset_env_is_not_disabled(self, monkeypatch):
        monkeypatch.delenv("EXONIT_INJECTION_OPENCL", raising=False)
        assert opencl_serial.injection_opencl_disabled_by_env() is False


# ============================================================
# 4. Kill-switch at the bls_opencl module level
# ============================================================

@pytest.mark.skipif(not _OPENCL_MODULE_OK, reason="bls_opencl module unavailable")
class TestKillSwitch:
    @pytest.mark.parametrize("flag", ["0", "false", "False"])
    def test_is_opencl_available_false_when_disabled(self, monkeypatch, flag):
        monkeypatch.setenv("EXONIT_INJECTION_OPENCL", flag)
        assert bls_opencl.is_opencl_available() is False

    def test_bls_gpu_batch_raises_when_disabled(self, monkeypatch):
        monkeypatch.setenv("EXONIT_INJECTION_OPENCL", "0")
        with pytest.raises(RuntimeError, match="EXONIT_INJECTION_OPENCL"):
            bls_opencl.bls_gpu_batch(
                np.array([0.0, 1.0, 2.0]), [np.array([1.0, 1.0, 1.0])])

    def test_bls_gpu_single_flux_raises_when_disabled(self, monkeypatch):
        monkeypatch.setenv("EXONIT_INJECTION_OPENCL", "0")
        with pytest.raises(RuntimeError, match="EXONIT_INJECTION_OPENCL"):
            bls_opencl.bls_gpu(np.array([0.0, 1.0, 2.0]), np.array([1.0, 1.0, 1.0]))

    def test_unset_env_leaves_backend_enabled(self, monkeypatch):
        monkeypatch.delenv("EXONIT_INJECTION_OPENCL", raising=False)
        # Whether the real device is available depends on the host, but the
        # kill-switch itself must not be the reason it's off.
        assert bls_opencl.injection_opencl_disabled_by_env() is False


# ============================================================
# 5. Backward-compat aliases. Spec iteration 2 (2026-07-24) rewired
#    worker/injection.py's auto-detect chain to import the NEW names
#    (bls_gpu/bls_gpu_batch/is_opencl_available) directly -- it no longer
#    uses these legacy aliases. worker/bls.py:87 still does
#    (`from bls_opencl import bls_opencl, _init_opencl`), so the aliases
#    remain load-bearing and must keep working.
# ============================================================

@pytest.mark.skipif(not _OPENCL_MODULE_OK, reason="bls_opencl module unavailable")
class TestLegacyAliases:
    def test_bls_opencl_alias_is_bls_gpu(self):
        assert bls_opencl.bls_opencl is bls_opencl.bls_gpu

    def test_bls_opencl_batch_alias_is_bls_gpu_batch(self):
        assert bls_opencl.bls_opencl_batch is bls_opencl.bls_gpu_batch

    def test_bls_py_still_imports_legacy_names_cleanly(self):
        # worker/bls.py:87 does
        # `from bls_opencl import bls_opencl, _init_opencl` at import time --
        # if that import were broken, importing `bls` would already fail.
        # This asserts the specific names exist with the expected
        # callables, not just that some import succeeded.
        import bls  # noqa: F401 -- import-cleanliness check only
        from bls_opencl import bls_opencl as _legacy_single
        from bls_opencl import _init_opencl as _legacy_init
        assert callable(_legacy_single)
        assert callable(_legacy_init)

    def test_bls_opencl_batch_legacy_name_still_importable(self):
        # No longer imported by injection.py or bls.py, but still a public
        # alias on the module -- must not have silently regressed.
        from bls_opencl import bls_opencl_batch as _legacy_batch
        assert callable(_legacy_batch)


# ============================================================
# 5b. GPU device selection robustness (2026-07-24 review, Finding 1):
#     a broken platform must not hide a working GPU platform listed after
#     it, and "no GPU anywhere" must NEVER fall back to a silently-created
#     CPU OpenCL context. CPU-only -- monkeypatches ``bls_opencl.cl``, no
#     real device required.
# ============================================================

@pytest.mark.skipif(not _OPENCL_PY_BOUND,
                    reason="PyOpenCL not importable (bls_opencl.cl unbound)")
class TestSelectGpuDeviceRobustness:
    class _FakeDevice:
        def __init__(self, gpu):
            self.type = (bls_opencl.cl.device_type.GPU if gpu
                        else bls_opencl.cl.device_type.CPU)

    class _FakePlatformRaises:
        name = "Broken CPU ICD"

        def get_devices(self):
            raise RuntimeError("simulated broken pocl/rusticl enumeration")

    class _FakePlatformWithDevices:
        def __init__(self, name, devices):
            self.name = name
            self._devices = devices

        def get_devices(self):
            return self._devices

    def test_broken_platform_does_not_abort_search_for_later_gpu(
            self, monkeypatch):
        """A platform whose get_devices() raises must be skipped (logged),
        not abort the whole search -- a GPU platform listed AFTER it must
        still be found."""
        gpu_device = self._FakeDevice(gpu=True)
        fake_platforms = [
            self._FakePlatformRaises(),
            self._FakePlatformWithDevices("Intel(R) OpenCL Graphics",
                                          [gpu_device]),
        ]
        monkeypatch.setattr(bls_opencl.cl, "get_platforms",
                            lambda: fake_platforms)
        found = bls_opencl._select_gpu_device()
        assert found is gpu_device

    def test_no_gpu_anywhere_returns_none(self, monkeypatch):
        cpu_only = self._FakePlatformWithDevices(
            "pocl", [self._FakeDevice(gpu=False)])
        monkeypatch.setattr(bls_opencl.cl, "get_platforms", lambda: [cpu_only])
        assert bls_opencl._select_gpu_device() is None

    def test_get_platforms_itself_raising_returns_none(self, monkeypatch):
        def _raise():
            raise RuntimeError("simulated ICD loader failure")
        monkeypatch.setattr(bls_opencl.cl, "get_platforms", _raise)
        assert bls_opencl._select_gpu_device() is None

    def test_init_opencl_returns_false_and_never_calls_create_some_context(
            self, monkeypatch):
        """The core Finding 1 regression gate: when no GPU device is found,
        _init_opencl must return False and must NEVER call
        cl.create_some_context (that would silently hand back a CPU OpenCL
        context -- exactly the fallback CLAUDE.md #2 forbids)."""
        monkeypatch.setattr(bls_opencl, "OPENCL_AVAILABLE", True)
        monkeypatch.setattr(bls_opencl, "_ctx", None)
        monkeypatch.setattr(bls_opencl, "_queue", None)
        monkeypatch.setattr(bls_opencl, "_select_gpu_device", lambda: None)

        def _forbidden_create_some_context(*a, **k):
            raise AssertionError(
                "_init_opencl must NEVER call create_some_context when no "
                "GPU device was found (2026-07-24 review, Finding 1)")
        monkeypatch.setattr(bls_opencl.cl, "create_some_context",
                            _forbidden_create_some_context)

        assert bls_opencl._init_opencl() is False
        assert bls_opencl._ctx is None

    def test_init_opencl_uses_the_selected_gpu_device_when_found(
            self, monkeypatch):
        """Sanity check the opposite branch: a device found by
        _select_gpu_device is passed straight to cl.Context, never through
        create_some_context."""
        sentinel_device = object()
        captured = {}

        class _FakeContext:
            def __init__(self, devices):
                captured["devices"] = devices

        class _FakeQueue:
            def __init__(self, ctx):
                captured["ctx"] = ctx

        monkeypatch.setattr(bls_opencl, "OPENCL_AVAILABLE", True)
        monkeypatch.setattr(bls_opencl, "_ctx", None)
        monkeypatch.setattr(bls_opencl, "_queue", None)
        monkeypatch.setattr(bls_opencl, "_select_gpu_device",
                            lambda: sentinel_device)
        monkeypatch.setattr(bls_opencl.cl, "Context", _FakeContext)
        monkeypatch.setattr(bls_opencl.cl, "CommandQueue", _FakeQueue)

        assert bls_opencl._init_opencl() is True
        assert captured["devices"] == [sentinel_device]
        assert captured["ctx"] is bls_opencl._ctx


# ============================================================
# 6. Real Arc hardware parity (skipped when OpenCL/Arc unavailable)
# ============================================================

class TestBlsGpuBatchHardwareParity:
    """The real fused OpenCL kernel against the shared simulator and
    against bls_fast_batch, at the standard injection workload (spec §5
    validation protocol step 2: "parity cell 90 fluxos/N=600/5000 periodes
    a l'Arc real")."""

    @_opencl_only
    def test_period_selection_matches_bls_fast_batch(self):
        time, flux_list, pmin, pmax = _injection_like_batch(
            n_points=600, n_flux=90, seed=21)
        kwargs = dict(period_min=pmin, period_max=pmax, n_periods=5000,
                      n_bins=150, n_dur_steps=10)
        cpu = bls_fast_batch(time, flux_list, **kwargs)
        gpu = bls_opencl.bls_gpu_batch(time, flux_list, **kwargs)
        for i, (c, g) in enumerate(zip(cpu, gpu)):
            rel = abs(c["period"] - g["period"]) / c["period"]
            assert rel < 0.05, (
                f"flux #{i}: period mismatch cpu={c['period']:.4f} "
                f"gpu={g['period']:.4f} (rel {rel:.3f})")

    @_opencl_only
    @pytest.mark.skipif(not _METAL_BATCH_MODULE_OK,
                        reason="shared f32 kernel reference simulator unavailable")
    def test_value_tolerances_vs_shared_simulator(self):
        """Real Arc hardware vs the exact-f32 numpy simulator, at the
        standard workload, with the spec's stated tolerances (rel_power<
        1e-4, rel_depth<1e-3, rel_snr<5e-2). The simulator implements the
        SAME kernel arithmetic (see bls_metal_batch.kernel_reference_numpy's
        docstring), so real-hardware drift here should be sequencing/rounding
        noise only, not an algorithmic difference -- a wider drift than this
        would point at an OpenCL-C porting bug, not expected f32 rounding."""
        time, flux_list, pmin, pmax = _injection_like_batch(
            n_points=600, n_flux=90, seed=33)
        kwargs = dict(period_min=pmin, period_max=pmax, n_periods=5000,
                      n_bins=150, n_dur_steps=10)
        sim = bls_metal_batch.kernel_reference_numpy(time, flux_list, **kwargs)
        gpu = bls_opencl.bls_gpu_batch(time, flux_list, **kwargs)
        for i, (s, g) in enumerate(zip(sim, gpu)):
            rel_period = abs(s["period"] - g["period"]) / s["period"]
            assert rel_period < 0.05, (
                f"flux #{i}: period mismatch sim={s['period']:.4f} "
                f"gpu={g['period']:.4f} (rel {rel_period:.3f})")
            rel_power = abs(s["power"] - g["power"]) / max(abs(s["power"]), 1e-9)
            assert rel_power < 1e-4, f"flux #{i}: rel power drift {rel_power:.2e}"
            rel_depth = abs(s["depth"] - g["depth"]) / max(abs(s["depth"]), 1e-9)
            assert rel_depth < 1e-3, f"flux #{i}: rel depth drift {rel_depth:.2e}"
            rel_snr = abs(s["snr"] - g["snr"]) / max(abs(s["snr"]), 1e-9)
            assert rel_snr < 5e-2, f"flux #{i}: rel snr drift {rel_snr:.2e}"

    @_opencl_only
    def test_wrap_branch_on_real_hardware(self):
        """The wrap-boundary fixture run through the REAL kernel (not just
        the simulator) -- the branch most likely to carry an OpenCL-C
        porting typo (e.g. a wrong `n_bins` vs `n_bins - 1` index)."""
        time, flux_list, pmin, pmax = _wrap_boundary_batch()
        kwargs = dict(period_min=pmin, period_max=pmax, n_periods=200,
                      n_bins=40, n_dur_steps=5)
        cpu = bls_fast_batch(time, flux_list, **kwargs)
        gpu = bls_opencl.bls_gpu_batch(time, flux_list, **kwargs)

        j_cpu, w_cpu = _recovered_bin_window(cpu[0], n_bins=40)
        j_gpu, w_gpu = _recovered_bin_window(gpu[0], n_bins=40)
        assert j_cpu + w_cpu > 40, (
            f"fixture bug: cpu winning window (j={j_cpu}, w={w_cpu}) does "
            "not cross the phase-wrap boundary")
        assert j_gpu + w_gpu > 40, (
            f"OpenCL kernel did not select the wrap-boundary window "
            f"(j={j_gpu}, w={w_gpu}) -- possible wrap-branch porting bug")

        c, g = cpu[0], gpu[0]
        rel_power = abs(c["power"] - g["power"]) / max(abs(c["power"]), 1e-9)
        assert rel_power < 1e-3, f"rel power drift {rel_power:.2e}"

    @_opencl_only
    def test_tie_break_on_real_hardware_matches_numba_first_in_order_winner(self):
        """The exact-tie fixture run through the REAL kernel -- confirms the
        strict '>' comparison (not '>=') survived the OpenCL-C port."""
        time, flux_list, pmin, pmax = _tie_break_batch()
        kwargs = dict(period_min=pmin, period_max=pmax, n_periods=1,
                      n_bins=12, n_dur_steps=1,
                      duration_min_frac=0.25, duration_max_frac=0.25)
        cpu = bls_fast_batch(time, flux_list, **kwargs)[0]
        gpu = bls_opencl.bls_gpu_batch(time, flux_list, **kwargs)[0]

        j_cpu, w_cpu = _recovered_bin_window(cpu, n_bins=12)
        j_gpu, w_gpu = _recovered_bin_window(gpu, n_bins=12)
        assert (j_cpu, w_cpu) == (0, 3)
        assert (j_gpu, w_gpu) == (0, 3), (
            f"OpenCL kernel picked (j={j_gpu}, w={w_gpu}), expected the "
            "first-in-order tie winner (0, 3) -- possible '>' -> '>=' "
            "tie-break regression in the port")

    @_opencl_only
    def test_oversized_n_logs_info_but_still_executes_on_gpu(self, caplog):
        """N above _f32_max_points() logs the informational (not routing)
        note from bls_gpu_batch's real call path, and the call still
        completes on the GPU (no CPU-fallback wiring exists inside this
        module -- see _F32_MAX_POINTS_DEFAULT's docstring)."""
        import logging
        bls_opencl._size_note_logged = False
        n = bls_opencl._F32_MAX_POINTS_DEFAULT + 200
        rng = np.random.default_rng(5)
        time = np.linspace(0.0, 30.0, n).astype(np.float64)
        flux = 1.0 + rng.normal(0, 3e-4, n)
        with caplog.at_level(logging.INFO):
            result = bls_opencl.bls_gpu_batch(
                time, [flux], n_periods=50, n_bins=40, n_dur_steps=3)
        assert len(result) == 1
        infos = [r.message for r in caplog.records
                 if r.levelno == logging.INFO]
        assert any(str(n) in m for m in infos), (
            f"expected an INFO log naming N={n}, saw: {infos}")
        bls_opencl._size_note_logged = False


# ============================================================
# 7. Decisive gate: e2e completeness map bit-identical
# ============================================================

class TestE2eCompletenessMapBitIdentical:
    @_opencl_only
    def test_e2e_completeness_map_bit_identical(self):
        """The decisive gate from the spec's §5 ("Gate decisiu"): an
        end-to-end _run_batched_gpu completeness map must be BIT-IDENTICAL
        between the CPU and OpenCL fused-kernel batch_fn (max|Delta| ==
        0.0), not just close. Mirrors test_bls_metal_parity.py's /
        test_bls_cuda_parity.py's test_e2e_completeness_map_bit_identical
        line-for-line (same workload, same tolerances, same assertion) --
        the OpenCL analogue, gated on the real Arc device exactly like the
        other two are gated on their own hardware."""
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
            batch_fn=bls_opencl.bls_gpu_batch, backend_label="OpenCL/Arc-fused",
            bls_n_periods=2000, bls_n_bins=150, bls_n_dur_steps=10)

        cpu_map = np.array(cpu_result["completeness_map"])
        gpu_map = np.array(gpu_result["completeness_map"])
        max_delta = float(np.max(np.abs(cpu_map - gpu_map)))
        assert max_delta == 0.0, (
            f"completeness map max|Delta|={max_delta} -- expected 0.0 "
            "(bit-identical, per the spec's Gate decisiu)")


# ============================================================
# 8. Integrated fallback chain: injection.py's module-level auto-detect
#    (spec 2026-07-24 iteration 2 -- the rewiring itself). Mirrors
#    test_bls_metal_parity.py's TestIntegratedKillSwitchChain, OpenCL side.
#    Before iteration 2, injection.py's OpenCL hook gated on the legacy
#    _init_opencl() (context-creation only, no real submission) -- these
#    tests exercise the NEW hook (is_opencl_available(), a real minimal
#    kernel submission) at the injection.py auto-detect level, not just
#    inside bls_opencl.py in isolation (tier 4 above).
# ============================================================

def _force_no_cuda_or_metal(monkeypatch):
    """Monkeypatch every non-OpenCL GPU backend's own availability check to
    False, so the chain's OpenCL branch is exercised deterministically
    regardless of what hardware happens to be attached to the host running
    this test (<host> has no CUDA/Metal today, but pinning this explicitly
    keeps the test meaningful if that ever changes)."""
    import bls_cuda
    monkeypatch.setattr(bls_cuda, "is_cuda_available", lambda: False)
    if _METAL_BATCH_MODULE_OK:
        monkeypatch.setattr(bls_metal_batch, "is_available", lambda: False)
    try:
        import bls_metal
        monkeypatch.setattr(bls_metal, "is_metal_available", lambda: False)
    except Exception:
        pass


class TestIntegratedOpenclChain:
    @_opencl_only
    def test_resolves_to_opencl_when_no_cuda_or_metal(self, monkeypatch):
        """With real OpenCL available (Arc, <host>) and CUDA/Metal forced
        unavailable, injection.py's auto-detect chain must resolve
        get_backend_name() == "OpenCL" -- the exact end-to-end wiring
        iteration 1 left untested (it only covered bls_opencl.
        is_opencl_available() in isolation, not injection.py's hook that
        calls it)."""
        _force_no_cuda_or_metal(monkeypatch)
        try:
            importlib.reload(injection)
            resolved = injection.get_backend_name()
            assert resolved == "OpenCL", (
                f"expected OpenCL to be selected with no CUDA/Metal "
                f"available on a real-OpenCL host; got {resolved!r}")
        finally:
            # Revert the mocks BEFORE the final reload, so the module
            # re-detects the REAL backend for this host afterward --
            # injection.py's auto-detect runs once at import time, so a
            # mocked reload would otherwise leak into every other test in
            # this session (same discipline as test_bls_metal_parity.py's
            # equivalent chain test).
            monkeypatch.undo()
            importlib.reload(injection)

    def test_kill_switch_skips_opencl_and_falls_through_to_cpu(self, monkeypatch):
        """EXONIT_INJECTION_OPENCL=0 must make the auto-detect chain skip
        OpenCL entirely -- the central, defense-in-depth check added to
        injection.py in iteration 2 (mirrors the pre-existing Metal
        kill-switch's own integrated test in test_bls_metal_parity.py).
        Runs on EVERY host (not hardware-gated): with CUDA/Metal forced
        unavailable and OpenCL kill-switched, the chain has nothing left
        but CPU, independent of whether OpenCL hardware is actually
        present here."""
        _force_no_cuda_or_metal(monkeypatch)
        monkeypatch.setenv("EXONIT_INJECTION_OPENCL", "0")
        try:
            importlib.reload(injection)
            resolved = injection.get_backend_name()
            assert resolved != "OpenCL", (
                f"kill-switch failed to stop the integrated chain -- "
                f"resolved backend: {resolved!r}")
            assert resolved == "CPU", (
                f"expected CPU with CUDA/Metal disabled and OpenCL "
                f"kill-switched; got {resolved!r}")
        finally:
            monkeypatch.undo()
            importlib.reload(injection)
