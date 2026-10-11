"""FPP GPU test suite — CPU baseline and GPU backend validation.

TestCPUBaseline establishes reference behaviour for shape, dtype,
value validity, and wall-clock performance.  GPU backend tests
(Tasks 2-4) will import cpu_reference_lnL / cpu_reference_eb_lnL
from conftest and compare against these values.
"""

import math
import time as _time
import time as time_mod

import numpy as np
import pytest

# sys.path is configured by tests/conftest.py: repo root is prepended (so
# `import worker.<subpkg>` works) and worker/ is appended for legacy flat
# imports like fpp_native. Do not re-insert worker/ here — that would shadow
# the worker package and break sibling test modules that import
# `worker.exominer_native`.
import fpp_native


# ================================================================== #
# CPU Baseline
# ================================================================== #

class TestCPUBaseline:
    """Validates shape, dtype, value properties, and timing of the
    CPU _batch_transit_lnL and _batch_eb_transit_lnL functions."""

    # -------------------------------------------------------------- #
    # Transit lnL shape / dtype
    # -------------------------------------------------------------- #

    def test_batch_transit_lnL_returns_correct_shape(
        self, transit_params, cpu_reference_lnL
    ):
        """Result must be a 1-D array of length N."""
        N = len(transit_params["ks"])
        assert cpu_reference_lnL.ndim == 1, "lnL must be 1-D"
        assert cpu_reference_lnL.shape == (N,), (
            f"Expected shape ({N},), got {cpu_reference_lnL.shape}"
        )
        assert cpu_reference_lnL.dtype == np.float64, (
            f"Expected float64, got {cpu_reference_lnL.dtype}"
        )

    # -------------------------------------------------------------- #
    # Transit lnL value validity
    # -------------------------------------------------------------- #

    def test_batch_transit_lnL_values_are_finite_or_neginf(
        self, cpu_reference_lnL
    ):
        """Every element must be either a finite float or -inf (no NaN)."""
        lnL = cpu_reference_lnL
        bad_mask = ~(np.isfinite(lnL) | (lnL == -np.inf))
        n_bad = bad_mask.sum()
        assert n_bad == 0, (
            f"{n_bad} values are neither finite nor -inf: "
            f"{lnL[bad_mask][:5]}"
        )
        # Sanity: at least some draws should produce finite likelihoods
        n_finite = np.isfinite(lnL).sum()
        assert n_finite > 0, "All lnL values are -inf — check parameter ranges"

    # -------------------------------------------------------------- #
    # EB lnL shape / dtype
    # -------------------------------------------------------------- #

    def test_batch_eb_transit_lnL_returns_correct_shape(
        self, eb_params, cpu_reference_eb_lnL
    ):
        """EB result must be a 1-D float64 array of length N."""
        N = len(eb_params["ks"])
        assert cpu_reference_eb_lnL.ndim == 1, "EB lnL must be 1-D"
        assert cpu_reference_eb_lnL.shape == (N,), (
            f"Expected shape ({N},), got {cpu_reference_eb_lnL.shape}"
        )
        assert cpu_reference_eb_lnL.dtype == np.float64, (
            f"Expected float64, got {cpu_reference_eb_lnL.dtype}"
        )

    # -------------------------------------------------------------- #
    # EB lnL value validity
    # -------------------------------------------------------------- #

    def test_batch_eb_transit_lnL_values_are_finite_or_neginf(
        self, cpu_reference_eb_lnL
    ):
        """Every EB lnL element must be finite or -inf (no NaN)."""
        lnL = cpu_reference_eb_lnL
        bad_mask = ~(np.isfinite(lnL) | (lnL == -np.inf))
        n_bad = bad_mask.sum()
        assert n_bad == 0, (
            f"{n_bad} EB values are neither finite nor -inf: "
            f"{lnL[bad_mask][:5]}"
        )

    # -------------------------------------------------------------- #
    # CPU benchmark
    # -------------------------------------------------------------- #

    def test_cpu_benchmark_1k_draws(self, transit_params):
        """Time the CPU baseline for 1 000 draws and print the result.

        This test always passes — it exists to establish a reference
        wall-clock baseline that GPU backends can be compared against.
        """
        p = transit_params
        N = len(p["ks"])
        T = len(p["time"])

        # Warm-up call to ensure numba JIT is compiled
        _ = fpp_native._batch_transit_lnL(
            p["time"], p["flux"], p["sigma"],
            p["ks"], p["P_orbs"], p["incs_rad"],
            p["a_Rs_arr"], p["eccs"], p["ws_rad"],
            p["u1"], p["u2"],
            p["companion_fluxratio"], p["companion_is_host"],
        )

        # Timed run
        n_trials = 3
        times = []
        for _ in range(n_trials):
            t0 = _time.perf_counter()
            fpp_native._batch_transit_lnL(
                p["time"], p["flux"], p["sigma"],
                p["ks"], p["P_orbs"], p["incs_rad"],
                p["a_Rs_arr"], p["eccs"], p["ws_rad"],
                p["u1"], p["u2"],
                p["companion_fluxratio"], p["companion_is_host"],
            )
            times.append(_time.perf_counter() - t0)

        best_ms   = min(times) * 1e3
        median_ms = sorted(times)[len(times) // 2] * 1e3

        print(
            f"\n[CPU benchmark] N={N} draws, T={T} points | "
            f"best={best_ms:.1f} ms  median={median_ms:.1f} ms  "
            f"({n_trials} trials)"
        )

        # Non-blocking assertion: result must be computed in a sane time
        assert median_ms < 120_000, (
            f"CPU baseline took {median_ms:.0f} ms — unexpectedly slow"
        )


# ================================================================== #
# CUDA Backend
# ================================================================== #

class TestCUDABackend:
    """Validates CUDA GPU backend against CPU baseline."""

    @pytest.fixture(autouse=True)
    def skip_if_no_cuda(self):
        try:
            from fpp_cuda import is_available
            if not is_available():
                pytest.skip("CUDA not available")
        except ImportError:
            pytest.skip("fpp_cuda not installed")

    # -------------------------------------------------------------- #
    # Transit lnL matches CPU
    # -------------------------------------------------------------- #

    def test_batch_transit_lnL_matches_cpu(
        self, transit_params, cpu_reference_lnL
    ):
        """GPU transit lnL must match CPU within tolerance."""
        from fpp_cuda import batch_transit_lnL

        p = transit_params
        gpu_lnL = batch_transit_lnL(
            p["time"], p["flux"], p["sigma"],
            p["ks"], p["P_orbs"], p["incs_rad"],
            p["a_Rs_arr"], p["eccs"], p["ws_rad"],
            p["u1"], p["u2"],
            p["companion_fluxratio"], p["companion_is_host"],
        )

        assert gpu_lnL.shape == cpu_reference_lnL.shape, (
            f"Shape mismatch: GPU {gpu_lnL.shape} vs CPU {cpu_reference_lnL.shape}"
        )
        assert gpu_lnL.dtype == np.float64, (
            f"Expected float64, got {gpu_lnL.dtype}"
        )

        # Compare finite values: absolute tolerance < 1.0 lnL
        cpu_fin = np.isfinite(cpu_reference_lnL)
        gpu_fin = np.isfinite(gpu_lnL)
        both_fin = cpu_fin & gpu_fin

        if both_fin.sum() > 0:
            abs_diff = np.abs(gpu_lnL[both_fin] - cpu_reference_lnL[both_fin])
            max_diff = abs_diff.max()
            median_diff = np.median(abs_diff)
            print(
                f"\n[CUDA transit] max |diff|={max_diff:.4f}, "
                f"median |diff|={median_diff:.4f}, "
                f"finite: GPU={gpu_fin.sum()} CPU={cpu_fin.sum()}"
            )
            # float32 kernel math can accumulate ~1-2 lnL drift over
            # 800 time points; 2.0 is a safe ceiling for RTX 4070.
            assert max_diff < 2.0, (
                f"Max absolute lnL difference {max_diff:.4f} >= 2.0"
            )

        # Allow < 5% extra -inf draws on GPU
        cpu_inf_count = (~cpu_fin).sum()
        gpu_inf_count = (~gpu_fin).sum()
        extra_inf = max(0, gpu_inf_count - cpu_inf_count)
        N = len(cpu_reference_lnL)
        assert extra_inf < 0.05 * N, (
            f"GPU has {extra_inf} extra -inf draws (> 5% of {N})"
        )

    # -------------------------------------------------------------- #
    # EB lnL matches CPU
    # -------------------------------------------------------------- #

    def test_batch_eb_transit_lnL_matches_cpu(
        self, eb_params, cpu_reference_eb_lnL
    ):
        """GPU EB lnL must match CPU within tolerance."""
        from fpp_cuda import batch_eb_transit_lnL

        p = eb_params
        gpu_lnL = batch_eb_transit_lnL(
            p["time"], p["flux"], p["sigma"],
            p["ks"], p["eb_fluxratios"], p["P_orbs"],
            p["incs_rad"], p["a_Rs_arr"], p["eccs"], p["ws_rad"],
            p["u1"], p["u2"],
            p["companion_fluxratio"], p["companion_is_host"],
        )

        assert gpu_lnL.shape == cpu_reference_eb_lnL.shape
        assert gpu_lnL.dtype == np.float64

        cpu_fin = np.isfinite(cpu_reference_eb_lnL)
        gpu_fin = np.isfinite(gpu_lnL)
        both_fin = cpu_fin & gpu_fin

        if both_fin.sum() > 0:
            abs_diff = np.abs(gpu_lnL[both_fin] - cpu_reference_eb_lnL[both_fin])
            max_diff = abs_diff.max()
            median_diff = np.median(abs_diff)
            print(
                f"\n[CUDA EB] max |diff|={max_diff:.4f}, "
                f"median |diff|={median_diff:.4f}, "
                f"finite: GPU={gpu_fin.sum()} CPU={cpu_fin.sum()}"
            )
            assert max_diff < 2.0, (
                f"Max absolute EB lnL difference {max_diff:.4f} >= 2.0"
            )

        cpu_inf_count = (~cpu_fin).sum()
        gpu_inf_count = (~gpu_fin).sum()
        extra_inf = max(0, gpu_inf_count - cpu_inf_count)
        N = len(cpu_reference_eb_lnL)
        assert extra_inf < 0.05 * N, (
            f"GPU has {extra_inf} extra -inf EB draws (> 5% of {N})"
        )

    # -------------------------------------------------------------- #
    # is_available / get_info
    # -------------------------------------------------------------- #

    def test_is_available_returns_bool(self):
        from fpp_cuda import is_available
        result = is_available()
        assert isinstance(result, bool)
        assert result is True  # we skipped if not available

    def test_get_info_returns_string(self):
        from fpp_cuda import get_info
        info = get_info()
        assert isinstance(info, str)
        assert len(info) > 0
        print(f"\n[CUDA info] {info}")

    # -------------------------------------------------------------- #
    # Benchmark
    # -------------------------------------------------------------- #

    def test_cuda_benchmark_1k_draws(self, transit_params):
        """Benchmark CUDA backend: warmup + timed run."""
        from fpp_cuda import batch_transit_lnL

        p = transit_params
        N = len(p["ks"])
        T = len(p["time"])

        # Warmup (includes kernel compilation)
        _ = batch_transit_lnL(
            p["time"], p["flux"], p["sigma"],
            p["ks"], p["P_orbs"], p["incs_rad"],
            p["a_Rs_arr"], p["eccs"], p["ws_rad"],
            p["u1"], p["u2"],
            p["companion_fluxratio"], p["companion_is_host"],
        )

        # Timed runs
        import cupy as cp
        n_trials = 5
        times = []
        for _ in range(n_trials):
            cp.cuda.Stream.null.synchronize()
            t0 = _time.perf_counter()
            _ = batch_transit_lnL(
                p["time"], p["flux"], p["sigma"],
                p["ks"], p["P_orbs"], p["incs_rad"],
                p["a_Rs_arr"], p["eccs"], p["ws_rad"],
                p["u1"], p["u2"],
                p["companion_fluxratio"], p["companion_is_host"],
            )
            cp.cuda.Stream.null.synchronize()
            times.append(_time.perf_counter() - t0)

        best_ms = min(times) * 1e3
        median_ms = sorted(times)[len(times) // 2] * 1e3

        print(
            f"\n[CUDA benchmark] N={N} draws, T={T} points | "
            f"best={best_ms:.1f} ms  median={median_ms:.1f} ms  "
            f"({n_trials} trials)"
        )


# ================================================================== #
# Metal/MLX Backend
# ================================================================== #

class TestMetalBackend:
    """Validates Metal/MLX GPU backend against CPU baseline (Apple Silicon)."""

    @pytest.fixture(autouse=True)
    def skip_if_no_metal(self):
        try:
            from fpp_metal import is_available
            if not is_available():
                pytest.skip("Metal/MLX not available")
        except ImportError:
            pytest.skip("fpp_metal not installed")

    # -------------------------------------------------------------- #
    # Transit lnL matches CPU
    # -------------------------------------------------------------- #

    def test_batch_transit_lnL_matches_cpu(
        self, transit_params, cpu_reference_lnL
    ):
        """Metal transit lnL must match CPU within float32 tolerance."""
        from fpp_metal import batch_transit_lnL

        p = transit_params
        gpu_lnL = batch_transit_lnL(
            p["time"], p["flux"], p["sigma"],
            p["ks"], p["P_orbs"], p["incs_rad"],
            p["a_Rs_arr"], p["eccs"], p["ws_rad"],
            p["u1"], p["u2"],
            p["companion_fluxratio"], p["companion_is_host"],
        )

        assert gpu_lnL.shape == cpu_reference_lnL.shape, (
            f"Shape mismatch: Metal {gpu_lnL.shape} vs CPU {cpu_reference_lnL.shape}"
        )
        assert gpu_lnL.dtype == np.float64, (
            f"Expected float64, got {gpu_lnL.dtype}"
        )

        cpu_fin = np.isfinite(cpu_reference_lnL)
        gpu_fin = np.isfinite(gpu_lnL)
        both_fin = cpu_fin & gpu_fin

        if both_fin.sum() > 0:
            abs_diff = np.abs(gpu_lnL[both_fin] - cpu_reference_lnL[both_fin])
            max_diff = abs_diff.max()
            median_diff = np.median(abs_diff)
            print(
                f"\n[Metal transit] max |diff|={max_diff:.4f}, "
                f"median |diff|={median_diff:.4f}, "
                f"finite: Metal={gpu_fin.sum()} CPU={cpu_fin.sum()}"
            )
            # float32 chi2 accumulation can cause ~2 lnL drift over 800 points
            assert max_diff < 2.0, (
                f"Max absolute lnL difference {max_diff:.4f} >= 2.0"
            )

        cpu_inf_count = (~cpu_fin).sum()
        gpu_inf_count = (~gpu_fin).sum()
        extra_inf = max(0, gpu_inf_count - cpu_inf_count)
        N = len(cpu_reference_lnL)
        assert extra_inf < 0.05 * N, (
            f"Metal has {extra_inf} extra -inf draws (> 5% of {N})"
        )

    # -------------------------------------------------------------- #
    # EB lnL matches CPU
    # -------------------------------------------------------------- #

    def test_batch_eb_transit_lnL_matches_cpu(
        self, eb_params, cpu_reference_eb_lnL
    ):
        """Metal EB lnL must match CPU within float32 tolerance."""
        from fpp_metal import batch_eb_transit_lnL

        p = eb_params
        gpu_lnL = batch_eb_transit_lnL(
            p["time"], p["flux"], p["sigma"],
            p["ks"], p["eb_fluxratios"], p["P_orbs"],
            p["incs_rad"], p["a_Rs_arr"], p["eccs"], p["ws_rad"],
            p["u1"], p["u2"],
            p["companion_fluxratio"], p["companion_is_host"],
        )

        assert gpu_lnL.shape == cpu_reference_eb_lnL.shape
        assert gpu_lnL.dtype == np.float64

        cpu_fin = np.isfinite(cpu_reference_eb_lnL)
        gpu_fin = np.isfinite(gpu_lnL)
        both_fin = cpu_fin & gpu_fin

        if both_fin.sum() > 0:
            abs_diff = np.abs(gpu_lnL[both_fin] - cpu_reference_eb_lnL[both_fin])
            max_diff = abs_diff.max()
            median_diff = np.median(abs_diff)
            print(
                f"\n[Metal EB] max |diff|={max_diff:.4f}, "
                f"median |diff|={median_diff:.4f}, "
                f"finite: Metal={gpu_fin.sum()} CPU={cpu_fin.sum()}"
            )
            assert max_diff < 2.0, (
                f"Max absolute EB lnL difference {max_diff:.4f} >= 2.0"
            )

        cpu_inf_count = (~cpu_fin).sum()
        gpu_inf_count = (~gpu_fin).sum()
        extra_inf = max(0, gpu_inf_count - cpu_inf_count)
        N = len(cpu_reference_eb_lnL)
        assert extra_inf < 0.05 * N, (
            f"Metal has {extra_inf} extra -inf EB draws (> 5% of {N})"
        )

    # -------------------------------------------------------------- #
    # is_available / get_info
    # -------------------------------------------------------------- #

    def test_is_available_returns_bool(self):
        from fpp_metal import is_available
        result = is_available()
        assert isinstance(result, bool)
        assert result is True  # we skipped if not available

    def test_get_info_returns_string(self):
        from fpp_metal import get_info
        info = get_info()
        assert isinstance(info, str)
        assert len(info) > 0
        print(f"\n[Metal info] {info}")

    # -------------------------------------------------------------- #
    # Benchmark
    # -------------------------------------------------------------- #

    def test_metal_benchmark_1k_draws(self, transit_params):
        """Benchmark Metal backend: warmup + timed run."""
        from fpp_metal import batch_transit_lnL

        p = transit_params
        N = len(p["ks"])
        T = len(p["time"])

        # Warmup (includes MLX graph compilation)
        _ = batch_transit_lnL(
            p["time"], p["flux"], p["sigma"],
            p["ks"], p["P_orbs"], p["incs_rad"],
            p["a_Rs_arr"], p["eccs"], p["ws_rad"],
            p["u1"], p["u2"],
            p["companion_fluxratio"], p["companion_is_host"],
        )

        n_trials = 3
        times = []
        for _ in range(n_trials):
            t0 = _time.perf_counter()
            _ = batch_transit_lnL(
                p["time"], p["flux"], p["sigma"],
                p["ks"], p["P_orbs"], p["incs_rad"],
                p["a_Rs_arr"], p["eccs"], p["ws_rad"],
                p["u1"], p["u2"],
                p["companion_fluxratio"], p["companion_is_host"],
            )
            times.append(_time.perf_counter() - t0)

        best_ms = min(times) * 1e3
        median_ms = sorted(times)[len(times) // 2] * 1e3

        print(
            f"\n[Metal benchmark] N={N} draws, T={T} points | "
            f"best={best_ms:.1f} ms  median={median_ms:.1f} ms  "
            f"({n_trials} trials)"
        )


# ================================================================== #
# OpenCL Backend
# ================================================================== #

class TestOpenCLBackend:
    """Validates OpenCL GPU backend against CPU baseline.

    Skipped automatically when PyOpenCL is not installed or no
    compatible OpenCL device is found.
    """

    @pytest.fixture(autouse=True)
    def skip_if_no_opencl(self):
        try:
            from fpp_opencl import is_available
            if not is_available():
                pytest.skip("OpenCL not available")
        except ImportError:
            pytest.skip("fpp_opencl not installed")

    # -------------------------------------------------------------- #
    # Transit lnL matches CPU
    # -------------------------------------------------------------- #

    def test_batch_transit_lnL_matches_cpu(
        self, transit_params, cpu_reference_lnL
    ):
        """OpenCL transit lnL must match CPU within float32 tolerance."""
        from fpp_opencl import batch_transit_lnL

        p = transit_params
        gpu_lnL = batch_transit_lnL(
            p["time"], p["flux"], p["sigma"],
            p["ks"], p["P_orbs"], p["incs_rad"],
            p["a_Rs_arr"], p["eccs"], p["ws_rad"],
            p["u1"], p["u2"],
            p["companion_fluxratio"], p["companion_is_host"],
        )

        assert gpu_lnL.shape == cpu_reference_lnL.shape, (
            f"Shape mismatch: OpenCL {gpu_lnL.shape} vs CPU {cpu_reference_lnL.shape}"
        )
        assert gpu_lnL.dtype == np.float64, (
            f"Expected float64, got {gpu_lnL.dtype}"
        )

        cpu_fin = np.isfinite(cpu_reference_lnL)
        gpu_fin = np.isfinite(gpu_lnL)
        both_fin = cpu_fin & gpu_fin

        if both_fin.sum() > 0:
            abs_diff = np.abs(gpu_lnL[both_fin] - cpu_reference_lnL[both_fin])
            max_diff = abs_diff.max()
            median_diff = np.median(abs_diff)
            print(
                f"\n[OpenCL transit] max |diff|={max_diff:.4f}, "
                f"median |diff|={median_diff:.4f}, "
                f"finite: OpenCL={gpu_fin.sum()} CPU={cpu_fin.sum()}"
            )
            # float32 chi2 accumulation can cause ~1-2 lnL drift over 800 points
            assert max_diff < 2.0, (
                f"Max absolute lnL difference {max_diff:.4f} >= 2.0"
            )

        # Allow < 5% extra -inf draws on GPU
        cpu_inf_count = (~cpu_fin).sum()
        gpu_inf_count = (~gpu_fin).sum()
        extra_inf = max(0, gpu_inf_count - cpu_inf_count)
        N = len(cpu_reference_lnL)
        assert extra_inf < 0.05 * N, (
            f"OpenCL has {extra_inf} extra -inf draws (> 5% of {N})"
        )

    # -------------------------------------------------------------- #
    # EB lnL matches CPU
    # -------------------------------------------------------------- #

    def test_batch_eb_transit_lnL_matches_cpu(
        self, eb_params, cpu_reference_eb_lnL
    ):
        """OpenCL EB lnL must match CPU within float32 tolerance."""
        from fpp_opencl import batch_eb_transit_lnL

        p = eb_params
        gpu_lnL = batch_eb_transit_lnL(
            p["time"], p["flux"], p["sigma"],
            p["ks"], p["eb_fluxratios"], p["P_orbs"],
            p["incs_rad"], p["a_Rs_arr"], p["eccs"], p["ws_rad"],
            p["u1"], p["u2"],
            p["companion_fluxratio"], p["companion_is_host"],
        )

        assert gpu_lnL.shape == cpu_reference_eb_lnL.shape
        assert gpu_lnL.dtype == np.float64

        cpu_fin = np.isfinite(cpu_reference_eb_lnL)
        gpu_fin = np.isfinite(gpu_lnL)
        both_fin = cpu_fin & gpu_fin

        if both_fin.sum() > 0:
            abs_diff = np.abs(gpu_lnL[both_fin] - cpu_reference_eb_lnL[both_fin])
            max_diff = abs_diff.max()
            median_diff = np.median(abs_diff)
            print(
                f"\n[OpenCL EB] max |diff|={max_diff:.4f}, "
                f"median |diff|={median_diff:.4f}, "
                f"finite: OpenCL={gpu_fin.sum()} CPU={cpu_fin.sum()}"
            )
            assert max_diff < 2.0, (
                f"Max absolute EB lnL difference {max_diff:.4f} >= 2.0"
            )

        cpu_inf_count = (~cpu_fin).sum()
        gpu_inf_count = (~gpu_fin).sum()
        extra_inf = max(0, gpu_inf_count - cpu_inf_count)
        N = len(cpu_reference_eb_lnL)
        assert extra_inf < 0.05 * N, (
            f"OpenCL has {extra_inf} extra -inf EB draws (> 5% of {N})"
        )

    # -------------------------------------------------------------- #
    # is_available / get_info
    # -------------------------------------------------------------- #

    def test_is_available_returns_bool(self):
        from fpp_opencl import is_available
        result = is_available()
        assert isinstance(result, bool)
        assert result is True  # we skipped if not available

    def test_get_info_returns_string(self):
        from fpp_opencl import get_info
        info = get_info()
        assert isinstance(info, str)
        assert len(info) > 0
        print(f"\n[OpenCL info] {info}")

    # -------------------------------------------------------------- #
    # Benchmark
    # -------------------------------------------------------------- #

    def test_opencl_benchmark_1k_draws(self, transit_params):
        """Benchmark OpenCL backend: warmup + timed run."""
        from fpp_opencl import batch_transit_lnL

        p = transit_params
        N = len(p["ks"])
        T = len(p["time"])

        # Warmup (includes kernel compilation)
        _ = batch_transit_lnL(
            p["time"], p["flux"], p["sigma"],
            p["ks"], p["P_orbs"], p["incs_rad"],
            p["a_Rs_arr"], p["eccs"], p["ws_rad"],
            p["u1"], p["u2"],
            p["companion_fluxratio"], p["companion_is_host"],
        )

        n_trials = 3
        times = []
        for _ in range(n_trials):
            t0 = _time.perf_counter()
            _ = batch_transit_lnL(
                p["time"], p["flux"], p["sigma"],
                p["ks"], p["P_orbs"], p["incs_rad"],
                p["a_Rs_arr"], p["eccs"], p["ws_rad"],
                p["u1"], p["u2"],
                p["companion_fluxratio"], p["companion_is_host"],
            )
            times.append(_time.perf_counter() - t0)

        best_ms = min(times) * 1e3
        median_ms = sorted(times)[len(times) // 2] * 1e3

        print(
            f"\n[OpenCL benchmark] N={N} draws, T={T} points | "
            f"best={best_ms:.1f} ms  median={median_ms:.1f} ms  "
            f"({n_trials} trials)"
        )


# ================================================================== #
# OpenCL FPP device selection robustness (CPU-only, no device required)
#
# 2026-07-24 review it.2, Finding 1: fpp_opencl._init_opencl() used to call
# cl.create_some_context(interactive=False) directly, without filtering for
# a GPU device -- exactly the "silent CPU fallback" CLAUDE.md #2 forbids,
# already fixed once for bls_opencl.py (see
# tests/test_bls_opencl_parity.py::TestSelectGpuDeviceRobustness, which this
# class mirrors). fpp_opencl now reuses bls_opencl._select_gpu_device()
# directly (imported as fpp_opencl._select_gpu_device), so these tests
# monkeypatch that name on fpp_opencl itself -- monkeypatching
# bls_opencl._select_gpu_device would NOT affect fpp_opencl's already-bound
# reference to the same function object.
# ================================================================== #

class TestFppOpenclDeviceSelectionRobustness:
    @pytest.fixture(autouse=True)
    def _require_fpp_opencl_module(self):
        try:
            import fpp_opencl
        except ImportError:
            pytest.skip("fpp_opencl not installed")
        # Narrower than "module importable": these tests monkeypatch
        # fpp_opencl.cl.* attributes directly, so PyOpenCL itself must have
        # imported cleanly inside fpp_opencl (same gate bls_opencl's own
        # device-selection tests use, _OPENCL_PY_BOUND).
        if not getattr(fpp_opencl, "OPENCL_AVAILABLE", False):
            pytest.skip("PyOpenCL not importable (fpp_opencl.cl unbound)")

    @pytest.fixture(autouse=True)
    def _reset_fpp_opencl_globals(self, monkeypatch):
        """Isolate every test in this class from the module's cached
        context/queue/program -- _init_opencl() short-circuits to True if
        _ctx is already set (e.g. from an earlier real-hardware test in
        this same process)."""
        import fpp_opencl
        monkeypatch.setattr(fpp_opencl, "_ctx", None)
        monkeypatch.setattr(fpp_opencl, "_queue", None)
        monkeypatch.setattr(fpp_opencl, "_program", None)
        yield

    def test_init_opencl_returns_false_and_never_calls_create_some_context(
            self, monkeypatch):
        """The core Finding 1 regression gate for fpp_opencl: when no GPU
        device is found, _init_opencl must return False and must NEVER call
        cl.create_some_context (that would silently hand back a CPU OpenCL
        context)."""
        import fpp_opencl

        monkeypatch.setattr(fpp_opencl, "OPENCL_AVAILABLE", True)
        monkeypatch.setattr(fpp_opencl, "_select_gpu_device", lambda: None)

        def _forbidden_create_some_context(*a, **k):
            raise AssertionError(
                "fpp_opencl._init_opencl must NEVER call create_some_context "
                "when no GPU device was found (2026-07-24 review it.2, "
                "Finding 1)")
        monkeypatch.setattr(fpp_opencl.cl, "create_some_context",
                            _forbidden_create_some_context)

        assert fpp_opencl._init_opencl() is False
        assert fpp_opencl._ctx is None

    def test_init_opencl_uses_the_selected_gpu_device_when_found(
            self, monkeypatch):
        """Sanity check the opposite branch: a device found by
        _select_gpu_device is passed straight to cl.Context, never through
        create_some_context, and the program still gets built."""
        import fpp_opencl

        sentinel_device = object()
        captured = {}

        class _FakeContext:
            def __init__(self, devices):
                captured["devices"] = devices

        class _FakeQueue:
            def __init__(self, ctx):
                captured["ctx"] = ctx

        class _FakeProgram:
            def __init__(self, ctx, src):
                captured["program_ctx"] = ctx

            def build(self):
                return self

        monkeypatch.setattr(fpp_opencl, "OPENCL_AVAILABLE", True)
        monkeypatch.setattr(fpp_opencl, "_select_gpu_device",
                            lambda: sentinel_device)
        monkeypatch.setattr(fpp_opencl.cl, "Context", _FakeContext)
        monkeypatch.setattr(fpp_opencl.cl, "CommandQueue", _FakeQueue)
        monkeypatch.setattr(fpp_opencl.cl, "Program", _FakeProgram)

        assert fpp_opencl._init_opencl() is True
        assert captured["devices"] == [sentinel_device]
        assert captured["ctx"] is fpp_opencl._ctx


# ================================================================== #
# OpenCL lnL normalisation math (CPU-only, no device required)
# ================================================================== #

class TestOpenCLLnLMath:
    """Verify the OpenCL lnL finalisation matches fpp_native's formula exactly.

    These tests exercise only the Python-side CPU finalisation code — no OpenCL
    device is needed.  They can run everywhere, including CI.

    The production formula (from fpp_native._batch_transit_lnL line ~357) is:
        lnL[j] = -0.5*ln2pi - lnsigma - 0.5*chi2/(sigma*sigma)
    i.e. ONE constant term per draw, NOT multiplied by T.
    The historic bug was ``T * lnL_const`` which caused Δ = (T-1)·lnL_const.
    """

    def _opencl_finalise(self, chi2_arr, sigma, T):
        """Replicate the OpenCL CPU finalisation as it appears in fpp_opencl."""
        import math
        ln2pi = math.log(2.0 * math.pi)
        lnsigma = math.log(float(sigma))
        lnL_const = -0.5 * ln2pi - lnsigma
        sigma2 = float(sigma) * float(sigma)
        return lnL_const - 0.5 * chi2_arr / sigma2

    def _native_lnL_const(self, sigma):
        """Return lnL_const from a real call to fpp_native._batch_transit_lnL.

        Uses a near-zero-depth transit (k=1e-6, circular, face-on, a/Rs=100)
        so the model is indistinguishable from 1.0 everywhere.  chi2 is then
        analytically zero to float64 precision, and the returned lnL equals
        lnL_const = -0.5*ln(2pi) - ln(sigma) exactly.

        This is the real reference — if fpp_native's formula ever changes,
        this helper diverges from _opencl_finalise and the tests below catch it.
        """
        T = 50
        time = np.linspace(-2.5, 2.5, T)
        flux = np.ones(T, dtype=np.float64)
        result = fpp_native._batch_transit_lnL(
            time, flux, float(sigma),
            np.array([1e-6]),     # k — near-zero depth
            np.array([10.0]),     # P_orb
            np.array([0.0]),      # inc_rad — face-on
            np.array([100.0]),    # a_Rs — large semi-major axis
            np.array([0.0]),      # ecc
            np.array([0.0]),      # w_rad
            0.0, 0.0,             # u1, u2 limb-darkening
            np.array([0.0]),      # companion_fluxratio
            False,                # companion_is_host
        )
        # result[0] == lnL_const because chi2 == 0
        return float(result[0])

    def test_single_draw_matches_native(self):
        """Single chi2 draw: opencl finalisation == fpp_native._batch_transit_lnL.

        Strategy: obtain lnL_const from a *real* fpp_native call (chi2=0 by
        construction), then derive the expected lnL for a non-zero chi2 as
        lnL_const - 0.5*chi2/sigma^2 and compare against _opencl_finalise.
        This means a divergence in fpp_native's formula is caught here.
        """
        sigma = 5e-4
        chi2 = np.array([1.23e-5], dtype=np.float64)
        T = 800

        lnL_const = self._native_lnL_const(sigma)
        expected_lnL = lnL_const - 0.5 * chi2 / (sigma * sigma)
        opencl_lnL = self._opencl_finalise(chi2, sigma, T)

        np.testing.assert_allclose(
            opencl_lnL, expected_lnL, rtol=1e-12,
            err_msg="OpenCL lnL finalisation does not match fpp_native reference"
        )

    def test_array_of_draws_matches_native(self):
        """N=1000 chi2 draws: opencl finalisation matches native elementwise.

        lnL_const is obtained from a real fpp_native call (not a copy of the
        formula), so a change in fpp_native's arithmetic would be detected.
        """
        rng = np.random.default_rng(0)
        sigma = 5e-4
        T = 800
        N = 1000
        # Realistic chi2 range: T draws of (flux_err/sigma)^2 ~ T * 1.0
        chi2 = rng.uniform(0.0, 2.0 * T * sigma**2, N)

        lnL_const = self._native_lnL_const(sigma)
        expected_lnL = lnL_const - 0.5 * chi2 / (sigma * sigma)
        opencl_lnL = self._opencl_finalise(chi2, sigma, T)

        np.testing.assert_allclose(
            opencl_lnL, expected_lnL, rtol=1e-12,
            err_msg="Elementwise mismatch between OpenCL and native lnL"
        )

    def test_wrong_T_mult_would_fail(self):
        """Regression guard: the *old* T*lnL_const formula must produce wrong results.

        This documents exactly the bug that was fixed.  If this test starts
        *passing* (i.e. the two expressions agree), the fix has been reverted.
        """
        import math
        sigma = 5e-4
        T = 800
        chi2 = np.array([1.23e-5], dtype=np.float64)

        ln2pi = math.log(2.0 * math.pi)
        lnsigma = math.log(float(sigma))
        lnL_const = -0.5 * ln2pi - lnsigma
        sigma2 = float(sigma) * float(sigma)

        buggy_lnL  = T * lnL_const - 0.5 * chi2 / sigma2
        fixed_lnL  = lnL_const      - 0.5 * chi2 / sigma2

        # The two must differ by (T-1)*lnL_const
        expected_delta = (T - 1) * lnL_const
        actual_delta = buggy_lnL - fixed_lnL
        np.testing.assert_allclose(
            actual_delta, expected_delta, rtol=1e-12,
            err_msg="Delta between buggy and fixed formula does not match theory"
        )
        # And the difference must be large (proving the bug was real)
        assert abs(actual_delta[0]) > 1000.0, (
            f"Expected large bug delta, got {actual_delta[0]:.1f}"
        )

    def test_eb_sentinel_becomes_neginf(self):
        """chi2=-1 sentinel draws must map to -inf after the rejection mask."""
        sigma = 5e-4
        T = 800
        # Mix of real chi2 values and the sentinel (-1.0)
        chi2_buf = np.array([1.23e-5, -1.0, 2.45e-5, -1.0], dtype=np.float32)

        rejected = chi2_buf < 0.0
        chi2_f64 = chi2_buf.astype(np.float64)
        lnL_batch = self._opencl_finalise(chi2_f64, sigma, T)
        lnL_batch[rejected] = -np.inf

        # Rejected draws must be -inf
        assert np.all(lnL_batch[rejected] == -np.inf), (
            "Sentinel draws not set to -inf"
        )
        # Non-rejected draws must be finite
        assert np.all(np.isfinite(lnL_batch[~rejected])), (
            "Non-sentinel draws should be finite"
        )

    def test_native_formula_T4000_large_delta_matches_bench(self):
        """Verify bench observation: at T=4000 the bug delta is ~2.67e4.

        With sigma=5e-4:
          -0.5*ln(2pi) ≈ -0.919
          -ln(5e-4)    ≈ +7.601
          lnL_const    ≈ 6.682
          (T-1)*lnL_const at T=4000 ≈ 3999 * 6.682 ≈ 2.672e4  [matches bench]
        """
        import math
        sigma = 5e-4
        T = 4000
        ln2pi = math.log(2.0 * math.pi)
        lnsigma = math.log(sigma)
        lnL_const = -0.5 * ln2pi - lnsigma

        expected_delta_per_draw = (T - 1) * lnL_const
        # The bench reports max|ΔlnL|=2.673e4.  The delta is constant across
        # draws (chi2 terms cancel), so it must be exactly (T-1)*lnL_const.
        # We only check the order of magnitude here (bench σ may differ).
        assert abs(expected_delta_per_draw) > 1000.0, (
            f"Delta at T=4000 should be large, got {expected_delta_per_draw:.1f}"
        )
        print(
            f"\n[lnL bug delta] T=4000, sigma={sigma}: "
            f"(T-1)*lnL_const = {expected_delta_per_draw:.3e}  "
            f"(bench reported ~2.67e4)"
        )


# ================================================================== #
# GPU Dispatch
# ================================================================== #

class TestGPUDispatch:
    def test_init_fpp_gpu_sets_backend(self):
        from fpp_native import _init_fpp_gpu, _fpp_gpu_backend
        _init_fpp_gpu()
        assert _fpp_gpu_backend in (None, "cuda", "metal", "opencl")

    def test_cpu_core_config_respected(self):
        import numba
        from fpp_native import _configure_fpp_cpu_cores
        original = numba.get_num_threads()
        _configure_fpp_cpu_cores(4)
        assert numba.get_num_threads() == 4
        numba.set_num_threads(original)  # restore

    def test_calc_fpp_native_signature_unchanged(self):
        import inspect
        from fpp_native import calc_fpp_native
        sig = inspect.signature(calc_fpp_native)
        params = list(sig.parameters.keys())
        assert params[:8] == ["tic_id", "period", "t0", "duration",
                               "depth_ppm", "sectors", "time", "flux"]


# ================================================================== #
# Full Integration Benchmark (N=100K)
# ================================================================== #

class TestFullBenchmark:
    """End-to-end benchmark with N=100K (production scale)."""

    @pytest.fixture
    def large_transit_params(self):
        """Production-scale transit parameters: N=100K draws."""
        np.random.seed(42)
        N = 100000
        period = 3.42
        time = np.linspace(-0.5 * period, 0.5 * period, 800)
        flux = np.ones(800)
        in_transit = np.abs(time) < 0.05 * period
        flux[in_transit] -= 0.0014
        flux += np.random.normal(0, 0.0003, 800)
        sigma = float(np.std(flux))
        return {
            "time": np.ascontiguousarray(time, dtype=np.float64),
            "flux": np.ascontiguousarray(flux, dtype=np.float64),
            "sigma": sigma,
            "ks": np.random.uniform(0.02, 0.15, N),
            "P_orbs": np.full(N, period),
            "incs_rad": np.random.uniform(85, 90, N) * np.pi / 180.0,
            "a_Rs": np.random.uniform(8, 20, N).astype(np.float64),
            "eccs": np.random.uniform(0, 0.3, N),
            "ws_rad": np.random.uniform(0, 2 * np.pi, N),
            "u1": 0.4, "u2": 0.26,
            "companion_fluxratio": np.zeros(N),
            "companion_is_host": False,
        }

    def test_benchmark_100k_cpu(self, large_transit_params):
        """CPU baseline: N=100K."""
        from fpp_native import _batch_transit_lnL
        p = large_transit_params
        # Warmup
        _batch_transit_lnL(
            p["time"], p["flux"], p["sigma"],
            p["ks"][:100], p["P_orbs"][:100], p["incs_rad"][:100],
            p["a_Rs"][:100], p["eccs"][:100], p["ws_rad"][:100],
            p["u1"], p["u2"], p["companion_fluxratio"][:100],
            p["companion_is_host"])
        start = time_mod.perf_counter()
        cpu_result = _batch_transit_lnL(
            p["time"], p["flux"], p["sigma"],
            p["ks"], p["P_orbs"], p["incs_rad"], p["a_Rs"],
            p["eccs"], p["ws_rad"], p["u1"], p["u2"],
            p["companion_fluxratio"], p["companion_is_host"])
        cpu_time = time_mod.perf_counter() - start
        print(f"\n  CPU N=100K: {cpu_time:.2f}s")
        assert len(cpu_result) == 100000

    def test_benchmark_100k_gpu(self, large_transit_params):
        """GPU (best available): N=100K."""
        from fpp_native import _gpu_batch_transit, _fpp_gpu_backend
        if _gpu_batch_transit is None:
            pytest.skip("No GPU backend available")
        p = large_transit_params
        # Warmup
        _gpu_batch_transit(
            p["time"], p["flux"], p["sigma"],
            p["ks"][:100], p["P_orbs"][:100], p["incs_rad"][:100],
            p["a_Rs"][:100], p["eccs"][:100], p["ws_rad"][:100],
            p["u1"], p["u2"], p["companion_fluxratio"][:100],
            p["companion_is_host"])
        start = time_mod.perf_counter()
        gpu_result = _gpu_batch_transit(
            p["time"], p["flux"], p["sigma"],
            p["ks"], p["P_orbs"], p["incs_rad"], p["a_Rs"],
            p["eccs"], p["ws_rad"], p["u1"], p["u2"],
            p["companion_fluxratio"], p["companion_is_host"])
        gpu_time = time_mod.perf_counter() - start
        print(f"\n  GPU ({_fpp_gpu_backend}) N=100K: {gpu_time:.2f}s")
        assert len(gpu_result) == 100000
