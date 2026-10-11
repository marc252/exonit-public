"""Coherency tests for bls_opencl_batch.

The batched version must produce the same detections as N independent
bls_opencl() calls on the same fluxes, so injection-recovery results
are reproducible regardless of which backend path is used."""

import numpy as np
import pytest


pytest.importorskip("pyopencl")

# Skip if no OpenCL device is actually available (CI, hosts without iGPU).
from bls_opencl import _init_opencl  # noqa: E402
if not _init_opencl():
    pytest.skip("OpenCL device not initializable", allow_module_level=True)


def _synthetic_flux(n_points, period, depth, duration, t0=2.5, seed=0):
    rng = np.random.default_rng(seed)
    time = np.linspace(0, 60, n_points)
    flux = 1.0 + rng.normal(0, 0.001, n_points)
    phase = ((time - t0) % period) / period
    half_dur = (duration / period) / 2
    flux[(phase < half_dur) | (phase > 1.0 - half_dur)] -= depth
    return time, flux


class TestBatchCoherency:
    def test_single_flux_in_batch_matches_non_batch(self):
        """Batch with a single flux element must equal bls_opencl() called
        directly on that flux — this isolates the batching wrapper."""
        from bls_opencl import bls_opencl, bls_opencl_batch

        time, flux = _synthetic_flux(3000, period=10.0, depth=0.005,
                                     duration=0.2, seed=1)

        solo = bls_opencl(time, flux,
                          period_min=1.0, period_max=30.0,
                          n_periods=500, n_bins=150, n_dur_steps=10)
        batch = bls_opencl_batch(time, [flux],
                                 period_min=1.0, period_max=30.0,
                                 n_periods=500, n_bins=150, n_dur_steps=10)

        assert len(batch) == 1
        # Period must match exactly within the grid resolution
        grid_step = (30.0 - 1.0) / 500
        assert abs(batch[0]["period"] - solo["period"]) < 2 * grid_step

    def test_three_fluxes_each_recovers_its_own_period(self):
        """Three injected fluxes with distinct periods — the batch must
        return each period in the right slot."""
        from bls_opencl import bls_opencl_batch

        time = np.linspace(0, 60, 3000)
        fluxes = []
        periods_truth = [3.0, 7.0, 15.0]
        for i, p in enumerate(periods_truth):
            _, f = _synthetic_flux(3000, period=p, depth=0.01,
                                   duration=0.2, seed=10 + i)
            fluxes.append(f)

        results = bls_opencl_batch(time, fluxes,
                                   period_min=1.0, period_max=30.0,
                                   n_periods=1000, n_bins=150,
                                   n_dur_steps=10)

        assert len(results) == 3
        for i, (r, p_true) in enumerate(zip(results, periods_truth)):
            # Allow 1:2 / 2:1 harmonics (BLS freely picks them for clean
            # single-transit folds) — same tolerance as scorecard.
            cand = r["period"]
            match = any(abs(cand - p_true * k) / (p_true * k) < 0.05
                        for k in (0.5, 1.0, 2.0))
            assert match, (
                f"flux {i}: expected P~{p_true}, got {cand} (no harmonic)")

    def test_returns_minimum_fields(self):
        """Batch results must expose period and snr — injection-recovery
        consumes those two."""
        from bls_opencl import bls_opencl_batch
        time = np.linspace(0, 30, 1500)
        flux = np.ones(1500) + np.random.default_rng(0).normal(0, 0.001, 1500)

        results = bls_opencl_batch(time, [flux, flux],
                                   period_min=1.0, period_max=10.0,
                                   n_periods=200, n_bins=100, n_dur_steps=5)
        assert len(results) == 2
        for r in results:
            assert "period" in r
            assert "snr" in r
