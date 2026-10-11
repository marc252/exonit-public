"""TDD: bls_fast_batch — multi-flux BLS with shared phase-fold setup.

Contract:
  - Same signature shape as bls_fast, but accepts a list/2D array of
    fluxes and returns a list of result dicts, one per flux, in order.
  - Each element must be NUMERICALLY EQUAL to what bls_fast would return
    for that flux alone (same period grid, same dur_bins → same ops).
  - Edge cases: empty list → [], single flux → single-element list that
    matches bls_fast exactly.

Parity drives correctness; the speedup is validated later by a
per-host profile script (not a unit test — timing is platform-dependent
and would be flaky on CI).
"""
import numpy as np
import pytest

from bls_fast import bls_fast, bls_fast_batch


def _synth_lc(n=2000, period=3.2, depth=1e-3, dur_frac=0.02, seed=0):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 80.0, n).astype(np.float64)
    flux = 1.0 + rng.normal(0, 1e-4, n).astype(np.float64)
    # Inject a shallow transit
    phase = (t / period) % 1.0
    in_transit = (phase < dur_frac) | (phase > 1.0 - dur_frac)
    flux[in_transit] -= depth
    return t, flux


class TestBlsFastBatchContract:
    def test_empty_flux_list_returns_empty_list(self):
        t, _ = _synth_lc()
        assert bls_fast_batch(t, []) == []

    def test_single_flux_matches_single_call(self):
        t, f = _synth_lc(seed=1)
        kwargs = dict(period_min=2.0, period_max=5.0, n_periods=400,
                      n_bins=80, n_dur_steps=6)
        single = bls_fast(t, f, **kwargs)
        batch = bls_fast_batch(t, [f], **kwargs)
        assert len(batch) == 1
        r = batch[0]
        for k in ("period", "power", "duration", "t0",
                  "depth", "depth_ppm", "snr"):
            assert r[k] == pytest.approx(single[k], rel=1e-6, abs=1e-9), (
                f"mismatch in {k}: batch={r[k]} single={single[k]}")

    def test_multi_flux_matches_individual_calls(self):
        t, f1 = _synth_lc(period=3.2, seed=2)
        _, f2 = _synth_lc(period=5.1, depth=2e-3, seed=3)
        _, f3 = _synth_lc(period=7.3, depth=0.5e-3, dur_frac=0.015, seed=4)
        kwargs = dict(period_min=1.0, period_max=10.0, n_periods=500,
                      n_bins=100, n_dur_steps=8)
        singles = [bls_fast(t, f, **kwargs) for f in (f1, f2, f3)]
        batch = bls_fast_batch(t, [f1, f2, f3], **kwargs)
        assert len(batch) == 3
        for i, (b, s) in enumerate(zip(batch, singles)):
            for k in ("period", "power", "duration", "t0",
                      "depth", "depth_ppm", "snr"):
                assert b[k] == pytest.approx(s[k], rel=1e-6, abs=1e-9), (
                    f"flux #{i} mismatch in {k}: batch={b[k]} single={s[k]}")

    def test_accepts_2d_array(self):
        """A (n_flux, N) numpy array is a natural alternative to list."""
        t, f1 = _synth_lc(seed=5)
        _, f2 = _synth_lc(period=4.7, seed=6)
        arr = np.stack([f1, f2])  # shape (2, N)
        kwargs = dict(period_min=2.0, period_max=6.0, n_periods=300,
                      n_bins=80, n_dur_steps=6)
        res_list = bls_fast_batch(t, [f1, f2], **kwargs)
        res_arr = bls_fast_batch(t, arr, **kwargs)
        assert len(res_arr) == 2
        for r1, r2 in zip(res_list, res_arr):
            for k in ("period", "snr", "depth"):
                assert r1[k] == pytest.approx(r2[k], rel=1e-9, abs=1e-12)
