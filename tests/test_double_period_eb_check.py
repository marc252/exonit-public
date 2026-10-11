"""Unit tests for double_period_eb_check in scripts/discovery_postprocess.py.

These tests exercise the decision logic directly using synthetic light curves
constructed with numpy — no MAST network calls, no lightkurve dependency.
The function is imported from the script module and the MAST fetch path is
bypassed by monkeypatching the lightkurve import inside the function.
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Make the scripts directory importable
SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS))

from discovery_postprocess import double_period_eb_check  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers to build synthetic LCs
# ---------------------------------------------------------------------------

def _make_synthetic_lc(
    period: float,
    t0: float,
    dur: float,
    n_points: int = 5000,
    baseline_days: float = 40.0,
    depth_primary_ppm: float = 4500.0,
    depth_secondary_ppm: float = 4500.0,
    noise_ppm: float = 200.0,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (time, flux) for a synthetic double-period EB.

    The true EB period is 2×period.  Eclipses are injected at phases 0.0 and
    0.5 of the 2×period fold with depths *depth_primary_ppm* and
    *depth_secondary_ppm* respectively.  BLS would detect the half-period
    *period* when the two eclipses are of similar depth.
    """
    rng = np.random.default_rng(seed)
    time = np.sort(rng.uniform(t0, t0 + baseline_days, n_points))
    flux = 1.0 + rng.normal(0, noise_ppm / 1e6, n_points)

    P2 = 2.0 * period
    phase2 = ((time - t0) % P2) / P2
    win = dur / P2

    # Primary eclipse at phase 0.0
    in_primary = np.abs(phase2) < win
    in_primary |= np.abs(phase2 - 1.0) < win  # wrap
    flux[in_primary] -= depth_primary_ppm / 1e6

    # Secondary eclipse at phase 0.5
    in_secondary = np.abs(phase2 - 0.5) < win
    flux[in_secondary] -= depth_secondary_ppm / 1e6

    return time, flux


class _FakeLightkurveLc:
    """Minimal mock that satisfies the lightkurve LC interface used in the function."""

    def __init__(self, time_val, flux_val):
        self.time = MagicMock()
        self.time.value = time_val
        self.flux = MagicMock()
        self.flux.value = flux_val

    def remove_nans(self):
        return self

    def normalize(self):
        return self

    def flatten(self, window_length=401):
        return self


class _FakeLcCollection:
    def __init__(self, lc):
        self._lc = lc
        self.__len__ = lambda self: 1

    def __len__(self):
        return 1

    def stitch(self):
        return self._lc


class _FakeLightkurve:
    """Module-level mock for lightkurve."""

    @staticmethod
    def search_lightcurve(target, mission=None, author=None, cadence=None):
        return MagicMock(__len__=lambda s: 1)  # non-empty search result

    @classmethod
    def _inject(cls, time_val, flux_val):
        """Return a lightkurve mock that produces the given arrays."""
        lk = MagicMock()
        col = _FakeLcCollection(_FakeLightkurveLc(time_val, flux_val))
        lk.search_lightcurve.return_value.download_all.return_value = col
        return lk


def _make_lk_mock(time_val, flux_val):
    lk_mock = MagicMock()
    fake_lc = _FakeLightkurveLc(time_val, flux_val)
    fake_col = _FakeLcCollection(fake_lc)
    # search returns a non-empty result
    sr_mock = MagicMock()
    sr_mock.__len__ = MagicMock(return_value=1)
    sr_mock.download_all.return_value = fake_col
    lk_mock.search_lightcurve.return_value = sr_mock
    return lk_mock


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

class TestDoublePeriodEbCheckEb:
    """Cases that should return verdict='eb'."""

    def test_unequal_eclipses_detected_as_eb(self):
        """Classic double-period EB: primary 4500 ppm, secondary 3500 ppm (6 σ apart)."""
        period = 7.124
        t0 = 1329.18
        dur = 0.14

        time_arr, flux_arr = _make_synthetic_lc(
            period=period, t0=t0, dur=dur,
            depth_primary_ppm=4500, depth_secondary_ppm=3500, noise_ppm=50,
        )
        lk_mock = _make_lk_mock(time_arr, flux_arr)

        with patch.dict("sys.modules", {"lightkurve": lk_mock}):
            verdict, detail = double_period_eb_check(
                "278536652", period, t0, dur, sigma_thresh=3.0
            )

        assert verdict == "eb", f"Expected 'eb', got '{verdict}': {detail}"
        assert "ppm" in detail

    def test_highly_asymmetric_eclipses_eb(self):
        """Primary 6000 ppm, secondary 1500 ppm — very obvious EB."""
        period = 5.0
        t0 = 1300.0
        dur = 0.1

        time_arr, flux_arr = _make_synthetic_lc(
            period=period, t0=t0, dur=dur,
            depth_primary_ppm=6000, depth_secondary_ppm=1500, noise_ppm=100,
        )
        lk_mock = _make_lk_mock(time_arr, flux_arr)

        with patch.dict("sys.modules", {"lightkurve": lk_mock}):
            verdict, detail = double_period_eb_check(
                "999999", period, t0, dur, sigma_thresh=3.0
            )

        assert verdict == "eb", f"Expected 'eb', got '{verdict}': {detail}"

    def test_sigma_threshold_respected(self):
        """If sigma < threshold, verdict must be 'ok' even with modest depth difference."""
        period = 7.124
        t0 = 1329.18
        dur = 0.14

        # Nearly equal depths → sigma < 3
        time_arr, flux_arr = _make_synthetic_lc(
            period=period, t0=t0, dur=dur,
            depth_primary_ppm=4500, depth_secondary_ppm=4400, noise_ppm=300,
        )
        lk_mock = _make_lk_mock(time_arr, flux_arr)

        with patch.dict("sys.modules", {"lightkurve": lk_mock}):
            # Use a very high threshold so that marginal differences are not flagged
            verdict, detail = double_period_eb_check(
                "278536652", period, t0, dur, sigma_thresh=100.0
            )

        assert verdict in ("ok", "unknown"), (
            f"High threshold should yield 'ok' or 'unknown', got '{verdict}': {detail}"
        )


class TestDoublePeriodEbCheckOk:
    """Cases that should return verdict='ok'."""

    def test_equal_eclipses_are_ok(self):
        """Twin-eclipse EB with identical depths: not distinguishable as double-period EB."""
        period = 7.124
        t0 = 1329.18
        dur = 0.14

        # Both eclipses the same depth → sigma ≈ 0
        time_arr, flux_arr = _make_synthetic_lc(
            period=period, t0=t0, dur=dur,
            depth_primary_ppm=4500, depth_secondary_ppm=4500, noise_ppm=50,
        )
        lk_mock = _make_lk_mock(time_arr, flux_arr)

        with patch.dict("sys.modules", {"lightkurve": lk_mock}):
            verdict, detail = double_period_eb_check(
                "278536652", period, t0, dur, sigma_thresh=3.0
            )

        # With identical depths the delta is in the noise; verdict should be ok
        assert verdict in ("ok", "unknown"), (
            f"Identical depths should NOT be flagged as EB; got '{verdict}': {detail}"
        )

    def test_single_eclipse_no_secondary_ok(self):
        """Genuine planet signal: only primary eclipse, no secondary at phase 0.5."""
        period = 7.124
        t0 = 1329.18
        dur = 0.14

        rng = np.random.default_rng(0)
        time_arr = np.sort(rng.uniform(t0, t0 + 40.0, 5000))
        flux_arr = 1.0 + rng.normal(0, 200 / 1e6, 5000)

        P2 = 2 * period
        phase2 = ((time_arr - t0) % P2) / P2
        win = dur / P2
        in_primary = np.abs(phase2) < win
        in_primary |= np.abs(phase2 - 1.0) < win
        flux_arr[in_primary] -= 4500 / 1e6  # only primary

        lk_mock = _make_lk_mock(time_arr, flux_arr)

        with patch.dict("sys.modules", {"lightkurve": lk_mock}):
            verdict, detail = double_period_eb_check(
                "278536652", period, t0, dur, sigma_thresh=3.0
            )

        # Phase 0.5 is pure noise (depthB ≈ 0, depthB/errB ≪ 1.5); the
        # secondary-detection guard must prevent a false "eb" verdict here.
        # A genuine planet with no secondary MUST NOT be flagged as EB.
        assert verdict in ("ok", "unknown"), (
            f"Real planet (no secondary) must not be flagged as EB; got '{verdict}': {detail}"
        )


class TestDoublePeriodEbCheckUnknown:
    """Cases that should return verdict='unknown' without crashing."""

    def test_lightkurve_not_installed(self):
        """Missing lightkurve → unknown."""
        with patch.dict("sys.modules", {"lightkurve": None}):
            # When the module is set to None, importing it raises ImportError
            verdict, detail = double_period_eb_check("123456", 7.0, 1300.0, 0.1)
        assert verdict == "unknown"

    def test_empty_search_result(self):
        """Empty search result (no SPOC 2-min data) → unknown."""
        lk_mock = MagicMock()
        sr_mock = MagicMock()
        sr_mock.__len__ = MagicMock(return_value=0)
        lk_mock.search_lightcurve.return_value = sr_mock

        with patch.dict("sys.modules", {"lightkurve": lk_mock}):
            verdict, detail = double_period_eb_check("123456", 7.0, 1300.0, 0.1)

        assert verdict == "unknown"

    def test_download_all_returns_none(self):
        """download_all returning None → unknown."""
        lk_mock = MagicMock()
        sr_mock = MagicMock()
        sr_mock.__len__ = MagicMock(return_value=1)
        sr_mock.download_all.return_value = None
        lk_mock.search_lightcurve.return_value = sr_mock

        with patch.dict("sys.modules", {"lightkurve": lk_mock}):
            verdict, detail = double_period_eb_check("123456", 7.0, 1300.0, 0.1)

        assert verdict == "unknown"

    def test_fetch_exception_yields_unknown(self):
        """Any exception during fetch → unknown (pipeline must not crash)."""
        lk_mock = MagicMock()
        lk_mock.search_lightcurve.side_effect = RuntimeError("MAST connection refused")

        with patch.dict("sys.modules", {"lightkurve": lk_mock}):
            verdict, detail = double_period_eb_check("123456", 7.0, 1300.0, 0.1)

        assert verdict == "unknown"
        assert "RuntimeError" in detail or "error" in detail.lower()

    def test_too_few_points_yields_unknown(self):
        """Fewer than 100 finite points → unknown."""
        rng = np.random.default_rng(7)
        time_arr = rng.uniform(1300.0, 1320.0, 50)  # only 50 points
        flux_arr = np.ones(50)

        lk_mock = _make_lk_mock(time_arr, flux_arr)

        with patch.dict("sys.modules", {"lightkurve": lk_mock}):
            verdict, detail = double_period_eb_check("123456", 7.0, 1300.0, 0.1)

        assert verdict == "unknown"

    def test_insufficient_in_transit_points_yields_unknown(self):
        """LC with no points near the eclipse windows → unknown (not a crash)."""
        rng = np.random.default_rng(99)
        # Only OOT noise, phases 0.1-0.4 and 0.6-0.9 in the 2P fold
        period = 7.124
        t0 = 1329.18
        P2 = 2 * period
        # Generate points only in phases 0.1-0.4 and 0.6-0.9
        n = 500
        time_arr = np.array([
            t0 + (i / n) * 40 + (0.1 + 0.3 * rng.random()) * P2
            for i in range(n)
        ])
        flux_arr = 1.0 + rng.normal(0, 300 / 1e6, n)

        lk_mock = _make_lk_mock(time_arr, flux_arr)

        with patch.dict("sys.modules", {"lightkurve": lk_mock}):
            verdict, detail = double_period_eb_check(
                "123456", period, t0, 0.01, sigma_thresh=3.0  # very short dur
            )

        # With dur=0.01 d, win fraction = 0.01/(2*7.124) ≈ 0.0007 — barely any points
        assert verdict in ("unknown", "ok", "eb")  # must not crash


class TestDoublePeriodEbCheckEdgeCases:
    """Edge-case robustness tests."""

    def test_returns_tuple_always(self):
        """The function must always return a (str, str) tuple."""
        lk_mock = MagicMock()
        lk_mock.search_lightcurve.side_effect = Exception("boom")

        with patch.dict("sys.modules", {"lightkurve": lk_mock}):
            result = double_period_eb_check("0", 1.0, 0.0, 0.05)

        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], str)
        assert isinstance(result[1], str)

    def test_verdict_in_valid_set(self):
        """Verdict is always one of the three valid strings."""
        lk_mock = MagicMock()
        lk_mock.search_lightcurve.side_effect = ValueError("bad")

        with patch.dict("sys.modules", {"lightkurve": lk_mock}):
            verdict, _ = double_period_eb_check("0", 7.0, 1300.0, 0.1)

        assert verdict in {"eb", "ok", "unknown"}
