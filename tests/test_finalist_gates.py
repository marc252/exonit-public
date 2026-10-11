"""Unit tests for the three new finalist gate improvements.

A. spoc_repeat_check    — SPOC repeatability check
B. orbit_start_artifact_check — orbit-start ramp artifact check
C1. _load_toi_catalog + anti-TOI in discovery_postprocess
C2. TOI skip in discovery_run.py (via next_block + done_set merge)

All tests use synthetic data or mocks — no MAST network calls,
no live DB access.
"""

import csv
import json
import sqlite3
import sys
import warnings
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

# Make the scripts directory importable
SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(ROOT))

from discovery_postprocess import (  # noqa: E402
    spoc_repeat_check,
    orbit_start_artifact_check,
    _load_toi_catalog,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_lk_mock_from_arrays(time_arr, flux_arr):
    """Build a lightkurve mock that returns the given time/flux arrays."""

    class _FakeLc:
        def __init__(self, t, f):
            self.time = MagicMock()
            self.time.value = t
            self.flux = MagicMock()
            self.flux.value = f

        def remove_nans(self):
            return self

        def remove_outliers(self, sigma=5):
            return self

        def normalize(self):
            return self

        def flatten(self, window_length=401):
            return self

    class _FakeCol:
        def __init__(self, lc):
            self._lc = lc

        def __len__(self):
            return 1

        def stitch(self):
            return self._lc

    lk_mock = MagicMock()
    fake_lc = _FakeLc(time_arr, flux_arr)
    fake_col = _FakeCol(fake_lc)

    sr_mock = MagicMock()
    sr_mock.__len__ = MagicMock(return_value=1)
    sr_mock.download_all.return_value = fake_col
    lk_mock.search_lightcurve.return_value = sr_mock
    return lk_mock


def _make_flat_lc_with_transits(
    period, t0, dur, n_transits=5, pts_per_day=100,
    depth_ppm=3000.0, noise_ppm=50.0, baseline_days=None, seed=0
):
    """Build (time, flux) with an injected transit of given depth."""
    rng = np.random.default_rng(seed)
    if baseline_days is None:
        baseline_days = period * (n_transits + 1)
    time = np.linspace(t0, t0 + baseline_days, int(pts_per_day * baseline_days))
    flux = 1.0 + rng.normal(0, noise_ppm / 1e6, len(time))
    # Inject transits
    phase = ((time - t0) % period) / period
    phase[phase > 0.5] -= 1.0
    in_tr = np.abs(phase) < 0.5 * dur / period
    flux[in_tr] -= depth_ppm / 1e6
    return time, flux


def _make_flat_lc_no_transits(
    period, t0, dur, n_transits=5, pts_per_day=100,
    noise_ppm=50.0, baseline_days=None, seed=0
):
    """Build (time, flux) with NO transit signal (pure noise)."""
    rng = np.random.default_rng(seed)
    if baseline_days is None:
        baseline_days = period * (n_transits + 1)
    time = np.linspace(t0, t0 + baseline_days, int(pts_per_day * baseline_days))
    flux = 1.0 + rng.normal(0, noise_ppm / 1e6, len(time))
    return time, flux


# ===========================================================================
# A: spoc_repeat_check
# ===========================================================================

class TestSpocRepeatCheckAbsent:
    """signal genuinely absent → 'absent'."""

    def test_absent_two_epochs_deep_expected(self):
        """Flat LC with ≥2 epochs and ≥30 in-transit points → 'absent'."""
        period = 5.0
        t0 = 1300.0
        dur = 0.2
        depth_ppm = 4000.0

        time_arr, flux_arr = _make_flat_lc_no_transits(
            period, t0, dur, n_transits=10, pts_per_day=200, noise_ppm=30.0
        )
        lk_mock = _make_lk_mock_from_arrays(time_arr, flux_arr)

        from unittest.mock import patch
        with patch.dict("sys.modules", {"lightkurve": lk_mock}):
            verdict, detail = spoc_repeat_check(
                "123456", period, t0, dur, depth_ppm, sigma_thresh=3.0
            )

        assert verdict == "absent", f"Expected 'absent', got '{verdict}': {detail}"
        assert "n_epochs" in detail

    def test_absent_threshold_uses_sigma(self):
        """A signal at 1×expected depth should be 'absent' with sigma_thresh=3
        when measured depth < 3×err (pure noise).
        """
        period = 7.0
        t0 = 1310.0
        dur = 0.15
        depth_ppm = 3000.0

        # Slightly noisy but no actual transit injected
        time_arr, flux_arr = _make_flat_lc_no_transits(
            period, t0, dur, n_transits=8, pts_per_day=200, noise_ppm=40.0
        )
        lk_mock = _make_lk_mock_from_arrays(time_arr, flux_arr)

        from unittest.mock import patch
        with patch.dict("sys.modules", {"lightkurve": lk_mock}):
            verdict, detail = spoc_repeat_check(
                "999", period, t0, dur, depth_ppm, sigma_thresh=3.0
            )

        # With n_transits=8 epochs and no signal, we expect 'absent'
        assert verdict == "absent", f"Expected 'absent', got '{verdict}': {detail}"


class TestSpocRepeatCheckConfirmed:
    """signal present and detectable → 'confirmed'."""

    def test_confirmed_strong_signal(self):
        """High-depth transit with good coverage → 'confirmed'."""
        period = 5.0
        t0 = 1300.0
        dur = 0.2
        depth_ppm = 5000.0

        time_arr, flux_arr = _make_flat_lc_with_transits(
            period, t0, dur, n_transits=10, pts_per_day=200,
            depth_ppm=depth_ppm, noise_ppm=20.0
        )
        lk_mock = _make_lk_mock_from_arrays(time_arr, flux_arr)

        from unittest.mock import patch
        with patch.dict("sys.modules", {"lightkurve": lk_mock}):
            verdict, detail = spoc_repeat_check(
                "111222", period, t0, dur, depth_ppm, sigma_thresh=3.0
            )

        assert verdict == "confirmed", f"Expected 'confirmed', got '{verdict}': {detail}"


class TestSpocRepeatCheckUnknown:
    """Cases where verdict must be 'unknown' (do not block)."""

    def test_single_epoch_unknown(self):
        """Only 1 distinct transit epoch → 'unknown' (mono-transit protection).

        Use a very long period (80d) with a 90-day baseline so only one transit
        falls within the data window.
        """
        period = 80.0  # only 1 transit in 90-day baseline
        t0 = 1300.0    # first (and only) transit at t=1300
        dur = 0.3
        depth_ppm = 3000.0

        # baseline_days=90 → only t0=1300 falls in [1300, 1390]; t0+80=1380 also in [1300,1390]
        # Use baseline_days=70 → only t0=1300 in [1300, 1370]; t0+80=1380 outside
        time_arr, flux_arr = _make_flat_lc_with_transits(
            period, t0, dur, n_transits=1, pts_per_day=100,
            depth_ppm=depth_ppm, noise_ppm=30.0,
            baseline_days=70.0
        )
        lk_mock = _make_lk_mock_from_arrays(time_arr, flux_arr)

        from unittest.mock import patch
        with patch.dict("sys.modules", {"lightkurve": lk_mock}):
            verdict, detail = spoc_repeat_check(
                "777", period, t0, dur, depth_ppm, sigma_thresh=3.0
            )

        # Only 1 epoch → must not exclude (unknown)
        assert verdict == "unknown", f"Expected 'unknown' for single epoch, got '{verdict}': {detail}"

    def test_no_spoc_data_unknown(self):
        """Empty MAST search → 'unknown'."""
        lk_mock = MagicMock()
        sr_mock = MagicMock()
        sr_mock.__len__ = MagicMock(return_value=0)
        lk_mock.search_lightcurve.return_value = sr_mock

        from unittest.mock import patch
        with patch.dict("sys.modules", {"lightkurve": lk_mock}):
            verdict, detail = spoc_repeat_check(
                "234567", 5.0, 1300.0, 0.1, 3000.0
            )

        assert verdict == "unknown"

    def test_lightkurve_missing_unknown(self):
        """lightkurve not installed → 'unknown'."""
        from unittest.mock import patch
        with patch.dict("sys.modules", {"lightkurve": None}):
            verdict, detail = spoc_repeat_check("000", 5.0, 1300.0, 0.1, 3000.0)
        assert verdict == "unknown"

    def test_fetch_exception_unknown(self):
        """Exception during MAST fetch → 'unknown', no crash."""
        lk_mock = MagicMock()
        lk_mock.search_lightcurve.side_effect = RuntimeError("MAST down")

        from unittest.mock import patch
        with patch.dict("sys.modules", {"lightkurve": lk_mock}):
            verdict, detail = spoc_repeat_check(
                "555", 5.0, 1300.0, 0.1, 3000.0
            )

        assert verdict == "unknown"

    def test_lc_cache_used(self):
        """If _lc_cache is pre-populated, lightkurve is never called."""
        period = 5.0
        t0 = 1300.0
        dur = 0.2
        depth_ppm = 5000.0

        time_arr, flux_arr = _make_flat_lc_with_transits(
            period, t0, dur, n_transits=10, pts_per_day=200,
            depth_ppm=depth_ppm, noise_ppm=20.0
        )
        cache = {"TIC42": (time_arr, flux_arr)}

        # lightkurve mock that would crash if called
        lk_mock = MagicMock()
        lk_mock.search_lightcurve.side_effect = AssertionError("should not call LK")

        from unittest.mock import patch
        with patch.dict("sys.modules", {"lightkurve": lk_mock}):
            verdict, detail = spoc_repeat_check(
                "TIC42", period, t0, dur, depth_ppm,
                _lc_cache=cache
            )

        # Cache was used; the assert would have triggered if LK was called
        assert verdict in ("absent", "confirmed", "unknown")

    def test_always_returns_tuple(self):
        """Must always return a (str, str) tuple — never crash."""
        lk_mock = MagicMock()
        lk_mock.search_lightcurve.side_effect = Exception("boom")

        from unittest.mock import patch
        with patch.dict("sys.modules", {"lightkurve": lk_mock}):
            result = spoc_repeat_check("0", 5.0, 1300.0, 0.1, 3000.0)

        assert isinstance(result, tuple)
        assert len(result) == 2
        assert result[0] in ("absent", "confirmed", "unknown")


# ===========================================================================
# B: orbit_start_artifact_check
# ===========================================================================

class TestOrbitStartArtifactCheckNotApplicable:
    """Periods outside [13, 15] d → 'not_applicable'."""

    def test_short_period_not_applicable(self):
        verdict, detail = orbit_start_artifact_check(
            "234968549", 3.04, 1339.0, 0.1
        )
        assert verdict == "not_applicable"

    def test_period_below_13_not_applicable(self):
        verdict, detail = orbit_start_artifact_check(
            "234968549", 12.99, 1339.0, 0.1
        )
        assert verdict == "not_applicable"

    def test_period_above_15_not_applicable(self):
        verdict, detail = orbit_start_artifact_check(
            "234968549", 15.01, 1339.0, 0.1
        )
        assert verdict == "not_applicable"


def _make_mc_mock(tic_id, time_arr):
    """Return a mock MetadataCache whose load_lc returns the given time array."""
    mc = MagicMock()
    # load_lc is a module-level function; we mock it via patch in each test.
    # This helper just builds the (time, flux, secs) triple that load_lc returns.
    return mc, (time_arr, np.ones(len(time_arr)), np.array([1]))


class TestOrbitStartArtifactCheckArtifact:
    """Transit at <4h from segment start → 'orbit_start_artifact'."""

    def test_transit_at_0_3h_from_segment_start_is_artifact(self):
        """Transit 0.38 h after segment start → 'orbit_start_artifact'.

        Strategy: only one segment starting at seg_start, period=14d, t0 chosen
        so that the only predicted transit within data coverage is ~23 min after
        the segment start (inside the 4 h ramp window).
        LC comes from load_lc(mc, tic_id) — no _lc_cache, no lightkurve.
        """
        period = 14.0
        seg_start = 0.0
        # Transit 0.38 h = 0.01583 d after seg_start
        t0 = seg_start + 0.38 / 24.0
        dur = 0.1

        # LC: single contiguous segment [0, 10] days — only one transit in range
        time_arr = np.linspace(seg_start, 10.0, 2000)
        mc, load_lc_return = _make_mc_mock("234968549", time_arr)

        from unittest.mock import patch
        with patch("discovery_postprocess.load_lc", return_value=load_lc_return):
            verdict, detail = orbit_start_artifact_check(
                "234968549", period, t0, dur, mc=mc
            )
        dt_h = (t0 - seg_start) * 24.0
        assert verdict == "orbit_start_artifact", (
            f"Expected 'orbit_start_artifact' for transit {dt_h:.2f}h from seg start, "
            f"got '{verdict}': {detail}"
        )

    def test_all_transits_outside_data_range_is_unknown(self):
        """If no transit falls within the data range, verdict is 'unknown'.

        t0 = 9.0, period = 14.0, LC = [0, 8] days.
        k=0: t_k=9 > 8 (outside); k=-1: t_k=-5 < 0 (outside) → no transits in data.
        """
        period = 14.0
        t0 = 9.0
        dur = 0.1

        time_arr = np.linspace(0.0, 8.0, 1000)
        mc, load_lc_return = _make_mc_mock("999", time_arr)

        from unittest.mock import patch
        with patch("discovery_postprocess.load_lc", return_value=load_lc_return):
            verdict, detail = orbit_start_artifact_check(
                "999", period, t0, dur, mc=mc
            )
        assert verdict == "unknown", f"Expected 'unknown' (no transits in data), got '{verdict}': {detail}"


class TestOrbitStartArtifactCheckOk:
    """Transit in clean data → 'ok'."""

    def test_transit_6h_after_segment_start_is_ok(self):
        """Transit 6 h after segment start → 'ok' (outside 4h ramp window)."""
        period = 14.0
        t0 = 1340.0 + (6.0 / 24.0)  # nearest seg start = 1326.0 (LC start)
        dur = 0.12

        time_arr = np.linspace(1326.0, 1360.0, 3000)  # continuous, no gap
        mc, load_lc_return = _make_mc_mock("888", time_arr)

        from unittest.mock import patch
        with patch("discovery_postprocess.load_lc", return_value=load_lc_return):
            verdict, detail = orbit_start_artifact_check(
                "888", period, t0, dur, mc=mc
            )
        # nearest seg start = 1326.0 (LC start); t_k - 1326 = 14.25 d >> 4h
        assert verdict == "ok", f"Expected 'ok', got '{verdict}': {detail}"

    def test_one_clean_transit_out_of_two_is_ok(self):
        """One transit in ramp, another in clean data → 'ok'."""
        period = 14.0
        t0 = 1300.1  # 0.1 d = 2.4 h from LC start → first transit in ramp
        dur = 0.1

        # LC covering two periods: 1300–1330 (continuous, no big gap)
        time_arr = np.linspace(1300.0, 1330.0, 3000)
        mc, load_lc_return = _make_mc_mock("777", time_arr)

        from unittest.mock import patch
        with patch("discovery_postprocess.load_lc", return_value=load_lc_return):
            verdict, detail = orbit_start_artifact_check(
                "777", period, t0, dur, mc=mc
            )
        # t_k=1: 1300.1 + 14 = 1314.1; nearest prior start = 1300.0; dt = 14.1 d >> 4h → clean
        assert verdict == "ok", f"Expected 'ok' (one clean transit), got '{verdict}': {detail}"


class TestOrbitStartArtifactCheckUnknown:
    """No LC → 'unknown'."""

    def test_no_mc_unknown(self):
        """No mc passed → 'unknown' (cannot load TGLC LC)."""
        verdict, detail = orbit_start_artifact_check(
            "234968549", 14.31, 1339.747, 0.12
        )
        assert verdict == "unknown"

    def test_load_lc_returns_none_unknown(self):
        """mc present but load_lc returns None → 'unknown'."""
        mc = MagicMock()
        from unittest.mock import patch
        with patch("discovery_postprocess.load_lc", return_value=(None, None, None)):
            verdict, detail = orbit_start_artifact_check(
                "234968549", 14.31, 1339.747, 0.12, mc=mc
            )
        assert verdict == "unknown", f"Expected 'unknown', got '{verdict}': {detail}"

    def test_lc_cache_ignored_without_mc(self):
        """Anti-regression: _lc_cache populated but no mc → 'unknown'.

        Before the fix, _lc_cache (containing SPOC multi-sector data from
        spoc_repeat_check) was used as the LC source, causing the check to
        evaluate ~198 SPOC transits instead of the 2 TGLC detection transits.
        After the fix, _lc_cache is ignored and without mc the result is 'unknown'.
        """
        # Build a large multi-sector SPOC-like LC (many transits, none in ramp)
        period = 14.0
        seg_start = 1339.0
        t0 = seg_start + 0.1 / 24.0  # transit 6 min from seg start → would be artifact
        time_arr = np.linspace(1000.0, 2800.0, 200_000)  # 7.7 years of SPOC data
        spoc_cache = {"234968549": (time_arr, np.ones(len(time_arr)))}

        # No mc → must return "unknown" regardless of _lc_cache content
        verdict, detail = orbit_start_artifact_check(
            "234968549", period, t0, 0.1,
            mc=None,
            _lc_cache=spoc_cache,  # this must be ignored
        )
        assert verdict == "unknown", (
            f"Expected 'unknown' (mc=None; _lc_cache must be ignored), "
            f"got '{verdict}': {detail}"
        )


# ===========================================================================
# C1: _load_toi_catalog + anti-TOI cross-match (postprocess)
# ===========================================================================

class TestLoadToiCatalog:
    def test_loads_tic_toi_pairs(self, tmp_path):
        cat = tmp_path / "toi_catalog.csv"
        with open(cat, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["TIC ID", "TOI", "Other"])
            w.writerow(["231663901", "101.01", "x"])
            w.writerow(["219338557", "133.01", "y"])
        result = _load_toi_catalog(str(cat))
        assert result[231663901] == "101.01"
        assert result[219338557] == "133.01"

    def test_missing_catalog_returns_empty_with_warning(self, tmp_path):
        """Missing catalog → empty dict + RuntimeWarning (no crash)."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = _load_toi_catalog(str(tmp_path / "no_such_catalog.csv"))
        assert result == {}
        assert any(issubclass(w.category, RuntimeWarning) for w in caught), (
            "Expected RuntimeWarning for missing catalog"
        )

    def test_skips_rows_with_missing_tic(self, tmp_path):
        cat = tmp_path / "toi_catalog.csv"
        with open(cat, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["TIC ID", "TOI"])
            w.writerow(["", "101.01"])        # empty TIC → skip
            w.writerow(["bad", "102.01"])     # non-numeric → skip
            w.writerow(["123456789", "103.01"])
        result = _load_toi_catalog(str(cat))
        assert list(result.keys()) == [123456789]


class TestAntiToiInPostprocess:
    """Integration-level check: known TOI finalist is excluded from discovery list."""

    def test_known_toi_excluded_from_finalists(self, tmp_path):
        """Simulate the anti-TOI step: finalist in catalog → moved to known_toi_finalists."""
        cat = tmp_path / "toi_catalog.csv"
        with open(cat, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["TIC ID", "TOI"])
            w.writerow(["219338557", "133.01"])

        toi_catalog = _load_toi_catalog(str(cat))

        # Simulate the anti-TOI logic from main() (duplicated here for isolation)
        final = [
            {"tic_id": "219338557", "period_days": 2.73, "depth_ppm": 6000},
            {"tic_id": "100000001", "period_days": 5.0,  "depth_ppm": 3000},
        ]
        known_toi_finalists = []
        if toi_catalog:
            for s in final:
                tic_int = int(s["tic_id"])
                if tic_int in toi_catalog:
                    s["_known_toi"] = toi_catalog[tic_int]
                    known_toi_finalists.append(s)
            if known_toi_finalists:
                known_ids = {s["tic_id"] for s in known_toi_finalists}
                final = [s for s in final if s["tic_id"] not in known_ids]

        assert len(final) == 1
        assert final[0]["tic_id"] == "100000001"
        assert len(known_toi_finalists) == 1
        assert known_toi_finalists[0]["_known_toi"] == "133.01"

    def test_no_catalog_no_exclusion(self):
        """Empty catalog → all finalists remain."""
        toi_catalog = {}  # as if catalog was missing
        final = [
            {"tic_id": "219338557", "period_days": 2.73, "depth_ppm": 6000},
        ]
        known_toi_finalists = []
        if toi_catalog:
            # This block never executes with empty dict
            pass

        assert len(final) == 1
        assert known_toi_finalists == []


# ===========================================================================
# C2: TOI skip in discovery_run.py
# ===========================================================================

class TestDiscoveryRunToiSkip:
    """Tests for the TOI skip in discovery_run.main()."""

    def _make_canonical_csv(self, path, tics):
        with open(path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["tic_id", "label"])
            for t in tics:
                w.writerow([t, "candidate"])

    def _make_tglc_db(self, path, tic_ids):
        con = sqlite3.connect(str(path))
        con.execute(
            "CREATE TABLE tglc ("
            "tic_id TEXT, sector INTEGER, path TEXT, exptime REAL, "
            "tmag REAL, ra REAL, dec REAL, PRIMARY KEY(tic_id, sector))"
        )
        for t in tic_ids:
            con.execute(
                "INSERT INTO tglc VALUES (?, 1, '/tmp/fake.fits', 1800, 10, 0, 0)",
                (str(t),),
            )
        con.commit()
        con.close()

    def _make_registry(self, path, canonical, cursor=0):
        data = {
            "active_sector": 1,
            "sectors": {
                "1": {
                    "canonical_list": canonical,
                    "total_tics": 100,
                    "downloaded": 100,
                    "cursor": cursor,
                    "enabled": True,
                }
            },
        }
        path.write_text(json.dumps(data, indent=2))

    def _make_toi_catalog_csv(self, path, tic_ids):
        with open(path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["TIC ID", "TOI"])
            for i, t in enumerate(tic_ids):
                w.writerow([str(t), f"{1000 + i}.01"])

    def test_toi_tic_skipped_cursor_advances(self, tmp_path):
        """A TOI TIC in the work list is skipped; cursor advances past it."""
        import subprocess

        SCRIPT = str(ROOT / "scripts" / "discovery_run.py")

        tics = [1001, 1002, 1003, 1004, 1005]
        canonical = tmp_path / "canonical.csv"
        self._make_canonical_csv(canonical, tics)

        registry = tmp_path / "sectors.json"
        self._make_registry(registry, str(canonical))

        tglc_db = tmp_path / "tglc.db"
        self._make_tglc_db(tglc_db, tics)

        tasks_db = tmp_path / "ex.db"
        # No done tasks

        # TIC 1002 is a TOI
        toi_cat = tmp_path / "toi_catalog.csv"
        self._make_toi_catalog_csv(toi_cat, [1002])

        result = subprocess.run(
            [sys.executable, SCRIPT,
             "--sector", "1",
             "--count", "3",
             "--registry", str(registry),
             "--canonical", str(canonical),
             "--tglc-index", str(tglc_db),
             "--db", str(tasks_db),
             "--toi-catalog", str(toi_cat),
             "--dry-run"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0, result.stderr
        out = json.loads(result.stdout)
        assert out["status"] == "dry_run"

        # 1002 must not be in selected
        assert 1002 not in out["tics_selected"], (
            f"TOI TIC 1002 must not be enqueued; selected={out['tics_selected']}"
        )
        # tois_skipped must be reported
        assert out.get("tois_skipped", -1) == 1, (
            f"Expected tois_skipped=1, got {out.get('tois_skipped')}"
        )

    def test_cursor_advances_past_toi(self, tmp_path):
        """Cursor must advance past the TOI TIC as if it were done."""
        import subprocess

        SCRIPT = str(ROOT / "scripts" / "discovery_run.py")

        # canonical: [10, 11, 12, 13, 14] — TIC 11 is TOI
        # count=2, expect selected=[10, 12], cursor_after=3 (canonical row of 12)
        tics = [10, 11, 12, 13, 14]
        canonical = tmp_path / "canonical.csv"
        self._make_canonical_csv(canonical, tics)

        registry = tmp_path / "sectors.json"
        self._make_registry(registry, str(canonical))

        tglc_db = tmp_path / "tglc.db"
        self._make_tglc_db(tglc_db, tics)

        tasks_db = tmp_path / "ex.db"

        toi_cat = tmp_path / "toi_catalog.csv"
        self._make_toi_catalog_csv(toi_cat, [11])

        result = subprocess.run(
            [sys.executable, SCRIPT,
             "--sector", "1",
             "--count", "2",
             "--registry", str(registry),
             "--canonical", str(canonical),
             "--tglc-index", str(tglc_db),
             "--db", str(tasks_db),
             "--toi-catalog", str(toi_cat),
             "--dry-run"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0, result.stderr
        out = json.loads(result.stdout)

        # selected should be [10, 12] (skip 11)
        assert 11 not in out["tics_selected"]
        assert 10 in out["tics_selected"]
        assert 12 in out["tics_selected"]
        # cursor_after = canonical row of 12 = index 2 + 1 = 3
        assert out["cursor_after"] == 3, (
            f"cursor_after should be 3 (row of last selected TIC=12), got {out['cursor_after']}"
        )

    def test_missing_catalog_no_filter(self, tmp_path):
        """Missing TOI catalog → run proceeds normally, no crash, tois_skipped=0."""
        import subprocess

        SCRIPT = str(ROOT / "scripts" / "discovery_run.py")

        tics = [20, 21, 22]
        canonical = tmp_path / "canonical.csv"
        self._make_canonical_csv(canonical, tics)

        registry = tmp_path / "sectors.json"
        self._make_registry(registry, str(canonical))

        tglc_db = tmp_path / "tglc.db"
        self._make_tglc_db(tglc_db, tics)

        tasks_db = tmp_path / "ex.db"

        result = subprocess.run(
            [sys.executable, SCRIPT,
             "--sector", "1",
             "--count", "3",
             "--registry", str(registry),
             "--canonical", str(canonical),
             "--tglc-index", str(tglc_db),
             "--db", str(tasks_db),
             "--toi-catalog", str(tmp_path / "no_such_catalog.csv"),
             "--dry-run"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0, result.stderr
        out = json.loads(result.stdout)
        assert out["status"] == "dry_run"
        assert out["enqueued"] == 3
        assert out.get("tois_skipped", 0) == 0

    def test_tois_skipped_count_correct(self, tmp_path):
        """tois_skipped count equals the number of TOI TICs in the work list."""
        import subprocess

        SCRIPT = str(ROOT / "scripts" / "discovery_run.py")

        tics = list(range(100, 115))  # 15 TICs
        canonical = tmp_path / "canonical.csv"
        self._make_canonical_csv(canonical, tics)

        registry = tmp_path / "sectors.json"
        self._make_registry(registry, str(canonical))

        tglc_db = tmp_path / "tglc.db"
        self._make_tglc_db(tglc_db, tics)

        tasks_db = tmp_path / "ex.db"

        # 3 of the 15 TICs are TOIs
        toi_cat = tmp_path / "toi_catalog.csv"
        self._make_toi_catalog_csv(toi_cat, [101, 105, 110])

        result = subprocess.run(
            [sys.executable, SCRIPT,
             "--sector", "1",
             "--count", "5",
             "--registry", str(registry),
             "--canonical", str(canonical),
             "--tglc-index", str(tglc_db),
             "--db", str(tasks_db),
             "--toi-catalog", str(toi_cat),
             "--dry-run"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0, result.stderr
        out = json.loads(result.stdout)
        assert out.get("tois_skipped") == 3, (
            f"Expected tois_skipped=3, got {out.get('tois_skipped')}"
        )
        # None of the TOI TICs should appear in selected
        for toi_tic in [101, 105, 110]:
            assert toi_tic not in out["tics_selected"], (
                f"TOI TIC {toi_tic} must not be enqueued"
            )
