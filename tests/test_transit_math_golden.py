"""Golden tests for exonit.transit_math (D6, docs/reviews/2026-07-02-opus48-
cleanup-review.md): "fold-and-measure-depth reimplemented 4x" was collapsed
into one shared ``fold_and_measure_depth`` engine. The bar for this refactor
is BIT-IDENTICAL behaviour at every call site.

Each of the 4 pre-refactor implementations is pasted VERBATIM below as an
``_old_*`` helper (copied from the worktree's pre-edit
scripts/discovery_postprocess.py / exonit/ffi_finalist_checks.py) and
compared against the actual post-refactor code, using ``==``/
``np.array_equal`` -- never ``allclose`` -- over a battery of seeded
synthetic light curves (V-shape, box, secondary eclipse at phase 0.5, gapped
coverage, few-point windows) plus one real SPOC light curve fixture.

Sites covered:
  1. secondary_depth               (scripts/discovery_postprocess.py)
  2. double_period_eb_check's      (scripts/discovery_postprocess.py)
     _measure_depth closure
  3. spoc_repeat_check's inline    (scripts/discovery_postprocess.py)
     fold block
  4. _fold_and_measure             (exonit/ffi_finalist_checks.py)

For sites 2 and 3 (which have surrounding guard/control-flow, not just a
pure leaf computation) this file also pastes the OLD *whole function*
verbatim and compares its (verdict, detail) output end-to-end against the
real, post-refactor function -- not just the inner fold arithmetic.
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from astropy.io import fits

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SCRIPTS))

from exonit.transit_math import fold_and_measure_depth  # noqa: E402
from exonit.ffi_finalist_checks import _fold_and_measure  # noqa: E402
from discovery_postprocess import (  # noqa: E402
    secondary_depth,
    double_period_eb_check,
    spoc_repeat_check,
)


# ===========================================================================
# OLD verbatim implementations (frozen copies -- do NOT "clean up" these to
# match the new code; their entire purpose is to be an untouched fossil of
# the pre-refactor behaviour).
# ===========================================================================

def _old_secondary_depth(time, flux, period, t0, dur):
    """Verbatim copy of scripts/discovery_postprocess.py::secondary_depth
    as it existed before the D6 refactor."""
    phase = ((time - t0) % period) / period
    win = max(dur / period, 0.01)
    sec = np.abs(phase - 0.5) < win
    base = (np.abs(phase - 0.25) < win) | (np.abs(phase - 0.75) < win)
    if sec.sum() < 5 or base.sum() < 5:
        return None
    return float((np.median(flux[base]) - np.median(flux[sec])) * 1e6)


def _old_measure_depth(centre_phase, phase_arr, flux_arr, win, baseline_frac=0.1):
    """Verbatim copy of the _measure_depth closure that used to live inside
    double_period_eb_check."""
    ph = phase_arr.copy()
    # Circular wrap so that the centre is at the middle
    ph_c = (ph - centre_phase + 0.5) % 1.0 - 0.5
    in_tr = np.abs(ph_c) < win
    oot = ((ph > 0.2) & (ph < 0.3)) | ((ph > 0.7) & (ph < 0.8))
    if in_tr.sum() < 5 or oot.sum() < 10:
        return None, None
    f_in = flux_arr[in_tr]
    f_oot = flux_arr[oot]
    depth = (np.median(f_oot) - np.median(f_in)) * 1e6
    mad = np.median(np.abs(f_in - np.median(f_in))) * 1.4826
    err = (mad / np.sqrt(in_tr.sum())) * 1e6
    err = max(err, 1.0)  # floor at 1 ppm to avoid divide-by-zero
    return depth, err


def _old_fold_and_measure(time, flux, P, t0, phase_centre, half_width):
    """Verbatim copy of exonit/ffi_finalist_checks.py::_fold_and_measure."""
    phase = ((time - t0) % P) / P
    phase[phase > 0.5] -= 1.0  # centre on 0

    in_window = np.abs(phase - phase_centre) < half_width
    oot_excl_hw = max(half_width * 3, 0.02)
    oot_mask = np.abs(phase) > oot_excl_hw

    n_in = int(in_window.sum())
    n_oot = int(oot_mask.sum())
    if n_in < 3 or n_oot < 5:
        return None

    med_in = float(np.median(flux[in_window]))
    med_oot = float(np.median(flux[oot_mask]))
    if med_oot == 0:
        return None
    return (1.0 - med_in / med_oot) * 1_000_000.0  # ppm


def _old_spoc_repeat_fold_block(time_arr, flux_arr, period, t0, dur):
    """Verbatim copy of the fold/mask/depth/err block that used to live
    inline inside spoc_repeat_check (before the guard `if n_base < 5`
    early-return). Guards are reported via a ``guard`` field instead of an
    early ``return`` so the test can assert equality of every field that
    WAS computed, and distinguish fields that never got computed because the
    original code returned before reaching them.
    """
    phase = ((time_arr - t0) % period) / period
    phase[phase > 0.5] -= 1.0  # centre on 0 -> range [-0.5, 0.5)
    half_win = 0.5 * dur / period
    in_tr = np.abs(phase) < half_win
    base_lo, base_hi = 0.75 * dur / period, 3.0 * dur / period
    baseline_mask = (np.abs(phase) > base_lo) & (np.abs(phase) < base_hi)

    n_in = int(in_tr.sum())
    n_base = int(baseline_mask.sum())

    if n_base < 5:
        return dict(guard="n_base", n_in=n_in, n_base=n_base,
                    in_tr=in_tr, baseline_mask=baseline_mask,
                    err_ppm=None, depth_measured_ppm=None)

    baseline_std = float(np.std(flux_arr[baseline_mask]))
    err_ppm = (baseline_std / max(1, np.sqrt(n_in))) * 1e6

    if n_in == 0:
        return dict(guard="n_in", n_in=n_in, n_base=n_base,
                    in_tr=in_tr, baseline_mask=baseline_mask,
                    err_ppm=err_ppm, depth_measured_ppm=None)

    depth_measured_ppm = float(
        (np.median(flux_arr[baseline_mask]) - np.median(flux_arr[in_tr])) * 1e6
    )
    return dict(guard=None, n_in=n_in, n_base=n_base,
                in_tr=in_tr, baseline_mask=baseline_mask,
                err_ppm=err_ppm, depth_measured_ppm=depth_measured_ppm)


def _old_double_period_eb_check(tic_id, period, t0, dur, sigma_thresh=3.0):
    """Verbatim copy of scripts/discovery_postprocess.py::double_period_eb_check
    as it existed before the D6 refactor (including its inline _measure_depth
    closure). Only the *fold/depth* logic matters for this test -- the fetch
    wrapper (mocked identically for both old and new below) is untouched."""
    try:
        import lightkurve as lk  # lazy import — not available on all machines
    except ImportError:
        return ("unknown", "lightkurve not installed")

    try:
        import threading as _threading

        _lc_box: list = []
        _exc_box: list = []

        def _fetch():
            try:
                sr = lk.search_lightcurve(
                    f"TIC {tic_id}", mission="TESS", author="SPOC", cadence="short"
                )
                if sr is None or len(sr) == 0:
                    return
                col = sr.download_all()
                if col is None or len(col) == 0:
                    return
                lc = col.stitch().remove_nans().normalize().flatten(window_length=401)
                _lc_box.append(lc)
            except Exception as exc:  # noqa: BLE001
                _exc_box.append(exc)

        t = _threading.Thread(target=_fetch, daemon=True)
        t.start()
        t.join(timeout=120)  # 2-minute wall-clock ceiling

        if t.is_alive():
            return ("unknown", "MAST timeout after 120 s")
        if _exc_box:
            return ("unknown", f"fetch error: {type(_exc_box[0]).__name__}: {_exc_box[0]}")
        if not _lc_box:
            return ("unknown", "no SPOC 2-min data found")

        lc = _lc_box[0]
        time_arr = np.array(lc.time.value, dtype=float)
        flux_arr = np.array(lc.flux.value, dtype=float)
        mask = np.isfinite(time_arr) & np.isfinite(flux_arr)
        time_arr, flux_arr = time_arr[mask], flux_arr[mask]
        if len(time_arr) < 100:
            return ("unknown", f"too few SPOC points: {len(time_arr)}")

        baseline = time_arr[-1] - time_arr[0]
        period_refined = period
        if baseline > 4 * period:
            try:
                from astropy.timeseries import BoxLeastSquares
                p_lo = period * 0.995
                p_hi = period * 1.005
                n_p = 200
                periods_grid = np.linspace(p_lo, p_hi, n_p)
                bls = BoxLeastSquares(time_arr, flux_arr)
                res = bls.power(periods_grid, dur)
                period_refined = float(periods_grid[np.argmax(res.power)])
            except Exception:  # noqa: BLE001
                pass

        P2 = 2.0 * period_refined
        win = max(dur / P2, 0.005)
        phase2 = ((time_arr - t0) % P2) / P2

        def _measure_depth(centre_phase, phase_arr, flux_arr, win, baseline_frac=0.1):
            ph = phase_arr.copy()
            ph_c = (ph - centre_phase + 0.5) % 1.0 - 0.5
            in_tr = np.abs(ph_c) < win
            oot = ((ph > 0.2) & (ph < 0.3)) | ((ph > 0.7) & (ph < 0.8))
            if in_tr.sum() < 5 or oot.sum() < 10:
                return None, None
            f_in = flux_arr[in_tr]
            f_oot = flux_arr[oot]
            depth = (np.median(f_oot) - np.median(f_in)) * 1e6
            mad = np.median(np.abs(f_in - np.median(f_in))) * 1.4826
            err = (mad / np.sqrt(in_tr.sum())) * 1e6
            err = max(err, 1.0)
            return depth, err

        depthA, errA = _measure_depth(0.0, phase2, flux_arr, win)
        depthB, errB = _measure_depth(0.5, phase2, flux_arr, win)

        if depthA is None or depthB is None:
            return ("unknown", f"insufficient in-transit points (winA={depthA}, winB={depthB})")

        diff = abs(depthA - depthB)
        combined_err = float(np.hypot(errA, errB))
        sigma = diff / combined_err if combined_err > 0 else 0.0

        detail = (
            f"fold_at_2P={P2:.4f}d; depthA={depthA:.0f}ppm±{errA:.0f}; "
            f"depthB={depthB:.0f}ppm±{errB:.0f}; diff={diff:.0f}ppm; "
            f"sigma={sigma:.1f} (thresh={sigma_thresh})"
        )

        if sigma >= sigma_thresh and depthA > 0 and depthB > 1.5 * errB:
            return ("eb", detail)
        return ("ok", detail)

    except Exception as exc:  # noqa: BLE001 — never crash the pipeline
        return ("unknown", f"unexpected error: {type(exc).__name__}: {exc}")


def _old_spoc_repeat_check(tic_id, period, t0, dur, depth_ppm, sigma_thresh=3.0,
                          _lc_cache: dict | None = None):
    """Verbatim copy of scripts/discovery_postprocess.py::spoc_repeat_check
    as it existed before the D6 refactor."""
    if _lc_cache is not None and tic_id in _lc_cache:
        time_arr, flux_arr = _lc_cache[tic_id]
        _from_cache = True
    else:
        _from_cache = False
        try:
            import lightkurve as lk  # lazy import — not available on all machines
        except ImportError:
            return ("unknown", "lightkurve not installed")

        _lc_box: list = []
        _exc_box: list = []

        def _fetch():
            try:
                sr = lk.search_lightcurve(
                    f"TIC {tic_id}", mission="TESS", author="SPOC", cadence="short"
                )
                if sr is None or len(sr) == 0:
                    return
                col = sr.download_all()
                if col is None or len(col) == 0:
                    return
                lc = (col.stitch()
                      .remove_nans()
                      .remove_outliers(sigma=8)
                      .normalize())
                _lc_box.append(lc)
            except Exception as exc:  # noqa: BLE001
                _exc_box.append(exc)

        import threading as _threading
        t = _threading.Thread(target=_fetch, daemon=True)
        t.start()
        t.join(timeout=120)
        if t.is_alive():
            return ("unknown", "MAST timeout after 120 s")
        if _exc_box:
            return ("unknown", f"fetch error: {type(_exc_box[0]).__name__}: {_exc_box[0]}")
        if not _lc_box:
            return ("unknown", "no SPOC 2-min data found")

        lc = _lc_box[0]
        time_arr = np.array(lc.time.value, dtype=float)
        flux_arr = np.array(lc.flux.value, dtype=float)

    mask = np.isfinite(time_arr) & np.isfinite(flux_arr)
    time_arr, flux_arr = time_arr[mask], flux_arr[mask]
    if len(time_arr) < 50:
        return ("unknown", f"too few SPOC points: {len(time_arr)}")

    med = np.median(flux_arr)
    if med != 0:
        flux_arr = flux_arr / med

    phase = ((time_arr - t0) % period) / period
    phase[phase > 0.5] -= 1.0
    half_win = 0.5 * dur / period
    in_tr = np.abs(phase) < half_win
    base_lo, base_hi = 0.75 * dur / period, 3.0 * dur / period
    baseline_mask = (np.abs(phase) > base_lo) & (np.abs(phase) < base_hi)

    n_in = int(in_tr.sum())
    n_base = int(baseline_mask.sum())

    if n_base < 5:
        return ("unknown", f"insufficient baseline points: {n_base}")

    baseline_std = float(np.std(flux_arr[baseline_mask]))
    err_ppm = (baseline_std / max(1, np.sqrt(n_in))) * 1e6

    if n_in == 0:
        return ("unknown", "no in-transit points")

    depth_measured_ppm = float(
        (np.median(flux_arr[baseline_mask]) - np.median(flux_arr[in_tr])) * 1e6
    )

    epoch_nums = np.round((time_arr[in_tr] - t0) / period).astype(int)
    from collections import Counter
    epoch_counts = Counter(epoch_nums)
    n_epochs = sum(1 for cnt in epoch_counts.values() if cnt >= 3)

    detail = (
        f"n_in={n_in}, n_epochs={n_epochs}, "
        f"depth_meas={depth_measured_ppm:.0f}ppm, err={err_ppm:.0f}ppm, "
        f"depth_exp={depth_ppm:.0f}ppm, thresh={sigma_thresh}"
    )

    absent_depth_thresh = max(sigma_thresh * err_ppm, 0.25 * abs(depth_ppm))

    if (n_epochs >= 2 and n_in >= 30
            and depth_measured_ppm < absent_depth_thresh):
        return ("absent", detail)

    if (n_epochs >= 2
            and depth_measured_ppm >= 3.0 * err_ppm
            and depth_measured_ppm >= 0.25 * abs(depth_ppm)):
        return ("confirmed", detail)

    return ("unknown", detail)


# ===========================================================================
# NEW-engine call helpers, mirroring the EXACT parameters the real (post-
# refactor) call sites pass into fold_and_measure_depth.
# ===========================================================================

def _new_measure_depth_via_engine(centre_phase, time_arr, flux_arr, t0, P2, win):
    res = fold_and_measure_depth(
        time_arr, flux_arr, P2, t0,
        centre_phase=centre_phase, half_width=win,
        phase_mode="raw", window_mode="circular",
        oot_mode="bands", oot_bands=((0.2, 0.3), (0.7, 0.8)),
        depth_formula="diff", error_mode="mad_in",
        min_in=5, min_oot=10, guard_before_compute=True,
    )
    return res.depth_ppm, res.err_ppm


def _new_spoc_fold_via_engine(time_arr, flux_arr, period, t0, dur):
    return fold_and_measure_depth(
        time_arr, flux_arr, period, t0,
        centre_phase=0.0, half_width=0.5 * dur / period,
        phase_mode="recentered", window_mode="direct",
        oot_mode="annulus", oot_annulus=(0.75 * dur / period, 3.0 * dur / period),
        depth_formula="diff", error_mode="std_oot",
        guard_before_compute=False,
    )


# ===========================================================================
# Seeded synthetic light curve generators
# ===========================================================================

def _make_vshape_lc(period=5.0, t0=2.5, dur=0.12, n_pts=2000, baseline_days=40.0,
                    depth_ppm=3000.0, noise_ppm=200.0, seed=1):
    rng = np.random.default_rng(seed)
    time = np.sort(rng.uniform(t0 - baseline_days / 2, t0 + baseline_days / 2, n_pts))
    flux = 1.0 + rng.normal(0, noise_ppm / 1e6, n_pts)
    phase = ((time - t0) % period) / period
    phase[phase > 0.5] -= 1.0
    hw = dur / period
    frac = np.clip(1.0 - np.abs(phase) / hw, 0.0, 1.0)  # triangular (V-shape)
    flux -= (depth_ppm / 1e6) * frac
    return time, flux


def _make_box_lc(period=5.0, t0=2.5, dur=0.12, n_pts=2000, baseline_days=40.0,
                 depth_ppm=3000.0, noise_ppm=200.0, seed=2):
    rng = np.random.default_rng(seed)
    time = np.sort(rng.uniform(t0 - baseline_days / 2, t0 + baseline_days / 2, n_pts))
    flux = 1.0 + rng.normal(0, noise_ppm / 1e6, n_pts)
    phase = ((time - t0) % period) / period
    phase[phase > 0.5] -= 1.0
    hw = 0.5 * dur / period
    in_tr = np.abs(phase) < hw
    flux[in_tr] -= depth_ppm / 1e6
    return time, flux


def _make_secondary_at_half_lc(period=5.0, t0=2.5, dur=0.12, n_pts=2000,
                               baseline_days=40.0, depth_primary_ppm=3000.0,
                               depth_secondary_ppm=1200.0, noise_ppm=100.0, seed=3):
    rng = np.random.default_rng(seed)
    time = np.sort(rng.uniform(t0 - baseline_days / 2, t0 + baseline_days / 2, n_pts))
    flux = 1.0 + rng.normal(0, noise_ppm / 1e6, n_pts)
    phase = ((time - t0) % period) / period  # [0, 1)
    win = max(dur / period, 0.01)
    primary = (np.abs(phase - 0.0) < win) | (np.abs(phase - 1.0) < win)
    secondary = np.abs(phase - 0.5) < win
    flux[primary] -= depth_primary_ppm / 1e6
    flux[secondary] -= depth_secondary_ppm / 1e6
    return time, flux


def _make_gapped_lc(period=5.0, t0=2.5, dur=0.12, seed=4, depth_ppm=2500.0,
                    noise_ppm=150.0):
    rng = np.random.default_rng(seed)
    seg1 = rng.uniform(t0, t0 + 10.0, 900)
    seg2 = rng.uniform(t0 + 15.0, t0 + 30.0, 900)  # 5-day downlink-style gap
    time = np.sort(np.concatenate([seg1, seg2]))
    flux = 1.0 + rng.normal(0, noise_ppm / 1e6, len(time))
    phase = ((time - t0) % period) / period
    phase[phase > 0.5] -= 1.0
    hw = 0.5 * dur / period
    in_tr = np.abs(phase) < hw
    flux[in_tr] -= depth_ppm / 1e6
    return time, flux


def _make_few_points_lc(t0=2.5, seed=5, n_pts=8):
    """Too few points for any of the 4 sites' in-window/baseline count
    guards -- exercises the "insufficient sample" (None / "unknown") path
    identically in old and new code."""
    rng = np.random.default_rng(seed)
    time = np.sort(rng.uniform(t0, t0 + 3.0, n_pts))
    flux = 1.0 + rng.normal(0, 200e-6, n_pts)
    return time, flux


_REAL_FIXTURE = ROOT / "tests" / "fixtures" / "exominer_native" / "tic_143143769_s0032.lc.fits"


def _load_real_lc():
    """One real SPOC 2-min light curve (TIC 143143769, sector 32), quality-
    filtered and median-normalised -- not a real transit for this TIC/period,
    just real cadence/gap/noise structure to fold real data through both old
    and new code."""
    with fits.open(_REAL_FIXTURE) as hdul:
        data = hdul[1].data
        t = np.array(data["TIME"], dtype=float)
        f = np.array(data["PDCSAP_FLUX"], dtype=float)
        q = np.array(data["QUALITY"], dtype=int)
    mask = np.isfinite(t) & np.isfinite(f) & (q == 0)
    t, f = t[mask], f[mask]
    order = np.argsort(t)
    t, f = t[order], f[order]
    f = f / np.median(f)
    return t, f


def _build_scenarios():
    scenarios = {}
    scenarios["vshape"] = (*_make_vshape_lc(), 5.0, 2.5, 0.12)
    scenarios["box"] = (*_make_box_lc(), 5.0, 2.5, 0.12)
    scenarios["secondary_half"] = (*_make_secondary_at_half_lc(), 5.0, 2.5, 0.12)
    scenarios["gapped"] = (*_make_gapped_lc(), 5.0, 2.5, 0.12)
    scenarios["few_points"] = (*_make_few_points_lc(), 5.0, 2.5, 0.12)
    t, f = _load_real_lc()
    scenarios["real"] = (t, f, 3.0, float(t[50]), 0.1)
    return scenarios


# name -> (time, flux, period, t0, dur)
SCENARIOS = _build_scenarios()
SCENARIO_NAMES = list(SCENARIOS.keys())


# ===========================================================================
# Site 1: secondary_depth
# ===========================================================================

class TestSecondaryDepthGolden:
    @pytest.mark.parametrize("name", SCENARIO_NAMES)
    def test_matches_old(self, name):
        time, flux, period, t0, dur = SCENARIOS[name]
        old = _old_secondary_depth(time.copy(), flux.copy(), period, t0, dur)
        new = secondary_depth(time.copy(), flux.copy(), period, t0, dur)
        if old is None:
            assert new is None, f"[{name}] old=None but new={new}"
        else:
            assert new is not None, f"[{name}] old={old} but new=None"
            assert new == old, f"[{name}] old={old!r} != new={new!r}"


# ===========================================================================
# Site 2: double_period_eb_check's _measure_depth closure
# ===========================================================================

class TestMeasureDepthGolden:
    """Leaf-level: the engine call with site 2's exact parameters vs the
    verbatim old closure, at both centre_phase values it is called with."""

    @pytest.mark.parametrize("name", SCENARIO_NAMES)
    @pytest.mark.parametrize("centre_phase", [0.0, 0.5])
    def test_matches_old(self, name, centre_phase):
        time, flux, period, t0, dur = SCENARIOS[name]
        P2 = 2.0 * period
        win = max(dur / P2, 0.005)
        phase2 = ((time - t0) % P2) / P2

        old_depth, old_err = _old_measure_depth(centre_phase, phase2, flux, win)
        new_depth, new_err = _new_measure_depth_via_engine(
            centre_phase, time, flux, t0, P2, win)

        if old_depth is None:
            assert new_depth is None and new_err is None, (
                f"[{name}, cp={centre_phase}] old=(None,None) new=({new_depth},{new_err})")
        else:
            assert new_depth == old_depth, f"[{name}, cp={centre_phase}] depth mismatch"
            assert new_err == old_err, f"[{name}, cp={centre_phase}] err mismatch"


def _make_lk_mock(time_val, flux_val):
    """Minimal lightkurve mock: search + download_all().stitch() -> (time,flux)."""
    class _FakeLc:
        def __init__(self, t, f):
            self.time = MagicMock()
            self.time.value = t
            self.flux = MagicMock()
            self.flux.value = f

        def remove_nans(self):
            return self

        def remove_outliers(self, sigma=8):
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
    fake_col = _FakeCol(_FakeLc(time_val, flux_val))
    sr_mock = MagicMock()
    sr_mock.__len__ = MagicMock(return_value=1)
    sr_mock.download_all.return_value = fake_col
    lk_mock.search_lightcurve.return_value = sr_mock
    return lk_mock


class TestDoublePeriodEbCheckEndToEnd:
    """Whole-function comparison: OLD verbatim double_period_eb_check vs the
    real, post-refactor function, both fed the SAME light curve via an
    identically-mocked lightkurve. Proves the refactor didn't just get the
    inner fold arithmetic right but the surrounding guard/verdict logic too.
    """

    @pytest.mark.parametrize("name", SCENARIO_NAMES)
    def test_matches_old(self, name):
        time, flux, period, t0, dur = SCENARIOS[name]
        lk_mock = _make_lk_mock(time, flux)

        with patch.dict("sys.modules", {"lightkurve": lk_mock}):
            old_result = _old_double_period_eb_check("999", period, t0, dur)
        with patch.dict("sys.modules", {"lightkurve": lk_mock}):
            new_result = double_period_eb_check("999", period, t0, dur)

        assert new_result == old_result, f"[{name}] old={old_result!r} != new={new_result!r}"


# ===========================================================================
# Site 3: spoc_repeat_check's inline fold block
# ===========================================================================

class TestSpocFoldBlockGolden:
    """Leaf-level: engine call with site 3's exact parameters vs the verbatim
    old fold block, comparing every field the real call site reads."""

    @pytest.mark.parametrize("name", SCENARIO_NAMES)
    def test_matches_old(self, name):
        time, flux, period, t0, dur = SCENARIOS[name]
        old = _old_spoc_repeat_fold_block(time.copy(), flux.copy(), period, t0, dur)
        new = _new_spoc_fold_via_engine(time.copy(), flux.copy(), period, t0, dur)

        assert np.array_equal(new.in_mask, old["in_tr"]), f"[{name}] in_mask mismatch"
        assert np.array_equal(new.oot_mask, old["baseline_mask"]), f"[{name}] oot_mask mismatch"
        assert new.n_in == old["n_in"], f"[{name}] n_in mismatch"
        assert new.n_oot == old["n_base"], f"[{name}] n_oot/n_base mismatch"

        if old["guard"] == "n_base":
            # Original never computed err/depth past this guard; the engine
            # always computes them (guard_before_compute=False) but the real
            # call site never reads them in this branch either -- see
            # fold_and_measure_depth's docstring.
            pass
        else:
            assert new.err_ppm == old["err_ppm"], f"[{name}] err_ppm mismatch"
            if old["guard"] is None:
                assert new.depth_ppm == old["depth_measured_ppm"], (
                    f"[{name}] depth_ppm mismatch")


class TestSpocRepeatCheckEndToEnd:
    """Whole-function comparison via _lc_cache (no MAST mocking needed)."""

    @pytest.mark.parametrize("name", SCENARIO_NAMES)
    def test_matches_old(self, name):
        time, flux, period, t0, dur = SCENARIOS[name]
        tic = "999"
        cache = {tic: (time.copy(), flux.copy())}
        depth_ppm = 3000.0

        old_result = _old_spoc_repeat_check(tic, period, t0, dur, depth_ppm, _lc_cache=cache)
        new_result = spoc_repeat_check(tic, period, t0, dur, depth_ppm, _lc_cache=cache)

        assert new_result == old_result, f"[{name}] old={old_result!r} != new={new_result!r}"


# ===========================================================================
# Site 4: exonit/ffi_finalist_checks.py::_fold_and_measure
# ===========================================================================

class TestFoldAndMeasureGolden:
    @pytest.mark.parametrize("name", SCENARIO_NAMES)
    @pytest.mark.parametrize("phase_centre", [0.0, 0.15, 0.25, -0.15, -0.35])
    def test_matches_old(self, name, phase_centre):
        time, flux, period, t0, dur = SCENARIOS[name]
        half_width = (dur / period) * 0.6

        old = _old_fold_and_measure(time.copy(), flux.copy(), period, t0,
                                    phase_centre, half_width)
        new = _fold_and_measure(time.copy(), flux.copy(), period, t0,
                                phase_centre, half_width)

        if old is None:
            assert new is None, f"[{name}, cp={phase_centre}] old=None but new={new}"
        else:
            assert new is not None, f"[{name}, cp={phase_centre}] old={old} but new=None"
            assert new == old, f"[{name}, cp={phase_centre}] old={old!r} != new={new!r}"


# ===========================================================================
# fold_and_measure_depth: minimal contract checks (invalid mode -> ValueError)
# ===========================================================================

class TestFoldAndMeasureDepthContract:
    def test_unknown_phase_mode_raises(self):
        time, flux, period, t0, _dur = SCENARIOS["box"]
        with pytest.raises(ValueError):
            fold_and_measure_depth(time, flux, period, t0, half_width=0.02,
                                   phase_mode="bogus", oot_mode="annulus",
                                   oot_annulus=(0.1, 0.2))

    def test_unknown_oot_mode_raises(self):
        time, flux, period, t0, _dur = SCENARIOS["box"]
        with pytest.raises(ValueError):
            fold_and_measure_depth(time, flux, period, t0, half_width=0.02,
                                   oot_mode="bogus")
