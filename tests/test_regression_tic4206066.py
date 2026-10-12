"""Fixed recall regression case: TIC 4206066, sector 6 (P4).

Replicates the worker's load -> detrend -> BLS multi-planet chain OFFLINE on a
committed TGLC FITS fixture, without refactoring worker.py.  Line refs into
worker/worker.py (may drift; re-check when the worker changes):
  - ``_load_star`` ~2293-2380: remove_nans/outliers, fill_gaps<5d, wotan, quick
    bls_fast pre-check + masked re-detrend if snr>=7, median renorm, sort.
  - BLS loop ~2560-2760 (``for i in range(max_planets)`` at ~2607): run_bls +
    resolve_harmonic, snr gate (~2701), accumulated mask, re-detrend, median fill.
Remaining deviations from the worker:
  - physical-T14 gate (reject -> mask -> continue, ~2705-2730) omitted: needs
    stellar R/M from server metadata.
  - wotan-failure counter / abort-after-2 (~2575-2600) omitted (failure is
    just skipped).
  - harmonic_long_period_* and snr_threshold / n_periods / period range use the
    worker's DEFAULTS (config.get defaults), not a live config.
  - medium-period protected slot and long-period sweep (after the loop) omitted.
  - single FITS: no multi-file stitch, data_source="ffi" fixed (the worker gets
    it from ``_detect_data_source``).
Determinism: np.random is seeded before fill_gaps (lightkurve adds random
noise to filled gaps; the worker does not seed it) and NUMBA_NUM_THREADS=1 is pinned so the regression test is
reproducible (production uses parallel numba prange in bls_fast, which makes
the quick pre-check / masked re-detrend, and hence borderline low-SNR
candidates, host- and run-dependent).

Expected values (ephemeris, T14, tolerances) come from
data/samples/regression_cases.csv.

What each test tracks:
  - test_prepare_and_search_deterministic: bitwise reproducibility of the chain.
  - test_tic4206066_s6_recovered: the STRONGEST (max SNR) period-matched BLS
    candidate has epoch offset <= epoch_tol_min (90 min; raw BLS box is biased).
  - test_tic4206066_s6_refined_epoch: after P2 refine_duration the strongest
    candidate meets epoch_tol_refined_min (45 min) and duration within 40 % of
    t14_hours; merge_same_period_candidates leaves one candidate at that period.
  - test_tic4206066_s6_bls_epoch_target_45min (strict xfail): raw BLS meets
    45 min; XPASS (failure) signals BLS itself improved -> drop the xfail.
"""
import os

# Must precede any numba/bls import (see Determinism note above).
os.environ.setdefault("NUMBA_NUM_THREADS", "1")
os.environ.setdefault("EXONIT_BLS_BACKEND", "numba")

import csv
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
CASES_CSV = REPO / "data" / "samples" / "regression_cases.csv"
PERIOD_TOL_D = 0.02   # absolute at the fundamental; relative (0.63 %) for harmonics
MAX_ITER = 5          # worker default max_planets


def _load_case(tic=4206066):
    with open(CASES_CSV, newline="") as fh:
        for row in csv.DictReader(fh):
            if int(row["tic_id"]) == tic:
                return row
    raise AssertionError(f"TIC {tic} not in {CASES_CSV}")


def _prepare(path):
    from worker.worker import _read_lightcurve
    from worker.wotan_detrend import detrend_with_wotan
    from bls_fast import bls_fast

    lc = _read_lightcurve(str(path))
    lc = lc.remove_nans().remove_outliers(sigma_upper=5, sigma_lower=float("inf"))
    tv = lc.time.value
    if len(tv) > 1 and np.max(np.diff(tv)) < 5:
        # lightkurve fill_gaps injects Gaussian noise from the global numpy RNG
        # (production is therefore run-to-run nondeterministic on gappy LCs);
        # seed it so this regression test is reproducible.
        np.random.seed(20261011)
        lc = lc.fill_gaps()
    t_raw = np.array(lc.time.value, dtype=np.float64)
    f_raw = np.array(lc.flux.value, dtype=np.float64)
    raw = f_raw.copy()
    fin = np.isfinite(t_raw) & np.isfinite(f_raw)
    flat = np.full_like(f_raw, np.nan)
    flat[fin] = detrend_with_wotan(t_raw[fin], f_raw[fin])
    try:
        m = np.isfinite(t_raw) & np.isfinite(flat)
        q = bls_fast(t_raw[m], flat[m], period_min=0.5, period_max=50.0,
                     n_periods=2000, n_bins=100, n_dur_steps=5)
        if q["snr"] >= 7.0:
            ph = ((t_raw - q["t0"]) % q["period"]) / q["period"]
            hd = q["duration"] / q["period"]
            tm = (ph < hd) | (ph > 1.0 - hd)
            flat[fin] = detrend_with_wotan(t_raw[fin], f_raw[fin],
                                           transit_mask=tm[fin])
    except Exception:
        pass
    med = np.nanmedian(flat)
    if med > 1.5 or med < 0.5:
        flat = flat / med
    idx = np.argsort(t_raw)
    t, flat, raw = t_raw[idx], flat[idx], raw[idx]
    ok = np.isfinite(t) & np.isfinite(flat) & np.isfinite(raw)
    return t[ok], flat[ok], raw[ok]


def _search(time_arr, flux_arr, raw_flux_arr):
    from worker.worker import resolve_harmonic
    from worker.wotan_detrend import detrend_with_wotan
    from bls import run_bls

    cands = []
    current = flux_arr.copy()
    acc = np.zeros(len(time_arr), dtype=bool)
    for i in range(MAX_ITER):
        if i >= 1 and acc.any() and acc.sum() / len(time_arr) <= 0.30:
            try:
                re = detrend_with_wotan(time_arr, raw_flux_arr, transit_mask=acc)
                fill = ~np.isfinite(re) | acc
                good = np.isfinite(re) & ~acc
                fv = np.nanmedian(re[good]) if good.any() else 1.0
                re = re.copy()
                re[fill] = fv
                current = re
            except Exception:
                pass
        res = run_bls(time_arr, current, period_min=0.5, period_max=50.0,
                      n_periods=50000)
        res = resolve_harmonic(time_arr, current, res, period_min=0.5,
                               period_max=50.0, bls_fn=run_bls,
                               data_source="ffi")
        if res["snr"] < 7.0:
            break
        cands.append({
            "n": i + 1,
            "period_days": round(float(res["period"]), 4),
            "t0": round(float(res["t0"]), 4),
            "duration_h": round(float(res["duration"]), 4) * 24.0,
            "depth_ppm": round(float(res["depth_ppm"]), 1),
            "snr": round(float(res["snr"]), 1),
        })
        ph = ((time_arr - res["t0"]) % res["period"]) / res["period"]
        hd = (res["duration"] / res["period"]) / 2
        it = (ph < hd) | (ph > 1.0 - hd)
        acc |= it
        current = current.copy()
        current[it] = np.nanmedian(current)
    return cands


def _period_ok(p, p_pub):
    rel = PERIOD_TOL_D / p_pub
    return any(abs(p - f * p_pub) <= rel * f * p_pub for f in (1.0, 2.0, 0.5))


def _epoch_offset_d(c, p_pub, t0_pub, tmin, tmax):
    """Min over the candidate's in-span transits of the distance to the
    nearest published transit (published ephemeris back-propagated from the
    CSV t0_btjd with period_days)."""
    p, t0 = c["period_days"], c["t0"]
    k = np.arange(np.floor((tmin - t0) / p), np.ceil((tmax - t0) / p) + 1)
    tt = t0 + k * p
    tt = tt[(tt >= tmin) & (tt <= tmax)]
    if len(tt) == 0:
        return float("inf")
    n = np.round((tt - t0_pub) / p_pub)
    return float(np.min(np.abs(tt - (t0_pub + n * p_pub))))


def _run(case):
    path = REPO / case["fixture"]
    assert path.exists(), f"regression fixture missing: {path}"
    t, flat, raw = _prepare(path)
    return t, flat, raw, _search(t, flat, raw)


@pytest.mark.slow
def test_prepare_and_search_deterministic():
    case = _load_case()
    t1, f1, r1, c1 = _run(case)
    t2, f2, r2, c2 = _run(case)
    assert np.array_equal(t1, t2)
    assert np.array_equal(f1, f2, equal_nan=True)
    assert np.array_equal(r1, r2, equal_nan=True)
    assert c1 == c2, f"candidate lists differ:\n{c1}\n{c2}"


@pytest.mark.slow
def test_tic4206066_s6_recovered():
    case = _load_case()
    p_pub = float(case["period_days"])
    t0_pub = float(case["t0_btjd"])
    tol_d = float(case["epoch_tol_min"]) / 1440.0
    t14_h = float(case["t14_hours"])

    t, flat, raw, cands = _run(case)
    tmin, tmax = float(t.min()), float(t.max())
    lines = [f"  P={c['period_days']:.4f} t0={c['t0']:.4f} dur={c['duration_h']:.2f}h "
             f"depth={c['depth_ppm']:.0f}ppm snr={c['snr']:.1f}" for c in cands]
    detail = "\n".join(lines) or "  (no candidates)"
    print(f"\nTIC {case['tic_id']} s{case['sector']}: {len(cands)} candidates\n{detail}")

    per_ok = [c for c in cands if _period_ok(c["period_days"], p_pub)]
    for c in per_ok:
        c["epoch_off_min"] = _epoch_offset_d(c, p_pub, t0_pub, tmin, tmax) * 1440
        print(f"  P={c['period_days']:.4f}: min epoch offset {c['epoch_off_min']:.1f} min")
    assert per_ok, f"no candidate matches P~{p_pub} (or 2x/0.5x); candidates:\n{detail}"
    best = max(per_ok, key=lambda c: c["snr"])
    print(f"INFO strongest period-matched candidate duration {best['duration_h']:.2f} h "
          f"vs published T14 {t14_h:.2f} h")
    assert best["epoch_off_min"] <= float(case["epoch_tol_min"]), (
        f"strongest period-matched candidate (snr={best['snr']}) epoch offset "
        f"{best['epoch_off_min']:.1f} min > {case['epoch_tol_min']} min "
        f"({case['notes']}); candidates:\n{detail}")


def _strongest_matched(case):
    p_pub = float(case["period_days"])
    t0_pub = float(case["t0_btjd"])
    t, flat, raw, cands = _run(case)
    per_ok = [c for c in cands if _period_ok(c["period_days"], p_pub)]
    assert per_ok, f"no candidate matches P~{p_pub}: {cands}"
    best = max(per_ok, key=lambda c: c["snr"])
    off = _epoch_offset_d(best, p_pub, t0_pub, float(t.min()), float(t.max())) * 1440
    return t, flat, cands, best, off


@pytest.mark.slow
def test_tic4206066_s6_refined_epoch():
    """P2 duration_refine on the strongest BLS candidate reaches the 45-min
    epoch target and a physical duration; same-period slots merge to one.

    Duration: TGLC FFI is 30-min cadence, which smears the published 1.87 h
    transit to ~2.4 h, hence the generous 40 % tolerance on t14_hours.
    Flux used is the flattened LC of the search (iteration 1 for the
    strongest candidate), as the worker does.
    """
    from worker.duration_refine import refine_duration, merge_same_period_candidates

    case = _load_case()
    p_pub = float(case["period_days"])
    t0_pub = float(case["t0_btjd"])
    t14_h = float(case["t14_hours"])
    t, flat, cands, best, off = _strongest_matched(case)
    assert best["n"] == 1, f"strongest candidate is not the first BLS slot: {cands}"
    ref = refine_duration(t, flat, best["period_days"], best["t0"],
                          best["duration_h"] / 24.0)
    rc = {"period_days": best["period_days"], "t0": ref["t0"]}
    roff = _epoch_offset_d(rc, p_pub, t0_pub, float(t.min()), float(t.max())) * 1440
    dur_h = ref["duration"] * 24.0
    print(f"\nREFINED: t0={ref['t0']:.4f} offset {roff:.1f} min (BLS {off:.1f}), "
          f"dur {dur_h:.2f} h, depth {ref['depth_ppm']} ppm")
    assert roff <= float(case["epoch_tol_refined_min"]), roff
    assert abs(dur_h - t14_h) <= 0.40 * t14_h, (dur_h, t14_h)

    bls_list = [dict(c, candidate_num=c["n"]) for c in cands]
    merged = merge_same_period_candidates(bls_list)
    same = [c for c in merged if abs(c["period_days"] - best["period_days"])
            / best["period_days"] <= 2e-3]
    assert len(same) == 1, merged


@pytest.mark.slow
@pytest.mark.xfail(strict=True, reason="raw BLS epoch offset of the strongest "
                   "candidate is 67.7 min (> 45 min target); passes only if BLS "
                   "itself improves (refinement is covered by "
                   "test_tic4206066_s6_refined_epoch)")
def test_tic4206066_s6_bls_epoch_target_45min():
    case = _load_case()
    _, _, _, best, off = _strongest_matched(case)
    print(f"\nUNREFINED strongest candidate epoch offset {off:.1f} min")
    assert off <= float(case["epoch_tol_refined_min"]), off
