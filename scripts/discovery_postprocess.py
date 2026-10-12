#!/usr/bin/env python3
"""Post-process a finished discovery search: veto + vetting + FPP + dossier.

Turns the manual vetting steps of Discovery Run v1 into one reproducible command,
so a discovery search needs no manual post-processing. Run after the primary +
backfills have completed.

Steps:
  1. Systematic-period veto (exonit.systematic_veto): reclassify instrumental
     false positives (a period in many unrelated stars with clustered epoch).
  2. For candidates with v6 (exominer_ffi_score) above --v6-min and verdict
     planet_candidate: layer-3 vetting (odd/even, V-shape, secondary) from the
     cached light curve.
  3. TRICERATOPS FPP (layer 4) on the layer-3 survivors (skip with --skip-fpp).
  4. Write a vetting dossier markdown.

Usage:
  python3 scripts/discovery_postprocess.py --sample data/samples/discovery_v1_seed42.csv \
      --label candidate --v6-min 0.5 --out docs/discovery-v1-vetting-dossier.md
"""
import argparse
import csv
import json
import os
import re
import sqlite3
import sys
import warnings
from itertools import chain
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
# Worker modules live in worker/ and are not installed as a package.
sys.path.insert(0, str(ROOT / "worker"))

import numpy as np  # noqa: E402
from astropy.io import fits  # noqa: E402

from exonit.config import ExoNitConfig  # noqa: E402
from exonit.db_backend import op_session, select_in_chunks, select_in_chunks_iter  # noqa: E402
from exonit.ffi_finalist_checks import (  # noqa: E402
    folded_depth_control_test, stellar_consistency_check,
    angular_separation_deg,
    shared_signal_groups,
)
from exonit.metadata_cache import MetadataCache  # noqa: E402
from exonit.systematic_veto import (  # noqa: E402
    find_systematic_periods, find_systematic_periods_columnar, ColumnarPopulation,
    is_vetoed, apply_depth_gate, find_deep_eclipse_hosts,
    load_toi_protection, is_toi_protected, filter_anti_toi_protected,
    partition_deep_veto_targets, MAX_PLANET_RADIUS_RJUP,
)
from exonit.systematic_families_io import load_families  # noqa: E402
from exonit.progress_client import ProgressReporter  # noqa: E402
from exonit.transit_math import fold_and_measure_depth  # noqa: E402


def load_lc(mc, tic):
    e = mc.get_entry(tic)
    if not e:
        return None, None, None
    T, F, secs = [], [], set()
    for fn in e["fits_files"]:
        p = mc.get_fits_path(fn)
        if not p:
            continue
        m = re.search(r"_s(\d{4})-", fn) or re.search(r"-s(\d{4})-", fn)
        try:
            with fits.open(p) as h:
                cl = {c.name.lower(): c.name for c in h[1].columns}
                d = h[1].data
                # flux column preference: QLP, SPOC, then TGLC (cal_aper_flux)
                fc = next((cl[k] for k in ("kspsap_flux", "sap_flux", "cal_aper_flux")
                           if k in cl), None)
                if not fc or "time" not in cl:
                    continue
                t = np.array(d[cl["time"]], float)
                f = np.array(d[fc], float)
                if "quality" in cl:
                    q = np.array(d[cl["quality"]], int)
                elif "tess_flags" in cl:  # TGLC quality columns
                    q = np.array(d[cl["tess_flags"]], int)
                    if "tglc_flags" in cl:
                        q = q | np.array(d[cl["tglc_flags"]], int)
                else:
                    q = np.zeros(len(t), int)
                msk = (q == 0) & np.isfinite(t) & np.isfinite(f) & (f > 0)
                if msk.sum() > 50:
                    T.append(t[msk]); F.append(f[msk] / np.median(f[msk]))
                    if m:
                        secs.add(int(m.group(1)))
        except Exception:
            pass
    if not T:
        return None, None, None
    return np.concatenate(T), np.concatenate(F), np.array(sorted(secs))


def _fetch_spoc_lc(tic_id, process, timeout=120):
    """Thread-bounded MAST fetch of all SPOC 2-min sectors for *tic_id*.

    Postprocess dedup (2026-07-02 cleanup review, item 4): the "search_lightcurve
    + download_all + apply a post-download chain + wall-clock join(timeout)"
    block was hand-copied between double_period_eb_check and spoc_repeat_check —
    only the post-download processing chain differed (``.flatten()`` vs
    ``.remove_outliers()``). *process* runs INSIDE the same background thread
    as the download (NOT after joining) so a slow/erroring processing step is
    caught by the SAME try/except and the SAME timeout as a slow/erroring
    download — matching the pre-extraction behaviour of both call sites
    exactly (an exception during ``.stitch()...`` used to be swallowed into
    "fetch error: ...", not raised to the caller).

    We run the download in a thread (lightkurve uses requests which cannot be
    interrupted with signal on non-main threads) and join with a generous
    timeout so slow MAST does not block the pipeline.

    Parameters
    ----------
    tic_id  : str
    process : callable(LightCurveCollection) -> LightCurve
        Caller-specific post-download chain, e.g.
        ``lambda col: col.stitch().remove_nans().normalize().flatten(window_length=401)``.
    timeout : float — wall-clock ceiling in seconds (default 120, matching
        both pre-extraction call sites).

    Returns
    -------
    (lc, None) on success, or (None, (verdict, detail)) on any failure — the
    caller can just ``return err`` directly.
    """
    try:
        import lightkurve as lk  # lazy import — not available on all machines
    except ImportError:
        return None, ("unknown", "lightkurve not installed")

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
            _lc_box.append(process(col))
        except Exception as exc:  # noqa: BLE001
            _exc_box.append(exc)

    t = _threading.Thread(target=_fetch, daemon=True)
    t.start()
    t.join(timeout=timeout)

    if t.is_alive():
        # Thread still running — MAST is too slow; return unknown
        return None, ("unknown", f"MAST timeout after {timeout:.0f} s")
    if _exc_box:
        return None, ("unknown", f"fetch error: {type(_exc_box[0]).__name__}: {_exc_box[0]}")
    if not _lc_box:
        return None, ("unknown", "no SPOC 2-min data found")
    return _lc_box[0], None


def double_period_eb_check(tic_id, period, t0, dur, sigma_thresh=3.0):
    """Check whether a finalist candidate is a double-period eclipsing binary.

    Downloads the SPOC 2-min light curve for *tic_id* from any available sector
    via lightkurve, folds at 2×period, and compares eclipse depths at phase 0.0
    (primary) and phase 0.5 (secondary).  If the two depths are significantly
    different (|depthA - depthB| >= sigma_thresh × hypot(errA, errB)) the system
    is a double-period EB: the BLS detected the period at half the true value
    because the two eclipses have similar depths.

    Parameters
    ----------
    tic_id : str
    period : float  — BLS period in days (half the true EB period)
    t0     : float  — BLS epoch (BTJD)
    dur    : float  — transit duration in days
    sigma_thresh : float  — significance threshold (default 3.0)

    Returns
    -------
    (verdict, detail) where verdict ∈ {"eb", "ok", "unknown"}
      "eb"      — double-period EB confirmed; exclude from finalist list
      "ok"      — depths are consistent; not an obvious double-period EB
      "unknown" — SPOC 2-min data unavailable or fetch failed; do not block
    """
    try:
        lc, err = _fetch_spoc_lc(
            tic_id,
            process=lambda col: col.stitch().remove_nans().normalize().flatten(window_length=401),
        )
        if err is not None:
            return err

        time_arr = np.array(lc.time.value, dtype=float)
        flux_arr = np.array(lc.flux.value, dtype=float)
        mask = np.isfinite(time_arr) & np.isfinite(flux_arr)
        time_arr, flux_arr = time_arr[mask], flux_arr[mask]
        if len(time_arr) < 100:
            return ("unknown", f"too few SPOC points: {len(time_arr)}")

        # ---------------------------------------------------------------
        # Optional BLS period refinement around the BLS value (±0.5 %).
        # Skipped if the LC is very short; the original period is used.
        # ---------------------------------------------------------------
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
                pass  # keep original period if BLS refinement fails

        # ---------------------------------------------------------------
        # Fold at 2×P and measure depth at phase 0.0 and phase 0.5.
        # Window half-width: max(dur/(2×P), 0.005) in phase units.
        # Baseline (OOT): phases 0.2–0.3 and 0.7–0.8 in the 2×P fold.
        # ---------------------------------------------------------------
        P2 = 2.0 * period_refined
        win = max(dur / P2, 0.005)  # phase fraction

        # D6 (docs/reviews/2026-07-02-opus48-cleanup-review.md): depth/err at
        # centre_phase 0.0 (primary) and 0.5 (secondary) via the shared
        # exonit.transit_math engine — circular-wrap in-window test (works at
        # the phase-domain-edge centre 0.0 without an edge artefact), fixed
        # literal OOT bands (0.2,0.3)/(0.7,0.8), MAD-based error. Bit-identical
        # to the pre-refactor _measure_depth closure — see
        # tests/test_transit_math_golden.py.
        def _measure_depth(centre_phase):
            res = fold_and_measure_depth(
                time_arr, flux_arr, P2, t0,
                centre_phase=centre_phase, half_width=win,
                phase_mode="raw", window_mode="circular",
                oot_mode="bands", oot_bands=((0.2, 0.3), (0.7, 0.8)),
                depth_formula="diff", error_mode="mad_in",
                min_in=5, min_oot=10, guard_before_compute=True,
            )
            return res.depth_ppm, res.err_ppm

        depthA, errA = _measure_depth(0.0)
        depthB, errB = _measure_depth(0.5)

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


def spoc_repeat_check(tic_id, period, t0, dur, depth_ppm, sigma_thresh=3.0,
                      _lc_cache: dict | None = None):
    """Check whether a transit signal repeats consistently in multi-sector SPOC 2-min data.

    Downloads ALL available SPOC 2-min sectors for *tic_id* via lightkurve,
    stitches them, and checks whether the folded depth is consistent with the
    expected depth at the candidate period.

    Parameters
    ----------
    tic_id       : str
    period       : float  — period in days
    t0           : float  — epoch (BTJD)
    dur          : float  — transit duration in days
    depth_ppm    : float  — expected transit depth in ppm (from BLS)
    sigma_thresh : float  — significance threshold for "absent" (default 3.0)
    _lc_cache    : optional dict {tic_id: (time_arr, flux_arr)} for tests/reuse

    Returns
    -------
    (verdict, detail)
      verdict ∈ {"absent", "confirmed", "unknown"}
        "absent"    — EXCLUDES the finalist: ≥2 distinct epochs, ≥30 in-transit
                      points, and measured depth < max(sigma_thresh×err, 0.25×expected)
        "confirmed" — depth_measured ≥ 3σ AND ≥ 0.25×expected
        "unknown"   — insufficient data, single epoch, timeout, error — do NOT block
    """
    # Reuse pre-loaded LC (e.g. from orbit_start_artifact_check) if available
    if _lc_cache is not None and tic_id in _lc_cache:
        time_arr, flux_arr = _lc_cache[tic_id]
        _from_cache = True
    else:
        _from_cache = False
        lc, err = _fetch_spoc_lc(
            tic_id,
            process=lambda col: col.stitch().remove_nans().remove_outliers(sigma=8).normalize(),
        )
        if err is not None:
            return err
        time_arr = np.array(lc.time.value, dtype=float)
        flux_arr = np.array(lc.flux.value, dtype=float)

    mask = np.isfinite(time_arr) & np.isfinite(flux_arr)
    time_arr, flux_arr = time_arr[mask], flux_arr[mask]
    if len(time_arr) < 50:
        return ("unknown", f"too few SPOC points: {len(time_arr)}")

    # Normalise by median (raw LC from cache may not be normalised)
    med = np.median(flux_arr)
    if med != 0:
        flux_arr = flux_arr / med

    # Fold at P; in-transit = |phase| < 0.5*dur/P; baseline = 0.75–3× dur away.
    # D6 (docs/reviews/2026-07-02-opus48-cleanup-review.md): the fold/mask/
    # depth/err arithmetic now runs through the shared exonit.transit_math
    # engine (guard_before_compute=False: this function's own guard order —
    # n_base checked before n_in, err computed unconditionally via its own
    # max(1, ...) floor, n_in guarded only right before the depth is read —
    # doesn't fit the simple "guard-then-compute" shape of the engine's other
    # 3 callers, so depth/err are always computed here and the guards below
    # are unchanged from before the refactor). in_tr is reused below for
    # per-epoch counting, exactly as before. Bit-identical to the
    # pre-refactor inline block — see tests/test_transit_math_golden.py.
    fold_res = fold_and_measure_depth(
        time_arr, flux_arr, period, t0,
        centre_phase=0.0, half_width=0.5 * dur / period,
        phase_mode="recentered", window_mode="direct",
        oot_mode="annulus", oot_annulus=(0.75 * dur / period, 3.0 * dur / period),
        depth_formula="diff", error_mode="std_oot",
        guard_before_compute=False,
    )
    in_tr = fold_res.in_mask
    baseline_mask = fold_res.oot_mask
    n_in = fold_res.n_in
    n_base = fold_res.n_oot

    if n_base < 5:
        return ("unknown", f"insufficient baseline points: {n_base}")

    err_ppm = fold_res.err_ppm

    if n_in == 0:
        return ("unknown", "no in-transit points")

    depth_measured_ppm = fold_res.depth_ppm

    # Count distinct epochs with ≥3 in-transit points
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

    # Only declare "confirmed" when we have ≥2 distinct epochs.  A single
    # epoch (mono-transit) is not enough to confirm repeatability even if the
    # measured depth looks right — keep as "unknown" to be conservative.
    if (n_epochs >= 2
            and depth_measured_ppm >= 3.0 * err_ppm
            and depth_measured_ppm >= 0.25 * abs(depth_ppm)):
        return ("confirmed", detail)

    return ("unknown", detail)


def orbit_start_artifact_check(tic_id, period, t0, dur, mc=None,
                                _lc_cache: dict | None = None):
    """Check whether all predicted transits land in the orbit-start ramp window.

    TESS re-acquires after each ~13.7-day downlink gap. The first ~4 hours of
    a new data segment can be affected by pointing ramps. Any transit that
    falls within [0, 4 h] after a data-gap start is suspect.

    This check only applies when 13.0 ≤ P ≤ 15.0 days (the TESS orbital
    family). Periods outside this range immediately return "not_applicable".

    IMPORTANT: the ramp veto must judge transits over the DETECTION window —
    i.e. the TGLC LC from the discovery sector — NOT over multi-sector SPOC
    history (the multi-sector repeatability is already covered by check A /
    spoc_repeat_check). Using SPOC here dilutes the check: a 2-transit
    detection spread over 5 sectors would see 198 transits, of which only 2
    fall in a ramp, masking the true artefact.

    Therefore: LC source is ONLY the local TGLC via load_lc(mc, tic_id).
    The _lc_cache parameter is accepted but intentionally ignored (it holds
    SPOC multi-sector data from spoc_repeat_check and must NOT be reused here).
    Without mc or without a valid TGLC LC → "unknown" (finalist kept;
    spoc_repeat_check is the backstop).

    Parameters
    ----------
    tic_id : str
    period : float  — period in days
    t0     : float  — epoch (BTJD)
    dur    : float  — transit duration in days (used only for logging)
    mc     : MetadataCache instance; used to load the local TGLC LC
    _lc_cache : ignored (kept for backward-compatible call signature)

    Returns
    -------
    (verdict, detail)
      verdict ∈ {"orbit_start_artifact", "ok", "unknown", "not_applicable"}
        "orbit_start_artifact" — ALL predicted transits with data fall in
                                 orbit-start ramp windows (EXCLUDES finalist)
        "ok"                   — at least one transit falls in clean data
        "unknown"              — no TGLC LC available (do not block)
        "not_applicable"       — period outside [13, 15] d range
    """
    RAMP_HOURS = 4.0
    RAMP_DAYS = RAMP_HOURS / 24.0
    GAP_THRESHOLD = 0.2  # days; dt > this → new segment

    if not (13.0 <= period <= 15.0):
        return ("not_applicable", f"P={period:.4f}d outside [13,15] d range")

    # Source: ONLY the local TGLC LC from the discovery sector.
    # Do NOT fall back to SPOC/lightkurve — that would evaluate 5+ sectors
    # and dilute the ramp check (see docstring).
    time_arr = None
    if mc is not None:
        t_lc, _, _ = load_lc(mc, tic_id)
        if t_lc is not None and len(t_lc) > 0:
            time_arr = np.sort(t_lc)

    if time_arr is None or len(time_arr) < 10:
        return ("unknown", "no TGLC LC available to check orbit-start ramp")

    time_arr = np.sort(time_arr[np.isfinite(time_arr)])

    # Build segment starts: time[0] plus any point where dt > GAP_THRESHOLD
    dt = np.diff(time_arr)
    gap_indices = np.where(dt > GAP_THRESHOLD)[0]
    # Segment starts: index 0 (absolute start) + index after each gap
    seg_starts = np.concatenate([[time_arr[0]],
                                  time_arr[gap_indices + 1]])

    t_min = time_arr[0]
    t_max = time_arr[-1]

    # Enumerate all predicted transit centres within data range
    k_min = int(np.floor((t_min - t0) / period))
    k_max = int(np.ceil((t_max - t0) / period))
    transit_times = [t0 + k * period for k in range(k_min, k_max + 1)
                     if t_min <= t0 + k * period <= t_max]

    if not transit_times:
        return ("unknown", "no predicted transits within data range")

    # For each transit, find closest segment start that is ≤ t_k
    ramp_count = 0
    data_count = 0
    ramp_details = []
    for t_k in transit_times:
        # Segment starts that are ≤ t_k
        prior_starts = seg_starts[seg_starts <= t_k]
        if len(prior_starts) == 0:
            # t_k is before any data — skip (no data coverage)
            continue
        nearest_start = prior_starts[-1]
        dt_from_start = t_k - nearest_start
        data_count += 1
        if 0.0 <= dt_from_start <= RAMP_DAYS:
            ramp_count += 1
            ramp_details.append(
                f"t_k={t_k:.3f} dt={dt_from_start * 24:.1f}h from seg start {nearest_start:.3f}"
            )

    detail = (
        f"P={period:.4f}d, {data_count} transits with data, "
        f"{ramp_count} in ramp window [0,{RAMP_HOURS}h]; "
        + ("; ".join(ramp_details) if ramp_details else "none in ramp")
    )

    if data_count == 0:
        return ("unknown", "no predicted transits with data coverage")
    if ramp_count == data_count:
        return ("orbit_start_artifact", detail)
    return ("ok", detail)


def _load_toi_catalog(catalog_path: str) -> dict:
    """Load TOI catalog into {tic_id_int: toi_str} dict.

    Postprocess dedup (2026-07-02 cleanup review, item 4): was a hand-rolled
    csv.DictReader loop with a manual per-row try/except int()/float() cast.
    Now a single vectorized pandas read (the same idiom exonit.scorecard's own
    load_toi_catalog uses: pd.read_csv + coercion, no per-row Python loop).

    NOT literally built from exonit.scorecard.load_toi_catalog(catalog_path)'s
    output: that function's normalized DataFrame deliberately drops the raw
    "TOI" label column (not needed for scorecard metrics). Reusing it here
    would either lose the "TOI-101.01" label the dossier prints (a quality
    regression per CLAUDE.md's quality bar) or require extending
    exonit/scorecard.py, which is out of this change's scope (scripts/*.py +
    tests only) and, per the project's "one change per iteration,
    scorecard-gated" convention, any edit to scorecard.py's own output would
    need its own before/after scorecard run — not bundled into this
    behaviour-neutral dedup pass.

    The catalog CSV uses 'TIC ID' and 'TOI' column names.  Returns empty
    dict with a warning if the file is missing (non-fatal: no filter applied),
    or if either required column is absent (same silent no-op the old
    per-row ``row.get(..., "")`` default produced for a malformed catalog).
    """
    import pandas as pd

    try:
        df = pd.read_csv(catalog_path, dtype=str, low_memory=False)
    except FileNotFoundError:
        import warnings as _warnings
        _warnings.warn(
            f"[anti-toi] TOI catalog not found at {catalog_path!r}; "
            "anti-TOI filter disabled for this run.",
            RuntimeWarning,
            stacklevel=2,
        )
        return {}

    if "TIC ID" not in df.columns or "TOI" not in df.columns:
        return {}

    tic_num = pd.to_numeric(df["TIC ID"], errors="coerce")
    toi_str = df["TOI"].fillna("").str.strip()
    mask = tic_num.notna() & (toi_str != "")
    return {int(t): s for t, s in zip(tic_num[mask], toi_str[mask])}


def secondary_depth(time, flux, period, t0, dur):
    """Depth at the phase-0.5 secondary-eclipse window vs a 0.25/0.75 baseline.

    D6 (docs/reviews/2026-07-02-opus48-cleanup-review.md): now a thin wrapper
    over the shared exonit.transit_math engine; behaviour is bit-identical to
    the pre-refactor inline version (see tests/test_transit_math_golden.py).
    """
    win = max(dur / period, 0.01)
    res = fold_and_measure_depth(
        time, flux, period, t0,
        centre_phase=0.5, half_width=win,
        phase_mode="raw", window_mode="direct",
        oot_mode="two_windows", oot_windows=(0.25, 0.75),
        depth_formula="diff", error_mode=None,
        min_in=5, min_oot=5, guard_before_compute=True,
    )
    return res.depth_ppm


def _select_in_chunks(con, sql_before, ids, sql_after="", chunk_size=30000):
    """Run ``{sql_before} (<placeholders>){sql_after}`` over *ids* in chunks.

    D2 (docs/reviews/2026-07-02-opus48-cleanup-review.md): this was the
    original (correctly-sized, 30k) implementation of the chunked-IN pattern;
    it has since been promoted to ``exonit.db_backend.select_in_chunks`` as
    the canonical shared version (other callers had un-chunked or
    wrong-sized copies — see that function's docstring). Kept as a thin local
    alias, rather than rewriting the 2 call sites in this file, since
    ``con`` here is an ``OperationalSession`` (from ``op_session()``) whose
    ``.execute(sql, params)`` already matches ``select_in_chunks``'s
    ``runner(sql, params) -> Iterable[Mapping]`` contract directly.
    """
    return select_in_chunks(con.execute, sql_before, ids, sql_after,
                            chunk_size=chunk_size)


def _select_in_chunks_iter(con, sql_before, ids, sql_after="", params_extra=(),
                           chunk_size=30000):
    """Streaming (per-chunk, never-concatenated) sibling of ``_select_in_chunks``.

    2026-08-01 (postprocess-streaming F3, §3.C): thin local alias over
    ``exonit.db_backend.select_in_chunks_iter``, mirroring ``_select_in_chunks``
    above — ``con`` here is an ``OperationalSession`` whose ``.execute(sql,
    params)`` already matches the ``runner`` contract. Yields each chunk's
    rows as a separate list instead of one concatenated list, so callers can
    process (and discard) one chunk at a time — see the module changelog /
    docs/superpowers/specs/2026-08-01-postprocess-streaming.md §3.C for why
    this replaces the un-chunked, fully-materialized queries in Steps 1/1b/1c/2+3.
    """
    return select_in_chunks_iter(con.execute, sql_before, ids, sql_after=sql_after,
                                 params_extra=params_extra, chunk_size=chunk_size)


#: Default per-candidate wall-clock ceiling (seconds) for the TRICERATOPS FPP
#: Monte-Carlo call in Step 3. Generous (Monte-Carlo N=100000 legitimately
#: takes minutes on some targets) but bounded: B3
#: (docs/reviews/2026-07-02-opus48-cleanup-review.md) — without it, one
#: pathological candidate could hang the whole end-of-run POSTPROCESS tail
#: forever, with the fleet already stopped (nothing left to notice/restart
#: it). Overridable via --fpp-timeout-seconds / EXONIT_POSTPROCESS_FPP_TIMEOUT.
DEFAULT_FPP_TIMEOUT_SECONDS = 900.0


def _run_fpp_with_timeout(calc_fpp_fn, tic_id, period, t0, dur, depth_ppm,
                          secs, time_arr, flux_arr, timeout_seconds):
    """Run *calc_fpp_fn* (TRICERATOPS Monte-Carlo) with a wall-clock ceiling.

    B3: same daemon-thread + join(timeout) pattern already used by
    double_period_eb_check/spoc_repeat_check's MAST fetch (via
    _fetch_spoc_lc), just with a much larger default timeout — calc_fpp's
    N=100000 Monte-Carlo draw legitimately takes minutes on some targets,
    unlike a MAST network round-trip (120s there vs 900s default here).

    Returns (result_or_None, timed_out: bool). Any exception raised by
    *calc_fpp_fn* is re-raised on the caller's thread (calc_fpp itself
    already catches its own internal exceptions and returns None on
    failure — see worker/triceratops_fpp.py — so this is a defensive
    backstop, not the expected path; the caller's existing
    ``except Exception`` around this call is left in place to catch it,
    matching pre-change behaviour).
    """
    import threading as _threading

    result_box: list = []
    exc_box: list = []

    def _run():
        try:
            result_box.append(calc_fpp_fn(
                tic_id, period, t0, dur, depth_ppm, secs, time_arr, flux_arr,
                N=100000, workdir="/tmp/tri_work"))
        except Exception as exc:  # noqa: BLE001
            exc_box.append(exc)

    th = _threading.Thread(target=_run, daemon=True)
    th.start()
    th.join(timeout=timeout_seconds)

    if th.is_alive():
        return None, True
    if exc_box:
        raise exc_box[0]
    return (result_box[0] if result_box else None), False


def _fpp_timeout_summary_line(timed_out_tics: list, n_timeout: int,
                              timeout_seconds: float) -> str:
    """Build the single, deliberately greppable "WARNING fpp-timeout: ..."
    stdout line for Step 3's end-of-loop summary.

    Extracted as its own function purely for testability (the caller just
    ``print()``s the result) — review follow-up (2026-07-02): this line lets
    ``step_postprocess`` (discovery_orchestrator.py) scan stdout for the
    exact ``"WARNING fpp-timeout:"`` prefix and surface it into the
    orchestrator's own log/state on both success and failure, instead of the
    fact only ever reaching the git-committed dossier.
    """
    tics_str = ", ".join(timed_out_tics[:20])
    if len(timed_out_tics) > 20:
        tics_str += f", ... ({len(timed_out_tics) - 20} more)"
    return (f"WARNING fpp-timeout: {n_timeout} candidate(s) skipped FPP "
            f"(exceeded {timeout_seconds:.0f}s): TIC {tics_str}")


def _strong_candidate_passes(row, sysp, eb_hosts, depth_gate_ids, toi_protection):
    """Step 2+3 survivor-selection row predicate — extracted for testability.

    Mirrors exactly the ``strong`` list-comprehension filter in ``main()``:
    DEPTH_GATE (``depth_gate_ids``) is a DIFFERENT veto and is never
    overridden by anti-TOI protection (deliberately out of scope). But
    SYSTEMATIC_VETO (``is_vetoed``/``sysp``) and DEEP_SYSTEMATIC_VETO
    (``eb_hosts``) membership — the SAME two predicates Steps 1/1c already
    protect at the DB-write level — must ALSO be overridden here when the row
    is a confirmed (KP/CP) TOI (``is_toi_protected``).

    Review round 2 (2026-07-03): without this override, a protected row would
    keep ``claude_verdict='planet_candidate'`` in the DB (correct) yet be
    silently and permanently excluded from ever becoming a survivor/finalist
    (the bug) — and the dossier's "Deep-systematic veto" section could list
    the same TIC as "excluded from finalists" right next to the
    "[anti-toi] PROTECTED ..." stdout line from Steps 1/1c, a direct
    self-contradiction.

    ``row`` needs ``id``, ``tic_id``, ``period_days``, ``t0`` keys (works with
    a raw DB row or a plain dict).
    """
    if row["id"] in depth_gate_ids:
        return False
    protected = is_toi_protected(row["tic_id"], row["period_days"], toi_protection)
    if is_vetoed(row["period_days"], row["t0"], sysp) and not protected:
        return False
    if row["tic_id"] in eb_hosts and not protected:
        return False
    return True


def _resolve_systematic_periods(pop, args, common_period_min_tics):
    """Return the ``sysp`` list Steps 1/1c veto against.

    Injection-recovery e2e harness (docs/superpowers/specs/2026-07-11-injection
    -recovery-e2e.md, deliverable 2): when ``args.systematic_periods`` is set,
    LOAD the frozen family set from that Parquet path (written by
    ``scripts/eval/export_systematic_families.py`` /
    ``exonit.systematic_families_io.dump_families``) instead of recomputing it
    live — this lets an isolated-DB run with synthetic candidates veto against
    the SAME campaign-wide families as production, without synthetics
    distorting the population ``find_systematic_periods`` would otherwise scan.

    Without the flag (``args.systematic_periods`` falsy, the default), this
    calls ``find_systematic_periods`` exactly as before — byte-identical
    production behaviour. Extracted to its own function purely so the branch
    is unit-testable without running the whole of ``main()``.

    2026-08-01 (postprocess-streaming F3, §3.C wiring of F2): ``pop`` is now
    usually a ``ColumnarPopulation`` (built by ``main()`` — see Step 1), in
    which case this dispatches to ``find_systematic_periods_columnar``
    instead, REQUIRED to be bit-identical to ``find_systematic_periods`` on
    the equivalent list-of-dicts (see that function's own docstring/parity
    tests). ``pop`` can still be a plain list — the existing unit tests
    (tests/test_postprocess_systematic_periods_flag.py) and the injection-
    recovery harness call this function directly with an in-memory list —
    so the dispatch is a type check, not a hard requirement on the caller.
    """
    if args.systematic_periods:
        sysp = load_families(args.systematic_periods)
        print(f"[veto] loaded {len(sysp)} frozen systematic families from "
              f"{args.systematic_periods} (--systematic-periods)")
        return sysp
    # min_concentration=0.4 because with the full-campaign population (tens of
    # thousands of candidates) even a modest concentration is statistically
    # significant (3000 TICs × conc=0.44 >> noise floor). Using 0.5 missed
    # the P~1.68d systematic that only reaches conc=0.445 at campaign scale.
    kwargs = dict(min_stars=10, min_concentration=0.4,
                  common_period_min_tics=common_period_min_tics,
                  common_period_max_period=args.common_period_max_period)
    if isinstance(pop, ColumnarPopulation):
        return find_systematic_periods_columnar(pop, **kwargs)
    return find_systematic_periods(pop, **kwargs)


def _run_cross_sector_step(con, tics, sector, out_path, sample, stem_arg, min_snr):
    """Step 1d. -> (n_tics or None, hits). None = skipped/failed (non-fatal)."""
    if sector is None:
        print("[cross-sector] --sector not given: step skipped")
        return None, {}
    try:
        from exonit.cross_sector import (
            find_cross_sector_hits, render_review_queue_md, review_queue_json)
        cs_rows = []
        for chunk in _select_in_chunks_iter(
                con,
                "SELECT id,tic_id,sector,period_days,t0,snr,claude_verdict,"
                "exominer_ffi_score FROM candidates WHERE tic_id IN",
                list(tics), sql_after="AND period_days IS NOT NULL"):
            cs_rows.extend(dict(r) for r in chunk)
        hits = find_cross_sector_hits(cs_rows, sector, min_snr=min_snr)
        stem = stem_arg or (
            out_path[:-len("_dossier.md")] + "_cross_sector"
            if out_path.endswith("_dossier.md")
            else str(Path(out_path).with_suffix("")) + "_cross_sector")
        label = Path(sample).stem
        Path(stem + ".md").write_text(render_review_queue_md(hits, label, sector))
        Path(stem + ".json").write_text(json.dumps(
            review_queue_json(hits, label, sector), indent=2, default=str))
        print(f"[cross-sector] {len(hits)} TIC(s) with recurrence -> {stem}.{{md,json}}")
        return len(hits), hits
    except Exception as exc:  # non-fatal: report-only tail step
        print(f"[cross-sector] WARNING step failed (non-fatal): {exc!r}")
        return None, {}


def _cross_sector_dossier_lines(n, hits):
    if n is None:
        return []
    lines = ["", "## Cross-sector recurrence (review queue)", "",
             f"Cross-sector review queue: {n} TIC(s)."]
    for h in sorted(hits.values(),
                    key=lambda h: (-h["cross_sector_hits"],
                                   -(h["best"].get("snr") or 0), h["tic_id"]))[:20]:
        m0 = h["matches"][0]
        lines.append(f"- TIC {h['tic_id']}: {h['cross_sector_hits']} other sector(s), "
                     f"P={m0['cur_period']:.4f} d ~ s{m0['other_sector']} "
                     f"P={m0['other_period']:.4f} d ({m0['kind']})")
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", required=True)
    ap.add_argument("--label", default="candidate")
    ap.add_argument("--v6-min", type=float, default=0.5)
    ap.add_argument("--db", default=None)
    ap.add_argument("--out", default="docs/discovery-vetting-dossier.md")
    ap.add_argument("--skip-fpp", action="store_true")
    ap.add_argument("--sector", type=int, default=None,
                    help="current sector of the lot; enables the cross-sector "
                         "recurrence review queue (skipped when omitted)")
    ap.add_argument("--cross-sector-out", default=None,
                    help="output stem for the cross-sector queue (.md/.json); "
                         "default: --out with _dossier.md replaced by _cross_sector")
    ap.add_argument("--cross-sector-min-snr", type=float, default=0.0)
    ap.add_argument("--apply-veto", action="store_true", help="reclassify vetoed candidates in DB")
    ap.add_argument("--veto-scope", choices=["sample", "all", "campaign"], default="campaign",
                    help="population for systematic-period detection: 'sample' = current "
                         "batch only (legacy, may miss low-N systematics in small lots), "
                         "'campaign' = all discovery candidates in DB (default, recommended), "
                         "'all' = all candidates in DB (alias for campaign)")
    ap.add_argument("--common-period-min-tics", type=int, default=10,
                    help="veto any period bin with ≥ N distinct TICs regardless of epoch "
                         "clustering (epoch-agnostic artefacts such as P≈0.71 d). "
                         "Set to 0 to disable. Default: 10.")
    # NOTE (2026-07-02 cleanup review, item 5): this CLI flag itself is not
    # exercised by any test — the orchestrator (step_postprocess) never passes
    # it, always running with the default. Its underlying parameter
    # (common_period_max_period on find_systematic_periods) IS covered
    # directly by tests/test_systematic_veto.py. Escape hatch for manual
    # re-runs, not a production knob.
    ap.add_argument("--common-period-max-period", type=float, default=2.0,
                    help="upper period bound (days) for the epoch-agnostic common-period "
                         "veto. Bins at P ≥ this value are NOT flagged by the agnostic "
                         "path (only the epoch-clustered path applies there). "
                         "Default: 2.0 d — covers the known short-period artefacts "
                         "(P<1.7 d) without risking genuine planets at longer periods.")
    ap.add_argument("--apply-depth-gate", action="store_true",
                    help="reclassify planet_candidates with Rp > %.1f R_Jup as likely_false_positive" %
                         MAX_PLANET_RADIUS_RJUP)
    ap.add_argument("--deep-eclipse-ppm", type=float, default=40_000.0,
                    help="absolute depth threshold (ppm) for the TIC-level deep-systematic "
                         "veto (default: 40000). Signals above this value are examined "
                         "against the commonality (is_vetoed) and geometry (Rp) arms.")
    ap.add_argument("--apply-deep-eclipse-veto", action="store_true",
                    help="reclassify deep-systematic-host TICs' planet_candidate rows to "
                         "likely_false_positive with cross_source_name='DEEP_SYSTEMATIC_VETO'. "
                         "By default (off) the veto is report-only: EB/artefact hosts are "
                         "excluded from finalists but the DB is not modified.")
    ap.add_argument("--fetch-missing-stellar", action="store_true",
                    help="fill missing R* for the finalist-candidate survivors "
                         "(NOT every planet_candidate) from the TIC via MAST — "
                         "chunked/bounded, server-side, graceful fallback for "
                         "FFI-only TICs. Used together with --apply-depth-gate.")
    ap.add_argument("--double-period-check", action="store_true", default=True,
                    help="run double-period EB check on finalists via SPOC 2-min LC "
                         "(default: ON; disable with --no-double-period-check)")
    ap.add_argument("--no-double-period-check", dest="double_period_check",
                    action="store_false",
                    help="skip double-period EB check (finalists unchanged)")
    ap.add_argument("--double-period-sigma", type=float, default=3.0,
                    help="sigma threshold for the double-period EB check (default: 3.0)")
    ap.add_argument("--spoc-repeat-check", action="store_true", default=True,
                    help="run SPOC repeatability check on finalists (default: ON)")
    # NOTE (2026-07-02 cleanup review, item 5): --no-spoc-repeat-check is not
    # exercised by any test — the orchestrator always runs with the default
    # (check ON). spoc_repeat_check() itself IS covered directly by
    # tests/test_finalist_gates.py. Escape hatch for manual re-runs.
    ap.add_argument("--no-spoc-repeat-check", dest="spoc_repeat_check",
                    action="store_false",
                    help="skip SPOC repeatability check")
    ap.add_argument("--spoc-repeat-sigma", type=float, default=3.0,
                    help="sigma threshold for SPOC repeat check (default: 3.0)")
    ap.add_argument("--orbit-start-check", action="store_true", default=True,
                    help="run orbit-start ramp artifact check on finalists (default: ON)")
    # NOTE (2026-07-02 cleanup review, item 5): --no-orbit-start-check is not
    # exercised by any test — the orchestrator always runs with the default
    # (check ON). orbit_start_artifact_check() itself IS covered directly by
    # tests/test_finalist_gates.py. Escape hatch for manual re-runs.
    ap.add_argument("--no-orbit-start-check", dest="orbit_start_check",
                    action="store_false",
                    help="skip orbit-start ramp artifact check")
    # Shared-signal cross-check (Step 5e). ON by default: it fired on real
    # data the day it was written (4 of 11 finalists in the s14 07-22 lot were
    # one signal), and a duplicate finalist is a false discovery claim.
    ap.add_argument("--shared-signal-check", action="store_true", default=True,
                    help="cross-check finalists against EACH OTHER for one "
                         "signal seen on several nearby stars (default: ON)")
    ap.add_argument("--no-shared-signal-check", dest="shared_signal_check",
                    action="store_false",
                    help="skip the shared-signal cross-check between finalists")
    ap.add_argument("--shared-signal-period-rtol", type=float, default=0.01,
                    help="relative period tolerance for the shared-signal "
                         "cross-check (default: 0.01 = 1%%)")
    ap.add_argument("--shared-signal-max-sep-deg", type=float, default=1.0,
                    help="max sky separation in degrees for two finalists to "
                         "count as the same signal (default: 1.0)")
    ap.add_argument("--toi-catalog", default=None,
                    help="path to TOI catalog CSV (default: data/toi_catalog.csv "
                         "relative to repo root)")
    ap.add_argument("--systematic-periods", default=None,
                    help="path to a frozen systematic-families Parquet file "
                         "(written by scripts/eval/export_systematic_families.py "
                         "or exonit.systematic_families_io.dump_families). When "
                         "given, Steps 1/1c LOAD this family set instead of "
                         "computing it live via find_systematic_periods over "
                         "the DB population — used by the injection-recovery "
                         "e2e harness to freeze prod families for an isolated "
                         "run so synthetic candidates never contaminate the "
                         "systematic-period detection. Default: None (live "
                         "computation, unchanged production behaviour).")
    ap.add_argument("--fpp-timeout-seconds", type=float,
                    default=float(os.environ.get(
                        "EXONIT_POSTPROCESS_FPP_TIMEOUT",
                        str(DEFAULT_FPP_TIMEOUT_SECONDS))),
                    help="wall-clock ceiling (seconds) for each TRICERATOPS FPP "
                         "Monte-Carlo call in Step 3; a candidate that exceeds this "
                         "is recorded as fpp-timeout and excluded from finalists "
                         "instead of hanging the run (B3; default: 900s, or "
                         "$EXONIT_POSTPROCESS_FPP_TIMEOUT)")
    args = ap.parse_args()

    db = args.db or ExoNitConfig().db_path
    # Operational DB access goes through the backend seam so a postgresql:// DSN
    # works exactly like a SQLite path. ``con`` keeps the sqlite3-flavoured
    # interface (execute/fetchall/fetchone/commit) the steps below rely on.
    con = op_session(db)
    mc = MetadataCache(ExoNitConfig().metadata_cache_path)
    sample = {r["tic_id"].strip(): r["label"] for r in csv.DictReader(open(args.sample))}
    tics = {t for t, l in sample.items() if l == args.label}

    # --- Anti-TOI override (always on, no CLI flag) ---
    # SYSTEMATIC_VETO (Step 1) and DEEP_SYSTEMATIC_VETO (Step 1c) below must
    # never demote a CONFIRMED planet (TOI TFOPWG Disposition KP/CP) — its
    # real period can coincide with an instrumental systematic by chance.
    # Known victims (2026-06-30 miss-analyst, s1): TIC 176956893 (CP,
    # P=6.907d), TIC 336732544 (KP, P=3.277d), TIC 364395234 (CP, P=14.255d).
    # Same default catalog path resolution as _load_toi_catalog (Step 5d
    # anti-TOI cross-match) below.
    toi_protection_path = args.toi_catalog or str(ROOT / "data" / "toi_catalog.csv")
    toi_protection = load_toi_protection(toi_protection_path)

    # --- Step 1: systematic-period veto ---
    # Build the population for systematic-period detection.
    #
    # 'campaign' / 'all': use every candidate in the DB (recommended). This
    # accumulates signal across all discovery lots and finds systematics like
    # P≈0.71 d or 1.68 d that appear in <10 TICs within a single lot but in
    # hundreds of TICs across the full campaign.
    #
    # 'sample': legacy mode — only the current lot/batch. Kept for back-compat
    # but may miss low-N systematics in small lots.
    #
    # 2026-08-01 (postprocess-streaming F3, §3.B/§3.C): the population is
    # loaded as a ColumnarPopulation (3 numpy arrays) instead of a 22 M-row
    # list-of-dicts — kills the ~13 GB `allc` (H1/H7 in the spec's hot-spot
    # inventory). `allc` itself is GONE: nothing downstream needs the full
    # campaign as dicts any more (`sample_cands`, below, gets its own
    # lot-scoped query instead of filtering `allc` — see H2 in the spec).
    if args.veto_scope == "sample":
        # sample scope: population IS the lot — same tic_id IN (...) chunked
        # query used everywhere else in this file, fed straight into the
        # columnar builder (chain.from_iterable flattens the per-chunk lists
        # select_in_chunks_iter yields into the flat row iterable
        # ColumnarPopulation.from_rows expects).
        pop = ColumnarPopulation.from_rows(chain.from_iterable(
            _select_in_chunks_iter(
                con,
                "SELECT tic_id,period_days,t0 FROM candidates WHERE tic_id IN",
                list(tics),
                sql_after="AND period_days IS NOT NULL AND t0 IS NOT NULL")))
    else:
        # campaign / all: full DB population for detection; apply veto only to
        # the current sample's planet_candidates (we don't re-process old lots
        # here). execute_stream drains the whole `candidates` table without
        # ever materializing it — same flattening as the sample-scope branch.
        pop = ColumnarPopulation.from_rows(chain.from_iterable(
            con.execute_stream(
                "SELECT tic_id,period_days,t0 FROM candidates "
                "WHERE period_days IS NOT NULL AND t0 IS NOT NULL")))
    print(f"[veto] scope={args.veto_scope}: {len(pop)} candidates in detection population")
    common_period_min_tics = args.common_period_min_tics if args.common_period_min_tics > 0 else None
    # See _resolve_systematic_periods above for the live-vs-frozen branch
    # (--systematic-periods) and the min_concentration=0.4 rationale.
    sysp = _resolve_systematic_periods(pop, args, common_period_min_tics)
    # F2.1 (2026-08-01): `pop` (the campaign-wide columnar population, ~530 MB
    # at 22 M rows) is never touched again after this point — the veto is
    # evaluated with is_vetoed(period, t0, sysp) over `sample_cands` below,
    # not over `pop` itself. Drop the reference now so the GC can free it
    # BEFORE `sample_cands` is loaded, instead of both being resident at once.
    del pop
    # Evaluate veto over ALL candidates in the current sample (not just the
    # population) — lot-scoped query directly (H2 in the spec: this used to
    # be a filter over `allc`, which pinned the full 13 GB campaign list
    # alive until here; now it's its own query and `allc` never exists).
    sample_cands = select_in_chunks(
        con.execute,
        "SELECT id,tic_id,period_days,t0,claude_verdict FROM candidates WHERE tic_id IN",
        list(tics),
        sql_after="AND period_days IS NOT NULL AND t0 IS NOT NULL")
    vetoed_pc_raw = [c for c in sample_cands if c["claude_verdict"] == "planet_candidate"
                     and is_vetoed(c["period_days"], c["t0"], sysp)]
    # Anti-TOI override: drop confirmed (KP/CP) planets from the veto list.
    vetoed_pc, n_anti_toi_protected = filter_anti_toi_protected(
        vetoed_pc_raw, toi_protection, "SYSTEMATIC_VETO")
    print(f"[veto] {len(sysp)} systematic periods; {len(vetoed_pc)} planet_candidate vetoed"
          + (f" ({n_anti_toi_protected} anti-TOI protected)" if n_anti_toi_protected else ""))
    if args.apply_veto:
        for c in vetoed_pc:
            con.execute("UPDATE candidates SET claude_verdict='likely_false_positive', "
                        "cross_source_name='SYSTEMATIC_VETO' WHERE id=?", (c["id"],))
        con.commit()
        print(f"[veto] reclassified {len(vetoed_pc)}")

    # --- Step 1b: depth gate (Rp > MAX_PLANET_RADIUS_RJUP → likely EB) ---
    # Load stellar radii from metadata cache for all sample TICs.
    mc_con = sqlite3.connect(ExoNitConfig().metadata_cache_path)
    mc_con.row_factory = sqlite3.Row
    stellar_radii: dict[str, float] = {}
    for tic in tics:
        row = mc_con.execute(
            "SELECT stellar_radius FROM metadata WHERE tic_id=?", (tic,)
        ).fetchone()
        if row and row["stellar_radius"] is not None:
            stellar_radii[tic] = float(row["stellar_radius"])
    mc_con.close()
    print(f"[depth_gate] stellar radii available for {len(stellar_radii)}/{len(tics)} TICs")
    # NOTE: this early depth gate uses LOCAL R* only (FITS/SPOC).  The MAST R*
    # fallback for FFI-only TICs is applied later, bounded to the
    # finalist-candidate `survivors` set (Step 2+3) — NOT to every
    # planet_candidate, which would flood MAST and hang postprocess (the
    # 2026-06-29 incident).  So here SPOC giants are caught + reclassified; the
    # FFI-only giants are caught at the survivor stage where the set is small.
    #
    # 2026-08-01 (postprocess-streaming F3, §3.C): apply_depth_gate needs
    # depth_ppm; reload from DB with that column, lot-scoped + chunked (H3 in
    # the spec) instead of pulling every planet_candidate in the DB and
    # filtering `tic_id in tics` in Python. apply_depth_gate is per-row with
    # no cross-row state (verified against exonit/systematic_veto.py before
    # this change), so it's safe to run per chunk and just accumulate the
    # (few) flagged rows — never materializing the full depth_gate_input.
    depth_gate_flags = []
    for chunk in _select_in_chunks_iter(
            con,
            "SELECT id,tic_id,period_days,t0,depth_ppm,claude_verdict FROM candidates "
            "WHERE tic_id IN",
            list(tics),
            sql_after="AND claude_verdict='planet_candidate' AND depth_ppm IS NOT NULL"):
        depth_gate_flags.extend(apply_depth_gate(chunk, stellar_radii))
    # exclude already-vetoed by period
    vetoed_ids = {c["id"] for c in vetoed_pc}
    depth_gate_flags = [c for c in depth_gate_flags if c["id"] not in vetoed_ids]
    print(f"[depth_gate] {len(depth_gate_flags)} planet_candidates flagged as EB "
          f"(Rp > {MAX_PLANET_RADIUS_RJUP:.1f} R_Jup)")
    if args.apply_depth_gate:
        for c in depth_gate_flags:
            con.execute("UPDATE candidates SET claude_verdict='likely_false_positive', "
                        "cross_source_name='DEPTH_GATE' WHERE id=?", (c["id"],))
        con.commit()
        print(f"[depth_gate] reclassified {len(depth_gate_flags)}")

    # --- Step 1c: TIC-level deep-systematic veto (artefact/EB host) ---
    # Load ALL signals (any verdict, including t0 and period_days) for the
    # sample's TICs so that a deep systematic signal on a likely_fp row can
    # veto a shallow planet_candidate from the same star — the per-signal gap
    # that the common-period veto misses (it removes the deep signal but never
    # propagates to the TIC level).
    # Reuses sysp already built above (no extra find_systematic_periods call)
    # and stellar_radii already loaded; no extra DB connection needed.
    #
    # 2026-08-01 (postprocess-streaming F3, §3.D wiring of F2): `all_sigs` is
    # no longer materialized (H4 in the spec, ~0.5-1 GB at a 300k lot) —
    # find_deep_eclipse_hosts is per-signal (a TIC is flagged if ANY of its
    # signals qualify, no cross-signal state beyond the output set), so it
    # accepts a plain iterable and this feeds it the chunked query directly,
    # never holding more than one chunk in memory at a time.
    # Identify which sample TICs already have a planet_candidate row
    # (used for the non-alarming diagnostic print below).
    planet_candidate_tics = {c["tic_id"] for c in sample_cands
                             if c["claude_verdict"] == "planet_candidate"}
    eb_hosts = find_deep_eclipse_hosts(
        chain.from_iterable(_select_in_chunks_iter(
            con,
            "SELECT tic_id, period_days, t0, depth_ppm FROM candidates WHERE tic_id IN",
            list(tics),
        )),
        sysp, stellar_radii,
        deep_eclipse_ppm=args.deep_eclipse_ppm,
    )
    print(f"[deep-systematic] {len(eb_hosts)} artefact/EB-host TICs in sample "
          f"({len(eb_hosts & planet_candidate_tics)} with active planet_candidate rows)")
    if args.apply_deep_eclipse_veto and eb_hosts:
        # Reclassify planet_candidate rows for deep-systematic-host TICs in the DB.
        # Mirror the veto and depth-gate UPDATE pattern exactly.
        #
        # Only TICs that actually have an active planet_candidate row can be
        # changed; the UPDATE on any other TIC matches 0 rows (no-op). At sector
        # scale eb_hosts is huge (29 120 in lot 20260606T104546) while the ones
        # with a live planet_candidate row are a handful — so restricting the
        # loop to (eb_hosts & planet_candidate_tics) is behaviour-identical but
        # turns tens of thousands of no-op UPDATEs into a few real ones. (Before
        # idx_candidates_tic existed each of those no-op UPDATEs full-scanned the
        # 665k-row table — the dominant cause of the multi-minute postprocess
        # stalls + lock contention.)
        to_update = eb_hosts & planet_candidate_tics
        # Anti-TOI override, row-level: a TIC can host several planet_candidate
        # signals (different periods); only the row(s) matching a confirmed
        # KP/CP TOI period are protected — the TIC's other rows still veto
        # normally. A TIC whose EVERY planet_candidate row is protected drops
        # out of to_update entirely (no UPDATE at all for it).
        to_update, protected_ids_by_tic = partition_deep_veto_targets(
            to_update, sample_cands, toi_protection)
        for tic in to_update:
            protected_ids = protected_ids_by_tic.get(tic)
            if protected_ids:
                placeholders = ",".join("?" * len(protected_ids))
                con.execute(
                    "UPDATE candidates SET claude_verdict='likely_false_positive', "
                    "cross_source_name='DEEP_SYSTEMATIC_VETO' "
                    "WHERE tic_id=? AND claude_verdict='planet_candidate' "
                    f"AND id NOT IN ({placeholders})",
                    (tic, *protected_ids),
                )
            else:
                con.execute(
                    "UPDATE candidates SET claude_verdict='likely_false_positive', "
                    "cross_source_name='DEEP_SYSTEMATIC_VETO' "
                    "WHERE tic_id=? AND claude_verdict='planet_candidate'",
                    (tic,),
                )
        con.commit()
        print(f"[deep-systematic] reclassified planet_candidate rows for "
              f"{len(to_update)} TICs ({len(eb_hosts)} hosts flagged)")

    # --- Step 1d: cross-sector recurrence review queue (report only) ---
    n_cross_sector, cross_sector_hits = _run_cross_sector_step(
        con, tics, args.sector, args.out, args.sample,
        args.cross_sector_out, args.cross_sector_min_snr)

    # --- Step 2+3: layer-3 vetting + layer-4 FPP on strong candidates ---
    from vetting import check_odd_even, check_v_shape  # worker module
    depth_gate_ids = {c["id"] for c in depth_gate_flags}
    # Anti-TOI override: _strong_candidate_passes re-applies the SAME
    # SYSTEMATIC_VETO (is_vetoed/sysp) and DEEP_SYSTEMATIC_VETO (eb_hosts)
    # predicates already protected at the DB-write level in Steps 1/1c — a
    # protected KP/CP row must survive here too (see its docstring; review
    # round 2, 2026-07-03).
    #
    # 2026-08-01 (postprocess-streaming F3, §3.C): lot-scoped + chunked (H5 in
    # the spec) instead of pulling every planet_candidate above v6_min in the
    # DB and filtering `tic_id in tics` in Python — the `IN` chunking already
    # does that filter in SQL. "best candidate per TIC" is computed
    # chunk-locally: select_in_chunks_iter's chunking is BY tic_id, so every
    # row for a given TIC lands in the SAME chunk (never split across two),
    # and each chunk keeps the ORDER BY exominer_ffi_score DESC — so the
    # first row seen per tic_id within a chunk is exactly the row the old
    # single global-ORDER-BY query would have kept via
    # `best.setdefault(tic_id, r)`. `survivors` (small — one row per
    # surviving TIC) is then re-sorted globally by
    # (-exominer_ffi_score, tic_id) to restore the priority order the old
    # global ORDER BY gave the dossier/finalist steps below, AND to make tied
    # scores deterministic (previously broken by backend row-arrival order —
    # the one accepted behaviour divergence in this change, see spec §5).
    best = {}
    for chunk in _select_in_chunks_iter(
            con,
            "SELECT id,tic_id,period_days,t0,duration_days,depth_ppm,snr,"
            "exominer_ffi_score,triceratops_fpp,triceratops_verdict "
            "FROM candidates WHERE tic_id IN",
            list(tics),
            sql_after="AND claude_verdict='planet_candidate' AND exominer_ffi_score>? "
                      "ORDER BY exominer_ffi_score DESC",
            params_extra=(args.v6_min,)):
        for r in chunk:
            if _strong_candidate_passes(r, sysp, eb_hosts, depth_gate_ids, toi_protection):
                best.setdefault(r["tic_id"], r)
    survivors = sorted(best.values(), key=lambda r: (-r["exominer_ffi_score"], r["tic_id"]))
    print(f"[layer3] {len(survivors)} strong candidates (post-veto) to vet")
    # TICs whose eb_host / systematic-veto membership was overridden above by
    # anti-TOI protection — a protected row survived despite the TIC being a
    # deep-systematic/EB host. Used to keep the dossier's "Deep-systematic
    # veto" section (Step 6) truthful instead of contradicting these
    # survivors / the "[anti-toi] PROTECTED ..." stdout lines from Steps 1/1c.
    eb_hosts_protected_survivors = {s["tic_id"] for s in survivors if s["tic_id"] in eb_hosts}

    # FFI-only giant gate (bounded MAST R* fallback).  The early depth gate
    # (Step 1b) only had local R* (FITS/SPOC), so FFI-only survivors with no
    # local R* slipped through.  Here we fetch R* from MAST ONLY for the
    # survivors — finalist-candidate scale (hundreds/lot), one query per TIC —
    # NOT for every planet_candidate (~thousands), which floods MAST and hangs
    # (2026-06-29).  Fail-open: a TIC whose R* can't be fetched is never demoted.
    if args.fetch_missing_stellar and survivors:
        from exonit.mast_stellar import fetch_stellar_params_mast as _fetch_mast
        _seen_tic: set = set()
        _surv_missing = []
        for s in survivors:  # preserve em_ffi-DESC priority order; dedup per TIC
            t = s["tic_id"]
            if t not in stellar_radii and t not in _seen_tic:
                _seen_tic.add(t)
                _surv_missing.append(t)
        if _surv_missing:
            print(f"[depth_gate] MAST fallback: R* for {len(_surv_missing)} "
                  f"survivor TIC(s) missing local R* ...")
            _mast = _fetch_mast(_surv_missing)
            _nf = 0
            for _tid, _p in _mast.items():
                if _p.get("rad") is not None and _tid not in stellar_radii:
                    stellar_radii[_tid] = _p["rad"]
                    _nf += 1
            print(f"[depth_gate] MAST fallback: filled R* for {_nf}/{len(_surv_missing)} TIC(s)")
            _giants = apply_depth_gate(survivors, stellar_radii)
            if _giants:
                _gids = {c["id"] for c in _giants}
                if args.apply_depth_gate:
                    for c in _giants:
                        con.execute(
                            "UPDATE candidates SET claude_verdict='likely_false_positive', "
                            "cross_source_name='DEPTH_GATE' WHERE id=?", (c["id"],))
                    con.commit()
                survivors = [s for s in survivors if s["id"] not in _gids]
                print(f"[depth_gate] MAST-backed gate flagged {len(_giants)} giant "
                      f"FFI-only FP(s); {len(survivors)} survivor(s) remain")

    rows = []
    rep = ProgressReporter(unit="candidates", total=len(survivors),
                           substep="postprocess", interval_s=15)
    for i, s in enumerate(survivors, 1):
        t, f, secs = load_lc(mc, s["tic_id"])
        if t is None:
            continue
        P, t0, dur = s["period_days"], s["t0"], s["duration_days"]
        oe = check_odd_even(t, f, P, t0, dur)
        phase = ((t - t0) % P) / P; phase[phase > 0.5] -= 1
        iw = np.abs(phase) < (dur / P) * 2
        vs = check_v_shape(phase[iw], f[iw]) if iw.sum() >= 10 else {"is_v_shaped": False}
        sec = secondary_depth(t, f, P, t0, dur)
        flags = []
        if not oe["odd_even_consistent"]:
            flags.append("odd/even")
        if vs["is_v_shaped"]:
            flags.append("V-shape")
        if sec is not None and s["depth_ppm"] and sec > 0.3 * s["depth_ppm"] and sec > 300:
            flags.append("secondary")
        s["_eb_flags"] = flags
        s["_sec"] = sec
        rows.append(s)
        rep.report(i, message=f"TIC {s['tic_id']}")
    rep.finish()

    l3_pass = [s for s in rows if not s["_eb_flags"]]
    print(f"[layer3] {len(l3_pass)}/{len(rows)} pass (no EB flags)")

    # --- Step 3 (layer 4): TRICERATOPS ---
    fpp_res = {}
    if not args.skip_fpp and l3_pass:
        n_from_db = n_recomputed = n_failed = n_timeout = 0
        timed_out_tics: list[str] = []
        from triceratops_fpp import calc_fpp  # SSL patch applied on import
        os.makedirs("/tmp/tri_work", exist_ok=True)
        for s in l3_pass:
            db_fpp = s.get("triceratops_fpp")
            if db_fpp is not None and db_fpp >= 0:
                fpp_res[s["tic_id"]] = {"fpp": float(db_fpp),
                                        "verdict": s.get("triceratops_verdict") or "db"}
                n_from_db += 1
                continue
            t, f, secs = load_lc(mc, s["tic_id"])
            if t is None:
                n_failed += 1
                continue
            if secs is None or len(secs) == 0:
                secs = np.array([1])
            try:
                # B3: bounded by args.fpp_timeout_seconds (default 900s) — a
                # pathological candidate must not hang the whole POSTPROCESS
                # tail forever with the fleet already stopped.
                fpp, timed_out = _run_fpp_with_timeout(
                    calc_fpp, s["tic_id"], s["period_days"], s["t0"], s["duration_days"],
                    s["depth_ppm"], secs, t, f, timeout_seconds=args.fpp_timeout_seconds)
                if timed_out:
                    print(f"[fpp] {s['tic_id']} TIMEOUT after "
                          f"{args.fpp_timeout_seconds:.0f}s — recording as "
                          f"fpp-timeout and continuing")
                    fpp_res[s["tic_id"]] = {"fpp": 1.0, "verdict": "fpp-timeout"}
                    n_timeout += 1
                    timed_out_tics.append(s["tic_id"])
                elif fpp:
                    fpp_res[s["tic_id"]] = fpp
                    n_recomputed += 1
                else:
                    n_failed += 1
            except Exception as e:
                print(f"[fpp] {s['tic_id']} error: {type(e).__name__}")
                n_failed += 1
        print(f"[fpp] {n_from_db} from DB, {n_recomputed} recomputed, "
              f"{n_timeout} timeout ({n_failed} failed)")
        if timed_out_tics:
            # Review follow-up (2026-07-02): a timed-out candidate previously
            # only surfaced as a plain table cell in the dossier — a human
            # would only notice it by opening/scanning that file after the
            # fact, unlike the double-period-EB / SPOC-repeat / orbit-start
            # checks which each get a loud dedicated dossier section. This
            # line is a deliberately greppable, single-line WARNING so
            # step_postprocess (discovery_orchestrator.py) can scan stdout
            # for it and surface it into the orchestrator's own log stream —
            # on a SUCCESSFUL (rc=0) run too, not only on failure.
            print(_fpp_timeout_summary_line(
                timed_out_tics, n_timeout, args.fpp_timeout_seconds))

    # --- Step 4: compute finalists (l3_pass + FPP gate) ---
    final = [s for s in l3_pass if fpp_res.get(s["tic_id"], {}).get("fpp", 1.0) < 0.5] if not args.skip_fpp else l3_pass

    # --- Step 5: double-period EB check on finalists ---
    # Applied ONLY to finalists (few candidates, ~0-2 per lot) because MAST
    # fetches are expensive.  A verdict of "unknown" (MAST unavailable / no
    # SPOC data) does NOT block the finalist — conservative for sensitivity.
    dp_verdicts: dict[str, tuple[str, str]] = {}  # tic_id → (verdict, detail)
    if args.double_period_check and final:
        print(f"[double-period] checking {len(final)} finalist(s) for double-period EB ...")
        for s in final:
            verdict, detail = double_period_eb_check(
                s["tic_id"], s["period_days"], s["t0"], s["duration_days"],
                sigma_thresh=args.double_period_sigma,
            )
            dp_verdicts[s["tic_id"]] = (verdict, detail)
            print(f"[double-period] TIC {s['tic_id']}: {verdict} — {detail}")

        n_eb = sum(1 for v, _ in dp_verdicts.values() if v == "eb")
        n_ok = sum(1 for v, _ in dp_verdicts.values() if v == "ok")
        n_unk = sum(1 for v, _ in dp_verdicts.values() if v == "unknown")
        print(f"[double-period] {len(final)} finalists checked: "
              f"{n_eb} excluded as EB, {n_ok} ok, {n_unk} unknown")

        # Exclude confirmed double-period EBs from the finalist list
        final = [s for s in final if dp_verdicts.get(s["tic_id"], ("ok", ""))[0] != "eb"]

    # --- Step 5b: SPOC repeatability check on finalists ---
    # Verifies that the transit signal actually repeats across multiple SPOC sectors.
    # A verdict of "absent" (≥2 distinct epochs, ≥30 in-transit points, depth ~0)
    # removes the finalist.  "unknown" (no SPOC, only 1 epoch, timeout) keeps it.
    # This catches orbit-ramp artefacts whose transit is only "seen" once in SPOC.
    spoc_repeat_verdicts: dict[str, tuple[str, str]] = {}  # tic_id → (verdict, detail)
    # Shared SPOC LC cache: the orbit-start check (step 5c) can reuse these downloads
    _spoc_lc_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    if args.spoc_repeat_check and final:
        print(f"[spoc-repeat] checking {len(final)} finalist(s) for SPOC repeatability ...")
        for s in final:
            verdict, detail = spoc_repeat_check(
                s["tic_id"], s["period_days"], s["t0"], s["duration_days"],
                s["depth_ppm"] or 0.0,
                sigma_thresh=args.spoc_repeat_sigma,
                _lc_cache=_spoc_lc_cache,
            )
            spoc_repeat_verdicts[s["tic_id"]] = (verdict, detail)
            print(f"[spoc-repeat] TIC {s['tic_id']}: {verdict} — {detail}")

        n_absent = sum(1 for v, _ in spoc_repeat_verdicts.values() if v == "absent")
        n_conf = sum(1 for v, _ in spoc_repeat_verdicts.values() if v == "confirmed")
        n_unk = sum(1 for v, _ in spoc_repeat_verdicts.values() if v == "unknown")
        print(f"[spoc-repeat] {len(final)} finalists checked: "
              f"{n_absent} excluded (absent), {n_conf} confirmed, {n_unk} unknown")

        final = [s for s in final
                 if spoc_repeat_verdicts.get(s["tic_id"], ("unknown", ""))[0] != "absent"]

    # --- Step 5c: orbit-start ramp artifact check on finalists ---
    # Checks whether all predicted transits (13–15 d period family) fall within
    # the first 4 hours of a TESS data segment (post-downlink ramp).
    # "orbit_start_artifact" (all transits in ramp) excludes the finalist.
    # "unknown" (no LC) keeps it.
    orbit_verdicts: dict[str, tuple[str, str]] = {}  # tic_id → (verdict, detail)
    if args.orbit_start_check and final:
        print(f"[orbit-start] checking {len(final)} finalist(s) for orbit-start ramp ...")
        for s in final:
            verdict, detail = orbit_start_artifact_check(
                s["tic_id"], s["period_days"], s["t0"], s["duration_days"],
                mc=mc,
                _lc_cache=_spoc_lc_cache,
            )
            orbit_verdicts[s["tic_id"]] = (verdict, detail)
            print(f"[orbit-start] TIC {s['tic_id']}: {verdict} — {detail}")

        n_art = sum(1 for v, _ in orbit_verdicts.values() if v == "orbit_start_artifact")
        n_ok_v = sum(1 for v, _ in orbit_verdicts.values() if v == "ok")
        n_na = sum(1 for v, _ in orbit_verdicts.values() if v == "not_applicable")
        n_unk = sum(1 for v, _ in orbit_verdicts.values() if v == "unknown")
        print(f"[orbit-start] {len(final)} finalists checked: "
              f"{n_art} excluded (artifact), {n_ok_v} ok, {n_na} not_applicable, {n_unk} unknown")

        final = [s for s in final
                 if orbit_verdicts.get(s["tic_id"], ("ok", ""))[0] != "orbit_start_artifact"]

    # --- Step 5d-pre: FFI-only finalist hardening (spec §3.1 + §3.3) ---
    # Runs ONLY on has_spoc=0 finalists (no-SPOC, FFI-only TICs).
    # SPOC finalists skip this gate entirely — they have 2-min data and their own
    # checks (double-period, spoc-repeat, orbit-start) which are more powerful.
    #
    # Controlled by env flag EXONIT_FFI_FINALIST_GATE:
    #   OFF (default): compute the check, write a dossier section, leave `final` unchanged.
    #   ON (=1):        actually remove would-demote finalists from `final`.
    #
    # Conservative fallback: missing LC or stellar params → keep the finalist.
    ffi_gate_enabled = os.environ.get("EXONIT_FFI_FINALIST_GATE", "0").strip() == "1"

    # Build has_spoc map for all current finalists from the stars table.
    finalist_tic_ids = [s["tic_id"] for s in final]
    has_spoc_map: dict[str, int] = {}
    if finalist_tic_ids:
        rows_spoc = _select_in_chunks(
            con,
            "SELECT tic_id, has_spoc FROM stars WHERE tic_id IN",
            finalist_tic_ids,
        )
        for r in rows_spoc:
            has_spoc_map[r["tic_id"]] = r["has_spoc"] if r["has_spoc"] is not None else -1

    # Helper: stellar mass from logg + R* (same formula as server.derive_mass_from_logg_rad)
    def _mass_from_logg_rad(logg, rad):
        """R² × 10^(logg − 4.438); returns None on invalid input."""
        if logg is None or rad is None:
            return None
        try:
            return float(rad) ** 2 * 10 ** (float(logg) - 4.438)
        except (TypeError, ValueError):
            return None

    ffi_hardening_results: dict[str, dict] = {}   # tic_id → {passes_3_1, r31, r33, detail}
    ffi_would_demote: set[str] = set()

    for s in final:
        hs = has_spoc_map.get(s["tic_id"], -1)
        if hs != 0:
            # SPOC finalist — skip this gate
            ffi_hardening_results[s["tic_id"]] = {"skipped": True,
                                                   "reason": "has_spoc != 0"}
            continue

        P = s["period_days"]
        t0_val = s["t0"]
        dur = s["duration_days"]

        # --- §3.1: load LC + run folded-depth control test ---
        t_lc, f_lc, _ = load_lc(mc, s["tic_id"])
        r31 = folded_depth_control_test(t_lc, f_lc, P=P, t0=t0_val,
                                        dur_days=dur, n_sigma=2.0)

        # --- §3.3: stellar-consistency annotation (informative only) ---
        entry = mc.get_entry(s["tic_id"])
        r_star = entry["stellar_radius"] if entry else None
        teff = entry["teff"] if entry else None
        logg = entry["logg"] if entry else None
        m_star = _mass_from_logg_rad(logg, r_star)
        # Use measured folded depth for §3.3 (more honest than BLS depth)
        fd_ppm = r31.get("folded_depth_ppm") if r31.get("folded_depth_ppm") else s.get("depth_ppm")
        r33 = stellar_consistency_check(
            folded_depth_ppm=fd_ppm,
            R_star=r_star, Teff=teff, M_star=m_star,
            P=P, dur_days=dur,
        )

        # would-demote: only if passes_3_1=False (§3.1 fails)
        # §3.3 flags are informative; they NEVER demote alone.
        passes = r31["passes_3_1"]  # True = keep; conservative default on error
        if not passes:
            ffi_would_demote.add(s["tic_id"])

        ffi_hardening_results[s["tic_id"]] = {
            "skipped": False,
            "passes_3_1": passes,
            "r31": r31,
            "r33": r33,
        }
        print(
            f"[ffi-gate] TIC {s['tic_id']}: "
            f"eff_snr={r31.get('effective_snr', 'N/A')} "
            f"passes_3_1={passes} "
            f"{'→ would-demote' if not passes else '→ keep'}"
            + (f" (flag OFF)" if not ffi_gate_enabled else "")
        )

    if ffi_would_demote:
        n_wd = len(ffi_would_demote)
        print(f"[ffi-gate] {n_wd} FFI-only finalist(s) would-demote "
              f"(flag={'ON — removing' if ffi_gate_enabled else 'OFF — report-only'}): "
              + ", ".join(f"TIC {t}" for t in sorted(ffi_would_demote)))
        if ffi_gate_enabled:
            final = [s for s in final if s["tic_id"] not in ffi_would_demote]
    else:
        print("[ffi-gate] 0 FFI-only finalists would-demote")

    # --- Step 5e: shared-signal cross-check BETWEEN finalists of this lot ---
    # Every gate above judges one candidate in isolation. That left one failure
    # mode uncovered, and it fired for real: in lot
    # discovery_s0014_lot20260722T073004, FOUR of eleven finalists were a single
    # signal — same period and epoch to four decimals, all inside 0.36° of sky
    # (two of them 2.3 arcmin apart), on four bright stars. Four neighbours
    # dimming at the same instant with the same cadence are not four planets:
    # either a systematic shared across a CCD region, or one real eclipsing
    # source bleeding into neighbouring TGLC apertures.
    #
    # This does NOT decide which member is real — it keeps the strongest (by
    # SNR) and demotes the rest, recording the whole group so Step 6 can print
    # it and a human can look. Runs LAST of the finalist gates so it only ever
    # de-duplicates candidates that already survived every other check.
    shared_groups: list[list[dict]] = []
    shared_demoted: dict[str, str] = {}   # tic_id → tic_id it duplicates
    if args.shared_signal_check and len(final) > 1:
        # ra/dec are not in the candidates table — fetch them for the handful
        # of finalists (never for the full lot).
        coords: dict[str, tuple] = {}
        rows_pos = _select_in_chunks(
            con, "SELECT tic_id, ra, dec FROM stars WHERE tic_id IN",
            [s["tic_id"] for s in final])
        for r in rows_pos:
            coords[r["tic_id"]] = (r["ra"], r["dec"])
        enriched = []
        for s in final:
            ra_v, dec_v = coords.get(s["tic_id"], (None, None))
            enriched.append({**s, "ra": ra_v, "dec": dec_v})
        n_nopos = sum(1 for c in enriched if c["ra"] is None or c["dec"] is None)
        if n_nopos:
            # Never silently: a finalist with no coordinates cannot be
            # cross-checked, and that is exactly the case where a duplicate
            # would slip through unnoticed.
            print(f"[shared-signal] WARNING: {n_nopos} finalist(s) have no ra/dec "
                  f"in `stars` — they are kept, but NOT cross-checked")
        shared_groups = shared_signal_groups(
            enriched,
            period_rtol=args.shared_signal_period_rtol,
            max_sep_deg=args.shared_signal_max_sep_deg,
        )
        for group in shared_groups:
            keeper = group[0]
            for dup in group[1:]:
                shared_demoted[dup["tic_id"]] = keeper["tic_id"]
            print(f"[shared-signal] group of {len(group)} sharing "
                  f"P≈{keeper['period_days']:.4f} d, t0≈{keeper['t0']:.4f}: "
                  + ", ".join(f"TIC {c['tic_id']}(snr={c.get('snr') or 0:.1f})" for c in group)
                  + f" → keeping TIC {keeper['tic_id']}, demoting {len(group) - 1}")
        if shared_demoted:
            final = [s for s in final if s["tic_id"] not in shared_demoted]
            print(f"[shared-signal] {len(shared_demoted)} finalist(s) demoted as "
                  f"duplicates of a shared signal; {len(final)} finalist(s) remain")
        else:
            print(f"[shared-signal] no shared signals among {len(enriched)} finalist(s)")

    # --- Step 5d: anti-TOI cross-match on finalists (C1) ---
    # Finalists that match a known TOI are NOT discoveries; report them separately
    # but exclude from the discovery finalist list.
    toi_catalog_path = args.toi_catalog or str(ROOT / "data" / "toi_catalog.csv")
    toi_catalog = _load_toi_catalog(toi_catalog_path)
    known_toi_finalists: list[dict] = []  # finalists identified as known TOIs
    if toi_catalog:
        for s in final:
            try:
                tic_int = int(s["tic_id"])
            except (ValueError, TypeError):
                continue
            if tic_int in toi_catalog:
                s["_known_toi"] = toi_catalog[tic_int]
                known_toi_finalists.append(s)
        if known_toi_finalists:
            known_toi_ids = {s["tic_id"] for s in known_toi_finalists}
            final = [s for s in final if s["tic_id"] not in known_toi_ids]
            print(f"[anti-toi] {len(known_toi_finalists)} finalist(s) matched known TOI(s): "
                  + ", ".join(f"TIC {s['tic_id']} = TOI-{s['_known_toi']}"
                              for s in known_toi_finalists))
        else:
            print(f"[anti-toi] 0 finalists matched known TOIs "
                  f"(catalog: {len(toi_catalog)} entries)")
    else:
        print("[anti-toi] TOI catalog empty or missing — cross-match skipped")

    # --- Step 6: dossier ---
    out = [f"# Discovery vetting dossier — {args.sample}", "",
           f"Strong candidates (v6>{args.v6_min}, post-veto): {len(rows)}.", "",
           "| TIC | P(d) | depth(ppm) | SNR | v6 | EB flags | FPP | verdict |",
           "|---|---|---|---|---|---|---|---|"]
    for s in sorted(rows, key=lambda x: -x["exominer_ffi_score"]):
        fpp = fpp_res.get(s["tic_id"], {})
        out.append(f"| {s['tic_id']} | {s['period_days']:.2f} | {s['depth_ppm']:.0f} | "
                   f"{s['snr']:.0f} | {s['exominer_ffi_score']:.2f} | "
                   f"{', '.join(s['_eb_flags']) or 'ok'} | "
                   f"{fpp.get('fpp', '-') if fpp else '-'} | {fpp.get('verdict', '-') if fpp else '-'} |")
    out += ["", f"**Finalists (pass all + FPP<0.5): {len(final)}**: "
            + (", ".join(s["tic_id"] for s in final) or "none")]

    # Shared-signal cross-check (Step 5e). Printed even when it found nothing,
    # so "no duplicates" is a recorded result and not an absent section that
    # could equally mean "the check never ran".
    if args.shared_signal_check:
        out += ["", "## Shared-signal cross-check between finalists", ""]
        if shared_groups:
            out.append(f"{len(shared_groups)} group(s) of finalists were the SAME signal "
                       f"seen on nearby stars; {len(shared_demoted)} demoted. "
                       f"At most one member of each group can be real — the pipeline keeps "
                       f"the strongest by SNR and does NOT claim which is the true source.")
            out.append("")
            out.append("| kept | demoted | P(d) | t0 | sep max (°) |")
            out.append("|---|---|---|---|---|")
            for group in shared_groups:
                keeper = group[0]
                seps = [angular_separation_deg(
                    float(keeper["ra"]), float(keeper["dec"]),
                    float(c["ra"]), float(c["dec"])) for c in group[1:]]
                out.append(
                    f"| {keeper['tic_id']} (snr {keeper.get('snr') or 0:.1f}) "
                    f"| {', '.join(c['tic_id'] for c in group[1:])} "
                    f"| {keeper['period_days']:.4f} | {keeper['t0']:.4f} "
                    f"| {max(seps):.3f} |")
        else:
            out.append(f"No shared signals among the finalists "
                       f"(period tol {args.shared_signal_period_rtol:.1%}, "
                       f"max separation {args.shared_signal_max_sep_deg:g}°).")

    # Double-period EB check summary for the dossier
    if args.double_period_check and dp_verdicts:
        out += ["", "## Double-period EB check (SPOC 2-min)", ""]
        for tic, (verdict, detail) in dp_verdicts.items():
            if verdict == "eb":
                out.append(f"- TIC {tic}: **double-period EB (excluded)** — {detail}")
            elif verdict == "ok":
                out.append(f"- TIC {tic}: ok — {detail}")
            else:
                out.append(f"- TIC {tic}: SPOC unavailable / unknown — {detail}")

    # SPOC repeatability check summary
    if args.spoc_repeat_check and spoc_repeat_verdicts:
        out += ["", "## SPOC repeatability check", ""]
        for tic, (verdict, detail) in spoc_repeat_verdicts.items():
            if verdict == "absent":
                out.append(f"- TIC {tic}: **absent — transit not confirmed in SPOC (excluded)** — {detail}")
            elif verdict == "confirmed":
                out.append(f"- TIC {tic}: confirmed — {detail}")
            else:
                out.append(f"- TIC {tic}: unknown (kept) — {detail}")

    # Orbit-start ramp artifact check summary
    if args.orbit_start_check and orbit_verdicts:
        out += ["", "## Orbit-start ramp artifact check", ""]
        for tic, (verdict, detail) in orbit_verdicts.items():
            if verdict == "orbit_start_artifact":
                out.append(f"- TIC {tic}: **orbit-start ramp artifact (excluded)** — {detail}")
            elif verdict == "ok":
                out.append(f"- TIC {tic}: ok — {detail}")
            elif verdict == "not_applicable":
                out.append(f"- TIC {tic}: not applicable — {detail}")
            else:
                out.append(f"- TIC {tic}: unknown (kept) — {detail}")

    # FFI-only finalist hardening (§3.1 + §3.3) dossier section
    if ffi_hardening_results:
        out += ["", "## FFI-only finalist hardening (has_spoc=0)", ""]
        out.append(
            f"EXONIT_FFI_FINALIST_GATE={'ON (hard-demote active)' if ffi_gate_enabled else 'OFF (report-only)'}. "
            f"SPOC finalists skip this gate entirely."
        )
        out.append("")
        out.append("| TIC | P(d) | folded_depth(ppm) | ctrl_std(ppm) | eff_snr | n_transits | Rp(R⊕) | dur_ratio | §3.3 flags | would-demote |")
        out.append("|---|---|---|---|---|---|---|---|---|---|")
        for tic, rec in ffi_hardening_results.items():
            if rec.get("skipped"):
                out.append(f"| {tic} | — | — | — | — | — | — | — | — | skipped (has_spoc≠0) |")
                continue
            r31 = rec["r31"]
            r33 = rec["r33"]
            fd = f"{r31['folded_depth_ppm']:.0f}" if r31.get("folded_depth_ppm") is not None else "N/A"
            cs = f"{r31['control_std_ppm']:.0f}" if r31.get("control_std_ppm") is not None else "N/A"
            es = f"{r31['effective_snr']:.2f}" if r31.get("effective_snr") is not None else "N/A"
            nt = str(r31.get("n_transits", "N/A"))
            rp = f"{r33['Rp_earth']:.1f}" if r33.get("Rp_earth") is not None else "N/A"
            dr = f"{r33['duration_ratio']:.2f}" if r33.get("duration_ratio") is not None else "N/A"
            flags33 = "; ".join(r33.get("flags", [])) or "ok"
            wd = "**YES**" if tic in ffi_would_demote else "no"
            # Find P from the finalist
            p_val = next((s["period_days"] for s in final if s["tic_id"] == tic), None)
            if p_val is None:
                # finalist was demoted — find in original rows
                p_val = next((s["period_days"] for s in rows if s["tic_id"] == tic), 0.0)
            out.append(
                f"| {tic} | {p_val:.4f} | {fd} | {cs} | {es} | {nt} | {rp} | {dr} | {flags33} | {wd} |"
            )
        if ffi_would_demote:
            out.append("")
            if ffi_gate_enabled:
                out.append(
                    f"**{len(ffi_would_demote)} finalist(s) REMOVED** by FFI gate (passes_3_1=False): "
                    + ", ".join(f"TIC {t}" for t in sorted(ffi_would_demote))
                )
            else:
                out.append(
                    f"**{len(ffi_would_demote)} finalist(s) WOULD BE removed** if gate were ON: "
                    + ", ".join(f"TIC {t}" for t in sorted(ffi_would_demote))
                    + " — set EXONIT_FFI_FINALIST_GATE=1 to activate."
                )

    # Known TOIs re-detected
    if known_toi_finalists:
        out += ["", "## Known TOIs re-detected (excluded from discovery finalists)", ""]
        for s in known_toi_finalists:
            out.append(
                f"- TIC {s['tic_id']}: **KNOWN TOI-{s['_known_toi']}** "
                f"(P={s['period_days']:.4f}d, depth={s['depth_ppm']:.0f}ppm) — "
                "re-detection, not a new discovery"
            )

    # Deep-systematic veto note for the dossier
    if eb_hosts:
        # Determine which eb_hosts had active planet_candidate rows (bounded set, safe to list).
        eb_with_pc = eb_hosts & planet_candidate_tics
        # Anti-TOI override (review round 2): a TIC in eb_with_pc whose
        # protected row survived into `survivors` (eb_hosts_protected_survivors)
        # is NOT actually excluded from finalists — reporting it as excluded
        # would contradict the "[anti-toi] PROTECTED ..." stdout line from
        # Step 1c for the same TIC.
        eb_with_pc_excluded = eb_with_pc - eb_hosts_protected_survivors
        out += ["", "## Deep-systematic veto (TIC-level artefact/EB host)", ""]
        out.append(
            f"Deep-systematic veto: {len(eb_hosts)} host-TIC(s) in sample "
            f"({len(eb_with_pc)} with planet_candidate rows); "
            f"excluded from finalists: "
            + (", ".join(sorted(eb_with_pc_excluded)) if eb_with_pc_excluded else "none")
            + f"  (threshold ≥{args.deep_eclipse_ppm:.0f} ppm on systematic period or "
            f"Rp>{MAX_PLANET_RADIUS_RJUP:.1f} R_Jup)"
        )
        anti_toi_survivors_in_eb = eb_with_pc & eb_hosts_protected_survivors
        if anti_toi_survivors_in_eb:
            out.append(
                "Anti-TOI override: "
                + ", ".join(f"TIC {t}" for t in sorted(anti_toi_survivors_in_eb))
                + " kept a surviving planet_candidate row (confirmed KP/CP TOI) "
                "despite the TIC being a deep-systematic/EB host — NOT excluded "
                "from finalists."
            )
        if args.apply_deep_eclipse_veto:
            out.append("DB reclassification applied (cross_source_name='DEEP_SYSTEMATIC_VETO').")
        else:
            out.append("Report-only (--apply-deep-eclipse-veto not set; DB not modified).")

    out += _cross_sector_dossier_lines(n_cross_sector, cross_sector_hits)

    Path(args.out).write_text("\n".join(out) + "\n")
    print(f"[dossier] escrit {args.out}")
    con.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
