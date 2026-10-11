"""Cross-source photometric confirmation (Codex #3).

A genuine transit should show up in MORE than one photometry source. Two model
scores agreeing (CNN + ExoMiner-FFI) is not independent evidence about the
*signal*; checking the same ephemeris in a different photometry source is.

v1 confirms SPOC-detected candidates against FFI (QLP/TGLC), which the server
already caches and serves via /api/ffi (workers never hit MAST — the server is
the only proxy, per project rule). For each candidate we do a FIXED-ephemeris
box fit (no blind search) at the candidate's (period, t0, duration) on the
alternate-source light curve and emit features. These are FEATURES for the
verdict/ranking, never a deterministic gate (cf. the no-hard-FPP-gate rule).

Mirrors worker/phase3_runner.py: mutates candidates in place, never raises,
fetch/detrend deps are injectable for testing.
"""
from __future__ import annotations

import logging
from typing import Callable, Optional

import numpy as np

# Dual-shim import: this module is loaded both as `worker.cross_source`
# (package context, e.g. worker.py's own dual-shim at its call site) and as
# flat `cross_source` (standalone worker deploy) — mirrors _import_fetch_fn /
# _import_detrend_fn below, which use the same pattern for the same reason.
try:
    from worker.phase_fold import phase_fold
except ImportError:
    from phase_fold import phase_fold

# Box-fit SNR in the alternate source above which we call the signal "seen".
XS_MIN_SNR = 5.0
# Plausible depth-ratio band (alt/primary) for agreement: same-ish depth.
XS_DEPTH_RATIO_LO = 0.3
XS_DEPTH_RATIO_HI = 3.0
# Minimum in-transit / out-of-transit points for a meaningful fit.
XS_MIN_IN_TRANSIT = 3
XS_MIN_OOT = 20


def fixed_period_box_fit(time, flux, period, t0, duration_days):
    """Box depth + SNR at a FIXED ephemeris (no period search).

    Phase-folds on (period, t0); compares in-transit to out-of-transit flux.
    Assumes `flux` is ~normalized to 1 (detrended). Returns a dict with
    depth_ppm (>0 for a dip), snr, n_in_transit, and is_dip.
    """
    t = np.asarray(time, dtype=np.float64)
    f = np.asarray(flux, dtype=np.float64)
    good = np.isfinite(t) & np.isfinite(f)
    t, f = t[good], f[good]
    null = {"depth_ppm": 0.0, "snr": 0.0, "n_in_transit": 0, "is_dip": False}
    if t.size < (XS_MIN_IN_TRANSIT + XS_MIN_OOT) or period <= 0 \
            or duration_days <= 0:
        return null

    # See docs/reviews/2026-07-02-opus48-cleanup-review.md item D5 and
    # tests/worker/test_phase_fold_golden.py for the bit-identical proof.
    phase = phase_fold(t, t0, period, center="shift")   # [-0.5, 0.5), 0 = center
    half = (duration_days / period) / 2.0
    in_tr = np.abs(phase) <= half
    oot = np.abs(phase) >= 2.0 * half                  # clear of the transit
    n_in = int(in_tr.sum())
    if n_in < XS_MIN_IN_TRANSIT or int(oot.sum()) < XS_MIN_OOT:
        return null

    f_oot = float(np.median(f[oot]))
    f_in = float(np.median(f[in_tr]))
    depth = f_oot - f_in                               # >0 for a dip
    # Robust out-of-transit scatter (MAD -> sigma).
    scatter = 1.4826 * float(np.median(np.abs(f[oot] - f_oot)))
    snr = (depth / (scatter / np.sqrt(n_in))) if scatter > 0 else 0.0
    return {
        "depth_ppm": depth * 1e6,
        "snr": float(snr),
        "n_in_transit": n_in,
        "is_dip": depth > 0,
    }


def _classify_agreement(xs_snr, depth_ratio, is_dip):
    """Soft agreement flag: signal seen, a dip, and a compatible depth."""
    if not is_dip or xs_snr < XS_MIN_SNR:
        return False
    if depth_ratio is None:
        return True  # SNR already checked above
    return XS_DEPTH_RATIO_LO <= depth_ratio <= XS_DEPTH_RATIO_HI


def _import_fetch_fn() -> Callable:
    try:
        from worker.server_ffi_client import fetch_ffi
    except ImportError:
        from server_ffi_client import fetch_ffi
    return fetch_ffi


def _import_detrend_fn() -> Callable:
    try:
        from worker.wotan_detrend import detrend_with_wotan
    except ImportError:
        from wotan_detrend import detrend_with_wotan
    return detrend_with_wotan


def run_cross_source_check(
    *, tic_id: str, candidates: list, data_source: str, config: dict,
    logger: logging.Logger,
    fetch_fn: Optional[Callable] = None,
    detrend_fn: Optional[Callable] = None,
) -> None:
    """Confirm SPOC candidates against the FFI light curve. Mutates in place.

    Writes per candidate: cross_source_name, cross_source_snr,
    cross_source_depth_ppm, cross_source_depth_ratio, cross_source_agreement.
    Never raises (logs and returns on any failure).
    """
    if not candidates or not tic_id:
        return
    # v1: only SPOC candidates get confirmed against FFI. FFI-primary candidates
    # already live in the FFI domain — cross-FFI-source is future work.
    if data_source == "ffi":
        return

    try:
        if fetch_fn is None:
            fetch_fn = _import_fetch_fn()
        res = fetch_fn(
            config["server_url"], str(tic_id),
            token=config.get("worker_token")
                  or config.get("token", "exonit-default-token"),
            timeout=config.get("ffi_fetch_timeout", 30.0),
        )
        if res is None:
            logger.info("cross-source skipped TIC %s — no FFI available", tic_id)
            return
        ffi_t, ffi_f, _q = res
        if ffi_t is None or ffi_f is None or len(ffi_t) < 100:
            return

        # Detrend the alternate source the same way the primary was processed,
        # then normalize to ~1 so depths are comparable in ppm.
        if detrend_fn is None:
            detrend_fn = _import_detrend_fn()
        ffi_t = np.asarray(ffi_t, dtype=np.float64)
        ffi_f = np.asarray(ffi_f, dtype=np.float64)
        flat = detrend_fn(ffi_t, ffi_f)
        m = np.isfinite(ffi_t) & np.isfinite(flat)
        ft, ff = ffi_t[m], flat[m]

        n_conf = 0
        for c in candidates:
            p = c.get("period_days")
            t0 = c.get("t0")
            dur = c.get("duration_days")
            if not p or t0 is None or not dur:
                continue
            fit = fixed_period_box_fit(ft, ff, float(p), float(t0), float(dur))
            prim_depth = c.get("depth_ppm") or 0.0
            ratio = (fit["depth_ppm"] / prim_depth
                     if prim_depth > 0 and fit["depth_ppm"] > 0 else None)
            agree = _classify_agreement(fit["snr"], ratio, fit["is_dip"])
            c["cross_source_name"] = "ffi"
            c["cross_source_snr"] = round(fit["snr"], 2)
            c["cross_source_depth_ppm"] = round(fit["depth_ppm"], 1)
            c["cross_source_depth_ratio"] = (round(ratio, 3)
                                             if ratio is not None else None)
            c["cross_source_agreement"] = bool(agree)
            if agree:
                n_conf += 1
        logger.info("  cross-source TIC %s: %d/%d candidates confirmed in FFI",
                    tic_id, n_conf, len(candidates))
    except Exception as exc:
        logger.warning("cross-source check failed for TIC %s: %s", tic_id, exc)
