"""Post-BLS refinement of transit duration / t0 and same-period merging.

BLS evaluates durations on a coarse grid of phase-bin widths (0.01*P steps),
so the box it returns can be 2x too long and mis-centred. That mis-centred box
then masks the wrong window and the same transit is re-found with a narrower
box in the next multi-planet slot (TIC 4206066 s6: P=3.1737 d at 3.81 / 1.52 /
0.76 h, true T14 = 1.87 h).

``refine_duration`` fixes the box at fixed period; ``merge_same_period_candidates``
drops any residual same-period duplicates. Both are pure, deterministic,
numpy-only functions (no I/O, no randomness) so the Swift/Rust ports can
replicate them exactly.

Rollout: the worker.py config flags ``duration_refine`` and
``merge_same_period`` default to False (opt-in). Swift (5 M4) and Rust/native
workers do not implement this yet, and enabling it only on Python hosts would
split one run into two candidate populations. The default flips to True in the
iteration that ships Swift/Rust parity plus a scorecard A/B.
"""
from __future__ import annotations

import numpy as np


def refine_duration(time, flux, period, t0, duration, *, cadence=None,
                    dur_min_h=0.5, dur_max_frac=0.15, n_dur=40,
                    t0_frac=1.0, n_t0=25):
    """Refine box duration, t0 and depth at a FIXED period.

    The light curve is folded at ``period``. A box model is evaluated on a
    grid of durations (geometric spacing from ``dur_min_h``/24 d to
    ``min(dur_max_frac*period, max(6 h, 2*input duration))``, plus the input duration) and t0 offsets
    (linear in +-``t0_frac``*(input duration); step <= 10 min, so the point
    count is max(``n_t0``, range/10 min) capped at 241; offset 0 always
    included). A second pass re-centres on the pass-1 t0 with range
    +-0.5*refined duration and keeps the better chi2. Depth is fitted analytically (mean out-of-transit minus mean
    in-transit; only dips, depth > 0, qualify). With a constant noise sigma the
    chi2 reduction of a box is depth^2 * n_in*n_out/(n_in+n_out) / sigma^2, which
    is maximised. At least 3 in-transit points are required; if no grid point
    qualifies the inputs are returned unchanged with chi2_improvement = 0.

    ``period`` and ``snr`` are deliberately NOT changed: detection SNR stays
    BLS's. Only duration, t0 and depth are refined.

    ``cadence`` (days, optional) floors the minimum duration at one cadence.
    NaNs in time/flux are masked.

    Returns dict(duration [d], t0 [d], depth_ppm, chi2_improvement) where
    chi2_improvement is the chi2 reduction of the best box relative to the
    input box (>= 0). When nothing qualifies, depth_ppm is None (the caller
    keeps its own depth).
    """
    unchanged = {"duration": float(duration), "t0": float(t0),
                 "depth_ppm": None, "chi2_improvement": 0.0}
    t = np.asarray(time, dtype=np.float64)
    y = np.asarray(flux, dtype=np.float64)
    ok = np.isfinite(t) & np.isfinite(y)
    period = float(period)
    duration = float(duration)
    if ok.sum() < 10 or not (period > 0) or not (duration > 0):
        return unchanged
    t = t[ok]
    y = y[ok]
    ymed = float(np.median(y))
    y = y - ymed

    # Phase in days relative to t0, in [-P/2, P/2); sort once.
    x = (t - float(t0) + 0.5 * period) % period - 0.5 * period
    order = np.argsort(x, kind="stable")
    x = x[order]
    y = y[order]
    n = len(x)
    csum = np.concatenate(([0.0], np.cumsum(y)))
    total = csum[-1]

    # Robust sigma of out-of-transit flux (outside the input box).
    outside = np.abs(x) > 0.5 * duration
    yo = y[outside] if outside.sum() >= 10 else y
    sigma = 1.4826 * float(np.median(np.abs(yo - np.median(yo))))
    if not (sigma > 0):
        sigma = float(np.std(yo))
    if not (sigma > 0):
        return unchanged

    d_lo = dur_min_h / 24.0
    if cadence is not None and cadence > 0:
        d_lo = max(d_lo, float(cadence))
    # Cap: 6 h, or 2x the input duration so long transits (long-period sweep,
    # 12-24 h) can still be refined, but never above dur_max_frac * period.
    d_hi = min(dur_max_frac * period, max(6.0 / 24.0, 2.0 * duration))
    if d_hi <= d_lo:
        d_hi = d_lo * 1.0001
    durs = np.geomspace(d_lo, d_hi, int(n_dur))
    durs = np.unique(np.concatenate((durs, [duration])))

    def _best_for(d, offs):
        lo = np.searchsorted(x, offs - 0.5 * d, side="left")
        hi = np.searchsorted(x, offs + 0.5 * d, side="left")
        n_in = (hi - lo).astype(np.float64)
        n_out = n - n_in
        s_in = csum[hi] - csum[lo]
        valid = (n_in >= 3) & (n_out >= 3)
        with np.errstate(divide="ignore", invalid="ignore"):
            depth = (total - s_in) / n_out - s_in / n_in
            chi2 = depth ** 2 * n_in * n_out / n / sigma ** 2
        valid &= depth > 0
        chi2 = np.where(valid, chi2, -np.inf)
        return chi2, depth

    # Baseline: the input box.
    c0, _ = _best_for(duration, np.array([0.0]))
    base = float(c0[0]) if np.isfinite(c0[0]) else 0.0

    step = 10.0 / 1440.0  # t0 grid step <= 10 min

    def _offsets(center, half):
        n_o = int(min(241, max(int(n_t0), int(np.ceil(2.0 * half / step)) + 1)))
        offs = center + np.linspace(-half, half, n_o)
        if not np.any(offs == 0.0):
            offs = np.concatenate((offs, [0.0]))
        return offs

    def _search(center, half):
        best = (-np.inf, None, None, None)  # chi2, dur, off, depth
        for d in durs:
            offs = _offsets(center, half)
            chi2, depth = _best_for(d, offs)
            k = int(np.argmax(chi2))
            if chi2[k] > best[0]:
                best = (float(chi2[k]), float(d), float(offs[k]), float(depth[k]))
        return best

    # Pass 1: +-t0_frac * input duration around the input t0.
    best = _search(0.0, t0_frac * duration)
    if best[1] is None or not np.isfinite(best[0]):
        return unchanged
    # Pass 2: re-centre on the pass-1 t0, +-0.5 * refined duration; keep the
    # better chi2 (pass 1 wins ties).
    best2 = _search(best[2], 0.5 * best[1])
    if best2[1] is not None and best2[0] > best[0]:
        best = best2
    return {"duration": best[1], "t0": float(t0) + best[2],
            "depth_ppm": best[3] * 1e6,
            "chi2_improvement": max(best[0] - base, 0.0)}


def merge_same_period_candidates(candidates, rel_tol=2e-3):
    """Collapse candidates sharing a period (ratio ~ 1) to the max-SNR one.

    Candidates are grouped greedily in the given order: a candidate joins the
    first group whose representative (first member) period agrees within
    ``rel_tol`` (relative). Per group the max-``snr`` member survives (ties:
    lowest ``candidate_num``). Survivors keep their original relative order and
    are renumbered ``candidate_num`` 1..N. Harmonics (ratio != 1) are not
    merged. Input dicts are not mutated; survivors are shallow copies.
    """
    groups = []
    for c in candidates:
        p = float(c["period_days"])
        for g in groups:
            p0 = float(g[0]["period_days"])
            if abs(p - p0) / max(p0, 1e-12) <= rel_tol:
                g.append(c)
                break
        else:
            groups.append([c])
    keep = set()
    for g in groups:
        winner = min(g, key=lambda c: (-float(c.get("snr", 0.0)),
                                       c.get("candidate_num", 0)))
        keep.add(id(winner))
    out = []
    for c in candidates:
        if id(c) in keep:
            c2 = dict(c)
            c2["candidate_num"] = len(out) + 1
            out.append(c2)
    return out


def apply_duration_refine(result, time, flux, logger=None, cadence=None):
    """Refine a BLS ``result`` dict in place (duration, t0, depth_ppm).

    Period and snr are left untouched. Logs one INFO line when the duration
    changes by more than 10 %. Returns ``result``.
    """
    ref = refine_duration(time, flux, result["period"], result["t0"],
                          result["duration"], cadence=cadence)
    if ref["depth_ppm"] is None:
        return result
    old = float(result["duration"])
    if logger is not None and abs(ref["duration"] - old) / max(old, 1e-12) > 0.10:
        logger.info(
            f"Duration refine: P={result['period']:.4f}d "
            f"dur {old * 24:.2f}h -> {ref['duration'] * 24:.2f}h, "
            f"t0 {result['t0']:.4f} -> {ref['t0']:.4f} "
            f"(dchi2={ref['chi2_improvement']:.1f})")
    result["duration"] = ref["duration"]
    result["t0"] = ref["t0"]
    result["depth_ppm"] = ref["depth_ppm"]
    return result
