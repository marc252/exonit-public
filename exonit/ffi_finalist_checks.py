"""Pure helpers for FFI-only finalist gate hardening (spec §3.1 + §3.3).

§3.1  Control-phase / folded-depth test
    Measures the transit depth by folding the *detrended* LC at (P, t0), then
    compares against "depths" measured at several off-transit control phases.
    Demote if the in-transit depth is NOT clearly above the control-phase
    spread.  Robust to detrending choices because every control phase is
    detrended identically.

§3.3  Stellar-consistency check (catalogue params, NOT LoRA prose)
    Infers Rp and expected transit duration from catalogue R*, Teff, M*.
    Flags implausible radius or implied high-b duration.  INFORMATIVE ONLY —
    never demotes on its own.

Conservative fallback: any missing input returns a "unknown / kept" result.
No randomness.  All maths are deterministic.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np

from exonit.transit_math import fold_and_measure_depth

# ---------------------------------------------------------------------------
# Physical constants (CGS-free)
# ---------------------------------------------------------------------------
# Solar radius in same units as stellar_radius column (R_sun)
_RSUN_M = 6.957e8          # m
_REARTH_M = 6.371e6        # m
_MSUN_KG = 1.989e30        # kg
_G = 6.674e-11             # SI
_DAY_S = 86400.0


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _median_detrend(time: np.ndarray, flux: np.ndarray,
                    window_days: float) -> np.ndarray:
    """Subtract a running-median trend, returning detrended flux.

    The window is a *sliding* median over *window_days* centred on each
    cadence.  Transit dips are preserved because the window is wider than
    the transit (caller guarantees window ≥ 3 × dur).

    Gaps > 0.5 d are treated as segment boundaries; the normalisation is done
    segment-by-segment so a jump in flux level across a gap does not create a
    spurious trend.  The returned array has the same length as *flux*.
    """
    result = np.zeros_like(flux, dtype=float)
    # Identify segment boundaries (gaps > 0.5 d)
    dt = np.diff(time)
    gap_mask = np.concatenate([[True], dt > 0.5])
    seg_ids = np.cumsum(gap_mask)
    for seg in np.unique(seg_ids):
        idx = np.where(seg_ids == seg)[0]
        if len(idx) < 3:
            result[idx] = flux[idx]
            continue
        t_s = time[idx]
        f_s = flux[idx]
        trend = np.empty_like(f_s)
        hw = window_days / 2.0
        for i, tc in enumerate(t_s):
            mask = np.abs(t_s - tc) <= hw
            trend[i] = np.median(f_s[mask])
        # Normalise: detrended = flux / trend (trend ≈ 1 for normalised input)
        # Guard against zero/negative trend
        good = trend > 0
        result[idx[good]] = f_s[good] / trend[good]
        result[idx[~good]] = f_s[~good]
    return result


def _fold_and_measure(time: np.ndarray, flux: np.ndarray,
                      P: float, t0: float,
                      phase_centre: float,
                      half_width: float) -> Optional[float]:
    """Measure 'depth' (1 - median_in / median_oot) at a given fold phase.

    *phase_centre* is in units of the period (−0.5 … +0.5).
    *half_width* is in the same phase units.

    Returns depth in ppm, or None if fewer than 3 in-window points.
    A positive depth means flux goes down (transit-like).

    D6 (docs/reviews/2026-07-02-opus48-cleanup-review.md): now a thin wrapper
    over the shared exonit.transit_math engine; behaviour is bit-identical to
    the pre-refactor inline version (see tests/test_transit_math_golden.py).
    OOT baseline: everything outside a one-sided exclusion around phase 0
    (NOT around phase_centre — this must stay clear of the real transit even
    when *phase_centre* probes an off-transit "control" phase elsewhere).
    """
    oot_excl_hw = max(half_width * 3, 0.02)
    res = fold_and_measure_depth(
        time, flux, P, t0,
        centre_phase=phase_centre, half_width=half_width,
        phase_mode="recentered", window_mode="direct",
        oot_mode="exclude_centre", oot_excl_half_width=oot_excl_hw,
        depth_formula="relative", error_mode=None,
        min_in=3, min_oot=5, guard_before_compute=True,
    )
    return res.depth_ppm


def _robust_std(arr: np.ndarray) -> float:
    """MAD-based robust std (σ ≈ 1.4826 × MAD)."""
    if len(arr) == 0:
        return 0.0
    return 1.4826 * float(np.median(np.abs(arr - np.median(arr))))


# ---------------------------------------------------------------------------
# §3.1  Control-phase / folded-depth test (the HARD gate)
# ---------------------------------------------------------------------------

def folded_depth_control_test(
    time: np.ndarray,
    flux: np.ndarray,
    P: float,
    t0: float,
    dur_days: float,
    control_phases: tuple = (0.15, 0.25, 0.35, -0.15, -0.25, -0.35),
    n_sigma: float = 2.0,
) -> dict:
    """Control-phase folded-depth test (§3.1).

    Parameters
    ----------
    time, flux : raw (or pre-cleaned) arrays from load_lc.  Flux must be
                 median-normalised (values near 1.0).
    P          : orbital period in days
    t0         : transit epoch (BTJD)
    dur_days   : transit duration in days
    control_phases : off-transit phases to probe (phase units, centred on 0).
                 Default is the fixed 6-tuple (0.15, 0.25, 0.35, -0.15, -0.25,
                 -0.35), calibrated on real TGLC sector-1 LCs (see calibration
                 note at passes_3_1 below).  Caller may pass a different tuple
                 for testing or reproducibility; the tuple is used as-is.
    n_sigma    : detection threshold (default 2.0; calibrated on real TGLC
                 sector-1 LCs — see note at passes_3_1 below)

    Returns
    -------
    dict with keys:
      folded_depth_ppm   — measured depth at phase 0 (transit) in ppm
      control_std_ppm    — robust std of control-phase depths
      control_depths_ppm — list of depths at each control phase
      n_transits         — number of complete transit windows in the data
      empty_transits     — number of transit windows with no in-window flux dip
      per_transit_depths_ppm — list of per-transit depths
      per_transit_snr    — list of per-transit SNR (depth / mad_oot_ppm per transit)
      effective_snr      — folded_depth_ppm / control_std_ppm (0 if denom=0)
      passes_3_1         — True if depth > n_sigma*control_std AND depth > max(control)
      error              — None on success, string on failure (conservative: keep)
    """
    result: dict = {
        "folded_depth_ppm": None,
        "control_std_ppm": None,
        "control_depths_ppm": [],
        "n_transits": 0,
        "empty_transits": 0,
        "per_transit_depths_ppm": [],
        "per_transit_snr": [],
        "effective_snr": None,
        "passes_3_1": True,   # conservative default: keep on any error
        "error": None,
    }

    # --- Input validation ---
    if time is None or flux is None or len(time) < 10:
        result["error"] = "insufficient data"
        return result
    if P is None or P <= 0 or not math.isfinite(P):
        result["error"] = f"invalid period {P}"
        return result
    if dur_days is None or dur_days <= 0 or not math.isfinite(dur_days):
        result["error"] = f"invalid duration {dur_days}"
        return result
    if t0 is None or not math.isfinite(t0):
        result["error"] = f"invalid t0 {t0}"
        return result

    # Ensure arrays are finite and sorted
    finite = np.isfinite(time) & np.isfinite(flux)
    time = time[finite]
    flux = flux[finite]
    if len(time) < 10:
        result["error"] = "insufficient finite points"
        return result
    order = np.argsort(time)
    time = time[order]
    flux = flux[order]

    # --- Detrend ---
    # Window: max(0.5 d, 3 * duration_days) so the transit is preserved.
    window_days = max(0.5, 3.0 * dur_days)
    try:
        f_det = _median_detrend(time, flux, window_days)
    except Exception as exc:
        result["error"] = f"detrend failed: {exc}"
        return result

    # Phase half-width for in-transit window
    hw_phase = (dur_days / P) * 0.6   # 20% wider than the nominal duration

    # Use the provided (or default) fixed control phases directly
    _phases = control_phases

    # --- In-transit depth ---
    depth_transit = _fold_and_measure(time, f_det, P, t0, 0.0, hw_phase)
    if depth_transit is None:
        result["error"] = "no in-transit points after detrend"
        return result
    result["folded_depth_ppm"] = depth_transit

    # --- Control-phase depths ---
    # Each control phase requires ≥3 in-window points (_fold_and_measure returns
    # None for phases that fall in cadence gaps), so phases in data gaps are
    # automatically excluded.
    ctrl_depths: list[float] = []
    for cp in _phases:
        d = _fold_and_measure(time, f_det, P, t0, cp, hw_phase)
        if d is not None:
            ctrl_depths.append(d)
    result["control_depths_ppm"] = ctrl_depths

    if len(ctrl_depths) < 2:
        # Too few usable control phases (gaps in cadence, very wide transit,
        # or caller provided too few phases) — cannot reliably estimate noise
        # floor.  Conservative keep.
        result["error"] = (
            f"insufficient control sampling: {len(ctrl_depths)} usable phases "
            f"(need ≥2); conservative keep"
        )
        result["effective_snr"] = None
        return result

    ctrl_arr = np.array(ctrl_depths)
    ctrl_std = max(_robust_std(ctrl_arr), 1.0)   # floor at 1 ppm to avoid /0
    result["control_std_ppm"] = float(ctrl_std)

    eff_snr = depth_transit / ctrl_std
    result["effective_snr"] = float(eff_snr)

    # --- Per-transit depths ---
    # Identify transit windows (fold at P, find epochs within the data span)
    t_min, t_max = float(time[0]), float(time[-1])
    # First transit epoch within data: find n such that t0 + n*P >= t_min
    n_first = math.ceil((t_min - t0) / P)
    n_last = math.floor((t_max - t0) / P)
    n_transits = 0
    empty_transits = 0
    per_depths: list[float] = []
    per_snr: list[float] = []

    # OOT MAD for per-transit SNR (global, cheaper than per-transit)
    phase_all = ((time - t0) % P) / P
    phase_all[phase_all > 0.5] -= 1.0
    oot_global = np.abs(phase_all) > (dur_days / P * 1.5)
    if oot_global.sum() > 5:
        oot_mad_ppm = float(
            1.4826 * np.median(np.abs(f_det[oot_global] - np.median(f_det[oot_global])))
        ) * 1e6
        oot_mad_ppm = max(oot_mad_ppm, 1.0)
    else:
        oot_mad_ppm = 1000.0  # fallback

    for n in range(n_first, n_last + 1):
        tc = t0 + n * P
        hw_t = dur_days * 0.6
        in_t = (time >= tc - hw_t) & (time <= tc + hw_t)
        n_transits += 1
        if in_t.sum() < 2:
            empty_transits += 1
            continue
        # Local OOT: within ±5 * dur but outside the transit
        local_hw = dur_days * 5
        local_oot = (time >= tc - local_hw) & (time <= tc + local_hw) & ~in_t
        if local_oot.sum() < 3:
            local_oot = oot_global  # fallback to global
        med_in = float(np.median(f_det[in_t]))
        med_oot_local = float(np.median(f_det[local_oot])) if local_oot.sum() >= 3 else 1.0
        if med_oot_local == 0:
            med_oot_local = 1.0
        d = (1.0 - med_in / med_oot_local) * 1_000_000.0
        per_depths.append(d)
        snr_t = d / oot_mad_ppm
        per_snr.append(snr_t)

    result["n_transits"] = n_transits
    result["empty_transits"] = empty_transits
    result["per_transit_depths_ppm"] = per_depths
    result["per_transit_snr"] = per_snr

    # --- passes_3_1 criterion ---
    # True (keep) iff:
    #   1) folded_depth > n_sigma * control_std  (above noise floor), AND
    #   2) folded_depth > max(control_depths)    (no single control is as deep)
    #
    # CALIBRATION NOTE (2026-06-18, 6-phase design, real TGLC sector-1 LCs):
    #   n_sigma=2.0, control_phases=(0.15, 0.25, 0.35, -0.15, -0.25, -0.35).
    #
    #   Measured eff_snr_6 on 6 real candidates (the only LCs available on this
    #   machine at calibration time):
    #     TIC 469851877  (P=2.86 d):  eff_snr_6 = 33.88  → passes ✓
    #     TIC 403289867  (P=2.32 d):  eff_snr_6 = 40.22  → passes ✓
    #     TIC 263003342  (P=3.53 d):  eff_snr_6 = 14.89  → passes ✓
    #     TIC 265591657  (P=2.27 d):  eff_snr_6 =  2.92  → passes ✓
    #     TIC 280832898  (P=1.00 d):  eff_snr_6 =  2.73  → passes ✓  (weakest survivor)
    #     TIC 24693554   (P=9.57 d):  eff_snr_6 =  1.10  → FAILS  ✓  (target FP)
    #
    #   n_sigma=2.0 design window:
    #     - Weakest survivor cushion: 2.73 − 2.0 = +0.73σ
    #     - FP (24693554) cushion:    2.0 − 1.10 = +0.90σ below floor
    #     - Clean separation window:  1.10 – 2.73
    #
    #   12-phase design was evaluated on real LCs and REJECTED: eff_snr_12 for
    #   TIC 24693554 = 4.57 (floor inert vs this FP class — rejection rests
    #   entirely on the max_ctrl absolute test with no σ-redundancy).
    #
    #   IMPORTANT: calibration rests on ~6 real data points.  eff_snr_6 MUST be
    #   re-checked on the actual run finalists before enabling
    #   EXONIT_FFI_FINALIST_GATE=1.  Do NOT enable the flag without that step.
    max_ctrl = float(np.max(ctrl_arr)) if len(ctrl_arr) else 0.0
    passes = (depth_transit > n_sigma * ctrl_std) and (depth_transit > max_ctrl)
    result["passes_3_1"] = bool(passes)
    return result


# ---------------------------------------------------------------------------
# §3.3  Stellar-consistency check (INFORMATIVE ONLY — never demotes)
# ---------------------------------------------------------------------------

def stellar_consistency_check(
    folded_depth_ppm: float,
    R_star: Optional[float],
    Teff: Optional[float],
    M_star: Optional[float],
    P: float,
    dur_days: float,
) -> dict:
    """Stellar-consistency annotation (§3.3) — informative, never demotes.

    Parameters
    ----------
    folded_depth_ppm : in-transit depth measured by folded_depth_control_test
    R_star  : stellar radius in R_sun (from catalogue, NOT LoRA prose)
    Teff    : effective temperature in K
    M_star  : stellar mass in M_sun (may be None → skip duration check)
    P       : period in days
    dur_days: observed transit duration in days

    Returns
    -------
    dict with keys:
      Rp_earth           — implied planet radius in R_earth (None if R_star missing)
      expected_dur_h     — expected central-transit duration in hours (None if M* missing)
      observed_dur_h     — observed duration in hours
      duration_ratio     — observed / expected (None if expected unavailable)
      implied_b          — implied impact parameter (None if unconstrained)
      flags              — list of informative flag strings (empty if all OK)
      error              — None or string describing why checks were skipped
    """
    # Guard dur_days early so computed observed_dur_h is not misleading.
    # A zero or negative duration would otherwise produce obs_dur_h=0,
    # ratio=0, b²=1, b=1.0 and trigger a spurious b>0.9 flag from a corrupt
    # DB row.  Treat the same as missing data: error + conservative skip.
    _dur_valid = (
        dur_days is not None
        and math.isfinite(dur_days)
        and dur_days > 0
    )
    result: dict = {
        "Rp_earth": None,
        "expected_dur_h": None,
        "observed_dur_h": dur_days * 24.0 if _dur_valid else None,
        "duration_ratio": None,
        "implied_b": None,
        "flags": [],
        "error": None,
    }

    if not _dur_valid:
        result["error"] = f"invalid dur_days {dur_days!r} — duration check skipped"
        return result

    if folded_depth_ppm is None or not math.isfinite(folded_depth_ppm) or folded_depth_ppm <= 0:
        result["error"] = "invalid folded_depth_ppm"
        return result

    # --- Rp from transit depth + R* ---
    if R_star is not None and math.isfinite(R_star) and R_star > 0:
        # depth = (Rp/R*)²  →  Rp = R* * sqrt(depth/1e6)
        rp_rsun = R_star * math.sqrt(folded_depth_ppm / 1_000_000.0)
        rp_earth = rp_rsun * (_RSUN_M / _REARTH_M)
        result["Rp_earth"] = rp_earth
        # Flag implausibly large radius (> 2 R_Jup ≈ 22.5 R_earth) — likely EB
        if rp_earth > 22.5:
            result["flags"].append(
                f"implausible radius: Rp={rp_earth:.1f} R_earth (>2 R_Jup — likely EB)"
            )
        # Flag very large but sub-EB (10–22.5 R_earth)
        elif rp_earth > 10.0:
            result["flags"].append(
                f"large radius: Rp={rp_earth:.1f} R_earth (planet-to-giant boundary)"
            )
    else:
        result["error"] = "R_star missing — Rp and duration check skipped"
        return result

    # --- Expected central-transit duration from Kepler's 3rd law ---
    # T_dur(central) = (P / π) * arcsin(R*/a)
    # a/R* from Kepler: a = (G M_star / (4π²) * P²)^(1/3) in metres
    if M_star is None or not math.isfinite(M_star) or M_star <= 0:
        result["error"] = "M_star missing — duration check skipped"
        return result

    try:
        P_s = P * _DAY_S
        R_m = R_star * _RSUN_M
        M_kg = M_star * _MSUN_KG
        a_m = ((_G * M_kg / (4 * math.pi ** 2)) * P_s ** 2) ** (1.0 / 3.0)
        r_over_a = R_m / a_m
        if r_over_a >= 1.0:
            # Star fills the orbit (degenerate) — skip
            result["error"] = "stellar radius ≥ semi-major axis (degenerate)"
            return result
        # Central transit duration (b=0): T = (P/π) * arcsin(R*/a)
        t_central_s = (P_s / math.pi) * math.asin(r_over_a)
        t_central_h = t_central_s / 3600.0
        result["expected_dur_h"] = t_central_h

        obs_dur_h = dur_days * 24.0
        if t_central_h > 0:
            ratio = obs_dur_h / t_central_h
            result["duration_ratio"] = ratio

            # Infer impact parameter from duration ratio:
            # T_obs = T_central * sqrt(1 - b²)  →  b² = 1 - ratio²
            b2 = 1.0 - ratio ** 2
            if b2 < 0:
                # Observed duration longer than central — can happen with grazing
                # geometry in the *other* direction or starspot; informative flag only
                result["flags"].append(
                    f"observed duration ({obs_dur_h:.2f} h) > central ({t_central_h:.2f} h) "
                    f"ratio={ratio:.2f} — possible blend or alias"
                )
                result["implied_b"] = None
            else:
                b = math.sqrt(b2)
                result["implied_b"] = b
                if b > 0.9:
                    result["flags"].append(
                        f"duration too short (b={b:.2f} > 0.9): "
                        f"obs={obs_dur_h:.2f} h vs central={t_central_h:.2f} h — "
                        f"possible blend/grazing"
                    )
    except Exception as exc:
        result["error"] = f"duration check failed: {exc}"

    return result


# ---------------------------------------------------------------------------
# §3.4  Shared-signal cross-check between finalists of the SAME lot
#
# Every other gate in this pipeline judges a candidate on its own. That leaves
# one failure mode completely uncovered, and it is not hypothetical: in lot
# discovery_s0014_lot20260722T073004, FOUR of the eleven finalists were one
# signal counted four times —
#
#   TIC 202948052  P=10.3437  t0=1693.8075  1714 ppm  Tmag 8.50
#   TIC 416877345  P=10.3437  t0=1693.8075  2055 ppm  Tmag 9.28
#   TIC 202806474  P=10.3468  t0=1693.8106  2351 ppm  Tmag 8.70
#   TIC 202944495  P=10.3609  t0=1693.7727  2057 ppm  Tmag 10.05
#
# — identical period AND epoch to four decimals, all four within 0.36° of sky
# (two of them 2.3 arcmin apart). Four neighbouring bright stars dimming at the
# same instant with the same cadence are not four planets: either a systematic
# shared across a CCD region, or one real eclipsing source bleeding into the
# neighbouring TGLC apertures. At most one of them can be real.
#
# This check does NOT decide which. It collapses each such group to its single
# strongest member and hands the rest back to the caller with the group spelled
# out, so the dossier records what was collapsed and a human can look.
# ---------------------------------------------------------------------------

def angular_separation_deg(ra1, dec1, ra2, dec2):
    """Great-circle separation in degrees. Haversine, so it stays accurate for
    the small separations that matter here (a naive cos-formula loses precision
    below ~1 arcmin, which is exactly the regime of aperture bleed)."""
    lat1, lat2 = math.radians(dec1), math.radians(dec2)
    dlat = lat2 - lat1
    dlon = math.radians(ra2 - ra1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2)
    return math.degrees(2 * math.asin(min(1.0, math.sqrt(a))))


def _epochs_agree(t0_a, t0_b, period, tol_days):
    """True if two epochs describe the same transit train.

    The two t0 values can be many cycles apart, so the comparison is done on
    the phase difference folded into [0, P), taking the shorter way round.
    """
    if period <= 0:
        return False
    phase = math.fmod(abs(t0_a - t0_b), period)
    return min(phase, period - phase) <= tol_days


def shared_signal_groups(candidates, *, period_rtol=0.01,
                         epoch_tol_frac=0.5, max_sep_deg=1.0,
                         default_duration_days=0.15):
    """Group finalists that are one signal seen on several nearby stars.

    *candidates* — dicts with ``tic_id``, ``period_days``, ``t0``, ``ra``,
    ``dec``, and optionally ``duration_days`` and ``snr``.

    Two candidates are the same signal when ALL THREE hold:
      * periods agree to *period_rtol* (relative),
      * epochs agree modulo the period to *epoch_tol_frac* × duration,
      * they are within *max_sep_deg* of each other on the sky.

    Grouping is transitive (A~B and B~C ⇒ one group of three), which is what
    a shared systematic across a CCD region actually looks like.

    Returns a list of groups with ≥2 members, each sorted strongest-first by
    ``snr`` (ties broken by depth, then TIC, so the result is deterministic and
    does not depend on input order). Singletons are not returned — nothing to
    say about them.

    Conservative by construction: a candidate missing ra/dec/period/t0 is never
    grouped, so incomplete data can only ever KEEP a finalist, never drop one.
    """
    usable = [c for c in candidates
              if c.get("ra") is not None and c.get("dec") is not None
              and c.get("period_days") and c.get("t0") is not None]

    parent = {c["tic_id"]: c["tic_id"] for c in usable}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra_, rb = find(a), find(b)
        if ra_ != rb:
            parent[rb] = ra_

    for i, a in enumerate(usable):
        for b in usable[i + 1:]:
            pa, pb = float(a["period_days"]), float(b["period_days"])
            mean_p = 0.5 * (pa + pb)
            if mean_p <= 0 or abs(pa - pb) / mean_p > period_rtol:
                continue
            dur = (a.get("duration_days") or b.get("duration_days")
                   or default_duration_days)
            if not _epochs_agree(float(a["t0"]), float(b["t0"]), mean_p,
                                 epoch_tol_frac * float(dur)):
                continue
            if angular_separation_deg(float(a["ra"]), float(a["dec"]),
                                      float(b["ra"]), float(b["dec"])) > max_sep_deg:
                continue
            union(a["tic_id"], b["tic_id"])

    by_root = {}
    for c in usable:
        by_root.setdefault(find(c["tic_id"]), []).append(c)

    def strength(c):
        return (-(c.get("snr") or 0.0), -(c.get("depth_ppm") or 0.0), str(c["tic_id"]))

    return [sorted(g, key=strength) for g in by_root.values() if len(g) > 1]
