"""Shared transit fold-and-measure-depth engine.

D6 (docs/reviews/2026-07-02-opus48-cleanup-review.md): before this module,
"fold a light curve at a period and measure an in-window depth vs an
out-of-transit/baseline level" was implemented independently 4x, with
genuinely DIFFERENT phase representations, in-window tests, and OOT/baseline
window conventions:

  1. scripts/discovery_postprocess.py::secondary_depth
     Raw (unrecentered) phase in [0, 1). Signal window is a fixed direct
     window at phase 0.5 (the secondary-eclipse phase). Baseline is the union
     of two direct windows at phase 0.25 and 0.75, using the SAME half-width
     as the signal window. Depth = median(baseline) - median(signal), ppm.
     No error estimate.

  2. scripts/discovery_postprocess.py::double_period_eb_check's inline
     ``_measure_depth`` closure. Raw phase (of the *2xP* fold) in [0, 1).
     In-window test is a CIRCULAR wrap around an arbitrary centre_phase (so
     it works correctly at centre_phase=0.0, which sits at the phase-domain
     edge, without an edge artefact). Baseline is two FIXED literal bands,
     (0.2, 0.3) and (0.7, 0.8), independent of centre_phase and of the
     transit duration. Depth = median(baseline) - median(signal), ppm, plus
     a MAD-based error on the in-window points.

  3. scripts/discovery_postprocess.py::spoc_repeat_check's inline fold block.
     Phase recentred to (-0.5, 0.5]. Signal window is direct, centred on 0.
     Baseline is an ANNULUS around phase 0, [0.75, 3.0] x (dur/period) on
     each side -- i.e. scaled to the transit duration, not a fixed band.
     Depth = median(baseline) - median(signal), ppm, plus a std-based error
     on the baseline points.

  4. exonit/ffi_finalist_checks.py::_fold_and_measure. Phase recentred to
     (-0.5, 0.5]. Signal window is direct, centred on an arbitrary
     centre_phase (used to probe several off-transit "control" phases, not
     just the transit itself). Baseline is everything OUTSIDE a one-sided
     exclusion half-width around phase 0 (not phase centre_phase!) -- i.e. a
     single EXCLUSION, not a two-sided annulus, because it must stay clear of
     the real transit even when measuring a "control" depth elsewhere.
     Depth = (1 - median(signal)/median(baseline)) x 1e6, a RELATIVE ratio,
     not the other three's absolute median difference. No error estimate.

None of these are "the same algorithm with different constants": the window
*topology* differs (two direct windows / fixed bands / annulus / one-sided
exclusion), the phase representation differs (raw [0,1) vs recentred
(-0.5,0.5]), the in-window test differs (direct vs circular-wrap), and even
the depth formula differs (absolute difference vs relative ratio). Collapsing
them into one "more clever" generic formula would silently reorder
floating-point operations at one or more call sites, which fails the
bit-identical bar this refactor is held to (the task that produced this
module; see tests/test_transit_math_golden.py). So ``fold_and_measure_depth``
below is ONE function with EXPLICIT mode parameters that select, per call
site, the exact verbatim arithmetic the original inline/closure code used --
not a reinterpretation of it.

Every call site's wrapper (secondary_depth, the two centre_phase calls inside
double_period_eb_check, the fold block inside spoc_repeat_check, and
_fold_and_measure) is proven bit-identical to the pre-refactor code by
tests/test_transit_math_golden.py, which pastes each OLD implementation
verbatim and asserts exact (``==`` / ``np.array_equal``, never ``allclose``)
agreement against the call into this module.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class FoldDepthResult:
    """Everything one ``fold_and_measure_depth`` call computed.

    ``phase``/``in_mask``/``oot_mask``/``n_in``/``n_oot`` are always
    populated (even when the caller's count guard would reject the result)
    so a caller that needs to reuse the in-window mask downstream -- e.g.
    spoc_repeat_check reuses its in-transit mask for per-epoch counting --
    can do so without re-folding. ``depth_ppm``/``err_ppm`` are ``None``
    when not computed (guard failed and ``guard_before_compute=True``, or
    the "relative" formula hit a zero baseline).
    """

    phase: np.ndarray
    in_mask: np.ndarray
    oot_mask: np.ndarray
    n_in: int
    n_oot: int
    depth_ppm: Optional[float]
    err_ppm: Optional[float]


def fold_and_measure_depth(
    time: np.ndarray,
    flux: np.ndarray,
    period: float,
    t0: float,
    *,
    centre_phase: float = 0.0,
    half_width: float,
    phase_mode: str = "raw",
    window_mode: str = "direct",
    oot_mode: str,
    oot_windows: Optional[Tuple[float, float]] = None,
    oot_bands: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None,
    oot_annulus: Optional[Tuple[float, float]] = None,
    oot_excl_half_width: Optional[float] = None,
    depth_formula: str = "diff",
    error_mode: Optional[str] = None,
    min_in: int = 0,
    min_oot: int = 0,
    guard_before_compute: bool = True,
) -> FoldDepthResult:
    """Fold (time, flux) at (period, t0) and measure a signal-window depth
    against an out-of-transit/baseline window, per one of the 4 conventions
    catalogued in this module's docstring.

    Parameters
    ----------
    time, flux : arrays (same length). Not copied, not NaN-filtered, not
        sorted -- callers that need that do it themselves (matches every
        original call site: none of the 4 pre-refactor implementations did
        it inside the fold/depth block either; it happened earlier, at LC
        load time, in the caller).
    period, t0 : fold period (days) and epoch (BTJD). For site 2
        (double_period_eb_check) the caller passes 2xP as *period*.
    centre_phase : phase-units centre of the signal window (0.0 for a
        primary transit; 0.5 for a secondary eclipse fold at 2xP; an
        arbitrary off-transit "control" phase for site 4).
    half_width : phase-units half-width of the signal window.
    phase_mode : "raw" -- phase = ((time-t0) % period) / period, in [0, 1).
                 "recentered" -- as above, then ``phase[phase>0.5] -= 1.0``,
                 giving (-0.5, 0.5].
    window_mode : "direct" -- in_mask = abs(phase - centre_phase) < half_width.
                  "circular" -- wraps around centre_phase before testing, so
                  a centre near the raw-phase domain edge (e.g. 0.0) does not
                  lose points that wrapped to the other edge (~1.0). Only
                  meaningful with phase_mode="raw" (matches site 2, the only
                  caller).
    oot_mode : "two_windows" -- union of two direct windows at *oot_windows*,
                  using the SAME half_width as the signal window (site 1).
               "bands" -- union of two fixed open-interval bands, independent
                  of half_width or centre_phase (site 2).
               "annulus" -- abs(phase) in (lo, hi), i.e. a symmetric ring
                  around phase 0, independent of centre_phase (site 3).
               "exclude_centre" -- abs(phase) > oot_excl_half_width, i.e.
                  everything outside a one-sided exclusion around phase 0
                  (not around centre_phase!) (site 4).
    depth_formula : "diff" -- (median(oot) - median(in)) x 1e6 ppm.
                    "relative" -- (1 - median(in)/median(oot)) x 1e6 ppm;
                    returns None if median(oot) == 0 (matches site 4's
                    explicit zero-division guard).
    error_mode : None -- no error computed (sites 1 and 4).
                 "mad_in" -- MAD-based robust error on the in-window points,
                    scaled by sqrt(n_in), floored at 1 ppm (site 2).
                 "std_oot" -- std-based error on the OOT/baseline points,
                    scaled by sqrt(max(1, n_in)) (site 3).
    min_in, min_oot : count guard applied when ``guard_before_compute=True``.
        depth/err are only computed if n_in >= min_in AND n_oot >= min_oot;
        otherwise they stay ``None`` -- this exactly reproduces the
        "if n_in < X or n_oot < Y: return None[, None]" guards of sites 1,
        2 and 4, which never call ``np.median``/``np.std`` on an
        insufficient sample.
    guard_before_compute : if False, depth/err are always computed
        (subject only to the "relative" zero-baseline guard), and the
        min_in/min_oot counts are returned for the CALLER to apply its own
        guard afterwards -- this matches site 3 (spoc_repeat_check), whose
        guard order/control-flow does not fit the simple
        "guard-then-compute" shape of the other three (it checks n_oot
        before n_in, computes the error unconditionally via its own
        ``max(1, ...)`` floor, and only guards n_in right before reading the
        depth). numpy's median/std of an empty selection return NaN with a
        (globally suppressed, see discovery_postprocess.py's
        ``warnings.filterwarnings("ignore")``) RuntimeWarning rather than
        raising, so eagerly computing these here and letting the caller
        discard them on a failed guard is observably identical to never
        having computed them.

    Returns
    -------
    FoldDepthResult
    """
    # --- phase -------------------------------------------------------
    if phase_mode == "raw":
        phase = ((time - t0) % period) / period
    elif phase_mode == "recentered":
        phase = ((time - t0) % period) / period
        phase[phase > 0.5] -= 1.0
    else:
        raise ValueError(f"unknown phase_mode {phase_mode!r}")

    # --- signal (in-window) mask --------------------------------------
    if window_mode == "direct":
        in_mask = np.abs(phase - centre_phase) < half_width
    elif window_mode == "circular":
        ph_c = (phase - centre_phase + 0.5) % 1.0 - 0.5
        in_mask = np.abs(ph_c) < half_width
    else:
        raise ValueError(f"unknown window_mode {window_mode!r}")

    # --- OOT / baseline mask ------------------------------------------
    if oot_mode == "two_windows":
        c_a, c_b = oot_windows
        oot_mask = (np.abs(phase - c_a) < half_width) | (np.abs(phase - c_b) < half_width)
    elif oot_mode == "bands":
        (lo1, hi1), (lo2, hi2) = oot_bands
        oot_mask = ((phase > lo1) & (phase < hi1)) | ((phase > lo2) & (phase < hi2))
    elif oot_mode == "annulus":
        lo, hi = oot_annulus
        oot_mask = (np.abs(phase) > lo) & (np.abs(phase) < hi)
    elif oot_mode == "exclude_centre":
        oot_mask = np.abs(phase) > oot_excl_half_width
    else:
        raise ValueError(f"unknown oot_mode {oot_mode!r}")

    n_in = int(in_mask.sum())
    n_oot = int(oot_mask.sum())

    depth_ppm: Optional[float] = None
    err_ppm: Optional[float] = None

    should_compute = True
    if guard_before_compute:
        should_compute = (n_in >= min_in) and (n_oot >= min_oot)

    if should_compute:
        f_in = flux[in_mask]
        f_oot = flux[oot_mask]

        if depth_formula == "diff":
            depth_ppm = float((np.median(f_oot) - np.median(f_in)) * 1e6)
        elif depth_formula == "relative":
            med_in = float(np.median(f_in))
            med_oot = float(np.median(f_oot))
            if med_oot == 0:
                depth_ppm = None
            else:
                depth_ppm = (1.0 - med_in / med_oot) * 1_000_000.0
        else:
            raise ValueError(f"unknown depth_formula {depth_formula!r}")

        if error_mode == "mad_in":
            mad = np.median(np.abs(f_in - np.median(f_in))) * 1.4826
            err = (mad / np.sqrt(n_in)) * 1e6
            err_ppm = max(err, 1.0)
        elif error_mode == "std_oot":
            baseline_std = float(np.std(f_oot))
            err_ppm = (baseline_std / max(1, np.sqrt(n_in))) * 1e6
        elif error_mode is None:
            pass
        else:
            raise ValueError(f"unknown error_mode {error_mode!r}")

    return FoldDepthResult(
        phase=phase, in_mask=in_mask, oot_mask=oot_mask,
        n_in=n_in, n_oot=n_oot, depth_ppm=depth_ppm, err_ppm=err_ppm,
    )
