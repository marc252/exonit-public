"""Statistical false positive vetting for transit candidates.

Implements multiple checks to filter out common false positives:
- Harmonics: detect period aliases (P/2, P/3, 2P, 3P)
- Odd/Even: check transit depth consistency across alternating transits
- V-shape: distinguish planet-like (U) from binary-like (V) transits
"""

import numpy as np

from phase_fold import phase_fold

HARMONIC_RATIOS = [0.5, 1/3, 2.0, 3.0, 2/3, 3/2]
HARMONIC_TOLERANCE = 0.02


def check_harmonics(candidate, all_candidates):
    """Check if a candidate's period is a harmonic of a stronger-or-equal signal.

    Sorts parents by SNR descending so the strongest matching parent wins
    when several candidates share a harmonic relationship with ``candidate``.
    Uses ``<`` (not ``<=``) so that an equal-SNR duplicate still gets flagged —
    e.g. two copies of the same signal at P=X and P=2X both reporting SNR=20
    previously both slipped through as independent detections.
    """
    p = candidate["period_days"]
    parents = sorted(
        (o for o in all_candidates
         if o["candidate_num"] != candidate["candidate_num"]
         and o["snr"] >= candidate["snr"]),
        key=lambda o: (-o["snr"], o["candidate_num"]),
    )
    for other in parents:
        if other["snr"] == candidate["snr"] and \
                other["candidate_num"] > candidate["candidate_num"]:
            continue  # break symmetric tie: only the later one gets flagged
        p_other = other["period_days"]
        for ratio in HARMONIC_RATIOS:
            if abs(p - p_other * ratio) / p < HARMONIC_TOLERANCE:
                return {"is_harmonic": True, "harmonic_of": other["candidate_num"],
                        "harmonic_ratio": ratio}
    return {"is_harmonic": False, "harmonic_of": None, "harmonic_ratio": None}


def check_odd_even(time, flux, period, t0, duration):
    """Check if odd and even transits have consistent depths.
    Eclipsing binaries show alternating primary/secondary eclipse depths."""
    half_dur = duration / 2
    odd_fluxes = []
    even_fluxes = []

    n_transits = int((time[-1] - time[0]) / period) + 1
    for i in range(n_transits):
        epoch = t0 + i * period
        in_transit = np.abs(time - epoch) < half_dur
        transit_flux = flux[in_transit]
        if len(transit_flux) < 3:
            continue
        depth = 1.0 - np.median(transit_flux)
        if i % 2 == 0:
            even_fluxes.append(depth)
        else:
            odd_fluxes.append(depth)

    if len(odd_fluxes) < 2 or len(even_fluxes) < 2:
        return {"odd_even_consistent": True, "odd_depth": None, "even_depth": None}

    odd_depth = np.median(odd_fluxes)
    even_depth = np.median(even_fluxes)
    max_d = max(odd_depth, even_depth)
    min_d = min(odd_depth, even_depth)
    consistent = (min_d > 0 and max_d / min_d < 3.0) or max_d < 1e-5

    return {"odd_even_consistent": consistent,
            "odd_depth": round(float(odd_depth * 1e6), 1),
            "even_depth": round(float(even_depth * 1e6), 1)}


def check_v_shape(phase, flux):
    """Check if a phase-folded transit is V-shaped (binary) or U-shaped (planet)."""
    sort_idx = np.argsort(phase)
    phase = phase[sort_idx]
    flux = flux[sort_idx]

    max_phase = np.max(np.abs(phase))
    if max_phase == 0:
        return {"is_v_shaped": False, "v_metric": 0.0}

    center_mask = np.abs(phase) < max_phase * 0.3
    edge_mask = np.abs(phase) > max_phase * 0.6

    if np.sum(center_mask) < 3 or np.sum(edge_mask) < 3:
        return {"is_v_shaped": False, "v_metric": 0.0}

    center_depth = 1.0 - np.median(flux[center_mask])
    edge_depth = 1.0 - np.median(flux[edge_mask])

    if center_depth <= 0:
        return {"is_v_shaped": False, "v_metric": 0.0}

    v_metric = edge_depth / center_depth
    is_v = v_metric > 0.7

    return {"is_v_shaped": is_v, "v_metric": round(float(v_metric), 3)}


def vet_candidate(candidate, all_candidates, time, flux):
    """Run all statistical vetting checks on a candidate."""
    result = {}

    harmonics = check_harmonics(candidate, all_candidates)
    result.update(harmonics)

    odd_even = check_odd_even(time, flux, candidate["period_days"],
                               candidate["t0"], candidate["duration_days"])
    result.update(odd_even)

    period = candidate["period_days"]
    t0 = candidate["t0"]
    duration = candidate["duration_days"]
    # Shared with cnn.py (identical convention) — see
    # docs/reviews/2026-07-02-opus48-cleanup-review.md item D5 and
    # tests/worker/test_phase_fold_golden.py for the bit-identical proof.
    phase = phase_fold(time, t0, period, fold_order="mod_div", center="mask")
    transit_window = (duration / period) * 2
    in_window = np.abs(phase) < transit_window
    if np.sum(in_window) >= 10:
        v_result = check_v_shape(phase[in_window], flux[in_window])
    else:
        v_result = {"is_v_shaped": False, "v_metric": 0.0}
    result.update(v_result)

    result["passed"] = (not result["is_harmonic"]
                        and result["odd_even_consistent"]
                        and not result["is_v_shaped"])
    return result
