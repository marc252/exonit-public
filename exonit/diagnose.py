"""Classify why a known KP/CP was not detected by ExoNit.

Heuristics use candidate list + stellar metadata + processing telemetry
to assign a single reason per miss. The reasons are actionable:

- no_bls_candidate    : BLS found nothing (maybe detrending destroyed it)
- wrong_period        : BLS found something, but not at the right period
- shallow_depth       : known transit is <500 ppm, precision marginal
- long_period         : known P > 50d, outside BLS search range (0.5-50d)
- short_period        : known P < 0.5d, outside BLS search range
- slow_flatten        : processing took > 5 min (bottleneck signal)
- insufficient_cover  : <3 sectors available, hard to detect
- unknown             : fallback when heuristics do not apply
"""

import pandas as pd


BLS_PERIOD_MIN = 0.5
BLS_PERIOD_MAX = 50.0
SHALLOW_DEPTH_PPM = 500
SLOW_FLATTEN_SECONDS = 300  # 5 min
INSUFFICIENT_SECTORS = 2


def classify_miss(*, known_period, known_depth_ppm, n_sectors,
                  time_span_days, processing_time_s, candidates,
                  tolerance=0.01):
    """Return (reason, detail_dict) explaining why this KP/CP was missed.

    `candidates` is a pandas DataFrame with columns: period_days, depth_ppm,
    snr, cnn_score — the candidates ExoNit found for this TIC.
    """
    detail = {
        "known_period": known_period,
        "known_depth_ppm": known_depth_ppm,
        "n_sectors": n_sectors,
        "time_span_days": time_span_days,
        "processing_time_s": processing_time_s,
        "n_candidates": int(len(candidates)),
    }

    if known_period is not None and known_period > BLS_PERIOD_MAX:
        return "long_period", detail
    if known_period is not None and 0 < known_period < BLS_PERIOD_MIN:
        return "short_period", detail

    if known_depth_ppm is not None and known_depth_ppm < SHALLOW_DEPTH_PPM:
        return "shallow_depth", detail

    if processing_time_s is not None and processing_time_s > SLOW_FLATTEN_SECONDS:
        return "slow_flatten", detail

    if n_sectors is not None and n_sectors <= INSUFFICIENT_SECTORS:
        return "insufficient_cover", detail

    if candidates is None or len(candidates) == 0:
        return "no_bls_candidate", detail

    from exonit.scorecard import match_period
    any_match = False
    for _, cand in candidates.iterrows():
        matched, _ = match_period(cand["period_days"], known_period,
                                  tolerance=tolerance)
        if matched:
            any_match = True
            break
    if not any_match:
        return "wrong_period", detail

    return "unknown", detail


def diagnose_misses_report(toi, stars, candidates, processing_times=None,
                           tolerance=0.01):
    """Build a per-miss report for KP/CP in TOI that ExoNit failed to detect.

    Returns a list of dicts sorted by known_depth_ppm descending (most
    suspicious first — deep transits that ExoNit missed).
    """
    from exonit.scorecard import _best_match_for_known

    processed = stars.set_index("tic_id")
    kp_cp = toi[toi["disposition"].isin(["KP", "CP"])].copy()
    kp_cp = kp_cp[kp_cp["tic_id"].isin(processed.index)]

    rows = []
    for _, row in kp_cp.iterrows():
        tic = str(row["tic_id"])
        tic_candidates = candidates[candidates["tic_id"] == tic]
        match, _ = _best_match_for_known(tic_candidates, row["period_days"],
                                         tolerance=tolerance)
        if match is not None:
            continue
        star_row = processed.loc[tic]
        reason, detail = classify_miss(
            known_period=row["period_days"],
            known_depth_ppm=row["depth_ppm"],
            n_sectors=int(star_row.get("n_sectors") or 0),
            time_span_days=float(star_row.get("time_span_days") or 0),
            processing_time_s=(processing_times or {}).get(tic),
            candidates=tic_candidates,
            tolerance=tolerance,
        )
        rows.append({
            "tic_id": tic,
            "disposition": row["disposition"],
            "reason": reason,
            **detail,
        })
    rows.sort(key=lambda r: (r["known_depth_ppm"] or 0), reverse=True)
    return rows


def summarize_reasons(rows):
    """Aggregate report rows into reason → count, sorted descending."""
    from collections import Counter
    return Counter(r["reason"] for r in rows).most_common()
