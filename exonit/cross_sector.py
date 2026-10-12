"""Cross-sector period recurrence (pure, no DB).

A candidate seen in the current sector whose period (direct or harmonic)
recurs in a DIFFERENT sector of the same TIC is a strong independent hint
that the signal is astrophysical rather than a sector-specific systematic.
Deliberately independent of ``claude_verdict`` so it can rescue rows the
classifier / vetoes labelled likely_false_positive.

Rows are dicts: {id, tic_id, sector, period_days, t0, snr, claude_verdict,
exominer_ffi_score}.
"""
from __future__ import annotations

from exonit.scorecard import match_period

LEGACY_DEFAULT_SECTOR = 1  # candidates.sector DEFAULT 1 -> label is suspect


def _snr(r):
    v = r.get("snr")
    return float(v) if v is not None else 0.0


def _match(a, b, tol):
    """Symmetric period match: try both argument orders. -> kind or None."""
    for x, y in ((a, b), (b, a)):
        ok, kind = match_period(x, y, tolerance=tol, inclusive=True)
        if ok:
            return kind
    return None


def find_cross_sector_hits(rows, current_sector, tol=0.01, min_snr=0.0):
    cur_by_tic, other_by_tic = {}, {}
    for r in rows:
        if r.get("period_days") is None or r["period_days"] <= 0:
            continue
        tic = str(r["tic_id"])
        if r.get("sector") == current_sector:
            if _snr(r) >= min_snr:
                cur_by_tic.setdefault(tic, []).append(r)
        else:
            other_by_tic.setdefault(tic, []).append(r)

    hits = {}
    for tic, curs in cur_by_tic.items():
        others = other_by_tic.get(tic)
        if not others:
            continue
        matches = []
        for c in curs:
            for o in others:
                kind = _match(c["period_days"], o["period_days"], tol)
                if kind is None:
                    continue
                matches.append({
                    "cur_id": c.get("id"), "cur_period": c["period_days"],
                    "cur_snr": c.get("snr"), "cur_verdict": c.get("claude_verdict"),
                    "other_id": o.get("id"), "other_sector": o.get("sector"),
                    "other_period": o["period_days"], "other_snr": o.get("snr"),
                    "other_verdict": o.get("claude_verdict"), "kind": kind,
                    "suspect_sector1": o.get("sector") == LEGACY_DEFAULT_SECTOR
                                       or current_sector == LEGACY_DEFAULT_SECTOR,
                })
        if not matches:
            continue
        matched_ids = {m["cur_id"] for m in matches}
        best_pool = [c for c in curs if c.get("id") in matched_ids] or curs
        best = max(best_pool, key=_snr)
        hits[tic] = {
            "tic_id": tic,
            "cross_sector_hits": len({m["other_sector"] for m in matches}),
            "best": best,
            "matches": matches,
        }
    return hits


def _sorted(hits):
    return sorted(hits.values(),
                  key=lambda h: (-h["cross_sector_hits"], -_snr(h["best"]), h["tic_id"]))


def _f(v, fmt):
    return "-" if v is None else format(v, fmt)


def render_review_queue_md(hits, run_label, current_sector):
    items = _sorted(hits)
    out = [f"# Cross-sector review queue - {run_label}", "",
           f"Current sector: {current_sector}. TICs whose period recurs "
           f"(direct/harmonic) in another sector: {len(items)}. "
           "Independent of claude_verdict by design.", ""]
    if not items:
        return "\n".join(out + ["(none)", ""])
    out += ["| TIC | other sectors | best SNR | cur P(d) | other sector | other P(d) | kind | cur verdict | other verdict | flag |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    for h in items:
        for m in h["matches"]:
            out.append(
                f"| {h['tic_id']} | {h['cross_sector_hits']} | {_f(_snr(h['best']), '.1f')} "
                f"| {_f(m['cur_period'], '.4f')} | {m['other_sector']} "
                f"| {_f(m['other_period'], '.4f')} | {m['kind']} "
                f"| {m['cur_verdict'] or '-'} | {m['other_verdict'] or '-'} "
                f"| {'suspect_sector1' if m['suspect_sector1'] else ''} |")
    return "\n".join(out + [""])


def review_queue_json(hits, run_label=None, current_sector=None):
    return {
        "run": run_label, "current_sector": current_sector,
        "n_tics": len(hits),
        "tics": [{**h, "best": dict(h["best"])} for h in _sorted(hits)],
    }
