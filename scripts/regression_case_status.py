#!/usr/bin/env python3
"""Read-only observability for fixed regression cases.

Reads data/samples/regression_cases.csv and, per case, reports what production
(EXONIT_DATABASE_URL) holds for that TIC/sector.  Always exits 0.
"""
import csv
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
CASES_CSV = REPO / "data" / "samples" / "regression_cases.csv"
MATCH_TOL = 0.01


def load_cases(path=CASES_CSV):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def _match(period, known):
    from exonit.scorecard import match_period
    return bool(match_period(period, known, tolerance=MATCH_TOL)[0])


def t0_offset_min(t0, period, t0_pub, p_pub):
    """Offset (min) of candidate t0 to the nearest published transit."""
    n = round((t0 - t0_pub) / p_pub)
    return abs(t0 - (t0_pub + n * p_pub)) * 1440.0


def summarize_case(case, rows):
    """rows: dicts with period_days,t0,snr,claude_verdict,verdict_proba(opt)."""
    p_pub = float(case["period_days"])
    t0_pub = float(case["t0_btjd"])
    matched = [r for r in rows if r.get("period_days") is not None
               and _match(float(r["period_days"]), p_pub)]
    if matched:
        best = max(matched, key=lambda r: r.get("snr") or 0)
        off = t0_offset_min(float(best["t0"]), float(best["period_days"]),
                            t0_pub, p_pub) if best.get("t0") is not None else None
        verdict = best.get("claude_verdict")
        proba = best.get("verdict_proba")
        promoted = verdict == "planet_candidate"
        pstr = f"{float(best['period_days']):.4f}"
        ostr = "n/a" if off is None else f"{off:.0f}"
        vstr = str(verdict)
        prstr = "n/a" if proba is None else f"{float(proba):.3f}"
    else:
        promoted, pstr, ostr, vstr, prstr = False, "n/a", "n/a", "n/a", "n/a"
    flag = "PROMOTED" if promoted else "NOT PROMOTED"
    return (f"TIC {case['tic_id']} s{case['sector']}: {len(rows)} candidates, "
            f"best P={pstr}, t0 offset={ostr} min, claude_verdict={vstr}, "
            f"verdict_proba={prstr} → {flag}")


def fetch_rows(conn, tic, sector):
    cur = conn.cursor()
    cur.execute("set statement_timeout='60s'")
    cur.execute("select 1 from information_schema.columns where "
                "table_name='candidates' and column_name='verdict_proba'")
    has_proba = cur.fetchone() is not None
    sel = ("period_days, t0, snr, claude_verdict, "
           + ("verdict_proba" if has_proba else "NULL AS verdict_proba"))
    cur.execute(f"select {sel} from candidates where tic_id=%s and sector=%s",
                (str(tic), int(sector)))
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, r)) for r in cur.fetchall()]


def main():
    cases = load_cases()
    url = os.environ.get("EXONIT_DATABASE_URL")
    if not url:
        print("EXONIT_DATABASE_URL not set; nothing to report")
        return 0
    try:
        import psycopg
        with psycopg.connect(url) as conn:
            for case in cases:
                rows = fetch_rows(conn, case["tic_id"], case["sector"])
                print(summarize_case(case, rows))
    except Exception as e:  # observability only, never a gate
        print(f"regression_case_status: error {type(e).__name__}: {e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
