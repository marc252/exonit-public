#!/usr/bin/env python3
"""Diagnose why known KP/CP transits were missed by ExoNit.

Reads the processed DB + TOI catalog, classifies every miss, and writes
a markdown report grouped by reason.
"""

import argparse
from pathlib import Path

from exonit.scorecard import load_toi_catalog, load_db_processed_for_toi
from exonit.diagnose import diagnose_misses_report, summarize_reasons
from exonit.db_backend import op_session


def parse_args():
    p = argparse.ArgumentParser(description="Diagnose KP/CP misses")
    p.add_argument("--db", required=True)
    p.add_argument("--toi", required=True)
    p.add_argument("--out", required=True, help="Output markdown path")
    p.add_argument("--tolerance", type=float, default=0.01)
    return p.parse_args()


def render_markdown(rows, summary):
    lines = ["# Diagnose misses report", ""]
    lines.append(f"Total misses: **{len(rows)}**")
    lines.append("")
    lines.append("## Summary by reason")
    lines.append("")
    lines.append("| Reason | Count |")
    lines.append("|--------|-------|")
    for reason, count in summary:
        lines.append(f"| {reason} | {count} |")
    lines.append("")
    lines.append("## Detailed rows (sorted by depth desc)")
    lines.append("")
    lines.append("| TIC | Disposition | Reason | Depth (ppm) | Period (d) | Sectors | Proc time (s) |")
    lines.append("|-----|-------------|--------|-------------|------------|---------|---------------|")
    for r in rows:
        lines.append(
            f"| {r['tic_id']} | {r['disposition']} | {r['reason']} | "
            f"{r.get('known_depth_ppm') or ''} | {r.get('known_period') or ''} | "
            f"{r.get('n_sectors') or ''} | {r.get('processing_time_s') or ''} |"
        )
    return "\n".join(lines)


def main():
    args = parse_args()
    toi = load_toi_catalog(args.toi)
    toi_tics = set(toi["tic_id"])
    stars, candidates, _ = load_db_processed_for_toi(args.db, toi_tics)

    # Routed through exonit.db_backend.op_session (same pattern as
    # discovery_postprocess.py / augment_scorecard_exominer_from_db.py) so a
    # postgresql:// DSN works exactly like a SQLite path — a bare
    # sqlite3.connect(args.db) against a DSN string raises "unable to open
    # database file" instead of talking to PG.
    con = op_session(args.db)
    try:
        cur = con.execute(
            "SELECT tic_id, processing_time FROM tasks "
            "WHERE processing_time IS NOT NULL"
        )
        processing_times = {str(row["tic_id"]): float(row["processing_time"])
                            for row in cur.fetchall()}
    finally:
        con.close()

    rows = diagnose_misses_report(toi, stars, candidates, processing_times,
                                  tolerance=args.tolerance)
    summary = summarize_reasons(rows)
    md = render_markdown(rows, summary)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(md)
    print(f"Wrote {len(rows)} misses to {out_path}")
    for reason, count in summary:
        print(f"  {reason}: {count}")


if __name__ == "__main__":
    main()
