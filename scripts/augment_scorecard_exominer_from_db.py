#!/usr/bin/env python3
"""Augment a scorecard with ExoMiner per-candidate scores read from the DB.

Companion to ``augment_scorecard_exominer.py``. That script re-runs the
ExoMiner Docker container for each TIC in the sample (good for re-deriving
from scratch but expensive — minutes per TIC, GB of data downloaded). This
one just reads what is already in the candidates table:

  * ``exominer_score``      — Docker server-side path (SPOC DV products)
  * ``exominer_ffi_score``  — Phase 3 worker-side path (FFI fallback)

So once a backfill wave has finished populating those columns, this script
generates the same kind of ``{out}.candidates.csv`` + ``{out}.md`` in
seconds instead of hours, with no Docker dependency.

Sample CSV format is the same as the other augment script:
  tic_id,disposition,period_days,depth_ppm,duration_hours,role
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

from exonit.db_backend import op_session  # noqa: E402


def load_sample(path: Path) -> Dict[str, dict]:
    with open(path) as f:
        return {
            row["tic_id"].strip(): {
                "disposition": row["disposition"],
                "period_sample": (
                    float(row["period_days"]) if row.get("period_days") else None
                ),
                "depth_sample": (
                    float(row["depth_ppm"]) if row.get("depth_ppm") else None
                ),
                "duration_sample": (
                    float(row["duration_hours"]) if row.get("duration_hours") else None
                ),
                "role": row.get("role", ""),
            }
            for row in csv.DictReader(f) if row.get("tic_id", "").strip()
        }


_CANDIDATE_FIELDS = [
    "tic_id", "candidate_num", "period_days", "t0", "duration_days",
    "depth_ppm", "snr", "cnn_score", "triceratops_fpp",
    "claude_verdict", "exominer_score", "exominer_ffi_score",
    "exominer_sector_run", "exominer_ffi_model_version",
]


def load_candidates(db_path: str, tic_ids: List[str]) -> Dict[str, List[dict]]:
    """Read candidates for *tic_ids* from the operational DB.

    Routed through ``exonit.db_backend.op_session`` (same pattern as
    ``discovery_postprocess.py``) so a ``postgresql://`` DSN works exactly
    like a SQLite path — a raw ``sqlite3.connect(db_path)`` against a DSN
    string silently creates an empty local file named after the DSN and this
    script would report zero candidates found (B9).
    """
    con = op_session(db_path)
    out: Dict[str, List[dict]] = {}
    field_list = ", ".join(_CANDIDATE_FIELDS)
    # Chunk to avoid the SQLite/PG bound-parameter limit when the sample is
    # large (never a single unbounded IN(...) — discovery_postprocess.
    # _select_in_chunks convention).
    chunk = 500
    for i in range(0, len(tic_ids), chunk):
        batch = tic_ids[i:i + chunk]
        placeholders = ",".join("?" for _ in batch)
        rows = con.execute(
            f"SELECT {field_list} FROM candidates WHERE tic_id IN ({placeholders})",
            batch,
        )
        for row in rows.fetchall():
            d = dict(row)
            d["tic_id"] = str(d["tic_id"])
            out.setdefault(d["tic_id"], []).append(d)
    con.close()
    return out


def summarize(sample: Dict[str, dict], cands_by_tic: Dict[str, List[dict]],
              threshold: float) -> dict:
    """Per-disposition rollup, mirroring what scorecard.compute_exominer_metrics
    does at the scorecard level but with a flat structure that's easy to
    eyeball in the markdown report."""
    buckets = {}
    for tic, info in sample.items():
        d = info.get("disposition") or ""
        if not d:
            continue
        b = buckets.setdefault(d, {"n": 0, "em_pos": 0, "em_null": 0,
                                   "ffi_pos": 0, "ffi_null": 0,
                                   "combined_pos": 0})
        b["n"] += 1
        cands = cands_by_tic.get(tic, [])
        # Per-TIC max — collapse multi-candidate TICs the same way the
        # catalogue-level scorecard does.
        em_vals = [c["exominer_score"] for c in cands
                   if c["exominer_score"] is not None]
        ffi_vals = [c["exominer_ffi_score"] for c in cands
                    if c["exominer_ffi_score"] is not None]
        em = max(em_vals) if em_vals else None
        em_ffi = max(ffi_vals) if ffi_vals else None
        em_ok = em is not None and em >= threshold
        ffi_ok = em_ffi is not None and em_ffi >= threshold
        if em is None:        b["em_null"] += 1
        elif em_ok:           b["em_pos"] += 1
        if em_ffi is None:    b["ffi_null"] += 1
        elif ffi_ok:          b["ffi_pos"] += 1
        if em_ok or ffi_ok:   b["combined_pos"] += 1
    return buckets


def to_markdown(buckets: dict, threshold: float, db: str) -> str:
    lines = [
        f"# ExoMiner augmentation (from DB)",
        "",
        f"- DB: `{db}`",
        f"- Threshold: {threshold:.2f}",
        "",
        "| disp | n | em≥t | em<t | em-null | ffi≥t | ffi<t | ffi-null | em|ffi≥t |",
        "|------|---|------|------|---------|-------|-------|----------|----------|",
    ]
    for d in ["KP", "CP", "PC", "APC", "FP", "FA"]:
        if d not in buckets:
            continue
        b = buckets[d]
        em_neg = b["n"] - b["em_pos"] - b["em_null"]
        ffi_neg = b["n"] - b["ffi_pos"] - b["ffi_null"]
        lines.append(
            f"| {d} | {b['n']} | {b['em_pos']} | {em_neg} | {b['em_null']} | "
            f"{b['ffi_pos']} | {ffi_neg} | {b['ffi_null']} | {b['combined_pos']} |"
        )
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", required=True,
                    help="CSV with tic_id,disposition,period_days,...")
    ap.add_argument("--db", required=True,
                    help="exonit.db with candidates table")
    ap.add_argument("--out", required=True,
                    help="Output basename (without extension)")
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()

    sample = load_sample(Path(args.sample))
    print(f"Loaded {len(sample)} TICs from sample")
    cands_by_tic = load_candidates(args.db, list(sample.keys()))
    total = sum(len(v) for v in cands_by_tic.values())
    print(f"Loaded {total} candidates over {len(cands_by_tic)} TICs from DB")

    # candidates.csv — one row per candidate
    out_base = Path(args.out)
    out_base.parent.mkdir(parents=True, exist_ok=True)
    csv_path = out_base.with_suffix(".candidates.csv")
    with open(csv_path, "w", newline="") as f:
        fields = ["tic_id", "disposition", "candidate_num", "period_days",
                  "depth_ppm", "duration_days", "snr", "cnn_score",
                  "triceratops_fpp", "claude_verdict", "exominer_score",
                  "exominer_ffi_score", "exominer_sector_run",
                  "exominer_ffi_model_version"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for tic, cands in cands_by_tic.items():
            disp = sample[tic]["disposition"]
            for c in cands:
                row = {k: c.get(k) for k in fields if k != "disposition"}
                row["tic_id"] = tic
                row["disposition"] = disp
                w.writerow(row)
    print(f"Wrote candidates → {csv_path}")

    buckets = summarize(sample, cands_by_tic, args.threshold)
    md = to_markdown(buckets, args.threshold, args.db)
    md_path = out_base.with_suffix(".exominer.md")
    md_path.write_text(md)
    print(f"Wrote report → {md_path}")

    json_path = out_base.with_suffix(".exominer.json")
    json_path.write_text(json.dumps(
        {"threshold": args.threshold, "buckets": buckets}, indent=2))
    print(f"Wrote summary → {json_path}")


if __name__ == "__main__":
    main()
