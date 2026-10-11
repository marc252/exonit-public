#!/usr/bin/env python3
"""Enqueue ExoNit targets from a stratified-sample CSV.

Reads a CSV produced by scripts/sample_stratified.py and posts each
target to the ExoNit server queue via exonit.enqueue.enqueue_targets.
"""

import argparse
import json
import os

import pandas as pd

from exonit.config import ExoNitConfig
from exonit.db import ExoNitDB
from exonit.enqueue import enqueue_targets


def load_targets(csv_path):
    """Read a stratified-sample CSV and return a list of target dicts."""
    df = pd.read_csv(csv_path, dtype={"tic_id": str})
    df = df[df["tic_id"].notna()]
    df = df[df["tic_id"] != "nan"]
    targets = []
    for _, row in df.iterrows():
        targets.append({"tic_id": str(row["tic_id"])})
    return targets


def parse_args():
    p = argparse.ArgumentParser(description="Enqueue targets from a CSV sample")
    p.add_argument("--csv", required=True, help="Path to stratified sample CSV")
    p.add_argument("--server", default="http://localhost:8400")
    p.add_argument("--token",
                   default=os.environ.get("EXONIT_WORKER_TOKEN", "exonit-default-token"))
    p.add_argument("--db", default=None, help="DB path (default: from config)")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--force-rescan", action="store_true",
                   help="Deep-rescan: re-enqueue TICs already completed "
                        "(spec 2026-07-23). Never the default.")
    p.add_argument("--sector", type=int, default=None,
                   help="TESS sector this lot belongs to (done-state design, "
                        "2026-09-26). Tags tasks.sector at insert time and, "
                        "with EXONIT_DONE_STATE=1, feeds the exhausted(t, "
                        "sector) filter. Optional: omitted -> tasks.sector "
                        "stays NULL (legacy behaviour).")
    p.add_argument("--run-id", default=None,
                   help="This lot's run identity (S4/S3, docs/superpowers/"
                        "specs/2026-09-26-enqueue-s1-s4-design.md). Minted "
                        "once by scripts/discovery_run.py and passed through "
                        "here so /api/worker/enqueue_bulk can tag "
                        "tasks.run_id and report/adopt live tasks from a "
                        "dead prior attempt at this same lot. Optional: "
                        "omitted -> tasks.run_id stays NULL (legacy "
                        "behaviour).")
    return p.parse_args()


def main():
    args = parse_args()
    targets = load_targets(args.csv)
    print(f"Loaded {len(targets)} targets from {args.csv}")

    db_path = args.db or ExoNitConfig().db_path
    db = ExoNitDB(db_path)

    summary = enqueue_targets(
        targets, db,
        server_url=args.server,
        token=args.token,
        max_workers=args.workers,
        force_rescan=args.force_rescan,
        sector=args.sector,
        run_id=args.run_id,
    )
    print(summary)

    # Machine-parseable line for scripts/discovery_run.py: the plain
    # print(summary) above is for humans (dict repr, not valid JSON).
    # Filter to JSON-serializable scalars only, defensively, so a future
    # non-serializable value added to the summary dict can't crash this
    # CLI's exit path.
    json_safe_summary = {
        k: v for k, v in summary.items()
        if isinstance(v, (int, float, str, bool)) or v is None
    }
    print("ENQUEUE_SUMMARY_JSON: " + json.dumps(json_safe_summary))


if __name__ == "__main__":
    main()
