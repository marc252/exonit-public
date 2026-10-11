#!/usr/bin/env python3
"""Compute an ExoNit scorecard and write JSON + markdown report."""

import argparse
import csv
import json
from pathlib import Path

from exonit.scorecard import (
    load_toi_catalog,
    load_db_processed_for_toi,
    build_scorecard,
    scorecard_to_markdown,
)
from exonit.progress_client import ProgressReporter


def parse_args():
    p = argparse.ArgumentParser(description="ExoNit scorecard generator")
    p.add_argument("--db", required=True, help="Path to ExoNit SQLite DB")
    p.add_argument("--toi", required=True, help="Path to toi_catalog.csv")
    p.add_argument("--out", required=True, help="Output JSON path")
    p.add_argument("--run-id", required=True)
    p.add_argument("--sample-size", type=int, required=True)
    p.add_argument("--stratification", default="unknown")
    p.add_argument("--pipeline-version", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tolerance", type=float, default=0.01)
    p.add_argument("--sample", default=None,
                   help="Lot CSV (tic_id column) — scopes the run's OWN recall "
                        "(metrics.lot). Without it the scorecard can only "
                        "report the cumulative figure, which does not measure "
                        "this run. Same CSV discovery_postprocess.py takes.")
    p.add_argument("--report", action="store_true",
                   help="Also print markdown report to stdout")
    return p.parse_args()


def _read_lot_tics(path):
    with open(path, newline="") as fh:
        return {str(row["tic_id"]) for row in csv.DictReader(fh)
                if row.get("tic_id")}


def main():
    args = parse_args()
    stages = ["parse arguments", "load inputs", "build scorecard",
              "write JSON", "write markdown report"]
    rep = ProgressReporter(unit="steps", total=len(stages),
                           substep="scorecard", interval_s=5)
    rep.report(1, message=stages[0])
    toi = load_toi_catalog(args.toi)
    toi_tics = set(toi["tic_id"])
    stars, candidates, coverage_aggregates = load_db_processed_for_toi(args.db, toi_tics)

    lot_tics = _read_lot_tics(args.sample) if args.sample else None
    rep.report(2, message=stages[1])

    card = build_scorecard(
        toi, stars, candidates, None,
        run_id=args.run_id, sample_size=args.sample_size,
        stratification=args.stratification,
        pipeline_version=args.pipeline_version,
        seed=args.seed, tolerance=args.tolerance,
        coverage_aggregates=coverage_aggregates,
        lot_tics=lot_tics, lot_csv=args.sample,
    )
    rep.report(3, message=stages[2])

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(card, indent=2))
    rep.report(4, message=stages[3])

    if args.report:
        print(scorecard_to_markdown(card))
    rep.report(5, message=stages[4])
    rep.finish()


if __name__ == "__main__":
    main()
