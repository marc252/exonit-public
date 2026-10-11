#!/usr/bin/env python3
"""Generate a reproducible stratified sample from the TOI catalog."""

import argparse
from pathlib import Path

from exonit.scorecard import load_toi_catalog
from exonit.stratified import stratified_sample


def parse_args():
    p = argparse.ArgumentParser(description="Stratified TOI sample generator")
    p.add_argument("--catalog", required=True)
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--recall-fraction", type=float, default=0.80)
    p.add_argument("--out", required=True)
    return p.parse_args()


def main():
    args = parse_args()
    catalog = load_toi_catalog(args.catalog)
    sample = stratified_sample(
        catalog, n=args.n, seed=args.seed,
        recall_fraction=args.recall_fraction,
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(out_path, index=False)
    print(f"Wrote {len(sample)} rows to {out_path}")
    print(sample["role"].value_counts().to_string())


if __name__ == "__main__":
    main()
