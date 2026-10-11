"""CLI for the ExoMiner++ 2.0 FFI integration.

Subcommands:

  contract            Print the 56-input contract (names + shapes).
  gate TIC            Decide the gate for a TIC (queries MAST DV unless
                      --dv-runs is given).
  normstats           Report bundled normalization-stat completeness.
  smoke               Load the CV ensemble and score a zero batch (proves the
                      inference path; run in the TF 2.13 venv).

Examples:
  python -m exonit.exominer_pp contract
  python -m exonit.exominer_pp gate 468178519 --dv-runs 67-67
  python -m exonit.exominer_pp normstats
  <tf-venv>/python -m exonit.exominer_pp smoke
"""
from __future__ import annotations

import argparse
import sys
from typing import List, Tuple


def _parse_runs(arg: str) -> List[Tuple[int, int]]:
    runs = []
    for tok in arg.split(","):
        tok = tok.strip()
        if not tok:
            continue
        a, b = tok.split("-")
        runs.append((int(a), int(b)))
    return runs


def cmd_contract(_args) -> int:
    from .contract import EXOMINER_PP_INPUTS, IMAGE_BRANCHES

    print(f"ExoMiner++ 2.0 FFI contract: {len(EXOMINER_PP_INPUTS)} inputs")
    for n, s in EXOMINER_PP_INPUTS.items():
        tag = " [IMAGE]" if n in IMAGE_BRANCHES else ""
        print(f"  {n:50s} {str(s):14s}{tag}")
    return 0


def cmd_gate(args) -> int:
    from .gate import decide_gate

    dv_runs = _parse_runs(args.dv_runs) if args.dv_runs else None
    d = decide_gate(args.tic, dv_runs=dv_runs)
    print(f"TIC {args.tic}: use_exominer_pp={d.use_exominer_pp} "
          f"reason={d.reason} sector_run={d.sector_run} dv_runs={list(d.dv_runs)}")
    return 0


def cmd_normstats(_args) -> int:
    from .norm_stats import check_completeness, load_norm_stats

    stats = load_norm_stats()
    print(f"scalar_params: {len(stats['scalar_params'])} features")
    print(f"centroid:      {list(stats['centroid'])}")
    print(f"diff_img:      {list(stats['diff_img'])}")
    comp = check_completeness(stats)
    print("required diff-image stats present:")
    for k, ok in comp.items():
        print(f"  {k:16s} {'OK' if ok else 'MISSING (derive from eval TFRecords)'}")
    return 0


def cmd_smoke(_args) -> int:
    import numpy as np

    from .contract import EXOMINER_PP_INPUTS
    from .inference import EXOMINER_PP_MODEL_VERSION, ExoMinerPPEnsemble

    ens = ExoMinerPPEnsemble().load()
    batch = {n: np.zeros((2,) + s, dtype=np.float32)
             for n, s in EXOMINER_PP_INPUTS.items()}
    mean, folds = ens.predict(batch, per_fold=True)
    print(f"model_version: {EXOMINER_PP_MODEL_VERSION}")
    print(f"folds loaded:  {len(ens._models)}")
    print(f"per-fold zero-input scores: {folds[:, 0].tolist()}")
    print(f"CV-mean zero-input score:   {float(mean[0]):.6f}")
    print("OK: inference path works (contract-valid forward pass -> [0,1] score).")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="exonit.exominer_pp")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("contract").set_defaults(func=cmd_contract)

    g = sub.add_parser("gate")
    g.add_argument("tic")
    g.add_argument("--dv-runs", default=None,
                   help="comma-sep start-end runs, e.g. '1-13,67-67' (skip MAST)")
    g.set_defaults(func=cmd_gate)

    sub.add_parser("normstats").set_defaults(func=cmd_normstats)
    sub.add_parser("smoke").set_defaults(func=cmd_smoke)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
