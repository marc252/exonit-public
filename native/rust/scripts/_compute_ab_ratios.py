#!/usr/bin/env python3
"""Stdlib-only helper for `run_ab_bench.sh`: reads the interleaved round
JSON files it produced and computes, per input file and per stage, the
MEDIAN of the python/rust ratio computed independently for each adjacent
(rust_i, python_i) round pair.

Only adjacent-round ratios are ever computed — never a python round against
a rust round from a different (non-adjacent) iteration — because only
measurements taken close together in wall-clock time cancel out this host's
background-load drift (see docs/DESIGN.md's <host> section and
`exonit_bench.rs`'s / `benchmark_python_stages.py`'s own module doc comments
for why: qwen-llm.service, PostgreSQL, MinIO, and TGLC downloads all share
this host's CPU). The median of N such ratios is the number this whole
harness exists to produce.

Usage: _compute_ab_ratios.py ROUND_JSON [ROUND_JSON ...]
  where ROUND_JSON files are named rust_round<N>.json / python_round<N>.json
  (as scripts/run_ab_bench.sh writes them) and passed in round order,
  alternating rust/python — exactly the order run_ab_bench.sh passes them.
Prints a JSON summary to stdout.
"""
import json
import statistics
import sys

STAGES = [
    "fits_read", "preprocess_detrend", "bls_two_pass_cpu",
    "bls_two_pass_cuda", "pipeline_full", "onnx_cnn", "onnx_exominer",
]


def load_rounds(paths):
    rounds = []
    for path in paths:
        with open(path) as fh:
            doc = json.load(fh)
        name = "rust" if "exonit-bench" in doc.get("tool", "") else "python"
        rounds.append((name, path, doc))
    return rounds


def index_by_path(doc):
    return {f["path"].rsplit("/", 1)[-1]: f for f in doc.get("files", [])}


def main():
    if len(sys.argv) < 3:
        print(__doc__, file=sys.stderr)
        return 2
    rounds = load_rounds(sys.argv[1:])

    pairs = []
    i = 0
    while i + 1 < len(rounds):
        a_name, a_path, a_doc = rounds[i]
        b_name, b_path, b_doc = rounds[i + 1]
        if {a_name, b_name} != {"rust", "python"}:
            print(f"warning: round pair {i}/{i+1} is not one rust + one python "
                  f"({a_name}, {b_name}) -- skipping", file=sys.stderr)
            i += 1
            continue
        rust_doc = a_doc if a_name == "rust" else b_doc
        python_doc = a_doc if a_name == "python" else b_doc
        pairs.append((rust_doc, python_doc))
        i += 2

    if not pairs:
        print("no valid adjacent rust/python round pairs found", file=sys.stderr)
        return 1

    # file basename -> stage -> [ratios across pairs]
    per_file_stage = {}
    for rust_doc, python_doc in pairs:
        rust_files = index_by_path(rust_doc)
        python_files = index_by_path(python_doc)
        common = set(rust_files) & set(python_files)
        for basename in common:
            rf = rust_files[basename]["stages"]
            pf = python_files[basename]["stages"]
            per_file_stage.setdefault(basename, {})
            for stage in STAGES:
                r = rf.get(stage)
                p = pf.get(stage)
                if not r or not p:
                    continue
                rust_s = r.get("min_s")
                python_s = p.get("min_s")
                if not rust_s or rust_s <= 0:
                    continue
                ratio = python_s / rust_s
                per_file_stage[basename].setdefault(stage, []).append(ratio)

    summary = {"pairs_used": len(pairs), "files": {}}
    for basename, stages in sorted(per_file_stage.items()):
        summary["files"][basename] = {}
        for stage, ratios in stages.items():
            summary["files"][basename][stage] = {
                "median_ratio_python_over_rust": statistics.median(ratios),
                "n_pairs": len(ratios),
                "ratios": ratios,
            }

    # Overall median ratio per stage across all files (secondary summary).
    overall = {}
    for basename, stages in per_file_stage.items():
        for stage, ratios in stages.items():
            overall.setdefault(stage, []).extend(ratios)
    summary["overall_median_ratio_python_over_rust"] = {
        stage: statistics.median(values) for stage, values in overall.items()
    }

    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
