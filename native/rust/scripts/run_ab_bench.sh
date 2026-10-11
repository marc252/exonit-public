#!/usr/bin/env bash
# Interleaved Rust-vs-Python A/B benchmark orchestrator (docs/PLAN.md F6:
# "Benchmark A/B al mateix host (finestres de rellotge, no temps-per-TIC
# derivat)"). Runs N rounds alternating rust, python, rust, python... on the
# SAME input FITS files, timestamps every round's JSON, then computes the
# median of the python/rust ratio between ADJACENT round pairs per file per
# stage (see scripts/_compute_ab_ratios.py's doc comment for why only
# adjacent pairs are valid evidence on this host).
#
# Meant to run AT NVIDIARACK (Ryzen 7 5800X, RTX 5060 Ti) -- this script is
# NOT run by the implementing agent, only written by it; the operator runs it.
#
# Usage:
#   scripts/run_ab_bench.sh --fits-dir DIR [--rounds N] [--iterations N] \
#       [--models-dir DIR] [--output-dir DIR] [--rust-bin PATH] \
#       [--python PATH] [--worker-dir PATH]
#
# Environment:
#   EXONIT_BENCH_GPU=0 (default)  CPU-only pass: runs directly, no gpuq.
#   EXONIT_BENCH_GPU=1            GPU pass: per the project rule "GPU work on
#                                  <host> ALWAYS goes through gpuq" (never launch
#                                  CUDA directly), this script does NOT run
#                                  anything itself -- it PRINTS the exact
#                                  `gpuq run` command and exits 0. Copy-paste
#                                  and run that command yourself; it
#                                  re-invokes this same script with an
#                                  internal marker set, which is what
#                                  actually performs the GPU-inclusive run
#                                  (requires --rust-bin to point at a binary
#                                  built with `cargo build --release
#                                  --features cuda`).
#   EXONIT_BENCH_GPU_CONFIRMED=1  Internal marker set by the printed `gpuq
#                                  run ... bash $0 ...` command itself so the
#                                  re-invocation actually runs instead of
#                                  printing again. Do not set this by hand
#                                  outside that command.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

FITS_DIR=""
ROUNDS=5
ITERATIONS=3
MODELS_DIR="~/exonit/worker/models"
OUTPUT_DIR=""
RUST_BIN="$REPO_ROOT/target/release/exonit-bench"
PYTHON_BIN="~/exonit/.venv/bin/python3"
WORKER_DIR="~/exonit/worker"

usage() {
    sed -n '2,33p' "${BASH_SOURCE[0]}"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --fits-dir) FITS_DIR="$2"; shift 2 ;;
        --rounds) ROUNDS="$2"; shift 2 ;;
        --iterations) ITERATIONS="$2"; shift 2 ;;
        --models-dir) MODELS_DIR="$2"; shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --rust-bin) RUST_BIN="$2"; shift 2 ;;
        --python) PYTHON_BIN="$2"; shift 2 ;;
        --worker-dir) WORKER_DIR="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "run_ab_bench.sh: unknown argument: $1" >&2; usage; exit 2 ;;
    esac
done

if [[ -z "$FITS_DIR" ]]; then
    echo "run_ab_bench.sh: --fits-dir is required" >&2
    exit 2
fi

EXONIT_BENCH_GPU="${EXONIT_BENCH_GPU:-0}"

# --- GPU pass: never self-launch, print the gpuq command and exit. -------
if [[ "$EXONIT_BENCH_GPU" == "1" && "${EXONIT_BENCH_GPU_CONFIRMED:-0}" != "1" ]]; then
    cat <<EOF
run_ab_bench.sh: EXONIT_BENCH_GPU=1 requested. Per the project's hard rule
("GPU work on <host> ALWAYS goes through gpuq"), this script will NOT launch anything
itself. Review and run this command yourself (it re-invokes this script
with the internal confirmation marker set, which performs the actual
GPU-inclusive interleaved run; --rust-bin must point at a binary built with
'cargo build --release --features cuda'):

EXONIT_BENCH_GPU=1 EXONIT_BENCH_GPU_CONFIRMED=1 gpuq run --name rust-ab-bench --idle-mib 1000 --max-wait 600 bash $REPO_ROOT/scripts/run_ab_bench.sh --fits-dir $FITS_DIR --rounds $ROUNDS --iterations $ITERATIONS --models-dir $MODELS_DIR --rust-bin $RUST_BIN --python $PYTHON_BIN --worker-dir $WORKER_DIR${OUTPUT_DIR:+ --output-dir $OUTPUT_DIR}

Note: NO "--" before "bash" in the gpuq invocation -- gpuq's own argument
parser fails with one.
EOF
    exit 0
fi

if [[ -z "$OUTPUT_DIR" ]]; then
    OUTPUT_DIR="$REPO_ROOT/bench_results/ab_$(date +%Y%m%dT%H%M%S)_gpu${EXONIT_BENCH_GPU}"
fi
mkdir -p "$OUTPUT_DIR"

if [[ ! -x "$RUST_BIN" ]]; then
    echo "run_ab_bench.sh: rust binary not found/executable at $RUST_BIN" >&2
    echo "  build it first: cargo build --release --bin exonit-bench$( [[ "$EXONIT_BENCH_GPU" == "1" ]] && echo ' --features cuda' )" >&2
    exit 1
fi
if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "run_ab_bench.sh: python interpreter not found/executable at $PYTHON_BIN" >&2
    exit 1
fi

echo "run_ab_bench.sh: $ROUNDS interleaved rounds, $ITERATIONS iterations/stage, GPU=$EXONIT_BENCH_GPU" >&2
echo "run_ab_bench.sh: output dir: $OUTPUT_DIR" >&2

ROUND_FILES=()
for round in $(seq 1 "$ROUNDS"); do
    ts="$(date +%Y%m%dT%H%M%S)"

    rust_out="$OUTPUT_DIR/rust_round${round}_${ts}.json"
    echo "run_ab_bench.sh: round $round/$ROUNDS -- rust" >&2
    "$RUST_BIN" --fits-dir "$FITS_DIR" --iterations "$ITERATIONS" --models-dir "$MODELS_DIR" --output "$rust_out"
    ROUND_FILES+=("$rust_out")

    python_out="$OUTPUT_DIR/python_round${round}_${ts}.json"
    echo "run_ab_bench.sh: round $round/$ROUNDS -- python" >&2
    EXONIT_WORKER_DIR="$WORKER_DIR" "$PYTHON_BIN" "$SCRIPT_DIR/benchmark_python_stages.py" \
        --fits-dir "$FITS_DIR" --iterations "$ITERATIONS" --models-dir "$MODELS_DIR" \
        --worker-dir "$WORKER_DIR" --output "$python_out"
    ROUND_FILES+=("$python_out")
done

summary_out="$OUTPUT_DIR/summary.json"
echo "run_ab_bench.sh: computing adjacent-pair ratios -> $summary_out" >&2
"$PYTHON_BIN" "$SCRIPT_DIR/_compute_ab_ratios.py" "${ROUND_FILES[@]}" > "$summary_out"
cat "$summary_out" >&2

echo "run_ab_bench.sh: done. Round JSONs + summary.json under $OUTPUT_DIR" >&2
