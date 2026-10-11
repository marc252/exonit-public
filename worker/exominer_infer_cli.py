"""Standalone CLI for ExoMiner inference, to run from an isolated TF venv.

Why a separate CLI: TensorFlow 2.13 pins numpy<1.25, which conflicts
with scipy/wotan/triceratops in the worker's main venv. We put TF in
its own venv (``venv_tf``) and expose inference through this CLI so
the main worker doesn't need to import TF at all. The main worker
calls:

    subprocess.run([venv_tf_python, "exominer_infer_cli.py",
                    "--tfrecord", TFRECORD, "--model", MODEL,
                    "--config", CONFIG])

and parses the JSON on stdout.

Exit codes:
  0 — success (JSON printed to stdout)
  2 — missing inputs / TF not available
  3 — model load / inference error
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict


def main() -> int:
    ap = argparse.ArgumentParser(description="Run ExoMiner inference on a TFRecord.")
    ap.add_argument("--tfrecord", required=True, help="Path to the TFRecord file")
    ap.add_argument("--model", help="Path to the .keras model (default: worker/models/exominer_plusplus_single.keras)")
    ap.add_argument("--config", help="Path to predict config YAML")
    ap.add_argument("--tic-id", help="Optional TIC to include in each result record")
    args = ap.parse_args()

    if not os.path.exists(args.tfrecord):
        print(json.dumps({"error": f"tfrecord not found: {args.tfrecord}"}), file=sys.stderr)
        return 2

    try:
        # importing locally so a --help invocation doesn't pay for TF load
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from exominer_inference import ExoMinerInferer, DEFAULT_MODEL_PATH, DEFAULT_CONFIG_PATH
    except ImportError as e:
        print(json.dumps({"error": f"exominer_inference import failed: {e}"}), file=sys.stderr)
        return 2

    model_path = args.model or DEFAULT_MODEL_PATH
    config_path = args.config or DEFAULT_CONFIG_PATH

    try:
        inferer = ExoMinerInferer(model_path=model_path, config_path=config_path)
        tces = inferer.infer_from_tfrecord(args.tfrecord)
    except RuntimeError as e:
        print(json.dumps({"error": f"runtime: {e}"}), file=sys.stderr)
        return 2
    except Exception as e:  # noqa: BLE001 — CLI boundary
        print(json.dumps({"error": f"inference failed: {type(e).__name__}: {e}"}), file=sys.stderr)
        return 3

    output = {
        "tic_id": args.tic_id,
        "model": os.path.basename(model_path),
        "n_tces": len(tces),
        "tces": [asdict(t) for t in tces],
    }
    print(json.dumps(output))
    return 0


if __name__ == "__main__":
    sys.exit(main())
