#!/usr/bin/env python3
"""Versioned JSON bridge for Python-only ExoNit science components.

Secrets and large arrays arrive on stdin, never in argv. Stdout is exactly one
JSON object; third-party chatter is redirected to stderr.
"""
from __future__ import annotations

import contextlib
import json
import shutil
import sys
from pathlib import Path

PROTOCOL = 1
FPP_N_DEFAULT = 100_000


def emit(value: dict) -> None:
    sys.stdout.write(json.dumps(value, separators=(",", ":"), allow_nan=False) + "\n")
    sys.stdout.flush()


def configure_imports(payload: dict) -> Path:
    worker_dir = Path(payload["worker_python_dir"]).resolve(strict=True)
    if not worker_dir.is_dir():
        raise ValueError("worker_python_dir is not a directory")
    sys.path.insert(0, str(worker_dir))
    sys.path.insert(0, str(worker_dir.parent))
    return worker_dir


def run_preflight(payload: dict) -> dict:
    if payload.get("require_fpp"):
        with contextlib.redirect_stdout(sys.stderr):
            from fpp_native import calc_fpp_native  # noqa: F401
    if payload.get("require_exominer"):
        model = Path(payload["onnx_path"])
        if not model.is_file():
            raise FileNotFoundError(f"ExoMiner++ model not found: {model}")
        with contextlib.redirect_stdout(sys.stderr):
            try:
                from exominer_native.hook import (  # noqa: F401
                    prepare_exominer_inputs, score_prepared_exominer)
            except ImportError:
                from worker.exominer_native.hook import (  # noqa: F401
                    prepare_exominer_inputs, score_prepared_exominer)
    return {"protocol": PROTOCOL, "status": "ok"}


def run_fpp(payload: dict) -> dict:
    import numpy as np
    with contextlib.redirect_stdout(sys.stderr):
        from fpp_native import calc_fpp_native
    task = payload["task"]
    time_values = np.asarray(payload["time"], dtype=np.float64)
    flux_values = np.asarray(payload["flux"], dtype=np.float64)
    if time_values.ndim != 1 or flux_values.shape != time_values.shape or len(time_values) < 6:
        raise ValueError("invalid light curve arrays")
    sectors = np.arange(1, max(int(payload["sector_count"]), int(task.get("n_sectors", 1))) + 1)
    fpp_n = int(payload.get("fpp_n", FPP_N_DEFAULT))
    if fpp_n <= 0:
        raise ValueError("fpp_n must be positive")
    with contextlib.redirect_stdout(sys.stderr):
        result = calc_fpp_native(
            str(task["tic_id"]), float(task["period_days"]), float(task["t0"]),
            float(task["duration_days"]), float(task["depth_ppm"]),
            sectors, time_values, flux_values,
            N=fpp_n,
            workdir=str(payload["workdir"]), server_url=str(payload["server_url"]),
            token=str(payload["token"]), abort_check=lambda: False,
        )
    if not result:
        return {"protocol": PROTOCOL, "status": "failed", "fpp": -1.0,
                "nfpp": -1.0, "verdict": "failed"}
    return {
        "protocol": PROTOCOL, "status": "ok",
        "fpp": float(result["fpp"]), "nfpp": float(result["nfpp"]),
        "probability_planet": (None if result.get("prob_planet") is None
                               else float(result["prob_planet"])),
        "verdict": result.get("verdict"),
    }


def run_exominer(payload: dict) -> dict:
    with contextlib.redirect_stdout(sys.stderr):
        try:
            from exominer_native.hook import prepare_exominer_inputs, score_prepared_exominer
        except ImportError:
            from worker.exominer_native.hook import prepare_exominer_inputs, score_prepared_exominer
    task = payload["task"]
    candidate = {
        "candidate_num": int(task["candidate_num"]),
        "period_days": float(task["period_days"]), "t0": float(task["t0"]),
        "duration_days": float(task["duration_days"]),
        "depth_ppm": float(task["depth_ppm"]),
    }
    prepared = None
    try:
        with contextlib.redirect_stdout(sys.stderr):
            prepared = prepare_exominer_inputs(
                str(task["tic_id"]), [candidate],
                server_url=str(payload["server_url"]), token=str(payload["token"]),
            )
        if prepared is None:
            return {"protocol": PROTOCOL, "status": "skipped", "score": -1.0}
        with contextlib.redirect_stdout(sys.stderr):
            score_prepared_exominer(
                [candidate], prepared, onnx_path=str(payload["onnx_path"])
            )
        score = candidate.get("exominer_score")
        if score is None:
            return {"protocol": PROTOCOL, "status": "skipped", "score": -1.0}
        return {
            "protocol": PROTOCOL, "status": "ok", "score": float(score),
            "tce_period": candidate.get("exominer_tce_period"),
            "sector_run": candidate.get("exominer_sector_run"),
        }
    finally:
        if prepared and prepared.get("tmp_root"):
            shutil.rmtree(prepared["tmp_root"], ignore_errors=True)


def main() -> int:
    if sys.argv[1:] == ["--self-test"]:
        emit({"protocol": PROTOCOL, "status": "ok"})
        return 0
    if sys.argv[1:]:
        raise ValueError("no positional arguments accepted")
    payload = json.load(sys.stdin)
    if payload.get("protocol") != PROTOCOL:
        raise ValueError("unsupported protocol")
    configure_imports(payload)
    mode = payload.get("mode")
    if mode == "preflight":
        result = run_preflight(payload)
    elif mode == "fpp":
        result = run_fpp(payload)
    elif mode == "exominer":
        result = run_exominer(payload)
    else:
        raise ValueError("unsupported mode")
    emit(result)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # one structured failure, without echoing secret input
        emit({"protocol": PROTOCOL, "status": "error",
              "error_type": type(exc).__name__, "error": str(exc)[:2048]})
        raise SystemExit(1)
