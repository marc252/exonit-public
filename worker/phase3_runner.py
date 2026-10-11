"""Always-on Phase 3 runner — extracted from worker.py for testability.

Called once per task, regardless of data_source. For data_source='ffi',
the loaded LC is used directly; for 'spoc', the FFI is fetched from
the server. Failures are logged and never propagate.

Score and fetch dependencies are injectable for testing — production
callers leave them None and the function imports them lazily.
"""
from __future__ import annotations

import logging
import os
from typing import Callable, Optional

DEFAULT_ONNX_FILENAME = "exominer_ffi_v6_ensemble.onnx"


def _resolve_onnx_path(config: dict) -> str:
    cfg_path = config.get("exominer_ffi_onnx_path")
    if cfg_path:
        return cfg_path
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, "models", DEFAULT_ONNX_FILENAME)


def _import_score_fn() -> Callable:
    try:
        from worker.exominer_ffi_inference import score_candidates_ffi
    except ImportError:
        from exominer_ffi_inference import score_candidates_ffi
    return score_candidates_ffi


def _import_fetch_fn() -> Callable:
    try:
        from worker.server_ffi_client import fetch_ffi
    except ImportError:
        from server_ffi_client import fetch_ffi
    return fetch_ffi


def run_phase3_always_on(
    *, tic_id: str, candidates: list,
    loaded_time, loaded_flux,
    data_source: str, star_meta: dict, config: dict,
    logger: logging.Logger,
    score_fn: Optional[Callable] = None,
    fetch_fn: Optional[Callable] = None,
) -> None:
    """Score every candidate with Phase 3 ExoMiner-FFI. Mutates in-place.

    On any failure (FFI unavailable, inference exception), logs a warning
    and leaves candidates without `exominer_ffi_score`.

    score_fn and fetch_fn are injectable for testing. Production callers
    leave them None — they will be lazily imported.
    """
    if not candidates or not tic_id:
        return

    if data_source == "ffi":
        ffi_t, ffi_f = loaded_time, loaded_flux
    else:
        if fetch_fn is None:
            fetch_fn = _import_fetch_fn()
        res = fetch_fn(
            config["server_url"], str(tic_id),
            token=config.get("worker_token")
                  or config.get("token", "exonit-default-token"),
            timeout=config.get("ffi_fetch_timeout", 30.0),
        )
        if res is None:
            logger.info("phase3 skipped TIC %s — no FFI available", tic_id)
            return
        ffi_t, ffi_f, _q = res

    if ffi_t is None or ffi_f is None:
        return

    if score_fn is None:
        score_fn = _import_score_fn()
    onnx_path = _resolve_onnx_path(config)
    try:
        summary = score_fn(
            tic_id=str(tic_id), candidates=candidates,
            time=ffi_t, flux=ffi_f, star_meta=star_meta,
            onnx_path=onnx_path,
            # Forward the worker's GPU flag so the ONNX session picks
            # CUDAExecutionProvider on hosts with onnxruntime-gpu.
            gpu_enabled=bool(config.get("gpu_enabled", True)),
        )
        if summary:
            logger.info(
                "  Phase3 TIC %s: %d/%d scored (data_source=%s)",
                tic_id, summary.get("n_matched", 0),
                summary.get("n_tces", 0), data_source,
            )
    except Exception as exc:
        logger.warning("phase3 scoring failed for TIC %s: %s", tic_id, exc)
