"""ExoMiner-FFI worker-side inferer.

Mirror of worker/cnn_ffi.py but for the ExoMiner-FFI multi-branch model
(global + local + centroid + stellar + tce). Used when the routing
detects ``data_source == "ffi"``; the SPOC ExoMiner++ pipeline (DV XML
+ orchestrate.run_tic) is not invoked because FFI products lack DV.
"""
from __future__ import annotations

import logging
import json
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import onnxruntime as ort

# Reuse view computation from cnn_ffi_views (same in Phase 2 worker layout).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    from exonit.cnn_ffi_views import (
        GLOBAL_BINS, LOCAL_BINS, ViewError, aux_views, global_view, local_view)
except ImportError:
    from cnn_ffi_views import (
        GLOBAL_BINS, LOCAL_BINS, ViewError, aux_views, global_view, local_view)

from onnx_providers import select_providers, describe_active_provider

MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "models", "exominer_ffi_v6_ensemble.onnx")
# Bumped when the default FFI model changes; written to each candidate.
FFI_MODEL_VERSION = "phase3_v6_ensemble_cal"

_INFERER_CACHE: dict[str, "ExoMinerFFIInferer"] = {}


def _get_inferer(path: str, gpu_enabled: bool = True,
                 intra_op_num_threads: Optional[int] = None) -> "ExoMinerFFIInferer":
    """Return a cached ExoMinerFFIInferer; lazy load on first call.

    The cache key includes the GPU flag (and the thread cap) so a CPU-only
    and a GPU-enabled session -- or two different thread-cap configs -- for
    the same model don't collide (in practice a worker only ever uses one
    mode, but explicit is safer than implicit).
    """
    resolved = path or MODEL_PATH
    cache_key = f"{resolved}|gpu={gpu_enabled}|threads={intra_op_num_threads}"
    if cache_key not in _INFERER_CACHE:
        _INFERER_CACHE[cache_key] = ExoMinerFFIInferer(
            resolved, gpu_enabled=gpu_enabled,
            intra_op_num_threads=intra_op_num_threads)
    return _INFERER_CACHE[cache_key]


def _normalise_stellar(tmag, teff, logg, rstar, mstar=None):
    """Match the same normalisation as exonit.exominer_ffi.preprocess._normalise_stellar."""
    def _z(v, mean, scale):
        if v is None:
            return 0.0
        try:
            v = float(v)
        except (TypeError, ValueError):
            return 0.0
        if not np.isfinite(v):
            return 0.0
        return float((v - mean) / scale)

    if mstar is None or (isinstance(mstar, float) and not np.isfinite(mstar)):
        try:
            l = float(logg) if logg is not None else float("nan")
            r = float(rstar) if rstar is not None else float("nan")
        except (TypeError, ValueError):
            l = r = float("nan")
        if np.isfinite(l) and np.isfinite(r):
            mstar = (10 ** l) * r * r / (10 ** 4.44)
        else:
            mstar = float("nan")

    return np.array([
        _z(tmag, 12.0, 2.5),
        _z(teff, 5500.0, 1500.0),
        _z(logg, 4.4, 0.5),
        _z(rstar, 1.0, 0.5),
        _z(mstar, 1.0, 0.5),
    ], dtype=np.float32)


def _normalise_tce(period_d, depth_ppm, duration_d, snr):
    return np.array([
        np.log10(max(float(period_d), 0.1)),
        np.log10(max(float(depth_ppm), 1.0)),
        np.log10(max(float(duration_d) * 24.0, 0.1)),
        np.log10(max(float(snr), 1.0)),
    ], dtype=np.float32)


class ExoMinerFFIInferer:
    """Lazy-loaded ONNX runtime session + scoring helper.

    All inputs are derived from data the worker already has after BLS:
    light curve (time, flux), TCE params (period, t0, duration, depth, snr),
    and stellar params from server metadata.

    Centroid is currently unavailable to the worker (Phase 1 dataset has
    no centroid columns); we always feed zeros + mask=0. The model was
    trained that way and ignores the centroid branch under that mask.
    """

    def __init__(self, model_path: Optional[str] = None, gpu_enabled: bool = True,
                intra_op_num_threads: Optional[int] = None):
        path = model_path or MODEL_PATH
        if not os.path.exists(path):
            raise FileNotFoundError(f"ExoMiner-FFI model not found at {path}")
        # Default to GPU-on so a CUDA host (cloud T4, <host>, <host>)
        # with onnxruntime-gpu installed picks the CUDAExecutionProvider
        # automatically (gpu_enabled=true ending up on CPU is flagged by the
        # worker startup GPU consistency check for the eager classifiers;
        # this one inits lazily, so log the backend explicitly here).
        providers = select_providers(gpu_enabled=gpu_enabled)
        # `intra_op_num_threads` is None (ORT default) for every existing
        # caller -- zero behaviour change unless a caller opts in. Added for
        # scripts/eval/injection_e2e_runner.py's multiprocessing pool (see
        # cnn_ffi.py::FFITransitClassifier's matching parameter docstring).
        sess_options = None
        if intra_op_num_threads is not None:
            sess_options = ort.SessionOptions()
            sess_options.intra_op_num_threads = intra_op_num_threads
            sess_options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(path, sess_options=sess_options,
                                            providers=providers)
        self.backend = f"ONNX-{describe_active_provider(self.session)}"
        logging.getLogger("exonit").info(
            "ExoMiner-FFI inferer loaded (%s, gpu_enabled=%s)", self.backend, gpu_enabled)
        if gpu_enabled and "CPU" in self.backend:
            logging.getLogger("exonit").error(
                "GPU CONSISTENCY: gpu_enabled=true but ExoMiner-FFI inference on %s",
                self.backend)
        self.model_path = path
        # Models trained with discriminative odd/even/secondary views expose
        # an ``aux_views`` input; older models do not. Feed it only when present.
        self.input_names = {i.name for i in self.session.get_inputs()}
        self.expects_aux = "aux_views" in self.input_names
        # Optional monotone calibration map living next to the model
        # (``<model>.calib.json``). Maps the raw sigmoid prob to a
        # calibrated prob so the 0.5 gate matches the validated recall
        # operating point. Absent (e.g. v1) -> raw scores, unchanged.
        self.calib_x = self.calib_y = None
        calib_path = os.path.splitext(path)[0] + ".calib.json"
        if os.path.exists(calib_path):
            try:
                cal = json.load(open(calib_path))
                self.calib_x = np.asarray(cal["x"], dtype=np.float64)
                self.calib_y = np.asarray(cal["y"], dtype=np.float64)
            except Exception:
                self.calib_x = self.calib_y = None

    def score(self, time, flux, period, t0, duration, depth_ppm, snr,
              stellar: Optional[dict] = None) -> float:
        try:
            g = global_view(time, flux, period, t0, duration).astype(np.float32)
            l = local_view(time, flux, period, t0, duration).astype(np.float32)
        except ViewError:
            return 0.0  # sparse → conservative reject

        c = np.zeros(LOCAL_BINS, dtype=np.float32)
        mask = np.zeros((1,), dtype=np.float32)

        stellar = stellar or {}
        s = _normalise_stellar(
            stellar.get("tmag"), stellar.get("teff"), stellar.get("logg"),
            stellar.get("rstar"), stellar.get("mstar"),
        )
        t = _normalise_tce(period, depth_ppm, duration, snr)

        feeds = {
            "global_view": g.reshape(1, 1, -1),
            "local_view": l.reshape(1, 1, -1),
            "centroid_view": c.reshape(1, 1, -1),
            "centroid_mask": mask.reshape(1, 1),
            "stellar": s.reshape(1, -1),
            "tce": t.reshape(1, -1),
        }
        if self.expects_aux:
            try:
                a = aux_views(time, flux, period, t0, duration).astype(np.float32)
            except ViewError:
                a = np.zeros((3, LOCAL_BINS), dtype=np.float32)
            feeds["aux_views"] = a.reshape(1, 3, -1)
        out = self.session.run(None, feeds)[0]
        logit = float(np.array(out).flatten()[0])
        logit = max(min(logit, 60.0), -60.0)  # guard exp overflow
        prob = 1.0 / (1.0 + np.exp(-logit))
        if self.calib_x is not None:
            prob = float(np.interp(prob, self.calib_x, self.calib_y))
        return prob


def score_candidates_ffi(*, tic_id: str, candidates: list, time, flux,
                         star_meta: dict, onnx_path: Optional[str] = None,
                         gpu_enabled: bool = True,
                         intra_op_num_threads: Optional[int] = None) -> dict:
    """Score every candidate in-place with ``exominer_ffi_score``.

    Mirror of `worker.exominer_native.hook.score_candidates_native` but
    much simpler — no DV XML, no orchestrate.run_tic, no TCE matching
    (we score each BLS candidate directly).

    Writes ``exominer_ffi_score`` (float [0,1]) and
    ``exominer_ffi_model_version`` (str) to each candidate dict.
    Deliberately does NOT write ``exominer_score`` so the two sources
    of truth (SPOC ExoMiner++ vs FFI ONNX) stay separated.

    ``intra_op_num_threads``: forwarded to the ONNX session (None = ORT
    default, unchanged behaviour for the live worker). See
    ``ExoMinerFFIInferer``'s matching parameter docstring.
    """
    if not candidates:
        return {"n_tces": 0, "n_matched": 0}

    inferer = _get_inferer(onnx_path, gpu_enabled=gpu_enabled,
                           intra_op_num_threads=intra_op_num_threads)
    stellar = {
        "tmag": star_meta.get("tmag"),
        "teff": star_meta.get("teff"),
        "logg": star_meta.get("logg"),
        "rstar": star_meta.get("stellar_radius"),
        "mstar": None,
    }

    n_matched = 0
    for cand in candidates:
        period = float(cand.get("period_days") or cand.get("period") or 0.0)
        t0 = float(cand.get("t0") or 0.0)
        duration = float(cand.get("duration_days") or 0.0)
        depth_ppm = float(cand.get("depth_ppm") or 0.0)
        snr = float(cand.get("snr") or 0.0)
        if period <= 0 or duration <= 0:
            continue
        score = inferer.score(time, flux, period, t0, duration,
                              depth_ppm, snr, stellar=stellar)
        cand["exominer_ffi_score"] = score
        cand["exominer_ffi_model_version"] = FFI_MODEL_VERSION
        n_matched += 1

    # ExoMiner-FFI v8 shadow scoring was RETIRED 2026-06-27: the binding hold-out
    # on 107k real sector-1 candidates showed v8 fails the gate (floods the faint
    # tmag>13.5 domain 2.5x more than v6 and is MORE brightness-biased there), so
    # v8 was never promoted and writing exominer_ffi_score_v8 only added per-
    # candidate overhead. The v8 ensemble ONNX was removed from worker/models/.
    # TODO: retry the v7-unblock retrain with MORE / cleaner training data before
    # any future shadow run (see memory project_ffi_v8_holdout_verdict).

    return {"n_tces": len(candidates), "n_matched": n_matched}
