"""FFI CNN classifier — drop-in replacement for cnn_astronet on FFI data.

Mirror of cnn_astronet.py but loads the FFI-trained ONNX model.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort

# Reuse view computation. Two layouts supported:
#   <host>/<host>: ~/exonit/exonit/cnn_ffi_views.py
#   <host>/<host>:    ~/exonit-worker/cnn_ffi_views.py (flat layout)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    from exonit.cnn_ffi_views import global_view, local_view, ViewError
except ImportError:
    from cnn_ffi_views import global_view, local_view, ViewError

from onnx_providers import select_providers, describe_active_provider

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "cnn_ffi_v1.onnx")


class FFITransitClassifier:
    def __init__(self, model_path: str | None = None, gpu_enabled: bool = True,
                intra_op_num_threads: int | None = None):
        path = model_path or MODEL_PATH
        if not os.path.exists(path):
            raise FileNotFoundError(f"FFI CNN model not found at {path}")
        # `gpu_enabled` defaults to True so workers that opt in (cloud T4,
        # <host>, <host>) automatically use CUDA once onnxruntime-gpu is
        # installed. A CUDA host ending up on CPU is flagged by the worker's
        # startup GPU consistency check (worker.py).
        providers = select_providers(gpu_enabled=gpu_enabled)
        # `intra_op_num_threads` is None (ORT default: pick a thread count
        # itself, usually = physical cores) for every existing caller
        # (worker.py, rescore_cnn_ffi.py) -- zero behaviour change unless a
        # caller opts in. Added for scripts/eval/injection_e2e_runner.py's
        # multiprocessing pool, where each of N pool processes building its
        # own uncapped session would undermine the pool's own core-budget
        # cap (CLAUDE.md norm 2 -- never use 100% of a worker's CPUs).
        sess_options = None
        if intra_op_num_threads is not None:
            sess_options = ort.SessionOptions()
            sess_options.intra_op_num_threads = intra_op_num_threads
            sess_options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(path, sess_options=sess_options,
                                            providers=providers)
        self.backend = f"ONNX-{describe_active_provider(self.session)}"

    def classify(self, time, flux, period, t0, duration) -> float:
        try:
            g = global_view(time, flux, period, t0, duration).astype(np.float32)
            l = local_view(time, flux, period, t0, duration).astype(np.float32)
        except ViewError:
            return 0.0  # sparse → conservative reject
        feed = {
            "global_view": g.reshape(1, 1, -1),
            "local_view": l.reshape(1, 1, -1),
        }
        out = self.session.run(None, feed)[0]
        # Model exports logits; apply sigmoid for [0,1] probability.
        logit = float(np.array(out).flatten()[0])
        logit = max(min(logit, 60.0), -60.0)  # guard exp overflow
        return 1.0 / (1.0 + np.exp(-logit))
