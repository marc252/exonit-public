"""Baseline inference smoke test — loads the .onnx and runs a dummy forward pass."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pytest

from worker.exominer_native.inference import ExoMinerOnnxInferer

ONNX = Path(__file__).resolve().parents[3] / "worker" / "models" / "exominer_plusplus_single.onnx"

@pytest.mark.skipif(not ONNX.exists(), reason="ONNX model missing; regenerate with scripts/exominer_onnx_convert.py")
def test_inferer_loads_and_returns_score_for_dummy_feed():
    infr = ExoMinerOnnxInferer(str(ONNX), providers=["CPUExecutionProvider"])
    # Build a feed of zeros matching the ORT session's declared shapes
    sess = ort.InferenceSession(str(ONNX), providers=["CPUExecutionProvider"])
    feeds = {}
    for inp in sess.get_inputs():
        shape = [1 if isinstance(d, str) or d is None else d for d in inp.shape]
        feeds[inp.name] = np.zeros(shape, dtype=np.float32)
    score = infr.score(feeds)
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0
