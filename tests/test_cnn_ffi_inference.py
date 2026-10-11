"""Tests for worker/cnn_ffi.py inference."""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "worker"))


@pytest.fixture
def dummy_onnx(tmp_path):
    sys.path.insert(0, str(ROOT))
    from exonit.cnn_ffi_model import AstroNetTriage
    m = AstroNetTriage(); m.eval()
    out = tmp_path / "cnn_ffi.onnx"
    g = torch.zeros(1, 1, 201); l = torch.zeros(1, 1, 61)
    torch.onnx.export(m, (g, l), out,
        input_names=["global_view", "local_view"], output_names=["score"],
        dynamic_axes={"global_view": {0: "batch"}, "local_view": {0: "batch"}, "score": {0: "batch"}},
        opset_version=17)
    return out


def test_classifier_returns_score_in_unit_range(dummy_onnx):
    from cnn_ffi import FFITransitClassifier
    rng = np.random.default_rng(0)
    time = np.linspace(0, 27, 3000)
    flux = np.ones_like(time) + 0.001 * rng.standard_normal(3000)
    in_t = np.abs((time - 1.0) % 2.0) < 0.05
    flux[in_t] -= 0.02
    clf = FFITransitClassifier(model_path=str(dummy_onnx))
    score = clf.classify(time, flux, period=2.0, t0=1.0, duration=0.1)
    assert 0.0 <= score <= 1.0
