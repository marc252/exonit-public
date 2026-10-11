"""Unit tests for ExoMinerFFI (Phase 3b)."""
from __future__ import annotations

import numpy as np
import torch


# =============================================================================
# Construction + forward pass shape
# =============================================================================

def test_model_construction():
    """The model can be constructed with default args."""
    from exonit.exominer_ffi.model import ExoMinerFFI

    model = ExoMinerFFI()
    assert isinstance(model, torch.nn.Module)


def test_forward_pass_shape():
    """Forward pass: (B, ...) inputs → (B,) logits."""
    from exonit.exominer_ffi.model import ExoMinerFFI

    model = ExoMinerFFI()
    model.eval()
    B = 4
    out = model(
        global_view=torch.zeros(B, 1, 201),
        local_view=torch.zeros(B, 1, 61),
        centroid_view=torch.zeros(B, 1, 61),
        centroid_mask=torch.zeros(B, 1),
        stellar=torch.zeros(B, 5),
        tce=torch.zeros(B, 4),
        aux_views=torch.zeros(B, 3, 61),
    )
    assert out.shape == (B,)


# =============================================================================
# Parameter budget
# =============================================================================

def test_param_count_under_budget():
    """Spec §4: ≤ 1.5M total parameters."""
    from exonit.exominer_ffi.model import ExoMinerFFI

    model = ExoMinerFFI()
    n = sum(p.numel() for p in model.parameters())
    assert n <= 1_500_000, f"too many params: {n}"
    assert n >= 50_000, f"suspiciously few params: {n}"


# =============================================================================
# Centroid mask gating
# =============================================================================

def test_zero_mask_centroid_contributes_zero():
    """A zero-mask centroid branch must not influence the logit.

    With mask=0, the centroid branch output is multiplied by 0 → drops
    out entirely. Two batches differing only in centroid_view (mask=0
    in both) must produce identical logits.
    """
    from exonit.exominer_ffi.model import ExoMinerFFI

    torch.manual_seed(0)
    model = ExoMinerFFI()
    model.eval()

    B = 2
    base = dict(
        global_view=torch.randn(B, 1, 201),
        local_view=torch.randn(B, 1, 61),
        centroid_mask=torch.zeros(B, 1),
        stellar=torch.randn(B, 5),
        tce=torch.randn(B, 4),
        aux_views=torch.randn(B, 3, 61),
    )
    centroid_a = torch.randn(B, 1, 61)
    centroid_b = torch.randn(B, 1, 61)
    out_a = model(centroid_view=centroid_a, **base)
    out_b = model(centroid_view=centroid_b, **base)
    assert torch.allclose(out_a, out_b, atol=1e-6), \
        "zero-mask centroid still influences logit — gating is broken"


def test_unit_mask_centroid_changes_logit():
    """With mask=1, centroid input must influence the logit."""
    from exonit.exominer_ffi.model import ExoMinerFFI

    torch.manual_seed(0)
    model = ExoMinerFFI()
    model.eval()

    B = 2
    base = dict(
        global_view=torch.randn(B, 1, 201),
        local_view=torch.randn(B, 1, 61),
        centroid_mask=torch.ones(B, 1),
        stellar=torch.randn(B, 5),
        tce=torch.randn(B, 4),
        aux_views=torch.randn(B, 3, 61),
    )
    centroid_a = torch.zeros(B, 1, 61)
    centroid_b = torch.randn(B, 1, 61)
    out_a = model(centroid_view=centroid_a, **base)
    out_b = model(centroid_view=centroid_b, **base)
    assert not torch.allclose(out_a, out_b, atol=1e-3), \
        "unit-mask centroid does not influence logit — branch is dead"


# =============================================================================
# ONNX export round-trip
# =============================================================================

def test_onnx_export_round_trip(tmp_path):
    """Model exports to ONNX and reproduces the same output ±1e-4."""
    import onnxruntime as ort

    from exonit.exominer_ffi.model import ExoMinerFFI, export_to_onnx

    torch.manual_seed(0)
    model = ExoMinerFFI()
    model.eval()

    onnx_path = tmp_path / "exominer_ffi.onnx"
    export_to_onnx(model, onnx_path)
    assert onnx_path.exists()

    B = 2
    feeds = {
        "global_view": np.random.randn(B, 1, 201).astype(np.float32),
        "local_view": np.random.randn(B, 1, 61).astype(np.float32),
        "centroid_view": np.random.randn(B, 1, 61).astype(np.float32),
        "centroid_mask": np.ones((B, 1), dtype=np.float32),
        "stellar": np.random.randn(B, 5).astype(np.float32),
        "tce": np.random.randn(B, 4).astype(np.float32),
        "aux_views": np.random.randn(B, 3, 61).astype(np.float32),
    }

    with torch.no_grad():
        ref = model(**{k: torch.from_numpy(v) for k, v in feeds.items()}).numpy()

    sess = ort.InferenceSession(str(onnx_path),
                                providers=["CPUExecutionProvider"])
    onnx_out = sess.run(None, feeds)[0]
    assert onnx_out.shape == (B,)
    np.testing.assert_allclose(onnx_out, ref, atol=1e-4)
