import numpy as np
import pytest
import torch

from exonit.cnn_ffi_model import AstroNetTriage


def test_forward_shape():
    m = AstroNetTriage()
    g = torch.randn(4, 1, 201)
    l = torch.randn(4, 1, 61)
    out = m(g, l)
    assert out.shape == (4,), f"Expected (4,), got {out.shape}"


def test_forward_outputs_finite_logits():
    """Model returns logits (no sigmoid in forward). Sigmoid is applied at
    inference (worker/cnn_ffi.py) and training (BCEWithLogitsLoss).
    Switching to logits is required for autocast bf16 safety."""
    m = AstroNetTriage()
    m.eval()
    g = torch.randn(2, 1, 201)
    l = torch.randn(2, 1, 61)
    with torch.no_grad():
        out = m(g, l)
    assert out.shape == (2,)
    assert torch.isfinite(out).all()


def test_param_count_in_range():
    """Architecture restored to plan-original (16,32,64)+(16,32) → ~600K params."""
    m = AstroNetTriage()
    n = sum(p.numel() for p in m.parameters() if p.requires_grad)
    assert n < 1_500_000, f"Too many params: {n}"
    assert n > 200_000, f"Too few params: {n}"
