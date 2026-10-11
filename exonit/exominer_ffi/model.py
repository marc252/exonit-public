"""PyTorch implementation of ExoMiner-FFI (Phase 3 spec §4).

Multi-branch network: three 1D-conv branches (global, local, centroid)
+ one MLP branch (stellar + tce features). The centroid branch is
multiplicatively gated by ``centroid_mask`` so QLP rows contribute
zero to the head — the model learns to weight that branch only when
TGLC data is present.

The forward pass returns logits (no sigmoid). Apply ``torch.sigmoid()``
at inference.
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn


def _conv_stack(in_channels, channels=(16, 32), kernel_size=5, pool_size=2):
    layers = []
    c_in = in_channels
    for c_out in channels:
        layers += [
            nn.Conv1d(c_in, c_out, kernel_size=kernel_size,
                      padding=kernel_size // 2),
            nn.ReLU(),
            nn.Conv1d(c_out, c_out, kernel_size=kernel_size,
                      padding=kernel_size // 2),
            nn.ReLU(),
            nn.MaxPool1d(pool_size),
        ]
        c_in = c_out
    return nn.Sequential(*layers), c_in


class ExoMinerFFI(nn.Module):
    """ExoMiner-FFI multi-branch classifier."""

    def __init__(self, dropout: float = 0.2, conv_dropout: float = 0.1):
        super().__init__()

        # Convolutional branches. ``aux_conv`` ingests 3 channels
        # (odd-local, even-local, secondary-local) — the discriminative
        # views that separate planets from eclipsing binaries / FPs.
        self.global_conv, _ = _conv_stack(1, channels=(16, 32, 64),
                                          kernel_size=5)
        self.local_conv, _ = _conv_stack(1, channels=(16, 32),
                                         kernel_size=5)
        self.centroid_conv, _ = _conv_stack(1, channels=(16, 32),
                                            kernel_size=5)
        self.aux_conv, _ = _conv_stack(3, channels=(16, 32),
                                       kernel_size=5)

        # Compute flattened sizes via dummy passes
        with torch.no_grad():
            g_flat = self.global_conv(torch.zeros(1, 1, 201)).numel()
            l_flat = self.local_conv(torch.zeros(1, 1, 61)).numel()
            c_flat = self.centroid_conv(torch.zeros(1, 1, 61)).numel()
            a_flat = self.aux_conv(torch.zeros(1, 3, 61)).numel()

        self._g_flat = g_flat
        self._l_flat = l_flat
        self._c_flat = c_flat
        self._a_flat = a_flat

        # Meta MLP for stellar (5) + tce (4)
        self.meta_mlp = nn.Sequential(
            nn.Linear(5 + 4, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
        )

        self.conv_dropout = nn.Dropout(conv_dropout)

        # Head over concat of all branches
        head_in = g_flat + l_flat + c_flat + a_flat + 32
        self.head = nn.Sequential(
            nn.Linear(head_in, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, global_view, local_view, centroid_view,
                centroid_mask, stellar, tce, aux_views):
        """Returns logits of shape (B,).

        ``centroid_mask`` is (B, 1) float in {0., 1.}. The centroid
        contribution is multiplied by the mask, so a zero-mask row
        passes zeros through the centroid branch. ``aux_views`` is
        (B, 3, 61): odd-local, even-local, secondary-local.
        """
        g = self.global_conv(global_view).flatten(1)
        l = self.local_conv(local_view).flatten(1)
        c = self.centroid_conv(centroid_view).flatten(1)
        c = c * centroid_mask
        a = self.aux_conv(aux_views).flatten(1)
        m = self.meta_mlp(torch.cat([stellar, tce], dim=1))
        x = torch.cat([g, l, c, a, m], dim=1)
        x = self.conv_dropout(x)
        return self.head(x).squeeze(-1)


def export_to_onnx(model: ExoMinerFFI, path) -> None:
    """Export an ExoMinerFFI model to ONNX with named inputs/outputs.

    Worker code (Phase 3d) will load the resulting .onnx via
    ``onnxruntime``.
    """
    model.eval()
    dummy = (
        torch.zeros(1, 1, 201),
        torch.zeros(1, 1, 61),
        torch.zeros(1, 1, 61),
        torch.zeros(1, 1),
        torch.zeros(1, 5),
        torch.zeros(1, 4),
        torch.zeros(1, 3, 61),
    )
    input_names = ["global_view", "local_view", "centroid_view",
                   "centroid_mask", "stellar", "tce", "aux_views"]
    output_names = ["logit"]
    dynamic_axes = {n: {0: "batch"} for n in input_names + output_names}

    torch.onnx.export(
        model,
        dummy,
        str(Path(path)),
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=17,
    )
