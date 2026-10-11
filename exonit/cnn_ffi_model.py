"""PyTorch implementation of AstroNet-Triage (Yu+ 2019).

Dual-view (global 201 + local 61). 1D conv stacks per view, concat,
dense head, sigmoid output.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def _conv_stack(in_channels, channels=(16, 32), kernel_size=5, pool_size=2):
    layers = []
    c_in = in_channels
    for c_out in channels:
        layers += [
            nn.Conv1d(c_in, c_out, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.ReLU(),
            nn.Conv1d(c_out, c_out, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.ReLU(),
            nn.MaxPool1d(pool_size),
        ]
        c_in = c_out
    return nn.Sequential(*layers), c_in


class AstroNetTriage(nn.Module):
    def __init__(self, dropout: float = 0.2, conv_dropout: float = 0.1):
        super().__init__()
        self.global_conv, g_out = _conv_stack(1, channels=(16, 32, 64), kernel_size=5)
        self.local_conv, l_out = _conv_stack(1, channels=(16, 32), kernel_size=5)
        # Compute flattened sizes via dummy pass
        with torch.no_grad():
            g = self.global_conv(torch.zeros(1, 1, 201))
            l = self.local_conv(torch.zeros(1, 1, 61))
            self._g_flat = g.numel()
            self._l_flat = l.numel()
        self.conv_dropout = nn.Dropout(conv_dropout)
        self.fc = nn.Sequential(
            nn.Linear(self._g_flat + self._l_flat, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, global_view, local_view):
        """Returns logits (no sigmoid). Apply torch.sigmoid() at inference."""
        g = self.global_conv(global_view).flatten(1)
        l = self.local_conv(local_view).flatten(1)
        x = self.conv_dropout(torch.cat([g, l], dim=1))
        return self.fc(x).squeeze(-1)
