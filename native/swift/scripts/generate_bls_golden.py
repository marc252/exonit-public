#!/usr/bin/env python3
"""Generate the deterministic BLS golden vector used by Swift contract tests.

This is local-only and does not contact the ExoNit server.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "reference-python"))
from bls_fast import bls_fast  # noqa: E402

period = 10.0 / 3.0
time = np.arange(360, dtype=np.float64) / 12.0
phase = np.mod(time / period, 1.0)
flux = np.ones_like(time)
flux[(phase >= 0.20) & (phase < 0.28)] -= 0.012
result = bls_fast(
    time,
    flux,
    period_min=2.0,
    period_max=5.0,
    n_periods=121,
    n_bins=50,
    n_dur_steps=7,
    duration_min_frac=0.04,
    duration_max_frac=0.12,
)
keys = ("period", "power", "duration", "t0", "depth", "depth_ppm", "snr")
print(json.dumps({key: result[key] for key in keys}, sort_keys=True, indent=2))
