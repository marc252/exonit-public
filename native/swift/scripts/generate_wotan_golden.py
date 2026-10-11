#!/usr/bin/env python3
"""Generate a deterministic Wotan biweight golden vector; local only."""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT / "reference-python"))
from wotan_detrend import detrend_with_wotan

rng = np.random.RandomState(24680)
time = np.arange(300, dtype=np.float64) * 0.02
time[150:] += 1.0
flux = 1.0 + 0.008 * np.sin(time * 1.7) + 0.002 * np.cos(time * 0.31)
flux += rng.normal(0.0, 0.0004, size=time.size)
phase = np.mod(time - 0.17, 1.23)
transit = phase < 0.08
flux[transit] -= 0.012
flux[73] = np.nan
time[219] = np.nan
mask = transit.copy()

plain = detrend_with_wotan(time, flux)
masked = detrend_with_wotan(time, flux, transit_mask=mask)

def values(a):
    return [None if not np.isfinite(x) else float(x) for x in a]

payload = {
    "time": values(time),
    "flux": values(flux),
    "transit_mask": mask.tolist(),
    "plain": values(plain),
    "masked": values(masked),
    "parameters": {"window_length": 0.5, "edge_cutoff": 0.5, "break_tolerance": 0.5, "cval": 5.0},
}
out = ROOT / "Fixtures" / "wotan_biweight_golden.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(payload, separators=(",", ":")) + "\n")
print(out)
