#!/usr/bin/env python3
"""Generate iterative quick-BLS + transit-masked Wotan golden."""
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "reference-python"))
from bls_fast import bls_fast
from wotan_detrend import detrend_with_wotan

n = 800
time = np.arange(n, dtype=np.float64) * 0.02
period = 2.37
t0 = 0.31
phase = ((time - t0) % period) / period
transit = (phase < 0.035) | (phase > 1 - 0.035)
trend = 1 + 0.003*np.sin(2*np.pi*time/4.7) + 0.0004*np.cos(2*np.pi*time/0.83)
flux = trend.copy()
flux[transit] -= 0.012
first = detrend_with_wotan(time, flux)
finite = np.isfinite(first)
quick = bls_fast(time[finite], first[finite], period_min=0.5, period_max=50, n_periods=2000, n_bins=100, n_dur_steps=5)
quick_phase = ((time - quick["t0"]) % quick["period"]) / quick["period"]
half_dur = quick["duration"] / quick["period"]
mask = (quick_phase < half_dur) | (quick_phase > 1-half_dur)
final = detrend_with_wotan(time, flux, transit_mask=mask)

def encode(a): return [None if not np.isfinite(x) else float(x) for x in a]
out = {
  "time": encode(time), "flux": encode(flux), "first": encode(first), "final": encode(final),
  "quick": {k: float(quick[k]) for k in ("period","power","duration","t0","depth","snr")},
  "mask": [bool(x) for x in mask],
}
path = ROOT / "Fixtures" / "iterative_detrend_golden.json"
path.write_text(json.dumps(out, separators=(",", ":")) + "\n")
print(json.dumps({"path":str(path),"period":quick["period"],"snr":quick["snr"],"masked":int(mask.sum())}))
