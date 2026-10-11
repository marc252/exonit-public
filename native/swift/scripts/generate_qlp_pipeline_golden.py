#!/usr/bin/env python3
"""Generate deterministic QLP stitch/clean/Wotan golden; no network."""
import json
from pathlib import Path
import sys

import lightkurve as lk
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "reference-python"))
from wotan_detrend import detrend_with_wotan
from bls_fast import bls_fast

paths = [
    "~/exonit-worker/fits_cache_7/hlsp_qlp_tess_ffi_s0013-0000000300868410_tess_v01_llc.fits",
    "~/exonit-worker/fits_cache_7/hlsp_qlp_tess_ffi_s0039-0000000300868410_tess_v01_llc.fits",
]
lcs = [lk.read(p) for p in paths]
lc = lk.LightCurveCollection(lcs).stitch()
lc = lc.remove_nans().remove_outliers(sigma_upper=5, sigma_lower=float("inf"))
max_gap = float(np.max(np.diff(lc.time.value)))
assert max_gap >= 5, max_gap

time = np.asarray(lc.time.value, np.float64)
raw = np.asarray(lc.flux.value, np.float64)
flat = detrend_with_wotan(time, raw)
finite = np.isfinite(time) & np.isfinite(flat)
quick = bls_fast(time[finite], flat[finite], period_min=0.5, period_max=50.0, n_periods=2000, n_bins=100, n_dur_steps=5)
if quick["snr"] >= 7.0:
    phase = ((time - quick["t0"]) % quick["period"]) / quick["period"]
    half_dur = quick["duration"] / quick["period"]
    transit_mask = (phase < half_dur) | (phase > 1.0 - half_dur)
    flat = detrend_with_wotan(time, raw, transit_mask=transit_mask)
order = np.argsort(time)
time, raw, flat = time[order], raw[order], flat[order]
valid = np.isfinite(time) & np.isfinite(raw) & np.isfinite(flat)
time, raw, flat = time[valid], raw[valid], flat[valid]

def vals(a): return [float(x) for x in a]
payload = {
    "paths": paths,
    "max_gap": max_gap,
    "time": vals(time),
    "raw": vals(raw),
    "flat": vals(flat),
}
out = ROOT / "Fixtures" / "qlp_pipeline_golden.json"
out.write_text(json.dumps(payload, separators=(",", ":")) + "\n")
print(json.dumps({"path": str(out), "n": len(time), "max_gap": max_gap, "sum_raw": float(raw.sum()), "sum_flat": float(flat.sum())}))
