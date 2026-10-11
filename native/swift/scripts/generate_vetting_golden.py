#!/usr/bin/env python3
import json
import sys

import numpy as np

sys.path.insert(0, "reference-python")
from vetting import vet_candidate


time = np.arange(0, 40, 0.05)
flux = np.ones_like(time)
for epoch in np.arange(1, 40, 5):
    flux[np.abs(time - epoch) < 0.15] -= 0.002
candidates = [
    {"candidate_num": 1, "period_days": 5.0, "t0": 1.0, "duration_days": 0.3, "depth_ppm": 2000.0, "snr": 20.0},
    {"candidate_num": 2, "period_days": 2.5, "t0": 1.0, "duration_days": 0.3, "depth_ppm": 2000.0, "snr": 10.0},
]

flux_odd_even = np.ones_like(time)
for index, epoch in enumerate(np.arange(1, 40, 4)):
    flux_odd_even[np.abs(time - epoch) < 0.15] -= 0.001 if index % 2 == 0 else 0.005
odd_even_candidates = [
    {"candidate_num": 1, "period_days": 4.0, "t0": 1.0, "duration_days": 0.3, "depth_ppm": 3000.0, "snr": 15.0}
]

result = {
    "time": time.tolist(),
    "flux": flux.tolist(),
    "candidates": candidates,
    "expected": [vet_candidate(candidate, candidates, time, flux) for candidate in candidates],
    "odd_even_flux": flux_odd_even.tolist(),
    "odd_even_candidates": odd_even_candidates,
    "odd_even_expected": vet_candidate(odd_even_candidates[0], odd_even_candidates, time, flux_odd_even),
}
with open("Fixtures/vetting_golden.json", "w", encoding="utf-8") as handle:
    json.dump(result, handle, separators=(",", ":"), default=lambda value: value.item())
