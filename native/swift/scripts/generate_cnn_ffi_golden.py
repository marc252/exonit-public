#!/usr/bin/env python3
import json
import sys

import numpy as np
import onnxruntime as ort

sys.path.insert(0, "reference-python")
from cnn_ffi_views import aux_views, global_view, local_view

period = 7.25
t0 = 1.125
duration = 0.22
time = np.arange(0.0, 80.0, 0.02, dtype=np.float64)
flux = 1.0 + 0.00015 * np.sin(time * 1.731) + 0.00008 * np.cos(time * 0.311)
center = t0 + duration / 2.0
phase_days = np.mod(time - center + period / 2.0, period) - period / 2.0
flux[np.abs(phase_days) < duration / 2.0] -= 0.003

global_values = global_view(time, flux, period, t0, duration)
local_values = local_view(time, flux, period, t0, duration)
aux_values = aux_views(time, flux, period, t0, duration)
stellar_raw = {"tmag": 10.7, "teff": 5772.0, "logg": 4.35, "rstar": 1.12}
mstar = (10 ** stellar_raw["logg"]) * stellar_raw["rstar"] ** 2 / (10 ** 4.44)
stellar = np.asarray([
    (stellar_raw["tmag"] - 12.0) / 2.5,
    (stellar_raw["teff"] - 5500.0) / 1500.0,
    (stellar_raw["logg"] - 4.4) / 0.5,
    (stellar_raw["rstar"] - 1.0) / 0.5,
    (mstar - 1.0) / 0.5,
], dtype=np.float32)
tce = np.asarray([
    np.log10(max(period, 0.1)),
    np.log10(3000.0),
    np.log10(max(duration * 24.0, 0.1)),
    np.log10(18.0),
], dtype=np.float32)
session = ort.InferenceSession(
    "~/exonit-worker/models/cnn_ffi_v1.onnx",
    providers=["CPUExecutionProvider"],
)
logit = float(session.run(None, {
    "global_view": global_values.reshape(1, 1, -1),
    "local_view": local_values.reshape(1, 1, -1),
})[0].reshape(-1)[0])
probability = 1.0 / (1.0 + np.exp(-np.clip(logit, -60.0, 60.0)))
exominer_session = ort.InferenceSession(
    "~/exonit-worker/models/exominer_ffi_v6_ensemble.onnx",
    providers=["CPUExecutionProvider"],
)
exominer_logit = float(exominer_session.run(None, {
    "global_view": global_values.reshape(1, 1, -1),
    "local_view": local_values.reshape(1, 1, -1),
    "centroid_view": np.zeros((1, 1, 61), dtype=np.float32),
    "centroid_mask": np.zeros((1, 1), dtype=np.float32),
    "stellar": stellar.reshape(1, -1),
    "tce": tce.reshape(1, -1),
    "aux_views": aux_values.reshape(1, 3, -1),
})[0].reshape(-1)[0])
exominer_raw_probability = 1.0 / (1.0 + np.exp(-np.clip(exominer_logit, -60.0, 60.0)))
calibration_x = np.asarray([0.0, 0.38930313157914914, 1.0])
calibration_y = np.asarray([0.0, 0.5, 1.0])
exominer_probability = float(np.interp(exominer_raw_probability, calibration_x, calibration_y))
with open("Fixtures/cnn_ffi_golden.json", "w", encoding="utf-8") as handle:
    json.dump({
        "period": period,
        "t0": t0,
        "duration": duration,
        "time": time.tolist(),
        "flux": flux.tolist(),
        "global_view": global_values.tolist(),
        "local_view": local_values.tolist(),
        "aux_views": aux_values.tolist(),
        "stellar_raw": stellar_raw,
        "stellar": stellar.tolist(),
        "tce": tce.tolist(),
        "logit": logit,
        "probability": probability,
        "exominer_logit": exominer_logit,
        "exominer_raw_probability": exominer_raw_probability,
        "exominer_probability": exominer_probability,
    }, handle, separators=(",", ":"))
