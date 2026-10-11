"""CNN-based transit candidate classification using ONNX Runtime.

On macOS with CoreML installed and a sibling `.mlpackage` next to the
ONNX file, inference is routed through Apple's Neural Engine via
coremltools. See `cnn_coreml.try_load_mlpackage`.
"""

import os
import numpy as np
import onnxruntime as ort

from cnn_coreml import try_load_mlpackage
from onnx_providers import select_providers, describe_active_provider
from phase_fold import phase_fold

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "transit_cnn.onnx")
N_BINS = 200


class TransitClassifier:
    def __init__(self, model_path=None, use_gpu=False):
        path = model_path or MODEL_PATH
        if not os.path.exists(path):
            raise FileNotFoundError(f"CNN model not found at {path}")
        self._coreml_run = try_load_mlpackage(path)
        if self._coreml_run:
            self.backend = "CoreML"
        else:
            # Honour the caller's `use_gpu` flag. With onnxruntime-gpu
            # installed and `use_gpu=True`, ORT routes through CUDA;
            # otherwise it falls back to the CPU EP transparently.
            providers = select_providers(gpu_enabled=use_gpu)
            self.session = ort.InferenceSession(path, providers=providers)
            self.backend = f"ONNX-{describe_active_provider(self.session)}"

    def classify(self, time, flux, period, t0, duration):
        phase_flux = self._phase_fold_and_bin(time, flux, period, t0, duration)
        input_data = phase_flux.reshape(1, 1, N_BINS).astype(np.float32)
        inputs = {"phase_folded_flux": input_data}
        if self._coreml_run is not None:
            result = self._coreml_run(inputs)
        else:
            result = self.session.run(None, inputs)
        return float(result[0][0][0])

    def _phase_fold_and_bin(self, time, flux, period, t0, duration):
        # Shared with vetting.py (identical convention) — see
        # docs/reviews/2026-07-02-opus48-cleanup-review.md item D5 and
        # tests/worker/test_phase_fold_golden.py for the bit-identical proof.
        phase = phase_fold(time, t0, period, fold_order="mod_div", center="mask")
        bins = np.linspace(-0.5, 0.5, N_BINS + 1)
        binned = np.ones(N_BINS)
        for i in range(N_BINS):
            mask = (phase >= bins[i]) & (phase < bins[i + 1])
            if np.sum(mask) >= 1:
                binned[i] = np.median(flux[mask])
        med = np.median(binned)
        std = np.std(binned)
        if std > 0:
            binned = (binned - med) / std
        return binned
