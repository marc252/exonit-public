"""CoreML wrapper for the CNN classifiers.

Exposes `try_load_mlpackage(onnx_path)` that returns a callable with the
same interface as `onnxruntime.InferenceSession.run`, or None if we are
not on macOS or the .mlpackage alongside the ONNX file does not exist.

Using the MLProgram format lets the model land on the Apple Neural
Engine (ANE) on M1+ hosts — the `cnn.TransitClassifier` runtime drops
from milliseconds to microseconds per inference, and more importantly
the CPU is freed for BLS and FPP on workers where `gpu_enabled=false`.
"""

from __future__ import annotations

import os
import platform
from typing import Callable, Optional

import numpy as np


def mlpackage_path_for(onnx_path: str) -> str:
    """Convention: same directory, same base name, `.mlpackage` suffix."""
    base, _ = os.path.splitext(onnx_path)
    return base + ".mlpackage"


def try_load_mlpackage(onnx_path: str):
    """Return a (run, input_name_map) pair or None.

    `run(inputs_dict)` returns a list of output numpy arrays so callers
    can swap it for `onnxruntime.InferenceSession.run(None, inputs)`.
    """
    if platform.system() != "Darwin":
        return None
    mlpath = mlpackage_path_for(onnx_path)
    if not os.path.exists(mlpath):
        return None
    try:
        import coremltools as ct
    except ImportError:
        return None

    model = ct.models.MLModel(mlpath, compute_units=ct.ComputeUnit.ALL)
    # Map from ONNX input names (what the rest of the codebase uses) to
    # the CoreML input names (which may be identical or mangled).
    spec = model.get_spec()
    ct_input_names = [inp.name for inp in spec.description.input]
    ct_output_names = [out.name for out in spec.description.output]
    onnx_input_names_meta = model.user_defined_metadata.get(
        "onnx_input_names", "")
    onnx_input_names = onnx_input_names_meta.split(",") if onnx_input_names_meta else ct_input_names
    name_map = dict(zip(onnx_input_names, ct_input_names))

    def run(inputs: dict) -> list:
        payload = {}
        for onnx_name, arr in inputs.items():
            ct_name = name_map.get(onnx_name, onnx_name)
            payload[ct_name] = np.asarray(arr, dtype=np.float32)
        out = model.predict(payload)
        # Preserve the output ordering from the CoreML spec so callers
        # can index with result[0] the way they do with onnxruntime.
        return [np.asarray(out[name]) for name in ct_output_names]

    return run
