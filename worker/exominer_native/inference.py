"""ONNX Runtime wrapper for the ExoMiner++ single-ensemble model."""
from __future__ import annotations

from typing import List, Optional

import numpy as np


class ExoMinerOnnxInferer:
    """Lazy-loaded ONNX Runtime inference session for ExoMiner++.

    Providers preference (first available wins):
      CUDA -> CoreML -> CPU.
    """

    _DEFAULT_PROVIDERS: List[str] = [
        "CUDAExecutionProvider",
        "CoreMLExecutionProvider",
        "CPUExecutionProvider",
    ]

    def __init__(self, onnx_path: str, providers: Optional[List[str]] = None):
        self._onnx_path = onnx_path
        self._providers = providers or self._DEFAULT_PROVIDERS
        self._session = None  # type: ignore[assignment]

    def _load(self) -> None:
        """Build the inference session.

        Tries the configured provider list as-is first; on failure (e.g.
        Apple's CoreML/Espresso compiler choking on degenerate shapes —
        observed on M1: ``Zero dimension in output shape``), retry with
        only CPUExecutionProvider so we still get a score.
        """
        import onnxruntime as ort
        try:
            self._session = ort.InferenceSession(
                self._onnx_path, providers=self._providers)
        except Exception:
            if self._providers == ["CPUExecutionProvider"]:
                raise
            self._session = ort.InferenceSession(
                self._onnx_path, providers=["CPUExecutionProvider"])

    def active_provider(self) -> str:
        if self._session is None:
            self._load()
        return self._session.get_providers()[0]  # type: ignore[union-attr]

    def score(self, feeds: dict) -> float:
        """Run a single forward pass. Returns the scalar sigmoid output.

        Feeds are auto-adjusted to match the session's declared input shapes:
          - extra batch dims prepended when ``arr.ndim < expected_rank``
          - NaN/Inf replaced with 0.0 so the graph can't propagate NaNs
        """
        if self._session is None:
            self._load()
        fixed = {}
        for inp in self._session.get_inputs():  # type: ignore[union-attr]
            name = inp.name
            if name not in feeds:
                raise KeyError(f"missing ONNX feed: {name!r}")
            arr = np.asarray(feeds[name], dtype=np.float32)
            expected_rank = len(inp.shape) if inp.shape is not None else arr.ndim
            while arr.ndim < expected_rank:
                arr = arr[np.newaxis, ...]
            if not np.isfinite(arr).all():
                arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
            fixed[name] = arr
        out = self._session.run(None, fixed)  # type: ignore[union-attr]
        return float(np.asarray(out[0]).reshape(-1)[0])
