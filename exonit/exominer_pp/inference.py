"""Inference wrapper for the ExoMiner++ 2.0 FFI CV ensemble.

Loads NASA's released 5-fold CV ensemble
(``models_cv/cv_iter_{0..4}/ensemble_model/ensemble_avg_model.keras``) and
scores a *normalized* feature dict (one that satisfies
``exominer_pp.contract``). Each fold is itself an averaging ensemble; the
final ExoMiner++ score is the mean of the 5 folds' sigmoid outputs (this is
the ``cv-super-mean-ensemble`` NASA documents).

Serving notes
-------------
The model is Keras 2.13 / TF 2.13 with a custom ``IdentityConv1DInitializer``
layer, and a 56-input dict signature. ONNX export was attempted; the
multi-input dict + custom initializer + the way the FC head is wired make a
clean ONNX export non-trivial and, more importantly, *unnecessary* for this
path: ExoMiner++ FFI scoring is an opt-in, low-volume server-side step on
s>=36 TICs (a handful of TCEs per TIC), not a per-target worker hot loop.
We therefore serve the native Keras model in the dedicated TF 2.13 venv —
identical to how NASA's own Docker pipeline serves it — which guarantees
numerical parity with the published reference scores. (The ExoNit *worker*
ONNX stack is reserved for the high-volume flux models; ExoMiner++ never runs
on the worker because it needs DV products the worker doesn't fetch.)

This module imports TensorFlow lazily so the rest of ExoNit (and the test
suite) can import the package without a TF dependency.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np

from .contract import EXOMINER_PP_INPUTS, verify_feature_dict

from exonit.config import config as _config

_PROJECT_ROOT = _config.project_root

# Pinned model identifier written alongside every score.
EXOMINER_PP_MODEL_VERSION = "exominerpp2.0_ffi_cv-super-mean-ensemble_zenodo-17707413"

DEFAULT_ENSEMBLE_DIR = Path(
    os.environ.get(
        "EXONIT_EXOMINER_PP_MODELS",
        os.path.join(_PROJECT_ROOT, "extern", "exominerpp_zenodo", "models_cv"),
    )
)
CV_FOLDS = 5
_FOLD_REL = "ensemble_model/ensemble_avg_model.keras"


def _import_tf():
    import tensorflow as tf  # noqa: WPS433  (lazy)

    tf.config.set_visible_devices([], "GPU")  # CPU-only; tiny inference
    return tf


def _register_custom_layers() -> None:
    """Make NASA's custom Keras objects importable for model deserialization.

    The model uses ``IdentityConv1DInitializer`` from
    ``models.custom_layers`` in NASA's repo. We add the cloned repo to
    sys.path (env override ``EXONIT_EXOMINER_REPO``) and import it so Keras
    can resolve the custom object on load.
    """
    import sys

    repo = os.environ.get(
        "EXONIT_EXOMINER_REPO", os.path.join(_PROJECT_ROOT, "extern", "ExoMiner"))
    if repo not in sys.path:
        sys.path.insert(0, repo)
    try:
        import models.custom_layers  # noqa: F401  (registers custom objects)
    except Exception as e:  # pragma: no cover - environment dependent
        raise RuntimeError(
            f"could not import models.custom_layers from {repo!r}; set "
            "EXONIT_EXOMINER_REPO to the cloned NASA ExoMiner repo"
        ) from e


def fold_paths(ensemble_dir: Optional[Path] = None) -> List[Path]:
    """Return the 5 CV-fold model paths (cv_iter_0..4)."""
    base = ensemble_dir or DEFAULT_ENSEMBLE_DIR
    return [base / f"cv_iter_{i}" / _FOLD_REL for i in range(CV_FOLDS)]


class ExoMinerPPEnsemble:
    """Lazy-loaded 5-fold ExoMiner++ FFI CV ensemble scorer."""

    def __init__(self, ensemble_dir: Optional[Path] = None, use_predict: bool = False):
        self.ensemble_dir = Path(ensemble_dir) if ensemble_dir else DEFAULT_ENSEMBLE_DIR
        # use_predict=True forces the heavier model.predict() path; the default
        # (eager __call__) is faster for the small batches this scores.
        self.use_predict = use_predict
        self._models: Optional[list] = None
        self._input_names: Optional[List[str]] = None

    def load(self) -> "ExoMinerPPEnsemble":
        """Load all available CV folds. Idempotent."""
        if self._models is not None:
            return self
        tf = _import_tf()
        _register_custom_layers()
        models = []
        for fp in fold_paths(self.ensemble_dir):
            if not fp.exists():
                raise FileNotFoundError(f"ExoMiner++ CV fold missing: {fp}")
            models.append(tf.keras.models.load_model(str(fp), compile=False))
        if not models:
            raise FileNotFoundError(
                f"no CV folds found under {self.ensemble_dir}"
            )
        self._models = models
        # Unique input layer names from the first fold (authoritative).
        names = []
        for t in models[0].inputs:
            names.append(t.name.split("/")[0].split(":")[0])
        # de-dup preserving order
        seen = set()
        self._input_names = [n for n in names if not (n in seen or seen.add(n))]
        return self

    @property
    def input_names(self) -> List[str]:
        if self._input_names is None:
            self.load()
        return list(self._input_names or [])

    def _prepare_batch(
        self, features: Mapping[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """Build the model-input dict (batched, float32) from a feature dict.

        ``features`` may be a single example (unbatched) or already batched.
        Verifies the contract first.
        """
        verify_feature_dict(features)
        batch: Dict[str, np.ndarray] = {}
        for name, shape in EXOMINER_PP_INPUTS.items():
            arr = np.asarray(features[name], dtype=np.float32)
            if tuple(arr.shape) == tuple(shape):  # unbatched -> add batch dim
                arr = arr[None, ...]
            batch[name] = arr
        return batch

    def predict(
        self, features: Mapping[str, np.ndarray], *, per_fold: bool = False
    ):
        """Score one (or a batch of) normalized feature dict(s).

        Args:
            features: name -> array mapping satisfying the contract.
            per_fold: if True also return the per-fold scores.

        Returns:
            ``mean_score`` (np.ndarray, shape (B,)) — the CV-super-mean
            ExoMiner++ score in [0, 1]. If ``per_fold`` also returns the
            (5, B) per-fold array.
        """
        self.load()
        batch = self._prepare_batch(features)
        fold_scores = []
        for m in self._models:  # type: ignore[union-attr]
            # Direct __call__ avoids the per-invocation tf.data + graph-trace
            # overhead of model.predict(), which dominates for the small
            # batches ExoMiner++ scores (a handful of TCEs per TIC). Errors are
            # NOT swallowed — a failed forward pass must surface, not silently
            # fall back to a path that could score differently. To use the
            # heavier predict() path explicitly, set use_predict=True on the
            # ensemble.
            if self.use_predict:
                out = np.asarray(m.predict(batch, verbose=0)).reshape(-1)
            else:
                out = np.asarray(m(batch, training=False)).reshape(-1)
            fold_scores.append(out)
        folds = np.stack(fold_scores, axis=0)  # (n_folds, B)
        mean = folds.mean(axis=0)
        if per_fold:
            return mean, folds
        return mean


def score_features(
    features: Mapping[str, np.ndarray],
    ensemble_dir: Optional[Path] = None,
) -> float:
    """One-shot: score a single normalized feature dict → scalar [0,1]."""
    ens = ExoMinerPPEnsemble(ensemble_dir).load()
    mean = ens.predict(features)
    return float(np.asarray(mean).reshape(-1)[0])
