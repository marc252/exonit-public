"""ExoMiner inference on the worker side.

Loads the original NASA ExoMiner `.keras` model (``exominer++_single``
by default) and runs inference against a TFRecord file produced by the
server-side preprocessor (``exonit.exominer_preprocess``).

No model conversion (ONNX, TFLite, etc.). The worker runs the exact
same TensorFlow 2.13 + Keras 2.13 forward pass the container does, so
scores are numerically identical up to float precision. See the
Phase 5.4 validation gate (`scripts/exominer_validate_worker_inference.py`).

TensorFlow is imported lazily so tests and tooling that don't need
inference can import this module freely.
"""
from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

try:  # light dependency for tests
    import yaml  # type: ignore
except ImportError:  # pragma: no cover
    yaml = None  # type: ignore


DEFAULT_CONFIG_PATH = str(Path(__file__).resolve().parent / "models" / "exominer_predict_config.yaml")
# Default points at the SavedModel directory (more portable across platforms
# than the .keras file, which hits TF 2.13 load bugs on Windows). The .keras
# file can also be used by passing its path explicitly.
DEFAULT_MODEL_PATH = str(Path(__file__).resolve().parent / "models" / "exominer_saved")
DEFAULT_MODEL_KERAS = str(Path(__file__).resolve().parent / "models" / "exominer_plusplus_single.keras")


@dataclass
class TCEInfererResult:
    tic_id: str
    tce_num: int
    period: float
    t0: float
    depth_ppm: float
    score: float


# --------- config helpers ---------

def load_predict_config(path: str = DEFAULT_CONFIG_PATH) -> Dict[str, Any]:
    """Load the ExoMiner predict config YAML.

    Returns a dict with at least ``features_set`` and ``data_fields``.
    Raises RuntimeError if PyYAML is not available.
    """
    if yaml is None:
        raise RuntimeError("PyYAML not installed — required for loading predict config")
    with open(path) as f:
        return yaml.safe_load(f)


def build_feature_spec(features_set: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Return a name → {dim, dtype} spec dict suitable for TFRecord parsing.

    This is a pure-Python mapping — mapping to actual ``tf.*`` types
    happens in :func:`tf_feature_types` (which requires TF).
    """
    out: Dict[str, Dict[str, Any]] = {}
    for name, feat in features_set.items():
        out[name] = {"dim": list(feat["dim"]), "dtype": feat["dtype"]}
    return out


def _import_tf():
    """Import TensorFlow lazily; return the module or None if unavailable."""
    try:
        import tensorflow as tf  # type: ignore
        return tf
    except ImportError:
        return None


def tf_feature_types(features_set: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Map the string dtype in features_set to actual tf.* dtypes. Requires TF."""
    tf = _import_tf()
    if tf is None:
        raise RuntimeError("TensorFlow not installed — required for tf_feature_types")
    type_map = {"float": tf.float32, "int": tf.int64, "string": tf.string}
    return {name: type_map[feat["dtype"]] for name, feat in features_set.items()}


def parse_tce_table(csv_path: str) -> List[Dict[str, Any]]:
    """Parse the container's processed TCE CSV into TCE metadata records."""
    out: List[Dict[str, Any]] = []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            try:
                out.append({
                    "tic_id": str(row["target_id"]),
                    "tce_num": int(row["tce_plnt_num"]),
                    "period": float(row["tce_period"]),
                    "t0": float(row.get("tce_time0bk") or 0.0),
                    "duration": float(row.get("tce_duration") or 0.0),
                })
            except (KeyError, ValueError):
                continue
    return out


# --------- inference ---------

class ExoMinerInferer:
    """Load ExoMiner model once (lazy) and run inference against TFRecord files."""

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        config_path: str = DEFAULT_CONFIG_PATH,
    ):
        self.model_path = model_path
        self.config_path = config_path
        self._model = None
        self._config: Optional[Dict[str, Any]] = None

    @property
    def config(self) -> Dict[str, Any]:
        if self._config is None:
            self._config = load_predict_config(self.config_path)
        return self._config

    def load_model(self) -> None:
        if self._model is not None:
            return
        tf = _import_tf()
        if tf is None:
            raise RuntimeError("TensorFlow not installed — required for ExoMiner inference")
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"ExoMiner model not found at {self.model_path}")
        # ExoMiner's saved config refers to custom layer classes by qualified
        # name (e.g. "models.models_keras.Time2Vec"). We reproduce that
        # package at worker_dir/models/ (see worker/models/__init__.py +
        # models_keras.py + utils_models.py, copied from the NASA container)
        # and put worker_dir at the front of sys.path so the loader can
        # resolve the names.
        #
        # For the .keras archive format, TF 2.13 on Windows has a load bug
        # with this specific model (expected N variables, received 0). The
        # workaround is to ship the SavedModel directory instead — that's
        # what DEFAULT_MODEL_PATH points at.
        import sys as _sys
        model_path = Path(self.model_path)
        if model_path.is_file():
            worker_dir = str(model_path.resolve().parent.parent)
        else:
            worker_dir = str(model_path.resolve().parent.parent)
        if worker_dir not in _sys.path:
            _sys.path.insert(0, worker_dir)
        from models.models_keras import Time2Vec, SplitLayer, StdLayer
        custom_objects = {"Time2Vec": Time2Vec, "SplitLayer": SplitLayer, "StdLayer": StdLayer}
        with tf.keras.utils.custom_object_scope(custom_objects):
            self._model = tf.keras.models.load_model(str(model_path), compile=False)

    def _build_parse_fn(self):
        """Build a tf.io.parse_single_example fn based on the config.

        Mirrors the container's ``_prepare_data_fields`` + ``_example_parser``
        logic. N-D features with last dim > 1 are stored either as
        serialized tf.Tensor strings (the "unfolded_*" views) or as
        flattened float arrays. Scalar features follow ``data_fields``.
        """
        import functools as _ft
        tf = _import_tf()
        if tf is None:  # pragma: no cover
            raise RuntimeError("TensorFlow not installed")
        features_set = self.config["features_set"]
        data_fields = self.config.get("data_fields", {})
        type_map = {"float": tf.float32, "int": tf.int64, "string": tf.string}
        scalar_type_map = {
            "string": tf.string,
            "int_scalar": tf.int64,
            "float_scalar": tf.float32,
        }

        feature_desc: Dict[str, Any] = {}
        tensor_feats = set()
        for name, feat in features_set.items():
            dim = list(feat["dim"])
            dtype = type_map[feat["dtype"]]
            if len(dim) > 1 and dim[-1] > 1:
                # Container stores ALL such features as serialized tf.Tensor
                # strings (1-element bytes_list). Regardless of whether the
                # name contains "unfolded".
                feature_desc[name] = tf.io.FixedLenFeature(1, tf.string)
                tensor_feats.add(name)
            else:
                feature_desc[name] = tf.io.FixedLenFeature(dim, dtype)
        for name, fld in data_fields.items():
            feature_desc[name] = tf.io.FixedLenFeature(shape=[], dtype=scalar_type_map[fld])

        def _parse(serialized):
            parsed = tf.io.parse_single_example(serialized, feature_desc)
            inputs = {}
            for name, feat in features_set.items():
                value = parsed[name]
                if name in tensor_feats:
                    tensor = tf.io.parse_tensor(value[0], out_type=type_map[feat["dtype"]])
                    value = tf.reshape(tensor, feat["dim"])
                inputs[name] = value
            metadata = {name: parsed[name] for name in data_fields}
            return inputs, metadata

        return _parse

    def infer_from_tfrecord(self, tfrecord_path: str) -> List[TCEInfererResult]:
        """Run inference over all examples in ``tfrecord_path``.

        Returns one :class:`TCEInfererResult` per TCE example in the file.
        """
        self.load_model()
        tf = _import_tf()
        if tf is None:  # pragma: no cover
            raise RuntimeError("TensorFlow not installed")
        parse_fn = self._build_parse_fn()
        dataset = tf.data.TFRecordDataset(tfrecord_path).map(parse_fn)

        results: List[TCEInfererResult] = []
        for features, metadata in dataset:
            # Add batch dim for predict
            batched = {k: tf.expand_dims(v, axis=0) for k, v in features.items()}
            score_arr = self._model(batched, training=False).numpy()
            score = float(score_arr.reshape(-1)[0])

            tic_id = str(int(metadata["target_id"].numpy())) if "target_id" in metadata else ""
            tce_num = int(metadata["tce_plnt_num"].numpy()) if "tce_plnt_num" in metadata else 0
            period = float(metadata["tce_period"].numpy()) if "tce_period" in metadata else 0.0
            t0 = float(metadata["tce_time0bk"].numpy()) if "tce_time0bk" in metadata else 0.0
            depth_ppm = float(metadata["tce_depth"].numpy()) if "tce_depth" in metadata else 0.0

            results.append(TCEInfererResult(
                tic_id=tic_id, tce_num=tce_num, period=period, t0=t0,
                depth_ppm=depth_ppm, score=score,
            ))

        return results
