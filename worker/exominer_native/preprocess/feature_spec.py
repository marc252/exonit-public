"""Feature spec loader for the ExoMiner ONNX model.

Reads ``worker/models/exominer_predict_config.yaml`` and exposes the
``features_set`` mapping so that downstream preprocessing code can stay
in sync with the model's expected inputs without hard-coding names or
shapes.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def default_feature_spec_path() -> Path:
    """Return the canonical path to the ExoMiner prediction config YAML.

    Resolves relative to this file's location so the path is correct
    regardless of the working directory.

    Layout assumption::

        <repo_root>/                          ← parents[3]
            worker/                           ← parents[2]
                models/
                    exominer_predict_config.yaml   ← target
                exominer_native/
                    preprocess/
                        feature_spec.py            ← this file
    """
    return Path(__file__).resolve().parents[2] / "models" / "exominer_predict_config.yaml"


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def load_feature_spec(yaml_path: "Path | str") -> dict[str, dict]:
    """Load the ``features_set`` mapping from *yaml_path*.

    Parameters
    ----------
    yaml_path:
        Path to ``exominer_predict_config.yaml`` (or any YAML that contains
        a top-level ``features_set`` key).

    Returns
    -------
    dict[str, dict]
        Mapping of feature name → ``{'dim': list[int], 'dtype': str}``.

    Raises
    ------
    FileNotFoundError
        If *yaml_path* does not exist.
    KeyError
        If the YAML does not contain a ``features_set`` key.
    """
    yaml_path = Path(yaml_path)
    if not yaml_path.exists():
        raise FileNotFoundError(
            f"ExoMiner feature spec not found at: {yaml_path}"
        )

    with yaml_path.open("r") as fh:
        config = yaml.safe_load(fh)

    return config["features_set"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_onnx_input_names(features_spec: dict) -> list[str]:
    """Return the list of ONNX input tensor names (== feature keys).

    Parameters
    ----------
    features_spec:
        Mapping returned by :func:`load_feature_spec`.
    """
    return list(features_spec.keys())


# Dtype mapping from YAML string to numpy dtype
_DTYPE_MAP: dict[str, type] = {
    "float": np.float32,
    "float_scalar": np.float32,
    "int": np.int64,
    "int_scalar": np.int64,
}


def empty_feed(features_spec: dict) -> dict[str, np.ndarray]:
    """Build a zero-filled feed dict matching the declared feature shapes.

    Useful as a sanity-check input to ``onnxruntime.InferenceSession.run``.

    Parameters
    ----------
    features_spec:
        Mapping returned by :func:`load_feature_spec`.

    Returns
    -------
    dict[str, np.ndarray]
        ``{feature_name: np.zeros(tuple(dim), dtype=<mapped_dtype>)}`` for
        every feature in *features_spec*.  Unknown dtype strings fall back to
        ``np.float32``.
    """
    feed: dict[str, np.ndarray] = {}
    for name, spec in features_spec.items():
        shape = tuple(spec["dim"])
        dtype = _DTYPE_MAP.get(str(spec["dtype"]), np.float32)
        feed[name] = np.zeros(shape, dtype=dtype)
    return feed
