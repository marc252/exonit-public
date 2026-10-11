"""Tests for worker.exominer_inference.

Tests are split into two tiers:

- **Structural tests** (always run): load config, dataclasses, feature
  spec construction, no-ops without TensorFlow installed.

- **TF tests** (skipped when TF is missing): actual model loading and
  inference. These run only where ``tensorflow`` is importable — the
  workers, CI nodes with TF, or the validation script.
"""
import csv
from pathlib import Path

import pytest

try:
    import tensorflow as tf  # noqa
    HAS_TF = True
except ImportError:
    HAS_TF = False

import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "worker"))

from exominer_inference import (
    ExoMinerInferer,
    TCEInfererResult,
    build_feature_spec,
    load_predict_config,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_FP = ROOT / "worker" / "models" / "exominer_predict_config.yaml"


# ---------- structural ----------

def test_config_yaml_exists():
    assert CONFIG_FP.exists(), f"Predict config missing at {CONFIG_FP}"


def test_load_predict_config_has_expected_keys():
    cfg = load_predict_config(str(CONFIG_FP))
    assert "features_set" in cfg
    assert "data_fields" in cfg
    # Must include at least the core views
    expected = {
        "unfolded_local_flux_view_fluxnorm",
        "global_flux_view_fluxnorm",
        "local_flux_view_fluxnorm",
        "local_flux_odd_view_fluxnorm",
        "local_flux_even_view_fluxnorm",
    }
    assert expected.issubset(cfg["features_set"].keys())


def test_load_predict_config_features_have_dim_and_dtype():
    cfg = load_predict_config(str(CONFIG_FP))
    for name, feat in cfg["features_set"].items():
        assert "dim" in feat, f"{name} missing dim"
        assert "dtype" in feat, f"{name} missing dtype"
        assert feat["dtype"] in ("float", "int")


def test_build_feature_spec_without_tf_returns_dict():
    """build_feature_spec builds a name→spec dict suitable for TFRecord parsing."""
    features_set = {
        "global_flux_view_fluxnorm": {"dim": [301, 1], "dtype": "float"},
        "tce_period": {"dim": [1], "dtype": "float"},
    }
    spec = build_feature_spec(features_set)
    assert "global_flux_view_fluxnorm" in spec
    assert "tce_period" in spec
    # Each spec entry should carry dim + dtype (not yet tf-typed)
    assert spec["global_flux_view_fluxnorm"]["dim"] == [301, 1]
    assert spec["global_flux_view_fluxnorm"]["dtype"] == "float"


def test_tce_inferer_result_dataclass():
    r = TCEInfererResult(
        tic_id="382437043", tce_num=1, period=14.64, t0=1337.75,
        depth_ppm=245.1, score=0.87,
    )
    assert r.tic_id == "382437043"
    assert r.score == pytest.approx(0.87)


def test_inferer_instantiates_without_loading_model(tmp_path):
    """Constructor is lazy — doesn't touch the model file until inference."""
    # Fake model file path (doesn't exist); constructor should NOT fail
    inf = ExoMinerInferer(
        model_path=str(tmp_path / "nonexistent.keras"),
        config_path=str(CONFIG_FP),
    )
    assert inf.model_path.endswith(".keras")
    assert inf._model is None  # not yet loaded


def test_inferer_reports_tf_missing_before_load(tmp_path, monkeypatch):
    """If TF is unavailable, load_model raises a clear error."""
    inf = ExoMinerInferer(
        model_path=str(tmp_path / "x.keras"),
        config_path=str(CONFIG_FP),
    )
    # Force the "TF missing" code path regardless of local state
    monkeypatch.setattr("exominer_inference._import_tf", lambda: None)
    with pytest.raises(RuntimeError, match="TensorFlow"):
        inf.load_model()


# ---------- TF-gated ----------

@pytest.mark.skipif(not HAS_TF, reason="TensorFlow not installed on this host")
def test_feature_spec_maps_to_tf_types():
    """With TF available, the dtype strings map to tf.float32 / tf.int64."""
    from exominer_inference import tf_feature_types
    types = tf_feature_types({
        "a": {"dim": [1], "dtype": "float"},
        "b": {"dim": [1], "dtype": "int"},
    })
    assert types["a"] == tf.float32
    assert types["b"] == tf.int64


@pytest.mark.skipif(not HAS_TF, reason="TensorFlow not installed on this host")
def test_parse_tce_table_returns_metadata(tmp_path):
    """Parse the container's tce_table CSV and return per-TCE metadata."""
    from exominer_inference import parse_tce_table
    csv_fp = tmp_path / "tces.csv"
    with open(csv_fp, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["target_id", "tce_plnt_num", "tce_period", "tce_time0bk", "tce_duration"])
        w.writerow(["382437043", 1, "14.64", "1337.75", "0.106"])
        w.writerow(["382437043", 2, "1142.82", "2041.79", "0.543"])
    tces = parse_tce_table(str(csv_fp))
    assert len(tces) == 2
    assert tces[0]["tic_id"] == "382437043"
    assert tces[0]["tce_num"] == 1
    assert tces[0]["period"] == pytest.approx(14.64)
