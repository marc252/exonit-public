"""Smoke tests for worker/exominer_native/preprocess/feature_spec.py."""
from __future__ import annotations

import numpy as np
import pytest

from worker.exominer_native.preprocess.feature_spec import (
    default_feature_spec_path,
    empty_feed,
    get_onnx_input_names,
    load_feature_spec,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def spec() -> dict:
    """Load the canonical feature spec once for all tests in this module."""
    return load_feature_spec(default_feature_spec_path())


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestLoadFeatureSpec:
    def test_returns_non_empty_dict(self, spec):
        """Sanity: the YAML must have at least one feature entry."""
        assert isinstance(spec, dict)
        assert len(spec) > 0

    def test_each_entry_has_dim_and_dtype(self, spec):
        """Every feature must carry 'dim' (list) and 'dtype' (str) keys."""
        for name, entry in spec.items():
            assert "dim" in entry, f"Feature '{name}' missing 'dim'"
            assert "dtype" in entry, f"Feature '{name}' missing 'dtype'"
            assert isinstance(entry["dim"], list), (
                f"Feature '{name}' dim must be a list, got {type(entry['dim'])}"
            )
            assert len(entry["dim"]) >= 1, (
                f"Feature '{name}' dim must have at least one element"
            )

    def test_file_not_found_raises(self, tmp_path):
        """load_feature_spec must raise FileNotFoundError for missing paths."""
        missing = tmp_path / "nonexistent.yaml"
        with pytest.raises(FileNotFoundError, match="ExoMiner feature spec not found"):
            load_feature_spec(missing)


class TestGetOnnxInputNames:
    def test_returns_list_of_strings(self, spec):
        names = get_onnx_input_names(spec)
        assert isinstance(names, list)
        assert all(isinstance(n, str) for n in names)

    def test_names_match_spec_keys(self, spec):
        names = get_onnx_input_names(spec)
        assert names == list(spec.keys())

    def test_order_preserved(self, spec):
        """Keys must come back in insertion order (Python 3.7+)."""
        names = get_onnx_input_names(spec)
        assert names == list(spec.keys())


class TestEmptyFeed:
    def test_returns_dict_of_arrays(self, spec):
        feed = empty_feed(spec)
        assert isinstance(feed, dict)
        for name, arr in feed.items():
            assert isinstance(arr, np.ndarray), (
                f"Expected ndarray for '{name}', got {type(arr)}"
            )

    def test_shapes_match_spec(self, spec):
        feed = empty_feed(spec)
        for name, arr in feed.items():
            expected_shape = tuple(spec[name]["dim"])
            assert arr.shape == expected_shape, (
                f"Feature '{name}': expected shape {expected_shape}, got {arr.shape}"
            )

    def test_all_zeros(self, spec):
        feed = empty_feed(spec)
        for name, arr in feed.items():
            assert np.all(arr == 0), f"Feature '{name}' should be zero-filled"

    def test_float_features_are_float32(self, spec):
        feed = empty_feed(spec)
        for name, arr in feed.items():
            dtype_str = str(spec[name]["dtype"])
            if dtype_str in ("float", "float_scalar"):
                assert arr.dtype == np.float32, (
                    f"Feature '{name}': expected float32, got {arr.dtype}"
                )

    def test_int_features_are_int64(self, spec):
        """If there are any int-typed features, they should be int64."""
        feed = empty_feed(spec)
        for name, arr in feed.items():
            dtype_str = str(spec[name]["dtype"])
            if dtype_str in ("int", "int_scalar"):
                assert arr.dtype == np.int64, (
                    f"Feature '{name}': expected int64, got {arr.dtype}"
                )

    def test_keys_match_spec(self, spec):
        feed = empty_feed(spec)
        assert set(feed.keys()) == set(spec.keys())
