"""Tests for exonit/verdict_routing.py (FFI/SPOC per-candidate routing) and
the EXONIT_VERDICT_FFI_VERSION branch it feeds in
exonit/verdict_classifier.py::predict_verdict.

docs/superpowers/specs/2026-09-19-verdict-v3-tfopwg-design.md §5.
"""
from __future__ import annotations

import sys
import types
from unittest.mock import patch

import pytest

from exonit.verdict_routing import (
    FFI_EXPTIME_THRESHOLD_S,
    data_source_from_exptime,
    resolve_data_source,
)


@pytest.fixture(autouse=True)
def _pin_verdict_env(monkeypatch):
    """run_nightly.py's load_dotenv() at import time can leak production
    env vars (EXONIT_VERDICT_VERSION=v2.1, etc.) into these tests — same
    rationale as tests/test_verdict_classifier.py's _pin_verdict_v1 fixture.
    Pin both dispatcher env vars to a known-empty/known-default state;
    individual tests override as needed."""
    monkeypatch.delenv("EXONIT_VERDICT_FFI_VERSION", raising=False)
    monkeypatch.setenv("EXONIT_VERDICT_VERSION", "v1")


# ---------------------------------------------------------------------------
# data_source_from_exptime
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("exptime,expected", [
    (120, "spoc"),
    (274, "spoc"),
    (400, "ffi"),
    (623, "ffi"),
    (1870, "ffi"),
    (None, "spoc"),
    ("x", "spoc"),
])
def test_data_source_from_exptime(exptime, expected):
    assert data_source_from_exptime(exptime) == expected


def test_threshold_constant_value():
    """Threshold is exactly 400.0s, as the spec's routing rule states."""
    assert FFI_EXPTIME_THRESHOLD_S == 400.0


def test_data_source_from_exptime_accepts_float_boundary():
    assert data_source_from_exptime(399.999) == "spoc"
    assert data_source_from_exptime(400.0) == "ffi"


# ---------------------------------------------------------------------------
# resolve_data_source
# ---------------------------------------------------------------------------

class _FakeCache:
    def __init__(self, entries=None, raise_exc=None):
        self._entries = entries or {}
        self._raise_exc = raise_exc

    def get_entry(self, tic_id):
        if self._raise_exc is not None:
            raise self._raise_exc
        return self._entries.get(str(tic_id))


def test_resolve_data_source_ffi_entry():
    cache = _FakeCache({"123": {"exptime": 1870.0}})
    assert resolve_data_source("123", cache) == "ffi"


def test_resolve_data_source_spoc_entry():
    cache = _FakeCache({"123": {"exptime": 120.0}})
    assert resolve_data_source("123", cache) == "spoc"


def test_resolve_data_source_no_entry_defaults_spoc():
    cache = _FakeCache({})
    assert resolve_data_source("999", cache) == "spoc"


def test_resolve_data_source_lookup_raises_defaults_spoc(caplog):
    cache = _FakeCache(raise_exc=RuntimeError("connection lost"))
    with caplog.at_level("WARNING"):
        result = resolve_data_source("123", cache)
    assert result == "spoc"
    assert any("resolve_data_source" in rec.message or "connection lost" in rec.message
               for rec in caplog.records)


# ---------------------------------------------------------------------------
# Dispatcher: exonit.verdict_classifier.predict_verdict data_source routing
# ---------------------------------------------------------------------------

def _sample_candidate():
    return {
        "period_days": 7.2, "depth_ppm": 800, "duration_days": 0.1, "snr": 12.5,
        "cnn_score": 0.82, "triceratops_fpp": 0.03, "exominer_score": 0.91,
        "exominer_tce_period": 7.2, "tmag": 10.5, "ra": 100.0, "dec": -30.0,
        "n_sectors": 3, "time_span_days": 730.0,
    }


@pytest.fixture
def _stub_v3_module(monkeypatch):
    """Install a fake exonit.verdict_classifier_v3_tfopwg module in
    sys.modules so the dispatcher's lazy `from exonit.verdict_classifier_v3_tfopwg
    import predict_v3_tfopwg` resolves without that module existing on disk
    yet (implemented by a parallel agent). Returns the mock so tests can
    assert on call args."""
    from unittest.mock import MagicMock
    fake_result = {"verdict": "planet_candidate", "confidence": 0.9,
                   "model_used": "v3_tfopwg"}
    mock_predict = MagicMock(return_value=fake_result)
    fake_module = types.ModuleType("exonit.verdict_classifier_v3_tfopwg")
    fake_module.predict_v3_tfopwg = mock_predict
    monkeypatch.setitem(sys.modules, "exonit.verdict_classifier_v3_tfopwg", fake_module)
    return mock_predict


def test_dispatcher_routes_ffi_to_v3_tfopwg(monkeypatch, _stub_v3_module):
    from exonit.verdict_classifier import predict_verdict

    monkeypatch.setenv("EXONIT_VERDICT_FFI_VERSION", "v3_tfopwg")
    candidate = _sample_candidate()
    vetting = {"passed": True}

    result = predict_verdict(candidate, vetting, star_info={}, data_source="ffi")

    _stub_v3_module.assert_called_once()
    assert result["verdict"] == "planet_candidate"
    assert result["model_used"] == "v3_tfopwg"


def test_dispatcher_ffi_version_set_but_spoc_source_keeps_v2_1(monkeypatch, _stub_v3_module):
    from exonit.verdict_classifier import predict_verdict

    monkeypatch.setenv("EXONIT_VERDICT_FFI_VERSION", "v3_tfopwg")
    monkeypatch.setenv("EXONIT_VERDICT_VERSION", "v2.1")
    candidate = _sample_candidate()
    vetting = {"passed": True}

    fake_v2_1_result = {"verdict": "uncertain", "confidence": 0.5,
                        "model_used": "v2.1"}
    with patch("exonit.verdict_classifier_v2_1.predict_v2_1",
              return_value=fake_v2_1_result) as mock_v2_1:
        result = predict_verdict(candidate, vetting, star_info={}, data_source="spoc")

    mock_v2_1.assert_called_once()
    _stub_v3_module.assert_not_called()
    assert result["model_used"] == "v2.1"


def test_dispatcher_empty_ffi_env_behaviour_unchanged(monkeypatch, _stub_v3_module):
    """EXONIT_VERDICT_FFI_VERSION unset (default) -> data_source is ignored,
    dispatcher falls straight through to EXONIT_VERDICT_VERSION (v2.1 here),
    never touching the v3_tfopwg branch even when data_source='ffi'."""
    from exonit.verdict_classifier import predict_verdict

    monkeypatch.delenv("EXONIT_VERDICT_FFI_VERSION", raising=False)
    monkeypatch.setenv("EXONIT_VERDICT_VERSION", "v2.1")
    candidate = _sample_candidate()
    vetting = {"passed": True}

    fake_v2_1_result = {"verdict": "planet_candidate", "confidence": 0.8,
                        "model_used": "v2.1"}
    with patch("exonit.verdict_classifier_v2_1.predict_v2_1",
              return_value=fake_v2_1_result) as mock_v2_1:
        result = predict_verdict(candidate, vetting, star_info={}, data_source="ffi")

    mock_v2_1.assert_called_once()
    _stub_v3_module.assert_not_called()
    assert result["model_used"] == "v2.1"


def test_dispatcher_no_data_source_arg_behaviour_unchanged(monkeypatch, _stub_v3_module):
    """Callers that don't pass data_source at all (pre-existing call sites)
    keep working exactly as before — data_source defaults to None, which is
    never 'ffi', so the new branch never triggers."""
    from exonit.verdict_classifier import predict_verdict

    monkeypatch.setenv("EXONIT_VERDICT_FFI_VERSION", "v3_tfopwg")
    monkeypatch.setenv("EXONIT_VERDICT_VERSION", "v2.1")
    candidate = _sample_candidate()
    vetting = {"passed": True}

    fake_v2_1_result = {"verdict": "uncertain", "confidence": 0.5,
                        "model_used": "v2.1"}
    with patch("exonit.verdict_classifier_v2_1.predict_v2_1",
              return_value=fake_v2_1_result) as mock_v2_1:
        result = predict_verdict(candidate, vetting, star_info={})

    mock_v2_1.assert_called_once()
    _stub_v3_module.assert_not_called()
    assert result["model_used"] == "v2.1"


def test_dispatcher_bad_ffi_version_raises(monkeypatch, _stub_v3_module):
    from exonit.verdict_classifier import predict_verdict

    monkeypatch.setenv("EXONIT_VERDICT_FFI_VERSION", "v99_bogus")
    candidate = _sample_candidate()
    vetting = {"passed": True}

    with pytest.raises(ValueError, match="v99_bogus"):
        predict_verdict(candidate, vetting, star_info={}, data_source="ffi")

    _stub_v3_module.assert_not_called()
