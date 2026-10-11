"""Tests for ``worker.exominer_native.preprocess.orchestrate``.

Smoke-test the full pipeline on real fixtures:
  - tests/fixtures/exominer_native/tic_143143769_s1-36.dvr.xml
  - tests/fixtures/exominer_native/tic_143143769_s0032.lc.fits

These tests do NOT carry a @pytest.mark.run_parity marker because
orchestrate.py is glue (spec §5: "NO parity test").
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from worker.exominer_native.preprocess.orchestrate import (
    TCEFeatures,
    run_tic,
)
from worker.exominer_native.preprocess import feature_spec as _feature_spec

# ---------------------------------------------------------------------------
# Fixtures paths
# ---------------------------------------------------------------------------

_FIXTURE_DIR = Path(__file__).resolve().parents[3] / "fixtures" / "exominer_native"
_DV_XML_PATH = _FIXTURE_DIR / "tic_143143769_s1-36.dvr.xml"
_LC_FITS_PATH = _FIXTURE_DIR / "tic_143143769_s0032.lc.fits"

_TIC_ID = "143143769"
_SECTOR_RUN = "1-36"

# Minimal ticv8 and gaia dicts (real values not needed for smoke-test)
_TICV8 = {
    "Teff": 4970.0,
    "logg": 4.55,
    "MH": 0.0,
    "rad": 0.86,
    "mass": 0.83,
    "rho": 2.38,
    "Tmag": 12.4,
}
_GAIA = {"ruwe": 1.01}

# Expected ONNX input names (from feature_spec)
_ONNX_KEYS = _feature_spec.get_onnx_input_names(
    _feature_spec.load_feature_spec(_feature_spec.default_feature_spec_path())
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_fixtures():
    """Return (dv_xml_bytes, [lc_fits_path])."""
    return _DV_XML_PATH.read_bytes(), [_LC_FITS_PATH]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_module_imports():
    """orchestrate module imports without error."""
    from worker.exominer_native.preprocess import orchestrate  # noqa: F401


def test_tce_features_dataclass():
    """TCEFeatures is a dataclass with the expected fields."""
    feat = TCEFeatures(
        uid="1-1-S1", period=1.0, t0=0.0, duration_days=0.1, depth_ppm=1000.0
    )
    assert feat.uid == "1-1-S1"
    assert feat.feeds == {}


class TestRunTicFixtures:
    """Integration smoke tests using real fixtures."""

    @pytest.fixture(autouse=True)
    def skip_if_missing(self):
        if not _DV_XML_PATH.exists() or not _LC_FITS_PATH.exists():
            pytest.skip("fixture files not present")

    def test_returns_at_least_one_result(self):
        dv_xml_bytes, lc_paths = _load_fixtures()
        results = run_tic(
            _TIC_ID, _SECTOR_RUN,
            dv_xml_bytes=dv_xml_bytes,
            lc_fits_paths=lc_paths,
            ticv8_data=_TICV8,
            gaia_data=_GAIA,
        )
        assert len(results) >= 1, "Expected at least one TCE result"

    def test_result_type(self):
        dv_xml_bytes, lc_paths = _load_fixtures()
        results = run_tic(
            _TIC_ID, _SECTOR_RUN,
            dv_xml_bytes=dv_xml_bytes,
            lc_fits_paths=lc_paths,
            ticv8_data=_TICV8,
            gaia_data=_GAIA,
        )
        for r in results:
            assert isinstance(r, TCEFeatures)

    def test_period_is_finite_and_positive(self):
        dv_xml_bytes, lc_paths = _load_fixtures()
        results = run_tic(
            _TIC_ID, _SECTOR_RUN,
            dv_xml_bytes=dv_xml_bytes,
            lc_fits_paths=lc_paths,
            ticv8_data=_TICV8,
            gaia_data=_GAIA,
        )
        for r in results:
            assert np.isfinite(r.period), f"{r.uid}: period is not finite"
            assert r.period > 0, f"{r.uid}: period is not positive"

    def test_uid_format(self):
        dv_xml_bytes, lc_paths = _load_fixtures()
        results = run_tic(
            _TIC_ID, _SECTOR_RUN,
            dv_xml_bytes=dv_xml_bytes,
            lc_fits_paths=lc_paths,
            ticv8_data=_TICV8,
            gaia_data=_GAIA,
        )
        for r in results:
            assert r.uid.startswith(_TIC_ID), f"uid {r.uid!r} doesn't start with tic_id"
            assert _SECTOR_RUN in r.uid, f"uid {r.uid!r} doesn't contain sector_run"

    def test_feeds_dict_has_onnx_keys(self):
        """Every ONNX key present in the spec must be in feeds (zero-fill fallback)."""
        dv_xml_bytes, lc_paths = _load_fixtures()
        results = run_tic(
            _TIC_ID, _SECTOR_RUN,
            dv_xml_bytes=dv_xml_bytes,
            lc_fits_paths=lc_paths,
            ticv8_data=_TICV8,
            gaia_data=_GAIA,
        )
        for r in results:
            missing = [k for k in _ONNX_KEYS if k not in r.feeds]
            assert not missing, f"{r.uid}: missing feeds keys: {missing}"

    def test_feeds_shapes_match_spec(self):
        """Every feed array must have the shape declared in the feature spec."""
        spec = _feature_spec.load_feature_spec(_feature_spec.default_feature_spec_path())
        dv_xml_bytes, lc_paths = _load_fixtures()
        results = run_tic(
            _TIC_ID, _SECTOR_RUN,
            dv_xml_bytes=dv_xml_bytes,
            lc_fits_paths=lc_paths,
            ticv8_data=_TICV8,
            gaia_data=_GAIA,
        )
        for r in results:
            for feat_name, feat_spec in spec.items():
                expected_shape = tuple(feat_spec["dim"])
                actual_shape = r.feeds[feat_name].shape
                assert actual_shape == expected_shape, (
                    f"{r.uid}: {feat_name} shape {actual_shape} != expected {expected_shape}"
                )

    def test_flux_views_are_not_all_zero(self):
        """Core flux views must contain non-zero data (spline detrend + phase-fold ran)."""
        dv_xml_bytes, lc_paths = _load_fixtures()
        results = run_tic(
            _TIC_ID, _SECTOR_RUN,
            dv_xml_bytes=dv_xml_bytes,
            lc_fits_paths=lc_paths,
            ticv8_data=_TICV8,
            gaia_data=_GAIA,
        )
        for r in results:
            assert np.any(r.feeds["global_flux_view_fluxnorm"] != 0), (
                f"{r.uid}: global_flux_view_fluxnorm is all-zero"
            )

    def test_empty_ticv8_and_gaia_does_not_crash(self):
        """Missing stellar/gaia data must be handled gracefully (NaN → median)."""
        dv_xml_bytes, lc_paths = _load_fixtures()
        results = run_tic(
            _TIC_ID, _SECTOR_RUN,
            dv_xml_bytes=dv_xml_bytes,
            lc_fits_paths=lc_paths,
            ticv8_data={},
            gaia_data={},
        )
        assert len(results) >= 1

    def test_momentum_dump_view_nonzero(self):
        """Momentum-dump local view has non-trivial signal when DQ flags fire.

        The fixture (sector 32) does contain momentum-dump events, so the view
        is expected to be non-zero after the full pipeline.
        """
        dv_xml_bytes, lc_paths = _load_fixtures()
        results = run_tic(
            _TIC_ID, _SECTOR_RUN,
            dv_xml_bytes=dv_xml_bytes,
            lc_fits_paths=lc_paths,
            ticv8_data=_TICV8,
            gaia_data=_GAIA,
        )
        for r in results:
            assert np.any(r.feeds["local_momentum_dump_view"] != 0), (
                f"{r.uid}: local_momentum_dump_view is all-zero"
            )

    def test_centroid_and_diff_img_feeds_populated(self):
        """Phase D.2/D.3 wiring: centroid views and diff-image tensors must be
        populated with real values (not the zero-fill placeholder)."""
        dv_xml_bytes, lc_paths = _load_fixtures()
        results = run_tic(
            _TIC_ID, _SECTOR_RUN,
            dv_xml_bytes=dv_xml_bytes,
            lc_fits_paths=lc_paths,
            ticv8_data=_TICV8,
            gaia_data=_GAIA,
        )
        assert results, "Expected at least one TCE"
        for r in results:
            # Centroid views (Phase D.2)
            assert np.any(r.feeds["local_centr_view_std_noclip"] != 0), (
                f"{r.uid}: local_centr_view_std_noclip is all-zero"
            )
            assert np.any(r.feeds["local_centr_view_std_noclip_var"] != 0), (
                f"{r.uid}: local_centr_view_std_noclip_var is all-zero"
            )
            # Diff-image tensors (Phase D.3)
            assert np.any(r.feeds["diff_imgs_std_trainset"] != 0), (
                f"{r.uid}: diff_imgs_std_trainset is all-zero"
            )
            assert np.any(r.feeds["oot_imgs_std_trainset"] != 0), (
                f"{r.uid}: oot_imgs_std_trainset is all-zero"
            )
            assert np.any(r.feeds["target_imgs"] != 0), (
                f"{r.uid}: target_imgs is all-zero (no pixel marked)"
            )
            assert np.any(r.feeds["quality"] != 0), (
                f"{r.uid}: quality is all-zero"
            )
            # target_imgs are one-hot maps; each channel should have
            # exactly one cell set to 1 (the target pixel location).
            tgt = r.feeds["target_imgs"]
            for ch in range(tgt.shape[-1]):
                assert tgt[..., ch].sum() <= 1.0 + 1e-6, (
                    f"{r.uid}: target_imgs channel {ch} has sum > 1"
                )
