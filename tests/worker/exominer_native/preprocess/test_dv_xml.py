"""Fast unit tests for ``worker.exominer_native.preprocess.dv_xml`` (no Docker)."""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pytest

from worker.exominer_native.preprocess.dv_xml import (
    N_IMGS_IN_DIFF,
    get_data_from_tess_dv_xml,
)


REPO_ROOT = Path(__file__).resolve().parents[4]
FIXTURE = REPO_ROOT / "tests/fixtures/exominer_native/tic_143143769_s1-36.dvr.xml"


def _silent_logger() -> logging.Logger:
    lg = logging.getLogger("test_dv_xml")
    lg.addHandler(logging.NullHandler())
    lg.propagate = False
    return lg


@pytest.mark.skipif(not FIXTURE.exists(), reason=f"fixture missing: {FIXTURE}")
def test_returns_one_tce_entry_for_tic_143143769():
    data = get_data_from_tess_dv_xml(
        FIXTURE, None, "1-36", None, 0.0, _silent_logger()
    )

    assert isinstance(data, dict)
    assert list(data.keys()) == ["143143769-1-S1-36"]


@pytest.mark.skipif(not FIXTURE.exists(), reason=f"fixture missing: {FIXTURE}")
def test_tce_entry_has_expected_shape():
    data = get_data_from_tess_dv_xml(
        FIXTURE, None, "1-36", None, 0.0, _silent_logger()
    )
    tce = data["143143769-1-S1-36"]

    assert set(tce) == {"target_ref_centroid", "image_data", "mag", "image_number", "quality_metric"}
    assert isinstance(tce["mag"], float)
    assert len(tce["image_data"]) == len(tce["image_number"]) == len(tce["target_ref_centroid"])
    assert len(tce["image_data"]) >= 1

    for img in tce["image_data"]:
        assert isinstance(img, np.ndarray)
        assert img.ndim == 4
        assert img.shape[2] == N_IMGS_IN_DIFF
        assert img.shape[3] == 2


@pytest.mark.skipif(not FIXTURE.exists(), reason=f"fixture missing: {FIXTURE}")
def test_no_neighbor_data_when_neighbors_dir_is_none():
    data = get_data_from_tess_dv_xml(
        FIXTURE, None, "1-36", None, 0.0, _silent_logger()
    )
    for tce in data.values():
        assert "neighbor_data" not in tce
