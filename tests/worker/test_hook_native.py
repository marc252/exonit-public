"""Tests for worker.exominer_native.hook — per-TIC native scoring.

Covers:
  * ``_match_period`` — exact, 2× alias, ½× alias, no-match, zero/None guards
  * ``score_candidates_native`` — end-to-end with HTTP mocked via in-repo
    fixtures (DV XML + LC FITS) + mocked ONNX inferer
  * ``score_candidates_native`` returns None on DV XML 404
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

from worker.exominer_native.hook import (
    PERIOD_FACTORS,
    PERIOD_TOL,
    _match_period,
    score_candidates_native,
)


FIXDIR = Path(__file__).resolve().parents[1] / "fixtures" / "exominer_native"
XML_FIXTURE = FIXDIR / "tic_143143769_s1-36.dvr.xml"
FITS_FIXTURE = FIXDIR / "tic_143143769_s0032.lc.fits"


# ---------------------------------------------------------------------------
# _match_period — pure-function unit tests
# ---------------------------------------------------------------------------

def _tce(uid: str, period: float):
    """Build a TCE-ish object compatible with both hook branches."""
    return SimpleNamespace(uid=uid, period=period)


def test_match_period_exact():
    tces = [_tce("a", 1.0), _tce("b", 2.7), _tce("c", 10.0)]
    m = _match_period(tces, 2.71)
    assert m is not None and m.uid == "b"


def test_match_period_2x_alias():
    # Target 5.0 d, TCE at 2.5 d → 2× alias matches within 2%.
    tces = [_tce("a", 2.5), _tce("b", 4.0)]
    m = _match_period(tces, 5.0)
    assert m is not None and m.uid == "a"


def test_match_period_half_alias():
    # Target 3.0 d, TCE at 6.0 d → ½× alias matches.
    tces = [_tce("a", 6.0), _tce("b", 9.99)]
    m = _match_period(tces, 3.0)
    assert m is not None and m.uid == "a"


def test_match_period_no_match():
    # Target 3.5 d, TCEs at 1.0 and 8.0 — none of the 1×/2×/½× aliases match.
    tces = [_tce("a", 1.0), _tce("b", 8.0)]
    m = _match_period(tces, 3.5)
    assert m is None


def test_match_period_none_target():
    tces = [_tce("a", 1.0)]
    assert _match_period(tces, None) is None


def test_match_period_empty_tces():
    assert _match_period([], 1.0) is None


def test_match_period_constants_are_sane():
    assert PERIOD_TOL == 0.02
    # 3× and 4× aliases (with their 1/3 and 1/4 reciprocals) added
    # 2026-05-16 to recover strong candidates whose BLS period was an N×
    # multiple of the ExoMiner TCE period.
    assert {1.0, 2.0, 0.5}.issubset(set(PERIOD_FACTORS))
    assert {3.0, 4.0, 0.25}.issubset(set(PERIOD_FACTORS))
    assert any(abs(f - 1.0 / 3.0) < 1e-9 for f in PERIOD_FACTORS)


def test_match_period_dict_shape():
    # Must also accept the legacy dict shape from exominer_client.
    tces = [{"period": 1.0, "score": 0.1}, {"period": 5.0, "score": 0.9}]
    m = _match_period(tces, 5.01)
    assert m is not None and m["score"] == 0.9


# ---------------------------------------------------------------------------
# Fake HTTP infrastructure for score_candidates_native
# ---------------------------------------------------------------------------

def _fake_response(*, status: int = 200, content: bytes = b"",
                   json_data=None):
    """Build a minimal MagicMock that quacks like requests.Response."""
    m = MagicMock()
    m.status_code = status
    m.content = content
    if json_data is not None:
        m.json.return_value = json_data
    else:
        m.json.side_effect = ValueError("no json")

    # iter_content for FITS streaming — yield the whole content in one chunk.
    def _iter(chunk_size=65536):
        if content:
            yield content

    m.iter_content.side_effect = _iter
    return m


def _build_http_router(dv_status: int = 200):
    """Return a callable usable as ``requests.get`` side_effect.

    Routes requests by URL suffix to in-memory fixtures.
    """
    xml_bytes = XML_FIXTURE.read_bytes()
    fits_bytes = FITS_FIXTURE.read_bytes()
    fits_name = FITS_FIXTURE.name  # 'tic_143143769_s0032.lc.fits'

    def _get(url, headers=None, params=None, timeout=None, stream=False):
        if "/api/mast/dv/" in url:
            if dv_status == 200:
                return _fake_response(status=200, content=xml_bytes)
            return _fake_response(status=dv_status)
        if "/api/ticv8/" in url:
            return _fake_response(status=200, json_data={
                "Teff": 5776.0, "rad": 1.0, "mass": 1.0,
                "Tmag": 11.3, "logg": 4.4, "MH": 0.0,
            })
        if "/api/gaia/" in url:
            return _fake_response(status=200, json_data={
                "source_id": "abc", "ruwe": 1.02, "parallax": 12.5,
            })
        if "/api/metadata/" in url:
            return _fake_response(status=200, json_data={
                "fits_files": [fits_name], "sectors": [32],
            })
        if "/api/fits/" in url:
            return _fake_response(status=200, content=fits_bytes)
        return _fake_response(status=404)

    return _get


# ---------------------------------------------------------------------------
# End-to-end score_candidates_native
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not XML_FIXTURE.exists() or not FITS_FIXTURE.exists(),
    reason="exominer-native fixtures missing",
)
def test_score_candidates_native_end_to_end():
    """Full happy-path with all HTTP and the ONNX inferer mocked.

    Uses the in-repo DV XML + LC FITS fixtures so the preprocessing
    orchestrator runs for real and produces TCEFeatures. Inference itself is
    stubbed to avoid the remaining scalar-rank issue tracked for Phase D.2.
    """
    candidates = [
        # TIC 143143769 TCE 1 period from fixture DV XML ≈ 1.3372 d
        {"period_days": 1.3372, "candidate_num": 1},
    ]

    router = _build_http_router(dv_status=200)

    # Stub the ONNX inferer so the test does not depend on the .onnx file
    # nor on the scalar-rank bug being fixed.
    fake_inferer = MagicMock()
    fake_inferer.score.return_value = 0.873

    with patch("worker.exominer_native.hook.requests.get", side_effect=router), \
         patch("worker.exominer_native.hook._get_inferer", return_value=fake_inferer):
        summary = score_candidates_native(
            tic_id="143143769",
            candidates=candidates,
            server_url="http://fake-server:8400",
            token="test-token",
            onnx_path="/nonexistent.onnx",
            config={},
        )

    assert summary is not None
    assert summary["tic_id"] == "143143769"
    assert summary["sector_run"] == "1-36"
    assert summary["n_tces"] >= 1
    assert summary["n_candidates"] == 1
    assert summary["n_matched"] == 1
    assert summary["per_tce_scores"]
    # Candidate dict must have been mutated in place.
    c = candidates[0]
    assert c["exominer_score"] == pytest.approx(0.873)
    assert c["exominer_sector_run"] == "1-36"
    assert c["exominer_tce_period"] == pytest.approx(1.3372, rel=0.02)

    # Auth header and ONNX inferer were actually used.
    assert fake_inferer.score.call_count >= 1


def test_score_candidates_native_returns_none_when_no_dv_xml():
    """A 404 from /api/mast/dv/ → return None (skip this TIC permanently)."""
    router = _build_http_router(dv_status=404)
    candidates = [{"period_days": 1.34}]

    with patch("worker.exominer_native.hook.requests.get", side_effect=router):
        summary = score_candidates_native(
            tic_id="999999999",
            candidates=candidates,
            server_url="http://fake-server:8400",
            token="test-token",
            onnx_path="/nonexistent.onnx",
            config={},
        )

    assert summary is None
    assert "exominer_score" not in candidates[0]


def test_score_candidates_native_returns_none_when_transient_error():
    """A 202 from /api/mast/dv/ → return None (transient, retry later)."""
    router = _build_http_router(dv_status=202)
    candidates = [{"period_days": 1.34}]

    with patch("worker.exominer_native.hook.requests.get", side_effect=router):
        summary = score_candidates_native(
            tic_id="143143769",
            candidates=candidates,
            server_url="http://fake-server:8400",
            token="test-token",
            onnx_path="/nonexistent.onnx",
            config={},
        )

    assert summary is None
    assert "exominer_score" not in candidates[0]


def test_score_candidates_native_returns_none_on_empty_candidates():
    """Empty candidate list → None, no HTTP calls."""
    with patch("worker.exominer_native.hook.requests.get") as mock_get:
        summary = score_candidates_native(
            tic_id="143143769",
            candidates=[],
            server_url="http://fake-server:8400",
            token="test-token",
            onnx_path="/nonexistent.onnx",
            config={},
        )
    assert summary is None
    assert mock_get.call_count == 0
