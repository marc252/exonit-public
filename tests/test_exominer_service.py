"""Tests for exonit.exominer_service — Docker-backed ExoMiner inference."""
import csv
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from exonit.exominer_service import (
    ExoMinerService,
    ExoMinerTCE,
    parse_predictions_csv,
)


# --------- parse_predictions_csv ---------

def _write_pred_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            ["uid", "target_id", "tce_plnt_num", "tce_period", "tce_duration",
             "tce_time0bk", "tce_depth", "ruwe", "tce_prad", "tce_max_mult_ev", "score"]
        )
        for r in rows:
            w.writerow(r)


def test_parse_predictions_csv_empty(tmp_path):
    p = tmp_path / "predictions.csv"
    _write_pred_csv(p, [])
    tces = parse_predictions_csv(str(p))
    assert tces == []


def test_parse_predictions_csv_multiple_tces(tmp_path):
    p = tmp_path / "predictions.csv"
    _write_pred_csv(
        p,
        [
            ["382437043-1-S1-69", "382437043", 1, 14.64, 0.10, 1337.75, 245.13, 1.72, 3.48, 11.4, 4.43e-08],
            ["382437043-2-S1-69", "382437043", 2, 1142.8, 0.54, 2041.79, 528.84, 1.72, 4.81, 9.08, 8.5e-18],
        ],
    )
    tces = parse_predictions_csv(str(p))
    assert len(tces) == 2
    assert tces[0].tic_id == "382437043"
    assert tces[0].tce_num == 1
    assert tces[0].period == pytest.approx(14.64)
    assert tces[0].depth_ppm == pytest.approx(245.13)
    assert tces[0].score == pytest.approx(4.43e-08)
    assert tces[1].score == pytest.approx(8.5e-18)


def test_parse_predictions_csv_missing_file(tmp_path):
    assert parse_predictions_csv(str(tmp_path / "missing.csv")) == []


# --------- ExoMinerService — setup & cache ---------

def test_service_creates_cache_db(tmp_path):
    svc = ExoMinerService(
        cache_db=str(tmp_path / "cache.db"),
        work_dir=str(tmp_path / "work"),
    )
    svc.close()
    assert (tmp_path / "cache.db").exists()


def test_service_cache_roundtrip(tmp_path):
    svc = ExoMinerService(
        cache_db=str(tmp_path / "cache.db"),
        work_dir=str(tmp_path / "work"),
    )
    tces = [
        ExoMinerTCE(tic_id="1", tce_num=1, period=2.3, duration=0.09, t0=1000.0,
                    depth_ppm=30000.0, ruwe=1.0, prad=13.7, mes=187.0, score=0.999),
    ]
    svc._cache_put("1", "2-72", "exominer++_single", tces)
    cached = svc._cache_get("1", "2-72", "exominer++_single")
    svc.close()
    assert cached is not None
    assert len(cached) == 1
    assert cached[0].score == pytest.approx(0.999)


def test_service_cache_miss_returns_none(tmp_path):
    svc = ExoMinerService(
        cache_db=str(tmp_path / "cache.db"),
        work_dir=str(tmp_path / "work"),
    )
    assert svc._cache_get("1", "2-72", "exominer++_single") is None
    svc.close()


def test_service_cache_keyed_by_sector_run_and_model(tmp_path):
    svc = ExoMinerService(
        cache_db=str(tmp_path / "cache.db"),
        work_dir=str(tmp_path / "work"),
    )
    tces_a = [ExoMinerTCE("1", 1, 2.3, 0.09, 1000, 30000, 1.0, 13.7, 187, 0.9)]
    tces_b = [ExoMinerTCE("1", 1, 2.3, 0.09, 1000, 30000, 1.0, 13.7, 187, 0.1)]
    svc._cache_put("1", "1-39", "exominer++_single", tces_a)
    svc._cache_put("1", "14-55", "exominer++_single", tces_b)
    assert svc._cache_get("1", "1-39", "exominer++_single")[0].score == pytest.approx(0.9)
    assert svc._cache_get("1", "14-55", "exominer++_single")[0].score == pytest.approx(0.1)
    svc.close()


# --------- predict_tic — integration with mocked MAST + docker ---------

def test_predict_tic_returns_cached_result_without_docker(tmp_path):
    svc = ExoMinerService(
        cache_db=str(tmp_path / "cache.db"),
        work_dir=str(tmp_path / "work"),
    )
    tces = [ExoMinerTCE("9", 1, 2.3, 0.09, 1000, 30000, 1.0, 13.7, 187, 0.999)]
    svc._cache_put("9", "1-39", "exominer++_single", tces)

    with patch("exonit.exominer_service.fetch_dv_runs", return_value=[(1, 39)]), \
         patch.object(svc, "_run_docker", side_effect=AssertionError("should not run docker")):
        result = svc.predict_tic("9")

    svc.close()
    assert len(result.tces) == 1
    assert result.tces[0].score == pytest.approx(0.999)
    assert result.from_cache is True
    assert result.sector_run == "1-39"


def test_predict_tic_no_dv_returns_empty_no_docker(tmp_path):
    svc = ExoMinerService(
        cache_db=str(tmp_path / "cache.db"),
        work_dir=str(tmp_path / "work"),
    )
    with patch("exonit.exominer_service.fetch_dv_runs", return_value=[]), \
         patch.object(svc, "_run_docker", side_effect=AssertionError("should not run docker")):
        result = svc.predict_tic("42")
    svc.close()
    assert result.tces == []
    assert result.sector_run is None
    assert result.reason == "no_dv_runs"


def test_predict_tic_runs_docker_on_cache_miss(tmp_path):
    svc = ExoMinerService(
        cache_db=str(tmp_path / "cache.db"),
        work_dir=str(tmp_path / "work"),
    )

    def fake_run_docker(tic_id, sector_run, model):
        # simulate container writing predictions csv and return its path
        out = Path(svc.work_dir) / "predictions_outputs.csv"
        _write_pred_csv(
            out,
            [
                ["8-1-S1-39", "8", 1, 2.3, 0.09, 1000, 30000, 1.0, 13.7, 187, 0.87]
            ],
        )
        return str(out)

    with patch("exonit.exominer_service.fetch_dv_runs", return_value=[(1, 39)]), \
         patch.object(svc, "_run_docker", side_effect=fake_run_docker):
        result = svc.predict_tic("8")

    assert len(result.tces) == 1
    assert result.tces[0].score == pytest.approx(0.87)
    assert result.from_cache is False

    # Next call should hit cache (no docker)
    with patch("exonit.exominer_service.fetch_dv_runs", return_value=[(1, 39)]), \
         patch.object(svc, "_run_docker", side_effect=AssertionError("should not run docker")):
        result2 = svc.predict_tic("8")
    assert result2.from_cache is True
    assert len(result2.tces) == 1
    svc.close()


def test_predict_tic_respects_requested_sector_run(tmp_path):
    svc = ExoMinerService(
        cache_db=str(tmp_path / "cache.db"),
        work_dir=str(tmp_path / "work"),
    )

    def fake_run_docker(tic_id, sector_run, model):
        assert sector_run == "14-55", f"expected 14-55, got {sector_run}"
        out = Path(svc.work_dir) / "predictions_outputs.csv"
        _write_pred_csv(out, [["8-1-S14-55", "8", 1, 2.3, 0.09, 1000, 30000, 1.0, 13.7, 187, 0.5]])
        return str(out)

    with patch.object(svc, "_run_docker", side_effect=fake_run_docker):
        result = svc.predict_tic("8", sector_run="14-55")

    assert result.sector_run == "14-55"
    assert len(result.tces) == 1
    svc.close()


# --------- match_tce_to_period ---------

def test_match_tce_to_period():
    from exonit.exominer_service import match_tce_to_period

    tces = [
        ExoMinerTCE("1", 1, 11.62, 0.22, 2000, 440, 1.0, 3.1, 13.5, 0.97),
        ExoMinerTCE("1", 2, 487.6, 0.26, 2002, 1644, 1.0, 5.8, 22.3, 0.18),
    ]
    # direct match
    m = match_tce_to_period(tces, 11.62)
    assert m.tce_num == 1
    # half-period alias
    m = match_tce_to_period(tces, 5.81)
    assert m.tce_num == 1
    # double-period alias
    m = match_tce_to_period(tces, 23.24)
    assert m.tce_num == 1
    # no match
    assert match_tce_to_period(tces, 100.0) is None
    # empty
    assert match_tce_to_period([], 1.0) is None
