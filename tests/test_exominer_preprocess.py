"""Tests for exonit.exominer_preprocess — Docker preprocessing + NFS cache."""
import csv
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from exonit.exominer_preprocess import (
    ExoMinerPreprocessor,
    PreprocessResult,
)


def _simulate_docker_outputs(run_dir: Path, tic_id: str, sector_run: str):
    """Create the file structure the container produces after a num_jobs=1 run."""
    job = run_dir / "output" / "job_0"
    (job / "tfrecord_data_diffimg_normalized").mkdir(parents=True, exist_ok=True)
    (job / "tce_table").mkdir(parents=True, exist_ok=True)

    tfrecord = job / "tfrecord_data_diffimg_normalized" / "shard-tess_diffimg_TESS_0"
    tfrecord.write_bytes(b"\x00\x01\x02\x03")  # fake TFRecord bytes

    tce_csv = job / "tce_table" / "tess-spoc-dv_tces_0_processed.csv"
    with open(tce_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["target_id", "tce_plnt_num", "tce_period", "tce_time0bk"])
        w.writerow([tic_id, 1, "14.64", "1337.75"])

    # also write prediction csv at output root so parse_predictions_csv works
    preds = run_dir / "output" / "predictions_outputs.csv"
    with open(preds, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            ["uid", "target_id", "tce_plnt_num", "tce_period", "tce_duration",
             "tce_time0bk", "tce_depth", "ruwe", "tce_prad", "tce_max_mult_ev", "score"]
        )
        w.writerow([f"{tic_id}-1-S{sector_run}", tic_id, 1, 14.64, 0.1, 1337.75, 245.1, 1.72, 3.48, 11.4, 0.87])


def test_preprocessor_creates_cache_dirs(tmp_path):
    pre = ExoMinerPreprocessor(
        nfs_records_dir=str(tmp_path / "records"),
        nfs_tce_table_dir=str(tmp_path / "tces"),
        work_dir=str(tmp_path / "work"),
    )
    pre.close()
    assert (tmp_path / "records").exists()
    assert (tmp_path / "tces").exists()


def test_preprocess_tic_runs_docker_copies_outputs(tmp_path):
    pre = ExoMinerPreprocessor(
        nfs_records_dir=str(tmp_path / "records"),
        nfs_tce_table_dir=str(tmp_path / "tces"),
        work_dir=str(tmp_path / "work"),
    )

    def fake_run(tic_id, sector_run, model, run_dir):
        _simulate_docker_outputs(Path(run_dir), tic_id, sector_run)

    with patch("exonit.exominer_preprocess.fetch_dv_runs", return_value=[(1, 39)]), \
         patch.object(pre, "_run_docker", side_effect=fake_run):
        result = pre.preprocess_tic("382437043")

    assert isinstance(result, PreprocessResult)
    assert result.tic_id == "382437043"
    assert result.sector_run == "1-39"
    assert result.from_cache is False
    assert Path(result.tfrecord_path).exists()
    assert Path(result.tce_table_path).exists()
    # Naming convention: <tic>_<sector_run>.<ext>
    assert "382437043_1-39" in result.tfrecord_path
    assert "382437043_1-39" in result.tce_table_path
    pre.close()


def test_preprocess_tic_hits_cache_second_call(tmp_path):
    pre = ExoMinerPreprocessor(
        nfs_records_dir=str(tmp_path / "records"),
        nfs_tce_table_dir=str(tmp_path / "tces"),
        work_dir=str(tmp_path / "work"),
    )

    def fake_run(tic_id, sector_run, model, run_dir):
        _simulate_docker_outputs(Path(run_dir), tic_id, sector_run)

    with patch("exonit.exominer_preprocess.fetch_dv_runs", return_value=[(1, 39)]), \
         patch.object(pre, "_run_docker", side_effect=fake_run):
        first = pre.preprocess_tic("1")
    assert first.from_cache is False

    with patch("exonit.exominer_preprocess.fetch_dv_runs", return_value=[(1, 39)]), \
         patch.object(pre, "_run_docker", side_effect=AssertionError("should not run docker")):
        second = pre.preprocess_tic("1")
    assert second.from_cache is True
    assert second.tfrecord_path == first.tfrecord_path
    assert second.tce_table_path == first.tce_table_path
    pre.close()


def test_preprocess_tic_no_dv_returns_empty(tmp_path):
    pre = ExoMinerPreprocessor(
        nfs_records_dir=str(tmp_path / "records"),
        nfs_tce_table_dir=str(tmp_path / "tces"),
        work_dir=str(tmp_path / "work"),
    )
    with patch("exonit.exominer_preprocess.fetch_dv_runs", return_value=[]), \
         patch.object(pre, "_run_docker", side_effect=AssertionError("should not run docker")):
        result = pre.preprocess_tic("404")
    assert result.tfrecord_path is None
    assert result.tce_table_path is None
    assert result.reason == "no_dv_runs"
    pre.close()


def test_preprocess_tic_respects_explicit_sector_run(tmp_path):
    pre = ExoMinerPreprocessor(
        nfs_records_dir=str(tmp_path / "records"),
        nfs_tce_table_dir=str(tmp_path / "tces"),
        work_dir=str(tmp_path / "work"),
    )

    def fake_run(tic_id, sector_run, model, run_dir):
        assert sector_run == "14-55"
        _simulate_docker_outputs(Path(run_dir), tic_id, sector_run)

    with patch.object(pre, "_run_docker", side_effect=fake_run):
        result = pre.preprocess_tic("1", sector_run="14-55")
    assert result.sector_run == "14-55"
    pre.close()


def test_preprocess_tic_docker_failure(tmp_path):
    pre = ExoMinerPreprocessor(
        nfs_records_dir=str(tmp_path / "records"),
        nfs_tce_table_dir=str(tmp_path / "tces"),
        work_dir=str(tmp_path / "work"),
    )
    import subprocess

    def fake_run(*args, **kw):
        raise subprocess.CalledProcessError(1, "docker")

    with patch("exonit.exominer_preprocess.fetch_dv_runs", return_value=[(1, 39)]), \
         patch.object(pre, "_run_docker", side_effect=fake_run):
        result = pre.preprocess_tic("1")
    assert result.tfrecord_path is None
    assert result.reason and "docker_failed" in result.reason
    pre.close()


def test_preprocess_tic_missing_outputs_flagged(tmp_path):
    """If docker runs but produces no TFRecord (e.g. TIC had no TCEs), flag it."""
    pre = ExoMinerPreprocessor(
        nfs_records_dir=str(tmp_path / "records"),
        nfs_tce_table_dir=str(tmp_path / "tces"),
        work_dir=str(tmp_path / "work"),
    )

    def fake_run_empty(tic_id, sector_run, model, run_dir):
        # Docker runs but produces nothing (e.g. TIC filtered out)
        (Path(run_dir) / "output" / "job_0").mkdir(parents=True, exist_ok=True)

    with patch("exonit.exominer_preprocess.fetch_dv_runs", return_value=[(1, 39)]), \
         patch.object(pre, "_run_docker", side_effect=fake_run_empty):
        result = pre.preprocess_tic("1")
    assert result.tfrecord_path is None
    assert result.reason == "no_tfrecord_produced"
    pre.close()
