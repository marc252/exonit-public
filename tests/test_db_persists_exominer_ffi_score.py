"""Task 6: verify exominer_ffi_score + exominer_ffi_model_version round-trip
through add_candidate into the DB."""
from pathlib import Path
from exonit.db import ExoNitDB


def test_add_candidate_persists_phase3_score(tmp_path):
    db = ExoNitDB(str(tmp_path / "test.db"))
    db.add_star("111", ra=0.0, dec=0.0, tmag=11.0, n_sectors=2,
                n_datapoints=100, time_span_days=27.0)
    db.add_candidate(
        "111", candidate_num=1,
        period_days=2.0, t0=1.0, duration_days=0.1,
        depth_ppm=1000.0, snr=12.0, cnn_score=0.8,
        exominer_ffi_score=0.72,
        exominer_ffi_model_version="phase3c_seed42_v1",
    )
    rows = db._query(
        "SELECT exominer_ffi_score, exominer_ffi_model_version "
        "FROM candidates WHERE tic_id='111'")
    assert len(rows) == 1
    assert abs(rows[0]["exominer_ffi_score"] - 0.72) < 1e-6
    assert rows[0]["exominer_ffi_model_version"] == "phase3c_seed42_v1"


def test_add_candidate_update_preserves_phase3_score(tmp_path):
    """Re-running add_candidate must not erase an existing FFI score."""
    db = ExoNitDB(str(tmp_path / "test.db"))
    db.add_star("222", ra=0.0, dec=0.0, tmag=11.0, n_sectors=2,
                n_datapoints=100, time_span_days=27.0)
    # First insertion: with score
    db.add_candidate(
        "222", candidate_num=1,
        period_days=2.0, t0=1.0, duration_days=0.1,
        depth_ppm=1000.0, snr=12.0,
        exominer_ffi_score=0.65,
        exominer_ffi_model_version="phase3c_seed42_v1",
    )
    # Second call (re-run without score): must preserve existing
    db.add_candidate(
        "222", candidate_num=1,
        period_days=2.0, t0=1.0, duration_days=0.1,
        depth_ppm=1000.0, snr=12.0,
    )
    rows = db._query(
        "SELECT exominer_ffi_score, exominer_ffi_model_version "
        "FROM candidates WHERE tic_id='222'")
    assert len(rows) == 1
    assert abs(rows[0]["exominer_ffi_score"] - 0.65) < 1e-6
    assert rows[0]["exominer_ffi_model_version"] == "phase3c_seed42_v1"
