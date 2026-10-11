"""Tests for exonit.diagnose."""

import pandas as pd
import pytest

from exonit.diagnose import classify_miss


def test_classify_no_bls_candidate():
    reason, detail = classify_miss(
        known_period=5.0,
        known_depth_ppm=3000,
        n_sectors=5,
        time_span_days=27.0,
        processing_time_s=45.0,
        candidates=pd.DataFrame(columns=[
            "period_days", "depth_ppm", "snr", "cnn_score"
        ]),
    )
    assert reason == "no_bls_candidate"


def test_classify_wrong_period():
    candidates = pd.DataFrame([
        {"period_days": 12.5, "depth_ppm": 2500, "snr": 10.0, "cnn_score": 0.6}
    ])
    reason, detail = classify_miss(
        known_period=5.0, known_depth_ppm=3000,
        n_sectors=5, time_span_days=27.0,
        processing_time_s=45.0,
        candidates=candidates,
    )
    assert reason == "wrong_period"


def test_classify_shallow_depth():
    candidates = pd.DataFrame([
        {"period_days": 12.5, "depth_ppm": 200, "snr": 10.0, "cnn_score": 0.6}
    ])
    reason, detail = classify_miss(
        known_period=5.0, known_depth_ppm=300,
        n_sectors=5, time_span_days=27.0,
        processing_time_s=45.0,
        candidates=candidates,
    )
    assert reason == "shallow_depth"


def test_classify_long_period_out_of_bls_range():
    reason, detail = classify_miss(
        known_period=55.0, known_depth_ppm=2000,
        n_sectors=3, time_span_days=81.0,
        processing_time_s=30.0,
        candidates=pd.DataFrame(columns=[
            "period_days", "depth_ppm", "snr", "cnn_score"
        ]),
    )
    assert reason == "long_period"


def test_classify_slow_flatten():
    reason, detail = classify_miss(
        known_period=5.0, known_depth_ppm=2000,
        n_sectors=30, time_span_days=810.0,
        processing_time_s=360.0,
        candidates=pd.DataFrame(columns=[
            "period_days", "depth_ppm", "snr", "cnn_score"
        ]),
    )
    assert reason == "slow_flatten"


from exonit.diagnose import diagnose_misses_report


def _toi_row(tic_id, period, depth, disposition="KP"):
    return {"tic_id": tic_id, "disposition": disposition,
            "period_days": period, "depth_ppm": depth,
            "duration_hours": 2.0, "n_observations": 5}


def test_diagnose_misses_report_groups_by_reason():
    toi = pd.DataFrame([
        _toi_row("A", 5.0, 3000),
        _toi_row("B", 55.0, 2000),
        _toi_row("C", 5.0, 300),
    ])
    stars = pd.DataFrame([
        {"tic_id": "A", "n_sectors": 5, "n_datapoints": 15000,
         "time_span_days": 27.0, "status": "completed"},
        {"tic_id": "B", "n_sectors": 3, "n_datapoints": 9000,
         "time_span_days": 81.0, "status": "completed"},
        {"tic_id": "C", "n_sectors": 5, "n_datapoints": 15000,
         "time_span_days": 27.0, "status": "completed"},
    ])
    candidates = pd.DataFrame([
        {"tic_id": "A", "period_days": 5.01, "depth_ppm": 2950,
         "snr": 12.0, "cnn_score": 0.9}
    ])
    processing_times = {"A": 30.0, "B": 20.0, "C": 40.0}
    report = diagnose_misses_report(toi, stars, candidates, processing_times)
    assert "A" not in {r["tic_id"] for r in report}
    reasons = {r["tic_id"]: r["reason"] for r in report}
    assert reasons["B"] == "long_period"
    assert reasons["C"] == "shallow_depth"


import subprocess
import sys
from pathlib import Path


def test_diagnose_misses_cli_on_backup(tmp_path):
    backup = Path("data/backups/exonit_v2_10k_20260414.db")
    toi = Path("data/toi_catalog.csv")
    if not (backup.exists() and toi.exists()):
        pytest.skip("backup DB or TOI catalog missing")

    out_md = tmp_path / "diag.md"
    result = subprocess.run(
        [sys.executable, "scripts/diagnose_misses.py",
         "--db", str(backup),
         "--toi", str(toi),
         "--out", str(out_md)],
        capture_output=True, text=True, cwd=Path.cwd(),
    )
    assert result.returncode == 0, result.stderr
    assert out_md.exists()
    text = out_md.read_text()
    assert "reason" in text.lower()
    assert "summary" in text.lower() or "resum" in text.lower()


def _write_toi_csv(path, rows):
    """Write a minimal TOI-catalog-shaped CSV (headers load_toi_catalog expects)."""
    import csv as _csv
    with open(path, "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=[
            "TIC ID", "TFOPWG Disposition", "Period (days)", "Depth (ppm)",
            "Duration (hours)", "Time Series Observations", "Epoch (BJD)",
        ])
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _seed_operational_db(db_path, completeness_path=None):
    """Seed stars/candidates/tasks through exonit.db.ExoNitDB (backend-agnostic:
    works whether db_path is a SQLite file or a postgresql:// DSN). completeness
    always lives on SQLite (hybrid backend, see ExoNitDB docstring) — callers on
    the PG path must supply a real filesystem completeness_path."""
    from exonit.db import ExoNitDB

    db = ExoNitDB(str(db_path), completeness_path=str(completeness_path or db_path))
    # TIC A: known 5.0d transit found by BLS near the right period -> not a miss.
    db.add_star("A", 100.0, 20.0, 12.0, 5, 15000, 27.0, status="completed")
    db.add_candidate("A", 1, 5.01, 1500.0, 2.0 / 24, 2950, 12.0)
    db.add_task("task-A", "A")
    db.complete_task("task-A", 42.0)
    # TIC B: known long-period (55d) transit outside the BLS search range -> miss.
    db.add_star("B", 101.0, 21.0, 13.0, 3, 9000, 81.0, status="completed")
    db.add_task("task-B", "B")
    db.complete_task("task-B", 20.0)
    db.close()
    return db_path


def test_diagnose_misses_cli_seeded_sqlite(tmp_path):
    """Regression: op_session-routed processing-time query still matches the
    previous sqlite3.connect() behaviour on a plain SQLite path."""
    db_path = tmp_path / "seeded.db"
    _seed_operational_db(db_path)

    toi_csv = tmp_path / "toi.csv"
    _write_toi_csv(toi_csv, [
        {"TIC ID": "A", "TFOPWG Disposition": "KP", "Period (days)": 5.0,
         "Depth (ppm)": 3000, "Duration (hours)": 2.0,
         "Time Series Observations": 5, "Epoch (BJD)": 1500.0},
        {"TIC ID": "B", "TFOPWG Disposition": "CP", "Period (days)": 55.0,
         "Depth (ppm)": 2000, "Duration (hours)": 3.0,
         "Time Series Observations": 3, "Epoch (BJD)": 1550.0},
    ])

    out_md = tmp_path / "diag.md"
    result = subprocess.run(
        [sys.executable, "scripts/diagnose_misses.py",
         "--db", str(db_path),
         "--toi", str(toi_csv),
         "--out", str(out_md)],
        capture_output=True, text=True, cwd=Path.cwd(),
    )
    assert result.returncode == 0, result.stderr
    text = out_md.read_text()
    assert "| B | CP | long_period |" in text
    assert "| A |" not in text  # A was matched, not a miss
    assert "Total misses: **1**" in text


def test_diagnose_misses_cli_pg(tmp_path):
    """Same seeded scenario against a throwaway PostgreSQL DSN: proves the
    --db argument now accepts a postgresql:// DSN (the crash this change
    fixes) instead of only a SQLite file path. Skips when no throwaway PG
    test DB is reachable (see tests/_pg_test_db.py)."""
    from tests._pg_test_db import pg_test_available, resolve_test_dsn

    dsn = resolve_test_dsn()
    if not pg_test_available(dsn):
        pytest.skip("EXONIT_TEST_DATABASE_URL (throwaway PG) unset or not reachable")

    import psycopg
    with psycopg.connect(dsn, connect_timeout=5, autocommit=True) as c:
        c.execute("TRUNCATE stars, candidates, tasks CASCADE")
    try:
        _seed_operational_db(dsn, completeness_path=tmp_path / "cmpl_pg.db")

        toi_csv = tmp_path / "toi.csv"
        _write_toi_csv(toi_csv, [
            {"TIC ID": "A", "TFOPWG Disposition": "KP", "Period (days)": 5.0,
             "Depth (ppm)": 3000, "Duration (hours)": 2.0,
             "Time Series Observations": 5, "Epoch (BJD)": 1500.0},
            {"TIC ID": "B", "TFOPWG Disposition": "CP", "Period (days)": 55.0,
             "Depth (ppm)": 2000, "Duration (hours)": 3.0,
             "Time Series Observations": 3, "Epoch (BJD)": 1550.0},
        ])

        out_md = tmp_path / "diag_pg.md"
        result = subprocess.run(
            [sys.executable, "scripts/diagnose_misses.py",
             "--db", dsn,
             "--toi", str(toi_csv),
             "--out", str(out_md)],
            capture_output=True, text=True, cwd=Path.cwd(),
        )
        assert result.returncode == 0, result.stderr
        text = out_md.read_text()
        assert "| B | CP | long_period |" in text
        assert "Total misses: **1**" in text
    finally:
        with psycopg.connect(dsn, connect_timeout=5, autocommit=True) as c:
            c.execute("TRUNCATE stars, candidates, tasks CASCADE")
