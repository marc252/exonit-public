"""Tests for exonit.scorecard."""

import pytest
from exonit.scorecard import match_period


@pytest.mark.parametrize("det, known, expected_match, expected_kind", [
    (5.154, 5.154, True, "direct"),
    (5.158, 5.154, True, "direct"),          # within 1%
    (5.30, 5.154, False, None),              # outside 1%
    (10.308, 5.154, True, "harmonic_2"),     # 2x
    (2.577, 5.154, True, "sub_harmonic_0.5"),  # 1/2
    (1.718, 5.154, True, "sub_harmonic_0.3333333333333333"),  # 1/3
    (25.77, 5.154, True, "harmonic_5"),      # 5x
    (4.0, 5.154, False, None),               # clearly unrelated
])
def test_match_period(det, known, expected_match, expected_kind):
    matched, kind = match_period(det, known, tolerance=0.01)
    assert matched is expected_match
    assert kind == expected_kind


def test_match_period_rejects_zero():
    matched, kind = match_period(5.0, 0, tolerance=0.01)
    assert matched is False
    assert kind is None


from exonit.scorecard import bin_depth, bin_period, bin_sectors


@pytest.mark.parametrize("depth_ppm, expected", [
    (300, "<500"),
    (500, "500-1000"),
    (999, "500-1000"),
    (1000, "1000-5000"),
    (4999, "1000-5000"),
    (5000, ">5000"),
    (20000, ">5000"),
])
def test_bin_depth(depth_ppm, expected):
    assert bin_depth(depth_ppm) == expected


@pytest.mark.parametrize("period_days, expected", [
    (0.5, "<3d"),
    (2.99, "<3d"),
    (3.0, "3-10d"),
    (9.99, "3-10d"),
    (10.0, "10-30d"),
    (29.99, "10-30d"),
    (30.0, ">30d"),
    (100.0, ">30d"),
])
def test_bin_period(period_days, expected):
    assert bin_period(period_days) == expected


@pytest.mark.parametrize("n_sectors, expected", [
    (1, "1-2"),
    (2, "1-2"),
    (3, "3-6"),
    (6, "3-6"),
    (7, "7-15"),
    (15, "7-15"),
    (16, ">15"),
    (49, ">15"),
])
def test_bin_sectors(n_sectors, expected):
    assert bin_sectors(n_sectors) == expected


import pandas as pd
from pathlib import Path

from exonit.scorecard import load_toi_catalog


def test_load_toi_catalog_real():
    catalog_path = Path("data/toi_catalog.csv")
    if not catalog_path.exists():
        pytest.skip("toi_catalog.csv not present")
    df = load_toi_catalog(catalog_path)
    for col in ("tic_id", "disposition", "period_days", "depth_ppm",
                "duration_hours", "n_observations"):
        assert col in df.columns, f"missing column {col}"
    assert pd.api.types.is_string_dtype(df["tic_id"])
    assert set(df["disposition"].dropna().unique()).issubset(
        {"KP", "CP", "PC", "FP", "APC", "FA"}
    )


def test_load_toi_catalog_dispositions_counts():
    catalog_path = Path("data/toi_catalog.csv")
    if not catalog_path.exists():
        pytest.skip("toi_catalog.csv not present")
    df = load_toi_catalog(catalog_path)
    kp_cp = df[df["disposition"].isin(["KP", "CP"])]
    assert len(kp_cp) > 100, "expected at least 100 KP+CP in TOI catalog"


from exonit.scorecard import load_db_processed


def test_load_db_processed_backup():
    db_path = Path("data/backups/exonit_v2_10k_20260414.db")
    if not db_path.exists():
        pytest.skip("backup DB not present")
    stars, candidates, completeness = load_db_processed(db_path)
    assert len(stars) > 1000, f"expected >1000 stars, got {len(stars)}"
    assert len(candidates) > 1000
    for col in ("tic_id", "n_sectors", "status"):
        assert col in stars.columns
    for col in ("tic_id", "period_days", "depth_ppm", "snr",
                "cnn_score", "triceratops_fpp", "claude_verdict"):
        assert col in candidates.columns
    assert "tic_id" in completeness.columns


# --------------------------------------------------------------------------- #
# Fase 4a Task 4: load_db_processed_for_toi's PG + EXONIT_COMPLETENESS_PG=1
# branch — distinct_completeness_tics must come from count_completeness_tics()
# (T1), not the hard-coded 0 the flag-OFF/pure-PG path reports (still covered
# by tests/test_scorecard_pg_parity.py::test_pg_completeness_count_is_zero).
# Real throwaway PG (tests/_pg_test_db.py); skips cleanly when unavailable.
# --------------------------------------------------------------------------- #

from exonit.db import ExoNitDB
from exonit.scorecard import load_db_processed_for_toi
from tests._pg_test_db import pg_test_available, resolve_test_dsn

_SCORECARD_CMPL_PG_DSN = resolve_test_dsn()

_SCORECARD_CMPL_PG_TABLES = (
    "worker_telemetry", "worker_debug_mode", "candidates",
    "stars", "tasks", "workers_registry",
    "completeness_staging", "completeness_tics", "completeness_agg_bucket",
)


def _truncate_scorecard_cmpl_pg(dsn):
    import psycopg
    with psycopg.connect(dsn, connect_timeout=5, autocommit=True) as c:
        c.execute("TRUNCATE " + ", ".join(_SCORECARD_CMPL_PG_TABLES) + " CASCADE")


@pytest.mark.skipif(
    not pg_test_available(_SCORECARD_CMPL_PG_DSN),
    reason="EXONIT_TEST_DATABASE_URL (throwaway PG) unset or not reachable",
)
def test_load_db_processed_for_toi_pg_flag_on_uses_count_completeness_tics(
        monkeypatch):
    monkeypatch.setenv("EXONIT_COMPLETENESS_PG", "1")
    _truncate_scorecard_cmpl_pg(_SCORECARD_CMPL_PG_DSN)
    try:
        db = ExoNitDB(_SCORECARD_CMPL_PG_DSN)
        try:
            db.add_star("1", 0.0, 0.0, 12.0, 5, 1000, 27.0, status="completed")
            db.add_star("2", 0.0, 0.0, 12.0, 5, 1000, 27.0, status="completed")
            db.add_completeness_many("1", [(1.0, 5.0, 0.5)])
            db.add_completeness_many("2", [(1.0, 5.0, 0.5)])
            db.add_completeness_many("2", [(2.0, 10.0, 0.6)])  # same tic
        finally:
            db.close()

        _, _, agg = load_db_processed_for_toi(_SCORECARD_CMPL_PG_DSN, {"1", "2"})
        assert agg["distinct_completeness_tics"] == 2
    finally:
        _truncate_scorecard_cmpl_pg(_SCORECARD_CMPL_PG_DSN)


def test_load_db_processed_for_toi_pg_flag_on_zero_when_empty(monkeypatch):
    psycopg_dsn = _SCORECARD_CMPL_PG_DSN
    if not pg_test_available(psycopg_dsn):
        pytest.skip("EXONIT_TEST_DATABASE_URL (throwaway PG) unset or not reachable")
    monkeypatch.setenv("EXONIT_COMPLETENESS_PG", "1")
    _truncate_scorecard_cmpl_pg(psycopg_dsn)
    try:
        db = ExoNitDB(psycopg_dsn)
        db.add_star("1", 0.0, 0.0, 12.0, 5, 1000, 27.0, status="completed")
        db.close()
        _, _, agg = load_db_processed_for_toi(psycopg_dsn, {"1"})
        assert agg["distinct_completeness_tics"] == 0
    finally:
        _truncate_scorecard_cmpl_pg(psycopg_dsn)


from exonit.scorecard import compute_recall_and_coverage


def _make_mini_toi():
    return pd.DataFrame([
        {"tic_id": "1", "disposition": "KP", "period_days": 5.0,
         "depth_ppm": 1500, "duration_hours": 2.0, "n_observations": 5},
        {"tic_id": "2", "disposition": "KP", "period_days": 10.0,
         "depth_ppm": 3000, "duration_hours": 3.0, "n_observations": 5},
        {"tic_id": "3", "disposition": "CP", "period_days": 1.5,
         "depth_ppm": 800, "duration_hours": 1.0, "n_observations": 3},
        {"tic_id": "4", "disposition": "FP", "period_days": 2.0,
         "depth_ppm": 500, "duration_hours": 1.0, "n_observations": 2},
    ])


def _make_mini_stars():
    return pd.DataFrame([
        {"tic_id": "1", "n_sectors": 5, "status": "completed"},
        {"tic_id": "2", "n_sectors": 5, "status": "completed"},
        {"tic_id": "3", "n_sectors": 3, "status": "completed"},
        {"tic_id": "4", "n_sectors": 2, "status": "completed"},
    ])


def _make_mini_candidates():
    return pd.DataFrame([
        {"tic_id": "1", "period_days": 5.02, "depth_ppm": 1480,
         "duration_days": 2.0/24, "snr": 12.0, "cnn_score": 0.9,
         "triceratops_fpp": 0.01, "claude_verdict": "planet_candidate"},
        {"tic_id": "2", "period_days": 20.0, "depth_ppm": 2900,
         "duration_days": 3.0/24, "snr": 10.0, "cnn_score": 0.8,
         "triceratops_fpp": 0.05, "claude_verdict": "planet_candidate"},
        {"tic_id": "3", "period_days": 7.3, "depth_ppm": 500,
         "duration_days": 0.8/24, "snr": 8.0, "cnn_score": 0.5,
         "triceratops_fpp": 0.3, "claude_verdict": "UNCERTAIN"},
        {"tic_id": "4", "period_days": 2.01, "depth_ppm": 480,
         "duration_days": 1.0/24, "snr": 9.0, "cnn_score": 0.8,
         "triceratops_fpp": 0.4, "claude_verdict": "UNCERTAIN"},
    ])


def test_compute_recall_and_coverage_minimal():
    toi = _make_mini_toi()
    stars = _make_mini_stars()
    candidates = _make_mini_candidates()
    completeness = pd.DataFrame([
        {"tic_id": "1"}, {"tic_id": "2"}, {"tic_id": "3"}
    ])
    result = compute_recall_and_coverage(toi, stars, candidates, completeness)
    # Schema v2: the cumulative block is what recall_global used to be.
    assert result["cumulative"]["recall"] == pytest.approx(2/3)
    assert result["cumulative"]["n_kp_cp"] == 3
    # No lot given -> the run's own recall is explicitly NOT measurable,
    # never a number inherited from the cumulative figure.
    assert result["lot"]["recall"] is None
    assert result["lot"]["measurable"] is False
    assert result["coverage"]["completed"] == pytest.approx(1.0)
    assert result["coverage"]["with_candidate"] == pytest.approx(1.0)
    assert result["coverage"]["with_completeness"] == pytest.approx(3/4)


from exonit.scorecard import (
    compute_recall_by_bin,
    compute_parameter_precision,
    compute_fp_rate,
)


def test_compute_recall_by_bin_minimal():
    toi = _make_mini_toi()
    stars = _make_mini_stars()
    candidates = _make_mini_candidates()
    bins = compute_recall_by_bin(toi, stars, candidates)
    assert bins["recall_by_depth"]["1000-5000"] == pytest.approx(1.0)
    assert bins["recall_by_depth"]["500-1000"] == pytest.approx(0.0)
    assert bins["recall_by_period"]["<3d"] == pytest.approx(0.0)
    assert bins["recall_by_period"]["3-10d"] == pytest.approx(1.0)


def test_compute_parameter_precision_minimal():
    toi = _make_mini_toi()
    stars = _make_mini_stars()
    candidates = _make_mini_candidates()
    prec = compute_parameter_precision(toi, stars, candidates)
    assert prec["period_precision"]["median_rel_err"] < 0.02
    assert prec["depth_precision"]["median_rel_err"] < 0.05


def test_compute_exominer_metrics_minimal():
    from exonit.scorecard import compute_exominer_metrics
    toi = _make_mini_toi()
    stars = _make_mini_stars()
    cands = _make_mini_candidates()
    # Add ExoMiner columns: KP TIC 1 has good Docker score, KP TIC 2 missing
    # Docker but has FFI, CP TIC 3 has neither, FP TIC 4 has high (bad) FFI.
    cands = cands.assign(
        exominer_score=[0.9, None, None, 0.2],
        exominer_ffi_score=[0.8, 0.7, None, 0.85],
    )
    m = compute_exominer_metrics(toi, stars, cands, threshold=0.5)["exominer"]
    assert m["n_kp_cp_processed"] == 3   # 2 KP + 1 CP
    assert m["n_fp_fa_processed"] == 1   # 1 FP
    # Docker recall: only TIC 1 hits among the 3 KP/CP
    assert m["recall_docker"] == pytest.approx(1/3)
    # FFI recall: TIC 1 + TIC 2 hit, TIC 3 null
    assert m["recall_ffi"] == pytest.approx(2/3)
    # Combined: TIC 1 (both) + TIC 2 (FFI) → 2 of 3
    assert m["recall_combined"] == pytest.approx(2/3)
    # Specificity: FP TIC 4 — Docker says 0.2 (correct → counts), FFI 0.85 (wrong → doesn't)
    assert m["specificity_docker"] == pytest.approx(1.0)
    assert m["specificity_ffi"] == pytest.approx(0.0)
    # Null counts
    assert m["kp_cp_em_null"] == 2   # TIC 2 + TIC 3
    assert m["kp_cp_ffi_null"] == 1  # TIC 3


def test_compute_exominer_metrics_minus1_counts_as_null():
    """exominer_score=-1 (no-DV cache-negative sentinel) must land in the
    em_null bucket, exactly like exominer_score=None — not fall through
    uncounted (the pre-fix bug: -1 was neither null nor a real positive, so
    it silently vanished from both em_pos and em_null, making
    recall_docker/specificity_docker misreport actual SPOC coverage). See
    progress/investigation-exominer-minus1-2026-07-02.md."""
    from exonit.scorecard import compute_exominer_metrics
    toi = _make_mini_toi()
    stars = _make_mini_stars()
    cands = _make_mini_candidates()
    # TIC1 (KP): -1 sentinel, no FFI -> "not attempted", not a real score.
    # TIC2 (KP): None, real FFI -> unaffected baseline (null + ffi hit).
    # TIC3 (CP): real Docker score -> unaffected baseline (real positive).
    # TIC4 (FP): -1 sentinel, no FFI -> "not attempted", NOT a correctly-
    #            identified negative (specificity must not credit it).
    cands = cands.assign(
        exominer_score=[-1, None, 0.9, -1],
        exominer_ffi_score=[None, 0.7, None, None],
    )
    m = compute_exominer_metrics(toi, stars, cands, threshold=0.5)["exominer"]

    # -1 is not a real score, so it never counts as a Docker hit.
    assert m["recall_docker"] == pytest.approx(1 / 3)  # only TIC3
    # Both KP/CP -1 (TIC1) and None (TIC2) land in em_null now.
    assert m["kp_cp_em_null"] == 2
    # The FP's -1 (TIC4) must be counted as "not attempted", not silently
    # dropped from every bucket.
    assert m["fp_fa_em_null"] == 1
    # Honest accounting: specificity_docker must NOT credit a never-
    # attempted TIC as a correctly-identified negative.
    assert m["specificity_docker"] == pytest.approx(0.0)


def test_compute_fp_rate_minimal():
    toi = _make_mini_toi()
    stars = _make_mini_stars()
    candidates = _make_mini_candidates()
    fp = compute_fp_rate(toi, stars, candidates)
    assert fp["fp_rate_cnn"] == pytest.approx(1.0)
    assert fp["fp_rate_claude"] == pytest.approx(0.0)
    assert fp["n_fp_processed"] == 1


import json
from exonit.scorecard import build_scorecard, scorecard_to_markdown


def test_build_scorecard_minimal(tmp_path):
    toi = _make_mini_toi()
    stars = _make_mini_stars()
    candidates = _make_mini_candidates()
    completeness = pd.DataFrame([{"tic_id": "1"}, {"tic_id": "2"}])
    card = build_scorecard(
        toi, stars, candidates, completeness,
        run_id="test-run-1", sample_size=4,
        stratification="test", pipeline_version="abc123", seed=42,
    )
    assert card["run_id"] == "test-run-1"
    assert card["sample_size"] == 4
    assert card["metrics"]["cumulative"]["recall"] == pytest.approx(2/3)
    assert card["schema_version"] == 2
    assert "recall_by_depth" in card["metrics"]
    assert "period_precision" in card["metrics"]
    assert "fp_rate_cnn" in card["metrics"]
    path = tmp_path / "card.json"
    path.write_text(json.dumps(card))
    reloaded = json.loads(path.read_text())
    assert reloaded["run_id"] == "test-run-1"


import subprocess
import sys


def test_scorecard_cli_on_backup(tmp_path):
    backup = Path("data/backups/exonit_v2_10k_20260414.db")
    toi = Path("data/toi_catalog.csv")
    if not (backup.exists() and toi.exists()):
        pytest.skip("backup DB or TOI catalog missing")

    out_json = tmp_path / "card.json"
    result = subprocess.run(
        [sys.executable, "scripts/scorecard.py",
         "--db", str(backup),
         "--toi", str(toi),
         "--out", str(out_json),
         "--run-id", "test-backup",
         "--sample-size", "1057",
         "--stratification", "full-backup",
         "--pipeline-version", "v2-backup",
         "--seed", "0",
         "--report"],
        capture_output=True, text=True, cwd=Path.cwd(),
    )
    assert result.returncode == 0, result.stderr
    assert out_json.exists()
    card = json.loads(out_json.read_text())
    assert 0.0 <= card["metrics"]["cumulative"]["recall"] <= 1.0
    assert card["metrics"]["cumulative"]["n_kp_cp"] > 0
    assert "Scorecard:" in result.stdout
    assert "period_precision" in card["metrics"]
    assert 0.0 <= card["coverage"]["completed"] <= 1.0
    assert card["metrics"]["cumulative"]["n_kp_cp"] >= 50


import sqlite3

from exonit.scorecard import load_db_processed, load_db_processed_for_toi


def _build_synthetic_db(tmp_path):
    """Create a minimal SQLite DB with both TOI and non-TOI TICs.

    TOI TICs: "1", "2", "3", "4"  (matches _make_mini_toi)
    Non-TOI TICs: "99", "100", "101" (extra rows that the filtered loader skips)

    The non-TOI rows make total_stars, distinct candidates/completeness differ
    from the filtered subset — the critical invariant the bit-identity test
    checks.
    """
    db = tmp_path / "synthetic.db"
    con = sqlite3.connect(str(db))
    con.executescript("""
        CREATE TABLE stars (
            tic_id TEXT, n_sectors INTEGER, status TEXT
        );
        CREATE TABLE candidates (
            tic_id TEXT, period_days REAL, depth_ppm REAL,
            duration_days REAL, snr REAL, cnn_score REAL,
            triceratops_fpp REAL, claude_verdict TEXT,
            exominer_score REAL, exominer_ffi_score REAL
        );
        CREATE TABLE completeness (tic_id TEXT);
    """)
    # TOI stars
    con.executemany("INSERT INTO stars VALUES (?,?,?)", [
        ("1", 5, "completed"),
        ("2", 5, "completed"),
        ("3", 3, "completed"),
        ("4", 2, "completed"),
    ])
    # Non-TOI stars (extra bulk that should NOT affect per-TIC metrics)
    con.executemany("INSERT INTO stars VALUES (?,?,?)", [
        ("99",  1, "completed"),
        ("100", 2, "failed"),
        ("101", 3, "completed"),
    ])
    # TOI candidates
    con.executemany(
        "INSERT INTO candidates VALUES (?,?,?,?,?,?,?,?,?,?)", [
            ("1", 5.02, 1480, 2.0/24, 12.0, 0.9, 0.01, "planet_candidate", 0.9, 0.8),
            ("2", 20.0, 2900, 3.0/24, 10.0, 0.8, 0.05, "planet_candidate", None, 0.7),
            ("3", 7.3,  500,  0.8/24,  8.0, 0.5, 0.3,  "UNCERTAIN",        None, None),
            ("4", 2.01, 480,  1.0/24,  9.0, 0.8, 0.4,  "UNCERTAIN",        0.2,  0.85),
        ]
    )
    # Non-TOI candidates
    con.executemany(
        "INSERT INTO candidates VALUES (?,?,?,?,?,?,?,?,?,?)", [
            ("99",  3.0, 1000, 1.0/24, 7.0, 0.3, 0.5, "UNCERTAIN", None, None),
            ("100", 5.0,  800, 1.5/24, 6.0, 0.4, 0.6, "UNCERTAIN", None, None),
        ]
    )
    # TOI completeness
    con.executemany("INSERT INTO completeness VALUES (?)", [
        ("1",), ("2",), ("3",),
    ])
    # Non-TOI completeness
    con.executemany("INSERT INTO completeness VALUES (?)", [
        ("99",), ("100",), ("101",), ("102",),
    ])
    con.commit()
    con.close()
    return db


def test_bit_identity_old_vs_new_loader(tmp_path):
    """Filtered loader produces bit-identical scorecard to the full loader.

    The DB has non-TOI stars/candidates/completeness rows that make the raw
    global counts (total_stars, distinct candidates/completeness) differ from
    the filtered subset. The test verifies that coverage uses the correct
    global counts via coverage_aggregates, and that all other metrics
    (per-TIC recall, bins, precision, fp, exominer) are unaffected because
    they only ever touch TOI TICs.
    """
    db = _build_synthetic_db(tmp_path)
    toi = _make_mini_toi()
    toi_tics = set(toi["tic_id"])

    # --- old path (full load) ---
    stars_old, cands_old, comp_old = load_db_processed(db)
    card_old = build_scorecard(
        toi, stars_old, cands_old, comp_old,
        run_id="test-oom", sample_size=4,
        stratification="test", pipeline_version="memfix", seed=0,
    )

    # --- new path (filtered load) ---
    stars_new, cands_new, aggs = load_db_processed_for_toi(db, toi_tics)
    card_new = build_scorecard(
        toi, stars_new, cands_new, None,
        run_id="test-oom", sample_size=4,
        stratification="test", pipeline_version="memfix", seed=0,
        coverage_aggregates=aggs,
    )

    # Discard generated_at before comparing (differs by wall-clock seconds)
    card_old.pop("generated_at", None)
    card_new.pop("generated_at", None)

    assert card_old == card_new, (
        "Scorecard mismatch between full and filtered loader.\n"
        f"Old coverage: {card_old['coverage']}\n"
        f"New coverage: {card_new['coverage']}\n"
        f"Old metrics snapshot: cumulative={card_old['metrics']['cumulative']}\n"
        f"New metrics snapshot: cumulative={card_new['metrics']['cumulative']}"
    )


def test_scorecard_to_markdown_minimal():
    card = {
        "run_id": "r1", "sample_size": 4, "pipeline_version": "abc",
        "metrics": {
            "lot": {"recall": 0.5, "measurable": True,
                    "n_kp_cp": 2, "n_detected": 1, "lot_size": 100},
            "cumulative": {"recall": 0.67, "measurable": True,
                           "n_kp_cp": 3, "n_detected": 2},
            "recall_by_depth": {"<500": 0.0, "500-1000": 0.0,
                                "1000-5000": 1.0, ">5000": 0.0},
            "recall_by_period": {"<3d": 0.0, "3-10d": 1.0,
                                 "10-30d": 0.0, ">30d": 0.0},
            "recall_by_sectors": {"1-2": 0.0, "3-6": 0.0,
                                  "7-15": 1.0, ">15": 0.0},
            "period_precision": {"median_rel_err": 0.004, "sigma": 0.001, "n": 2},
            "depth_precision": {"median_rel_err": 0.02, "sigma": 0.01, "n": 2},
            "duration_precision": {"median_rel_err": 0.05, "sigma": 0.03, "n": 2},
            "fp_rate_cnn": 1.0, "fp_rate_claude": 0.0, "n_fp_processed": 1,
        },
        "coverage": {"completed": 1.0, "with_candidate": 1.0,
                     "with_completeness": 0.5},
    }
    md = scorecard_to_markdown(card)
    assert "recall" in md.lower()
    assert "0.500" in md          # the LOT recall leads the report
    assert "0.670" in md          # the cumulative one is reported, but labelled
    assert "depth" in md.lower()


# ---------------------------------------------------------------------------
# Schema v2 (2026-07-14): the scorecard must MEASURE THE LOT.
#
# The bug this pins down: metrics.recall_global was a COUNT over every KP/CP
# ever processed into the DB, with no filter on the run or its lot. Three
# consecutive runs over 100k / 300k / 400k completely different stars therefore
# reported recall=0.7231270358306189, n=307 — identical to the last decimal —
# because the sector's confirmed planets were already all processed and fresh
# stars add virtually none. Every A/B run against that number was measuring
# nothing.
# ---------------------------------------------------------------------------

def test_lot_recall_reacts_to_the_lot_while_cumulative_does_not():
    """The whole point: change the lot, the lot recall changes; the cumulative
    one does not budge. This is the regression that made the metric useless."""
    toi = _make_mini_toi()
    stars = _make_mini_stars()
    candidates = _make_mini_candidates()

    # TIC 1 is a KP we DO detect; TIC 3 is a CP we do NOT (see the fixtures).
    lot_hit = compute_recall_and_coverage(
        toi, stars, candidates, None, lot_tics={"1"},
        coverage_aggregates={"total_stars": 4, "completed_stars": 4,
                             "distinct_candidate_tics": 4,
                             "distinct_completeness_tics": 3})
    lot_miss = compute_recall_and_coverage(
        toi, stars, candidates, None, lot_tics={"3"},
        coverage_aggregates={"total_stars": 4, "completed_stars": 4,
                             "distinct_candidate_tics": 4,
                             "distinct_completeness_tics": 3})

    assert lot_hit["lot"]["recall"] == pytest.approx(1.0)
    assert lot_miss["lot"]["recall"] == pytest.approx(0.0)

    # ...while the cumulative figure is byte-identical across the two lots.
    assert lot_hit["cumulative"] == lot_miss["cumulative"]


def test_lot_with_no_confirmed_planets_reports_null_not_zero():
    """A 50k discovery lot normally contains ZERO KP/CP. Reporting 0.0 would
    read as 'we missed every planet'; the truth is 'this lot cannot measure
    recall'. The dossier has to say so in words, too."""
    toi = _make_mini_toi()
    stars = _make_mini_stars()
    candidates = _make_mini_candidates()

    rc = compute_recall_and_coverage(
        toi, stars, candidates, None,
        lot_tics={"9999"},          # a lot with no TOI in it at all
        coverage_aggregates={"total_stars": 4, "completed_stars": 4,
                             "distinct_candidate_tics": 4,
                             "distinct_completeness_tics": 3})

    assert rc["lot"]["recall"] is None
    assert rc["lot"]["measurable"] is False
    assert rc["lot"]["n_kp_cp"] == 0
    assert "reason" in rc["lot"]

    card = build_scorecard(
        toi, stars, candidates,
        pd.DataFrame([{"tic_id": "1"}, {"tic_id": "2"}]),
        run_id="r", sample_size=1, stratification="discovery",
        pipeline_version="v", seed=0, lot_tics={"9999"},
        lot_csv="data/discovery/s0014_lot_x.csv")
    md = scorecard_to_markdown(card)
    assert "n/a" in md
    assert "NO" in md and "valida" in md      # states it cannot validate the run
    assert card["metrics"]["lot"]["lot_csv"] == "data/discovery/s0014_lot_x.csv"


def test_a_tic_in_the_lot_but_never_processed_is_not_counted_as_a_miss():
    """Scoping is lot ∩ processed: a TIC the run never finished is not a miss
    against it (that would make an aborted run look like a recall regression)."""
    toi = _make_mini_toi()
    stars = _make_mini_stars()                     # TICs 1..4 processed
    candidates = _make_mini_candidates()

    rc = compute_recall_and_coverage(
        toi, stars, candidates, None,
        lot_tics={"1", "77777"},                   # 77777 never processed
        coverage_aggregates={"total_stars": 4, "completed_stars": 4,
                             "distinct_candidate_tics": 4,
                             "distinct_completeness_tics": 3})

    assert rc["lot"]["n_kp_cp"] == 1               # only TIC 1 counts
    assert rc["lot"]["recall"] == pytest.approx(1.0)
    assert rc["lot"]["lot_size"] == 2              # but the lot's size is honest
