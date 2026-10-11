"""Tests for scripts/augment_scorecard_exominer_from_db.py.

Covers the B9 DSN-footgun fix (2026-07-02): load_candidates() must route
through exonit.db_backend.op_session so a postgresql:// DSN behaves like a
SQLite path, instead of a bare sqlite3.connect() silently creating an empty
local file named after the DSN.
"""
from __future__ import annotations

import csv
import sqlite3

import pytest


@pytest.fixture
def candidates_db(tmp_path):
    """A small operational DB with a handful of candidates over 2 TICs."""
    db_path = tmp_path / "aug.db"
    conn = sqlite3.connect(str(db_path))
    conn.executescript("""
        CREATE TABLE candidates (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            tic_id TEXT,
            candidate_num INTEGER,
            period_days REAL,
            t0 REAL,
            duration_days REAL,
            depth_ppm REAL,
            snr REAL,
            cnn_score REAL,
            triceratops_fpp REAL,
            claude_verdict TEXT,
            exominer_score REAL,
            exominer_ffi_score REAL,
            exominer_sector_run TEXT,
            exominer_ffi_model_version TEXT
        );
    """)
    conn.execute(
        "INSERT INTO candidates (tic_id, candidate_num, cnn_score, "
        "exominer_score, exominer_ffi_score) VALUES ('100', 1, 0.9, 0.8, NULL)"
    )
    conn.execute(
        "INSERT INTO candidates (tic_id, candidate_num, cnn_score, "
        "exominer_score, exominer_ffi_score) VALUES ('200', 1, 0.85, NULL, 0.65)"
    )
    conn.commit()
    conn.close()
    return db_path


def test_load_candidates_reads_rows_by_tic(candidates_db):
    from scripts.augment_scorecard_exominer_from_db import load_candidates

    out = load_candidates(str(candidates_db), ["100", "200", "300"])
    assert set(out.keys()) == {"100", "200"}
    assert out["100"][0]["exominer_score"] == 0.8
    assert out["200"][0]["exominer_ffi_score"] == 0.65


def test_load_candidates_queries_every_chunk(candidates_db, monkeypatch):
    """With more tic_ids than the chunk size (500), every chunk must still
    be queried (never a single unbounded IN(...))."""
    from scripts import augment_scorecard_exominer_from_db as mod

    calls = []
    real_op_session = mod.op_session

    class _CountingSession:
        def __init__(self, inner):
            self._inner = inner

        def execute(self, sql, params=()):
            calls.append(len(params))
            return self._inner.execute(sql, params)

        def close(self):
            self._inner.close()

    monkeypatch.setattr(mod, "op_session", lambda db: _CountingSession(real_op_session(db)))
    tic_ids = ["100", "200"] + [str(1000 + i) for i in range(499)]  # 501 total
    out = mod.load_candidates(str(candidates_db), tic_ids)
    assert set(out.keys()) == {"100", "200"}
    # 501 ids at chunk=500 -> two queries, sized 500 then 1.
    assert calls == [500, 1]


def test_load_candidates_routes_through_op_session(candidates_db, monkeypatch):
    """B9 regression: must go through exonit.db_backend.op_session, not a
    bare sqlite3.connect (which would silently create an empty local file
    for a postgresql:// DSN)."""
    from scripts import augment_scorecard_exominer_from_db as mod

    calls = []
    real_op_session = mod.op_session

    def _spy(db_path):
        calls.append(db_path)
        return real_op_session(db_path)

    monkeypatch.setattr(mod, "op_session", _spy)
    mod.load_candidates(str(candidates_db), ["100"])
    assert calls == [str(candidates_db)]


def test_summarize_and_markdown_roundtrip(candidates_db, tmp_path):
    """End-to-end sanity: load_sample + load_candidates + summarize produce
    a coherent per-disposition rollup."""
    from scripts.augment_scorecard_exominer_from_db import (
        load_candidates, load_sample, summarize,
    )

    sample_csv = tmp_path / "sample.csv"
    with open(sample_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["tic_id", "disposition", "period_days",
                                          "depth_ppm", "duration_hours", "role"])
        w.writeheader()
        w.writerow({"tic_id": "100", "disposition": "CP", "period_days": "3.5",
                    "depth_ppm": "500", "duration_hours": "2", "role": "test"})
        w.writerow({"tic_id": "200", "disposition": "PC", "period_days": "1.2",
                    "depth_ppm": "300", "duration_hours": "1.5", "role": "test"})

    sample = load_sample(sample_csv)
    cands = load_candidates(str(candidates_db), list(sample.keys()))
    buckets = summarize(sample, cands, threshold=0.5)

    assert buckets["CP"]["em_pos"] == 1   # TIC 100: exominer_score=0.8 >= 0.5
    assert buckets["PC"]["ffi_pos"] == 1  # TIC 200: exominer_ffi_score=0.65 >= 0.5
