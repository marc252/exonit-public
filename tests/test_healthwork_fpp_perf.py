"""Parity test for /api/health/work's FPP-backfill counting rewrite.

Root cause (progress/impl-healthwork-perf.md): the OR-gate predicate
(cnn_score>=0.7 OR exominer_ffi_score>=0.5) widened the FPP-backfill
selection query from a few hundred rows to ~410k, and the endpoint's Python
loop then called ``MetadataCache.get_entry(tic_id)`` once PER CANDIDATE ROW
(one locked SQLite SELECT * + two json.loads each) — 18-30 s at that scale,
enough to blow the discovery orchestrator's poll timeout and fail a live
run.

The fix (exonit/routes/admin.py `_compute()`) replaces that per-row loop
with: GROUP BY tic_id on the SQL side (candidate COUNT per tic_id, not per
row), then one batched `MetadataCache.cached_tic_ids()` call (chunked
`tic_id IN (...)` lookups, no JSON parsing) instead of N separate
`get_entry()` calls.

This test proves the new code reports IDENTICAL `fpp_backfill.actionable` /
`fpp_backfill.unrecoverable` numbers to the OLD (pre-fix) per-row algorithm,
across a mix of scenarios that exercise every corner the rewrite has to get
right:
  - a tic_id with only an fpp_todo-arm candidate (cached -> actionable)
  - a tic_id with only an nfpp_todo-arm candidate (uncached -> unrecoverable)
  - a tic_id with candidates in BOTH arms (must sum counts across arms
    instead of only keeping the last-seen arm's count)
  - a tic_id eligible only via the exominer_ffi_score>=0.5 OR-gate branch
    (uncached -> unrecoverable)
  - candidates that must be EXCLUDED entirely (snr too low; star not
    'completed') — these must not leak into either bucket
"""

from __future__ import annotations

import os

from fastapi.testclient import TestClient

from exonit.db import ExoNitDB
from exonit.metadata_cache import MetadataCache


def _build_app_and_db(tmp_path, monkeypatch):
    db_path = str(tmp_path / "healthwork.db")
    mc_path = str(tmp_path / "metadata_cache.db")
    monkeypatch.setenv("EXONIT_DB_PATH", db_path)
    monkeypatch.setenv("EXONIT_REPORTS_DIR", str(tmp_path / "reports"))
    monkeypatch.setenv("EXONIT_METADATA_CACHE", mc_path)
    os.makedirs(tmp_path / "reports", exist_ok=True)

    from exonit.server import create_app
    app = create_app()
    client = TestClient(app)
    db = ExoNitDB(db_path)
    mc = MetadataCache(mc_path)
    return client, db, mc


def _add_star(db, tic_id, status="completed"):
    db.add_star(
        tic_id, ra=10.0, dec=-20.0, tmag=11.0,
        n_sectors=1, n_datapoints=1000, time_span_days=27.0,
        status=status,
    )


def _add_candidate(db, tic_id, candidate_num, snr=15.0, cnn_score=0.85,
                   exominer_ffi_score=None, triceratops_fpp=None,
                   triceratops_nfpp=None):
    return db.add_candidate(
        tic_id, candidate_num=candidate_num, period_days=3.0, t0=0.0,
        duration_days=0.1, depth_ppm=1000, snr=snr, cnn_score=cnn_score,
        exominer_ffi_score=exominer_ffi_score,
        triceratops_fpp=triceratops_fpp, triceratops_nfpp=triceratops_nfpp,
    )


def _reference_fpp_counts(db, mc):
    """The OLD (pre-fix) per-row algorithm, reproduced verbatim for parity.

    Mirrors exactly what exonit/routes/admin.py's `_compute()` did before
    this fix: one un-grouped row per eligible candidate, then a per-row
    `mc.get_entry(tic_id) is not None` check. This is the "ground truth"
    the rewritten (grouped + batched) code must match exactly.
    """
    fpp_candidates = db._query(
        """SELECT c.id, c.tic_id FROM candidates c
           JOIN stars s ON c.tic_id = s.tic_id
           WHERE c.triceratops_fpp IS NULL
             AND c.snr >= 10.0
             AND (c.cnn_score >= 0.7 OR c.exominer_ffi_score >= 0.5)
             AND s.status = 'completed'""")
    fpp_candidates += db._query(
        """SELECT c.id, c.tic_id FROM candidates c
           JOIN stars s ON c.tic_id = s.tic_id
           WHERE c.triceratops_nfpp IS NULL
             AND c.triceratops_fpp >= 0 AND c.triceratops_fpp < 0.5
             AND c.snr >= 10.0
             AND (c.cnn_score >= 0.7 OR c.exominer_ffi_score >= 0.5)
             AND s.status = 'completed'""")
    actionable = sum(1 for c in fpp_candidates if mc.get_entry(c["tic_id"]))
    unrecoverable = len(fpp_candidates) - actionable
    return actionable, unrecoverable, len(fpp_candidates)


def test_fpp_counting_parity_mixed_scenarios(tmp_path, monkeypatch):
    client, db, mc = _build_app_and_db(tmp_path, monkeypatch)

    # T_A: one fpp_todo-arm candidate, cached -> actionable.
    _add_star(db, "T_A")
    _add_candidate(db, "T_A", candidate_num=1, snr=15.0, cnn_score=0.8,
                    triceratops_fpp=None)
    mc.add_entry("T_A", sectors=[1], exptime=120.0, fits_files=["a.fits"])

    # T_B: one nfpp_todo-arm candidate, NOT cached -> unrecoverable.
    _add_star(db, "T_B")
    _add_candidate(db, "T_B", candidate_num=1, snr=12.0, cnn_score=0.75,
                    triceratops_fpp=0.2, triceratops_nfpp=None)

    # T_C: multi-candidate star with one candidate in EACH arm, cached ->
    # both must count as actionable (tests that counts are summed across
    # arms per tic_id, not overwritten).
    _add_star(db, "T_C")
    _add_candidate(db, "T_C", candidate_num=1, snr=14.0, cnn_score=0.9,
                    triceratops_fpp=None)
    _add_candidate(db, "T_C", candidate_num=2, snr=13.0, cnn_score=0.8,
                    triceratops_fpp=0.1, triceratops_nfpp=None)
    mc.add_entry("T_C", sectors=[1], exptime=120.0, fits_files=["c.fits"])

    # T_D: eligible only via the exominer_ffi_score>=0.5 OR-gate branch
    # (cnn_score below 0.7), NOT cached -> unrecoverable.
    _add_star(db, "T_D")
    _add_candidate(db, "T_D", candidate_num=1, snr=11.0, cnn_score=0.3,
                    exominer_ffi_score=0.6, triceratops_fpp=None)

    # T_E: snr below the 10.0 gate -> must be excluded entirely.
    _add_star(db, "T_E")
    _add_candidate(db, "T_E", candidate_num=1, snr=5.0, cnn_score=0.9,
                    triceratops_fpp=None)

    # T_F: star not 'completed' -> must be excluded entirely.
    _add_star(db, "T_F", status="processing")
    _add_candidate(db, "T_F", candidate_num=1, snr=20.0, cnn_score=0.9,
                    triceratops_fpp=None)

    ref_actionable, ref_unrecoverable, ref_total = _reference_fpp_counts(db, mc)

    # Hand-computed expectation, spelled out for readability:
    #   T_A: 1 candidate, cached      -> +1 actionable
    #   T_B: 1 candidate, not cached  -> +1 unrecoverable
    #   T_C: 2 candidates, cached     -> +2 actionable
    #   T_D: 1 candidate, not cached  -> +1 unrecoverable
    #   T_E, T_F: excluded
    assert ref_total == 5
    assert ref_actionable == 3
    assert ref_unrecoverable == 2

    resp = client.get("/api/health/work?nocache=1")
    assert resp.status_code == 200
    fpp = resp.json()["fpp_backfill"]

    assert fpp["actionable"] == ref_actionable == 3
    assert fpp["unrecoverable"] == ref_unrecoverable == 2

    db.close()
    mc.close()


def test_fpp_counting_parity_empty(tmp_path, monkeypatch):
    """No eligible candidates at all -> both counts are 0, endpoint doesn't
    error on an empty tic_id set (the batched cache lookup must tolerate
    zero ids)."""
    client, db, mc = _build_app_and_db(tmp_path, monkeypatch)

    resp = client.get("/api/health/work?nocache=1")
    assert resp.status_code == 200
    fpp = resp.json()["fpp_backfill"]
    assert fpp["actionable"] == 0
    assert fpp["unrecoverable"] == 0

    db.close()
    mc.close()


def test_fpp_counting_parity_large_queue(tmp_path, monkeypatch):
    """Reproduces the production shape at smaller scale: many distinct
    tic_ids (>5000, forcing multiple cached_tic_ids() chunks), most cached,
    a minority not — proving the grouped+batched path still matches the
    reference per-row algorithm once the cache lookup itself has to chunk."""
    client, db, mc = _build_app_and_db(tmp_path, monkeypatch)

    n_cached = 4000
    n_uncached = 2500
    for i in range(n_cached):
        tic = f"CACHED_{i}"
        _add_star(db, tic)
        _add_candidate(db, tic, candidate_num=1, snr=15.0, cnn_score=0.8,
                        triceratops_fpp=None)
        mc.add_entry(tic, sectors=[1], exptime=120.0, fits_files=[])
    for i in range(n_uncached):
        tic = f"UNCACHED_{i}"
        _add_star(db, tic)
        _add_candidate(db, tic, candidate_num=1, snr=15.0, cnn_score=0.8,
                        triceratops_fpp=None)

    ref_actionable, ref_unrecoverable, ref_total = _reference_fpp_counts(db, mc)
    assert ref_total == n_cached + n_uncached
    assert ref_actionable == n_cached
    assert ref_unrecoverable == n_uncached

    resp = client.get("/api/health/work?nocache=1")
    assert resp.status_code == 200
    fpp = resp.json()["fpp_backfill"]
    assert fpp["actionable"] == ref_actionable == n_cached
    assert fpp["unrecoverable"] == ref_unrecoverable == n_uncached

    db.close()
    mc.close()
