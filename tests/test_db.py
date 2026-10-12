"""Tests for SQLite database module."""

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from exonit.db import ExoNitDB, _is_concurrent_ddl_race, _to_epoch, utc_now_iso


@pytest.fixture
def db(tmp_path):
    db_path = str(tmp_path / "test.db")
    database = ExoNitDB(db_path)
    yield database
    database.close()


class TestExoNitDB:
    def test_creates_tables(self, db):
        # db.list_tables() was removed (2026-07-02 cleanup, minor item 8:
        # zero-caller sqlite_master-only helper) — query sqlite_master
        # directly; this fixture is always the SQLite path.
        tables = {r["name"] for r in db._query(
            "SELECT name FROM sqlite_master WHERE type='table'")}
        assert "stars" in tables
        assert "candidates" in tables
        assert "completeness" in tables

    def test_creates_has_spoc_partial_index(self, db):
        """P3 (docs/reviews/2026-07-02-opus48-cleanup-review.md): stars.has_spoc
        must have a partial index on the SQLite side (schema_pg.sql carries the
        PG equivalent — not exercised here, that file is hand-authored SQL)."""
        indexes = {r["name"] for r in db._query(
            "SELECT name FROM sqlite_master WHERE type='index'")}
        assert "idx_stars_has_spoc" in indexes

    def test_add_star(self, db):
        db.add_star("123456", ra=180.0, dec=-30.0, tmag=10.5,
                    n_sectors=4, n_datapoints=50000, time_span_days=100.0)
        star = db.get_star("123456")
        assert star is not None
        assert star["tic_id"] == "123456"
        assert star["status"] == "completed"

    def test_add_stars_bulk_inserts_all_rows(self, db):
        """Q9 (perf review 2026-09-26): add_stars_bulk must insert one row
        per TIC id, all sharing the given field values, in a single
        executemany call."""
        db.add_stars_bulk(
            ["100", "200", "300"], ra=0, dec=0, tmag=0,
            n_sectors=0, n_datapoints=0, time_span_days=0, status="no_fits")
        for tic in ("100", "200", "300"):
            star = db.get_star(tic)
            assert star is not None
            assert star["status"] == "no_fits"
            assert star["ra"] == 0
            assert star["tmag"] == 0

    def test_add_stars_bulk_empty_list_is_noop(self, db):
        """Must not raise, and must not touch the DB, for an empty list —
        the exact call shape enqueue_targets makes when nothing is no_fits."""
        db.add_stars_bulk(
            [], ra=0, dec=0, tmag=0,
            n_sectors=0, n_datapoints=0, time_span_days=0, status="no_fits")
        count = db._query_scalar("SELECT COUNT(*) FROM stars")
        assert count == 0

    def test_add_stars_bulk_upserts_existing_row(self, db):
        """A TIC already present in stars must be upserted (same REPLACE
        semantics as add_star), not duplicated."""
        db.add_star("500", ra=1.0, dec=2.0, tmag=9.0,
                    n_sectors=1, n_datapoints=100, time_span_days=10.0,
                    status="completed")
        db.add_stars_bulk(
            ["500"], ra=0, dec=0, tmag=0,
            n_sectors=0, n_datapoints=0, time_span_days=0, status="no_fits")
        star = db.get_star("500")
        assert star["status"] == "no_fits"
        count = db._query_scalar(
            "SELECT COUNT(*) FROM stars WHERE tic_id = '500'")
        assert count == 1

    def test_add_stars_bulk_matches_add_star_result(self, db, tmp_path):
        """A row written by add_stars_bulk must be indistinguishable from one
        written by add_star for the same inputs (aside from analyzed_at,
        which is a fresh timestamp either way)."""
        db.add_star("601", ra=5.0, dec=6.0, tmag=7.0,
                    n_sectors=2, n_datapoints=200, time_span_days=20.0,
                    status="no_fits")
        db.add_stars_bulk(
            ["602"], ra=5.0, dec=6.0, tmag=7.0,
            n_sectors=2, n_datapoints=200, time_span_days=20.0,
            status="no_fits")
        via_add_star = db.get_star("601")
        via_bulk = db.get_star("602")
        for key in ("ra", "dec", "tmag", "n_sectors", "n_datapoints",
                    "time_span_days", "status", "sectors_observed"):
            assert via_add_star[key] == via_bulk[key]

    def test_is_analyzed(self, db):
        assert db.is_analyzed("123456") is False
        db.add_star("123456", ra=0, dec=0, tmag=10.0,
                    n_sectors=1, n_datapoints=1000, time_span_days=30.0)
        assert db.is_analyzed("123456") is True

    def test_add_candidate(self, db):
        db.add_star("123456", ra=0, dec=0, tmag=10.0,
                    n_sectors=1, n_datapoints=1000, time_span_days=30.0)
        db.add_candidate("123456", candidate_num=1, period_days=5.0,
                         t0=1000.0, duration_days=0.1, depth_ppm=500.0, snr=12.5)
        candidates = db.get_candidates("123456")
        assert len(candidates) == 1
        assert candidates[0]["period_days"] == 5.0

    def test_add_candidate_default_sector_is_1(self, db):
        """Omitting sector keeps the historical default (1) — no regression."""
        db.add_star("777", ra=0, dec=0, tmag=10.0,
                    n_sectors=1, n_datapoints=1000, time_span_days=30.0)
        db.add_candidate("777", candidate_num=1, period_days=5.0,
                         t0=0, duration_days=0.1, depth_ppm=500.0, snr=12.5)
        row = db._query("SELECT sector FROM candidates WHERE tic_id='777'")[0]
        assert row["sector"] == 1

    def test_add_candidate_records_explicit_sector(self, db):
        """An explicit sector (e.g. an active s23 discovery run) is persisted."""
        db.add_star("888", ra=0, dec=0, tmag=10.0,
                    n_sectors=1, n_datapoints=1000, time_span_days=30.0)
        db.add_candidate("888", candidate_num=1, period_days=5.0,
                         t0=0, duration_days=0.1, depth_ppm=500.0, snr=12.5,
                         sector=23)
        row = db._query("SELECT sector FROM candidates WHERE tic_id='888'")[0]
        assert row["sector"] == 23

    def test_add_candidate_ignores_retired_v8_kwarg(self, db):
        """v8 residue removal (2026-07-02 cleanup, item 5 + 11(d)): the
        exominer_ffi_score_v8 kwarg is still ACCEPTED for backward
        compatibility (server.py's own call site no longer passes it as of
        11(d)) but no longer written anywhere; the column stays NULL and
        every other field still saves."""
        db.add_star("999", ra=0, dec=0, tmag=10.0,
                    n_sectors=1, n_datapoints=1000, time_span_days=30.0)
        db.add_candidate("999", candidate_num=1, period_days=5.0,
                         t0=0, duration_days=0.1, depth_ppm=500.0, snr=12.5,
                         exominer_ffi_score=0.8, exominer_ffi_score_v8=0.9)
        row = db._query(
            "SELECT period_days, exominer_ffi_score, exominer_ffi_score_v8 "
            "FROM candidates WHERE tic_id='999'")[0]
        assert row["period_days"] == 5.0
        assert row["exominer_ffi_score"] == 0.8
        assert row["exominer_ffi_score_v8"] is None  # never written

    def test_get_stats(self, db):
        db.add_star("111", ra=0, dec=0, tmag=10.0,
                    n_sectors=1, n_datapoints=1000, time_span_days=30.0)
        db.add_star("222", ra=0, dec=0, tmag=11.0,
                    n_sectors=2, n_datapoints=2000, time_span_days=60.0)
        db.add_candidate("111", candidate_num=1, period_days=3.0,
                         t0=1000.0, duration_days=0.1, depth_ppm=200.0, snr=8.0)
        stats = db.get_stats()
        assert stats["total_stars"] == 2
        assert stats["total_candidates"] == 1

    # --- add_completeness_many: batched completeness writes (Phase 2) ---
    # The injection-recovery grid is ~72 cells; the old code called
    # add_completeness() (one commit each) in a loop — the dominant source of
    # write amplification per result. add_completeness_many writes them all in
    # ONE transaction. Behaviour must be identical to the per-cell loop.

    def test_add_completeness_many_inserts_all_cells(self, db):
        cells = [(r, p, 0.5 + 0.001 * (r + p))
                 for r in (1, 2, 4, 8) for p in (5, 10, 20)]  # 12 cells
        db.add_completeness_many("123456", cells)
        rows = db._query("SELECT * FROM completeness WHERE tic_id = ?", ("123456",))
        assert len(rows) == len(cells)

    def test_add_completeness_many_matches_individual_loop(self, db):
        """Identical result to calling add_completeness() per cell."""
        cells = [(1.0, 5.0, 0.1), (2.0, 10.0, 0.2), (4.0, 20.0, 0.3)]
        for r, p, d in cells:
            db.add_completeness("AAA", r, p, d)
        db.add_completeness_many("BBB", cells)
        a = sorted((r["radius_earth"], r["period_days"], r["detection_prob"])
                   for r in db._query("SELECT * FROM completeness WHERE tic_id='AAA'"))
        b = sorted((r["radius_earth"], r["period_days"], r["detection_prob"])
                   for r in db._query("SELECT * FROM completeness WHERE tic_id='BBB'"))
        assert a == b

    def test_add_completeness_many_replaces_on_rerun(self, db):
        """INSERT OR REPLACE semantics: same (tic,radius,period) key overwrites."""
        db.add_completeness_many("123456", [(1.0, 5.0, 0.10)])
        db.add_completeness_many("123456", [(1.0, 5.0, 0.99)])  # same key, new prob
        rows = db._query("SELECT * FROM completeness WHERE tic_id='123456'")
        assert len(rows) == 1
        assert rows[0]["detection_prob"] == 0.99

    def test_add_completeness_many_empty_is_noop(self, db):
        db.add_completeness_many("123456", [])
        rows = db._query("SELECT * FROM completeness WHERE tic_id='123456'")
        assert rows == []

    # --- wal_checkpoint (Phase 3 #7) ---

    def test_wal_checkpoint_returns_triplet(self, db):
        # write something so there are WAL frames to checkpoint
        db.add_star("123456", ra=0, dec=0, tmag=10.0,
                    n_sectors=1, n_datapoints=1000, time_span_days=30.0)
        result = db.wal_checkpoint("TRUNCATE")
        assert result is not None and len(result) == 3
        busy, log_frames, checkpointed = result
        assert busy in (0, 1)  # 0 = fully checkpointed, 1 = a reader blocked truncate

    def test_wal_checkpoint_bad_mode_defaults_safe(self, db):
        db.add_star("999", ra=0, dec=0, tmag=10.0,
                    n_sectors=1, n_datapoints=1000, time_span_days=30.0)
        # An unknown mode must not raise (defaults to TRUNCATE).
        assert db.wal_checkpoint("NONSENSE") is not None

    # --- telemetry swap_used_gb (Phase 3 #8) ---

    def test_insert_telemetry_stores_swap_and_rss(self, db):
        db.insert_telemetry("w1", {
            "status": "processing",
            "system": {"swap_used_gb": 3.5, "pid_rss_mb": 800.0,
                       "ram_used_gb": 12.0, "cpu_percent": 55.0},
        })
        rows = db._query(
            "SELECT swap_used_gb, pid_rss_mb FROM worker_telemetry "
            "WHERE worker_id='w1'")
        assert rows, "telemetry row not written"
        assert rows[0]["swap_used_gb"] == 3.5
        assert rows[0]["pid_rss_mb"] == 800.0

    def test_insert_telemetry_swap_optional(self, db):
        # Old worker payloads without a system block must still insert (NULL).
        db.insert_telemetry("w2", {"status": "idle"})
        rows = db._query(
            "SELECT swap_used_gb FROM worker_telemetry WHERE worker_id='w2'")
        assert rows and rows[0]["swap_used_gb"] is None


class TestUpdateCandidate:
    def test_update_fpp(self, db):
        db.add_star("100", ra=0, dec=0, tmag=10, n_sectors=1,
                    n_datapoints=1000, time_span_days=30)
        db.add_candidate("100", candidate_num=1, period_days=5.0, t0=0,
                         duration_days=0.1, depth_ppm=500, snr=15.0)
        db.update_candidate_fpp(candidate_id=1, fpp=0.023, nfpp=0.01,
                                prob_planet=0.85)
        c = db.get_candidates("100")[0]
        assert c["triceratops_fpp"] == 0.023
        assert c["triceratops_nfpp"] == 0.01
        assert c["triceratops_prob_planet"] == 0.85
        assert c["triceratops_verdict"] == "candidate"

    def test_failed_nfpp_backfill_preserves_existing_fpp(self, db):
        db.add_star("110", ra=0, dec=0, tmag=10, n_sectors=1,
                    n_datapoints=1000, time_span_days=30)
        candidate_id = db.add_candidate(
            "110", candidate_num=1, period_days=5.0, t0=0,
            duration_days=0.1, depth_ppm=500, snr=15.0)
        db.update_candidate_fpp(candidate_id, fpp=0.01, nfpp=None)
        db.update_candidate_fpp(candidate_id, fpp=-1.0, nfpp=-1.0,
                                verdict="failed")
        c = db.get_candidates("110")[0]
        assert c["triceratops_fpp"] == pytest.approx(0.01)
        assert c["triceratops_nfpp"] == pytest.approx(-1.0)
        assert c["triceratops_verdict"] == "candidate"

    def test_update_claude(self, db):
        db.add_star("200", ra=0, dec=0, tmag=10, n_sectors=1,
                    n_datapoints=1000, time_span_days=30)
        db.add_candidate("200", candidate_num=1, period_days=5.0, t0=0,
                         duration_days=0.1, depth_ppm=500, snr=15.0)
        db.update_candidate_claude(candidate_id=1, verdict="planet_candidate",
                                   analysis="Strong transit signal")
        c = db.get_candidates("200")[0]
        assert c["claude_verdict"] == "planet_candidate"
        assert c["claude_analysis"] == "Strong transit signal"

    def test_verdict_proba_column_and_writes(self, db):
        """P3: column exists; kwarg writes in both UPDATE branches; legacy
        call leaves it untouched."""
        cols = [r["name"] for r in db._query("PRAGMA table_info(candidates)")]
        assert "verdict_proba" in cols
        db.add_star("201", ra=0, dec=0, tmag=10, n_sectors=1,
                    n_datapoints=1000, time_span_days=30)
        db.add_candidate("201", candidate_num=1, period_days=5.0, t0=0,
                         duration_days=0.1, depth_ppm=500, snr=15.0)
        assert db.get_candidates("201")[0]["verdict_proba"] is None
        # normal branch
        db.update_candidate_claude(1, "planet_candidate", "",
                                   verdict_proba=0.42)
        assert db.get_candidates("201")[0]["verdict_proba"] == pytest.approx(0.42)
        # legacy call (no kwarg) does not touch it
        db.update_candidate_claude(1, "uncertain", "")
        c = db.get_candidates("201")[0]
        assert c["claude_verdict"] == "uncertain"
        assert c["verdict_proba"] == pytest.approx(0.42)
        # explicit None clears it
        db.update_candidate_claude(1, "uncertain", "", verdict_proba=None)
        assert db.get_candidates("201")[0]["verdict_proba"] is None
        # keep-lora branch
        db.update_candidate_claude(1, "uncertain", "[lora-v2] good text here")
        db.update_candidate_claude(1, "planet_candidate", "", verdict_proba=0.7)
        c = db.get_candidates("201")[0]
        assert c["claude_analysis"] == "[lora-v2] good text here"
        assert c["claude_verdict"] == "planet_candidate"
        assert c["verdict_proba"] == pytest.approx(0.7)

    def test_update_exominer(self, db):
        db.add_star("250", ra=0, dec=0, tmag=10, n_sectors=1,
                    n_datapoints=1000, time_span_days=30)
        db.add_candidate("250", candidate_num=1, period_days=14.64, t0=0,
                         duration_days=0.1, depth_ppm=245, snr=11.4)
        db.update_candidate_exominer(
            candidate_id=1, score=0.87, tce_period=14.64, sector_run="1-69")
        c = db.get_candidates("250")[0]
        assert c["exominer_score"] == pytest.approx(0.87)
        assert c["exominer_tce_period"] == pytest.approx(14.64)
        assert c["exominer_sector_run"] == "1-69"

    def test_exominer_columns_default_null(self, db):
        """Newly-added candidates default to NULL on ExoMiner columns."""
        db.add_star("260", ra=0, dec=0, tmag=10, n_sectors=1,
                    n_datapoints=1000, time_span_days=30)
        db.add_candidate("260", candidate_num=1, period_days=5.0, t0=0,
                         duration_days=0.1, depth_ppm=500, snr=15.0)
        c = db.get_candidates("260")[0]
        assert c["exominer_score"] is None
        assert c["exominer_tce_period"] is None
        assert c["exominer_sector_run"] is None

    def test_add_candidate_preserves_analysis_on_reprocess(self, db):
        db.add_star("270", ra=0, dec=0, tmag=10, n_sectors=1,
                    n_datapoints=1000, time_span_days=30)
        candidate_id = db.add_candidate(
            "270", candidate_num=1, period_days=5.0, t0=0,
            duration_days=0.1, depth_ppm=500, snr=15.0, cnn_score=0.91)
        db.update_candidate_fpp(candidate_id, fpp=0.01, nfpp=0.0005,
                                prob_planet=0.98)
        db.update_candidate_exominer(candidate_id, score=0.87,
                                     tce_period=5.0, sector_run="1-69")
        db.update_candidate_claude(candidate_id, verdict="planet_candidate",
                                   analysis="Looks clean")

        db.add_candidate("270", candidate_num=1, period_days=5.1, t0=0.2,
                         duration_days=0.12, depth_ppm=520, snr=16.0)

        candidates = db.get_candidates("270")
        assert len(candidates) == 1
        c = candidates[0]
        assert c["period_days"] == pytest.approx(5.1)
        assert c["triceratops_fpp"] == pytest.approx(0.01)
        assert c["triceratops_nfpp"] == pytest.approx(0.0005)
        assert c["triceratops_verdict"] == "validated"
        assert c["exominer_score"] == pytest.approx(0.87)
        assert c["claude_verdict"] == "planet_candidate"

    def test_get_candidates_needing_analysis(self, db):
        db.add_star("300", ra=0, dec=0, tmag=10, n_sectors=1,
                    n_datapoints=1000, time_span_days=30)
        db.add_candidate("300", candidate_num=1, period_days=5.0, t0=0,
                         duration_days=0.1, depth_ppm=500, snr=15.0,
                         cnn_score=0.95)
        db.add_candidate("300", candidate_num=2, period_days=3.0, t0=0,
                         duration_days=0.1, depth_ppm=200, snr=8.0,
                         cnn_score=0.5)
        pending = db.get_candidates_needing_analysis(min_snr=10.0,
                                                      min_cnn=0.7)
        assert len(pending) == 1
        assert pending[0]["candidate_num"] == 1


    def test_get_candidates_needing_analysis_backfills_missing_nfpp(self, db):
        db.add_star("310", ra=0, dec=0, tmag=10, n_sectors=1,
                    n_datapoints=1000, time_span_days=30)
        candidate_id = db.add_candidate(
            "310", candidate_num=1, period_days=5.0, t0=0,
            duration_days=0.1, depth_ppm=500, snr=15.0, cnn_score=0.95)
        db._exec(
            """UPDATE candidates SET triceratops_fpp = 0.01,
                      triceratops_nfpp = NULL, triceratops_verdict = 'candidate'
               WHERE id = ?""",
            (candidate_id,))
        pending = db.get_candidates_needing_analysis(min_snr=10.0,
                                                      min_cnn=0.7)
        assert len(pending) == 1
        assert pending[0]["candidate_num"] == 1


class TestUpdateCandidateClaudeBatch:
    """Tests for db.update_candidate_claude_batch."""

    def _add_candidate(self, db, tic, num=1):
        db.add_star(tic, ra=0, dec=0, tmag=10, n_sectors=1,
                    n_datapoints=1000, time_span_days=30)
        return db.add_candidate(tic, candidate_num=num, period_days=5.0,
                                t0=0, duration_days=0.1, depth_ppm=500,
                                snr=15.0)

    def test_empty_batch_returns_zero(self, db):
        n = db.update_candidate_claude_batch([])
        assert n == 0

    def test_single_update_writes_verdict_and_analysis(self, db):
        cid = self._add_candidate(db, "400")
        n = db.update_candidate_claude_batch([{
            "candidate_id": cid,
            "verdict": "planet_candidate",
            "analysis": "[lora-v2] Good transit",
        }])
        assert n == 1
        c = db.get_candidates("400")[0]
        assert c["claude_verdict"] == "planet_candidate"
        assert c["claude_analysis"] == "[lora-v2] Good transit"

    def test_batch_writes_all_in_one_transaction(self, db):
        """Two candidates updated in a single call; both should be present."""
        cid1 = self._add_candidate(db, "410", num=1)
        cid2 = self._add_candidate(db, "420", num=1)
        n = db.update_candidate_claude_batch([
            {"candidate_id": cid1, "verdict": "likely_fp",
             "analysis": "[lora-v2] FP signal"},
            {"candidate_id": cid2, "verdict": "planet_candidate",
             "analysis": "[lora-v2] Clean transit"},
        ])
        assert n == 2
        assert db.get_candidates("410")[0]["claude_verdict"] == "likely_fp"
        assert db.get_candidates("420")[0]["claude_verdict"] == "planet_candidate"

    def test_lora_preservation_incoming_not_lora(self, db):
        """Existing [lora-v2] analysis must survive if incoming is not tagged."""
        cid = self._add_candidate(db, "430")
        # Set up existing good [lora-v2] analysis.
        db.update_candidate_claude(cid, verdict="planet_candidate",
                                   analysis="[lora-v2] Existing good analysis text")
        # Attempt to overwrite with a shorter, untagged analysis.
        n = db.update_candidate_claude_batch([{
            "candidate_id": cid,
            "verdict": "unknown",
            "analysis": "Short fallback",
        }])
        assert n == 1
        c = db.get_candidates("430")[0]
        # Verdict updated, analysis preserved.
        assert c["claude_verdict"] == "unknown"
        assert c["claude_analysis"] == "[lora-v2] Existing good analysis text"

    def test_lora_incoming_overwrites_non_lora_existing(self, db):
        """Incoming [lora-v2] analysis must overwrite a non-lora existing one."""
        cid = self._add_candidate(db, "440")
        db.update_candidate_claude(cid, verdict="unknown",
                                   analysis="Old plain text")
        db.update_candidate_claude_batch([{
            "candidate_id": cid,
            "verdict": "planet_candidate",
            "analysis": "[lora-v2] Fresh LoRA explanation",
        }])
        c = db.get_candidates("440")[0]
        assert c["claude_analysis"] == "[lora-v2] Fresh LoRA explanation"

    def test_null_verdict_only_updates_analysis(self, db):
        """verdict=None (key present) should still update the analysis column."""
        cid = self._add_candidate(db, "450")
        db.update_candidate_claude_batch([{
            "candidate_id": cid,
            "verdict": None,
            "analysis": "[lora-v2] Analysis without verdict",
        }])
        c = db.get_candidates("450")[0]
        assert c["claude_verdict"] is None
        assert c["claude_analysis"] == "[lora-v2] Analysis without verdict"

    def test_absent_verdict_key_preserves_existing_verdict(self, db):
        """CRITICAL regression guard: when 'verdict' key is absent from the
        batch entry, an existing claude_verdict must NOT be clobbered.

        This is the exact bug the reviewer flagged: a candidate already has
        claude_verdict='planet_candidate' from a previous run; the LoRA
        backfill only regenerates the analysis and sends the batch entry
        WITHOUT a 'verdict' key.  The existing verdict must survive.
        """
        cid = self._add_candidate(db, "460")
        # Establish a pre-existing verdict.
        db.update_candidate_claude(cid, verdict="planet_candidate",
                                   analysis="Old plain text")

        # Batch entry without "verdict" key — only update analysis.
        db.update_candidate_claude_batch([{
            "candidate_id": cid,
            "analysis": "[lora-v2] Fresh LoRA explanation",
        }])
        c = db.get_candidates("460")[0]
        # verdict must be PRESERVED, analysis must be updated.
        assert c["claude_verdict"] == "planet_candidate", (
            "claude_verdict was clobbered; 'verdict' key absent should mean "
            "no write to claude_verdict"
        )
        assert c["claude_analysis"] == "[lora-v2] Fresh LoRA explanation"

    def test_absent_verdict_key_with_lora_existing_preserves_both(self, db):
        """When verdict key absent AND existing analysis is [lora-v2] that
        wins preservation (incoming is also [lora-v2] so it overwrites);
        more importantly: if existing is lora and incoming is shorter
        non-lora, and verdict key absent → nothing should be written at all,
        preserving both columns.
        """
        cid = self._add_candidate(db, "470")
        db.update_candidate_claude(cid, verdict="planet_candidate",
                                   analysis="[lora-v2] Long existing lora explanation")

        # Incoming: no verdict key, shorter non-lora analysis.
        # keep_existing_analysis=True AND has_verdict=False → nothing written.
        db.update_candidate_claude_batch([{
            "candidate_id": cid,
            "analysis": "Short plain",
        }])
        c = db.get_candidates("470")[0]
        assert c["claude_verdict"] == "planet_candidate"
        assert c["claude_analysis"] == "[lora-v2] Long existing lora explanation"


# ------------------------------------------------------------------ #
# Tests for reassign_stale_tasks with worker_current_tic
# ------------------------------------------------------------------ #

class TestReassignStaleTasksCurrentTic:
    """Anti-regression tests for the worker_current_tic extension to
    reassign_stale_tasks.

    Scenario: a worker loses a result-POST and moves to the next TIC.
    The old task stays 'assigned' to a worker that is still alive (heartbeat
    recent), so the old alive-set guard would protect it forever.
    The fix: compare current_tic against the task's tic_id and reassign
    when they differ.
    """

    OLD_TIME = "2020-01-01T00:00:00"  # far in the past, always > any timeout

    def _add_stale_task(self, db, task_id, tic_id, worker_id):
        """Insert a task and backdate its assigned_at so it is always stale."""
        db.add_task(task_id, tic_id)
        db.assign_task(task_id, worker_id)
        db.conn.execute(
            "UPDATE tasks SET assigned_at=? WHERE task_id=?",
            (self.OLD_TIME, task_id))
        db.conn.commit()

    # Case 1: worker alive, current_tic == task's tic, below abandon → keep
    def test_alive_worker_on_same_tic_not_reassigned(self, db):
        self._add_stale_task(db, "t1", "TIC-A", "w1")
        reassigned = db.reassign_stale_tasks(
            timeout_seconds=100,
            alive_worker_ids={"w1"},
            worker_current_tic={"w1": "TIC-A"},
        )
        assert reassigned == 0
        row = db.conn.execute(
            "SELECT status FROM tasks WHERE task_id='t1'").fetchone()
        assert row["status"] == "assigned"

    # Case 2: worker alive, current_tic == None (idle) → reassign (THE BUG FIX)
    def test_alive_worker_idle_current_tic_none_reassigned(self, db):
        self._add_stale_task(db, "t2", "TIC-B", "w2")
        reassigned = db.reassign_stale_tasks(
            timeout_seconds=100,
            alive_worker_ids={"w2"},
            worker_current_tic={"w2": None},
        )
        assert reassigned == 1
        row = db.conn.execute(
            "SELECT status, assigned_to FROM tasks WHERE task_id='t2'").fetchone()
        assert row["status"] == "pending"
        assert row["assigned_to"] is None

    # Case 3: worker alive, current_tic == different TIC → reassign
    def test_alive_worker_on_different_tic_reassigned(self, db):
        self._add_stale_task(db, "t3", "TIC-C", "w3")
        reassigned = db.reassign_stale_tasks(
            timeout_seconds=100,
            alive_worker_ids={"w3"},
            worker_current_tic={"w3": "TIC-OTHER"},
        )
        assert reassigned == 1
        row = db.conn.execute(
            "SELECT status FROM tasks WHERE task_id='t3'").fetchone()
        assert row["status"] == "pending"

    # Case 4: worker NOT alive (not in alive set), worker_current_tic provided → reassign
    def test_dead_worker_reassigned_with_current_tic_map(self, db):
        self._add_stale_task(db, "t4", "TIC-D", "w4")
        reassigned = db.reassign_stale_tasks(
            timeout_seconds=100,
            alive_worker_ids=set(),          # w4 not alive
            worker_current_tic={"w4": "TIC-D"},  # even though same tic
        )
        assert reassigned == 1
        row = db.conn.execute(
            "SELECT status FROM tasks WHERE task_id='t4'").fetchone()
        assert row["status"] == "pending"

    # Case 5: age < timeout → never reassigned regardless of state
    def test_fresh_task_never_reassigned(self, db):
        db.add_task("t5", "TIC-E")
        db.assign_task("t5", "w5")
        # assigned_at is NOW — well within timeout
        reassigned = db.reassign_stale_tasks(
            timeout_seconds=9999,
            alive_worker_ids=set(),          # even dead worker
            worker_current_tic={"w5": None},
        )
        assert reassigned == 0
        row = db.conn.execute(
            "SELECT status FROM tasks WHERE task_id='t5'").fetchone()
        assert row["status"] == "assigned"

    # Case 6: age > abandon_timeout → reassigned even when current_tic matches
    def test_abandon_timeout_overrides_matching_tic(self, db):
        self._add_stale_task(db, "t6", "TIC-F", "w6")
        # Task was assigned in 2020 → age is years; abandon_timeout=1s triggers
        reassigned = db.reassign_stale_tasks(
            timeout_seconds=100,
            alive_worker_ids={"w6"},
            worker_current_tic={"w6": "TIC-F"},  # same tic — would normally skip
            abandon_timeout_seconds=1,             # anything < task age → abandon
        )
        assert reassigned == 1
        row = db.conn.execute(
            "SELECT status FROM tasks WHERE task_id='t6'").fetchone()
        assert row["status"] == "pending"

    # Case 7: backward-compat — worker_current_tic=None, worker alive → skip (old behaviour)
    def test_backward_compat_alive_worker_no_current_tic_map_not_reassigned(self, db):
        self._add_stale_task(db, "t7", "TIC-G", "w7")
        reassigned = db.reassign_stale_tasks(
            timeout_seconds=100,
            alive_worker_ids={"w7"},
            worker_current_tic=None,   # old callers omit this
        )
        assert reassigned == 0
        row = db.conn.execute(
            "SELECT status FROM tasks WHERE task_id='t7'").fetchone()
        assert row["status"] == "assigned"

    # Extra: int vs str tic_id normalisation — current_tic as int should still match str in DB
    def test_tic_id_type_normalisation_int_vs_str(self, db):
        """worker_current_tic may carry int TIC IDs (JSON-decoded heartbeat);
        must NOT cause a false-mismatch that would reassign a live task."""
        self._add_stale_task(db, "t8", "12345", "w8")  # DB stores tic_id as TEXT
        reassigned = db.reassign_stale_tasks(
            timeout_seconds=100,
            alive_worker_ids={"w8"},
            worker_current_tic={"w8": 12345},   # int, same logical value
        )
        assert reassigned == 0, (
            "int current_tic must compare equal to str tic_id in DB "
            "— reassigning a live task would be a double-assignment bug"
        )


# ------------------------------------------------------------------ #
# Regression (2026-07-02): naive-local timestamp bug.
#
# db.py used to write ``datetime.now().isoformat()`` — NAIVE LOCAL time
# (Europe/Madrid, UTC+1/+2) — into DB timestamp columns. On PostgreSQL
# (timestamptz) a naive string is interpreted per the session timezone
# (UTC), so the local wall-clock value landed ~1-2h in the future — which
# silently defeated the orphan-task auto-release (age = now() - assigned_at
# came out negative and never crossed the 1800s stale threshold). SQLite
# itself wasn't broken (writer and reader both assumed local time
# consistently), but the fix (utc_now_iso(), tz-aware UTC everywhere) must
# not regress SQLite, and old naive-format rows must keep working during
# the mixed-data transition (see progress/impl-tzfix-2026-07-02.md).
# ------------------------------------------------------------------ #

class TestUtcNowIso:
    def test_returns_tz_aware_utc(self):
        s = utc_now_iso()
        parsed = datetime.fromisoformat(s)
        assert parsed.tzinfo is not None
        assert parsed.utcoffset().total_seconds() == 0

    def test_round_trips_through_to_epoch_as_current_time(self):
        s = utc_now_iso()
        epoch = _to_epoch(s)
        now = datetime.now(timezone.utc).timestamp()
        assert abs(epoch - now) < 2  # generous wall-clock-call tolerance


class TestAssignedAtIsTzAwareAndAgeNonNegative:
    """Anti-regression: the exact production symptom (orphan-release age
    silently going negative) must not happen via the public write paths."""

    def test_assign_task_assigned_at_parses_tz_aware(self, db):
        db.add_task("ta1", "TIC-TZ1")
        db.assign_task("ta1", "wtz1")
        row = db.get_task("ta1")
        parsed = datetime.fromisoformat(row["assigned_at"])
        assert parsed.tzinfo is not None

    def test_get_and_assign_task_assigned_at_parses_tz_aware(self, db):
        db.add_task("ta2", "TIC-TZ2")
        task = db.get_and_assign_task("wtz2")
        parsed = datetime.fromisoformat(task["assigned_at"])
        assert parsed.tzinfo is not None

    def test_assign_task_age_is_non_negative(self, db):
        """The exact orphan-release computation (reassign_stale_tasks'
        cutoff logic) applied to a task assigned "just now" must never see
        a negative age — that was the production bug's signature."""
        db.add_task("ta3", "TIC-TZ3")
        db.assign_task("ta3", "wtz3")
        row = db.get_task("ta3")
        assigned_epoch = _to_epoch(row["assigned_at"])
        now_epoch = datetime.now(timezone.utc).timestamp()
        age = now_epoch - assigned_epoch
        assert age >= 0, f"orphan-release age went negative ({age:.3f}s) — regressed"
        assert age < 5

    def test_add_star_analyzed_at_parses_tz_aware(self, db):
        db.add_star("TIC-TZ4", ra=0, dec=0, tmag=10.0,
                    n_sectors=1, n_datapoints=1000, time_span_days=30.0)
        star = db.get_star("TIC-TZ4")
        parsed = datetime.fromisoformat(star["analyzed_at"])
        assert parsed.tzinfo is not None

    def test_complete_task_completed_at_parses_tz_aware(self, db):
        db.add_task("ta5", "TIC-TZ5")
        db.assign_task("ta5", "wtz5")
        db.complete_task("ta5", processing_time=1.0)
        row = db.get_task("ta5")
        parsed = datetime.fromisoformat(row["completed_at"])
        assert parsed.tzinfo is not None

    def test_requeue_or_fail_task_completed_at_parses_tz_aware(self, db):
        db.add_task("ta6", "TIC-TZ6")
        for _ in range(3):  # exhaust MAX_TASK_ATTEMPTS to hit the failed path
            db.assign_task("ta6", "wtz6")
            db.requeue_or_fail_task("ta6", error="boom")
        row = db.get_task("ta6")
        assert row["status"] == "failed"
        parsed = datetime.fromisoformat(row["completed_at"])
        assert parsed.tzinfo is not None


class TestMixedNaiveAndTzAwareRows:
    """Old rows written before this fix are NAIVE-local strings (e.g. the
    OLD_TIME fixture above, ``"2020-01-01T00:00:00"``). New rows are
    tz-aware UTC. Both must keep working through _to_epoch() /
    reassign_stale_tasks() during the transition — this is what makes the
    fix deployable without a data migration or a live-run interruption."""

    def test_to_epoch_handles_naive_legacy_string(self):
        # Old-format string (no offset) — _to_epoch must not raise/return None.
        assert _to_epoch("2020-01-01T00:00:00") is not None

    def test_to_epoch_handles_new_tz_aware_string(self):
        assert _to_epoch(utc_now_iso()) is not None

    def test_reassign_stale_tasks_reassigns_old_naive_format_row(self, db):
        """A pre-fix (naive-format) stale row must still be recognised as
        stale and reassigned — old data isn't stranded by the format change."""
        db.add_task("tm1", "TIC-MIX1")
        db.assign_task("tm1", "wmix1")
        db.conn.execute(
            "UPDATE tasks SET assigned_at=? WHERE task_id=?",
            ("2020-01-01T00:00:00", "tm1"))
        db.conn.commit()
        reassigned = db.reassign_stale_tasks(timeout_seconds=100)
        assert reassigned == 1

    def test_reassign_stale_tasks_reassigns_new_tz_aware_format_row(self, db):
        """A post-fix (tz-aware) stale row must be recognised as stale too."""
        db.add_task("tm2", "TIC-MIX2")
        db.assign_task("tm2", "wmix2")
        old_aware = datetime(2020, 1, 1, tzinfo=timezone.utc).isoformat()
        db.conn.execute(
            "UPDATE tasks SET assigned_at=? WHERE task_id=?",
            (old_aware, "tm2"))
        db.conn.commit()
        reassigned = db.reassign_stale_tasks(timeout_seconds=100)
        assert reassigned == 1


# --------------------------------------------------------------------------- #
# Fase 4a Task 4 follow-up (e): _ensure_completeness_pg_schema concurrent
# CREATE-IF-NOT-EXISTS race guard. No live PostgreSQL needed here — a fake
# ``self.backend`` plus real psycopg error CLASSES (for their ``sqlstate``
# class attribute) is enough to exercise the guard in isolation. Skips
# cleanly if psycopg is not installed (matches the project's existing
# "skip cleanly when PG/psycopg unavailable" convention for PG-adjacent
# tests, e.g. tests/_pg_test_db.py).
# --------------------------------------------------------------------------- #

class _FakeCmplBackend:
    """Records every ``execute`` call; raises *exc* on the first call only."""

    def __init__(self, exc=None):
        self.calls = []
        self._exc = exc
        self._raised_once = False

    def execute(self, sql, params=()):
        self.calls.append(sql)
        if self._exc is not None and not self._raised_once:
            self._raised_once = True
            raise self._exc


def _fake_db(exc=None):
    """A bare object with just enough shape for _ensure_completeness_pg_schema
    (an ExoNitDB instance method) to run: ``self.backend.execute(stmt)``."""
    return SimpleNamespace(backend=_FakeCmplBackend(exc))


class TestIsConcurrentDdlRace:
    def test_matches_duplicate_table(self):
        psycopg = pytest.importorskip("psycopg")
        assert _is_concurrent_ddl_race(psycopg.errors.DuplicateTable("x")) is True

    def test_matches_duplicate_object(self):
        psycopg = pytest.importorskip("psycopg")
        assert _is_concurrent_ddl_race(psycopg.errors.DuplicateObject("x")) is True

    def test_matches_unique_violation(self):
        psycopg = pytest.importorskip("psycopg")
        assert _is_concurrent_ddl_race(psycopg.errors.UniqueViolation("x")) is True

    def test_does_not_match_unrelated_error(self):
        assert _is_concurrent_ddl_race(RuntimeError("nope")) is False

    def test_does_not_match_plain_exception_without_sqlstate(self):
        psycopg = pytest.importorskip("psycopg")
        # A generic psycopg DatabaseError with no sqlstate set must not match.
        assert _is_concurrent_ddl_race(psycopg.DatabaseError("x")) is False


class TestEnsureCompletenessPgSchemaRaceGuard:
    def test_duplicate_table_race_is_ignored(self):
        psycopg = pytest.importorskip("psycopg")
        fake = _fake_db(psycopg.errors.DuplicateTable("relation already exists"))
        ExoNitDB._ensure_completeness_pg_schema(fake)  # must not raise
        # All 5 DDL statements are still attempted despite the first raising.
        assert len(fake.backend.calls) == 5

    def test_duplicate_object_race_is_ignored(self):
        psycopg = pytest.importorskip("psycopg")
        fake = _fake_db(psycopg.errors.DuplicateObject("index already exists"))
        ExoNitDB._ensure_completeness_pg_schema(fake)  # must not raise

    def test_unique_violation_race_is_ignored(self):
        psycopg = pytest.importorskip("psycopg")
        fake = _fake_db(psycopg.errors.UniqueViolation("duplicate key"))
        ExoNitDB._ensure_completeness_pg_schema(fake)  # must not raise

    def test_unrelated_error_still_propagates(self):
        fake = _fake_db(RuntimeError("disk full or something unrelated"))
        with pytest.raises(RuntimeError, match="disk full"):
            ExoNitDB._ensure_completeness_pg_schema(fake)


# ------------------------------------------------------------------ #
# get_task_stats_cached — cross-process cache for get_task_stats()
# (2026-09-25, fix A: /api/status, /api/worker/list and get_stats() were
# each recomputing the same GROUP-BY-status seq scan on their own TTL).
# ------------------------------------------------------------------ #

class TestGetTaskStatsCached:

    def test_cold_cache_computes_inline_and_seeds_cache(self, db):
        """No cache entry yet: compute inline (nothing to serve) and write
        one for the next call."""
        from exonit.shared_state import get_shared_state
        assert db.get_task_stats_cached() == {}
        assert get_shared_state().get(db._TASK_STATS_CACHE_KEY) is not None

    def test_cache_hit_serves_stale_value_within_ttl(self, db):
        db.add_task("t1", "111")
        assert db.get_task_stats_cached() == {"pending": 1}

        db.add_task("t2", "222")  # a second pending task lands after caching
        assert db.get_task_stats_cached() == {"pending": 1}, (
            "within TTL the cached (now stale) value must be served, not "
            "a fresh query"
        )

    def test_cache_miss_recomputes_after_ttl_expiry(self, db):
        db.add_task("t1", "111")
        assert db.get_task_stats_cached(ttl_s=0.0) == {"pending": 1}
        db.add_task("t2", "222")
        # ttl_s=0 -> immediately stale on the very next call; no contender
        # holds the refresh lock, so this call wins it and recomputes.
        assert db.get_task_stats_cached(ttl_s=0.0) == {"pending": 2}

    def test_env_var_overrides_default_ttl(self, db, monkeypatch):
        monkeypatch.setenv("EXONIT_TASK_STATS_CACHE_TTL_S", "0")
        db.add_task("t1", "111")
        assert db.get_task_stats_cached() == {"pending": 1}
        db.add_task("t2", "222")
        assert db.get_task_stats_cached() == {"pending": 2}

    def test_single_flight_serves_stale_when_lock_already_held(self, db, monkeypatch):
        """A sibling process already refreshing (lock held) must be served
        the stale value, not trigger a second concurrent recompute."""
        db.add_task("t1", "111")
        assert db.get_task_stats_cached(ttl_s=0.0) == {"pending": 1}
        db.add_task("t2", "222")

        from exonit.shared_state import get_shared_state
        st = get_shared_state()
        assert st.set_if_absent(db._TASK_STATS_LOCK_KEY, "1", ttl=120.0)

        calls = {"n": 0}
        original = db.get_task_stats

        def counting():
            calls["n"] += 1
            return original()

        monkeypatch.setattr(db, "get_task_stats", counting)

        assert db.get_task_stats_cached(ttl_s=0.0) == {"pending": 1}, (
            "lock already held elsewhere -> serve the stale cached value"
        )
        assert calls["n"] == 0

    def test_falls_back_to_direct_query_when_shared_state_get_broken(
        self, db, monkeypatch
    ):
        from exonit.shared_state import get_shared_state
        st = get_shared_state()
        monkeypatch.setattr(
            st, "get", lambda k: (_ for _ in ()).throw(RuntimeError("boom")))
        db.add_task("t1", "111")
        assert db.get_task_stats_cached() == {"pending": 1}

    def test_falls_back_to_direct_query_when_shared_state_set_broken(
        self, db, monkeypatch
    ):
        """A write failure must not stop the caller from getting a correct
        (uncached) answer — best-effort caching, not best-effort correctness."""
        from exonit.shared_state import get_shared_state
        st = get_shared_state()
        monkeypatch.setattr(
            st, "set", lambda k, v, **kw: (_ for _ in ()).throw(RuntimeError("boom")))
        db.add_task("t1", "111")
        assert db.get_task_stats_cached() == {"pending": 1}

    def test_get_stats_routes_through_the_shared_cache(self, db):
        """get_stats() (used by /api/stats AND /api/downloads/status) must
        share the same cached task-count aggregate, not run its own query."""
        db.add_star("111", ra=0, dec=0, tmag=10.0, n_sectors=1,
                    n_datapoints=1000, time_span_days=30.0)
        db.add_task("t1", "111")
        assert db.get_stats()["pending_tasks"] == 1

        db.add_task("t2", "222")
        assert db.get_stats()["pending_tasks"] == 1, (
            "within TTL get_stats() must see the same cached count as "
            "get_task_stats_cached(), not re-query"
        )

    def test_get_task_stats_itself_is_never_cached(self, db):
        """The raw get_task_stats() stays live — callers that need
        freshness (should one ever be added) must keep using it directly."""
        db.add_task("t1", "111")
        assert db.get_task_stats() == {"pending": 1}
        db.add_task("t2", "222")
        assert db.get_task_stats() == {"pending": 2}


def test_add_candidate_is_per_sector(tmp_path):
    db = ExoNitDB(str(tmp_path / "t.db"))
    i6 = db.add_candidate("9", 1, 3.0, 0.0, 0.1, 100.0, 10.0, sector=6)
    i7 = db.add_candidate("9", 1, 6.01, 0.0, 0.1, 100.0, 10.0, sector=7)
    i6b = db.add_candidate("9", 1, 3.5, 0.0, 0.1, 100.0, 10.0, sector=6)
    assert i6 == i6b and i6 != i7
    got = {r["sector"]: r["period_days"] for r in db.get_candidates("9")}
    assert got == {6: 3.5, 7: 6.01}
