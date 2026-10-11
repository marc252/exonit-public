"""Tests for exonit.discovery_state.sector_gap_stats's V2 dispatch (S2,
docs/superpowers/specs/2026-09-26-enqueue-s1-s4-design.md §5.2/§5.3).

These never touch a real PostgreSQL server — they exercise the flag/dispatch
logic against a throwaway SQLite DB (which always takes the legacy path,
since ``postgres_fdw`` does not exist on SQLite — "SQLite keeps legacy
path" per the spec). The real fdw-backed parity test lives in
tests/test_gap_stats_v2_pg.py (skips without a provisioned ``meta_fdw.tglc``).
"""

import csv
import warnings

import pytest

from exonit import done_state
from exonit.db import ExoNitDB


@pytest.fixture
def gap_fixture(tmp_path):
    db = ExoNitDB(str(tmp_path / "test.db"))
    db.backend.execute("""
        CREATE TABLE IF NOT EXISTS discovery_no_data (
            tic_id TEXT NOT NULL,
            sector INTEGER NOT NULL,
            reason TEXT NOT NULL DEFAULT 'no_fits',
            checked_at TEXT DEFAULT (datetime('now')),
            PRIMARY KEY (tic_id, sector)
        )""")
    tics = [3001, 3002, 3003]
    canonical_csv = tmp_path / "canonical.csv"
    with open(canonical_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["tic_id"])
        for t in tics:
            w.writerow([t])
    db.add_task("t1", "3001", sector=9)
    db.complete_task("t1", processing_time=1.0)
    # Dual-write, same as routes/workers.py's worker_result path — keeps
    # tic_sector_state in sync so `completed` is 1 regardless of whether
    # EXONIT_DONE_STATE is on (done_state path) or off (tasks-table path).
    done_state.mark_tic_done(db.backend, "3001", "t1")
    return {
        "db_path": str(tmp_path / "test.db"),
        "canonical_csv": str(canonical_csv),
        "tglc_index": str(tmp_path / "no_tglc.db"),
        "toi_catalog": str(tmp_path / "no_toi.csv"),
    }


def _call(gap_fixture):
    from exonit.discovery_state import sector_gap_stats
    return sector_gap_stats(
        9, db_path=gap_fixture["db_path"],
        canonical_csv=gap_fixture["canonical_csv"],
        tglc_index=gap_fixture["tglc_index"],
        toi_catalog=gap_fixture["toi_catalog"])


class TestGapStatsV2FlagOffUnchanged:
    def test_flag_off_byte_identical_to_pre_s2(self, gap_fixture, monkeypatch):
        """With EXONIT_GAP_STATS_V2 unset (default), the result must be
        exactly what the pre-S2 legacy path produces — no V2 code runs at
        all (verified by NOT needing any PG/fdw infra for this test)."""
        monkeypatch.delenv("EXONIT_GAP_STATS_V2", raising=False)
        monkeypatch.delenv("EXONIT_DONE_STATE", raising=False)
        gs = _call(gap_fixture)
        assert gs["work_list_size"] == 3
        assert gs["completed"] == 1
        assert "failed_terminal" not in gs


class TestGapStatsV2RequiresDoneState:
    def test_v2_without_done_state_warns_and_uses_legacy(
        self, gap_fixture, monkeypatch,
    ):
        monkeypatch.setenv("EXONIT_GAP_STATS_V2", "1")
        monkeypatch.delenv("EXONIT_DONE_STATE", raising=False)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            gs = _call(gap_fixture)
        assert any(
            issubclass(x.category, RuntimeWarning)
            and "EXONIT_DONE_STATE" in str(x.message)
            for x in w
        ), "misconfigured V2 (no done-state) must warn"
        # Legacy result, unaffected by the V2 flag being on:
        assert gs["work_list_size"] == 3
        assert gs["completed"] == 1
        assert "failed_terminal" not in gs


class TestGapStatsV2SqliteFallsBackSilently:
    def test_v2_on_sqlite_backend_falls_back_without_crashing(
        self, gap_fixture, monkeypatch,
    ):
        """SQLite has no postgres_fdw — V2 + done-state both on, but the
        operational DB is SQLite, must fall back to legacy without raising
        (spec §5.2: "SQLite keeps legacy path")."""
        monkeypatch.setenv("EXONIT_GAP_STATS_V2", "1")
        monkeypatch.setenv("EXONIT_DONE_STATE", "1")
        gs = _call(gap_fixture)
        assert gs["work_list_size"] == 3
        assert gs["completed"] == 1


# ---------------------------------------------------------------------------
# Review fix (2026-09-26, progress/review-enqueue-s2.md item 1): gap_stats_v2's
# probe for "which of these candidate TICs are done(t)" must NEVER fall back
# to a whole-tic_sector_state scan, no matter how many candidate ids are
# passed in (exonit.done_state._done_ids_scoped does exactly that above
# 100k ids — this module keeps its own, always-chunked, local probe instead).
# ---------------------------------------------------------------------------

class _SpyBackend:
    """Minimal Backend stand-in: records every ``.query()`` call and RAISES
    if ``.iter_id_column()`` is ever called — that method is the vehicle for
    ``done_state._done_ids_scoped``'s whole-table-scan fallback
    (``_done_ids(backend)``), so any call to it here is exactly the bug this
    test guards against."""

    def __init__(self, dialect_name="postgres"):
        self.query_calls = []
        self.dialect = type("D", (), {"name": dialect_name})()

    def query(self, sql, params=()):
        self.query_calls.append((sql, tuple(params)))
        return []  # no rows "done" — irrelevant to what this test checks

    def iter_id_column(self, sql, params=(), **kwargs):
        raise AssertionError(
            "gap_stats_v2's chunked probe must NEVER call iter_id_column "
            "(that is done_state._done_ids_scoped's whole-table-scan "
            "fallback path) -- got sql=%r" % (sql,))


class TestProbeDoneChunkedNeverFullScans:
    def test_over_100k_ids_stays_chunked_no_full_scan(self):
        """150k candidate ids -- well past done_state's 100k chunk/full-scan
        crossover -- must still produce ONLY chunked `tic_id IN (...)`
        queries via `.query()`, and must never touch `.iter_id_column()`."""
        from exonit.gap_stats_v2 import _probe_done_chunked

        backend = _SpyBackend()
        ids = list(range(1, 150_001))  # 150k ids > done_state._CHUNK_PROBE_MAX_IDS (100k)

        result = _probe_done_chunked(backend, ids, chunk_size=5000)

        assert result == set()
        assert len(backend.query_calls) == 30, (
            f"expected 150000/5000 = 30 chunked queries, got "
            f"{len(backend.query_calls)}")
        for sql, params in backend.query_calls:
            assert "tic_id IN" in sql
            assert "status = 'completed'" in sql
            assert len(params) <= 5000

    def test_default_chunk_size_is_5000_or_less(self):
        from exonit.gap_stats_v2 import _PROBE_DONE_CHUNK_SIZE
        assert _PROBE_DONE_CHUNK_SIZE <= 5000

    def test_empty_ids_issues_no_query(self):
        from exonit.gap_stats_v2 import _probe_done_chunked
        backend = _SpyBackend()
        assert _probe_done_chunked(backend, []) == set()
        assert backend.query_calls == []
