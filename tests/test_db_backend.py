"""Tests for the DB backend abstraction (exonit/db_backend.py).

Two layers:
  * pure unit tests of the SQL translation (no database needed)
  * parity tests parametrised over [sqlite, pg] that run the *same* operations
    on both backends and assert identical observable behaviour.

The PG parity tests use an isolated schema ``exonit_test`` inside the local
``eventseli`` database (credentials from ~/.pgpass). They skip cleanly when PG
or psycopg is unavailable, so the suite still runs on a SQLite-only box.
"""

import os
from pathlib import Path

import pytest

from exonit.db_backend import (
    Dialect,
    PgDialect,
    is_pg_url,
    make_backend,
    qmark_to_pyformat,
    select_in_chunks,
    select_in_chunks_iter,
)


# --------------------------------------------------------------------------- #
# Unit tests: SQL translation (no DB)
# --------------------------------------------------------------------------- #

def test_is_pg_url():
    assert is_pg_url("postgresql://u:p@h/db")
    assert is_pg_url("postgres://u:p@h/db")
    assert not is_pg_url("data/exonit.db")
    assert not is_pg_url("")
    assert not is_pg_url(None)


def test_qmark_basic():
    assert qmark_to_pyformat("SELECT * FROM t WHERE a=? AND b=?") == \
        "SELECT * FROM t WHERE a=%s AND b=%s"


def test_qmark_respects_string_literal():
    # A ? inside a string literal must NOT become %s.
    sql = "SELECT * FROM t WHERE name='a?b' AND x=?"
    assert qmark_to_pyformat(sql) == "SELECT * FROM t WHERE name='a?b' AND x=%s"


def test_qmark_escapes_percent_in_literal():
    # LIKE '%foo%' must be escaped to %%foo%% for psycopg's format parser.
    sql = "SELECT * FROM t WHERE name LIKE '%foo%' AND x=?"
    assert qmark_to_pyformat(sql) == \
        "SELECT * FROM t WHERE name LIKE '%%foo%%' AND x=%s"


def test_qmark_doubled_quote_escape():
    sql = "SELECT * FROM t WHERE s='it''s ok?' AND x=?"
    assert qmark_to_pyformat(sql) == \
        "SELECT * FROM t WHERE s='it''s ok?' AND x=%s"


def test_pg_translate_datetime_now():
    d = PgDialect()
    assert d.translate("INSERT INTO t(at) VALUES(datetime('now'))") == \
        "INSERT INTO t(at) VALUES(NOW())"


def test_pg_translate_indexed_by_removed():
    d = PgDialect()
    out = d.translate("SELECT * FROM candidates INDEXED BY idx_cand_fpp_todo WHERE x=?")
    assert "INDEXED BY" not in out
    assert "%s" in out


def test_sqlite_dialect_is_identity():
    d = Dialect()
    sql = "SELECT * FROM t INDEXED BY i WHERE a=? AND at<datetime('now')"
    assert d.translate(sql) == sql  # unchanged on SQLite


def test_pg_translate_boolean_literal():
    p = PgDialect()
    # BOOLEAN columns compared with int literals -> TRUE/FALSE on PG.
    assert p.translate("WHERE s.has_spoc = 1") == "WHERE s.has_spoc = TRUE"
    assert p.translate("WHERE has_spoc = 0") == "WHERE has_spoc = FALSE"
    assert p.translate("WHERE ctoi_submitted = 1 AND gpu_used = 0") == \
        "WHERE ctoi_submitted = TRUE AND gpu_used = FALSE"
    # IS NULL and non-boolean columns are left untouched.
    assert p.translate("WHERE s.has_spoc IS NULL") == "WHERE s.has_spoc IS NULL"
    assert p.translate("WHERE depth_ppm = 1000") == "WHERE depth_ppm = 1000"


def test_sqlite_boolean_literal_untouched():
    # SQLite stores booleans as 0/1: the comparison must stay an int comparison.
    s = Dialect()
    assert s.translate("WHERE s.has_spoc = 1") == "WHERE s.has_spoc = 1"


def test_dialect_helpers_differ():
    s, p = Dialect(), PgDialect()
    assert s.least("a", "b") == "min(a, b)"
    assert p.least("a", "b") == "LEAST(a, b)"
    assert s.now() == "datetime('now')"
    assert p.now() == "NOW()"
    assert s.telemetry_ts() == "timestamp"
    assert p.telemetry_ts() == "ts"
    assert s.bool_param(True) == 1
    assert p.bool_param(True) is True


# --------------------------------------------------------------------------- #
# B5: nulls_last() — same fragment on both dialects (identical syntax
# supported on PG and SQLite>=3.30; see progress/impl-dblayer-2026-07-02.md)
# --------------------------------------------------------------------------- #

def test_nulls_last_fragment():
    s, p = Dialect(), PgDialect()
    assert s.nulls_last("cnn_score") == "cnn_score DESC NULLS LAST"
    assert p.nulls_last("cnn_score") == "cnn_score DESC NULLS LAST"
    assert s.nulls_last("cnn_score", "ASC") == "cnn_score ASC NULLS LAST"


def test_nulls_last_puts_nulls_after_values_on_both_backends(backend):
    """The actual dialect-drift bug: without NULLS LAST, PG puts NULLs FIRST
    on DESC (SQLite puts them last). nulls_last() must force LAST on both."""
    kind, b = backend
    b.execute("INSERT INTO probe(tic_id, score) VALUES(?,?)", ("has_score", 5.0))
    b.execute("INSERT INTO probe(tic_id, score) VALUES(?,?)", ("null_score", None))
    order = b.dialect.nulls_last("score")
    rows = b.query(f"SELECT tic_id FROM probe ORDER BY {order}")
    assert [r["tic_id"] for r in rows] == ["has_score", "null_score"]


# --------------------------------------------------------------------------- #
# D13: Dialect.upsert() raw_values — DB-clock timestamp columns
# --------------------------------------------------------------------------- #

def test_sqlite_upsert_raw_values_in_place_of_placeholder():
    s = Dialect()
    sql = s.upsert("t", ("id", "state", "updated_at"), conflict_cols=("id",),
                   raw_values={"updated_at": "datetime('now')"})
    assert sql == (
        "INSERT OR REPLACE INTO t (id, state, updated_at) "
        "VALUES (?, ?, datetime('now'))")


def test_pg_upsert_raw_values_in_values_and_set_clause():
    p = PgDialect()
    sql = p.upsert("t", ("id", "state", "updated_at"), conflict_cols=("id",),
                   raw_values={"updated_at": "datetime('now')"})
    assert "VALUES (?, ?, datetime('now'))" in sql
    assert "ON CONFLICT (id) DO UPDATE SET" in sql
    assert "state = EXCLUDED.state" in sql
    assert "updated_at = datetime('now')" in sql
    assert "updated_at = EXCLUDED.updated_at" not in sql


def test_pg_upsert_raw_values_roundtrip(backend):
    """The raw_values VALUES entry is exempt from a bound placeholder — the
    params tuple must OMIT it, matching how discovery_state.py calls this."""
    kind, b = backend
    # ON CONFLICT needs a real unique constraint on PG (the base probe DDL
    # only has one on id); add one on tic_id for this upsert test.
    b.execute("CREATE UNIQUE INDEX IF NOT EXISTS probe_tic_uniq ON probe(tic_id)")
    sql = b.dialect.upsert(
        "probe", ("tic_id", "score", "at"), conflict_cols=("tic_id",),
        raw_values={"at": b.dialect.now()})
    b.execute(sql, ("raw_ts", 1.0))
    row = b.query_one("SELECT tic_id, at FROM probe WHERE tic_id=?", ("raw_ts",))
    assert row["tic_id"] == "raw_ts"
    assert row["at"] is not None  # DB clock populated it, not a bound param


# --------------------------------------------------------------------------- #
# D13: Dialect.upsert_on_conflict() — genuine PARTIAL update on conflict,
# identical on SQLite and PG (unlike upsert(), which is REPLACE on SQLite).
# --------------------------------------------------------------------------- #

def test_upsert_on_conflict_same_sql_both_dialects():
    """Same SQL text on SQLite and PG — no per-dialect override needed."""
    s, p = Dialect(), PgDialect()
    kwargs = dict(table="t", columns=("id", "state", "detail"),
                  conflict_cols=("id",))
    assert s.upsert_on_conflict(**kwargs) == p.upsert_on_conflict(**kwargs)
    assert s.upsert_on_conflict(**kwargs) == (
        "INSERT INTO t (id, state, detail) VALUES (?, ?, ?) "
        "ON CONFLICT (id) DO UPDATE SET state = EXCLUDED.state, "
        "detail = EXCLUDED.detail")


def test_upsert_on_conflict_partial_update_preserves_untouched_columns(backend):
    """The D13 bug this seam exists to prevent: on SQLite, Dialect.upsert()
    (INSERT OR REPLACE) would reset every column not in `columns` to its
    default on conflict. upsert_on_conflict() must NOT do that — the same
    contract set_autofill's tests exercise (test_discovery_autofill_manager.
    py::test_external_stop_during_poll_is_honored)."""
    kind, b = backend
    b.execute("CREATE UNIQUE INDEX IF NOT EXISTS probe_tic_uniq2 ON probe(tic_id)")

    # First insert: both score and flag set.
    sql_full = b.dialect.upsert_on_conflict(
        "probe", ("tic_id", "score", "flag"), conflict_cols=("tic_id",))
    b.execute(sql_full, ("partial", 1.0, b.dialect.bool_param(True)))

    # Second call: only score is written — flag must be preserved, not reset.
    sql_partial = b.dialect.upsert_on_conflict(
        "probe", ("tic_id", "score"), conflict_cols=("tic_id",))
    b.execute(sql_partial, ("partial", 2.0))

    row = b.query_one("SELECT score, flag FROM probe WHERE tic_id=?", ("partial",))
    assert abs(row["score"] - 2.0) < 1e-9
    assert bool(row["flag"]) is True  # preserved, NOT clobbered to NULL/0


def test_upsert_on_conflict_no_update_cols_does_nothing():
    s = Dialect()
    sql = s.upsert_on_conflict("t", ("id", "state"), conflict_cols=("id", "state"))
    assert "DO NOTHING" in sql
    assert "DO UPDATE" not in sql


# --------------------------------------------------------------------------- #
# D2: select_in_chunks() — chunked IN(...) queries (PG's 65535-param limit)
# --------------------------------------------------------------------------- #

def test_select_in_chunks_splits_into_multiple_calls():
    calls = []

    def fake_runner(sql, params):
        calls.append((sql, params))
        return [{"id": p} for p in params]

    ids = list(range(1, 11))  # 10 ids, chunk_size=3 -> 4 chunks (3,3,3,1)
    rows = select_in_chunks(
        fake_runner, "SELECT id FROM t WHERE id IN", ids, chunk_size=3)

    assert len(calls) == 4
    assert [len(p) for _, p in calls] == [3, 3, 3, 1]
    assert sorted(r["id"] for r in rows) == ids
    # Every chunk's SQL has the right number of placeholders and the shared
    # sql_before/sql_after text.
    for sql, params in calls:
        assert sql == f"SELECT id FROM t WHERE id IN ({','.join('?' * len(params))})"


def test_select_in_chunks_single_chunk_matches_unchunked():
    def fake_runner(sql, params):
        return [{"id": p} for p in params]

    ids = [1, 2, 3]
    rows = select_in_chunks(fake_runner, "SELECT id FROM t WHERE id IN", ids)
    assert sorted(r["id"] for r in rows) == ids


def test_select_in_chunks_empty_ids_no_calls():
    calls = []

    def fake_runner(sql, params):
        calls.append((sql, params))
        return []

    assert select_in_chunks(fake_runner, "SELECT id FROM t WHERE id IN", []) == []
    assert calls == []


def test_select_in_chunks_sql_after_and_params_extra():
    calls = []

    def fake_runner(sql, params):
        calls.append((sql, params))
        return []

    select_in_chunks(
        fake_runner, "SELECT id FROM t WHERE id IN", [1, 2],
        sql_after=" AND status = ?", params_extra=("active",))
    sql, params = calls[0]
    assert sql == "SELECT id FROM t WHERE id IN (?,?) AND status = ?"
    assert params == (1, 2, "active")


def test_select_in_chunks_dict_coerces_backend_rows(backend):
    """End-to-end against a real backend (both dialects): rows returned by
    ``backend.query`` (already dict) survive the dict() coercion unchanged,
    and chunking below the real row count still concatenates correctly."""
    kind, b = backend
    ids = []
    for i in range(5):
        rid = b.execute_returning_id(
            "INSERT INTO probe(tic_id, score) VALUES(?,?)", (f"t{i}", float(i)))
        ids.append(rid)

    rows = select_in_chunks(
        b.query, "SELECT id, tic_id FROM probe WHERE id IN", ids, chunk_size=2)
    assert sorted(r["tic_id"] for r in rows) == [f"t{i}" for i in range(5)]


# --------------------------------------------------------------------------- #
# F1 (2026-08-01 postprocess-streaming spec, §3.A):
# select_in_chunks_iter() — per-chunk generator variant of select_in_chunks()
# --------------------------------------------------------------------------- #

def test_select_in_chunks_iter_parity_with_select_in_chunks():
    """Flattening the iter variant's yields must equal select_in_chunks()'s
    concatenated list — same rows, same order — proving the reimplementation
    of select_in_chunks() on top of it is behaviour-preserving."""
    def fake_runner(sql, params):
        return [{"id": p} for p in params]

    ids = list(range(1, 11))  # 10 ids, chunk_size=3 -> 4 chunks (3,3,3,1)
    flattened = [row for chunk in select_in_chunks_iter(
        fake_runner, "SELECT id FROM t WHERE id IN", ids, chunk_size=3)
        for row in chunk]
    expected = select_in_chunks(
        fake_runner, "SELECT id FROM t WHERE id IN", ids, chunk_size=3)
    assert flattened == expected


def test_select_in_chunks_iter_batch_sizes():
    def fake_runner(sql, params):
        return [{"id": p} for p in params]

    ids = list(range(1, 11))  # 10 ids
    chunks = list(select_in_chunks_iter(
        fake_runner, "SELECT id FROM t WHERE id IN", ids, chunk_size=3))
    assert [len(c) for c in chunks] == [3, 3, 3, 1]


def test_select_in_chunks_iter_chunk_locality():
    """Invariant the spec calls out: chunking is by id, so ALL rows for a
    given id (here: multiple synthetic DB rows per id, like several
    candidates per tic_id) land in exactly one chunk — never split across
    two. That's what makes a chunk-local per-id groupby safe."""
    def fake_runner(sql, params):
        # 3 synthetic rows per id in this chunk, e.g. 3 candidates per tic_id.
        return [{"tic_id": p, "row": k} for p in params for k in range(3)]

    ids = list(range(1, 11))  # 10 unique ids
    chunks = list(select_in_chunks_iter(
        fake_runner, "SELECT * FROM t WHERE tic_id IN", ids, chunk_size=3))
    assert len(chunks) == 4

    seen_ids = set()
    for chunk in chunks:
        chunk_ids = {r["tic_id"] for r in chunk}
        assert not (chunk_ids & seen_ids), "an id reappeared in a later chunk"
        seen_ids |= chunk_ids
        for tid in chunk_ids:
            assert sum(1 for r in chunk if r["tic_id"] == tid) == 3
    assert seen_ids == set(ids)


def test_select_in_chunks_iter_sql_after_and_params_extra():
    calls = []

    def fake_runner(sql, params):
        calls.append((sql, params))
        return []

    list(select_in_chunks_iter(
        fake_runner, "SELECT id FROM t WHERE id IN", [1, 2],
        sql_after=" AND status = ?", params_extra=("active",)))
    sql, params = calls[0]
    assert sql == "SELECT id FROM t WHERE id IN (?,?) AND status = ?"
    assert params == (1, 2, "active")


def test_select_in_chunks_iter_empty_ids_no_calls():
    calls = []

    def fake_runner(sql, params):
        calls.append((sql, params))
        return []

    assert list(select_in_chunks_iter(
        fake_runner, "SELECT id FROM t WHERE id IN", [])) == []
    assert calls == []


def test_select_in_chunks_iter_dict_coerces_backend_rows(backend):
    """End-to-end against a real backend (both dialects), mirroring
    test_select_in_chunks_dict_coerces_backend_rows for the iter variant."""
    kind, b = backend
    ids = []
    for i in range(5):
        rid = b.execute_returning_id(
            "INSERT INTO probe(tic_id, score) VALUES(?,?)", (f"t{i}", float(i)))
        ids.append(rid)

    chunks = list(select_in_chunks_iter(
        b.query, "SELECT id, tic_id FROM probe WHERE id IN", ids, chunk_size=2))
    assert [len(c) for c in chunks] == [2, 2, 1]
    flattened = [row for c in chunks for row in c]
    assert sorted(r["tic_id"] for r in flattened) == [f"t{i}" for i in range(5)]


# --------------------------------------------------------------------------- #
# Parity fixtures
# --------------------------------------------------------------------------- #

def _pg_dsn():
    """Build a PG DSN pointing at the isolated exonit_test schema, or None."""
    try:
        import psycopg  # noqa: F401
    except Exception:
        return None
    pgpass = Path.home() / ".pgpass"
    if not pgpass.exists():
        return None
    parts = pgpass.read_text().strip().splitlines()[0].split(":")
    if len(parts) < 5:
        return None
    host, port, db, user = parts[:4]
    # Password is read from ~/.pgpass by libpq/psycopg — never inline it in the
    # DSN (it would leak into tracebacks/logs).
    # search_path isolates all test DDL/DML into the exonit_test schema.
    return (f"postgresql://{user}@{host}:{port}/{db}"
            f"?options=-c%20search_path%3Dexonit_test")


_PROBE_DDL = {
    "sqlite": (
        "CREATE TABLE probe ("
        " id INTEGER PRIMARY KEY AUTOINCREMENT,"
        " tic_id TEXT, score REAL, flag INTEGER, at TEXT)"
    ),
    "pg": (
        "CREATE TABLE probe ("
        " id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,"
        " tic_id TEXT, score DOUBLE PRECISION, flag BOOLEAN, at TIMESTAMPTZ)"
    ),
}


@pytest.fixture(params=["sqlite", "pg"])
def backend(request, tmp_path):
    kind = request.param
    if kind == "sqlite":
        b = make_backend(str(tmp_path / "probe.db"))
        b.execute(_PROBE_DDL["sqlite"])
        yield "sqlite", b
        b.close()
    else:
        dsn = _pg_dsn()
        if not dsn:
            pytest.skip("PostgreSQL/psycopg/.pgpass not available")
        b = make_backend(dsn)
        # Fresh isolated schema per test run.
        b.execute("CREATE SCHEMA IF NOT EXISTS exonit_test")
        b.execute("DROP TABLE IF EXISTS probe")
        b.execute(_PROBE_DDL["pg"])
        yield "pg", b
        try:
            b.execute("DROP TABLE IF EXISTS probe")
        finally:
            b.close()


# --------------------------------------------------------------------------- #
# Parity tests
# --------------------------------------------------------------------------- #

def test_insert_query_roundtrip(backend):
    kind, b = backend
    b.execute("INSERT INTO probe(tic_id, score, at) VALUES(?,?,datetime('now'))",
              ("12345", 0.91))
    rows = b.query("SELECT tic_id, score FROM probe WHERE tic_id=?", ("12345",))
    assert len(rows) == 1
    assert rows[0]["tic_id"] == "12345"
    assert abs(rows[0]["score"] - 0.91) < 1e-9


def test_query_one_and_scalar(backend):
    kind, b = backend
    b.execute("INSERT INTO probe(tic_id, score) VALUES(?,?)", ("a", 1.0))
    b.execute("INSERT INTO probe(tic_id, score) VALUES(?,?)", ("b", 2.0))
    one = b.query_one("SELECT tic_id FROM probe WHERE tic_id=?", ("b",))
    assert one == {"tic_id": "b"}
    assert b.query_one("SELECT tic_id FROM probe WHERE tic_id=?", ("zzz",)) is None
    n = b.query_scalar("SELECT COUNT(*) FROM probe")
    assert n == 2


def test_returning_id(backend):
    kind, b = backend
    rid = b.execute_returning_id(
        "INSERT INTO probe(tic_id, score) VALUES(?,?)", ("ret", 3.0))
    assert isinstance(rid, int) and rid > 0
    got = b.query_scalar("SELECT score FROM probe WHERE id=?", (rid,))
    assert abs(got - 3.0) < 1e-9


def test_executemany(backend):
    kind, b = backend
    b.executemany("INSERT INTO probe(tic_id, score) VALUES(?,?)",
                  [("x", 1.0), ("y", 2.0), ("z", 3.0)])
    assert b.query_scalar("SELECT COUNT(*) FROM probe") == 3


def test_least_scalar_parity(backend):
    kind, b = backend
    d = b.dialect
    b.execute("INSERT INTO probe(tic_id, score) VALUES(?,?)", ("m", 0.7))
    # min(score, 0.9) -> 0.7 on both backends via the dialect helper.
    val = b.query_scalar(f"SELECT {d.least('score', '0.9')} FROM probe WHERE tic_id=?",
                         ("m",))
    assert abs(val - 0.7) < 1e-9


def test_bool_param_parity(backend):
    kind, b = backend
    d = b.dialect
    b.execute("INSERT INTO probe(tic_id, flag) VALUES(?,?)",
              ("f", d.bool_param(True)))
    row = b.query_one("SELECT flag FROM probe WHERE tic_id=?", ("f",))
    # SQLite stores 1, PG stores True; both are truthy.
    assert bool(row["flag"]) is True


# --------------------------------------------------------------------------- #
# F1 (2026-08-01 postprocess-streaming spec, §3.A):
# OperationalSession.execute_stream() — batched, non-materializing SELECT
# --------------------------------------------------------------------------- #

def test_execute_stream_matches_fetchall(backend):
    """Concatenating all streamed batches must equal execute(...).fetchall()
    over the same query/data."""
    kind, b = backend
    for i in range(23):
        b.execute("INSERT INTO probe(tic_id, score) VALUES(?,?)", (f"t{i:02d}", float(i)))
    con = b.op_session()
    try:
        expected = con.execute(
            "SELECT tic_id, score FROM probe ORDER BY tic_id").fetchall()
        got = []
        for batch in con.execute_stream(
                "SELECT tic_id, score FROM probe ORDER BY tic_id", batch_rows=7):
            got.extend(batch)
    finally:
        con.close()
    assert got == expected


def test_execute_stream_batch_sizes_with_partial_last_batch(backend):
    """7 rows, batch_rows=3 -> batches of 3, 3, 1."""
    kind, b = backend
    for i in range(7):
        b.execute("INSERT INTO probe(tic_id, score) VALUES(?,?)", (f"t{i}", float(i)))
    con = b.op_session()
    try:
        batches = list(con.execute_stream(
            "SELECT tic_id FROM probe ORDER BY tic_id", batch_rows=3))
    finally:
        con.close()
    assert [len(batch) for batch in batches] == [3, 3, 1]


def test_execute_stream_empty_result_yields_nothing(backend):
    kind, b = backend
    con = b.op_session()
    try:
        batches = list(con.execute_stream(
            "SELECT tic_id FROM probe WHERE tic_id = ?", ("nope",), batch_rows=10))
    finally:
        con.close()
    assert batches == []


def test_execute_stream_batch_rows_larger_than_result_single_batch(backend):
    kind, b = backend
    for i in range(4):
        b.execute("INSERT INTO probe(tic_id, score) VALUES(?,?)", (f"t{i}", float(i)))
    con = b.op_session()
    try:
        batches = list(con.execute_stream(
            "SELECT tic_id FROM probe ORDER BY tic_id", batch_rows=1000))
    finally:
        con.close()
    assert len(batches) == 1
    assert len(batches[0]) == 4


def test_execute_stream_rows_are_dicts(backend):
    kind, b = backend
    b.execute("INSERT INTO probe(tic_id, score) VALUES(?,?)", ("solo", 5.0))
    con = b.op_session()
    try:
        batches = list(con.execute_stream(
            "SELECT tic_id, score FROM probe", batch_rows=10))
    finally:
        con.close()
    assert len(batches) == 1
    row = batches[0][0]
    assert isinstance(row, dict)
    assert row["tic_id"] == "solo"
    assert abs(row["score"] - 5.0) < 1e-9


def test_execute_stream_respects_params(backend):
    kind, b = backend
    for i in range(5):
        b.execute("INSERT INTO probe(tic_id, score) VALUES(?,?)", (f"t{i}", float(i)))
    con = b.op_session()
    try:
        batches = list(con.execute_stream(
            "SELECT tic_id FROM probe WHERE score >= ? ORDER BY tic_id",
            (2.0,), batch_rows=100))
    finally:
        con.close()
    got = [r["tic_id"] for batch in batches for r in batch]
    assert got == ["t2", "t3", "t4"]


def test_execute_stream_early_abandon_closes_cursor_cleanly(backend):
    """Abandoning the generator mid-scan (GeneratorExit at the suspended
    yield) must close the cursor without leaving the session unusable — the
    named PG cursor in particular must not dangle inside the transaction."""
    kind, b = backend
    for i in range(10):
        b.execute("INSERT INTO probe(tic_id, score) VALUES(?,?)", (f"t{i}", float(i)))
    con = b.op_session()
    try:
        gen = con.execute_stream(
            "SELECT tic_id FROM probe ORDER BY tic_id", batch_rows=2)
        first = next(gen)
        assert len(first) == 2
        gen.close()
        # Session must still be fully usable after abandoning the stream.
        row = con.execute("SELECT COUNT(*) AS c FROM probe").fetchone()
        assert row["c"] == 10
    finally:
        con.close()


# --------------------------------------------------------------------------- #
# Q3 (perf review 2026-09-26): iter_id_column -- id-only queries streamed as
# tuple rows (never dict_row/Row) via a server-side cursor (PG) / fetchmany
# (SQLite). Uses the throwaway `exonit_test` DB (tests/_pg_test_db.py) for
# the PG side, NOT the `backend` fixture above (that one points at the local
# `eventseli` DB via ~/.pgpass' first line, which is unrelated to this
# review and not guaranteed reachable/authenticated on every dev box).
# --------------------------------------------------------------------------- #

from tests._pg_test_db import pg_test_available, resolve_test_dsn  # noqa: E402

_ITER_ID_PG_DSN = resolve_test_dsn()


def test_sqlite_iter_id_column_streams_tuples_in_order(tmp_path):
    b = make_backend(str(tmp_path / "idcol.db"))
    try:
        b.execute("CREATE TABLE probe_id (tic_id TEXT)")
        b.executemany(
            "INSERT INTO probe_id VALUES (?)",
            [(f"t{i}",) for i in range(5)])
        ids = list(b.iter_id_column(
            "SELECT tic_id FROM probe_id ORDER BY tic_id"))
        assert ids == [f"t{i}" for i in range(5)]
        assert all(isinstance(v, str) for v in ids)
    finally:
        b.close()


def test_sqlite_iter_id_column_batch_boundary(tmp_path):
    b = make_backend(str(tmp_path / "idcol2.db"))
    try:
        b.execute("CREATE TABLE probe_id (tic_id TEXT)")
        b.executemany(
            "INSERT INTO probe_id VALUES (?)",
            [(f"t{i}",) for i in range(7)])
        ids = list(b.iter_id_column(
            "SELECT tic_id FROM probe_id ORDER BY tic_id", batch_rows=3))
        assert ids == [f"t{i}" for i in range(7)]
    finally:
        b.close()


def test_sqlite_iter_id_column_empty_result(tmp_path):
    b = make_backend(str(tmp_path / "idcol3.db"))
    try:
        b.execute("CREATE TABLE probe_id (tic_id TEXT)")
        assert list(b.iter_id_column("SELECT tic_id FROM probe_id")) == []
    finally:
        b.close()


@pytest.fixture
def pg_idcol_backend():
    if not pg_test_available(_ITER_ID_PG_DSN):
        pytest.skip("EXONIT_TEST_DATABASE_URL (throwaway PG) unset or not reachable")
    b = make_backend(_ITER_ID_PG_DSN)
    b.execute("CREATE TABLE IF NOT EXISTS q3_probe_id (tic_id TEXT)")
    b.execute("TRUNCATE q3_probe_id")
    yield b
    b.execute("DROP TABLE IF EXISTS q3_probe_id")
    b.close()


def test_pg_iter_id_column_streams_tuples_in_order(pg_idcol_backend):
    b = pg_idcol_backend
    b.executemany(
        "INSERT INTO q3_probe_id VALUES (?)", [(f"t{i}",) for i in range(5)])
    ids = list(b.iter_id_column(
        "SELECT tic_id FROM q3_probe_id ORDER BY tic_id"))
    assert ids == [f"t{i}" for i in range(5)]
    assert all(isinstance(v, str) for v in ids)


def test_pg_iter_id_column_batch_boundary(pg_idcol_backend):
    """batch_rows smaller than the result set must still yield every row via
    the named server-side cursor + tuple_row override (not the pool
    connection's dict_row default)."""
    b = pg_idcol_backend
    b.executemany(
        "INSERT INTO q3_probe_id VALUES (?)", [(f"t{i}",) for i in range(7)])
    ids = list(b.iter_id_column(
        "SELECT tic_id FROM q3_probe_id ORDER BY tic_id", batch_rows=3))
    assert ids == [f"t{i}" for i in range(7)]


def test_pg_iter_id_column_empty_result(pg_idcol_backend):
    assert list(pg_idcol_backend.iter_id_column(
        "SELECT tic_id FROM q3_probe_id")) == []


def test_pg_iter_id_column_matches_query_result(pg_idcol_backend):
    """Same rows as the eager query() surface -- iter_id_column is a
    drop-in replacement for a `{row["col"] for row in query(...)}`
    comprehension, not a different result."""
    b = pg_idcol_backend
    b.executemany(
        "INSERT INTO q3_probe_id VALUES (?)",
        [(f"t{i}",) for i in range(10)])
    via_query = {r["tic_id"] for r in b.query("SELECT tic_id FROM q3_probe_id")}
    via_iter = set(b.iter_id_column("SELECT tic_id FROM q3_probe_id"))
    assert via_iter == via_query
