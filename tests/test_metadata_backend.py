"""Tests for exonit/metadata_backend.py — the TGLC-index SQLite/PG seam.

Covers: URL detection, the qmark->pyformat SQL translator, the Dialect/
PgDialect upsert() builders, and SqliteBackend's execute/query/transaction
contract. PgBackend construction itself is exercised (opt-in, against a
throwaway DB only) in tests/test_metadata_backend_pg_parity.py.
"""
import pytest

from exonit.metadata_backend import (
    Dialect,
    PgDialect,
    SqliteBackend,
    is_pg_url,
    make_metadata_backend,
    qmark_to_pyformat,
)


# --------------------------------------------------------------------------- #
# is_pg_url / make_metadata_backend dispatch
# --------------------------------------------------------------------------- #

def test_is_pg_url_true_for_postgres_schemes():
    assert is_pg_url("postgresql://u@h/db")
    assert is_pg_url("postgres://u@h/db")


def test_is_pg_url_false_for_path_or_none():
    assert not is_pg_url("data/metadata_cache.db")
    assert not is_pg_url(None)
    assert not is_pg_url("")


def test_make_metadata_backend_path_returns_sqlite(tmp_path):
    backend = make_metadata_backend(str(tmp_path / "meta.db"))
    try:
        assert isinstance(backend, SqliteBackend)
        assert backend.dialect.name == "sqlite"
    finally:
        backend.close()


# --------------------------------------------------------------------------- #
# qmark_to_pyformat
# --------------------------------------------------------------------------- #

def test_qmark_to_pyformat_basic():
    assert qmark_to_pyformat("SELECT * FROM t WHERE a = ? AND b = ?") == \
        "SELECT * FROM t WHERE a = %s AND b = %s"


def test_qmark_to_pyformat_preserves_question_mark_in_string_literal():
    sql = "SELECT * FROM t WHERE note = 'is this ok?' AND a = ?"
    out = qmark_to_pyformat(sql)
    assert "'is this ok?'" in out
    assert out.endswith("a = %s")


def test_qmark_to_pyformat_escapes_percent_outside_and_inside_literals():
    sql = "SELECT * FROM t WHERE filename LIKE '%-s0023-%' AND path = ?"
    out = qmark_to_pyformat(sql)
    assert "'%%-s0023-%%'" in out
    assert out.endswith("path = %s")


def test_qmark_to_pyformat_handles_escaped_single_quote():
    sql = "SELECT * FROM t WHERE name = 'O''Brien' AND a = ?"
    out = qmark_to_pyformat(sql)
    assert "'O''Brien'" in out
    assert out.endswith("a = %s")


# --------------------------------------------------------------------------- #
# Dialect.translate()
# --------------------------------------------------------------------------- #

def test_sqlite_dialect_translate_is_identity():
    d = Dialect()
    sql = "SELECT * FROM tglc WHERE tic_id = ? AND path LIKE '%foo%'"
    assert d.translate(sql) == sql


def test_pg_dialect_translate_converts_placeholders_and_datetime():
    d = PgDialect()
    sql = "INSERT INTO metadata (tic_id, updated_at) VALUES (?, datetime('now'))"
    out = d.translate(sql)
    assert "%s" in out
    assert "NOW()" in out
    assert "datetime(" not in out
    assert "?" not in out


def test_pg_dialect_translate_datetime_now_case_insensitive_and_spacing():
    d = PgDialect()
    assert "NOW()" in d.translate("SELECT DATETIME( 'now' )")


# --------------------------------------------------------------------------- #
# Dialect.upsert() — SQLite: INSERT OR REPLACE
# --------------------------------------------------------------------------- #

def test_sqlite_upsert_is_insert_or_replace():
    d = Dialect()
    sql = d.upsert("fits_paths", ["filename", "full_path"], ["filename"])
    assert sql == "INSERT OR REPLACE INTO fits_paths (filename, full_path) VALUES (?, ?)"


def test_sqlite_upsert_raw_values_embeds_expression_not_placeholder():
    d = Dialect()
    sql = d.upsert("metadata", ["tic_id", "updated_at"], ["tic_id"],
                    raw_values={"updated_at": d.now()})
    assert sql == ("INSERT OR REPLACE INTO metadata (tic_id, updated_at) "
                    "VALUES (?, datetime('now'))")


# --------------------------------------------------------------------------- #
# PgDialect.upsert() — ON CONFLICT DO UPDATE (REPLACE semantics, D4 of spec)
# --------------------------------------------------------------------------- #

def test_pg_upsert_updates_every_non_conflict_column():
    d = PgDialect()
    sql = d.upsert("tglc", ["tic_id", "sector", "path", "exptime"],
                    ["tic_id", "sector"])
    assert sql == (
        "INSERT INTO tglc (tic_id, sector, path, exptime) VALUES (?, ?, ?, ?) "
        "ON CONFLICT (tic_id, sector) DO UPDATE SET "
        "path = EXCLUDED.path, exptime = EXCLUDED.exptime"
    )


def test_pg_upsert_all_columns_are_conflict_cols_uses_do_nothing():
    d = PgDialect()
    sql = d.upsert("fits_paths", ["filename"], ["filename"])
    assert sql.endswith("ON CONFLICT (filename) DO NOTHING")


def test_pg_upsert_raw_values_still_uses_placeholder_for_translate_step():
    """raw_values columns must NOT get a '?' in VALUES, but translate() (not
    upsert()) is what turns the remaining '?'s into '%s' — upsert() itself
    always emits '?' placeholders (see its docstring)."""
    d = PgDialect()
    sql = d.upsert("metadata", ["tic_id", "updated_at"], ["tic_id"],
                    raw_values={"updated_at": d.now()})
    assert "?" in sql          # tic_id's placeholder, not yet translated
    assert "NOW()" in sql      # updated_at is the raw expression, not '?'
    assert sql.count("?") == 1


# --------------------------------------------------------------------------- #
# No double-translation: upsert() output must survive translate() cleanly
# --------------------------------------------------------------------------- #

def test_pg_upsert_then_translate_produces_single_percent_s_placeholders():
    """Regression guard for the double-translation bug documented in
    exonit/db.py (add_run_tics_bulk's comment): calling dialect.translate()
    on a string that already contains '%s' would corrupt it to '%%s'. The
    codebase's convention is: upsert() emits '?', and callers pass that
    STRAIGHT to backend.execute()/executemany() (which translates internally,
    exactly once) — never pre-translate. This test locks that contract by
    checking translate() is idempotent-safe when applied to upsert()'s
    '?'-based output exactly once."""
    d = PgDialect()
    sql = d.upsert("fits_blobs",
                    ["filename", "container_key", "data_offset", "data_length"],
                    ["filename"])
    translated = d.translate(sql)
    assert translated.count("%s") == 4
    assert "%%s" not in translated


# --------------------------------------------------------------------------- #
# SqliteBackend contract (execute/query/query_one/query_scalar/executemany/tx)
# --------------------------------------------------------------------------- #

@pytest.fixture
def sqlite_backend(tmp_path):
    backend = SqliteBackend(str(tmp_path / "t.db"))
    backend.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, name TEXT)")
    yield backend
    backend.close()


def test_sqlite_backend_execute_and_query(sqlite_backend):
    sqlite_backend.execute("INSERT INTO t (id, name) VALUES (?, ?)", (1, "a"))
    rows = sqlite_backend.query("SELECT * FROM t")
    assert rows == [{"id": 1, "name": "a"}]


def test_sqlite_backend_query_one_and_missing(sqlite_backend):
    sqlite_backend.execute("INSERT INTO t (id, name) VALUES (?, ?)", (1, "a"))
    assert sqlite_backend.query_one("SELECT * FROM t WHERE id = ?", (1,)) == {"id": 1, "name": "a"}
    assert sqlite_backend.query_one("SELECT * FROM t WHERE id = ?", (2,)) is None


def test_sqlite_backend_query_scalar(sqlite_backend):
    sqlite_backend.execute("INSERT INTO t (id, name) VALUES (?, ?)", (1, "a"))
    assert sqlite_backend.query_scalar("SELECT COUNT(*) FROM t") == 1
    assert sqlite_backend.query_scalar("SELECT name FROM t WHERE id = ?", (99,)) is None


def test_sqlite_backend_executemany(sqlite_backend):
    sqlite_backend.executemany(
        "INSERT INTO t (id, name) VALUES (?, ?)", [(1, "a"), (2, "b")])
    assert sqlite_backend.query_scalar("SELECT COUNT(*) FROM t") == 2


def test_sqlite_backend_transaction_commits_on_success(sqlite_backend):
    with sqlite_backend.transaction() as tx:
        tx.execute("INSERT INTO t (id, name) VALUES (?, ?)", (1, "a"))
        tx.execute("INSERT INTO t (id, name) VALUES (?, ?)", (2, "b"))
    assert sqlite_backend.query_scalar("SELECT COUNT(*) FROM t") == 2


def test_sqlite_backend_transaction_rolls_back_on_exception(sqlite_backend):
    sqlite_backend.execute("INSERT INTO t (id, name) VALUES (?, ?)", (1, "a"))
    with pytest.raises(RuntimeError):
        with sqlite_backend.transaction() as tx:
            tx.execute("DELETE FROM t WHERE id = ?", (1,))
            raise RuntimeError("boom")
    # The DELETE inside the failed transaction must not have stuck.
    assert sqlite_backend.query_scalar("SELECT COUNT(*) FROM t") == 1


def test_sqlite_backend_raw_connection_for_ddl(tmp_path):
    backend = SqliteBackend(str(tmp_path / "raw.db"))
    try:
        with backend.raw_connection() as conn:
            conn.execute("CREATE TABLE IF NOT EXISTS x (a INTEGER)")
            conn.commit()
        assert backend.query("SELECT * FROM x") == []
    finally:
        backend.close()


# --------------------------------------------------------------------------- #
# Q3 (perf review 2026-09-26): iter_id_column -- id-only streaming without
# dict_row/Row materialization (docs/reviews/2026-09-26-enqueue-path-
# performance-review.md).
# --------------------------------------------------------------------------- #

def test_sqlite_iter_id_column_matches_query_values(sqlite_backend):
    sqlite_backend.executemany(
        "INSERT INTO t (id, name) VALUES (?, ?)",
        [(1, "a"), (2, "b"), (3, "c")])
    ids = list(sqlite_backend.iter_id_column("SELECT id FROM t ORDER BY id"))
    assert ids == [1, 2, 3]


def test_sqlite_iter_id_column_respects_params(sqlite_backend):
    sqlite_backend.executemany(
        "INSERT INTO t (id, name) VALUES (?, ?)",
        [(1, "a"), (2, "b"), (3, "c")])
    ids = list(sqlite_backend.iter_id_column(
        "SELECT id FROM t WHERE name = ?", ("b",)))
    assert ids == [2]


def test_sqlite_iter_id_column_batch_boundary(sqlite_backend):
    """batch_rows smaller than the result set must still yield every row,
    in order -- the whole point is looping fetchmany() to exhaustion."""
    sqlite_backend.executemany(
        "INSERT INTO t (id, name) VALUES (?, ?)",
        [(i, str(i)) for i in range(1, 8)])
    ids = list(sqlite_backend.iter_id_column(
        "SELECT id FROM t ORDER BY id", batch_rows=3))
    assert ids == list(range(1, 8))


def test_sqlite_iter_id_column_empty_result(sqlite_backend):
    assert list(sqlite_backend.iter_id_column("SELECT id FROM t")) == []


def test_sqlite_iter_id_column_is_generator_not_dict_rows(sqlite_backend):
    """Rows must come back as plain scalars, not dict/Row-wrapped -- the
    memory-saving whole point of this method vs. query()."""
    sqlite_backend.execute("INSERT INTO t (id, name) VALUES (?, ?)", (1, "a"))
    for value in sqlite_backend.iter_id_column("SELECT id FROM t"):
        assert isinstance(value, int)
