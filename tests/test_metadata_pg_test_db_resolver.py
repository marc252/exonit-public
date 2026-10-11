"""Tests for tests/_metadata_pg_test_db.py's DSN resolver — the "database
name must contain 'test'" hard guard for the TGLC-index PG-parity tests
(docs/superpowers/specs/2026-07-20-tglc-index-to-pg.md, item (f): "PG-
touching tests are opt-in ... and must refuse to run unless the DSN's
database name contains 'test'").

These tests never touch a real PostgreSQL connection — they exercise the
resolver's string logic only, so they always run (no EXONIT_TEST_META_PG_URL
/ psycopg / network dependency).
"""
import pytest

from tests._metadata_pg_test_db import resolve_test_meta_dsn


def test_unset_env_returns_none(monkeypatch):
    monkeypatch.delenv("EXONIT_TEST_META_PG_URL", raising=False)
    assert resolve_test_meta_dsn() is None


def test_db_name_containing_test_is_accepted(monkeypatch):
    monkeypatch.setenv(
        "EXONIT_TEST_META_PG_URL",
        "postgresql://exonit@<postgres-host>:5432/exonit_meta_test")
    assert resolve_test_meta_dsn() == \
        "postgresql://exonit@<postgres-host>:5432/exonit_meta_test"


def test_db_name_without_test_is_refused(monkeypatch):
    monkeypatch.setenv(
        "EXONIT_TEST_META_PG_URL",
        "postgresql://exonit@<postgres-host>:5432/exonit_meta")
    with pytest.raises(RuntimeError, match="does not contain 'test'"):
        resolve_test_meta_dsn()


def test_production_db_name_is_refused(monkeypatch):
    """The exact gotcha this guard exists for: a DSN accidentally pointing
    at the real exonit_meta (or exonit) database must never be usable here."""
    monkeypatch.setenv(
        "EXONIT_TEST_META_PG_URL",
        "postgresql://exonit@<postgres-host>:5432/exonit")
    with pytest.raises(RuntimeError):
        resolve_test_meta_dsn()


def test_case_insensitive_test_match(monkeypatch):
    monkeypatch.setenv(
        "EXONIT_TEST_META_PG_URL",
        "postgresql://exonit@<postgres-host>:5432/Exonit_Meta_TEST")
    assert resolve_test_meta_dsn() is not None


def test_dsn_with_query_string_extracts_db_name_correctly(monkeypatch):
    monkeypatch.setenv(
        "EXONIT_TEST_META_PG_URL",
        "postgresql://exonit@<postgres-host>:5432/exonit_meta_test?sslmode=disable")
    assert resolve_test_meta_dsn() is not None
