"""Unit tests for the PG connection-pool timeout/keepalive kwargs added to
``exonit/db_backend.py`` and ``exonit/metadata_backend.py`` (incident
2026-08-15: a network blip left psycopg connections in a "black hole",
piling up ASGI threads until the server hung — see ``_pg_connect_kwargs``'s
docstring in both modules).

No real PostgreSQL is needed: ``psycopg_pool.ConnectionPool`` is monkeypatched
with a fake that just records the kwargs it was called with, so these tests
run on a SQLite-only box like the rest of the pure-unit layer.
"""
import pytest

import exonit.db_backend as db_backend
import exonit.metadata_backend as metadata_backend

pytest.importorskip("psycopg_pool")


class _FakeConnectionPool:
    """Stand-in for psycopg_pool.ConnectionPool: records the constructor
    call instead of opening a real connection."""

    last_call = None

    def __init__(self, url, min_size=1, max_size=8, kwargs=None, open=True):
        self.url = url
        self.min_size = min_size
        self.max_size = max_size
        self.kwargs = kwargs
        self.open = open
        _FakeConnectionPool.last_call = self

    def close(self):
        pass


@pytest.fixture(autouse=True)
def _fake_connection_pool(monkeypatch):
    """Patch psycopg_pool.ConnectionPool for every test in this module —
    both db_backend.PgBackend and metadata_backend.PgBackend do a local
    ``from psycopg_pool import ConnectionPool`` inside __init__, which reads
    this module attribute at call time."""
    import psycopg_pool
    monkeypatch.setattr(psycopg_pool, "ConnectionPool", _FakeConnectionPool)
    _FakeConnectionPool.last_call = None
    yield


# --------------------------------------------------------------------------- #
# Both modules expose the same PgBackend(url, ...) constructor + the same
# _pg_connect_kwargs() shape, so the tests are parametrised over the module.
# --------------------------------------------------------------------------- #

MODULES = pytest.mark.parametrize("mod", [db_backend, metadata_backend], ids=["db_backend", "metadata_backend"])


@MODULES
def test_defaults(mod, monkeypatch):
    monkeypatch.delenv("EXONIT_PG_CONNECT_TIMEOUT_S", raising=False)
    monkeypatch.delenv("EXONIT_PG_STATEMENT_TIMEOUT_MS", raising=False)

    mod.PgBackend("postgresql://u:p@h/db")
    kwargs = _FakeConnectionPool.last_call.kwargs

    assert kwargs["connect_timeout"] == 10
    assert kwargs["keepalives"] == 1
    assert kwargs["keepalives_idle"] == 30
    assert kwargs["keepalives_interval"] == 10
    assert kwargs["keepalives_count"] == 3
    assert kwargs["options"] == "-c statement_timeout=600000"


@MODULES
def test_statement_timeout_override(mod, monkeypatch):
    monkeypatch.setenv("EXONIT_PG_STATEMENT_TIMEOUT_MS", "30000")
    monkeypatch.delenv("EXONIT_PG_CONNECT_TIMEOUT_S", raising=False)

    mod.PgBackend("postgresql://u:p@h/db")
    kwargs = _FakeConnectionPool.last_call.kwargs

    assert kwargs["options"] == "-c statement_timeout=30000"


@MODULES
def test_connect_timeout_override(mod, monkeypatch):
    monkeypatch.setenv("EXONIT_PG_CONNECT_TIMEOUT_S", "5")
    monkeypatch.delenv("EXONIT_PG_STATEMENT_TIMEOUT_MS", raising=False)

    mod.PgBackend("postgresql://u:p@h/db")
    kwargs = _FakeConnectionPool.last_call.kwargs

    assert kwargs["connect_timeout"] == 5


@MODULES
def test_statement_timeout_zero_disables_options(mod, monkeypatch):
    monkeypatch.setenv("EXONIT_PG_STATEMENT_TIMEOUT_MS", "0")
    monkeypatch.delenv("EXONIT_PG_CONNECT_TIMEOUT_S", raising=False)

    mod.PgBackend("postgresql://u:p@h/db")
    kwargs = _FakeConnectionPool.last_call.kwargs

    assert "options" not in kwargs
    # keepalives stay on regardless — they're unconditional, not tied to
    # the statement timeout.
    assert kwargs["keepalives"] == 1


@MODULES
@pytest.mark.parametrize("bad_value", ["abc", "-5", ""])
def test_malformed_env_falls_back_to_defaults(mod, monkeypatch, bad_value):
    monkeypatch.setenv("EXONIT_PG_STATEMENT_TIMEOUT_MS", bad_value)
    monkeypatch.setenv("EXONIT_PG_CONNECT_TIMEOUT_S", bad_value)

    # Must not raise — these backends are constructed at server startup.
    mod.PgBackend("postgresql://u:p@h/db")
    kwargs = _FakeConnectionPool.last_call.kwargs

    assert kwargs["connect_timeout"] == 10
    assert kwargs["options"] == "-c statement_timeout=600000"


@MODULES
def test_row_factory_still_present(mod, monkeypatch):
    """The new kwargs must merge with, not replace, the existing
    row_factory=dict_row kwarg."""
    monkeypatch.delenv("EXONIT_PG_CONNECT_TIMEOUT_S", raising=False)
    monkeypatch.delenv("EXONIT_PG_STATEMENT_TIMEOUT_MS", raising=False)

    mod.PgBackend("postgresql://u:p@h/db")
    kwargs = _FakeConnectionPool.last_call.kwargs

    assert "row_factory" in kwargs
