import logging
import os

import exonit.config as config_mod
from exonit.config import ExoNitConfig


def test_default_config(monkeypatch):
    monkeypatch.delenv("EXONIT_NFS_CACHE", raising=False)
    monkeypatch.delenv("EXONIT_TESS_CACHE_DIR", raising=False)
    cfg = ExoNitConfig()
    assert cfg.port == 8400
    assert cfg.db_path.endswith("exonit.db")
    assert cfg.worker_token != ""
    # neutral default: <project>/data/tess_cache (no site-specific mount)
    assert cfg.nfs_cache_path == os.path.join(cfg.project_root, "data", "tess_cache")


def test_nfs_cache_env_names(monkeypatch):
    monkeypatch.delenv("EXONIT_NFS_CACHE", raising=False)
    monkeypatch.setenv("EXONIT_TESS_CACHE_DIR", "/srv/tess")
    assert ExoNitConfig().nfs_cache_path == "/srv/tess"
    monkeypatch.setenv("EXONIT_NFS_CACHE", "/srv/nfs")  # long-standing name wins
    assert ExoNitConfig().nfs_cache_path == "/srv/nfs"


def test_config_from_env(monkeypatch):
    monkeypatch.setenv("EXONIT_PORT", "9000")
    monkeypatch.setenv("EXONIT_WORKER_TOKEN", "my-secret-token")
    cfg = ExoNitConfig()
    assert cfg.port == 9000
    assert cfg.worker_token == "my-secret-token"


def test_config_project_root():
    cfg = ExoNitConfig()
    assert os.path.isdir(cfg.project_root)


# --------------------------------------------------------------------- #
# metadata_db_url (2026-07-20 TGLC->PG migration, F1)
# --------------------------------------------------------------------- #

def test_metadata_db_url_unset_by_default(monkeypatch):
    monkeypatch.delenv("EXONIT_METADATA_DB_URL", raising=False)
    cfg = ExoNitConfig()
    assert cfg.metadata_db_url is None
    # metadata_cache_path is unaffected — separate setting (D2 of the spec).
    assert cfg.metadata_cache_path.endswith("metadata_cache.db")


def test_metadata_db_url_reads_env_var(monkeypatch):
    monkeypatch.setenv(
        "EXONIT_METADATA_DB_URL",
        "postgresql://exonit@pg-host.example:5432/exonit_meta")
    cfg = ExoNitConfig()
    assert cfg.metadata_db_url == "postgresql://exonit@pg-host.example:5432/exonit_meta"
    # metadata_cache_path stays the SQLite-path setting, independent of the
    # new DSN var — the two are read separately (D2).
    assert cfg.metadata_cache_path.endswith("metadata_cache.db")


def test_db_path_default_is_sqlite(monkeypatch):
    monkeypatch.delenv("EXONIT_DATABASE_URL", raising=False)
    monkeypatch.delenv("EXONIT_DB_PATH", raising=False)
    cfg = ExoNitConfig()
    assert cfg.db_path.endswith("exonit.db")


def test_db_path_respects_db_path_env(monkeypatch):
    monkeypatch.delenv("EXONIT_DATABASE_URL", raising=False)
    monkeypatch.setenv("EXONIT_DB_PATH", "/tmp/custom.db")
    cfg = ExoNitConfig()
    assert cfg.db_path == "/tmp/custom.db"


def test_db_path_returns_dsn_when_database_url_set(monkeypatch):
    # EXONIT_DATABASE_URL wins over EXONIT_DB_PATH: db_path is the DSN.
    monkeypatch.setenv("EXONIT_DATABASE_URL",
                       "postgresql://exonit@pg-host.example:5432/exonit")
    monkeypatch.setenv("EXONIT_DB_PATH", "/tmp/custom.db")
    cfg = ExoNitConfig()
    assert cfg.db_path == "postgresql://exonit@pg-host.example:5432/exonit"


# --------------------------------------------------------------------- #
# Footgun B9: loud warning on *implicit* SQLite fallback (2026-07-03)
# --------------------------------------------------------------------- #

def test_warns_on_implicit_sqlite_fallback_when_legacy_file_exists(
    monkeypatch, caplog, capsys
):
    monkeypatch.delenv("EXONIT_DATABASE_URL", raising=False)
    monkeypatch.delenv("EXONIT_DB_PATH", raising=False)
    monkeypatch.setattr(config_mod, "_sqlite_fallback_file_exists", lambda path: True)

    with caplog.at_level(logging.WARNING, logger="exonit.config"):
        cfg = ExoNitConfig()

    assert cfg.db_path.endswith("exonit.db")
    assert any("EXONIT_DATABASE_URL" in r.message for r in caplog.records)
    assert any(cfg.db_path in r.message for r in caplog.records)
    err = capsys.readouterr().err
    assert "WARNING" in err
    assert cfg.db_path in err
    assert "EXONIT_DATABASE_URL" in err
    assert "--db" in err


def test_no_warning_when_legacy_file_absent(monkeypatch, caplog):
    """Implicit fallback, but nothing on disk to silently shadow — no warning."""
    monkeypatch.delenv("EXONIT_DATABASE_URL", raising=False)
    monkeypatch.delenv("EXONIT_DB_PATH", raising=False)
    monkeypatch.setattr(config_mod, "_sqlite_fallback_file_exists", lambda path: False)

    with caplog.at_level(logging.WARNING, logger="exonit.config"):
        ExoNitConfig()

    assert caplog.records == []


def test_no_warning_when_db_path_explicitly_set(monkeypatch, caplog):
    """An explicit EXONIT_DB_PATH is a deliberate choice, not a footgun."""
    monkeypatch.delenv("EXONIT_DATABASE_URL", raising=False)
    monkeypatch.setenv("EXONIT_DB_PATH", "/tmp/custom.db")
    monkeypatch.setattr(config_mod, "_sqlite_fallback_file_exists", lambda path: True)

    with caplog.at_level(logging.WARNING, logger="exonit.config"):
        ExoNitConfig()

    assert caplog.records == []


def test_no_warning_when_database_url_set(monkeypatch, caplog):
    """PG explicitly configured — no SQLite fallback warning, even if the
    legacy file happens to still exist on disk."""
    monkeypatch.setenv("EXONIT_DATABASE_URL",
                       "postgresql://exonit@pg-host.example:5432/exonit")
    monkeypatch.setenv("EXONIT_NFS_CACHE", "/srv/tess")
    monkeypatch.delenv("EXONIT_DB_PATH", raising=False)
    monkeypatch.setattr(config_mod, "_sqlite_fallback_file_exists", lambda path: True)

    with caplog.at_level(logging.WARNING, logger="exonit.config"):
        ExoNitConfig()

    assert caplog.records == []


# --------------------------------------------------------------------- #
# SPOC cache root: loud ERROR when PostgreSQL is configured but no cache
# variable is (production used to get it from a code default)
# --------------------------------------------------------------------- #

def test_errors_when_pg_and_no_cache_variable(monkeypatch, caplog, capsys):
    monkeypatch.delenv("EXONIT_NFS_CACHE", raising=False)
    monkeypatch.delenv("EXONIT_TESS_CACHE_DIR", raising=False)
    monkeypatch.setenv("EXONIT_DATABASE_URL", "postgresql://u@db.invalid:5432/x")  # never connected
    with caplog.at_level(logging.ERROR, logger="exonit.config"):
        ExoNitConfig()
    assert any(r.levelno == logging.ERROR and "EXONIT_NFS_CACHE" in r.message
               for r in caplog.records)
    assert "ERROR" in capsys.readouterr().err


def test_no_cache_error_for_sqlite_or_when_set(monkeypatch, caplog):
    monkeypatch.delenv("EXONIT_NFS_CACHE", raising=False)
    monkeypatch.delenv("EXONIT_TESS_CACHE_DIR", raising=False)
    monkeypatch.delenv("EXONIT_DATABASE_URL", raising=False)
    with caplog.at_level(logging.ERROR, logger="exonit.config"):
        ExoNitConfig()
    assert not [r for r in caplog.records if r.levelno >= logging.ERROR]
    monkeypatch.setenv("EXONIT_DATABASE_URL", "postgresql://u@db.invalid:5432/x")
    monkeypatch.setenv("EXONIT_NFS_CACHE", "/srv/x")
    caplog.clear()
    with caplog.at_level(logging.ERROR, logger="exonit.config"):
        ExoNitConfig()
    assert not [r for r in caplog.records if r.levelno >= logging.ERROR]
