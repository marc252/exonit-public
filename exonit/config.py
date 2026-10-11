"""Centralized configuration for ExoNit."""

import logging
import os
import sys

LOG = logging.getLogger("exonit.config")


def _sqlite_fallback_file_exists(path: str) -> bool:
    """Indirection so tests can monkeypatch the existence check without
    touching a real file on disk."""
    return os.path.exists(path)


def _is_pg_dsn(url) -> bool:
    return bool(url) and str(url).lower().startswith(("postgresql://", "postgres://"))


class ExoNitConfig:
    """Configuration loaded from environment variables with sensible defaults."""

    def __init__(self):
        self.project_root = os.path.dirname(
            os.path.dirname(os.path.abspath(__file__)))

        self.port = int(os.environ.get("EXONIT_PORT", "8400"))
        # db_path is the value handed straight to ExoNitDB(...) / make_backend(...),
        # which discriminate a PostgreSQL DSN from a filesystem path (is_pg_url).
        # EXONIT_DATABASE_URL (a postgresql:// DSN) wins over the SQLite path so a
        # single env var flips the whole process to PG; absent it, the historical
        # SQLite path (EXONIT_DB_PATH or the default file) is returned unchanged.
        _database_url = os.environ.get("EXONIT_DATABASE_URL")
        _explicit_db_path = os.environ.get("EXONIT_DB_PATH")
        _default_sqlite_path = os.path.join(self.project_root, "data", "exonit.db")
        self.db_path = _database_url or _explicit_db_path or _default_sqlite_path

        # Footgun B9 (bit us 2026-07-03): outside systemd — which sets
        # EXONIT_DATABASE_URL via the .env EnvironmentFile — a script that
        # instantiates this config with neither EXONIT_DATABASE_URL nor
        # EXONIT_DB_PATH set silently falls back to the legacy SQLite file
        # and reads a dead snapshot with zero indication anything is wrong.
        # Warn loudly, exactly once per instantiation, but only when that
        # implicit fallback actually shadows a real file — an explicitly
        # provided path (EXONIT_DB_PATH) or DSN (EXONIT_DATABASE_URL) is a
        # deliberate choice and must stay silent, and a fresh environment
        # with no legacy file has nothing to silently shadow.
        _implicit_fallback = _database_url is None and _explicit_db_path is None
        if _implicit_fallback and _sqlite_fallback_file_exists(_default_sqlite_path):
            _msg = (
                f"EXONIT_DATABASE_URL is not set — falling back to the legacy "
                f"SQLite file at {_default_sqlite_path!r}, which may be a stale "
                f"snapshot. To use PostgreSQL, set EXONIT_DATABASE_URL (or pass "
                f"--db explicitly to this script)."
            )
            LOG.warning(_msg)
            print(f"WARNING: exonit.config: {_msg}", file=sys.stderr)

        self.metadata_cache_path = os.environ.get(
            "EXONIT_METADATA_CACHE",
            os.path.join(self.project_root, "data", "metadata_cache.db"))
        # 2026-07-20 TGLC-index-to-PG migration (F1): the PG DSN for the
        # metadata/tglc/fits_blobs/fits_paths/nearby_stars index, a SEPARATE
        # database (exonit_meta) from the operational EXONIT_DATABASE_URL —
        # see docs/superpowers/specs/2026-07-20-tglc-index-to-pg.md D1/D2.
        # Unset (default None) -> exonit/metadata_cache.py's MetadataCache
        # stays on SQLite at metadata_cache_path, byte-for-byte unchanged.
        # NOT read via ExoNitConfig by MetadataCache itself (it reads the env
        # var directly, so every caller across the codebase — which all pass
        # metadata_cache_path today — flips to PG together without each call
        # site needing to change); exposed on the config object too, for any
        # caller that wants to branch on it explicitly (e.g. the migration
        # tooling in a later phase).
        self.metadata_db_url = os.environ.get("EXONIT_METADATA_DB_URL")
        self.reports_dir = os.environ.get(
            "EXONIT_REPORTS_DIR",
            os.path.join(self.project_root, "reports"))
        self.catalog_path = os.environ.get(
            "EXONIT_CATALOG",
            os.path.join(self.project_root, "data", "toi_catalog.csv"))
        self.berger_db_path = os.environ.get(
            "EXONIT_BERGER_DB",
            os.path.join(self.project_root, "data", "catalogs",
                         "berger_2023_stellar.db"))
        # SPOC FITS cache root (an NFS mount on a multi-host deployment).
        # EXONIT_NFS_CACHE is the long-standing name; EXONIT_TESS_CACHE_DIR
        # is accepted as an alias. Neutral default: <project>/data/tess_cache.
        _cache_env = (os.environ.get("EXONIT_NFS_CACHE")
                      or os.environ.get("EXONIT_TESS_CACHE_DIR"))
        self.nfs_cache_path = _cache_env or os.path.join(
            self.project_root, "data", "tess_cache")
        # Same footgun class as B9: production (PostgreSQL) used to get its SPOC
        # root from a code default; with no cache variable set the process would
        # silently read an empty <project>/data/tess_cache. PostgreSQL is the
        # production marker, so only then is the absence worth an ERROR.
        if not _cache_env and _is_pg_dsn(_database_url):
            _msg = (
                f"neither EXONIT_NFS_CACHE nor EXONIT_TESS_CACHE_DIR is set while "
                f"EXONIT_DATABASE_URL is PostgreSQL — using {self.nfs_cache_path!r}, "
                f"which is probably empty. Set EXONIT_NFS_CACHE to the SPOC FITS cache."
            )
            LOG.error(_msg)
            print(f"ERROR: exonit.config: {_msg}", file=sys.stderr)
        self.worker_token = os.environ.get(
            "EXONIT_WORKER_TOKEN", "exonit-default-token")

        # Pipeline defaults
        # (max_sectors/snr_threshold/max_planets removed 2026-06-09: ghost
        # config — no pipeline code ever read them; the real thresholds live
        # in the worker and in run_nightly backfill args.)


# Singleton
config = ExoNitConfig()


def deployment() -> str:
    """Deployment role for this process: ``home`` (default) or ``aws``.

    Read live from ``EXONIT_DEPLOYMENT`` (not cached on the singleton) so the
    same codebase behaves as home or AWS purely by environment. Drives UI panel
    gating and which control endpoints are active.
    """
    return os.environ.get("EXONIT_DEPLOYMENT", "home").lower()


def worker_token() -> str:
    """Worker/enqueue auth token, read live from ``EXONIT_WORKER_TOKEN``.

    Home and AWS each set their own; enqueue and worker auth must use this so a
    cross-deployment token mismatch (401) cannot happen.
    """
    return os.environ.get("EXONIT_WORKER_TOKEN", "exonit-default-token")
