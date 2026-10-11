"""Tests for exonit/object_store.py.

Two layers:
  * unit tests of the enable/config logic (no network)
  * a round-trip test against the real MinIO on the <host>, which skips
    cleanly when EXONIT_S3_* credentials are not in the environment so the suite
    still runs on a box without MinIO access.
"""

import os

import pytest

from exonit import object_store as ostore


# --------------------------------------------------------------------------- #
# Unit: enable + config (no network)
# --------------------------------------------------------------------------- #

def test_disabled_without_creds(monkeypatch):
    for k in ("EXONIT_S3_ENDPOINT", "EXONIT_S3_ACCESS_KEY",
              "EXONIT_S3_SECRET_KEY", "EXONIT_S3_USE_AWS"):
        monkeypatch.delenv(k, raising=False)
    assert ostore.is_enabled() is False
    assert ostore.store() is None


def test_enabled_with_minio_endpoint(monkeypatch):
    monkeypatch.setenv("EXONIT_S3_ENDPOINT", "http://<postgres-host>:9000")
    monkeypatch.setenv("EXONIT_S3_ACCESS_KEY", "k")
    monkeypatch.setenv("EXONIT_S3_SECRET_KEY", "s")
    assert ostore.is_enabled() is True


def test_endpoint_alone_not_enough(monkeypatch):
    # An endpoint with no credentials must NOT enable (would 403 at runtime).
    monkeypatch.setenv("EXONIT_S3_ENDPOINT", "http://<postgres-host>:9000")
    monkeypatch.delenv("EXONIT_S3_ACCESS_KEY", raising=False)
    monkeypatch.delenv("EXONIT_S3_SECRET_KEY", raising=False)
    assert ostore.is_enabled() is False


def test_default_bucket(monkeypatch):
    monkeypatch.delenv("EXONIT_S3_FITS_BUCKET", raising=False)
    assert ostore.fits_bucket() == "exonit-fits"
    monkeypatch.setenv("EXONIT_S3_FITS_BUCKET", "custom")
    assert ostore.fits_bucket() == "custom"


# --------------------------------------------------------------------------- #
# Configurable client timeouts/retries + reset_client (2026-07-21 — see
# scripts/discovery_download.py's _pack_and_upload_chunk, which constructs a
# dedicated ObjectStore with these overridden rather than mutating the
# shared singleton's defaults).
# --------------------------------------------------------------------------- #

def test_default_client_config_unchanged(monkeypatch):
    """The shared-singleton-facing defaults (no constructor overrides) stay
    byte-for-byte what they were before this class grew the new params —
    every existing caller of object_store.store() must be unaffected."""
    monkeypatch.setenv("EXONIT_S3_ENDPOINT", "http://<postgres-host>:9000")
    monkeypatch.setenv("EXONIT_S3_ACCESS_KEY", "k")
    monkeypatch.setenv("EXONIT_S3_SECRET_KEY", "s")
    s = ostore.ObjectStore()
    client = s.client
    assert client.meta.config.read_timeout == 60
    assert client.meta.config.connect_timeout == 10
    # botocore normalises max_attempts -> total_max_attempts (= max_attempts+1,
    # the retry count plus the original try).
    assert client.meta.config.retries["total_max_attempts"] == 6
    assert client.meta.config.retries["mode"] == "standard"


def test_overridden_client_config(monkeypatch):
    """A dedicated instance can raise read_timeout and disable botocore's
    own per-part retries (the pack/upload retry loop handles retries
    itself, at the whole-chunk level)."""
    monkeypatch.setenv("EXONIT_S3_ENDPOINT", "http://<postgres-host>:9000")
    monkeypatch.setenv("EXONIT_S3_ACCESS_KEY", "k")
    monkeypatch.setenv("EXONIT_S3_SECRET_KEY", "s")
    s = ostore.ObjectStore(read_timeout=300, retries_max_attempts=0)
    client = s.client
    assert client.meta.config.read_timeout == 300
    # max_attempts=0 -> total_max_attempts=1: no retries, one try only.
    assert client.meta.config.retries["total_max_attempts"] == 1


def test_reset_client_drops_cached_client(monkeypatch):
    """reset_client() forces the NEXT .client access to build a fresh boto3
    client instead of reusing a possibly-poisoned connection."""
    monkeypatch.setenv("EXONIT_S3_ENDPOINT", "http://<postgres-host>:9000")
    monkeypatch.setenv("EXONIT_S3_ACCESS_KEY", "k")
    monkeypatch.setenv("EXONIT_S3_SECRET_KEY", "s")
    s = ostore.ObjectStore()
    first = s.client
    assert s.client is first, "unchanged: repeated access returns the same client"
    s.reset_client()
    second = s.client
    assert second is not first, "reset_client must force a fresh client"


# --------------------------------------------------------------------------- #
# Round-trip against real MinIO (skips without creds)
# --------------------------------------------------------------------------- #

def _minio_available():
    return bool(os.environ.get("EXONIT_S3_ENDPOINT")
                and os.environ.get("EXONIT_S3_ACCESS_KEY")
                and os.environ.get("EXONIT_S3_SECRET_KEY"))


@pytest.mark.skipif(not _minio_available(),
                    reason="MinIO/EXONIT_S3_* not configured in env")
def test_minio_round_trip():
    import urllib.request

    s = ostore.ObjectStore()
    key = "_pytest/round_trip.bin"
    payload = b"exonit-object-store-pytest"
    try:
        s.put_bytes(key, payload)
        assert s.exists(key) is True
        assert s.get_bytes(key) == payload
        # Presigned GET must be fetchable with a plain HTTP client — this is the
        # exact path a worker uses after following the server's 307 redirect.
        url = s.presign_get(key, expires=120)
        assert s.bucket in url
        fetched = urllib.request.urlopen(url, timeout=10).read()
        assert fetched == payload
    finally:
        s.delete_object(key)
    assert s.exists(key) is False
