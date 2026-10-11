"""Object storage (S3-compatible) for ExoNit FITS and artifacts.

Unified S3 API per the architecture redesign (docs/superpowers/plans/
2026-06-20-architecture-redesign.md, Phase 4): bulky artifacts (FITS/LC/plots/
models) live in object storage — MinIO on the <host> 8 TB at home, AWS S3 in
the cloud — and are served to workers via presigned URLs so the control-plane
server stops streaming heavy data.

This module is ADDITIVE and safe to deploy dark: with no ``EXONIT_S3_ENDPOINT``
(or no credentials) configured, ``is_enabled()`` is False and every caller falls
back to the historical local-disk path. Turning it on is a pure env-var change.

Config (environment):
  EXONIT_S3_ENDPOINT     MinIO endpoint, e.g. http://minio-host:9000.
                         Empty/unset with keys present → real AWS S3.
  EXONIT_S3_FITS_BUCKET  bucket for TGLC FITS (default: exonit-fits).
  EXONIT_S3_ACCESS_KEY   access key (MinIO root or a scoped service account).
  EXONIT_S3_SECRET_KEY   secret key.
  EXONIT_S3_REGION       region (default us-east-1; MinIO ignores it).

MinIO needs path-style addressing (``http://host:9000/bucket/key``), not the
virtual-host style boto3 defaults to for AWS, so the client is built with
``addressing_style='path'`` and SigV4.
"""

import os
import threading
import time
from typing import Iterator, Optional


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default)


def is_enabled() -> bool:
    """True when object storage is configured (endpoint or AWS keys + creds)."""
    has_creds = bool(_env("EXONIT_S3_ACCESS_KEY") and _env("EXONIT_S3_SECRET_KEY"))
    # An explicit endpoint (MinIO) or AWS creds both enable it; the bucket always
    # has a default, so creds are the real switch.
    return has_creds and bool(_env("EXONIT_S3_ENDPOINT") or _env("EXONIT_S3_USE_AWS"))


def fits_bucket() -> str:
    return _env("EXONIT_S3_FITS_BUCKET", "exonit-fits")


class ObjectStore:
    """Thin S3 wrapper (boto3) for one process. Thread-safe; lazily connects.

    Works against MinIO (endpoint set) and AWS S3 (endpoint absent) with the same
    API, matching the redesign's "unified S3 API" requirement.
    """

    def __init__(self, bucket: Optional[str] = None, *,
                 connect_timeout: int = 10, read_timeout: int = 60,
                 retries_max_attempts: int = 5, retries_mode: str = "standard"):
        """*connect_timeout*/*read_timeout*/*retries_** override the boto3
        client Config (defaults unchanged — every existing caller of the
        shared ``store()`` singleton keeps its historical behaviour).

        A caller that needs a different timeout/retry policy for one
        specific, long-running operation (e.g.
        scripts/discovery_download.py's per-chunk pack/upload, which runs
        its OWN whole-sequence retry+backoff — see ``_pack_and_upload_chunk``
        — and therefore wants botocore's per-part retries OFF, plus a longer
        read_timeout, so a stalled multipart PUT surfaces to ITS retry loop
        instead of racing it) constructs a dedicated instance with these
        overridden, rather than mutating the shared singleton — so the
        change stays scoped to that call site instead of silently affecting
        every other consumer of object storage (server FITS serving, worker
        fetches, ...). See 2026-07-21 MinIO multipart read-timeout incident.
        """
        self.bucket = bucket or fits_bucket()
        self.endpoint = _env("EXONIT_S3_ENDPOINT") or None
        self.region = _env("EXONIT_S3_REGION", "us-east-1")
        self._access = _env("EXONIT_S3_ACCESS_KEY") or None
        self._secret = _env("EXONIT_S3_SECRET_KEY") or None
        self._connect_timeout = connect_timeout
        self._read_timeout = read_timeout
        self._retries_max_attempts = retries_max_attempts
        self._retries_mode = retries_mode
        self._client = None
        self._lock = threading.Lock()

    # -- connection ------------------------------------------------------- #
    @property
    def client(self):
        if self._client is None:
            with self._lock:
                if self._client is None:
                    self._client = self._make_client()
        return self._client

    def reset_client(self) -> None:
        """Drop the cached boto3 client so the next access re-creates it.

        Used by a caller retrying a failed multi-part operation (e.g.
        discovery_download.py's per-chunk pack/upload retry loop): a
        read-timeout mid-upload (MinIO under load) can leave the underlying
        HTTP connection pool in a bad state, and retrying on the SAME client
        would just re-hit it. Each retry gets a fresh client instead.
        """
        with self._lock:
            self._client = None

    def _make_client(self):
        import boto3
        from botocore.config import Config

        cfg = Config(
            signature_version="s3v4",
            s3={"addressing_style": "path"},   # MinIO requires path-style
            retries={"max_attempts": self._retries_max_attempts,
                     "mode": self._retries_mode},
            connect_timeout=self._connect_timeout,
            read_timeout=self._read_timeout,
            # Default pool is 10; parallel uploaders/fetchers use more threads
            # than that, and an exhausted pool surfaced as IncompleteBody PUTs.
            max_pool_connections=int(_env("EXONIT_S3_POOL", "48")),
        )
        return boto3.client(
            "s3",
            endpoint_url=self.endpoint,
            region_name=self.region,
            aws_access_key_id=self._access,
            aws_secret_access_key=self._secret,
            config=cfg,
        )

    # -- operations ------------------------------------------------------- #
    def put_file(self, key: str, local_path: str,
                 content_type: str = "application/octet-stream") -> None:
        """Upload a local file to ``key`` (idempotent overwrite).

        Large containers (multi-GB) go up as multipart; use big 64 MB parts and
        bounded concurrency so the part count stays low — MinIO under load
        occasionally returns IncompleteBody on a part, and fewer/larger parts
        plus the client retry policy make a clean upload far more likely.
        """
        from boto3.s3.transfer import TransferConfig
        cfg = TransferConfig(
            multipart_threshold=64 * 1024 * 1024,
            multipart_chunksize=64 * 1024 * 1024,
            max_concurrency=8,
            use_threads=True,
        )
        self.client.upload_file(
            local_path, self.bucket, key,
            ExtraArgs={"ContentType": content_type}, Config=cfg)

    def put_bytes(self, key: str, data: bytes,
                  content_type: str = "application/octet-stream") -> None:
        self.client.put_object(
            Bucket=self.bucket, Key=key, Body=data, ContentType=content_type)

    def get_bytes(self, key: str) -> bytes:
        return self.client.get_object(Bucket=self.bucket, Key=key)["Body"].read()

    def delete_object(self, key: str) -> None:
        """Delete one object (idempotent — S3/MinIO does not error on a
        missing key). Used e.g. by the completeness compaction job
        (``scripts/compact_completeness.py``) to clean up its ``.tmp-<runid>``
        verification copy after the real object is written."""
        self.client.delete_object(Bucket=self.bucket, Key=key)

    def get_range(self, key: str, offset: int, length: int) -> bytes:
        """Read ``length`` bytes starting at ``offset`` from object ``key``.

        Used to serve one FITS out of a packed tar container via an HTTP Range
        read, so only the member's bytes cross the wire (not the whole container).
        """
        end = offset + length - 1
        resp = self.client.get_object(
            Bucket=self.bucket, Key=key, Range=f"bytes={offset}-{end}")
        return resp["Body"].read()

    def download_to(self, key: str, local_path: str) -> None:
        self.client.download_file(self.bucket, key, local_path)

    def exists(self, key: str) -> bool:
        from botocore.exceptions import ClientError
        try:
            self.client.head_object(Bucket=self.bucket, Key=key)
            return True
        except ClientError as e:
            if e.response.get("Error", {}).get("Code") in ("404", "NoSuchKey", "NotFound"):
                return False
            raise

    def presign_get(self, key: str, expires: int = 3600) -> str:
        """Presigned GET URL — workers fetch the object directly, off-server."""
        return self.client.generate_presigned_url(
            "get_object",
            Params={"Bucket": self.bucket, "Key": key},
            ExpiresIn=expires)

    def list_keys(self, prefix: str = "", max_keys: int = 1000) -> Iterator[str]:
        paginator = self.client.get_paginator("list_objects_v2")
        seen = 0
        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                yield obj["Key"]
                seen += 1
                if seen >= max_keys:
                    return


# --------------------------------------------------------------------------- #
# Shared retry+backoff for whole S3/MinIO write (or read) sequences.
# --------------------------------------------------------------------------- #

# Retry schedule shared by every caller that wraps a whole S3 operation
# sequence in retry+backoff (2026-07-21 MinIO multipart read-timeout
# incident — a live s7 fleet run hit 56 chunks of
# scripts/discovery_download.py's ``_pack_and_upload_chunk`` this way, and
# the completeness compactor's remote fast-path, on the SAME class of
# failure — MinIO under load from a concurrent download — died with rc=1
# after 28 min and fell back to local execution). One initial attempt plus
# four retries, each preceded by an exponential-backoff sleep.
S3_RETRY_DELAYS_S = (30, 60, 120, 240)


def with_s3_retries(fn, *, ostore, delays=S3_RETRY_DELAYS_S, on_retry=None):
    """Run ``fn()`` with retry+backoff against transient S3/MinIO failures.

    Total attempts = ``len(delays) + 1`` (one initial attempt, then one
    retry per entry in *delays*). Before every retry (never the first
    attempt) this calls ``ostore.reset_client()`` — a read-timeout mid-
    request can leave the underlying HTTP connection pool in a bad state,
    and retrying on the SAME client would just re-hit it (see
    ``ObjectStore.reset_client``'s docstring) — then sleeps for the
    corresponding entry of *delays*.

    ``fn`` takes no arguments. It should perform the ENTIRE operation being
    retried, not just part of it — the two current callers
    (``scripts/discovery_download.py::_pack_and_upload_chunk`` and
    ``scripts/compact_completeness.py``'s bucket read/upload) each retry a
    self-contained sequence: every individual S3 call inside it (PUT/GET/
    HEAD/exists) is safe to redo in full because those calls are themselves
    idempotent (a PUT overwrites the same key with the same bytes; a GET/
    HEAD has no side effect) — a caller with a non-idempotent step (e.g. a
    delete that must run at most once) must not fold it into ``fn``.

    On every attempt but the last, an exception is swallowed here (only
    surfaced to the caller via *on_retry*, if given) and the next attempt
    runs; on the LAST attempt, the exception propagates — this function
    never masks a genuinely final failure, only the transient ones in
    between it and the first attempt.

    *on_retry*, if given, is called as ``on_retry(attempt, attempts, delay,
    exc)`` right before the backoff sleep (``attempt`` is 1-indexed, the
    attempt that just failed). Callers use it to log/emit their own event
    shape instead of this helper hardcoding one — e.g. the download emits a
    structured ``pack_retry`` event, the compactor logs a warning.
    """
    attempts = len(delays) + 1
    last_exc: Optional[BaseException] = None
    for attempt in range(1, attempts + 1):
        try:
            if attempt > 1:
                ostore.reset_client()
            return fn()
        except Exception as exc:
            last_exc = exc
            if attempt < attempts:
                delay = delays[attempt - 1]
                if on_retry is not None:
                    on_retry(attempt, attempts, delay, exc)
                time.sleep(delay)
    assert last_exc is not None
    raise last_exc


# Process-wide singleton (created lazily on first access).
_store_lock = threading.Lock()
_store: Optional[ObjectStore] = None


def store() -> Optional[ObjectStore]:
    """Return the shared ObjectStore, or None when object storage is disabled."""
    global _store
    if not is_enabled():
        return None
    if _store is None:
        with _store_lock:
            if _store is None:
                _store = ObjectStore()
    return _store
