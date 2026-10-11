"""Shared state wrapper for multi-process coordination.

Backed by Redis when available; falls back to an in-memory implementation for
development and single-process testing.  The in-memory backend is NOT safe for
multi-process deployments — each process would maintain its own isolated state.

A3 (implemented): When ``EXONIT_SERVER_WORKERS`` > 1, ``get_shared_state``
hard-asserts that Redis is reachable.  Falling back to MemoryBackend in
multi-process mode would cause split-brain: each process maintains its own
isolated state, leading to double task assignment, broken cross-process
semaphores, and N concurrent ExoMiner Docker containers (OOM).

Single-process mode (``EXONIT_SERVER_WORKERS`` absent or 1) preserves the
pre-A3 behaviour: Redis tried first, MemoryBackend fallback on failure.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from abc import ABC, abstractmethod
from typing import Any, Callable

logger = logging.getLogger(__name__)

DEFAULT_REDIS_URL = "redis://localhost:6379/0"

# ---------------------------------------------------------------------------
# Abstract interface
# ---------------------------------------------------------------------------


class SharedStateBackend(ABC):
    """Protocol shared by ``RedisBackend`` and ``MemoryBackend``."""

    @abstractmethod
    def ping(self) -> bool:
        """Return True if the backend is reachable."""

    # -- scalars -------------------------------------------------------------

    @abstractmethod
    def get(self, key: str) -> str | None:
        """Return the string value for *key*, or None if absent/expired."""

    @abstractmethod
    def set(self, key: str, value: str, *, ttl: float | None = None) -> None:
        """Set *key* to *value*.  *ttl* is time-to-live in seconds (float ok)."""

    @abstractmethod
    def delete(self, key: str) -> None:
        """Delete *key* (no-op if absent)."""

    @abstractmethod
    def set_if_absent(
        self, key: str, value: str, *, ttl: float | None = None
    ) -> bool:
        """Set *key* = *value* only if *key* does not exist (NX semantics).

        Returns True if the key was written, False if it already existed.
        """

    # -- int counters --------------------------------------------------------

    @abstractmethod
    def get_int(self, key: str, default: int = 0) -> int:
        """Return the integer value for *key*, or *default* if absent/expired."""

    @abstractmethod
    def set_int(self, key: str, value: int) -> None:
        """Store an integer as a string (compatible with ``incr``)."""

    @abstractmethod
    def incr(self, key: str, amount: int = 1) -> int:
        """Atomically increment *key* by *amount* and return the new value.

        If the key does not exist it is initialised to 0 before incrementing.
        """

    # -- hash of JSON dicts --------------------------------------------------

    @abstractmethod
    def hset_json(self, key: str, field: str, value: dict) -> None:
        """Store *value* (a JSON-serialisable dict) under hash *key* / *field*."""

    @abstractmethod
    def hget_json(self, key: str, field: str) -> dict | None:
        """Return the dict stored at hash *key* / *field*, or None if absent."""

    @abstractmethod
    def hgetall_json(self, key: str) -> dict[str, dict]:
        """Return all fields of hash *key* as a {field: dict} mapping."""

    @abstractmethod
    def hdel(self, key: str, field: str) -> None:
        """Delete *field* from hash *key* (no-op if absent)."""

    @abstractmethod
    def hkeys(self, key: str) -> list[str]:
        """Return all field names of hash *key* (empty list if key absent)."""

    @abstractmethod
    def hsetnx_json(self, key: str, field: str, value: dict) -> bool:
        """Set *field* in hash *key* to *value* ONLY if the field does not exist.

        Atomic check-and-set: returns True if the field was written (it was
        absent), False if it already existed (value unchanged).  This is the
        cross-process claim primitive for task assignment — one caller wins,
        all concurrent callers lose.

        Redis implementation: ``HSETNX key field JSON(value)``.
        MemoryBackend implementation: check-and-set under the global lock.
        """

    @abstractmethod
    def hlen(self, key: str) -> int:
        """Return the number of fields in hash *key* (0 if absent).

        Redis implementation: ``HLEN key``.
        MemoryBackend implementation: ``len`` of the inner dict.
        """

    # -- atomic sub-document mutation (B6, 2026-07-02) -----------------------
    #
    # ``hget_json`` + mutate-in-Python + ``hset_json`` (the pattern used by
    # ``worker_registry._WorkerEntryView`` prior to this change) is a classic
    # read-modify-write race: two callers can both read the same blob, apply
    # their own change, and the second ``hset_json`` silently clobbers the
    # first caller's write.  The four methods below perform GET+mutate+SET as
    # ONE atomic operation on the backend (a single Lua script on Redis —
    # Redis executes scripts atomically, so concurrent callers serialize;
    # a ``threading.Lock``-guarded critical section on MemoryBackend, which
    # is single-process so a lock is sufficient).
    #
    # CAVEAT — Redis backend only, verified empirically against Redis 7.0.15 /
    # its embedded lua-cjson (see ``tests/test_shared_state.py::TestAtomicSubDocument``
    # for the reproduction): every one of these calls round-trips the ENTIRE
    # blob through ``cjson.decode``/``cjson.encode`` inside the Lua script
    # (not just the field(s) being touched), which has two known quirks:
    #   1. A whole-number float (e.g. ``0.0``) loses its decimal point on
    #      re-encode (``0.0`` -> JSON text ``"0"``), so it comes back to
    #      Python as ``int`` rather than ``float``.  Numerically identical
    #      (``0 == 0.0``) and harmless for arithmetic — MemoryBackend does
    #      NOT have this quirk (plain ``json.dumps``/``json.loads``, which
    #      preserve the float/int distinction), so it is Redis-backend-only
    #      type drift, not a value bug.
    #   2. An EMPTY list (``[]``) round-trips as an empty dict (``{}``) — a
    #      real, silent corruption (a Lua table can't distinguish "empty
    #      array" from "empty object" once round-tripped through cjson
    #      unless the *exact same* table object survives untouched, which a
    #      merge-and-re-encode does not guarantee).  Guarded against: the
    #      Redis implementation raises ``ValueError`` if any value passed to
    #      these methods is (or contains) an empty list — callers must not
    #      store empty-list values through this path.  Non-empty lists
    #      round-trip correctly (arrays are unambiguous once they have at
    #      least one element).  No worker-registry field is ever an empty
    #      list today (verified against every ``workers_state[...]`` write
    #      site in ``exonit/server.py``); the guard exists so a future field
    #      that violates this fails loudly at the call site instead of
    #      silently corrupting the registry.

    @abstractmethod
    def hupdate_json(self, key: str, field: str, updates: dict) -> dict:
        """Atomically merge *updates* into the dict at hash *key* / *field*.

        Equivalent to::

            entry = hget_json(key, field) or {}
            entry.update(updates)
            hset_json(key, field, entry)

        but performed as a single atomic GET+merge+SET.  Returns the merged
        entry (post-update).  A missing field is treated as ``{}``.
        """

    @abstractmethod
    def hincr_json_field(
        self, key: str, field: str, subfield: str, delta: float
    ) -> float:
        """Atomically add *delta* to the numeric *subfield* of the dict at
        hash *key* / *field*.  A missing or non-numeric *subfield* is
        treated as 0 before adding.  Stores the result back and returns the
        new value.  Fixes the classic ``entry[k] = entry.get(k, 0) + delta``
        two-call race (read and write are two SEPARATE backend round trips
        even when each one is individually atomic).
        """

    @abstractmethod
    def hdelete_json_field(self, key: str, field: str, subfield: str) -> dict:
        """Atomically remove *subfield* from the dict at hash *key* / *field*.

        Returns the entry as it was BEFORE the deletion (so callers can
        check ``subfield in previous`` for KeyError semantics and recover
        the removed value for ``pop()``).  No-op (previous == {}) if the
        hash field or the subfield was already absent.
        """

    @abstractmethod
    def hsetdefault_json_field(
        self, key: str, field: str, subfield: str, default: Any
    ) -> Any:
        """Atomically set *subfield* to *default* ONLY if absent, and return
        the value now stored (either the pre-existing value, unchanged, or
        *default* if it was just written).  Mirrors ``dict.setdefault``.
        """


# ---------------------------------------------------------------------------
# In-memory backend (thread-safe, TTL lazy-expired)
# ---------------------------------------------------------------------------


class MemoryBackend(SharedStateBackend):
    """Dictionary-backed implementation for development and single-process tests.

    Thread-safe via a single ``threading.Lock``.  TTL expiry is *lazy*: expired
    entries are evicted on the next access, not by a background sweep.

    Parameters
    ----------
    clock:
        Callable returning the current time as a float (default: ``time.monotonic``).
        Inject a fake clock in tests to make TTL behaviour deterministic.
    """

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._lock = threading.Lock()
        # scalars: key -> (str_value, expiry_float_or_None)
        self._scalars: dict[str, tuple[str, float | None]] = {}
        # hashes: key -> {field: str(JSON)}
        self._hashes: dict[str, dict[str, str]] = {}

    # -- helpers -------------------------------------------------------------

    def _is_expired(self, expiry: float | None) -> bool:
        return expiry is not None and self._clock() >= expiry

    def _expiry(self, ttl: float | None) -> float | None:
        return None if ttl is None else self._clock() + ttl

    # -- SharedStateBackend --------------------------------------------------

    def ping(self) -> bool:
        return True

    def get(self, key: str) -> str | None:
        with self._lock:
            entry = self._scalars.get(key)
            if entry is None:
                return None
            value, expiry = entry
            if self._is_expired(expiry):
                del self._scalars[key]
                return None
            return value

    def set(self, key: str, value: str, *, ttl: float | None = None) -> None:
        with self._lock:
            self._scalars[key] = (value, self._expiry(ttl))

    def delete(self, key: str) -> None:
        with self._lock:
            self._scalars.pop(key, None)

    def set_if_absent(
        self, key: str, value: str, *, ttl: float | None = None
    ) -> bool:
        with self._lock:
            entry = self._scalars.get(key)
            if entry is not None:
                _, expiry = entry
                if not self._is_expired(expiry):
                    return False
                # Expired — treat as absent.
            self._scalars[key] = (value, self._expiry(ttl))
            return True

    def get_int(self, key: str, default: int = 0) -> int:
        raw = self.get(key)
        if raw is None:
            return default
        return int(raw)

    def set_int(self, key: str, value: int) -> None:
        self.set(key, str(value))

    def incr(self, key: str, amount: int = 1) -> int:
        with self._lock:
            entry = self._scalars.get(key)
            if entry is None or self._is_expired(entry[1]):
                current = 0
            else:
                current = int(entry[0])
            new_val = current + amount
            # Keep existing TTL when incrementing an existing key.
            existing_expiry = entry[1] if (entry and not self._is_expired(entry[1])) else None
            self._scalars[key] = (str(new_val), existing_expiry)
            return new_val

    def hset_json(self, key: str, field: str, value: dict) -> None:
        with self._lock:
            if key not in self._hashes:
                self._hashes[key] = {}
            self._hashes[key][field] = json.dumps(value)

    def hget_json(self, key: str, field: str) -> dict | None:
        with self._lock:
            bucket = self._hashes.get(key)
            if bucket is None:
                return None
            raw = bucket.get(field)
            if raw is None:
                return None
            return json.loads(raw)

    def hgetall_json(self, key: str) -> dict[str, dict]:
        with self._lock:
            bucket = self._hashes.get(key, {})
            return {f: json.loads(v) for f, v in bucket.items()}

    def hdel(self, key: str, field: str) -> None:
        with self._lock:
            bucket = self._hashes.get(key)
            if bucket is not None:
                bucket.pop(field, None)

    def hkeys(self, key: str) -> list[str]:
        with self._lock:
            return list(self._hashes.get(key, {}).keys())

    def hsetnx_json(self, key: str, field: str, value: dict) -> bool:
        with self._lock:
            bucket = self._hashes.setdefault(key, {})
            if field in bucket:
                return False
            bucket[field] = json.dumps(value)
            return True

    def hlen(self, key: str) -> int:
        with self._lock:
            return len(self._hashes.get(key, {}))

    # -- atomic sub-document mutation (B6) -----------------------------------
    #
    # Single-process backend: the module-level ``threading.Lock`` already
    # serialises every access, so a plain read-modify-write under the lock
    # IS atomic here (no cjson involved — plain ``json.dumps``/``json.loads``
    # round-trip exactly, so none of the Redis-backend caveats documented on
    # the abstract methods apply to this implementation).

    def hupdate_json(self, key: str, field: str, updates: dict) -> dict:
        with self._lock:
            bucket = self._hashes.setdefault(key, {})
            entry = json.loads(bucket[field]) if field in bucket else {}
            entry.update(updates)
            bucket[field] = json.dumps(entry)
            return entry

    def hincr_json_field(
        self, key: str, field: str, subfield: str, delta: float
    ) -> float:
        with self._lock:
            bucket = self._hashes.setdefault(key, {})
            entry = json.loads(bucket[field]) if field in bucket else {}
            current = entry.get(subfield)
            if not isinstance(current, (int, float)) or isinstance(current, bool):
                current = 0
            new_val = current + delta
            entry[subfield] = new_val
            bucket[field] = json.dumps(entry)
            return new_val

    def hdelete_json_field(self, key: str, field: str, subfield: str) -> dict:
        with self._lock:
            bucket = self._hashes.setdefault(key, {})
            entry = json.loads(bucket[field]) if field in bucket else {}
            previous = dict(entry)
            entry.pop(subfield, None)
            bucket[field] = json.dumps(entry)
            return previous

    def hsetdefault_json_field(
        self, key: str, field: str, subfield: str, default: Any
    ) -> Any:
        with self._lock:
            bucket = self._hashes.setdefault(key, {})
            entry = json.loads(bucket[field]) if field in bucket else {}
            if subfield not in entry:
                entry[subfield] = default
                bucket[field] = json.dumps(entry)
                return default
            return entry[subfield]


# ---------------------------------------------------------------------------
# Redis backend
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Atomic sub-document mutation — Lua script (B6, 2026-07-02)
# ---------------------------------------------------------------------------
#
# One script covers hupdate_json / hincr_json_field / hdelete_json_field /
# hsetdefault_json_field: each Python method just picks which of the four
# ARGV maps/lists is non-empty for its use case, so every call is a single
# EVALSHA round trip (GET, decode, mutate, encode, SET, all inside Redis'
# single-threaded script execution — concurrent callers serialise, exactly
# like MULTI/EXEC would, without the WATCH-retry-loop complexity that would
# be needed to keep a transaction-based approach live under contention).
#
# Application order when a caller combines more than one kind of update in
# a single call (no current call site does — documented for completeness):
# increments, then sets, then set-if-absent, then deletes.
_HMUTATE_LUA = r"""
local raw = redis.call('HGET', KEYS[1], ARGV[1])
local entry
if raw then
    entry = cjson.decode(raw)
else
    entry = {}
end

local incr_updates = cjson.decode(ARGV[2])
for k, delta in pairs(incr_updates) do
    local cur = entry[k]
    if type(cur) ~= 'number' then
        cur = 0
    end
    entry[k] = cur + delta
end

local set_updates = cjson.decode(ARGV[3])
for k, v in pairs(set_updates) do
    entry[k] = v
end

local setnx_updates = cjson.decode(ARGV[4])
for k, v in pairs(setnx_updates) do
    if entry[k] == nil then
        entry[k] = v
    end
end

local delete_keys = cjson.decode(ARGV[5])
for _, k in ipairs(delete_keys) do
    entry[k] = nil
end

local final_json = cjson.encode(entry)
redis.call('HSET', KEYS[1], ARGV[1], final_json)

local previous_json = raw or '{}'
return {previous_json, final_json}
"""


def _reject_empty_list_values(value: Any, *, path: str = "") -> None:
    """Raise ``ValueError`` if *value* is, or (recursively) contains, an
    empty list.

    Redis' embedded lua-cjson cannot distinguish an empty JSON array from an
    empty JSON object once it has been decoded into a Lua table and
    re-encoded (verified empirically against Redis 7.0.15: ``[]`` round-trips
    as ``{}``) — see the caveat on ``SharedStateBackend.hupdate_json`` et al.
    Every ``_HMUTATE_LUA`` call fully decodes+re-encodes the ENTIRE resident
    blob, not just the field(s) that call's own arguments mention — so this
    guard must run at every path that can introduce a value into that blob,
    not only the four atomic methods' own incoming arguments. Two call
    sites, both required (review finding, B6 follow-up 2026-07-02):
    ``RedisBackend._hmutate`` (guards the incr/set/setnx payloads of a given
    atomic call) AND ``worker_registry.WorkerRegistry.__setitem__`` (guards
    the whole-entry overwrite used by new-worker registration — the path
    that can SEED an empty-list field in the first place; without a guard
    there too, such a field would survive that write untouched — it doesn't
    go through this Lua script — only to be silently corrupted by the NEXT,
    unrelated, atomic call touching that same entry). Non-empty lists are
    unaffected (arrays with at least one element round-trip correctly) and
    are not rejected. Deliberately NOT applied inside ``hset_json`` itself,
    which is shared by unrelated hash keys that legitimately store empty
    lists (e.g. pipeline state's ``log: {"lines": []}``, never touched by
    ``_HMUTATE_LUA``) — a blanket guard there would break those call sites
    for a corruption mode that only exists for the sub-document mutation
    path.
    """
    if isinstance(value, list):
        if len(value) == 0:
            raise ValueError(
                f"shared_state: refusing to store an empty list at "
                f"{path or '<value>'!r} — Redis' embedded cjson corrupts "
                "empty JSON arrays into empty JSON objects on the atomic "
                "sub-document mutation path (hupdate_json/hincr_json_field/"
                "hdelete_json_field/hsetdefault_json_field). Use a "
                "non-empty sentinel or None instead."
            )
        for i, item in enumerate(value):
            _reject_empty_list_values(item, path=f"{path}[{i}]")
    elif isinstance(value, dict):
        for k, v in value.items():
            _reject_empty_list_values(v, path=f"{path}.{k}" if path else str(k))


class RedisBackend(SharedStateBackend):
    """Redis-backed implementation using ``redis-py`` with ``decode_responses=True``.

    Uses a ``BlockingConnectionPool`` so that threads wait for a free
    connection instead of raising ``MaxConnectionsError`` when the pool is
    exhausted.  Under the default server workload (many background threads
    polling via ``DistributedSemaphore``) a regular ``ConnectionPool`` with
    max_connections=100 was exhausted in production, producing 245×
    ``MaxConnectionsError``.

    Parameters
    ----------
    url:
        Redis URL, e.g. ``"redis://localhost:6379/0"``.

    Environment variables
    ---------------------
    EXONIT_REDIS_MAX_CONNECTIONS:
        Maximum number of connections in the pool.  Parsed as int; falls back
        to 256 if absent or invalid.
    EXONIT_REDIS_POOL_TIMEOUT:
        Seconds to wait for a free connection before raising
        ``redis.exceptions.ConnectionError``.  Parsed as float; falls back to
        30 s if absent or invalid.
    """

    _DEFAULT_MAX_CONNECTIONS = 256
    _DEFAULT_POOL_TIMEOUT = 30.0

    def __init__(self, url: str = DEFAULT_REDIS_URL) -> None:
        import redis as redis_lib  # local import keeps the module importable without redis

        try:
            max_connections = int(os.environ.get("EXONIT_REDIS_MAX_CONNECTIONS", ""))
            if max_connections <= 0:
                raise ValueError
        except (ValueError, TypeError):
            max_connections = self._DEFAULT_MAX_CONNECTIONS

        try:
            pool_timeout = float(os.environ.get("EXONIT_REDIS_POOL_TIMEOUT", ""))
            if pool_timeout <= 0:
                raise ValueError
        except (ValueError, TypeError):
            pool_timeout = self._DEFAULT_POOL_TIMEOUT

        pool = redis_lib.BlockingConnectionPool.from_url(
            url,
            decode_responses=True,
            max_connections=max_connections,
            timeout=pool_timeout,
        )
        self._client = redis_lib.Redis(connection_pool=pool)
        # redis-py's Script wrapper caches the SHA and transparently falls
        # back from EVALSHA to EVAL on NOSCRIPT (e.g. after a Redis restart
        # flushed the script cache) — no manual SHA bookkeeping needed here.
        self._hmutate_script = self._client.register_script(_HMUTATE_LUA)
        logger.debug(
            "RedisBackend: BlockingConnectionPool max_connections=%d timeout=%.1fs url=%s",
            max_connections,
            pool_timeout,
            url,
        )

    @property
    def raw_client(self):
        """Raw redis.Redis client for purpose-built structures (streams).

        Work-stream queues (exonit/work_streams.py) need XADD/XREADGROUP,
        which don't fit the generic KV/hash surface of this backend.
        """
        return self._client

    def ping(self) -> bool:
        try:
            return bool(self._client.ping())
        except Exception:
            return False

    def get(self, key: str) -> str | None:
        return self._client.get(key)

    def set(self, key: str, value: str, *, ttl: float | None = None) -> None:
        if ttl is not None:
            px = max(1, int(ttl * 1000))
            self._client.set(key, value, px=px)
        else:
            self._client.set(key, value)

    def delete(self, key: str) -> None:
        self._client.delete(key)

    def set_if_absent(
        self, key: str, value: str, *, ttl: float | None = None
    ) -> bool:
        if ttl is not None:
            px = max(1, int(ttl * 1000))
            result = self._client.set(key, value, nx=True, px=px)
        else:
            result = self._client.set(key, value, nx=True)
        return bool(result)

    def get_int(self, key: str, default: int = 0) -> int:
        raw = self._client.get(key)
        if raw is None:
            return default
        return int(raw)

    def set_int(self, key: str, value: int) -> None:
        self._client.set(key, str(value))

    def incr(self, key: str, amount: int = 1) -> int:
        return int(self._client.incrby(key, amount))

    def hset_json(self, key: str, field: str, value: dict) -> None:
        self._client.hset(key, field, json.dumps(value))

    def hget_json(self, key: str, field: str) -> dict | None:
        raw = self._client.hget(key, field)
        if raw is None:
            return None
        return json.loads(raw)

    def hgetall_json(self, key: str) -> dict[str, dict]:
        raw = self._client.hgetall(key)
        return {f: json.loads(v) for f, v in raw.items()}

    def hdel(self, key: str, field: str) -> None:
        self._client.hdel(key, field)

    def hkeys(self, key: str) -> list[str]:
        return self._client.hkeys(key)

    def hsetnx_json(self, key: str, field: str, value: dict) -> bool:
        result = self._client.hsetnx(key, field, json.dumps(value))
        return bool(result)

    def hlen(self, key: str) -> int:
        return int(self._client.hlen(key))

    # -- atomic sub-document mutation (B6) -----------------------------------

    def _hmutate(
        self,
        key: str,
        field: str,
        *,
        incr_updates: dict | None = None,
        set_updates: dict | None = None,
        setnx_updates: dict | None = None,
        delete_keys: list | None = None,
    ) -> tuple[dict, dict]:
        """Run ``_HMUTATE_LUA`` and return ``(previous_entry, final_entry)``."""
        incr_updates = incr_updates or {}
        set_updates = set_updates or {}
        setnx_updates = setnx_updates or {}
        delete_keys = delete_keys or []
        _reject_empty_list_values(incr_updates)
        _reject_empty_list_values(set_updates)
        _reject_empty_list_values(setnx_updates)
        previous_json, final_json = self._hmutate_script(
            keys=[key],
            args=[
                field,
                json.dumps(incr_updates),
                json.dumps(set_updates),
                json.dumps(setnx_updates),
                json.dumps(delete_keys),
            ],
        )
        return json.loads(previous_json), json.loads(final_json)

    def hupdate_json(self, key: str, field: str, updates: dict) -> dict:
        _, final_entry = self._hmutate(key, field, set_updates=updates)
        return final_entry

    def hincr_json_field(
        self, key: str, field: str, subfield: str, delta: float
    ) -> float:
        _, final_entry = self._hmutate(
            key, field, incr_updates={subfield: delta}
        )
        return final_entry[subfield]

    def hdelete_json_field(self, key: str, field: str, subfield: str) -> dict:
        previous_entry, _ = self._hmutate(key, field, delete_keys=[subfield])
        return previous_entry

    def hsetdefault_json_field(
        self, key: str, field: str, subfield: str, default: Any
    ) -> Any:
        _, final_entry = self._hmutate(
            key, field, setnx_updates={subfield: default}
        )
        return final_entry[subfield]


# ---------------------------------------------------------------------------
# Singleton factory
# ---------------------------------------------------------------------------

_state_lock = threading.Lock()
_shared_state_instance: SharedStateBackend | None = None


def _read_server_workers() -> int:
    """Return the value of ``EXONIT_SERVER_WORKERS`` as an int (default 1).

    Returns 1 on any parse error so callers can safely compare ``>1``.
    """
    raw = os.environ.get("EXONIT_SERVER_WORKERS", "1")
    try:
        return int(raw)
    except (ValueError, TypeError):
        return 1


def get_shared_state(
    url: str | None = None,
    *,
    force_memory: bool = False,
) -> SharedStateBackend:
    """Return the process-level singleton ``SharedStateBackend``.

    Resolution order (single-process, ``EXONIT_SERVER_WORKERS`` absent or 1):
    1. ``force_memory=True`` → always ``MemoryBackend``.
    2. ``EXONIT_SHARED_STATE_BACKEND=memory`` env var → ``MemoryBackend``.
    3. Otherwise: try ``RedisBackend`` at *url* (falling back to
       ``EXONIT_REDIS_URL`` or ``DEFAULT_REDIS_URL``).  If Redis is
       unreachable, log a WARNING and return ``MemoryBackend``.

    Multi-process guard (``EXONIT_SERVER_WORKERS`` > 1, set by server.main):
        In-memory backend is **forbidden** — it would cause split-brain (each
        process maintains independent state → double task assignment, N
        concurrent ExoMiner Docker containers → OOM, broken semaphores).

        * ``force_memory=True`` or ``EXONIT_SHARED_STATE_BACKEND=memory`` →
          ``RuntimeError``.
        * Redis unreachable → ``RuntimeError`` (no silent fallback).
        * Redis reachable → ``RedisBackend`` (normal path).
    """
    global _shared_state_instance

    with _state_lock:
        if _shared_state_instance is not None:
            return _shared_state_instance

        num_workers = _read_server_workers()
        multi_process = num_workers > 1

        if multi_process:
            # Hard-assert: in-memory backend is unsafe in multi-process mode.
            if force_memory or os.environ.get("EXONIT_SHARED_STATE_BACKEND") == "memory":
                raise RuntimeError(
                    f"multi-process server (EXONIT_SERVER_WORKERS={num_workers}) "
                    "requires Redis; in-memory backend is unsafe — each process "
                    "would have isolated state → double task assignment and "
                    f"{num_workers} concurrent ExoMiner Docker containers → OOM. "
                    "Start redis-server and set EXONIT_REDIS_URL (or use the "
                    "default redis://localhost:6379/0)."
                )

            redis_url = (
                url
                or os.environ.get("EXONIT_REDIS_URL")
                or DEFAULT_REDIS_URL
            )
            try:
                backend = RedisBackend(redis_url)
                if not backend.ping():
                    raise ConnectionError(f"Redis ping failed at {redis_url!r}")
                _shared_state_instance = backend
            except Exception as exc:
                raise RuntimeError(
                    f"multi-process server (EXONIT_SERVER_WORKERS={num_workers}) "
                    f"requires Redis, but connection to {redis_url!r} failed: {exc}. "
                    "Start redis-server (apt install redis-server) or set "
                    "EXONIT_REDIS_URL to a reachable Redis instance."
                ) from exc

            return _shared_state_instance

        # Single-process path — pre-A3 behaviour preserved.
        if force_memory or os.environ.get("EXONIT_SHARED_STATE_BACKEND") == "memory":
            _shared_state_instance = MemoryBackend()
            return _shared_state_instance

        redis_url = (
            url
            or os.environ.get("EXONIT_REDIS_URL")
            or DEFAULT_REDIS_URL
        )
        try:
            backend = RedisBackend(redis_url)
            if not backend.ping():
                raise ConnectionError(f"Redis ping failed at {redis_url}")
            _shared_state_instance = backend
        except Exception as exc:
            logger.warning(
                "shared_state: Redis unavailable (%s) — falling back to "
                "MemoryBackend.  This is only safe for single-process "
                "deployments.  Set EXONIT_REDIS_URL or start redis-server.",
                exc,
            )
            _shared_state_instance = MemoryBackend()

        return _shared_state_instance


def reset_shared_state() -> None:
    """Clear the singleton so the next call to ``get_shared_state`` creates a fresh one.

    Intended for use in tests only.
    """
    global _shared_state_instance
    with _state_lock:
        _shared_state_instance = None
