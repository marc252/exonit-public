"""Dict-compatible proxy for the workers_state registry.

``WorkerRegistry`` wraps a ``SharedStateBackend`` (Redis or in-memory) and
exposes the same interface as the plain ``dict`` it replaces, so every existing
call-site in ``server.py`` and the tests works without modification.

The registry is backed by a Redis hash (or MemoryBackend hash): each worker_id
is a *field* and its entry is stored as a JSON string.

NOTE (B6, 2026-07-02): every mutating method on ``_WorkerEntryView``
(``__setitem__``, ``__delitem__``, ``update``, ``pop``, ``setdefault``) now
goes through ``SharedStateBackend``'s atomic sub-document primitives
(``hupdate_json`` / ``hincr_json_field`` / ``hdelete_json_field`` /
``hsetdefault_json_field`` — see ``exonit/shared_state.py``) instead of a
hand-rolled ``hget_json`` + mutate-in-Python + ``hset_json`` cycle.  On the
Redis backend each of those is a single Lua script (GET+mutate+SET executed
atomically by Redis); on MemoryBackend each is a lock-guarded critical
section (single-process, so a lock is sufficient).  Concurrent writers to
the SAME worker entry (e.g. a heartbeat and a task-result callback landing
at the same time, or two server processes under A3) now serialise instead
of racing to silently clobber each other's field.

NOTE (B6 follow-up, 2026-07-02, review CHANGES_REQUESTED): the atomic
primitives' empty-list guard only inspected the dict passed to *that* call.
On the Redis backend, ``_HMUTATE_LUA`` round-trips the ENTIRE resident entry
through cjson on every call, not just the field(s) it touches — so an empty
list seeded into a worker entry via ``WorkerRegistry.__setitem__`` (the
whole-entry overwrite used by new-worker registration,
``server.py:worker_register``) would survive that write untouched, then get
silently corrupted into ``{}`` by the very next unrelated
``hupdate_json``/``increment``/``pop``/``setdefault`` call for that worker.
Reproduced empirically against Redis db=15 by the reviewer. Fixed by running
``_reject_empty_list_values`` at EVERY mutating entry point of this module
— ``WorkerRegistry.__setitem__`` (the whole-entry overwrite path that seeded
the reviewer's repro) AND ``_WorkerEntryView.__setitem__`` / ``.update()`` /
``.setdefault()`` (the per-field paths — the Redis backend's own
``hupdate_json``/``hsetdefault_json_field`` already rejected these for its
own cjson-corruption reasons, but MemoryBackend's did not, since it has no
cjson quirk of its own; guarding here too, unconditionally on both backends,
means the registry API's invariant — "no worker entry can ever contain an
empty list, checked by anything that goes through ``WorkerRegistry``" —
does not depend on which backend happens to be plugged in, and a test
written against ``MemoryBackend`` alone still catches a violation). Not a
blanket guard on ``hset_json``/``hupdate_json`` themselves at the
``shared_state`` layer (those stay backend-native — Redis guards for its own
cjson reasons, MemoryBackend does not need to): other hash keys, e.g. the
pipeline state's ``log: {"lines": []}``, legitimately store empty lists
through those generic primitives and must not be broken by a guard that
isn't worker-registry-specific.
"""

from __future__ import annotations

import collections.abc
from typing import Any, Iterator

from .shared_state import SharedStateBackend, _reject_empty_list_values


class _WorkerEntryView(collections.abc.MutableMapping):
    """Proxy for a single worker's entry dict stored inside a registry hash.

    Every mutation (``__setitem__``, ``__delitem__``, ``update``, ``pop``,
    ``setdefault``) persists via ONE atomic GET+mutate+SET call on the
    backend (``SharedStateBackend.hupdate_json`` / ``hincr_json_field`` /
    ``hdelete_json_field`` / ``hsetdefault_json_field`` — B6, 2026-07-02).
    Reads (``__getitem__``, ``__contains__``, ``__iter__``, ``__len__``,
    ``get``) still do a plain ``hget_json`` and always hit the backend for
    freshness — read-then-mutate-in-Python is exactly the race this class
    used to have, so no method here reads the whole entry and computes a new
    value locally before writing it back (``increment`` — not part of the
    ``MutableMapping`` protocol but exposed for exactly this reason — is the
    one exception call sites should reach for instead of
    ``view[k] = view.get(k, 0) + delta``, which is still racy even now that
    each individual read and each individual write is atomic on its own:
    the read and the write remain two separate backend round trips).

    This is intentionally NOT cached in-process because heartbeats from multiple
    threads must see each other's writes (and in multi-process mode, different
    OS processes share the same Redis hash).
    """

    def __init__(
        self, backend: SharedStateBackend, hash_key: str, worker_id: str
    ) -> None:
        self._backend = backend
        self._hash_key = hash_key
        self._worker_id = worker_id

    # -- helpers ------------------------------------------------------------------

    def _load(self) -> dict:
        return self._backend.hget_json(self._hash_key, self._worker_id) or {}

    # -- MutableMapping -----------------------------------------------------------

    def __getitem__(self, key: str) -> Any:
        entry = self._load()
        if key not in entry:
            raise KeyError(key)
        return entry[key]

    def __setitem__(self, key: str, value: Any) -> None:
        """Set *key* to *value*.

        Rejects (``ValueError``) an empty-list *value* (or one containing an
        empty list) — same invariant as ``WorkerRegistry.__setitem__``,
        enforced here too and unconditionally on every backend so the
        registry API's guarantee ("no worker entry can ever contain an empty
        list") does not depend on which backend happens to be plugged in.
        The Redis backend's own ``hupdate_json`` already rejects this for
        its own reasons (cjson corruption of the resident blob); this call
        adds the same check at the registry layer for MemoryBackend too,
        which has no cjson quirk of its own but must still uphold the same
        business invariant so a test written against MemoryBackend catches
        a violation without needing a live Redis.
        """
        _reject_empty_list_values(value, path=key)
        self._backend.hupdate_json(self._hash_key, self._worker_id, {key: value})

    def __delitem__(self, key: str) -> None:
        previous = self._backend.hdelete_json_field(
            self._hash_key, self._worker_id, key)
        if key not in previous:
            raise KeyError(key)

    def __contains__(self, key: object) -> bool:
        return key in self._load()

    def __iter__(self) -> Iterator[str]:
        return iter(self._load())

    def __len__(self) -> int:
        return len(self._load())

    # -- dict-like convenience methods --------------------------------------------

    def get(self, key: str, default: Any = None) -> Any:
        return self._load().get(key, default)

    def pop(self, key: str, *args: Any) -> Any:
        """Remove *key* and return its value.

        Raises ``KeyError`` if *key* absent and no default given (dict semantics).
        """
        previous = self._backend.hdelete_json_field(
            self._hash_key, self._worker_id, key)
        if key in previous:
            return previous[key]
        if args:
            return args[0]
        raise KeyError(key)

    def update(self, other: Any = None, **kwargs: Any) -> None:
        updates: dict = {}
        if other is not None:
            if hasattr(other, "keys"):
                for k in other:
                    updates[k] = other[k]
            else:
                for k, v in other:
                    updates[k] = v
        updates.update(kwargs)
        if not updates:
            return
        _reject_empty_list_values(updates)
        self._backend.hupdate_json(self._hash_key, self._worker_id, updates)

    def setdefault(self, key: str, default: Any = None) -> Any:
        _reject_empty_list_values(default, path=key)
        return self._backend.hsetdefault_json_field(
            self._hash_key, self._worker_id, key, default)

    def increment(self, key: str, delta: float = 1) -> float:
        """Atomically add *delta* to numeric field *key* and return the new
        value.  NOT part of ``MutableMapping`` — replaces the racy
        ``view[key] = view.get(key, 0) + delta`` pattern (two separate
        backend round trips: a read, then a write, with no atomicity between
        them even though each one is individually atomic).  A missing or
        non-numeric *key* is treated as 0 before adding.
        """
        return self._backend.hincr_json_field(
            self._hash_key, self._worker_id, key, delta)

    def __repr__(self) -> str:  # pragma: no cover
        return f"_WorkerEntryView(worker_id={self._worker_id!r}, entry={self._load()!r})"


class WorkerRegistry(collections.abc.MutableMapping):
    """Dict-compatible registry of worker entries backed by a ``SharedStateBackend``.

    The registry stores each worker_id as a *field* in a single backend hash
    (``key`` parameter, default ``"workers:registry"``).  The value is the JSON
    of the worker entry dict.

    Usage is identical to a plain ``dict``::

        registry["wid"]["status"] = "idle"     # persists immediately
        registry["wid"].update({...})           # persists immediately
        del registry["wid"]                     # removes from backend
        "wid" in registry                       # backend lookup
        for wid, entry in registry.items():     # snapshot (each entry is a dict)

    **Snapshot semantics for iteration:**
    ``keys()``, ``values()``, and ``items()`` snapshot the backend in a single
    ``hgetall_json`` call.  The *values* in those snapshots are plain dicts —
    mutating them does NOT persist.  Use ``registry[wid][field] = v`` or
    ``registry[wid] = entry`` to persist.  This is documented here so callers
    that iterate-and-mutate know to convert their loops (two such loops exist in
    ``server.py`` and are the only call-sites changed in A1).

    **NOT a snapshot for direct access:**
    ``registry[wid]`` returns a ``_WorkerEntryView`` that persists writes
    immediately — this covers all the heartbeat / result / register paths
    that access a specific worker_id.

    **TTL:**
    No TTL is applied to the hash.  Workers are considered offline based on
    ``last_heartbeat`` age, not backend expiry; this matches the original dict
    behaviour exactly.
    """

    def __init__(
        self, backend: SharedStateBackend, key: str = "workers:registry"
    ) -> None:
        self._backend = backend
        self._key = key

    # -- MutableMapping -----------------------------------------------------------

    def __getitem__(self, worker_id: str) -> _WorkerEntryView:
        """Return a live view of *worker_id*'s entry.

        Raises ``KeyError`` if the worker is not registered.
        """
        if worker_id not in self:
            raise KeyError(worker_id)
        return _WorkerEntryView(self._backend, self._key, worker_id)

    def __setitem__(self, worker_id: str, entry: dict) -> None:
        """Store (or overwrite) *worker_id*'s entry as a copy of *entry*.

        Rejects (``ValueError``) any *entry* that is, or (recursively)
        contains, an empty list — see the module docstring's B6-follow-up
        note. This write itself goes via plain ``hset_json`` (no cjson
        round-trip, so an empty list would survive THIS call untouched on
        the Redis backend), but the guard runs anyway, unconditionally on
        every backend: this is the only path that can SEED such a field into
        a worker entry in the first place (new-worker registration), and
        without the guard here the corruption would only surface later, on
        the next unrelated atomic-primitive call for this worker — silently,
        with no error at the point the bad value was actually introduced.
        Applying it on MemoryBackend too (which has no cjson quirk at all)
        keeps the two backends' behaviour identical, so a test written
        against ``MemoryBackend`` (no local Redis needed) still catches a
        future worker field that violates this invariant.
        """
        entry = dict(entry)
        _reject_empty_list_values(entry)
        self._backend.hset_json(self._key, worker_id, entry)

    def __delitem__(self, worker_id: str) -> None:
        """Delete *worker_id* from the registry.

        Raises ``KeyError`` if the worker is not present (dict semantics).
        """
        if worker_id not in self:
            raise KeyError(worker_id)
        self._backend.hdel(self._key, worker_id)

    def __contains__(self, worker_id: object) -> bool:
        return worker_id in self._backend.hkeys(self._key)

    def __iter__(self) -> Iterator[str]:
        return iter(self._backend.hkeys(self._key))

    def __len__(self) -> int:
        return len(self._backend.hkeys(self._key))

    # -- snapshot accessors -------------------------------------------------------

    def keys(self):  # type: ignore[override]
        """Return a list snapshot of all registered worker_ids."""
        return list(self._backend.hkeys(self._key))

    def values(self):  # type: ignore[override]
        """Return a snapshot list of all worker entry dicts (plain dicts).

        WARNING: mutating entries in this snapshot does NOT persist.
        Use ``registry[wid][field] = v`` for persistent per-field updates.
        """
        return list(self._backend.hgetall_json(self._key).values())

    def items(self):  # type: ignore[override]
        """Return a snapshot list of ``(worker_id, entry_dict)`` pairs.

        WARNING: mutating the entry dicts in this snapshot does NOT persist.
        Use ``registry[wid][field] = v`` for persistent per-field updates.
        """
        return list(self._backend.hgetall_json(self._key).items())

    # -- other dict methods -------------------------------------------------------

    def get(self, worker_id: str, default: Any = None) -> Any:
        """Return the entry dict for *worker_id*, or *default* if absent.

        Returns a plain dict snapshot (not a ``_WorkerEntryView``).  For
        read-only callers that need a fallback (e.g. ``.get(wid, {})``) this
        is sufficient and cheaper than constructing a view.
        """
        return self._backend.hget_json(self._key, worker_id) or default

    def clear(self) -> None:
        """Remove all entries from the registry (used in tests and admin resets)."""
        for wid in self._backend.hkeys(self._key):
            self._backend.hdel(self._key, wid)

    def __repr__(self) -> str:  # pragma: no cover
        snapshot = self._backend.hgetall_json(self._key)
        return f"WorkerRegistry(key={self._key!r}, workers={list(snapshot.keys())!r})"
