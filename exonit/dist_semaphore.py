"""Distributed bounded semaphore backed by a SharedStateBackend.

This module provides :class:`DistributedSemaphore`, a cross-process,
bounded counting semaphore implemented via *lease slots* stored in a
:class:`~exonit.shared_state.SharedStateBackend` (Redis or in-memory
fallback).

Purpose
-------
In a multi-process ``uvicorn --workers N`` deployment, each OS process has
its own memory — ``threading.BoundedSemaphore`` is per-process, so N
processes each holding limit=1 would allow N concurrent Docker containers
simultaneously.  :class:`DistributedSemaphore` coordinates across processes
through a shared backend so the global concurrency cap is always respected.

Design
------
Slots are Redis keys ``sem:<name>:slot:<i>`` for ``i`` in
``range(max(1, limit))``.  Each slot holds a unique *token* set with a
short TTL that is continuously renewed by a daemon thread ("renewer") while
the holder is alive.

Invariant (critical safety property)
-------------------------------------
At any point, at most ``limit`` distinct processes/threads hold a slot
simultaneously.  Mode of failure for a *hung* holder: the renewer thread
stops renewing (e.g. process killed) → the slot key expires after
``lease_ttl`` seconds.  The slot is then naturally available for
re-acquisition.  This means a hung holder retains exclusion for up to
``lease_ttl`` seconds — NOT that another holder is admitted (no
over-admission).

Usage
-----
DistributedSemaphore is a context manager with identical protocol to
``threading.BoundedSemaphore``::

    sem = DistributedSemaphore(backend, "exominer_docker", limit=1)
    with sem:
        ...  # at most 1 holder across all processes

Re-entrant use is not supported (matches current ExoNit usage).
"""

from __future__ import annotations

import logging
import secrets
import threading
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from exonit.shared_state import SharedStateBackend

logger = logging.getLogger(__name__)

__all__ = ["DistributedSemaphore"]


class DistributedSemaphore:
    """Cross-process bounded counting semaphore via lease slots.

    Parameters
    ----------
    backend:
        A :class:`~exonit.shared_state.SharedStateBackend` instance
        (``RedisBackend`` for true cross-process coordination;
        ``MemoryBackend`` for single-process or testing).
    name:
        Logical semaphore name.  Slot keys are
        ``sem:<name>:slot:<i>`` for ``i`` in ``range(max(1, limit))``.
    limit:
        Maximum number of simultaneous holders.  Must be >= 1.
    lease_ttl:
        Time-to-live in seconds for each slot key.  The renewer thread
        extends the lease every ``lease_ttl / 3`` seconds, so a live
        holder will never lose its slot.  A dead process loses its slot
        after at most ``lease_ttl`` seconds.
    poll_interval:
        Seconds to sleep between retry attempts during blocking
        ``acquire``.  Lower values reduce latency at the cost of more
        backend round-trips.
    renew:
        If True (default), start a background renewer thread on acquire.
        Set to False to disable renewal (used in tests to simulate a dead
        process / crashed holder without actually killing a thread).
    """

    def __init__(
        self,
        backend: "SharedStateBackend",
        name: str,
        limit: int,
        *,
        lease_ttl: float = 120.0,
        poll_interval: float = 0.5,
        renew: bool = True,
    ) -> None:
        if limit < 1:
            raise ValueError(f"limit must be >= 1, got {limit!r}")
        self._backend = backend
        self._name = name
        self._limit = limit
        self._lease_ttl = lease_ttl
        self._poll_interval = poll_interval
        self._renew = renew

        # Slot keys, indexed 0..limit-1.
        self._slot_keys: list[str] = [
            f"sem:{name}:slot:{i}" for i in range(limit)
        ]

        # Per-thread acquisition state.  Each thread that calls acquire()
        # stores its own slot_key, token, stop_event, and renewer thread here.
        # Using threading.local() prevents concurrent threads that share this
        # singleton instance (e.g. spoc_check with limit=4) from clobbering
        # each other's state when they acquire/release simultaneously.
        self._local = threading.local()

    # ------------------------------------------------------------------
    # Public acquire / release / context-manager API
    # ------------------------------------------------------------------

    def acquire(
        self,
        *,
        blocking: bool = True,
        timeout: float | None = None,
    ) -> bool:
        """Try to acquire a slot.

        Parameters
        ----------
        blocking:
            If False, return immediately with False if no slot is free.
        timeout:
            Maximum wall-clock seconds to wait (None = wait forever, same
            behaviour as ``threading.BoundedSemaphore``).

        Returns
        -------
        bool
            True if a slot was acquired, False if not (only possible when
            ``blocking=False`` or ``timeout`` expires).
        """
        deadline = None if timeout is None else time.monotonic() + timeout

        while True:
            token = secrets.token_hex(16)
            for slot_key in self._slot_keys:
                if self._backend.set_if_absent(
                    slot_key, token, ttl=self._lease_ttl
                ):
                    # Slot acquired — store in the calling thread's local storage.
                    self._local.slot = slot_key
                    self._local.token = token
                    self._local.stop_event = None
                    self._local.renewer = None
                    if self._renew:
                        self._start_renewer(slot_key, token)
                    logger.debug(
                        "dist_sem(%s): acquired slot %s token=%s",
                        self._name, slot_key, token[:8],
                    )
                    return True

            # No slot was free on this pass.
            if not blocking:
                return False

            if deadline is not None and time.monotonic() >= deadline:
                return False

            time.sleep(self._poll_interval)

    def release(self) -> None:
        """Release the currently held slot.

        Check-and-delete: only deletes the slot key if the stored token
        still matches ours.  This prevents a late release from evicting
        a slot that was re-acquired by another holder after our lease
        expired.

        Idempotent — calling ``release`` without a prior ``acquire``
        (or calling it twice) is a no-op.
        """
        # Read and clear this thread's local acquisition state atomically
        # within the thread (no other thread touches our _local namespace).
        slot_key = getattr(self._local, "slot", None)
        token = getattr(self._local, "token", None)
        stop_event = getattr(self._local, "stop_event", None)
        renewer = getattr(self._local, "renewer", None)
        self._local.slot = None
        self._local.token = None
        self._local.stop_event = None
        self._local.renewer = None

        if slot_key is None:
            return  # Nothing to release.

        # Stop the renewer before deleting the slot so it cannot
        # re-create the key after we delete it.
        if stop_event is not None:
            stop_event.set()
        if renewer is not None:
            renewer.join(timeout=max(0.2, self._poll_interval))

        # Check-and-delete: only delete if we still own the slot.
        current = self._backend.get(slot_key)
        if current == token:
            self._backend.delete(slot_key)
            logger.debug(
                "dist_sem(%s): released slot %s token=%s",
                self._name, slot_key, token[:8],
            )
        else:
            # Our token is gone (expired + re-acquired by someone else, or
            # already deleted).  Do not touch the backend.
            logger.debug(
                "dist_sem(%s): release skipped (token mismatch) slot=%s "
                "our=%s current=%s",
                self._name, slot_key, token[:8],
                (current or "")[:8],
            )

    def __enter__(self) -> "DistributedSemaphore":
        """Blocking acquire with no timeout (mirrors BoundedSemaphore)."""
        self.acquire(blocking=True, timeout=None)
        return self

    def __exit__(self, *args: object) -> None:
        self.release()

    # ------------------------------------------------------------------
    # Renewer thread
    # ------------------------------------------------------------------

    def _start_renewer(self, slot_key: str, token: str) -> None:
        """Start a daemon thread that periodically extends the slot lease.

        The stop_event and thread handle are stored in the CALLING thread's
        thread-local storage so that concurrent acquirers on the same instance
        each manage their own renewer independently.
        """
        stop_event = threading.Event()
        # Store stop_event in the calling thread's local storage immediately,
        # before starting the background thread, so release() can find it.
        self._local.stop_event = stop_event
        interval = self._lease_ttl / 3.0

        def _renew() -> None:
            while not stop_event.wait(timeout=interval):
                # Check we still own the slot before renewing.
                current = self._backend.get(slot_key)
                if current != token:
                    # Slot was lost (expired + claimed by another holder,
                    # or deleted by a race in release).  Stop quietly.
                    logger.warning(
                        "dist_sem(%s): renewer lost ownership of %s "
                        "(stored=%s, ours=%s) — stopping renewer",
                        self._name, slot_key,
                        (current or "")[:8], token[:8],
                    )
                    return
                self._backend.set(slot_key, token, ttl=self._lease_ttl)
                logger.debug(
                    "dist_sem(%s): renewed slot %s token=%s",
                    self._name, slot_key, token[:8],
                )

        t = threading.Thread(target=_renew, daemon=True, name=f"sem-renew-{self._name}")
        # Store the thread handle in the calling thread's local storage.
        self._local.renewer = t
        t.start()
