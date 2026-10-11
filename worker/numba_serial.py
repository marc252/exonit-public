"""Serialize entry into numba ``parallel=True`` regions on platforms whose only
available numba threading layer is the non-thread-safe ``workqueue``.

Apple Silicon (macOS arm64) ships numba with only the ``workqueue`` layer — the
thread-safe ``tbb``/``omp`` layers have no arm64 wheels (``pip install tbb`` /
``intel-openmp`` resolve to nothing on arm64). When two ``parallel=True`` regions
are entered concurrently from different threads — e.g. the background injection
BLS (``bls_fast_batch``) and a main-thread FPP (``_batch_transit_lnL``) — the
workqueue layer aborts the whole process with::

    Numba workqueue threading layer is terminating: Concurrent access has been detected.

A single process-wide lock, shared across every numba parallel entry point,
serializes those entries and prevents the crash. The regions competed for the
same CPU cores anyway, so the wall-clock cost on a CPU-bound Mac is small.

No-op everywhere else: on Linux (``omp``) and Windows the layer is thread-safe,
and the GPU paths don't use numba parallel, so the guard is a null context with
zero overhead.
"""
from __future__ import annotations

import contextlib
import sys
import threading

# Apple Silicon → only the non-thread-safe workqueue layer is available.
_SERIALIZE = sys.platform == "darwin"
# Reentrant: still blocks *other* threads while held (the cross-thread crash we
# prevent), but a single thread may re-enter without deadlocking if guarded
# regions ever nest.
_LOCK = threading.RLock()


def numba_parallel_guard():
    """Context manager guarding entry into a numba parallel region.

    On Apple Silicon returns a shared process-wide lock (serializes concurrent
    parallel regions across threads); elsewhere a no-op null context.
    """
    return _LOCK if _SERIALIZE else contextlib.nullcontext()
