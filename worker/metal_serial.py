"""Serialize all MLX/Metal GPU submissions within the worker process.

MLX (Apple's array framework, used for the fused Metal FPP and injection-
recovery kernels on Apple Silicon) is **not thread-safe**: its command-buffer
/ command-queue state is shared process-wide, and submitting work (a custom
``mx.fast.metal_kernel`` launch, or any lazy op graph forced by ``mx.eval``)
from two threads at once corrupts that shared state and aborts the process::

    -[_MTLCommandBuffer addCompletedHandler:]:1011: failed assertion
      `Completed handler provided after commit call'
    failed assertion _status < MTLCommandBufferStatusCommitted at line 323
      in -[IOGPUMetalCommandBuffer setCurrentCommandEncoder:]

On worker.py's architecture, FPP Monte Carlo (``fpp_metal_kernel`` /
``fpp_metal``) runs on the main thread while injection-recovery
(``bls_metal_batch`` / ``bls_metal``) runs on a background daemon thread
concurrently (see worker.py ~line 2550, "Injection started in background").
Once both submit through MLX, that cross-thread race killed Mac worker
processes hours into production (2026-07-12) — the exact same failure class
as the 2026-04 M1 hang, which was wrongly declared "does not reproduce" on
the strength of a 4-round contention bench (~8s of GPU work): it is a race,
not a deterministic failure, and short benches routinely miss it.

A single process-wide reentrant lock, shared across every MLX submission
site (``fpp_metal_kernel``, ``bls_metal_batch``, and the legacy
``fpp_metal`` / ``bls_metal`` fallback paths — see ``worker/injection.py``
and ``worker/fpp_native.py`` for how each can still be reached), serializes
those submissions and prevents the crash. Mirrors ``numba_serial.py``'s
``numba_parallel_guard()``, which solves the analogous problem for numba's
non-thread-safe ``workqueue`` layer.

Keep every guarded region as small as correctness allows: hold the lock only
around the actual kernel launch + ``mx.eval`` (or kernel/pipeline build)
that submits work to the GPU, never around CPU-only numpy pre/post-
processing — the goal is to serialize GPU *submission*, not the pipeline.
"""
from __future__ import annotations

import os
import threading

# Reentrant: a guarded call may invoke another guarded helper on the same
# thread (e.g. a kernel-build helper called from within an already-guarded
# batch function) without deadlocking; it still blocks *other* threads while
# held, which is the cross-thread race this module exists to prevent.
_LOCK = threading.RLock()


def metal_guard():
    """Context manager guarding a region that submits work to MLX/Metal.

    Returns the single process-wide lock shared by every MLX submission site.
    Callers must import and use THIS function (not construct their own lock)
    so ``fpp_metal_kernel``, ``bls_metal_batch``, and any other Metal module
    all serialize against the same object — a second, independent lock would
    not prevent the cross-module race.
    """
    return _LOCK


def injection_metal_disabled_by_env() -> bool:
    """True iff ``EXONIT_INJECTION_METAL`` explicitly disables Metal
    injection-recovery backends. Env ``0``/``false`` (case-insensitive,
    whitespace-tolerant) disables; anything else, including unset, does not.

    Single source of truth for the "off" check, shared by every module that
    gates a Metal injection backend on this switch (``bls_metal.
    is_metal_available``, ``bls_metal_batch.is_available``, and
    ``injection.py``'s own central auto-detect gate) so the kill-switch's
    documented contract — "disables this backend unconditionally" — means
    the same thing everywhere, instead of independently-typed string
    comparisons silently drifting apart. Lives here (rather than in
    ``bls_metal.py`` or ``bls_metal_batch.py``) because those two already
    import from each other (``bls_metal_batch`` reuses ``bls_metal``'s
    ``_phase_fold_cpu``) — a helper defined in either would create or risk
    an import cycle; both already import this module for ``metal_guard()``.
    See the 2026-07-12 re-review, Finding 1: ``bls_metal.is_metal_available``
    used to not read this var at all, so the kill-switch only stopped
    Metal injection incidentally, via a worker.py check conditioned on
    FPP's own backend resolution — not the same check.
    """
    override = os.environ.get("EXONIT_INJECTION_METAL", "").strip().lower()
    return override in ("0", "false")
