"""Serialize all PyOpenCL GPU submissions within the worker process.

Mirrors ``metal_serial.py`` line for line (same rationale, different
backend): the fused OpenCL injection-recovery kernel (``bls_opencl.py``)
keeps its context/command-queue/compiled-program cache in module-level
globals (``_ctx``/``_queue``/kernel cache), and this module runs on a
background daemon thread while ``worker.py``'s FPP path (``fpp_opencl.py``,
wired in ``fpp_native.py:108`` as the "universal fallback") can submit
OpenCL work concurrently on the main thread. Two threads submitting through
the SAME PyOpenCL context/queue at once is exactly the MLX incident class
this codebase already hit once (2026-07-12, see ``metal_serial.py``'s
docstring): the fix there was a single process-wide reentrant lock around
every submission site, not a hope that the underlying library happens to be
thread-safe. PyOpenCL's own thread-safety guarantees are weaker than "none
of our code races the driver" requires in practice on a shared iGPU with a
long injection kernel running next to FPP bursts -- treat every submission
through ``_ctx``/``_queue`` the same way MLX submissions are treated: never
race two of them.

A single process-wide reentrant lock, shared across every OpenCL submission
site (``bls_opencl.py``'s fused kernel and ``fpp_opencl.py``'s FPP kernel --
see ``worker/injection.py`` and ``worker/fpp_native.py`` for how each is
reached), serializes those submissions and prevents the same crash class
from ever reproducing on Arc.

Keep every guarded region as small as correctness allows: hold the lock only
around the actual kernel launch + buffer read-back (or context/program
build) that submits work to the GPU, never around CPU-only numpy pre/post-
processing -- the goal is to serialize GPU *submission*, not the pipeline
(same rule as ``metal_serial.metal_guard``'s docstring).
"""
from __future__ import annotations

import os
import threading

# Reentrant: a guarded call may invoke another guarded helper on the same
# thread (e.g. a context-init helper called from within an already-guarded
# batch function) without deadlocking; it still blocks *other* threads while
# held, which is the cross-thread race this module exists to prevent.
_LOCK = threading.RLock()


def opencl_guard():
    """Context manager guarding a region that submits work to PyOpenCL.

    Returns the single process-wide lock shared by every OpenCL submission
    site. Callers must import and use THIS function (not construct their
    own lock) so ``bls_opencl``, ``fpp_opencl``, and any other OpenCL module
    all serialize against the same object -- a second, independent lock
    would not prevent the cross-module race.
    """
    return _LOCK


def injection_opencl_disabled_by_env() -> bool:
    """True iff ``EXONIT_INJECTION_OPENCL`` explicitly disables OpenCL
    injection-recovery backends. Env ``0``/``false`` (case-insensitive,
    whitespace-tolerant) disables; anything else, including unset, does not.

    Single source of truth for the "off" check, shared by every module that
    gates an OpenCL injection backend on this switch (``bls_opencl.
    is_opencl_available``, ``bls_opencl.bls_gpu_batch``, and any future
    central injection.py auto-detect gate) -- same contract as
    ``metal_serial.injection_metal_disabled_by_env`` for
    ``EXONIT_INJECTION_METAL``, kept in its own module for the identical
    reason: a helper defined inside ``bls_opencl.py`` would work today, but
    the moment a second OpenCL-consuming module needs the same check
    (``fpp_opencl.py`` already exists and will need one), duplicating the
    string-comparison risks the two independently drifting apart -- exactly
    the class of bug the 2026-07-12 Metal re-review found (Finding 1:
    ``bls_metal.is_metal_available`` used to not read its own kill-switch
    var at all).
    """
    override = os.environ.get("EXONIT_INJECTION_OPENCL", "").strip().lower()
    return override in ("0", "false")
