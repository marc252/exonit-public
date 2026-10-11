"""ONNX Runtime execution-provider selection helper.

Picks the best available execution provider (EP) for an ONNX inference
session given the worker's `gpu_enabled` flag and what onnxruntime can
actually offer on the host. Always appends CPU as last resort so session
construction never fails — but a GPU host ending up on CPU is NOT silent:
the worker's startup GPU consistency check (worker.py) logs an ERROR.

Preference order, when `gpu_enabled=True`:
    1. ``CUDAExecutionProvider``  — NVIDIA (requires onnxruntime-gpu;
       the worker requirements list it as an opt-in extra for CUDA hosts).
    2. ``DmlExecutionProvider``   — DirectML, Intel/AMD iGPU on Windows
       (worker-HP).
    3. ``CPUExecutionProvider``   — always-available fallback so sessions
       never fail to construct.

CoreML on macOS is handled by a separate path (``worker/cnn_coreml.py``)
that loads a sibling ``.mlpackage`` and bypasses onnxruntime entirely, so
this helper does not include ``CoreMLExecutionProvider``.

The previous behaviour hard-coded ``["CPUExecutionProvider"]`` in
``cnn.py``, ``cnn_ffi.py`` and ``exominer_ffi_inference.py``, meaning
that even on workers with an NVIDIA GPU and ``onnxruntime-gpu``
installed, CNN and ExoMiner-FFI inference ran on CPU. The big run
``bg-20260513-1505`` on a T4 Spot was effectively CPU-bound for those
stages — see ``docs/PENDING_AUDIT.md`` (2026-05-14 entry).
"""
from __future__ import annotations

from typing import List, Sequence

try:
    import onnxruntime as ort  # type: ignore
except Exception:  # pragma: no cover - import guarded for tests
    ort = None  # type: ignore


def _preload_cuda_dlls() -> None:
    """Pre-load CUDA/cuDNN DLLs shipped as pip wheels (nvidia-* packages).

    On Windows/Linux hosts where onnxruntime-gpu is built against CUDA 12 but
    the system CUDA Toolkit is absent or is a different major version, the
    CUDAExecutionProvider fails to initialise because the DLLs
    (libcudnn*.dll, libcublas*.dll, …) are not on PATH/LD_LIBRARY_PATH.
    onnxruntime >= 1.21 provides ``ort.preload_dlls()`` which locates and
    loads these DLLs from the pip ``nvidia-*`` wheels already present in the
    venv's site-packages — before any InferenceSession is created.

    This is a no-op on:
    - ORT < 1.21 (``preload_dlls`` attribute missing)
    - CPU-only ORT builds (no CUDA DLLs to load, call silently succeeds)
    - macOS (CUDA does not exist; the attribute is absent)
    - Hosts where onnxruntime is not importable (ort is None)
    """
    if ort is None:
        return
    try:
        ort.preload_dlls()  # type: ignore[attr-defined]
    except Exception:  # pragma: no cover - host-specific, not testable in CI
        pass


_preload_cuda_dlls()


# CUDA goes first because workers with onnxruntime-gpu installed will pay a
# real performance cost when stuck on CPU. DirectML covers HP's Intel Xe-LP
# iGPU on Windows. CPU is always appended as the final fallback so the
# session construction never raises on a host without any accelerator EP.
_GPU_PROVIDER_PRIORITY: Sequence[str] = (
    "CUDAExecutionProvider",
    "DmlExecutionProvider",
)


def select_providers(gpu_enabled: bool = True) -> List[str]:
    """Return the EP list to pass to ``ort.InferenceSession(providers=...)``.

    Always ends with ``CPUExecutionProvider``. When ``gpu_enabled`` is
    False or onnxruntime is missing, returns ``["CPUExecutionProvider"]``.
    """
    if ort is None or not gpu_enabled:
        return ["CPUExecutionProvider"]
    available = set(ort.get_available_providers())
    providers = [p for p in _GPU_PROVIDER_PRIORITY if p in available]
    providers.append("CPUExecutionProvider")
    return providers


def describe_active_provider(session) -> str:
    """Return a short label for the EP actually backing this session.

    ``session.get_providers()`` returns the ordered list ORT will try; the
    first entry is the one in use. We strip the ``ExecutionProvider``
    suffix for readability in telemetry / log output (e.g. ``"CUDA"`` or
    ``"CPU"`` rather than ``"CUDAExecutionProvider"``).
    """
    try:
        providers = session.get_providers()
    except Exception:
        return "unknown"
    if not providers:
        return "unknown"
    name = providers[0]
    if name.endswith("ExecutionProvider"):
        name = name[: -len("ExecutionProvider")]
    return name or "unknown"
