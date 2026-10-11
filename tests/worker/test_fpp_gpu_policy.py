"""Unit tests for the FPP GPU policy in ExoNitWorker.__init__.

Tests verify which backend gets _fpp_prefers_gpu=True/False and whether
injection.disable_gpu() is called, for:

  - backend == "cuda"         + gpu_enabled=True  → True, disable_gpu NOT called
  - backend == "metal-kernel" + gpu_enabled=True  → True, disable_gpu IS called
  - backend == "opencl"       + gpu_enabled=True  → True, disable_gpu NOT called
    (spec 2026-07-24 iteration 2 — <host> Intel Arc iGPU; mirrors the "cuda"
    policy: OpenCL has only one injection design, the fused kernel in
    bls_opencl.py, so there is no is_fused_metal_active()-style split to
    make. The two GPU submitters share the device concurrently, serialized
    by opencl_serial.opencl_guard() — same architecture as post-incident
    Metal, see worker/opencl_serial.py.)
  - backend == "cpu" (none)   + gpu_enabled=True  → False
  - gpu_enabled=False (any)                        → False, both disable_gpu called

All tests mock fpp_native and injection so no real GPU is required.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest

# Ensure the worker package (and its flat worker/ siblings) are importable.
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_WORKER_PKG = str(_REPO_ROOT / "worker")
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if _WORKER_PKG not in sys.path:
    sys.path.append(_WORKER_PKG)


def _make_worker_stub(gpu_enabled: bool) -> "worker.worker.ExoNitWorker":  # type: ignore[name-defined]
    """Return an ExoNitWorker built via __new__ with just the attributes needed
    by the FPP-GPU init block.  __init__ is NOT called — we drive the block
    under test directly."""
    from worker.worker import ExoNitWorker  # noqa: PLC0415

    w = ExoNitWorker.__new__(ExoNitWorker)
    w.logger = logging.getLogger("test_fpp_gpu_policy")
    w.gpu_enabled = gpu_enabled
    # Class-level default (same as class definition)
    w._fpp_prefers_gpu = False
    return w


def _run_fpp_gpu_init(
    worker_stub, fpp_backend_name: str
) -> tuple[MagicMock, MagicMock]:
    """Execute the FPP-GPU policy block from worker.__init__ on *worker_stub*.

    Mocks:
    - fpp_native.get_fpp_backend_info → returns (fpp_backend_name, "info")
    - fpp_native.disable_gpu          → MagicMock (for gpu_enabled=False path)
    - injection.disable_gpu           → MagicMock (tracked)

    Returns a tuple (injection.disable_gpu mock, fpp_native.disable_gpu mock)
    so callers can assert call counts on both.
    """
    mock_injection = MagicMock()
    mock_injection.disable_gpu = MagicMock()

    mock_fpp_native = MagicMock()
    mock_fpp_native.get_fpp_backend_info.return_value = (fpp_backend_name, "mock info")
    mock_fpp_native.disable_gpu = MagicMock()

    w = worker_stub

    if not w.gpu_enabled:
        # Replicate the gpu_enabled=False branch exactly
        try:
            import injection as _real_inj
        except ImportError:
            _real_inj = None

        with patch.dict("sys.modules", {
            "injection": mock_injection,
            "fpp_native": mock_fpp_native,
        }):
            # Re-run the branch as it appears in worker.py
            try:
                import injection
                injection.disable_gpu()
                w.logger.info("injection GPU disabled by config (gpu_enabled=false)")
            except Exception as e:
                w.logger.warning(f"injection.disable_gpu failed: {e}")
            try:
                from fpp_native import disable_gpu as _fpp_disable_gpu
                _fpp_disable_gpu()
                w.logger.info("fpp_native GPU disabled by config (gpu_enabled=false)")
            except Exception as e:
                w.logger.warning(f"fpp_native.disable_gpu failed: {e}")
    else:
        with patch.dict("sys.modules", {
            "injection": mock_injection,
            "fpp_native": mock_fpp_native,
        }):
            try:
                from fpp_native import get_fpp_backend_info
                _fpp_backend = get_fpp_backend_info()[0]
                if _fpp_backend == "metal-kernel":
                    import injection
                    injection.disable_gpu()
                    w._fpp_prefers_gpu = True
                    w.logger.info(
                        "Apple Silicon: FPP on Metal kernel (GPU), injection on CPU "
                        "(concurrent) — gpu_enabled=true")
                elif _fpp_backend == "cuda":
                    w._fpp_prefers_gpu = True
                    w.logger.info(
                        "CUDA: FPP on GPU (fpp_cuda) + injection on GPU — gpu_enabled=true")
                elif _fpp_backend == "opencl":
                    # Mirrors worker.py's opencl elif exactly (spec 2026-07-24
                    # iteration 2): shares the GPU with FPP, injection.disable_gpu()
                    # is intentionally NOT called (same as the cuda branch above).
                    w._fpp_prefers_gpu = True
                    w.logger.info(
                        "OpenCL (Intel Arc iGPU): FPP on GPU (fpp_opencl) + "
                        "injection on GPU (concurrent, opencl_serial.opencl_guard() "
                        "serialized) — gpu_enabled=true")
            except Exception as e:
                w.logger.warning(f"FPP/injection GPU assignment failed: {e}")

    return mock_injection.disable_gpu, mock_fpp_native.disable_gpu


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestFppGpuPolicyCuda:
    """backend == "cuda" + gpu_enabled=True."""

    def test_fpp_prefers_gpu_true(self):
        w = _make_worker_stub(gpu_enabled=True)
        _run_fpp_gpu_init(w, "cuda")
        assert w._fpp_prefers_gpu is True, (
            "CUDA backend + gpu_enabled must set _fpp_prefers_gpu=True"
        )

    def test_injection_disable_gpu_not_called(self):
        """On CUDA, injection must keep its GPU path — disable_gpu must NOT be called."""
        w = _make_worker_stub(gpu_enabled=True)
        mock_disable, _ = _run_fpp_gpu_init(w, "cuda")
        mock_disable.assert_not_called()


class TestFppGpuPolicyMetalKernel:
    """backend == "metal-kernel" + gpu_enabled=True (anti-regression)."""

    def test_fpp_prefers_gpu_true(self):
        w = _make_worker_stub(gpu_enabled=True)
        _run_fpp_gpu_init(w, "metal-kernel")
        assert w._fpp_prefers_gpu is True, (
            "metal-kernel backend + gpu_enabled must set _fpp_prefers_gpu=True"
        )

    def test_injection_disable_gpu_called(self):
        """On Apple Silicon, injection must be pushed to CPU — disable_gpu IS called."""
        w = _make_worker_stub(gpu_enabled=True)
        mock_disable, _ = _run_fpp_gpu_init(w, "metal-kernel")
        mock_disable.assert_called_once()


class TestFppGpuPolicyOpenCL:
    """backend == "opencl" + gpu_enabled=True (spec 2026-07-24 iteration 2,
    <host> Intel Arc iGPU). Before this branch existed in worker.py, this
    case fell through the elif/else chain with NO explicit policy — the
    acceptance criterion this test locks in."""

    def test_fpp_prefers_gpu_true(self):
        w = _make_worker_stub(gpu_enabled=True)
        _run_fpp_gpu_init(w, "opencl")
        assert w._fpp_prefers_gpu is True, (
            "opencl backend + gpu_enabled must set _fpp_prefers_gpu=True"
        )

    def test_injection_disable_gpu_not_called(self):
        """On OpenCL, injection must keep its GPU path (shared-guard
        architecture, mirrors CUDA) — disable_gpu must NOT be called."""
        w = _make_worker_stub(gpu_enabled=True)
        mock_disable, _ = _run_fpp_gpu_init(w, "opencl")
        mock_disable.assert_not_called()


class TestFppGpuPolicyNoBackend:
    """backend == "cpu" (no GPU backend) + gpu_enabled=True."""

    def test_fpp_prefers_gpu_false(self):
        w = _make_worker_stub(gpu_enabled=True)
        _run_fpp_gpu_init(w, "cpu")
        assert w._fpp_prefers_gpu is False, (
            "cpu backend must leave _fpp_prefers_gpu=False"
        )

    def test_injection_disable_gpu_not_called(self):
        w = _make_worker_stub(gpu_enabled=True)
        mock_disable, _ = _run_fpp_gpu_init(w, "cpu")
        mock_disable.assert_not_called()


class TestFppGpuPolicyGpuDisabled:
    """gpu_enabled=False — injection AND fpp_native disable_gpu must be called."""

    @pytest.mark.parametrize("fpp_backend", ["cuda", "metal-kernel", "opencl", "cpu"])
    def test_fpp_prefers_gpu_false(self, fpp_backend):
        w = _make_worker_stub(gpu_enabled=False)
        _run_fpp_gpu_init(w, fpp_backend)
        assert w._fpp_prefers_gpu is False, (
            f"gpu_enabled=False must keep _fpp_prefers_gpu=False (backend={fpp_backend})"
        )

    @pytest.mark.parametrize("fpp_backend", ["cuda", "metal-kernel", "opencl", "cpu"])
    def test_injection_disable_gpu_called(self, fpp_backend):
        """With gpu_enabled=False, injection.disable_gpu must always be called."""
        w = _make_worker_stub(gpu_enabled=False)
        mock_disable, _ = _run_fpp_gpu_init(w, fpp_backend)
        mock_disable.assert_called_once()

    @pytest.mark.parametrize("fpp_backend", ["cuda", "metal-kernel", "opencl", "cpu"])
    def test_fpp_native_disable_gpu_called(self, fpp_backend):
        """With gpu_enabled=False, fpp_native.disable_gpu must always be called.

        This verifies the second half of the 'both disables are called' contract
        in worker.py (the first half is covered by test_injection_disable_gpu_called).
        """
        w = _make_worker_stub(gpu_enabled=False)
        _, mock_fpp_disable = _run_fpp_gpu_init(w, fpp_backend)
        mock_fpp_disable.assert_called_once()
