"""Unit tests for worker/onnx_providers.py.

Covers the ``_preload_cuda_dlls`` helper and ``select_providers``
without requiring an actual ONNX Runtime GPU installation.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# worker/ must be on sys.path (conftest.py appends it, but guard here too)
_WORKER = str(Path(__file__).resolve().parent.parent / "worker")
if _WORKER not in sys.path:
    sys.path.append(_WORKER)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fresh_import():
    """Remove onnx_providers from sys.modules and re-import it fresh."""
    sys.modules.pop("onnx_providers", None)
    import onnx_providers  # noqa: PLC0415 — intentional dynamic import
    return onnx_providers


# ---------------------------------------------------------------------------
# _preload_cuda_dlls — called once at module import time
# ---------------------------------------------------------------------------

class TestPreloadCudaDlls:
    """``_preload_cuda_dlls`` must survive every failure mode silently."""

    def test_preload_dlls_called_when_attribute_exists(self):
        """If ort.preload_dlls exists it is called during module import."""
        mock_ort = MagicMock()
        mock_ort.preload_dlls = MagicMock()
        mock_ort.get_available_providers = MagicMock(return_value=[])

        with patch.dict(sys.modules, {"onnxruntime": mock_ort}):
            mod = _fresh_import()

        mock_ort.preload_dlls.assert_called_once()
        # Module must still expose the public API
        assert callable(mod.select_providers)
        assert callable(mod.describe_active_provider)

    def test_preload_dlls_missing_attribute_does_not_raise(self):
        """ORT < 1.21: no preload_dlls attribute → import succeeds silently."""
        mock_ort = MagicMock(spec=[])  # spec=[] → no attributes at all
        # get_available_providers is needed by select_providers
        mock_ort.get_available_providers = MagicMock(return_value=[])

        with patch.dict(sys.modules, {"onnxruntime": mock_ort}):
            mod = _fresh_import()

        # No exception, public API intact
        assert callable(mod.select_providers)

    def test_preload_dlls_raises_does_not_raise(self):
        """If preload_dlls raises (e.g. DLL not found), import still succeeds."""
        mock_ort = MagicMock()
        mock_ort.preload_dlls.side_effect = OSError("libcuda.so.1: cannot open")
        mock_ort.get_available_providers = MagicMock(return_value=[])

        with patch.dict(sys.modules, {"onnxruntime": mock_ort}):
            mod = _fresh_import()

        # Exception was swallowed, module is functional
        assert callable(mod.select_providers)

    def test_ort_none_preload_is_skipped(self):
        """When onnxruntime is not importable, _preload_cuda_dlls is a no-op."""
        with patch.dict(sys.modules, {"onnxruntime": None}):
            # Force re-import with onnxruntime absent
            sys.modules.pop("onnx_providers", None)
            # Temporarily make 'import onnxruntime' raise ImportError
            orig = sys.modules.get("onnxruntime", "MISSING")
            sys.modules["onnxruntime"] = None  # type: ignore[assignment]
            try:
                # Can't patch a None entry the same way — call function directly
                sys.modules.pop("onnx_providers", None)
                import onnx_providers as mod  # noqa: PLC0415
                # ort will be None (import guard catches the AttributeError
                # that fires when accessing None as a module)
                mod2 = _fresh_import()
                mod2._preload_cuda_dlls()  # must not raise
            finally:
                if orig == "MISSING":
                    sys.modules.pop("onnxruntime", None)
                else:
                    sys.modules["onnxruntime"] = orig  # type: ignore[assignment]

    def test_preload_cuda_dlls_function_directly_no_attr(self):
        """Call _preload_cuda_dlls() with a mock lacking preload_dlls."""
        import onnx_providers as mod
        original_ort = mod.ort
        try:
            stub = types.SimpleNamespace(get_available_providers=lambda: [])
            # stub has no preload_dlls attr → AttributeError swallowed
            mod.ort = stub  # type: ignore[attr-defined]
            mod._preload_cuda_dlls()  # must not raise
        finally:
            mod.ort = original_ort  # type: ignore[attr-defined]

    def test_preload_cuda_dlls_function_directly_raises(self):
        """Call _preload_cuda_dlls() with a mock whose preload_dlls raises."""
        import onnx_providers as mod
        original_ort = mod.ort
        try:
            stub = types.SimpleNamespace(
                get_available_providers=lambda: [],
                preload_dlls=lambda: (_ for _ in ()).throw(RuntimeError("boom")),
            )
            mod.ort = stub  # type: ignore[attr-defined]
            mod._preload_cuda_dlls()  # must not raise
        finally:
            mod.ort = original_ort  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# select_providers — existing behaviour must be unchanged
# ---------------------------------------------------------------------------

class TestSelectProviders:
    """Regression tests: select_providers returns the right EP list."""

    def test_gpu_disabled_returns_cpu_only(self):
        import onnx_providers as mod
        assert mod.select_providers(gpu_enabled=False) == ["CPUExecutionProvider"]

    def test_ort_none_returns_cpu_only(self):
        import onnx_providers as mod
        original_ort = mod.ort
        try:
            mod.ort = None  # type: ignore[attr-defined]
            result = mod.select_providers(gpu_enabled=True)
        finally:
            mod.ort = original_ort  # type: ignore[attr-defined]
        assert result == ["CPUExecutionProvider"]

    def test_cuda_available_returns_cuda_first(self):
        import onnx_providers as mod
        original_ort = mod.ort
        try:
            stub = MagicMock()
            stub.get_available_providers.return_value = [
                "CUDAExecutionProvider",
                "CPUExecutionProvider",
            ]
            mod.ort = stub  # type: ignore[attr-defined]
            result = mod.select_providers(gpu_enabled=True)
        finally:
            mod.ort = original_ort  # type: ignore[attr-defined]
        assert result[0] == "CUDAExecutionProvider"
        assert result[-1] == "CPUExecutionProvider"

    def test_cpu_always_last(self):
        import onnx_providers as mod
        original_ort = mod.ort
        try:
            stub = MagicMock()
            stub.get_available_providers.return_value = ["DmlExecutionProvider"]
            mod.ort = stub  # type: ignore[attr-defined]
            result = mod.select_providers(gpu_enabled=True)
        finally:
            mod.ort = original_ort  # type: ignore[attr-defined]
        assert result[-1] == "CPUExecutionProvider"
