"""Unit tests for the GPU-opt-out path in injection + fpp_native.

Context: the AMD Vega iGPU on ``<host>`` triggers the amdgpu
kernel timeout (``The CS has cancelled because the context is lost``)
when the injection OpenCL kernel runs over large light curves. The
worker config flag ``gpu_enabled=false`` must force both modules off
GPU; prior to the fix those modules ignored the flag (auto-detect at
import time).
"""
from __future__ import annotations


def test_injection_disable_gpu_clears_backend_and_batch():
    import injection
    # Whatever the auto-detected backend is at import, calling
    # disable_gpu() must tear it down.
    injection.disable_gpu()
    assert injection._bls_gpu is None
    assert injection._bls_gpu_batch is None
    assert injection.get_backend_name() == "CPU (disabled by config)"


def test_fpp_native_disable_gpu_clears_batch_fns():
    from worker.exominer_native import inference  # no-op import just to warm module system
    _ = inference
    import fpp_native
    fpp_native.disable_gpu()
    assert fpp_native._gpu_batch_transit is None
    assert fpp_native._gpu_batch_eb_transit is None
    assert fpp_native._fpp_gpu_backend is None
    # get_fpp_backend_info must report CPU after disable
    backend_name, info = fpp_native.get_fpp_backend_info()
    assert backend_name == "cpu"


def test_disable_gpu_is_idempotent():
    import injection
    import fpp_native
    injection.disable_gpu()
    injection.disable_gpu()  # second call must not raise
    fpp_native.disable_gpu()
    fpp_native.disable_gpu()
    assert injection._bls_gpu is None
    assert fpp_native._gpu_batch_transit is None
