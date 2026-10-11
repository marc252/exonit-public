"""Unified BLS dispatch — picks the best available backend automatically.

Priority: CUDA two-pass > Numba two-pass (CPU) > Metal/MLX > OpenCL > numpy

CUDA two-pass beats numba two-pass by ~2.76× on a T4 with bit-exact
period/SNR/depth (validated 2026-05-14 on 5 real TESS 2-min LCs after
bls_cuda's SDE/grid bug was fixed — the GPU was previously demoted to
fallback because single-pass GPU with a linear-period grid produced
collapsed SDEs that would zero injection recall). Now that the grid
is uniform-in-frequency (Kovács 2002) and the two-pass refine strategy
runs on the GPU, CUDA wins outright.

Override: set ``EXONIT_BLS_BACKEND`` (numba | metal | cuda | opencl |
numpy) to force a specific backend. Useful e.g. on Apple M1 where
numba runs single-thread due to workqueue thread-safety issues; Metal
via MLX can be faster in that case even though numba wins on M3+.
"""

import logging
import os

logger = logging.getLogger(__name__)

# --- Detect available backends at import time ---

_backend = None
_backend_name = None
_gpu_info_fn = None

_forced = (os.environ.get("EXONIT_BLS_BACKEND") or "").strip().lower() or None

# 1. Numba (preferred unless overridden)
try:
    from bls_fast import bls_two_pass, bls_fast, HAS_NUMBA
except ImportError:
    bls_fast = None
    bls_two_pass = None
    HAS_NUMBA = False

# 1. CUDA two-pass — fastest path when a CUDA GPU is present. Earlier code
#    preferred numba over CUDA because the single-pass GPU implementation was
#    a) buggy (linear-period grid → SDE collapse, see bls_cuda commit notes)
#    and b) slower than numba two_pass for the worker's default 50k periods
#    config. With the SNR fix and bls_gpu_two_pass on the GPU we measure
#    2.76× median speedup over 5 real TESS LCs with bit-exact period/SNR/depth
#    on a T4. Numba stays as the fallback for hosts without a CUDA GPU.
_cuda_available = False
_cuda_gpu_info_fn = None
try:
    from bls_cuda import bls_gpu_two_pass as _cuda_bls, is_cuda_available, get_gpu_info as _cuda_gpu_info
    if is_cuda_available():
        _cuda_available = True
        _cuda_gpu_info_fn = _cuda_gpu_info
        if _forced in (None, "cuda"):
            _backend = _cuda_bls
            _backend_name = "cuda"
            _gpu_info_fn = _cuda_gpu_info
except Exception as _e:
    # Never silent: a CUDA host falling back to CPU is the #1 fleet gotcha
    # (libcuda/driver/cupy import errors). worker.py escalates to ERROR when
    # gpu_enabled=true ends up on a CPU backend.
    logger.warning("BLS CUDA backend unavailable: %s", _e)

# 2. Numba two-pass — CPU fallback when CUDA is unavailable (or disabled).
if _backend is None and HAS_NUMBA and _forced in (None, "numba"):
    _backend = bls_two_pass
    _backend_name = "numba"

# 3. Metal/MLX
_metal_available = False
_metal_gpu_info_fn = None
try:
    from bls_metal import bls_gpu as _metal_bls, is_metal_available, get_gpu_info as _metal_gpu_info
    if is_metal_available():
        _metal_available = True
        _metal_gpu_info_fn = _metal_gpu_info
        if _backend is None and _forced in (None, "metal"):
            _backend = _metal_bls
            _backend_name = "metal"
            _gpu_info_fn = _metal_gpu_info
except Exception as _e:
    logger.warning("BLS Metal backend unavailable: %s", _e)

# 4. OpenCL
_opencl_available = False
_opencl_gpu_info_fn = None
try:
    from bls_opencl import bls_opencl, _init_opencl, get_gpu_info as _opencl_gpu_info
    if _init_opencl():
        _opencl_available = True
        _opencl_gpu_info_fn = _opencl_gpu_info
        if _backend is None and _forced in (None, "opencl"):
            _backend = bls_opencl
            _backend_name = "opencl"
except Exception as _e:
    logger.warning("BLS OpenCL backend unavailable: %s", _e)

# 5. Fallback: numpy CPU
if _backend is None and bls_fast is not None and _forced in (None, "numpy"):
    _backend = bls_fast
    _backend_name = "numpy"

if _forced is not None and _backend_name != _forced:
    logger.warning(
        "EXONIT_BLS_BACKEND=%s requested but not available; using %s",
        _forced, _backend_name)


def run_bls(time, flux, **kwargs):
    """Run BLS with the best available backend.

    Accepts all kwargs that bls_fast/bls_gpu accept:
    period_min, period_max, n_periods, n_bins, n_dur_steps, etc.
    """
    if _backend is None:
        raise RuntimeError("No BLS backend available (need at least numpy + bls_fast.py)")
    return _backend(time, flux, **kwargs)


def get_backend_name():
    """Return the active backend name: 'numba', 'cuda', 'metal', 'opencl', or 'numpy'."""
    return _backend_name or "none"


def get_gpu_info():
    """Return GPU info string, or None if using CPU.

    OpenCL is intentionally checked even when it isn't the ACTIVE backend
    (_gpu_info_fn only covers cuda/metal, whichever won backend selection):
    the fleet status table wants "what GPU is on this box", and an OpenCL
    device can be real and named even on a host where numba still won the
    BLS backend race (<host>: numba is faster for BLS, but the Arc iGPU
    _opencl_available flag/device are both real -- reporting it beats the
    previous hardcoded 'AMD OpenCL iGPU' string, which was wrong on Intel
    hosts and gave no real device name on any host)."""
    if _gpu_info_fn:
        try:
            return _gpu_info_fn()
        except Exception:
            pass
    if _opencl_available and _opencl_gpu_info_fn:
        try:
            name = _opencl_gpu_info_fn()
            if name:
                return name
        except Exception:
            pass
    if _backend_name == "opencl":
        return "OpenCL GPU (name unavailable)"
    return None


def is_gpu():
    """Return True if BLS is using a GPU backend."""
    return _backend_name in ("cuda", "metal", "opencl")


def get_hardware_info():
    """Return detected hardware accelerators, independent of the chosen BLS backend.

    Returns a dict like:
      {"cuda": "NVIDIA GeForce RTX 4070", "metal": None, "opencl": True}
    A value is truthy when the accelerator is usable on this machine.
    """
    info = {"cuda": False, "metal": False, "opencl": False}
    if _cuda_available:
        try:
            info["cuda"] = _cuda_gpu_info_fn() if _cuda_gpu_info_fn else True
        except Exception:
            info["cuda"] = True
    if _metal_available:
        try:
            info["metal"] = _metal_gpu_info_fn() if _metal_gpu_info_fn else True
        except Exception:
            info["metal"] = True
    if _opencl_available:
        info["opencl"] = "AMD/Intel OpenCL iGPU"
    return info


def get_hardware_summary():
    """One-line textual summary for logs/dashboards."""
    hw = get_hardware_info()
    parts = [k for k, v in hw.items() if v]
    return "+".join(parts) if parts else "cpu"
