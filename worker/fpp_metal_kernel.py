"""Apple Silicon (Metal) FPP transit-model evaluation via a custom fused kernel.

This is the FAST Metal path for FPP, replacing the array-op ``fpp_metal`` module
(which materialised ~500 (N,T) intermediates per batch and was memory-bandwidth-
bound, ~4x slower than numba). Here a single GPU thread handles one parameter draw
and runs the Kepler solve + 100-ring integration over all T timepoints in registers
(no (N,T) materialisation), mirroring the CUDA kernel in ``fpp_cuda.py``.

Measured on M4 (Metal 4): ~5x faster than numba for N>=10k draws, T=1000.
float32 throughout (Apple GPUs have no float64) — identical precision to the CUDA
float32 path already used in production on the NVIDIA workers.

Activated by ``fpp_native._init_fpp_gpu`` when MLX + ``mx.fast.metal_kernel`` are
available. Requires MLX >= 0.18 (metal_kernel API).
"""
import logging
import math
import numpy as np

from metal_serial import metal_guard

try:
    import mlx.core as mx
    _MLX_OK = hasattr(mx, "fast") and hasattr(mx.fast, "metal_kernel")
except ImportError:
    _MLX_OK = False

_LN2PI = 1.8378770664093453

# --- shared MSL fragment: per-timepoint transit model flux (no dilution) ------
# Computes z(time[i]) and the 100-ring integrated model flux into `model`.
_MODEL_FRAG = r"""
        float t_val = time[i];
        float phase = t_val / period;
        float M = two_pi * (phase - floor(phase)) - w;
        float nu;
        if (ecc < 1e-10f) {
            nu = M;
        } else {
            float E = M;
            for (int it = 0; it < 10; it++) {
                E -= (E - ecc * sin(E) - M) / (1.0f - ecc * cos(E));
            }
            nu = atan2(sqrt_1me2 * sin(E), cos(E) - ecc);
        }
        float r_factor = (1.0f - e2) / (1.0f + ecc * cos(nu));
        float f = nu + w;
        float sin_f = sin(f);
        float cos_f = cos(f);
        float z = a_Rs * r_factor * sqrt(sin_f*sin_f + cos_i*cos_i*cos_f*cos_f);

        float model = 1.0f;
        if (z < 1.0f + k && k > 0.0f) {
            float dr = 0.01f;
            float blocked = 0.0f;
            float z2 = z * z;
            float p2 = k * k;
            for (int ri = 0; ri < 100; ri++) {
                float r = ((float)ri + 0.5f) * dr;
                float r2 = r * r;
                float mu = sqrt(fmax(1.0f - r2, 0.0f));
                float omu = 1.0f - mu;
                float I_r = 1.0f - u1 * omu - u2 * omu * omu;
                float frac;
                if (z < 1e-12f) {
                    frac = (r <= k) ? 1.0f : 0.0f;
                } else if (r >= z + k) {
                    frac = 0.0f;
                } else if (r + k <= z) {
                    frac = 0.0f;
                } else if (r <= fabs(z - k)) {
                    frac = (k > z) ? 1.0f : 0.0f;
                } else {
                    float arg = (r2 + z2 - p2) / (2.0f * r * z);
                    if (arg >= 1.0f) frac = 0.0f;
                    else if (arg <= -1.0f) frac = 1.0f;
                    else frac = acos(arg) / 3.14159265358979323846f;
                }
                blocked += I_r * frac * 2.0f * r * dr;
            }
            model = 1.0f - blocked / omega;
        }
"""

_TRANSIT_SRC = r"""
    uint j = thread_position_in_grid.x;
    int N = (int)params[2];
    int T = (int)params[3];
    if (j >= (uint)N) return;
    float u1 = params[0];
    float u2 = params[1];
    int companion_is_host = (int)params[4];

    float k = ks[j];
    float period = P_orbs[j];
    float inc = incs_rad[j];
    float a_Rs = a_Rs_arr[j];
    float ecc = eccs[j];
    float w = ws_rad[j];
    float fr = comp_fr[j];

    float two_pi = 6.2831853071795864f;
    float cos_i = cos(inc);
    float omega = 1.0f - u1 / 3.0f - u2 / 6.0f;
    if (omega <= 0.0f) omega = 1.0f;
    float e2 = ecc * ecc;
    float sqrt_1me2 = sqrt(fmax(1.0f - e2, 1e-12f));

    float chi2 = 0.0f;
    for (int i = 0; i < T; i++) {
""" + _MODEL_FRAG + r"""
        if (fr > 0.0f) {
            float F_dilute;
            if (companion_is_host) F_dilute = 1.0f / (fr / (1.0f - fr));
            else                   F_dilute = fr / (1.0f - fr);
            model = (model + F_dilute) / (1.0f + F_dilute);
        }
        float diff = flux[i] - model;
        chi2 += diff * diff;
    }
    chi2_out[j] = chi2;
"""

_EB_SRC = r"""
    uint j = thread_position_in_grid.x;
    int N = (int)params[2];
    int T = (int)params[3];
    if (j >= (uint)N) return;
    float u1 = params[0];
    float u2 = params[1];
    int companion_is_host = (int)params[4];

    float k = ks[j];
    if (fabs(k - 1.0f) < 1e-6f) k *= 0.999f;
    float period = P_orbs[j];
    float inc = incs_rad[j];
    float a_Rs = a_Rs_arr[j];
    float ecc = eccs[j];
    float w = ws_rad[j];

    float fr_eb = eb_fluxratios[j];
    float F_EB = (fr_eb < 1.0f) ? fr_eb / (1.0f - fr_eb) : 1e6f;
    float fr_comp = comp_fr[j];
    float F_comp = (fr_comp > 0.0f) ? fr_comp / (1.0f - fr_comp) : 0.0f;

    float two_pi = 6.2831853071795864f;
    float cos_i = cos(inc);
    float omega = 1.0f - u1 / 3.0f - u2 / 6.0f;
    if (omega <= 0.0f) omega = 1.0f;
    float e2 = ecc * ecc;
    float sqrt_1me2 = sqrt(fmax(1.0f - e2, 1e-12f));

    // secdepth + dilution params (i-independent): compute once
    float secdepth;
    float d1, d2;  // two dilution stages
    int host = companion_is_host;
    if (host) {
        d1 = F_EB / fmax(F_comp, 1e-30f);
        d2 = 1.0f / (F_comp + F_EB);
        secdepth = fr_eb / (1.0f + d2);
    } else {
        d1 = F_EB;
        d2 = F_comp / (1.0f + F_EB);
        secdepth = (d2 > 0.0f) ? fr_eb / (1.0f + d2) : fr_eb;
    }
    secdepth_out[j] = secdepth;

    float chi2 = 0.0f;
    for (int i = 0; i < T; i++) {
""" + _MODEL_FRAG + r"""
        model = (model + d1) / (1.0f + d1);
        model = (model + d2) / (1.0f + d2);
        float diff = flux[i] - model;
        chi2 += diff * diff;
    }
    chi2_out[j] = chi2;
"""

_transit_kernel = None
_eb_kernel = None


def _build():
    global _transit_kernel, _eb_kernel
    if _transit_kernel is None:
        _transit_kernel = mx.fast.metal_kernel(
            name="fpp_transit_chi2",
            input_names=["time", "flux", "ks", "P_orbs", "incs_rad", "a_Rs_arr",
                         "eccs", "ws_rad", "comp_fr", "params"],
            output_names=["chi2_out"],
            source=_TRANSIT_SRC,
        )
        _eb_kernel = mx.fast.metal_kernel(
            name="fpp_eb_transit_chi2",
            input_names=["time", "flux", "ks", "eb_fluxratios", "P_orbs",
                         "incs_rad", "a_Rs_arr", "eccs", "ws_rad", "comp_fr",
                         "params"],
            output_names=["chi2_out", "secdepth_out"],
            source=_EB_SRC,
        )


def is_available():
    if not _MLX_OK:
        return False
    try:
        with metal_guard():
            _build()
            x = mx.array([1.0, 2.0], dtype=mx.float32)
            mx.eval(mx.sum(x))
        return True
    except Exception as e:
        logging.getLogger(__name__).warning("MLX present but Metal kernel unusable: %s", e)
        return False


def get_info():
    try:
        import subprocess
        r = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                           capture_output=True, text=True, timeout=5)
        chip = r.stdout.strip() if r.returncode == 0 else "Apple Silicon"
    except Exception:
        chip = "Apple Silicon"
    return f"{chip} (Metal fused kernel)"


def _f32(x):
    return mx.array(np.ascontiguousarray(x, dtype=np.float32))


def _params(u1, u2, N, T, companion_is_host):
    return mx.array(np.array([u1, u2, float(N), float(T),
                              1.0 if companion_is_host else 0.0], dtype=np.float32))


def _lnL(chi2_np, sigma):
    sigma2 = float(sigma) ** 2
    return -0.5 * _LN2PI - math.log(float(sigma)) - 0.5 * chi2_np / sigma2


def batch_transit_lnL(time, flux, sigma, ks, P_orbs, incs_rad, a_Rs_arr,
                      eccs, ws_rad, u1, u2, companion_fluxratio, companion_is_host):
    """Transit lnL for N draws via fused Metal kernel. Returns float64 (N,)."""
    N = len(ks); T = len(time)
    # metal_guard(): MLX is not thread-safe -- injection-recovery submits
    # concurrently from a background thread (bls_metal_batch). The build
    # (first call only, then cached) + kernel launch + eval is where the
    # command buffer gets committed, so all of it must be serialized against
    # the same process-wide lock injection uses.
    with metal_guard():
        _build()
        (chi2,) = _transit_kernel(
            inputs=[_f32(time), _f32(flux), _f32(ks), _f32(P_orbs), _f32(incs_rad),
                    _f32(a_Rs_arr), _f32(eccs), _f32(ws_rad), _f32(companion_fluxratio),
                    _params(u1, u2, N, T, companion_is_host)],
            grid=(N, 1, 1), threadgroup=(min(256, max(N, 1)), 1, 1),
            output_shapes=[(N,)], output_dtypes=[mx.float32])
        mx.eval(chi2)
    # np.array() readout runs outside the guard, but that's safe: mx.eval()
    # above already materialized chi2 inside the guard -- this only copies
    # realized memory, no new command-buffer submission (2026-07-12
    # re-review, "outside the guard" note).
    return _lnL(np.array(chi2, dtype=np.float64), sigma)


def batch_eb_transit_lnL(time, flux, sigma, ks, eb_fluxratios, P_orbs,
                         incs_rad, a_Rs_arr, eccs, ws_rad, u1, u2,
                         companion_fluxratio, companion_is_host):
    """EB transit lnL for N draws via fused Metal kernel, with secondary-eclipse
    rejection (secdepth >= 1.5*sigma -> lnL=-inf). Returns float64 (N,)."""
    N = len(ks); T = len(time)
    # metal_guard(): see batch_transit_lnL -- same process-wide lock, shared
    # with injection's bls_metal_batch, serializes the command-buffer commit.
    with metal_guard():
        _build()
        chi2, secdepth = _eb_kernel(
            inputs=[_f32(time), _f32(flux), _f32(ks), _f32(eb_fluxratios), _f32(P_orbs),
                    _f32(incs_rad), _f32(a_Rs_arr), _f32(eccs), _f32(ws_rad),
                    _f32(companion_fluxratio), _params(u1, u2, N, T, companion_is_host)],
            grid=(N, 1, 1), threadgroup=(min(256, max(N, 1)), 1, 1),
            output_shapes=[(N,), (N,)], output_dtypes=[mx.float32, mx.float32])
        mx.eval(chi2, secdepth)
    # Both readouts below run outside the guard, but that's safe: mx.eval()
    # above already materialized chi2/secdepth inside the guard -- this only
    # copies realized memory, no new command-buffer submission (2026-07-12
    # re-review, "outside the guard" note).
    lnL = _lnL(np.array(chi2, dtype=np.float64), sigma)
    reject = np.array(secdepth, dtype=np.float64) >= 1.5 * float(sigma)
    lnL[reject] = -np.inf
    return lnL
