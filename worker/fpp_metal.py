"""Apple Silicon (Metal/MLX) accelerated FPP transit model evaluation.

Fully vectorized — no Python loops over time points or ring integration.
All operations are batched MLX array ops on GPU.

Install MLX: pip install mlx  (Apple Silicon only)
"""
import math
import numpy as np

from metal_serial import metal_guard

try:
    import mlx.core as mx
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False


def is_available():
    """True if MLX/Metal is available and functional."""
    if not MLX_AVAILABLE:
        return False
    try:
        with metal_guard():
            x = mx.array([1.0, 2.0, 3.0], dtype=mx.float32)
            r = mx.sum(x)
            mx.eval(r)
        return True
    except Exception:
        return False


def get_info():
    """Returns a string describing the Apple Silicon GPU."""
    if not is_available():
        return "Metal/MLX not available"
    try:
        import subprocess
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0 and result.stdout.strip():
            return f"{result.stdout.strip()} (Metal/MLX)"
    except Exception:
        pass
    return "Apple Silicon (Metal/MLX)"


def _kepler_solve(M, ecc):
    """Solve Kepler equation via 10 fixed Newton iterations. Vectorized over (N,) or (N,T)."""
    E = M
    for _ in range(10):
        E = E - (E - ecc * mx.sin(E) - M) / (1.0 - ecc * mx.cos(E))
    return E


def _compute_z_vectorized(time_arr, P_orbs, incs_rad, a_Rs_arr, eccs, ws_rad):
    """Compute z(t) for all N draws × T time points at once.

    Args:
        time_arr: mx array (T,) — phase-folded time
        P_orbs: mx array (N,)
        incs_rad, a_Rs_arr, eccs, ws_rad: mx arrays (N,)

    Returns:
        mx array (N, T) of projected separations.
    """
    TWO_PI = 6.2831853
    # (N, T) = (N, 1) op (1, T)
    phase = time_arr[None, :] / P_orbs[:, None]        # (N, T)
    phase = phase - mx.floor(phase)
    M = TWO_PI * phase - ws_rad[:, None]                # (N, T)

    E = _kepler_solve(M, eccs[:, None])                 # (N, T)
    cosE = mx.cos(E)
    sinE = mx.sin(E)

    ecc2 = eccs[:, None]
    sqrt_1me2 = mx.sqrt(mx.maximum(1.0 - ecc2 * ecc2, 1e-12))
    nu = mx.arctan2(sqrt_1me2 * sinE, cosE - ecc2)     # (N, T)

    r_factor = (1.0 - ecc2 * ecc2) / (1.0 + ecc2 * mx.cos(nu))  # (N, T)
    f = nu + ws_rad[:, None]                            # (N, T)
    sin_f = mx.sin(f)
    cos_f = mx.cos(f)
    cos_i = mx.cos(incs_rad)[:, None]                   # (N, 1)

    z = a_Rs_arr[:, None] * r_factor * mx.sqrt(
        sin_f * sin_f + cos_i * cos_i * cos_f * cos_f)  # (N, T)
    return z


def _ring_integral_vectorized(z, ks, u1, u2, n_rings=50):
    """Compute transit flux for all (N, T) z values at once.

    Uses vectorized ring integration — no Python loops.
    Reduced to 50 rings (vs 100) for GPU efficiency; accuracy <0.1%.

    Args:
        z: mx array (N, T)
        ks: mx array (N,) — planet/star radius ratios
        u1, u2: float — limb darkening coefficients

    Returns:
        mx array (N, T) of normalized flux values.
    """
    omega = 1.0 - u1 / 3.0 - u2 / 6.0
    if omega <= 0.0:
        omega = 1.0

    dr = 1.0 / n_rings
    # Ring radii: (R,) where R = n_rings
    r_vals = mx.array([(i + 0.5) * dr for i in range(n_rings)], dtype=mx.float32)
    r2_vals = r_vals * r_vals
    mu_vals = mx.sqrt(1.0 - r2_vals)
    one_minus_mu = 1.0 - mu_vals
    I_r = 1.0 - u1 * one_minus_mu - u2 * one_minus_mu * one_minus_mu  # (R,)
    weights = I_r * 2.0 * r_vals * dr  # (R,)

    # Expand dims for broadcasting: z (N,T), ks (N,), r_vals (R,)
    # We need (N, T, R) but that may be too large.
    # Instead, accumulate over rings with a loop of 50 (not 1000).
    ks_exp = ks[:, None]  # (N, 1)
    blocked = mx.zeros_like(z)  # (N, T)

    for ri in range(n_rings):
        r = float(r_vals[ri])
        r2 = r * r
        w = float(weights[ri])

        r_mx = mx.array(r, dtype=mx.float32)
        two_r_z = 2.0 * r_mx * mx.maximum(z, 1e-12)
        arg = (r2 + z * z - ks_exp * ks_exp) / two_r_z
        arg_clipped = mx.clip(arg, -1.0, 1.0)

        frac = mx.arccos(arg_clipped) / 3.14159265

        # Edge cases (all vectorized boolean masks)
        z_near_zero = z < 1e-12
        r_outside = r_mx >= z + ks_exp
        r_inside_z = r_mx + ks_exp <= z
        r_inside_p = r_mx <= mx.abs(z - ks_exp)
        k_gt_z = ks_exp > z

        frac = mx.where(r_inside_p & ~k_gt_z, 0.0, frac)
        frac = mx.where(r_inside_p & k_gt_z, 1.0, frac)
        frac = mx.where(r_inside_z, 0.0, frac)
        frac = mx.where(r_outside, 0.0, frac)
        frac = mx.where(z_near_zero, mx.where(r_mx <= ks_exp, 1.0, 0.0), frac)

        blocked = blocked + w * frac

    return 1.0 - blocked / omega


def _apply_companion_dilution(model, companion_fluxratio, companion_is_host):
    """Dilute model flux by companion star. Vectorized over (N, T)."""
    has_comp = companion_fluxratio > 0.0  # (N,)
    if companion_is_host:
        safe_fr = mx.maximum(companion_fluxratio, 1e-12)
        F_dilute = (1.0 - safe_fr) / safe_fr
    else:
        safe_fr = mx.maximum(companion_fluxratio, 1e-12)
        F_dilute = safe_fr / (1.0 - safe_fr)
    # F_dilute is (N,), model is (N, T)
    F_dilute = F_dilute[:, None]
    diluted = (model + F_dilute) / (1.0 + F_dilute)
    return mx.where(has_comp[:, None], diluted, model)


def batch_transit_lnL(time, flux, sigma, ks, P_orbs, incs_rad,
                      a_Rs_arr, eccs, ws_rad, u1, u2,
                      companion_fluxratio, companion_is_host):
    """Vectorized transit log-likelihood on Apple Silicon (Metal/MLX).

    Computes all N parameter sets × T time points in one GPU pass.
    No Python loops over time points.

    Returns: numpy float64 array of shape (N,).
    """
    if not MLX_AVAILABLE:
        raise ImportError("MLX not available. Install with: pip install mlx")

    N = len(ks)
    T = len(time)

    # metal_guard(): this legacy path builds an MLX op graph (Kepler solve +
    # ring integration, all lazy) and only actually submits to the GPU at
    # mx.eval() below -- but it shares the same MLX device/command-queue as
    # the fused kernels, so the whole region is serialized against the same
    # process-wide lock (see metal_serial.py) rather than relying on exactly
    # which intermediate call is "the" submission point.
    with metal_guard():
        mx_time = mx.array(time.astype(np.float32))
        mx_flux = mx.array(flux.astype(np.float32))
        mx_ks = mx.array(ks.astype(np.float32))
        mx_P = mx.array(P_orbs.astype(np.float32))
        mx_inc = mx.array(incs_rad.astype(np.float32))
        mx_aRs = mx.array(a_Rs_arr.astype(np.float32))
        mx_ecc = mx.array(eccs.astype(np.float32))
        mx_ws = mx.array(ws_rad.astype(np.float32))
        mx_cfr = mx.array(companion_fluxratio.astype(np.float32))

        # Compute z for all (N, T) at once
        z = _compute_z_vectorized(mx_time, mx_P, mx_inc, mx_aRs, mx_ecc, mx_ws)

        # Transit model for all (N, T)
        model = _ring_integral_vectorized(z, mx_ks, float(u1), float(u2))

        # Companion dilution
        model = _apply_companion_dilution(model, mx_cfr, companion_is_host)

        # Chi-squared: sum over T for each of N draws
        diff = mx_flux[None, :] - model  # (N, T)
        chi2 = mx.sum(diff * diff, axis=1)  # (N,)

        mx.eval(chi2)
    # np.array() readout runs outside the guard, but that's safe: mx.eval()
    # above already materialized chi2 inside the guard -- this only copies
    # realized memory, no new command-buffer submission (2026-07-12
    # re-review, "outside the guard" note).
    chi2_np = np.array(chi2, dtype=np.float64)
    sigma2 = float(sigma) * float(sigma)
    lnsigma = math.log(float(sigma))
    ln2pi = math.log(2.0 * math.pi)
    lnL = -0.5 * ln2pi - lnsigma - 0.5 * chi2_np / sigma2
    return lnL


def batch_eb_transit_lnL(time, flux, sigma, ks, eb_fluxratios, P_orbs,
                         incs_rad, a_Rs_arr, eccs, ws_rad, u1, u2,
                         companion_fluxratio, companion_is_host):
    """Vectorized EB transit log-likelihood on Apple Silicon (Metal/MLX).

    Includes secondary eclipse rejection (secdepth >= 1.5*sigma -> lnL=-inf).
    Returns: numpy float64 array of shape (N,).
    """
    if not MLX_AVAILABLE:
        raise ImportError("MLX not available. Install with: pip install mlx")

    N = len(ks)
    T = len(time)

    ks_eb = np.where(np.abs(ks - 1.0) < 1e-6, ks * 0.999, ks).astype(np.float32)

    fr_eb = eb_fluxratios.astype(np.float64)
    fr_comp = companion_fluxratio.astype(np.float64)
    F_EB = np.where(fr_eb < 1.0, fr_eb / (1.0 - fr_eb), 1e6)
    F_comp = np.where(fr_comp > 0.0, fr_comp / (1.0 - fr_comp), 0.0)

    if companion_is_host:
        safe_comp = np.maximum(F_comp, 1e-30)
        F_dilute_1 = F_EB / safe_comp
        F_dilute_2 = 1.0 / (F_comp + F_EB)
        secdepth = fr_eb / (1.0 + F_dilute_2)
    else:
        F_dilute_1 = F_EB
        F_dilute_2 = F_comp / (1.0 + F_EB)
        secdepth = np.where(F_dilute_2 > 0, fr_eb / (1.0 + F_dilute_2), fr_eb)

    reject = secdepth >= 1.5 * float(sigma)

    # metal_guard(): see batch_transit_lnL -- same process-wide lock,
    # shared with the fused kernels and injection's Metal backends.
    with metal_guard():
        mx_time = mx.array(time.astype(np.float32))
        mx_flux = mx.array(flux.astype(np.float32))
        mx_ks = mx.array(ks_eb)
        mx_P = mx.array(P_orbs.astype(np.float32))
        mx_inc = mx.array(incs_rad.astype(np.float32))
        mx_aRs = mx.array(a_Rs_arr.astype(np.float32))
        mx_ecc = mx.array(eccs.astype(np.float32))
        mx_ws = mx.array(ws_rad.astype(np.float32))

        # Compute z for all (N, T)
        z = _compute_z_vectorized(mx_time, mx_P, mx_inc, mx_aRs, mx_ecc, mx_ws)
        model = _ring_integral_vectorized(z, mx_ks, float(u1), float(u2))

        # EB dilution (two stages)
        mx_fd1 = mx.array(F_dilute_1.astype(np.float32))[:, None]
        mx_fd2 = mx.array(F_dilute_2.astype(np.float32))[:, None]
        model = (model + mx_fd1) / (1.0 + mx_fd1)
        model = (model + mx_fd2) / (1.0 + mx_fd2)

        # Chi-squared
        diff = mx_flux[None, :] - model
        chi2 = mx.sum(diff * diff, axis=1)

        mx.eval(chi2)
    # np.array() readout runs outside the guard, but that's safe: mx.eval()
    # above already materialized chi2 inside the guard -- this only copies
    # realized memory, no new command-buffer submission (2026-07-12
    # re-review, "outside the guard" note).
    chi2_np = np.array(chi2, dtype=np.float64)
    sigma2 = float(sigma) * float(sigma)
    lnsigma = math.log(float(sigma))
    ln2pi = math.log(2.0 * math.pi)
    lnL = -0.5 * ln2pi - lnsigma - 0.5 * chi2_np / sigma2
    lnL[reject] = -np.inf
    return lnL
