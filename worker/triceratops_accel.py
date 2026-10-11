"""Accelerated replacements for slow TRICERATOPS functions.

Replaces:
1. calc_depths: dblquad (numerical integration) → analytical erf (~688x faster)
2. Gauss2D integral over rectangle has exact solution using error functions

These monkey-patch TRICERATOPS at import time.

Note: as of triceratops master (2026-04), upstream calc_depths still
uses scipy.integrate.dblquad. PR #45 was merged but the analytical
implementation is not yet in the target class. This patch remains
necessary for production performance.
"""

import numpy as np
from scipy.special import erf

# Try OpenCL for batch gaussian flux computation
_opencl_available = False
try:
    import pyopencl as cl
    _ctx = cl.create_some_context(interactive=False)
    _queue = cl.CommandQueue(_ctx)
    _opencl_available = True
except Exception:
    pass


def gaussian_flux_analytical(mu_x, mu_y, sigma, A, x_lo, x_hi, y_lo, y_hi):
    """Analytical integral of 2D Gaussian over a rectangle.

    The integral of A/(2*pi*sigma^2) * exp(-((x-mu_x)^2+(y-mu_y)^2)/(2*sigma^2))
    over [x_lo, x_hi] x [y_lo, y_hi] equals:

    (A/4) * [erf((x_hi-mu_x)/(sqrt(2)*sigma)) - erf((x_lo-mu_x)/(sqrt(2)*sigma))]
           * [erf((y_hi-mu_y)/(sqrt(2)*sigma)) - erf((y_lo-mu_y)/(sqrt(2)*sigma))]
    """
    s = np.sqrt(2) * sigma
    return (A / 4.0) * (erf((x_hi - mu_x) / s) - erf((x_lo - mu_x) / s)) * \
                        (erf((y_hi - mu_y) / s) - erf((y_lo - mu_y) / s))


def gaussian_flux_batch(mu_x, mu_y, sigma, A, pixels):
    """Compute total flux for a star through all aperture pixels at once.

    Args:
        mu_x, mu_y: Star position in pixel coords (scalars)
        sigma: PSF width (scalar)
        A: Star amplitude (scalar)
        pixels: Nx2 array of pixel center coordinates

    Returns:
        Total flux through all pixels (scalar)
    """
    x_lo = pixels[:, 0] - 0.5
    x_hi = pixels[:, 0] + 0.5
    y_lo = pixels[:, 1] - 0.5
    y_hi = pixels[:, 1] + 0.5
    return np.sum(gaussian_flux_analytical(mu_x, mu_y, sigma, A,
                                           x_lo, x_hi, y_lo, y_hi))


if _opencl_available:
    _KERNEL_SRC = """
    __kernel void gaussian_flux(
        __global const float *mu_x,    // [N_stars]
        __global const float *mu_y,    // [N_stars]
        __global const float *A,       // [N_stars]
        __global const float *pix_x,   // [N_pix] pixel centers
        __global const float *pix_y,   // [N_pix]
        const float sigma,
        const int N_pix,
        __global float *flux_out       // [N_stars] total flux per star
    ) {
        int sid = get_global_id(0);  // star index
        float s = sqrt(2.0f) * sigma;
        float total = 0.0f;
        for (int p = 0; p < N_pix; p++) {
            float dx_hi = (pix_x[p] + 0.5f - mu_x[sid]) / s;
            float dx_lo = (pix_x[p] - 0.5f - mu_x[sid]) / s;
            float dy_hi = (pix_y[p] + 0.5f - mu_y[sid]) / s;
            float dy_lo = (pix_y[p] - 0.5f - mu_y[sid]) / s;
            total += (A[sid] / 4.0f)
                   * (erf(dx_hi) - erf(dx_lo))
                   * (erf(dy_hi) - erf(dy_lo));
        }
        flux_out[sid] = total;
    }
    """
    _program = cl.Program(_ctx, _KERNEL_SRC).build()


def calc_depths_fast(target, tdepth, all_ap_pixels=None):
    """Fast replacement for triceratops.target.calc_depths.

    Uses analytical erf instead of numerical dblquad integration.
    With OpenCL GPU: batches all stars × pixels in one kernel call.
    Without GPU: vectorized numpy (still ~100x faster than dblquad).
    """
    stars = target.stars
    pix_coords = target.pix_coords

    if all_ap_pixels is None:
        all_ap_pixels = []
        for i in range(len(pix_coords)):
            target_pixel = np.round(pix_coords[i][0])
            this_ap = np.array([
                np.repeat(
                    np.arange(target_pixel[0] - 2, target_pixel[0] + 3, 1),
                    5),
                np.tile(
                    np.arange(target_pixel[1] - 2, target_pixel[1] + 3, 1),
                    5)
            ]).T
            all_ap_pixels.append(this_ap)

    N_stars = len(stars)
    N_apertures = len(all_ap_pixels)
    sigma = 0.75

    # Precompute star amplitudes (relative to brightest)
    Tmags = stars["Tmag"].values
    A_vals = 10 ** ((np.min(Tmags) - Tmags) / 2.5)

    rel_flux_per_aperture = np.zeros((N_apertures, N_stars))
    flux_ratio_per_aperture = np.zeros((N_apertures, N_stars))

    for k in range(N_apertures):
        pixels = all_ap_pixels[k]

        if _opencl_available and N_stars > 5:
            # GPU path: one kernel call for all stars
            mu_xs = np.array(pix_coords[k][:, 1], dtype=np.float32)
            mu_ys = np.array(pix_coords[k][:, 0], dtype=np.float32)
            A_arr = np.array(A_vals, dtype=np.float32)
            pix_x = np.array(pixels[:, 1], dtype=np.float32)
            pix_y = np.array(pixels[:, 0], dtype=np.float32)
            flux_out = np.zeros(N_stars, dtype=np.float32)

            mf = cl.mem_flags
            buf_mu_x = cl.Buffer(_ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=mu_xs)
            buf_mu_y = cl.Buffer(_ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=mu_ys)
            buf_A = cl.Buffer(_ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=A_arr)
            buf_px = cl.Buffer(_ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=pix_x)
            buf_py = cl.Buffer(_ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=pix_y)
            buf_out = cl.Buffer(_ctx, mf.WRITE_ONLY, flux_out.nbytes)

            _program.gaussian_flux(
                _queue, (N_stars,), None,
                buf_mu_x, buf_mu_y, buf_A, buf_px, buf_py,
                np.float32(sigma), np.int32(len(pixels)), buf_out)
            cl.enqueue_copy(_queue, flux_out, buf_out)

            rel_flux_per_aperture[k, :] = flux_out
        else:
            # CPU vectorized path
            for i in range(N_stars):
                mu_x = pix_coords[k][i, 1]
                mu_y = pix_coords[k][i, 0]
                rel_flux_per_aperture[k, i] = gaussian_flux_batch(
                    mu_x, mu_y, sigma, A_vals[i], pixels)

        total = np.sum(rel_flux_per_aperture[k])
        if total > 0:
            flux_ratio_per_aperture[k, :] = rel_flux_per_aperture[k] / total

    # Average across apertures
    flux_ratios = np.mean(flux_ratio_per_aperture, axis=0)
    stars["fluxratio"] = flux_ratios

    # Calculate transit depth each star would need
    tdepths = np.where(flux_ratios > 0, tdepth / flux_ratios, 0)
    stars["tdepth"] = tdepths

    target.stars = stars


def apply_patches():
    """Monkey-patch TRICERATOPS with accelerated functions."""
    from triceratops import triceratops as tri

    # Replace calc_depths
    original_calc_depths = tri.target.calc_depths

    def patched_calc_depths(self, tdepth, all_ap_pixels=None):
        calc_depths_fast(self, tdepth, all_ap_pixels)

    tri.target.calc_depths = patched_calc_depths

    gpu_status = "OpenCL GPU" if _opencl_available else "CPU vectorized"
    print(f"  TRICERATOPS accelerated: analytical erf ({gpu_status})")
