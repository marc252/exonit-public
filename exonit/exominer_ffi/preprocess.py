"""View packaging for ExoMiner-FFI inputs.

Mirrors the upstream ExoMiner SPOC view sizes for global/local, adds
a TGLC-only centroid view with a mask flag, and packs stellar + TCE
scalar features alongside.
"""
from __future__ import annotations

import numpy as np

from exonit.cnn_ffi_views import AUX_CHANNELS, aux_views, global_view, local_view


def em_view_spec() -> dict:
    """Return the canonical input spec for ExoMiner-FFI.

    Used by tests, the dataset builder, and the model architecture
    (Phase 3b) so all three stay aligned.

    Sizes follow Phase 2's cnn_ffi_views (global 201, local 61) which
    are tuned for FFI cadence; upstream ExoMiner uses 2001/201 for the
    much denser SPOC cadence. We pin centroid_view to local_view length
    so the model can concat the two cleanly.
    """
    from exonit.cnn_ffi_views import GLOBAL_BINS, LOCAL_BINS

    return {
        "global_view_len": int(GLOBAL_BINS),
        "local_view_len": int(LOCAL_BINS),
        "centroid_view_len": int(LOCAL_BINS),
        # odd-local, even-local, secondary-local (each LOCAL_BINS)
        "aux_n_channels": int(AUX_CHANNELS),
        "aux_view_len": int(LOCAL_BINS),
        # Tmag, Teff, logg, Rstar_solar, Mstar_solar (Mstar derived if missing)
        "stellar_n_features": 5,
        # period_d, depth_norm, duration_norm, snr
        "tce_n_features": 4,
        "mask_flags": ("centroid_mask",),
    }


def _normalise_stellar(tmag, teff, logg, rstar, mstar):
    """Centred and roughly unit-scale stellar params for stable training.

    Missing values become 0 (the centred mean). Mstar inferred from
    log(g) and Rstar when not provided (Mstar = g * R^2 / G_sun).
    """
    def _z(v, mean, scale):
        if v is None:
            return 0.0
        try:
            v = float(v)
        except (TypeError, ValueError):
            return 0.0
        if not np.isfinite(v):
            return 0.0
        return float((v - mean) / scale)

    if mstar is None or (isinstance(mstar, float) and not np.isfinite(mstar)):
        try:
            l = float(logg) if logg is not None else float("nan")
            r = float(rstar) if rstar is not None else float("nan")
        except (TypeError, ValueError):
            l = r = float("nan")
        if np.isfinite(l) and np.isfinite(r):
            mstar = (10 ** l) * r * r / (10 ** 4.44)
        else:
            mstar = float("nan")

    return np.array([
        _z(tmag, 12.0, 2.5),       # TESS magnitude
        _z(teff, 5500.0, 1500.0),
        _z(logg, 4.4, 0.5),
        _z(rstar, 1.0, 0.5),
        _z(mstar, 1.0, 0.5),
    ], dtype=np.float32)


def _normalise_tce(period_d, depth_ppm, duration_d, snr):
    """Normalise TCE features to roughly unit scale (log-space)."""
    return np.array([
        np.log10(max(float(period_d), 0.1)),
        np.log10(max(float(depth_ppm), 1.0)),
        np.log10(max(float(duration_d) * 24.0, 0.1)),  # to hours
        np.log10(max(float(snr), 1.0)),
    ], dtype=np.float32)


def _bin_centroid_view(centroid, period_d, t0_d, view_len):
    """Phase-fold centroid magnitude to a fixed-length view (in-transit zoom).

    Returns (view_array, mask) where mask is 1 if centroid contributed
    a non-zero view, 0 otherwise.
    """
    if centroid is None:
        return np.zeros(view_len, dtype=np.float32), 0
    t = np.asarray(centroid["time"])
    x = np.asarray(centroid["x"])
    y = np.asarray(centroid["y"])
    if t.size == 0 or period_d <= 0:
        return np.zeros(view_len, dtype=np.float32), 0
    mag = np.sqrt(x * x + y * y)
    # Phase fold so transit (t == t0) lands at phase=0, range [-0.5, 0.5).
    raw = ((t - t0_d) % period_d) / period_d
    phase = np.where(raw > 0.5, raw - 1.0, raw)
    in_local = np.abs(phase) < 0.05
    if not in_local.any():
        return np.zeros(view_len, dtype=np.float32), 0
    p = phase[in_local]
    m = mag[in_local]
    bin_edges = np.linspace(-0.05, 0.05, view_len + 1)
    out = np.zeros(view_len, dtype=np.float32)
    for i in range(view_len):
        sel = (p >= bin_edges[i]) & (p < bin_edges[i + 1])
        if sel.any():
            out[i] = float(np.median(m[sel]))
    if np.all(out == 0):
        return out, 0
    out = out - np.median(out)
    span = float(np.max(np.abs(out)))
    if span < 1e-12:
        # All bins identical → no information; treat as no centroid.
        return np.zeros(view_len, dtype=np.float32), 0
    out = (out / span).astype(np.float32)
    return out, 1


def make_em_record(*, time, flux, tce_period_d, tce_t0_d, tce_duration_d,
                   tce_depth_ppm, tce_snr,
                   stellar_tmag=None, stellar_teff=None, stellar_logg=None,
                   stellar_rstar=None, stellar_mstar=None,
                   centroid=None) -> dict:
    """Build one ExoMiner-FFI input record from a light curve and a TCE.

    Returns dict with keys: global_view, local_view, centroid_view,
    centroid_mask, stellar, tce.
    """
    spec = em_view_spec()
    g = np.asarray(global_view(time, flux, tce_period_d, tce_t0_d, tce_duration_d),
                   dtype=np.float32)
    l = np.asarray(local_view(time, flux, tce_period_d, tce_t0_d, tce_duration_d),
                   dtype=np.float32)
    c, mask = _bin_centroid_view(centroid, tce_period_d, tce_t0_d, spec["centroid_view_len"])
    aux = np.asarray(
        aux_views(time, flux, tce_period_d, tce_t0_d, tce_duration_d),
        dtype=np.float32)
    return {
        "global_view": g,
        "local_view": l,
        "centroid_view": c,
        "centroid_mask": int(mask),
        "aux_views": aux,
        "stellar": _normalise_stellar(stellar_tmag, stellar_teff, stellar_logg,
                                      stellar_rstar, stellar_mstar),
        "tce": _normalise_tce(tce_period_d, tce_depth_ppm, tce_duration_d, tce_snr),
    }
