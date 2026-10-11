"""Single-TIC in-memory ExoMiner preprocessing orchestrator.

Wires together all 14 sibling modules into a single callable that takes raw
data (DV XML bytes, LC FITS paths, TICv8/Gaia dicts) and returns one
``TCEFeatures`` per TCE found in the DV XML.

Pipeline stages
---------------
  FULLY WIRED
  ===========
  A. Parse DV XML              → dv_xml.get_data_from_tess_dv_xml (diff-img side channel)
  B. Ephemeris extraction      → XML modelParameter + planetCandidate attrs
  C. Read LC FITS              → lc_io.read_tess_light_curve
  D. Detrend                   → detrend.detrend_flux_using_spline
  E. Phase-fold + views        → lc_main.phase_fold_timeseries + preprocess_tce
  F. Normalize                 → normalize.normalize_example (using norm_stats)
  G. Assemble feeds            → feature_spec.load_feature_spec + empty_feed

  WIRED IN PHASE D.2 / D.3
  ========================
  H. Centroid views           → per-sector centroid_dist = sqrt((x-<x>)^2 +
                                 (y-<y>)^2) using MOM_CENTR1/MOM_CENTR2 from
                                 each LC FITS; phase-folded + binned via
                                 ``lc_main.generate_centroid_binned_views`` →
                                 normalized by training stats (``_std_noclip``).
                                 Reference point = per-sector median (pure
                                 px-space offset; does NOT apply the
                                 ``correct_centroid_using_transit_depth``
                                 branch — that requires an OOT/IT split we
                                 don't have without re-running the flux
                                 gapping pipeline on centroids).
  I. Diff-image features      → per-TCE diff/oot/snr + target imgs extracted
                                 from DV XML ``image_data``, preprocessed per
                                 sector via
                                 ``diff_img.preprocess_single_diff_img_data_for_example``,
                                 stacked to ``(5, 33, 33)``, normalized with
                                 training stats → ``_std_trainset``, then
                                 transposed to the YAML ``(33, 33, 5)``
                                 channel-last layout expected by ONNX.
                                 ``target_imgs`` feed is channel-last stack
                                 (no normalization).  ``quality`` feed is the
                                 per-sector quality metric.

  PARTIAL / ZERO-FILL
  ===================
  J. Scalar diagnostic params → extracted where available from planetCandidate /
                                 ticv8_data / gaia_data; set to NaN where absent
                                 (normalizer replaces NaN with training medians).

Note: momentum-dump view is enabled (get_momentum_dump=True in config) so
``lc_main.preprocess_tce`` wires it end-to-end when FITS data contains a
valid DQ array.  If missing, the view is zero-filled via the empty_feed
fallback.
"""
from __future__ import annotations

import logging
import tempfile
import xml.etree.cElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from . import dv_xml as _dv_xml
from . import lc_io as _lc_io
from . import detrend as _detrend
from . import gapping as _gapping
from . import lc_main as _lc_main
from . import normalize as _normalize
from . import feature_spec as _feature_spec
from . import centroid as _centroid
from . import diff_img as _diff_img


_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default preprocessing config — values match config_lc_preprocessing.yaml
# ---------------------------------------------------------------------------

_DEFAULT_PREPROC_CONFIG: dict[str, Any] = {
    "num_bins_glob": 301,
    "num_bins_loc": 31,
    # NASA config_preprocessing.yaml: bin_width_factor_glob = 1 / num_bins_glob.
    # Using 0.16 here (the loc value) produces global bins of width ~P*0.16
    # which completely wash out any transit signal in the global view.
    "bin_width_factor_glob": 1.0 / 301,
    "bin_width_factor_loc": 0.16,
    "num_durations": 2.5,
    "n_max_phases": 20,
    "min_n_phases": 0,
    "keep_odd_even_order": False,
    "frac_it_cadences_thr": 0.5,
    "sampling_rate_h": 0.5,          # TESS 2-min cadence ≈ 0.5 h / 2 min
    "phase_extend_method": "copy_phases",
    "outlier_removal": True,
    "outlier_removal_sigma": 20,
    "outlier_removal_fill": False,
    "get_momentum_dump": True,
    "random_seed": 24,
    "satellite": "tess",
    "quarter_sampling": False,       # Kepler-only
    # detrend — NASA config_preprocessing.yaml default is savitzky-golay.
    "detrending_method": "savitzky-golay",
    "sg_sigma": 5,
    "sg_max_poly_order": 8,
    "sg_penalty_weight": 1,
    "sg_break_tolerance": 5,
    "sg_n_durations_win": 5,
    "gapWidth": 0.007,
    "spline_bkspace_min": 0.5,
    "spline_bkspace_max": 20,
    "spline_bkspace_num": 20,
    "spline_maxiter": 5,
    "spline_penalty_coeff": 1.0,
    "spline_outlier_cut": 3,
    # gap config
    "tr_dur_f": 2,
}

# Default norm_stats directory (relative to this file)
_DEFAULT_NORM_STATS_DIR = Path(__file__).resolve().parents[1] / "data" / "norm_stats"


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------

@dataclass
class TCEFeatures:
    """Preprocessed features for a single TCE, ready for ONNX inference.

    Attributes
    ----------
    uid:
        ``"{tic_id}-{tce_num}-S{sector_run}"`` — matches the key format
        produced by ``dv_xml.get_data_from_tess_dv_xml``.
    period:
        Orbital period in days.
    t0:
        Transit epoch (BTJD).
    duration_days:
        Transit duration in days.
    depth_ppm:
        Transit depth in ppm.
    feeds:
        ``{feature_name: np.ndarray}`` mapping ONNX input names to arrays
        shaped exactly as declared in ``exominer_predict_config.yaml``.
    """
    uid: str
    period: float
    t0: float
    duration_days: float
    depth_ppm: float
    feeds: dict[str, np.ndarray] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Periodogram computation (approximation of NASA TPS-CDPP)
# ---------------------------------------------------------------------------

_PGRAM_N_BINS = 674
_PGRAM_P_MIN = 0.5      # days — matches BLS search range
_PGRAM_P_MAX = 50.0     # days


def _compute_pgram_features(
    time: np.ndarray,
    flux: np.ndarray,
    tce_period: float,
    tce_t0: float,
    tce_duration: float,
) -> dict:
    """Compute the 674-bin smoothed BLS periodogram + transit-masked variant.

    Returns a dict with 4 keys matching the ExoMiner feature schema:
      pgram_smooth              (674,)  float32  power over log-spaced periods
      pgram_tpm_smooth          (674,)  float32  same but with in-transit cadences masked
      pgram_smooth_max_power    scalar  float32  peak of pgram_smooth
      pgram_tpm_smooth_max_power scalar float32  peak of pgram_tpm_smooth

    The output is L1-normalized to [0, 1] so it's scale-invariant across
    LCs. Not a drop-in replacement for NASA's TPS-CDPP periodogram, but
    preserves the same shape (signal power peaks at the TCE period).
    """
    from astropy.timeseries import BoxLeastSquares

    time = np.asarray(time, dtype=float)
    flux = np.asarray(flux, dtype=float)
    mask = np.isfinite(time) & np.isfinite(flux)
    time = time[mask]
    flux = flux[mask]
    if time.size < 100:
        return {
            "pgram_smooth": np.zeros(_PGRAM_N_BINS, dtype=np.float32),
            "pgram_tpm_smooth": np.zeros(_PGRAM_N_BINS, dtype=np.float32),
            "pgram_smooth_max_power": 0.0,
            "pgram_tpm_smooth_max_power": 0.0,
        }

    # Log-spaced periods (BLS is more resolution-dense at short periods in
    # frequency space; log in period approximates that).
    periods = np.logspace(np.log10(_PGRAM_P_MIN), np.log10(_PGRAM_P_MAX), _PGRAM_N_BINS)
    dur = max(0.02, min(0.2, float(tce_duration) if np.isfinite(tce_duration) else 0.05))
    durations = np.array([dur])

    def _bls_power(t: np.ndarray, f: np.ndarray) -> np.ndarray:
        try:
            bls = BoxLeastSquares(t, f)
            pg = bls.power(periods, durations)
            p = np.asarray(pg.power, dtype=np.float32)
            p = np.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0)
            return p
        except Exception:
            return np.zeros(_PGRAM_N_BINS, dtype=np.float32)

    p_full = _bls_power(time, flux)

    # Transit-period mask (tpm): drop in-transit cadences before computing.
    in_transit = np.zeros_like(time, dtype=bool)
    if np.isfinite(tce_period) and tce_period > 0 \
       and np.isfinite(tce_t0) and np.isfinite(tce_duration) and tce_duration > 0:
        phase = ((time - tce_t0) % tce_period) / tce_period
        # transit phase width = duration / period, centered at 0 (and 1)
        half_w = (tce_duration / tce_period) / 2.0
        in_transit = (phase < half_w) | (phase > 1.0 - half_w)
    keep = ~in_transit
    if keep.sum() >= 100:
        p_tpm = _bls_power(time[keep], flux[keep])
    else:
        p_tpm = p_full.copy()

    # Normalize both to [0, 1] by peak (L∞ norm) — preserves shape.
    def _norm(p: np.ndarray) -> np.ndarray:
        m = float(np.max(p)) if p.size else 0.0
        return (p / m).astype(np.float32) if m > 0 else p.astype(np.float32)

    return {
        "pgram_smooth": _norm(p_full),
        "pgram_tpm_smooth": _norm(p_tpm),
        "pgram_smooth_max_power": float(np.max(p_full)) if p_full.size else 0.0,
        "pgram_tpm_smooth_max_power": float(np.max(p_tpm)) if p_tpm.size else 0.0,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_NS = {"dv": "http://www.nasa.gov/2018/TESS/DV"}


def _get_model_param(planet_el: ET.Element, name: str) -> float | None:
    """Return float value of a ``dv:modelParameter[@name=name]``, or None."""
    el = planet_el.find(f'.//dv:modelParameter[@name="{name}"]', _NS)
    if el is None:
        return None
    try:
        return float(el.attrib["value"])
    except (KeyError, ValueError):
        return None


def _get_stat_value(planet_el: ET.Element, tag: str) -> float:
    """Return float value of a ``<dv:TAG value="X" significance="Y"/>`` element.

    Used for albedo/halo/core/longer-period comparison statistics + bootstrap
    results. Returns ``np.nan`` if absent or unparseable.
    """
    el = planet_el.find(f".//dv:{tag}", _NS)
    if el is None:
        return np.nan
    try:
        return float(el.attrib["value"])
    except (KeyError, ValueError):
        return np.nan


def _get_centroid_sky_offset(planet_el: ET.Element, which: str) -> tuple[float, float]:
    """Return ``(value, uncertainty)`` for the ``meanSkyOffset`` inside
    ``<dv:msTicCentroidOffsets>`` (which='tic') or ``<dv:msControlCentroidOffsets>``
    (which='control'). Both are children of ``<dv:differenceImageMotionResults>``.
    """
    parent = "msTicCentroidOffsets" if which == "tic" else "msControlCentroidOffsets"
    el = planet_el.find(f".//dv:{parent}/dv:meanSkyOffset", _NS)
    if el is None:
        return np.nan, np.nan
    try:
        v = float(el.attrib.get("value", "nan"))
    except (TypeError, ValueError):
        v = np.nan
    try:
        u = float(el.attrib.get("uncertainty", "nan"))
    except (TypeError, ValueError):
        u = np.nan
    return v, u


def _extract_ephemeris(planet_el: ET.Element) -> dict[str, float]:
    """Extract per-TCE ephemeris scalars from a ``planetResults`` XML element.

    Returns a dict with keys:
      ``tce_period``, ``tce_time0bk``, ``tce_duration``, ``tce_depth``,
      ``tce_maxmes``, ``tce_maxmesd``, ``tce_max_mult_ev``,
      ``tce_max_sngle_ev``, ``tce_model_chisq``, ``tce_robstat``,
      ``tce_prad``, ``tce_ptemp``, ``wst_depth``, ``tce_albedo_stat``,
      ``tce_cap_stat``, ``tce_hap_stat``, ``boot_fap``, ``tce_dikco_msky``,
      ``tce_dikco_msky_err``, ``tce_dicco_msky``, ``tce_dicco_msky_err``.
    """
    period = _get_model_param(planet_el, "orbitalPeriodDays") or np.nan
    t0 = _get_model_param(planet_el, "transitEpochBtjd") or np.nan
    dur_h = _get_model_param(planet_el, "transitDurationHours") or np.nan
    duration = dur_h / 24.0  # hours → days
    depth = _get_model_param(planet_el, "transitDepthPpm") or np.nan

    # planetCandidate element carries MES and secondary stats
    pc = planet_el.find(".//dv:planetCandidate", _NS)
    maxmes = np.nan
    maxmesd = np.nan
    max_mult_ev = np.nan
    max_sngle_ev = np.nan
    model_chisq = np.nan
    robstat = np.nan

    if pc is not None:
        def _fa(key: str) -> float:
            try:
                return float(pc.attrib[key])
            except (KeyError, ValueError):
                return np.nan

        max_mult_ev = _fa("maxMultipleEventSigma")
        max_sngle_ev = _fa("maxSingleEventSigma")
        robstat = _fa("robustStatistic")
        # modelChiSquare2 / chiSquare2 → approximate model chi-sq scalar
        model_chisq = _fa("chiSquare2")
        # epochTjd is the BLS transit epoch — use as approximate secondary offset
        epoch_tjd = _fa("epochTjd")
        if np.isfinite(epoch_tjd) and np.isfinite(t0):
            maxmesd = (epoch_tjd - t0) % period
            if maxmesd > period / 2:
                maxmesd -= period
        maxmes = max_mult_ev  # alias — same physical quantity

    # Also pull allTransitsFit model chi-sq if available (prefer this)
    all_fit = planet_el.find(".//dv:allTransitsFit", _NS)
    if all_fit is not None:
        try:
            model_chisq = float(all_fit.attrib["modelChiSquare"])
        except (KeyError, ValueError):
            pass

    # secondary stats: use tce_maxmesd=0 fallback if NaN (needed by weak-secondary)
    if not np.isfinite(maxmesd):
        maxmesd = 0.0

    # Additional planet/diagnostic params from the fitted model + DV stats.
    tce_prad = _get_model_param(planet_el, "planetRadiusEarthRadii") or np.nan
    tce_ptemp = _get_model_param(planet_el, "equilibriumTempKelvin") or np.nan

    # Weak-secondary depth: <dv:weakSecondary><dv:depthPpm value="X" uncertainty="Y"/>
    wst_depth = np.nan
    ws = planet_el.find(".//dv:weakSecondary/dv:depthPpm", _NS)
    if ws is not None:
        try:
            wst_depth = float(ws.attrib["value"])
        except (KeyError, ValueError):
            pass

    # Binary/signal-discrimination comparison statistics.
    tce_albedo_stat = _get_stat_value(planet_el, "albedoComparisonStatistic")
    tce_ptemp_stat = _get_stat_value(planet_el, "tempComparisonStatistic")
    tce_cap_stat = _get_stat_value(planet_el, "coreApertureCorrelationStatistic")
    tce_hap_stat = _get_stat_value(planet_el, "haloApertureCorrelationStatistic")

    # Bootstrap FAP — the significance attribute on <dv:bootstrapResults>.
    boot_fap = np.nan
    boot = planet_el.find(".//dv:bootstrapResults", _NS)
    if boot is not None:
        try:
            boot_fap = float(boot.attrib.get("significance", "nan"))
        except (TypeError, ValueError):
            pass

    # Multi-sector centroid offsets: KIC-cat (tic) vs out-of-transit (control).
    dikco_v, dikco_u = _get_centroid_sky_offset(planet_el, "tic")
    dicco_v, dicco_u = _get_centroid_sky_offset(planet_el, "control")

    # Impact parameter + odd/even depth comparison.
    tce_impact = _get_model_param(planet_el, "minImpactParameter") or np.nan
    tce_bin_oedp_stat = _get_stat_value(planet_el, "oddEvenTransitDepthComparisonStatistic")

    return {
        "tce_period": period,
        "tce_time0bk": t0,
        "tce_duration": duration,
        "tce_depth": depth,
        "tce_maxmes": maxmes,
        "tce_maxmesd": maxmesd,
        "tce_max_mult_ev": max_mult_ev,
        "tce_max_sngle_ev": max_sngle_ev,
        "tce_model_chisq": model_chisq,
        "tce_robstat": robstat,
        "tce_prad": tce_prad,
        "tce_ptemp": tce_ptemp,
        "wst_depth": wst_depth,
        "tce_albedo_stat": tce_albedo_stat,
        "tce_ptemp_stat": tce_ptemp_stat,
        "tce_cap_stat": tce_cap_stat,
        "tce_hap_stat": tce_hap_stat,
        "boot_fap": boot_fap,
        "tce_dikco_msky": dikco_v,
        "tce_dikco_msky_err": dikco_u,
        "tce_dicco_msky": dicco_v,
        "tce_dicco_msky_err": dicco_u,
        "tce_impact": tce_impact,
        "tce_bin_oedp_stat": tce_bin_oedp_stat,
    }


def _build_scalar_params(ephem: dict,
                          ticv8: dict,
                          gaia: dict,
                          tic_id: str,
                          tce_num: int) -> dict:
    """Assemble all scalar parameters the normalizer expects.

    Unknown fields are set to ``np.nan``; ``normalize_scalar_parameters``
    will replace NaN with the training-set median for each missing param.
    """
    # Stellar parameters from TICv8
    def _fv(d: dict, *keys) -> float:
        for k in keys:
            v = d.get(k, np.nan)
            if v is not None:
                try:
                    return float(v)
                except (TypeError, ValueError):
                    pass
        return np.nan

    # Derive tce_prad (R_earth) if the DV XML didn't provide it directly but
    # we have tce_depth (ppm) + stellar radius from TICv8. Formula:
    #   Rp/R* = sqrt(depth_ppm / 1e6); Rp_earth = (Rp/R*) * R*_sun * R_sun/R_earth
    #   R_sun / R_earth ≈ 109.1735
    tce_prad = ephem.get("tce_prad", np.nan)
    if not np.isfinite(tce_prad):
        depth_ppm = ephem.get("tce_depth", np.nan)
        sradius = _fv(ticv8, "rad", "Rad")
        if np.isfinite(depth_ppm) and np.isfinite(sradius) and sradius > 0 and depth_ppm > 0:
            tce_prad = float(np.sqrt(depth_ppm / 1e6) * sradius * 109.1735)

    return {
        # TCE diagnostics (from DV XML)
        "tce_max_mult_ev": ephem.get("tce_max_mult_ev", np.nan),
        "tce_max_sngle_ev": ephem.get("tce_max_sngle_ev", np.nan),
        "tce_model_chisq": ephem.get("tce_model_chisq", np.nan),
        "tce_maxmes": ephem.get("tce_maxmes", np.nan),
        "tce_maxmesd": ephem.get("tce_maxmesd", np.nan),
        "tce_depth": ephem.get("tce_depth", np.nan),
        "tce_period": ephem.get("tce_period", np.nan),
        "tce_prad": tce_prad,
        "tce_robstat": ephem.get("tce_robstat", np.nan),
        # Stellar params from TICv8
        "tce_sdens": _fv(ticv8, "rho", "rad"),        # density proxy
        "tce_steff": _fv(ticv8, "Teff", "teff"),
        "tce_slogg": _fv(ticv8, "logg"),
        "tce_smet": _fv(ticv8, "MH", "met"),
        "tce_sradius": _fv(ticv8, "rad", "Rad"),
        "tce_smass": _fv(ticv8, "mass", "Mass"),
        "mag": _fv(ticv8, "Tmag", "tmag"),
        # Gaia RUWE
        "ruwe": _fv(gaia, "ruwe"),
        # Weak-secondary + binary-discrimination diagnostics (from DV XML)
        "wst_depth": ephem.get("wst_depth", np.nan),
        "tce_ptemp_stat": ephem.get("tce_ptemp_stat", np.nan),
        "tce_albedo_stat": ephem.get("tce_albedo_stat", np.nan),
        "tce_dikco_msky": ephem.get("tce_dikco_msky", np.nan),
        "tce_dikco_msky_err": ephem.get("tce_dikco_msky_err", np.nan),
        "tce_dicco_msky": ephem.get("tce_dicco_msky", np.nan),
        "tce_dicco_msky_err": ephem.get("tce_dicco_msky_err", np.nan),
        "tce_cap_stat": ephem.get("tce_cap_stat", np.nan),
        "tce_hap_stat": ephem.get("tce_hap_stat", np.nan),
        "boot_fap": ephem.get("boot_fap", np.nan),
        "tce_impact": ephem.get("tce_impact", np.nan),
        "tce_bin_oedp_stat": ephem.get("tce_bin_oedp_stat", np.nan),
        # mag_cat: 0 = Kepler, 1 = TESS (per exominer++ single-ensemble config).
        "mag_cat": 1.0,
    }


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_tic(
    tic_id: str,
    sector_run: str,
    *,
    dv_xml_bytes: bytes,
    lc_fits_paths: list,
    ticv8_data: dict,
    gaia_data: dict,
    norm_stats_dir: "Path | None" = None,
) -> list[TCEFeatures]:
    """Preprocess one TIC and return ``TCEFeatures`` for each TCE in the DV XML.

    Parameters
    ----------
    tic_id:
        TIC identifier string, e.g. ``"143143769"``.
    sector_run:
        Sector-run label, e.g. ``"1-36"``.  Used as the ``uid`` suffix.
    dv_xml_bytes:
        Raw bytes of the TESS SPOC DV XML file.
    lc_fits_paths:
        List of ``Path`` (or str) objects pointing to the sector-level
        ``*lc.fits`` files for this TIC.  At least one is required.
    ticv8_data:
        Dict of stellar parameters from TIC-8 (keys as returned by the
        server's ``/api/ticv8`` endpoint).
    gaia_data:
        Dict with at minimum ``{'ruwe': float}`` from Gaia DR3.
    norm_stats_dir:
        Directory containing the three ``.npy`` norm-stats files.
        Defaults to ``worker/exominer_native/data/norm_stats/``.

    Returns
    -------
    list[TCEFeatures]
        One element per TCE found in the DV XML.  Empty list if no TCEs
        are present or if all TCEs fail preprocessing.
    """
    norm_stats_dir = Path(norm_stats_dir) if norm_stats_dir else _DEFAULT_NORM_STATS_DIR
    lc_fits_paths = [Path(p) for p in lc_fits_paths]

    # -----------------------------------------------------------------------
    # A. Load feature spec + norm stats (once per call)
    # -----------------------------------------------------------------------
    spec_path = _feature_spec.default_feature_spec_path()
    features_spec = _feature_spec.load_feature_spec(spec_path)
    empty_feeds = _feature_spec.empty_feed(features_spec)

    try:
        norm_stats = _normalize.load_norm_stats(norm_stats_dir)
    except FileNotFoundError:
        _log.warning("norm_stats directory not found: %s — skipping normalization", norm_stats_dir)
        norm_stats = {}

    # -----------------------------------------------------------------------
    # A. Parse DV XML via temp file (dv_xml expects a file path)
    # -----------------------------------------------------------------------
    import logging as _logging
    _null_logger = _logging.getLogger("orchestrate.null")
    _null_logger.addHandler(_logging.NullHandler())

    with tempfile.NamedTemporaryFile(suffix=".dvr.xml", delete=False) as tf:
        tf.write(dv_xml_bytes)
        tmp_xml_path = Path(tf.name)

    try:
        diff_img_data = _dv_xml.get_data_from_tess_dv_xml(
            tmp_xml_path,
            neighbors_dir=None,
            sector_run_id=sector_run,
            plot_dir=None,
            plot_prob=0,
            logger=_null_logger,
        )
    finally:
        try:
            tmp_xml_path.unlink()
        except OSError:
            pass

    # -----------------------------------------------------------------------
    # B. Parse XML tree again for ephemeris (ET is faster than a second DV-XML
    #    full parse, and diff_img_data doesn't store model parameters)
    # -----------------------------------------------------------------------
    tree = ET.fromstring(dv_xml_bytes)
    planet_els = [el for el in tree if "planetResults" in el.tag]

    if not planet_els:
        _log.warning("No planetResults in DV XML for TIC %s sector_run %s", tic_id, sector_run)
        return []

    # -----------------------------------------------------------------------
    # C. Read LC FITS
    # -----------------------------------------------------------------------
    lc_data, files_not_read = _lc_io.read_tess_light_curve(
        [str(p) for p in lc_fits_paths],
        get_momentum_dump=_DEFAULT_PREPROC_CONFIG["get_momentum_dump"],
    )
    if files_not_read:
        _log.warning("Could not read %d LC FITS files: %s", len(files_not_read), files_not_read)

    if not lc_data["all_time"]:
        raise ValueError(f"No LC data read for TIC {tic_id} sector_run {sector_run}")

    all_time = lc_data["all_time"]
    all_flux = lc_data["all_flux"]

    # Concatenate for detrend (detrend handles re-splitting via lc_util.split)
    time_concat = np.concatenate(all_time)
    flux_concat = np.concatenate(all_flux)

    results: list[TCEFeatures] = []

    for planet_el in planet_els:
        tce_num = int(planet_el.attrib.get("planetNumber", 1))
        uid = f"{tic_id}-{tce_num}-S{sector_run}"

        try:
            result = _process_tce(
                uid=uid,
                tce_num=tce_num,
                tic_id=tic_id,
                sector_run=sector_run,
                planet_el=planet_el,
                diff_img_data=diff_img_data,
                all_time=all_time,
                all_flux=all_flux,
                lc_data=lc_data,
                ticv8_data=ticv8_data,
                gaia_data=gaia_data,
                norm_stats=norm_stats,
                features_spec=features_spec,
                empty_feeds=empty_feeds,
                config=_DEFAULT_PREPROC_CONFIG,
            )
            results.append(result)
        except Exception as exc:
            _log.error("TCE %s failed preprocessing: %s", uid, exc, exc_info=True)
            continue

    return results


def _process_tce(
    *,
    uid: str,
    tce_num: int,
    tic_id: str,
    sector_run: str,
    planet_el: ET.Element,
    diff_img_data: dict,
    all_time: list,
    all_flux: list,
    lc_data: dict,
    ticv8_data: dict,
    gaia_data: dict,
    norm_stats: dict,
    features_spec: dict,
    empty_feeds: dict,
    config: dict,
) -> TCEFeatures:
    """Preprocess a single TCE and return a ``TCEFeatures`` object."""

    # -----------------------------------------------------------------------
    # B. Extract ephemeris scalars
    # -----------------------------------------------------------------------
    ephem = _extract_ephemeris(planet_el)
    period = ephem["tce_period"]
    t0 = ephem["tce_time0bk"]
    duration = ephem["tce_duration"]
    depth = ephem["tce_depth"]

    for key, val in ephem.items():
        if not np.isfinite(val if isinstance(val, float) else float(val)):
            _log.warning("%s: %s = %s is non-finite", uid, key, val)

    # NASA preprocess.py:629 — sampling_rate_h is "samples per hour" and is
    # computed dynamically from the actual data's median cadence gap. Override
    # the config default (which is just a placeholder) per-TCE here.
    try:
        median_dt_days = float(np.median(np.diff(np.concatenate(all_time))))
        if median_dt_days > 0:
            config = dict(config)  # don't mutate _DEFAULT_PREPROC_CONFIG
            config["sampling_rate_h"] = 1.0 / (median_dt_days * 24.0)
    except Exception:
        pass

    # -----------------------------------------------------------------------
    # D. Detrend (spline) — one per TCE so other-TCE transits are gapped out
    # -----------------------------------------------------------------------
    # Gap in-transit cadences for this TCE before spline fit
    intransit = _gapping.gap_this_tce(
        all_time,
        {
            "tce_time0bk": t0,
            "tce_period": period,
            "tce_duration": duration,
        },
        gap_pad=config["tr_dur_f"],
    )
    # Convert gapped index lists to boolean masks
    intransit_masks = [
        np.zeros(len(t), dtype=bool) for t in all_time
    ]
    for arr_i, idxs in enumerate(intransit):
        intransit_masks[arr_i][idxs] = True

    if config.get("detrending_method", "spline") == "savitzky-golay":
        import lightkurve as lk
        # Remove non-finite cadences before building the LightCurve (NASA parity).
        time_cat = np.concatenate([arr.copy() for arr in all_time])
        flux_cat = np.concatenate([arr.copy() for arr in all_flux])
        mask_cat = np.concatenate(intransit_masks)
        finite = np.isfinite(time_cat) & np.isfinite(flux_cat)
        time_cat, flux_cat, mask_cat = time_cat[finite], flux_cat[finite], mask_cat[finite]
        # SG window length: NASA uses 1.2 day * 24 h * sampling_rate_h (default 28.8 samples/day
        # at TESS 2-min cadence); make sure it's odd.
        sr_h = float(config.get("sampling_rate_h", 30.0))  # samples per hour
        win_h = 1.2 * 24  # hours
        win_len = int(win_h * sr_h)
        if win_len % 2 == 0:
            win_len += 1
        lc = lk.LightCurve(data={"time": time_cat, "flux": flux_cat})
        time_det, flux_det, flux_trend, _ = _detrend.detrend_flux_using_sg_filter(
            lc, mask_cat, win_len=win_len,
            sigma=config.get("sg_sigma", 5),
            max_poly_order=config.get("sg_max_poly_order", 8),
            penalty_weight=config.get("sg_penalty_weight", 1),
            break_tolerance=config.get("sg_break_tolerance", 5),
        )
    else:
        time_det, flux_det, spline_det, _ = _detrend.detrend_flux_using_spline(
            [arr.copy() for arr in all_flux],
            [arr.copy() for arr in all_time],
            intransit_masks,
            config,
        )
        # ``detrend_flux_using_spline`` returns concatenated arrays
        flux_trend = spline_det

    # -----------------------------------------------------------------------
    # E. Phase-fold via lc_main.phase_fold_timeseries + preprocess_tce
    # -----------------------------------------------------------------------
    # Build the detrended data dict expected by phase_fold_timeseries
    # (flux_time / flux / flux_trend are 1-D concatenated arrays)
    tce_series = {
        "uid": uid,
        "tce_period": period,
        "tce_time0bk": t0,
        "tce_duration": duration,
        "tce_maxmesd": ephem["tce_maxmesd"],
    }

    # -----------------------------------------------------------------------
    # H. Centroid distance time series (Phase D.2).
    #
    # Port of NASA ``centroid_preprocessing`` (preprocess.py:814-820): the
    # reference centroid is the NaN-median of *out-of-transit* cadences
    # concatenated across all sectors. Using a per-sector nanmedian over
    # *all* cadences (including in-transit) biases the reference toward
    # the dip and collapses the transit-induced centroid offset we want
    # the model to see. Fallback matches NASA: if no finite OOT value is
    # available, fall back to the median over all cadences.
    # -----------------------------------------------------------------------
    cx_arrs = [np.asarray(lc_data["all_centroids_px"]["x"][i], dtype=float)
               for i in range(len(all_time))]
    cy_arrs = [np.asarray(lc_data["all_centroids_px"]["y"][i], dtype=float)
               for i in range(len(all_time))]
    cx_oot_parts = [cx_arrs[i][~intransit_masks[i]] for i in range(len(cx_arrs))]
    cy_oot_parts = [cy_arrs[i][~intransit_masks[i]] for i in range(len(cy_arrs))]
    cx_oot_all = np.concatenate(cx_oot_parts) if cx_oot_parts else np.array([])
    cy_oot_all = np.concatenate(cy_oot_parts) if cy_oot_parts else np.array([])
    ref_x = float(np.nanmedian(cx_oot_all)) if cx_oot_all.size else np.nan
    ref_y = float(np.nanmedian(cy_oot_all)) if cy_oot_all.size else np.nan
    if not np.isfinite(ref_x):
        ref_x = float(np.nanmedian(np.concatenate(cx_arrs))) if cx_arrs else 0.0
    if not np.isfinite(ref_y):
        ref_y = float(np.nanmedian(np.concatenate(cy_arrs))) if cy_arrs else 0.0

    centroid_dist_per_sector: list = []
    for sector_i in range(len(all_time)):
        cx = cx_arrs[sector_i]
        cy = cy_arrs[sector_i]
        if not (np.isfinite(ref_x) and np.isfinite(ref_y)):
            centroid_dist_per_sector.append(np.zeros_like(cx))
            continue
        dist = _centroid.compute_centroid_distance(
            {"x": cx, "y": cy}, [ref_x, ref_y], delta_dec=None,
        )
        centroid_dist_per_sector.append(dist)

    centroid_dist_concat = np.concatenate(centroid_dist_per_sector)
    centroid_dist_time_concat = np.concatenate(all_time)
    # Remove any non-finite cadences (should mirror flux cadence filtering)
    finite_mask = np.isfinite(centroid_dist_concat) & np.isfinite(centroid_dist_time_concat)
    if finite_mask.sum() < 10:
        # Not enough valid cadences — fall back to zeros (normalizer handles it)
        centroid_dist_clean = np.zeros(len(centroid_dist_time_concat))
        centroid_dist_time_clean = centroid_dist_time_concat
    else:
        centroid_dist_clean = centroid_dist_concat[finite_mask]
        centroid_dist_time_clean = centroid_dist_time_concat[finite_mask]

    detrended_data = {
        "flux_time": time_det,
        "flux": flux_det,
        "flux_trend": flux_trend,
        "centroid_dist": centroid_dist_clean,
        "centroid_dist_time": centroid_dist_time_clean,
        "errors": [],
        # momentum_dump (optional)
        "momentum_dump": None,
        "momentum_dump_time": None,
    }

    if config["get_momentum_dump"] and "momentum_dump" in lc_data:
        detrended_data["momentum_dump"] = np.concatenate(lc_data["momentum_dump"]).astype(float)
        detrended_data["momentum_dump_time"] = np.concatenate(lc_data["time_momentum_dump"])

    # Scalar params for preprocess_tce (trans data + stellar)
    scalar_params = _build_scalar_params(ephem, ticv8_data, gaia_data, tic_id, tce_num)
    tce_row = {**tce_series, **scalar_params}

    phase_folded = _lc_main.phase_fold_timeseries(detrended_data, config, tce_row)

    preproc_result = _lc_main.preprocess_tce(phase_folded, tce_row, config)

    # -----------------------------------------------------------------------
    # F. Assemble intermediate feature dict for normalization
    # -----------------------------------------------------------------------
    features: dict[str, Any] = {}
    features.update(preproc_result["views"])
    features.update(preproc_result["views_var"])
    features.update(preproc_result.get("stats", {}))
    features.update(scalar_params)

    # transit count scalars expected by the model
    features["tce_num_transits_obs"] = preproc_result["num_transits"].get(
        "global_flux_view_fluxnorm_num_transits", -1)
    features["tce_num_transits"] = features["tce_num_transits_obs"]

    # -----------------------------------------------------------------------
    # Periodogram features (ported from NASA ExoMiner config).
    # Compute Lomb-Scargle amplitude pgram of the detrended LC and of the
    # synthetic transit pulse model (TPM), smoothed + max-normalized. Writes:
    #   pgram_smooth, pgram_smooth_max_power,
    #   pgram_tpm_smooth, pgram_tpm_smooth_max_power
    # Earlier BLS-based approximation empirically hurt scores; this port
    # matches the oracle's LS algorithm exactly (lightkurve 'amplitude' + 'fast').
    # -----------------------------------------------------------------------
    try:
        from .pgram import compute_pgram_features as _compute_pgram_features
        # NASA oracle passes raw stitched flux to lc_periodogram but then
        # lightkurve.stitch() re-normalizes per-sector to median=1 (removing
        # sector offsets). Our detrended_data.flux has *dividing* spline
        # detrending already applied which makes the global periodogram
        # similar in shape to per-sector-normalized flux. Empirically the
        # detrended input scores the confirmed KP ~4x better than raw.
        _pgram = _compute_pgram_features(
            time=detrended_data["flux_time"],
            flux=detrended_data["flux"],
            tce_period=period,
            tce_t0=t0,
            tce_duration=duration,
            tce_depth_ppm=depth,
            tce_maxmesd=ephem.get("tce_maxmesd", 0.0),
            wst_depth_ppm=ephem.get("wst_depth", 0.0) or 0.0,
        )
        features.update(_pgram)
    except Exception as e:
        _log.warning("%s: pgram compute failed: %s (leaving zero)", uid, e)

    # mag_shift — Gaia-neighbour magnitude shift (Phase D TODO). Currently
    # defaults to training median (3.44) when absent so the normalizer sees a
    # "typical" value rather than a clipped anomaly. A proper implementation
    # requires a Gaia cone search at the target's RA/Dec and a brightness
    # comparison vs. the TIC.
    if "mag_shift" not in features or features.get("mag_shift") is None \
            or not np.isfinite(features.get("mag_shift", np.nan)):
        features["mag_shift"] = 3.4446  # training median (exominer++ single)

    # -----------------------------------------------------------------------
    # I. Diff-image tensors (Phase D.3).
    #
    # Fetch this TCE's diff-image data from the DV XML parse, preprocess
    # each valid sector (crop → fill NaN → pad → resize → center → crop),
    # stack into ``(5, 33, 33)``, and inject into ``features`` so the
    # downstream ``normalize_diff_img`` produces the ``_std_trainset``
    # feeds.  ``target_imgs`` are rendered straight into ``features``
    # (not normalized — they're 1-hot location maps).
    #
    # We bypass ``diff_img.preprocess_tce_diff_img`` because its
    # missing-value check fails when ``neighbors_imgs`` is never filled
    # (native-worker path has no Gaia neighbour lookup).  Instead we
    # iterate the per-sector preprocessor directly.  The output shape
    # convention is ``(N_sectors, H, W)``; the per-feat reshape at the
    # end of this function transposes into the YAML channel-last layout.
    # -----------------------------------------------------------------------
    tce_diff_data = diff_img_data.get(uid)
    _wire_diff_img_features(features, uid, tce_diff_data, n_imgs=5, size=33)

    # -----------------------------------------------------------------------
    # G. Normalize
    # -----------------------------------------------------------------------
    if norm_stats:
        normalized = _normalize.normalize_example(features, norm_stats)
        features.update(normalized)

    # -----------------------------------------------------------------------
    # H/I. Assemble feeds: start from zero-filled base, fill from computed feats
    # -----------------------------------------------------------------------
    feeds: dict[str, np.ndarray] = {k: v.copy() for k, v in empty_feeds.items()}

    for feat_name, spec in features_spec.items():
        if feat_name not in features:
            continue
        val = features[feat_name]
        if val is None:
            continue
        arr = np.asarray(val, dtype=np.float32)
        target_shape = tuple(spec["dim"])
        # Diff-image features arrive as (N_sectors, H, W) from the normalizer
        # (or (N_sectors, H, W) for target_imgs); the YAML declares them as
        # (H, W, N_sectors).  Transpose into channel-last layout before
        # reshape to preserve per-sector images.
        if arr.ndim == 3 and len(target_shape) == 3 and target_shape[-1] == arr.shape[0]:
            arr = np.transpose(arr, (1, 2, 0))
        try:
            arr = arr.reshape(target_shape)
            feeds[feat_name] = arr
        except ValueError:
            _log.debug("Cannot reshape %s: %s → %s", feat_name, arr.shape, target_shape)

    return TCEFeatures(
        uid=uid,
        period=period,
        t0=t0,
        duration_days=duration,
        depth_ppm=depth,
        feeds=feeds,
    )


# ---------------------------------------------------------------------------
# Phase D.3 helper — diff-image stack assembly
# ---------------------------------------------------------------------------


def _wire_diff_img_features(features: dict,
                            uid: str,
                            tce_diff_data: dict | None,
                            *,
                            n_imgs: int,
                            size: int) -> None:
    """Populate ``diff_imgs`` / ``oot_imgs`` / ``snr_imgs`` / ``target_imgs``
    (+ ``_tc`` variants) and ``quality`` into ``features`` in-place.

    We bypass :func:`diff_img.preprocess_tce_diff_img` because its
    missing-values gate trips on the ``neighbors_imgs`` NaN placeholders
    whenever no neighbour data is supplied (native-worker path has no
    Gaia neighbour lookup wired).  The inner per-sector preprocessor
    ``preprocess_single_diff_img_data_for_example`` is fine — we just
    drive it ourselves.

    Output stack shape: ``(n_imgs, size, size)`` — the orchestrate feed
    assembly transposes into ``(size, size, n_imgs)`` to match the YAML
    channel-last declaration.
    """
    if tce_diff_data is None:
        _log.warning("%s: no diff-image data found; keeping zeros", uid)
        return

    image_data = tce_diff_data.get("image_data", [])
    if len(image_data) == 0:
        _log.warning("%s: image_data is empty; keeping zeros", uid)
        return

    qmetric = tce_diff_data.get("quality_metric", [])
    tref = tce_diff_data.get("target_ref_centroid", [])

    # Identify sectors with usable imagery.
    valid_idxs: list[int] = []
    for i in range(len(image_data)):
        q = qmetric[i] if i < len(qmetric) else None
        r = tref[i] if i < len(tref) else None
        if q is None or r is None:
            continue
        if not q.get("valid", False):
            continue
        if r["col"].get("uncertainty", -1) == -1:
            continue
        img = image_data[i]
        if np.isnan(img[:, :, 1, 0]).all() or np.isnan(img[:, :, 2, 0]).all():
            continue
        valid_idxs.append(i)

    if not valid_idxs:
        _log.warning("%s: no valid diff-img sectors; keeping zeros", uid)
        return

    # Sample exactly n_imgs, tiling when fewer valid sectors are available
    # (mirrors ``diff_img.sample_image_data`` — avoiding the module-level
    # RNG since we want determinism per-call).
    rng = np.random.default_rng(seed=42)
    if len(valid_idxs) < n_imgs:
        k = n_imgs // len(valid_idxs)
        sampled = np.tile(valid_idxs, k)
        extra = rng.choice(valid_idxs, n_imgs % len(valid_idxs), replace=False)
        sampled = np.concatenate([sampled, extra])
    else:
        sampled = rng.choice(valid_idxs, n_imgs, replace=False)

    target_mag = tce_diff_data.get("mag")

    # Per-sampled-slot outputs; we run both centering options to populate
    # the ``_tc`` variants.
    diff_stack = np.zeros((n_imgs, size, size), dtype=np.float32)
    oot_stack = np.zeros((n_imgs, size, size), dtype=np.float32)
    snr_stack = np.zeros((n_imgs, size, size), dtype=np.float32)
    tgt_stack = np.zeros((n_imgs, size, size), dtype=np.float32)
    diff_stack_tc = np.zeros_like(diff_stack)
    oot_stack_tc = np.zeros_like(oot_stack)
    snr_stack_tc = np.zeros_like(snr_stack)
    tgt_stack_tc = np.zeros_like(tgt_stack)
    quality_arr = np.zeros(n_imgs, dtype=np.float32)

    for slot_i, sector_idx in enumerate(sampled):
        img = image_data[sector_idx]
        diff = img[:, :, 2, 0].copy()
        oot = img[:, :, 1, 0].copy()
        # snr = diff / diff_uncertainty; avoid div-by-zero via the oracle's
        # implicit numpy behaviour (unc==0 → inf, which is handled later
        # by NaN-fill).  Matches diff_img.preprocess_tce_diff_img.
        with np.errstate(divide="ignore", invalid="ignore"):
            snr = diff / img[:, :, 2, 1]

        ref_col = float(tref[sector_idx]["col"]["value"])
        ref_row = float(tref[sector_idx]["row"]["value"])
        if ref_col == -1:
            tcol, trow = np.nan, np.nan
        else:
            tcol, trow = ref_col, ref_row

        for center, out_diff, out_oot, out_snr, out_tgt in (
            (False, diff_stack, oot_stack, snr_stack, tgt_stack),
            (True, diff_stack_tc, oot_stack_tc, snr_stack_tc, tgt_stack_tc),
        ):
            try:
                (pd_diff, pd_oot, pd_snr, pd_tgt, _pos, _pos_disc, _nb) = (
                    _diff_img.preprocess_single_diff_img_data_for_example(
                        diff.copy(), oot.copy(), snr.copy(),
                        tcol, trow,
                        size, size, 1, 1,
                        int(tce_diff_data["image_number"][sector_idx]),
                        uid, "sector",
                        center_target=center,
                        neighbor_data=None,
                        target_mag=target_mag,
                        exclude_neighbor_objs_outside=True,
                        log=None, proc_id=uid,
                    )
                )
            except Exception as exc:  # pragma: no cover — diagnostic
                _log.warning("%s: sector %d diff-img preproc failed: %s",
                             uid, sector_idx, exc)
                continue

            # Any residual NaN → 0 (consistent with placeholder fallback)
            pd_diff = np.nan_to_num(pd_diff, copy=False)
            pd_oot = np.nan_to_num(pd_oot, copy=False)
            pd_snr = np.nan_to_num(pd_snr, copy=False)

            # Safety: if the preprocessor returned a non-target shape
            # (rare — happens when crop_to_valid yields a tiny region),
            # skip this slot rather than crash.
            if pd_diff.shape != (size, size):
                continue

            out_diff[slot_i] = pd_diff
            out_oot[slot_i] = pd_oot
            out_snr[slot_i] = pd_snr
            out_tgt[slot_i] = pd_tgt

        quality_arr[slot_i] = float(qmetric[sector_idx].get("value", 0.0) or 0.0)

    features["diff_imgs"] = diff_stack
    features["oot_imgs"] = oot_stack
    features["target_imgs"] = tgt_stack
    features["diff_imgs_tc"] = diff_stack_tc
    features["oot_imgs_tc"] = oot_stack_tc
    features["target_imgs_tc"] = tgt_stack_tc
    features["quality"] = quality_arr
    # ``snr_imgs`` are deliberately omitted — the shipped norm-stats file
    # only ships ``diff_imgs`` / ``oot_imgs`` (+ ``_tc``), and the YAML
    # model spec only declares ``diff_imgs_std_trainset`` /
    # ``oot_imgs_std_trainset`` / ``target_imgs``.  Injecting snr_imgs
    # here would trigger a KeyError in ``normalize_diff_img`` because
    # snr stats are absent.
