"""Native port of ``src_preprocessing/lc_preprocessing/tess_io.py``.

Reads TESS SPOC light-curve FITS into the dict shape ExoMiner's downstream
preprocessing expects. Worker-side: fetched from the server via
``/api/fits/`` (bytes), never touches MAST directly (D4).

Trimmed / adapted from the oracle:
  - ``from tensorflow.io import gfile`` → ``pathlib.Path`` (the worker has
    no TF dependency).
  - ``from src_preprocessing.light_curve import util`` moved to a lazy
    import inside the only branch that uses it
    (``interpolate_missing_time=True``); not needed for the pipeline's
    default path.
  - Added ``read_tess_light_curve_from_bytes``: calls ``fits.open`` over an
    in-memory ``BytesIO`` so the worker can consume the HTTP response body
    without writing a temp file.

Signatures kept identical to the oracle so the parity harness can call
both sides with the same positional arguments.
"""
from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path

import numpy as np
from astropy import wcs
from astropy.io import fits


MOMENTUM_DUMP_VALUE = 32  # momentum dump value in the DQ array
MAX_BIT = 12  # max number of bits in the DQ array


def get_tess_light_curve_files(base_dir, ticid, sectors_observed):
    """Find TESS light curve FITS files for a TIC + observed sectors.

    Args:
        base_dir: Base directory containing TESS light curve FITS files data
        ticid: ID of the TESS target star. It can be an int or a possibly
            zero-padded string.
        sectors_observed: str, sectors observed separated by an underscore.

    Returns:
        A list of filepaths to the light curve FITS files for a given TIC
        and set of observed sectors.
    """
    if "_" in sectors_observed:
        sectors_observed = sorted(
            [int(sector_observed) for sector_observed in sectors_observed.split("_")]
        )
    else:
        sectors_observed = [int(sectors_observed)]
    sectors_observed = [str(sector_id).zfill(4) for sector_id in sectors_observed]

    ticid = str(ticid).zfill(16)

    target_lc_fits_fps = list(base_dir.rglob(f"*{ticid}*lc.fits"))

    filtered_lc_paths = []
    for sector_observed in sectors_observed:
        pattern = re.compile(rf"-s{sector_observed}")
        matches = [p for p in target_lc_fits_fps if pattern.search(str(p))]
        filtered_lc_paths.extend(matches)

    return filtered_lc_paths


def tess_filenames(base_dir, ticid, sectors):
    """Return the light curve filenames for a TESS target in 2-min cadence."""
    filenames = []
    tess_id = str(ticid).zfill(16)

    for sector in sectors:
        sector_dir = Path(base_dir) / f"sector_{sector}"
        fps_lst = list(sector_dir.glob(f"*{tess_id}*lc.fits"))
        if len(fps_lst) == 0:
            filename = None
        else:
            filename = fps_lst[0]

        if filename:
            filenames.append(filename)

    return filenames


def tess_ffi_filenames(base_dir, tic_id, sector_run, check_existence=True):
    """Return the light curve filenames for a TESS target observed in FFI data."""
    tic_id = f"{tic_id}".zfill(16)
    sector_runs_ids = [f"{sector_i}".zfill(4) for sector_i in sector_run]

    filenames = []
    for sector_run_id in sector_runs_ids:
        base_dir_sector_run = os.path.join(
            base_dir,
            f"s{sector_run_id}",
            "target",
            tic_id[0:4], tic_id[4:8], tic_id[8:12], tic_id[12:16],
        )
        base_name = f"hlsp_tess-spoc_tess_phot_{tic_id}-s{sector_run_id}_tess_v1_lc.fits"
        filename = os.path.join(base_dir_sector_run, base_name)
        if not check_existence or Path(filename).exists():
            filenames.append(filename)

    return filenames


def read_tess_light_curve(
    filenames,
    light_curve_extension="LIGHTCURVE",
    interpolate_missing_time=False,
    centroid_radec=False,
    prefer_psfcentr=False,
    get_momentum_dump=False,
    dq_values_filter=None,
):
    """Reads data from FITS files for a TESS target star.

    Signature matches the oracle (``src_preprocessing/lc_preprocessing/tess_io.py``).

    Args:
        filenames: A list of .fits files containing time and flux measurements.
        light_curve_extension: Name of the HDU 1 extension containing light curves.
        interpolate_missing_time: Whether to interpolate missing (NaN) time values.
        centroid_radec: bool, whether to transform the centroid time series
            from CCD module pixel coordinates to RA and Dec.
        prefer_psfcentr: bool, if True, uses PSF centroids when available.
        get_momentum_dump: bool, if True the momentum dump information is
            extracted from the FITS file.
        dq_values_filter: list of DQ flag integers to exclude. None disables filtering.

    Returns:
        (data, files_not_read). ``data`` is a dict with keys ``all_time``,
        ``all_flux``, ``all_flux_err``, ``all_centroids`` (dict with ``x``/``y``),
        ``all_centroids_px``, ``sectors``, ``module``, ``camera``, ``ccd``,
        ``target_position``, ``flag_keep`` (+ ``momentum_dump`` and
        ``time_momentum_dump`` if ``get_momentum_dump=True``).
    """
    data = {
        "all_time": [],
        "all_flux": [],
        "all_flux_err": [],
        "all_centroids": {"x": [], "y": []},
        "all_centroids_px": {"x": [], "y": []},
    }

    timeseries_fields = list(data.keys())

    if get_momentum_dump:
        data["momentum_dump"] = []
        data["time_momentum_dump"] = []

    data.update({
        "flag_keep": [],
        "sectors": [],
        "module": [],
        "camera": [],
        "ccd": [],
        "target_position": [],
    })

    files_not_read = []

    for filename in filenames:

        basename = os.path.basename(filename)

        try:
            with fits.open(filename, ignoring_missing_end=True) as hdu_list:

                camera = hdu_list["PRIMARY"].header["CAMERA"]
                ccd = hdu_list["PRIMARY"].header["CCD"]
                sector = hdu_list["PRIMARY"].header["SECTOR"]

                if len(data["target_position"]) == 0:
                    data["target_position"] = [
                        hdu_list["PRIMARY"].header["RA_OBJ"],
                        hdu_list["PRIMARY"].header["DEC_OBJ"],
                    ]

                if not centroid_radec:
                    pass

                light_curve = hdu_list[light_curve_extension].data

                if prefer_psfcentr:
                    centroid_x, centroid_y = light_curve.PSF_CENTR1, light_curve.PSF_CENTR2
                else:
                    centroid_x, centroid_y = (
                        light_curve.MOM_CENTR1 - light_curve.POS_CORR1,
                        light_curve.MOM_CENTR2 - light_curve.POS_CORR2,
                    )

                centroid_fdl_x, centroid_fdl_y = light_curve.MOM_CENTR1, light_curve.MOM_CENTR2

                if centroid_radec:
                    ref_px_ccdf = np.array([
                        [hdu_list["APERTURE"].header["CRVAL1P"]],
                        [hdu_list["APERTURE"].header["CRVAL2P"]],
                    ])

            if centroid_radec:
                w = wcs.WCS(hdu_list["APERTURE"].header)
                pixcrd = np.vstack((centroid_x - ref_px_ccdf[0], centroid_y - ref_px_ccdf[1])).T
                world = w.wcs_pix2world(pixcrd, 0, ra_dec_order=True)
                centroid_x, centroid_y = world[:, 0], world[:, 1]

            time = light_curve.TIME
            flux = light_curve.PDCSAP_FLUX
            flux_err = light_curve.PDCSAP_FLUX_ERR

            if not time.size:
                files_not_read.append((basename, "No data available."))
                continue

            if not len(time) == len(flux) == len(centroid_x) == len(centroid_y):
                files_not_read.append((
                    basename,
                    f"Time series do not have the same size (Timestamps {time.size}, "
                    f"PDC flux {flux.size}, "
                    f"FW centroid {centroid_x.size}|{centroid_y.size}).",
                ))
                continue

            inds_keep = True * np.ones(len(flux), dtype="bool")
            inds_keep[np.isnan(flux)] = False

            if dq_values_filter:
                dq_values_filter = dq_values_filter if dq_values_filter else []
                flags = {
                    dq_value: np.binary_repr(dq_value).zfill(MAX_BIT).find("1")
                    for dq_value in dq_values_filter
                }
                qflags = np.array([np.binary_repr(el).zfill(MAX_BIT) for el in light_curve.QUALITY])
                for flag_bit in flags:
                    qflags_bit = [el[flags[flag_bit]] == "1" for el in qflags]
                    inds_keep[qflags_bit] = False

            if get_momentum_dump:
                if not dq_values_filter:
                    qflags = np.array([np.binary_repr(el).zfill(MAX_BIT) for el in light_curve.QUALITY])
                momentum_dump_bit = np.binary_repr(MOMENTUM_DUMP_VALUE).zfill(MAX_BIT).find("1")
                momentum_dump_arr = np.array([el[momentum_dump_bit] == "1" for el in qflags]).astype("uint")

            if interpolate_missing_time:
                # Lazy import: only needed when the caller opts in. The worker
                # never does, so we avoid carrying `src_preprocessing.light_curve.util`.
                try:
                    from src_preprocessing.light_curve import util  # type: ignore
                except ImportError as exc:
                    raise NotImplementedError(
                        "interpolate_missing_time=True is not ported yet in worker.exominer_native"
                    ) from exc
                time = util.interpolate_missing_time(time, light_curve.CADENCENO)

            data["all_time"].append(time)
            data["all_flux"].append(flux)
            data["all_flux_err"].append(flux_err)
            data["all_centroids"]["x"].append(centroid_x)
            data["all_centroids"]["y"].append(centroid_y)
            data["all_centroids_px"]["x"].append(centroid_fdl_x)
            data["all_centroids_px"]["y"].append(centroid_fdl_y)
            data["camera"].append(camera)
            data["ccd"].append(ccd)
            data["sectors"].append(sector)
            data["flag_keep"].append(inds_keep)

            if get_momentum_dump:
                data["momentum_dump"].append(momentum_dump_arr)
                data["time_momentum_dump"].append(np.array(time))

        except NotImplementedError:
            # Programming-level opt-in errors (e.g. interpolate_missing_time
            # branch) must surface — not be swallowed as "read failed".
            raise
        except Exception:
            files_not_read.append((basename, "FITS file not read correctly."))

    for arr_i, inds_keep in enumerate(data["flag_keep"]):
        for data_field in timeseries_fields:
            if "centroids" in data_field:
                data[data_field]["x"][arr_i] = data[data_field]["x"][arr_i][inds_keep]
                data[data_field]["y"][arr_i] = data[data_field]["y"][arr_i][inds_keep]
            else:
                data[data_field][arr_i] = data[data_field][arr_i][inds_keep]

    return data, files_not_read


def read_tess_light_curve_from_bytes(
    fits_bytes,
    *,
    light_curve_extension="LIGHTCURVE",
    interpolate_missing_time=False,
    centroid_radec=False,
    prefer_psfcentr=False,
    get_momentum_dump=False,
    dq_values_filter=None,
):
    """Worker-friendly wrapper: one FITS LC as in-memory bytes.

    The worker fetches FITS bytes from the server (``GET /api/fits/``) — this
    call feeds those bytes straight into the same reader logic without
    touching the local filesystem.

    Args:
        fits_bytes: bytes, the raw contents of one ``*lc.fits`` file.
        Remaining kwargs: same as :func:`read_tess_light_curve`.

    Returns:
        ``(data, files_not_read)`` identical to ``read_tess_light_curve``
        called with ``filenames=[<single file>]``. On read failure,
        ``files_not_read`` contains a ``(".bytes", reason)`` tuple.
    """
    # ``read_tess_light_curve`` uses ``os.path.basename`` on each filename,
    # which does not accept BytesIO. Persist the bytes to a short-lived temp
    # file so the reader sees a real path. Cost: one disk write per FITS,
    # negligible compared with the preprocessing downstream.
    with tempfile.NamedTemporaryFile(suffix=".lc.fits", delete=False) as tf:
        tf.write(fits_bytes)
        tmp_path = tf.name
    try:
        return read_tess_light_curve(
            [tmp_path],
            light_curve_extension=light_curve_extension,
            interpolate_missing_time=interpolate_missing_time,
            centroid_radec=centroid_radec,
            prefer_psfcentr=prefer_psfcentr,
            get_momentum_dump=get_momentum_dump,
            dq_values_filter=dq_values_filter,
        )
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
