"""Native port of ``src_preprocessing/diff_img/extracting/utils_diff_img.py``.

Scope: extraction of TCE difference-image data from a TESS SPOC DV XML file.
This is the first module of the native ExoMiner worker pipeline and the one
validated end-to-end at Spike 4 (2026-04-18): oracle (NASA container, numpy
1.24) vs port (worker venv, numpy 2.x) produces bitwise-identical dicts for
TIC 143143769 s1-36.

Trimmed from the container source:
  - ``plot_diff_img_data`` → replaced by a no-op stub (matplotlib is not a
    worker dependency; the plot branch fires only when ``plot_prob > 0``,
    which the worker never does).
  - Kepler variants (``get_data_from_kepler_dv_xml``) — out of scope.
  - ``get_data_from_tess_dv_xml_main`` (multiprocessing + file-logging
    wrapper) — the worker orchestrator does its own job control.

Signatures kept identical to the oracle so the parity harness can call both
sides with the same positional arguments.
"""
from __future__ import annotations

import xml.etree.cElementTree as et

import numpy as np
import pandas as pd


N_IMGS_IN_DIFF = 4  # diff, oot, it, snr
MAX_MAG = 25
MIN_MAG = 1
MAG_RANGE = MIN_MAG - MAX_MAG
TESS_MAG_SAT = 7


def plot_diff_img_data(*args, **kwargs):  # noqa: D401
    """No-op stub — the native worker never plots (no matplotlib dep)."""
    return None


def get_neighbors_for_target_in_sector(sectors_obs, neighbors_dir, tic_id, sector_run_id, proc_id, logger):
    """Get data on neighbors for target ``tic_id`` in sectors in ``sectors_obs``.

    :param sectors_obs: list[int], observed TESS sectors (0-indexed from the
        DV XML ``sectorsObserved`` bitmask).
    :param neighbors_dir: Path, directory containing the neighbors mapping
        tables (``S{sector}/mapping_results/neighbors_pxcoords_S{sector}.csv``).
    :param tic_id: str, target TIC ID.
    :param sector_run_id: str, sector run ID (used only for log messages).
    :param proc_id: int, process ID (used only for log messages).
    :param logger: ``logging.Logger``.

    :return: list of ``pd.DataFrame`` per sector, indexed by neighbor TIC.
    """
    neighbors_lst = []
    for sector_obs in sectors_obs:
        neighbors_tbl = pd.read_csv(
            neighbors_dir / f"S{sector_obs}" / "mapping_results"
            / f"neighbors_pxcoords_S{sector_obs}.csv"
        )

        neighbors_target = neighbors_tbl.loc[neighbors_tbl["target_id"] == int(tic_id)]

        if len(neighbors_target) == 0:
            raise ValueError(
                f"[{proc_id}] [Sector run {sector_run_id}] Target {tic_id}"
                f" not found in the neighbors table for sector {sector_obs}"
            )

        neighbors_target = neighbors_target.set_index("ID")
        neighbors_lst.append(neighbors_target)

        logger.info(
            f"[{proc_id}] [Sector run {sector_run_id}] Found {len(neighbors_target)} neighbors for "
            f"target {tic_id} in sector {sector_obs}."
        )

    return neighbors_lst


def get_data_from_tess_dv_xml(dv_xml_fp, neighbors_dir, sector_run_id, plot_dir, plot_prob, logger, proc_id=-1):
    """Extract difference image data from the TESS target DV XML file for the
    set of TCEs detected in that star for that TESS SPOC sector run.

    :param dv_xml_fp: Path, filepath to DV XML file.
    :param neighbors_dir: Path or None, path to directory containing target
        neighbors data. ``None`` disables the neighbors-enrichment branch.
    :param sector_run_id: str, sector run ID.
    :param plot_dir: Path or None, plot directory. The native port never
        writes plots regardless of this value.
    :param plot_prob: float, probability to plot difference image for a given
        example. The native port evaluates the condition (to keep the PRNG
        call-count identical to the oracle) but never actually plots.
    :param logger: ``logging.Logger``.
    :param proc_id: int, process ID.

    :return: dict, each item is the difference image data for a given TCE.
        The TCE is identified by the string key
        ``'{tic_id}-{tce_plnt_num}-S{sector_run}'``. The value is a dictionary
        that contains at least:
          - ``target_ref_centroid``: list of dicts ``{col, row}`` with
            ``value``/``uncertainty`` in the pixel frame.
          - ``image_data``: list of ``np.ndarray`` shaped
            ``(n_rows, n_cols, n_imgs=4, 2)`` — in-transit, out-of-transit,
            difference, SNR; last axis is value/uncertainty.
          - ``image_number``: list of sector numbers.
          - ``mag``: target magnitude.
          - ``quality_metric``: list of per-sector quality dicts.
        If ``neighbors_dir`` is not None, also ``neighbor_data``.
    """
    data = {}

    try:
        tree = et.parse(dv_xml_fp)
    except Exception as e:
        raise Exception(f"{proc_id}] [Sector run {sector_run_id}] Exception found when reading {dv_xml_fp}: {e}.")

    root = tree.getroot()

    tic_id = root.attrib["ticId"]

    tmag = float([el for el in root if "tessMag" in el.tag][0].attrib["value"])

    planet_res_lst = [el for el in root if "planetResults" in el.tag]

    n_sectors_expected = root.attrib["sectorsObserved"].count("1")
    sectors_obs = [i for i, char in enumerate(root.attrib["sectorsObserved"]) if char == "1"]

    # get neighboring stars
    if neighbors_dir:
        logger.info(
            f"{proc_id}] [Sector run {sector_run_id}] Getting neighbors information for target {tic_id} in "
            f"sectors {sectors_obs}..."
        )
        neighbors_lst = get_neighbors_for_target_in_sector(
            sectors_obs, neighbors_dir, tic_id, sector_run_id, proc_id, logger
        )

    n_tces = len(planet_res_lst)
    tce_i = 0
    for planet_res in planet_res_lst:

        tce_i += 1

        uid = f'{root.attrib["ticId"]}-{planet_res.attrib["planetNumber"]}-S{sector_run_id}'

        if neighbors_dir:
            # filter neighbors based on transit depth - could these objects cause the observed transit depth?
            tce_depth = max(
                0,
                float(
                    planet_res.find(
                        './/dv:modelParameter[@name="transitDepthPpm"]',
                        {"dv": "http://www.nasa.gov/2018/TESS/DV"},
                    ).attrib["value"]
                ),
            ) + 1
            tce_depth /= 1e6
            delta_mag = tmag - np.log10(tce_depth) * 2.5
            tce_neighbors_lst = []
            for neighbors_sector in neighbors_lst:
                tce_neighbors_sector = {}
                for neighbor_id, neighbor_data in neighbors_sector.iterrows():
                    if np.isnan(neighbor_data["row_px"]) or np.isnan(neighbor_data["col_px"]):
                        continue
                    if neighbor_data["Tmag"] <= delta_mag:
                        tce_neighbors_sector.update({neighbor_id: dict(neighbor_data)})
                tce_neighbors_lst.append(tce_neighbors_sector)

        data[uid] = {
            "target_ref_centroid": [],
            "image_data": [],
            "mag": tmag,
            "image_number": [],
            "quality_metric": [],
        }
        if neighbors_dir:
            data[uid]["neighbor_data"] = tce_neighbors_lst

        logger.info(
            f"[{proc_id}] [Sector run {sector_run_id}] Getting difference image data for TCE TIC "
            f"{uid} ({tce_i}/{n_tces} TCEs)..."
        )

        # get difference image results
        diff_img_res = [el for el in planet_res if "differenceImageResults" in el.tag]

        n_sectors = len(diff_img_res)

        if n_sectors < n_sectors_expected:
            logger.info(
                f"[{proc_id}] [Sector run {sector_run_id}] TCE TIC {uid} has less than "
                f"{n_sectors_expected} "
                f"sectors ({n_sectors})"
            )

        # iterate over sectors
        for sector_i in range(n_sectors):

            img_res_s = diff_img_res[sector_i]

            # get quality metric data
            q_metric_s = [el.attrib for el in img_res_s if "qualityMetric" in el.tag][0]
            q_metric_s["value"] = float(q_metric_s["value"])
            q_metric_s["attempted"] = True if q_metric_s["attempted"] == "true" else False
            q_metric_s["valid"] = True if q_metric_s["valid"] == "true" else False
            data[uid]["quality_metric"].append(q_metric_s)

            # get sector id
            sector = int(img_res_s.attrib["sector"])
            data[uid]["image_number"].append(sector)

            # get difference image pixel data
            img_px_data = [el for el in img_res_s if "differenceImagePixelData" in el.tag]

            px_dict = {(int(el.attrib["ccdRow"]), int(el.attrib["ccdColumn"])): list(el) for el in img_px_data}

            # get max and min row and col
            px_row_lst, px_col_lst = [], []
            for px_row, px_col in px_dict.keys():
                px_row_lst.append(px_row)
                px_col_lst.append(px_col)

            min_row, max_row = min(px_row_lst), max(px_row_lst)
            min_col, max_col = min(px_col_lst), max(px_col_lst)

            # determine size of images
            row_size = max_row - min_row + 1
            col_size = max_col - min_col + 1

            # populate array with pixel values
            diff_imgs = np.nan * np.ones((row_size, col_size, N_IMGS_IN_DIFF, 2), dtype="float")

            for px_coord, diff_imgs_q in px_dict.items():
                diff_imgs[px_coord[0] - min_row, px_coord[1] - min_col, :, 0] = [
                    float(el.attrib["value"]) for el in diff_imgs_q
                ]
                diff_imgs[px_coord[0] - min_row, px_coord[1] - min_col, :, 1] = [
                    float(el.attrib["uncertainty"]) for el in diff_imgs_q
                ]

            # get target position in pixel frame
            tic_centroid_ref = [el for el in img_res_s if "ticReferenceCentroid" in el.tag][0]
            tic_centroid_ref_col = [el for el in tic_centroid_ref if "col" in el.tag][0].attrib
            tic_centroid_ref_row = [el for el in tic_centroid_ref if "row" in el.tag][0].attrib

            # check for missing value
            if float(tic_centroid_ref_col["uncertainty"]) == -1 or float(tic_centroid_ref_row["uncertainty"]) == -1:
                tic_centroid_ref_dict = {
                    "col": {k: float(v) if k == "value" else float(v) for k, v in tic_centroid_ref_col.items()},
                    "row": {k: float(v) if k == "value" else float(v) for k, v in tic_centroid_ref_row.items()},
                }
                logger.info(
                    f"[{proc_id}] [Sector run {sector_run_id}] TCE TIC {uid} has missing reference "
                    f'centroid for target in sector {img_res_s.attrib["sector"]}.'
                )
            else:
                tic_centroid_ref_dict = {
                    "col": {
                        k: float(v) - min_col if k == "value" else float(v) for k, v in tic_centroid_ref_col.items()
                    },
                    "row": {
                        k: float(v) - min_row if k == "value" else float(v) for k, v in tic_centroid_ref_row.items()
                    },
                }

            # plot difference image — evaluated to keep the PRNG call-count
            # identical to the oracle; actual plotting is disabled at the port
            if np.random.uniform() <= plot_prob:

                if neighbors_dir:
                    neighbors_coords = [
                        (neighbor_data["col_px"], neighbor_data["row_px"])
                        for _, neighbor_data in data[uid]["neighbor_data"][sector_i].items()
                    ]
                    neighbors_mags = [
                        neighbor_data["Tmag"] for _, neighbor_data in tce_neighbors_lst[sector_i].items()
                    ]
                else:
                    neighbors_coords, neighbors_mags, delta_mag = None, None, None

                plot_diff_img_data(
                    diff_imgs,
                    plot_dir / f"tic_{uid}_sector_{sector}.png" if plot_dir is not None else None,
                    target_coords=(
                        tic_centroid_ref_dict["col"]["value"],
                        tic_centroid_ref_dict["row"]["value"],
                    )
                    if tic_centroid_ref_dict["col"]["uncertainty"] != -1
                    else None,
                    neighbors_coords=neighbors_coords,
                    logscale=True,
                    target_mag=data[uid]["mag"],
                    neighbors_mag=neighbors_mags,
                    mag_sat=TESS_MAG_SAT,
                    min_mag=delta_mag,
                )

            data[uid]["target_ref_centroid"].append(tic_centroid_ref_dict)
            data[uid]["image_data"].append(diff_imgs)

    return data
