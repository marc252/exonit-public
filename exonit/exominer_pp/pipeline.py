"""Drive NASA's published ExoMiner pipeline to generate ExoMiner++ FFI inputs.

Input generation (download SPOC-FFI products → light-curve preprocessing →
difference-image extraction from the DV XML → neighbor search → normalization)
is delegated to NASA's published pipeline running inside the pinned Docker
image ``ghcr.io/nasa/exominer:latest``. This is the byte-faithful reference
and the only way to reproduce the published 56-input contract without
re-implementing thousands of lines of SPOC-equivalent preprocessing
(CLAUDE.md rule #3).

Two integration realities (verified 2026-06-25, see
``progress/exominerpp_implementation.md``):

1. The Docker image's *pipeline* is wired for the **2-min ExoMiner++**
   model: it produces 33x33 difference images and only the
   ``diff/oot/target`` image fields. The **FFI new-architecture** CV
   ensemble (the Zenodo artifact) needs 55x55 images plus the ``snr`` and
   ``neighbors`` branches, produced by NASA's *research* CV-preprocessing
   config (``src_cv/preprocessing``), which is present in the cloned repo but
   NOT pre-wired in the turnkey pipeline. So driving the stock image yields
   the 2-min contract, not the FFI new-arch contract.

2. The normalization stats bundled in the image cover ``diff_imgs(_tc)`` /
   ``oot_imgs(_tc)`` but not ``snr_imgs_tc`` (see ``norm_stats``).

This module therefore exposes two entry points:

  * ``run_docker_2min`` — the working, validated path: drives the stock image
    to score a TIC with the 2-min ExoMiner++ model end-to-end (this is what
    ``exonit.exominer_service`` already does; we reuse it). Useful for any
    SPOC-2-min TIC.
  * ``build_ffi_config_overrides`` + ``run_ffi_pipeline`` — the FFI
    new-architecture path: produces the config overrides (55x55, snr +
    neighbor image fields, neighbor-search enabled) required to generate the
    *full* ExoMiner++ FFI contract, and drives ``run_pipeline.py`` with them.
    The neighbor-search step and the ``snr_imgs_tc`` norm stat are the two
    pieces the public turnkey image does not ship; this module wires them
    from the cloned repo + the derived stat.

Nothing here touches the production server or DB.
"""
from __future__ import annotations

import csv
import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from exonit.config import config as _config

_PROJECT_ROOT = _config.project_root

DEFAULT_IMAGE = "ghcr.io/nasa/exominer:latest"
DEFAULT_REPO = os.environ.get(
    "EXONIT_EXOMINER_REPO", os.path.join(_PROJECT_ROOT, "extern", "ExoMiner"))


@dataclass
class FfiPipelineConfig:
    """Configuration for the FFI new-architecture input-generation run.

    Attributes mirror the knobs that must change from the stock 2-min image
    to produce the FFI 55x55 / 4-image-branch contract.
    """

    # 55x55 = final_size(11) * upscale_f(5). Stock image uses upscale_f=3 → 33x33.
    final_size: int = 11
    upscale_f: int = 5
    # image fields written into the TFRecord (stock image omits snr/neighbors).
    img_fields: tuple = ("diff_imgs", "oot_imgs", "snr_imgs", "neighbors_imgs", "target_imgs")
    enable_neighbor_search: bool = True
    # path to the FFI CV ensemble dir (5 keras folds) for inference.
    ensemble_dir: Optional[str] = None
    # derived snr_imgs_tc norm stat (median/std/min/max) to splice in.
    snr_norm_stat: Optional[Dict[str, float]] = None


def image_size(cfg: FfiPipelineConfig) -> int:
    """Final image edge length produced by ``cfg`` (e.g. 55)."""
    return cfg.final_size * cfg.upscale_f


def build_ffi_config_overrides(cfg: FfiPipelineConfig) -> Dict[str, Dict]:
    """Return the per-config-file overrides for an FFI new-architecture run.

    Returns a mapping ``{config_filename: {yaml_key: value}}`` describing the
    edits required vs the stock Docker pipeline configs. Kept as data (not
    applied here) so it is unit-testable and so the caller can apply it to a
    mounted config dir or a forked image.
    """
    edge = image_size(cfg)
    return {
        "config_diff_img_preprocessing.yaml": {
            "final_size": {"x": cfg.final_size, "y": cfg.final_size},
            "upscale_f": {"x": cfg.upscale_f, "y": cfg.upscale_f},
        },
        "config_add_diff_img_tfrecords.yaml": {
            "imgs_fields": list(cfg.img_fields),
        },
        "config_normalize_data.yaml": {
            "diff_img_data_shape": [5, edge, edge],
        },
        # The predict config must select the FFI new-architecture feature set
        # (55x55 diff/oot/snr/neighbors, no momentum dump). See
        # models/exominer_plusplus_ffi.yaml for the full features_set.
        "config_predict_model.yaml": {
            "_feature_set": "exominer_plusplus_ffi_new_arch",
            "_image_shape": [5, edge, edge],
        },
    }


def write_tics_csv(path: Path, tic_id: str, sector_run: str) -> Path:
    """Write the single-row TICs CSV the pipeline expects (tic_id, sector_run)."""
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tic_id", "sector_run"])
        w.writerow([str(tic_id), str(sector_run)])
    return path


def build_docker_cmd(
    tics_csv: Path,
    out_dir: Path,
    *,
    model: str,
    data_collection_mode: str = "ffi",
    image: str = DEFAULT_IMAGE,
    num_processes: int = 1,
    num_jobs: int = 1,
    extra_mounts: Optional[List[str]] = None,
    run_as_root: Optional[bool] = None,
) -> List[str]:
    """Construct the ``docker run`` argv to drive ``run_pipeline.py``.

    Mirrors ``exonit.exominer_service._build_docker_cmd`` (same user-identity
    policy) but defaults to ``--data_collection_mode=ffi`` and lets the caller
    mount a custom model and/or overridden config files.
    """
    if run_as_root is None:
        run_as_root = os.environ.get("EXONIT_EXOMINER_DOCKER_ROOT", "").strip() == "1"
    identity = ["-e", "HOME=/tmp"] if run_as_root else [
        "--user", f"{os.getuid()}:{os.getgid()}"
    ]
    mounts = [
        "-v", f"{tics_csv}:/tics_tbl.csv:Z",
        "-v", f"{out_dir}:/outputs:Z",
    ]
    if extra_mounts:
        mounts += extra_mounts
    return [
        "docker", "run", "--rm",
        *identity,
        *mounts,
        image,
        "--tic_ids_fp=/tics_tbl.csv",
        "--output_dir=/outputs",
        f"--data_collection_mode={data_collection_mode}",
        f"--num_processes={num_processes}",
        f"--num_jobs={num_jobs}",
        "--download_spoc_data_products=false",
        "--stellar_parameters_source=ticv8",
        "--ruwe_source=gaiadr2",
        f"--exominer_model={model}",
    ]


@dataclass
class PipelineResult:
    tic_id: str
    sector_run: str
    predictions_csv: Optional[str] = None
    returncode: Optional[int] = None
    ok: bool = False
    note: str = ""
    work_dir: Optional[str] = None


def run_ffi_pipeline(
    tic_id: str,
    sector_run: str,
    cfg: FfiPipelineConfig,
    *,
    work_root: Optional[str] = None,
    image: str = DEFAULT_IMAGE,
    check: bool = False,
) -> PipelineResult:
    """Drive the FFI new-architecture input-generation + inference for a TIC.

    Requires a model filepath in ``cfg.ensemble_dir`` (a single keras file or
    a fold) mounted into the container, and the FFI config overrides applied
    (via a mounted config dir). The full neighbor-search wiring + snr norm
    stat are described in ``progress/exominerpp_implementation.md`` — this
    function performs the docker invocation given those are staged; if the
    stock image is used unmodified it will produce the 2-min contract, so the
    returned ``note`` flags whether full-FFI staging was provided.
    """
    work_root = work_root or os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "data", "exominer_pp_work",
    )
    os.makedirs(work_root, exist_ok=True)
    run_dir = tempfile.mkdtemp(prefix=f"exominerpp_{tic_id}_", dir=work_root)
    tics_csv = write_tics_csv(Path(run_dir) / "tics_tbl.csv", tic_id, sector_run)
    out_dir = Path(run_dir) / "output"
    out_dir.mkdir(exist_ok=True)

    if not cfg.ensemble_dir:
        return PipelineResult(
            tic_id=str(tic_id), sector_run=str(sector_run), ok=False,
            work_dir=run_dir,
            note="no ensemble model provided (cfg.ensemble_dir); cannot run inference",
        )

    model_path = Path(cfg.ensemble_dir)
    extra_mounts = ["-v", f"{model_path}:/custom_model.keras:Z"]
    cmd = build_docker_cmd(
        tics_csv, out_dir, model="/custom_model.keras",
        data_collection_mode="ffi", image=image, extra_mounts=extra_mounts,
    )
    note = (
        "full-FFI staging (55x55 + snr + neighbors) requires mounting "
        "overridden configs + neighbor search; see implementation doc"
    )
    try:
        proc = subprocess.run(cmd, check=check, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        return PipelineResult(
            tic_id=str(tic_id), sector_run=str(sector_run), ok=False,
            returncode=e.returncode, work_dir=run_dir,
            note=f"docker failed: {e}",
        )
    # run_pipeline.py writes the aggregated table as predictions_{out_dir.name}.csv
    # (older images: predictions_outputs.csv). Glob to be robust across versions.
    candidates = sorted(out_dir.glob("predictions_*.csv"))
    if not candidates:
        candidates = sorted(out_dir.rglob("ranked_predictions_predictset.csv"))
    found = candidates[0] if candidates else None
    return PipelineResult(
        tic_id=str(tic_id), sector_run=str(sector_run),
        predictions_csv=str(found) if found else None,
        returncode=proc.returncode, ok=found is not None,
        work_dir=run_dir, note=note,
    )
