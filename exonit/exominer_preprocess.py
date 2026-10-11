"""Preprocessing-only half of the ExoMiner container pipeline.

Runs the NASA container with ``num_jobs=1`` so the outputs always land
at a deterministic location (``job_0/...``), then copies the TFRecord
(normalized + diff-image) and the processed TCE table to a shared NFS
directory where workers can read them.

The split between preprocessing (CPU + I/O, server) and inference
(GPU, worker) is the architectural principle of Phase 5 — see
``docs/superpowers/plans/2026-04-18-exominer-phase5-gpu-workers.md``.

Cache is filesystem-based: if the output files already exist at the
expected NFS paths, no Docker is run. Use ``invalidate`` to force a
rerun.

Typical usage::

    pre = ExoMinerPreprocessor(
        nfs_records_dir="<tess-cache>/exominer_records",
        nfs_tce_table_dir="<tess-cache>/exominer_tce_tables",
    )
    result = pre.preprocess_tic("382437043")
    # → TFRecord at result.tfrecord_path, TCE table at result.tce_table_path
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from exonit.mast_dv import MastDVCache, fetch_dv_runs, pick_best_run


DEFAULT_IMAGE = "ghcr.io/nasa/exominer:latest"
DEFAULT_MODEL = "exominer++_single"

_SHARD_NAME = "shard-tess_diffimg_TESS_0"
_TCE_TABLE_NAME = "tess-spoc-dv_tces_0_processed.csv"


@dataclass
class PreprocessResult:
    tic_id: str
    sector_run: Optional[str]
    tfrecord_path: Optional[str]
    tce_table_path: Optional[str]
    from_cache: bool = False
    reason: Optional[str] = None


class ExoMinerPreprocessor:
    def __init__(
        self,
        nfs_records_dir: Optional[str] = None,
        nfs_tce_table_dir: Optional[str] = None,
        work_dir: Optional[str] = None,
        image: str = DEFAULT_IMAGE,
        model: str = DEFAULT_MODEL,
        mast_cache: Optional[MastDVCache] = None,
    ):
        root = Path(__file__).resolve().parents[1]
        self.nfs_records_dir = Path(nfs_records_dir or root / "data" / "exominer_records")
        self.nfs_tce_table_dir = Path(nfs_tce_table_dir or root / "data" / "exominer_tce_tables")
        self.work_dir = Path(work_dir or root / "data" / "exominer_prep_work")
        self.nfs_records_dir.mkdir(parents=True, exist_ok=True)
        self.nfs_tce_table_dir.mkdir(parents=True, exist_ok=True)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.image = image
        self.model = model
        self._owns_mast_cache = mast_cache is None
        self._mast_cache = mast_cache

    # -- path helpers --

    def _tfrecord_dest(self, tic_id: str, sector_run: str) -> Path:
        return self.nfs_records_dir / f"{tic_id}_{sector_run}.tfrecord"

    def _tce_table_dest(self, tic_id: str, sector_run: str) -> Path:
        return self.nfs_tce_table_dir / f"{tic_id}_{sector_run}.csv"

    # -- cache --

    def is_cached(self, tic_id: str, sector_run: str) -> bool:
        return self._tfrecord_dest(tic_id, sector_run).exists() and self._tce_table_dest(tic_id, sector_run).exists()

    def invalidate(self, tic_id: str, sector_run: str) -> None:
        for p in (self._tfrecord_dest(tic_id, sector_run), self._tce_table_dest(tic_id, sector_run)):
            if p.exists():
                p.unlink()

    # -- docker --

    def _run_docker(self, tic_id: str, sector_run: str, model: str, run_dir: str) -> None:
        """Execute docker run with num_jobs=1 so outputs land at ``job_0/``."""
        import csv

        tics_csv = Path(run_dir) / "tics_tbl.csv"
        out_dir = Path(run_dir) / "output"
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(tics_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["tic_id", "sector_run"])
            w.writerow([tic_id, sector_run])

        cmd = [
            "docker", "run", "--rm",
            "-v", f"{tics_csv}:/tics_tbl.csv:Z",
            "-v", f"{out_dir}:/outputs:Z",
            self.image,
            "--tic_ids_fp=/tics_tbl.csv",
            "--output_dir=/outputs",
            "--data_collection_mode=2min",
            "--num_processes=1",
            "--num_jobs=1",
            "--download_spoc_data_products=false",
            "--stellar_parameters_source=ticv8",
            "--ruwe_source=gaiadr2",
            f"--exominer_model={model}",
        ]
        subprocess.run(cmd, check=True)

    # -- main entry point --

    def preprocess_tic(
        self,
        tic_id: str,
        sector_run: Optional[str] = None,
        model: Optional[str] = None,
    ) -> PreprocessResult:
        model = model or self.model
        tic_id = str(tic_id)

        # Resolve sector_run from MAST if needed
        if sector_run is None:
            mast_cache = self._mast_cache or MastDVCache()
            try:
                runs = fetch_dv_runs(tic_id, cache=mast_cache)
            finally:
                if self._mast_cache is None:
                    mast_cache.close()
            if not runs:
                return PreprocessResult(tic_id, None, None, None, reason="no_dv_runs")
            sector_run = pick_best_run(runs)
            if sector_run is None:  # pragma: no cover
                return PreprocessResult(tic_id, None, None, None, reason="no_dv_runs")

        # Cache hit?
        if self.is_cached(tic_id, sector_run):
            return PreprocessResult(
                tic_id=tic_id,
                sector_run=sector_run,
                tfrecord_path=str(self._tfrecord_dest(tic_id, sector_run)),
                tce_table_path=str(self._tce_table_dest(tic_id, sector_run)),
                from_cache=True,
            )

        # Run docker
        run_dir = tempfile.mkdtemp(prefix=f"exominer_prep_{tic_id}_", dir=self.work_dir)
        try:
            try:
                self._run_docker(tic_id, sector_run, model, run_dir)
            except subprocess.CalledProcessError as e:
                return PreprocessResult(
                    tic_id, sector_run, None, None,
                    reason=f"docker_failed: {e}",
                )

            # Expected outputs
            src_tfrecord = Path(run_dir) / "output" / "job_0" / "tfrecord_data_diffimg_normalized" / _SHARD_NAME
            src_tce_table = Path(run_dir) / "output" / "job_0" / "tce_table" / _TCE_TABLE_NAME
            if not src_tfrecord.exists():
                return PreprocessResult(tic_id, sector_run, None, None, reason="no_tfrecord_produced")

            dest_tfrecord = self._tfrecord_dest(tic_id, sector_run)
            dest_tce_table = self._tce_table_dest(tic_id, sector_run)
            shutil.copy2(src_tfrecord, dest_tfrecord)
            if src_tce_table.exists():
                shutil.copy2(src_tce_table, dest_tce_table)
            else:
                # The TCE table is nice-to-have but not strictly required for inference
                dest_tce_table.write_text("")

            return PreprocessResult(
                tic_id=tic_id,
                sector_run=sector_run,
                tfrecord_path=str(dest_tfrecord),
                tce_table_path=str(dest_tce_table),
                from_cache=False,
            )
        finally:
            shutil.rmtree(run_dir, ignore_errors=True)

    def close(self) -> None:
        pass  # nothing to close — filesystem-based cache
