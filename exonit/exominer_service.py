"""ExoMiner++ inference service — Docker wrapper + SQLite cache.

Runs the NASA ExoMiner container (``ghcr.io/nasa/exominer:latest``) to
score TCEs for a TIC, and caches the result. TCEs are stable (pinned
pipeline + pinned model + pinned DV products), so cache entries never
expire automatically — call ``invalidate`` to force a rerun.

Typical use::

    svc = ExoMinerService()
    result = svc.predict_tic("382437043")
    for tce in result.tces:
        print(tce.period, tce.score)

Matching a BLS candidate to a SPOC TCE is exposed via
``match_tce_to_period``, which handles the period aliases in _PERIOD_FACTORS.
"""
from __future__ import annotations

import csv
import json
import os
import shutil
import sqlite3
import subprocess
import tempfile
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import List, Optional

from exonit.mast_dv import MastDVCache, fetch_dv_runs, pick_best_run


DEFAULT_IMAGE = "ghcr.io/nasa/exominer:latest"
DEFAULT_MODEL = "exominer++_single"

_CREATE_SQL = """
CREATE TABLE IF NOT EXISTS exominer_cache (
    tic_id       TEXT NOT NULL,
    sector_run   TEXT NOT NULL,
    model        TEXT NOT NULL,
    tces_json    TEXT NOT NULL,
    created_at   TEXT DEFAULT (datetime('now')),
    PRIMARY KEY (tic_id, sector_run, model)
)
"""


@dataclass
class ExoMinerTCE:
    tic_id: str
    tce_num: int
    period: float
    duration: float  # days
    t0: float  # BKJD (tce_time0bk)
    depth_ppm: float
    ruwe: float
    prad: float  # Earth radii
    mes: float  # max multiple event sigma (tce_max_mult_ev)
    score: float


@dataclass
class PredictResult:
    tic_id: str
    sector_run: Optional[str]
    model: str
    tces: List[ExoMinerTCE] = field(default_factory=list)
    from_cache: bool = False
    reason: Optional[str] = None  # "no_dv_runs", "docker_failed", etc.


# ---------- parsing ----------

def parse_predictions_csv(path: str) -> List[ExoMinerTCE]:
    if not os.path.exists(path):
        return []
    out: List[ExoMinerTCE] = []
    with open(path) as f:
        for row in csv.DictReader(f):
            try:
                out.append(
                    ExoMinerTCE(
                        tic_id=str(row["target_id"]),
                        tce_num=int(row["tce_plnt_num"]),
                        period=float(row["tce_period"]),
                        duration=float(row["tce_duration"]),
                        t0=float(row["tce_time0bk"]),
                        depth_ppm=float(row["tce_depth"]),
                        ruwe=float(row["ruwe"]) if row.get("ruwe") not in ("", None) else float("nan"),
                        prad=float(row["tce_prad"]),
                        mes=float(row["tce_max_mult_ev"]),
                        score=float(row["score"]),
                    )
                )
            except (KeyError, ValueError):
                continue
    return out


# ---------- period matching ----------

# Server-side mirror of worker/exominer_period_match.py (the worker dir is
# deployed standalone, so this cannot import it) — keep BOTH in sync.
# 3× and 4× aliases added 2026-05-16 to recover
# strong candidates whose BLS period was an N× multiple of the ExoMiner
# TCE period (see hook.py for the failure cases).
_PERIOD_TOL = 0.02  # 2% relative tolerance
_PERIOD_FACTORS = (1.0, 2.0, 0.5, 3.0, 1.0 / 3.0, 4.0, 0.25)


def match_tce_to_period(
    tces: List[ExoMinerTCE],
    target_period: float,
    tol: float = _PERIOD_TOL,
) -> Optional[ExoMinerTCE]:
    """Return the TCE whose period matches ``target_period`` within tol.

    Matches against the _PERIOD_FACTORS aliases (1, 2, ½, 3, ⅓, 4, ¼). Returns the
    first best match (by relative error) if multiple TCEs qualify.
    """
    if not tces:
        return None
    best: Optional[ExoMinerTCE] = None
    best_err = float("inf")
    for tce in tces:
        for k in _PERIOD_FACTORS:
            p1 = target_period
            p2 = k * tce.period
            denom = max(abs(p1), abs(p2))
            if denom == 0:
                continue
            err = abs(p1 - p2) / denom
            if err < tol and err < best_err:
                best = tce
                best_err = err
    return best


# ---------- service ----------

class ExoMinerService:
    def __init__(
        self,
        cache_db: Optional[str] = None,
        work_dir: Optional[str] = None,
        image: str = DEFAULT_IMAGE,
        model: str = DEFAULT_MODEL,
        mast_cache: Optional[MastDVCache] = None,
        num_processes: int = 1,
        num_jobs: int = 1,
    ):
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.cache_db = cache_db or os.path.join(root, "data", "exominer_cache.db")
        self.work_dir = work_dir or os.path.join(root, "data", "exominer_work")
        os.makedirs(os.path.dirname(os.path.abspath(self.cache_db)), exist_ok=True)
        os.makedirs(self.work_dir, exist_ok=True)

        self.image = image
        self.model = model
        self.num_processes = num_processes
        self.num_jobs = num_jobs
        self._owns_mast_cache = mast_cache is None
        self._mast_cache = mast_cache

        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.cache_db, check_same_thread=False)
        self._conn.execute(_CREATE_SQL)
        self._conn.commit()
        self._sweep_stale_workdirs(max_age_seconds=3600)

    def _sweep_stale_workdirs(self, max_age_seconds: int = 3600) -> int:
        """Delete leftover ``exominer_<tic>_*`` dirs older than max_age.

        Defends against crashes/restarts that leave docker outputs behind
        (each can hold hundreds of MB of FITS + tfrecords). Returns count
        of dirs removed.
        """
        if not os.path.isdir(self.work_dir):
            return 0
        cutoff = time.time() - max_age_seconds
        removed = 0
        for name in os.listdir(self.work_dir):
            if not name.startswith("exominer_"):
                continue
            path = os.path.join(self.work_dir, name)
            try:
                if os.path.getmtime(path) < cutoff:
                    shutil.rmtree(path, ignore_errors=True)
                    if not os.path.exists(path):
                        removed += 1
            except OSError:
                continue
        return removed

    # -- cache --

    def _cache_get(self, tic_id: str, sector_run: str, model: str) -> Optional[List[ExoMinerTCE]]:
        with self._lock:
            row = self._conn.execute(
                "SELECT tces_json FROM exominer_cache WHERE tic_id=? AND sector_run=? AND model=?",
                (str(tic_id), sector_run, model),
            ).fetchone()
        if row is None:
            return None
        return [ExoMinerTCE(**t) for t in json.loads(row[0])]

    def _cache_put(self, tic_id: str, sector_run: str, model: str, tces: List[ExoMinerTCE]) -> None:
        payload = json.dumps([asdict(t) for t in tces])
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO exominer_cache (tic_id, sector_run, model, tces_json) "
                "VALUES (?, ?, ?, ?)",
                (str(tic_id), sector_run, model, payload),
            )
            self._conn.commit()

    def invalidate(self, tic_id: str, sector_run: Optional[str] = None) -> int:
        """Drop cache entries for a TIC. Returns rows deleted."""
        with self._lock:
            if sector_run is None:
                cur = self._conn.execute(
                    "DELETE FROM exominer_cache WHERE tic_id=?", (str(tic_id),)
                )
            else:
                cur = self._conn.execute(
                    "DELETE FROM exominer_cache WHERE tic_id=? AND sector_run=?",
                    (str(tic_id), sector_run),
                )
            self._conn.commit()
            return cur.rowcount

    # -- docker orchestration --

    def _build_docker_cmd(
        self,
        tics_csv: str,
        out_dir: str,
        model: str,
        resource_flags: list,
    ) -> list:
        """Return the full ``docker run`` argument list (without actually running it).

        Extracted for testability: callers can inspect the command without
        invoking Docker.

        User-identity policy
        --------------------
        By default the container runs with ``--user uid:gid`` so output files
        are owned by the calling process and ``shutil.rmtree`` works without
        sudo.

        The current ``ghcr.io/nasa/exominer:latest`` image (as of 2026-06-20)
        has its pipeline code at ``/app/exominer_pipeline/run_pipeline.py``
        with permissions readable only by root, so running as a non-root uid
        produces an instant [Errno 13] Permission denied.  Set
        ``EXONIT_EXOMINER_DOCKER_ROOT=1`` to omit ``--user`` and add
        ``-e HOME=/tmp`` instead.  In that mode the output files will be
        root-owned; cleanup is still attempted but uses ``ignore_errors=True``
        so a PermissionError on ``shutil.rmtree`` does not crash the server.

        When ``EXONIT_EXOMINER_DOCKER_ROOT`` is unset or not ``"1"`` the
        command is byte-identical to the pre-2026-06-20 behaviour.
        """
        run_as_root = os.environ.get("EXONIT_EXOMINER_DOCKER_ROOT", "").strip() == "1"

        if run_as_root:
            # Root mode: drop --user, add writable HOME for conda/pip caches.
            # Cleanup of root-owned output files is best-effort (ignore_errors).
            identity_flags: list = ["-e", "HOME=/tmp"]
        else:
            # Default: run as calling uid:gid so output files are user-owned.
            identity_flags = ["--user", f"{os.getuid()}:{os.getgid()}"]

        return [
            "docker", "run", "--rm",
            *identity_flags,
            *resource_flags,
            "-v", f"{tics_csv}:/tics_tbl.csv:Z",
            "-v", f"{out_dir}:/outputs:Z",
            self.image,
            "--tic_ids_fp=/tics_tbl.csv",
            "--output_dir=/outputs",
            "--data_collection_mode=2min",
            f"--num_processes={self.num_processes}",
            f"--num_jobs={self.num_jobs}",
            "--download_spoc_data_products=false",
            "--stellar_parameters_source=ticv8",
            "--ruwe_source=gaiadr2",
            f"--exominer_model={model}",
        ]

    def _run_docker(self, tic_id: str, sector_run: str, model: str) -> str:
        """Execute docker run synchronously and return path to predictions_outputs.csv.

        Returned path lives under ``self.work_dir/exominer_<tic>_<rand>/output/``;
        caller is responsible for cleanup via ``shutil.rmtree`` on the run_dir.

        See ``_build_docker_cmd`` for the user-identity policy (default:
        ``--user uid:gid``; opt-in root mode via EXONIT_EXOMINER_DOCKER_ROOT=1).
        """
        run_dir = tempfile.mkdtemp(prefix=f"exominer_{tic_id}_", dir=self.work_dir)
        tics_csv = os.path.join(run_dir, "tics_tbl.csv")
        out_dir = os.path.join(run_dir, "output")
        os.makedirs(out_dir, exist_ok=True)
        with open(tics_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["tic_id", "sector_run"])
            w.writerow([tic_id, sector_run])

        # Cap RAM/CPU per container so a runaway invocation cannot OOM the
        # host. On 2026-05-15 the GCP server (16 GB) crashed because multiple
        # concurrent docker runs each grew unbounded. Pair with the
        # `_exominer_docker_semaphore` in server.py which caps concurrency.
        # Empty env var disables the corresponding flag.
        mem_limit = os.environ.get("EXONIT_EXOMINER_MEMORY_LIMIT", "4g").strip()
        cpu_limit = os.environ.get("EXONIT_EXOMINER_CPU_LIMIT", "2").strip()
        resource_flags: list[str] = []
        if mem_limit:
            resource_flags += ["--memory", mem_limit]
        if cpu_limit:
            resource_flags += ["--cpus", cpu_limit]

        cmd = self._build_docker_cmd(tics_csv, out_dir, model, resource_flags)
        subprocess.run(cmd, check=True)
        return os.path.join(out_dir, "predictions_outputs.csv")

    # -- high-level --

    def predict_tic(
        self,
        tic_id: str,
        sector_run: Optional[str] = None,
        model: Optional[str] = None,
    ) -> PredictResult:
        """Score all TCEs for ``tic_id``, using cache and falling back to docker."""
        model = model or self.model
        tic_id = str(tic_id)

        # Resolve sector_run from MAST if not given
        if sector_run is None:
            mast_cache = self._mast_cache or MastDVCache()
            try:
                runs = fetch_dv_runs(tic_id, cache=mast_cache)
            finally:
                if self._mast_cache is None:
                    mast_cache.close()
            if not runs:
                return PredictResult(
                    tic_id=tic_id, sector_run=None, model=model,
                    tces=[], from_cache=False, reason="no_dv_runs",
                )
            sector_run = pick_best_run(runs)
            if sector_run is None:  # pragma: no cover — pick_best_run should return a string given non-empty runs
                return PredictResult(
                    tic_id=tic_id, sector_run=None, model=model,
                    tces=[], from_cache=False, reason="no_dv_runs",
                )

        # Cache hit?
        cached = self._cache_get(tic_id, sector_run, model)
        if cached is not None:
            return PredictResult(
                tic_id=tic_id, sector_run=sector_run, model=model,
                tces=cached, from_cache=True,
            )

        # Run docker. Always clean run_dir via try/finally so a docker failure
        # doesn't leak the workdir (which can grow to hundreds of MB per call).
        t0 = time.time()
        run_dir: Optional[str] = None
        try:
            pred_csv = self._run_docker(tic_id, sector_run, model)
            run_dir = os.path.dirname(os.path.dirname(pred_csv))
        except subprocess.CalledProcessError as e:
            return PredictResult(
                tic_id=tic_id, sector_run=sector_run, model=model,
                tces=[], from_cache=False, reason=f"docker_failed: {e}",
            )

        try:
            elapsed = time.time() - t0
            all_tces = parse_predictions_csv(pred_csv)
            tces = [t for t in all_tces if t.tic_id == tic_id]
            self._cache_put(tic_id, sector_run, model, tces)
            return PredictResult(
                tic_id=tic_id, sector_run=sector_run, model=model,
                tces=tces, from_cache=False,
                reason=f"docker_run_{elapsed:.1f}s",
            )
        finally:
            if run_dir:
                shutil.rmtree(run_dir, ignore_errors=True)

    def close(self) -> None:
        self._conn.close()
