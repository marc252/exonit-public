"""Worker-side ExoMiner integration.

The worker does BLS + CNN + TRICERATOPS + Claude in its regular venv.
For ExoMiner, we call out to the isolated ``venv_tf`` via subprocess so
the main venv stays untouched (TF 2.13 pins numpy<1.25 which is
incompatible with scipy/wotan >=1.26).

Flow per TIC:

 1. ``GET /api/exominer/tfrecord/{tic_id}`` on the central server.
    - 200 → save TFRecord locally, continue.
    - 202 → server is preprocessing; skip ExoMiner this run, cache warms for
       the next one.
    - 404 → TIC has no DV XML at MAST; skip permanently.
 2. Spawn ``venv_tf/python exominer_infer_cli.py --tfrecord ... --tic-id ...``.
 3. Parse the JSON scores.
 4. Match each BLS candidate to the closest TCE by period (with ½/2×
    aliases) and decorate it with ``exominer_score`` + ``exominer_tce_period``.

The worker config keys that drive this are:
  - ``exominer_enabled`` (bool, default ``False``)
  - ``exominer_venv_python`` (str, full path to the TF-capable python)
  - ``exominer_cli_path``   (str, path to exominer_infer_cli.py — default
     ``worker_dir/exominer_infer_cli.py``)
  - ``exominer_timeout_sec`` (int, default 120)
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import requests


# Period matching: shared single source of truth for the worker side
# (exominer_period_match.py — the server mirrors it in exonit/exominer_service.py).
from exominer_period_match import (  # noqa: F401
    PERIOD_TOL, PERIOD_FACTORS, match_period as _match_period,
)


@dataclass
class ExoMinerScore:
    tce_num: int
    period: float
    t0: float
    depth_ppm: float
    score: float


def fetch_tfrecord(
    server_url: str,
    tic_id: str,
    token: str,
    dest_dir: str,
    timeout: int = 30,
) -> Optional[Dict[str, str]]:
    """GET /api/exominer/tfrecord/{tic_id}. Returns dict with ``path`` and
    ``sector_run`` on cache hit, ``None`` on 202/404/error.
    """
    url = f"{server_url.rstrip('/')}/api/exominer/tfrecord/{tic_id}"
    headers = {"Authorization": f"Bearer {token}"}
    try:
        r = requests.get(url, headers=headers, timeout=timeout, stream=True)
    except requests.RequestException as e:
        logging.warning("exominer fetch_tfrecord %s: request failed: %s", tic_id, e)
        return None

    if r.status_code == 200:
        sector_run = r.headers.get("x-sector-run", "")
        os.makedirs(dest_dir, exist_ok=True)
        fname = f"{tic_id}_{sector_run or 'unknown'}.tfrecord"
        fp = os.path.join(dest_dir, fname)
        with open(fp, "wb") as f:
            for chunk in r.iter_content(chunk_size=65536):
                if chunk:
                    f.write(chunk)
        return {"path": fp, "sector_run": sector_run}
    if r.status_code in (202, 404):
        return None
    logging.warning("exominer fetch_tfrecord %s: status=%d", tic_id, r.status_code)
    return None


def run_inference_cli(
    venv_python: str,
    cli_path: str,
    tfrecord_path: str,
    tic_id: str,
    timeout: int = 120,
) -> Optional[List[Dict[str, Any]]]:
    """Invoke exominer_infer_cli.py. Returns the ``tces`` list or None on error."""
    cmd = [
        venv_python, cli_path,
        "--tfrecord", tfrecord_path,
        "--tic-id", str(tic_id),
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        logging.warning("exominer inference TIC %s: timeout after %ds", tic_id, timeout)
        return None
    except FileNotFoundError as e:
        logging.warning("exominer inference TIC %s: venv python not found: %s", tic_id, e)
        return None
    if r.returncode != 0:
        logging.warning("exominer inference TIC %s: exit=%d stderr=%s",
                        tic_id, r.returncode, (r.stderr or "")[-300:])
        return None
    try:
        payload = json.loads(r.stdout.strip().splitlines()[-1])
        return payload.get("tces", [])
    except (json.JSONDecodeError, IndexError) as e:
        logging.warning("exominer inference TIC %s: bad JSON: %s", tic_id, e)
        return None


def score_candidates(
    tic_id: str,
    candidates: List[Dict[str, Any]],
    *,
    server_url: str,
    token: str,
    venv_python: str,
    cli_path: str,
    work_dir: Optional[str] = None,
    timeout: int = 120,
) -> Optional[Dict[str, Any]]:
    """Score a TIC's BLS candidates with ExoMiner.

    Mutates each candidate dict in place to add ``exominer_score``,
    ``exominer_tce_period``, ``exominer_sector_run`` when a matching
    TCE is found. Returns a small summary dict or None on skip.
    """
    if not candidates:
        return None
    work_dir = work_dir or tempfile.gettempdir()
    fetched = fetch_tfrecord(server_url, tic_id, token, work_dir, timeout=timeout)
    if fetched is None:
        return None  # 202/404/error — skip this run
    tfrecord_path = fetched["path"]
    sector_run = fetched["sector_run"]

    tces = run_inference_cli(venv_python, cli_path, tfrecord_path, tic_id, timeout=timeout)
    try:
        os.remove(tfrecord_path)
    except OSError:
        pass

    if tces is None:
        return None
    if not tces:
        return {"tic_id": tic_id, "sector_run": sector_run, "n_matched": 0, "n_tces": 0}

    n_matched = 0
    for c in candidates:
        matched = _match_period(tces, c.get("period_days"))
        if matched is None:
            continue
        c["exominer_score"] = float(matched.get("score", 0.0))
        c["exominer_tce_period"] = float(matched.get("period", 0.0))
        c["exominer_sector_run"] = sector_run
        n_matched += 1

    return {
        "tic_id": tic_id,
        "sector_run": sector_run,
        "n_tces": len(tces),
        "n_matched": n_matched,
    }
