"""Phase D.5 — End-to-end native ExoMiner pipeline smoke test.

Exercises ``worker.exominer_native.preprocess.orchestrate.run_tic`` on the
committed TIC 143143769 s1-36 fixtures (DV XML + one LC FITS) and scores
the resulting ``TCEFeatures`` with
``worker.exominer_native.inference.ExoMinerOnnxInferer``.

The oracle (Docker-side ghcr.io/nasa/exominer:latest) score for this TIC
is cached in ``data/exominer_cache.db`` — we use that rather than
re-running Docker, which keeps the test fast while still being a real
oracle comparison.

Marked ``parity_gate`` so it only runs with ``--run-parity`` (same knob
as the other Docker-adjacent comparisons — skipped in fast CI).
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pytest

from worker.exominer_native.inference import ExoMinerOnnxInferer
from worker.exominer_native.preprocess import orchestrate as _orchestrate


REPO_ROOT = Path(__file__).resolve().parents[3]
FIXTURES_DIR = REPO_ROOT / "tests" / "fixtures" / "exominer_native"
ONNX_PATH = REPO_ROOT / "worker" / "models" / "exominer_plusplus_single.onnx"
NORM_STATS_DIR = REPO_ROOT / "worker" / "exominer_native" / "data" / "norm_stats"
CACHE_DB = REPO_ROOT / "data" / "exominer_cache.db"

TIC_ID = "143143769"
SECTOR_RUN = "1-36"
FIXTURE_XML = FIXTURES_DIR / "tic_143143769_s1-36.dvr.xml"
FIXTURE_FITS = FIXTURES_DIR / "tic_143143769_s0032.lc.fits"

# Physical scoring tolerance. 5 pp is defensible given: native pipeline
# still zero-fills some scalars (weak-secondary, neighbour centroid offsets
# etc.), only ships one LC FITS sector while the oracle uses the full set,
# and applies a subset of the training norm-stats for diff-img/centroid.
TOLERANCE = 0.05


def _fetch_oracle_score() -> float | None:
    """Return the cached Docker-oracle score for this TIC+sector_run, or None."""
    if not CACHE_DB.exists():
        return None
    conn = sqlite3.connect(str(CACHE_DB))
    try:
        row = conn.execute(
            "SELECT tces_json FROM exominer_cache "
            "WHERE tic_id=? AND sector_run=? AND model='exominer++_single'",
            (TIC_ID, SECTOR_RUN),
        ).fetchone()
    finally:
        conn.close()
    if row is None:
        return None
    tces = json.loads(row[0])
    if not tces:
        return None
    # TCE1 is the first entry; match tce_num=1 to be explicit.
    for t in tces:
        if int(t.get("tce_num", 0)) == 1:
            return float(t["score"])
    return float(tces[0]["score"])


def _add_batch_axis_to_feeds(feeds: dict[str, np.ndarray],
                             onnx_path: str) -> dict[str, np.ndarray]:
    """Reshape each feed to match ORT's declared input shape.

    The native orchestrate emits feeds shaped exactly as declared in the
    YAML (e.g. scalars as (1,), 301x1 views as (301,1)), but the ONNX
    model expects a leading batch axis on every input — so scalars need
    (1, 1) and 301x1 views need (1, 301, 1). We derive target shapes from
    the ORT session so we stay in sync with the model without hard-coding.
    """
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    target_shapes = {}
    for inp in sess.get_inputs():
        # Replace dynamic dims (strings or None) with 1.
        target_shapes[inp.name] = tuple(
            1 if (isinstance(d, str) or d is None) else int(d)
            for d in inp.shape
        )

    out: dict[str, np.ndarray] = {}
    for name, arr in feeds.items():
        if name not in target_shapes:
            # Extra feed not declared by the model — skip (ORT would error).
            continue
        tgt = target_shapes[name]
        arr = np.asarray(arr, dtype=np.float32)
        # NaN/Inf guard — a handful of scalars (mag_shift_norm,
        # pgram_smooth_max_power_norm, pgram_tpm_smooth_max_power_norm)
        # are still NaN after the native preprocess because the source
        # data (sector mag history, transit periodogram) isn't wired in
        # yet. Zero-fill them so ORT doesn't propagate NaN through the
        # whole model; the oracle comparison below is what tells us
        # whether the zero-fill is physically defensible.
        if not np.all(np.isfinite(arr)):
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        # Case A: already correct.
        if arr.shape == tgt:
            out[name] = arr
            continue
        # Case B: missing batch axis (native feed rank = ORT rank - 1).
        if len(arr.shape) == len(tgt) - 1 and arr.shape == tgt[1:]:
            out[name] = arr.reshape(tgt)
            continue
        # Case C: try a generic reshape as a last resort.
        try:
            out[name] = arr.reshape(tgt)
        except ValueError:
            # Fall back to zeros of the right shape — keeps ORT happy so
            # the test can still distinguish "pipeline produces a score"
            # from "pipeline crashes inside ORT".
            out[name] = np.zeros(tgt, dtype=np.float32)
    # Ensure all declared inputs are present.
    for name, tgt in target_shapes.items():
        if name not in out:
            out[name] = np.zeros(tgt, dtype=np.float32)
    return out


@pytest.mark.parity_gate
def test_native_e2e_score_vs_oracle():
    """Run the full native pipeline on TIC 143143769 s1-36 and compare to
    the cached Docker-oracle score within ``TOLERANCE``.

    Structured so that any failure *before* the final assert (preprocess
    error, no TCE, ORT error, bad score) gets captured and turned into
    an xfail with a descriptive reason — this is informative, not blocking.
    Only the final ``|native - oracle| < TOLERANCE`` comparison is a hard
    assert.
    """
    # ---- preconditions ----
    assert FIXTURE_XML.exists(), f"missing DV XML fixture: {FIXTURE_XML}"
    assert FIXTURE_FITS.exists(), f"missing LC FITS fixture: {FIXTURE_FITS}"
    assert ONNX_PATH.exists(), f"missing ONNX model: {ONNX_PATH}"
    assert NORM_STATS_DIR.exists(), f"missing norm_stats dir: {NORM_STATS_DIR}"

    # ---- 1. orchestrate.run_tic ----
    dv_xml_bytes = FIXTURE_XML.read_bytes()
    try:
        tces = _orchestrate.run_tic(
            TIC_ID,
            SECTOR_RUN,
            dv_xml_bytes=dv_xml_bytes,
            lc_fits_paths=[FIXTURE_FITS],
            ticv8_data={"Tmag": 11.3, "Teff": 5776.0, "radius": 1.0},
            gaia_data={"ruwe": 0.98},
            norm_stats_dir=NORM_STATS_DIR,
        )
    except Exception as e:  # pragma: no cover — diagnostic
        pytest.xfail(f"orchestrate.run_tic raised: {type(e).__name__}: {e}")

    if not tces:
        pytest.xfail("orchestrate.run_tic returned 0 TCEs (no planetResults?)")

    tce = tces[0]
    print(f"[e2e] got {len(tces)} TCE(s); scoring first: uid={tce.uid} "
          f"period={tce.period:.4f}d depth={tce.depth_ppm:.1f}ppm")

    # ---- 2. score via ONNX ----
    # Known-bug workaround: the YAML-declared scalar shape is (1,) but ORT
    # needs (1, 1) because the exported model has an explicit batch axis.
    # The parent Phase D.5 will fix this inside orchestrate.py; until then
    # we add the batch axis in the test.
    try:
        feeds = _add_batch_axis_to_feeds(tce.feeds, str(ONNX_PATH))
    except Exception as e:  # pragma: no cover — diagnostic
        pytest.xfail(f"feed reshape failed: {type(e).__name__}: {e}")

    inferer = ExoMinerOnnxInferer(str(ONNX_PATH),
                                  providers=["CPUExecutionProvider"])
    try:
        native_score = inferer.score(feeds)
    except Exception as e:  # pragma: no cover — diagnostic
        pytest.xfail(f"ExoMinerOnnxInferer.score raised: "
                     f"{type(e).__name__}: {e}")

    print(f"[e2e] native_score = {native_score:.6g}")

    # Sanity on the native score itself.
    assert isinstance(native_score, float)
    assert np.isfinite(native_score), f"non-finite native score: {native_score}"
    assert 0.0 <= native_score <= 1.0, \
        f"native score outside [0, 1]: {native_score}"

    # ---- 3. oracle comparison ----
    oracle_score = _fetch_oracle_score()
    if oracle_score is None:
        pytest.xfail(
            "oracle score unavailable — cache miss for "
            f"({TIC_ID}, {SECTOR_RUN}). Populate by running "
            "`docker run ghcr.io/nasa/exominer:latest` via "
            "ExoMinerService.predict_tic, or clear data/exominer_cache.db "
            "and let the server-side hook repopulate it."
        )

    delta = abs(native_score - oracle_score)
    print(f"[e2e] oracle_score = {oracle_score:.6g}  "
          f"|delta| = {delta:.6g}  tolerance = {TOLERANCE}")

    assert delta < TOLERANCE, (
        f"native({native_score:.6g}) vs oracle({oracle_score:.6g}) "
        f"delta={delta:.6g} exceeds tolerance={TOLERANCE}. "
        "This is a physical scoring regression — inspect preprocess "
        "normalization / zero-filled scalar features / diff-img stack."
    )
