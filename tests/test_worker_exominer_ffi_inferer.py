"""Worker-side ExoMiner-FFI inferer tests using a fresh in-memory ONNX export."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

# Make worker module importable for the test
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "worker"))


def _export_test_onnx(tmp_path):
    """Export a freshly-initialised ExoMinerFFI to ONNX for the test."""
    from exonit.exominer_ffi.model import ExoMinerFFI, export_to_onnx

    torch.manual_seed(0)
    model = ExoMinerFFI()
    model.eval()
    p = tmp_path / "exominer_ffi_test.onnx"
    export_to_onnx(model, p)
    return p


def _synthetic_lc(period=5.0, t0=2.5, depth_ppm=10000.0, duration=0.2):
    """Return time, flux for a simple synthetic transit."""
    t = np.linspace(0, 27, 5400)
    flux = np.ones_like(t)
    in_transit = ((t - t0) % period < duration / 2) | \
                 ((period - ((t - t0) % period)) < duration / 2)
    flux[in_transit] -= depth_ppm / 1e6
    return t, flux


def test_inferer_score_returns_probability(tmp_path):
    from worker.exominer_ffi_inference import ExoMinerFFIInferer

    onnx_path = _export_test_onnx(tmp_path)
    infer = ExoMinerFFIInferer(model_path=str(onnx_path))
    t, flux = _synthetic_lc()
    score = infer.score(
        t, flux, period=5.0, t0=2.5, duration=0.2,
        depth_ppm=10000.0, snr=20.0,
        stellar={"tmag": 12, "teff": 5500, "logg": 4.4, "rstar": 1.0},
    )
    assert 0.0 <= score <= 1.0


def test_score_candidates_ffi_mutates_in_place(tmp_path):
    from worker.exominer_ffi_inference import score_candidates_ffi

    onnx_path = _export_test_onnx(tmp_path)
    t, flux = _synthetic_lc()
    candidates = [
        {"period_days": 5.0, "t0": 2.5, "duration_days": 0.2,
         "depth_ppm": 10000.0, "snr": 20.0},
        {"period_days": 3.7, "t0": 1.0, "duration_days": 0.15,
         "depth_ppm": 5000.0, "snr": 10.0},
    ]
    star_meta = {"tmag": 12, "teff": 5500, "logg": 4.4, "stellar_radius": 1.0}
    summary = score_candidates_ffi(
        tic_id="999999",
        candidates=candidates,
        time=t, flux=flux,
        star_meta=star_meta,
        onnx_path=str(onnx_path),
    )
    assert summary["n_tces"] == 2
    assert summary["n_matched"] == 2
    assert "exominer_ffi_score" in candidates[0]
    assert 0.0 <= candidates[0]["exominer_ffi_score"] <= 1.0


def test_score_candidates_skips_zero_period(tmp_path):
    from worker.exominer_ffi_inference import score_candidates_ffi

    onnx_path = _export_test_onnx(tmp_path)
    t, flux = _synthetic_lc()
    candidates = [{"period_days": 0.0, "t0": 0, "duration_days": 0.2,
                   "depth_ppm": 1000, "snr": 5.0}]
    summary = score_candidates_ffi(
        tic_id="x", candidates=candidates, time=t, flux=flux,
        star_meta={}, onnx_path=str(onnx_path),
    )
    assert summary["n_matched"] == 0
    assert "exominer_ffi_score" not in candidates[0]
