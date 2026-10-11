"""TDD test: score_candidates_ffi must write exominer_ffi_score, not exominer_score."""
import numpy as np


def test_score_candidates_ffi_writes_to_exominer_ffi_score(monkeypatch):
    """Score must land in exominer_ffi_score, not exominer_score."""
    from worker.exominer_ffi_inference import score_candidates_ffi

    # Stub the inferer so the test does not need the real ONNX model.
    class _StubInferer:
        def score(self, *args, **kwargs):
            return 0.42

    import worker.exominer_ffi_inference as mod
    # `_get_inferer` now takes an extra `gpu_enabled` kwarg so the cached
    # session honours the worker's GPU flag — accept and ignore here.
    monkeypatch.setattr(mod, "_get_inferer",
                        lambda path, gpu_enabled=True: _StubInferer())

    cands = [{
        "candidate_num": 1,
        "period_days": 2.0,
        "t0": 1.0,
        "duration_days": 0.1,
        "depth_ppm": 1000.0,
        "snr": 12.0,
    }]
    rng = np.random.default_rng(0)
    time = np.linspace(0, 27, 3000)
    flux = np.ones_like(time) + 0.001 * rng.standard_normal(3000)
    in_t = np.abs((time - 1.0) % 2.0) < 0.05
    flux[in_t] -= 0.02

    # NOTE: existing star_meta uses keys "teff" and "stellar_radius"
    # (NOT "teff_k" and "rstar_solar"). Match the contract the
    # production code already accepts.
    summary = score_candidates_ffi(
        tic_id="12345",
        candidates=cands,
        time=time, flux=flux,
        star_meta={"tmag": 11.0, "teff": 5500.0,
                   "logg": 4.4, "stellar_radius": 1.0},
        onnx_path="/dev/null",  # stubbed away
    )
    assert "exominer_ffi_score" in cands[0], \
        f"missing exominer_ffi_score; got keys {list(cands[0].keys())}"
    assert "exominer_score" not in cands[0], \
        "exominer_score must NOT be written by the FFI inferer"
    assert 0.0 <= cands[0]["exominer_ffi_score"] <= 1.0
    from exominer_ffi_inference import FFI_MODEL_VERSION
    # assert against the constant, not a hardcoded snapshot: the model version
    # changes on every promoted retrain (was stale at "phase3c_seed42_v1").
    assert cands[0].get("exominer_ffi_model_version") == FFI_MODEL_VERSION
