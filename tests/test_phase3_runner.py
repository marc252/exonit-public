"""Free function run_phase3_always_on must score candidates regardless of
data_source, fetching FFI from server when SPOC is the loaded LC.

Tests use dependency injection (score_fn/fetch_fn parameters) instead of
monkeypatch so they exercise the production code path."""
import logging
import numpy as np


def test_phase3_runs_on_ffi_data_source_using_loaded_lc():
    from phase3_runner import run_phase3_always_on
    captured = {"score_calls": []}

    def fake_score_ffi(*, tic_id, candidates, time, flux, star_meta, onnx_path,
                       gpu_enabled=True):
        captured["score_calls"].append({"tic_id": tic_id, "n_pts": len(time),
                                         "gpu_enabled": gpu_enabled})
        for c in candidates:
            c["exominer_ffi_score"] = 0.7
            c["exominer_ffi_model_version"] = "phase3c_seed42_v1"
        return {"n_matched": len(candidates), "n_tces": len(candidates)}

    candidates = [{"candidate_num": 1, "period_days": 2.0, "t0": 1.0,
                    "duration_days": 0.1, "depth_ppm": 1000.0, "snr": 12.0}]
    run_phase3_always_on(
        tic_id="123",
        candidates=candidates,
        loaded_time=np.linspace(0, 27, 100), loaded_flux=np.ones(100),
        data_source="ffi",
        star_meta={"tmag": 11.0, "teff": 5500.0, "logg": 4.4,
                   "stellar_radius": 1.0},
        config={"server_url": "http://srv:8400",
                "worker_token": "tok",
                "exominer_ffi_onnx_path": "/dev/null"},
        logger=logging.getLogger("test"),
        score_fn=fake_score_ffi,
    )
    assert len(captured["score_calls"]) == 1
    assert candidates[0]["exominer_ffi_score"] == 0.7


def test_phase3_runs_on_spoc_tic_fetching_ffi_from_server():
    from phase3_runner import run_phase3_always_on
    captured = {"fetched": False, "score_calls": []}

    def fake_fetch(server_url, tic_id, *, token, timeout=30.0):
        captured["fetched"] = True
        return (np.linspace(0, 27, 100),
                np.ones(100), np.zeros(100, dtype=int))

    def fake_score_ffi(*, tic_id, candidates, time, flux, star_meta, onnx_path,
                       gpu_enabled=True):
        captured["score_calls"].append(tic_id)
        for c in candidates:
            c["exominer_ffi_score"] = 0.55
            c["exominer_ffi_model_version"] = "phase3c_seed42_v1"
        return {"n_matched": 1, "n_tces": 1}

    candidates = [{"candidate_num": 1, "period_days": 2.0, "t0": 1.0,
                    "duration_days": 0.1, "depth_ppm": 1000.0, "snr": 12.0}]
    run_phase3_always_on(
        tic_id="123", candidates=candidates,
        loaded_time=np.linspace(0, 27, 100), loaded_flux=np.ones(100),
        data_source="spoc",
        star_meta={"tmag": 11.0, "teff": 5500.0, "logg": 4.4,
                   "stellar_radius": 1.0},
        config={"server_url": "http://srv:8400",
                "worker_token": "tok",
                "exominer_ffi_onnx_path": "/dev/null"},
        logger=logging.getLogger("test"),
        score_fn=fake_score_ffi,
        fetch_fn=fake_fetch,
    )
    assert captured["fetched"], "SPOC TIC must trigger server FFI fetch"
    assert candidates[0]["exominer_ffi_score"] == 0.55


def test_phase3_skips_on_ffi_fetch_failure():
    from phase3_runner import run_phase3_always_on
    score_called = []

    def fake_fetch_returns_none(*args, **kwargs):
        return None  # simulate 404 / timeout

    candidates = [{"candidate_num": 1, "period_days": 2.0, "t0": 1.0,
                    "duration_days": 0.1, "depth_ppm": 1000.0, "snr": 12.0}]
    run_phase3_always_on(
        tic_id="123", candidates=candidates,
        loaded_time=None, loaded_flux=None,
        data_source="spoc", star_meta={},
        config={"server_url": "http://srv:8400", "worker_token": "tok"},
        logger=logging.getLogger("test"),
        score_fn=lambda **kw: score_called.append(kw),
        fetch_fn=fake_fetch_returns_none,
    )
    assert score_called == []
    assert "exominer_ffi_score" not in candidates[0]


def test_phase3_skips_on_ffi_data_source_with_none_lc():
    """Defensive guard: data_source=ffi but LC arrays are None → skip."""
    from phase3_runner import run_phase3_always_on
    score_called = []

    candidates = [{"candidate_num": 1, "period_days": 2.0, "t0": 1.0,
                    "duration_days": 0.1, "depth_ppm": 1000.0, "snr": 12.0}]
    run_phase3_always_on(
        tic_id="123", candidates=candidates,
        loaded_time=None, loaded_flux=None,
        data_source="ffi", star_meta={},
        config={"server_url": "http://srv:8400", "worker_token": "tok"},
        logger=logging.getLogger("test"),
        score_fn=lambda **kw: score_called.append(kw),
    )
    assert score_called == []
    assert "exominer_ffi_score" not in candidates[0]
