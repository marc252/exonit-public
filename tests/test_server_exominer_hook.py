"""Tests for the ExoMiner background hook on /api/worker/result.

The hook is gated by the EXONIT_ENABLE_EXOMINER_AUTO env var; we test
that enabling it triggers a background ExoMiner run that updates the
candidate's exominer_score, and that disabling it leaves the candidate
untouched.
"""
import threading
import time
from unittest.mock import patch, MagicMock

import pytest

import exonit.server as server_module
from exonit.dist_semaphore import DistributedSemaphore
from exonit.server import _run_exominer_background, create_app
from exonit.db import ExoNitDB
from exonit.exominer_service import ExoMinerTCE, PredictResult


@pytest.fixture
def db(tmp_path):
    d = ExoNitDB(str(tmp_path / "test.db"))
    yield d
    d.close()


def test_run_exominer_background_updates_matching_candidate(db):
    db.add_star("600", ra=0, dec=0, tmag=10, n_sectors=1,
                n_datapoints=1000, time_span_days=30)
    db.add_candidate("600", candidate_num=1, period_days=14.64, t0=1337.75,
                     duration_days=0.1, depth_ppm=245, snr=11.4)

    fake_tce = ExoMinerTCE(
        tic_id="600", tce_num=1, period=14.64, duration=0.1, t0=1337.75,
        depth_ppm=245, ruwe=1.72, prad=3.48, mes=11.4, score=0.873,
    )
    fake_result = PredictResult(
        tic_id="600", sector_run="1-69", model="exominer++_single",
        tces=[fake_tce], from_cache=True,
    )

    with patch("exonit.exominer_service.ExoMinerService") as MockSvc:
        svc = MockSvc.return_value
        svc.predict_tic.return_value = fake_result
        _run_exominer_background(db, "600", [{"candidate_num": 1, "period_days": 14.64}])

    c = db.get_candidates("600")[0]
    assert c["exominer_score"] == pytest.approx(0.873)
    assert c["exominer_tce_period"] == pytest.approx(14.64)
    assert c["exominer_sector_run"] == "1-69"


def test_run_exominer_background_no_tces_leaves_candidate_untouched(db):
    db.add_star("601", ra=0, dec=0, tmag=10, n_sectors=1,
                n_datapoints=1000, time_span_days=30)
    db.add_candidate("601", candidate_num=1, period_days=5.0, t0=0,
                     duration_days=0.1, depth_ppm=500, snr=15)

    empty_result = PredictResult(
        tic_id="601", sector_run=None, model="exominer++_single",
        tces=[], from_cache=False, reason="no_dv_runs",
    )

    with patch("exonit.exominer_service.ExoMinerService") as MockSvc:
        svc = MockSvc.return_value
        svc.predict_tic.return_value = empty_result
        _run_exominer_background(db, "601", [{"candidate_num": 1, "period_days": 5.0}])

    c = db.get_candidates("601")[0]
    assert c["exominer_score"] is None


def test_run_exominer_background_no_matching_period(db):
    """TCE with very different period should not update the BLS candidate."""
    db.add_star("602", ra=0, dec=0, tmag=10, n_sectors=1,
                n_datapoints=1000, time_span_days=30)
    db.add_candidate("602", candidate_num=1, period_days=5.0, t0=0,
                     duration_days=0.1, depth_ppm=500, snr=15)

    unrelated_tce = ExoMinerTCE(
        tic_id="602", tce_num=1, period=100.0, duration=0.5, t0=0,
        depth_ppm=1000, ruwe=1.0, prad=5.0, mes=20.0, score=0.9,
    )
    fake_result = PredictResult(
        tic_id="602", sector_run="1-39", model="exominer++_single",
        tces=[unrelated_tce], from_cache=True,
    )

    with patch("exonit.exominer_service.ExoMinerService") as MockSvc:
        svc = MockSvc.return_value
        svc.predict_tic.return_value = fake_result
        _run_exominer_background(db, "602", [{"candidate_num": 1, "period_days": 5.0}])

    c = db.get_candidates("602")[0]
    assert c["exominer_score"] is None


def test_run_exominer_background_swallows_exceptions(db):
    """A failing ExoMinerService must not crash the hook thread."""
    db.add_star("603", ra=0, dec=0, tmag=10, n_sectors=1,
                n_datapoints=1000, time_span_days=30)
    db.add_candidate("603", candidate_num=1, period_days=5.0, t0=0,
                     duration_days=0.1, depth_ppm=500, snr=15)

    with patch("exonit.exominer_service.ExoMinerService") as MockSvc:
        svc = MockSvc.return_value
        svc.predict_tic.side_effect = RuntimeError("docker down")
        # Must not raise
        _run_exominer_background(db, "603", [{"candidate_num": 1, "period_days": 5.0}])

    c = db.get_candidates("603")[0]
    assert c["exominer_score"] is None


# --------- ExoMiner Docker concurrency semaphore ---------

def _reset_exominer_semaphore():
    server_module._exominer_docker_semaphore = None


def test_exominer_docker_semaphore_is_bounded(monkeypatch):
    """Default cap is 1 so the OOM-2026-05-15 regression cannot reappear.

    A2b: semaphore is now a DistributedSemaphore (cross-process lease) rather
    than a threading.BoundedSemaphore, but the external API is identical.
    """
    _reset_exominer_semaphore()
    monkeypatch.delenv("EXONIT_EXOMINER_MAX_CONCURRENT", raising=False)
    sem = server_module._get_exominer_docker_semaphore()
    assert isinstance(sem, DistributedSemaphore)
    # Same instance returned on subsequent calls (lazy singleton).
    assert server_module._get_exominer_docker_semaphore() is sem
    # Default = 1: a single non-blocking acquire succeeds, a second fails.
    assert sem.acquire(blocking=False) is True
    assert sem.acquire(blocking=False) is False
    sem.release()
    _reset_exominer_semaphore()


def test_exominer_docker_semaphore_respects_env(monkeypatch):
    """EXONIT_EXOMINER_MAX_CONCURRENT controls the DistributedSemaphore limit."""
    _reset_exominer_semaphore()
    monkeypatch.setenv("EXONIT_EXOMINER_MAX_CONCURRENT", "3")
    sem = server_module._get_exominer_docker_semaphore()
    assert sem.acquire(blocking=False) is True
    assert sem.acquire(blocking=False) is True
    assert sem.acquire(blocking=False) is True
    assert sem.acquire(blocking=False) is False
    for _ in range(3):
        sem.release()
    _reset_exominer_semaphore()


def test_run_exominer_background_serializes_on_semaphore(db, monkeypatch):
    """Concurrent invocations must not both enter the docker section at once.

    With max_concurrent=1, the second thread must wait for the first to
    release before the inner function (which spawns docker) is entered.
    """
    _reset_exominer_semaphore()
    monkeypatch.delenv("EXONIT_EXOMINER_MAX_CONCURRENT", raising=False)

    db.add_star("700", ra=0, dec=0, tmag=10, n_sectors=1,
                n_datapoints=1000, time_span_days=30)
    db.add_candidate("700", candidate_num=1, period_days=5.0, t0=0,
                     duration_days=0.1, depth_ppm=500, snr=15)

    in_inner = threading.Event()
    can_finish = threading.Event()
    overlap_seen = threading.Event()
    active = {"count": 0}
    lock = threading.Lock()

    def fake_inner(db_arg, tic_id, cands):
        with lock:
            active["count"] += 1
            if active["count"] > 1:
                overlap_seen.set()
        in_inner.set()
        can_finish.wait(timeout=2.0)
        with lock:
            active["count"] -= 1

    with patch.object(server_module, "_run_exominer_background_inner",
                      side_effect=fake_inner):
        t1 = threading.Thread(
            target=_run_exominer_background,
            args=(db, "700", [{"candidate_num": 1, "period_days": 5.0}]),
        )
        t2 = threading.Thread(
            target=_run_exominer_background,
            args=(db, "700", [{"candidate_num": 1, "period_days": 5.0}]),
        )
        t1.start()
        in_inner.wait(timeout=2.0)
        # t1 is inside the semaphore. Start t2 and give it a moment to try.
        t2.start()
        time.sleep(0.2)
        # t2 must still be blocked (active count never exceeded 1).
        assert overlap_seen.is_set() is False, \
            "semaphore did not serialize: both threads entered _run_docker"
        # Release t1, then t2.
        can_finish.set()
        t1.join(timeout=2.0)
        t2.join(timeout=2.0)

    _reset_exominer_semaphore()


# --------- ExoMiner Docker resource limits ---------

def test_run_docker_includes_memory_and_cpu_flags(tmp_path, monkeypatch):
    """`docker run` must include --memory and --cpus so a single container
    cannot exhaust the host (regression: 2026-05-15 GCP server OOM)."""
    from exonit.exominer_service import ExoMinerService

    monkeypatch.setenv("EXONIT_EXOMINER_MEMORY_LIMIT", "3g")
    monkeypatch.setenv("EXONIT_EXOMINER_CPU_LIMIT", "1.5")

    svc = ExoMinerService(
        work_dir=str(tmp_path), cache_db=str(tmp_path / "cache.db"),
    )
    try:
        with patch("exonit.exominer_service.subprocess.run") as mock_run:
            svc._run_docker("700", "1-69", "exominer++_single")
        cmd = mock_run.call_args[0][0]
        assert "--memory" in cmd
        assert cmd[cmd.index("--memory") + 1] == "3g"
        assert "--cpus" in cmd
        assert cmd[cmd.index("--cpus") + 1] == "1.5"
    finally:
        svc.close()


def test_run_docker_omits_flags_when_env_empty(tmp_path, monkeypatch):
    """Setting an empty env var disables the corresponding flag."""
    from exonit.exominer_service import ExoMinerService

    monkeypatch.setenv("EXONIT_EXOMINER_MEMORY_LIMIT", "")
    monkeypatch.setenv("EXONIT_EXOMINER_CPU_LIMIT", "")

    svc = ExoMinerService(
        work_dir=str(tmp_path), cache_db=str(tmp_path / "cache.db"),
    )
    try:
        with patch("exonit.exominer_service.subprocess.run") as mock_run:
            svc._run_docker("701", "1-69", "exominer++_single")
        cmd = mock_run.call_args[0][0]
        assert "--memory" not in cmd
        assert "--cpus" not in cmd
    finally:
        svc.close()
