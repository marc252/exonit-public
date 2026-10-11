"""Tests for the SPOC-executor fast-shutdown hook (2026-08-01).

A recycled uvicorn worker must NOT linger draining its queued SPOC/MAST
checks after lifespan shutdown: _shutdown_spoc_executor_fast cancels the
never-started backlog so the process can exit as soon as the in-flight
checks (one per pool thread) finish, and create_app registers it as a
Starlette shutdown handler so it fires once per worker process.
"""
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

import exonit.server as server_module
from exonit.server import _shutdown_spoc_executor_fast


@pytest.fixture
def spoc_executor(monkeypatch):
    """Install a real 1-thread executor as the module-level SPOC executor."""
    ex = ThreadPoolExecutor(max_workers=1, thread_name_prefix="spoc-test")
    monkeypatch.setattr(server_module, "_spoc_executor", ex)
    yield ex
    ex.shutdown(wait=False, cancel_futures=True)


def test_fast_shutdown_cancels_queued_not_running(spoc_executor):
    release = threading.Event()
    started = threading.Event()

    def blocker():
        started.set()
        release.wait(timeout=10)
        return "ran"

    running = spoc_executor.submit(blocker)
    assert started.wait(timeout=5)
    queued = [spoc_executor.submit(lambda: "ran") for _ in range(5)]

    _shutdown_spoc_executor_fast()

    # Queued-but-never-started work is dropped; the in-flight one completes.
    assert all(f.cancelled() for f in queued)
    release.set()
    assert running.result(timeout=5) == "ran"

    # The executor no longer accepts work — a late submit must raise, never
    # silently spawn fresh threads in a shutting-down process.
    with pytest.raises(RuntimeError):
        spoc_executor.submit(lambda: None)


def test_fast_shutdown_noop_without_executor(monkeypatch):
    monkeypatch.setattr(server_module, "_spoc_executor", None)
    _shutdown_spoc_executor_fast()  # must not raise


def test_create_app_registers_shutdown_handler(monkeypatch, tmp_path):
    monkeypatch.setenv("EXONIT_DB_PATH", str(tmp_path / "test.db"))
    monkeypatch.setenv("EXONIT_REPORTS_DIR", str(tmp_path / "reports"))
    monkeypatch.setenv("EXONIT_PIPELINE_STATE_FILE",
                       str(tmp_path / "pipeline_state.json"))
    # Isolate from the frozen production metadata index (see
    # test_discovery_api.py's fixture for the full rationale).
    monkeypatch.setenv("EXONIT_METADATA_CACHE",
                       str(tmp_path / "metadata_cache.db"))
    monkeypatch.delenv("EXONIT_METADATA_DB_URL", raising=False)

    app = server_module.create_app()
    assert _shutdown_spoc_executor_fast in app.router.on_shutdown
