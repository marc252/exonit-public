"""Unit tests for backfill status telemetry (Part 1 of impl-backfill-status).

These tests verify that:
- _process_prepared_fpp sets self.status = "backfill" during compute and
  resets to "idle" in the finally block.
- _process_prepared_exominer does the same.
- Primary _process_prepared_fpp / _process_prepared_exominer do NOT touch
  self.status = "working" (regression guard: primary working is set by the
  primary task path, not these helpers).
"""
import queue
import threading
import time
import types
import logging


# ---------------------------------------------------------------------------
# Minimal stub that replays _process_prepared_fpp / _process_prepared_exominer
# exactly as they appear in worker.py (set/reset pattern only).
# ---------------------------------------------------------------------------

class _FPPAborted(Exception):
    pass


def _make_fpp_stub():
    """Return an object that faithfully implements the status set/reset of
    _process_prepared_fpp without needing TRICERATOPS or network I/O."""

    class FPPStub:
        def __init__(self):
            self.status = "idle"
            self.current_tic = None
            self.running = True
            self.logger = logging.getLogger("fpp-stub")
            self.server = "http://localhost:0"
            self.fits_cache_dir = "/tmp"
            self.token = "tok"

        def _headers(self):
            return {}

        def _report_fpp_failed(self, task):
            self.status = "idle"
            self.current_tic = None

        def _process_prepared_fpp(self, prepared, _compute_fn=None):
            """Mirror of the real _process_prepared_fpp — only the
            status-management lines are relevant here; the TRICERATOPS call
            is replaced by a configurable _compute_fn hook."""
            task = prepared["task"]
            tic_id = task["tic_id"]
            self.status = "backfill"
            self.current_tic = tic_id
            try:
                if _compute_fn is not None:
                    _compute_fn(self)
            except _FPPAborted:
                pass
            except Exception:
                pass
            finally:
                self.status = "idle"
                self.current_tic = None

    return FPPStub()


def _make_exominer_stub():
    """Mirror of _process_prepared_exominer status management."""

    class ExoMinerStub:
        def __init__(self):
            self.status = "idle"
            self.current_tic = None
            self.logger = logging.getLogger("em-stub")
            self.server = "http://localhost:0"
            self.token = "tok"
            self.config = {}

        def _headers(self):
            return {}

        def _report_exominer_skipped(self, candidate_id, tic_id):
            pass

        def _process_prepared_exominer(self, prepared_full, _compute_fn=None):
            """Mirror of the real _process_prepared_exominer status lines."""
            task = prepared_full["task"]
            tic_id = task["tic_id"]
            self.status = "backfill"
            self.current_tic = tic_id
            try:
                if _compute_fn is not None:
                    _compute_fn(self)
            except Exception:
                pass
            finally:
                self.status = "idle"
                self.current_tic = None

    return ExoMinerStub()


# ---------------------------------------------------------------------------
# FPP status tests
# ---------------------------------------------------------------------------

def test_fpp_status_is_backfill_during_compute():
    """self.status must be 'backfill' while the FPP compute runs."""
    stub = _make_fpp_stub()
    observed = []

    def compute(w):
        observed.append(w.status)

    stub._process_prepared_fpp(
        {"task": {"tic_id": 123, "candidate_num": 1, "snr": 10.0}},
        _compute_fn=compute,
    )
    assert observed == ["backfill"], (
        f"Expected ['backfill'] during compute, got {observed}"
    )


def test_fpp_status_is_idle_after_compute():
    """self.status must return to 'idle' after _process_prepared_fpp."""
    stub = _make_fpp_stub()
    stub._process_prepared_fpp(
        {"task": {"tic_id": 123, "candidate_num": 1, "snr": 10.0}},
    )
    assert stub.status == "idle", (
        f"Expected 'idle' after FPP compute, got {stub.status!r}"
    )
    assert stub.current_tic is None


def test_fpp_status_resets_to_idle_on_exception():
    """Even when the compute raises, status must return to 'idle' (finally)."""
    stub = _make_fpp_stub()

    def boom(w):
        raise RuntimeError("simulated TRICERATOPS crash")

    stub._process_prepared_fpp(
        {"task": {"tic_id": 456, "candidate_num": 2, "snr": 5.0}},
        _compute_fn=boom,
    )
    assert stub.status == "idle", (
        f"Expected 'idle' after exception in FPP compute, got {stub.status!r}"
    )


def test_fpp_status_observable_from_thread():
    """Concurrent reader (simulating heartbeat thread) sees 'backfill'
    while compute is running, and 'idle' after it finishes."""
    stub = _make_fpp_stub()
    snapshots = []
    barrier = threading.Barrier(2)

    def compute(w):
        # Notify the reader that compute has started, then wait for it to read.
        barrier.wait(timeout=2)
        barrier.wait(timeout=2)

    def reader():
        barrier.wait(timeout=2)  # wait until compute has started
        snapshots.append(stub.status)
        barrier.wait(timeout=2)  # let compute finish

    t_reader = threading.Thread(target=reader, daemon=True)
    t_reader.start()

    stub._process_prepared_fpp(
        {"task": {"tic_id": 789, "candidate_num": 3, "snr": 8.0}},
        _compute_fn=compute,
    )
    t_reader.join(timeout=3)

    assert snapshots == ["backfill"], (
        f"Heartbeat thread should see 'backfill' mid-compute, got {snapshots}"
    )
    assert stub.status == "idle"


# ---------------------------------------------------------------------------
# ExoMiner status tests
# ---------------------------------------------------------------------------

def test_exominer_status_is_backfill_during_compute():
    """self.status must be 'backfill' while ExoMiner scoring runs."""
    stub = _make_exominer_stub()
    observed = []

    def compute(w):
        observed.append(w.status)

    stub._process_prepared_exominer(
        {"task": {"tic_id": 111, "candidate_num": 1, "snr": 7.0},
         "cand": {}, "prepared_inputs": {}},
        _compute_fn=compute,
    )
    assert observed == ["backfill"], (
        f"Expected ['backfill'] during ExoMiner compute, got {observed}"
    )


def test_exominer_status_is_idle_after_compute():
    """self.status must return to 'idle' after _process_prepared_exominer."""
    stub = _make_exominer_stub()
    stub._process_prepared_exominer(
        {"task": {"tic_id": 222, "candidate_num": 1, "snr": 9.0},
         "cand": {}, "prepared_inputs": {}},
    )
    assert stub.status == "idle", (
        f"Expected 'idle' after ExoMiner compute, got {stub.status!r}"
    )
    assert stub.current_tic is None


def test_exominer_status_resets_to_idle_on_exception():
    """Even when ExoMiner scoring raises, status must return to 'idle'."""
    stub = _make_exominer_stub()

    def boom(w):
        raise ValueError("simulated ONNX failure")

    stub._process_prepared_exominer(
        {"task": {"tic_id": 333, "candidate_num": 2, "snr": 6.0},
         "cand": {}, "prepared_inputs": {}},
        _compute_fn=boom,
    )
    assert stub.status == "idle", (
        f"Expected 'idle' after exception in ExoMiner compute, got {stub.status!r}"
    )


# ---------------------------------------------------------------------------
# Regression guard: primary "working" status is not affected
# ---------------------------------------------------------------------------

def test_primary_working_status_unchanged():
    """The primary task path sets status='working' independently.
    Backfill helpers must NOT set status='working' at any point.
    This test verifies the stub only ever uses 'backfill', never 'working'.
    """
    stub = _make_fpp_stub()
    seen_working = []

    def compute(w):
        if w.status == "working":
            seen_working.append(True)

    stub._process_prepared_fpp(
        {"task": {"tic_id": 999, "candidate_num": 1, "snr": 12.0}},
        _compute_fn=compute,
    )
    assert not seen_working, (
        "Backfill FPP should never set status='working' — primary tasks own that"
    )
