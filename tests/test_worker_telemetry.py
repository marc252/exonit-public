"""Tests for worker telemetry payload construction."""
import time
from collections import deque


def _make_worker_stub():
    """Create a minimal Worker-like object for testing heartbeat payload."""
    class WorkerStub:
        def __init__(self):
            self.worker_id = "w-test123"
            self.status = "working"
            self.current_tic = 254113311
            self.current_phase = "bls"
            self.phase_progress = 0.42
            self._star_start_time = time.time() - 15.0
            self.stars_completed = 10
            self.total_processing_time = 300.0
            self.last_error = None
            self.debug_mode = False
            self.recent_timings = deque(maxlen=5)
            self.recent_timings.append({
                "tic": 123, "total_s": 30.0,
                "phases": {"load": 1.0, "bls": 10.0, "vetting": 1.0,
                           "cnn": 0.5, "triceratops": 12.0, "injection": 5.5}
            })

        def _star_elapsed(self):
            if self._star_start_time is None:
                return 0.0
            return round(time.time() - self._star_start_time, 1)

        def _build_heartbeat_payload(self):
            payload = {
                "worker_id": self.worker_id,
                "status": self.status,
                "current_tic": self.current_tic,
                "phase": self.current_phase,
                "phase_progress": self.phase_progress,
                "star_elapsed_s": self._star_elapsed(),
                "stars_done": self.stars_completed,
                "avg_time_per_star": round(
                    self.total_processing_time / max(1, self.stars_completed), 1),
                "last_error": self.last_error,
            }
            if self.debug_mode:
                import psutil
                proc = psutil.Process()
                vm = psutil.virtual_memory()
                temps = psutil.sensors_temperatures() if hasattr(psutil, 'sensors_temperatures') else {}
                cpu_temp = None
                if temps:
                    for entries in temps.values():
                        if entries:
                            cpu_temp = entries[0].current
                            break
                payload["system"] = {
                    "cpu_percent": psutil.cpu_percent(interval=None),
                    "ram_used_gb": round(vm.used / (1024**3), 2),
                    "ram_total_gb": round(vm.total / (1024**3), 2),
                    "cpu_temp_c": cpu_temp,
                    "pid_rss_mb": round(proc.memory_info().rss / (1024**2), 1),
                }
                payload["recent_timings"] = list(self.recent_timings)
            return payload

    return WorkerStub()


def test_baseline_payload_has_phase_fields():
    w = _make_worker_stub()
    payload = w._build_heartbeat_payload()
    assert payload["phase"] == "bls"
    assert payload["phase_progress"] == 0.42
    assert payload["stars_done"] == 10
    assert payload["avg_time_per_star"] == 30.0
    assert payload["star_elapsed_s"] > 0
    assert "system" not in payload
    assert "recent_timings" not in payload


def test_debug_payload_includes_system():
    w = _make_worker_stub()
    w.debug_mode = True
    payload = w._build_heartbeat_payload()
    assert "system" in payload
    assert "cpu_percent" in payload["system"]
    assert "ram_used_gb" in payload["system"]
    assert "pid_rss_mb" in payload["system"]
    assert "recent_timings" in payload
    assert len(payload["recent_timings"]) == 1


def test_idle_worker_has_null_phase():
    w = _make_worker_stub()
    w.status = "idle"
    w.current_phase = None
    w.phase_progress = None
    w.current_tic = None
    w._star_start_time = None
    payload = w._build_heartbeat_payload()
    assert payload["phase"] is None
    assert payload["phase_progress"] is None
    assert payload["current_tic"] is None
