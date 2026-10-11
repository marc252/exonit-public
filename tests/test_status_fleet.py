"""Tests for the Status fleet map feature (Task 6 of ui-status-fleet-map).

Covers:
1. DB round-trip: insert_telemetry with a ``gpu`` block → get_latest_telemetry
   returns the five GPU metric columns.
2. ``GET /api/status/fleet``: HTTP endpoint with fake workers in workers_state,
   assertions on shape, grouping (two instances → one computer entry), and
   contract keys.
3. ``GET /status``: HTML route returns 200 with the page content.
4. Worker GPU collector: ``_get_gpu_telemetry`` returns all-None when no GPU
   hardware detected; correctly parses nvidia-smi output when CUDA is present.

Constraints:
- NEVER touches the live DB or production PostgreSQL.
- Uses FastAPI TestClient with a temp SQLite DB.
- No real subprocess calls (subprocess.run is monkeypatched).
"""

from __future__ import annotations

import json
import os
import shutil as _shutil_real
import subprocess as _subprocess_real
import sys
import time as _time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def db(tmp_path: Path, monkeypatch):
    """Path to a temp DB with tables already created (matches other tests' style)."""
    db_file = str(tmp_path / "test.db")
    reports_dir = str(tmp_path / "reports")
    os.makedirs(reports_dir, exist_ok=True)

    monkeypatch.setenv("EXONIT_DB_PATH", db_file)
    monkeypatch.setenv("EXONIT_REPORTS_DIR", reports_dir)
    monkeypatch.setenv(
        "EXONIT_PIPELINE_STATE_FILE",
        str(tmp_path / "pipeline_state.json"),
    )

    # Create tables (same as create_app does)
    from exonit.db import ExoNitDB
    d = ExoNitDB(db_file)
    del d  # close

    return Path(db_file)



@pytest.fixture()
def temp_db(tmp_path: Path, monkeypatch):
    """FastAPI TestClient with an isolated temp DB + exposed tmp_path."""
    db_file = str(tmp_path / "test.db")
    reports_dir = str(tmp_path / "reports")
    os.makedirs(reports_dir, exist_ok=True)

    monkeypatch.setenv("EXONIT_DB_PATH", db_file)
    monkeypatch.setenv("EXONIT_REPORTS_DIR", reports_dir)
    monkeypatch.setenv(
        "EXONIT_PIPELINE_STATE_FILE",
        str(tmp_path / "pipeline_state.json"),
    )

    from exonit.server import create_app
    app = create_app()

    from fastapi.testclient import TestClient
    with TestClient(app) as client:
        yield client, tmp_path, db_file


# ---------------------------------------------------------------------------
# 1. DB round-trip: GPU columns persist and round-trip
# ---------------------------------------------------------------------------


class TestDBRoundTrip:

    def test_gpu_columns_exist_after_create(self, db):
        """After init, the worker_telemetry table must have all five GPU columns."""
        from exonit.db import ExoNitDB
        db = ExoNitDB(str(db))  # re-create using fixture-returned path
        rows = db._query(
            "PRAGMA table_info(worker_telemetry)"
        )
        col_names = {r["name"] for r in rows}
        for col in ("gpu_util_pct", "gpu_mem_used_gb", "gpu_mem_total_gb",
                     "gpu_temp_c", "gpu_power_w"):
            assert col in col_names, f"Missing column: {col}"

    def test_gpu_round_trip(self, db, monkeypatch):
        """Insert telemetry with a gpu block → get_latest_telemetry returns it."""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from exonit.db import ExoNitDB

        db_path = str(db)  # fixture returns tmp_path
        d = ExoNitDB(db_path)

        dummy_telemetry = {
            "status": "working", "current_tic": 254113311,
            "phase": "bls", "phase_progress": 0.5,
            "star_elapsed_s": 30.0, "stars_done": 5,
            "avg_time_per_star": 6.0, "last_error": None,
            "recent_timings": [],
        }
        d.insert_telemetry("test-gpu-worker", dummy_telemetry)

        result = d.get_latest_telemetry("test-gpu-worker")
        assert result is not None
        # The Row-like object should have .get() or dict keys.
        row = dict(result) if hasattr(result, "keys") else result
        assert row.get("gpu_util_pct") is None
        assert row.get("gpu_mem_used_gb") is None

    def test_gpu_block_round_trip(self, db, monkeypatch):
        """Insert telemetry with real GPU values → they round-trip correctly."""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from exonit.db import ExoNitDB

        db_path = str(db)
        d = ExoNitDB(db_path)

        telemetry_with_gpu = {
            "status": "working", "current_tic": 254113311,
            "phase": "bls", "phase_progress": 0.8,
            "star_elapsed_s": 12.0, "stars_done": 100,
            "avg_time_per_star": 5.2, "last_error": None,
            "recent_timings": [],
            "system": {"cpu_percent": 45.0, "ram_used_gb": 8.0,
                        "ram_total_gb": 16.0},
            "gpu": {
                "util_pct": 73.5,
                "mem_used_gb": 4.2,
                "mem_total_gb": 16.0,
                "temp_c": 62.0,
                "power_w": 115.3,
            },
        }
        d.insert_telemetry("gpu-roundtrip-wid", telemetry_with_gpu)

        result = d.get_latest_telemetry("gpu-roundtrip-wid")
        assert result is not None
        row = dict(result) if hasattr(result, "keys") else result
        assert row.get("gpu_util_pct") == 73.5
        assert row.get("gpu_mem_used_gb") == pytest.approx(4.2)
        assert row.get("gpu_mem_total_gb") == 16.0
        assert row.get("gpu_temp_c") == 62.0
        assert row.get("gpu_power_w") == pytest.approx(115.3)


# ---------------------------------------------------------------------------
# 2. GET /api/status/fleet endpoint
# ---------------------------------------------------------------------------


class TestStatusFleetEndpoint:

    def test_returns_200_with_shape(self, temp_db):
        """Even with no workers, the endpoint returns 200 + valid shape."""
        client, tmp_path, _ = temp_db
        r = client.get("/api/status/fleet")
        assert r.status_code == 200
        data = r.json()
        assert "server" in data
        assert "computers" in data
        assert isinstance(data["computers"], list)
        assert data["server"]["name"] == "ExoNit server"

    def test_groups_multiple_instances_into_one(self, temp_db):
        """Two instances worker-x-1 + worker-x-2 → ONE entry worker-x."""
        client, tmp_path, _ = temp_db
        ws = client.app.state.workers_state
        now = _time.time()
        ws["worker-x-1"] = {
            "name": "worker-x-1", "last_heartbeat": now - 10,
            "status": "working", "tasks_completed": 42,
            "capabilities": {"bls_gpu_info": "RTX 5060 Ti", "bls_backend": "cuda",
                             "cpu_cores": 16},
        }
        ws["worker-x-2"] = {
            "name": "worker-x-2", "last_heartbeat": now - 15,
            "status": "idle", "tasks_completed": 7,
            "capabilities": {"bls_gpu_info": "RTX 5060 Ti", "bls_backend": "cuda",
                             "cpu_cores": 16},
        }

        r = client.get("/api/status/fleet")
        assert r.status_code == 200
        data = r.json()
        computer_ids = [c["id"] for c in data["computers"]]
        assert "worker-x" in computer_ids
        # Only one entry for both instances
        x_entry = [c for c in data["computers"] if c["id"] == "worker-x"][0]
        assert x_entry["instances_online"] == 2
        assert x_entry["tasks_completed"] == 49

    def test_contract_keys_present(self, temp_db):
        """Each computer object must have all contract keys."""
        client, tmp_path, _ = temp_db

        workers_state = client.app.state.workers_state
        now = _time.time()
        workers_state["worker-test-1"] = {
            "name": "worker-test-1", "last_heartbeat": now - 5,
            "status": "idle", "tasks_completed": 0,
            "capabilities": {"bls_gpu_info": "M4", "bls_backend": "metal",
                             "cpu_cores": 10},
        }

        r = client.get("/api/status/fleet")
        data = r.json()
        comp = data["computers"][0]

        required_keys = (
            "id", "friendly_name", "platform", "ip", "gpu_name",
            "gpu_backend", "cpu_cores", "instances_total", "instances_online",
            "online", "status", "last_heartbeat_ago",
            "cpu_percent", "ram_used_gb", "ram_total_gb",
            "gpu_util_pct", "gpu_mem_used_gb", "gpu_mem_total_gb",
            "gpu_temp_c", "gpu_power_w", "tasks_completed",
        )
        for key in required_keys:
            assert key in comp, f"Missing key: {key}"

    def test_offline_workers_excluded_from_online(self, temp_db):
        """Offline workers should not count towards instances_online."""
        client, tmp_path, _ = temp_db

        workers_state = client.app.state.workers_state
        now = _time.time()
        # Online
        workers_state["worker-y-1"] = {
            "name": "worker-y-1", "last_heartbeat": now - 20,
            "status": "idle", "tasks_completed": 0,
            "capabilities": {"bls_gpu_info": "none"},
        }
        # Offline (heartbeat > 600s ago)
        workers_state["worker-y-2"] = {
            "name": "worker-y-2", "last_heartbeat": now - 9999,
            "status": "offline", "tasks_completed": 0,
            "capabilities": {"bls_gpu_info": "none"},
        }

        r = client.get("/api/status/fleet")
        data = r.json()
        y_entry = [c for c in data["computers"] if c["id"] == "worker-y"][0]
        assert y_entry["instances_online"] == 1
        assert y_entry["online"] is True


class TestGpuIdentity:
    """gpu_name / gpu_backend must describe the machine's HARDWARE, not the
    BLS backend (owner report 2026-07-25).

    The capability payloads below are copied from the live fleet, because
    the bug only showed up on a shape nobody had written a test for: every
    Mac reports bls_backend "numba" and bls_gpu_info "CPU only (numba)"
    while running injection and FPP on Metal, so the Status page called six
    busy Apple GPUs "No GPU reported (numba)".
    """

    def _entry(self, client, name, caps_list):
        ws = client.app.state.workers_state
        now = _time.time()
        for i, caps in enumerate(caps_list, start=1):
            ws[f"{name}-{i}"] = {
                "name": f"{name}-{i}", "last_heartbeat": now - 5,
                "status": "working", "tasks_completed": 0,
                "capabilities": caps,
            }
        data = client.get("/api/status/fleet").json()
        return [c for c in data["computers"] if c["id"] == name][0]

    def test_apple_silicon_with_mlx_named_from_hardware_probe(self, temp_db):
        client, _tmp, _ = temp_db
        comp = self._entry(client, "worker-m4", [{
            "gpu": "metal", "bls_backend": "numba", "bls_gpu_info": "numba",
            "hardware": {"cuda": False, "opencl": False,
                         "metal": "Apple M4 (Metal/MLX)"},
            "fpp_backend": "metal-kernel", "cpu_cores": 10,
        }])
        assert comp["gpu_name"] == "Apple M4 (Metal/MLX)"
        assert comp["gpu_backend"] == "metal"

    def test_apple_silicon_without_mlx_still_not_called_cpu_only(self, temp_db):
        """The M1/M3 Macs report hardware.metal == False (no MLX installed)
        yet run fpp_backend 'metal-kernel'. Naming them from the platform
        is honest; "No GPU reported" was not."""
        client, _tmp, _ = temp_db
        comp = self._entry(client, "worker-<host>", [{
            "gpu": "cpu", "bls_backend": "numba",
            "bls_gpu_info": "CPU only (numba)",
            "hardware": {"cuda": False, "opencl": False, "metal": False},
            "fpp_backend": "metal-kernel", "cpu_cores": 8,
        }])
        assert comp["gpu_backend"] == "metal"
        assert "numba" not in comp["gpu_name"].lower()

    def test_specific_device_string_beats_generic_probe_label(self, temp_db):
        """bls_gpu_info carries "Intel(R) Arc(TM) Graphics"; the hardware
        probe only says "AMD/Intel OpenCL iGPU" — the specific one wins."""
        client, _tmp, _ = temp_db
        comp = self._entry(client, "worker-arc", [{
            "gpu": "opencl", "bls_backend": "numba",
            "bls_gpu_info": "Intel(R) Arc(TM) Graphics",
            "hardware": {"cuda": False, "opencl": "AMD/Intel OpenCL iGPU",
                         "metal": False},
            "fpp_backend": "opencl", "cpu_cores": 16,
        }])
        assert comp["gpu_name"] == "Intel(R) Arc(TM) Graphics"
        assert comp["gpu_backend"] == "opencl"

    def test_mixed_host_reports_its_accelerator_whichever_instance_answers(self, temp_db):
        """<host> deliberately runs 3 CPU instances + 1 Arc instance. Which
        instance happens to be the representative one must not change what
        the machine is reported to be."""
        client, _tmp, _ = temp_db
        cpu_inst = {"gpu": "cpu", "bls_backend": "numba",
                    "bls_gpu_info": "CPU only (numba)",
                    "hardware": {"cuda": False, "opencl": False, "metal": False},
                    "fpp_backend": "numba", "cpu_cores": 16}
        gpu_inst = {"gpu": "opencl", "bls_backend": "numba",
                    "bls_gpu_info": "Intel(R) Arc(TM) Graphics",
                    "hardware": {"cuda": False, "opencl": "AMD/Intel OpenCL iGPU",
                                 "metal": False},
                    "fpp_backend": "opencl", "cpu_cores": 16}
        comp = self._entry(client, "worker-mixed", [cpu_inst, cpu_inst, gpu_inst])
        assert comp["gpu_name"] == "Intel(R) Arc(TM) Graphics"
        assert comp["gpu_backend"] == "opencl"

    def test_genuinely_cpu_only_machine_still_reads_as_one(self, temp_db):
        """The honest negative case must survive: no accelerator anywhere
        means the page should still say so, not invent a GPU."""
        client, _tmp, _ = temp_db
        comp = self._entry(client, "worker-plain", [{
            "gpu": "cpu", "bls_backend": "numba",
            "bls_gpu_info": "CPU only (numba)",
            "hardware": {"cuda": False, "opencl": False, "metal": False},
            "fpp_backend": "numba", "cpu_cores": 4,
        }])
        assert comp["gpu_backend"] == "numba"
        assert comp["gpu_name"] in ("—", "CPU only (numba)")


# ---------------------------------------------------------------------------
# 3. GET /status HTML route
# ---------------------------------------------------------------------------


class TestStatusPage:

    def test_returns_200(self, temp_db):
        client, tmp_path, _ = temp_db
        r = client.get("/status")
        assert r.status_code == 200

    def test_contains_fleet_map_elements(self, temp_db):
        client, tmp_path, _ = temp_db
        r = client.get("/status")
        body = r.text
        assert "Fleet Map" in body or "fleetSvg" in body
        assert "/api/status/fleet" in body


# ---------------------------------------------------------------------------
# 4. Worker GPU collector
# ---------------------------------------------------------------------------


class TestWorkerGPUCollector:

    def test_returns_all_none_when_no_hardware(self):
        """When _bls_hardware_info returns {}, _get_gpu_telemetry is all-None."""
        from worker.worker import ExoNitWorker

        # Create a minimal instance — we only need the _get_gpu_telemetry method.
        # It doesn't depend on __init__ state beyond self.debug_mode for the
        # heartbeat payload, not the GPU method itself.
        class MinimalWorker:
            _bls_hardware_info = None

            def __init__(self):
                # Import the actual methods from ExoNitWorker
                for name in ("_get_gpu_telemetry", "_gpu_telem_nvidia",
                              "_gpu_telem_macmon", "_gpu_telem_amd_sysfs"):
                    setattr(self, name,
                            getattr(ExoNitWorker, name).__get__(self))
                # Staticmethod -- assign directly, no __get__(self) binding.
                self._round_gpu_telemetry = ExoNitWorker._round_gpu_telemetry

        w = MinimalWorker()

        with patch("worker.worker._bls_hardware_info", return_value={}):
            result = w._get_gpu_telemetry()

        assert result == {
            "util_pct": None, "mem_used_gb": None, "mem_total_gb": None,
            "temp_c": None, "power_w": None,
        }

    def test_returns_all_none_on_subprocess_failure(self):
        """When nvidia-smi fails (no binary), degrades to None → all-None."""
        from worker.worker import ExoNitWorker

        class MinimalWorker:
            def __init__(self):
                for name in ("_get_gpu_telemetry", "_gpu_telem_nvidia",
                              "_gpu_telem_macmon", "_gpu_telem_amd_sysfs"):
                    setattr(self, name,
                            getattr(ExoNitWorker, name).__get__(self))
                # Staticmethod -- assign directly, no __get__(self) binding.
                self._round_gpu_telemetry = ExoNitWorker._round_gpu_telemetry

        w = MinimalWorker()

        with patch("worker.worker._bls_hardware_info", return_value={"cuda": "fake"}):
            with patch("worker.worker.shutil.which", return_value=None):
                result = w._get_gpu_telemetry()

        # Should be all-None (not crash)
        assert result["util_pct"] is None

    def test_nvidia_smi_parse(self):
        """Canned nvidia-smi output is correctly parsed into five numbers."""
        from worker.worker import ExoNitWorker

        class MinimalWorker:
            def __init__(self):
                for name in ("_get_gpu_telemetry", "_gpu_telem_nvidia",
                              "_gpu_telem_macmon", "_gpu_telem_amd_sysfs"):
                    setattr(self, name,
                            getattr(ExoNitWorker, name).__get__(self))
                # Staticmethod -- assign directly, no __get__(self) binding.
                self._round_gpu_telemetry = ExoNitWorker._round_gpu_telemetry

        w = MinimalWorker()

        canned_csv = "72.5, 61, 8192, 16384, 120.5\n"
        mock_result = MagicMock()
        mock_result.stdout = canned_csv

        with patch("worker.worker._bls_hardware_info", return_value={"cuda": "RTX 5060 Ti"}):
            with patch("worker.worker.shutil.which", return_value="/usr/bin/nvidia-smi"):
                with patch("worker.worker.subprocess.run", return_value=mock_result):
                    result = w._gpu_telem_nvidia()

        assert result is not None
        assert result["util_pct"] == 72.5
        assert result["temp_c"] == 61.0
        assert pytest.approx(result["mem_used_gb"]) == 8192 / 1024  # 8.0 GiB
        assert pytest.approx(result["mem_total_gb"]) == 16384 / 1024  # 16.0 GiB
        assert result["power_w"] == 120.5

    def test_nvidia_smi_n_a_values_become_none(self):
        """N/A / [N/A] fields from nvidia-smi become None."""
        from worker.worker import ExoNitWorker

        class MinimalWorker:
            def __init__(self):
                for name in ("_get_gpu_telemetry", "_gpu_telem_nvidia",
                              "_gpu_telem_macmon", "_gpu_telem_amd_sysfs"):
                    setattr(self, name,
                            getattr(ExoNitWorker, name).__get__(self))
                # Staticmethod -- assign directly, no __get__(self) binding.
                self._round_gpu_telemetry = ExoNitWorker._round_gpu_telemetry

        w = MinimalWorker()

        canned_csv = "45.0, N/A, 4096, 8192, [N/A]\n"
        mock_result = MagicMock()
        mock_result.stdout = canned_csv

        with patch("worker.worker._bls_hardware_info", return_value={"cuda": "Tesla T4"}):
            with patch("worker.worker.shutil.which", return_value="/usr/bin/nvidia-smi"):
                with patch("worker.worker.subprocess.run", return_value=mock_result):
                    result = w._gpu_telem_nvidia()

        assert result is not None
        assert result["util_pct"] == 45.0
        assert result["temp_c"] is None
        assert result["power_w"] is None

    def test_macmon_idle_gpu_temp_and_power(self):
        """Real idle-M4 macmon JSON: bogus ~0.8C gpu_temp_avg -> None;
        gpu_power (not sys_power) is used, and 0.0 stays 0.0 (valid idle GPU power)."""
        from worker.worker import ExoNitWorker

        class MinimalWorker:
            def __init__(self):
                for name in ("_get_gpu_telemetry", "_gpu_telem_nvidia",
                              "_gpu_telem_macmon", "_gpu_telem_amd_sysfs",
                              "_find_macmon"):
                    setattr(self, name,
                            getattr(ExoNitWorker, name).__get__(self))
                # Staticmethod -- assign directly, no __get__(self) binding.
                self._round_gpu_telemetry = ExoNitWorker._round_gpu_telemetry

        w = MinimalWorker()

        canned_json = json.dumps({
            "all_power": 6.11, "gpu_power": 0.0, "gpu_usage": [0, 0.0],
            "memory": {"ram_total": 17179869184, "ram_usage": 4626563072},
            "sys_power": 18.0,
            "temp": {"cpu_temp_avg": 61.1, "gpu_temp_avg": 0.8000000715255737},
        })
        mock_result = MagicMock()
        mock_result.stdout = canned_json + "\n"

        with patch("worker.worker.shutil.which", return_value="/opt/homebrew/bin/macmon"):
            with patch("worker.worker.subprocess.run", return_value=mock_result):
                result = w._gpu_telem_macmon()

        assert result is not None
        assert result["temp_c"] is None  # bogus 0.8C idle reading discarded
        assert result["power_w"] == 0.0  # valid idle GPU power, not None

    def test_get_gpu_telemetry_rounds_to_2_decimals(self):
        """_get_gpu_telemetry() must round every numeric value to 2 decimals,
        even though the raw macmon/nvidia-smi backends return long floats
        (e.g. 0.8000000715255737 -> the UI showed that verbatim, 2026-07-10)."""
        from worker.worker import ExoNitWorker

        class MinimalWorker:
            def __init__(self):
                for name in ("_get_gpu_telemetry", "_gpu_telem_nvidia",
                              "_gpu_telem_amd_sysfs"):
                    setattr(self, name,
                            getattr(ExoNitWorker, name).__get__(self))
                # Staticmethod -- assign directly, no __get__(self) binding.
                self._round_gpu_telemetry = ExoNitWorker._round_gpu_telemetry
                self._gpu_telem_macmon = MagicMock(return_value={
                    "util_pct": 22.765304565429688,
                    "mem_used_gb": 4.311111111,
                    "mem_total_gb": 16.0,
                    "temp_c": None,
                    "power_w": 22.765304565429688,
                })

        w = MinimalWorker()

        with patch("worker.worker._bls_hardware_info", return_value={"metal": True}):
            result = w._get_gpu_telemetry()

        assert result == {
            "util_pct": 22.77,
            "mem_used_gb": 4.31,
            "mem_total_gb": 16.0,
            "temp_c": None,
            "power_w": 22.77,
        }

    def test_round_gpu_telemetry_helper_direct(self):
        """_round_gpu_telemetry() rounds numeric fields, passes None through."""
        from worker.worker import ExoNitWorker

        raw = {"util_pct": None, "mem_used_gb": 0.0, "mem_total_gb": 16.0,
               "temp_c": 0.8000000715255737, "power_w": 115.34999999}
        rounded = ExoNitWorker._round_gpu_telemetry(raw)
        assert rounded == {"util_pct": None, "mem_used_gb": 0.0,
                            "mem_total_gb": 16.0, "temp_c": 0.8,
                            "power_w": 115.35}


# ---------------------------------------------------------------------------
# 5. Auto-update multi-instance race (Task 2 of ui-status-fleet-map)
# ---------------------------------------------------------------------------


class TestAutoUpdateMultiInstanceRace:
    """Guards against the bug confirmed 2026-07-10 on worker-<host>:
    instances 1-2 restarted on a new manifest, instances 3-4 never did
    because by the time they checked, disk already matched the remote
    hash (another instance had already downloaded it) -- so `to_update`
    was empty and they silently kept running stale in-memory code
    forever. Fix: compare remote_hash against self._loaded_hashes
    (snapshotted at process start), not against on-disk state."""

    def _make_worker(self, worker_dir, loaded_hashes, update_restart_count=0):
        from worker.worker import ExoNitWorker

        class MinimalWorker:
            def __init__(self):
                self.server = "http://fake-server"
                self.token = "fake-token"
                self.worker_id = "worker-test-1"
                self.logger = MagicMock()
                self._loaded_hashes = loaded_hashes
                self._update_restart_count = update_restart_count
                for name in ("check_for_updates", "_headers",
                              "_snapshot_worker_dir_hashes"):
                    setattr(self, name, getattr(ExoNitWorker, name).__get__(self))
                # Staticmethod -- don't rebind through __get__(self), that
                # would smuggle self in as the first positional arg.
                self._file_sha256_normalized = ExoNitWorker._file_sha256_normalized

        return MinimalWorker()

    def test_disk_current_but_loaded_code_stale_triggers_restart(self, tmp_path, monkeypatch):
        """Disk already matches remote (another instance updated it), but
        THIS process's _loaded_hashes (snapshotted at its own start) is
        still the old hash -> must os.execv restart anyway."""
        worker_file = tmp_path / "worker.py"
        new_content = b"# new code v2\n"
        worker_file.write_bytes(new_content)
        import hashlib
        new_hash = hashlib.sha256(new_content).hexdigest()
        old_hash = hashlib.sha256(b"# old code v1\n").hexdigest()

        w = self._make_worker(str(tmp_path), {"worker.py": old_hash})

        manifest_resp = MagicMock()
        manifest_resp.json.return_value = {
            "files": [{"filename": "worker.py", "sha256": new_hash}]
        }

        with patch("worker.worker.os.path.abspath", return_value=str(worker_file)):
            with patch("worker.worker.requests.get", return_value=manifest_resp):
                with patch("worker.worker.os.execv") as mock_execv:
                    w.check_for_updates()

        mock_execv.assert_called_once()

    def test_disk_and_loaded_code_both_current_no_restart(self, tmp_path):
        """Normal steady state: disk matches remote AND matches what this
        process loaded at start -> no restart."""
        worker_file = tmp_path / "worker.py"
        content = b"# current code\n"
        worker_file.write_bytes(content)
        import hashlib
        h = hashlib.sha256(content).hexdigest()

        w = self._make_worker(str(tmp_path), {"worker.py": h})

        manifest_resp = MagicMock()
        manifest_resp.json.return_value = {
            "files": [{"filename": "worker.py", "sha256": h}]
        }

        with patch("worker.worker.os.path.abspath", return_value=str(worker_file)):
            with patch("worker.worker.requests.get", return_value=manifest_resp):
                with patch("worker.worker.os.execv") as mock_execv:
                    result = w.check_for_updates()

        mock_execv.assert_not_called()
        assert result is False

    def test_snapshot_includes_models_subdir_with_slash_key(self, tmp_path):
        """_snapshot_worker_dir_hashes must key worker/models/<name> entries
        as 'models/<name>' -- matching the server manifest's _syncable()
        layout (exonit/routes/workers.py). Before the 2026-07-10 fix, only
        top-level files were hashed, so self._loaded_hashes.get('models/x')
        was always None != remote_hash -> loaded_code_stale=True on every
        poll -> infinite execv restart loop (observed on worker-<host>)."""
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        (tmp_path / "worker.py").write_bytes(b"# code\n")
        (models_dir / "fake.onnx").write_bytes(b"fake-model-bytes")

        w = self._make_worker(str(tmp_path), {})

        with patch("worker.worker.os.path.dirname", return_value=str(tmp_path)):
            with patch("worker.worker.os.path.abspath", return_value=str(tmp_path)):
                hashes = w._snapshot_worker_dir_hashes()

        assert "worker.py" in hashes
        assert "models/fake.onnx" in hashes

    def test_stale_restart_loop_guard_skips_execv_past_threshold(self, tmp_path):
        """If loaded_code_stale keeps being true (e.g. a manifest/snapshot
        key mismatch) with nothing left to download, the process must NOT
        execv-restart forever. Once this process lineage has already
        restarted twice for this exact reason (EXONIT_UPDATE_RESTART_COUNT
        >= 2), check_for_updates() must log a warning and skip the restart."""
        worker_file = tmp_path / "worker.py"
        content = b"# current code\n"
        worker_file.write_bytes(content)
        import hashlib
        h = hashlib.sha256(content).hexdigest()

        # loaded_hashes deliberately missing the key present in the manifest
        # (simulating the models/ mismatch) -> loaded_code_stale=True even
        # though disk is current (to_update stays empty).
        w = self._make_worker(str(tmp_path), {}, update_restart_count=2)

        manifest_resp = MagicMock()
        manifest_resp.json.return_value = {
            "files": [{"filename": "worker.py", "sha256": h}]
        }

        with patch("worker.worker.os.path.abspath", return_value=str(worker_file)):
            with patch("worker.worker.requests.get", return_value=manifest_resp):
                with patch("worker.worker.os.execv") as mock_execv:
                    result = w.check_for_updates()

        mock_execv.assert_not_called()
        assert result is False
        w.logger.warning.assert_called()


# ---------------------------------------------------------------------------
# 5. The server box (worker-default) is a machine like any other
# ---------------------------------------------------------------------------


class TestServerBoxIsReported:
    """worker-default must report platform and ip, not two nulls.

    Regression guard (2026-07-25): status_fleet skipped "worker-default" when
    building the static lookup from ops/workers.json, while the matching skip
    in the grouping loop never fired — instances register as
    "worker-default-1".."-N". Net effect: the machine appeared in the fleet
    table with platform=null and ip=null, the only one missing both, and the
    dashboard rendered "—" for its OS.
    """

    def _manifest(self, tmp_path, monkeypatch):
        manifest = tmp_path / "workers.json"
        manifest.write_text(json.dumps({"workers": [
            {"name": "worker-default", "ssh_host": None, "platform": "linux",
             "instances": 4},
            {"name": "worker-remote", "ssh_host": "user@10.0.0.9",
             "platform": "macos", "instances": 2},
        ]}))
        import exonit.routes.workers as workers_routes
        monkeypatch.setattr(workers_routes, "_WORKERS_MANIFEST", manifest)
        monkeypatch.setattr(workers_routes, "_SERVER_IP", "10.0.0.1")
        return manifest

    def test_worker_default_reports_platform_and_server_ip(self, temp_db,
                                                           monkeypatch):
        client, tmp_path, _ = temp_db
        self._manifest(tmp_path, monkeypatch)
        now = _time.time()
        ws = client.app.state.workers_state
        for i in (1, 2):
            ws[f"worker-default-{i}"] = {
                "name": f"worker-default-{i}", "last_heartbeat": now - 5,
                "status": "idle", "tasks_completed": 3,
                "capabilities": {"bls_gpu_info": "numba", "bls_backend": "numba",
                                 "cpu_cores": 16},
            }

        data = client.get("/api/status/fleet").json()
        entry = [c for c in data["computers"] if c["id"] == "worker-default"]
        assert entry, "the server box must appear in the fleet listing"
        entry = entry[0]
        assert entry["platform"] == "linux"
        assert entry["ip"] == "10.0.0.1", (
            "a worker with no ssh_host runs on the server box — it should "
            "report the server's own address, not null")
        assert entry["instances_total"] == 4

    def test_remote_worker_ip_still_comes_from_ssh_host(self, temp_db,
                                                       monkeypatch):
        client, tmp_path, _ = temp_db
        self._manifest(tmp_path, monkeypatch)
        now = _time.time()
        client.app.state.workers_state["worker-remote-1"] = {
            "name": "worker-remote-1", "last_heartbeat": now - 5,
            "status": "idle", "tasks_completed": 1,
            "capabilities": {"bls_gpu_info": "M4", "bls_backend": "metal",
                             "cpu_cores": 10},
        }
        data = client.get("/api/status/fleet").json()
        entry = [c for c in data["computers"] if c["id"] == "worker-remote"][0]
        assert entry["ip"] == "10.0.0.9"
        assert entry["platform"] == "macos"


# ---------------------------------------------------------------------------
# 6. Machine-family merging (2026-08-16 owner report: 25 cards for ~11
#    physical machines because worker-<host>-rust, worker-<host>-rust-py and
#    the retired worker-<host> each opened their own card).
# ---------------------------------------------------------------------------


class TestMachineFamilyMerging:

    def _manifest(self, tmp_path, monkeypatch, workers):
        manifest = tmp_path / "workers.json"
        manifest.write_text(json.dumps({"workers": workers}))
        import exonit.routes.workers as workers_routes
        monkeypatch.setattr(workers_routes, "_WORKERS_MANIFEST", manifest)
        monkeypatch.setattr(workers_routes, "_SERVER_IP", "10.0.0.1")
        return manifest

    def test_rust_rust_py_swift_and_base_families_merge_into_one_card(
            self, temp_db, monkeypatch):
        """A single physical box registering under four different family
        names (the retired plain-Python family, the native Rust port, its
        Python FPP sidecar, and — hypothetically, to prove the suffix
        stripping is general — a Swift port) must produce ONE computer
        entry, not four."""
        client, tmp_path, _ = temp_db
        self._manifest(tmp_path, monkeypatch, [
            {"name": "worker-<host>", "ssh_host": "user@192.0.2.51",
             "platform": "windows", "instances": 8, "instances_fpp": 10},
        ])
        now = _time.time()
        ws = client.app.state.workers_state
        for suffix, tasks in (
            ("worker-<host>-1", 3),          # retired plain-Python family
            ("worker-<host>-rust-1", 5),     # native Rust port
            ("worker-<host>-rust-py-1", 7),  # Rust port's Python FPP sidecar
            ("worker-<host>-swift-1", 2),    # hypothetical Swift port
        ):
            ws[suffix] = {
                "name": suffix, "last_heartbeat": now - 5,
                "status": "idle", "tasks_completed": tasks,
                "capabilities": {"bls_gpu_info": "cuda", "bls_backend": "cuda",
                                 "cpu_cores": 16},
            }

        data = client.get("/api/status/fleet").json()
        computer_ids = [c["id"] for c in data["computers"]]
        assert computer_ids.count("worker-<host>") == 1
        assert "worker-<host>-rust" not in computer_ids
        assert "worker-<host>-rust-py" not in computer_ids
        assert "worker-<host>-swift" not in computer_ids

        entry = [c for c in data["computers"] if c["id"] == "worker-<host>"][0]
        assert entry["instances_online"] == 4
        assert entry["tasks_completed"] == 3 + 5 + 7 + 2

    def test_instances_total_is_max_of_manifest_instances_and_instances_fpp(
            self, temp_db, monkeypatch):
        """instances_total must come from the manifest's CONFIGURED capacity,
        as the MAX of `instances` (scan-phase pool) and `instances_fpp`
        (FPP-phase pool) — the two never run concurrently (farmctl prunes
        down to whichever the current phase calls for), so summing them
        would overcount: <host> is a ceiling of 10 (its FPP pool), not 18."""
        client, tmp_path, _ = temp_db
        self._manifest(tmp_path, monkeypatch, [
            {"name": "worker-<host>", "ssh_host": "user@192.0.2.51",
             "platform": "windows", "instances": 8, "instances_fpp": 10},
        ])
        now = _time.time()
        ws = client.app.state.workers_state
        ws["worker-<host>-rust-1"] = {
            "name": "worker-<host>-rust-1", "last_heartbeat": now - 5,
            "status": "idle", "tasks_completed": 0,
            "capabilities": {"bls_gpu_info": "cuda", "bls_backend": "cuda"},
        }

        data = client.get("/api/status/fleet").json()
        entry = [c for c in data["computers"] if c["id"] == "worker-<host>"][0]
        assert entry["instances_total"] == 10

    def test_instances_total_takes_max_across_two_manifest_entries_per_field(
            self, temp_db, monkeypatch):
        """A machine with TWO manifest entries after key-stripping (e.g. a
        legacy Python sizing row plus a dedicated Swift sizing row) sums each
        field separately across entries, THEN takes the max of the two sums
        — matching the M4 worked example: python entry instances=0/
        instances_fpp=9, swift entry instances=5/instances_fpp=0 ->
        max(0+5, 9+0) = max(5, 9) = 9."""
        client, tmp_path, _ = temp_db
        self._manifest(tmp_path, monkeypatch, [
            {"name": "worker-<host>", "ssh_host": "user@192.0.2.60",
             "platform": "macos", "instances": 0, "instances_fpp": 9},
            {"name": "worker-<host>-swift", "ssh_host": "user@192.0.2.60",
             "platform": "macos", "instances": 5, "instances_fpp": 0},
        ])
        now = _time.time()
        ws = client.app.state.workers_state
        ws["worker-<host>-swift-1"] = {
            "name": "worker-<host>-swift-1", "last_heartbeat": now - 5,
            "status": "idle", "tasks_completed": 0,
            "capabilities": {"bls_gpu_info": "metal", "bls_backend": "metal"},
        }

        data = client.get("/api/status/fleet").json()
        entry = [c for c in data["computers"] if c["id"] == "worker-<host>"][0]
        assert entry["instances_total"] == 9

    def test_machine_absent_from_manifest_and_stale_over_7_days_is_omitted(
            self, temp_db, monkeypatch):
        """A retired family/host (no manifest entry, no heartbeat in over a
        week) must fall off the fleet map entirely."""
        client, tmp_path, _ = temp_db
        self._manifest(tmp_path, monkeypatch, [
            {"name": "worker-<host>", "ssh_host": "user@192.0.2.51",
             "platform": "windows", "instances": 8, "instances_fpp": 10},
        ])
        now = _time.time()
        ws = client.app.state.workers_state
        ws["worker-HP-1"] = {
            "name": "worker-HP-1", "last_heartbeat": now - 9 * 24 * 3600,
            "status": "offline", "tasks_completed": 100,
            "capabilities": {"bls_gpu_info": "none"},
        }

        data = client.get("/api/status/fleet").json()
        computer_ids = [c["id"] for c in data["computers"]]
        assert "worker-HP" not in computer_ids

    def test_machine_in_manifest_but_fully_offline_is_kept(
            self, temp_db, monkeypatch):
        """A machine still in the manifest (configured fleet) must NOT be
        dropped just because every instance is currently offline — e.g.
        <host>, paused for days but still part of the fleet."""
        client, tmp_path, _ = temp_db
        self._manifest(tmp_path, monkeypatch, [
            {"name": "worker-<host>", "ssh_host": "user@192.0.2.36",
             "platform": "macos", "instances": 3},
        ])
        now = _time.time()
        ws = client.app.state.workers_state
        ws["worker-<host>-1"] = {
            "name": "worker-<host>-1", "last_heartbeat": now - 13 * 24 * 3600,
            "status": "offline", "tasks_completed": 50,
            "capabilities": {"bls_gpu_info": "none"},
        }

        data = client.get("/api/status/fleet").json()
        entries = [c for c in data["computers"] if c["id"] == "worker-<host>"]
        assert entries, "manifest machines must be kept even fully offline"
        assert entries[0]["online"] is False
        assert entries[0]["instances_online"] == 0

    def test_telemetry_falls_back_to_a_family_that_actually_reports_it(
            self, temp_db, monkeypatch):
        """The Rust family answers first (more recent heartbeat) but never
        sends system/GPU telemetry; the Python FPP sidecar family does. The
        merged card must show the sidecar's numbers, not blanks."""
        client, tmp_path, dbfile = temp_db
        self._manifest(tmp_path, monkeypatch, [
            {"name": "worker-<host>", "ssh_host": "user@192.0.2.51",
             "platform": "windows", "instances": 8, "instances_fpp": 10},
        ])
        now = _time.time()
        ws = client.app.state.workers_state
        ws["worker-<host>-rust-1"] = {
            "name": "worker-<host>-rust-1", "last_heartbeat": now - 2,
            "status": "idle", "tasks_completed": 0,
            "capabilities": {"bls_gpu_info": "cuda", "bls_backend": "cuda"},
        }
        ws["worker-<host>-rust-py-1"] = {
            "name": "worker-<host>-rust-py-1", "last_heartbeat": now - 5,
            "status": "idle", "tasks_completed": 0,
            "capabilities": {"bls_gpu_info": "NVIDIA GeForce RTX 4070 (12.0 GB)",
                             "bls_backend": "cuda"},
        }

        from exonit.db import ExoNitDB
        d = ExoNitDB(dbfile)
        # No telemetry row for worker-<host>-rust-1 at all (the Rust worker
        # never posts /api/telemetry). The sidecar does.
        d.insert_telemetry("worker-<host>-rust-py-1", {
            "status": "idle", "current_tic": None, "phase": None,
            "phase_progress": None, "star_elapsed_s": 0.0, "stars_done": 0,
            "avg_time_per_star": 0.0, "last_error": None, "recent_timings": [],
            "system": {"cpu_percent": 12.5, "ram_used_gb": 4.0,
                        "ram_total_gb": 32.0},
            "gpu": {"util_pct": 33.0, "mem_used_gb": 2.0, "mem_total_gb": 12.0,
                    "temp_c": 55.0, "power_w": 80.0},
        })

        data = client.get("/api/status/fleet").json()
        entry = [c for c in data["computers"] if c["id"] == "worker-<host>"][0]
        assert entry["cpu_percent"] == 12.5
        assert entry["ram_total_gb"] == 32.0
        assert entry["gpu_util_pct"] == 33.0

    def test_macmini_m4_does_not_merge_with_macmini_m1(self, temp_db,
                                                        monkeypatch):
        """worker-<host> (the M4 mac mini) and worker-<host> (the M1
        mac mini) are two DIFFERENT physical machines; only the listed
        family suffixes (-rust, -rust-py, -swift) may be stripped, never an
        arbitrary trailing segment like '-m4'."""
        client, tmp_path, _ = temp_db
        self._manifest(tmp_path, monkeypatch, [
            {"name": "worker-<host>", "ssh_host": "user@192.0.2.60",
             "platform": "macos", "instances": 4, "instances_fpp": 9},
            {"name": "worker-<host>", "ssh_host": "user@192.0.2.106",
             "platform": "macos", "instances": 2},
        ])
        now = _time.time()
        ws = client.app.state.workers_state
        ws["worker-<host>-1"] = {
            "name": "worker-<host>-1", "last_heartbeat": now - 5,
            "status": "idle", "tasks_completed": 1,
            "capabilities": {"bls_gpu_info": "numba", "bls_backend": "numba"},
        }
        ws["worker-<host>-1"] = {
            "name": "worker-<host>-1", "last_heartbeat": now - 5,
            "status": "idle", "tasks_completed": 1,
            "capabilities": {"bls_gpu_info": "numba", "bls_backend": "numba"},
        }

        data = client.get("/api/status/fleet").json()
        computer_ids = {c["id"] for c in data["computers"]}
        assert "worker-<host>" in computer_ids
        assert "worker-<host>" in computer_ids
        # Two separate cards, each with exactly its own single instance.
        m4_entry = [c for c in data["computers"] if c["id"] == "worker-<host>"][0]
        m1_entry = [c for c in data["computers"] if c["id"] == "worker-<host>"][0]
        assert m4_entry["instances_online"] == 1
        assert m1_entry["instances_online"] == 1

    def test_real_gpu_name_beats_backend_token_after_merge(self, temp_db,
                                                            monkeypatch):
        """Once families are merged, the backend TOKEN some workers report
        as their "gpu" ('cuda', 'opencl', 'sycl', 'metal', 'ispc') must never
        win over a real device string reported by a sibling family."""
        client, tmp_path, _ = temp_db
        self._manifest(tmp_path, monkeypatch, [
            {"name": "worker-<host>", "ssh_host": "user@192.0.2.51",
             "platform": "windows", "instances": 8, "instances_fpp": 10},
        ])
        now = _time.time()
        ws = client.app.state.workers_state
        ws["worker-<host>-rust-1"] = {
            "name": "worker-<host>-rust-1", "last_heartbeat": now - 5,
            "status": "idle", "tasks_completed": 0,
            "capabilities": {"bls_gpu_info": "cuda", "bls_backend": "cuda",
                             "cpu_cores": 32},
        }
        ws["worker-<host>-rust-py-1"] = {
            "name": "worker-<host>-rust-py-1", "last_heartbeat": now - 5,
            "status": "idle", "tasks_completed": 0,
            "capabilities": {"bls_gpu_info": "NVIDIA GeForce RTX 4070 (12.0 GB)",
                             "bls_backend": "cuda", "cpu_cores": 16},
        }

        data = client.get("/api/status/fleet").json()
        entry = [c for c in data["computers"] if c["id"] == "worker-<host>"][0]
        assert entry["gpu_name"] == "NVIDIA GeForce RTX 4070 (12.0 GB)"
        assert entry["gpu_backend"] == "cuda"
        # cpu_cores: max across the merged families, not just whichever
        # instance happened to be the representative.
        assert entry["cpu_cores"] == 32
