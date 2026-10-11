"""Tests for the AWS concurrent worker bootstrap code-sync fix.

Covers:
1. Bootstrap script text assertions (no network, no systemd, no real execution):
   - early chown -R ubuntu:ubuntu /home/ubuntu/exonit
   - exonit-codesync.service oneshot unit with Type=oneshot, RemainAfterExit=yes, User=ubuntu
   - per-worker units include After=exonit-codesync.service
   - per-worker units include Requires=exonit-codesync.service
   - per-worker units set Environment=EXONIT_SKIP_STARTUP_CODESYNC=1
   - the oneshot unit does NOT set EXONIT_SKIP_STARTUP_CODESYNC (it must actually sync)

2. Worker.run() startup-codesync guard:
   - EXONIT_SKIP_STARTUP_CODESYNC=1 → check_for_updates is NOT called
   - EXONIT_SKIP_STARTUP_CODESYNC unset → check_for_updates IS called (default behaviour)
   - startup_code_sync=False in config → check_for_updates NOT called regardless of env var
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest

# ---------------------------------------------------------------------------
# Locate bootstrap script
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parent.parent
_BOOTSTRAP = _REPO_ROOT / "exonit_aws" / "cloud-init" / "worker-bootstrap-concurrent.sh"


# ---------------------------------------------------------------------------
# 1. Bootstrap script assertions
# ---------------------------------------------------------------------------

class TestBootstrapScript:
    """Read the bootstrap script as text and assert structural properties."""

    @pytest.fixture(scope="class")
    def bootstrap_text(self):
        assert _BOOTSTRAP.exists(), f"Bootstrap script not found: {_BOOTSTRAP}"
        return _BOOTSTRAP.read_text()

    def test_early_chown_ubuntu(self, bootstrap_text):
        """Bootstrap must chown the entire exonit source tree to ubuntu early."""
        assert "chown -R ubuntu:ubuntu /home/ubuntu/exonit" in bootstrap_text, (
            "Missing early 'chown -R ubuntu:ubuntu /home/ubuntu/exonit' in bootstrap"
        )

    def test_chown_before_worker_units(self, bootstrap_text):
        """The chown must appear before any per-worker unit is written."""
        chown_pos = bootstrap_text.find("chown -R ubuntu:ubuntu /home/ubuntu/exonit")
        worker_unit_pos = bootstrap_text.find("exonit-worker-w${i}.service")
        assert chown_pos != -1, "chown line not found"
        assert worker_unit_pos != -1, "worker unit template not found"
        assert chown_pos < worker_unit_pos, (
            "chown must appear before the per-worker unit template"
        )

    def test_codesync_unit_present(self, bootstrap_text):
        """Bootstrap must create an exonit-codesync.service unit file."""
        assert "exonit-codesync.service" in bootstrap_text, (
            "Bootstrap does not reference exonit-codesync.service"
        )

    def test_codesync_type_oneshot(self, bootstrap_text):
        """The codesync unit must be Type=oneshot."""
        assert "Type=oneshot" in bootstrap_text, (
            "exonit-codesync.service must declare Type=oneshot"
        )

    def test_codesync_remain_after_exit(self, bootstrap_text):
        """The codesync unit must set RemainAfterExit=yes so systemd knows it succeeded."""
        assert "RemainAfterExit=yes" in bootstrap_text, (
            "exonit-codesync.service must declare RemainAfterExit=yes"
        )

    def test_codesync_user_ubuntu(self, bootstrap_text):
        """The codesync unit must run as ubuntu (not root) so it can write files."""
        # We look for User=ubuntu inside the codesync unit definition block.
        # The block is delimited by the COSEOF heredoc markers.
        coseof_start = bootstrap_text.find("<<COSEOF")
        coseof_end = bootstrap_text.find("COSEOF", coseof_start + 8)
        assert coseof_start != -1 and coseof_end != -1, "COSEOF heredoc not found"
        unit_block = bootstrap_text[coseof_start:coseof_end]
        assert "User=ubuntu" in unit_block, (
            "exonit-codesync.service must declare User=ubuntu inside the unit block"
        )

    def test_worker_units_after_codesync(self, bootstrap_text):
        """Per-worker units must declare After=exonit-codesync.service."""
        assert "After=network-online.target exonit-codesync.service" in bootstrap_text, (
            "Worker units must include 'After=...exonit-codesync.service' in the [Unit] section"
        )

    def test_worker_units_require_codesync(self, bootstrap_text):
        """Per-worker units must declare Requires=exonit-codesync.service."""
        assert "Requires=exonit-codesync.service" in bootstrap_text, (
            "Worker units must include 'Requires=exonit-codesync.service'"
        )

    def test_worker_units_skip_env(self, bootstrap_text):
        """Per-worker units must set EXONIT_SKIP_STARTUP_CODESYNC=1."""
        assert "Environment=EXONIT_SKIP_STARTUP_CODESYNC=1" in bootstrap_text, (
            "Worker units must set Environment=EXONIT_SKIP_STARTUP_CODESYNC=1"
        )

    def test_codesync_unit_does_not_skip_itself(self, bootstrap_text):
        """The codesync oneshot unit must NOT set EXONIT_SKIP_STARTUP_CODESYNC.

        If the oneshot skipped the sync, it would be a no-op and the whole fix
        would be broken.
        """
        coseof_start = bootstrap_text.find("<<COSEOF")
        coseof_end = bootstrap_text.find("COSEOF", coseof_start + 8)
        unit_block = bootstrap_text[coseof_start:coseof_end]
        assert "EXONIT_SKIP_STARTUP_CODESYNC" not in unit_block, (
            "exonit-codesync.service must NOT set EXONIT_SKIP_STARTUP_CODESYNC "
            "(it would defeat the whole point of the oneshot)"
        )

    def test_codesync_started_before_workers(self, bootstrap_text):
        """The bootstrap must start exonit-codesync.service before the worker loop."""
        codesync_start_pos = bootstrap_text.find("systemctl start exonit-codesync.service")
        worker_start_loop_pos = bootstrap_text.find(
            'systemctl start "exonit-worker-w${i}.service"'
        )
        assert codesync_start_pos != -1, "systemctl start exonit-codesync.service not found"
        assert worker_start_loop_pos != -1, "worker start loop not found"
        assert codesync_start_pos < worker_start_loop_pos, (
            "exonit-codesync.service must be started before the worker start loop"
        )


# ---------------------------------------------------------------------------
# 2. Worker.run() startup-codesync guard
# ---------------------------------------------------------------------------

def _make_minimal_config(tmp_path: Path, startup_code_sync=True) -> Path:
    """Write a minimal config.json for Worker construction."""
    cfg = {
        "server_url": "http://testserver:8400",
        "token": "test-token",
        "worker_name": "test-worker",
        "gpu_enabled": False,
        "startup_code_sync": startup_code_sync,
        "gui_enabled": False,
        "cpu_threads": 1,
        "poll_interval_seconds": 5,
        "heartbeat_interval_seconds": 30,
    }
    p = tmp_path / "config.json"
    p.write_text(json.dumps(cfg))
    return p


def _make_worker(tmp_path, startup_code_sync=True):
    """Construct a minimal ExoNitWorker via __new__ (bypasses __init__ and GPU probes).

    We set just the attributes that worker.run() reads before the sync block and
    through to the end of the first loop iteration.
    """
    import logging
    import sys
    from collections import deque

    # worker.worker lives inside the worker package; the package dir must be on
    # sys.path so intra-package imports inside worker.py resolve.
    worker_pkg = str(_REPO_ROOT / "worker")
    if worker_pkg not in sys.path:
        sys.path.insert(0, worker_pkg)

    from worker.worker import ExoNitWorker  # noqa: PLC0415

    w = ExoNitWorker.__new__(ExoNitWorker)

    # Minimal attributes read by run() up to and including the sync block.
    w.logger = logging.getLogger("test_worker_bootstrap")
    w.ui = None
    w.running = False  # causes the main while loop to exit immediately
    w.paused = False
    w.status = "idle"
    w.current_phase = None
    w.phase_progress = None
    w.current_tic = None
    w.last_error = None
    w.server = "http://testserver:8400"
    w.token = "test-token"
    w.name = "test-worker"
    w.gpu_enabled = False
    w.poll_interval = 5
    w.hb_interval = 30
    w.fits_cache_dir = str(tmp_path)
    w.config = {
        "server_url": "http://testserver:8400",
        "token": "test-token",
        "startup_code_sync": startup_code_sync,
        "gui_enabled": False,
    }
    w._needs_reregister = False
    w.debug_mode = False
    return w


class TestWorkerStartupCodesyncGuard:
    """Unit tests for the EXONIT_SKIP_STARTUP_CODESYNC guard in worker.run()."""

    def test_skip_env_set_does_not_call_check_for_updates(self, tmp_path, monkeypatch):
        """With EXONIT_SKIP_STARTUP_CODESYNC=1, check_for_updates must not be called."""
        monkeypatch.setenv("EXONIT_SKIP_STARTUP_CODESYNC", "1")

        worker = _make_worker(tmp_path)

        # Patch everything that run() calls after the sync block so run()
        # returns immediately after the sync decision.
        with patch.object(worker, "check_for_updates") as mock_sync, \
             patch.object(worker, "register", return_value=None), \
             patch("threading.Thread") as mock_thread, \
             patch.object(worker, "get_task", return_value=None), \
             patch.object(worker, "_pool_enabled", return_value=False):
            # Make the run() loop exit on first iteration
            worker.running = False
            worker.run()

        mock_sync.assert_not_called(), (
            "check_for_updates must NOT be called when EXONIT_SKIP_STARTUP_CODESYNC=1"
        )

    def test_skip_env_unset_calls_check_for_updates(self, tmp_path, monkeypatch):
        """With EXONIT_SKIP_STARTUP_CODESYNC unset, check_for_updates must be called."""
        monkeypatch.delenv("EXONIT_SKIP_STARTUP_CODESYNC", raising=False)

        worker = _make_worker(tmp_path, startup_code_sync=True)

        with patch.object(worker, "check_for_updates", return_value=False) as mock_sync, \
             patch.object(worker, "register", return_value=None), \
             patch("threading.Thread") as mock_thread, \
             patch.object(worker, "get_task", return_value=None), \
             patch.object(worker, "_pool_enabled", return_value=False):
            worker.running = False
            worker.run()

        mock_sync.assert_called_once(), (
            "check_for_updates must be called at startup when EXONIT_SKIP_STARTUP_CODESYNC is unset"
        )

    def test_startup_code_sync_false_skips_regardless(self, tmp_path, monkeypatch):
        """startup_code_sync=False in config skips the sync regardless of env var."""
        monkeypatch.delenv("EXONIT_SKIP_STARTUP_CODESYNC", raising=False)

        worker = _make_worker(tmp_path, startup_code_sync=False)

        with patch.object(worker, "check_for_updates", return_value=False) as mock_sync, \
             patch.object(worker, "register", return_value=None), \
             patch("threading.Thread") as mock_thread, \
             patch.object(worker, "get_task", return_value=None), \
             patch.object(worker, "_pool_enabled", return_value=False):
            worker.running = False
            worker.run()

        mock_sync.assert_not_called(), (
            "check_for_updates must NOT be called when startup_code_sync=False in config"
        )

    def test_skip_env_any_other_value_does_not_skip(self, tmp_path, monkeypatch):
        """Only the exact string '1' triggers the skip; other values behave normally."""
        monkeypatch.setenv("EXONIT_SKIP_STARTUP_CODESYNC", "true")

        worker = _make_worker(tmp_path, startup_code_sync=True)

        with patch.object(worker, "check_for_updates", return_value=False) as mock_sync, \
             patch.object(worker, "register", return_value=None), \
             patch("threading.Thread") as mock_thread, \
             patch.object(worker, "get_task", return_value=None), \
             patch.object(worker, "_pool_enabled", return_value=False):
            worker.running = False
            worker.run()

        mock_sync.assert_called_once(), (
            "Only '1' triggers the skip; 'true' must still call check_for_updates"
        )
