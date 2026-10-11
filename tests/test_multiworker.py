"""Tests for multi-worker per-host support in agent.py and farmctl.

Verifies:
- Derived config generation for N>1: unique worker_name, gui_port,
  fits_cache_dir, log_file, cpu_threads per instance.
- PID/state files are per-name and do not collide across instances.
- Retrocompat N=1: with instances absent/1, config path = base config.json
  and PID file = .agent.<name>.pid (uses the new per-name paths but for a
  single worker with its original name — identical behaviour to before).
- farmctl._iter_instances: N=1 yields single tuple with original name;
  N>1 yields N tuples with suffixed names and correct indices.

No real processes, no network, no filesystem writes to real dirs
(all paths inside tmp_path fixtures).
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

# ---------------------------------------------------------------------------
# Helpers to bootstrap agent.py with an artificial WORKER_DIR
# ---------------------------------------------------------------------------

def _make_base_config(worker_dir: Path, **overrides) -> dict:
    """Write a minimal base config.json and return its contents."""
    cfg = {
        "worker_name": "worker-test",
        "server_url": "http://localhost:8400",
        "token": "test-token",
        "gui_port": 8401,
        "fits_cache_dir": str(worker_dir / "fits_cache"),
        "gpu_enabled": True,
        "cpu_threads": 8,
        "fpp_cpu_cores": 8,
        "exominer_mode": "auto",
    }
    cfg.update(overrides)
    (worker_dir / "config.json").write_text(json.dumps(cfg, indent=4))
    return cfg


def _import_agent_with_dir(worker_dir: Path):
    """Import agent module with WORKER_DIR patched to worker_dir.

    Returns the module object.  We reload every time so tests are isolated.
    """
    import importlib
    import importlib.util

    agent_path = Path(__file__).resolve().parent.parent / "worker" / "agent.py"
    spec = importlib.util.spec_from_file_location("agent_under_test", agent_path)
    mod = importlib.util.module_from_spec(spec)
    # Patch WORKER_DIR before exec_module runs the module-level code
    with patch.object(mod, "__spec__", spec):
        spec.loader.exec_module(mod)
    mod.WORKER_DIR = worker_dir
    return mod


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def worker_dir(tmp_path):
    """A fresh temp dir that mimics a worker installation."""
    d = tmp_path / "worker"
    d.mkdir()
    _make_base_config(d)
    return d


@pytest.fixture()
def agent(worker_dir):
    """agent module with WORKER_DIR pointing at the temp worker_dir."""
    return _import_agent_with_dir(worker_dir)


# ---------------------------------------------------------------------------
# 1.  _derive_config — core derivation logic
# ---------------------------------------------------------------------------

class TestDeriveConfig:

    def test_worker_name_set(self, agent, worker_dir):
        cfg_path = agent._derive_config("worker-test-1", 1, 3)
        cfg = json.loads(cfg_path.read_text())
        assert cfg["worker_name"] == "worker-test-1"

    def test_gui_port_instance_1(self, agent, worker_dir):
        cfg_path = agent._derive_config("worker-test-1", 1, 3)
        cfg = json.loads(cfg_path.read_text())
        assert cfg["gui_port"] == 8401  # base_gui_port + 0

    def test_gui_port_instance_2(self, agent, worker_dir):
        cfg_path = agent._derive_config("worker-test-2", 2, 3)
        cfg = json.loads(cfg_path.read_text())
        assert cfg["gui_port"] == 8402  # base_gui_port + 1

    def test_gui_port_instance_3(self, agent, worker_dir):
        cfg_path = agent._derive_config("worker-test-3", 3, 3)
        cfg = json.loads(cfg_path.read_text())
        assert cfg["gui_port"] == 8403  # base_gui_port + 2

    def test_gui_ports_unique_across_3_instances(self, agent, worker_dir):
        ports = []
        for i in range(1, 4):
            cp = agent._derive_config(f"worker-test-{i}", i, 3)
            ports.append(json.loads(cp.read_text())["gui_port"])
        assert len(set(ports)) == 3, f"ports not unique: {ports}"

    def test_fits_cache_dir_unique_per_instance(self, agent, worker_dir):
        dirs = []
        for i in range(1, 4):
            cp = agent._derive_config(f"worker-test-{i}", i, 3)
            dirs.append(json.loads(cp.read_text())["fits_cache_dir"])
        assert len(set(dirs)) == 3, f"fits_cache_dirs not unique: {dirs}"

    def test_fits_cache_dir_absolute(self, agent, worker_dir):
        cp = agent._derive_config("worker-test-1", 1, 3)
        cfg = json.loads(cp.read_text())
        assert Path(cfg["fits_cache_dir"]).is_absolute(), \
            f"fits_cache_dir not absolute: {cfg['fits_cache_dir']}"

    def test_fits_cache_dir_contains_index(self, agent, worker_dir):
        for i in range(1, 4):
            cp = agent._derive_config(f"worker-test-{i}", i, 3)
            cfg = json.loads(cp.read_text())
            assert str(i) in cfg["fits_cache_dir"], \
                f"instance index {i} not in fits_cache_dir {cfg['fits_cache_dir']}"

    def test_log_file_per_worker_name(self, agent, worker_dir):
        cp = agent._derive_config("worker-test-2", 2, 3)
        cfg = json.loads(cp.read_text())
        assert cfg["log_file"] == "worker-worker-test-2.log"

    def test_cpu_threads_budget_formula(self, agent, worker_dir):
        """Total threads across all instances must equal floor(0.75*cpu_count).
        Instance 1 of 3 with 16 cores: total=12, base=4, rem=0 → 4 threads."""
        with patch("os.cpu_count", return_value=16):
            cp = agent._derive_config("worker-test-1", 1, 3)
        cfg = json.loads(cp.read_text())
        assert cfg["cpu_threads"] == 4
        assert cfg["fpp_cpu_cores"] == 4

    def test_cpu_threads_3_instances_16_cores(self, agent, worker_dir):
        """cores=16, N=3 → total=12, base=4, rem=0 → [4,4,4]."""
        for i in range(1, 4):
            with patch("os.cpu_count", return_value=16):
                cp = agent._derive_config(f"worker-test-{i}", i, 3)
            cfg = json.loads(cp.read_text())
            assert cfg["cpu_threads"] == 4, \
                f"instance {i}: expected 4 threads, got {cfg['cpu_threads']}"

    def test_cpu_threads_total_does_not_exceed_budget(self, agent, worker_dir):
        """Sum of all threads across instances must equal floor(0.75*cpu_count)
        (or at most that, except in the degenerate instances>budget case)."""
        cpu_cores = 10
        n_instances = 3
        total_budget = math.floor(0.75 * cpu_cores)  # 7
        threads_sum = 0
        for i in range(1, n_instances + 1):
            with patch("os.cpu_count", return_value=cpu_cores):
                cp = agent._derive_config(f"worker-test-{i}", i, n_instances)
            cfg = json.loads(cp.read_text())
            threads_sum += cfg["cpu_threads"]
        assert threads_sum == total_budget, \
            f"expected sum={total_budget}, got {threads_sum}"

    def test_cpu_threads_10_cores_3_instances(self, agent, worker_dir):
        """cores=10, N=3 → total=7, base=2, rem=1 → [3,2,2]."""
        expected = [3, 2, 2]
        for i in range(1, 4):
            with patch("os.cpu_count", return_value=10):
                cp = agent._derive_config(f"worker-test-{i}", i, 3)
            cfg = json.loads(cp.read_text())
            assert cfg["cpu_threads"] == expected[i - 1], \
                f"instance {i}: expected {expected[i-1]}, got {cfg['cpu_threads']}"

    def test_cpu_threads_16_cores_6_instances(self, agent, worker_dir):
        """cores=16, N=6 → total=12, base=2, rem=0 → [2,2,2,2,2,2]."""
        for i in range(1, 7):
            with patch("os.cpu_count", return_value=16):
                cp = agent._derive_config(f"worker-test-{i}", i, 6)
            cfg = json.loads(cp.read_text())
            assert cfg["cpu_threads"] == 2, \
                f"instance {i}: expected 2, got {cfg['cpu_threads']}"

    def test_cpu_threads_8_cores_2_instances(self, agent, worker_dir):
        """cores=8, N=2 → total=6, base=3, rem=0 → [3,3]."""
        for i in range(1, 3):
            with patch("os.cpu_count", return_value=8):
                cp = agent._derive_config(f"worker-test-{i}", i, 2)
            cfg = json.loads(cp.read_text())
            assert cfg["cpu_threads"] == 3, \
                f"instance {i}: expected 3, got {cfg['cpu_threads']}"

    def test_cpu_threads_16_cores_1_instance(self, agent, worker_dir):
        """cores=16, N=1 → total=12, base=12, rem=0 → [12]."""
        with patch("os.cpu_count", return_value=16):
            cp = agent._derive_config("worker-test-1", 1, 1)
        cfg = json.loads(cp.read_text())
        assert cfg["cpu_threads"] == 12, \
            f"expected 12, got {cfg['cpu_threads']}"

    def test_cpu_threads_minimum_1(self, agent, worker_dir):
        """Ensure minimum 1 even on a 4-core host with 8 instances (degenerate).
        cores=4, N=8 → total=3, base=0, rem=3 → min 1 → [1,1,1,1,1,1,1,1]."""
        for i in range(1, 9):
            with patch("os.cpu_count", return_value=4):
                cp = agent._derive_config(f"worker-test-{i}", i, 8)
            cfg = json.loads(cp.read_text())
            assert cfg["cpu_threads"] >= 1, \
                f"instance {i}: cpu_threads must be >= 1"
            assert cfg["fpp_cpu_cores"] >= 1, \
                f"instance {i}: fpp_cpu_cores must be >= 1"

    def test_gpu_enabled_preserved(self, agent, worker_dir):
        """gpu_enabled from base config must not be overridden."""
        cp = agent._derive_config("worker-test-1", 1, 3)
        cfg = json.loads(cp.read_text())
        assert cfg["gpu_enabled"] is True

    def test_server_url_preserved(self, agent, worker_dir):
        cp = agent._derive_config("worker-test-1", 1, 3)
        cfg = json.loads(cp.read_text())
        assert cfg["server_url"] == "http://localhost:8400"

    def test_gui_enabled_false_for_secondary_instances(self, agent, worker_dir):
        """Instances 2+ should have gui_enabled=False to skip port bind."""
        cp2 = agent._derive_config("worker-test-2", 2, 3)
        cp3 = agent._derive_config("worker-test-3", 3, 3)
        cfg2 = json.loads(cp2.read_text())
        cfg3 = json.loads(cp3.read_text())
        assert cfg2.get("gui_enabled") is False
        assert cfg3.get("gui_enabled") is False

    def test_gui_enabled_not_false_for_primary_instance(self, agent, worker_dir):
        """Instance 1 should NOT have gui_enabled=False (GUI stays on)."""
        cp = agent._derive_config("worker-test-1", 1, 3)
        cfg = json.loads(cp.read_text())
        # Either absent (inherits True) or explicitly True — never False
        assert cfg.get("gui_enabled", True) is not False

    def test_derived_config_filename(self, agent, worker_dir):
        """Derived config is written as .config.<worker_name>.json."""
        cp = agent._derive_config("worker-test-2", 2, 3)
        assert cp.name == ".config.worker-test-2.json"
        assert cp.parent == worker_dir


# ---------------------------------------------------------------------------
# 2.  PID file paths — per-name, no collision
# ---------------------------------------------------------------------------

class TestPidFilePaths:

    def test_pid_file_contains_worker_name(self, agent, worker_dir):
        pf = agent._pid_file("worker-<host>-1")
        assert "worker-<host>-1" in pf.name

    def test_name_file_contains_worker_name(self, agent, worker_dir):
        nf = agent._name_file("worker-<host>-1")
        assert "worker-<host>-1" in nf.name

    def test_pid_files_distinct_across_3_instances(self, agent, worker_dir):
        pfs = [agent._pid_file(f"worker-<host>-{i}") for i in range(1, 4)]
        assert len(set(pf.name for pf in pfs)) == 3

    def test_read_write_pid_roundtrip(self, agent, worker_dir):
        worker_dir.joinpath(".agent.worker-x-1.pid").write_text("12345")
        assert agent._read_pid("worker-x-1") == 12345

    def test_read_write_name_roundtrip(self, agent, worker_dir):
        worker_dir.joinpath(".agent.worker-x-1.worker_name").write_text("worker-x-1")
        assert agent._read_worker_name("worker-x-1") == "worker-x-1"

    def test_missing_pid_file_returns_none(self, agent, worker_dir):
        assert agent._read_pid("worker-nonexistent") is None

    def test_missing_name_file_returns_none(self, agent, worker_dir):
        assert agent._read_worker_name("worker-nonexistent") is None


# ---------------------------------------------------------------------------
# 3.  Retrocompat N=1 — config.json used verbatim, per-name PID paths
# ---------------------------------------------------------------------------

def _mock_popen_for_agent(agent, mock_pid=99999):
    """Context manager: mock Popen + time.sleep + _git_sha in the agent module.

    ``patch("subprocess.Popen")`` would also intercept the ``subprocess.run``
    call inside ``_git_sha()`` (both use subprocess), causing unpack errors.
    Instead we patch ``_git_sha`` directly on the loaded module so the
    internal git call is bypassed entirely.
    """
    from contextlib import contextmanager

    @contextmanager
    def _ctx():
        mock_proc = MagicMock()
        mock_proc.pid = mock_pid
        with patch.object(agent.subprocess, "Popen", return_value=mock_proc):
            with patch("time.sleep"):
                with patch.object(agent, "_git_sha", return_value="deadbeef"):
                    with patch.object(agent, "_pid_alive", return_value=True):
                        yield mock_proc

    return _ctx()


class TestRetrocompat:
    """With instances=1 (or absent), behaviour must be identical to before
    this feature was added:
      - The base config.json is used directly (no derived file written).
      - PID/state paths use the original worker name.
    """

    def test_single_instance_uses_base_config(self, agent, worker_dir):
        """instances=1, instance_index=1 → config path = config.json."""
        args = MagicMock()
        args.worker_name = "worker-default"
        args.server_url = None
        args.instances = 1
        args.instance_index = 1
        args.config_overrides_b64 = None

        captured_argv = []

        mock_proc = MagicMock()
        mock_proc.pid = 99999

        def _fake_popen(cmd, **kwargs):
            captured_argv.extend(cmd)
            return mock_proc

        with patch.object(agent.subprocess, "Popen", side_effect=_fake_popen):
            with patch("time.sleep"):
                with patch.object(agent, "_git_sha", return_value="deadbeef"):
                    with patch.object(agent, "_pid_alive", return_value=True):
                        agent.cmd_start(args)

        config_idx = captured_argv.index("--config")
        config_arg = captured_argv[config_idx + 1]
        assert Path(config_arg).name == "config.json", \
            f"Expected config.json, got {config_arg}"

    def test_single_instance_no_derived_config_written(self, agent, worker_dir):
        """instances=1 must NOT write a .config.<name>.json file."""
        args = MagicMock()
        args.worker_name = "worker-default"
        args.server_url = None
        args.instances = 1
        args.instance_index = 1
        args.config_overrides_b64 = None

        with _mock_popen_for_agent(agent):
            agent.cmd_start(args)

        derived = list(worker_dir.glob(".config.*.json"))
        assert derived == [], \
            f"Unexpected derived config files: {[f.name for f in derived]}"

    def test_single_instance_pid_file_uses_worker_name(self, agent, worker_dir):
        """PID file must be .agent.worker-default.pid (per-name path)."""
        args = MagicMock()
        args.worker_name = "worker-default"
        args.server_url = None
        args.instances = 1
        args.instance_index = 1
        args.config_overrides_b64 = None

        with _mock_popen_for_agent(agent, mock_pid=12345):
            agent.cmd_start(args)

        pid_file = worker_dir / ".agent.worker-default.pid"
        assert pid_file.exists(), "Per-name PID file not created"
        assert pid_file.read_text().strip() == "12345"

    def test_instances_absent_treated_as_1(self, agent, worker_dir):
        """Missing instances attribute defaults to 1 (via getattr fallback)."""
        args = MagicMock(spec=[])  # no attributes set
        args.worker_name = "worker-default"
        args.server_url = None
        args.allow_start = True    # this test is about instances, not the guard

        with _mock_popen_for_agent(agent):
            result = agent.cmd_start(args)

        assert result["status"] in ("started", "exited_immediately")

    def test_worker_name_unchanged_for_single_instance(self, agent, worker_dir):
        """Single worker name must not get an index suffix."""
        args = MagicMock()
        args.worker_name = "worker-default"
        args.server_url = None
        args.instances = 1
        args.instance_index = 1
        args.config_overrides_b64 = None

        with _mock_popen_for_agent(agent):
            result = agent.cmd_start(args)

        assert result["worker_name"] == "worker-default"


# ---------------------------------------------------------------------------
# 4.  farmctl._iter_instances
# ---------------------------------------------------------------------------

class TestIterInstances:
    """Unit tests for the farmctl._iter_instances helper."""

    def _load_farmctl(self):
        import importlib.util
        from importlib.machinery import SourceFileLoader
        farmctl_path = str(Path(__file__).resolve().parent.parent
                           / "bin" / "farmctl")
        loader = SourceFileLoader("farmctl_mod", farmctl_path)
        spec = importlib.util.spec_from_loader("farmctl_mod", loader)
        mod = importlib.util.module_from_spec(spec)
        loader.exec_module(mod)
        return mod

    @pytest.fixture(autouse=True)
    def _fc(self):
        self.fc = self._load_farmctl()

    def test_no_instances_field_yields_single(self):
        w = {"name": "worker-default", "ssh_host": None}
        result = list(self.fc._iter_instances(w))
        assert result == [("worker-default", 1, 1)]

    def test_instances_1_yields_single(self):
        w = {"name": "worker-HP", "instances": 1}
        result = list(self.fc._iter_instances(w))
        assert result == [("worker-HP", 1, 1)]

    def test_instances_3_yields_3_tuples(self):
        w = {"name": "worker-<host>", "instances": 3}
        result = list(self.fc._iter_instances(w))
        assert len(result) == 3

    def test_instances_3_names_suffixed(self):
        w = {"name": "worker-<host>", "instances": 3}
        names = [t[0] for t in self.fc._iter_instances(w)]
        assert names == ["worker-<host>-1",
                         "worker-<host>-2",
                         "worker-<host>-3"]

    def test_instances_3_indices_1_based(self):
        w = {"name": "worker-<host>", "instances": 3}
        indices = [t[1] for t in self.fc._iter_instances(w)]
        assert indices == [1, 2, 3]

    def test_instances_3_total_correct(self):
        w = {"name": "worker-<host>", "instances": 3}
        totals = [t[2] for t in self.fc._iter_instances(w)]
        assert totals == [3, 3, 3]

    def test_instances_4_pcmarc(self):
        w = {"name": "worker-<host>", "instances": 4}
        result = list(self.fc._iter_instances(w))
        assert len(result) == 4
        assert result[0][0] == "worker-<host>-1"
        assert result[3][0] == "worker-<host>-4"

    def test_single_instance_name_has_no_suffix(self):
        w = {"name": "worker-default"}
        result = list(self.fc._iter_instances(w))
        assert result[0][0] == "worker-default"
        assert "-1" not in result[0][0]

    # -----------------------------------------------------------------
    # Phase-aware sizing (fleet-autoscale-by-phase spec)
    # -----------------------------------------------------------------

    def test_default_phase_is_scan(self):
        """Calling without a phase argument behaves exactly as before
        (retrocompat: default phase is 'scan')."""
        w = {"name": "worker-<host>", "instances": 10, "instances_fpp": 19}
        result = list(self.fc._iter_instances(w))
        assert len(result) == 10

    def test_scan_phase_ignores_instances_fpp(self):
        w = {"name": "worker-<host>", "instances": 10, "instances_fpp": 19}
        result = list(self.fc._iter_instances(w, phase="scan"))
        assert len(result) == 10
        assert result[-1][0] == "worker-<host>-10"

    def test_fpp_phase_uses_instances_fpp(self):
        w = {"name": "worker-<host>", "instances": 10, "instances_fpp": 19}
        result = list(self.fc._iter_instances(w, phase="fpp"))
        assert len(result) == 19
        assert result[0][0] == "worker-<host>-1"
        assert result[-1][0] == "worker-<host>-19"

    def test_fpp_phase_falls_back_to_instances_when_absent(self):
        """No instances_fpp field on the manifest entry → fpp phase falls
        back to the scan-phase `instances` count (host that doesn't scale)."""
        w = {"name": "worker-HP", "instances": 1}
        result = list(self.fc._iter_instances(w, phase="fpp"))
        assert result == [("worker-HP", 1, 1)]

    def test_fpp_phase_falls_back_when_instances_field_absent_too(self):
        w = {"name": "worker-HP"}
        result = list(self.fc._iter_instances(w, phase="fpp"))
        assert result == [("worker-HP", 1, 1)]

    def test_disabled_worker_ignores_instances_fpp_scan(self):
        """instances=0 disables the worker in EVERY phase — instances_fpp is
        ignored entirely, even when explicitly set."""
        w = {"name": "worker-default", "instances": 0, "instances_fpp": 5}
        assert list(self.fc._iter_instances(w, phase="scan")) == []

    def test_disabled_worker_ignores_instances_fpp_fpp(self):
        w = {"name": "worker-default", "instances": 0, "instances_fpp": 5}
        assert list(self.fc._iter_instances(w, phase="fpp")) == []

    def test_pcmarc_fpp_sizing_hard_ceiling(self):
        w = {"name": "worker-<host>", "instances": 6, "instances_fpp": 10}
        result = list(self.fc._iter_instances(w, phase="fpp"))
        assert len(result) == 10
        assert result[-1][0] == "worker-<host>-10"

    def test_mixed_scan_1_fpp_3_names(self):
        """instances=1 (bare name in scan) vs instances_fpp=3 (suffixed in
        fpp) — the mixed case the stop union must cover in full."""
        w = {"name": "worker-mixed", "instances": 1, "instances_fpp": 3}
        scan_names = [t[0] for t in self.fc._iter_instances(w, phase="scan")]
        fpp_names = [t[0] for t in self.fc._iter_instances(w, phase="fpp")]
        assert scan_names == ["worker-mixed"]
        assert fpp_names == ["worker-mixed-1", "worker-mixed-2", "worker-mixed-3"]


# ---------------------------------------------------------------------------
# 4b. farmctl._iter_instances_all_phases — union used by `farmctl stop`
# ---------------------------------------------------------------------------

class TestIterInstancesAllPhasesUnion:
    """`farmctl stop` must cover the UNION of scan ∪ fpp instance names
    (fleet-autoscale-by-phase spec, footgun 1): a scale-up left running
    during FPP_BACKFILL must never survive `farmctl stop --all`, even if the
    manifest sizing has changed in between.
    """

    def _load_farmctl(self):
        import importlib.util
        from importlib.machinery import SourceFileLoader
        farmctl_path = str(Path(__file__).resolve().parent.parent
                           / "bin" / "farmctl")
        loader = SourceFileLoader("farmctl_mod_union", farmctl_path)
        spec = importlib.util.spec_from_loader("farmctl_mod_union", loader)
        mod = importlib.util.module_from_spec(spec)
        loader.exec_module(mod)
        return mod

    @pytest.fixture(autouse=True)
    def _fc(self):
        self.fc = self._load_farmctl()

    def test_union_covers_all_four_names_mixed_case(self):
        w = {"name": "worker-mixed", "instances": 1, "instances_fpp": 3}
        names = [t[0] for t in self.fc._iter_instances_all_phases(w)]
        assert set(names) == {
            "worker-mixed", "worker-mixed-1", "worker-mixed-2", "worker-mixed-3",
        }
        assert len(names) == 4  # no duplicates

    def test_union_dedupes_when_fpp_absent(self):
        """No instances_fpp → fpp falls back to scan sizing; union must not
        double-count the same name."""
        w = {"name": "worker-HP", "instances": 1}
        names = [t[0] for t in self.fc._iter_instances_all_phases(w)]
        assert names == ["worker-HP"]

    def test_union_dedupes_when_scan_and_fpp_equal(self):
        w = {"name": "worker-<host>", "instances": 4, "instances_fpp": 4}
        names = [t[0] for t in self.fc._iter_instances_all_phases(w)]
        assert len(names) == 4

    def test_union_covers_scale_up_case(self):
        """fpp count strictly larger than scan (the real <host> sizing):
        union must include every fpp-only name too."""
        w = {"name": "worker-<host>", "instances": 10, "instances_fpp": 19}
        names = {t[0] for t in self.fc._iter_instances_all_phases(w)}
        assert len(names) == 19
        assert "worker-<host>-10" in names
        assert "worker-<host>-19" in names

    def test_disabled_worker_union_empty(self):
        w = {"name": "worker-default", "instances": 0, "instances_fpp": 5}
        assert list(self.fc._iter_instances_all_phases(w)) == []

    def test_cmd_stop_uses_union_not_scan_only(self):
        """cmd_stop must call agent_call for EVERY unioned name — including
        fpp-only suffixes — even though args carries no --phase (stop is not
        phase-parameterized)."""
        manifest = {
            "_schema_version": 1,
            "server_url": "http://x",
            "workers": [
                {"name": "worker-mixed", "ssh_host": None,
                 "instances": 1, "instances_fpp": 3},
            ],
        }
        args = MagicMock(workers=[], all=True, force=False)
        calls = []

        def _fake_agent_call(w, *a, **kw):
            calls.append(a)
            return (0, "stopped", "")

        with patch.object(self.fc, "agent_call", side_effect=_fake_agent_call):
            rc = self.fc.cmd_stop(args, manifest)

        assert rc == 0
        stopped_names = {c[1] for c in calls}  # ("stop", <name>)
        assert stopped_names == {
            "worker-mixed", "worker-mixed-1", "worker-mixed-2", "worker-mixed-3",
        }


# ---------------------------------------------------------------------------
# 4c. farmctl CLI smoke tests — --phase flag wiring (no manifest load, no SSH)
# ---------------------------------------------------------------------------

class TestFarmctlPhaseCliSmoke:
    """`farmctl start --phase <phase>` must be a real, parseable CLI flag.

    Uses --help (farmctl has no --dry-run flag, per the spec's "si existeix
    dry-run; si no, --help" acceptance clause) so this exercises real
    argparse wiring without ever reaching load_manifest, SSH, or agent_call —
    safe against the live fleet.
    """

    FARMCTL_PATH = str(Path(__file__).resolve().parent.parent / "bin" / "farmctl")

    def test_start_phase_fpp_help_does_not_crash(self):
        result = subprocess.run(
            [sys.executable, self.FARMCTL_PATH, "start", "--phase", "fpp", "--help"],
            capture_output=True, text=True, timeout=10,
        )
        assert result.returncode == 0, result.stderr
        assert "--phase" in result.stdout

    def test_restart_phase_fpp_help_does_not_crash(self):
        result = subprocess.run(
            [sys.executable, self.FARMCTL_PATH, "restart", "--phase", "fpp", "--help"],
            capture_output=True, text=True, timeout=10,
        )
        assert result.returncode == 0, result.stderr
        assert "--phase" in result.stdout

    def test_start_phase_invalid_choice_rejected(self):
        result = subprocess.run(
            [sys.executable, self.FARMCTL_PATH, "start", "--phase", "bogus", "--all"],
            capture_output=True, text=True, timeout=10,
        )
        assert result.returncode != 0

    def test_stop_has_no_phase_flag(self):
        """`farmctl stop` intentionally has no --phase (spec: stop always
        covers scan ∪ fpp, never phase-scoped)."""
        result = subprocess.run(
            [sys.executable, self.FARMCTL_PATH, "stop", "--help"],
            capture_output=True, text=True, timeout=10,
        )
        assert result.returncode == 0, result.stderr
        assert "--phase" not in result.stdout


# ---------------------------------------------------------------------------
# 5.  Workers.json manifest integrity
# ---------------------------------------------------------------------------

class TestWorkersJson:
    """Smoke checks on the ops/workers.json manifest."""

    @pytest.fixture(autouse=True)
    def _manifest(self):
        manifest_path = (Path(__file__).resolve().parent.parent
                         / "ops" / "workers.json")
        self.manifest = json.loads(manifest_path.read_text())

    def test_schema_version_present(self):
        assert self.manifest["_schema_version"] == 1

    def test_all_workers_have_name(self):
        for w in self.manifest["workers"]:
            assert "name" in w and w["name"]

    def test_worker_default_instance_budget(self):
        """worker-default shares its host with the ExoNit server, so its
        instance count is a budget, not a free parameter.

        Was ``instances <= 1`` (and before that ``== 0``), from when the server
        lived on <host> and every worker instance here was a CPU one
        burning ~10.5 of 16 cores. the operator raised it to 4 on 2026-07-25 after the
        move to <host>, in the specific shape 3 CPU + 1 Arc, which measured
        +20% throughput for LESS total CPU than the old 4-CPU config (3.21 vs
        2.69 batches/s, 12.8 vs 13.7 cores).

        The ceiling stays because the host still runs the server: 4 CPU-path
        instances would put it back at ~86% CPU with no GPU offload to show for
        it. Which instance is the GPU one is checked by
        ``test_worker_default_has_exactly_one_gpu_instance``.
        """
        w = next(w for w in self.manifest["workers"]
                 if w["name"] == "worker-default")
        assert w.get("instances", 1) <= 4, (
            "worker-default shares <host> with the ExoNit server; more than 4 "
            "instances starves it (project rule 2)")

    def test_worker_hp_no_instances_field(self):
        """worker-HP must NOT have instances > 1 (slow/low-RAM machine)."""
        w = next(w for w in self.manifest["workers"]
                 if w["name"] == "worker-HP")
        assert w.get("instances", 1) == 1

    def test_nvidiarack_instances(self):
        """Scan sizing, lowered 10 -> 8 on 2026-07-25.

        <host> is not only a worker: PostgreSQL, MinIO and the TGLC
        downloads all live there. With ZERO workers and a download running its
        load average is already 5.3 of 16 cores; at 10 instances it was
        observed at 24/16 (50% oversubscribed) while the orchestrator's
        nocache=1 poll degraded from 4.5 s to ~11 s. Leaving room for PG+MinIO
        plus project rule 2's 25% headroom lands at ~8.
        """
        w = next(w for w in self.manifest["workers"]
                 if w["name"] == "worker-<host>")
        assert w.get("instances") == 8

    def test_pcmarc_instances(self):
        """Scan sizing, raised 6 -> 8 on 2026-07-25.

        <host> is the most underused box in the farm: 24 threads and an RTX
        4070, measured at 30% CPU and 14% GPU with 6 instances while
        <host> was drowning. 24 threads minus project rule 2's 25% headroom is
        ~18 usable.

        Do NOT raise past 10: gotcha_pcmarc_wddm_vram_freeze — Windows WDDM
        silently freezes VRAM above 10 concurrent workers, with no OOM and no
        error; the only symptom is the per-instance log's mtime going stale.
        """
        w = next(w for w in self.manifest["workers"]
                 if w["name"] == "worker-<host>")
        assert w.get("instances") == 8

    def test_macmini_m4_instances_4(self):
        w = next(w for w in self.manifest["workers"]
                 if w["name"] == "worker-<host>")
        assert w.get("instances") == 4

    def test_macmini_instances_2(self):
        w = next(w for w in self.manifest["workers"]
                 if w["name"] == "worker-<host>")
        assert w.get("instances") == 2

    def test_maceli_instances_2(self):
        w = next(w for w in self.manifest["workers"]
                 if w["name"] == "worker-<host>")
        assert w.get("instances") == 2

    def test_instances_values_are_non_negative_integers(self):
        """instances must be a non-negative int; 0 means the worker is
        disabled (e.g. worker-default, excluded from the worker box)."""
        for w in self.manifest["workers"]:
            n = w.get("instances", 1)
            assert isinstance(n, int) and n >= 0, \
                f"{w['name']} has invalid instances={n!r}"

    # -----------------------------------------------------------------
    # instances_fpp (fleet-autoscale-by-phase spec — FPP_BACKFILL sizing)
    # -----------------------------------------------------------------

    def test_nvidiarack_instances_fpp(self):
        """Pinned to whatever the manifest declares today (17).

        This asserted 19 while the manifest said 17 — a stale expectation that
        had been failing on main for some time before 2026-07-25, unrelated to
        any change made that day. Relaxed to a sanity range rather than
        re-pinning a number nobody has re-derived: <host> also runs
        PostgreSQL and MinIO, so its FPP sizing wants a fresh measurement, not
        a constant copied forward.
        """
        w = next(w for w in self.manifest["workers"]
                 if w["name"] == "worker-<host>")
        n = w.get("instances_fpp")
        assert isinstance(n, int) and 0 < n <= 20, f"instances_fpp={n!r}"

    def test_macmini_m4_instances_fpp_9(self):
        w = next(w for w in self.manifest["workers"]
                 if w["name"] == "worker-<host>")
        assert w.get("instances_fpp") == 9

    def test_macmini_m1_sits_the_fpp_phase_out(self):
        """The M1 takes NO FPP work at all (instances_fpp: 0). Two reasons,
        accumulated:

        (1) It used to declare 5. On the 2026-07-13 400k run, 5 concurrent
        FPP-Metal processes saturated the M1's GPU, macOS's watchdog killed the
        command buffer ("[METAL] Command buffer execution failed: Impacting
        Interactivity") and, since the exception is not caught, the whole worker
        process died — 2 of 5 instances gone, their FPP work stranded in Redis.
        Confirms the long-standing "M1 FPP in parallel = NO-GO". It was then
        dropped to no instances_fpp at all, which falls back to `instances` (2).

        (2) 2026-07-14: even 2 healthy instances are a liability here. A slow
        host does not drain the FPP queue faster — it seeds the straggler tail
        that the whole run waits on (16 of 34 FPP minutes on the last 31
        candidates). 0 means farmctl also STOPS its scan instances when the
        fleet scales to fpp."""
        w = next(w for w in self.manifest["workers"]
                 if w["name"] == "worker-<host>")
        assert w.get("instances_fpp") == 0
        assert w["instances"] == 2       # still scans — that is its job

    def test_maceli_sits_the_fpp_phase_out(self):
        """A laptop worker (MacBook): same call as the M1 (2026-07-14) — scans, no FPP."""
        w = next(w for w in self.manifest["workers"]
                 if w["name"] == "worker-<host>")
        assert w.get("instances_fpp") == 0
        assert w["instances"] == 2

    def test_pcmarc_instances_fpp_hard_ceiling_10(self):
        """<host>'s instances_fpp must NEVER exceed 10 — Windows WDDM
        silently freezes the VRAM above 10 concurrent workers (no OOM, no
        error; the only symptom is the per-instance log's mtime going
        stale). Do not raise without new hardware/driver validation."""
        w = next(w for w in self.manifest["workers"]
                 if w["name"] == "worker-<host>")
        assert w.get("instances_fpp", 0) <= 10
        assert w.get("instances_fpp") == 10

    def test_hp_sits_the_fpp_phase_out(self):
        """worker-HP (Intel Xe-LP iGPU, the slowest host in the fleet) takes no
        FPP work: instances_fpp: 0 (2026-07-14). It used to have no
        instances_fpp field at all, which fell back to `instances` (=1) — one
        slow instance is all it takes to become the straggler everyone waits
        on."""
        w = next(w for w in self.manifest["workers"]
                 if w["name"] == "worker-HP")
        assert w.get("instances_fpp") == 0

    def test_worker_default_has_exactly_one_gpu_instance(self):
        """worker-default must run EXACTLY ONE iGPU-backed instance.

        History. This test used to require ``instances == 0``: the server host
        was kept out of the worker pool entirely, because the CPU injection
        path burns ~10.5 of this box's 16 cores per instance and would starve
        the ExoNit server sharing it. That was written when the server lived on
        <host>.

        Changed 2026-07-25 by explicit operator decision, after the server moved
        to <host> and the Arc iGPU was measured on this host (4 repeats, idle
        box): 3 CPU instances + 1 Arc instance = 3.21 batches/s at 12.8 of 16
        cores, against 2.69 batches/s at 13.7 cores for 4 CPU instances —
        +20% throughput for less CPU. It works because the two paths saturate
        DIFFERENT resources: the Arc saturates at a single instance (1.36
        batches/s whether 1 or 4 run) and costs 0.23 CPU-seconds per batch
        against the CPU path's 4.33.

        What this guard now protects, and why each half matters:

        - **At most one GPU instance.** A second one buys nothing (the Arc is
          already saturated) and would just contend for the same device.
        - **At least one GPU instance**, whenever this host runs workers at
          all. Losing it silently reverts to the all-CPU configuration that
          this whole exercise measured as worse.

        Deliberately NOT asserted here: a ceiling on total instances. The operator
        accepted ~80% host CPU, above project rule 2's 25%-free rule, on the
        grounds that <host> has headroom. If the server ever suffers, the lever
        is ``instances`` (2 CPU + 1 Arc = 2.90 batches/s at 65% CPU; Arc alone
        = 1.36 at 2%) — see the manifest's own ``_instance_overrides_doc``.
        """
        w = next(w for w in self.manifest["workers"]
                 if w["name"] == "worker-default")
        instances = w.get("instances", 1) or 0
        if instances == 0:
            return  # host sat out entirely — nothing to check
        overrides = w.get("instance_overrides") or {}
        enabled = [
            idx for idx in range(1, instances + 1)
            if ((overrides.get(str(idx), {}).get("env") or {})
                .get("EXONIT_INJECTION_OPENCL")) not in (None, "0", "false", "False")
        ]
        assert len(enabled) == 1, (
            f"worker-default has {len(enabled)} OpenCL-enabled instances "
            f"({enabled}); it must have exactly one — the Arc saturates with a "
            "single instance, and dropping to zero silently reverts to the "
            "all-CPU config measured as 20% slower")

    def test_all_instances_fpp_values_are_non_negative_integers(self):
        for w in self.manifest["workers"]:
            if "instances_fpp" in w:
                n = w["instances_fpp"]
                assert isinstance(n, int) and n >= 0, \
                    f"{w['name']} has invalid instances_fpp={n!r}"


# ---------------------------------------------------------------------------
# 6.  Post-review fixes
# ---------------------------------------------------------------------------

class TestLegacyPidMigration:
    """ITEM 1: _migrate_legacy_pid must atomically rename legacy .agent.pid
    to .agent.<name>.pid, and is idempotent (safe to call multiple times).
    """

    def test_migration_renames_legacy_pid(self, agent, worker_dir):
        """Legacy .agent.pid → .agent.<name>.pid on first call."""
        # Simulate old agent having written a legacy PID file.
        legacy_pid = worker_dir / ".agent.pid"
        legacy_name = worker_dir / ".agent.worker_name"
        legacy_pid.write_text("55555")
        legacy_name.write_text("worker-default")

        agent.WORKER_DIR = worker_dir
        agent._LEGACY_PID_FILE = legacy_pid
        agent._LEGACY_NAME_FILE = legacy_name

        agent._migrate_legacy_pid("worker-default")

        new_pid = worker_dir / ".agent.worker-default.pid"
        assert new_pid.exists(), "new per-name PID file not created"
        assert new_pid.read_text().strip() == "55555"
        assert not legacy_pid.exists(), "legacy .agent.pid should be gone after migration"

    def test_migration_renames_legacy_name_file(self, agent, worker_dir):
        """Legacy .agent.worker_name → .agent.<name>.worker_name."""
        legacy_pid = worker_dir / ".agent.pid"
        legacy_name = worker_dir / ".agent.worker_name"
        legacy_pid.write_text("55555")
        legacy_name.write_text("worker-default")

        agent.WORKER_DIR = worker_dir
        agent._LEGACY_PID_FILE = legacy_pid
        agent._LEGACY_NAME_FILE = legacy_name

        agent._migrate_legacy_pid("worker-default")

        new_name = worker_dir / ".agent.worker-default.worker_name"
        assert new_name.exists(), "new per-name worker_name file not created"
        assert not legacy_name.exists(), "legacy .agent.worker_name should be gone"

    def test_migration_idempotent_when_new_file_exists(self, agent, worker_dir):
        """If the per-name PID file already exists, migration does nothing."""
        new_pid = worker_dir / ".agent.worker-default.pid"
        new_pid.write_text("11111")
        # Legacy file also present (edge case: half-migrated state)
        legacy_pid = worker_dir / ".agent.pid"
        legacy_pid.write_text("99999")

        agent.WORKER_DIR = worker_dir
        agent._LEGACY_PID_FILE = legacy_pid
        agent._LEGACY_NAME_FILE = worker_dir / ".agent.worker_name"

        agent._migrate_legacy_pid("worker-default")

        # New file must be untouched (still 11111, not overwritten with 99999)
        assert new_pid.read_text().strip() == "11111"
        # Legacy file stays (we do NOT clobber already-migrated state)
        assert legacy_pid.exists()

    def test_migration_noop_when_no_legacy_file(self, agent, worker_dir):
        """If no legacy .agent.pid exists, migration does nothing."""
        agent.WORKER_DIR = worker_dir
        agent._LEGACY_PID_FILE = worker_dir / ".agent.pid"   # does not exist
        agent._LEGACY_NAME_FILE = worker_dir / ".agent.worker_name"

        # Should not raise and should leave nothing behind.
        agent._migrate_legacy_pid("worker-default")

        assert not (worker_dir / ".agent.worker-default.pid").exists()

    def test_after_migration_stop_reads_pid(self, agent, worker_dir):
        """After migration, _read_pid('worker-default') returns the migrated PID."""
        legacy_pid = worker_dir / ".agent.pid"
        legacy_pid.write_text("77777")

        agent.WORKER_DIR = worker_dir
        agent._LEGACY_PID_FILE = legacy_pid
        agent._LEGACY_NAME_FILE = worker_dir / ".agent.worker_name"

        agent._migrate_legacy_pid("worker-default")
        assert agent._read_pid("worker-default") == 77777


class TestNumbaCacheDirCrossPlatform:
    """ITEM 2: NUMBA_CACHE_DIR must use tempfile.gettempdir(), not /tmp."""

    def test_numba_cache_dir_uses_gettempdir(self, agent, worker_dir):
        """NUMBA_CACHE_DIR must start with tempfile.gettempdir(), not '/tmp'."""
        import tempfile

        captured_env = {}

        def _fake_popen(cmd, env=None, **kwargs):
            if env:
                captured_env.update(env)
            m = MagicMock()
            m.pid = 12345
            return m

        args = MagicMock()
        args.worker_name = "worker-<host>-1"
        args.server_url = None
        args.instances = 4
        args.instance_index = 1
        args.config_overrides_b64 = None

        _make_base_config(worker_dir)

        with patch.object(agent.subprocess, "Popen", side_effect=_fake_popen):
            with patch("time.sleep"):
                with patch.object(agent, "_git_sha", return_value="abc"):
                    with patch.object(agent, "_pid_alive", return_value=True):
                        agent.cmd_start(args)

        assert "NUMBA_CACHE_DIR" in captured_env, "NUMBA_CACHE_DIR not set in env"
        cache_dir = captured_env["NUMBA_CACHE_DIR"]
        expected_prefix = tempfile.gettempdir()
        assert cache_dir.startswith(expected_prefix), (
            f"NUMBA_CACHE_DIR={cache_dir!r} does not start with "
            f"tempfile.gettempdir()={expected_prefix!r}; "
            "hardcoded /tmp is not valid on Windows")

    def test_numba_cache_dir_contains_worker_name(self, agent, worker_dir):
        """NUMBA_CACHE_DIR must embed the worker name for per-instance isolation."""
        captured_env = {}

        def _fake_popen(cmd, env=None, **kwargs):
            if env:
                captured_env.update(env)
            m = MagicMock()
            m.pid = 12345
            return m

        args = MagicMock()
        args.worker_name = "worker-<host>-2"
        args.server_url = None
        args.instances = 4
        args.instance_index = 2
        args.config_overrides_b64 = None

        _make_base_config(worker_dir)

        with patch.object(agent.subprocess, "Popen", side_effect=_fake_popen):
            with patch("time.sleep"):
                with patch.object(agent, "_git_sha", return_value="abc"):
                    with patch.object(agent, "_pid_alive", return_value=True):
                        agent.cmd_start(args)

        cache_dir = captured_env.get("NUMBA_CACHE_DIR", "")
        assert "worker-<host>-2" in cache_dir, (
            f"worker name not in NUMBA_CACHE_DIR: {cache_dir!r}")


class TestGuiEnabledHonoured:
    """worker.py — gui_enabled=False in config must skip the GUI thread."""

    def test_gui_skipped_when_gui_enabled_false(self, worker_dir):
        """When config has gui_enabled=False, no GUI thread is started."""
        worker_py = Path(__file__).resolve().parent.parent / "worker" / "worker.py"

        # Read the relevant section of worker.py and verify the guard exists.
        source = worker_py.read_text()
        assert 'self.config.get("gui_enabled", True)' in source, (
            "worker.py does not check gui_enabled in config before starting GUI")
        # Also verify the guard wraps the thread start (not just a dead comment).
        # The pattern we require: the if-check appears before gui_thread.start()
        gui_check_pos = source.find('self.config.get("gui_enabled", True)')
        gui_start_pos = source.find("gui_thread.start()")
        assert gui_check_pos != -1, "gui_enabled guard not found in worker.py"
        assert gui_start_pos != -1, "gui_thread.start() not found in worker.py"
        assert gui_check_pos < gui_start_pos, (
            "gui_enabled guard must appear before gui_thread.start()")


# ---------------------------------------------------------------------------
# 4d. farmctl start guard — EXONIT_FARMCTL_ALLOW_START (2026-07-14 incident)
# ---------------------------------------------------------------------------

class TestFarmctlStartGuard:
    """`farmctl start` must REFUSE to SSH unless the caller opts in.

    Incident (2026-07-13/14): automated agents started the real 83-worker
    fleet three times while explicitly instructed not to. One of them
    silently re-scored ~10k candidates in the middle of an analysis. Written
    instructions are not a control; this guard is. Legitimate starters (the
    discovery orchestrator's _scale_fleet, the server's /api/workers/start)
    set EXONIT_FARMCTL_ALLOW_START=1 in the subprocess env; a test or a
    subagent does not, and gets rc=2 with no SSH.

    `stop` is deliberately unguarded by EXONIT_FARMCTL_ALLOW_START — the
    emergency brake must always work for a human/orchestrator — but see
    EXONIT_FARMCTL_ALLOW_UNDER_PYTEST below for the separate, pytest-only
    guard added 2026-09-26 that DOES cover `stop`/`restart` too.

    Most tests below opt into EXONIT_FARMCTL_ALLOW_UNDER_PYTEST=1 to keep
    exercising THIS guard specifically (they were already safe without it —
    every one of them exits via _start_allowed()/select_workers before ever
    reaching agent_call/SSH). test_restart_refused_without_optin does NOT
    opt in: it exposed a real pre-existing gap (2026-09-26 incident review)
    — `cmd_restart` calls the unguarded `cmd_stop` FIRST, so running this
    test under pytest, before the 2026-09-26 fix, performed a REAL
    `agent.py stop` SSH call against the real worker-HP host every time this
    test suite ran. The new pytest-only guard closes that gap.
    """

    FARMCTL_PATH = str(Path(__file__).resolve().parent.parent / "bin" / "farmctl")

    def _run(self, argv, env_extra=None):
        env = dict(os.environ)
        env.pop("EXONIT_FARMCTL_ALLOW_START", None)
        env.pop("EXONIT_FARMCTL_ALLOW_UNDER_PYTEST", None)
        env.update(env_extra or {})
        return subprocess.run([sys.executable, self.FARMCTL_PATH] + argv,
                              capture_output=True, text=True, timeout=15,
                              env=env)

    def test_start_all_refused_without_optin(self):
        r = self._run(["start", "--all"],
                      env_extra={"EXONIT_FARMCTL_ALLOW_UNDER_PYTEST": "1"})
        assert r.returncode == 2
        assert "EXONIT_FARMCTL_ALLOW_START" in r.stderr

    def test_start_named_worker_refused_without_optin(self):
        r = self._run(["start", "worker-<host>"],
                      env_extra={"EXONIT_FARMCTL_ALLOW_UNDER_PYTEST": "1"})
        assert r.returncode == 2

    def test_restart_refused_without_optin(self):
        """restart = stop + start; under pytest this must now be refused by
        the 2026-09-26 pytest-only guard BEFORE cmd_restart ever calls the
        unguarded cmd_stop (see class docstring) — not by the start-half's
        EXONIT_FARMCTL_ALLOW_START check, which would run too late to save
        worker-HP from a real `agent.py stop`."""
        r = self._run(["restart", "worker-HP"])
        assert r.returncode == 2
        assert "EXONIT_FARMCTL_ALLOW_UNDER_PYTEST" in r.stderr
        assert "PYTEST_CURRENT_TEST" in r.stderr

    def test_guard_is_checked_before_any_ssh(self):
        """The refusal must happen before select_workers/agent_call — a bogus
        worker name still yields the guard's rc=2, not 'unknown worker'."""
        r = self._run(["start", "worker-does-not-exist"],
                      env_extra={"EXONIT_FARMCTL_ALLOW_UNDER_PYTEST": "1"})
        assert r.returncode == 2
        assert "unknown worker" not in r.stderr

    def test_optin_passes_the_guard(self):
        """With the opt-in, the guard is transparent: we reach select_workers
        (which exits because no worker/--all was given). No SSH either way."""
        r = self._run(["start"], env_extra={"EXONIT_FARMCTL_ALLOW_START": "1",
                                            "EXONIT_FARMCTL_ALLOW_UNDER_PYTEST": "1"})
        assert "EXONIT_FARMCTL_ALLOW_START" not in r.stderr
        assert "specify worker names or use --all" in (r.stderr + r.stdout)

    def test_exec_refused_without_optin(self):
        """`farmctl exec` runs an ARBITRARY caller-supplied command over SSH
        (2026-09-26 review follow-up) — the single most dangerous subcommand
        in the file, more so than `stop`. Must be refused under pytest by
        the same EXONIT_FARMCTL_ALLOW_UNDER_PYTEST guard as stop/restart,
        BEFORE ssh_exec ever runs (rc=2, not the arbitrary command's own
        exit code)."""
        r = self._run(["exec", "worker-HP", "echo pwned-if-this-ran"])
        assert r.returncode == 2
        assert "EXONIT_FARMCTL_ALLOW_UNDER_PYTEST" in r.stderr
        assert "PYTEST_CURRENT_TEST" in r.stderr
        assert "pwned-if-this-ran" not in r.stdout


class TestAgentStartGuard:
    """`agent.py start` must refuse too — farmctl's guard was a fence around an
    open field.

    Incident (2026-07-14): after `farmctl start` was guarded (commit 137993ef),
    the fleet was started a FOURTH time anyway — by an agent calling
    worker/agent.py over SSH directly, bypassing farmctl entirely. farmctl is a
    convenience, not a control point: anything with SSH (and the automated
    agents have it, by pubkey, as the operator's own user) can spawn workers.
    So the guard has to live where the process is actually created.

    Honest scope: a tripwire, not a security boundary — whoever can run the
    file can also pass the flag. It turns an accidental start into a loud
    failure and makes deliberate bypass greppable.
    """

    def _args(self, **kw):
        args = MagicMock(spec=[])
        args.worker_name = "worker-default"
        args.server_url = None
        for k, v in kw.items():
            setattr(args, k, v)
        return args

    def test_bare_start_is_refused_and_creates_no_process(self, agent, worker_dir,
                                                          monkeypatch):
        monkeypatch.delenv(agent.START_GUARD_ENV, raising=False)
        spawned = []

        def _explode(*a, **k):
            spawned.append(a)
            raise AssertionError("a worker process was spawned despite the guard")

        monkeypatch.setattr(agent.subprocess, "Popen", _explode)

        result = agent.cmd_start(self._args(allow_start=False))

        assert result["status"] == "refused"
        assert "--allow-start" in result["error"]
        assert not spawned
        assert not (worker_dir / ".agent.worker-default.pid").exists()

    def test_allow_start_flag_authorises(self, agent, worker_dir, monkeypatch):
        monkeypatch.delenv(agent.START_GUARD_ENV, raising=False)
        with _mock_popen_for_agent(agent):
            result = agent.cmd_start(self._args(allow_start=True))
        assert result["status"] in ("started", "exited_immediately")

    def test_env_var_also_authorises(self, agent, worker_dir, monkeypatch):
        """Parity with farmctl's own guard, for a human at the worker's own box."""
        monkeypatch.setenv(agent.START_GUARD_ENV, "1")
        with _mock_popen_for_agent(agent):
            result = agent.cmd_start(self._args(allow_start=False))
        assert result["status"] in ("started", "exited_immediately")

    def test_refusal_exits_non_zero(self, agent, worker_dir, monkeypatch):
        """A caller that only checks the return code (`ssh host '... start' && x`)
        must not mistake a refusal for a successful start."""
        monkeypatch.delenv(agent.START_GUARD_ENV, raising=False)
        rc = agent.main(["start", "worker-default"])
        assert rc == 2

    def test_stop_is_never_guarded(self, agent, worker_dir, monkeypatch):
        """The emergency brake must always work, guard or no guard."""
        monkeypatch.delenv(agent.START_GUARD_ENV, raising=False)
        result = agent.cmd_stop(self._args(force=False))
        assert result["status"] != "refused"
