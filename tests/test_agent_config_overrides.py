"""Tests for per-host config overrides (config-overrides-per-host feature).

Closes farmctl-deploy-clobbers-worker-config: `ops/workers.json` may declare
a per-worker `config` dict (e.g. <host>'s ``{"gpu_enabled": true}``).
`bin/farmctl` base64-encodes it and passes ``--config-overrides-b64`` to
`worker/agent.py`, which merges it into the derived ``.config.<name>.json``
on every start — surviving a `farmctl deploy` clobber of the tracked
config.json.

Verifies:
- Multi-instance derivation with overrides: the override value appears in
  the derived config, but per-instance derived fields (gui_port, cpu_threads,
  fits_cache_dir, log_file) still win — an override can never break those.
- Single-instance WITH overrides: writes .config.<name>.json = base +
  overrides, with NO multi-instance redistribution of cpu_threads/ports.
- Single-instance WITHOUT overrides: unchanged base-config-verbatim path
  (bit-identical retrocompat — no regression from this feature).
- bin/farmctl's b64 encoding helper round-trips JSON with quotes/spaces.
- Malformed overrides (bad b64, bad JSON, JSON that isn't a dict) refuse the
  start with a clear error rather than silently ignoring them.

No real processes, no network, no filesystem writes outside tmp_path.
"""
from __future__ import annotations

import base64
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

# ---------------------------------------------------------------------------
# Helpers to bootstrap agent.py with an artificial WORKER_DIR (mirrors
# tests/test_multiworker.py's pattern).
# ---------------------------------------------------------------------------

def _make_base_config(worker_dir: Path, **overrides) -> dict:
    cfg = {
        "worker_name": "worker-test",
        "server_url": "http://localhost:8400",
        "token": "test-token",
        "gui_port": 8401,
        "fits_cache_dir": str(worker_dir / "fits_cache"),
        "gpu_enabled": False,
        "cpu_threads": 8,
        "fpp_cpu_cores": 8,
        "exominer_mode": "auto",
    }
    cfg.update(overrides)
    (worker_dir / "config.json").write_text(json.dumps(cfg, indent=4))
    return cfg


def _import_agent_with_dir(worker_dir: Path):
    import importlib.util

    agent_path = Path(__file__).resolve().parent.parent / "worker" / "agent.py"
    spec = importlib.util.spec_from_file_location("agent_overrides_under_test", agent_path)
    mod = importlib.util.module_from_spec(spec)
    with patch.object(mod, "__spec__", spec):
        spec.loader.exec_module(mod)
    mod.WORKER_DIR = worker_dir
    return mod


def _load_farmctl():
    import importlib.util
    from importlib.machinery import SourceFileLoader

    farmctl_path = str(Path(__file__).resolve().parent.parent / "bin" / "farmctl")
    loader = SourceFileLoader("farmctl_config_overrides_mod", farmctl_path)
    spec = importlib.util.spec_from_loader("farmctl_config_overrides_mod", loader)
    mod = importlib.util.module_from_spec(spec)
    loader.exec_module(mod)
    return mod


@pytest.fixture()
def worker_dir(tmp_path):
    d = tmp_path / "worker"
    d.mkdir()
    _make_base_config(d)
    return d


@pytest.fixture()
def agent(worker_dir):
    return _import_agent_with_dir(worker_dir)


def _mock_popen_for_agent(agent, mock_pid=99999):
    """Same pattern as test_multiworker.py: mock Popen + sleep + _git_sha so
    cmd_start runs to completion without spawning a real process."""
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


def _b64(obj) -> str:
    return base64.b64encode(json.dumps(obj).encode("utf-8")).decode("ascii")


# ---------------------------------------------------------------------------
# 1. Multi-instance derivation with overrides — per-instance fields still win
# ---------------------------------------------------------------------------

class TestDeriveConfigWithOverrides:

    def test_override_value_applied(self, agent, worker_dir):
        """An override key not touched by per-instance derivation passes
        through to the derived config."""
        cfg_path = agent._derive_config(
            "worker-test-1", 1, 3, overrides={"gpu_enabled": True})
        cfg = json.loads(cfg_path.read_text())
        assert cfg["gpu_enabled"] is True

    def test_per_instance_gui_port_wins_over_override(self, agent, worker_dir):
        """Even if an override tried to set gui_port, the per-instance value
        must win (ports must stay unique across instances)."""
        cfg_path = agent._derive_config(
            "worker-test-2", 2, 3, overrides={"gui_port": 1})
        cfg = json.loads(cfg_path.read_text())
        assert cfg["gui_port"] == 8402  # base_gui_port + (2-1), NOT 1

    def test_per_instance_cpu_threads_wins_over_override(self, agent, worker_dir):
        """cpu_threads override must be overridden by the 25%-free-CPU
        per-instance distribution (correctness rule, non-negotiable)."""
        with patch("os.cpu_count", return_value=16):
            cfg_path = agent._derive_config(
                "worker-test-1", 1, 3, overrides={"cpu_threads": 999})
        cfg = json.loads(cfg_path.read_text())
        assert cfg["cpu_threads"] == 4  # floor(0.75*16)/3 == 4, NOT 999
        assert cfg["fpp_cpu_cores"] == 4

    def test_per_instance_fits_cache_dir_wins_over_override(self, agent, worker_dir):
        cfg_path = agent._derive_config(
            "worker-test-1", 1, 3,
            overrides={"fits_cache_dir": "/should/not/be/used"})
        cfg = json.loads(cfg_path.read_text())
        assert "/should/not/be/used" not in cfg["fits_cache_dir"]
        assert "fits_cache" in cfg["fits_cache_dir"]

    def test_per_instance_log_file_wins_over_override(self, agent, worker_dir):
        cfg_path = agent._derive_config(
            "worker-test-2", 2, 3, overrides={"log_file": "custom.log"})
        cfg = json.loads(cfg_path.read_text())
        assert cfg["log_file"] == "worker-worker-test-2.log"

    def test_overrides_none_behaves_like_absent(self, agent, worker_dir):
        """overrides=None must not change any existing multi-instance
        behaviour (backward compat for the pre-feature call signature)."""
        cfg_path = agent._derive_config("worker-test-1", 1, 3, overrides=None)
        cfg = json.loads(cfg_path.read_text())
        assert cfg["gpu_enabled"] is False  # from base config, unmodified

    def test_overrides_empty_dict_behaves_like_absent(self, agent, worker_dir):
        cfg_path = agent._derive_config("worker-test-1", 1, 3, overrides={})
        cfg = json.loads(cfg_path.read_text())
        assert cfg["gpu_enabled"] is False


# ---------------------------------------------------------------------------
# 2. Single-instance WITH overrides — base + overrides, no redistribution
# ---------------------------------------------------------------------------

class TestSingleInstanceWithOverrides:

    def test_apply_overrides_merges_into_base(self, agent, worker_dir):
        cfg_path = agent._apply_overrides("worker-test", {"gpu_enabled": True})
        cfg = json.loads(cfg_path.read_text())
        assert cfg["gpu_enabled"] is True
        # Unrelated base fields untouched.
        assert cfg["server_url"] == "http://localhost:8400"
        assert cfg["token"] == "test-token"

    def test_apply_overrides_no_cpu_redistribution(self, agent, worker_dir):
        """Single-instance semantics: cpu_threads/fpp_cpu_cores stay exactly
        as in the base config (no floor(0.75*cpu_count) recompute)."""
        cfg_path = agent._apply_overrides("worker-test", {"gpu_enabled": True})
        cfg = json.loads(cfg_path.read_text())
        assert cfg["cpu_threads"] == 8  # unchanged from base
        assert cfg["fpp_cpu_cores"] == 8

    def test_apply_overrides_no_gui_port_shift(self, agent, worker_dir):
        cfg_path = agent._apply_overrides("worker-test", {"gpu_enabled": True})
        cfg = json.loads(cfg_path.read_text())
        assert cfg["gui_port"] == 8401  # unchanged from base

    def test_apply_overrides_no_fits_cache_suffix(self, agent, worker_dir):
        cfg_path = agent._apply_overrides("worker-test", {"gpu_enabled": True})
        cfg = json.loads(cfg_path.read_text())
        assert cfg["fits_cache_dir"] == str(worker_dir / "fits_cache")

    def test_apply_overrides_writes_derived_filename(self, agent, worker_dir):
        cfg_path = agent._apply_overrides("worker-<host>", {"gpu_enabled": True})
        assert cfg_path.name == ".config.worker-<host>.json"
        assert cfg_path.parent == worker_dir

    def test_cmd_start_single_instance_with_overrides_writes_derived_config(
            self, agent, worker_dir):
        """End-to-end via cmd_start: instances=1 + overrides → derived config
        file is written (NOT the base config.json path)."""
        args = MagicMock()
        args.worker_name = "worker-<host>"
        args.server_url = None
        args.instances = 1
        args.instance_index = 1
        args.allow_start = True
        args.config_overrides_b64 = _b64({"gpu_enabled": True})

        captured_argv = []
        mock_proc = MagicMock()
        mock_proc.pid = 12345

        def _fake_popen(cmd, **kwargs):
            captured_argv.extend(cmd)
            return mock_proc

        with patch.object(agent.subprocess, "Popen", side_effect=_fake_popen):
            with patch("time.sleep"):
                with patch.object(agent, "_git_sha", return_value="deadbeef"):
                    with patch.object(agent, "_pid_alive", return_value=True):
                        result = agent.cmd_start(args)

        assert result["status"] in ("started", "exited_immediately")
        config_idx = captured_argv.index("--config")
        config_arg = Path(captured_argv[config_idx + 1])
        assert config_arg.name == ".config.worker-<host>.json"
        cfg = json.loads(config_arg.read_text())
        assert cfg["gpu_enabled"] is True
        # Single-instance semantics preserved even with overrides applied.
        assert cfg["cpu_threads"] == 8
        assert cfg["gui_port"] == 8401


# ---------------------------------------------------------------------------
# 3. Single-instance WITHOUT overrides — unchanged bit-identical retrocompat
# ---------------------------------------------------------------------------

class TestSingleInstanceWithoutOverridesUnchanged:

    def test_cmd_start_no_overrides_uses_base_config_verbatim(self, agent, worker_dir):
        args = MagicMock()
        args.worker_name = "worker-default"
        args.server_url = None
        args.instances = 1
        args.instance_index = 1
        args.allow_start = True
        args.config_overrides_b64 = None

        captured_argv = []
        mock_proc = MagicMock()
        mock_proc.pid = 12345

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
        assert Path(config_arg).name == "config.json"

    def test_no_config_overrides_b64_attr_defaults_to_none(self, agent, worker_dir):
        """Missing --config-overrides-b64 attribute on args (e.g. an older
        farmctl or a test double) must be treated as absent, not an error."""
        args = MagicMock(spec=["worker_name", "server_url", "instances",
                                "instance_index", "allow_start"])
        args.worker_name = "worker-default"
        args.server_url = None
        args.instances = 1
        args.instance_index = 1
        args.allow_start = True

        with _mock_popen_for_agent(agent):
            result = agent.cmd_start(args)

        assert result["status"] in ("started", "exited_immediately")
        derived = list(worker_dir.glob(".config.*.json"))
        assert derived == []

    def test_no_derived_config_file_written(self, agent, worker_dir):
        args = MagicMock()
        args.worker_name = "worker-default"
        args.server_url = None
        args.instances = 1
        args.instance_index = 1
        args.allow_start = True
        args.config_overrides_b64 = None

        with _mock_popen_for_agent(agent):
            agent.cmd_start(args)

        derived = list(worker_dir.glob(".config.*.json"))
        assert derived == []


# ---------------------------------------------------------------------------
# 4. bin/farmctl's b64 encoding helper — round-trip, including conflictive chars
# ---------------------------------------------------------------------------

class TestFarmctlEncodingHelper:

    @pytest.fixture(autouse=True)
    def _fc(self):
        self.fc = _load_farmctl()

    def test_roundtrip_simple_dict(self):
        overrides = {"gpu_enabled": True}
        encoded = self.fc._encode_config_overrides_b64(overrides)
        decoded = json.loads(base64.b64decode(encoded).decode("utf-8"))
        assert decoded == overrides

    def test_roundtrip_with_quotes_and_spaces(self):
        overrides = {"note": 'has "quotes" and spaces', "path": "C:\\Users\\a b"}
        encoded = self.fc._encode_config_overrides_b64(overrides)
        decoded = json.loads(base64.b64decode(encoded).decode("utf-8"))
        assert decoded == overrides

    def test_encoded_output_is_pure_base64_alphabet(self):
        """Must be safe to embed as a single shell token on cmd.exe/PowerShell/
        POSIX without quoting — i.e. alphanumeric plus +/=."""
        import re
        overrides = {"note": 'quotes " and spaces and $(cmd) and & and |'}
        encoded = self.fc._encode_config_overrides_b64(overrides)
        assert re.fullmatch(r"[A-Za-z0-9+/=]+", encoded)

    def test_matches_agent_decoder(self, agent, worker_dir):
        """farmctl's encoder output must be exactly what agent.py's decoder
        accepts — the cross-module contract this feature depends on."""
        overrides = {"gpu_enabled": True, "note": "<host> RTX 5060 Ti"}
        encoded = self.fc._encode_config_overrides_b64(overrides)
        decoded = agent._decode_config_overrides_b64(encoded)
        assert decoded == overrides


# ---------------------------------------------------------------------------
# 5. Malformed overrides refuse the start (no silent ignore)
# ---------------------------------------------------------------------------

class TestMalformedOverridesRefuseStart:

    def test_invalid_base64_raises(self, agent, worker_dir):
        with pytest.raises(ValueError):
            agent._decode_config_overrides_b64("not-valid-base64!!!")

    def test_valid_base64_invalid_json_raises(self, agent, worker_dir):
        bad_json_b64 = base64.b64encode(b"{not json}").decode("ascii")
        with pytest.raises(ValueError):
            agent._decode_config_overrides_b64(bad_json_b64)

    def test_json_list_not_dict_raises(self, agent, worker_dir):
        list_b64 = base64.b64encode(json.dumps([1, 2, 3]).encode()).decode("ascii")
        with pytest.raises(ValueError):
            agent._decode_config_overrides_b64(list_b64)

    def test_json_scalar_not_dict_raises(self, agent, worker_dir):
        scalar_b64 = base64.b64encode(json.dumps("just a string").encode()).decode("ascii")
        with pytest.raises(ValueError):
            agent._decode_config_overrides_b64(scalar_b64)

    def test_cmd_start_refuses_on_malformed_b64(self, agent, worker_dir, monkeypatch):
        spawned = []

        def _explode(*a, **k):
            spawned.append(a)
            raise AssertionError("a worker process was spawned despite malformed overrides")

        monkeypatch.setattr(agent.subprocess, "Popen", _explode)

        args = MagicMock()
        args.worker_name = "worker-default"
        args.server_url = None
        args.instances = 1
        args.instance_index = 1
        args.allow_start = True
        args.config_overrides_b64 = "%%%not-base64%%%"

        result = agent.cmd_start(args)

        assert result["status"] == "refused"
        assert "error" in result and result["error"]
        assert not spawned
        # No PID file, no derived config: refusal must be a clean no-op.
        assert not (worker_dir / ".agent.worker-default.pid").exists()
        assert list(worker_dir.glob(".config.*.json")) == []

    def test_cmd_start_refuses_on_malformed_json(self, agent, worker_dir, monkeypatch):
        spawned = []
        monkeypatch.setattr(
            agent.subprocess, "Popen",
            lambda *a, **k: spawned.append(a) or MagicMock())

        args = MagicMock()
        args.worker_name = "worker-default"
        args.server_url = None
        args.instances = 1
        args.instance_index = 1
        args.allow_start = True
        args.config_overrides_b64 = base64.b64encode(b"{broken").decode("ascii")

        result = agent.cmd_start(args)

        assert result["status"] == "refused"
        assert not spawned

    def test_cmd_start_refuses_on_non_dict_json(self, agent, worker_dir, monkeypatch):
        spawned = []
        monkeypatch.setattr(
            agent.subprocess, "Popen",
            lambda *a, **k: spawned.append(a) or MagicMock())

        args = MagicMock()
        args.worker_name = "worker-default"
        args.server_url = None
        args.instances = 1
        args.instance_index = 1
        args.allow_start = True
        args.config_overrides_b64 = base64.b64encode(json.dumps([1, 2]).encode()).decode("ascii")

        result = agent.cmd_start(args)

        assert result["status"] == "refused"
        assert not spawned

    def test_refusal_via_main_exits_nonzero(self, agent, worker_dir, monkeypatch):
        """main()'s existing refused-status→rc=2 wiring must cover this
        refusal path too (same contract as the --allow-start guard)."""
        monkeypatch.setattr(
            agent.subprocess, "Popen",
            lambda *a, **k: (_ for _ in ()).throw(
                AssertionError("must not spawn on malformed overrides")))
        rc = agent.main([
            "start", "worker-default", "--allow-start",
            "--config-overrides-b64", "%%%not-base64%%%",
        ])
        assert rc == 2


# ---------------------------------------------------------------------------
# 6. bin/farmctl cmd_start wiring — config field triggers the CLI arg
# ---------------------------------------------------------------------------

class TestFarmctlCmdStartWiring:
    """cmd_start must append --config-overrides-b64 for a worker whose
    manifest entry declares a non-empty `config` dict, and must NOT append
    it otherwise (retrocompat for every host without a `config` field)."""

    @pytest.fixture(autouse=True)
    def _fc(self):
        self.fc = _load_farmctl()

    def _run_cmd_start(self, monkeypatch, workers):
        manifest = {
            "_schema_version": 1,
            "server_url": "http://x",
            "workers": workers,
        }
        args = MagicMock(workers=[], all=True, phase="scan", no_prune=True)
        calls = []

        def _fake_agent_call(w, *a, **kw):
            calls.append(a)
            return (0, "started", "")

        monkeypatch.setenv(self.fc.START_GUARD_ENV, "1")
        with patch.object(self.fc, "agent_call", side_effect=_fake_agent_call):
            self.fc.cmd_start(args, manifest)
        return calls

    def test_config_field_adds_overrides_flag(self, monkeypatch):
        workers = [{
            "name": "worker-<host>", "ssh_host": "user@x",
            "instances": 1, "config": {"gpu_enabled": True},
        }]
        calls = self._run_cmd_start(monkeypatch, workers)
        assert len(calls) == 1
        assert "--config-overrides-b64" in calls[0]
        idx = calls[0].index("--config-overrides-b64")
        encoded = calls[0][idx + 1]
        decoded = json.loads(base64.b64decode(encoded).decode("utf-8"))
        assert decoded == {"gpu_enabled": True}

    def test_no_config_field_no_overrides_flag(self, monkeypatch):
        workers = [{
            "name": "worker-HP", "ssh_host": "hp@x", "instances": 1,
        }]
        calls = self._run_cmd_start(monkeypatch, workers)
        assert len(calls) == 1
        assert "--config-overrides-b64" not in calls[0]

    def test_empty_config_dict_no_overrides_flag(self, monkeypatch):
        workers = [{
            "name": "worker-HP", "ssh_host": "hp@x", "instances": 1,
            "config": {},
        }]
        calls = self._run_cmd_start(monkeypatch, workers)
        assert "--config-overrides-b64" not in calls[0]

    def test_config_field_applies_to_multi_instance(self, monkeypatch):
        """Multi-instance host: every instance call gets the flag."""
        workers = [{
            "name": "worker-<host>", "ssh_host": "user@x",
            "instances": 3, "config": {"gpu_enabled": True},
        }]
        calls = self._run_cmd_start(monkeypatch, workers)
        assert len(calls) == 3
        for c in calls:
            assert "--config-overrides-b64" in c

    def test_config_field_applies_to_local_worker_no_ssh_host(self, monkeypatch):
        """Local worker (ssh_host=None) still gets the flag — the feature
        applies regardless of local/remote."""
        workers = [{
            "name": "worker-default", "ssh_host": None,
            "instances": 1, "config": {"gpu_enabled": True},
        }]
        calls = self._run_cmd_start(monkeypatch, workers)
        assert "--config-overrides-b64" in calls[0]
