"""Per-instance env/config overrides in ops/workers.json + bin/farmctl
(2026-07-25, farmctl-per-instance-config).

Unblocks a measured +20% throughput win on <host> (3 CPU instances + 1
OpenCL/Arc-iGPU instance = 3.21 batches/s vs 2.69 for 4 identical CPU
instances) by letting a single host run a MIX of env/config across its own
instances, instead of one identical override for all of them.

Design:
  - Manifest key: `instance_overrides`, a dict keyed by the 1-based instance
    INDEX as a JSON string ("1", "2", ...) — not the instance name. See
    bin/farmctl's `_instance_override` docstring for why index, not name.
  - Merge is per-KEY: host-level `env`/`config` is the base, the per-instance
    block overrides individual keys on top (dict.update semantics).
  - No `instance_overrides` block at all must produce byte-identical
    behaviour to before this feature existed.
  - An override key that matches no instance in a given phase resolution is
    not a silent no-op: `_iter_instances` warns to stderr every time it
    resolves that phase.
"""
from __future__ import annotations

from importlib.machinery import SourceFileLoader
from importlib.util import module_from_spec, spec_from_loader
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def farmctl():
    path = str(Path(__file__).resolve().parent.parent / "bin" / "farmctl")
    loader = SourceFileLoader("farmctl_mod", path)
    spec = spec_from_loader("farmctl_mod", loader)
    mod = module_from_spec(spec)
    loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Merge semantics: _merged_env / _merged_config
# ---------------------------------------------------------------------------

def test_no_instance_overrides_block_is_byte_identical_to_host_level(farmctl):
    """A worker with no `instance_overrides` key at all: _merged_env/_config
    must return exactly the host-level dict contents, for every index."""
    w = {"name": "worker-x", "env": {"A": "1", "B": "2"},
         "config": {"gpu_enabled": True}}

    for idx in (1, 2, 3):
        assert farmctl._merged_env(w, idx) == {"A": "1", "B": "2"}
        assert farmctl._merged_config(w, idx) == {"gpu_enabled": True}


def test_no_host_level_env_and_no_override_is_empty_dict(farmctl):
    w = {"name": "worker-x"}
    assert farmctl._merged_env(w, 1) == {}
    assert farmctl._merged_config(w, 1) == {}


def test_partial_key_override_keeps_untouched_keys(farmctl):
    """Per-instance override touches ONE key; the other host-level key
    must survive — this is the "not a wholesale replacement" contract."""
    w = {
        "name": "worker-x",
        "env": {"A": "1", "B": "2"},
        "instance_overrides": {"1": {"env": {"A": "3"}}},
    }

    assert farmctl._merged_env(w, 1) == {"A": "3", "B": "2"}
    # Index 2 has no override block at all -> host-level env untouched.
    assert farmctl._merged_env(w, 2) == {"A": "1", "B": "2"}


def test_partial_config_override_keeps_untouched_keys(farmctl):
    w = {
        "name": "worker-x",
        "config": {"gpu_enabled": True, "fpp_cpu_cores": 4},
        "instance_overrides": {"2": {"config": {"gpu_enabled": False}}},
    }

    assert farmctl._merged_config(w, 2) == {
        "gpu_enabled": False, "fpp_cpu_cores": 4}
    assert farmctl._merged_config(w, 1) == {
        "gpu_enabled": True, "fpp_cpu_cores": 4}


def test_override_can_add_a_key_not_present_at_host_level(farmctl):
    w = {
        "name": "worker-x",
        "instance_overrides": {"4": {"env": {"EXONIT_INJECTION_OPENCL": "1"}}},
    }
    assert farmctl._merged_env(w, 4) == {"EXONIT_INJECTION_OPENCL": "1"}
    assert farmctl._merged_env(w, 1) == {}


# ---------------------------------------------------------------------------
# Instance identity: index vs name, bare-name single-instance hosts
# ---------------------------------------------------------------------------

def test_single_instance_bare_name_host_keys_on_index_1(farmctl):
    """A host with no `instances` (or `instances: 1`) resolves under its bare
    name, not `<name>-1` — the override must still be reachable via index 1,
    not via the literal string "worker-HP"."""
    w = {
        "name": "worker-HP",
        "instance_overrides": {"1": {"env": {"EXONIT_WOTAN_BACKEND": "cpu"}}},
    }
    [(iname, idx, total)] = list(farmctl._iter_instances(w, phase="scan"))
    assert iname == "worker-HP"
    assert idx == 1
    assert farmctl._merged_env(w, idx) == {"EXONIT_WOTAN_BACKEND": "cpu"}


def test_index_from_instance_name_bare_and_suffixed(farmctl):
    w = {"name": "worker-HP"}
    assert farmctl._index_from_instance_name(w, "worker-HP") == 1
    w2 = {"name": "worker-default"}
    assert farmctl._index_from_instance_name(w2, "worker-default-3") == 3


# ---------------------------------------------------------------------------
# Orphan-key warning (constraint 3: never a silent no-op)
# ---------------------------------------------------------------------------

def test_orphan_override_key_warns_on_stderr(farmctl, capsys):
    """instances: 2, but instance_overrides declares index "3" — no
    instance 3 exists in this phase. Must warn, not silently no-op."""
    w = {
        "name": "worker-typo",
        "instances": 2,
        "instance_overrides": {"3": {"env": {"X": "1"}}},
    }
    list(farmctl._iter_instances(w, phase="scan"))
    captured = capsys.readouterr()
    assert "worker-typo" in captured.err
    assert "instance_overrides key '3'" in captured.err
    assert "no instance" in captured.err


def test_valid_key_does_not_warn(farmctl, capsys):
    w = {
        "name": "worker-ok",
        "instances": 2,
        "instance_overrides": {"1": {"env": {"X": "1"}}},
    }
    list(farmctl._iter_instances(w, phase="scan"))
    captured = capsys.readouterr()
    assert captured.err == ""


def test_non_integer_key_warns(farmctl, capsys):
    w = {
        "name": "worker-bad-key",
        "instances": 2,
        "instance_overrides": {"first": {"env": {"X": "1"}}},
    }
    list(farmctl._iter_instances(w, phase="scan"))
    captured = capsys.readouterr()
    assert "not an integer instance index" in captured.err


# ---------------------------------------------------------------------------
# Phase awareness (constraint 4): scan-vs-fpp sizing
# ---------------------------------------------------------------------------

def test_key_valid_in_scan_orphaned_in_smaller_fpp_sizing(farmctl, capsys):
    """instances: 4 (scan), instances_fpp: 2 — an override for index 4 is a
    real instance during scan but orphaned during fpp. Both facts are true
    at once and neither resolution is wrong; fpp resolution warns."""
    w = {
        "name": "worker-mixed-phase",
        "instances": 4,
        "instances_fpp": 2,
        "instance_overrides": {"4": {"env": {"X": "1"}}},
    }

    list(farmctl._iter_instances(w, phase="scan"))
    assert capsys.readouterr().err == ""  # valid for scan (4 instances)

    list(farmctl._iter_instances(w, phase="fpp"))
    err = capsys.readouterr().err
    assert "worker-mixed-phase" in err
    assert "instance_overrides key '4'" in err

    # And the override itself is only reachable at index 4, which only scan
    # phase actually starts:
    scan_names = [i for i, _x, _y in farmctl._iter_instances(w, phase="scan")]
    fpp_names = [i for i, _x, _y in farmctl._iter_instances(w, phase="fpp")]
    assert scan_names == [f"worker-mixed-phase-{i}" for i in range(1, 5)]
    assert fpp_names == ["worker-mixed-phase-1", "worker-mixed-phase-2"]
    assert farmctl._merged_env(w, 4) == {"X": "1"}  # reachable, just unused by fpp


def test_key_orphaned_in_scan_valid_in_larger_fpp_sizing(farmctl, capsys):
    """The inverse: instances: 2 (scan), instances_fpp: 4 — index 4 is real
    only once the host scales up for fpp."""
    w = {
        "name": "worker-scale-up",
        "instances": 2,
        "instances_fpp": 4,
        "instance_overrides": {"4": {"env": {"X": "1"}}},
    }

    list(farmctl._iter_instances(w, phase="scan"))
    err_scan = capsys.readouterr().err
    assert "worker-scale-up" in err_scan
    assert "instance_overrides key '4'" in err_scan

    list(farmctl._iter_instances(w, phase="fpp"))
    assert capsys.readouterr().err == ""


# ---------------------------------------------------------------------------
# Merge with EXISTING host-level env/config: config-overrides-per-host feature
# is not broken by the new per-instance layer (uses the real manifest).
# ---------------------------------------------------------------------------

def test_real_manifest_nvidiarack_host_level_env_untouched_by_absent_overrides(farmctl):
    """worker-<host> has host-level env/config and NO instance_overrides
    — must be untouched (regression guard for the merge helper itself)."""
    manifest = farmctl.load_manifest()
    by_name = {w["name"]: w for w in manifest["workers"]}
    w = by_name["worker-<host>"]

    assert farmctl._merged_env(w, 1) == w["env"]
    assert farmctl._merged_env(w, 7) == w["env"]
    assert farmctl._merged_config(w, 1) == w["config"]


# ---------------------------------------------------------------------------
# Exact command strings for worker-default's 4 instances (the deployed
# artefact) — this is what decides whether the OpenCL config is actually
# correct on the wire.
# ---------------------------------------------------------------------------

def _build_start_cmd(farmctl, monkeypatch, w, iname, idx, total):
    """Reproduce cmd_start's per-instance agent_args + agent_call exactly,
    capturing the shell command ssh_exec would have run, without running it."""
    captured = {}

    def fake_ssh_exec(worker, command, timeout=30):
        captured["command"] = command
        return 0, "", ""

    monkeypatch.setattr(farmctl, "ssh_exec", fake_ssh_exec)

    if w["ssh_host"] is None:
        agent_args = ["start", iname]
    else:
        agent_args = ["start", iname, "--server-url", "http://<server-ip>:8400"]
    if total > 1:
        agent_args += ["--instances", str(total), "--instance-index", str(idx)]
    config_overrides = farmctl._merged_config(w, idx)
    if isinstance(config_overrides, dict) and config_overrides:
        agent_args += ["--config-overrides-b64",
                       farmctl._encode_config_overrides_b64(config_overrides)]
    agent_args.append("--allow-start")
    farmctl.agent_call(w, *agent_args, instance_index=idx, timeout=30)
    return captured["command"]


def test_worker_default_four_instances_exact_command_strings(farmctl, monkeypatch):
    manifest = farmctl.load_manifest()
    by_name = {w["name"]: w for w in manifest["workers"]}
    w = by_name["worker-default"]
    assert w["instances"] == 4

    py = w["python"]
    agent_path = w["worker_dir"] + "/agent.py"

    # The Arc instance is index 1 since 2026-07-27 (it was index 4 before):
    # instances_fpp: 1 keeps ONLY that instance during FPP_BACKFILL, and
    # farmctl sizes phases as "instances 1..N", so the GPU one has to be
    # first or the FPP prune would stop precisely the instance we want kept.
    # It carries BOTH the env var and config.gpu_enabled=true (b64-encoded as
    # --config-overrides-b64): worker.py refuses the GPU path on the config
    # flag alone, so pinning only the env var here would let a silent
    # fallback-to-CPU regression through (2026-07-25 correction).
    gpu_b64 = farmctl._encode_config_overrides_b64({"gpu_enabled": True})
    expected = {
        1: (f'EXONIT_INJECTION_OPENCL=1 {py} {agent_path} start worker-default-1 '
            f'--instances 4 --instance-index 1 '
            f'--config-overrides-b64 {gpu_b64} --allow-start'),
        2: (f'EXONIT_INJECTION_OPENCL=0 {py} {agent_path} start worker-default-2 '
            f'--instances 4 --instance-index 2 --allow-start'),
        3: (f'EXONIT_INJECTION_OPENCL=0 {py} {agent_path} start worker-default-3 '
            f'--instances 4 --instance-index 3 --allow-start'),
        4: (f'EXONIT_INJECTION_OPENCL=0 {py} {agent_path} start worker-default-4 '
            f'--instances 4 --instance-index 4 --allow-start'),
    }

    for iname, idx, total in farmctl._iter_instances(w, phase="scan"):
        cmd = _build_start_cmd(farmctl, monkeypatch, w, iname, idx, total)
        assert cmd == expected[idx], (
            f"instance {idx} command mismatch:\n  got:      {cmd}\n"
            f"  expected: {expected[idx]}")
