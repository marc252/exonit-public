"""farmctl `sidecar_fpp` manifest field (2026-08-07, python-fpp-sidecar).

The Rust worker has no TRICERATOPS/FPP implementation
(`WorkerRuntimeError::BackfillPoolsUnimplemented`), so a host running it
needed a SEPARATE manifest entry running the Python worker just for the
`fpp` phase. `sidecar_fpp: true` on a Rust-worker manifest entry collapses
this into ONE entry: `farmctl start --phase fpp` issues a single
`agent start <name> --fpp-instances N --allow-start` call to the Rust
agent (which spawns N Python `worker.py` sidecars itself — see
`exonit-rust-worker`'s `docs/RUNBOOK_PYTHON_FPP_SIDECAR.md`) instead of
farmctl's own per-instance `<name>-i` loop. See `bin/farmctl`'s
`_iter_instances` (the `sidecar_fpp` branch) and `_start_sidecar_fpp`.

A manifest entry with NO `sidecar_fpp` key is byte-identical to before this
feature — covered by the existing `test_farmctl_phase_prune.py` /
`test_farmctl_instance_overrides.py` suites, not re-asserted here.
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


def test_sidecar_fpp_yields_bare_name_only_in_fpp_phase(farmctl):
    """8 native scan instances, 17 FPP sidecars — the fpp resolution must be
    ONE tuple for the bare name, not a 17-way `<name>-i` split."""
    w = {"name": "worker-<host>-rust", "instances": 8,
         "instances_fpp": 17, "sidecar_fpp": True}

    assert list(farmctl._iter_instances(w, phase="fpp")) == [
        ("worker-<host>-rust", 1, 1)]
    # scan phase is completely untouched by the sidecar_fpp flag.
    assert [i for i, _x, _y in farmctl._iter_instances(w, phase="scan")] == [
        f"worker-<host>-rust-{i}" for i in range(1, 9)]


def test_sidecar_fpp_instances_fpp_zero_sits_fpp_out(farmctl):
    """instances_fpp: 0 still means "sits FPP out entirely", same as any
    other host — sidecar_fpp does not change that check."""
    w = {"name": "worker-x-rust", "instances": 4,
         "instances_fpp": 0, "sidecar_fpp": True}

    assert list(farmctl._iter_instances(w, phase="fpp")) == []


def test_sidecar_fpp_transition_prunes_all_native_scan_instances(farmctl):
    """Starting the fpp phase on a sidecar_fpp host must prune EVERY native
    scan instance (the Rust worker has no FPP capability at all) — mirrors
    the pre-existing `instances_fpp: 0` convention for Rust-native hosts,
    but derived automatically instead of needing a separate manifest field."""
    w = {"name": "worker-<host>-rust", "instances": 8,
         "instances_fpp": 17, "sidecar_fpp": True}

    assert farmctl._instances_out_of_phase(w, "fpp") == [
        f"worker-<host>-rust-{i}" for i in range(1, 9)]
    # ...and the sidecar entity itself is untouched by an fpp-phase start.
    assert "worker-<host>-rust" not in farmctl._instances_out_of_phase(w, "fpp")


def test_scan_transition_prunes_the_sidecar_entity(farmctl):
    """Starting the scan phase must stop any sidecars a previous fpp phase
    left running (the bare name)."""
    w = {"name": "worker-<host>-rust", "instances": 8,
         "instances_fpp": 17, "sidecar_fpp": True}

    assert farmctl._instances_out_of_phase(w, "scan") == [
        "worker-<host>-rust"]


def test_stop_covers_both_native_instances_and_the_sidecar_entity(farmctl):
    """`farmctl stop` (phase-agnostic — _iter_instances_all_phases) must
    cover every native scan instance name AND the bare sidecar name, so a
    stop always fully tears down whichever phase was last running."""
    w = {"name": "worker-<host>-rust", "instances": 8,
         "instances_fpp": 17, "sidecar_fpp": True}

    names = {iname for iname, _i, _t in farmctl._iter_instances_all_phases(w)}
    assert names == {f"worker-<host>-rust-{i}" for i in range(1, 9)} | {
        "worker-<host>-rust"}


def _capture_ssh(farmctl, monkeypatch):
    captured = []

    def fake_ssh_exec(worker, command, timeout=30):
        captured.append((command, timeout))
        return 0, '{"status": "started_sidecars"}', ""

    monkeypatch.setattr(farmctl, "ssh_exec", fake_ssh_exec)
    return captured


def test_start_sidecar_fpp_builds_the_single_fpp_instances_command(farmctl, monkeypatch):
    captured = _capture_ssh(farmctl, monkeypatch)
    w = {"name": "worker-<host>-rust", "ssh_host": "user@<postgres-host>",
         "platform": "linux", "worker_dir": "~/exonit-rust-run",
         "python": "~/exonit-rust-worker/target/release/exonit-rust-worker",
         "instances": 8, "instances_fpp": 17, "sidecar_fpp": True}

    ok = farmctl._start_sidecar_fpp(w, "http://<server-ip>:8400")

    assert ok is True
    assert len(captured) == 1
    command, timeout = captured[0]
    assert command == (
        f'{w["python"]} {w["worker_dir"]}/agent.py start worker-<host>-rust '
        f'--fpp-instances 17 --server-url http://<server-ip>:8400 --allow-start')
    assert timeout == 45


def test_start_sidecar_fpp_forwards_config_overrides(farmctl, monkeypatch):
    captured = _capture_ssh(farmctl, monkeypatch)
    w = {"name": "worker-<host>-rust", "ssh_host": "user@<postgres-host>",
         "platform": "linux", "worker_dir": "~/exonit-rust-run",
         "python": "~/exonit-rust-worker/target/release/exonit-rust-worker",
         "instances": 8, "instances_fpp": 17, "sidecar_fpp": True,
         "config": {"gpu_enabled": True}}

    farmctl._start_sidecar_fpp(w, "http://<server-ip>:8400")

    command, _timeout = captured[0]
    b64 = farmctl._encode_config_overrides_b64({"gpu_enabled": True})
    assert f"--config-overrides-b64 {b64}" in command


def test_start_sidecar_fpp_falls_back_to_instances_when_instances_fpp_absent(farmctl, monkeypatch):
    captured = _capture_ssh(farmctl, monkeypatch)
    w = {"name": "worker-x-rust", "ssh_host": "user@<lan-ip>",
         "platform": "linux", "worker_dir": "~/exonit-rust-run",
         "python": "~/exonit-rust-worker/target/release/exonit-rust-worker",
         "instances": 5, "sidecar_fpp": True}

    farmctl._start_sidecar_fpp(w, "http://<server-ip>:8400")

    command, _timeout = captured[0]
    assert "--fpp-instances 5" in command


def test_start_sidecar_fpp_zero_instances_fpp_is_a_noop(farmctl, monkeypatch):
    captured = _capture_ssh(farmctl, monkeypatch)
    w = {"name": "worker-x-rust", "ssh_host": "user@<lan-ip>",
         "platform": "linux", "worker_dir": "~/exonit-rust-run",
         "python": "~/exonit-rust-worker/target/release/exonit-rust-worker",
         "instances": 5, "instances_fpp": 0, "sidecar_fpp": True}

    ok = farmctl._start_sidecar_fpp(w, "http://<server-ip>:8400")

    assert ok is True
    assert captured == [], "instances_fpp: 0 must not call agent.py at all"


def test_start_sidecar_fpp_reports_failure(farmctl, monkeypatch):
    def fake_ssh_exec(worker, command, timeout=30):
        return 1, "", "boom"

    monkeypatch.setattr(farmctl, "ssh_exec", fake_ssh_exec)
    w = {"name": "worker-x-rust", "ssh_host": "user@<lan-ip>",
         "platform": "linux", "worker_dir": "~/exonit-rust-run",
         "python": "~/exonit-rust-worker/target/release/exonit-rust-worker",
         "instances": 5, "instances_fpp": 5, "sidecar_fpp": True}

    ok = farmctl._start_sidecar_fpp(w, "http://<server-ip>:8400")

    assert ok is False


def test_cmd_start_fpp_phase_calls_sidecar_helper_not_per_instance_loop(farmctl, monkeypatch):
    """End-to-end through cmd_start: --phase fpp on a sidecar_fpp worker must
    issue exactly ONE agent.py invocation (the --fpp-instances call), not
    17 per-instance ones, and must NOT be gated by --no-prune (prune still
    runs first, targeting the native scan instances)."""
    monkeypatch.setenv("EXONIT_FARMCTL_ALLOW_START", "1")
    calls = []

    def fake_agent_call(worker, *args, instance_index=1, timeout=30):
        calls.append(list(args))
        return 0, '{"status": "ok"}', ""

    monkeypatch.setattr(farmctl, "agent_call", fake_agent_call)

    manifest = {
        "server_url": "http://<server-ip>:8400",
        "workers": [{
            "name": "worker-<host>-rust", "ssh_host": "user@<postgres-host>",
            "platform": "linux", "worker_dir": "~/exonit-rust-run",
            "python": "~/exonit-rust-worker/target/release/exonit-rust-worker",
            "instances": 8, "instances_fpp": 17, "sidecar_fpp": True,
        }],
    }

    class Args:
        workers = []
        all = True
        phase = "fpp"
        no_prune = True  # isolate the start half from the prune SSH calls

    rc = farmctl.cmd_start(Args(), manifest)

    assert rc == 0
    start_calls = [c for c in calls if c and c[0] == "start"]
    assert len(start_calls) == 1, f"expected exactly one start call, got {start_calls}"
    assert start_calls[0][:2] == ["start", "worker-<host>-rust"]
    assert "--fpp-instances" in start_calls[0]
    assert start_calls[0][start_calls[0].index("--fpp-instances") + 1] == "17"


# --- _expected_online_names (fleet-verify resolution, 2026-08-07) ----------

def test_expected_online_names_sidecar_fpp_yields_py_names(farmctl):
    w = {"name": "worker-X-rust", "instances": 8, "instances_fpp": 17,
         "sidecar_fpp": True, "ssh_host": "u@h", "platform": "linux"}
    assert farmctl._expected_online_names(w, phase="fpp") == {
        f"worker-X-rust-py-{i}" for i in range(1, 18)}


def test_expected_online_names_sidecar_scan_matches_iter_instances(farmctl):
    w = {"name": "worker-X-rust", "instances": 3, "instances_fpp": 5,
         "sidecar_fpp": True, "ssh_host": "u@h", "platform": "linux"}
    assert farmctl._expected_online_names(w, phase="scan") == {
        iname for iname, _i, _t in farmctl._iter_instances(w, phase="scan")}


def test_expected_online_names_non_sidecar_matches_iter_instances(farmctl):
    w = {"name": "worker-Y", "instances": 0, "instances_fpp": 3,
         "ssh_host": "u@h", "platform": "linux"}
    for phase in ("scan", "fpp"):
        assert farmctl._expected_online_names(w, phase=phase) == {
            iname for iname, _i, _t in farmctl._iter_instances(w, phase=phase)}


def test_expected_online_names_sidecar_fpp_zero_is_empty(farmctl):
    w = {"name": "worker-Z-rust", "instances": 2, "instances_fpp": 0,
         "sidecar_fpp": True, "ssh_host": "u@h", "platform": "linux"}
    assert farmctl._expected_online_names(w, phase="fpp") == set()
