"""farmctl start --phase sizes the fleet BOTH ways (2026-07-14).

Before this, `start --phase fpp` was purely additive: it started the FPP-sized
instances and left everything else running. So the slow hosts (the M1 mac mini,
the HP laptop, a laptop worker) kept taking FPP work all through FPP_BACKFILL —
and a slow host on FPP does not speed the queue up, it seeds the straggler tail
that the entire run then waits on (run s0014/lot20260714T175216: 16 of 34 FPP
minutes spent on the last 31 candidates).

`instances_fpp: 0` now means "this host sits the phase out", and cmd_start
enforces it by stopping the instances the previous phase left behind.
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


def test_slow_host_is_pruned_out_of_the_fpp_phase(farmctl):
    """instances: 2 / instances_fpp: 0 — the real shape of worker-<host> (M1),
    worker-<host> and worker-HP after this change."""
    w = {"name": "worker-<host>", "instances": 2, "instances_fpp": 0}

    assert [i for i, _x, _y in farmctl._iter_instances(w, phase="fpp")] == []
    assert farmctl._instances_out_of_phase(w, "fpp") == [
        "worker-<host>-1", "worker-<host>-2"]
    # ...and in its own phase it is left completely alone.
    assert farmctl._instances_out_of_phase(w, "scan") == []


def test_single_instance_slow_host_is_pruned_by_its_bare_name(farmctl):
    """worker-HP declares no `instances` at all (defaults to 1), so it runs
    under its bare name, not `-1`."""
    w = {"name": "worker-HP", "instances_fpp": 0}

    assert farmctl._instances_out_of_phase(w, "fpp") == ["worker-HP"]


def test_scaling_up_prunes_nothing(farmctl):
    """<host>: 10 scan instances, 19 for FPP. Nothing to stop — the scan
    names are a subset of the FPP names."""
    w = {"name": "worker-<host>", "instances": 10, "instances_fpp": 19}

    assert farmctl._instances_out_of_phase(w, "fpp") == []
    # Coming back DOWN to scan sizing does stop the extra nine.
    assert farmctl._instances_out_of_phase(w, "scan") == [
        f"worker-<host>-{i}" for i in range(11, 20)]


def test_disabled_worker_is_never_touched_in_any_phase(farmctl):
    """`instances: 0` (worker-default on <host>: it must never compete with
    the prod server) means disabled everywhere — it is not started, and it is
    not "pruned" either, because it owns no instance name at all."""
    w = {"name": "worker-default", "instances": 0, "instances_fpp": 5}

    assert list(farmctl._iter_instances(w, phase="fpp")) == []
    assert farmctl._instances_out_of_phase(w, "fpp") == []
    assert farmctl._instances_out_of_phase(w, "scan") == []


def test_host_without_fpp_sizing_still_falls_back_to_instances(farmctl):
    """Absent instances_fpp = "does not scale for FPP", NOT "sits FPP out".
    The distinction matters: absent must keep the old fallback behaviour."""
    w = {"name": "worker-x", "instances": 3}

    assert [i for i, _x, _y in farmctl._iter_instances(w, phase="fpp")] == [
        "worker-x-1", "worker-x-2", "worker-x-3"]
    assert farmctl._instances_out_of_phase(w, "fpp") == []


def test_real_manifest_keeps_the_three_slow_hosts_out_of_fpp(farmctl):
    """The manifest itself is the operator-visible contract — assert on it, not
    just on the helper."""
    manifest = farmctl.load_manifest()
    by_name = {w["name"]: w for w in manifest["workers"]}

    for name in ("worker-<host>", "worker-<host>", "worker-HP"):
        w = by_name[name]
        assert w.get("instances_fpp") == 0, f"{name} must sit FPP out"
        assert list(farmctl._iter_instances(w, phase="fpp")) == []
        assert farmctl._instances_out_of_phase(w, "fpp"), (
            f"{name} must be STOPPED when the fleet scales to fpp, not just "
            f"left out of the start list")
        # It still scans — that is where a slow host earns its keep.
        assert list(farmctl._iter_instances(w, phase="scan"))
