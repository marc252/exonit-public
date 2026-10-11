import json

import pytest

from autonomy.fleet_health import (
    parse_df_kb,
    parse_vm_stat,
    parse_free_m,
    parse_wmic,
    parse_sensors,
    evaluate,
    probe_host,
    collect,
    main,
)
from autonomy.types import HostHealth
from autonomy.config import Thresholds

_FLEET = {
    "worker-default": "local",
    "worker-win-1": "user@win-host-51",
    "worker-win-2": "user@win-host-47",
    "worker-mac-1": "user@mac-host-1",
    "worker-linux-gpu": "user@linux-gpu-host",
}
_PLATFORMS = {"worker-default": "linux", "worker-win-1": "windows", "worker-win-2": "windows",
              "worker-mac-1": "mac", "worker-linux-gpu": "linux"}


@pytest.fixture(autouse=True)
def _neutral_fleet(monkeypatch):
    """Neutral fixture fleet (the real one is deployment config, ops/autonomy_fleet.json)."""
    monkeypatch.setattr("autonomy.fleet_health.FLEET_WORKERS", _FLEET)
    monkeypatch.setattr("autonomy.fleet_health.FLEET_PLATFORMS", _PLATFORMS)


DF = "Filesystem 1K-blocks Used Available Use% Mounted on\n/dev/sda1 500000000 400000000 100000000 80% /\n"
VMSTAT = "Mach Virtual Memory Statistics: (page size of 16384 bytes)\nPages free: 100000.\nPages inactive: 50000.\n"
FREE = "               total        used        free      shared  buff/cache   available\nMem:           31900       20000        2000         100        9900       11000\n"
WMIC = "FreeSpace=100000000000\nSize=500000000000\n\nFreePhysicalMemory=8000000\n"
SENSORS = "coretemp-isa-0000\nPackage id 0:  +45.0\xb0C  (high = +100.0\xb0C, crit = +100.0\xb0C)\n"


def test_parse_df():
    free_gb, pct = parse_df_kb(DF)
    assert round(free_gb) == 95 and pct == 80.0


def test_parse_vm_stat():
    assert round(parse_vm_stat(VMSTAT, memsize=32_000_000_000), 1) == round((150000 * 16384) / 2**30, 1)


def test_parse_free():
    assert round(parse_free_m(FREE), 1) == round(11000 / 1024, 1)


def test_parse_wmic():
    disk_free, _, ram = parse_wmic(WMIC)
    assert round(disk_free) == 93 and round(ram, 1) == round(8000000 / 1024 / 1024, 1)


def test_parse_sensors_first_match():
    assert parse_sensors(SENSORS) == 45.0


def test_parse_sensors_no_match_returns_none():
    assert parse_sensors("no temp data here\n") is None


def test_probe_host_failure_is_contained():
    def boom(target, cmd):
        raise TimeoutError("ssh timeout")

    h = probe_host("worker-mac-1", "user@mac-host-1", runner=boom)
    assert h.reachable is False and "timeout" in h.error


def test_probe_host_linux_ok():
    text = DF + FREE

    def fake(target, cmd):
        return text

    h = probe_host("worker-linux-gpu", "user@linux-gpu-host", runner=fake)
    assert h.reachable is True
    assert round(h.disk_free_gb) == 95
    assert h.disk_pct_used == 80.0
    assert round(h.ram_avail_gb, 1) == round(11000 / 1024, 1)


def test_probe_host_windows_ok():
    def fake(target, cmd):
        return WMIC

    h = probe_host("worker-win-1", "user@win-host-51", runner=fake)
    assert h.reachable is True
    assert round(h.disk_free_gb) == 93


def test_evaluate_mac_disk_crit_suggests_clean_cache():
    h = HostHealth("worker-mac-1", "x", True, 3.0, 99.0, 8.0, None, None)
    advs = evaluate([h], Thresholds())
    crit = [a for a in advs if a.kind == "disk_crit"]
    assert crit and crit[0].suggested_playbook == "clean_fits_cache"


def test_evaluate_disk_warn_no_playbook():
    h = HostHealth("worker-linux-gpu", "x", True, 20.0, 90.0, 8.0, None, None)
    advs = evaluate([h], Thresholds())
    warn = [a for a in advs if a.kind == "disk_warn"]
    assert warn and warn[0].severity == "warn" and warn[0].suggested_playbook == ""


def test_evaluate_ram_crit_suggests_restart_worker():
    h = HostHealth("worker-mac-1", "x", True, 100.0, 10.0, 1.0, None, None)
    advs = evaluate([h], Thresholds())
    crit = [a for a in advs if a.kind == "ram_crit"]
    assert crit and crit[0].suggested_playbook == "restart_worker"
    assert crit[0].params == {"worker": "worker-mac-1"}


def test_evaluate_temp_warn_no_playbook():
    h = HostHealth("worker-linux-gpu", "x", True, 100.0, 10.0, 8.0, 99.0, None)
    advs = evaluate([h], Thresholds())
    warn = [a for a in advs if a.kind == "temp_warn"]
    assert warn and warn[0].severity == "warn" and warn[0].suggested_playbook == ""


def test_evaluate_unreachable_no_playbook():
    h = HostHealth("worker-win-2", "x", False, error="ssh timeout")
    advs = evaluate([h], Thresholds())
    unreach = [a for a in advs if a.kind == "unreachable"]
    assert unreach and unreach[0].suggested_playbook == ""


def test_evaluate_healthy_host_no_advisories():
    h = HostHealth("worker-linux-gpu", "x", True, 100.0, 10.0, 16.0, 40.0, None)
    advs = evaluate([h], Thresholds())
    assert advs == []


def test_collect_tolerates_per_host_failure():
    def flaky(target, cmd):
        if "win-host-51" in target:
            raise TimeoutError("ssh timeout")
        return WMIC if "win-host-47" in target else DF + FREE

    healths = collect(runner=flaky)
    by_name = {h.worker_name: h for h in healths}
    assert by_name["worker-win-1"].reachable is False
    assert by_name["worker-win-2"].reachable is True
    assert len(healths) == len(_FLEET)  # all the FLEET_WORKERS entries


def test_main_writes_json_output(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "autonomy.fleet_health.collect",
        lambda runner=None: [HostHealth("worker-linux-gpu", "x", True, 100.0, 10.0, 16.0, 40.0, None)],
    )

    data_dir = tmp_path / "data" / "autonomy"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(f"data_dir: {data_dir}\n")

    main(["--config", str(config_path)])

    out_path = data_dir / "fleet_health.json"
    assert out_path.exists()
    payload = json.loads(out_path.read_text())
    assert "ts" in payload and "hosts" in payload and "advisories" in payload
    assert payload["hosts"][0]["worker_name"] == "worker-linux-gpu"
