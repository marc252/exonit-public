"""Fleet health probe over read-only SSH (phase 1 — shadow mode).

`collect()` walks `FLEET_WORKERS` (playbooks.py) and runs, per host, a
single READ command (disk + RAM + temperature if available) via SSH
(`BatchMode=yes`, `ConnectTimeout=10`) or locally for `worker-default`.
No remote writes, no kill, no restart — read only. `evaluate()`
compares the results against the `Thresholds` from config.py and produces
`Advisory` items (some with a suggested playbook from the closed catalog in
playbooks.py, others informational only in v1).

The parsers (`parse_df_kb`, `parse_free_m`, `parse_vm_stat`, `parse_wmic`,
`parse_sensors`) are pure functions over text — tested with real captured
outputs, with no SSH in the tests. The only real SSH/local execution
happens in `_run_ssh`, injectable into `probe_host`/`collect` via the
`runner` parameter.

CLI: `python -m autonomy.fleet_health [--json]` — collect+evaluate, writes
`data/autonomy/fleet_health.json` and prints a short table.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

from autonomy.config import REPO_ROOT, Thresholds, load_config
from autonomy.playbooks import FLEET_PLATFORMS, FLEET_WORKERS, is_mac_host
from autonomy.types import Advisory, HostHealth

_LOG = logging.getLogger("autonomy.fleet_health")

# READ commands per platform (no writes, no kill/restart).
_COMMANDS: dict[str, str] = {
    "linux": "df -k /home; free -m; command -v sensors >/dev/null && sensors 2>/dev/null || true",
    "mac": "df -k /; vm_stat; sysctl -n hw.memsize",
    "windows": (
        'wmic logicaldisk where "DeviceID=\'C:\'" get FreeSpace,Size /value '
        "& wmic OS get FreePhysicalMemory /value"
    ),
}

_DF_LINE_RE = re.compile(r"^\S+\s+\d+\s+\d+\s+(\d+)\s+(\d+)%\s+\S+", re.MULTILINE)
_FREE_LINE_RE = re.compile(r"^Mem:\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)", re.MULTILINE)
_VM_STAT_PAGESIZE_RE = re.compile(r"page size of (\d+) bytes")
_VM_STAT_FREE_RE = re.compile(r"Pages free:\s*(\d+)")
_VM_STAT_INACTIVE_RE = re.compile(r"Pages inactive:\s*(\d+)")
_STANDALONE_NUMBER_RE = re.compile(r"^\s*(\d+)\s*$", re.MULTILINE)
_WMIC_FREESPACE_RE = re.compile(r"FreeSpace=(\d+)")
_WMIC_SIZE_RE = re.compile(r"Size=(\d+)")
_WMIC_RAM_RE = re.compile(r"FreePhysicalMemory=(\d+)")
_SENSORS_TEMP_RE = re.compile(r"\+([0-9.]+)\xb0C")


def parse_df_kb(text: str) -> tuple[Optional[float], Optional[float]]:
    """Parse a line of `df -k` output (linux or mac). Returns (free_gb, pct_used)."""
    m = _DF_LINE_RE.search(text)
    if not m:
        return None, None
    avail_kb = int(m.group(1))
    pct_used = float(m.group(2))
    return avail_kb / 1024 / 1024, pct_used


def parse_free_m(text: str) -> Optional[float]:
    """Parse the `Mem:` line of `free -m`. Returns ram_avail_gb ('available' column)."""
    m = _FREE_LINE_RE.search(text)
    if not m:
        return None
    available_mb = int(m.group(6))
    return available_mb / 1024


def parse_vm_stat(text: str, memsize: int) -> Optional[float]:
    """Parse `vm_stat` (mac). ram_avail_gb = (Pages free + Pages inactive) x pagesize.

    `memsize` (bytes, from `sysctl -n hw.memsize`) is only used as a
    safety cap (we never return more available RAM than the total).
    """
    free_m = _VM_STAT_FREE_RE.search(text)
    inactive_m = _VM_STAT_INACTIVE_RE.search(text)
    if not free_m or not inactive_m:
        return None
    pagesize_m = _VM_STAT_PAGESIZE_RE.search(text)
    pagesize = int(pagesize_m.group(1)) if pagesize_m else 4096

    avail_bytes = (int(free_m.group(1)) + int(inactive_m.group(1))) * pagesize
    if memsize:
        avail_bytes = min(avail_bytes, memsize)
    return avail_bytes / 2**30


def parse_wmic(text: str) -> tuple[Optional[float], Optional[float], Optional[float]]:
    """Parse `wmic` output (windows). Returns (disk_free_gb, pct_used|None, ram_avail_gb)."""
    free_m = _WMIC_FREESPACE_RE.search(text)
    ram_m = _WMIC_RAM_RE.search(text)
    if not free_m or not ram_m:
        return None, None, None

    free_bytes = int(free_m.group(1))
    disk_free_gb = free_bytes / 1024**3

    pct_used = None
    size_m = _WMIC_SIZE_RE.search(text)
    if size_m:
        size_bytes = int(size_m.group(1))
        if size_bytes > 0:
            pct_used = (size_bytes - free_bytes) / size_bytes * 100

    ram_kb = int(ram_m.group(1))
    ram_avail_gb = ram_kb / 1024 / 1024
    return disk_free_gb, pct_used, ram_avail_gb


def parse_sensors(text: str) -> Optional[float]:
    """First temperature match (`+NN.N°C`) in `sensors` output (linux). None if there is none."""
    m = _SENSORS_TEMP_RE.search(text)
    if not m:
        return None
    return float(m.group(1))


def _parse_memsize(text: str) -> int:
    """Extract `sysctl -n hw.memsize` (a digits-only line) from the combined mac text."""
    matches = _STANDALONE_NUMBER_RE.findall(text)
    return int(matches[-1]) if matches else 0


def _run_ssh(target: str, cmd: str) -> str:
    """Run `cmd` over read-only SSH (or locally for `worker-default`). Never writes anything remotely."""
    if target == "local":
        argv = ["bash", "-lc", cmd]
    else:
        argv = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", target, cmd]

    proc = subprocess.run(argv, capture_output=True, text=True, timeout=30)
    if proc.returncode != 0:
        raise RuntimeError(f"command failed (rc={proc.returncode}): {proc.stderr.strip()[:500]}")
    return proc.stdout


def probe_host(
    worker_name: str,
    target: str,
    runner: Callable[[str, str], str] = _run_ssh,
) -> HostHealth:
    """Probe a host: run its platform's command and parse the result.

    Never raises — any failure (timeout, SSH error, parse) is translated to
    `HostHealth(reachable=False, error=...)`.
    """
    platform = FLEET_PLATFORMS.get(worker_name, "linux")
    cmd = _COMMANDS[platform]

    try:
        text = runner(target, cmd)
    except Exception as exc:  # noqa: BLE001 - any probe failure is contained here
        return HostHealth(worker_name=worker_name, host=target, reachable=False, error=str(exc))

    try:
        temp_c = None
        if platform == "linux":
            disk_free_gb, pct_used = parse_df_kb(text)
            ram_avail_gb = parse_free_m(text)
            temp_c = parse_sensors(text)
        elif platform == "mac":
            disk_free_gb, pct_used = parse_df_kb(text)
            memsize = _parse_memsize(text)
            ram_avail_gb = parse_vm_stat(text, memsize)
        elif platform == "windows":
            disk_free_gb, pct_used, ram_avail_gb = parse_wmic(text)
        else:
            disk_free_gb = pct_used = ram_avail_gb = None
    except Exception as exc:  # noqa: BLE001 - defensive parse, never propagate an exception upwards
        return HostHealth(worker_name=worker_name, host=target, reachable=False, error=f"parse error: {exc!r}")

    return HostHealth(
        worker_name=worker_name,
        host=target,
        reachable=True,
        disk_free_gb=disk_free_gb,
        disk_pct_used=pct_used,
        ram_avail_gb=ram_avail_gb,
        temp_c=temp_c,
        error=None,
    )


def collect(runner: Callable[[str, str], str] = _run_ssh) -> list[HostHealth]:
    """Probe the whole fleet (`FLEET_WORKERS`) sequentially, tolerating per-host failures."""
    results = []
    for worker_name, target in FLEET_WORKERS.items():
        try:
            results.append(probe_host(worker_name, target, runner=runner))
        except Exception as exc:  # noqa: BLE001 - probe_host should already contain this, double safety net
            results.append(HostHealth(worker_name=worker_name, host=target, reachable=False, error=str(exc)))
    return results


def _is_mac_host(worker_name: str) -> bool:
    return is_mac_host(worker_name)


def evaluate(healths: list[HostHealth], th: Thresholds) -> list[Advisory]:
    """Generate `Advisory` items from the `HostHealth` entries outside thresholds (`Thresholds`)."""
    advisories: list[Advisory] = []

    for h in healths:
        if not h.reachable:
            advisories.append(
                Advisory(
                    worker_name=h.worker_name,
                    kind="unreachable",
                    severity="warn",
                    suggested_playbook="",
                    params={},
                    detail=f"{h.worker_name} unreachable: {h.error}",
                )
            )
            continue

        if h.disk_free_gb is not None:
            if h.disk_free_gb < th.disk_free_gb_crit:
                is_mac = _is_mac_host(h.worker_name)
                advisories.append(
                    Advisory(
                        worker_name=h.worker_name,
                        kind="disk_crit",
                        severity="crit",
                        suggested_playbook="clean_fits_cache" if is_mac else "",
                        params={"host": h.worker_name} if is_mac else {},
                        detail=f"disk_free_gb={h.disk_free_gb:.1f} < crit={th.disk_free_gb_crit}",
                    )
                )
            elif h.disk_free_gb < th.disk_free_gb_warn:
                advisories.append(
                    Advisory(
                        worker_name=h.worker_name,
                        kind="disk_warn",
                        severity="warn",
                        suggested_playbook="",
                        params={},
                        detail=f"disk_free_gb={h.disk_free_gb:.1f} < warn={th.disk_free_gb_warn}",
                    )
                )

        if h.ram_avail_gb is not None and h.ram_avail_gb < th.ram_avail_gb_crit:
            advisories.append(
                Advisory(
                    worker_name=h.worker_name,
                    kind="ram_crit",
                    severity="crit",
                    suggested_playbook="restart_worker",
                    params={"worker": h.worker_name},
                    detail=f"ram_avail_gb={h.ram_avail_gb:.1f} < crit={th.ram_avail_gb_crit}",
                )
            )

        if h.temp_c is not None and h.temp_c > th.temp_c_warn:
            advisories.append(
                Advisory(
                    worker_name=h.worker_name,
                    kind="temp_warn",
                    severity="warn",
                    suggested_playbook="",
                    params={},
                    detail=f"temp_c={h.temp_c:.1f} > warn={th.temp_c_warn}",
                )
            )

    return advisories


def _print_table(healths: list[HostHealth], advisories: list[Advisory]) -> None:
    print(f"{'worker':<22} {'reachable':<10} {'disk_gb':<8} {'ram_gb':<8} {'temp_c':<7}")
    for h in healths:
        disk = f"{h.disk_free_gb:.1f}" if h.disk_free_gb is not None else "-"
        ram = f"{h.ram_avail_gb:.1f}" if h.ram_avail_gb is not None else "-"
        temp = f"{h.temp_c:.1f}" if h.temp_c is not None else "-"
        print(f"{h.worker_name:<22} {str(h.reachable):<10} {disk:<8} {ram:<8} {temp:<7}")

    if advisories:
        print("\nAdvisories:")
        for a in advisories:
            pb = a.suggested_playbook or "(no playbook)"
            print(f"  [{a.severity}] {a.worker_name}: {a.kind} -> {pb} — {a.detail}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fleet health probe over read-only SSH.")
    parser.add_argument("--json", action="store_true", help="print the full JSON to stdout")
    parser.add_argument(
        "--config",
        default=None,
        help="path to the YAML config (default: data/autonomy/config.yaml)",
    )
    return parser


def main(argv: Optional[list] = None) -> dict:
    logging.basicConfig(level=logging.INFO)
    parser = _build_parser()
    args = parser.parse_args(argv)

    config_path = Path(args.config) if args.config else REPO_ROOT / "data" / "autonomy" / "config.yaml"
    cfg = load_config(config_path)

    healths = collect()
    advisories = evaluate(healths, cfg.thresholds)

    payload = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "hosts": [h.to_dict() for h in healths],
        "advisories": [a.to_dict() for a in advisories],
    }

    out_path = cfg.data_dir / "fleet_health.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))

    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        _print_table(healths, advisories)

    return payload


if __name__ == "__main__":
    main()
