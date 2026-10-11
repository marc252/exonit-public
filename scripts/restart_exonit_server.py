#!/usr/bin/env python3
"""Restart `exonit.service` by killing the whole cgroup and waiting for
`systemd` to restart it (`Restart=on-failure`) and for `/` to respond.

Incident 2026-08-15: the server (systemd `exonit.service`, port 8400)
hung with the port still listening but no uvicorn worker accepting
requests (~150 threads blocked on a futex). Killing only the `MainPID`
is not enough -- systemd stayed in "deactivating" waiting for hung
children (the parent process never goes away because the children hold
it). That is why this script kills ALL pids of the service cgroup
(`cgroup.procs`), not only the `MainPID`.

Before killing, if `py-spy` (PYSPY_BIN) is present, a stack dump of each
cgroup pid is captured to `data/diagnostics/serverhang_<timestamp>/`
(bounded to 60s in total, best-effort -- never prevents the kill).

Worst-case time budget (Executor playbook: subprocess timeout of 300s):
`systemctl show` for MainPID (10s) + forensics (60s, a real hard cap --
the per-pid timeout is capped to whatever remains until the deadline) +
kill (~instant) + health poll (`--timeout-s`, default 210s) =
10 + 60 + 210 = **280s**, a 20s margin below the required 290s threshold
and 20s below the Executor's real 300s timeout.

Usage:
    python scripts/restart_exonit_server.py [--timeout-s 210] [--dry-run] [--skip-forensics]

Exit 0 if the server responds again within the timeout; 1 otherwise.
No dependencies outside the stdlib -- meant to be called from
`autonomy/playbooks.py::CATALOG["restart_server"]` (guarded Executor)
but also runnable by hand during a real incident.
"""
from __future__ import annotations

import argparse
import os
import shutil
import signal
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

ROOT = Path(__file__).resolve().parent.parent

SERVICE_NAME = "exonit.service"
CGROUP_PROCS_PATH = Path(f"/sys/fs/cgroup/system.slice/{SERVICE_NAME}/cgroup.procs")
HEALTH_URL = "http://localhost:8400/"

#: Poll interval while waiting for `/` to respond after the restart.
_POLL_INTERVAL_S = 5

#: py-spy binary (symlink into a dedicated venv with cap_sys_ptrace -- no sudo).
PYSPY_BIN = (os.environ.get("EXONIT_PYSPY_BIN") or shutil.which("py-spy")
             or str(Path.home() / ".local" / "bin" / "py-spy"))

#: Root directory where forensic dumps are stored (one subdir per incident).
DIAGNOSTICS_ROOT = ROOT / "data" / "diagnostics"

#: Global deadline of the forensic capture (incident 2026-08-15: the
#: Executor playbook has a 300s subprocess timeout and margin must be left
#: for the kill + health poll -- forensics can never eat the whole budget).
#: It is a real hard limit: the timeout passed to each `py-spy dump` is
#: capped to the time remaining until this deadline (see
#: `_capture_forensics`), not merely checked before LAUNCHING each dump.
_FORENSICS_DEADLINE_S = 60.0

#: Maximum timeout per individual pid (`py-spy dump --pid <pid>`) -- also
#: capped to the time remaining until `_FORENSICS_DEADLINE_S`.
_PYSPY_PER_PID_TIMEOUT_S = 20

#: Timeout of `systemctl show -p MainPID` (see `_read_main_pid`) -- included
#: in the worst-case time budget documented in the module docstring.
_MAIN_PID_TIMEOUT_S = 10


def _read_main_pid(run_cmd: Callable) -> Optional[int]:
    """`systemctl show -p MainPID --value exonit.service` -- no sudo, the
    service runs as the invoking user. Returns `None` if it cannot be
    read/parsed (the caller carries on anyway with the cgroup pids, if
    any)."""
    try:
        proc = run_cmd(
            ["systemctl", "show", "-p", "MainPID", "--value", SERVICE_NAME],
            capture_output=True, text=True, timeout=_MAIN_PID_TIMEOUT_S,
        )
    except Exception:  # noqa: BLE001 - systemctl unavailable/failing: carry on without MainPID
        return None
    out = (getattr(proc, "stdout", "") or "").strip()
    try:
        pid = int(out)
    except ValueError:
        return None
    return pid if pid > 0 else None


def _read_cgroup_pids(cgroup_procs_path: Path) -> list[int]:
    """All live pids of the service cgroup -- `cgroup.procs` is
    readable without root under cgroup v2. Empty/missing -> `[]` (never
    blocks; only what was found gets killed)."""
    try:
        text = cgroup_procs_path.read_text()
    except OSError:
        return []
    pids: list[int] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            pids.append(int(line))
        except ValueError:
            continue
    return pids


def _capture_forensics(
    pids: list[int],
    *,
    run_cmd: Callable,
    monotonic: Callable[[], float],
    print_fn: Callable[[str], None],
    pyspy_bin: str = PYSPY_BIN,
    diagnostics_root: Path = DIAGNOSTICS_ROOT,
) -> None:
    """`py-spy dump --pid <pid>` for each cgroup pid BEFORE the SIGKILL,
    saved to `data/diagnostics/serverhang_<UTC timestamp>/pid<N>.txt`.

    Incident 2026-08-15: killing without capturing stacks first left us
    with no forensics of the futex-lock cause. ALL of this is best-effort
    and bounded to `_FORENSICS_DEADLINE_S` in total (a real hard limit: the
    timeout of each dump is capped to the time remaining until the
    deadline, not merely checked before launching it) -- no exception
    (missing binary, timeout, pid already dead, full disk,
    `PermissionError` when checking the binary, failing `monotonic()`...)
    may prevent the kill that follows --
    NOTE: this function can still raise at points not wrapped by an inner
    `try/except` (e.g. the deadline computation); the caller
    (`restart_server`) is responsible for wrapping the WHOLE call in its
    own `try/except`. The first pids of the cgroup get priority if the
    deadline runs out before all of them are done."""
    if not Path(pyspy_bin).exists():
        print_fn("[restart] py-spy not available; skipping forensics")
        return
    if not pids:
        return

    try:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        out_dir = diagnostics_root / f"serverhang_{ts}"
        out_dir.mkdir(parents=True, exist_ok=True)
    except Exception as exc:  # noqa: BLE001 - do not block the kill over a full disk/permissions
        print_fn(f"[restart] could not create the forensics directory: {exc!r}")
        return

    deadline = monotonic() + _FORENSICS_DEADLINE_S
    for pid in pids:
        now = monotonic()
        remaining = deadline - now
        if remaining <= 0:
            print_fn("[restart] forensics: 60s deadline exhausted, skipping the remaining pids")
            break
        # The 60s deadline is a REAL hard limit, not just a check before
        # launching each dump: if 8s remain until the deadline, the dump
        # itself is cut at 8s (not at the 20s of _PYSPY_PER_PID_TIMEOUT_S).
        per_pid_timeout = min(_PYSPY_PER_PID_TIMEOUT_S, remaining)
        try:
            proc = run_cmd(
                [pyspy_bin, "dump", "--pid", str(pid)],
                capture_output=True, text=True, timeout=per_pid_timeout,
            )
            stdout = getattr(proc, "stdout", "") or ""
            stderr = getattr(proc, "stderr", "") or ""
            (out_dir / f"pid{pid}.txt").write_text(stdout + stderr)
            print_fn(f"[restart] forensics: dump of pid {pid} saved to {out_dir / f'pid{pid}.txt'}")
        except Exception as exc:  # noqa: BLE001 - best-effort: never prevent the kill that follows
            print_fn(f"[restart] forensics: dump of pid {pid} failed (ok, carrying on): {exc!r}")


def _kill_pids(pids: list[int], kill_fn: Callable[[int, int], None], print_fn: Callable[[str], None]) -> None:
    """`SIGKILL` each pid, ignoring `ProcessLookupError` (already dead --
    normal, the cgroup may include children that have already exited) and
    `PermissionError` (should never happen, the service runs as the same
    user, but we do not want one problematic pid to stop the others)."""
    for pid in pids:
        try:
            kill_fn(pid, signal.SIGKILL)
            print_fn(f"[restart] kill -9 {pid}")
        except ProcessLookupError:
            print_fn(f"[restart] pid {pid} no longer existed (ok)")
        except PermissionError as exc:
            print_fn(f"[restart] no permission to kill {pid}: {exc!r}")


def _wait_for_health(
    *,
    timeout_s: int,
    urlopen: Callable,
    sleep_fn: Callable[[float], None],
    monotonic: Callable[[], float],
    print_fn: Callable[[str], None],
) -> bool:
    """Poll `GET HEALTH_URL` every `_POLL_INTERVAL_S` until it responds
    (any HTTP response counts -- someone only needs to be listening and
    accepting, not a specific status) or `timeout_s` runs out."""
    deadline = monotonic() + timeout_s
    while True:
        try:
            with urlopen(HEALTH_URL, timeout=5) as resp:
                resp.read()
            print_fn(f"[restart] {HEALTH_URL} responds -- server restarted")
            return True
        except Exception as exc:  # noqa: BLE001 - still going down/restarting: retry
            print_fn(f"[restart] not responding yet ({exc!r}); retrying...")
        if monotonic() >= deadline:
            return False
        sleep_fn(_POLL_INTERVAL_S)


def restart_server(
    *,
    run_cmd: Callable = subprocess.run,
    sleep_fn: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
    urlopen: Callable = urllib.request.urlopen,
    kill_fn: Callable[[int, int], None] = os.kill,
    cgroup_procs_path: Path = CGROUP_PROCS_PATH,
    pyspy_bin: str = PYSPY_BIN,
    diagnostics_root: Path = DIAGNOSTICS_ROOT,
    timeout_s: int = 210,
    dry_run: bool = False,
    skip_forensics: bool = False,
    print_fn: Callable[[str], None] = print,
) -> int:
    """Kill the whole `exonit.service` cgroup (MainPID + `cgroup.procs`) and
    wait for systemd to restart it and for `HEALTH_URL` to respond. All
    dependencies are injectable for tests -- no real call to
    `subprocess`/`os.kill`/network outside of here.

    Before killing (if `py-spy` is present and `skip_forensics=False`), a
    `py-spy dump` of each cgroup pid is captured (bounded to
    `_FORENSICS_DEADLINE_S` = 60s in total, a real hard limit -- see
    `_capture_forensics`) to have a forensic stack of the hang. The default
    `timeout_s` is 210s (not 240) so that the worst case -- `systemctl show`
    (10s, `_MAIN_PID_TIMEOUT_S`) + forensics (60s) + kill (~instant) +
    health poll (210s) = 280s -- stays below the required 290s and the
    300s subprocess timeout of the Executor playbook (see the module
    docstring for the full breakdown).

    Returns 0 if the server responds again within `timeout_s`, 1 if not.
    `dry_run=True` only shows the pids it would kill and exits 0 without
    touching anything (nor capturing forensics)."""
    main_pid = _read_main_pid(run_cmd)
    cgroup_pids = _read_cgroup_pids(cgroup_procs_path)

    pids = sorted(set(cgroup_pids) | ({main_pid} if main_pid is not None else set()))

    print_fn(f"[restart] MainPID={main_pid} cgroup_pids={cgroup_pids} -> pids to kill={pids}")

    if dry_run:
        print_fn(f"[restart] --dry-run: nothing is killed (pids={pids})")
        return 0

    if not skip_forensics:
        # Belt-and-braces around the WHOLE call (not only the inner
        # try/excepts of _capture_forensics): a reviewer showed that an
        # exception in the deadline computation or in
        # `Path(pyspy_bin).exists()` (e.g. PermissionError, which
        # `pathlib.Path.exists()` does NOT swallow) escaped those inner
        # blocks and prevented the kills that follow. The invariant is
        # absolute: NO forensics path may prevent the kills.
        try:
            _capture_forensics(
                pids, run_cmd=run_cmd, monotonic=monotonic, print_fn=print_fn,
                pyspy_bin=pyspy_bin, diagnostics_root=diagnostics_root,
            )
        except Exception as exc:  # noqa: BLE001 - forensics always best-effort, never blocks the kill
            print_fn(f"[restart] forensics: capture failed entirely (ok, carrying on with the kill): {exc!r}")
    else:
        print_fn("[restart] --skip-forensics: py-spy is not captured")

    if not pids:
        print_fn("[restart] no pid found (service already stopped?); still waiting for it to respond")
    else:
        _kill_pids(pids, kill_fn, print_fn)

    print_fn(
        f"[restart] waiting for systemd to restart and {HEALTH_URL} to respond "
        f"(up to {timeout_s}s)..."
    )
    ok = _wait_for_health(
        timeout_s=timeout_s, urlopen=urlopen, sleep_fn=sleep_fn, monotonic=monotonic, print_fn=print_fn,
    )
    if ok:
        print_fn("[restart] OK: server restarted and responding")
        return 0
    print_fn(f"[restart] FAIL: the server did not respond within {timeout_s}s")
    return 1


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--timeout-s", type=int, default=210,
        help="maximum seconds to wait for the server to respond again (default 210)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="show the pids it would kill and exit 0 without touching anything (no forensics capture either)",
    )
    parser.add_argument(
        "--skip-forensics", action="store_true",
        help="do not capture py-spy stacks before killing",
    )
    return parser


def main(argv: Optional[list] = None) -> int:
    args = _build_parser().parse_args(argv)
    return restart_server(
        timeout_s=args.timeout_s, dry_run=args.dry_run, skip_forensics=args.skip_forensics,
    )


if __name__ == "__main__":
    sys.exit(main())
