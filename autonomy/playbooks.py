"""Playbook catalog and guarded executor (phase 1 — shadow/dry-run mode).

The catalog is a closed v1 (spec §4.4): no action outside this list can be
chosen, neither by rules nor by the local AI. The `Executor` is the only point of the
subsystem that would "execute" a real command — and in phase 1 it never does:

- `dry_run=True` by default.
- If `dry_run=False` is requested, the environment variable
  `EXONIT_AUTONOMY_ARMED=1` is also required (double key). Without it, the executor forces
  `dry_run=True` anyway and notes it in the detail.
- Every call — whether it succeeds or is refused — is recorded in `actions_log`
  (JSONL, one line per call) for auditing.
- Each playbook has an execution limit (per run, per lot, per 24h
  window, or unlimited) to avoid repair loops.

The limit counters are persisted in `state_path` (JSON
`{key: [iso_timestamps]}`). `reset_scope("run"|"lot")` clears only the
counters of the playbooks with that window (run/lot boundary).
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from autonomy.config import REPO_ROOT
from autonomy.types import PlaybookChoice

#: Default grace window for `detached` playbooks (fix 2026-09-19,
#: incident "resume_run kills the orchestrator it resumes"): just enough time
#: to detect an IMMEDIATE death of the subprocess (missing binary, invalid
#: argument, instant crash) -- the real orchestrator runs for hours, we never
#: wait for that to elapse.
_DEFAULT_DETACHED_GRACE_S = 20.0

#: Python interpreter of the subsystem (the same one running the supervisor).
PY = sys.executable

_LOG = logging.getLogger("autonomy.playbooks")

#: Fleet file (name -> {ssh, platform}); `ops/autonomy_fleet.example.json`
#: is the template. Overridable with EXONIT_AUTONOMY_FLEET.
FLEET_FILE = Path(os.environ.get("EXONIT_AUTONOMY_FLEET")
                  or REPO_ROOT / "ops" / "autonomy_fleet.json")


def load_fleet(path: Path = FLEET_FILE) -> dict[str, dict]:
    """Load the known fleet. Without a file: only `worker-default` (local)."""
    default = {"worker-default": {"ssh": "local", "platform": "linux"}}
    try:
        raw = json.loads(Path(path).read_text())
    except (OSError, ValueError) as exc:
        _LOG.warning("fleet file %s missing or unreadable (%s) — falling back to "
                     "worker-default only", path, exc)
        return default
    return {k: v for k, v in raw.items()
            if not k.startswith("_") and isinstance(v, dict) and "ssh" in v} or default


_FLEET = load_fleet()
#: worker -> SSH destination ("local" for the server host).
FLEET_WORKERS: dict[str, str] = {k: v["ssh"] for k, v in _FLEET.items()}
#: worker -> platform ("linux" | "mac" | "windows").
FLEET_PLATFORMS: dict[str, str] = {k: v.get("platform", "linux") for k, v in _FLEET.items()}


def is_mac_host(name: str) -> bool:
    """Mac according to the fleet platform; names outside the fleet: `worker-mac` prefix."""
    if name in FLEET_PLATFORMS:
        return FLEET_PLATFORMS[name] == "mac"
    return name.startswith("worker-mac")


@dataclass
class Playbook:
    """Definition of a playbook from the closed catalog."""

    name: str
    description: str
    allowed_params: set[str]
    required_params: set[str]
    limit_n: int
    limit_window: str  # "run" | "24h" | "lot" | "always"
    detached: bool = False  # True: subprocess.Popen (background), no waits


@dataclass
class ExecutionResult:
    """Result of a call to `Executor.execute`."""

    status: str  # dry_run | executed | refused_invalid | refused_limit
    detail: str


CATALOG: dict[str, Playbook] = {
    "rebuild_streams": Playbook(
        name="rebuild_streams",
        description=(
            "rebuild_work_streams.py --stage <phase> (Redis<->PG drift post-restart)"
        ),
        allowed_params={"stage"},
        required_params={"stage"},
        limit_n=2,
        limit_window="run",
    ),
    "requeue_orphans": Playbook(
        name="requeue_orphans",
        description=(
            "requeue of orphaned 'assigned' tasks (age>300s, cur_tic != assigned_tic)"
        ),
        allowed_params=set(),
        required_params=set(),
        limit_n=4,
        limit_window="24h",
    ),
    "restart_worker": Playbook(
        name="restart_worker",
        description="farmctl restart <worker> of ONE specific host",
        allowed_params={"worker"},
        required_params={"worker"},
        limit_n=3,
        limit_window="24h",
    ),
    "kill_zombies": Playbook(
        name="kill_zombies",
        description=(
            "kill zombie server/worker processes verified by pgrep on the host"
        ),
        allowed_params={"host"},
        required_params={"host"},
        limit_n=3,
        limit_window="24h",
    ),
    "clean_fits_cache": Playbook(
        name="clean_fits_cache",
        description="find fits_cache_* -delete on the Mac with a full disk (m4 incident 08-08)",
        allowed_params={"host"},
        required_params={"host"},
        limit_n=2,
        limit_window="24h",
    ),
    "wal_checkpoint": Playbook(
        name="wal_checkpoint",
        description="wal_checkpoint(TRUNCATE) between lots",
        allowed_params=set(),
        required_params=set(),
        limit_n=1,
        limit_window="lot",
    ),
    "resume_run": Playbook(
        name="resume_run",
        description="discovery_orchestrator.py --resume RUN_ID --force at the failed step",
        allowed_params={"run_id"},
        required_params={"run_id"},
        limit_n=3,
        limit_window="run",
        # Incident 2026-09-19: runs the orchestrator in the foreground for
        # hours -- with the foreground execution below (timeout=300) the
        # `Executor` SIGKILLed it at 300s, killing the very run it was
        # trying to resume. Must be detached (subprocess.Popen + a short
        # wait only to detect an immediate death).
        detached=True,
    ),
    "restart_server": Playbook(
        name="restart_server",
        description=(
            "kills the exonit.service cgroup (kill -9) and waits for systemd "
            "to restart it and /health to respond"
        ),
        allowed_params=set(),
        required_params=set(),
        limit_n=2,
        limit_window="24h",
    ),
    "pause_and_alert": Playbook(
        name="pause_and_alert",
        description="stop the loop, leave a structured note, send mail",
        allowed_params={"reason"},
        required_params=set(),
        limit_n=0,  # not applicable: limit_window=="always" is never checked
        limit_window="always",
    ),
}

# Playbooks whose limit is per specific host/worker/run (not global).
# The counting key includes the value of this param.
_SCOPED_PARAM: dict[str, str] = {
    "restart_worker": "worker",
    "clean_fits_cache": "host",
    # Incident 2026-09-19: `resume_run` limit_n=3 without scope counted
    # GLOBALLY (3 executions IN THE WHOLE LIFETIME of the `Executor`, not 3 per
    # run) -- with `state_path` persisting across lots, the 4th lot that ever
    # needed a resume already found it exhausted. `run_id` is always in
    # `params` (required, and `_tick_error` overrides it with `ctx.run_id`
    # -- never a value from the AI, see supervisor.py).
    "resume_run": "run_id",
}


def validate(choice: PlaybookChoice) -> Optional[str]:
    """Validate a playbook choice. Returns the rejection reason, or None if valid."""
    pb = CATALOG.get(choice.playbook)
    if pb is None:
        return f"unknown playbook: {choice.playbook!r}"

    params = choice.params or {}

    unknown = set(params) - pb.allowed_params
    if unknown:
        return f"params not allowed for {pb.name}: {sorted(unknown)}"

    missing = pb.required_params - set(params)
    if missing:
        return f"missing required params for {pb.name}: {sorted(missing)}"

    for key in ("worker", "host"):
        if key in params and params[key] not in FLEET_WORKERS:
            return f"{key} unknown in the fleet: {params[key]!r}"

    return None


def _missing_script(cmd: Optional[list[str]]) -> Optional[str]:
    """Path of the `scripts/*.py` target of `cmd` when that file does not exist, else None."""
    if not cmd or len(cmd) < 2 or not str(cmd[1]).startswith("scripts/"):
        return None
    return None if (REPO_ROOT / cmd[1]).is_file() else str(cmd[1])


def command_for(choice: PlaybookChoice) -> Optional[list[str]]:
    """Real command that would be executed for this choice (None for pause_and_alert).

    In phase 1 this command is never executed outside the `Executor`'s double
    arming guard; here it is only built for logging/dry-run.
    """
    p = choice.params or {}

    if choice.playbook == "rebuild_streams":
        return [PY, "scripts/rebuild_work_streams.py", "--stage", p.get("stage", "")]
    if choice.playbook == "requeue_orphans":
        return [PY, "scripts/requeue_orphans.py"]
    if choice.playbook == "restart_worker":
        return [PY, "bin/farmctl", "restart", p.get("worker", "")]
    if choice.playbook == "kill_zombies":
        return [PY, "scripts/kill_zombies.py", "--host", p.get("host", "")]
    if choice.playbook == "clean_fits_cache":
        return [PY, "scripts/clean_fits_cache.py", "--host", p.get("host", "")]
    if choice.playbook == "wal_checkpoint":
        return [PY, "scripts/wal_checkpoint.py"]
    if choice.playbook == "resume_run":
        return [
            PY,
            "scripts/discovery_orchestrator.py",
            "--resume",
            p.get("run_id", ""),
            "--force",
        ]
    if choice.playbook == "restart_server":
        return [PY, "scripts/restart_exonit_server.py"]
    if choice.playbook == "pause_and_alert":
        return None

    return None


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _tail_text(path: Path, n: int = 4000) -> str:
    """Last `n` characters of a log file, to attach to the detail of a
    `detached` playbook that died within the grace window. Never raises
    (missing/unreadable file -> empty string)."""
    try:
        text = path.read_text(errors="replace")
    except OSError:
        return ""
    return text[-n:]


def _count_key(choice: PlaybookChoice) -> str:
    scoped_param = _SCOPED_PARAM.get(choice.playbook)
    if scoped_param is None:
        return choice.playbook
    return f"{choice.playbook}:{(choice.params or {}).get(scoped_param, '')}"


class Executor:
    """Guarded executor: validates, checks limits, logs and (only if armed) executes."""

    def __init__(
        self,
        state_path: Path,
        actions_log: Path,
        dry_run: bool = True,
        detached_grace_s: float = _DEFAULT_DETACHED_GRACE_S,
    ):
        self.state_path = Path(state_path)
        self.actions_log = Path(actions_log)
        self.dry_run = dry_run
        self.detached_grace_s = detached_grace_s
        self.state: dict[str, list[str]] = self._load_state()

    def _load_state(self) -> dict[str, list[str]]:
        if not self.state_path.exists():
            return {}
        try:
            return json.loads(self.state_path.read_text())
        except (json.JSONDecodeError, OSError):
            return {}

    def _save_state(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(json.dumps(self.state))

    def _count_in_window(self, key: str, limit_window: str) -> int:
        timestamps = self.state.get(key, [])
        if limit_window != "24h":
            return len(timestamps)
        cutoff = _now() - timedelta(hours=24)
        count = 0
        for ts in timestamps:
            try:
                parsed = datetime.fromisoformat(ts)
            except ValueError:
                continue
            if parsed >= cutoff:
                count += 1
        return count

    def _record(self, key: str) -> None:
        self.state.setdefault(key, []).append(_now().isoformat())
        self._save_state()

    def _playbook_log_dir(self) -> Path:
        return self.actions_log.parent / "playbook_logs"

    def _run_detached(self, cmd: list[str], playbook_name: str) -> ExecutionResult:
        """Launch `cmd` as an independent background process (`start_new_session`)
        instead of `subprocess.run(..., timeout=300)`: a `detached` playbook
        (only `resume_run` today) runs the orchestrator for hours and
        `execute()` CANNOT wait for it in the foreground without the timeout
        killing it (incident 2026-09-19).

        It only waits `self.detached_grace_s` to detect an
        IMMEDIATE death (missing binary, invalid argument, instant crash) -- if
        it is still alive after that margin, it is considered "running" and
        returned with the PID; the real observation of whether the run has resumed is made
        by the supervisor polling the server/state, not by this subprocess.
        stdout+stderr are redirected to a file under
        `data/autonomy/playbook_logs/` (the process is independent -- there is no
        pipe to read from here)."""
        log_dir = self._playbook_log_dir()
        log_dir.mkdir(parents=True, exist_ok=True)
        ts = _now().strftime("%Y%m%dT%H%M%S%fZ")
        log_path = log_dir / f"{playbook_name}_{ts}.log"

        try:
            log_f = log_path.open("wb")
        except OSError as exc:
            return ExecutionResult(
                "executed", f"cmd={cmd} detached=True error opening log {log_path}: {exc!r}"
            )

        try:
            proc = subprocess.Popen(
                cmd,
                stdout=log_f,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
                close_fds=True,
            )
        except Exception as exc:  # noqa: BLE001 - we want to log any failure
            return ExecutionResult(
                "executed", f"cmd={cmd} detached=True error launching: {exc!r}"
            )
        finally:
            # Popen has duplicated the fd for the child (or it never got
            # launched); in both cases the parent no longer needs this
            # handle open.
            log_f.close()

        deadline = time.monotonic() + self.detached_grace_s
        rc = proc.poll()
        while rc is None and time.monotonic() < deadline:
            time.sleep(0.2)
            rc = proc.poll()

        if rc is not None:
            tail = _tail_text(log_path)
            return ExecutionResult(
                "executed",
                f"cmd={cmd} detached=True pid={proc.pid} DEAD_IMMEDIATELY rc={rc} "
                f"(within the grace window {self.detached_grace_s}s) log={log_path} "
                f"tail={tail!r}",
            )

        return ExecutionResult(
            "executed",
            f"cmd={cmd} detached=True pid={proc.pid} alive after {self.detached_grace_s}s "
            f"(running in background, real observation via server/state) log={log_path}",
        )

    def reset_scope(self, scope: str) -> None:
        """Clear the counters of the playbooks with `limit_window == scope`."""
        for key in list(self.state):
            playbook_name = key.split(":", 1)[0]
            pb = CATALOG.get(playbook_name)
            if pb is not None and pb.limit_window == scope:
                del self.state[key]
        self._save_state()

    def _log(
        self,
        choice: PlaybookChoice,
        trigger: str,
        result: ExecutionResult,
        effective_dry_run: bool,
    ) -> None:
        record = {
            "ts": _now().isoformat(),
            "trigger": trigger,
            "playbook": choice.playbook,
            "params": choice.params,
            "source": choice.source,
            "confidence": choice.confidence,
            "status": result.status,
            "detail": result.detail,
            "dry_run": effective_dry_run,
        }
        self.actions_log.parent.mkdir(parents=True, exist_ok=True)
        with self.actions_log.open("a") as f:
            f.write(json.dumps(record) + "\n")

    def execute(self, choice: PlaybookChoice, trigger: str) -> ExecutionResult:
        reason = validate(choice)
        if reason is not None:
            result = ExecutionResult("refused_invalid", reason)
            self._log(choice, trigger, result, self.dry_run)
            return result

        pb = CATALOG[choice.playbook]
        key = _count_key(choice)

        if pb.limit_window != "always":
            count = self._count_in_window(key, pb.limit_window)
            if count >= pb.limit_n:
                result = ExecutionResult(
                    "refused_limit",
                    f"limit {pb.limit_n}/{pb.limit_window} exceeded for {key!r} "
                    f"({count} in window)",
                )
                self._log(choice, trigger, result, self.dry_run)
                return result

        effective_dry_run = self.dry_run
        guard_note = ""
        if not effective_dry_run and os.environ.get("EXONIT_AUTONOMY_ARMED") != "1":
            effective_dry_run = True
            guard_note = (
                " [ARMED guard: EXONIT_AUTONOMY_ARMED != '1', dry_run forced]"
            )

        cmd = command_for(choice)

        if effective_dry_run:
            detail = f"would run: {cmd}{guard_note}" if cmd else f"pause_and_alert (no command){guard_note}"
            result = ExecutionResult("dry_run", detail)
        else:
            missing = _missing_script(cmd)
            if missing:
                # requeue_orphans / kill_zombies / clean_fits_cache / wal_checkpoint name scripts that are not
                # part of the repository. Say so clearly instead of letting python fail with "can't open file".
                result = ExecutionResult(
                    "executed",
                    f"cmd={cmd} error=script not found: {missing} (this playbook has no implementation in "
                    f"this checkout; nothing was run)",
                )
            elif cmd is None:
                result = ExecutionResult("executed", "pause_and_alert: no command; alert logged")
            elif pb.detached:
                # incident 2026-09-19: the `subprocess.run(..., timeout=300)` below
                # SIGKILLed `resume_run` (it runs for hours in the foreground).
                result = self._run_detached(cmd, choice.playbook)
            else:
                try:
                    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
                    result = ExecutionResult(
                        "executed",
                        f"cmd={cmd} rc={proc.returncode} "
                        f"stdout_tail={proc.stdout[-500:]!r} stderr_tail={proc.stderr[-500:]!r}",
                    )
                except Exception as exc:  # noqa: BLE001 - we want to log any failure
                    result = ExecutionResult("executed", f"cmd={cmd} error={exc!r}")

        self._record(key)
        self._log(choice, trigger, result, effective_dry_run)
        return result
