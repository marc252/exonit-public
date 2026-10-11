"""ExoNit Worker Agent — minimal cross-platform process controller.

This module is invoked over SSH from `bin/farmctl` on the server host. It
manages the worker.py lifecycle on a single host without depending on any
OS-specific service manager (no systemd, launchd, NSSM, Task Scheduler).

Each invocation runs a single command and exits, so there is NO long-lived
agent daemon — the SSH session lives only for the duration of the call.

Usage (invoked via ``ssh host 'python3 path/to/agent.py <command> [args]'``):

    python3 agent.py start <worker-name>
    python3 agent.py stop <worker-name> [--force]
    python3 agent.py status <worker-name>
    python3 agent.py logs [<worker-name>] [-n N]
    python3 agent.py version

All commands print a single JSON object to stdout (except `logs`, which
streams raw text). Exit code 0 on success, 1 on user error, 2 on internal
failure.

Multi-worker support (``--instances N --instance-index i``):
    When N>1 (or instance-index>1), agent generates a per-worker derived
    config ``.config.<worker-name>.json`` in the worker dir and uses it
    instead of the base ``config.json``.  PID/state files are always
    per-name: ``.agent.<worker-name>.pid`` / ``.agent.<worker-name>.worker_name``.
    This allows N workers to coexist in the same directory without collisions.

    With instances=1 (default, absent), the derived config path still
    uses the worker name for the PID file (e.g. ``.agent.worker-default.pid``)
    but the base ``config.json`` is used unchanged — ensuring bit-identical
    behaviour for existing single-worker setups.

Per-host config overrides (``--config-overrides-b64``):
    ``bin/farmctl`` may pass a base64(JSON) blob built from ``ops/workers.json``'s
    per-worker ``config`` field (e.g. <host>'s ``{"gpu_enabled": true}``,
    closing the farmctl-deploy-clobbers-worker-config gotcha: a `farmctl
    deploy` overwrites the tracked config.json, but the next start/restart
    re-applies the host override from the versioned manifest). Decoded and
    merged into the base config BEFORE the per-instance derived fields
    above — those always win over an override (they encode operational
    correctness: unique ports, unique caches, the 25%-free CPU rule).
    Malformed overrides (bad base64, bad JSON, JSON that isn't an object)
    REFUSE the start rather than silently ignoring them — see
    ``_decode_config_overrides_b64``.
"""
from __future__ import annotations

import argparse
import base64
import json
import math
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# Shared with worker.py's PID lock (2026-07-02 cleanup review, D13) — see
# procutil.py for the Windows os.execv SystemError edge-case docstring.
from procutil import pid_alive as _pid_alive

WORKER_DIR = Path(__file__).resolve().parent

# Legacy single-worker PID/name files — kept for backward reference only.
# All live code now uses per-name paths computed by _pid_file() / _name_file().
_LEGACY_PID_FILE = WORKER_DIR / ".agent.pid"
_LEGACY_NAME_FILE = WORKER_DIR / ".agent.worker_name"


def _pid_file(worker_name: str) -> Path:
    """Return the per-name PID file path."""
    return WORKER_DIR / f".agent.{worker_name}.pid"


def _name_file(worker_name: str) -> Path:
    """Return the per-name worker-name state file path."""
    return WORKER_DIR / f".agent.{worker_name}.worker_name"


def _read_pid(worker_name: str) -> int | None:
    pf = _pid_file(worker_name)
    if not pf.exists():
        return None
    try:
        return int(pf.read_text().strip())
    except (ValueError, OSError):
        return None


def _read_worker_name(worker_name: str) -> str | None:
    nf = _name_file(worker_name)
    if not nf.exists():
        return None
    try:
        return nf.read_text().strip() or None
    except OSError:
        return None


def _migrate_legacy_pid(worker_name: str) -> None:
    """One-time idempotent migration of legacy single-worker PID files.

    The old agent wrote ``.agent.pid`` and ``.agent.worker_name``; the new
    code uses ``.agent.<worker_name>.pid`` and ``.agent.<worker_name>.worker_name``.

    If the new per-name PID file already exists the migration is skipped
    (idempotent).  If the legacy file does not exist there is nothing to do.
    This covers the cutover scenario where worker-default was started with the
    old agent and its PID lives in the legacy files — the new ``stop``/
    ``status`` code would otherwise not find it and leave the process as an
    orphan.
    """
    new_pid = _pid_file(worker_name)
    if new_pid.exists():
        # Already migrated (or started fresh with new code).
        return
    if not _LEGACY_PID_FILE.exists():
        # Nothing to migrate.
        return
    # Rename legacy files to per-name paths.
    _LEGACY_PID_FILE.rename(new_pid)
    if _LEGACY_NAME_FILE.exists():
        _LEGACY_NAME_FILE.rename(_name_file(worker_name))


def _git_sha() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short=8", "HEAD"],
            cwd=WORKER_DIR, capture_output=True, text=True, timeout=2)
        if result.returncode == 0:
            return result.stdout.strip() or "unknown"
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass
    return "unknown"


def _python_executable() -> str:
    """Use the same python that started the agent — farmctl invokes agent.py
    via the per-host python declared in ops/workers.json, so sys.executable
    here is already the right venv. This keeps the manifest as the single
    source of truth for python paths and avoids agent.py re-discovering
    venvs on disk (which was a footgun on hosts with multiple stale venvs).
    """
    return sys.executable


def _decode_config_overrides_b64(b64_str: str) -> dict:
    """Decode ``--config-overrides-b64`` into a per-host config-override dict.

    Base64, not raw JSON, because this argument crosses SSH into cmd.exe,
    PowerShell and POSIX shells with mutually incompatible quoting rules for
    embedded double quotes — same reasoning as ``--allow-start`` being a flag
    rather than an env var (see START_GUARD_ENV usage below). The matching
    encoder is ``bin/farmctl``'s ``_encode_config_overrides_b64``.

    Raises ``ValueError`` (with a message identifying which stage failed) on
    ANY malformation: invalid base64, invalid UTF-8, invalid JSON, or JSON
    that decodes to something other than an object. This feature exists
    specifically to close a silent-config-drift bug
    (farmctl-deploy-clobbers-worker-config) — a malformed override must
    refuse the start loudly, never fall back to "ignore and continue".
    """
    try:
        raw = base64.b64decode(b64_str, validate=True)
    except Exception as exc:
        raise ValueError(f"invalid base64 in --config-overrides-b64: {exc}") from exc
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(
            f"--config-overrides-b64 decoded to invalid UTF-8: {exc}") from exc
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"--config-overrides-b64 is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(
            "--config-overrides-b64 must decode to a JSON object (dict), "
            f"got {type(data).__name__}")
    return data


def _apply_overrides(worker_name: str, overrides: dict) -> Path:
    """Single-instance-with-overrides path: base config + host overrides only.

    Used when ``instances == 1 and instance_index == 1`` AND overrides are
    present. Deliberately does NOT go through ``_derive_config``'s
    multi-instance derivation (gui_port shift, fits_cache_dir suffix, log_file
    rename, cpu_threads redistribution) — a single-instance host's semantics
    stay exactly what they were before this feature, with the override dict
    merged on top. When overrides are absent, ``cmd_start`` keeps using
    ``config.json`` verbatim (this function is not called at all) — that is
    the bit-identical retrocompat path.
    """
    base_cfg_path = WORKER_DIR / "config.json"
    if not base_cfg_path.exists():
        raise FileNotFoundError(f"base config not found: {base_cfg_path}")
    cfg = json.loads(base_cfg_path.read_text())
    cfg.update(overrides)
    derived_path = WORKER_DIR / f".config.{worker_name}.json"
    derived_path.write_text(json.dumps(cfg, indent=4))
    return derived_path


def _derive_config(worker_name: str, instance_index: int, instances: int,
                    overrides: dict | None = None) -> Path:
    """Build a per-worker derived config and write it to worker_dir.

    Reads ``config.json`` (base), applies host ``overrides`` (if any, from
    ops/workers.json's per-worker ``config`` field via
    ``--config-overrides-b64``), THEN applies per-instance derived fields,
    writes to ``.config.<worker_name>.json``.  Returns the derived config path.

    Precedence: per-instance derived fields below ALWAYS win over an entry in
    ``overrides`` with the same key — they encode operational correctness
    (unique ports, unique caches, the 25%-free-CPU rule), and a host override
    must never be able to break that. In practice the two sets of keys don't
    overlap (overrides carry things like ``gpu_enabled``), but the ordering
    here is the contract regardless.

    Per-instance fields applied:
      - worker_name   → <worker_name>
      - gui_port      → base_gui_port + (instance_index - 1)
      - fits_cache_dir → <worker_dir>/fits_cache_<instance_index>  (absolute)
      - log_file      → worker-<worker_name>.log
      - cpu_threads   → distributed share of floor(0.75 * cpu_count), min 1
                        (first ``total % instances`` instances get base+1,
                        the rest get base; total never exceeds 75% of cores)
      - fpp_cpu_cores → same as cpu_threads
      - gui_enabled   → False for instance_index > 1  (prevents OSError on
                        secondary workers; primary still gets the GUI)

    All other fields (gpu_enabled, server_url, token, exominer_mode, …)
    are inherited verbatim from the base config (or from ``overrides``, if
    present there).
    """
    base_cfg_path = WORKER_DIR / "config.json"
    if not base_cfg_path.exists():
        raise FileNotFoundError(f"base config not found: {base_cfg_path}")

    raw_base_cfg = json.loads(base_cfg_path.read_text())
    cfg = dict(raw_base_cfg)

    # --- host overrides (ops/workers.json `config`) — applied BEFORE the
    # per-instance derived fields below, so those always take precedence.
    # NOTE: the per-instance fields below deliberately read their "base"
    # reference values (base_gui_port, base_fits) from ``raw_base_cfg``
    # (pre-override), NOT from ``cfg`` (post-override) — so an override that
    # happens to name one of the reserved per-instance keys (gui_port,
    # fits_cache_dir, …) has literally zero effect on the derived value, not
    # just "gets overwritten by the same number it would have produced
    # anyway". ---
    if overrides:
        cfg.update(overrides)

    # --- worker identity ---
    cfg["worker_name"] = worker_name

    # --- GUI port: avoid bind collisions across instances ---
    base_gui_port = raw_base_cfg.get("gui_port", 8401)
    cfg["gui_port"] = base_gui_port + (instance_index - 1)

    # --- GUI enabled: only primary instance gets the GUI ---
    if instance_index > 1:
        cfg["gui_enabled"] = False

    # --- fits cache: unique dir per instance to prevent download races ---
    base_fits = raw_base_cfg.get("fits_cache_dir", "")
    if base_fits:
        # Normalise to absolute before appending suffix
        base_fits_path = Path(base_fits)
        if not base_fits_path.is_absolute():
            base_fits_path = WORKER_DIR / base_fits_path
        cfg["fits_cache_dir"] = str(base_fits_path.parent /
                                    f"{base_fits_path.name}_{instance_index}")
    else:
        cfg["fits_cache_dir"] = str(WORKER_DIR / f"fits_cache_{instance_index}")

    # --- log file: one log per worker ---
    cfg["log_file"] = f"worker-{worker_name}.log"

    # --- CPU threads: honour the 25%-free rule across N instances ---
    # Budget: floor(0.75 * cpu_count) threads total (never more than 75% of
    # physical cores — the 25%-free rule is enforced on the HOST total, not
    # per-instance, so floor-per-instance waste is eliminated).
    #
    # Distribution: base = total // instances.  The first (total % instances)
    # instances (1-based) each receive base+1; the remainder receive base.
    # Minimum 1 per instance regardless (degenerate case: many instances on
    # very few cores — each still gets at least 1 thread; this can exceed the
    # total budget, but it is the lesser evil when instances > 0.75*cpu_count).
    cpu_count = os.cpu_count() or 1
    total = math.floor(0.75 * cpu_count)
    base = total // instances
    remainder = total % instances
    threads = max(1, base + (1 if instance_index <= remainder else 0))
    cfg["cpu_threads"] = threads
    cfg["fpp_cpu_cores"] = threads

    derived_path = WORKER_DIR / f".config.{worker_name}.json"
    derived_path.write_text(json.dumps(cfg, indent=4))
    return derived_path


#: Same guard `bin/farmctl` enforces server-side (START_GUARD_ENV there), moved
#: down to the ONLY code path that can actually create a worker process.
#:
#: farmctl's guard turned out to be a fence around an open field: on 2026-07-14,
#: after it was in place, the fleet was started a fourth time anyway — by
#: calling THIS file over SSH directly, bypassing farmctl entirely. Anything
#: with SSH (and the automated agents have it, by pubkey, as the operator's own
#: user) could start 83 workers, and one such start drained the ExoMiner queue
#: mid-analysis and corrupted a running experiment.
#:
#: Honest about what this is: a tripwire, not a security boundary. Whoever can
#: run this file can also pass the flag. The value is that an ACCIDENTAL or
#: well-meaning start ("just verifying the fix") now fails loudly, and going
#: around it has to be deliberate and leaves a greppable trace.
#:
#: `stop` stays unguarded, deliberately: the emergency brake must always work.
START_GUARD_ENV = "EXONIT_FARMCTL_ALLOW_START"


def _start_authorised(args) -> bool:
    """True iff this start was explicitly authorised.

    Two channels, both accepted:
      * ``--allow-start`` — what farmctl passes. The flag (not the env var) is
        the primary channel because it survives SSH identically on Windows
        (PowerShell), macOS and Linux, whereas exporting an env var through
        ``ssh host 'cmd'`` is shell-dependent and silently drops on some hosts.
      * ``EXONIT_FARMCTL_ALLOW_START=1`` — for a human at the worker's own
        terminal, and for parity with farmctl's own guard.
    """
    return (getattr(args, "allow_start", False)
            or os.environ.get(START_GUARD_ENV) == "1")


def cmd_start(args) -> dict:
    worker_name = args.worker_name
    if not worker_name:
        return {"status": "error",
                "error": "start requires --worker-name <name>"}

    if not _start_authorised(args):
        return {
            "status": "refused",
            "error": (
                "REFUSED: starting a worker needs --allow-start (or "
                f"{START_GUARD_ENV}=1). `bin/farmctl start` passes it for you. "
                "This guard exists because automated agents started the real "
                "fleet four times on 2026-07-13/14 while explicitly instructed "
                "not to — the last one by calling agent.py over SSH to get "
                "around farmctl's own guard. `stop` is never guarded."
            ),
        }

    # Decode per-host config overrides (config-overrides-per-host feature)
    # BEFORE touching any PID/state file: a malformed override must refuse
    # cleanly, with no side effects, exactly like the auth guard above — see
    # _decode_config_overrides_b64 for why "refuse loudly" is the point here.
    overrides = None
    overrides_b64 = getattr(args, "config_overrides_b64", None)
    if overrides_b64:
        try:
            overrides = _decode_config_overrides_b64(overrides_b64)
        except ValueError as exc:
            return {"status": "refused", "error": f"REFUSED: {exc}"}

    # Migrate legacy .agent.pid → .agent.<worker_name>.pid on first run after
    # deploy, so that a worker started with the old code can still be stopped.
    _migrate_legacy_pid(worker_name)

    pid = _read_pid(worker_name)
    if pid is not None and _pid_alive(pid):
        return {"status": "already_running", "pid": pid,
                "worker_name": _read_worker_name(worker_name)}
    if pid is not None:
        # Stale PID file — clean it up.
        _pid_file(worker_name).unlink(missing_ok=True)

    instances = getattr(args, "instances", 1) or 1
    instance_index = getattr(args, "instance_index", 1) or 1

    # Determine config path:
    #   - instances=1 AND instance_index=1, NO overrides → use config.json
    #     verbatim (guarantees byte-identical retrocompat for single-worker
    #     setups that don't declare a `config` override in workers.json)
    #   - instances=1 AND instance_index=1, WITH overrides → base + overrides,
    #     no multi-instance derivation (_apply_overrides)
    #   - otherwise → generate derived config (_derive_config), overrides
    #     applied before the per-instance fields, which always win
    if instances == 1 and instance_index == 1:
        if overrides:
            config_path = _apply_overrides(worker_name, overrides)
        else:
            config_path = WORKER_DIR / "config.json"
    else:
        config_path = _derive_config(worker_name, instance_index, instances,
                                      overrides=overrides)

    log_path = WORKER_DIR / f"worker-{worker_name}.log"
    log_fh = open(log_path, "ab")

    cmd = [_python_executable(), "-u", str(WORKER_DIR / "worker.py"),
           "--config", str(config_path),
           "--worker-name", worker_name]
    if args.server_url:
        cmd.extend(["--server-url", args.server_url])

    # Build subprocess environment: inherit current env, add NUMBA_CACHE_DIR
    # so each worker gets its own JIT cache (avoids numba cache corruption
    # when N workers compile simultaneously in the same directory).
    # Use tempfile.gettempdir() for cross-platform correctness: on Windows
    # this resolves to %LOCALAPPDATA%\Temp (e.g. C:\Users\<user>\AppData\Local\Temp)
    # rather than the POSIX /tmp which does not exist on Windows natively.
    sub_env = os.environ.copy()
    sub_env["NUMBA_CACHE_DIR"] = os.path.join(
        tempfile.gettempdir(), f"numba-{worker_name}")

    popen_kwargs: dict = {
        "cwd": str(WORKER_DIR),
        "stdout": log_fh, "stderr": subprocess.STDOUT,
        "stdin": subprocess.DEVNULL,
        "close_fds": True,
        "env": sub_env,
    }
    if os.name == "nt":
        # Windows: full detach. Without CREATE_BREAKAWAY_FROM_JOB the worker
        # is killed when the parent SSH session ends if Windows associates
        # the SSH server with a job object that has JOB_OBJECT_LIMIT_KILL_ON
        # _JOB_CLOSE — observed empirically on Windows 11 OpenSSH.
        flags = (subprocess.CREATE_NEW_PROCESS_GROUP
                 | subprocess.DETACHED_PROCESS
                 | 0x00000008)  # CREATE_NO_WINDOW
        # CREATE_BREAKAWAY_FROM_JOB is 0x01000000; only honoured if the
        # parent job allows it. Try to set it; if Windows refuses with
        # ERROR_ACCESS_DENIED, retry without the flag.
        try:
            popen_kwargs["creationflags"] = flags | 0x01000000
            proc = subprocess.Popen(cmd, **popen_kwargs)
        except OSError:
            popen_kwargs["creationflags"] = flags
            proc = subprocess.Popen(cmd, **popen_kwargs)
    else:
        # POSIX: start_new_session detaches from the SSH controlling tty.
        popen_kwargs["start_new_session"] = True
        proc = subprocess.Popen(cmd, **popen_kwargs)

    _pid_file(worker_name).write_text(str(proc.pid))
    _name_file(worker_name).write_text(worker_name)

    # Brief sanity wait — if the worker dies in the first second, the user
    # should see it now rather than discover it on the next status call.
    time.sleep(1.0)
    alive = _pid_alive(proc.pid)
    return {
        "status": "started" if alive else "exited_immediately",
        "pid": proc.pid,
        "worker_name": worker_name,
        "git_sha": _git_sha(),
    }


def cmd_stop(args) -> dict:
    worker_name = getattr(args, "worker_name", None)
    if not worker_name:
        return {"status": "error",
                "error": "stop requires --worker-name <name>"}

    # Migrate legacy PID files so we can stop a worker started with old code.
    _migrate_legacy_pid(worker_name)

    pid = _read_pid(worker_name)
    name = _read_worker_name(worker_name)
    if pid is None or not _pid_alive(pid):
        if pid is not None:
            _pid_file(worker_name).unlink(missing_ok=True)
            _name_file(worker_name).unlink(missing_ok=True)
        return {"status": "not_running"}
    sig = signal.SIGKILL if (args.force and hasattr(signal, "SIGKILL")) \
        else signal.SIGTERM
    try:
        os.kill(pid, sig)
    except (ProcessLookupError, PermissionError, OSError) as exc:
        return {"status": "error", "error": str(exc), "pid": pid}
    # Wait up to 10 s for graceful exit, then escalate.
    deadline = time.time() + 10
    while _pid_alive(pid) and time.time() < deadline:
        time.sleep(0.5)
    if _pid_alive(pid) and not args.force:
        try:
            os.kill(pid, signal.SIGKILL if hasattr(signal, "SIGKILL")
                    else signal.SIGTERM)
        except (ProcessLookupError, PermissionError, OSError):
            pass
    _pid_file(worker_name).unlink(missing_ok=True)
    _name_file(worker_name).unlink(missing_ok=True)
    return {"status": "stopped", "pid": pid, "worker_name": name}


def cmd_status(args) -> dict:
    worker_name = getattr(args, "worker_name", None)
    if not worker_name:
        return {"status": "error",
                "error": "status requires --worker-name <name>"}

    pid = _read_pid(worker_name)
    name = _read_worker_name(worker_name)
    running = pid is not None and _pid_alive(pid)
    if pid is not None and not running:
        # Stale PID file — clean up so the next start succeeds.
        _pid_file(worker_name).unlink(missing_ok=True)
        _name_file(worker_name).unlink(missing_ok=True)
        pid = None
    return {
        "status": "running" if running else "stopped",
        "pid": pid if running else None,
        "worker_name": name if running else None,
        "git_sha": _git_sha(),
    }


def cmd_logs(args) -> str:
    worker_name = getattr(args, "worker_name", None)
    if worker_name:
        log_file = WORKER_DIR / f"worker-{worker_name}.log"
    else:
        # Fallback: legacy single log file
        log_file = WORKER_DIR / "worker.log"
    if not log_file.exists():
        # Also try the legacy path
        legacy = WORKER_DIR / "worker.log"
        if legacy.exists():
            log_file = legacy
        else:
            return ""
    try:
        with open(log_file, "rb") as f:
            data = f.read()
    except OSError as exc:
        return f"(log read error: {exc})"
    lines = data.decode("utf-8", errors="replace").splitlines()
    return "\n".join(lines[-args.n:]) if args.n > 0 else "\n".join(lines)


def cmd_version(args) -> dict:
    return {"git_sha": _git_sha(), "agent_version": "1.1",
            "python": sys.version.split()[0],
            "platform": "windows" if os.name == "nt" else "posix"}


def main(argv=None):
    p = argparse.ArgumentParser(prog="agent.py", description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="command", required=True)

    p_start = sub.add_parser("start", help="Start worker.py if not already running")
    p_start.add_argument("worker_name",
                         help="Logical worker name (overrides config.json)")
    p_start.add_argument("--server-url",
                         help="Server URL to register against (overrides "
                              "config.json server_url; passed straight to "
                              "worker.py --server-url)")
    p_start.add_argument("--instances", type=int, default=1,
                         help="Total number of worker instances on this host (default 1)")
    p_start.add_argument("--instance-index", type=int, default=1,
                         dest="instance_index",
                         help="1-based index of this instance (default 1)")
    p_start.add_argument("--allow-start", action="store_true",
                         dest="allow_start",
                         help="Authorise this start (see START_GUARD_ENV). "
                              "bin/farmctl passes it; a stray script or agent "
                              "calling agent.py over SSH does not.")
    p_start.add_argument("--config-overrides-b64", dest="config_overrides_b64",
                         default=None,
                         help="Base64(JSON) per-host config overrides, from "
                              "ops/workers.json's per-worker `config` field "
                              "(bin/farmctl encodes it). Merged into the "
                              "derived config BEFORE per-instance fields "
                              "(gui_port, fits_cache_dir, cpu_threads, …), "
                              "which always win. Malformed input REFUSES the "
                              "start rather than silently ignoring it.")
    p_start.set_defaults(func=cmd_start)

    p_stop = sub.add_parser("stop", help="Stop the worker via SIGTERM (then SIGKILL)")
    p_stop.add_argument("worker_name",
                        help="Logical worker name to stop")
    p_stop.add_argument("--force", action="store_true",
                        help="Send SIGKILL immediately instead of SIGTERM")
    p_stop.set_defaults(func=cmd_stop)

    p_status = sub.add_parser("status", help="Report worker liveness + git SHA")
    p_status.add_argument("worker_name",
                          help="Logical worker name to query")
    p_status.set_defaults(func=cmd_status)

    p_logs = sub.add_parser("logs", help="Print last N lines of the worker log")
    p_logs.add_argument("worker_name", nargs="?", default=None,
                        help="Worker name (uses worker-<name>.log; "
                             "falls back to worker.log if omitted)")
    p_logs.add_argument("-n", type=int, default=50,
                        help="Number of trailing lines (default 50; 0 = all)")
    p_logs.set_defaults(func=cmd_logs)

    p_version = sub.add_parser("version", help="Print agent + git version")
    p_version.set_defaults(func=cmd_version)

    args = p.parse_args(argv)
    result = args.func(args)
    # Windows default code page (cp1252) chokes on log lines with accented
    # chars or replacement codepoints. Write raw UTF-8 bytes to the stdout
    # buffer to bypass the legacy text codec entirely.
    if isinstance(result, str):
        payload = result if result.endswith("\n") else result + "\n"
        sys.stdout.buffer.write(payload.encode("utf-8", errors="replace"))
    else:
        sys.stdout.buffer.write(
            (json.dumps(result) + "\n").encode("utf-8", errors="replace"))
    sys.stdout.buffer.flush()
    # An unauthorised start exits NON-ZERO (2, matching farmctl's own refusal
    # code) so a caller that only checks the return code — a shell script, a
    # subagent's `ssh host 'python3 agent.py start ...' && ...` — cannot mistake
    # a refusal for a successful start. Every other command keeps returning 0
    # exactly as before; farmctl parses their JSON, not their rc.
    if isinstance(result, dict) and result.get("status") == "refused":
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
