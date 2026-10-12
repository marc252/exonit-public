#!/usr/bin/env python3
"""Discovery pipeline orchestrator — post-run state machine.

Spawned as a background subprocess by the POST /api/discovery/scan endpoint
once a lot has been successfully enqueued (status='started'). Runs on the
server host.

State machine (in order):
  IDLE → SCANNING → FPP_BACKFILL → EXOMINER_BACKFILL → POSTPROCESS → STOP_NVIDIARACK → LORA → FREE_GPU → FINALIZE → DONE
  Any step can transition to FAILED with a 'detail' field.

State is persisted atomically to --state-file (JSON) so the UI can poll it
via GET /api/discovery/pipeline and it survives a server restart.

Usage:
  python3 scripts/discovery_orchestrator.py \
      --lot-csv data/discovery/s0001_lot4.csv \
      --sector 1

See --help for all options.
"""

import argparse
import contextlib
import csv
import json
import logging
import os
import re
import shlex
import signal
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote, urlsplit, urlunsplit

from exonit import progress as progress_model
from exonit.progress_client import ProgressReporter

ROOT = Path(__file__).resolve().parent.parent

logger = logging.getLogger("discovery_orchestrator")

# ---------------------------------------------------------------------------
# CONSTANTS — auditable, assertable in tests, easy to adjust if infra changes
# ---------------------------------------------------------------------------

#: The worker (name in ops/workers.json) that shares its GPU with the manual
#: gpuq/vLLM LoRA fallback; stopped before LoRA so the GPU is fully available.
#: Deployment-specific -> EXONIT_GPU_HOST_WORKER (empty: the step is a no-op).
#: farmctl stop of a native worker's name also covers its python sidecars.
NVIDIARACK_WORKER_NAME = os.environ.get("EXONIT_GPU_HOST_WORKER", "").strip()

#: farmctl binary path (absolute, runs on <host>).
FARMCTL_BIN = str(ROOT / "bin" / "farmctl")

#: Manual-fallback GPU host (shared-GPU box with a `gpuq` job queue). All of
#: this is deployment config (see .env.example): ssh target, the gpuq binary,
#: the NVML-compat wrapper and the remote lora_via_gpuq.sh script. HARD RULE
#: of the deployment this was written for: ALL GPU work on that box goes
#: through gpuq (+ the wrapper) — never start vLLM/CUDA directly.
GPUQ_SSH_TARGET = os.environ.get("EXONIT_GPUQ_SSH_TARGET", "user@gpu-host")
GPUQ_BIN = os.environ.get("EXONIT_GPUQ_BIN", "gpuq")
GPUQ_NVML_WRAPPER = os.environ.get("EXONIT_GPUQ_NVML_WRAPPER", "").strip()
GPUQ_LORA_SCRIPT = os.environ.get("EXONIT_GPUQ_LORA_SCRIPT", "lora_via_gpuq.sh")

#: Canonical SSH+gpuq command that starts vLLM in a gpuq slot, runs the
#: generate_missing_explanations.py generator, then tears vLLM down
#: (discovery-campaign.md step 12). Layout is asserted by
#: tests/test_discovery_orchestrator.py::test_lora_ssh_cmd_canonical.
_GPUQ_WRAP = f"{GPUQ_NVML_WRAPPER} " if GPUQ_NVML_WRAPPER else ""
LORA_SSH_CMD = (
    f"ssh -o BatchMode=yes -o ConnectTimeout=8 {GPUQ_SSH_TARGET} "
    f'"bash -lc \'{_GPUQ_WRAP}'
    f"{GPUQ_BIN} run --name exonit-lora "
    f"--idle-mib 1000 --max-wait 0 bash {GPUQ_LORA_SCRIPT}'\""
)

#: Phase 3 (2026-09-26): the LoRA step's DEFAULT is now the 5 Mac mini M4
#: fleet (scripts/lora_macs_driver.py, mlx_lm.server + the same
#: /api/explain shim) — <host>/vLLM is no longer on the critical path
#: for every discovery lot (docs/superpowers/specs/2026-09-26-lora-macs-design.md).
#: The command is still OVERRIDABLE: EXONIT_LORA_CMD env var, else the first
#: non-empty non-comment line of data/lora_cmd.override (same file-flag
#: pattern as data/STOP_CHAIN: a live server/supervisor picks the change up
#: on the NEXT lot without a restart, because every lot spawns a fresh
#: orchestrator process that resolves this at import), else LORA_MACS_CMD.
#: Delete the file to fall back to the Macs.
#:
#: Both EXONIT_LORA_CMD and the override file accept a short ALIAS instead
#: of a full command string, so a manual fallback never requires retyping
#: the ssh+gpuq incantation by hand:
#:   gpuq       -> LORA_SSH_CMD      (manual fallback: ssh+gpuq+vLLM;
#:                                    legacy alias name: see LORA_CMD_ALIASES)
#:   llamacpp   -> local llama.cpp on <host>'s Arc iGPU (Phase 2, superseded)
#:   macs       -> LORA_MACS_CMD     (== the default, spelled out explicitly)
#: Anything else is used verbatim as the command (Phase 2 behaviour unchanged).
LORA_CMD_OVERRIDE_FILE = str(ROOT / "data" / "lora_cmd.override")

#: Phase 3 default: scripts/lora_macs_driver.py, run with the same
#: `.venv-server` interpreter as the orchestrator itself (<host>).
LORA_MACS_CMD = f"{sys.executable} {ROOT / 'scripts' / 'lora_macs_driver.py'}"

#: Phase 2 override target, kept as a named alias (`llamacpp`) — local
#: llama.cpp on the Arc iGPU, superseded by the Macs but still wired up.
LORA_LLAMACPP_CMD = f"bash {ROOT / 'ops' / 'lora_via_llamacpp.sh'}"

#: Alias table for EXONIT_LORA_CMD / data/lora_cmd.override (see doc above).
LORA_CMD_ALIASES = {
    "gpuq": LORA_SSH_CMD,
    "<host>": LORA_SSH_CMD,  # farm-literal-ok: legacy alias name still used by data/lora_cmd.override
    "llamacpp": LORA_LLAMACPP_CMD,
    "macs": LORA_MACS_CMD,
}


def _apply_lora_cmd_alias(value: str) -> str:
    return LORA_CMD_ALIASES.get(value.strip().lower(), value)


def _resolve_lora_cmd() -> str:
    env_cmd = os.environ.get("EXONIT_LORA_CMD", "").strip()
    if env_cmd:
        return _apply_lora_cmd_alias(env_cmd)
    try:
        with open(LORA_CMD_OVERRIDE_FILE, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line and not line.startswith("#"):
                    return _apply_lora_cmd_alias(line)
    except OSError:
        pass
    return LORA_MACS_CMD


LORA_CMD = _resolve_lora_cmd()


def _is_nvidiarack_cmd(cmd: str) -> bool:
    """True for the <host>/gpuq/vLLM path (LORA_SSH_CMD, or the
    `<host>` alias resolving to it) — every other command (the Macs
    driver, the llama.cpp override, or any ad-hoc override string) is local
    to this host and gets the killpg-based timeout handling in step_lora
    instead of the ssh-can't-reach-the-remote-job caveat.

    A substring check rather than `cmd == LORA_SSH_CMD` on purpose: matching
    on the script name alone (not the trailing quote) survives
    `_lora_cmd_for_sector`'s positional-argument rewrite (the trailing
    `bash .../lora_via_gpuq.sh'` becomes `lora_via_gpuq.sh 5 50000'` once a
    sector/backlog is appended), which an exact-equality check would not.
    """
    return os.path.basename(GPUQ_LORA_SCRIPT) in cmd


#: 2026-09-20 (operator): on top of the run's own sector, each LORA step also
#: explains up to this many candidates of the historic backlog (every other
#: sector, strong-tier gate first — generator ``--order strong --limit N``),
#: so the ~214k rows left unexplained by the v3_tfopwg promotion drain over
#: the next runs without a separate job. 0 = off. Budget: ~1.15 expl/s on
#: the RTX 5060 Ti → 50 000 ≈ 12 h of LoRA per lot (fleet idle meanwhile),
#: still under LORA_TIMEOUT_SECONDS (24 h). Read from the environment at
#: import like the other knobs — the supervisor's EnvironmentFile is the
#: production source, so a change there needs a supervisor restart.
LORA_BACKLOG_LIMIT = int(os.environ.get("EXONIT_LORA_BACKLOG_LIMIT", "0") or 0)


def _lora_cmd_for_sector(sector: int | None,
                         backlog_limit: int = 0) -> str:
    """The command step_lora actually runs, scoped to *sector* (+ backlog).

    2026-09-20 (s5 lot 1): the generator behind lora_via_gpuq.sh selected
    every unexplained planet_candidate in the DB — 222k after the v3_tfopwg
    recalculation promoted 221k historic rows — not the lot's ~400, i.e. 52 h
    of vLLM against the 24 h step timeout. On the default <host> path the
    sector is appended as a positional argument to lora_via_gpuq.sh, which
    forwards it as ``--sector N`` to generate_missing_explanations.py. An
    override command (EXONIT_LORA_CMD / data/lora_cmd.override) is left
    untouched — it gets EXONIT_LORA_SECTOR in its environment instead (see
    step_lora) — and so is ``sector=None`` (legacy callers: whole-DB scope).

    *backlog_limit* > 0 is appended as a SECOND positional argument
    (``lora_via_gpuq.sh SECTOR BACKLOG``): after the sector pass the script
    runs the generator once more, whole-DB, ``--order strong --limit N``.
    Only meaningful together with a sector (a whole-DB pass already covers
    everything).
    """
    if sector is None or not _is_nvidiarack_cmd(LORA_CMD):
        return LORA_CMD
    marker = f"{GPUQ_LORA_SCRIPT}'"
    assert marker in LORA_CMD, "LORA_SSH_CMD layout changed; update _lora_cmd_for_sector"
    args = f" {int(sector)}"
    if int(backlog_limit) > 0:
        args += f" {int(backlog_limit)}"
    return LORA_CMD.replace(marker, f"{GPUQ_LORA_SCRIPT}{args}'", 1)

#: Default state file location (relative to ROOT, overridable).
DEFAULT_STATE_FILE = str(ROOT / "data" / "discovery" / "pipeline_state.json")

#: Server URL this run reports progress to. Published into the process
#: environment at the top of run() from ``--server-url``, because `.env`
#: defines no EXONIT_SERVER_URL and both ProgressReporter and
#: progress.child_env() read it — left unset, every reporter would be
#: silently disabled and nothing would ever say so.
_SERVER_URL = ""

#: Ordered list of valid states (transition order matters).
#: STOP_WORKERS stops the WHOLE farm before POSTPROCESS so no worker writes to
#: the SQLite DB while the heavy postprocess (veto UPDATEs, TRICERATOPS,
#: scorecard) holds write transactions — eliminating the lock contention that
#: made past closeouts crawl.  STOP_NVIDIARACK afterwards is then idempotent
#: (already stopped) and just guarantees the GPU is free right before LoRA.
#: On AWS, after POSTPROCESS the run takes the AWS_FINALIZE path instead of
#: STOP_NVIDIARACK/LORA/FREE_GPU/FINALIZE (those are home-only steps).
STATES = [
    "IDLE",
    "SCANNING",
    "FPP_BACKFILL",
    "EXOMINER_BACKFILL",
    "STOP_WORKERS",
    "POSTPROCESS",
    "AWS_FINALIZE",    # AWS-only: export bundle → S3 → self-stop (home skips this)
    "STOP_NVIDIARACK",
    "LORA",
    "FREE_GPU",
    "FINALIZE",
    "DONE",
    "FAILED",
]

#: States that are in-progress (not terminal).  When entering one of these,
#: any leftover 'detail' from a previous FAILED run must be cleared.
_PROGRESS_STATES = {"IDLE", "SCANNING", "FPP_BACKFILL", "EXOMINER_BACKFILL",
                    "STOP_WORKERS", "POSTPROCESS", "AWS_FINALIZE",
                    "STOP_NVIDIARACK", "LORA", "FREE_GPU", "FINALIZE"}

#: Number of consecutive non-decreasing polls before declaring ExoMiner stuck.
EXOMINER_STUCK_POLLS = 30

#: SCANNING / FPP_BACKFILL safety nets so the orchestrator can NEVER hang
#: forever (the recurring "hangs at SCANNING after a server restart" bug):
#:  - SCANNING_STUCK_POLLS: consecutive polls with NO net progress (the running
#:    minimum of pending+assigned does not drop) before bailing.  Generous (120
#:    polls = 60 min at the 30 s default) so a slow single TIC or the 30-min
#:    orphan-reassign cycle never trips it — only a truly dead fleet does.
#:  - POLL_CONNECT_ERROR_LIMIT: consecutive failed polls (server unreachable /
#:    malformed JSON) before bailing (20 polls = 10 min).  A transient restart
#:    resets the counter on the first good poll.
#: Both bail with an actionable FAILED detail instead of looping silently.
SCANNING_STUCK_POLLS = 120
#: Consecutive polls with no net progress before a phase tries to heal itself
#: (see _poll_until_drained's stall_heal). At the 30 s poll interval this is
#: ~5 min — early enough that a drift is cleared while the run is still
#: healthy, far enough apart that a genuinely-slow tail is not reconciled on
#: every poll.
STALL_HEAL_POLLS = 10
POLL_CONNECT_ERROR_LIMIT = 20

#: Phase 3 Task 2 (durable orchestration): db.update_discovery_run(heartbeat=True)
#: from _poll_until_drained's loop is throttled to at most once per this many
#: seconds, regardless of --poll-interval, so a tight interval never hammers PG.
HEARTBEAT_MIN_INTERVAL_SECONDS = 30

#: B3 (docs/reviews/2026-07-02-opus48-cleanup-review.md): the subprocess.run
#: calls below had NO timeout at all — one pathological candidate inside
#: discovery_postprocess.py's TRICERATOPS loop (itself now bounded per-
#: candidate, see that script's --fpp-timeout-seconds) could otherwise hang
#: the whole end-of-run tail forever, with the fleet already stopped so
#: nothing was left to notice. These are generous backstops, not tripwires —
#: they exist so a genuinely wedged subprocess eventually fails the run
#: instead of hanging it silently. All overridable via env for ops.
#:
#: POSTPROCESS: discovery_postprocess.py + scorecard.py, each capped
#: separately at this ceiling (6h default).
POSTPROCESS_TIMEOUT_SECONDS = float(
    os.environ.get("EXONIT_POSTPROCESS_TIMEOUT_SECONDS", str(6 * 3600)))
#: FINALIZE completeness compaction (Phase 4a Task 4): scripts/
#: compact_completeness.py drains completeness_staging into bucketed Parquet
#: (MinIO) + PG aggregates. A full 256-bucket pass over real-scale buckets
#: (~720k rows each) benchmarks at ~16-20 min on <host>, so 60 min gives
#: real (>2x) headroom. NON-FATAL either way (see
#: _finalize_compact_completeness below): a genuinely wedged compaction
#: never blocks the run reaching DONE; it just leaves staging for the next
#: pass to absorb.
COMPLETENESS_COMPACTION_TIMEOUT_SECONDS = float(
    os.environ.get("EXONIT_COMPLETENESS_COMPACTION_TIMEOUT_SECONDS", "3600"))
#: FINALIZE completeness compaction, REMOTE fast-path (2026-07-17, "fast completeness
#: compaction" spec): both PG and MinIO for completeness live on
#: the storage host (PG + MinIO) — running the compactor there over loopback
#: instead of from <host> over LAN removes the network round-trip that
#: dominates a big pass. Tighter than the local 3600s default because a
#: wedged/unreachable remote must fall back to local quickly rather than
#: eating most of the local budget first; see _run_compact_remote.
EXONIT_COMPACT_REMOTE_TIMEOUT_SECONDS = float(
    os.environ.get("EXONIT_COMPACT_REMOTE_TIMEOUT_SECONDS", "2700"))
#: Where a FAILED remote compaction's full stdout+stderr is dumped (2026-07-21
#: fix — see _write_compact_remote_fail_log). A module-level path rather than
#: inlined so tests can monkeypatch it to a tmp_path instead of touching the
#: real data/discovery/ directory (which a live discovery run may be writing
#: to concurrently).
_COMPACT_REMOTE_FAIL_LOG_DIR = ROOT / "data" / "discovery"
#: Tail length (chars) logged inline after a remote-compaction failure — the
#: END of stdout/stderr, not the start: a Python traceback's informative
#: part is at the bottom. The previous behaviour kept only the first 500
#: chars, which lost the traceback entirely (2026-07-21 incident: the remote
#: fast-path died with rc=1 after 28 min with no usable diagnosis left).
_COMPACT_REMOTE_FAIL_TAIL_CHARS = 2000
#: FINALIZE completeness compaction, SERVICE fast-path (2026-09-09,
#: exonit-compactd — see docs/superpowers/specs/
#: 2026-09-09-compactd-service-design.md, Component 3). Interval between
#: polls of GET /jobs/<job_id>; kept short-lived and independent per request
#: (never one long-held connection) — the incident this replaces
#: (gotcha_compact_remote_ssh_dies_silently_staging_snowball) was exactly a
#: long-lived SSH session dying silently without keepalive.
EXONIT_COMPACT_SERVICE_POLL_INTERVAL_SECONDS = float(
    os.environ.get("EXONIT_COMPACT_SERVICE_POLL_INTERVAL_SECONDS", "15"))
#: Per-request HTTP timeout (POST /compact or one GET /jobs/<id> poll) for
#: the exonit-compactd service fast-path — NOT the total budget (that reuses
#: EXONIT_COMPACT_REMOTE_TIMEOUT_SECONDS above, shared with the SSH path).
EXONIT_COMPACT_SERVICE_HTTP_TIMEOUT_SECONDS = 30.0
#: LORA: the blocking SSH+gpuq call (vLLM cold-start + full explanation
#: generation) legitimately takes hour-scale — 24h so this is a backstop,
#: not a tripwire on a healthy run.
LORA_TIMEOUT_SECONDS = float(
    os.environ.get("EXONIT_LORA_TIMEOUT_SECONDS", str(24 * 3600)))

#: Phase 3 Task 5: best-effort remote cancel of the 'exonit-lora' gpuq job when
#: step_lora's local TimeoutExpired backstop above fires. shell=True means the
#: local kill only reaches the ssh shell (see KNOWN LIMITATION in step_lora),
#: leaving the remote gpuq job running and wedging the <host> GPU queue
#: for every other consumer (HARD RULE: all <host> GPU work goes through
#: gpuq). Signature verified over ssh on 2026-07-05 (read-only, no job
#: touched):
#:   `gpuq cancel --help` -> "usage: gpuq cancel [-h] [--kill] id" — cancel is
#:   by NUMERIC JOB ID, not by --name; `--kill` is required to SIGTERM a
#:   *running* job's process — without it, cancel only flips the DB row to
#:   'cancelled' and the underlying subprocess keeps running untouched.
#:   `gpuq status --help` -> "usage: gpuq status [-h] [--limit LIMIT]"; each
#:   job line looks like:
#:     "  #231 running exit=None name=exonit-lora created=<iso> started=<iso> note=..."
#:   so the id must be discovered by parsing `gpuq status` for a line with
#:   name=exonit-lora and status in {running, queued}.
LORA_CANCEL_STATUS_SSH_CMD = (
    f"ssh -o BatchMode=yes -o ConnectTimeout=8 {GPUQ_SSH_TARGET} "
    f'"{GPUQ_BIN} status --limit 50"'
)
LORA_CANCEL_SSH_CMD_TEMPLATE = (
    f"ssh -o BatchMode=yes -o ConnectTimeout=8 {GPUQ_SSH_TARGET} "
    f'"{GPUQ_BIN} cancel {{job_id}} --kill"'
)
LORA_CANCEL_TIMEOUT_SECONDS = 60.0
LORA_CANCEL_JOB_NAME = "exonit-lora"
_LORA_STATUS_LINE_RE = re.compile(
    r"^\s*#(?P<id>\d+)\s+(?P<status>\S+)\s+exit=\S+\s+name=(?P<name>\S+)"
)

#: AWS_FINALIZE: bundle export to S3 (FATAL on failure/timeout) and the
#: latest/ mirror sync (non-fatal), each bounded separately (2h default).
AWS_EXPORT_TIMEOUT_SECONDS = float(
    os.environ.get("EXONIT_AWS_EXPORT_TIMEOUT_SECONDS", str(2 * 3600)))
AWS_SYNC_TIMEOUT_SECONDS = float(
    os.environ.get("EXONIT_AWS_SYNC_TIMEOUT_SECONDS", str(2 * 3600)))

#: Fleet-autoscale-by-phase (docs/superpowers/specs/fleet-autoscale-by-phase.md):
#: `farmctl start --phase <phase>` fans out over SSH to up to 10 hosts; each
#: individual agent_call has its own 30s timeout inside farmctl, but the
#: overall subprocess.run needs headroom for the whole fan-out plus slow
#: WSL/Windows SSH round-trips. Generous backstop, not a tripwire — a timeout
#: here is fail-soft (see _scale_fleet): the run continues with whatever
#: fleet is already up.
FLEET_AUTOSCALE_TIMEOUT_SECONDS = float(
    os.environ.get("EXONIT_FLEET_AUTOSCALE_TIMEOUT_SECONDS", "300"))

#: Phase 3 Task 3 (dispatch table + --resume + checkpoint-skip): default FAILED
#: detail text when a step returns a falsy detail string. Preserves the exact
#: pre-Task-3 wording for the two states that had an `or "..."` fallback
#: (SCANNING, FPP_BACKFILL) — every other state passes its detail through
#: unchanged, matching the pre-refactor code path-for-path.
_FAILURE_DEFAULT_DETAIL = {
    "SCANNING": "scanning poll failed",
    "FPP_BACKFILL": "fpp_backfill poll failed",
}

# ---------------------------------------------------------------------------
# State-file helpers (atomic write so UI never reads a half-written file)
# ---------------------------------------------------------------------------


def _load_state(path: str) -> dict:
    """Load state file; return {} on missing/malformed."""
    try:
        return json.loads(Path(path).read_text())
    except Exception:
        return {}


def _save_state(path: str, state: dict) -> None:
    """Atomically write state dict to path (write+rename)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2))
    tmp.replace(p)


#: Phase 3 Task 2 — module-level orchestrator DB state. The orchestrator runs
#: as a standalone script (no ServerDeps/app context), and _persist_state's
#: signature is fixed by the Phase-3 plan (sf, run_id, state, extra, heartbeat
#: — no db-handle parameter), so run() stashes the resolved db path / dry-run
#: gate here once, at the top, exactly like the existing DEFAULT_STATE_FILE-
#: style module constants above. Defaults are the SAFE no-op state (no path,
#: dual-write disabled) so any test that bypasses run() and calls _set_state/
#: _poll_until_drained directly never opens a real DB connection.
_db_path: str | None = None
_pg_dual_write_enabled: bool = False
_db_instance = None  # lazily-built ExoNitDB, cached for the process lifetime


def _get_orchestrator_db():
    """Lazily build (and cache) this process's ExoNitDB handle.

    Returns None if dual-write is disabled (--dry-run — see
    _pg_dual_write_enabled) or _db_path was never set (run() hasn't executed
    yet). Both cases are silent no-ops, never a real DB connection — this is
    the single gate that keeps --dry-run invocations (used pervasively by the
    test suite, often without --db) from ever touching disk/network for PG.
    """
    global _db_instance
    if not _pg_dual_write_enabled or not _db_path:
        return None
    if _db_instance is None:
        from exonit.db import ExoNitDB
        _db_instance = ExoNitDB(_db_path)
    return _db_instance


def _ensure_discovery_run_row(run_id: str, sector: int, lot_csv: str,
                              dry_run: bool, chained: bool,
                              server_url: str) -> None:
    """Best-effort upsert: create the discovery_runs PG row if missing.

    Covers TWO paths uniformly:
      - Backward-compat manual CLI use (no --run-id from the scan endpoint) —
        no row exists yet, so it's created here.
      - The normal server-spawned path — the endpoint already created the row
        before spawning us, so the get-first check makes this a harmless
        no-op instead of hitting create_discovery_run's PK integrity error on
        a duplicate run_id.

    Gated by _get_orchestrator_db() (returns None under --dry-run), so this
    never opens a real DB connection during a dry run. Non-fatal: PG being
    down must never prevent the run from proceeding (the JSON mirror is the
    fallback source of truth regardless).
    """
    try:
        db = _get_orchestrator_db()
        if db is None:
            return
        if db.get_discovery_run(run_id) is None:
            db.create_discovery_run(run_id, sector=sector, lot_csv=lot_csv,
                                    dry_run=dry_run, chained=chained,
                                    server_url=server_url)
    except Exception as exc:
        logger.warning(
            "could not ensure discovery_runs row for run_id=%s (non-fatal): %s",
            run_id, exc)


def _ensure_run_scope(run_id: str, lot_csv: str) -> None:
    """Ensure this lot's TIC set is scoped for run_id (S3, docs/superpowers/
    specs/2026-09-26-enqueue-s1-s4-design.md §3.1 — was
    ``_materialise_lot_tics``, 2026-07-14).

    This is what lets the phase polls below ask "is MY LOT done?" instead of
    "is the whole DB done?" — see _health_work_url() and the table comment in
    scripts/migration/schema_pg.sql for the hang this prevents.

    S3: ``tasks.run_id`` (written at enqueue time by
    ``/api/worker/enqueue_bulk`` when the caller passes ``run_id``, see
    exonit/routes/workers.py) is now the primary scope source — this
    function's OWN job is materialising ``discovery_run_tics``, gated by
    ``EXONIT_RUN_TICS_WRITE`` (default 1, so a deployment that hasn't yet
    flipped to ``EXONIT_RUN_SCOPE=tasks`` keeps writing it exactly as
    before). Set to 0 only once >=2 lots have run clean under
    ``EXONIT_RUN_SCOPE=tasks`` (operator decision §9.2 / the spec's §3.5
    deploy windows W4-W5) — this table is NOT dropped by S3 itself. Logs
    ``db.run_scope_kind(run_id)`` either way so a run's log makes the active
    scope mechanism observable without a separate query.

    Idempotent (INSERT-ignore on the PK), so a --resume that re-enters
    ENQUEUEING re-inserts nothing.

    Non-fatal, same contract as _ensure_discovery_run_row: if PG is down (or
    --dry-run closed _get_orchestrator_db), the run proceeds — the health/work
    scope simply falls back to the global counters, which is exactly the
    pre-2026-07-14 behaviour, not a new failure mode.
    """
    try:
        db = _get_orchestrator_db()
        if db is None:
            return
        if os.environ.get("EXONIT_RUN_TICS_WRITE", "1") == "1":
            with open(lot_csv, newline="") as fh:
                tics = [row["tic_id"] for row in csv.DictReader(fh)
                        if row.get("tic_id")]
            n = db.add_discovery_run_tics(run_id, tics)
            logger.info(
                "run scope: %d TICs materialised for run_id=%s", n, run_id)
        else:
            logger.info(
                "run scope: EXONIT_RUN_TICS_WRITE=0 — skipping "
                "discovery_run_tics materialisation for run_id=%s "
                "(relying on tasks.run_id)", run_id)
        logger.info("run scope: kind=%s for run_id=%s",
                    db.run_scope_kind(run_id), run_id)
    except Exception as exc:
        logger.warning(
            "could not ensure run scope for run_id=%s (non-fatal — phase "
            "polls fall back to GLOBAL counters): %s", run_id, exc)


def _reconcile_stream(server_url: str, stage: str, dry_run: bool) -> None:
    """Top up the *stage* work stream from the DB before we start waiting on it.

    A backfill phase waits for a counter derived from PostgreSQL to reach zero,
    but workers are served from the Redis stream. Those two can disagree, and
    when they do the phase waits forever:

      * a row eligible in PG but in NO queue (a writer that bypassed the
        producer hooks, a late has_spoc flip) can never be served to anyone;
      * a row dead-lettered for a missing metadata entry has left the queue but
        its PG row is untouched, so it keeps counting as pending.

    Both happened on 2026-07-14. Calling this first means the phase never begins
    waiting on work it has no way to hand to a worker.

    Non-fatal, like every other pre-phase step here: if the server refuses, the
    phase proceeds exactly as it did before this existed.
    """
    if dry_run:
        logger.info("dry_run: skipping reconcile_stream(%s)", stage)
        return
    import urllib.error
    import urllib.request
    url = f"{server_url}/api/admin/reconcile_stream?stage={stage}"
    req = urllib.request.Request(url, method="POST", data=b"")
    token = os.environ.get("EXONIT_WORKER_TOKEN")
    if not token:
        # The orchestrator is spawned by the server (Popen) and inherits its
        # env; EXONIT_WORKER_TOKEN has never been in the production .env, so
        # every reconcile came back 401 (seen live 2026-08-01 and 2026-08-05).
        # worker/config.json on the server host carries the same token the
        # fleet authenticates with — use it instead of silently sending
        # nothing.
        try:
            with open(ROOT / "worker" / "config.json", encoding="utf-8") as fh:
                token = json.load(fh).get("token")
        except Exception as exc:
            logger.warning("reconcile(%s): no EXONIT_WORKER_TOKEN and "
                           "worker/config.json unreadable (%s) — request "
                           "will go out unauthenticated", stage, exc)
            token = None
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            body = json.loads(resp.read())
        if body.get("enqueued") or body.get("dlq_cleared"):
            logger.info(
                "reconcile(%s): enqueued=%s dlq_cleared=%s (eligible=%s, "
                "already_queued=%s) — these would otherwise have counted as "
                "pending forever with nothing able to serve them",
                stage, body.get("enqueued"), body.get("dlq_cleared"),
                body.get("eligible"), body.get("already_queued"))
        else:
            logger.info("reconcile(%s): queue already agrees with the DB "
                        "(eligible=%s)", stage, body.get("eligible"))
    except Exception as exc:
        logger.warning("reconcile(%s) failed (non-fatal): %s", stage, exc)


def _recycle_workers(server_url: str, dry_run: bool) -> None:
    """Best-effort trigger of post-run worker recycling (staggered SIGTERM
    self-restart of the A3 uvicorn worker processes), called at the very end
    of FINALIZE. See docs/superpowers/specs/2026-07-17-worker-recycle-postrun.md.

    A short timeout (10s) and a wide try/except, same non-fatal contract as
    every other FINALIZE sub-step and as ``_reconcile_stream`` above:
    recycling reclaims memory in the background over the following minutes,
    it must never hold up or fail a run's close-out.
    """
    if dry_run:
        logger.info("dry_run: skipping worker recycle trigger")
        return
    if not server_url:
        logger.debug("worker recycle trigger skipped: no server_url")
        return
    import urllib.error
    import urllib.request
    url = f"{server_url}/api/admin/recycle-workers"
    req = urllib.request.Request(url, method="POST", data=b"")
    # Same token resolution as the server's own _check_auth (config.worker_token
    # falls back to the built-in default when the env var is absent — reading
    # the env var directly sent NO header in prod and got a 401 on the first
    # live run, 2026-07-18).
    from exonit.config import worker_token
    req.add_header("Authorization", f"Bearer {worker_token()}")
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            body = json.loads(resp.read())
        logger.info("FINALIZE: worker recycle triggered (generation=%s)",
                    body.get("generation"))
    except Exception as exc:
        logger.warning(
            "FINALIZE: worker recycle trigger failed (non-fatal): %s", exc)


def _lot_size(lot_csv: str) -> int:
    """Number of TICs in the lot CSV (0 if unreadable).

    Replaces the hardcoded ``--sample-size 0`` the scorecard step used to pass,
    which is why every discovery scorecard on disk reports sample_size: 0.
    Non-fatal: a bad read costs the metadata field, never the run.
    """
    try:
        with open(lot_csv, newline="") as fh:
            return sum(1 for row in csv.DictReader(fh) if row.get("tic_id"))
    except Exception:
        return 0


def _health_work_url(server_url: str, run_id: str | None) -> str:
    """The /api/health/work URL a phase poll should read.

    nocache=1: the orchestrator must see ground-truth counts. The endpoint is
    cached 10 s for the UI's frequent polls; that stale cache made SCANNING
    read pending=0 immediately after a freshly-enqueued lot and declare
    "drained" without waiting (the gap-fill SCANNING race).

    run_id (2026-07-14): scopes the counters to THIS run's lot. Without it the
    counters are global, so a run waits on candidates from other sectors and
    other lots — on 2026-07-14 a 50k sector-14 run hung at EXOMINER_BACKFILL on
    43 leftover sector-2 candidates that can never be scored, and every future
    run would have hung on the same rows. If the run has no materialised TIC
    set the server falls back to the global counters (and says so in
    payload["scope"]), i.e. exactly the old behaviour.
    """
    url = f"{server_url}/api/health/work?nocache=1"
    return f"{url}&run_id={quote(str(run_id))}" if run_id else url


def _write_checkpoint(run_id: str | None, step: str, data: dict) -> None:
    """Best-effort ``discovery_runs.checkpoints[step]`` write (Phase 3 Task 3).

    Mirrors _ensure_discovery_run_row's non-fatal contract: PG being down (or
    the --dry-run gate closing _get_orchestrator_db()) must never fail an
    orchestrator step. Silent no-op if run_id is falsy.

    Used for two purposes: (1) the three real per-step checkpoints
    (postprocess/lora/finalize) so --resume can skip an already-completed
    expensive step, and (2) a reserved ``_failed_state`` key recording WHICH
    HOME_SEQUENCE state was in flight when a run transitioned to FAILED — the
    PG ``state`` column itself becomes "FAILED" at that point, so without this
    breadcrumb ``--resume --force`` would have no way to know where to
    restart from. Chosen over a schema migration (a new ``failed_state``
    column) as the least invasive option: checkpoints is already a free-form
    JSON blob nothing else reads by key name, so a reserved key is a pure
    addition with zero migration risk.
    """
    if not run_id:
        return
    try:
        db = _get_orchestrator_db()
        if db is None:
            return
        db.set_discovery_run_checkpoint(run_id, step, data)
    except Exception as exc:
        logger.warning(
            "checkpoint write failed for run_id=%s step=%s (non-fatal): %s",
            run_id, step, exc)


def _persist_state(sf: str, run_id: str | None, *, state: str | None = None,
                    extra: dict | None = None, heartbeat: bool = False) -> None:
    """Single choke point for ALL orchestrator state writes (Phase 3 Task 2).

    (1) Writes the JSON mirror file — unchanged atomic write+rename format,
        still the fallback/source of truth for every existing reader (UI,
        autofill poller, double-launch guard). Skipped if extra is None
        (heartbeat-only calls from _poll_until_drained have nothing new to
        write to JSON).
    (2) Best-effort dual-writes to the discovery_runs PG row via
        db.update_discovery_run(). Wrapped in try/except: a down/unreachable
        PG must NEVER fail an orchestrator step — the JSON mirror remains
        authoritative either way. Skipped entirely if run_id is falsy (no run
        minted yet) or _get_orchestrator_db() returns None (the --dry-run gate).

    state=None (used by _merge_state) means "leave the PG row's state column
    unchanged" — matches db.update_discovery_run's own None-means-unchanged
    convention, and mirrors _merge_state's JSON semantics exactly (it never
    touches the 'state' field either).

    Final-review fix F6: ``finished=True`` is passed whenever *state* is a
    terminal state (DONE/FAILED), so ``discovery_runs.finished_at`` gets
    stamped at the SAME choke point every state transition already goes
    through — no caller has to remember to pass it individually.
    """
    if extra is not None:
        _save_state(sf, extra)
    if not run_id:
        return
    try:
        db = _get_orchestrator_db()
        if db is None:
            return
        db.update_discovery_run(
            run_id,
            state=state,
            detail=(extra or {}).get("detail"),
            heartbeat=heartbeat,
            finished=state in ("DONE", "FAILED"),
        )
    except Exception as exc:
        logger.warning(
            "PG dual-write failed for run_id=%s (non-fatal): %s", run_id, exc)


def _progress_url(*, remote: bool = False) -> str:
    """Base URL a progress client should POST to. "" means "do not report".

    A LOCAL child shares this host with the server, so the server's own URL
    works even when it is ``http://localhost:8400`` — which is exactly what
    the server passes. A child on <host> does NOT: ``localhost`` there is
    <host>, and its marks would go to a port nothing is listening on.
    Reporting into the void is worse than not reporting, because the
    dashboard would show a phase that never goes stale and never advances —
    so the remote path stays OFF unless EXONIT_PROGRESS_URL names an address
    the remote host can actually reach.

    Mirrors _compact_remote_dsn's shape: an explicit env override first, then
    a mechanical decision about what the remote side can reach.
    """
    explicit = os.environ.get("EXONIT_PROGRESS_URL", "").strip()
    if explicit:
        return explicit
    base = _SERVER_URL or os.environ.get("EXONIT_SERVER_URL", "").strip()
    if not base:
        return ""
    if remote and (urlsplit(base).hostname or "") in ("localhost", "127.0.0.1", "::1"):
        logger.info(
            "progress: remote child will not report — server_url is %s, which "
            "means the remote host itself. Set EXONIT_PROGRESS_URL to a "
            "LAN-reachable address to enable it.", base)
        return ""
    return base


def _phase_reports_itself(state_name: str) -> bool:
    """Whether the orchestrator posts progress for *state_name* itself.

    Two kinds of phase are excluded. The queue phases already have real
    done/total derived SERVER-side from counters it already computes, and a
    mark from here carries source="post", which wins over source="server" —
    it would replace good numbers with a blank start/finish. FREE_GPU is
    excluded because its whole body is one logger.info: vLLM is torn down by
    lora_via_gpuq.sh's EXIT trap, so there is nothing happening to report.
    """
    return (state_name not in progress_model.QUEUE_PHASES
            and state_name != "FREE_GPU")


def _phase_reporter(phase: str, run_id: str | None, **kwargs):
    """A reporter for a phase THIS process executes itself.

    Children get their identity from the environment; these steps have no
    child to spawn, so the identity is passed explicitly rather than by
    mutating this long-lived process's own os.environ. Returns None when
    there is no run_id, or if the client cannot even be constructed —
    telemetry never breaks a run.
    """
    if not run_id:
        return None
    try:
        return ProgressReporter(run_id=run_id, phase=phase, **kwargs)
    except Exception:
        return None


def _set_state(path: str, state_name: str, *, fresh: bool = False,
                run_id: str | None = None, **extra) -> dict:
    """Transition to a new state, merge extra fields, persist (JSON + PG).

    With ``fresh=True`` the state starts EMPTY instead of loading the previous
    file — used only at the start of a run (SCANNING) so result fields from a
    prior run (finished_at / run_id / scorecard paths / a stray note) never
    linger in the UI mid-run. Result fields can't be cleared on every progress
    state because POSTPROCESS/FINALIZE legitimately set and read run_id and the
    scorecard paths.

    For non-terminal (progress) states, any leftover 'detail' field from a
    previous FAILED run is cleared so the UI never shows a stale error.
    For FAILED, the caller provides detail via **extra.

    Phase 3 Task 2: also best-effort dual-writes the same state to PG (see
    _persist_state). ``run_id`` need only be passed explicitly on the FIRST
    call of a run (run()'s initial fresh=True SCANNING transition) — once
    written, it persists in the JSON file, so every subsequent call
    (fresh=False, reloading that file) picks it up automatically from
    ``current.get("run_id")`` without having to re-pass it.

    Final-review fix F2: an explicitly-passed ``run_id`` now ALWAYS wins
    (explicit-wins), replacing the old write-once ``if run_id and "run_id"
    not in current`` guard. Write-once silently no-op'd whenever the mirror
    file already carried a (possibly STALE, belonging to a DIFFERENT run)
    ``run_id`` — exactly the case on a ``--resume`` if another run (e.g. a
    chained/autofill lot) wrote ``pipeline_state.json`` in between: resuming
    run A would leave run B's identity in the mirror forever. Harmless for
    every existing caller (run() always passes the SAME run_id on every
    loop iteration, so re-writing it is a same-value no-op); this also
    restores step_postprocess's legacy explicit-overwrite semantics that the
    write-once guard had silently blocked.
    """
    current = {} if fresh else _load_state(path)
    current["state"] = state_name
    current["updated_at"] = datetime.now(timezone.utc).isoformat()
    if state_name in _PROGRESS_STATES:
        # Clear stale detail so the UI doesn't show a previous failure message.
        current.pop("detail", None)
    current.update(extra)
    if run_id:
        current["run_id"] = run_id
    _persist_state(path, run_id or current.get("run_id"),
                   state=state_name, extra=current)
    logger.info("state → %s", state_name)
    return current


def _merge_state(path: str, *, run_id: str | None = None, **extra) -> None:
    """Merge extra fields into the state file WITHOUT changing the 'state' field.

    Used to persist ancillary data (e.g. scorecard paths) mid-step without
    emitting a spurious state transition that would confuse the UI or tests.

    Phase 3 Task 2: also dual-writes to PG with state=None (leaves the PG row's
    state column untouched — the exact same "don't change state" semantics
    as the JSON side).
    """
    current = _load_state(path)
    current["updated_at"] = datetime.now(timezone.utc).isoformat()
    current.update(extra)
    if run_id and "run_id" not in current:
        current["run_id"] = run_id
    _persist_state(path, run_id or current.get("run_id"), extra=current)


# ---------------------------------------------------------------------------
# Individual pipeline steps (each is independently testable)
# ---------------------------------------------------------------------------


def _poll_until_drained(
    url: str,
    poll_interval: int,
    *,
    label: str,
    fetch_metric,
    is_drained,
    drained_log: str,
    stuck_limit: int,
    stuck_detail,
    bail_on_connect_errors: bool,
    connect_error_limit: int = POLL_CONNECT_ERROR_LIMIT,
    on_stuck=None,
    stall_heal=None,
    stall_heal_polls: int = STALL_HEAL_POLLS,
    state_file: str | None = None,
    run_id: str | None = None,
) -> tuple[bool, str]:
    """Poll *url* every *poll_interval* seconds until *is_drained* is True.

    D4 (docs/reviews/2026-07-02-opus48-cleanup-review.md): shared skeleton for
    step_scanning / step_fpp_backfill / step_exominer_backfill — the same
    urllib.request + JSON-parse + running-minimum stuck-detection +
    connect-error-limit loop was hand-copied ~75 lines × 3. Each step
    wrapper supplies only what genuinely differs and is responsible for its
    own top-of-function log line, dry_run short-circuit, and any
    step-specific pre-check (e.g. EXOMINER_BACKFILL's env-disabled skip) —
    those differ in ordering/shape enough between steps (dry_run must be
    checked BEFORE EXOMINER_BACKFILL's disabled-skip, for instance) that
    folding them in here would obscure more than it dedups. BEHAVIOR-NEUTRAL:
    every log line/detail string produced below is byte-identical to the
    pre-extraction per-step copies (verified against
    tests/test_discovery_orchestrator.py + tests/test_aws_finalize.py, which
    assert on several of these substrings and on exact poll call-counts).

    Parameters
    ----------
    url : the /api/health/work?nocache=1 URL to poll (identical for all 3
        steps today, but passed in rather than hardcoded).
    poll_interval : seconds between polls.
    label : log-line / message prefix, e.g. "SCANNING", "FPP_BACKFILL".
    fetch_metric : callable(data: dict) -> (metric: int, log_suffix: str, extra: dict)
        Extracts the field(s) to watch from the parsed JSON response.
        *metric* is what "is_drained"/stuck-detection track; *log_suffix* is
        appended to "{label}: " for the per-poll INFO log; *extra* carries
        whatever additional fields the step's own stuck message needs.
    is_drained : callable(metric) -> bool — drained predicate.
    drained_log : suffix for the "{label}: {drained_log}" success log line.
    stuck_limit : consecutive non-decreasing polls before declaring stuck.
    stuck_detail : callable(extra, minutes) -> str — the FAILED/stuck detail
        message (step-specific wording, e.g. mentions pending vs actionable).
    bail_on_connect_errors : if True, bail with (False, detail) after
        *connect_error_limit* consecutive poll errors (SCANNING/FPP_BACKFILL).
        If False, poll errors are logged and retried forever
        (EXOMINER_BACKFILL's pre-extraction behaviour — no error limit).
    connect_error_limit : consecutive poll errors before bailing (only used
        when bail_on_connect_errors is True).
    stall_heal : optional callable(polls: int) -> None, called every
        *stall_heal_polls* consecutive polls with no net progress. For the
        backfill phases this re-runs the stream reconcile, which is the ONLY
        thing that can clear a PG↔Redis drift once the phase is already
        waiting. See step_fpp_backfill for why a once-on-entry reconcile is
        not enough. Best-effort by contract: it must never raise, and the
        stuck counter is deliberately NOT reset by it — healing that does not
        move the metric must still end in the stuck bail.
    stall_heal_polls : how many stalled polls between stall_heal attempts.
    on_stuck : optional callable(extra, minutes, msg) -> (bool, str) | None.
        Called when the stuck limit is reached, after the default (False,
        msg) has been logged as an error. Returning None keeps the default
        (False, msg); returning a tuple overrides it (EXOMINER_BACKFILL's
        AWS special-case: stuck is expected there, so it returns (True, ...)).
    state_file, run_id : Phase 3 Task 2 — when run_id is given, each loop
        iteration best-effort heartbeats via
        db.update_discovery_run(run_id, heartbeat=True), throttled to at most
        once per HEARTBEAT_MIN_INTERVAL_SECONDS regardless of poll_interval.
        Proves the ORCHESTRATOR process itself is alive, independent of
        whether the server poll below succeeds. Omitted/None (the default) —
        e.g. direct step_* tests whose state file carries no run_id — means
        no heartbeat is attempted at all.
    """
    import urllib.request

    last_min: int | None = None
    stuck_count = 0
    error_count = 0
    last_heartbeat_mono = 0.0

    while True:
        if run_id:
            now_mono = time.monotonic()
            if now_mono - last_heartbeat_mono >= HEARTBEAT_MIN_INTERVAL_SECONDS:
                _persist_state(state_file, run_id, heartbeat=True)
                last_heartbeat_mono = now_mono

        try:
            # timeout=60 (was 10, 2026-07-12, progress/impl-healthwork-perf.md):
            # /api/health/work?nocache=1 is a stats aggregate over the whole
            # candidates/tasks tables, not a ping — under load (e.g. the OR-
            # gate FPP-backfill queue explosion) it can legitimately take
            # several seconds. A 10 s timeout let one slow poll count toward
            # POLL_CONNECT_ERROR_LIMIT and, after 20 of them (10 min), killed
            # a live 100k-star run. 60 s is defence in depth on top of the
            # endpoint's own fix; POLL_CONNECT_ERROR_LIMIT is unchanged so a
            # genuinely dead server is still caught within the same 10 min.
            with urllib.request.urlopen(url, timeout=60) as resp:
                data = json.loads(resp.read())
            error_count = 0  # a good poll resets the connection-error streak
            metric, log_suffix, extra = fetch_metric(data)
            logger.info("%s: %s", label, log_suffix)

            if is_drained(metric):
                logger.info("%s: %s", label, drained_log)
                return True, ""

            if last_min is None or metric < last_min:
                last_min = metric
                stuck_count = 0
            else:
                stuck_count += 1
                # Self-heal long before the stuck bail. A backfill counter that
                # stops moving is usually not "the fleet is dead" — it is the
                # last few rows sitting in PG with nothing in the Redis stream
                # to serve them (2026-08-01: FPP sat at actionable=7 with the
                # fpp stream at queued=0 and 11 items dead-lettered, and would
                # have burned the full 68-minute stuck budget before FAILING a
                # run that was otherwise finished). Reconciling on entry only
                # covers drift that already existed when the phase began; this
                # covers drift that appears while it waits.
                if (stall_heal is not None and stall_heal_polls > 0
                        and stuck_count % stall_heal_polls == 0):
                    minutes = (stuck_count * poll_interval) // 60
                    logger.warning(
                        "%s: no net progress for %d min (metric=%s) — "
                        "attempting stream reconcile", label, minutes, metric)
                    try:
                        stall_heal(stuck_count)
                    except Exception:
                        logger.warning("%s: stall heal failed", label,
                                       exc_info=True)
                if stuck_count >= stuck_limit:
                    minutes = (stuck_count * poll_interval) // 60
                    msg = stuck_detail(extra, minutes)
                    logger.error(msg)
                    if on_stuck is not None:
                        override = on_stuck(extra, minutes, msg)
                        if override is not None:
                            return override
                    return False, msg

        except Exception as exc:
            if bail_on_connect_errors:
                error_count += 1
                logger.warning("%s: poll error %d/%d (will retry): %s",
                               label, error_count, connect_error_limit, exc)
                if error_count >= connect_error_limit:
                    minutes = (error_count * poll_interval) // 60
                    msg = (
                        f"{label}: server unreachable / bad response for {error_count} "
                        f"consecutive polls ({minutes} min) — last error: {exc}"
                    )
                    logger.error(msg)
                    return False, msg
            else:
                logger.warning("%s: poll error (will retry): %s", label, exc)

        time.sleep(poll_interval)


#: How long _verify_fleet waits for the started instances to show up online
#: in /api/worker/list before declaring them missing (first pass; the
#: post-retry pass uses the same value). Registration is seconds, but a Mac
#: waking from sleep or a slow WSL host can take a while.
FLEET_VERIFY_TIMEOUT_SECONDS = int(os.environ.get(
    "EXONIT_FLEET_VERIFY_TIMEOUT", "180"))
FLEET_VERIFY_POLL_SECONDS = 15


def _expected_fleet_names(phase: str, manifest_path: str | None = None) -> set:
    """Instance names ops/workers.json sizes for `phase`.

    Resolved by importing bin/farmctl and calling ITS `_iter_instances` —
    single source of truth, no duplicated sizing logic here (the sizing rules
    have already bitten once: see gotcha-farmctl-instances-zero-disabled-fpp,
    2026-08-05, where a spec-compliant farmctl silently dropped 62 FPP
    workers because nobody VERIFIED the started fleet against the manifest).
    """
    from importlib.machinery import SourceFileLoader
    from importlib.util import module_from_spec, spec_from_loader
    loader = SourceFileLoader("_exonit_farmctl_for_verify", FARMCTL_BIN)
    spec = spec_from_loader("_exonit_farmctl_for_verify", loader)
    module = module_from_spec(spec)
    loader.exec_module(module)
    if manifest_path is not None:
        module.MANIFEST = Path(manifest_path)
    manifest = module.load_manifest()
    names: set = set()
    for worker in manifest["workers"]:
        if hasattr(module, "_expected_online_names"):
            # sidecar_fpp hosts register `<name>-py-i` Python workers, not the
            # bare name _iter_instances yields as farmctl's start/stop unit —
            # see farmctl._expected_online_names (2026-08-07).
            names |= module._expected_online_names(worker, phase=phase)
        else:
            for iname, _idx, _total in module._iter_instances(worker, phase=phase):
                names.add(iname)
    return names


def _online_worker_names(server_url: str) -> set:
    """Worker names currently online per the server's heartbeat registry."""
    import urllib.request
    with urllib.request.urlopen(
            f"{server_url}/api/worker/list", timeout=30) as resp:
        payload = json.loads(resp.read())
    workers = payload["workers"] if isinstance(payload, dict) else payload
    return {w["name"] for w in workers
            if isinstance(w, dict)
            and (w.get("online") or w.get("status") == "online")}


def _verify_fleet(phase: str, server_url: str,
                  deadline_seconds: float) -> tuple[set, set]:
    """Poll until every manifest-expected instance is online, or time out.

    Returns (expected, missing). Missing hosts that are legitimately off
    (a sleeping laptop) stay in `missing` — that is exactly the information
    the operator wants surfaced, not noise to suppress.
    """
    expected = _expected_fleet_names(phase)
    deadline = time.time() + deadline_seconds
    missing = set(expected)
    while True:
        try:
            missing = expected - _online_worker_names(server_url)
        except Exception as exc:
            logger.warning("fleet verify(%s): worker/list unreachable "
                           "(non-fatal): %s", phase, exc)
        if not missing or time.time() >= deadline:
            return expected, missing
        time.sleep(FLEET_VERIFY_POLL_SECONDS)


def _scale_fleet(phase: str, dry_run: bool,
                 server_url: str | None = None) -> tuple[bool, str]:
    """Scale the worker farm to the ops/workers.json sizing declared for `phase`.

    Called at the start of step_scanning ("scan") and step_fpp_backfill
    ("fpp") — see docs/superpowers/specs/fleet-autoscale-by-phase.md. `farmctl
    start` is idempotent (only starts instances that aren't already running),
    so calling this on every entry into a phase — including a --resume that
    lands mid-phase — never duplicates a live worker (this also resolves the
    "chained runs start with an empty fleet" footgun: SCANNING now
    auto-starts it).

    NON-FATAL by design, same policy as step_stop_workers/step_stop_nvidiarack:
    a farmctl failure/timeout/exception must never kill the run — a slow run
    on whatever fleet is already up beats a dead one. Failure details carry a
    "(non-fatal)" marker so the caller can surface them into the pipeline
    state file for the UI.

    No-op fast paths (all return (True, <reason>) without touching farmctl):
      - EXONIT_FLEET_AUTOSCALE=0 (kill-switch; default is ON).
      - AWS deployment (the fleet is managed via terminate_fleet / the AWS
        worker API, never via farmctl SSH).
      - dry_run (mirrors every other step's dry-run skip).
    """
    if os.environ.get("EXONIT_FLEET_AUTOSCALE", "1") == "0":
        logger.info("_scale_fleet(%s): skipped (EXONIT_FLEET_AUTOSCALE=0)", phase)
        return True, "fleet autoscale disabled (EXONIT_FLEET_AUTOSCALE=0)"

    from exonit.config import deployment as _deployment
    if _deployment() == "aws":
        logger.info("_scale_fleet(%s): skipped (aws deployment — fleet managed "
                    "via terminate_fleet, not farmctl)", phase)
        return True, "fleet autoscale skipped (aws deployment)"

    # --all is REQUIRED: farmctl's argparse only starts named workers unless
    # --all is given (bin/farmctl select_workers sys.exit()s "specify worker
    # names or use --all" otherwise, rc=1) — without it this call is a
    # complete, silently-swallowed no-op (the fail-soft path below turns that
    # rc=1 into a "(non-fatal)" warning and the run proceeds with an EMPTY
    # fleet). Regression test: test_calls_farmctl_start_with_all_and_phase.
    cmd = [FARMCTL_BIN, "start", "--all", "--phase", phase]
    logger.info("_scale_fleet(%s): %s", phase, " ".join(cmd))

    if dry_run:
        logger.info("dry_run: skipping farmctl start --phase %s", phase)
        return True, f"dry_run: would run farmctl start --phase {phase}"

    try:
        # The orchestrator IS a legitimate fleet starter, so it opts in to the
        # farmctl start guard (bin/farmctl START_GUARD_ENV) explicitly. The
        # guard exists because automated agents started the real fleet three
        # times on 2026-07-13/14 despite written instructions not to; a caller
        # that cannot name itself here cannot start 83 workers by accident.
        env = dict(os.environ, EXONIT_FARMCTL_ALLOW_START="1")
        result = subprocess.run(cmd, capture_output=True, text=True,
                                timeout=FLEET_AUTOSCALE_TIMEOUT_SECONDS,
                                env=env)
        farmctl_detail = ""
        if result.returncode != 0:
            detail = (result.stderr or result.stdout or "")[:300]
            logger.warning("_scale_fleet(%s): farmctl rc=%d (non-fatal): %s",
                           phase, result.returncode, detail)
            farmctl_detail = (f"farmctl start --phase {phase} "
                              f"rc={result.returncode} (non-fatal): {detail}")
        else:
            logger.info("_scale_fleet(%s): farmctl OK", phase)

        if server_url is None:
            return True, farmctl_detail

        # Verify the fleet actually came up — farmctl's rc alone has already
        # proven insufficient twice (2026-08-05: rc=2 was warning noise while
        # 62 FPP workers were silently absent for a different reason). One
        # retry for stragglers, then surface whatever is still missing.
        expected, missing = _verify_fleet(phase, server_url,
                                          FLEET_VERIFY_TIMEOUT_SECONDS)
        if missing:
            logger.warning(
                "fleet verify(%s): %d/%d online after %ds; retrying farmctl "
                "for: %s", phase, len(expected) - len(missing), len(expected),
                FLEET_VERIFY_TIMEOUT_SECONDS, ", ".join(sorted(missing)))
            subprocess.run(cmd, capture_output=True, text=True,
                           timeout=FLEET_AUTOSCALE_TIMEOUT_SECONDS, env=env)
            expected, missing = _verify_fleet(phase, server_url,
                                              FLEET_VERIFY_TIMEOUT_SECONDS)
        if missing:
            verify_detail = (
                f"fleet verify --phase {phase}: "
                f"{len(expected) - len(missing)}/{len(expected)} online; "
                f"missing after retry (non-fatal): "
                f"{', '.join(sorted(missing))}")
            logger.warning("_scale_fleet(%s): %s", phase, verify_detail)
            return True, "; ".join(x for x in (farmctl_detail, verify_detail) if x)
        logger.info("fleet verify(%s): all %d expected instances online",
                    phase, len(expected))
        return True, ""
    except Exception as exc:
        logger.warning("_scale_fleet(%s) exception (non-fatal): %s", phase, exc)
        return True, f"farmctl start --phase {phase} exception (non-fatal): {exc}"


def step_scanning(server_url: str, state_file: str, poll_interval: int, dry_run: bool) -> tuple[bool, str]:
    """Poll /api/health/work until primary_tasks are fully drained.

    Returns (True, "") when pending == assigned == 0.
    Returns (False, detail) — instead of looping forever — when either:
      * the running minimum of pending+assigned shows no net progress for
        SCANNING_STUCK_POLLS consecutive polls (orphaned assigned tasks on a
        dead/paused fleet — the classic "hangs at SCANNING" after a restart), or
      * the server is unreachable / returns bad JSON for POLL_CONNECT_ERROR_LIMIT
        consecutive polls.
    Transient errors and slow-but-real progress are tolerated (counters reset).
    """
    # Fleet-autoscale-by-phase: auto-start the SCAN-sized fleet on entry —
    # resolves "chained runs start with an empty fleet"
    # (gotcha_chained_runs_need_fleet_restart). Non-fatal; a failure is
    # recorded into the state file for the UI but never blocks scanning.
    _scale_ok, _scale_detail = _scale_fleet("scan", dry_run, server_url=server_url)
    if _scale_detail and "(non-fatal)" in _scale_detail:
        _merge_state(state_file, fleet_autoscale_warning=_scale_detail)

    run_id = _load_state(state_file).get("run_id")
    url = _health_work_url(server_url, run_id)
    logger.info("SCANNING: polling %s every %ds", url, poll_interval)

    if dry_run:
        logger.info("dry_run: skipping actual poll, treating as drained")
        return True, ""

    def _metric(data):
        primary = data.get("primary_tasks", {})
        pending = primary.get("pending", 0)
        assigned = primary.get("assigned", 0)
        return (pending + assigned, f"pending={pending} assigned={assigned}",
                {"pending": pending, "assigned": assigned})

    def _stuck(extra, minutes):
        return (
            f"SCANNING stuck: pending={extra['pending']} assigned={extra['assigned']} with no "
            f"net progress for {minutes} min — likely orphaned assigned tasks "
            f"on a dead/paused fleet. Requeue stale tasks or restart the "
            f"workers, then re-run the orchestrator."
        )

    # Phase 3 Task 2: run_id (if the state file already carries one) enables
    # the heartbeat inside _poll_until_drained — and, since 2026-07-14, scopes
    # the counters to this lot (read once above, for the URL).
    return _poll_until_drained(
        url, poll_interval,
        label="SCANNING",
        fetch_metric=_metric,
        is_drained=lambda m: m == 0,
        drained_log="primary tasks drained",
        stuck_limit=SCANNING_STUCK_POLLS,
        stuck_detail=_stuck,
        bail_on_connect_errors=True,
        state_file=state_file,
        run_id=run_id,
    )


def step_fpp_backfill(server_url: str, state_file: str, poll_interval: int, dry_run: bool) -> tuple[bool, str]:
    """Poll /api/health/work until FPP backfill is drained (actionable == 0).

    NOTE: We do NOT wait for ExoMiner backfill. ExoMiner Docker runs on the
    server for SPOC candidates; FFI-only TGLC candidates will never produce
    a Docker ExoMiner score. Waiting would block indefinitely.
    (discovery-campaign.md step 12: 'trigger = primary + FPP backfill at 0;
    ExoMiner Docker is left to run in the background.')

    Same safety nets as step_scanning: returns (False, detail) rather than
    looping forever if FPP backfill makes no net progress for
    SCANNING_STUCK_POLLS polls, or the server is unreachable for
    POLL_CONNECT_ERROR_LIMIT consecutive polls.
    """
    # Fleet-autoscale-by-phase: scale the fleet UP to the FPP sizing on entry
    # (TRICERATOPS is far lighter per-instance than BLS+injection — ~2x
    # instances measured 07-11/07-12, see the spec). Non-fatal; a failure is
    # recorded into the state file for the UI but never blocks the backfill.
    _scale_ok, _scale_detail = _scale_fleet("fpp", dry_run, server_url=server_url)
    if _scale_detail and "(non-fatal)" in _scale_detail:
        _merge_state(state_file, fleet_autoscale_warning=_scale_detail)

    # Before waiting on a PG-derived counter, make sure the Redis stream the
    # workers are actually served from can move it (see _reconcile_stream).
    _reconcile_stream(server_url, "fpp", dry_run)

    run_id = _load_state(state_file).get("run_id")
    url = _health_work_url(server_url, run_id)
    logger.info("FPP_BACKFILL: polling %s every %ds", url, poll_interval)

    if dry_run:
        logger.info("dry_run: skipping FPP poll, treating as drained")
        return True, ""

    def _metric(data):
        fpp = data.get("fpp_backfill", {})
        actionable = fpp.get("actionable", 0)
        return actionable, f"actionable={actionable}", {"actionable": actionable}

    def _stuck(extra, minutes):
        return (
            f"FPP_BACKFILL stuck: actionable={extra['actionable']} with no net "
            f"progress for {minutes} min — workers may be down or unable to "
            f"compute FPP. Check the fleet, then re-run the orchestrator."
        )

    return _poll_until_drained(
        url, poll_interval,
        label="FPP_BACKFILL",
        fetch_metric=_metric,
        is_drained=lambda m: m == 0,
        drained_log="drained",
        stuck_limit=SCANNING_STUCK_POLLS,
        stuck_detail=_stuck,
        bail_on_connect_errors=True,
        # The entry reconcile above only covers drift that existed when the
        # phase began. Drift that appears DURING the wait — a row dead-lettered
        # after its metadata lookup kept failing, a late has_spoc flip — is
        # exactly as fatal and used to need a human with curl.
        stall_heal=lambda _polls: _reconcile_stream(server_url, "fpp", dry_run),
        state_file=state_file,
        run_id=run_id,
    )


def step_exominer_backfill(
    server_url: str, state_file: str, poll_interval: int, dry_run: bool
) -> tuple[bool, str]:
    """Poll /api/health/work until ExoMiner backfill is drained (pending == 0).

    The server-side Docker ExoMiner hook marks all strong candidates (including
    FFI-only ones, even if as -1.0) once primary tasks are done.  Draining this
    queue ensures ExoMiner scores are present before postprocess/scorecard runs.

    Returns (True, "") on success.
    Returns (False, detail) if the backfill appears stuck (pending does not
    decrease for EXOMINER_STUCK_POLLS consecutive polls = 15 min at default
    30s interval).  This prevents the orchestrator hanging forever when the
    Docker hook is misconfigured (EXONIT_ENABLE_EXOMINER_AUTO=1 missing or
    `docker ps | grep exominer` shows no running container).
    """
    _reconcile_stream(server_url, "exominer", dry_run)

    run_id = _load_state(state_file).get("run_id")
    url = _health_work_url(server_url, run_id)
    logger.info("EXOMINER_BACKFILL: polling %s every %ds", url, poll_interval)

    if dry_run:
        logger.info("dry_run: skipping ExoMiner poll, treating as drained")
        return True, ""

    # When the server-side ExoMiner backfill is disabled (EXONIT_EXOMINER_BACKFILL != "1"),
    # the queue will never drain because the Docker hook is not running.  This is the
    # expected state for home FFI-discovery runs started with the stopgap flag, and is
    # identical to the AWS situation: exominer_ffi_score (written always-on by the worker
    # during the scan phase) is the authoritative score for FFI-only TICs.  Skip the
    # phase entirely so the run can proceed to postprocess/scorecard/LoRA.
    if os.environ.get("EXONIT_EXOMINER_BACKFILL", "0") != "1":
        logger.info(
            "EXOMINER_BACKFILL: EXONIT_EXOMINER_BACKFILL!=1 — skipping phase; "
            "exominer_ffi_score (scan, always-on) is authoritative (same as AWS path)"
        )
        return True, "exominer backfill disabled (EXONIT_EXOMINER_BACKFILL!=1) — skipped"

    def _metric(data):
        em = data.get("exominer_backfill", {})
        pending = em.get("pending", 0)
        assigned = em.get("currently_assigned", 0)
        # Wait on 'actionable', not 'pending' (2026-07-14). A candidate whose
        # FITS metadata is not cached is held back by the serve loop on
        # purpose — no serve, no ack — so it can NEVER leave 'pending', and a
        # run waiting for pending==0 waits forever (that is exactly how the
        # 2026-07-14 run hung). 'actionable' counts only what a worker can
        # actually be served. .get(..., pending) keeps an older server (which
        # reports no 'actionable' key) on its previous behaviour rather than
        # reading a missing key as 0 and declaring the phase drained.
        actionable = em.get("actionable", pending)
        unrecoverable = em.get("unrecoverable", 0)
        suffix = f"actionable={actionable} currently_assigned={assigned}"
        if unrecoverable:
            suffix += f" (unrecoverable={unrecoverable}, not waited on)"
        return (actionable, suffix,
                {"pending": pending, "actionable": actionable,
                 "unrecoverable": unrecoverable})

    def _stuck(extra, minutes):
        # Report BOTH: 'actionable' is what we wait on, 'pending' is what an
        # operator sees in the UI. When they differ, the gap IS the diagnosis
        # (candidates whose metadata never cached and that can never be served).
        return (
            f"EXOMINER_BACKFILL stuck: actionable={extra['actionable']}"
            f" pending={extra['pending']} for {minutes} min"
            " — Docker hook may not be running"
            " (check EXONIT_ENABLE_EXOMINER_AUTO=1 and"
            " `docker ps | grep exominer`)"
        )

    def _on_stuck(extra, minutes, msg):
        # On AWS there is no SPOC ExoMiner (no Docker/model on workers),
        # so the queue will never drain to 0.  That is expected: the
        # exominer_ffi_score (written always-on during the scan phase) is
        # the authoritative score for FFI-only TICs.  Treat stuck == done.
        # On home the Docker hook SHOULD be running — stuck is a real
        # failure and must stay fatal (return None -> default False/msg).
        from exonit.config import deployment as _deployment
        if _deployment() == "aws":
            aws_msg = (
                msg + " — proceeding: ExoMiner not available on AWS,"
                " exominer_ffi_score (scan, always-on) is authoritative"
            )
            logger.warning("EXOMINER_BACKFILL: %s", aws_msg)
            return True, aws_msg
        return None

    return _poll_until_drained(
        url, poll_interval,
        label="EXOMINER_BACKFILL",
        fetch_metric=_metric,
        is_drained=lambda m: m == 0,
        drained_log="drained",
        stuck_limit=EXOMINER_STUCK_POLLS,
        stuck_detail=_stuck,
        bail_on_connect_errors=False,
        # Same PG↔Redis drift, same cure as FPP_BACKFILL above — this phase has
        # its own stream and its own entry-time reconcile.
        stall_heal=lambda _polls: _reconcile_stream(server_url, "exominer",
                                                    dry_run),
        on_stuck=_on_stuck,
        state_file=state_file,
        run_id=run_id,
    )


def _lot_number_from_csv(lot_csv: str) -> str:
    """Unique lot identifier for the run_id, from the CSV filename.

    - Auto lots ``s0001_lot_20260531T082820.csv`` → ``'20260531T082820'``
      (timestamp; unique per run, so two runs the same day no longer collide).
    - Numbered lots ``s0001_lot4.csv`` → ``'4'``.
    - Fallback → ``'X'``.
    """
    import re
    name = Path(lot_csv).name
    m = re.search(r"lot_(\d{8}T\d{6})", name)   # auto lots → timestamp
    if m:
        return m.group(1)
    m = re.search(r"lot(\d+)", name)             # numbered lots → number
    return m.group(1) if m else "X"


def _lot_prefix_from_csv(lot_csv: str) -> str:
    """``"lot"`` or ``"deep_lot"`` run_id segment prefix, from the lot CSV.

    FIX-1 (final-review, spec §4 deep-pass-multisector design): deep-rescan
    lots must mint a DISTINCTIVE run_id (``discovery_s0014_deep_lot<ts>_
    <date>``), separable from normal campaign lots at a glance. Deep lot
    CSVs are named ``s<sector>_deep_lot_<ts>.csv`` (see this module's own
    lot writer / discovery_run.py's --force-rescan path); anything else is
    a normal campaign lot.

    Kept separate from ``_lot_number_from_csv`` on purpose: that helper's
    bare ts/number return is relied on byte-identically by the existing
    tests below (test_lot_number_*), so the deep/normal distinction is
    layered on top here rather than changing its contract.

    MUST match exonit/routes/discovery.py's twin helper of the same name
    EXACTLY -- both-sites-must-agree invariant: the server mints run_id
    ONCE (via that helper) before spawning the orchestrator; this CLI-
    fallback derivation (used when no --run-id is passed) must never
    disagree with it for the same lot_csv.
    """
    return "deep_lot" if "_deep_lot_" in Path(lot_csv).name else "lot"


def _derive_run_id(sector, lot_csv: str) -> str:
    """Canonical run_id: discovery_s<sector:04d>_lot<lot>_<YYYY-MM-DD> (UTC
    today); deep-rescan lots mint discovery_s<sector:04d>_deep_lot<lot>_
    <YYYY-MM-DD> instead (FIX-1, spec §4 — see _lot_prefix_from_csv).

    S3 unification (docs/superpowers/specs/2026-09-26-enqueue-s1-s4-
    design.md §3.1): thin wrapper over the single canonical implementation,
    ``exonit.discovery_state.derive_run_id`` — kept as a same-name,
    same-signature module-level function (not inlined at call sites) so
    ``tests/test_discovery_orchestrator.py``'s
    ``patch.object(orch, "_derive_run_id")``/direct-call usages keep
    working unchanged.
    """
    from exonit.discovery_state import derive_run_id as _shared_derive_run_id
    return _shared_derive_run_id(sector, lot_csv)


def step_stop_workers(state_file: str, dry_run: bool) -> tuple[bool, str]:
    """Stop the WHOLE worker farm before POSTPROCESS.

    Why: the postprocess opens its own sqlite connection and runs veto UPDATEs +
    TRICERATOPS + scorecard while holding write transactions. If workers keep
    writing candidates/telemetry to the same DB file, the two contend on the
    SQLite write lock (busy_timeout 10 s, retries) and the postprocess crawls or
    hits "database is locked". By the time we reach this step, primary tasks and
    FPP backfill are already drained, so no worker has useful work left.

    On **home**: runs ``farmctl stop --all`` (non-fatal if a worker is offline).
    On **AWS**: terminates the EC2 fleet via ``exonit.aws_control.terminate_fleet``
    (also non-fatal — the per-run export is the source of truth).

    NON-FATAL: an offline/unreachable worker must not block the run.
    """
    from exonit.config import deployment as _deployment
    if _deployment() == "aws":
        if dry_run:
            logger.info("dry_run: skipping AWS terminate_fleet")
            return True, "dry_run: would terminate_fleet (aws)"
        try:
            from exonit.aws_control import terminate_fleet
            res = terminate_fleet()
            logger.info("AWS fleet terminated: %s", res)
            return True, f"aws fleet terminated: {res}"
        except Exception as e:
            logger.warning("terminate_fleet failed (non-fatal): %s", e)
            return True, f"aws terminate_fleet exception (non-fatal): {e}"
    # Home path: farmctl stop --all (unchanged)
    cmd = [FARMCTL_BIN, "stop", "--all"]
    logger.info("STOP_WORKERS: %s", " ".join(cmd))

    if dry_run:
        logger.info("dry_run: skipping farmctl stop --all")
        return True, ""

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        if result.returncode != 0:
            detail = (result.stderr or result.stdout or "")[:300]
            logger.warning("STOP_WORKERS: farmctl rc=%d (non-fatal): %s",
                           result.returncode, detail)
            return True, f"farmctl stop --all rc={result.returncode} (non-fatal): {detail}"
        logger.info("STOP_WORKERS: OK")
        return True, ""
    except Exception as exc:
        logger.warning("STOP_WORKERS exception (non-fatal): %s", exc)
        return True, f"farmctl stop --all exception (non-fatal): {exc}"


def step_postprocess(
    lot_csv: str,
    sector: int,
    db: str,
    label: str,
    state_file: str,
    dry_run: bool,
    run_id: str | None = None,
) -> tuple[bool, str]:
    """Run discovery_postprocess.py and scorecard.py.

    Returns (success, detail).  On dry_run, skips subprocess calls.
    Writes outputs to data/scorecards/ with naming convention:
        discovery_s<NNNN>_lot<N>_<YYYY-MM-DD>.{json,md}

    NOTE: augment_scorecard_exominer.py is intentionally NOT called here.
    That script expects TOI ground-truth columns (disposition, period_days,
    depth_ppm, duration_hours, role) which discovery lot CSVs do not have
    (they only carry tic_id,label).  Calling it would raise KeyError: 'disposition'.

    Phase 3 Task 2: run_id should be the ONE minted once at launch (run()'s
    args.run_id) — NOT re-derived here. Re-deriving via _derive_run_id at
    this point (hours into a run) was exactly the midnight-crossing bug this
    task fixes: POSTPROCESS/scorecard paths used to get a DIFFERENT date than
    SCANNING started with if the run crossed midnight. run_id=None is kept
    only for direct-caller/test backward compatibility (falls back to the
    legacy derivation, unchanged from before this task).
    """
    if run_id is None:
        run_id = _derive_run_id(sector, lot_csv)
    scorecards_dir = ROOT / "data" / "scorecards"
    scorecards_dir.mkdir(parents=True, exist_ok=True)

    dossier_out = str(scorecards_dir / f"{run_id}_dossier.md")
    scorecard_json = str(scorecards_dir / f"{run_id}.json")

    toi_catalog = str(ROOT / "data" / "toi_catalog.csv")

    scripts = ROOT / "scripts"

    steps = [
        # Step 1: veto + vetting + FPP + dossier
        (
            "postprocess",
            [
                sys.executable, str(scripts / "discovery_postprocess.py"),
                "--sample", lot_csv,
                "--label", label,
                "--v6-min", "0.5",
                "--out", dossier_out,
                "--sector", str(sector),
                "--apply-veto",
                "--apply-deep-eclipse-veto",
                "--apply-depth-gate",
                "--fetch-missing-stellar",
                *(["--db", db] if db else []),
            ],
        ),
        # Step 2: scorecard (metrics vs TOI catalogue)
        #
        # --sample <lot_csv> (2026-07-14) is what makes the scorecard measure
        # THIS LOT. Without it the only recall it can report is the cumulative
        # one over the whole DB — which does not move between runs (three
        # consecutive runs over 100k/300k/400k different stars all reported
        # 0.7231270358306189) and therefore cannot validate any change.
        (
            "scorecard",
            [
                sys.executable, str(scripts / "scorecard.py"),
                "--db", db or str(ROOT / "data" / "exonit.db"),
                "--toi", toi_catalog,
                "--out", scorecard_json,
                "--run-id", run_id,
                "--sample", lot_csv,
                "--sample-size", str(_lot_size(lot_csv)),
                "--stratification", "discovery",
                "--pipeline-version", "discovery-v1",
            ],
        ),
    ]

    if dry_run:
        logger.info("dry_run: skipping all postprocess subprocesses (%d steps)", len(steps))
        return True, ""

    # Review follow-up (2026-07-02): fpp-timeout candidates (see B3 in
    # discovery_postprocess.py) print a single greppable
    # "WARNING fpp-timeout: ..." line to stdout. Previously this was only
    # read/logged on a FAILED (rc!=0) subprocess — on a healthy (rc=0) run
    # the fact that N candidates timed out on FPP never reached the
    # orchestrator's own log at any level, only the git-committed dossier
    # discovered after the fact. Now scanned on EVERY subprocess result
    # (success or failure) and logged immediately.
    fpp_timeout_warnings: list[str] = []

    # Both POSTPROCESS children report into the SAME phase record; the substep
    # field ("postprocess" then "scorecard") is what says which one is running.
    # There is one live record per phase by design — the second step overwrites
    # the first (see the spec's "Magatzem" section).
    step_env = progress_model.child_env(run_id, "POSTPROCESS")

    for step_name, cmd in steps:
        logger.info("POSTPROCESS [%s]: %s", step_name, " ".join(cmd))
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, cwd=str(ROOT),
                timeout=POSTPROCESS_TIMEOUT_SECONDS,
                env=step_env,
            )
            step_warnings = [
                line for line in (result.stdout or "").splitlines()
                if line.startswith("WARNING fpp-timeout:")
            ]
            for line in step_warnings:
                logger.warning("POSTPROCESS [%s]: %s", step_name, line)
            fpp_timeout_warnings.extend(step_warnings)

            if result.returncode != 0:
                detail = (result.stderr or result.stdout or "")[:500]
                logger.error("POSTPROCESS [%s] failed rc=%d: %s", step_name, result.returncode, detail)
                return False, f"{step_name} rc={result.returncode}: {detail}"
            logger.info("POSTPROCESS [%s]: OK", step_name)
        except subprocess.TimeoutExpired:
            msg = (f"{step_name} exceeded timeout of "
                  f"{POSTPROCESS_TIMEOUT_SECONDS:.0f}s "
                  f"({POSTPROCESS_TIMEOUT_SECONDS / 3600:.1f}h) — killed. A "
                  f"pathological candidate or hung subprocess was blocking "
                  f"the run tail with the fleet already stopped (B3).")
            logger.error("POSTPROCESS [%s] TIMEOUT: %s", step_name, msg)
            return False, msg
        except Exception as exc:
            logger.exception("POSTPROCESS [%s] exception", step_name)
            return False, f"{step_name} exception: {exc}"

    # Persist scorecard paths into the state file so FINALIZE can find them
    # even if the run crosses midnight (POSTPROCESS at 22:45, LoRA ends at
    # 00:19 next day).  We merge without changing the 'state' field so we
    # don't emit a spurious extra transition to the UI.
    merge_fields = dict(
        run_id=run_id,
        scorecard_json=str(Path(scorecard_json).relative_to(ROOT)),
        dossier=str(Path(dossier_out).relative_to(ROOT)),
    )
    if fpp_timeout_warnings:
        # Also surface into the state file (not just the log stream) so the
        # UI can show it without a human having to open the dossier.
        merge_fields["fpp_timeout_warning"] = "; ".join(fpp_timeout_warnings)
    _merge_state(state_file, **merge_fields)
    if fpp_timeout_warnings:
        return True, "; ".join(fpp_timeout_warnings)
    return True, ""


def step_stop_nvidiarack(state_file: str, dry_run: bool) -> tuple[bool, str]:
    """Stop worker-<host> via farmctl to free the GPU for LoRA."""
    if not NVIDIARACK_WORKER_NAME:
        if _is_nvidiarack_cmd(LORA_CMD) and not dry_run:
            # The gpuq/vLLM LoRA path shares the GPU with that worker: running
            # it without stopping the worker is the contention the step exists
            # to prevent. Misconfiguration must fail loudly, not no-op.
            msg = ("EXONIT_GPU_HOST_WORKER is not set but the LoRA command is the "
                   "gpuq/GPU-host path — cannot stop the GPU-host worker")
            logger.error("STOP_NVIDIARACK: %s", msg)
            return False, msg
        logger.warning("STOP_NVIDIARACK: EXONIT_GPU_HOST_WORKER not set — nothing to stop")
        return True, ""
    cmd = [FARMCTL_BIN, "stop", NVIDIARACK_WORKER_NAME]
    logger.info("STOP_NVIDIARACK: %s", " ".join(cmd))

    if dry_run:
        logger.info("dry_run: skipping farmctl stop")
        return True, ""

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode != 0:
            detail = (result.stderr or result.stdout or "")[:300]
            logger.error("STOP_NVIDIARACK failed rc=%d: %s", result.returncode, detail)
            return False, f"farmctl stop rc={result.returncode}: {detail}"
        logger.info("STOP_NVIDIARACK: OK")
        return True, ""
    except Exception as exc:
        logger.exception("STOP_NVIDIARACK exception")
        return False, f"farmctl stop exception: {exc}"


def _cancel_remote_lora_job() -> None:
    """Best-effort: cancel the stuck 'exonit-lora' gpuq job on <host>
    after step_lora's local TimeoutExpired backstop fires (Phase 3 Task 5).

    HARD RULE: all <host> GPU work goes through gpuq; a job left running
    after our local kill (which per the KNOWN LIMITATION above only reaches
    the local ssh shell, not anything remote) wedges the queue for every
    other consumer until someone notices and clears it by hand.

    Two remote round-trips, each with its own LORA_CANCEL_TIMEOUT_SECONDS
    (60s) timeout — worst-case wall-clock is therefore ~120s (two sequential
    60s-capped round trips): `gpuq status` to find the numeric job id for
    name=exonit-lora (cancel is by id, not by name — see the constants above
    for the verified `--help` signatures), then `gpuq cancel <id> --kill`.

    Job selection: prefer the RUNNING match — under gpuq's strict FIFO at
    most one job can be running, and it's the one wedging the GPU. A newer
    queued 'exonit-lora' (e.g. a resume/retry enqueued behind the stuck one)
    always has a HIGHER id, so picking max(id) blindly would cancel the new
    job and leave the wedge alive. Only when no running match exists fall
    back to the highest-id queued one.

    Never raises and never blocks beyond its own timeouts — any failure
    (ssh timeout, non-zero rc, no matching job, unparseable output) is
    logged as a warning and swallowed. Does not affect step_lora's return
    value/state transition either way; those keep the exact pre-Task-5
    timeout semantics.
    """
    try:
        status = subprocess.run(
            LORA_CANCEL_STATUS_SSH_CMD,
            shell=True,
            capture_output=True,
            text=True,
            timeout=LORA_CANCEL_TIMEOUT_SECONDS,
        )
        if status.returncode != 0:
            logger.warning(
                "LORA cancel: gpuq status rc=%d, skipping remote cancel: %s",
                status.returncode,
                (status.stderr or status.stdout or "")[:300],
            )
            return

        # Prefer the RUNNING job (at most one under FIFO — it's the wedge);
        # a retry queued behind it has a higher id, so max(id) would target
        # the wrong job. Fall back to the newest queued match only if none
        # is running (see docstring).
        running_id = None
        queued_id = None
        for line in (status.stdout or "").splitlines():
            m = _LORA_STATUS_LINE_RE.match(line)
            if not m or m.group("name") != LORA_CANCEL_JOB_NAME:
                continue
            candidate = int(m.group("id"))
            if m.group("status") == "running":
                if running_id is None or candidate > running_id:
                    running_id = candidate
            elif m.group("status") == "queued":
                if queued_id is None or candidate > queued_id:
                    queued_id = candidate
        job_id = running_id if running_id is not None else queued_id
        if job_id is None:
            logger.warning(
                "LORA cancel: no running/queued '%s' gpuq job found in "
                "status output — nothing to cancel (may have already "
                "finished or been cleared by someone else).",
                LORA_CANCEL_JOB_NAME,
            )
            return

        cancel = subprocess.run(
            LORA_CANCEL_SSH_CMD_TEMPLATE.format(job_id=job_id),
            shell=True,
            capture_output=True,
            text=True,
            timeout=LORA_CANCEL_TIMEOUT_SECONDS,
        )
        if cancel.returncode != 0:
            logger.warning(
                "LORA cancel: gpuq cancel %d --kill rc=%d: %s",
                job_id, cancel.returncode,
                (cancel.stderr or cancel.stdout or "")[:300],
            )
            return
        logger.warning(
            "LORA cancel: gpuq job #%d (%s) cancelled on the GPU host after "
            "the local timeout.", job_id, LORA_CANCEL_JOB_NAME,
        )
    except Exception as exc:
        # Best-effort — must never raise out of step_lora's except handler.
        logger.warning("LORA cancel: remote cancel attempt failed (ignored): %s", exc)


def step_lora(state_file: str, dry_run: bool,
              sector: int | None = None,
              backlog_limit: int | None = None) -> tuple[bool, str]:
    """Run the LoRA explanation job, scoped to *sector* (see _lora_cmd_for_sector)
    plus up to *backlog_limit* historic rows (default: LORA_BACKLOG_LIMIT).

    Default (Phase 3, 2026-09-26): scripts/lora_macs_driver.py — the 5 Mac
    mini M4 fleet. Manual fallback (`<host>` alias / LORA_SSH_CMD): via
    gpuq on <host> — HARD RULE: ALL GPU work on <host> goes through
    with_nvidia_nvml_compat.sh + gpuq. Any other override (`llamacpp` alias,
    or an ad-hoc command) runs locally on <host>. Either way the command is
    blocking; it starts the engine(s), runs the generator, then tears
    everything down. Can take many minutes.
    """
    if backlog_limit is None:
        backlog_limit = LORA_BACKLOG_LIMIT
    backlog_limit = int(backlog_limit or 0)
    cmd = _lora_cmd_for_sector(sector, backlog_limit)
    is_nvidiarack = _is_nvidiarack_cmd(cmd)
    env = dict(os.environ)
    if sector is not None:
        env["EXONIT_LORA_SECTOR"] = str(int(sector))
        env["EXONIT_LORA_BACKLOG_LIMIT"] = str(backlog_limit)
    logger.info("LORA: running (blocking, %s, sector=%s, backlog_limit=%d): %s",
                "gpuq" if is_nvidiarack else "override",
                sector, backlog_limit, cmd)

    if dry_run:
        logger.info("dry_run: skipping LoRA call")
        return True, ""

    if is_nvidiarack:
        try:
            # KNOWN LIMITATION (review follow-up, 2026-07-02): shell=True means
            # Python's TimeoutExpired handling below only kills the immediate
            # local child (the `/bin/sh -c "..."` shell) — NOT the ssh client
            # process it spawned, nor (more importantly) anything already
            # running remotely on <host> as a result of it. LORA_SSH_CMD is
            # exactly that: ssh + gpuq + vLLM on <host>. If this 24h backstop
            # ever actually fires, the remote `gpuq run --name exonit-lora` job
            # (and vLLM under it) can be left running on <host> with no local
            # process left to tear it down — a stray local ssh process can also
            # linger. This is a HARD RULE concern (all <host> GPU work must
            # go through gpuq; a leaked job blocks the queue for every other
            # consumer). Minimum mitigation: the FAILED detail below tells the
            # operator to check/clear it manually (`gpuq status` /
            # `gpuq stop --name exonit-lora` on <host>). This path is now
            # the MANUAL FALLBACK only (Phase 3 default moved to the Macs
            # driver below, which does not have this limitation) — a real fix
            # here would need a remote-side watchdog/timeout in
            # lora_via_gpuq.sh itself, still out of scope.
            result = subprocess.run(
                cmd,
                shell=True,          # cmd is a pre-built shell string
                capture_output=True,
                text=True,
                timeout=LORA_TIMEOUT_SECONDS,
                env=env,
            )
            if result.returncode != 0:
                detail = (result.stderr or result.stdout or "")[:500]
                logger.error("LORA failed rc=%d: %s", result.returncode, detail)
                return False, f"lora rc={result.returncode}: {detail}"
            # Tail of the job's own stdout (the [job] lines from lora_via_gpuq.sh:
            # sector pass rc, backlog pass rc/count) — previously only surfaced on
            # failure, so a non-fatal backlog problem was invisible here.
            tail = "\n".join((result.stdout or "").strip().splitlines()[-6:])
            logger.info("LORA: OK%s", f"\n{tail}" if tail else "")
            return True, ""
        except subprocess.TimeoutExpired:
            msg = (f"lora exceeded timeout of {LORA_TIMEOUT_SECONDS:.0f}s "
                  f"({LORA_TIMEOUT_SECONDS / 3600:.1f}h) — killed locally. "
                  f"Backstop, not expected to fire on a healthy vLLM run. "
                  f"KNOWN LIMITATION: only the local ssh shell was killed — the "
                  f"remote gpuq job ('exonit-lora') and vLLM under it "
                  f"may still be running. Check/clear manually: "
                  f"`gpuq status` and `gpuq stop --name exonit-lora` on the GPU host "
                  f"({GPUQ_SSH_TARGET}) before starting any other GPU work there.")
            logger.error("LORA TIMEOUT: %s", msg)
            # Phase 3 Task 5: best-effort — attempt the remote gpuq cancel so the
            # GPU queue isn't left wedged. Never changes this step's outcome (see
            # _cancel_remote_lora_job's docstring): the returned (ok, msg) below
            # is identical whether the cancel succeeds, fails, or can't run at all.
            _cancel_remote_lora_job()
            return False, msg
        except Exception as exc:
            logger.exception("LORA exception")
            return False, f"lora exception: {exc}"

    # Non-<host> path (Phase 3 default: the Macs driver; also the Phase 2
    # llama.cpp override, or any ad-hoc command): everything the command
    # spawns is LOCAL to this host (ssh fan-out to the Macs happens INSIDE
    # lora_macs_driver.py, not here), so `start_new_session=True` +
    # `os.killpg` on timeout can actually reach every child process the
    # command started — unlike the <host> path above, whose remote gpuq
    # job survives a local kill by construction. This closes the KNOWN
    # LIMITATION for every path except that inherently-remote manual
    # fallback. The driver's own `finally`/signal handlers are expected to
    # tear their hosts down on the SIGTERM this sends; the `ps` sweep in the
    # driver's own teardown is the authoritative check, not this function.
    try:
        proc = subprocess.Popen(
            cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, env=env, start_new_session=True,
        )
    except Exception as exc:
        logger.exception("LORA exception")
        return False, f"lora exception: {exc}"
    try:
        stdout, _ = proc.communicate(timeout=LORA_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        msg = (f"lora exceeded timeout of {LORA_TIMEOUT_SECONDS:.0f}s "
              f"({LORA_TIMEOUT_SECONDS / 3600:.1f}h) — killing the local "
              f"process group (SIGTERM then SIGKILL). The command's own "
              f"teardown (lora_macs_driver.py's `finally` + signal handlers, "
              f"or ops/lora_via_llamacpp.sh's EXIT trap) should still run; "
              f"verify no mlx_lm.server/explainer shim/llama-server processes "
              f"are left on the Mac fleet or the server host before starting any other "
              f"LoRA job.")
        logger.error("LORA TIMEOUT: %s", msg)
        try:
            pgid = os.getpgid(proc.pid)
            os.killpg(pgid, signal.SIGTERM)
            proc.wait(timeout=10)
        except (ProcessLookupError, subprocess.TimeoutExpired, OSError):
            with contextlib.suppress(ProcessLookupError, OSError):
                os.killpg(pgid, signal.SIGKILL)
            with contextlib.suppress(subprocess.TimeoutExpired):
                proc.wait(timeout=5)
        return False, msg
    except Exception as exc:
        logger.exception("LORA exception")
        return False, f"lora exception: {exc}"

    if proc.returncode != 0:
        detail = (stdout or "")[-500:]
        logger.error("LORA failed rc=%d: %s", proc.returncode, detail)
        return False, f"lora rc={proc.returncode}: {detail}"
    tail = "\n".join((stdout or "").strip().splitlines()[-6:])
    logger.info("LORA: OK%s", f"\n{tail}" if tail else "")
    return True, ""


def _compact_remote_dsn(pg_dsn: str) -> str:
    """The DSN the REMOTE compactor (running ON <host>) must use to reach
    PostgreSQL — ``127.0.0.1``, never the LAN IP.

    Verified operationally (2026-07-17): <host>'s ``pg_hba.conf`` does
    not accept connections from its OWN LAN address even
    though that address is exactly what ``db_path`` correctly holds from
    <host>'s (the orchestrator's) point of view. ``EXONIT_COMPACT_REMOTE_DSN``
    overrides this entirely (e.g. a non-standard port or socket path) when
    the mechanical host-swap below is wrong.
    """
    override = os.environ.get("EXONIT_COMPACT_REMOTE_DSN", "").strip()
    if override:
        return override

    parsed = urlsplit(pg_dsn)
    netloc = parsed.netloc
    userinfo, sep, hostport = netloc.rpartition("@")
    host, colon, port = hostport.partition(":")
    new_hostport = f"127.0.0.1{colon}{port}"
    new_netloc = f"{userinfo}{sep}{new_hostport}" if sep else new_hostport
    return urlunsplit(parsed._replace(netloc=new_netloc))


def _compact_remote_cmd(host: str, pg_dsn: str, run_id: str | None = None) -> list[str]:
    """Argv for running ``compact_completeness.py`` on *host* via ssh.

    Verified interactively (2026-07-17, dry-run) against <host>. Not
    ``shell=True`` locally — ``host`` and the single remote-shell string are
    passed as discrete argv elements to ``ssh``, which forwards the latter
    verbatim to the remote login shell (mirrors LORA_SSH_CMD's pattern, but
    as a plain list since there is no LOCAL shell metacharacter to worry
    about here — the whole remote command is one argv element).
    ``shlex.quote`` on the DSN is defense in depth: a DSN doesn't normally
    contain shell metacharacters, but a wrong/overridden one
    (``EXONIT_COMPACT_REMOTE_DSN``) must never be able to break out of the
    remote command line.

    2026-07-27: the EXONIT_PROGRESS_* triplet is prepended so the remote
    process reports its bucket progress back to THIS server over HTTP. Set
    before ``. .env`` sourcing would be pointless (the file could override
    them), so they go after it, immediately before the interpreter — a plain
    env prefix on the command, which is why the URL and run_id are quoted.
    """
    remote_dsn = _compact_remote_dsn(pg_dsn)
    progress_env = ""
    progress_url = _progress_url(remote=True)
    if run_id and progress_url:
        progress_env = (
            f"EXONIT_PROGRESS_RUN_ID={shlex.quote(run_id)} "
            f"EXONIT_PROGRESS_PHASE=FINALIZE "
            f"EXONIT_PROGRESS_URL={shlex.quote(progress_url)} ")
    # Checkout on the REMOTE host; "$HOME" is expanded by its login shell.
    remote_repo = os.environ.get("EXONIT_COMPACT_REMOTE_REPO", "$HOME/exonit")
    remote_shell_cmd = (
        f"set -a && . {remote_repo}/.env && set +a && "
        f"{progress_env}"
        f"PYTHONPATH={remote_repo} {remote_repo}/.venv/bin/python "
        f"{remote_repo}/scripts/compact_completeness.py --pg {shlex.quote(remote_dsn)}"
    )
    return ["ssh", "-o", "ConnectTimeout=15", host, remote_shell_cmd]


def _write_compact_remote_fail_log(*, host: str, cmd: list, reason: str,
                                    returncode: int | None,
                                    stdout: str, stderr: str) -> Path | None:
    """Write the FULL stdout+stderr of a failed remote compaction attempt to
    ``data/discovery/compact_remote_fail_<YYYYmmddTHHMMSS>.log`` (2026-07-21
    fix). Previously the caller kept only the first 500 chars of stderr
    inline in the log line — useless for a Python traceback, whose
    informative part is at the END, not the start — and nothing else was
    ever persisted, so a remote failure's diagnosis was simply lost (the
    2026-07-21 incident: the remote fast-path died with rc=1 after 28 min
    with nothing left to debug from).

    Returns the log path so the caller can report it (and a tail excerpt)
    in its own warning line — or ``None`` if the write itself failed (disk
    full, permission error, ``data/discovery/`` unwritable, ...). NEVER
    raises: this is a best-effort diagnostic on top of a function
    (``_run_compact_remote``) whose own contract is "never raises" — a
    write failure here must degrade to "diagnosis lost" (logged as its own
    warning), never to "FINALIZE crashes" (2026-07-21 review fix). The
    caller still logs a tail excerpt inline regardless of whether this
    write succeeded, so the minimal diagnostic is never lost even when the
    full log can't be persisted.
    """
    try:
        _COMPACT_REMOTE_FAIL_LOG_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        log_path = _COMPACT_REMOTE_FAIL_LOG_DIR / f"compact_remote_fail_{ts}.log"
        with open(log_path, "w") as fh:
            fh.write(f"host: {host}\n")
            fh.write(f"cmd: {' '.join(cmd)}\n")
            fh.write(f"reason: {reason}\n")
            fh.write(f"returncode: {returncode}\n")
            fh.write("\n----- stdout -----\n")
            fh.write(stdout or "")
            fh.write("\n----- stderr -----\n")
            fh.write(stderr or "")
        return log_path
    except Exception as exc:
        logger.warning(
            "FINALIZE: could not write compact_remote_fail log for %s "
            "(reason=%s) — diagnostic log lost, but the run continues: %s",
            host, reason, exc)
        return None


def _run_compact_remote(host: str, pg_dsn: str, run_id: str | None = None) -> dict | None:
    """Run the compactor on *host* via ssh. Returns the parsed JSON summary
    on success, or ``None`` on ANY failure — unreachable host, non-zero rc,
    timeout, or unparseable stdout. Never raises: the caller
    (``_finalize_compact_completeness``) falls back to local execution on
    every ``None``, the same posture as the SSH stalls documented elsewhere
    in this codebase (feedback_ssh_silent_stalls_gcp_aws) — a
    wedged/unreachable <host> must never leave completeness_staging
    unprocessed for this run.

    Every failure branch below writes the FULL remote stdout+stderr to a
    ``compact_remote_fail_<ts>.log`` file (see
    ``_write_compact_remote_fail_log``) and logs its path plus a tail
    excerpt (2026-07-21 fix — the previous 500-char head-truncated inline
    log lost every traceback). That log write is itself best-effort and can
    never raise out of here (2026-07-21 review fix): an unwritable
    ``data/discovery/`` (disk full, permissions) degrades to "diagnosis
    lost" — a warning naming the failed write plus the tail excerpt still
    logged inline — never to breaking THIS function's own "never raises"
    contract.

    Safe to race with a concurrent local fallback or another run's
    compaction: the advisory lock inside ``compact_completeness.py`` is held
    in PostgreSQL itself, not locally, so it serializes correctly regardless
    of which host(s) are trying to compact at once.
    """
    cmd = _compact_remote_cmd(host, pg_dsn, run_id)
    logger.info("FINALIZE: completeness compaction (remote/%s): %s", host, " ".join(cmd))
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True,
            timeout=EXONIT_COMPACT_REMOTE_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        log_path = _write_compact_remote_fail_log(
            host=host, cmd=cmd, reason="timeout", returncode=None,
            stdout=stdout, stderr=stderr)
        tail = (stderr or stdout)[-_COMPACT_REMOTE_FAIL_TAIL_CHARS:]
        logger.warning(
            "FINALIZE: completeness compaction remote call to %s exceeded "
            "timeout of %.0fs — falling back to local execution. Full "
            "output: %s. Tail: %s",
            host, EXONIT_COMPACT_REMOTE_TIMEOUT_SECONDS,
            log_path or "<log write failed>", tail)
        return None
    except Exception as exc:
        log_path = _write_compact_remote_fail_log(
            host=host, cmd=cmd, reason=f"exception: {exc}", returncode=None,
            stdout="", stderr=str(exc))
        logger.warning(
            "FINALIZE: completeness compaction remote call to %s raised "
            "%s — falling back to local execution. Full output: %s",
            host, exc, log_path or "<log write failed>")
        return None

    if result.returncode != 0:
        log_path = _write_compact_remote_fail_log(
            host=host, cmd=cmd, reason="nonzero_rc",
            returncode=result.returncode,
            stdout=result.stdout or "", stderr=result.stderr or "")
        tail = (result.stderr or result.stdout or "")[-_COMPACT_REMOTE_FAIL_TAIL_CHARS:]
        logger.warning(
            "FINALIZE: completeness compaction remote call to %s rc=%d — "
            "falling back to local execution. Full output: %s. Tail: %s",
            host, result.returncode, log_path or "<log write failed>", tail)
        return None

    try:
        return json.loads(result.stdout)
    except (json.JSONDecodeError, TypeError):
        log_path = _write_compact_remote_fail_log(
            host=host, cmd=cmd, reason="bad_json_stdout",
            returncode=result.returncode,
            stdout=result.stdout or "", stderr=result.stderr or "")
        tail = (result.stdout or "")[-_COMPACT_REMOTE_FAIL_TAIL_CHARS:]
        logger.warning(
            "FINALIZE: completeness compaction remote call to %s: stdout "
            "was not valid JSON — falling back to local execution. Full "
            "output: %s. Tail: %s",
            host, log_path or "<log write failed>", tail)
        return None


def _compact_service_request(
    url: str, token: str, *, method: str = "GET", body: bytes | None = None,
) -> tuple[int, dict | None, str]:
    """One short, independent HTTP request to exonit-compactd.

    Returns ``(status, parsed_json_or_None, raw_text)``. An HTTP error
    status (4xx/5xx) is NOT raised here — the service always answers in
    JSON per the spec (Component 2), including its error bodies (e.g. the
    ``409 {"error": "busy", "job_id": ...}`` on ``/compact``), so the caller
    needs the status/body pair even on failure. Only a genuine
    connection-level failure (refused/unreachable/DNS/timeout) raises,
    via ``urllib.request.urlopen`` — the caller distinguishes that case
    (``service_unreachable``) from an in-band error response
    (``service_error``).
    """
    req = urllib.request.Request(url, data=body, method=method)
    req.add_header("Authorization", f"Bearer {token}")
    if body is not None:
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(
                req, timeout=EXONIT_COMPACT_SERVICE_HTTP_TIMEOUT_SECONDS) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            status = resp.status
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        status = exc.code
    try:
        parsed = json.loads(raw) if raw else None
    except (json.JSONDecodeError, TypeError):
        parsed = None
    return status, parsed, raw


def _run_compact_service(url: str, token: str, run_id: str | None = None) -> dict | None:
    """Run the compaction via the ``exonit-compactd`` HTTP service on
    <host> (2026-09-09, Component 3 of docs/superpowers/specs/
    2026-09-09-compactd-service-design.md). Returns the parsed JSON summary
    on success, or ``None`` on ANY failure — unreachable service, a
    non-202/409 response to the ``POST /compact``, a job that ends in
    ``state: error``, a ``404`` on a poll (job vanished — e.g. the service
    restarted mid-job), a total-timeout, or any exception. Never raises: the
    caller (``_finalize_compact_completeness``) falls back to the SSH path
    (and from there to local) on every ``None``, the exact same
    non-fatal-by-construction posture as ``_run_compact_remote``.

    This exists to replace the 2026-07-17 SSH fast path's failure mode from
    the 2026-09-09 incident (gotcha_compact_remote_ssh_dies_silently_
    staging_snowball): a long-lived SSH session with no keepalive died
    silently on a long compaction, leaving ``completeness_staging`` to
    snowball to 82.7M files before anyone noticed. Every request here — the
    initial ``POST /compact`` and every ``GET /jobs/<job_id>`` poll — is its
    own short, independent connection (default 30s each); nothing is ever
    held open across the whole compaction, so there is no long session left
    to die silently.

    Protocol (server side is Component 2, ``scripts/exonit_compactd.py``):
      1. ``POST {url}/compact`` with a JSON body carrying the progress
         identity (see below). ``202`` -> a fresh job just started,
         ``job_id`` in the body. ``409`` -> another caller (this run or a
         concurrent one) already has a job in flight; ``job_id`` in the body
         names it — following it to completion is a SUCCESS for this call
         (someone's compaction still drains the staging table; there is no
         reason to fail just because we weren't the one who started it).
      2. Poll ``GET {url}/jobs/<job_id>`` every
         ``EXONIT_COMPACT_SERVICE_POLL_INTERVAL_SECONDS`` (default 15s)
         until ``state`` is ``done``/``error``, a poll 404s, or the total
         elapsed time reaches ``EXONIT_COMPACT_REMOTE_TIMEOUT_SECONDS``
         (shared with the SSH path's budget — same 2700s default).
      3. ``done`` -> return ``summary`` (same shape as the SSH/local paths).
         Anything else -> write a fail-log (reusing
         ``_write_compact_remote_fail_log``, ``reason`` one of
         ``service_error`` / ``service_timeout`` / ``service_unreachable``)
         and return ``None``.

    Progress-reporting parity with the SSH path (2026-09-09 review, Finding
    2: the service path silently dropped ``EXONIT_PROGRESS_*`` and regressed
    dashboard visibility for what is now the PREFERRED path — see
    ``compact_completeness.py``'s ``_progress_reporter``, which is a no-op
    unless all three of ``EXONIT_PROGRESS_RUN_ID``/``_PHASE``/``_URL`` are
    set). Mirrors ``_compact_remote_cmd``'s own ``progress_env`` construction
    exactly: if ``run_id`` is given AND ``_progress_url(remote=True)``
    resolves to a LAN-reachable URL (<host> cannot reach ``localhost``
    on the caller's own host — see ``_progress_url``'s docstring), the POST
    body carries ``{"run_id": ..., "phase": "FINALIZE", "progress_url":
    ...}``; the service is expected to export these as
    ``EXONIT_PROGRESS_RUN_ID``/``_PHASE``/``_URL`` on the compaction
    subprocess's environment (Component 2). Otherwise the fields are simply
    omitted from the body (never sent as empty strings) — same "omit rather
    than point the remote host at itself" posture as the SSH path.
    """
    base = url.rstrip("/")
    started = time.monotonic()

    post_body: dict = {}
    progress_url = _progress_url(remote=True)
    if run_id and progress_url:
        post_body["run_id"] = run_id
        post_body["phase"] = "FINALIZE"
        post_body["progress_url"] = progress_url

    try:
        status, parsed, raw = _compact_service_request(
            f"{base}/compact", token, method="POST",
            body=json.dumps(post_body).encode("utf-8"))
    except Exception as exc:
        log_path = _write_compact_remote_fail_log(
            host=base, cmd=["POST", f"{base}/compact"],
            reason="service_unreachable", returncode=None,
            stdout="", stderr=str(exc))
        logger.warning(
            "FINALIZE: completeness compaction service POST %s/compact "
            "unreachable — falling back. Full output: %s. Tail: %s",
            base, log_path or "<log write failed>",
            str(exc)[-_COMPACT_REMOTE_FAIL_TAIL_CHARS:])
        return None

    if status == 202 and parsed and parsed.get("job_id"):
        job_id = parsed["job_id"]
    elif status == 409 and parsed and parsed.get("job_id"):
        job_id = parsed["job_id"]
        logger.info(
            "FINALIZE: completeness compaction service busy — following "
            "the active job %s to completion", job_id)
    else:
        log_path = _write_compact_remote_fail_log(
            host=base, cmd=["POST", f"{base}/compact"],
            reason="service_error", returncode=status,
            stdout=raw or "", stderr="")
        tail = (raw or "")[-_COMPACT_REMOTE_FAIL_TAIL_CHARS:]
        logger.warning(
            "FINALIZE: completeness compaction service POST %s/compact "
            "unexpected response (status=%s) — falling back. Full output: "
            "%s. Tail: %s",
            base, status, log_path or "<log write failed>", tail)
        return None

    jobs_url = f"{base}/jobs/{quote(str(job_id), safe='')}"
    while True:
        if time.monotonic() - started >= EXONIT_COMPACT_REMOTE_TIMEOUT_SECONDS:
            log_path = _write_compact_remote_fail_log(
                host=base, cmd=["GET", jobs_url], reason="service_timeout",
                returncode=None, stdout="", stderr="")
            logger.warning(
                "FINALIZE: completeness compaction service job %s exceeded "
                "total timeout of %.0fs — falling back. Full output: %s",
                job_id, EXONIT_COMPACT_REMOTE_TIMEOUT_SECONDS,
                log_path or "<log write failed>")
            return None

        time.sleep(EXONIT_COMPACT_SERVICE_POLL_INTERVAL_SECONDS)

        try:
            status, parsed, raw = _compact_service_request(jobs_url, token, method="GET")
        except Exception as exc:
            log_path = _write_compact_remote_fail_log(
                host=base, cmd=["GET", jobs_url], reason="service_unreachable",
                returncode=None, stdout="", stderr=str(exc))
            logger.warning(
                "FINALIZE: completeness compaction service poll of %s "
                "unreachable — falling back. Full output: %s. Tail: %s",
                jobs_url, log_path or "<log write failed>",
                str(exc)[-_COMPACT_REMOTE_FAIL_TAIL_CHARS:])
            return None

        if status == 404 or not parsed:
            log_path = _write_compact_remote_fail_log(
                host=base, cmd=["GET", jobs_url], reason="service_error",
                returncode=status, stdout=raw or "", stderr="")
            logger.warning(
                "FINALIZE: completeness compaction service job %s vanished "
                "(status=%s) — falling back. Full output: %s",
                job_id, status, log_path or "<log write failed>")
            return None

        state = parsed.get("state")
        if state == "done":
            summary = parsed.get("summary")
            if summary is None:
                log_path = _write_compact_remote_fail_log(
                    host=base, cmd=["GET", jobs_url], reason="service_error",
                    returncode=status, stdout=raw or "", stderr="")
                logger.warning(
                    "FINALIZE: completeness compaction service job %s "
                    "reported done with no summary — falling back. Full "
                    "output: %s", job_id, log_path or "<log write failed>")
                return None
            return summary
        if state == "error":
            log_tail = parsed.get("log_tail", "")
            log_path = _write_compact_remote_fail_log(
                host=base, cmd=["GET", jobs_url], reason="service_error",
                returncode=parsed.get("rc"), stdout=log_tail, stderr="")
            logger.warning(
                "FINALIZE: completeness compaction service job %s failed "
                "(rc=%s) — falling back. Full output: %s. Tail: %s",
                job_id, parsed.get("rc"), log_path or "<log write failed>",
                (log_tail or "")[-_COMPACT_REMOTE_FAIL_TAIL_CHARS:])
            return None
        # state == "running" (or anything else in-progress-shaped): keep polling.


def _finalize_compact_completeness(db_path: str, run_id: str | None = None) -> None:
    """Drain ``completeness_staging`` -> bucketed Parquet + PG aggregates.

    Phase 4a Task 4: invoked once, at the end of FINALIZE, only when the
    completeness-on-PG cutover flag (``EXONIT_COMPLETENESS_PG=1``) is ON AND
    the operational backend is PostgreSQL — the flag is meaningless with a
    SQLite main backend (see ``exonit/db.py``'s ``ExoNitDB.__init__``
    docstring). Flag OFF (the default): no-op, zero behaviour change.

    Precedence (2026-09-09, Component 3 of docs/superpowers/specs/
    2026-09-09-compactd-service-design.md): **service -> SSH -> local**.

    - **Service** (fastest, preferred): if BOTH ``EXONIT_COMPACT_SERVICE_URL``
      (e.g. ``http://storage-host:8646``) and
      ``EXONIT_COMPACT_SERVICE_TOKEN`` are set, the compaction is dispatched
      to the ``exonit-compactd`` HTTP service on <host> first
      (``_run_compact_service``) — a short ``POST`` plus 15s polls, never a
      long-lived connection. This exists because the 2026-07-17 SSH fast
      path below has a documented silent-death failure mode on long runs
      (2026-09-09 incident, gotcha_compact_remote_ssh_dies_silently_
      staging_snowball: an SSH session with no keepalive died mid-compaction
      without anyone noticing, and ``completeness_staging`` snowballed to
      82.7M files before it was caught). ANY failure of the service attempt
      (unreachable, bad response, job error, poll timeout) falls through to
      the SSH path below unconditionally — the reason was already logged
      inside ``_run_compact_service``. Either env var missing/empty: this
      step is skipped entirely, unchanged behaviour.
    - **SSH** (2026-07-17, fallback #1 — kept intact, retirement is a
      separate future change per the spec's "out of scope"): if
      ``EXONIT_COMPACT_REMOTE_HOST`` is set (e.g. ``user@storage-host``),
      the compaction is attempted ON <host> over SSH
      (``_run_compact_remote``) — PG and MinIO both live there, so this
      removes the LAN round-trip that dominates a full pass. ANY failure of
      the remote attempt (unreachable, non-zero rc, timeout, bad output)
      falls back to running locally below, unconditionally.
    - **Local** (fallback #2, always available): unchanged, byte-identical
      to before either fast path existed.

    This function's non-fatal-by-construction guarantee (below) is
    unaffected by which path (or combination of failed paths) was tried.

    NON-FATAL by construction — every branch below returns instead of
    raising. A failing/timed-out/erroring compaction must never fail the
    run: ``completeness_staging`` simply keeps accumulating rows and the
    NEXT run's (or a manually-invoked) compaction absorbs everything that
    piled up since the last successful pass (docs/superpowers/plans/
    2026-07-05-phase4-completeness-parquet.md, Task 4 WARNING note). The
    run's own health must never be gated on pending staging rows.
    """
    from exonit.completeness_store import completeness_pg_enabled
    from exonit.db_backend import is_pg_url

    if not (completeness_pg_enabled() and is_pg_url(db_path)):
        return

    service_url = os.environ.get("EXONIT_COMPACT_SERVICE_URL", "").strip()
    service_token = os.environ.get("EXONIT_COMPACT_SERVICE_TOKEN", "").strip()
    if service_url and service_token:
        summary = _run_compact_service(service_url, service_token, run_id)
        if summary is not None:
            logger.info(
                "FINALIZE: completeness compaction OK (path=service): %s",
                json.dumps(summary, default=str))
            return
        # Any service failure falls through to the SSH path (and from there
        # to local) below — the reason was already logged inside
        # _run_compact_service.
        logger.info(
            "FINALIZE: completeness compaction service fast-path failed — "
            "falling back to the SSH/local path(s)")

    remote_host = os.environ.get("EXONIT_COMPACT_REMOTE_HOST", "").strip()
    if remote_host:
        summary = _run_compact_remote(remote_host, db_path, run_id)
        if summary is not None:
            logger.info(
                "FINALIZE: completeness compaction OK (path=remote/%s): %s",
                remote_host, json.dumps(summary, default=str))
            return
        # Any remote failure falls through to the local path below — the
        # reason was already logged inside _run_compact_remote.
        path_label = "local-fallback"
    else:
        path_label = "local"

    cmd = [
        sys.executable, str(ROOT / "scripts" / "compact_completeness.py"),
        "--pg", db_path,
    ]
    logger.info("FINALIZE: completeness compaction (path=%s): %s",
                path_label, " ".join(cmd))
    env = progress_model.child_env(run_id, "FINALIZE")
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, cwd=str(ROOT),
            timeout=COMPLETENESS_COMPACTION_TIMEOUT_SECONDS, env=env,
        )
    except subprocess.TimeoutExpired:
        logger.warning(
            "FINALIZE: completeness compaction (path=%s) exceeded timeout "
            "of %.0fs (non-fatal) — staging left intact, absorbed by the "
            "next compaction", path_label, COMPLETENESS_COMPACTION_TIMEOUT_SECONDS)
        return
    except Exception as exc:
        logger.warning(
            "FINALIZE: completeness compaction (path=%s) exception "
            "(non-fatal): %s", path_label, exc)
        return

    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "")[:500]
        logger.warning(
            "FINALIZE: completeness compaction (path=%s) rc=%d (non-fatal): %s",
            path_label, result.returncode, detail)
        return

    try:
        summary = json.loads(result.stdout)
    except (json.JSONDecodeError, TypeError):
        logger.warning(
            "FINALIZE: completeness compaction (path=%s) OK but stdout was "
            "not valid JSON (non-fatal): %r", path_label, (result.stdout or "")[:300])
        return
    logger.info("FINALIZE: completeness compaction OK (path=%s): %s",
                path_label, json.dumps(summary, default=str))


def _publish_finalize_substep(state_file: str, substep: str) -> None:
    """Surface which FINALIZE sub-step is running in the pipeline state.

    The completeness compaction can hold FINALIZE for 15-40 min with no
    outward sign of life (operator, 2026-07-25) — the UI reads
    ``finalize_substep`` to say what the run is actually doing. Best-effort:
    a state-write failure must never affect the sub-step itself. Publish ""
    when the last sub-step ends so the field never lingers into DONE.
    """
    try:
        _set_state(state_file, "FINALIZE", finalize_substep=substep)
    except Exception as exc:
        logger.debug("finalize_substep publish failed (non-fatal): %s", exc)


def step_finalize(
    state_file: str,
    lot_csv: str,
    sector: int,
    db: str | None,
    dry_run: bool,
    *,
    server_url: str = "",
) -> tuple[bool, str]:
    """Mark the lot as DONE in CAMPAIGN.md and append a row to STATUS.md.

    Reads run_id / scorecard_json / dossier from the state file (written by
    step_postprocess).  If missing (e.g. a run started before this feature),
    attempts to reconstruct them using today's date — but warns that the
    scorecard file may not exist if midnight was crossed.

    This step is NON-FATAL: a failure here is logged as WARNING and the run
    proceeds to DONE regardless, because the science artifacts (scorecard,
    dossier, DB records) are already complete.

    ``server_url`` (2026-07-17, optional/keyword, defaults to "" = skip):
    if given, POSTs a best-effort worker-recycle trigger after the
    completeness compaction below — see ``_recycle_workers``. Left empty by
    the AWS_FINALIZE call site (that host self-stops right after, recycling
    its uvicorn workers would be pointless).
    """
    if dry_run:
        logger.info("dry_run: skipping FINALIZE")
        return True, ""

    # Pull paths persisted by step_postprocess
    state = _load_state(state_file)
    run_id = state.get("run_id")
    scorecard_json = state.get("scorecard_json")
    dossier = state.get("dossier")

    if not (run_id and scorecard_json and dossier):
        # Fallback: reconstruct from today's date (may be wrong if midnight crossed)
        logger.warning(
            "FINALIZE: run_id/scorecard_json/dossier missing from state file — "
            "reconstructing from current date (may fail if midnight was crossed)"
        )
        run_id = run_id or _derive_run_id(sector, lot_csv)
        base = f"data/scorecards/{run_id}"
        scorecard_json = scorecard_json or f"{base}.json"
        dossier = dossier or f"{base}_dossier.md"

    db_path = db or str(ROOT / "data" / "exonit.db")

    _publish_finalize_substep(state_file, "campaign_update")
    try:
        # Import here (same pattern as step_postprocess using sys.executable)
        sys.path.insert(0, str(ROOT / "scripts"))
        from discovery_finalize import finalize_run  # type: ignore[import]
        result = finalize_run(
            lot_csv=lot_csv,
            run_id=run_id,
            scorecard_json=scorecard_json,
            dossier=dossier,
            db_path=db_path,
        )
        action = result.get("campaign_action", "unknown")
        logger.info(
            "FINALIZE: campaign_action=%s n_completed=%s/%s n_strong=%s n_finalists=%s",
            action,
            result.get("n_completed"),
            result.get("n_lot"),
            result.get("n_strong"),
            result.get("n_finalists"),
        )
        ok, detail = True, action
    except Exception as exc:
        logger.exception("FINALIZE exception (non-fatal)")
        ok, detail = True, f"exception (non-fatal): {exc}"

    # Phase 4a Task 4: completeness compaction is an entirely separate concern
    # from the CAMPAIGN.md/STATUS.md update above (own try/except so neither
    # side can mask or block the other); a no-op when the cutover flag is
    # OFF, and non-fatal by construction when it's ON (see the helper's
    # docstring) — this outer try/except is pure defense in depth.
    _publish_finalize_substep(state_file, "compact_completeness")
    try:
        _finalize_compact_completeness(db_path, run_id)
    except Exception as exc:  # pragma: no cover - helper is non-fatal internally
        logger.warning(
            "FINALIZE: completeness compaction hook exception (non-fatal): %s",
            exc)

    # Post-run worker recycling (2026-07-17): own try/except, same "never
    # blocks FINALIZE" contract as the compaction hook above — this is an
    # entirely separate concern (memory reclaim on the server, not this
    # run's science artifacts).
    _publish_finalize_substep(state_file, "recycle_workers")
    try:
        _recycle_workers(server_url, dry_run)
    except Exception as exc:  # pragma: no cover - helper is non-fatal internally
        logger.warning(
            "FINALIZE: worker recycle hook exception (non-fatal): %s", exc)

    _publish_finalize_substep(state_file, "")
    return ok, detail


def step_aws_finalize(
    state_file: str,
    lot_csv: str,
    sector: int,
    db: str | None,
    dry_run: bool,
) -> tuple[bool, str]:
    """AWS-only finalize: export bundle → S3, mirror to latest/, run home FINALIZE,
    best-effort self-stop the server instance.

    This step is inserted after POSTPROCESS on AWS runs INSTEAD of the home
    STOP_NVIDIARACK → LORA → FREE_GPU → FINALIZE chain (those steps require LAN
    access to <host> which is unavailable from AWS).

    The S3 export IS FATAL: if the bundle cannot be uploaded, the run goes FAILED
    so we know data was not persisted.  The latest/ mirror and the self-stop are
    NON-FATAL (logged warnings only).

    S3 env vars — AWS→S3→home round-trip
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    AWS side (this function, writer):
        EXONIT_AWS_S3_RUNS_PREFIX  (default: s3://exonit-sector2/runs)
            Set this in the AWS server's .env.  Each run writes two S3 paths:
              <base>/<run_id>/   — permanent archival copy of this run's bundle
              <base>/latest/     — stable pointer, overwritten after every run

    Home side (repatriate button, reader, in server.py /api/aws/repatriate):
        EXONIT_AWS_S3_PREFIX  — must be set in the home server's .env.
            Point it at the latest/ pointer written above, e.g.:
              EXONIT_AWS_S3_PREFIX=s3://exonit-sector2/runs/latest/
            With this value, pressing the Repatriate button with an empty URI
            field automatically pulls the most recent run. To pull a specific
            historical run, set it (or pass it in the UI field) to
              s3://exonit-sector2/runs/<run_id>/
            Note: this is a DIFFERENT variable name from EXONIT_AWS_S3_RUNS_PREFIX
            and lives on a DIFFERENT machine's .env (<host>, not the AWS server).
    """
    if dry_run:
        logger.info("dry_run: skipping AWS finalize (export, mirror, self-stop)")
        # Still run step_finalize in dry_run so CAMPAIGN.md/STATUS.md are updated
        # (also dry_run, so it too will skip its real I/O).
        step_finalize(state_file, lot_csv, sector, db, dry_run)
        return True, "dry_run: would export+mirror+stop (aws)"

    # 1. Resolve paths from state file (written by step_postprocess)
    state = _load_state(state_file)
    run_id = state.get("run_id")
    scorecard_json_rel = state.get("scorecard_json")
    dossier_rel = state.get("dossier")

    if not (run_id and scorecard_json_rel):
        # Reconstruct if state was lost (e.g. run started before this feature)
        logger.warning(
            "AWS_FINALIZE: run_id/scorecard_json missing from state — "
            "reconstructing from current date (may fail if midnight was crossed)"
        )
        run_id = run_id or _derive_run_id(sector, lot_csv)
        base_rel = f"data/scorecards/{run_id}"
        scorecard_json_rel = scorecard_json_rel or f"{base_rel}.json"
        dossier_rel = dossier_rel or f"{base_rel}_dossier.md"

    scorecard_json_abs = str(ROOT / scorecard_json_rel) if scorecard_json_rel else None
    db_path = db or str(ROOT / "data" / "exonit.db")
    bundle_out = str(ROOT / "data" / "bundle.db")

    # 2. Compute S3 prefixes
    base = os.environ.get("EXONIT_AWS_S3_RUNS_PREFIX", "s3://exonit-sector2/runs")
    base = base.rstrip("/")
    run_prefix = f"{base}/{run_id}/"
    latest_prefix = f"{base}/latest/"

    # 3. Export bundle to S3 (FATAL on failure)
    # NOTE: use --all-sectors, NOT --sector <campaign_sector>.  The AWS DB is a
    # fresh per-run DB so every candidate belongs to this run.  candidates.sector
    # stores the TESS *observation* sector (typically 1), which does NOT match the
    # campaign/lot sector (e.g. 2) passed to this function.  Filtering by campaign
    # sector would yield an empty bundle.
    export_cmd = [
        sys.executable,
        str(ROOT / "scripts" / "export_run_bundle.py"),
        "--src-db", db_path,
        "--out", bundle_out,
        "--all-sectors",
        "--s3-prefix", run_prefix,
    ]
    if scorecard_json_abs and Path(scorecard_json_abs).exists():
        export_cmd += ["--scorecard", scorecard_json_abs]

    logger.info("AWS_FINALIZE: export → %s", run_prefix)
    logger.info("AWS_FINALIZE export cmd: %s", " ".join(export_cmd))
    try:
        result = subprocess.run(
            export_cmd, capture_output=True, text=True, cwd=str(ROOT),
            timeout=AWS_EXPORT_TIMEOUT_SECONDS,
        )
        if result.returncode != 0:
            detail = (result.stderr or result.stdout or "")[:500]
            logger.error("AWS_FINALIZE export failed rc=%d: %s",
                         result.returncode, detail)
            return False, f"export_run_bundle rc={result.returncode}: {detail}"
        logger.info("AWS_FINALIZE: export OK: %s", (result.stdout or "").strip())
    except subprocess.TimeoutExpired:
        msg = (f"export_run_bundle exceeded timeout of "
              f"{AWS_EXPORT_TIMEOUT_SECONDS:.0f}s "
              f"({AWS_EXPORT_TIMEOUT_SECONDS / 3600:.1f}h) — killed.")
        logger.error("AWS_FINALIZE export TIMEOUT: %s", msg)
        return False, msg
    except Exception as exc:
        logger.exception("AWS_FINALIZE export exception")
        return False, f"export_run_bundle exception: {exc}"

    # 4. Mirror to latest/ (NON-FATAL — run_prefix is the source of truth)
    try:
        sync_cmd = ["aws", "s3", "sync", run_prefix, latest_prefix, "--delete"]
        logger.info("AWS_FINALIZE: mirror to %s", latest_prefix)
        sync_result = subprocess.run(
            sync_cmd, capture_output=True, text=True, cwd=str(ROOT),
            timeout=AWS_SYNC_TIMEOUT_SECONDS,
        )
        if sync_result.returncode != 0:
            logger.warning(
                "AWS_FINALIZE: mirror to latest/ failed rc=%d (non-fatal): %s",
                sync_result.returncode,
                (sync_result.stderr or sync_result.stdout or "")[:300],
            )
    except subprocess.TimeoutExpired:
        logger.warning(
            "AWS_FINALIZE: mirror to latest/ exceeded timeout of %.0fs "
            "(%.1fh) (non-fatal)",
            AWS_SYNC_TIMEOUT_SECONDS, AWS_SYNC_TIMEOUT_SECONDS / 3600,
        )
    except Exception as exc:
        logger.warning("AWS_FINALIZE: mirror to latest/ exception (non-fatal): %s", exc)

    # 5. Run existing step_finalize for CAMPAIGN.md/STATUS.md (NON-FATAL)
    try:
        ok_fin, detail_fin = step_finalize(state_file, lot_csv, sector, db, dry_run)
        if detail_fin:
            logger.info("AWS_FINALIZE: step_finalize result: %s", detail_fin)
    except Exception as exc:
        logger.warning("AWS_FINALIZE: step_finalize exception (non-fatal): %s", exc)

    # 6. Best-effort self-stop (LAST — only after bundle is confirmed up)
    #    Query IMDSv2 for the instance-id, then stop ourselves.
    region = (os.environ.get("AWS_REGION")
              or os.environ.get("AWS_DEFAULT_REGION")
              or "eu-west-3")
    try:
        import urllib.request as _ur
        # IMDSv2: get a session token first
        token_req = _ur.Request(
            "http://169.254.169.254/latest/api/token",
            headers={"X-aws-ec2-metadata-token-ttl-seconds": "21600"},
            method="PUT",
        )
        with _ur.urlopen(token_req, timeout=2) as resp:
            imds_token = resp.read().decode().strip()

        iid_req = _ur.Request(
            "http://169.254.169.254/latest/meta-data/instance-id",
            headers={"X-aws-ec2-metadata-token": imds_token},
        )
        with _ur.urlopen(iid_req, timeout=2) as resp:
            instance_id = resp.read().decode().strip()

        logger.info("AWS_FINALIZE: self-stop %s in %s", instance_id, region)
        stop_result = subprocess.run(
            ["aws", "ec2", "stop-instances",
             "--instance-ids", instance_id,
             "--region", region],
            capture_output=True, text=True,
        )
        if stop_result.returncode != 0:
            logger.warning(
                "AWS_FINALIZE: self-stop failed rc=%d (non-fatal): %s",
                stop_result.returncode,
                (stop_result.stderr or stop_result.stdout or "")[:300],
            )
        else:
            logger.info("AWS_FINALIZE: self-stop issued OK for %s", instance_id)
    except Exception as exc:
        logger.warning("AWS_FINALIZE: self-stop exception (non-fatal): %s", exc)

    return True, f"export→{run_prefix} mirror→{latest_prefix} run_id={run_id}"


def step_free_gpu(dry_run: bool) -> None:
    """Record that the LoRA step's engine(s) have been torn down.

    Whichever LORA_CMD ran (the Macs driver's own `finally`/signal-handler
    teardown, the <host> path's `gpuq`-managed vLLM, or the llama.cpp
    override's EXIT trap) already tears itself down as part of step_lora
    returning — there is nothing left to do here. We keep the state
    transition for UI visibility.
    """
    logger.info("FREE_GPU: explainer engine(s) torn down by the LoRA "
               "command's own teardown — GPU/host free")


def step_wal_checkpoint(db: str | None, dry_run: bool) -> None:
    """Best-effort WAL TRUNCATE checkpoint at end of run (NON-FATAL).

    A run's writes leave the SQLite ``-wal`` file at its high-water mark (PASSIVE
    auto-checkpoint recycles frames but never shrinks the file — it had crept to
    156 MB). Now that the fleet is stopped (STOP_WORKERS) and the run is finished
    the DB is idle, so a TRUNCATE checkpoint reclaims it. Uses its own short-
    lived sqlite connection (the orchestrator must not import ExoNitDB — that
    would re-run schema init). Any failure is logged and ignored.

    Phase 4a Task 4 follow-up (f): with ``EXONIT_COMPLETENESS_PG=1`` there is
    no SQLite completeness sidecar left anywhere to checkpoint (retired in
    favour of PG staging + Parquet) — this is checked FIRST, unconditionally,
    so the flag alone guarantees ``data/exonit.db`` is never opened, instead
    of relying on ``db``/``is_pg_url`` happening to resolve the same way.
    Flag OFF: completely unchanged.
    """
    if dry_run:
        logger.info("dry_run: skipping WAL checkpoint")
        return
    from exonit.completeness_store import completeness_pg_enabled
    if completeness_pg_enabled():
        logger.info(
            "WAL checkpoint: no-op (EXONIT_COMPLETENESS_PG=1, completeness "
            "SQLite sidecar retired)")
        return
    import sqlite3
    from exonit.db_backend import is_pg_url
    db_path = db or str(ROOT / "data" / "exonit.db")
    if is_pg_url(db_path):
        # PG manages its own WAL — nothing to TRUNCATE, and opening the DSN as a
        # SQLite file would error. No-op.
        logger.info("WAL checkpoint: PG backend, no-op")
        return
    try:
        con = sqlite3.connect(db_path, timeout=30)
        try:
            row = con.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
            logger.info("WAL checkpoint (busy,log,checkpointed)=%s",
                        tuple(row) if row else None)
        finally:
            con.close()
    except Exception as exc:
        logger.warning("WAL checkpoint failed (non-fatal): %s", exc)


# ---------------------------------------------------------------------------
# Main state machine
# ---------------------------------------------------------------------------


def _write_pid_file(state_file: str) -> None:
    """Write this process's PID next to the state file (atexit-removed).

    Lets a run be stopped precisely — `kill $(cat <state>.pid)` — instead of
    `pkill -f discovery_orchestrator.py`, which is both slow (process-table
    scan) and dangerous (the pattern can match the controlling shell). The
    PID is also mirrored into the state file (orchestrator_pid) so the server
    / UI can surface and target it.
    """
    pid_file = str(Path(state_file).with_suffix(".pid"))
    try:
        Path(pid_file).parent.mkdir(parents=True, exist_ok=True)
        Path(pid_file).write_text(str(os.getpid()))
        import atexit
        atexit.register(lambda: Path(pid_file).unlink(missing_ok=True))
    except OSError as exc:
        logger.warning("could not write pid file %s: %s", pid_file, exc)


def run(args: argparse.Namespace) -> None:
    """Execute the full post-run pipeline state machine.

    Phase 3 Task 3 (dispatch table + --resume + checkpoint-skip): the linear
    step-by-step body has become a loop over a per-call HOME_SEQUENCE table
    built further down, once args/run_id/sf are settled. A fresh run still
    starts at index 0 exactly as before; ``--resume <run_id>`` instead reads
    the discovery_runs PG row and starts at the persisted (or, for a FAILED
    run with --force, the recorded-as-failed) state's index. The AWS branch
    and the ``--chained`` short-circuit are UNCHANGED in effect — they are
    special-cased at their exact point in the loop (tied to the POSTPROCESS
    / EXOMINER_BACKFILL entries respectively), so every existing state-order
    test (test_discovery_orchestrator.py, test_aws_finalize.py) still holds.
    """
    # Progress reporting reads the server URL from the environment, and `.env`
    # does not define EXONIT_SERVER_URL — so without this, every reporter in
    # this process and in its children would be disabled and nothing would say
    # so. --server-url is the value the server is actually serving on.
    global _SERVER_URL
    _SERVER_URL = getattr(args, "server_url", "") or ""
    if _SERVER_URL:
        os.environ.setdefault("EXONIT_SERVER_URL", _SERVER_URL)

    sf = args.state_file

    _write_pid_file(sf)

    # Resolve the operational DB once, here, so the whole pipeline (postprocess,
    # scorecard, finalize, WAL checkpoint) sees the same target. ``--db`` wins;
    # absent it we fall back to config.db_path, which returns EXONIT_DATABASE_URL
    # (a postgresql:// DSN) when set, else the default SQLite file. This is what
    # makes a PG run work without threading the DSN through every CLI flag.
    if args.db is None:
        from exonit.config import config as _config
        args.db = _config.db_path

    # Phase 3 Task 2/3: module-level dual-write gate. Set BEFORE run_id
    # resolution (Task 2 had it just after) because --resume (Task 3) needs a
    # live DB handle to read the discovery_runs row before a run_id even
    # exists locally — for --resume, the run_id itself comes FROM that row.
    global _db_path, _pg_dual_write_enabled
    _db_path = args.db
    _pg_dual_write_enabled = not args.dry_run

    resume_mode = bool(getattr(args, "resume", None))
    clear_detail_on_resume = False
    checkpoints: dict = {}
    resume_state: str | None = None

    if resume_mode:
        db = _get_orchestrator_db()
        if db is None:
            sys.exit(
                f"--resume {args.resume}: discovery_runs is unreachable "
                f"(PG down, or --dry-run was also passed — --resume always "
                f"needs a live PG row to read the persisted state from)"
            )
        row = db.get_discovery_run(args.resume)
        if row is None:
            sys.exit(f"--resume {args.resume}: no discovery_runs row found")

        if row["state"] == "DONE":
            sys.exit(f"--resume {args.resume}: run is already DONE — nothing to resume")

        if row["state"] == "FAILED":
            if not getattr(args, "force", False):
                sys.exit(
                    f"--resume {args.resume}: run is FAILED — pass "
                    f"'--resume {args.resume} --force' to retry from the "
                    f"step that failed"
                )
            failed_state = row["checkpoints"].get("_failed_state")
            if not failed_state:
                logger.warning(
                    "--resume --force %s: no _failed_state checkpoint found "
                    "(pre-Task-3 run?) — restarting from SCANNING; safe "
                    "because every step is re-runnable and the postprocess/"
                    "lora/finalize checkpoints (if any) are still honored",
                    args.resume,
                )
                failed_state = "SCANNING"
            resume_state = failed_state
            clear_detail_on_resume = True
        else:
            # A non-terminal state: the process died mid-step without ever
            # reaching FAILED (SIGKILL/OOM/host reboot). Re-execute that same
            # step — every step is re-runnable by construction.
            resume_state = row["state"]

        # Run identity comes from the row — NEVER re-typed/re-derived by the
        # operator. --dry-run is intentionally NOT inherited from the row: it
        # is a per-invocation runtime flag (governs THIS process's real vs.
        # simulated I/O), not part of the run's identity.
        args.run_id = args.resume
        args.sector = row["sector"]
        args.lot_csv = row["lot_csv"]
        args.chained = bool(row["chained"])
        args.server_url = row["server_url"]
        checkpoints = row["checkpoints"] or {}

        logger.info(
            "--resume %s: resuming at %s (row state=%s)%s",
            args.resume, resume_state, row["state"],
            " [FAILED --force]" if clear_detail_on_resume else "",
        )
    else:
        if not args.lot_csv:
            sys.exit("--lot-csv is required unless --resume is given")
        # Phase 3 Task 2: run_id is minted ONCE, either by the scan endpoint
        # (passed via --run-id, the normal server-spawned path) or — for a
        # bare manual CLI invocation — derived here the legacy way. NEVER
        # re-derived later (step_postprocess used to call _derive_run_id
        # itself, hours into a run — exactly the midnight-crossing bug this
        # fixes).
        if not getattr(args, "run_id", None):
            args.run_id = _derive_run_id(args.sector, args.lot_csv)

    run_id = args.run_id

    # Backward compat: a manual CLI run (no --run-id from the endpoint) has no
    # discovery_runs row yet — create one so the dual-writes below have
    # somewhere to land. Harmless no-op on the normal server-spawned path (the
    # endpoint already created the row) AND on --resume (the row already
    # exists — fetched/validated above).
    _ensure_discovery_run_row(
        run_id, args.sector, args.lot_csv, args.dry_run,
        bool(getattr(args, "chained", False)), args.server_url)

    # Materialise this lot's TIC set so the phase polls below can ask "is MY
    # LOT done?" instead of "is the whole DB done?". Idempotent, so --resume
    # and the chained path re-enter it harmlessly. Must run BEFORE the first
    # poll (i.e. before SCANNING), not at first use: a run whose scope is
    # missing silently falls back to the global counters.
    if not args.dry_run:
        _ensure_run_scope(run_id, args.lot_csv)

    # Final-review fix F1: stamp OUR OWN pid + heartbeat immediately, for
    # EVERY run() invocation (fresh, --resume, or chained/autofill-spawned
    # without a caller-side pid write). Without this:
    #   - a --resume --force run leaves the DEAD process's pid + hours-old
    #     heartbeat_at in the row: the watchdog would deterministically FAIL
    #     a LIVE resumed run within ~4 min for any state that emits no poll
    #     heartbeats (POSTPROCESS, LORA), reopening the anti-double-launch
    #     guard mid-postprocess.
    #   - a chained/autofill run spawned without --run-id (so the spawning
    #     endpoint never had a run_id to write a pid for) is otherwise
    #     invisible to the watchdog entirely.
    # Same non-fatal try/except policy as _ensure_discovery_run_row /
    # _write_checkpoint: PG being down must never fail an orchestrator step.
    try:
        _own_db = _get_orchestrator_db()
        if _own_db is not None:
            _own_db.update_discovery_run(run_id, pid=os.getpid(), heartbeat=True)
    except Exception as exc:
        logger.warning(
            "could not record own pid/heartbeat for run_id=%s (non-fatal): %s",
            run_id, exc)

    # ------------------------------------------------------------------
    # Dispatch table (Phase 3 Task 3): (state_name, step_fn, checkpoint_key).
    # step_fn is a zero-arg closure bound to this run's args/sf/run_id. Each
    # closure calls the module-level step_x by its BARE name (not a captured
    # reference) so tests that patch.object(orch, "step_x", ...) are honored
    # — Python resolves a bare global name at CALL time, not at closure-
    # definition time, so a patch applied before orch.run(args) is picked up
    # exactly as it was before this refactor (verified against every
    # patch.object-based test in test_discovery_orchestrator.py and
    # test_aws_finalize.py).
    #
    # STOP_WORKERS / FREE_GPU / FINALIZE are NON-FATAL by design (unchanged
    # from before this task) — their wrappers below always report (True, "")
    # to the generic loop regardless of the underlying step's own result, so
    # the loop's single fatal-handling branch needs no non-fatal special case.
    # ------------------------------------------------------------------

    def _stop_workers_step():
        ok, detail = step_stop_workers(sf, args.dry_run)
        if detail:
            logger.info("STOP_WORKERS detail: %s", detail)
        return True, ""

    def _free_gpu_step():
        step_free_gpu(args.dry_run)
        return True, ""

    def _finalize_step():
        try:
            ok, detail = step_finalize(
                sf, args.lot_csv, args.sector, args.db, args.dry_run,
                server_url=args.server_url)
            if not ok:
                # Should never happen (step_finalize always returns True), but be safe.
                logger.warning("FINALIZE non-fatal issue: %s", detail)
            elif detail and detail not in ("", "updated", "already_done"):
                logger.info("FINALIZE result: %s", detail)
        except Exception as exc:
            # FINALIZE is non-fatal — log and continue to DONE regardless.
            logger.warning("FINALIZE unexpected exception (non-fatal): %s", exc)
        return True, ""

    HOME_SEQUENCE = [
        ("SCANNING", lambda: step_scanning(
            args.server_url, sf, args.poll_interval, args.dry_run), None),
        ("FPP_BACKFILL", lambda: step_fpp_backfill(
            args.server_url, sf, args.poll_interval, args.dry_run), None),
        ("EXOMINER_BACKFILL", lambda: step_exominer_backfill(
            args.server_url, sf, args.poll_interval, args.dry_run), None),
        ("STOP_WORKERS", _stop_workers_step, None),
        ("POSTPROCESS", lambda: step_postprocess(
            args.lot_csv, args.sector, args.db, args.label, sf, args.dry_run,
            run_id=run_id), "postprocess"),
        ("STOP_NVIDIARACK", lambda: step_stop_nvidiarack(sf, args.dry_run), None),
        ("LORA", lambda: step_lora(sf, args.dry_run, sector=args.sector,
                                   backlog_limit=LORA_BACKLOG_LIMIT), "lora"),
        ("FREE_GPU", _free_gpu_step, None),
        ("FINALIZE", _finalize_step, "finalize"),
    ]
    state_to_index = {name: i for i, (name, _fn, _cp) in enumerate(HOME_SEQUENCE)}

    def _clear_failed_state_breadcrumb() -> None:
        """Task-3 review (Minor): after a successful forced resume reaches
        DONE, drop the ``_failed_state`` breadcrumb so a later inspection or
        a second failure/resume cycle never sees a stale value. (A second
        failure before DONE overwrites it anyway; this covers the success
        path.) No-op unless this invocation was a --resume --force recovery.
        Called at every DONE transition point in run().
        """
        if clear_detail_on_resume:
            _write_checkpoint(run_id, "_failed_state", None)

    if resume_mode and resume_state == "AWS_FINALIZE":
        # AWS_FINALIZE is not a HOME_SEQUENCE entry (home runs never reach
        # it) — re-execute it directly, the same "re-run the in-flight step"
        # policy as every other resume point.
        #
        # Final-review fix F2: this IS the first (and only) transition of
        # this resume, so rehydrate the mirror's identity from the PG row
        # (args.lot_csv/sector/server_url were set from `row` above) — same
        # rationale as the HOME_SEQUENCE loop's identity rehydration below.
        set_kwargs = {
            "run_id": run_id,
            "lot_csv": args.lot_csv,
            "sector": args.sector,
            "server_url": args.server_url,
        }
        if clear_detail_on_resume:
            set_kwargs["detail"] = ""
        _set_state(sf, "AWS_FINALIZE", **set_kwargs)
        ok, detail = step_aws_finalize(sf, args.lot_csv, args.sector, args.db, args.dry_run)
        if not ok:
            _set_state(sf, "FAILED", detail=detail)
            _write_checkpoint(run_id, "_failed_state", "AWS_FINALIZE")
            return
        _clear_failed_state_breadcrumb()
        _set_state(sf, "DONE", finished_at=datetime.now(timezone.utc).isoformat())
        logger.info("AWS_FINALIZE done: %s", detail)
        return

    if resume_mode:
        start_index = state_to_index.get(resume_state)
        if start_index is None:
            if resume_state == "ENQUEUEING":
                # Known-legitimate pre-loop state (the discovery_runs row
                # default): the scan endpoint created the row but the
                # orchestrator died before its first SCANNING transition.
                # Nothing has run yet, so starting at SCANNING is exact —
                # not a guess.
                start_index = 0
            else:
                # Task-3 review (Important): an unmapped state most likely
                # signals a corrupted row or schema drift. Refusing to guess
                # a restart point — silently re-running SCANNING + backfills
                # would burn hours of fleet work on a wrong assumption.
                sys.exit(
                    f"--resume {args.resume}: persisted state "
                    f"{resume_state!r} is not a resumable pipeline state — "
                    f"the discovery_runs row is likely corrupted or written "
                    f"by an incompatible schema version. Refusing to guess a "
                    f"restart point. Manual recourse: inspect and fix the "
                    f"row in PG (UPDATE discovery_runs SET state = '<known "
                    f"state>' WHERE run_id = '{args.resume}'), or delete the "
                    f"row and start a fresh full run with --lot-csv."
                )
    else:
        start_index = 0

    detail_cleared = not clear_detail_on_resume  # True == "nothing to clear"
    # Final-review fix F2: on the FIRST transition of ANY resume (forced or
    # not), rehydrate the mirror's lot_csv/sector/server_url from the PG row
    # (the source of truth args.* were populated from above). Without this: if
    # another run (e.g. a chained/autofill lot) wrote pipeline_state.json in
    # between this run's death and its resume, the mirror still carries THAT
    # other run's identity — _active_run_sector() (exonit/server.py) reads
    # the mirror's 'sector' during SCANNING to tag brand-new candidates, so a
    # SCANNING resume would silently mislabel them with the WRONG sector.
    resume_identity_pending = resume_mode

    for idx in range(start_index, len(HOME_SEQUENCE)):
        state_name, step_fn, cp_key = HOME_SEQUENCE[idx]

        if cp_key and checkpoints.get(cp_key, {}).get("done"):
            logger.info("%s: checkpoint %r already done — skipping", state_name, cp_key)
            continue

        set_kwargs = {"run_id": run_id}
        if idx == start_index and not resume_mode:
            # Fresh run only: establish the baseline record (unchanged from
            # before this task) — fresh=True discards any stale JSON left
            # over from a previous run's result fields.
            set_kwargs.update(
                fresh=True,
                lot_csv=args.lot_csv,
                sector=args.sector,
                server_url=args.server_url,
                orchestrator_pid=os.getpid(),
                started_at=datetime.now(timezone.utc).isoformat(),
                dry_run=args.dry_run,
            )
        else:
            if resume_identity_pending:
                set_kwargs.update(
                    lot_csv=args.lot_csv,
                    sector=args.sector,
                    server_url=args.server_url,
                )
                resume_identity_pending = False
            if not detail_cleared:
                # First transition of a --resume --force recovery: explicitly
                # clear PG's detail column. JSON already clears 'detail' for
                # progress states inside _set_state — PG does NOT (Task 2
                # review note) — so an explicit detail="" here is required,
                # or the old FAILED message would linger in the PG row
                # forever.
                set_kwargs["detail"] = ""
                detail_cleared = True

        _set_state(sf, state_name, **set_kwargs)

        if state_name == "EXOMINER_BACKFILL":
            # 3a. Bulk-mark unconfirmed FFI candidates before polling
            # ExoMiner (unchanged from before this task).
            #
            # The SPOC-gate (_run_spoc_check_background) fires per-TIC during
            # the scan phase but can crash / time-out under load, leaving
            # stars with has_spoc=NULL. Those candidates would otherwise fall
            # into the ExoMiner backfill selection (exominer_score IS NULL
            # AND cnn>=0.7), causing thousands of DV-fetch 404s.
            #
            # WHY THIS POINT IS RACE-SAFE:
            #   - Scan is done (SCANNING + FPP_BACKFILL both drained before
            #     we reached here).
            #   - No more worker_task_result POSTs for this lot → no new SPOC
            #     gate threads.
            #   - Any has_spoc=NULL remaining are gate-failures, not pending
            #     checks.
            #   - We call the server endpoint (POST
            #     /api/admin/mark_unconfirmed_ffi) rather than writing to the
            #     DB directly, so the server's connection / lock is used.
            if not args.dry_run:
                import urllib.request as _urllib_req
                import json as _json
                _url = f"{args.server_url}/api/admin/mark_unconfirmed_ffi"
                try:
                    _req = _urllib_req.Request(_url, data=b"", method="POST")
                    with _urllib_req.urlopen(_req, timeout=30) as _resp:
                        _body = _json.loads(_resp.read())
                    logger.info("mark_unconfirmed_ffi: %s", _body)
                except Exception as _exc:
                    # Non-fatal: the backfill query now filters has_spoc=1 so
                    # unconfirmed rows won't be DV-fetched; at worst they
                    # stay NULL (no regression vs the old behaviour) and
                    # drain slowly if something gets a real SPOC hit.
                    logger.warning("mark_unconfirmed_ffi call failed (non-fatal): %s", _exc)
            else:
                logger.info("dry_run: skipping mark_unconfirmed_ffi call")

        # One place instruments EVERY orchestrator-run phase (2026-07-27).
        # Editing each step function was the obvious route and the wrong one:
        # several have multiple success returns, so "report on the way out"
        # would have to be repeated at each of them. Here there is exactly one
        # entry and one exit for every phase.
        phase_rep = None
        if _phase_reports_itself(state_name):
            phase_rep = _phase_reporter(state_name, run_id,
                                        unit="phase", total=1, interval_s=30)
            if phase_rep is not None:
                try:
                    phase_rep.report(0, message=f"{state_name} started")
                except Exception:
                    pass

        ok, detail = step_fn()

        if phase_rep is not None and ok:
            try:
                phase_rep.finish(message=f"{state_name} done")
            except Exception:
                pass

        if not ok:
            fail_detail = detail
            if not fail_detail and state_name in _FAILURE_DEFAULT_DETAIL:
                fail_detail = _FAILURE_DEFAULT_DETAIL[state_name]
            fail_extra = {"nvidiarack_worker_stopped": True} if state_name == "LORA" else {}
            _set_state(sf, "FAILED", detail=fail_detail, **fail_extra)
            _write_checkpoint(run_id, "_failed_state", state_name)
            return

        if cp_key:
            cp_data = {"done": True}
            if cp_key == "postprocess":
                # step_postprocess already _merge_state'd these into the JSON
                # mirror — read them back so the checkpoint carries the same
                # paths (used by --resume to skip straight past it).
                state_now = _load_state(sf)
                cp_data["scorecard_json"] = state_now.get("scorecard_json")
                cp_data["dossier"] = state_now.get("dossier")
            _write_checkpoint(run_id, cp_key, cp_data)

        if state_name == "EXOMINER_BACKFILL" and getattr(args, "chained", False):
            # Chained gap-fill lot (UI auto-fill): the lot's TICs are now
            # fully scored (primary BLS/vetting/CNN + FPP + ExoMiner/FFI).
            # STOP here and let the auto-fill manager chain the next lot.
            # The end-of-campaign tail (STOP_WORKERS / POSTPROCESS /
            # STOP_NVIDIARACK / LORA / FREE_GPU / FINALIZE) is intentionally
            # skipped: running it after every 100-TIC lot would stop the
            # whole fleet and cold-start vLLM per lot (~10 min each), making
            # chained gap-fill self-defeating. Those run once when the
            # sector is actually complete (veto/scorecard/LoRA backfill
            # handled separately).
            _clear_failed_state_breadcrumb()
            _set_state(sf, "DONE", detail="chained gap-fill lot complete (tail skipped)")
            logger.info("Pipeline DONE (chained gap-fill lot)")
            return

        if state_name == "POSTPROCESS":
            # AWS branch: skip the home-only STOP_NVIDIARACK/LORA/FREE_GPU/
            # FINALIZE steps and go directly to AWS_FINALIZE (export → S3,
            # self-stop). The home path (the rest of the loop) is UNCHANGED.
            from exonit.config import deployment as _deployment
            if _deployment() == "aws":
                _set_state(sf, "AWS_FINALIZE")
                ok2, detail2 = step_aws_finalize(
                    sf, args.lot_csv, args.sector, args.db, args.dry_run)
                if not ok2:
                    _set_state(sf, "FAILED", detail=detail2)
                    _write_checkpoint(run_id, "_failed_state", "AWS_FINALIZE")
                    return
                _clear_failed_state_breadcrumb()
                _set_state(sf, "DONE", finished_at=datetime.now(timezone.utc).isoformat())
                logger.info("AWS_FINALIZE done: %s", detail2)
                return
            # Home path: STOP_NVIDIARACK → LORA → FREE_GPU → FINALIZE → DONE
            # (unchanged) continues via the next loop iterations.

    # 10. WAL checkpoint — reclaim the -wal file now that the run is done and
    #     the fleet is stopped (non-fatal cleanup; no state transition).
    step_wal_checkpoint(args.db, args.dry_run)

    _clear_failed_state_breadcrumb()
    _set_state(sf, "DONE",
               finished_at=datetime.now(timezone.utc).isoformat())
    logger.info("Pipeline DONE")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Discovery post-run pipeline orchestrator (state machine)"
    )
    p.add_argument("--server-url", default="http://localhost:8400",
                   help="ExoNit server base URL (default: http://localhost:8400)")
    p.add_argument("--lot-csv", default=None,
                   help="Path to the lot CSV enqueued by discovery_run.py "
                        "(not required with --resume — taken from the "
                        "discovery_runs row instead)")
    p.add_argument("--sector", type=int, default=None,
                   help="Sector number (default: registry active_sector — no "
                        "hardcoded 1: home=1, AWS=2)")
    p.add_argument("--state-file", default=DEFAULT_STATE_FILE,
                   help=f"JSON state file (default: {DEFAULT_STATE_FILE})")
    p.add_argument("--db", default=None,
                   help="Path to exonit.db (default: data/exonit.db via ExoNitConfig)")
    p.add_argument("--label", default="candidate",
                   help="Label filter for discovery_postprocess.py (default: candidate)")
    p.add_argument("--poll-interval", type=int, default=30,
                   help="Seconds between health/work polls (default: 30)")
    p.add_argument("--dry-run", action="store_true",
                   help="Transition states but skip real subprocess/ssh calls")
    p.add_argument("--chained", action="store_true",
                   help="Chained gap-fill mode (UI auto-fill): run SCANNING + "
                        "FPP_BACKFILL + EXOMINER_BACKFILL then stop at DONE. "
                        "Skips the end-of-campaign tail (STOP_WORKERS, "
                        "POSTPROCESS, STOP_NVIDIARACK, LORA, FREE_GPU, FINALIZE) "
                        "so chaining many small lots does not stop the fleet or "
                        "cold-start vLLM per lot.")
    p.add_argument("--run-id", default=None,
                   help="Run identifier minted once by the scan endpoint at "
                        "launch (discovery_s<sector>_lot<lot>_<date>), passed "
                        "through so this process NEVER re-derives it (Phase 3 "
                        "Task 2 — the midnight-crossing bug). Manual CLI use "
                        "may omit it: derived the legacy way (today's UTC "
                        "date) and a discovery_runs PG row created if missing.")
    p.add_argument("--resume", default=None, metavar="RUN_ID",
                   help="Resume a previously-started run by its run_id: reads "
                        "the discovery_runs PG row and restarts the pipeline "
                        "at the persisted step (sector/lot_csv/chained/"
                        "server_url come from the row, never re-typed/"
                        "re-derived — --lot-csv/--sector are not required "
                        "with this flag). A DONE run is a hard error; a "
                        "FAILED run needs --force (Phase 3 Task 3).")
    p.add_argument("--force", action="store_true",
                   help="With --resume, allow retrying a FAILED run — "
                        "resumes at the step recorded as having failed (the "
                        "'_failed_state' checkpoint). Without --force, "
                        "--resume of a FAILED run is a clear error telling "
                        "the operator to pass this flag.")
    args = p.parse_args(argv)
    if not args.resume and not args.lot_csv:
        p.error("--lot-csv is required unless --resume is given")
    return args


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    args = _parse_args()
    if args.sector is None:
        from exonit.discovery_state import active_sector
        args.sector = active_sector(str(ROOT / "data" / "discovery" / "sectors.json"))
    run(args)
