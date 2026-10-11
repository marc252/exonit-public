"""Autonomous-loop supervisor (phase 1/1.5 — shadow mode + armed mode, §4.2).

`tick(deps)` is the pure function that decides what the supervisor would do given
a `TickDeps` — it does not touch the infra, does not call a real AI, and writes nothing outside
what is passed to it as parameters. **`tick()` does not change between modes**: the only
difference between shadow and armed is whether the `Decision` it returns is really
executed or only logged.

`main()` (CLI `python -m autonomy.supervisor --shadow|--armed [--once]`)
is the real loop. In shadow mode (`run_shadow_loop`, UNTOUCHED since phase 1)
it does the read-only GET to `/api/discovery/pipeline`, builds the `TickDeps`
and logs the decision to `data/autonomy/shadow.log` without ever executing anything
(`ServerClient` always `read_only=True`).

In armed mode (`run_armed_loop`, phase 1.5) the same poll/heartbeat/log
is accompanied by a real execution of the decision (`_execute_decision`):
launching the next lot, closing a DONE lot, or remediating an error via
classifier/AI triage + `Executor`. Armed is ONLY activated with the double
key (`config.armed == True` **and** `EXONIT_AUTONOMY_ARMED == "1"`); without
both, `--armed` silently falls back (with a warning) to `run_shadow_loop`
— see `run_armed_loop`.

Decision order of `tick` (spec §4.2):
  1. `data/STOP_CHAIN` present -> `sleep_stop_chain`.
  2. `state == "DONE"` -> `would_close_lot`, EXCEPT if this DONE has already been
     closed (`pipeline.run_id == deps.state.last_closed_run_id`, only
     set in armed mode -- see review round 2, 2026-08-09): then it
     is treated as `IDLE` (`_tick_idle(deps)`) because the next lot must be
     chained, not re-closed. In shadow mode `last_closed_run_id` is never
     set, so shadow always sees `would_close_lot` on a DONE, with no
     change of behaviour.
  3. `state == "IDLE"` -> `would_launch_next_lot` (with an approved current plan) or
     `continuity` (queue with no approved plan).
  4. `state == "FAILED"` or `stuck` -> classify (rules -> local AI) and
     `would_remediate` / `pause`, EXCEPT if this FAILED is only the
     residue of an already-closed run (`pipeline.run_id == deps.state.
     last_closed_run_id`, same guard as the DONE above -- see
     incident 2026-08-11, plan exhausted after closing the last lot):
     then it is also treated as `IDLE` (`_tick_idle_residual(deps,
     "FAILED")`) because it can never be a real failure (a run that really fails
     never reaches `last_closed_run_id`, only a DONE closed via
     `_close_lot_armed` is marked there).
  4b. S4b (2026-09-26, docs/superpowers/specs/2026-09-26-enqueue-s1-s4-
     design.md §2.1/§9.4): `FAILED` without `run_id` whose `detail` starts
     with "scan timed out" (server HTTP timeout, not an exhausted plan) ->
     SINGLE automatic retry (`would_launch_next_lot` via `_tick_idle`,
     counter in `state.scan_timeout_retry_count`) instead of the classifier
     -- S4 (adoption) makes the retry safe (it adopts the dead run's tasks
     instead of duplicating them). Once the retry is exhausted -> `pause`.
  5. any other state ("running") -> `noop_running`, unless the current
     plan needs early planning (~75% or <=2 lots, with no approved
     successor): then `would_trigger_planning`, **only once per
     plan** (marked in `state.planning_triggered`).

Implementation note: the real response of GET /api/discovery/pipeline
(`exonit/routes/discovery.py::api_discovery_pipeline`) only guarantees the
`state` key (plus `progress`/`eta_total_s`, often None); `run_id`,
`detail`, `log_tail`, `tics_done`, `lots_done`, `lots_left`, `stuck` are not
there today — they are all read with defensive `.get()` and safe defaults
so the tick never blows up on the real response of a running run. The
`stuck` signal and the accumulated plan progress (`tics_done`/`lots_done`/`lots_left`)
are hooks for when a future task wires them up (staleness between polls,
plan counters); in their absence the tick behaves safely
(it never fires actions on a signal that does not exist).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

from autonomy.alerting import Mailer, run_alert_hook
from autonomy.classifier import classify
from autonomy.closeout import _resolve_db, close_lot
from autonomy.config import AlertConfig, AutonomyConfig, FleetConfig, REPO_ROOT, load_config
from autonomy.plans import Plan, PlansFile, current_plan, load_plans, needs_planning, plan_progress
from autonomy.playbooks import Executor, ExecutionResult
from autonomy.server_client import ServerClient, ServerHTTPError
from autonomy.triage_llm import spawn_and_choose
from autonomy.types import ErrorContext, PlaybookChoice
from exonit.discovery_state import (
    get_sector_state,
    load_registry,
    save_registry,
    set_sector_state,
)

_LOG = logging.getLogger("autonomy.supervisor")

#: Subsystem Python — same convention as closeout.py/playbooks.py.
PY = sys.executable

#: pipeline_state.json of the orchestrator (source of truth for `close_lot`).
_PIPELINE_STATE_PATH = REPO_ROOT / "data" / "discovery" / "pipeline_state.json"


@dataclass
class SupervisorState:
    """State persisted to `data/autonomy/supervisor_state.json` (spec §4.2)."""

    completed_plan_ids: set = field(default_factory=set)
    planning_triggered: dict = field(default_factory=dict)  # plan_id -> bool
    last_decision: Optional[str] = None
    last_poll_ts: Optional[float] = None
    # Phase 1.5 (armed mode): run_id of the last lot closed by `close_lot` —
    # avoids closing the same run twice if the pipeline stays at DONE
    # between polls (idempotency of `would_close_lot`). None in shadow mode
    # (never written) and in old phase-1 installs (field absent from the
    # persisted JSON -> `.get()` with default None in `from_dict`).
    last_closed_run_id: Optional[str] = None
    # S4b (docs/superpowers/specs/2026-09-26-enqueue-s1-s4-design.md §2.1/
    # §9.4, operator decision): counter of SINGLE automatic retries on a
    # FAILED without run_id whose detail starts with "scan timed out" (see
    # `tick`). Reset to 0 when `_execute_decision` sees `_launch_lot`
    # return "launched" (a real launch succeeded -- the next timeout,
    # if any, is a NEW incident, not a repeat of the same one). Absent from the
    # persisted JSON (installs older than S4b) -> default 0.
    scan_timeout_retry_count: int = 0

    def to_dict(self) -> dict:
        return {
            "completed_plan_ids": sorted(self.completed_plan_ids),
            "planning_triggered": dict(self.planning_triggered),
            "last_decision": self.last_decision,
            "last_poll_ts": self.last_poll_ts,
            "last_closed_run_id": self.last_closed_run_id,
            "scan_timeout_retry_count": self.scan_timeout_retry_count,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SupervisorState":
        return cls(
            completed_plan_ids=set(d.get("completed_plan_ids") or []),
            planning_triggered=dict(d.get("planning_triggered") or {}),
            last_decision=d.get("last_decision"),
            last_poll_ts=d.get("last_poll_ts"),
            last_closed_run_id=d.get("last_closed_run_id"),
            scan_timeout_retry_count=d.get("scan_timeout_retry_count", 0),
        )


@dataclass
class Decision:
    """Decision of a tick. `kind` is one of the 8 values of spec §4.2/task-8,
    plus `"observing"` (incident 2026-08-15 — 1st/2nd consecutive failure of a
    GET to the pipeline, generated in the loop, not by `tick()`; see
    `_handle_server_unreachable`)."""

    kind: str
    detail: str
    choice: Optional[PlaybookChoice] = None


@dataclass
class TickDeps:
    """Dependencies injected into `tick` — no real SSH/HTTP inside `tick`."""

    pipeline: dict
    plans: PlansFile
    stop_chain_exists: bool
    fleet: list  # list[HostHealth]
    classify_fn: Callable[[ErrorContext], Optional[PlaybookChoice]]
    triage_fn: Callable[[ErrorContext], PlaybookChoice]
    state: SupervisorState


def _tick_running(deps: TickDeps, state_name: str) -> Decision:
    """'Running' state: `noop_running`, unless early planning must be
    triggered for the current plan (only once per plan)."""
    plan = current_plan(deps.plans, deps.state.completed_plan_ids)
    if plan is not None and not deps.state.planning_triggered.get(plan.plan_id, False):
        pipeline = deps.pipeline or {}
        tics_done = pipeline.get("tics_done") or 0
        lots_done = pipeline.get("lots_done") or 0
        lots_left = pipeline.get("lots_left")
        progress = plan_progress(plan, tics_done, lots_done)
        if needs_planning(deps.plans, plan, progress, lots_left):
            deps.state.planning_triggered[plan.plan_id] = True
            return Decision(
                "would_trigger_planning",
                f"plan {plan.plan_id} at {progress:.0%} (or <=2 lots); shadow: would trigger "
                "a planning proposal (annex B) + mail 'proposal pending approval' "
                "(only once per plan).",
                None,
            )
    return Decision("noop_running", f"pipeline running (state={state_name}); shadow: does nothing.", None)


def _tick_idle(deps: TickDeps) -> Decision:
    """IDLE: would launch the next lot if there is an approved current plan, else the
    continuity policy (§4.6) — the full continuity computation
    needs `space_ok`/`sectors_with_data`/`deep_coverage`, not present in
    `TickDeps` in phase 1; the tick only reports what would apply."""
    plan = current_plan(deps.plans, deps.state.completed_plan_ids)
    if plan is not None:
        return Decision(
            "would_launch_next_lot",
            f"IDLE with current plan {plan.plan_id} (sector {plan.sector}, mode {plan.mode}); "
            "shadow: would do preflight (fleet+space) and POST /api/discovery/scan.",
            None,
        )
    return Decision(
        "continuity",
        "IDLE and the plan queue has no approved current plan; shadow: would apply the "
        "continuity policy (spec §4.6).",
        None,
    )


def _tick_idle_residual(deps: TickDeps, state_name: str) -> Decision:
    """Delegates to `_tick_idle` when `state_name` (DONE or FAILED) is only the
    residue of a run that `last_closed_run_id` has already closed -- never a real
    failure (see the docstring of `tick`, DONE-chaining guard 2026-08-09 and its
    FAILED-residual twin added later, same mechanism). Annotates the
    `detail` of `_tick_idle` with an observable trace of why it was
    reinterpreted as IDLE instead of closing/remediating."""
    decision = _tick_idle(deps)
    note = f" ({state_name} residual of an already-closed run, treated as IDLE)"
    return Decision(decision.kind, decision.detail + note, decision.choice)


def _tick_error(deps: TickDeps, state_name: str, pipeline: dict) -> Decision:
    """FAILED or stuck: classify (rules first, local AI after) and choose
    `would_remediate` (playbook != pause_and_alert) or `pause`."""
    fleet_snapshot = {h.worker_name: h.to_dict() for h in (deps.fleet or [])}
    ctx = ErrorContext(
        run_id=str(pipeline.get("run_id") or "unknown"),
        phase=str(pipeline.get("phase") or state_name),
        state=state_name,
        detail=str(pipeline.get("detail") or ""),
        log_tail=str(pipeline.get("log_tail") or ""),
        fleet_snapshot=fleet_snapshot,
        recent_actions=list(pipeline.get("recent_actions") or []),
    )

    choice = deps.classify_fn(ctx)
    if choice is None:
        choice = deps.triage_fn(ctx)

    if choice.playbook == "resume_run":
        # Incident 2026-08-11: the AI returned the literal placeholder of the
        # prompt instead of the real run_id -> the identity parameter of a
        # resume never comes from the AI, always from the authorized context (`ctx.run_id`).
        params = choice.params if isinstance(choice.params, dict) else {}
        params = dict(params, run_id=ctx.run_id)
        choice = replace(choice, params=params)

    kind = "pause" if choice.playbook == "pause_and_alert" else "would_remediate"
    detail = (
        f"playbook={choice.playbook} (source={choice.source}, confidence={choice.confidence}): "
        f"{choice.reasoning}"
    )
    return Decision(kind, detail, choice)


def tick(deps: TickDeps) -> Decision:
    """Pure function: given `deps`, returns the `Decision` the supervisor would
    take (spec §4.2). Does no I/O; it may only mutate `deps.state`
    (`planning_triggered`, `last_decision`) for persistence between ticks."""
    if deps.stop_chain_exists:
        decision = Decision(
            "sleep_stop_chain", "data/STOP_CHAIN present; supervisor paused (shadow mode).", None
        )
        deps.state.last_decision = decision.kind
        return decision

    pipeline = deps.pipeline or {}
    state_name = pipeline.get("state", "IDLE")
    stuck = bool(pipeline.get("stuck", False))

    if state_name == "DONE":
        run_id = pipeline.get("run_id")
        if run_id and run_id == deps.state.last_closed_run_id:
            # Review round 2 (2026-08-09): this DONE has already been closed
            # (only marked in armed mode, see `_close_lot_armed`) --
            # re-emitting `would_close_lot` here would stall the loop
            # forever (the executor turns it into a `noop` for idempotency
            # and the supervisor never moves on to the next lot). The next one must be chained: it
            # is treated exactly like IDLE.
            decision = _tick_idle_residual(deps, state_name)
        else:
            decision = Decision(
                "would_close_lot",
                "pipeline DONE; shadow: would close the lot (health check + scorecard + "
                "ExoMiner augment + CAMPAIGN.md row + weekly digest, spec §4.5).",
                None,
            )
    elif state_name == "IDLE":
        decision = _tick_idle(deps)
    elif state_name == "FAILED" and pipeline.get("run_id") and pipeline.get("run_id") == deps.state.last_closed_run_id:
        # Incident 2026-08-11: an exhausted plan (HTTP 400 reason=no_undone_tics
        # after closing the last lot) leaves `pipeline_state.json` in FAILED
        # but keeping the OLD `run_id`, already closed -- `last_closed_run_id`
        # is only marked in `_close_lot_armed` from a DONE with decision
        # `would_close_lot`, never by a run that really fails mid-run (its
        # run_id never reaches `last_closed_run_id`). So this
        # FAILED is always residue of the failed launch, not a failure of the
        # run that was just closed: same mechanism as the DONE guard
        # above, it is treated as IDLE instead of entering the classifier/pause.
        decision = _tick_idle_residual(deps, state_name)
    elif (state_name == "FAILED" and not pipeline.get("run_id")
          and pipeline.get("detail") in _PLAN_EXHAUSTED_REASONS):
        # Incident 2026-09-26: since 40ce1bc the POST /api/discovery/scan
        # writes ENQUEUEING with a CLEAN state (fresh=True), so the FAILED
        # of an exhausted plan NO longer keeps the old run_id of the closed lot and the
        # guard above (run_id == last_closed_run_id) stopped recognizing it:
        # tick() sent it to the classifier, the LLM triage did
        # pause_and_alert and the chain stalled when s5 finished. With no
        # run_id + an exhausted-plan reason (the same closed list that
        # `_interpret_post_error` uses to mark the plan completed) it is residue
        # of the launch, not a failure: IDLE -> next plan. Any other
        # FAILED without run_id (need_download, enqueue_failed, spawn failed...)
        # continues to the classifier/pause.
        decision = _tick_idle_residual(deps, state_name)
    elif (state_name == "FAILED" and not pipeline.get("run_id")
          and str(pipeline.get("detail") or "").startswith("scan timed out")):
        # S4b (docs/superpowers/specs/2026-09-26-enqueue-s1-s4-design.md
        # §2.1/§9.4, operator decision): an HTTP-level scan timeout (the server
        # killed the discovery_run.py/enqueue_from_csv.py subprocess before
        # they finished -- exonit/routes/discovery.py's asyncio.TimeoutError
        # branch, detail=f"scan timed out after {N} s") with no run_id (the
        # discovery_runs row was never created) is now SAFE to
        # retry automatically: S4 (adoption, EXONIT_LOT_ADOPT_ACTIVE)
        # makes the next /api/discovery/scan for the SAME plan adopt the
        # tasks the dead run had already created instead of duplicating them,
        # instead of escalating to `pause_and_alert` as it did before. A single
        # automatic retry (`state.scan_timeout_retry_count`, reset to 0 when
        # `_execute_decision` sees `_launch_lot` return "launched"); if it
        # fails again with the same reason, a human must look (pause) -- never an
        # infinite retry loop.
        if deps.state.scan_timeout_retry_count < 1:
            deps.state.scan_timeout_retry_count += 1
            _retried = _tick_idle(deps)
            decision = Decision(
                _retried.kind,
                _retried.detail + (
                    f" (automatic retry S4b {deps.state.scan_timeout_retry_count}"
                    f"/1 for FAILED={pipeline.get('detail')!r})"
                ),
                _retried.choice,
            )
        else:
            decision = Decision(
                "pause",
                f"FAILED for {pipeline.get('detail')!r} without run_id, and the single "
                "automatic retry (S4b) is already exhausted -- needs a human eye.",
                None,
            )
    elif state_name == "FAILED" or stuck:
        decision = _tick_error(deps, state_name, pipeline)
    else:
        decision = _tick_running(deps, state_name)

    deps.state.last_decision = decision.kind
    return decision


# ---------------------------------------------------------------------------
# Hung-server detection (incident 2026-08-15): the port was listening but
# no uvicorn accepted requests (~150 threads blocked on a futex). A single
# failed GET may be a transient network glitch -- only after
# 3 CONSECUTIVE failures is it treated as "really hung server". The
# counter lives only as a local variable of the loop (`run_shadow_loop`/
# `run_armed_loop`), never in `SupervisorState` nor on disk -- it resets if the
# supervisor process restarts, which is acceptable (the next OK GET
# also resets it to 0).
# ---------------------------------------------------------------------------

#: Extracts `rc=<N>` from the detail of a real `ExecutionResult` (`Executor.execute`,
#: format `f"cmd={cmd} rc={proc.returncode} ..."`). Absent (e.g. the script
#: could not even be started, detail="cmd=... error=...") -> never "ok".
_RC_RE = re.compile(r"\brc=(-?\d+)\b")


def _restart_result_ok(result: ExecutionResult) -> bool:
    """A `restart_server` is only considered to have "repaired the server" if
    the `Executor` really executed it (not `refused_limit`/`refused_invalid`)
    AND the script exited with `rc=0` (see `scripts/restart_exonit_server.py`
    — exit 0 only when `/` responds again within the timeout)."""
    if result.status != "executed":
        return False
    match = _RC_RE.search(result.detail)
    return match is not None and match.group(1) == "0"


def _handle_server_unreachable(
    exc: Exception,
    count: int,
    *,
    armed: bool,
    executor: Optional[Executor] = None,
    paused_path: Optional[Path] = None,
) -> tuple[Decision, int, Optional[str], Optional[str]]:
    """A GET to `/api/discovery/pipeline` failed: increments `count` and
    decides what to do. Returns `(decision, new_count, exec_kind, exec_detail)`
    — `exec_kind`/`exec_detail` are ONLY non-None when this function has already
    really executed a restart (`armed` case with a successful exit): the caller
    (`run_armed_loop`) uses it to skip `_execute_decision` entirely
    (the tick ends here, the next poll will see the real pipeline).

    - 1st/2nd failure (`count < 3`): `Decision("observing", ...)` — only
      waits, never touches anything nor pauses (could be a transient glitch).
    - 3rd+ failure in shadow mode (`armed=False`) or without `executor`:
      `Decision("pause", ...)` exactly as before the 2026-08-15
      incident (shadow never touches the server).
    - 3rd+ failure in armed mode WITH `data/autonomy/PAUSED` already present
      (blocking review finding, 2026-08-15): the hard invariant of
      `_execute_decision` ("while PAUSED exists, never execute anything") was ONLY
      satisfied if every path went through `_execute_decision` — this
      handler calls `executor.execute` BEFORE getting there, so it must
      check `paused_path` itself. If present: `Decision("pause",
      ...)` WITHOUT touching the executor (`count` is kept, not reset) —
      it flows to `_execute_decision`, which already detects `PAUSED` and does nothing
      (same "no action" as any other decision_kind while
      paused). Avoids two real scenarios: (a) supervisor paused for an
      UNRELATED reason + server unreachable 3+ polls -> undue autonomous
      SIGKILL; (b) a failed restart writes PAUSED -> the next tick (with
      the server still down) would try a REAL restart again before
      the 2/24h limit stopped it.
    - 3rd+ failure in armed mode WITHOUT PAUSED: tries `restart_server` via
      the `Executor` (same audited/limited path as `_remediate` —
      validation + limits + `actions_log`). If it succeeds (`rc=0`):
      `Decision("would_remediate", ...)` with `exec_kind="executed"` and
      `count` reset to 0. If the executor refuses it (limit exceeded) or the
      script exits with `rc!=0`: `Decision("pause", ...)` — flows to
      `_execute_decision` like any other pause (mail + PAUSED file),
      `count` is NOT reset."""
    count += 1

    if count < 3:
        return (
            Decision(
                "observing",
                f"server not responding (consecutive failure {count}/3); waiting before acting",
                None,
            ),
            count,
            None,
            None,
        )

    already_paused = bool(paused_path is not None and paused_path.exists())

    if armed and executor is not None and already_paused:
        return (
            Decision(
                "pause",
                f"server hung (3+ consecutive failed GETs) but {paused_path} already exists "
                "-- no action (PAUSED invariant: never execute anything while it is present)",
                None,
            ),
            count,
            None,
            None,
        )

    if armed and executor is not None:
        choice = PlaybookChoice(
            "restart_server",
            {},
            1.0,
            "server hung: 3+ consecutive failed GETs",
            "rules",
        )
        result = executor.execute(choice, trigger="armed_loop")
        if _restart_result_ok(result):
            return (
                Decision(
                    "would_remediate",
                    f"server hung: 3+ consecutive failed GETs — restart_server succeeded: {result.detail}",
                    choice,
                ),
                0,
                "executed",
                result.detail,
            )
        return (
            Decision(
                "pause",
                f"server hung: restart_server did NOT repair it ({result.status}): {result.detail}",
                None,
            ),
            count,
            None,
            None,
        )

    return (
        Decision("pause", f"GET /api/discovery/pipeline failed: {exc!r}", None),
        count,
        None,
        None,
    )


# ---------------------------------------------------------------------------
# Real loop (shadow mode) — the only part that does I/O.
# ---------------------------------------------------------------------------

def _shadow_triage_stub(ctx: ErrorContext) -> PlaybookChoice:
    """AI triage disabled in shadow mode (phase 1): never calls the operator's
    local qwen. Always `pause_and_alert` with `source='fallback'` — the reasoning
    leaves a record in the log that the AI would have been consulted here."""
    return PlaybookChoice(
        "pause_and_alert",
        {"reason": "shadow: AI triage disabled"},
        0.0,
        "shadow: AI triage disabled",
        "fallback",
    )


def _load_state(path: Path) -> SupervisorState:
    if not path.exists():
        return SupervisorState()
    try:
        return SupervisorState.from_dict(json.loads(path.read_text()))
    except (json.JSONDecodeError, OSError) as exc:
        _LOG.warning("supervisor_state.json unreadable (%s); starting from a fresh state", exc)
        return SupervisorState()


def _save_state(path: Path, state: SupervisorState) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state.to_dict(), indent=2))


def _load_plans_safe(path: Path) -> PlansFile:
    if not path.exists():
        _LOG.warning("plans.yaml does not exist at %s; empty queue", path)
        return PlansFile(plans=[], master_list=[])
    try:
        return load_plans(path)
    except Exception as exc:  # noqa: BLE001 - a malformed plans.yaml must never bring down the loop
        _LOG.warning("plans.yaml unreadable (%s); empty queue", exc)
        return PlansFile(plans=[], master_list=[])


def _log_shadow_line(
    log_path: Path,
    ts: str,
    pipeline_state: Optional[str],
    decision: Decision,
    *,
    mode: str = "shadow",
    exec_kind: Optional[str] = None,
    exec_detail: Optional[str] = None,
) -> None:
    """Writes a JSONL line to `shadow.log`. `mode`/`exec_kind`/`exec_detail`
    are new in phase 1.5 (armed) — additional fields, they never replace nor
    reorder the existing ones: an old shadow line (without `mode`) and a
    new one are compatible with the same parser (`.get("mode", "shadow")`)."""
    record = {
        "ts": ts,
        "mode": mode,
        "pipeline_state": pipeline_state,
        "decision_kind": decision.kind,
        "detail": decision.detail,
        "playbook": decision.choice.playbook if decision.choice else None,
        "params": decision.choice.params if decision.choice else None,
        "source": decision.choice.source if decision.choice else None,
        "exec_kind": exec_kind,
        "exec_detail": exec_detail,
    }
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as f:
        f.write(json.dumps(record) + "\n")


def run_shadow_loop(*, once: bool, config_path: Optional[Path]) -> None:
    """Real supervisor loop in shadow mode. Read-only GET, never POST."""
    cfg = load_config(config_path)
    client = ServerClient(cfg.server_url, read_only=True)  # ALWAYS read_only in phase 1

    state_path = cfg.data_dir / "supervisor_state.json"
    heartbeat_path = cfg.data_dir / "heartbeat"
    shadow_log_path = cfg.data_dir / "shadow.log"
    plans_path = cfg.data_dir / "plans.yaml"
    stop_chain_path = REPO_ROOT / "data" / "STOP_CHAIN"

    state = _load_state(state_path)
    server_unreachable_count = 0  # incident 2026-08-15 — lives only inside this process

    while True:
        ts = datetime.now(timezone.utc).isoformat()
        heartbeat_path.parent.mkdir(parents=True, exist_ok=True)
        heartbeat_path.write_text(str(time.time()))

        stop_chain_exists = stop_chain_path.exists()
        try:
            pipeline = client.get("/api/discovery/pipeline")
            pipeline_state = pipeline.get("state") if isinstance(pipeline, dict) else None
            decision = None
            server_unreachable_count = 0
        except Exception as exc:  # noqa: BLE001 - the GET must never bring down the shadow loop
            pipeline = {}
            pipeline_state = None
            # Shadow: never touches the server (armed=False, no executor) —
            # 3+ consecutive failures pause (logged, no mail — shadow
            # never alerts) exactly as before the 2026-08-15 incident.
            decision, server_unreachable_count, _, _ = _handle_server_unreachable(
                exc, server_unreachable_count, armed=False,
            )

        if decision is None:
            plans = _load_plans_safe(plans_path)
            deps = TickDeps(
                pipeline=pipeline,
                plans=plans,
                stop_chain_exists=stop_chain_exists,
                # fleet_health runs separately (cron/timer, spec §5.1); integrating its
                # snapshot (data/autonomy/fleet_health.json) into the tick is out of
                # scope for this task.
                fleet=[],
                classify_fn=classify,
                triage_fn=_shadow_triage_stub,
                state=state,
            )
            decision = tick(deps)

        state.last_poll_ts = time.time()
        _save_state(state_path, state)
        _log_shadow_line(shadow_log_path, ts, pipeline_state, decision)
        _LOG.info("[%s] state=%s decision=%s detail=%s", ts, pipeline_state, decision.kind, decision.detail)
        print(f"[{ts}] state={pipeline_state} decision={decision.kind} detail={decision.detail}")

        if once:
            return
        time.sleep(cfg.poll_interval_s)


# ---------------------------------------------------------------------------
# Real loop (armed mode, phase 1.5) — executes the decisions of `tick()`.
# ---------------------------------------------------------------------------

def _is_armed(cfg: AutonomyConfig) -> bool:
    """Double key (NON-NEGOTIABLE): `config.armed == True` AND
    `EXONIT_AUTONOMY_ARMED == "1"`. Without both, `--armed` falls back to shadow."""
    return bool(cfg.armed) and os.environ.get("EXONIT_AUTONOMY_ARMED") == "1"


def _write_paused(paused_path: Path, reason: str) -> None:
    paused_path.parent.mkdir(parents=True, exist_ok=True)
    paused_path.write_text(
        json.dumps(
            {"ts": datetime.now(timezone.utc).isoformat(), "reason": reason}, indent=2
        )
    )


def _pause_and_alert(
    mailer: Mailer, paused_path: Path, event_type: str, reason: str, alert_cfg: AlertConfig
) -> None:
    """Single pause point of the armed loop (spec §4.2/§5.3): mail (or
    dry_run, already handled by the `Mailer`) + optional external hook
    (`alert.command`, 2026-09-19 — see `run_alert_hook`, does not depend on SMTP)
    + `data/autonomy/PAUSED` with the reason. While this file exists,
    `_execute_decision` executes nothing — it only observes and logs, like
    `STOP_CHAIN` but of internal origin.

    `event_type` is composite (`"pause:<origin>"`, e.g.
    `pause:launch_blocked`/`pause:closeout_blocked`/
    `pause:remediation_refused`/`pause:continuity`) because the rate limit
    of 1 mail/hour of the `Mailer` is per `event_type` — two pauses of different
    origin within the same hour no longer silence each other (review
    round 1, 2026-08-09: before, all shared `event_type="pause"`). The
    external hook does NOT have this rate limit (it is called whenever it pauses) --
    `alert.command`, if configured, must be cheap to call."""
    subject = f"supervisor pause (armed mode) — {event_type}"
    mailer.send(event_type, subject, reason)
    run_alert_hook(alert_cfg, subject, reason)
    _write_paused(paused_path, reason)


def _append_planning_note(data_dir: Path, note: str) -> None:
    """Minimal note in the digest on `would_trigger_planning` (spec §4.1): the
    plan proposal itself is made by the operator outside this process (v1) —
    here we only record that one was needed. Does not touch `closeout.py`
    (same `weekly_digest.md` file, format compatible with the blocks that
    `close_lot` writes there)."""
    digest_path = data_dir / "weekly_digest.md"
    digest_path.parent.mkdir(parents=True, exist_ok=True)
    if not digest_path.exists():
        digest_path.write_text("# Weekly digest — autonomous runs\n", encoding="utf-8")
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
    with digest_path.open("a", encoding="utf-8") as fh:
        fh.write(f"\n## pending planning proposal — {ts} UTC\n- {note}\n")


def _scan_payload(plan: Plan) -> dict:
    payload: dict = {"sector": plan.sector, "count": plan.lot_size}
    if plan.mode == "deep_rescan":
        payload["mode"] = "deep_rescan"
        payload["tics_csv"] = plan.worklist
    return payload


#: Reasons of `{"status": "error", "reason": ...}` (HTTP 400,
#: `scripts/discovery_run.py`) that mean "no work left for this
#: plan" -- equivalent to `enqueued==0` of the 200 branch. ONLY these:
#: `sector_not_enabled`/`sector_not_found`/`deep_mode_flags` are
#: CONFIGURATION errors (malformed plan.yaml), not "plan exhausted" -- those must
#: keep blocking and pausing because they need human eyes, not move on by themselves.
_PLAN_EXHAUSTED_REASONS = frozenset({
    "no_undone_tics",
    "deep_worklist_exhausted",
    "deep_worklist_empty",
    "canonical_list_empty",
})


def _interpret_post_error(exc: Exception) -> tuple[str, str]:
    """Interprets a failure of `client.post('/api/discovery/scan', ...)`.

    Review round 1 (2026-08-09): `ServerClient.post` now raises
    `ServerHTTPError` (HTTP status + JSON body preserved) for any
    non-2xx response, instead of a bare `RuntimeError` -- this makes it possible to
    distinguish the TWO real meanings that `exonit/routes/discovery.py`
    gives to the same HTTP 409 (`already_running` from the anti-double-
    launch guard vs. `need_download` -- TGLC data missing, a real
    data-availability problem, NOT "someone got there first"), and to read the
    real `reason` of an HTTP 400 to detect a naturally exhausted plan
    (`no_undone_tics`, `deep_worklist_exhausted`, ...) instead of only
    knowing that "something failed".

    Returns `(kind, detail)` with `kind` in `{"already_running",
    "need_download", "plan_completed", "blocked"}`."""
    if isinstance(exc, ServerHTTPError):
        body = exc.body or {}
        status = body.get("status")
        reason = body.get("reason")

        if exc.status == 409:
            if status == "need_download":
                hint = body.get("hint") or ""
                return "need_download", (
                    "need_download (HTTP 409): TGLC data not downloaded -- "
                    "human decision (the space/download gate §5.2 is not yet "
                    f"wired): {hint or body}"
                )
            return "already_running", f"POST already in progress (HTTP 409, someone got there first): {body}"

        if reason in _PLAN_EXHAUSTED_REASONS:
            return "plan_completed", f"plan exhausted (HTTP {exc.status}, reason={reason}): {body}"

        return "blocked", f"POST /api/discovery/scan failed (HTTP {exc.status}): {body}"

    # Network failure/timeout with no HTTP response (no body to read).
    return "blocked", f"POST /api/discovery/scan failed: {exc}"


def _run_health_check_gate(repo_root: Path, run_cmd) -> tuple[bool, str]:
    """Preflight of `_launch_lot`: `run_health_check.py --quiet`. `rc==2`
    (CRITICAL) blocks the launch; `rc in (0, 1)` (ok/warning) lets it
    through — a warning does not stop a new lot, only CRITICAL does."""
    cmd = [PY, "scripts/run_health_check.py", "--quiet"]
    try:
        proc = run_cmd(cmd, cwd=str(repo_root), capture_output=True, text=True, timeout=60)
    except Exception as exc:  # noqa: BLE001
        return False, f"run_health_check.py could not be run: {exc!r}"
    rc = getattr(proc, "returncode", 2)
    tail = ((getattr(proc, "stdout", "") or "") + (getattr(proc, "stderr", "") or ""))[-500:]
    if rc == 2:
        return False, f"health check CRITICAL (rc=2): {tail}"
    return True, f"health check rc={rc}: ok to continue"


def _start_fleet_and_wait(
    client: ServerClient,
    fleet_cfg: FleetConfig,
    run_cmd,
    sleep_fn: Callable[[float], None],
    repo_root: Path,
) -> tuple[bool, str]:
    """`farmctl restart --all` (with the explicit `EXONIT_FARMCTL_ALLOW_START=1`
    guard, never `--all` unless the plan requires it — the scan phase
    always uses it, like `_scale_fleet` in the orchestrator) and polling of
    `/api/health/work` until `min_alive` live workers or `start_wait_s`.

    A `farmctl restart` that fails (rc!=0, timeout, exception) is NOT fatal
    by itself — it continues anyway checking `min_alive`, in case the
    fleet was already up from a previous cycle (same non-fatal policy
    as `_scale_fleet` in `discovery_orchestrator.py`); what does block
    the launch is not reaching `min_alive` within the deadline."""
    farmctl_bin = str(repo_root / "bin" / "farmctl")
    env = dict(os.environ, EXONIT_FARMCTL_ALLOW_START="1")
    try:
        proc = run_cmd(
            [farmctl_bin, "restart", "--all"], cwd=str(repo_root),
            capture_output=True, text=True, timeout=180, env=env,
        )
        rc = getattr(proc, "returncode", 1)
        if rc != 0:
            tail = ((getattr(proc, "stdout", "") or "") + (getattr(proc, "stderr", "") or ""))[-500:]
            _LOG.warning("farmctl restart --all rc=%s (continuing, min_alive will be checked): %s", rc, tail)
    except Exception as exc:  # noqa: BLE001 - non-fatal, min_alive is checked anyway
        _LOG.warning("farmctl restart --all failed to run (continuing): %r", exc)

    attempts = max(1, fleet_cfg.start_wait_s // 10)
    alive = 0
    for _ in range(attempts):
        try:
            health = client.get("/api/health/work")
            alive = int(((health or {}).get("workers") or {}).get("alive", 0) or 0)
        except Exception as exc:  # noqa: BLE001 - transient poll, retried
            _LOG.warning("poll /api/health/work failed (continuing): %r", exc)
            alive = 0
        if alive >= fleet_cfg.min_alive:
            return True, f"{alive}/{fleet_cfg.min_alive} workers alive"
        sleep_fn(10)
    return False, f"only {alive}/{fleet_cfg.min_alive} workers alive after {fleet_cfg.start_wait_s}s"


def _sync_sector_ui_state(sector: int, *, repo_root: Path = REPO_ROOT) -> None:
    """Syncs the sector state seen by the UI with the sector the
    supervisor has just launched.

    Incident 2026-08-09: the first armed lot of sector 2 ran fine, but
    the UI kept showing sector 14 -- `exonit/routes/discovery.py`
    (~L862) derives `active_sector` as the FIRST sector with `state=='active'`
    in PG (`discovery_sector_state`), with fallback to the `active_sector` field of
    `data/discovery/sectors.json`; `_launch_lot` touched neither.

    It does, in this order: (a) resolves the DB the same way as `close_lot`
    (`autonomy.closeout._resolve_db`, prioritizes `EXONIT_DATABASE_URL`),
    (b) reads the registry, (c) pauses any other sector that was
    `active` in PG (only one active sector at a time, because the UI takes the
    first), (d) marks `sector` as `active`, (e) updates
    `active_sector` in the registry (UI fallback) and saves it.

    Never raises: it is UI metadata, it cannot block nor kill the launch
    (remember the `launch_timeout` incident of this morning -- the supervisor is
    a long-running service) -- any exception is logged and ignored.
    """
    try:
        db = _resolve_db(repo_root)
        registry_path = str(repo_root / "data" / "discovery" / "sectors.json")
        reg = load_registry(registry_path)

        for key in reg.get("sectors", {}):
            other = int(key)
            if other == sector:
                continue
            if get_sector_state(other, db_path=db) == "active":
                set_sector_state(
                    other, "paused", db_path=db,
                    detail=f"armed supervisor: paused when launching sector {sector}",
                )

        set_sector_state(
            sector, "active", db_path=db,
            detail="armed supervisor: lot launched by _launch_lot",
        )

        reg["active_sector"] = sector
        save_registry(registry_path, reg)
    except Exception as exc:  # noqa: BLE001 - UI metadata, never blocks the launch
        _LOG.warning("sector UI state sync failed (non-fatal): %r", exc)


# Q6 (perf review 2026-09-26, docs/reviews/2026-09-26-enqueue-path-
# performance-review.md): the POST /api/discovery/scan client timeout used
# to be a bare 900s constant while the server's own enqueue deadline
# (EXONIT_SCAN_TIMEOUT_S, prod .env sets 1800) could be, and was, set
# higher -- the supervisor cut the socket at 900s on a lot that was
# enqueuing fine and would have finished well within its own 1800s budget,
# misreporting a healthy-but-slow enqueue as `launch_timeout`. Kept at 900
# by default (unchanged for every existing test/deployment that never sets
# EXONIT_SCAN_TIMEOUT_S) and grown to match the server's deadline plus a
# headroom margin whenever that env var says the server is willing to wait
# longer -- the client must never time out BEFORE the server's own
# enqueue-timeout logic (routes/discovery.py's api_discovery_scan) has had
# a chance to.
_LAUNCH_TIMEOUT_DEFAULT_S = 900.0
_LAUNCH_TIMEOUT_HEADROOM_S = 60.0


def _scan_post_timeout_s() -> float:
    """Client-side timeout for POST /api/discovery/scan (Q6).

    Reads the SAME env var the server checks for its own enqueue deadline
    (``EXONIT_SCAN_TIMEOUT_S`` -- see ``exonit/routes/discovery.py``'s
    ``api_discovery_scan``) so the two never drift the way they did in the
    2026-08-08/2026-09-26 incidents. Unset or non-numeric -> the
    pre-existing 900s default, byte-for-byte (every deployment/test that
    never sets this env var is unaffected).
    """
    raw = os.environ.get("EXONIT_SCAN_TIMEOUT_S")
    if not raw:
        return _LAUNCH_TIMEOUT_DEFAULT_S
    try:
        server_timeout_s = float(raw)
    except (TypeError, ValueError):
        return _LAUNCH_TIMEOUT_DEFAULT_S
    return max(_LAUNCH_TIMEOUT_DEFAULT_S,
               server_timeout_s + _LAUNCH_TIMEOUT_HEADROOM_S)


def _launch_lot(
    client: ServerClient,
    plan: Plan,
    state: SupervisorState,
    cfg: AutonomyConfig,
    *,
    run_cmd=subprocess.run,
    sleep_fn: Callable[[float], None] = time.sleep,
    repo_root: Path = REPO_ROOT,
    executor: Optional[Executor] = None,
) -> tuple[str, str]:
    """Executes `would_launch_next_lot` (spec §4.2): preflight -> fleet ->
    POST /api/discovery/scan. Returns `(kind, detail)` with `kind` in
    `{"launched", "already_running", "need_download", "plan_completed",
    "blocked", "launch_timeout"}` — never raises; the caller decides what to do
    with `"blocked"`/`"need_download"` (both end in `pause_and_alert`,
    see `_execute_decision`). `"launch_timeout"` (POST with no response
    within the timeout, a raw `TimeoutError`/`OSError` that escapes
    `ServerHTTPError`) does NOT pause -- it is self-healing, see the comment on the
    `except OSError` below.

    `executor`, if passed (real run boundary, incident 2026-09-19:
    `resume_run` limit_n=3 only makes sense PER run — see `_SCOPED_PARAM`
    in playbooks.py): when really launching a new lot (`kind == "launched"`)
    `executor.reset_scope("run")` is called, clearing the counters of all
    playbooks with `limit_window == "run"` (`resume_run`,
    `rebuild_streams`) so they do not carry over executions from a previous run.
    `None` (default, existing tests) = no reset -- only whoever has
    real control over the `Executor` resets (`run_armed_loop` via `_execute_decision`)."""
    hc_ok, hc_detail = _run_health_check_gate(repo_root, run_cmd)
    if not hc_ok:
        return "blocked", f"preflight (health check): {hc_detail}"

    fleet_ok, fleet_detail = _start_fleet_and_wait(client, cfg.fleet, run_cmd, sleep_fn, repo_root)
    if not fleet_ok:
        return "blocked", f"preflight (fleet): {fleet_detail}"

    payload = _scan_payload(plan)

    # Sync the UI sector state BEFORE the POST, not after:
    # from here on we are already committed to launching plan.sector (the
    # health and fleet preflights have passed) and the server rewrites
    # `sectors.json` during ENQUEUEING (advances the cursor) -- writing to it
    # after the POST races with that concurrent read-modify-write.
    _sync_sector_ui_state(plan.sector, repo_root=repo_root)

    scan_timeout_s = _scan_post_timeout_s()
    try:
        raw = client.post("/api/discovery/scan", payload, timeout_s=scan_timeout_s)
    except RuntimeError as exc:  # includes ServerHTTPError, subclass of RuntimeError
        kind, detail = _interpret_post_error(exc)
        if kind == "plan_completed":
            state.completed_plan_ids.add(plan.plan_id)
        return kind, detail
    except OSError as exc:
        # TimeoutError/ConnectionError/urllib.error.URLError are all
        # subclasses of OSError. Real incident 2026-08-08: a lot of 300k
        # TICs took more than 900s to enqueue; the server kept
        # enqueuing fine, but `h.getresponse()` (inside `server_client.py::post`,
        # after `h.request()`) raised a raw `TimeoutError` that `urllib`
        # ONLY wraps in `URLError` when `h.request()` fails,
        # not `h.getresponse()` -- so it escaped the `except RuntimeError`
        # above and killed the whole supervisor process (systemd restarted it
        # 30s later). We do NOT mark `plan.plan_id` as completed
        # here -- we do not know whether the POST ever reached the server.
        #
        # Self-healing in both directions: if the POST arrived and the
        # server is still enqueuing, the next tick will see it from the state of the
        # pipeline (ENQUEUEING/SCANNING); if it never arrived, the plan
        # stays uncompleted and the next tick will choose
        # `would_launch_next_lot` again for this same plan and retry the
        # launch.
        return "launch_timeout", (
            f"POST /api/discovery/scan did not respond within the timeout "
            f"({scan_timeout_s:.0f}s): {exc!r} -- "
            "the server is probably still enqueuing; the next tick will verify it "
            "from the pipeline state (ENQUEUEING/SCANNING) and, if needed, will "
            "retry the launch"
        )

    raw = raw or {}
    status = raw.get("status")
    enqueued = raw.get("enqueued")

    if status == "already_running":
        return "already_running", f"someone got there first: {raw}"

    if status in ("started", "dry_run") and enqueued:
        if executor is not None:
            executor.reset_scope("run")
        return "launched", f"lot launched: status={status} enqueued={enqueued} run_id={raw.get('run_id')}"

    # enqueued==0 or an error status equivalent to "no TICs left" (e.g.
    # deep_worklist_exhausted, canonical_list_empty): the plan is considered
    # completed — the next tick will move on to the next plan in the queue or to the
    # continuity policy (§4.6).
    state.completed_plan_ids.add(plan.plan_id)
    return "plan_completed", f"plan {plan.plan_id} completed: status={status} enqueued={enqueued} response={raw}"


def _remediate(
    choice: PlaybookChoice,
    executor: Executor,
    *,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> tuple[str, str]:
    """Executes `would_remediate` (spec §4.3/§4.4): `Executor.execute` already does
    the validation + limits + auditing; here we only translate the result and
    apply the 60s wait after `resume_run` that the spec asks for (giving
    the orchestrator time to resume before the next poll)."""
    result = executor.execute(choice, trigger="armed_loop")
    if result.status in ("refused_limit", "refused_invalid"):
        return "blocked", f"executor refused {choice.playbook}: {result.detail}"
    if choice.playbook == "resume_run" and result.status == "executed":
        sleep_fn(60)
    return "executed", f"{choice.playbook}: {result.status} — {result.detail}"


def _close_lot_armed(
    state: SupervisorState,
    pipeline: dict,
    *,
    run_cmd=subprocess.run,
    repo_root: Path = REPO_ROOT,
    pipeline_state_path: Path = _PIPELINE_STATE_PATH,
    executor: Optional[Executor] = None,
) -> tuple[str, str]:
    """Executes `would_close_lot` (spec §4.5) via the real `close_lot`
    (`dry_run=False, commit=True`). Idempotency: if the pipeline's `run_id`
    is the same as `state.last_closed_run_id`, it does not close again
    (avoids a double close between consecutive polls with the same
    DONE).

    `executor`, if passed (same run boundary as `_launch_lot`,
    incident 2026-09-19): on a real close (`kind == "closed"`)
    `executor.reset_scope("run")` is called — the run that was just closed will no
    longer need `resume_run`, and leaving the counter dirty until the
    next `_launch_lot` only bloats `executor_state.json` for no gain.
    `None` (default, existing tests) = no reset."""
    run_id = (pipeline or {}).get("run_id")
    if run_id and run_id == state.last_closed_run_id:
        return "noop", f"run_id {run_id} already closed (idempotency)"

    result = close_lot(
        pipeline_state_path, dry_run=False, commit=True, repo_root=repo_root, run_cmd=run_cmd,
    )
    if result.status == "blocked":
        blocked_steps = [s for s in result.steps if s["status"] == "blocked"]
        return "blocked", f"closeout blocked for {result.run_id!r}: {blocked_steps}"

    state.last_closed_run_id = run_id or result.run_id
    if executor is not None:
        executor.reset_scope("run")
    return "closed", f"lot {result.run_id} closed: {result.status}"


def _execute_decision(
    decision: Decision,
    pipeline: dict,
    state: SupervisorState,
    plans: PlansFile,
    cfg: AutonomyConfig,
    client: ServerClient,
    mailer: Mailer,
    executor: Executor,
    *,
    run_cmd=subprocess.run,
    sleep_fn: Callable[[float], None] = time.sleep,
    paused_path: Path,
    pipeline_state_path: Path = _PIPELINE_STATE_PATH,
    repo_root: Path = REPO_ROOT,
) -> tuple[str, str]:
    """Really executes the `Decision` of `tick()` in armed mode. Never calls
    `tick()` — it only dispatches its result. `PAUSED` (like `STOP_CHAIN`
    but of internal origin) blocks ANY executive dispatch until the
    operator removes it; the tick keeps being logged anyway."""
    if paused_path.exists():
        return "observing", f"PAUSED present ({paused_path}); no action (decision={decision.kind})"

    if decision.kind in ("sleep_stop_chain", "noop_running"):
        return "noop", decision.detail

    if decision.kind == "would_launch_next_lot":
        plan = current_plan(plans, state.completed_plan_ids)
        if plan is None:
            # plans.yaml is re-read at every decision boundary (spec §4.2):
            # between the pure tick and this dispatch it may have changed (plan
            # completed/removed). Defensive: never launches anything without a plan.
            reason = "would_launch_next_lot but no current plan when re-reading plans.yaml"
            _pause_and_alert(mailer, paused_path, "pause:launch_no_plan", reason, cfg.alert)
            return "paused", reason
        kind, detail = _launch_lot(
            client, plan, state, cfg,
            run_cmd=run_cmd, sleep_fn=sleep_fn, repo_root=repo_root, executor=executor,
        )
        if kind == "launched":
            # S4b: a real launch succeeded -- any FUTURE "scan timed out"
            # FAILED is a NEW incident, not a repeat of one this launch just
            # recovered from. Reset so the single-retry budget is available
            # again.
            state.scan_timeout_retry_count = 0
        if kind == "need_download":
            # TGLC data missing -- a human decision (download more
            # sectors from the Downloads page), not a technical
            # block; its own event_type so it is not confused with
            # "already_running" (both share HTTP 409 in the server's
            # real contract) nor with other pauses.
            _pause_and_alert(mailer, paused_path, "pause:launch_need_download", detail, cfg.alert)
            return "paused", detail
        if kind == "blocked":
            _pause_and_alert(mailer, paused_path, "pause:launch_blocked", detail, cfg.alert)
            return "paused", detail
        return kind, detail

    if decision.kind == "continuity":
        if plans.master_list:
            # The full continuity policy (§4.6) needs the space gate
            # (§5.2, GB/sector calibration) and `deep_coverage` per
            # sector — neither is wired yet (out of scope
            # for this phase 1.5 task; CONCERN in the report). Launching a new
            # sector without the space gate would violate the quality/reproducibility
            # rule of CLAUDE.md, so the safe answer is to pause and alert, not to fake the gate.
            reason = (
                "continuity policy (§4.6) requires the space gate (§5.2) "
                "and deep_coverage per sector, not yet wired — safety pause "
                "instead of launching a sector without a space gate"
            )
        else:
            reason = "no approved plan nor master_list in plans.yaml"
        _pause_and_alert(mailer, paused_path, "pause:continuity", reason, cfg.alert)
        return "paused", reason

    if decision.kind == "would_close_lot":
        kind, detail = _close_lot_armed(
            state, pipeline,
            run_cmd=run_cmd, repo_root=repo_root, pipeline_state_path=pipeline_state_path, executor=executor,
        )
        if kind == "blocked":
            _pause_and_alert(mailer, paused_path, "pause:closeout_blocked", detail, cfg.alert)
            return "paused", detail
        return kind, detail

    if decision.kind == "would_remediate":
        kind, detail = _remediate(decision.choice, executor, sleep_fn=sleep_fn)
        if kind == "blocked":
            _pause_and_alert(mailer, paused_path, "pause:remediation_refused", detail, cfg.alert)
            return "paused", detail
        return kind, detail

    if decision.kind == "pause":
        # Comes from `_tick_error` (no rule nor triage AI found an applicable
        # playbook -- it already escalated by itself) or from a failed pipeline GET
        # (`run_armed_loop`). Its own event_type so it does not share
        # the one-hour rate limit with the other pause origins.
        _pause_and_alert(mailer, paused_path, "pause:escalated", decision.detail, cfg.alert)
        return "paused", decision.detail

    if decision.kind == "would_trigger_planning":
        subject = "pending planning proposal"
        mailer.send("planning_needed", subject, decision.detail)
        run_alert_hook(cfg.alert, subject, decision.detail)
        _append_planning_note(cfg.data_dir, decision.detail)
        return "planning_notified", decision.detail

    return "noop", f"unknown decision_kind, no action: {decision.kind}"


def _heartbeat_status_line(
    client: ServerClient, pipeline_state: Optional[str], pipeline: dict, state: SupervisorState
) -> str:
    """One-line summary for the external heartbeat (`alert.heartbeat_hours`,
    2026-09-19): pipeline state, current run_id, lots closed by the
    supervisor and how many workers are alive right now -- exactly what
    a human would check to know whether the autonomous chain is still alive without
    looking at it. `alive` "unknown" if the GET fails (never blocks the heartbeat
    over a transient network glitch)."""
    run_id = (pipeline or {}).get("run_id")
    lots_done = len(state.completed_plan_ids)
    try:
        health = client.get("/api/health/work")
        alive = ((health or {}).get("workers") or {}).get("alive")
    except Exception:  # noqa: BLE001 - the heartbeat must never bring down the loop
        alive = "unknown"
    return (
        f"state={pipeline_state or 'unknown'} run_id={run_id or 'none'} "
        f"lots_done={lots_done} fleet_alive={alive}"
    )


def _maybe_send_alert_heartbeat(
    alert_cfg: AlertConfig,
    state_path: Path,
    status_line: str,
    *,
    now: Optional[datetime] = None,
) -> Optional[str]:
    """Calls `run_alert_hook` with subject `heartbeat` at most once every
    `alert_cfg.heartbeat_hours` (persisted in `state_path`, a plain ISO
    timestamp) -- so an absent human sees (via `alert.command`) that the autonomous
    chain is still alive. `heartbeat_hours <= 0` (default) = disabled,
    calls nothing. Never raises (a corrupt/missing state file is treated
    as "time to send")."""
    hours = alert_cfg.heartbeat_hours
    if not hours or hours <= 0:
        return None

    now = now or datetime.now(timezone.utc)
    last = None
    try:
        raw = state_path.read_text().strip()
        if raw:
            last = datetime.fromisoformat(raw)
    except (OSError, ValueError):
        last = None

    if last is not None and (now - last).total_seconds() < hours * 3600:
        return None

    result = run_alert_hook(alert_cfg, "heartbeat", status_line)
    try:
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text(now.isoformat())
    except OSError:  # noqa: BLE001 - the heartbeat must never bring down the loop
        pass
    return result


def run_armed_loop(*, once: bool, config_path: Optional[Path]) -> None:
    """Real supervisor loop in ARMED mode (phase 1.5). Without the double
    key (`config.armed == True` and `EXONIT_AUTONOMY_ARMED == "1"`) it falls
    straight to `run_shadow_loop` — shadow mode is EXACTLY the
    same code as in phase 1, never a parallel copy that could diverge."""
    cfg = load_config(config_path)

    if not _is_armed(cfg):
        _LOG.warning(
            "*** --armed requested without the double key (config.armed=%r, "
            "EXONIT_AUTONOMY_ARMED=%r) -- falling back to SHADOW MODE (read-only, no "
            "real execution). To really arm: armed: true in %s AND "
            "EXONIT_AUTONOMY_ARMED=1 in the process environment. ***",
            cfg.armed, os.environ.get("EXONIT_AUTONOMY_ARMED"), config_path,
        )
        run_shadow_loop(once=once, config_path=config_path)
        return

    client = ServerClient(cfg.server_url, read_only=False)  # validated by the double key above
    mailer = Mailer(cfg.mail, cfg.data_dir / "mail_state.json", dry_run=not cfg.mail.enabled)
    executor = Executor(cfg.data_dir / "executor_state.json", cfg.data_dir / "actions.log", dry_run=False)

    state_path = cfg.data_dir / "supervisor_state.json"
    heartbeat_path = cfg.data_dir / "heartbeat"
    alert_heartbeat_path = cfg.data_dir / "alert_heartbeat_last_sent"
    log_path = cfg.data_dir / "shadow.log"
    plans_path = cfg.data_dir / "plans.yaml"
    paused_path = cfg.data_dir / "PAUSED"
    stop_chain_path = REPO_ROOT / "data" / "STOP_CHAIN"

    state = _load_state(state_path)
    server_unreachable_count = 0  # incident 2026-08-15 — lives only inside this process

    while True:
        ts = datetime.now(timezone.utc).isoformat()
        heartbeat_path.parent.mkdir(parents=True, exist_ok=True)
        heartbeat_path.write_text(str(time.time()))

        stop_chain_exists = stop_chain_path.exists()
        restart_exec_kind: Optional[str] = None
        restart_exec_detail: Optional[str] = None
        try:
            pipeline = client.get("/api/discovery/pipeline")
            pipeline_state = pipeline.get("state") if isinstance(pipeline, dict) else None
            decision = None
            server_unreachable_count = 0
        except Exception as exc:  # noqa: BLE001 - the GET must never bring down the loop
            pipeline = {}
            pipeline_state = None
            # Armed: on the 3rd consecutive failure it tries to repair on its own via
            # restart_server before pausing (incident 2026-08-15). Only
            # `restart_exec_kind` is non-None when a successful restart has
            # already really been executed -- the tick ends here, `_execute_decision`
            # is skipped (avoids re-executing the same playbook).
            # `paused_path` is passed explicitly (review finding,
            # 2026-08-15): if PAUSED already exists, `_handle_server_unreachable`
            # never calls the executor -- same invariant as `_execute_decision`.
            decision, server_unreachable_count, restart_exec_kind, restart_exec_detail = (
                _handle_server_unreachable(
                    exc, server_unreachable_count, armed=True, executor=executor, paused_path=paused_path,
                )
            )

        plans = _load_plans_safe(plans_path)

        if cfg.alert.heartbeat_hours and cfg.alert.heartbeat_hours > 0:
            # Only computes the status line (makes an extra GET to /api/health/work)
            # when the heartbeat is enabled -- `heartbeat_hours <= 0` is the
            # default, we do not want an additional GET every tick for no reason.
            status_line = _heartbeat_status_line(client, pipeline_state, pipeline, state)
            _maybe_send_alert_heartbeat(cfg.alert, alert_heartbeat_path, status_line)

        if decision is None:
            deps = TickDeps(
                pipeline=pipeline,
                plans=plans,
                stop_chain_exists=stop_chain_exists,
                fleet=[],
                classify_fn=classify,
                triage_fn=lambda ctx: spawn_and_choose(ctx, cfg.triage_llm),
                state=state,
            )
            decision = tick(deps)

        if restart_exec_kind is not None:
            exec_kind, exec_detail = restart_exec_kind, restart_exec_detail
        else:
            exec_kind, exec_detail = _execute_decision(
                decision, pipeline, state, plans, cfg, client, mailer, executor,
                paused_path=paused_path,
            )

        state.last_poll_ts = time.time()
        _save_state(state_path, state)
        _log_shadow_line(
            log_path, ts, pipeline_state, decision,
            mode="armed", exec_kind=exec_kind, exec_detail=exec_detail,
        )
        _LOG.info(
            "[%s] state=%s decision=%s exec=%s detail=%s",
            ts, pipeline_state, decision.kind, exec_kind, exec_detail,
        )
        print(f"[{ts}] state={pipeline_state} decision={decision.kind} exec={exec_kind} detail={exec_detail}")

        if once:
            return
        time.sleep(cfg.poll_interval_s)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="ExoNit autonomous-loop supervisor (phase 1.5 — shadow or armed mode)."
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--shadow",
        action="store_true",
        help="shadow mode: only observes and logs, never executes anything (client always read-only).",
    )
    mode.add_argument(
        "--armed",
        action="store_true",
        help=(
            "armed mode: really executes launches/closeouts/remediations. "
            "Requires the double key (armed: true in the config AND "
            "EXONIT_AUTONOMY_ARMED=1 in the environment) — without them it falls back to shadow."
        ),
    )
    parser.add_argument("--once", action="store_true", help="a single tick and exit (for smoke tests).")
    parser.add_argument(
        "--config",
        default=None,
        help="path to the YAML config (default: data/autonomy/config.yaml)",
    )
    return parser


def main(argv: Optional[list] = None) -> None:
    logging.basicConfig(level=logging.INFO)
    parser = _build_parser()
    args = parser.parse_args(argv)

    config_path = Path(args.config) if args.config else REPO_ROOT / "data" / "autonomy" / "config.yaml"
    if args.armed:
        run_armed_loop(once=args.once, config_path=config_path)
    else:
        run_shadow_loop(once=args.once, config_path=config_path)


if __name__ == "__main__":
    main()
