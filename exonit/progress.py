"""Progress records for the discovery pipeline's phases.

One record per ``(run_id, phase)``, stored in ``shared_state`` (Redis in
production, MemoryBackend in tests) under ``progress:<run_id>:<phase>`` with a
TTL, so a finished or dead run's progress expires on its own instead of
sitting on the dashboard forever.

Two producers write records, both through ``record_mark``:

* **Scripts**, via ``POST /api/progress`` (``source="post"``) — the phases the
  server has no counter for: ENQUEUEING, POSTPROCESS, FINALIZE, and the
  orchestrator's own ssh/stop steps.
* **The server itself** (``source="server"``) — the queue phases, derived from
  the counters ``/api/health/work`` already computes. Nothing is sent over the
  wire for a number the server already holds, and progress does not freeze if
  the orchestrator dies.

Why ``shared_state`` and not ``pipeline_state.json``: the server runs
``uvicorn --workers 4`` (mode A3) and any of the four processes must answer
``GET /api/discovery/pipeline`` with the same value. A local file would
reintroduce the per-process-state bug class.

WHY THE SAMPLE RING IS A LIST OF PAIRS: everything here round-trips through
JSON on the way to Redis, and JSON turns dict keys into strings. A ring keyed
by timestamp would come back with string keys and every numeric lookup would
silently miss — the exact failure that made the sector gap stats unreadable
for months. ``samples`` is therefore a ``list[[float, int]]``, and no
structure in this module is ever indexed by a number.

This module deliberately imports no FastAPI: it is a pure model, unit-testable
without an app.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent

#: A record outlives the phase that wrote it by an hour, then expires. Long
#: enough that a phase which stops reporting is still visible as *stale* for a
#: good while; short enough that yesterday's run never shows up as today's.
PROGRESS_TTL_S = 3600.0

#: Sample ring bounds — same policy the browser used to apply in localStorage.
RING_MAX_SAMPLES = 40
RING_MAX_AGE_S = 45 * 60

#: A windowed rate needs at least this much elapsed time to mean anything.
RATE_MIN_WINDOW_S = 120.0

#: How far back the trailing rate window reaches. The ring still holds
#: RING_MAX_AGE_S of history (it is what a longer window would be computed
#: from), but the reported rate only looks at the last few minutes — see
#: _compute_rate for why "since this record began" was the wrong question.
RATE_WINDOW_S = 300.0

#: A record is stale once it is older than this many promised intervals.
STALE_INTERVALS = 3

#: Used when an emitter does not declare how often it intends to report.
DEFAULT_INTERVAL_S = 10.0

#: Message/field length ceiling — a record must stay small enough that writing
#: it is never itself a cost.
MAX_TEXT = 500

#: Phases the SERVER feeds from its own counters (see feed_queue_phase). Every
#: other phase is fed by a POST from the script running it.
QUEUE_PHASES = ("SCANNING", "FPP_BACKFILL", "EXOMINER_BACKFILL", "LORA")

#: ENQUEUEING is the one phase with no run_id to report under. The id is
#: derived from the lot CSV that discovery_run.py is in the middle of
#: producing, so it does not exist yet — and pipeline_state.json, which merges
#: rather than replaces, may still be carrying the PREVIOUS run's id at that
#: moment. Reporting under that would attribute this enqueue's progress to a
#: finished run. So the phase uses this fixed key and the reader looks it up
#: by phase instead of by run (see progress_lookup_run_id).
ENQUEUEING_RUN_ID = "__enqueueing__"


def progress_lookup_run_id(phase: str, state_run_id: str | None) -> str | None:
    """Which run key holds *phase*'s progress record.

    Everything reports under its own run_id except ENQUEUEING — see
    ENQUEUEING_RUN_ID for why it cannot.
    """
    if phase == "ENQUEUEING":
        return ENQUEUEING_RUN_ID
    return state_run_id


class ProgressError(ValueError):
    """A malformed progress mark. Callers answer 400; never a 500."""


def progress_key(run_id: str, phase: str) -> str:
    return f"progress:{run_id}:{phase}"


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _text(value, field: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, (str, int, float)):
        raise ProgressError(f"{field} must be a string")
    return str(value)[:MAX_TEXT]


def normalise_mark(payload: dict) -> dict:
    """Validate and coerce an incoming mark into the stored field set.

    Raises ProgressError on anything malformed — the endpoint turns that into
    a 400. Never mutates *payload*.
    """
    if not isinstance(payload, dict):
        raise ProgressError("payload must be an object")

    run_id = _text(payload.get("run_id"), "run_id")
    phase = _text(payload.get("phase"), "phase")
    if not run_id:
        raise ProgressError("run_id is required")
    if not phase:
        raise ProgressError("phase is required")

    try:
        done = int(payload.get("done", 0))
    except (TypeError, ValueError):
        raise ProgressError("done must be an integer")
    if done < 0:
        raise ProgressError("done must be >= 0")

    total = payload.get("total", None)
    if total is not None:
        try:
            total = int(total)
        except (TypeError, ValueError):
            raise ProgressError("total must be an integer or null")
        if total < 0:
            raise ProgressError("total must be >= 0")

    interval_s = payload.get("interval_s", DEFAULT_INTERVAL_S)
    try:
        interval_s = float(interval_s)
    except (TypeError, ValueError):
        raise ProgressError("interval_s must be a number")
    if interval_s <= 0:
        raise ProgressError("interval_s must be > 0")

    pid = payload.get("pid", None)
    if pid is not None:
        try:
            pid = int(pid)
        except (TypeError, ValueError):
            raise ProgressError("pid must be an integer or null")

    return {
        "run_id": run_id,
        "phase": phase,
        "substep": _text(payload.get("substep"), "substep"),
        "unit": _text(payload.get("unit"), "unit"),
        "done": done,
        "total": total,
        "message": _text(payload.get("message"), "message"),
        "host": _text(payload.get("host"), "host"),
        "pid": pid,
        "interval_s": interval_s,
    }


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------

def _iso(ts: float) -> str:
    from datetime import datetime, timezone
    return datetime.fromtimestamp(ts, timezone.utc).isoformat()


def _load(state, key: str) -> dict | None:
    try:
        raw = state.get(key)
    except Exception as exc:                     # Redis down
        logger.warning("progress read failed for %s: %s", key, exc)
        return None
    if not raw:
        return None
    try:
        record = json.loads(raw)
    except Exception:
        return None
    return record if isinstance(record, dict) else None


def _trim_samples(samples: list, now: float) -> list:
    fresh = [s for s in samples
             if isinstance(s, list) and len(s) == 2
             and now - s[0] <= RING_MAX_AGE_S]
    return fresh[-RING_MAX_SAMPLES:]


def record_mark(state, payload: dict, *, now: float | None = None,
                source: str = "post", observed_ts: float | None = None) -> dict:
    """Validate *payload*, merge it into the stored record and persist it.

    Monotonic guard: within one substep ``done`` never goes backwards (marks
    can arrive out of order, and every phase here is monotonic by nature). A
    CHANGE of substep is a legitimate restart of the counter, so it resets the
    record — that is how POSTPROCESS goes from ``discovery_postprocess`` to
    ``scorecard`` inside one phase.

    *observed_ts* is when the counters in *payload* were actually read, which
    is not always ``now``: a server-fed mark can be derived from a cached
    health payload computed minutes earlier (see ``computed_at`` in
    routes/admin.py). The sample ring is dated by observation, because that is
    the clock a RATE must be measured against — dating a stale reading "now"
    compresses the work it represents into the gap since the last mark and
    reports it as a burst. ``updated_ts`` stays on the wall clock: staleness
    means "nobody is reporting", which is about the reporter, not the data.

    Never raises for storage problems (a lost mark is irrelevant); only
    ProgressError from validation propagates.
    """
    now = time.time() if now is None else now
    sample_ts = now if observed_ts is None else float(observed_ts)
    # A reading cannot be from the future, and treating it as such would make
    # the next window's dt negative.
    sample_ts = min(sample_ts, now)
    mark = normalise_mark(payload)
    key = progress_key(mark["run_id"], mark["phase"])
    previous = _load(state, key)

    same_substep = bool(previous) and previous.get("substep") == mark["substep"]
    # Baseline epoch guard (2026-08-23): a change of ``total`` between marks
    # means the phase's baseline was re-seeded underneath us — e.g. the first
    # server-fed mark of a run derived from a STALE health payload still
    # carrying the previous run's baseline (deep-rescan s17: done=51,769 got
    # locked in against total=34,010, remaining pinned to 0 for the whole
    # phase). The monotonic guard exists to absorb out-of-order marks within
    # one counting epoch, not to survive a redefinition of what is being
    # counted — so a total change resets the record exactly like a substep
    # change does. Marks without a total (either side) can't prove an epoch
    # change and keep the old behaviour.
    prev_total = previous.get("total") if previous else None
    same_epoch = same_substep and not (
        prev_total is not None and mark["total"] is not None
        and int(prev_total) != int(mark["total"]))
    if same_epoch:
        mark["done"] = max(mark["done"], int(previous.get("done") or 0))
        samples = _trim_samples(previous.get("samples") or [], now)
        phase_started_ts = float(previous.get("phase_started_ts") or now)
        done_at_start = int(previous.get("done_at_start") or 0)
    else:
        samples = []
        phase_started_ts = sample_ts
        # What `done` already was the first time this record saw the phase.
        # Zero for a phase genuinely starting; non-zero when a record was
        # rebuilt mid-phase (its TTL expired, or a new run reused the key), and
        # then the since-start average MUST NOT count the work that happened
        # before this record existed as if this record had watched it.
        done_at_start = int(mark["done"])

    # Out-of-order arrival: an older observation than one already in the ring
    # tells us nothing new and would invert the window. Drop the sample, keep
    # the mark (done/total/message are still the latest known state).
    if not samples or sample_ts > samples[-1][0]:
        samples = _trim_samples(samples + [[sample_ts, mark["done"]]], now)

    record = dict(mark)
    record["samples"] = samples
    record["phase_started_ts"] = phase_started_ts
    record["done_at_start"] = done_at_start
    record["updated_ts"] = now
    record["updated_at"] = _iso(now)
    record["source"] = source

    try:
        state.set(key, json.dumps(record), ttl=PROGRESS_TTL_S)
    except Exception as exc:                     # Redis down
        logger.warning("progress write failed for %s: %s", key, exc)
    return record


def _compute_rate(record: dict, now: float) -> tuple[float | None, str | None]:
    """Items per second, preferring a windowed rate over a since-start average.

    Returns (rate, source) with source in {"window", "avg", None}. A rate is
    only reported when it is positive and derived from real movement — a phase
    that has not moved reports None rather than 0, so the UI can say "no
    movement" instead of showing an infinite ETA.

    The window is TRAILING (2026-08-01): the widest pair inside the last
    RATE_WINDOW_S, not the widest pair in the whole 45-minute ring. Anchoring
    on the ring's oldest sample meant "what has the rate been since this record
    began", which reads as a rate but is really a running average — it drifts
    for as long as the ring is deep, so anything odd in it (a baseline captured
    against a stale reading, a phase that was fast then slow) stayed on screen
    for up to 45 minutes, slowly decaying, instead of being replaced by what is
    happening now. A trailing window forgets in RATE_WINDOW_S, which is also
    what makes the number match what the logs show while you watch them.
    """
    samples = record.get("samples") or []

    def _pair_rate(pair_samples: list) -> float | None:
        if len(pair_samples) < 2:
            return None
        first, last = pair_samples[0], pair_samples[-1]
        dt = last[0] - first[0]
        d_done = last[1] - first[1]
        if dt >= RATE_MIN_WINDOW_S and d_done > 0:
            return d_done / dt
        return None

    if len(samples) >= 2:
        cutoff = samples[-1][0] - RATE_WINDOW_S
        trailing = [s for s in samples if s[0] >= cutoff]
        rate = _pair_rate(trailing)
        if rate is not None:
            return rate, "window"
        # Early in a phase the trailing window may not span RATE_MIN_WINDOW_S
        # yet. Fall back to the full ring rather than reporting nothing — it is
        # the same answer the trailing window converges to once it fills.
        rate = _pair_rate(samples)
        if rate is not None:
            return rate, "window"

    # Since-start average, net of whatever `done` already was when this record
    # first saw the phase (done_at_start). Missing on records written before
    # 2026-08-01, where 0 restores the previous behaviour exactly.
    done = int(record.get("done") or 0) - int(record.get("done_at_start") or 0)
    elapsed = now - float(record.get("phase_started_ts") or now)
    if done > 0 and elapsed >= 1.0:
        return done / elapsed, "avg"
    return None, None


def read_progress(state, run_id: str, phase: str, *,
                  now: float | None = None) -> dict | None:
    """Return the stored record enriched with age/staleness/rate/ETA.

    Returns None when there is no record (the caller falls back to the
    historic prior, exactly as before this feature existed).
    """
    now = time.time() if now is None else now
    record = _load(state, progress_key(run_id, phase))
    if not record:
        return None

    interval_s = float(record.get("interval_s") or DEFAULT_INTERVAL_S)
    age_s = max(0.0, now - float(record.get("updated_ts") or now))
    rate, rate_source = _compute_rate(record, now)

    total = record.get("total")
    remaining = None
    eta_phase_s = None
    if total is not None:
        remaining = max(0, int(total) - int(record.get("done") or 0))
        if rate and rate > 0:
            eta_phase_s = remaining / rate

    out = {k: v for k, v in record.items() if k != "samples"}
    out["age_s"] = age_s
    out["stale"] = age_s > STALE_INTERVALS * interval_s
    out["rate"] = rate
    out["rate_source"] = rate_source
    out["remaining"] = remaining
    out["eta_phase_s"] = eta_phase_s
    return out


# ---------------------------------------------------------------------------
# Current phase
# ---------------------------------------------------------------------------

def pipeline_state_path() -> str:
    """Same file, same env override, as exonit/routes/discovery.py uses."""
    return os.environ.get(
        "EXONIT_PIPELINE_STATE_FILE",
        str(ROOT / "data" / "discovery" / "pipeline_state.json"))


def current_pipeline_state() -> dict:
    """Read pipeline_state.json. Returns {} when missing or malformed."""
    try:
        path = Path(pipeline_state_path())
        if not path.exists():
            return {}
        data = json.loads(path.read_text())
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


# ---------------------------------------------------------------------------
# Identity handed to a spawned child
# ---------------------------------------------------------------------------

def child_env(run_id: str | None, phase: str, base: dict | None = None) -> dict:
    """Copy of *base* (default os.environ) with the progress identity added.

    Whoever spawns a long-running script is the only one who knows which run
    and which phase it belongs to, so the spawner injects it. Without a run_id
    nothing is added and the child's reporter stays a silent no-op — which is
    exactly what should happen when a script is run by hand.
    """
    env = dict(os.environ if base is None else base)
    if not run_id:
        return env
    env["EXONIT_PROGRESS_RUN_ID"] = run_id
    env["EXONIT_PROGRESS_PHASE"] = phase
    env.setdefault("EXONIT_PROGRESS_URL",
                   os.environ.get("EXONIT_SERVER_URL", "http://localhost:8400"))
    return env


# ---------------------------------------------------------------------------
# Server-side feeder for the queue phases
# ---------------------------------------------------------------------------
#
# SCANNING / FPP_BACKFILL / EXOMINER_BACKFILL / LORA already have exact
# counters inside /api/health/work. Sending those numbers back to the server
# over HTTP would be absurd, and tying them to the orchestrator's liveness
# would freeze the bar whenever it died. So the server writes them itself,
# on the request path of the endpoint that computes them — no background
# thread, no extra query, and it stays fresh for exactly as long as somebody
# (the orchestrator every 30 s, the dashboard every 15 s) is watching.

def feed_queue_phase(state, *, run_id: str, phase: str, health: dict,
                     now: float | None = None) -> dict | None:
    """Derive and store progress for a queue phase. Returns None if not applicable.

    Never overwrites a record whose source is "post": when a phase has both an
    emitter and a counter (LORA), the emitter is the better signal — it knows
    about substeps and about work the counters cannot see.
    """
    now = time.time() if now is None else now
    if phase not in QUEUE_PHASES or not run_id:
        return None

    existing = _load(state, progress_key(run_id, phase))
    if existing and existing.get("source") == "post":
        return None

    mark = None
    if phase == "SCANNING":
        p = health.get("primary_tasks") or {}
        if p.get("baseline") is not None:
            mark = {"unit": "tasks", "done": int(p.get("done") or 0),
                    "total": int(p["baseline"])}
    elif phase == "EXOMINER_BACKFILL":
        e = health.get("exominer_backfill") or {}
        if e.get("baseline") is not None:
            mark = {"unit": "candidates", "done": int(e.get("done") or 0),
                    "total": int(e["baseline"])}
    elif phase == "LORA":
        lo = health.get("lora_explanation") or {}
        if lo.get("baseline") is not None:
            mark = {"unit": "candidates", "done": int(lo.get("done") or 0),
                    "total": int(lo["baseline"])}
    elif phase == "FPP_BACKFILL":
        f = health.get("fpp_backfill") or {}
        if f.get("actionable") is not None:
            actionable = int(f["actionable"])
            # No server-side baseline exists for FPP: the queue depth the FIRST
            # time this run enters the phase IS the baseline. It is stored in
            # the record's own total, so it survives across processes and
            # across a dashboard reload — which the old client-side capture in
            # localStorage did not.
            baseline = (int(existing.get("total"))
                        if existing and existing.get("total") is not None
                        else actionable)
            mark = {"unit": "candidates",
                    "done": max(0, baseline - actionable),
                    "total": baseline}

    if mark is None:
        return None

    payload = {
        "run_id": run_id,
        "phase": phase,
        "substep": None,
        "message": None,
        "host": None,
        "pid": None,
        # The dashboard polls health/work every 15 s and the orchestrator every
        # 30 s; 60 s means a record only goes stale when BOTH have stopped.
        "interval_s": 60.0,
    }
    payload.update(mark)
    # Date the sample by when *health* was computed, not by when this call
    # happens: the payload may have come from the SWR response cache and be
    # older than the request that carried it (routes/admin.py stamps
    # "computed_at" for exactly this). Absent (an older server, a hand-built
    # health dict in a test) falls back to `now`, the previous behaviour.
    observed = health.get("computed_at")
    return record_mark(state, payload, now=now, source="server",
                       observed_ts=observed if isinstance(observed, (int, float))
                       else None)


# ---------------------------------------------------------------------------
# Total ETA
# ---------------------------------------------------------------------------
#
# This lived in the browser until 2026-07-27, which meant every tab had its own
# answer, a reload lost it, and nothing outside the dashboard — a phone, an
# alert, a curl — could ask "when does this finish?". The current phase's ETA
# comes from the live record; every phase after it still comes from the mined
# historic priors, which this feature deliberately does not touch.

#: Home pipeline order, terminal states excluded (see STATES in
#: scripts/discovery_orchestrator.py — AWS_FINALIZE is the AWS-only branch and
#: is not part of the home sequence).
PIPELINE_STAGES = ["SCANNING", "FPP_BACKFILL", "EXOMINER_BACKFILL",
                   "STOP_WORKERS", "POSTPROCESS", "STOP_NVIDIARACK",
                   "LORA", "FREE_GPU", "FINALIZE"]


def build_stages(current_state: str | None) -> list[str]:
    """Ordered stage list for a run currently in *current_state*."""
    stages = list(PIPELINE_STAGES)
    if current_state == "ENQUEUEING":
        return ["ENQUEUEING"] + stages
    if current_state == "AWS_FINALIZE":
        return ["SCANNING", "FPP_BACKFILL", "EXOMINER_BACKFILL",
                "STOP_WORKERS", "POSTPROCESS", "AWS_FINALIZE"]
    return stages


def phase_timings_path() -> str:
    return os.environ.get("EXONIT_PHASE_TIMINGS_FILE",
                          str(ROOT / "www" / "static" / "phase_timings.json"))


def load_priors() -> dict:
    """Mined historic phase durations. Returns {} when unavailable."""
    try:
        return json.loads(Path(phase_timings_path()).read_text())
    except Exception:
        return {}


def prior_estimate_s(priors: dict, phase: str, lot: int | None) -> float | None:
    phases = (priors or {}).get("phases") or {}
    p = phases.get(phase)
    if not p:
        return None
    if p.get("scales_with_lot") and p.get("latest_per_1k_tics_s") is not None and lot:
        return float(p["latest_per_1k_tics_s"]) * (lot / 1000.0)
    if p.get("latest_s") is not None:
        return float(p["latest_s"])
    if p.get("median_s") is not None:
        return float(p["median_s"])
    return None


def eta_total_s(*, state: str, progress: dict | None, priors: dict,
                lot: int | None, now: float | None = None) -> float | None:
    """Seconds until DONE, or None when any part of the sum is unknown.

    Same all-or-nothing policy the browser applied: a partial total is worse
    than no total, because it reads as a promise the pipeline cannot keep.
    """
    if not progress or progress.get("eta_phase_s") is None:
        return None
    stages = build_stages(state)
    if state not in stages:
        return None
    total = float(progress["eta_phase_s"])
    for later in stages[stages.index(state) + 1:]:
        est = prior_estimate_s(priors, later, lot)
        if est is None:
            return None
        total += est
    return total
