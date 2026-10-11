"""System/admin/pipeline-control endpoints (Phase D create_app() split).

Extracted verbatim from exonit.server.create_app() — see
docs/reviews/2026-07-02-opus48-cleanup-review.md and
progress/impl-createapp-split-2026-07-02.md for the module map. Endpoint
bodies below are byte-identical to the pre-split code (git move-detection
should match them); only this file's import/register() preamble is new.

Covers: /api/version, /api/health/work, /api/admin/*, /api/stats,
/api/backfill/fpp, /api/backfill/nfpp, /api/reports, /api/setup/*,
/api/start, /api/stop, /api/status (pipeline control).
"""

import os
import re
import signal
import subprocess
import threading
import time as _time
from collections import defaultdict
from pathlib import Path

from fastapi import Query, Request
from fastapi.responses import JSONResponse

from exonit import __version__
from exonit import progress as progress_model
from exonit import server as _server
from exonit import work_producers, work_streams
from exonit import worker_recycle
from exonit.procutil import reap_in_background as _reap_in_background
from exonit.server import (
    WORKER_OFFLINE_SECONDS,
    _EXOMINER_ASSIGNMENTS_KEY,
    _EXOMINER_BASELINE_PENDING_KEY,
    _EXOMINER_BASELINE_STARTED_AT_KEY,
    _FPP_ASSIGNMENTS_KEY,
    _FPP_BACKFILL_DONE_KEY,
    _FPP_BACKFILL_STARTED_AT_KEY,
    _LORA_BASELINE_PENDING_KEY,
    _LORA_BASELINE_STARTED_AT_KEY,
    _PRIMARY_BASELINE_PENDING_KEY,
    _PRIMARY_BASELINE_STARTED_AT_KEY,
    _PROJECT_ROOT,
    _lora_workload_counts,
    logger,
)
from exonit.routes.deps import ServerDeps
from exonit.shared_state import get_shared_state

# NOTE: _get_shared_metadata_cache is called as `_server.<name>` (not
# plain-imported) below — tests patch it via patch("exonit.server.<name>",
# ...) AFTER the app has already been built; see routes/proxies.py's NOTE
# for the full rationale.


def _backlog_query(db, sql, params=()):
    """Run one of /api/health/work's backlog counters (fpp_by_tic,
    em_by_tic) with planner settings pinned for THIS statement only.

    2026-10-05 (scripts/migration/2026-10-05_backlog_partial_indexes.sql):
    these were the top-3 statements by total time in pg_stat_statements
    (~2 s x ~2.5k calls/day each). Root cause on PG: the OR-of-nullable-
    columns predicate is misestimated by ~4 orders of magnitude (9.18M rows
    estimated vs ~5k actual on candidates_default), which leaves the
    partial-index plan and a Parallel Seq Scan of the 46M-row partition
    within ~2 % of each other in cost — so any ANALYZE/VACUUM flips the plan
    (measured: 72 ms index vs 4.5 s seq scan, same query, before/after a
    VACUUM ANALYZE), and the inflated cost also triggers JIT (~50-400 ms of
    pure compile overhead). ``random_page_cost = 1.1`` (NVMe, hot cache)
    makes the index path win by a wide margin; ``jit = off`` drops the
    compile. Both are ``SET LOCAL`` — scoped to this transaction, never
    leaked to the pooled connection or to any other query. Measured live:
    fpp unscoped 2119 -> 37 ms, em unscoped 2433 -> 225 ms (with the new
    partial indexes), identical row counts. SQLite: unchanged plain query.
    """
    if db.dialect.name != "postgres":
        return db._query(sql, params)
    backend = db.backend
    with backend.transaction() as conn:
        backend.tx_execute(conn, "SET LOCAL jit = off")
        backend.tx_execute(conn, "SET LOCAL random_page_cost = 1.1")
        return backend.tx_query(conn, sql, params)


def register(app, deps: ServerDeps) -> None:
    db = deps.db
    cfg = deps.cfg
    reports_dir = deps.reports_dir
    workers_state = deps.workers_state
    _check_auth = deps.check_auth
    _cached_json = deps.cached_json

    # --- Pipeline state (local Popen handle only) ---
    # Cross-process metadata (pid, log, progress) lives in shared_state under
    # _PIPELINE_STATE_KEY; the Popen handle itself is per-process (NOT
    # serialisable) and stays in this plain dict, exactly as it did as a
    # create_app() local before the split.
    _PIPELINE_STATE_KEY = "pipeline:run_state"          # fields: current / progress / log
    pipeline_state: dict = {
        "process": None,   # Popen handle — local to the spawning process
    }

    def _pipeline_shared(field, default=None):
        val = get_shared_state().hget_json(_PIPELINE_STATE_KEY, field)
        return val if val is not None else default

    def _pipeline_pid_alive():
        cur = _pipeline_shared("current") or {}
        pid = cur.get("pid")
        if not pid:
            return None
        try:
            os.kill(pid, 0)
            return pid
        except (OSError, ProcessLookupError):
            return None

    @app.get("/api/version")
    def api_version():
        return {"version": __version__}

    @app.get("/api/health/work")
    def api_health_work(nocache: int = 0, run_id: str = ""):
        """Single source of truth: every queue we know about, in one call.

        Lets the operator (or a CLI script) verify "is everything actually
        done?" before declaring a run complete. Without this, residual work
        hides in derived queues (FPP backfill, missing LoRA explanations)
        that the regular task table doesn't surface.

        ``run_id`` (2026-07-14) scopes the three work counters — primary_tasks,
        fpp_backfill, exominer_backfill — to the TIC set of that run's lot
        (``discovery_run_tics``). WITHOUT it the counters are global COUNT(*)s
        over the whole candidates/tasks tables, which is right for the UI and
        the Prometheus exporter (they ask "is the *system* idle?") but WRONG as
        a run's phase-exit condition, which asks "is *my lot* done?". A 50k
        sector-14 run hung at EXOMINER_BACKFILL on 43 leftover sector-2
        candidates that can never be scored (FFI targets, no SPOC DV report):
        the global counter could never reach 0, so every future run would have
        hung on the same rows. The orchestrator now passes run_id; every other
        caller keeps the global payload, byte-identical to before.

        A run_id with no materialised TIC set (a run enqueued by an older
        orchestrator, or a typo) falls back to the global counters rather than
        reporting a spurious 0 — reporting "drained" for a scope we cannot
        prove is empty would let a run skip a phase it never actually ran.
        """
        # Resolved OUTSIDE _compute() so a scoped request can never be served
        # from — or written into — the shared 10 s "health_work" cache entry,
        # which holds the GLOBAL payload the UI/exporter poll for.
        #
        # S3 (docs/superpowers/specs/2026-09-26-enqueue-s1-s4-design.md §3.1):
        # db.run_scope_kind() replaces the bare count_discovery_run_tics()
        # check — a run may now be scoped via tasks.run_id ('tasks',
        # EXONIT_RUN_SCOPE=tasks) OR the legacy discovery_run_tics table
        # ('run_tics', the default), or have no materialised scope at all
        # (None, unchanged fallback-to-global behaviour).
        scope_kind = db.run_scope_kind(run_id) if run_id else None
        scoped = scope_kind is not None

        def _compute():
            import shutil
            # Scope fragment (empty string when unscoped => the original SQL,
            # with no extra parameter to bind). Interpolated rather than
            # parameterised because a JOIN clause is structure, not a value;
            # run_id itself stays a bound parameter.
            #
            # cand_scope (fallback path only, joins on the raw `c.tic_id`
            # before aggregation — see fpp_by_tic below for why the fast path
            # scopes AFTER aggregation instead) and post_agg_scope (fast path,
            # joins on the already-aggregated `cand.tic_id`) are deliberately
            # two different fragments; see the 2026-09-25 comment block below.
            #
            # S3: the JOIN varies by scope_kind — the legacy 'run_tics'
            # table (unchanged SQL, run_id filtered IN the ON clause since
            # that table holds every run's rows) or, for 'tasks', a
            # non-correlated derived table that already filters by run_id
            # itself (evaluated once, not per outer row — no MATERIALIZED
            # hint needed), mirroring the run's TIC set straight off
            # tasks.run_id instead of discovery_run_tics.
            if scope_kind == "tasks":
                cand_scope = (
                    " JOIN (SELECT DISTINCT tic_id FROM tasks WHERE run_id = ?) rt "
                    "ON rt.tic_id = c.tic_id ")
                post_agg_scope = (
                    " JOIN (SELECT DISTINCT tic_id FROM tasks WHERE run_id = ?) rt "
                    "ON rt.tic_id = cand.tic_id ")
                scope_p: tuple = (run_id,)
            elif scope_kind == "run_tics":
                cand_scope = (
                    " JOIN discovery_run_tics rt "
                    "ON rt.tic_id = c.tic_id AND rt.run_id = ? ")
                post_agg_scope = (
                    " JOIN discovery_run_tics rt "
                    "ON rt.tic_id = cand.tic_id AND rt.run_id = ? ")
                scope_p = (run_id,)
            else:
                cand_scope = ""
                post_agg_scope = ""
                scope_p = ()

            # Primary BLS task queue.
            #
            # Q8 (perf review 2026-09-26, docs/reviews/2026-09-26-enqueue-
            # path-performance-review.md): the SCOPED (run_id) branch is the
            # orchestrator's phase-exit check (admin.py's own docstring
            # above, "orchestrator polls every 30 s") — it needs a
            # guaranteed-fresh count, so it keeps its own direct JOIN query
            # unchanged. Likewise an UNSCOPED request with nocache=1 (see
            # discovery_orchestrator.py's _health_work_url: run_id can be
            # None/unmaterialised, in which case the orchestrator's own poll
            # falls back to this exact unscoped URL and still expects
            # ground truth) also keeps the direct query — exactly matching
            # get_task_stats_cached()'s own documented caveat ("if [a
            # phase-exit check] is ever added, it MUST call
            # get_task_stats() directly"). Only the remaining case — UNSCOPED
            # and NOT nocache, i.e. the UI dashboard / health polling path
            # (measured 603 calls/7h, ~2.2s each, a parallel seq scan of the
            # whole 34.5M-row tasks table; this path is already served from
            # the 30s _cached_json wrapper below, so it never needed
            # per-poll freshness) — routes through get_task_stats_cached().
            if scoped or nocache:
                if scoped and scope_kind == "tasks":
                    # S3: tasks.run_id makes this a plain, index-only
                    # (idx_tasks_run_id) filter — no JOIN to
                    # discovery_run_tics needed at all.
                    rows = db._query(
                        "SELECT status, COUNT(*) AS n FROM tasks "
                        "WHERE run_id = ? GROUP BY status", (run_id,))
                elif scoped:
                    rows = db._query(
                        """SELECT t.status, COUNT(*) AS n FROM tasks t
                           JOIN discovery_run_tics rt
                             ON rt.tic_id = t.tic_id AND rt.run_id = ?
                           GROUP BY t.status""", (run_id,))
                else:
                    rows = db._query(
                        "SELECT status, COUNT(*) AS n FROM tasks GROUP BY status")
                tasks_by_status = {r["status"]: r["n"] for r in rows}
            else:
                tasks_by_status = db.get_task_stats_cached()

            # FPP backfill (mirror of worker_fpp_task SQL filter).
            # 'actionable' = the candidates the workers will actually pick up;
            # 'unrecoverable' is an estimate of those whose stellar params will
            # likely never be available (no metadata cache entry → no FITS).
            _mc = _server._get_shared_metadata_cache(cfg)
            # Candidates needing FPP, via the predicate-complete partial indexes
            # (idx_cand_fpp_todo / idx_cand_nfpp_todo). The two arms (fpp IS NULL;
            # low-fpp awaiting nfpp) are disjoint, so UNION ALL + GROUP BY is exact
            # and identical to the old single OR query — but ~0.3 ms instead of a
            # ~1 s scan of the cnn>=0.7-or-em_ffi>=0.5 rows when there is little/no
            # work. INDEXED BY is needed because the planner otherwise prefers the
            # broad cnn index; wrapped so a missing index (old DB) falls back to
            # the original query.
            # OR-gate (2026-07-11, docs/reviews/injection_e2e_results_2026-07-11.md
            # + docs/reviews/ffi_cnn_investigation_2026-07-11.md §5): mirrors the
            # literal predicate idx_cand_fpp_todo / idx_cand_nfpp_todo now carry
            # (exonit/db.py) — snr>=10 AND (cnn_score>=0.7 OR exominer_ffi_score>=0.5).
            #
            # GROUP BY c.tic_id (2026-07-12, progress/impl-healthwork-perf.md):
            # the OR-gate widened this query from a few hundred rows to ~410k
            # (309k distinct tic_ids), and the old per-row `_mc.get_entry(tic_id)`
            # loop below did 410k locked SQLite SELECT*+json.loads calls —
            # 18-30 s, enough to blow the orchestrator's 10 s poll timeout and
            # fail a live run. GROUP BY collapses each arm to one row per
            # tic_id with its candidate count, so the Python side never
            # iterates per-candidate — only per distinct tic_id, and the
            # per-tic_id "is it cached?" check is one batched
            # `_mc.cached_tic_ids(...)` (chunked IN queries) instead of N
            # separate locked SELECTs. The reported actionable/unrecoverable
            # counts are unchanged: summing `n` over tic_ids present in the
            # cache is exactly the old per-row `sum(1 for c in ... if
            # get_entry(c["tic_id"]))`, just grouped first.
            #
            # PG rewrite (2026-09-25, progress/impl-healthwork-pg-counters.md):
            # `INDEXED BY` is stripped on PG (db_backend.translate) and, on the
            # live 43M-row `candidates_default` partition, PG's planner mis-
            # estimates this OR-of-nullable-columns predicate by 4 orders of
            # magnitude (8.6M rows estimated, 0 actual — the JOIN to `stars`
            # then built a Parallel Hash from a Parallel Seq Scan of ALL
            # ~9.15M completed stars just to probe against that empty set,
            # ~2.6 s per call even fully idle). A fresh `ANALYZE candidates;
            # ANALYZE stars;` does not fix it — this is column correlation
            # (the OR-gate's own two arms are, by construction, mutually
            # exclusive with `triceratops_fpp`/`triceratops_nfpp` already
            # populated) that per-column planner statistics cannot see, the
            # same class of misestimate as idx_cand_fpp_todo's SQLite/PG
            # guard fix in workers.py's cheap `LIMIT 1` polls.
            #
            # Fix: a MATERIALIZED CTE computes the two (tiny) arms first via
            # UNION ALL + GROUP BY tic_id — this pre-aggregation is itself the
            # trigger that makes PG's planner recognise the CTE's own output
            # is small (a few hundred distinct tic_ids in the common case),
            # so it drives the join to `stars` from the CTE side (a Nested
            # Loop probing `stars`'s PRIMARY KEY per tic_id, or a Hash Join
            # hashing the tiny CTE result) instead of ever building a hash
            # table over the whole `stars` table. Verified live on
            # the production PG (2026-09-25): 2621 ms -> 306 ms unscoped,
            # 1927 ms -> 119 ms scoped (run_id join moved to AFTER the
            # aggregation, on `cand.tic_id` — see post_agg_scope above; moving
            # it there also lets the fpp/nfpp arms keep using their tiny
            # partial indexes regardless of scope, rather than driving a
            # per-partition tic_id lookup for every TIC in the run's lot).
            # Result set is byte-identical to the old two-query
            # Python-side-sum form (same disjoint-arms UNION, just summed in
            # SQL instead of in `fpp_count_by_tic` below) — verified via
            # matching md5(sorted tic_id:n) checksums against prod, both
            # empty (today's real fpp/nfpp backlog is fully drained) and
            # against a substituted non-empty stand-in predicate over the
            # same tables (10.48M tics, checksums equal). SQLite: `WITH ...
            # AS MATERIALIZED` needs sqlite3 >= 3.35 (2021); the bundled
            # Python 3 sqlite3 module on every supported OS is well above
            # that, and SQLite's planner already picks the partial indexes
            # unaided (unaffected either way — MATERIALIZED is accepted as a
            # hint there too, not a syntax addition it lacks).
            try:
                fpp_by_tic = _backlog_query(db,
                    f"""WITH cand AS MATERIALIZED (
                           SELECT tic_id, COUNT(*) AS n FROM (
                             SELECT c.tic_id
                               FROM candidates c INDEXED BY idx_cand_fpp_todo
                              WHERE c.triceratops_fpp IS NULL
                                AND c.snr >= 10.0
                                AND (c.cnn_score >= 0.7 OR c.exominer_ffi_score >= 0.5)
                             UNION ALL
                             SELECT c.tic_id
                               FROM candidates c INDEXED BY idx_cand_nfpp_todo
                              WHERE c.triceratops_nfpp IS NULL
                                AND c.triceratops_fpp >= 0 AND c.triceratops_fpp < 0.5
                                AND c.snr >= 10.0
                                AND (c.cnn_score >= 0.7 OR c.exominer_ffi_score >= 0.5)
                           ) u
                           GROUP BY tic_id
                       )
                       SELECT cand.tic_id, cand.n
                       FROM cand
                       JOIN stars s ON s.tic_id = cand.tic_id
                       {post_agg_scope}
                       WHERE s.status = 'completed'""", scope_p)
            except Exception:
                fpp_by_tic = _backlog_query(db,
                    f"""SELECT c.tic_id, COUNT(*) AS n
                       FROM candidates c
                       JOIN stars s ON c.tic_id = s.tic_id
                       {cand_scope}
                       WHERE c.snr >= 10.0
                         AND (c.cnn_score >= 0.7 OR c.exominer_ffi_score >= 0.5)
                         AND s.status = 'completed'
                         AND (c.triceratops_fpp IS NULL
                              OR (c.triceratops_fpp >= 0
                                  AND c.triceratops_fpp < 0.5
                                  AND c.triceratops_nfpp IS NULL))
                       GROUP BY c.tic_id""", scope_p)
            # The two arms can share a tic_id (a multi-candidate star with one
            # candidate in each state), so sum counts per tic_id rather than
            # assuming the rows are already unique.
            fpp_count_by_tic: dict = defaultdict(int)
            for r in fpp_by_tic:
                fpp_count_by_tic[r["tic_id"]] += r["n"]
            fpp_total = sum(fpp_count_by_tic.values())
            fpp_cached_tics = _mc.cached_tic_ids(fpp_count_by_tic.keys())
            fpp_actionable = sum(n for tic, n in fpp_count_by_tic.items()
                                 if tic in fpp_cached_tics)
            fpp_unrecoverable = fpp_total - fpp_actionable

            # ExoMiner backfill: candidates the verdict pipeline considers
            # strong (cnn_score >= 0.7) but ExoMiner never scored. Same partial-index
            # fast path (idx_cand_exominer_todo) — ~0.1 ms vs ~0.9 s; identical count.
            #
            # GROUP BY c.tic_id + the cached-TIC split (2026-07-14) mirrors the
            # FPP block above, and for the same hard-won reason: the ExoMiner
            # serve loop HOLDS BACK (no serve, no ack) any candidate whose FITS
            # metadata is not in the cache — deliberately, since serving it
            # would make the worker fail its LC load and persist a permanent -1
            # over what may be a real score. A candidate in that state can
            # therefore never leave `pending`, so a counter that includes it is
            # an exit condition that can never be met. 'actionable' counts only
            # what a worker can actually be served; 'unrecoverable' is the rest,
            # reported (not hidden) so the gap stays visible.
            #
            # PG investigation (2026-09-25, progress/impl-healthwork-pg-counters.md):
            # unlike the FPP/NFPP arms above, this query's PG cost is NOT the
            # fpp_by_tic join-side pathology (the `s.status='completed' AND
            # s.has_spoc=1` side is already small and cheap via
            # idx_stars_has_spoc, hashed or nested-loop-probed efficiently
            # either way). The remaining ~0.9-1.5 s (unscoped) is the
            # candidate-side scan itself: `exominer_score IS NULL AND
            # cnn_score >= 0.7` is a genuinely large, real backlog right now
            # (~540k rows in candidates_default alone — Docker-side ExoMiner
            # is OFF in prod, EXONIT_ENABLE_EXOMINER_AUTO=0, so it accumulates)
            # — confirmed via `SET LOCAL enable_seqscan=off`: forcing the
            # idx_cand_exominer_todo bitmap scan measured 1219 ms vs 1289 ms
            # unforced, i.e. no material win once the matching set is this
            # big; the cost is genuine I/O volume, not a bad plan. The scoped
            # variant (JOIN discovery_run_tics) is a DIFFERENT, already-good
            # shape: PG drives from the run's (smaller) TIC list and does a
            # per-partition indexed tic_id lookup into candidates — 1066 ms
            # measured live. Applying the fpp_by_tic-style "materialize the
            # candidate predicate unscoped, then filter by scope after"
            # rewrite here was tried and made the SCOPED case WORSE (1066 ->
            # 1852 ms, +74%): unlike the fpp/nfpp todo-sets, this backlog
            # doesn't shrink to near-nothing, so computing it in full before
            # applying the (usually much smaller) run scope throws away the
            # scope's own selectivity. Left as-is; no query-level fix
            # available without either (a) a genuinely smaller `has_spoc`-side
            # index (already in place) or (b) extended statistics
            # (`CREATE STATISTICS ... (dependencies)` on
            # (cnn_score, exominer_score)) to fix the planner's per-column-
            # independence assumption at the root — a schema/DDL change,
            # explicitly out of scope for this read-only-measurement task.
            # UPDATE 2026-10-05: fixed at the schema level instead — partial
            # (tic_id) indexes idx_cand_em_todo_tic + idx_stars_spoc_completed_tic
            # (scripts/migration/2026-10-05_backlog_partial_indexes.sql) turn
            # this into index-only scans: unscoped 2433 -> 225 ms; see
            # _backlog_query for the per-statement planner settings.
            try:
                em_by_tic = _backlog_query(db,
                    f"""SELECT c.tic_id, COUNT(*) AS n
                       FROM candidates c INDEXED BY idx_cand_exominer_todo
                       JOIN stars s ON c.tic_id = s.tic_id
                       {cand_scope}
                       WHERE c.exominer_score IS NULL
                         AND c.cnn_score >= 0.7
                         AND s.status = 'completed'
                         AND s.has_spoc = 1
                       GROUP BY c.tic_id""", scope_p)
            except Exception:
                em_by_tic = _backlog_query(db,
                    f"""SELECT c.tic_id, COUNT(*) AS n
                       FROM candidates c
                       JOIN stars s ON c.tic_id = s.tic_id
                       {cand_scope}
                       WHERE s.status = 'completed'
                         AND c.cnn_score >= 0.7
                         AND c.exominer_score IS NULL
                         AND s.has_spoc = 1
                       GROUP BY c.tic_id""", scope_p)
            em_count_by_tic: dict = defaultdict(int)
            for r in em_by_tic:
                em_count_by_tic[r["tic_id"]] += r["n"]
            exominer_pending = sum(em_count_by_tic.values())
            em_cached_tics = _mc.cached_tic_ids(em_count_by_tic.keys())
            exominer_actionable = sum(n for tic, n in em_count_by_tic.items()
                                      if tic in em_cached_tics)
            exominer_unrecoverable = exominer_pending - exominer_actionable

            # LoRA explanations split by category:
            #   stale  = has rich analysis but not from [lora-v2] (legacy Claude
            #            text or server-side write before the prefix was added).
            #            Worth regenerating to keep style consistent.
            #   missing = has no analysis at all (or shorter than the threshold).
            #            generate_missing_explanations.py targets these.
            #
            # When EXONIT_LORA_STRONG_ONLY=1 (discovery runs with --strong-only),
            # only planet_candidate rows are counted so the progress bar reflects
            # the actual workload instead of showing 0/N forever.
            # Read per-call so flipping the env var takes effect without restart.
            #
            # PG measurement (2026-09-25, progress/impl-healthwork-pg-counters.md):
            # this is the only caller of `_lora_workload_counts`
            # (exonit/server.py). Measured live on the production PG under the
            # actual prod config (EXONIT_LORA_STRONG_ONLY=1, per .env) —
            # lora_stale 826 ms, lora_missing 890 ms. Both already drive off
            # the pre-existing `idx_candidates_verdict_snr (claude_verdict,
            # snr DESC)` index without any planner misestimate (the
            # `claude_verdict='planet_candidate'` predicate is a plain,
            # accurately-estimated equality, not the OR-of-nullable-columns
            # correlation that hurts fpp_by_tic/em_by_tic above), and
            # lora_missing's stars join is already a cheap Nested Loop
            # probing `stars_pkey` from the small candidate-side result — the
            # same shape this task's fpp_by_tic rewrite forces for the other
            # arms. No rewrite applied: nothing to fix under the current prod
            # config. (Measured WITHOUT strong_only — i.e. the pre-2026-xx
            # EXONIT_LORA_STRONG_ONLY=0 shape, not currently active in prod —
            # lora_missing costs 6.66 s instead, because
            # `(cnn_score>=0.7 OR exominer_ffi_score>=0.5)` alone has no
            # matching partial index and the stars join reverts to the same
            # whole-`stars`-hash pattern as the unfixed fpp_by_tic; if that
            # flag is ever turned off in prod, this arm reopens THIS commit's
            # bug and needs the same fix — out of scope here as the flag is
            # off in every deployed config today.)
            lora_strong_only = os.environ.get("EXONIT_LORA_STRONG_ONLY") == "1"
            lora_stale, lora_missing = _lora_workload_counts(db, lora_strong_only)

            # Snapshot baselines on first call after server start so the UI
            # progress bars have a stable reference (both queues' pending count
            # can fluctuate as workers add new candidates).
            # A2a: set-once via set_if_absent; max-bump via get_int+set_int
            # (cosmetic race acceptable — see comment near key definitions above).
            _store = get_shared_state()
            _store.set_if_absent(
                _EXOMINER_BASELINE_PENDING_KEY, str(exominer_pending))
            _store.set_if_absent(
                _EXOMINER_BASELINE_STARTED_AT_KEY, str(_time.time()))
            _stored_em = _store.get_int(_EXOMINER_BASELINE_PENDING_KEY)
            if exominer_pending > _stored_em:
                _store.set_int(_EXOMINER_BASELINE_PENDING_KEY, exominer_pending)
            # LoRA workload counts both stale and missing — both need regeneration.
            lora_workload = lora_stale + lora_missing
            _store.set_if_absent(
                _LORA_BASELINE_PENDING_KEY, str(lora_workload))
            _store.set_if_absent(
                _LORA_BASELINE_STARTED_AT_KEY, str(_time.time()))
            _stored_lora = _store.get_int(_LORA_BASELINE_PENDING_KEY)
            if lora_workload > _stored_lora:
                _store.set_int(_LORA_BASELINE_PENDING_KEY, lora_workload)

            # Workers: stuck and duplicate-current-tic detection.
            now = _time.time()
            alive = []
            stuck = []
            cur_tics: dict = {}
            for wid, w in workers_state.items():
                if now - w.get("last_heartbeat", 0) >= WORKER_OFFLINE_SECONDS:
                    continue
                alive.append(wid)
                tic = w.get("current_tic")
                if tic:
                    cur_tics.setdefault(tic, []).append(w.get("name", wid))
                    tic_started = w.get("tic_started_at", 0)
                    if tic_started and now - tic_started > 1800:
                        stuck.append({
                            "name": w.get("name", wid),
                            "tic_id": tic,
                            "stuck_seconds": int(now - tic_started),
                        })
            # Backfill workers (FPP / ExoMiner) on different candidates of the
            # same multi-candidate TIC report identical current_tic at heartbeat
            # time. That's legitimate parallelism, not a fault — filter it out
            # by excluding TICs whose primary task is already completed.
            potential_dupes = [tic for tic, names in cur_tics.items()
                               if len(names) > 1]
            completed_dupe_tics: set = set()
            if potential_dupes:
                placeholders = ",".join("?" * len(potential_dupes))
                try:
                    completed_rows = db._query(
                        f"SELECT tic_id FROM tasks WHERE tic_id IN "
                        f"({placeholders}) AND status = 'completed'",
                        tuple(str(t) for t in potential_dupes))
                    completed_dupe_tics = {str(r["tic_id"])
                                           for r in completed_rows}
                except Exception:
                    completed_dupe_tics = set()
            duplicates = {tic: names for tic, names in cur_tics.items()
                          if len(names) > 1
                          and str(tic) not in completed_dupe_tics}

            # Disk space (root partition)
            try:
                du = shutil.disk_usage("/")
                disk_free_gb = round(du.free / (1024 ** 3), 1)
                disk_pct_used = round((du.used / du.total) * 100, 1)
            except OSError:
                disk_free_gb = None
                disk_pct_used = None

            # Health flags — used by UI to color the card.
            warnings = []
            critical = []
            if disk_free_gb is not None and disk_free_gb < 1:
                critical.append(f"disk_free_gb={disk_free_gb} (<1GB)")
            elif disk_free_gb is not None and disk_free_gb < 5:
                warnings.append(f"disk_free_gb={disk_free_gb} (<5GB)")
            if duplicates:
                critical.append(f"duplicate_tics={len(duplicates)}")
            if stuck:
                warnings.append(f"stuck_workers={len(stuck)}")
            if not alive:
                warnings.append("no_workers_alive")

            # Primary baseline snapshot (session-level "X done of Y in this run").
            # A2a: set-once + max-bump via shared store (cosmetic race acceptable).
            primary_remaining = (tasks_by_status.get("pending", 0)
                                 + tasks_by_status.get("assigned", 0))
            _store.set_if_absent(
                _PRIMARY_BASELINE_PENDING_KEY, str(primary_remaining))
            _store.set_if_absent(
                _PRIMARY_BASELINE_STARTED_AT_KEY, str(_time.time()))
            _stored_primary = _store.get_int(_PRIMARY_BASELINE_PENDING_KEY)
            if primary_remaining > _stored_primary:
                # Auto-bump on a new burst (e.g. catalog re-enqueue) so the
                # UI never shows "1275/888 = 144%".
                _store.set_int(_PRIMARY_BASELINE_PENDING_KEY, primary_remaining)
                _store.set(_PRIMARY_BASELINE_STARTED_AT_KEY, str(_time.time()))
                _stored_primary = primary_remaining
            primary_baseline = _stored_primary
            primary_done = max(0, primary_baseline - primary_remaining)

            # Read baseline values for display.
            # get() returns None if never set (first call before api_health),
            # allowing the done=0 branch exactly as the original None check did.
            _em_baseline_raw = _store.get(_EXOMINER_BASELINE_PENDING_KEY)
            _em_baseline_pending: int | None = (
                int(_em_baseline_raw) if _em_baseline_raw is not None else None)
            _lora_baseline_raw = _store.get(_LORA_BASELINE_PENDING_KEY)
            _lora_baseline_pending: int | None = (
                int(_lora_baseline_raw) if _lora_baseline_raw is not None else None)

            payload = {
                # When these counters were actually read (2026-08-01). The
                # payload can be served from the SWR cache long after that, and
                # a consumer which timestamps it "now" draws wrong conclusions:
                # the progress sample ring did exactly that, dating a 50-minute
                # -old FPP reading as current and turning the gap between it and
                # the next fresh reading into a fake burst of throughput. Any
                # consumer measuring a RATE from this payload must use this
                # field, not its own clock.
                "computed_at": _time.time(),
                "primary_tasks": {
                    "pending": tasks_by_status.get("pending", 0),
                    "assigned": tasks_by_status.get("assigned", 0),
                    "completed": tasks_by_status.get("completed", 0),
                    "failed": tasks_by_status.get("failed", 0),
                    "baseline": primary_baseline,
                    "done": primary_done,
                },
                "fpp_backfill": {
                    "actionable": fpp_actionable,
                    "unrecoverable": fpp_unrecoverable,
                    "currently_assigned": _store.hlen(_FPP_ASSIGNMENTS_KEY),
                },
                "exominer_backfill": {
                    # 'pending' keeps its original meaning (every unscored
                    # candidate, cached or not) so the UI and the Prometheus
                    # exporter are untouched. 'actionable' is the subset a
                    # worker can actually be served, and is what a run's exit
                    # condition must wait on — see the comment above the query.
                    "pending": exominer_pending,
                    "actionable": exominer_actionable,
                    "unrecoverable": exominer_unrecoverable,
                    "currently_assigned": _store.hlen(_EXOMINER_ASSIGNMENTS_KEY),
                    "baseline": _em_baseline_pending,
                    "done": (max(0, _em_baseline_pending - exominer_pending)
                             if _em_baseline_pending is not None else 0),
                },
                "lora_explanation": {
                    "stale": lora_stale,
                    "missing": lora_missing,
                    "baseline": _lora_baseline_pending,
                    "done": (max(0, _lora_baseline_pending - lora_workload)
                             if _lora_baseline_pending is not None else 0),
                },
                "workers": {
                    "alive": len(alive),
                    "stuck": stuck,
                    "duplicate_tics": duplicates,
                },
                "disk": {
                    "free_gb": disk_free_gb,
                    "pct_used": disk_pct_used,
                },
                "status": (
                    "critical" if critical else
                    "warning" if warnings else
                    "ok"
                ),
                "warnings": warnings,
                "critical": critical,
                # Explicit so a caller that ASKED for a scope can tell whether
                # it got one: run_id given but scoped=false means the run has
                # no materialised TIC set and these counters are global. A
                # silent fallback is how a run ends up waiting on (or skipping)
                # work that was never its own.
                "scope": {"run_id": run_id or None, "scoped": scoped,
                         "kind": scope_kind},
            }

            # Phase 1 (streams): the legacy claims-hash ``currently_assigned``
            # above reads a hash the streams path never writes, so with the
            # flag on it always reports 0 inflight and the DLQ — this
            # design's core safety-visibility promise — stays invisible
            # outside redis-cli. ADD a work_streams block with the real
            # per-stage queue depths instead of repurposing
            # currently_assigned, so the exporter's existing parsing of that
            # key is untouched; flag off -> payload identical to before.
            # Never let a Redis hiccup 500 this endpoint.
            if work_producers.streams_enabled():
                try:
                    stream_stats = {
                        stage: work_streams.get_stream(stage).stats()
                        for stage in ("fpp", "exominer", "primary")
                    }
                    payload["work_streams"] = stream_stats
                    payload["fpp_backfill"]["currently_assigned_streams"] = (
                        stream_stats["fpp"]["pending"])
                    payload["exominer_backfill"]["currently_assigned_streams"] = (
                        stream_stats["exominer"]["pending"])
                except Exception:
                    logger.warning(
                        "work_streams stats collection failed", exc_info=True)

            return payload

        def _feed_progress(payload: dict) -> None:
            """Feed the queue-phase progress record from *payload*.

            Only for a scoped request: the global payload is the "is the system
            idle?" view and is not tied to any run. Wrapped whole — telemetry
            may never take this endpoint down.

            2026-08-01: this used to live only on the cached path below, so the
            ONLY feeder was an open dashboard — the orchestrator's own poll
            carries nocache=1 and returned before ever reaching it. With no
            dashboard open, nothing refreshed the record for the length of a
            phase; it expired (PROGRESS_TTL_S, 1 h) and the next dashboard poll
            rebuilt it from scratch, re-capturing the FPP baseline from
            whatever `actionable` that poll happened to see. When that poll was
            served a stale SWR entry, the rebuild anchored the baseline minutes
            in the past and the next fresh payload moved `done` by tens of
            thousands in one step — a fake step the rate window then reported
            as ~7,000 TIC/min, decaying back to plausible as the window widened
            (observed on discovery_s0014_lot20260731T232258, FPP_BACKFILL:
            stored total=32,951 for a phase that entered at 53,429, and
            phase_started_ts 79 min after the phase actually began).

            Feeding from the nocache path too means the orchestrator's 30 s
            poll keeps the record alive for the whole phase whether or not a
            browser is open — so it never expires mid-phase and never needs
            rebuilding. That poll is also ground truth (it bypasses the cache),
            which makes it the better sample of the two.
            """
            if not scoped:
                return
            try:
                pipeline = progress_model.current_pipeline_state()
                if pipeline.get("run_id") == run_id:
                    progress_model.feed_queue_phase(
                        get_shared_state(), run_id=run_id,
                        phase=pipeline.get("state") or "", health=payload)
            except Exception as exc:
                logger.debug("queue progress not fed: %s", exc)

        # nocache=1 lets a backend caller (the discovery orchestrator's phase
        # polls) bypass the 10 s UI cache and read ground-truth task counts —
        # without it, SCANNING could read a stale pending=0 right after a lot
        # was enqueued and skip the wait (the gap-fill SCANNING race).
        if nocache:
            payload = _compute()
            _feed_progress(payload)
            return payload
        # The cache key MUST carry the scope. The docstring above already
        # promised that "a scoped request can never be served from — or
        # written into — the shared cache entry", but the key was a bare
        # "health_work" for every caller: the dashboard (scoped to the live
        # run) and the Prometheus exporter (global) then overwrote each
        # other's entry every 10 s, so the run's progress bar alternated
        # between the run's own counters and the global ones — reading as a
        # bar that jumps backwards or sits frozen. Keyed by scope, each
        # audience gets its own entry, at the cost of one extra cached
        # payload. (Every other parameterised endpoint here already does
        # this: see `sky_completed:{limit}` and `api_stars:{limit}:{offset}`.)
        cache_key = f"health_work:{run_id}" if scoped else "health_work"
        # A scoped request is cached too — under its OWN key, which is the
        # whole point of cache_key above. Calling _compute() directly for
        # scoped requests would re-run every COUNT(*) on tables of millions of
        # rows for each poll, and this endpoint is polled by the orchestrator
        # every 30 s AND by every open dashboard every 15 s.
        #
        # TTL raised 10.0 -> 30.0 on 2026-07-31: measured 2026-07-31 from
        # <host> against PG on <host>, the global _compute() here costs
        # 9-14s — LONGER than
        # the old 10s TTL, so the cache almost never actually hit and every
        # 15s UI poll paid the full cost. With _cached_json now
        # stale-while-revalidate (exonit/server.py), the TTL no longer gates
        # perceived latency (a stale entry is still served instantly) — it
        # only sets how often the background thread re-hits PG. 30s keeps
        # that refresh cadence sane relative to the 9-14s compute cost.
        # nocache=1 (line ~490 above) is unaffected either way.
        payload = _cached_json(cache_key, 30.0, _compute)

        # Feed the queue-phase progress record as a side effect of the call
        # that already computed these numbers (see _feed_progress above).
        _feed_progress(payload)
        return payload

    def _reset_baselines_impl() -> list:
        """Clear all progress-bar baselines. Shared by /api/admin/reset_baselines
        and /api/admin/reset_all (which used to carry a drifting copy).

        The body moved to exonit.server.reset_progress_baselines on
        2026-07-26, when /api/discovery/scan became a third caller: this is a
        closure over register()'s scope and a route module cannot reach it.
        Kept as a name because two endpoints here already call it."""
        return _server.reset_progress_baselines()

    def _reset_worker_stats_impl() -> int:
        """Zero per-worker counters (registry + DB). Shared by
        /api/admin/reset_worker_stats and /api/admin/reset_all."""
        n = 0
        for wid in workers_state.keys():
            # P6 (2026-07-02): one .update() round trip (read+write) instead
            # of three separate __setitem__ calls (three read+write pairs).
            workers_state[wid].update({
                "tasks_completed": 0,
                "tasks_failed": 0,
                "total_processing_time": 0.0,
            })
            n += 1
        db.reset_all_worker_stats()  # also zero persisted rows (incl. offline workers)
        return n

    @app.post("/api/admin/reconcile_stream")
    def api_reconcile_stream(request: Request, stage: str = "fpp"):
        """Top up a work stream with the DB rows it is missing.

        The queue and the DB can disagree in BOTH directions, and neither drift
        heals on its own (both stranded a run on 2026-07-14):

          * eligible in PG, in no queue — a writer that bypassed the producer
            hooks, or a late has_spoc flip. The counter says "pending" and no
            worker can ever be served it.
          * dead-lettered but still eligible in PG — an item held back for a
            missing metadata entry leaves the queue after max_deliveries, but
            its PG row is untouched, so it counts as pending forever. 56 FPP
            candidates had been in that state since 2026-07-12.

        The orchestrator calls this at the start of every backfill phase, so a
        phase never begins waiting on a counter it has no way to move.
        Non-destructive: in-flight claims are preserved (unlike
        scripts/rebuild_work_streams.py, which clears and rebuilds).
        """
        if not _check_auth(request):
            return JSONResponse({"error": "unauthorized"}, status_code=401)
        if stage not in ("fpp", "exominer", "primary"):
            return JSONResponse({"error": f"unknown stage {stage!r}"},
                                status_code=400)
        try:
            result = work_producers.reconcile_stream(db, stage)
        except Exception as exc:
            logger.warning("reconcile_stream(%s) failed", stage, exc_info=True)
            return JSONResponse({"error": str(exc), "stage": stage},
                                status_code=500)
        if result.get("enqueued") or result.get("dlq_cleared"):
            logger.info("reconcile_stream(%s): %s", stage, result)
        return result

    @app.post("/api/admin/recycle-workers")
    def api_recycle_workers(request: Request):
        """Bump ``recycle:generation`` so every A3 uvicorn worker process
        notices (within its next ~30s recycle-loop tick) that it should
        self-recycle — staggered one at a time via ``recycle:slot``'s
        SET NX EX=180 lock. See exonit/worker_recycle.py + docs/superpowers/
        specs/2026-07-17-worker-recycle-postrun.md.

        Callable by hand (any worker_token holder) or by the discovery
        orchestrator, best-effort, at the end of FINALIZE.

        Same Bearer worker_token auth pattern as /api/admin/reconcile_stream
        just above — unlike the UI-only reset_* endpoints below, this one
        eventually kills and respawns real server worker processes, so it
        is not left unauthenticated like those.
        """
        if not _check_auth(request):
            return JSONResponse({"error": "unauthorized"}, status_code=401)
        generation = get_shared_state().incr(worker_recycle.RECYCLE_GENERATION_KEY)
        logger.info("recycle-workers: generation bumped to %d", generation)
        return {"generation": generation}

    @app.post("/api/admin/reset_baselines")
    def api_reset_baselines():
        """Reset the primary/exominer/lora/fpp baselines so the UI
        progress bars show "0 of <current_pending>" again.

        Useful when launching a new bulk enqueue (e.g. catalog re-run) —
        otherwise the previous run's baseline stays frozen and the UI
        shows nonsense like "1275/888".

        Unauthenticated like /api/workers/start: the dashboard (browser) calls
        it with no worker_token. Only clears in-memory baselines, never raw data.
        """
        return {"ok": True, "reset": _reset_baselines_impl()}

    @app.post("/api/admin/reset_worker_stats")
    def api_reset_worker_stats():
        """Zero per-worker counters (tasks_completed, tasks_failed,
        total_processing_time) both in memory and in workers_registry, so the
        dashboard per-worker average time restarts at 0 for a new run.

        reset_baselines handles the progress bars; this handles the worker rows.
        Both are meant to be called together when launching a fresh run.

        Unauthenticated like /api/workers/start (dashboard has no worker_token);
        only zeroes counters, never deletes raw data.
        """
        return {"ok": True, "workers_reset": _reset_worker_stats_impl()}

    @app.post("/api/admin/mark_unconfirmed_ffi")
    def api_mark_unconfirmed_ffi():
        """Bulk-mark exominer_score=-1 for candidates whose star has has_spoc=NULL.

        This is the safe entry-point to the SPOC-gate bulk clean-up path.  It
        reproduces in one bulk UPDATE what the per-TIC background gate would have
        written if it had not crashed/timed-out during the scan phase.

        TIMING: call only at the entry to EXOMINER_BACKFILL (post-scan,
        post-FPP_BACKFILL).  At that point no more worker_task_result POSTs can
        arrive for this lot (primary tasks drained), so no race with an in-flight
        SPOC gate thread that might still resolve has_spoc=1.

        Returns the number of rows updated.  Idempotent: a second call on the
        same DB is a no-op (already-marked rows are filtered by IS NULL guard).

        Unauthenticated like /api/admin/reset_baselines — the orchestrator calls
        this from the server host with no worker_token.
        """
        n = db.mark_unconfirmed_ffi_candidates()
        logger.info(
            "mark_unconfirmed_ffi: bulk-marked %d candidates exominer_score=-1"
            " (has_spoc=NULL stars)", n)
        return {"ok": True, "marked": n}

    @app.get("/api/stats")
    def api_stats():
        def _compute():
            stats = db.get_stats()
            # COUNT via the idx_cand_strong partial index (~0.1 ms) instead of
            # SELECT * + len() over the ~355k cnn>=0.7 rows (~0.9 s).
            stats["strong_candidates"] = db.count_strong_candidates(min_snr=10)
            return stats
        # 60 s, not 10 s. Measured on <host> mid-run (2026-07-25) this
        # aggregate takes 4.6-12 s of PG time, and it was being recomputed on
        # roughly every other dashboard poll — each recompute pinning one of
        # the four uvicorn workers for that whole time, which is what made
        # every page feel slow while a run was scanning. The numbers it
        # serves are cumulative lifetime counters in the millions: a minute
        # of staleness is invisible, the 12 s stall was not.
        return _cached_json("stats", 60.0, _compute)

    @app.get("/api/backfill/fpp")
    def api_backfill_fpp():
        """Progress for the FPP backfill (candidates without TRICERATOPS FPP).

        Mirrors the worker_fpp_task SQL filter so the UI shows the same queue
        the workers are draining.

        ``elapsed_seconds`` and ``rate_per_min`` are measured from the moment
        the first FPP result is written back (seeded by the increment path at
        ~line 2336), NOT from the first UI poll.  Before any done has been
        recorded both values are 0.0 and the started_at key is absent.
        """
        def _compute():
            # Same cheap partial-index path as /api/health/work: two disjoint arms
            # (fpp IS NULL; low-fpp awaiting nfpp) summed = identical to the old OR
            # COUNT, but ~0.2 ms instead of a ~1 s full scan. Fallback on missing idx.
            # OR-gate (2026-07-11) — see /api/health/work above for the rationale.
            try:
                pending = db._query_scalar(
                    """SELECT COUNT(*) FROM candidates c INDEXED BY idx_cand_fpp_todo
                       JOIN stars s ON c.tic_id = s.tic_id
                       WHERE c.triceratops_fpp IS NULL
                         AND c.snr >= 10.0
                         AND (c.cnn_score >= 0.7 OR c.exominer_ffi_score >= 0.5)
                         AND s.status = 'completed'""") + db._query_scalar(
                    """SELECT COUNT(*) FROM candidates c INDEXED BY idx_cand_nfpp_todo
                       JOIN stars s ON c.tic_id = s.tic_id
                       WHERE c.triceratops_nfpp IS NULL
                         AND c.triceratops_fpp >= 0 AND c.triceratops_fpp < 0.5
                         AND c.snr >= 10.0
                         AND (c.cnn_score >= 0.7 OR c.exominer_ffi_score >= 0.5)
                         AND s.status = 'completed'""")
            except Exception:
                pending = db._query_scalar(
                    """SELECT COUNT(*) FROM candidates c
                       JOIN stars s ON c.tic_id = s.tic_id
                       WHERE c.snr >= 10.0
                         AND (c.cnn_score >= 0.7 OR c.exominer_ffi_score >= 0.5)
                         AND s.status = 'completed'
                         AND (c.triceratops_fpp IS NULL
                              OR (c.triceratops_fpp >= 0
                                  AND c.triceratops_fpp < 0.5
                                  AND c.triceratops_nfpp IS NULL))""")
            _st = get_shared_state()
            done = _st.get_int(_FPP_BACKFILL_DONE_KEY)
            # started_at is seeded by the first FPP result write-back (increment
            # path), NOT here.  Reading without set_if_absent prevents the GET from
            # anchoring the clock hours before any work has been done, which used to
            # dilute rate_per_min (e.g. 3.7/min reported instead of ~174/min real).
            raw_started_at = _st.get(_FPP_BACKFILL_STARTED_AT_KEY)
            if raw_started_at is None:
                elapsed = 0.0
                rate_per_min = 0.0
            else:
                elapsed = _time.time() - float(raw_started_at)
                # Baseline is "everything we've ever seen" — done + still-pending —
                # so the bar is monotonically non-decreasing and never stuck at 0
                # when the primary pipeline adds new FPP-actionable candidates.
                rate_per_min = (done / elapsed * 60.0) if elapsed > 0 else 0.0
            baseline = done + pending
            return {
                "baseline": baseline,
                "pending": pending,
                "done": done,
                "elapsed_seconds": round(elapsed, 1),
                "rate_per_min": round(rate_per_min, 2),
                "active_workers_assigned": _st.hlen(_FPP_ASSIGNMENTS_KEY),
            }
        # 60 s, not 5 s: measured 4.2 s of PG time per recompute on <host>
        # (2026-07-25) while the dashboard polled it every 10 s. Same story as
        # /api/stats above — the recompute cost, not the endpoint itself, was
        # the load. The FPP queue drains over tens of minutes; a minute of
        # staleness on a progress bar is not a decision-changing delay.
        return _cached_json("backfill_fpp", 60.0, _compute)

    @app.get("/api/backfill/nfpp")
    def api_backfill_nfpp():
        def _compute():
            total = db._query_scalar("SELECT COUNT(*) FROM candidates")
            with_nfpp = db._query_scalar(
                "SELECT COUNT(*) FROM candidates WHERE triceratops_nfpp IS NOT NULL")
            # triceratops_fpp IS NOT NULL is redundant here: in SQL,
            # NULL >= 0 evaluates to false, so the fpp>=0 predicate already
            # excludes NULL rows. Removing the explicit IS NOT NULL lets the
            # SQLite planner match idx_cand_nfpp_pending (same predicate).
            pending = db._query_scalar(
                "SELECT COUNT(*) FROM candidates "
                "WHERE triceratops_fpp >= 0 AND triceratops_fpp < 0.5 "
                "AND triceratops_nfpp IS NULL")
            pending_tics = db._query_scalar(
                "SELECT COUNT(DISTINCT tic_id) FROM candidates "
                "WHERE triceratops_fpp >= 0 AND triceratops_fpp < 0.5 "
                "AND triceratops_nfpp IS NULL")
            rows = db._query(
                "SELECT triceratops_verdict AS v, COUNT(*) AS n FROM candidates "
                "WHERE triceratops_verdict IS NOT NULL GROUP BY triceratops_verdict")
            by_verdict = {r["v"]: r["n"] for r in rows}
            return {
                "total_candidates": total,
                "with_nfpp": with_nfpp,
                "pending_candidates": pending,
                "pending_tics": pending_tics,
                "validated": by_verdict.get("validated", 0),
                "candidate": by_verdict.get("candidate", 0),
                "likely_fp": by_verdict.get("likely_fp", 0),
                "failed": by_verdict.get("failed", 0),
            }
        # 120 s, not 10 s: measured 3.6 s of PG time per recompute. This one
        # feeds a single "N candidates validated so far" footer line — the
        # slowest-moving number on the dashboard.
        return _cached_json("backfill_nfpp", 120.0, _compute)

    @app.get("/api/reports")
    def api_reports(limit: int = Query(50, ge=1, le=200)):
        rdir = Path(reports_dir)
        if not rdir.is_dir():
            return []
        files = sorted(
            [f.name for f in rdir.iterdir() if f.suffix == ".json"],
            reverse=True,
        )
        return files[:limit]

    # --- Setup wizard endpoints ---

    @app.get("/api/setup/status")
    def setup_status():
        """Check if first-run setup is needed and return progress."""
        from exonit.setup_wizard import is_first_run, get_state
        return {
            "first_run": is_first_run(cfg.metadata_cache_path, cfg.catalog_path),
            "setup": get_state(),
            "fits_dir": os.environ.get("EXONIT_FITS_DIR",
                str(_PROJECT_ROOT / "data" / "fits_cache")),
        }

    @app.post("/api/setup/start")
    async def setup_start(request: Request):
        """Start first-run setup: download catalog + metadata + FITS."""
        body = await request.json()
        n_stars = body.get("n_stars", 100)
        fits_dir = body.get("fits_dir",
            os.environ.get("EXONIT_FITS_DIR",
                str(_PROJECT_ROOT / "data" / "fits_cache")))

        # Validate fits_dir
        try:
            os.makedirs(fits_dir, exist_ok=True)
        except Exception as e:
            return JSONResponse({"error": f"Cannot create FITS directory: {e}"},
                                status_code=400)

        import shutil
        disk = shutil.disk_usage(fits_dir)
        free_gb = disk.free / 1e9

        from exonit.setup_wizard import start_setup
        ok = start_setup(cfg.catalog_path, cfg.metadata_cache_path,
                         fits_dir, n_stars)
        if not ok:
            return JSONResponse({"error": "Setup already running"},
                                status_code=409)
        return {
            "status": "started",
            "n_stars": n_stars,
            "fits_dir": fits_dir,
            "disk_free_gb": round(free_gb, 1),
        }

    # --- Pipeline control routes ---

    def _read_pipeline_output(proc):
        """Daemon thread: read pipeline stdout and mirror log+progress to
        shared_state (single writer — only the spawning process runs this
        thread; any uvicorn worker can read)."""
        st = get_shared_state()
        log: list = []
        progress = (_pipeline_shared("progress") or {})
        for raw_line in iter(proc.stdout.readline, b""):
            line = raw_line.decode("utf-8", errors="replace").rstrip()
            log.append(line)
            if len(log) > 500:
                log = log[-500:]

            # Parse progress: [N/M]
            m = re.search(r"\[(\d+)/(\d+)\]", line)
            if m:
                progress["current_star"] = int(m.group(1))
                progress["total_stars"] = int(m.group(2))

            # Parse current TIC: "Analyzing TIC XXXXXXX"
            m = re.search(r"Analyzing TIC[:\s]+(\S+)", line, re.IGNORECASE)
            if m:
                progress["current_tic"] = m.group(1)
                progress["phase"] = "analyzing"
            # Parse completed star: "RESULTS —"
            if "RESULTS" in line and "—" in line:
                progress["stars_completed"] = progress.get("stars_completed", 0) + 1
                progress["phase"] = "results"
            st.hset_json(_PIPELINE_STATE_KEY, "log", {"lines": log})
            st.hset_json(_PIPELINE_STATE_KEY, "progress", progress)

        # Process finished
        proc.wait()
        pipeline_state["process"] = None
        st.hdel(_PIPELINE_STATE_KEY, "current")

    @app.post("/api/start")
    async def api_start(request: Request):
        # Cross-process guard (PATTERN A3): check the shared pid, not just this
        # process's Popen handle — /api/start may land on any uvicorn worker.
        if _pipeline_pid_alive() is not None:
            return JSONResponse({"error": "pipeline already running"}, status_code=409)

        body = await request.json()
        stars = body.get("stars", 5)
        target = body.get("target", None)

        # The `run` pipeline is always distributed now (workers, including the
        # local one, pull tasks). The old no_injection/no_claude/no_hub options
        # were removed 2026-06-09: they were forwarded as CLI flags that
        # run_nightly parsed and silently ignored — the operator believed the
        # stages were disabled when nothing changed.
        cmd = ["python3", "-u", "run_nightly.py", "run", "--stars", str(stars)]
        if target:
            cmd.extend(["--target", str(target)])

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=str(_PROJECT_ROOT),
        )
        # Zombie-leak fix (2026-07-10 review F3): _read_pipeline_output
        # below already calls proc.wait() once its stdout readline loop
        # hits EOF, which reaps this child in the common case -- but that
        # loop skips the wait() if it raises partway through (e.g. a
        # shared_state write failure), leaving the child a <defunct> zombie
        # forever. This is a harmless-if-redundant safety net: same
        # daemon-thread-doing-wait() as the discovery orchestrator fix
        # (2026-07-07); Popen.wait() itself is safe to call concurrently
        # from two threads. See exonit/procutil.py's docstring.
        _reap_in_background(proc)

        # Local Popen handle + cross-process metadata in shared_state.
        pipeline_state["process"] = proc
        st = get_shared_state()
        st.hset_json(_PIPELINE_STATE_KEY, "current",
                     {"pid": proc.pid, "started_at": _time.time()})
        st.hset_json(_PIPELINE_STATE_KEY, "log", {"lines": []})
        st.hset_json(_PIPELINE_STATE_KEY, "progress", {
            "started_at": _time.time(),
            "total_stars": stars,
            "current_star": 0,
            "current_tic": "",
            "phase": "enqueuing",
            "stars_completed": 0,
        })

        t = threading.Thread(target=_read_pipeline_output, args=(proc,), daemon=True)
        t.start()

        return {"status": "started", "stars": stars}

    @app.post("/api/stop")
    def api_stop():
        # Local fast path when this process owns the Popen handle; otherwise
        # cross-process stop via the shared pid (the start may have landed on
        # a sibling uvicorn worker — the old code answered not_running there
        # and left the pipeline alive).
        proc = pipeline_state.get("process")
        if proc is not None and proc.poll() is None:
            proc.terminate()
            pipeline_state["process"] = None
            get_shared_state().hdel(_PIPELINE_STATE_KEY, "current")
            return {"status": "stopped"}
        pid = _pipeline_pid_alive()
        if pid is None:
            return {"status": "not_running"}
        try:
            os.kill(pid, signal.SIGTERM)
        except (OSError, ProcessLookupError):
            pass
        get_shared_state().hdel(_PIPELINE_STATE_KEY, "current")
        return {"status": "stopped"}

    @app.get("/api/status")
    def api_status():
        def _compute():
            dashboard_running = _pipeline_pid_alive() is not None

            # Always read real progress from DB (works for CLI-launched
            # pipelines too). Cached (2026-09-25, fix A): this endpoint's own
            # 3 s outer cache (below) was recomputing the underlying
            # GROUP-BY-status seq scan independently of /api/worker/list's,
            # which has a different TTL — see
            # ExoNitDB.get_task_stats_cached's docstring.
            task_stats = db.get_task_stats_cached()
            db_completed = task_stats.get("completed", 0)
            db_pending = task_stats.get("pending", 0)
            db_assigned = task_stats.get("assigned", 0)
            db_total = db_completed + db_pending + db_assigned

            # Determine if any work is happening (workers active or dashboard pipeline)
            active_workers = [w for w in workers_state.values()
                              if w.get("status") == "working"]
            running = dashboard_running or len(active_workers) > 0 or db_pending > 0

            # Current TIC from active workers
            current_tics = [w.get("current_tic") for w in active_workers
                            if w.get("current_tic")]
            current_tic = current_tics[0] if current_tics else ""

            # Phase
            if db_pending == 0 and db_assigned == 0 and db_total > 0:
                phase = "complete"
            elif len(active_workers) > 0:
                phase = f"processing ({len(active_workers)} workers)"
            elif db_pending > 0:
                phase = "waiting for workers"
            else:
                phase = ""

            # ETA from average processing time in DB
            avg_time = db._query_scalar(
                "SELECT AVG(processing_time) FROM tasks WHERE status='completed' AND processing_time > 0"
            ) or 0
            remaining = db_pending + db_assigned
            eta = int(avg_time * remaining / max(1, len(active_workers))) if active_workers else 0

            # Return camelCase keys to match dashboard JS expectations
            return {
                "running": running,
                "log": (_pipeline_shared("log") or {}).get("lines", [])[-100:],
                "progress": {
                    "totalStars": db_total,
                    "currentStar": db_completed,
                    "currentTIC": current_tic,
                    "phase": phase,
                    "starsCompleted": db_completed,
                    "elapsedSeconds": 0,  # not tracked since A3 (no per-run start time); JS tolerates 0
                    "etaSeconds": eta,
                    "avgSecondsPerStar": round(avg_time, 1),
                },
            }
        return _cached_json("status", 3.0, _compute)

    @app.post("/api/admin/reset_all")
    def api_reset_all():
        """Reset ALL baselines AND worker stats in one call.

        Calls the same _reset_baselines_impl/_reset_worker_stats_impl helpers
        as the two individual endpoints.  Covers all 5 flows:
          primary_baseline, exominer_baseline, lora_baseline,
          fpp_backfill_progress (from reset_baselines)
          + per-worker done/avg counters (from reset_worker_stats).

        Unauthenticated like /api/workers/start: the dashboard (browser) calls
        this with no worker_token and only in-memory baselines/counters are
        cleared (raw DB data is untouched).
        """
        return {
            "ok": True,
            "reset": _reset_baselines_impl(),
            "workers_reset": _reset_worker_stats_impl(),
        }

