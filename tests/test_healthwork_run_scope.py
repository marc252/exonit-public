"""Per-run scoping of /api/health/work's work counters (2026-07-14).

The incident this pins down
---------------------------
A discovery run's phase-exit conditions poll /api/health/work, whose
fpp/exominer/primary counters were GLOBAL COUNT(*)s over the whole
candidates/tasks tables. A run therefore waited on candidates belonging to
OTHER sectors and OTHER lots.

On 2026-07-14 a 50k sector-14 run hung at EXOMINER_BACKFILL on 43 leftover
sector-2 candidates. Two independent defects had to line up, and both are
covered here:

  1. Those candidates were not the run's own work at all (different lot) —
     yet they gated its exit. => scope the counters to the run's lot.

  2. They can NEVER be scored: their FITS metadata is not in the cache, and
     the ExoMiner serve loop deliberately HOLDS BACK such a candidate (no
     serve, no ack — serving it would make the worker fail its LC load and
     persist a permanent -1 over what may be a real score). A counter that
     includes them is an exit condition that can never be met. => wait on
     'actionable' (metadata-cached), not 'pending'.

Fixing only (1) would still hang the moment an un-cached candidate belongs to
the run's own lot, so both are asserted independently.
"""

from __future__ import annotations

import os

from fastapi.testclient import TestClient

from exonit.db import ExoNitDB
from exonit.metadata_cache import MetadataCache


def _build(tmp_path, monkeypatch):
    db_path = str(tmp_path / "scope.db")
    mc_path = str(tmp_path / "metadata_cache.db")
    monkeypatch.setenv("EXONIT_DB_PATH", db_path)
    monkeypatch.setenv("EXONIT_REPORTS_DIR", str(tmp_path / "reports"))
    monkeypatch.setenv("EXONIT_METADATA_CACHE", mc_path)
    os.makedirs(tmp_path / "reports", exist_ok=True)

    from exonit.server import create_app
    client = TestClient(create_app())
    db = ExoNitDB(db_path)
    mc = MetadataCache(mc_path)
    return client, db, mc


def _star(db, tic_id, has_spoc=True):
    db.add_star(tic_id, ra=10.0, dec=-20.0, tmag=11.0, n_sectors=1,
                n_datapoints=1000, time_span_days=27.0, status="completed")
    db.set_star_has_spoc(tic_id, has_spoc)


def _cand(db, tic_id, num, cnn_score=0.85, snr=15.0, exominer_score=None,
          triceratops_fpp=None):
    cid = db.add_candidate(
        tic_id, candidate_num=num, period_days=3.0, t0=0.0,
        duration_days=0.1, depth_ppm=1000, snr=snr, cnn_score=cnn_score,
        triceratops_fpp=triceratops_fpp)
    if exominer_score is not None:
        db.update_candidate_exominer(cid, exominer_score)
    return cid


def _cache(mc, tic_id):
    mc.add_entry(tic_id=tic_id, sectors=[14], exptime=1800.0,
                 fits_files=[f"{tic_id}.fits"])


def _work(client, run_id=None):
    url = "/api/health/work?nocache=1"
    if run_id is not None:
        url += f"&run_id={run_id}"
    resp = client.get(url)
    assert resp.status_code == 200
    return resp.json()


def test_exominer_counter_ignores_other_lots_candidates(tmp_path, monkeypatch):
    """The 2026-07-14 hang: another lot's leftovers must not gate my run."""
    client, db, mc = _build(tmp_path, monkeypatch)

    # MY lot: one candidate, already scored -> my ExoMiner work is done.
    _star(db, "MINE")
    _cand(db, "MINE", 1, exominer_score=0.4)
    _cache(mc, "MINE")

    # AN OLD LOT (the sector-2 leftovers): unscored, metadata never cached.
    _star(db, "LEFTOVER")
    _cand(db, "LEFTOVER", 1, exominer_score=None)

    db.add_discovery_run_tics("run_mine", ["MINE"])

    # Unscoped (UI / Prometheus): still sees the leftover. Unchanged behaviour.
    assert _work(client)["exominer_backfill"]["pending"] == 1

    # Scoped to my run: the leftover is not my work, so my phase is drained.
    scoped = _work(client, run_id="run_mine")
    assert scoped["scope"] == {"run_id": "run_mine", "scoped": True,
                               "kind": "run_tics"}
    assert scoped["exominer_backfill"]["pending"] == 0
    assert scoped["exominer_backfill"]["actionable"] == 0


def test_exominer_actionable_excludes_uncached_metadata(tmp_path, monkeypatch):
    """Defect 2: an un-cached candidate IN MY OWN LOT can never be served.

    It must land in 'unrecoverable', not in the number the run waits on —
    otherwise scoping alone just moves the same forever-hang inside the lot.
    """
    client, db, mc = _build(tmp_path, monkeypatch)

    _star(db, "CACHED")
    _cand(db, "CACHED", 1, exominer_score=None)
    _cache(mc, "CACHED")

    _star(db, "UNCACHED")          # eligible, but no metadata entry -> held back
    _cand(db, "UNCACHED", 1, exominer_score=None)

    db.add_discovery_run_tics("run_x", ["CACHED", "UNCACHED"])

    em = _work(client, run_id="run_x")["exominer_backfill"]
    assert em["pending"] == 2          # both are unscored...
    assert em["actionable"] == 1       # ...but only one can ever be served
    assert em["unrecoverable"] == 1    # and the other is reported, not hidden


def test_unknown_run_id_falls_back_to_global_counters(tmp_path, monkeypatch):
    """A run with no materialised scope must NOT report a spurious 0.

    Reporting "drained" for a scope we cannot prove is empty would let a run
    skip a phase it never actually ran (silent data loss, not a hang).
    """
    client, db, mc = _build(tmp_path, monkeypatch)

    _star(db, "T1")
    _cand(db, "T1", 1, exominer_score=None)
    _cache(mc, "T1")

    data = _work(client, run_id="never_enqueued")
    assert data["scope"] == {"run_id": "never_enqueued", "scoped": False,
                             "kind": None}
    assert data["exominer_backfill"]["pending"] == 1      # global, not 0
    assert data["exominer_backfill"]["actionable"] == 1


def test_fpp_and_primary_counters_are_scoped_too(tmp_path, monkeypatch):
    """The same leak exists on the other two phases; scope them together."""
    client, db, mc = _build(tmp_path, monkeypatch)

    _star(db, "MINE")
    _cache(mc, "MINE")
    # fpp >= 0.5 leaves BOTH arms (the low-fpp rows still owe an nFPP), so my
    # lot genuinely has no FPP work left.
    _cand(db, "MINE", 1, triceratops_fpp=0.9)
    db.add_task("task_mine", "MINE")
    db.complete_task("task_mine", processing_time=1.0)

    _star(db, "OTHER")
    _cache(mc, "OTHER")
    _cand(db, "OTHER", 1, triceratops_fpp=None)    # another lot's FPP work
    db.add_task("task_other", "OTHER")             # and another lot's scan work

    db.add_discovery_run_tics("run_mine", ["MINE"])

    glob = _work(client)
    assert glob["fpp_backfill"]["actionable"] == 1
    assert glob["primary_tasks"]["pending"] == 1

    scoped = _work(client, run_id="run_mine")
    assert scoped["fpp_backfill"]["actionable"] == 0
    assert scoped["primary_tasks"]["pending"] == 0


def test_add_discovery_run_tics_is_idempotent(tmp_path, monkeypatch):
    """--resume re-enters ENQUEUEING; re-materialising must not duplicate."""
    client, db, mc = _build(tmp_path, monkeypatch)

    db.add_discovery_run_tics("run_a", ["1", "2", "3"])
    db.add_discovery_run_tics("run_a", ["1", "2", "3"])
    assert db.count_discovery_run_tics("run_a") == 3

    db.add_discovery_run_tics("run_b", ["3", "4"])   # same TIC, different run
    assert db.count_discovery_run_tics("run_a") == 3
    assert db.count_discovery_run_tics("run_b") == 2
    assert db.count_discovery_run_tics("run_never") == 0

    assert db.add_discovery_run_tics("run_c", []) == 0


# --------------------------------------------------------------------- #
# S3 (docs/superpowers/specs/2026-09-26-enqueue-s1-s4-design.md §3): the
# 'tasks' scope kind (EXONIT_RUN_SCOPE=tasks, tasks.run_id populated at
# enqueue time) — must report the SAME counters as the legacy 'run_tics'
# scope for equivalent data (parity requirement, §3.4/§6 W3).
# --------------------------------------------------------------------- #

def _tag_task_run_id(db, task_id, run_id):
    """Test helper: set tasks.run_id directly (production writes it via
    /api/worker/enqueue_bulk's _insert_tasks_bulk_pg/_sqlite, exercised in
    tests/test_enqueue.py — this is the DB-level equivalent for a test that
    only cares about the health/work scoping, not the enqueue path)."""
    db._exec("UPDATE tasks SET run_id = ? WHERE task_id = ?", (run_id, task_id))


def test_run_scope_kind_is_tasks_when_env_set_and_column_populated(
        tmp_path, monkeypatch):
    """EXONIT_RUN_SCOPE=tasks + a tagged task -> db.run_scope_kind == 'tasks',
    and /api/health/work's scope block reports it."""
    client, db, mc = _build(tmp_path, monkeypatch)
    monkeypatch.setenv("EXONIT_RUN_SCOPE", "tasks")

    _star(db, "MINE")
    _cache(mc, "MINE")
    _cand(db, "MINE", 1, triceratops_fpp=None)
    db.add_task("task_mine", "MINE")
    _tag_task_run_id(db, "task_mine", "run_tasks_scope")

    scoped = _work(client, run_id="run_tasks_scope")
    assert scoped["scope"] == {"run_id": "run_tasks_scope", "scoped": True,
                               "kind": "tasks"}
    assert scoped["primary_tasks"]["pending"] == 1
    assert db.run_scope_kind("run_tasks_scope") == "tasks"


def test_tasks_scope_and_run_tics_scope_agree_on_the_same_lot(
        tmp_path, monkeypatch):
    """Parity (§3.4): tag the SAME TICs both ways (tasks.run_id AND
    discovery_run_tics) and confirm every counter this endpoint reports is
    identical regardless of which scope kind wins."""
    client, db, mc = _build(tmp_path, monkeypatch)

    _star(db, "MINE")
    _cache(mc, "MINE")
    _cand(db, "MINE", 1, exominer_score=None, triceratops_fpp=0.9)
    db.add_task("task_mine", "MINE")
    _tag_task_run_id(db, "task_mine", "run_parity")

    _star(db, "OTHER")
    _cache(mc, "OTHER")
    _cand(db, "OTHER", 1, exominer_score=None)
    db.add_task("task_other", "OTHER")

    db.add_discovery_run_tics("run_parity", ["MINE"])

    monkeypatch.setenv("EXONIT_RUN_SCOPE", "run_tics")
    via_run_tics = _work(client, run_id="run_parity")
    assert via_run_tics["scope"]["kind"] == "run_tics"

    monkeypatch.setenv("EXONIT_RUN_SCOPE", "tasks")
    via_tasks = _work(client, run_id="run_parity")
    assert via_tasks["scope"]["kind"] == "tasks"

    for key in ("primary_tasks", "fpp_backfill", "exominer_backfill"):
        assert via_run_tics[key] == via_tasks[key], key


def test_run_scope_kind_falls_back_to_run_tics_when_tasks_scope_empty(
        tmp_path, monkeypatch):
    """EXONIT_RUN_SCOPE=tasks but no task carries this run_id (an older run,
    or a chunk the enqueue-time dedup skipped) -> falls back to
    discovery_run_tics rather than reporting unscoped/global."""
    client, db, mc = _build(tmp_path, monkeypatch)
    monkeypatch.setenv("EXONIT_RUN_SCOPE", "tasks")

    db.add_discovery_run_tics("run_fallback", ["1", "2"])
    assert db.run_scope_kind("run_fallback") == "run_tics"


def test_run_scope_kind_none_when_neither_mechanism_has_this_run(
        tmp_path, monkeypatch):
    client, db, mc = _build(tmp_path, monkeypatch)
    assert db.run_scope_kind("nobody_enqueued_this") is None
