"""Tests for Fase 3 Task 3: dispatch table + --resume + checkpoint-skip.

Constraints (matching tests/test_discovery_orchestrator.py's header):
- NEVER real SSH, farmctl, server poll, or subprocess network/GPU calls.
- NEVER touch the live DB data/exonit.db or production PostgreSQL.
- All external I/O is mocked/monkeypatched; --resume tests use a REAL
  (tmp_path) sqlite ExoNitDB so get_discovery_run/checkpoints round-trip
  exactly as they will in production — same pattern as
  test_run_manual_cli_creates_discovery_runs_row_end_to_end in
  tests/test_discovery_orchestrator.py.
"""

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import discovery_orchestrator as orch  # noqa: E402
from exonit.db import ExoNitDB  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_orchestrator_db_globals():
    """Isolate module-level PG dual-write state between tests (Fase 3 Task 2)."""
    orch._db_path = None
    orch._pg_dual_write_enabled = False
    orch._db_instance = None
    yield
    orch._db_path = None
    orch._pg_dual_write_enabled = False
    orch._db_instance = None


def _lot_csv(tmp_path, name="s0014_lot20260704T220000.csv"):
    p = tmp_path / name
    p.write_text("tic_id,label\n12345,candidate\n")
    return str(p)


def _all_steps_ok_patches():
    """Patch every step_* the loop may call, all returning success — the
    fastest way to drive run() to DONE without any real I/O."""
    return [
        patch.object(orch, "step_scanning", return_value=(True, "")),
        patch.object(orch, "step_fpp_backfill", return_value=(True, "")),
        patch.object(orch, "step_exominer_backfill", return_value=(True, "")),
        patch.object(orch, "step_stop_workers", return_value=(True, "")),
        patch.object(orch, "step_postprocess", return_value=(True, "")),
        patch.object(orch, "step_stop_nvidiarack", return_value=(True, "")),
        patch.object(orch, "step_lora", return_value=(True, "")),
        patch.object(orch, "step_finalize", return_value=(True, "")),
        patch.object(orch, "step_wal_checkpoint", return_value=None),
        patch("urllib.request.urlopen", side_effect=Exception("no server")),
    ]


def _apply(patches):
    for p in patches:
        p.start()
    return patches


def _stop(patches):
    for p in patches[::-1]:
        p.stop()


# ---------------------------------------------------------------------------
# (a) fresh run: full HOME_SEQUENCE order parity with the pre-existing test
# ---------------------------------------------------------------------------

def test_fresh_run_full_sequence_order(tmp_path):
    lot_csv = _lot_csv(tmp_path)
    sf = str(tmp_path / "pipeline_state.json")

    seen_states = []
    original_set_state = orch._set_state

    def capturing_set_state(path, state_name, **extra):
        seen_states.append(state_name)
        return original_set_state(path, state_name, **extra)

    args = orch._parse_args([
        "--lot-csv", lot_csv,
        "--sector", "1",
        "--state-file", sf,
        "--poll-interval", "0",
        "--dry-run",
    ])

    with patch.object(orch, "_set_state", side_effect=capturing_set_state):
        orch.run(args)

    expected_order = ["SCANNING", "FPP_BACKFILL", "EXOMINER_BACKFILL",
                      "STOP_WORKERS", "POSTPROCESS", "STOP_NVIDIARACK", "LORA",
                      "FREE_GPU", "FINALIZE", "DONE"]
    assert seen_states == expected_order

    state = orch._load_state(sf)
    assert state["state"] == "DONE"


# ---------------------------------------------------------------------------
# (b) --resume with state=POSTPROCESS starts there (no re-scan)
# ---------------------------------------------------------------------------

def test_resume_at_postprocess_skips_scan_phases(tmp_path):
    db_path = str(tmp_path / "exonit.db")
    lot_csv = _lot_csv(tmp_path)
    sf = str(tmp_path / "pipeline_state.json")

    run_id = "discovery_s0014_lot20260704T220000_2026-07-04"
    db = ExoNitDB(db_path)
    db.create_discovery_run(run_id, sector=14, lot_csv=lot_csv,
                            dry_run=False, chained=False,
                            server_url="http://localhost:8400")
    db.update_discovery_run(run_id, state="POSTPROCESS")
    db.close()

    args = orch._parse_args([
        "--resume", run_id,
        "--state-file", sf,
        "--db", db_path,
        "--poll-interval", "0",
    ])

    patches = _apply(_all_steps_ok_patches())
    try:
        orch.run(args)
    finally:
        _stop(patches)

    state = orch._load_state(sf)
    assert state["state"] == "DONE"
    assert state["run_id"] == run_id


def test_resume_at_postprocess_does_not_call_scan_steps(tmp_path):
    db_path = str(tmp_path / "exonit.db")
    lot_csv = _lot_csv(tmp_path)
    sf = str(tmp_path / "pipeline_state.json")

    run_id = "discovery_s0014_lot20260704T221111_2026-07-04"
    db = ExoNitDB(db_path)
    db.create_discovery_run(run_id, sector=14, lot_csv=lot_csv,
                            dry_run=False, chained=False,
                            server_url="http://localhost:8400")
    db.update_discovery_run(run_id, state="POSTPROCESS")
    db.close()

    args = orch._parse_args([
        "--resume", run_id,
        "--state-file", sf,
        "--db", db_path,
        "--poll-interval", "0",
    ])

    with patch.object(orch, "step_scanning") as mock_scan, \
         patch.object(orch, "step_fpp_backfill") as mock_fpp, \
         patch.object(orch, "step_exominer_backfill") as mock_em, \
         patch.object(orch, "step_stop_workers", return_value=(True, "")), \
         patch.object(orch, "step_postprocess", return_value=(True, "")) as mock_pp, \
         patch.object(orch, "step_stop_nvidiarack", return_value=(True, "")), \
         patch.object(orch, "step_lora", return_value=(True, "")), \
         patch.object(orch, "step_finalize", return_value=(True, "")), \
         patch.object(orch, "step_wal_checkpoint", return_value=None):
        orch.run(args)

    mock_scan.assert_not_called()
    mock_fpp.assert_not_called()
    mock_em.assert_not_called()
    mock_pp.assert_called_once()

    state = orch._load_state(sf)
    assert state["state"] == "DONE"


# ---------------------------------------------------------------------------
# (c) --resume with checkpoints.postprocess.done=true skips to STOP_NVIDIARACK
# ---------------------------------------------------------------------------

def test_resume_skips_postprocess_when_checkpoint_done(tmp_path):
    db_path = str(tmp_path / "exonit.db")
    lot_csv = _lot_csv(tmp_path)
    sf = str(tmp_path / "pipeline_state.json")

    run_id = "discovery_s0014_lot20260704T222222_2026-07-04"
    db = ExoNitDB(db_path)
    db.create_discovery_run(run_id, sector=14, lot_csv=lot_csv,
                            dry_run=False, chained=False,
                            server_url="http://localhost:8400")
    db.update_discovery_run(run_id, state="POSTPROCESS")
    db.set_discovery_run_checkpoint(
        run_id, "postprocess",
        {"done": True, "scorecard_json": "data/scorecards/x.json",
         "dossier": "data/scorecards/x_dossier.md"})
    db.close()

    args = orch._parse_args([
        "--resume", run_id,
        "--state-file", sf,
        "--db", db_path,
        "--poll-interval", "0",
    ])

    with patch.object(orch, "step_postprocess") as mock_pp, \
         patch.object(orch, "step_stop_nvidiarack", return_value=(True, "")) as mock_stop_nv, \
         patch.object(orch, "step_lora", return_value=(True, "")), \
         patch.object(orch, "step_finalize", return_value=(True, "")), \
         patch.object(orch, "step_wal_checkpoint", return_value=None):
        orch.run(args)

    mock_pp.assert_not_called()
    mock_stop_nv.assert_called_once()

    state = orch._load_state(sf)
    assert state["state"] == "DONE"


# ---------------------------------------------------------------------------
# (d) --resume of a DONE run errors, no steps executed
# ---------------------------------------------------------------------------

def test_resume_of_done_run_errors_and_executes_nothing(tmp_path):
    db_path = str(tmp_path / "exonit.db")
    lot_csv = _lot_csv(tmp_path)
    sf = str(tmp_path / "pipeline_state.json")

    run_id = "discovery_s0014_lot20260704T223333_2026-07-04"
    db = ExoNitDB(db_path)
    db.create_discovery_run(run_id, sector=14, lot_csv=lot_csv,
                            dry_run=False, chained=False,
                            server_url="http://localhost:8400")
    db.update_discovery_run(run_id, state="DONE", finished=True)
    db.close()

    args = orch._parse_args([
        "--resume", run_id,
        "--state-file", sf,
        "--db", db_path,
        "--poll-interval", "0",
    ])

    with patch.object(orch, "step_scanning") as mock_scan, \
         patch.object(orch, "step_postprocess") as mock_pp:
        with pytest.raises(SystemExit):
            orch.run(args)

    mock_scan.assert_not_called()
    mock_pp.assert_not_called()
    # No state file must have been written by the rejected resume.
    assert not Path(sf).exists()


def test_resume_of_missing_run_id_errors(tmp_path):
    db_path = str(tmp_path / "exonit.db")
    sf = str(tmp_path / "pipeline_state.json")
    # Create the db file but no row for this run_id.
    ExoNitDB(db_path).close()

    args = orch._parse_args([
        "--resume", "discovery_s9999_lotX_2026-01-01",
        "--state-file", sf,
        "--db", db_path,
        "--poll-interval", "0",
    ])

    with pytest.raises(SystemExit):
        orch.run(args)


def test_resume_of_failed_run_without_force_errors(tmp_path):
    db_path = str(tmp_path / "exonit.db")
    lot_csv = _lot_csv(tmp_path)
    sf = str(tmp_path / "pipeline_state.json")

    run_id = "discovery_s0014_lot20260704T224444_2026-07-04"
    db = ExoNitDB(db_path)
    db.create_discovery_run(run_id, sector=14, lot_csv=lot_csv,
                            dry_run=False, chained=False,
                            server_url="http://localhost:8400")
    db.update_discovery_run(run_id, state="FAILED", detail="lora rc=3: boom",
                            finished=True)
    db.close()

    args = orch._parse_args([
        "--resume", run_id,
        "--state-file", sf,
        "--db", db_path,
        "--poll-interval", "0",
    ])

    with patch.object(orch, "step_lora") as mock_lora:
        with pytest.raises(SystemExit):
            orch.run(args)

    mock_lora.assert_not_called()


# ---------------------------------------------------------------------------
# --resume --force on a FAILED run: resumes at the recorded failed_state and
# clears the stale PG detail on the first transition.
# ---------------------------------------------------------------------------

def test_resume_force_on_failed_run_resumes_at_failed_state_and_clears_detail(tmp_path):
    db_path = str(tmp_path / "exonit.db")
    lot_csv = _lot_csv(tmp_path)
    sf = str(tmp_path / "pipeline_state.json")

    run_id = "discovery_s0014_lot20260704T225555_2026-07-04"
    db = ExoNitDB(db_path)
    db.create_discovery_run(run_id, sector=14, lot_csv=lot_csv,
                            dry_run=False, chained=False,
                            server_url="http://localhost:8400")
    # Simulate: SCANNING/FPP/EXOMINER/STOP_WORKERS/POSTPROCESS all completed,
    # then LORA failed.
    db.set_discovery_run_checkpoint(
        run_id, "postprocess",
        {"done": True, "scorecard_json": "data/scorecards/x.json",
         "dossier": "data/scorecards/x_dossier.md"})
    db.update_discovery_run(run_id, state="FAILED",
                            detail="lora rc=3: ENGINE FAILED",
                            finished=True)
    db.set_discovery_run_checkpoint(run_id, "_failed_state", "LORA")
    db.close()

    args = orch._parse_args([
        "--resume", run_id,
        "--force",
        "--state-file", sf,
        "--db", db_path,
        "--poll-interval", "0",
    ])

    with patch.object(orch, "step_scanning") as mock_scan, \
         patch.object(orch, "step_postprocess") as mock_pp, \
         patch.object(orch, "step_stop_nvidiarack") as mock_stop_nv, \
         patch.object(orch, "step_lora", return_value=(True, "")) as mock_lora, \
         patch.object(orch, "step_finalize", return_value=(True, "")), \
         patch.object(orch, "step_wal_checkpoint", return_value=None):
        orch.run(args)

    mock_scan.assert_not_called()
    mock_pp.assert_not_called()
    mock_stop_nv.assert_not_called()
    mock_lora.assert_called_once()

    state = orch._load_state(sf)
    assert state["state"] == "DONE"

    db = ExoNitDB(db_path)
    row = db.get_discovery_run(run_id)
    db.close()
    assert row["detail"] in ("", None) or "ENGINE FAILED" not in (row["detail"] or "")
    # Task-3 review item 2: the _failed_state breadcrumb must be cleared once
    # the forced resume completes successfully (reaches DONE), so a later
    # inspection or a second failure/resume cycle never sees a stale value.
    assert not row["checkpoints"].get("_failed_state"), (
        f"_failed_state must be cleared after a successful forced resume, "
        f"got: {row['checkpoints'].get('_failed_state')!r}"
    )


# ---------------------------------------------------------------------------
# Task-3 review item 1: unrecognized persisted state is a HARD ERROR
# (likely row corruption / schema drift) — never a silent SCANNING fallback.
# ---------------------------------------------------------------------------

def test_resume_with_unknown_state_hard_errors_and_executes_nothing(tmp_path):
    db_path = str(tmp_path / "exonit.db")
    lot_csv = _lot_csv(tmp_path)
    sf = str(tmp_path / "pipeline_state.json")

    run_id = "discovery_s0014_lot20260705T000001_2026-07-05"
    db = ExoNitDB(db_path)
    db.create_discovery_run(run_id, sector=14, lot_csv=lot_csv,
                            dry_run=False, chained=False,
                            server_url="http://localhost:8400")
    db.update_discovery_run(run_id, state="BOGUS_STATE")
    db.close()

    args = orch._parse_args([
        "--resume", run_id,
        "--state-file", sf,
        "--db", db_path,
        "--poll-interval", "0",
    ])

    with patch.object(orch, "step_scanning") as mock_scan, \
         patch.object(orch, "step_postprocess") as mock_pp, \
         patch.object(orch, "step_lora") as mock_lora:
        with pytest.raises(SystemExit) as excinfo:
            orch.run(args)

    mock_scan.assert_not_called()
    mock_pp.assert_not_called()
    mock_lora.assert_not_called()
    # The message must name the bogus state and point at a manual recourse.
    msg = str(excinfo.value)
    assert "BOGUS_STATE" in msg
    assert "--lot-csv" in msg


def test_resume_at_enqueueing_starts_at_scanning(tmp_path):
    """ENQUEUEING (the discovery_runs row default) is a known-legitimate
    pre-loop state, NOT corruption: the orchestrator died before its first
    SCANNING transition, so nothing has run yet — resume starts at SCANNING."""
    db_path = str(tmp_path / "exonit.db")
    lot_csv = _lot_csv(tmp_path)
    sf = str(tmp_path / "pipeline_state.json")

    run_id = "discovery_s0014_lot20260705T000002_2026-07-05"
    db = ExoNitDB(db_path)
    db.create_discovery_run(run_id, sector=14, lot_csv=lot_csv,
                            dry_run=False, chained=False,
                            server_url="http://localhost:8400")
    # Row stays at its default state: ENQUEUEING.
    db.close()

    args = orch._parse_args([
        "--resume", run_id,
        "--state-file", sf,
        "--db", db_path,
        "--poll-interval", "0",
    ])

    with patch.object(orch, "step_scanning", return_value=(True, "")) as mock_scan, \
         patch.object(orch, "step_fpp_backfill", return_value=(True, "")), \
         patch.object(orch, "step_exominer_backfill", return_value=(True, "")), \
         patch.object(orch, "step_stop_workers", return_value=(True, "")), \
         patch.object(orch, "step_postprocess", return_value=(True, "")), \
         patch.object(orch, "step_stop_nvidiarack", return_value=(True, "")), \
         patch.object(orch, "step_lora", return_value=(True, "")), \
         patch.object(orch, "step_finalize", return_value=(True, "")), \
         patch.object(orch, "step_wal_checkpoint", return_value=None), \
         patch("urllib.request.urlopen", side_effect=Exception("no server")):
        orch.run(args)

    mock_scan.assert_called_once()
    state = orch._load_state(sf)
    assert state["state"] == "DONE"
    assert state["run_id"] == run_id


# ---------------------------------------------------------------------------
# Task-3 review item 3: --dry-run + --resume is a hard error (resume needs a
# live PG row; --dry-run closes the DB gate).
# ---------------------------------------------------------------------------

def test_resume_with_dry_run_hard_errors(tmp_path):
    db_path = str(tmp_path / "exonit.db")
    lot_csv = _lot_csv(tmp_path)
    sf = str(tmp_path / "pipeline_state.json")

    run_id = "discovery_s0014_lot20260705T000003_2026-07-05"
    db = ExoNitDB(db_path)
    db.create_discovery_run(run_id, sector=14, lot_csv=lot_csv,
                            dry_run=False, chained=False,
                            server_url="http://localhost:8400")
    db.update_discovery_run(run_id, state="POSTPROCESS")
    db.close()

    args = orch._parse_args([
        "--resume", run_id,
        "--state-file", sf,
        "--db", db_path,
        "--poll-interval", "0",
        "--dry-run",
    ])

    with patch.object(orch, "step_scanning") as mock_scan, \
         patch.object(orch, "step_postprocess") as mock_pp:
        with pytest.raises(SystemExit) as excinfo:
            orch.run(args)

    mock_scan.assert_not_called()
    mock_pp.assert_not_called()
    assert "dry-run" in str(excinfo.value)


# ---------------------------------------------------------------------------
# F2 (final-review fix): resume rehydrates the mirror's identity from the PG
# row — a stale mirror left by a DIFFERENT run must not leak into run A's
# resumed state (this is what made _active_run_sector() mislabel new
# candidates with the WRONG sector during a SCANNING resume).
# ---------------------------------------------------------------------------

def test_resume_rehydrates_mirror_identity_over_stale_other_run(tmp_path):
    db_path = str(tmp_path / "exonit.db")
    lot_csv_a = _lot_csv(tmp_path, "s0014_lotA.csv")
    sf = str(tmp_path / "pipeline_state.json")

    run_id_a = "discovery_s0014_lotA_2026-07-05"
    db = ExoNitDB(db_path)
    db.create_discovery_run(run_id_a, sector=14, lot_csv=lot_csv_a,
                            dry_run=False, chained=False,
                            server_url="http://localhost:8400")
    db.update_discovery_run(run_id_a, state="POSTPROCESS")
    db.close()

    # Seed the mirror with a DIFFERENT run's identity (run B) — simulating a
    # chained/autofill lot that wrote pipeline_state.json while run A's
    # orchestrator was dead.
    Path(sf).write_text(json.dumps({
        "state": "SCANNING",
        "run_id": "discovery_s0099_lotB_2026-07-04",
        "sector": 99,
        "lot_csv": "s0099_lotB.csv",
        "server_url": "http://localhost:9999",
        "updated_at": "2026-07-04T00:00:00+00:00",
    }))

    args = orch._parse_args([
        "--resume", run_id_a,
        "--state-file", sf,
        "--db", db_path,
        "--poll-interval", "0",
    ])

    patches = _apply(_all_steps_ok_patches())
    try:
        orch.run(args)
    finally:
        _stop(patches)

    state = orch._load_state(sf)
    assert state["state"] == "DONE"
    assert state["run_id"] == run_id_a
    assert state["sector"] == 14
    assert state["lot_csv"] == lot_csv_a
    assert state["server_url"] == "http://localhost:8400"


# ---------------------------------------------------------------------------
# (e) resume never re-derives run_id
# ---------------------------------------------------------------------------

def test_resume_never_calls_derive_run_id(tmp_path):
    db_path = str(tmp_path / "exonit.db")
    lot_csv = _lot_csv(tmp_path)
    sf = str(tmp_path / "pipeline_state.json")

    run_id = "discovery_s0014_lot20260704T226666_2026-07-04"
    db = ExoNitDB(db_path)
    db.create_discovery_run(run_id, sector=14, lot_csv=lot_csv,
                            dry_run=False, chained=False,
                            server_url="http://localhost:8400")
    db.update_discovery_run(run_id, state="STOP_NVIDIARACK")
    db.close()

    args = orch._parse_args([
        "--resume", run_id,
        "--state-file", sf,
        "--db", db_path,
        "--poll-interval", "0",
    ])

    patches = _apply(_all_steps_ok_patches())
    try:
        with patch.object(orch, "_derive_run_id") as mock_derive:
            orch.run(args)
        mock_derive.assert_not_called()
    finally:
        _stop(patches)

    assert args.run_id == run_id
    state = orch._load_state(sf)
    assert state["run_id"] == run_id


# ---------------------------------------------------------------------------
# (f) AWS branch intact under the new dispatch loop
# ---------------------------------------------------------------------------

def test_aws_branch_intact_under_dispatch_loop(tmp_path, monkeypatch):
    monkeypatch.setenv("EXONIT_DEPLOYMENT", "aws")

    lot_csv = _lot_csv(tmp_path, "s0002_lot20260609T120000.csv")
    sf = str(tmp_path / "pipeline_state.json")

    seen_states = []
    original_set_state = orch._set_state

    def capturing_set_state(path, state_name, **extra):
        seen_states.append(state_name)
        return original_set_state(path, state_name, **extra)

    args = orch._parse_args([
        "--lot-csv", lot_csv,
        "--sector", "2",
        "--state-file", sf,
        "--db", str(tmp_path / "exonit.db"),
        "--poll-interval", "0",
        "--dry-run",
    ])

    with patch.object(orch, "_set_state", side_effect=capturing_set_state), \
         patch.object(orch, "step_stop_nvidiarack") as mock_stop_nv, \
         patch.object(orch, "step_lora") as mock_lora, \
         patch.object(orch, "step_aws_finalize",
                      return_value=(True, "dry_run: export done")) as mock_aws_fin:
        orch.run(args)

    assert "AWS_FINALIZE" in seen_states
    assert "DONE" in seen_states
    assert "STOP_NVIDIARACK" not in seen_states
    assert "LORA" not in seen_states
    mock_stop_nv.assert_not_called()
    mock_lora.assert_not_called()
    mock_aws_fin.assert_called_once()


def test_resume_at_aws_finalize_reexecutes_it(tmp_path, monkeypatch):
    monkeypatch.setenv("EXONIT_DEPLOYMENT", "aws")

    db_path = str(tmp_path / "exonit.db")
    lot_csv = _lot_csv(tmp_path, "s0002_lot20260609T130000.csv")
    sf = str(tmp_path / "pipeline_state.json")

    run_id = "discovery_s0002_lot20260609T130000_2026-06-09"
    db = ExoNitDB(db_path)
    db.create_discovery_run(run_id, sector=2, lot_csv=lot_csv,
                            dry_run=False, chained=False,
                            server_url="http://localhost:8400")
    db.update_discovery_run(run_id, state="AWS_FINALIZE")
    db.close()

    args = orch._parse_args([
        "--resume", run_id,
        "--state-file", sf,
        "--db", db_path,
        "--poll-interval", "0",
    ])

    with patch.object(orch, "step_scanning") as mock_scan, \
         patch.object(orch, "step_postprocess") as mock_pp, \
         patch.object(orch, "step_aws_finalize",
                      return_value=(True, "export ok")) as mock_aws_fin:
        orch.run(args)

    mock_scan.assert_not_called()
    mock_pp.assert_not_called()
    mock_aws_fin.assert_called_once()

    state = orch._load_state(sf)
    assert state["state"] == "DONE"
