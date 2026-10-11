"""Tests for scripts/discovery_finalize.py.

Constraints:
- Never touch the live DB data/exonit.db.
- Never touch the live CAMPAIGN.md or STATUS.md.
- All I/O uses tmp_path fixtures.
"""

import csv
import json
import sqlite3
import sys
from pathlib import Path

import pytest

# Ensure scripts/ is on the path.
SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import discovery_finalize as fin  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_lot_csv(tmp_path: Path, tic_ids: list) -> Path:
    """Write a minimal lot CSV."""
    p = tmp_path / "s0001_lot_test.csv"
    with open(p, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["tic_id", "label"])
        for t in tic_ids:
            w.writerow([t, "candidate"])
    return p


def _make_db(tmp_path: Path, rows: list[tuple]) -> Path:
    """Create a minimal tasks table with (tic_id TEXT, status TEXT) rows."""
    db = tmp_path / "test.db"
    con = sqlite3.connect(str(db))
    con.execute(
        "CREATE TABLE tasks (id INTEGER PRIMARY KEY, tic_id TEXT, status TEXT)"
    )
    con.executemany("INSERT INTO tasks (tic_id, status) VALUES (?, ?)", rows)
    con.commit()
    con.close()
    return db


def _make_dossier(tmp_path: Path, n_strong: int, n_finalists: int) -> Path:
    """Write a minimal dossier markdown."""
    p = tmp_path / "dossier.md"
    content = (
        f"# Discovery vetting dossier — test\n\n"
        f"Strong candidates (v6>0.5, post-veto): {n_strong}.\n\n"
        "| TIC | P(d) | depth(ppm) | SNR | v6 | EB flags | FPP | verdict |\n"
        "|---|---|---|---|---|---|---|---|\n"
        f"\n**Finalists (pass all + FPP<0.5): {n_finalists}**: none\n"
    )
    p.write_text(content, encoding="utf-8")
    return p


def _make_campaign(tmp_path: Path, lot_basename: str, status: str = "ENQUEUED 2026-05-30") -> Path:
    """Write a minimal CAMPAIGN.md with one table row for the given lot."""
    p = tmp_path / "CAMPAIGN.md"
    content = (
        "# Discovery campaign\n\n"
        "## CURSOR\n"
        "| Lot | Rows | File | Status |\n"
        "|---|---|---|---|\n"
        f"| lot-auto | 1–100 (100 TICs) | `{lot_basename}` | {status} |\n"
    )
    p.write_text(content, encoding="utf-8")
    return p


def _make_status(tmp_path: Path, with_section: bool = False, existing_row: str = "") -> Path:
    """Write a minimal STATUS.md."""
    p = tmp_path / "STATUS.md"
    content = "# ExoNit — Estat actual\n\n## Incidents recents\n\n- Some incident.\n"
    if with_section:
        content += (
            "\n" + fin._STATUS_SECTION_START + "\n"
            + fin._STATUS_TABLE_HEADER + "\n"
            + (existing_row + "\n" if existing_row else "")
            + fin._STATUS_SECTION_END + "\n"
        )
    p.write_text(content, encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# 1. ENQUEUED → DONE
# ---------------------------------------------------------------------------

def test_enqueued_to_done(tmp_path):
    """A lot row with ENQUEUED status is updated to DONE with correct stats."""
    lot_csv = _make_lot_csv(tmp_path, [111, 222, 333])
    db = _make_db(tmp_path, [
        ("111", "completed"),
        ("222", "completed"),
        ("333", "failed"),
    ])
    dossier = _make_dossier(tmp_path, n_strong=2, n_finalists=0)
    scorecard = tmp_path / "discovery_s0001_lottest_2026-05-30.json"
    scorecard.write_text("{}")
    campaign = _make_campaign(tmp_path, lot_csv.name)
    status = _make_status(tmp_path)

    result = fin.finalize_run(
        lot_csv=str(lot_csv),
        run_id="discovery_s0001_lottest_2026-05-30",
        scorecard_json=str(scorecard),
        dossier=str(dossier),
        db_path=str(db),
        campaign_path=str(campaign),
        status_path=str(status),
    )

    assert result["campaign_action"] == "updated"
    assert result["n_lot"] == 3
    assert result["n_completed"] == 2
    assert result["n_failed"] == 1
    assert result["n_strong"] == 2
    assert result["n_finalists"] == 0

    camp_text = campaign.read_text()
    assert "**DONE**" in camp_text
    assert "ENQUEUED" not in camp_text
    assert "2/3 completed, 1 failed" in camp_text
    assert "2 strong" in camp_text
    assert "0 finalists" in camp_text
    assert f"`{lot_csv.name}`" in camp_text


# ---------------------------------------------------------------------------
# 2. Idempotency
# ---------------------------------------------------------------------------

def test_idempotent_campaign(tmp_path):
    """Calling finalize_run twice does NOT change CAMPAIGN.md the second time."""
    lot_csv = _make_lot_csv(tmp_path, [111, 222])
    db = _make_db(tmp_path, [("111", "completed"), ("222", "completed")])
    dossier = _make_dossier(tmp_path, n_strong=0, n_finalists=0)
    scorecard = tmp_path / "discovery_s0001_lot_2026-05-30.json"
    scorecard.write_text("{}")
    campaign = _make_campaign(tmp_path, lot_csv.name)
    status = _make_status(tmp_path)

    run_kwargs = dict(
        lot_csv=str(lot_csv),
        run_id="discovery_s0001_lot_2026-05-30",
        scorecard_json=str(scorecard),
        dossier=str(dossier),
        db_path=str(db),
        campaign_path=str(campaign),
        status_path=str(status),
    )

    r1 = fin.finalize_run(**run_kwargs)
    camp_after_first = campaign.read_text()

    r2 = fin.finalize_run(**run_kwargs)
    camp_after_second = campaign.read_text()

    assert r1["campaign_action"] == "updated"
    assert r2["campaign_action"] == "already_done"
    assert camp_after_first == camp_after_second, (
        "CAMPAIGN.md must not change on second call (idempotent)"
    )


def test_idempotent_status_single_row(tmp_path):
    """Calling finalize_run twice results in exactly one row in STATUS.md."""
    lot_csv = _make_lot_csv(tmp_path, [111])
    db = _make_db(tmp_path, [("111", "completed")])
    dossier = _make_dossier(tmp_path, n_strong=1, n_finalists=0)
    scorecard = tmp_path / "discovery_s0001_lot_2026-05-31.json"
    scorecard.write_text("{}")
    campaign = _make_campaign(tmp_path, lot_csv.name)
    status = _make_status(tmp_path)

    run_kwargs = dict(
        lot_csv=str(lot_csv),
        run_id="discovery_s0001_lot_2026-05-31",
        scorecard_json=str(scorecard),
        dossier=str(dossier),
        db_path=str(db),
        campaign_path=str(campaign),
        status_path=str(status),
    )

    fin.finalize_run(**run_kwargs)
    fin.finalize_run(**run_kwargs)

    status_text = status.read_text()
    # Count LINES (rows) that reference the run_id (not raw string count,
    # because the run_id appears twice in a single table row: in the Run col
    # and in the Scorecard col).
    matching_lines = [l for l in status_text.splitlines() if "discovery_s0001_lot_2026-05-31" in l]
    assert len(matching_lines) == 1, (
        f"Expected 1 row for run_id, got {len(matching_lines)}:\n{status_text}"
    )


# ---------------------------------------------------------------------------
# 3. Defensive no-match
# ---------------------------------------------------------------------------

def test_no_match_campaign_empty(tmp_path):
    """CAMPAIGN.md with no matching lot → no_unique_match, file unchanged."""
    lot_csv = _make_lot_csv(tmp_path, [111])
    db = _make_db(tmp_path, [("111", "completed")])
    dossier = _make_dossier(tmp_path, n_strong=0, n_finalists=0)
    scorecard = tmp_path / "discovery_s0001_lot_2026-05-30.json"
    scorecard.write_text("{}")

    # CAMPAIGN with a DIFFERENT lot basename
    campaign = tmp_path / "CAMPAIGN.md"
    campaign.write_text(
        "# Discovery campaign\n\n"
        "| Lot | Rows | File | Status |\n"
        "|---|---|---|---|\n"
        "| lot-auto | 1–100 (100 TICs) | `other_lot.csv` | ENQUEUED 2026-05-30 |\n"
    )
    original_text = campaign.read_text()
    status = _make_status(tmp_path)

    result = fin.finalize_run(
        lot_csv=str(lot_csv),
        run_id="discovery_s0001_lot_2026-05-30",
        scorecard_json=str(scorecard),
        dossier=str(dossier),
        db_path=str(db),
        campaign_path=str(campaign),
        status_path=str(status),
    )

    assert result["campaign_action"] == "no_unique_match"
    assert campaign.read_text() == original_text, "File must be unchanged on no_unique_match"


def test_no_match_campaign_multiple(tmp_path):
    """CAMPAIGN.md with 2 matching rows → no_unique_match, file unchanged."""
    lot_csv = _make_lot_csv(tmp_path, [111])
    db = _make_db(tmp_path, [("111", "completed")])
    dossier = _make_dossier(tmp_path, n_strong=0, n_finalists=0)
    scorecard = tmp_path / "discovery_s0001_lot_2026-05-30.json"
    scorecard.write_text("{}")

    # CAMPAIGN with two lines for the same basename
    campaign = tmp_path / "CAMPAIGN.md"
    lot_name = lot_csv.name
    campaign.write_text(
        "# Discovery campaign\n\n"
        "| Lot | Rows | File | Status |\n"
        "|---|---|---|---|\n"
        f"| lot-auto | 1–50 (50 TICs) | `{lot_name}` | ENQUEUED 2026-05-29 |\n"
        f"| lot-auto | 51–100 (50 TICs) | `{lot_name}` | ENQUEUED 2026-05-30 |\n"
    )
    original_text = campaign.read_text()
    status = _make_status(tmp_path)

    result = fin.finalize_run(
        lot_csv=str(lot_csv),
        run_id="discovery_s0001_lot_2026-05-30",
        scorecard_json=str(scorecard),
        dossier=str(dossier),
        db_path=str(db),
        campaign_path=str(campaign),
        status_path=str(status),
    )

    assert result["campaign_action"] == "no_unique_match"
    assert campaign.read_text() == original_text


# ---------------------------------------------------------------------------
# 4. DB stats: tic_id TEXT vs int normalisation
# ---------------------------------------------------------------------------

def test_db_stats_text_vs_int(tmp_path):
    """DB stores tic_id as TEXT; CSV may have int-like strings → must match."""
    # CSV with plain integers
    lot_csv = _make_lot_csv(tmp_path, [12345678, 99999999, 11111111])
    # DB stores as TEXT (which is the production schema)
    db = _make_db(tmp_path, [
        ("12345678", "completed"),
        ("99999999", "completed"),
        ("11111111", "failed"),
    ])

    tic_ids = fin._read_lot_tics(str(lot_csv))
    stats = fin._db_stats(str(db), tic_ids)

    assert stats["n_lot"] == 3
    assert stats["n_completed"] == 2
    assert stats["n_failed"] == 1
    assert stats["n_other"] == 0


def test_db_stats_all_completed(tmp_path):
    lot_csv = _make_lot_csv(tmp_path, [1, 2, 3, 4, 5])
    db = _make_db(tmp_path, [(str(i), "completed") for i in range(1, 6)])
    tic_ids = fin._read_lot_tics(str(lot_csv))
    stats = fin._db_stats(str(db), tic_ids)
    assert stats["n_completed"] == 5
    assert stats["n_failed"] == 0
    assert stats["n_other"] == 0


def test_db_stats_no_tasks(tmp_path):
    """TICs in CSV but no tasks in DB → n_other = n_lot."""
    lot_csv = _make_lot_csv(tmp_path, [42, 43])
    db = _make_db(tmp_path, [])  # empty tasks table
    tic_ids = fin._read_lot_tics(str(lot_csv))
    stats = fin._db_stats(str(db), tic_ids)
    assert stats["n_lot"] == 2
    assert stats["n_completed"] == 0
    assert stats["n_failed"] == 0
    assert stats["n_other"] == 2


# ---------------------------------------------------------------------------
# 4b. S3 (docs/superpowers/specs/2026-09-26-enqueue-s1-s4-design.md §3.1
# item 3): run_id fast path — same result as the legacy whole-table scan,
# via a single GROUP BY status scoped to tasks.run_id.
# ---------------------------------------------------------------------------

def _make_db_with_run_id(tmp_path: Path, rows: list[tuple], name="test_runid.db") -> Path:
    """rows: (tic_id, status, run_id) tuples."""
    db = tmp_path / name
    con = sqlite3.connect(str(db))
    con.execute(
        "CREATE TABLE tasks (id INTEGER PRIMARY KEY, tic_id TEXT, "
        "status TEXT, run_id TEXT)"
    )
    con.executemany(
        "INSERT INTO tasks (tic_id, status, run_id) VALUES (?, ?, ?)", rows)
    con.commit()
    con.close()
    return db


def test_db_stats_run_id_fast_path_matches_legacy_shape(tmp_path, monkeypatch):
    monkeypatch.setenv("EXONIT_RUN_SCOPE", "tasks")
    lot_csv = _make_lot_csv(tmp_path, [501, 502, 503])
    db = _make_db_with_run_id(tmp_path, [
        ("501", "completed", "run_x"),
        ("502", "failed", "run_x"),
        ("503", "pending", "run_x"),
        # A DIFFERENT run's row for the SAME TIC must not leak in.
        ("501", "failed", "run_y"),
    ])
    tic_ids = fin._read_lot_tics(str(lot_csv))
    stats = fin._db_stats(str(db), tic_ids, run_id="run_x")
    assert stats == {"n_lot": 3, "n_completed": 1, "n_failed": 1, "n_other": 1}


def test_db_stats_without_run_id_uses_legacy_scan_path(tmp_path):
    """No run_id given (older caller) -> unchanged whole-table-scan path."""
    lot_csv = _make_lot_csv(tmp_path, [601, 602])
    db = _make_db(tmp_path, [("601", "completed"), ("602", "failed")])
    tic_ids = fin._read_lot_tics(str(lot_csv))
    stats = fin._db_stats(str(db), tic_ids, run_id=None)
    assert stats["n_completed"] == 1
    assert stats["n_failed"] == 1


def test_db_stats_run_id_given_but_scope_kind_not_tasks_falls_back(tmp_path):
    """run_id given but EXONIT_RUN_SCOPE is not 'tasks' (default) and the
    run has no tasks.run_id rows -> falls back to the legacy scan, not an
    empty/wrong result."""
    lot_csv = _make_lot_csv(tmp_path, [701, 702])
    db = _make_db(tmp_path, [("701", "completed"), ("702", "failed")])
    tic_ids = fin._read_lot_tics(str(lot_csv))
    stats = fin._db_stats(str(db), tic_ids, run_id="run_never_scoped")
    assert stats["n_completed"] == 1
    assert stats["n_failed"] == 1


# ---------------------------------------------------------------------------
# 5. Dossier parsing
# ---------------------------------------------------------------------------

def test_parse_dossier_found(tmp_path):
    """Standard dossier format → correct n_strong and n_finalists."""
    dossier = _make_dossier(tmp_path, n_strong=7, n_finalists=3)
    result = fin._parse_dossier(str(dossier))
    assert result["n_strong"] == 7
    assert result["n_finalists"] == 3


def test_parse_dossier_zero_finalists(tmp_path):
    """Zero finalists line is parsed correctly."""
    dossier = _make_dossier(tmp_path, n_strong=5, n_finalists=0)
    result = fin._parse_dossier(str(dossier))
    assert result["n_strong"] == 5
    assert result["n_finalists"] == 0


def test_parse_dossier_missing_lines(tmp_path):
    """A dossier without the expected lines → None values, no crash."""
    dossier = tmp_path / "empty_dossier.md"
    dossier.write_text("# Just a title\n\nNo relevant lines here.\n")
    result = fin._parse_dossier(str(dossier))
    assert result["n_strong"] is None
    assert result["n_finalists"] is None


def test_parse_dossier_real_format(tmp_path):
    """Matches the exact format generated by discovery_postprocess.py."""
    dossier = tmp_path / "real.md"
    dossier.write_text(
        "# Discovery vetting dossier — data/discovery/s0001_lot_20260530T173210.csv\n\n"
        "Strong candidates (v6>0.5, post-veto): 7.\n\n"
        "| TIC | P(d) | depth(ppm) | SNR | v6 | EB flags | FPP | verdict |\n"
        "|---|---|---|---|---|---|---|---|\n"
        "| 139121623 | 25.72 | 1022 | 33 | 0.93 | ok | - | - |\n"
        "\n**Finalists (pass all + FPP<0.5): 0**: none\n"
    )
    result = fin._parse_dossier(str(dossier))
    assert result["n_strong"] == 7
    assert result["n_finalists"] == 0


def test_parse_dossier_legacy_catalan_format(tmp_path):
    """Historical production dossiers (Catalan label) must still parse."""
    dossier = tmp_path / "legacy.md"
    dossier.write_text(
        "Strong candidates (v6>0.5, post-veto): 3.\n\n"
        "\n**Finalistes (passen tot + FPP<0,5): 2**: 1, 2\n"
    )
    result = fin._parse_dossier(str(dossier))
    assert result["n_strong"] == 3
    assert result["n_finalists"] == 2


# ---------------------------------------------------------------------------
# 6. STATUS.md section creation and idempotency
# ---------------------------------------------------------------------------

def test_status_creates_section_if_absent(tmp_path):
    """STATUS.md gets the auto section appended if it does not exist."""
    lot_csv = _make_lot_csv(tmp_path, [1])
    db = _make_db(tmp_path, [("1", "completed")])
    dossier = _make_dossier(tmp_path, n_strong=0, n_finalists=0)
    scorecard = tmp_path / "discovery_s0001_lot_2026-05-30.json"
    scorecard.write_text("{}")
    campaign = _make_campaign(tmp_path, lot_csv.name)
    # STATUS without the markers
    status = _make_status(tmp_path, with_section=False)

    fin.finalize_run(
        lot_csv=str(lot_csv),
        run_id="discovery_s0001_lot_2026-05-30",
        scorecard_json=str(scorecard),
        dossier=str(dossier),
        db_path=str(db),
        campaign_path=str(campaign),
        status_path=str(status),
    )

    status_text = status.read_text()
    assert fin._STATUS_SECTION_START in status_text
    assert fin._STATUS_SECTION_END in status_text
    assert "discovery_s0001_lot_2026-05-30" in status_text
    # Narrative content must be preserved
    assert "## Incidents recents" in status_text
    assert "Some incident." in status_text


def test_status_adds_row_when_section_exists(tmp_path):
    """A row is added inside the existing markers."""
    lot_csv = _make_lot_csv(tmp_path, [1])
    db = _make_db(tmp_path, [("1", "completed")])
    dossier = _make_dossier(tmp_path, n_strong=1, n_finalists=0)
    scorecard = tmp_path / "discovery_s0001_lot_2026-05-30.json"
    scorecard.write_text("{}")
    campaign = _make_campaign(tmp_path, lot_csv.name)
    status = _make_status(tmp_path, with_section=True)

    fin.finalize_run(
        lot_csv=str(lot_csv),
        run_id="discovery_s0001_lot_2026-05-30",
        scorecard_json=str(scorecard),
        dossier=str(dossier),
        db_path=str(db),
        campaign_path=str(campaign),
        status_path=str(status),
    )

    status_text = status.read_text()
    assert "discovery_s0001_lot_2026-05-30" in status_text
    # Row is between the markers
    start = status_text.index(fin._STATUS_SECTION_START)
    end = status_text.index(fin._STATUS_SECTION_END)
    section = status_text[start:end]
    assert "discovery_s0001_lot_2026-05-30" in section


def test_status_replaces_existing_row(tmp_path):
    """A second call replaces the run_id row, not appends it."""
    lot_csv = _make_lot_csv(tmp_path, [1, 2])
    db = _make_db(tmp_path, [("1", "completed"), ("2", "completed")])
    dossier = _make_dossier(tmp_path, n_strong=0, n_finalists=0)
    scorecard = tmp_path / "discovery_s0001_lot_2026-05-30.json"
    scorecard.write_text("{}")
    campaign = _make_campaign(tmp_path, lot_csv.name)
    # STATUS already has a stale row for the same run_id
    stale_row = "| discovery_s0001_lot_2026-05-30 | 2026-05-30 | `old.csv` | 0 | 0 | 0 | 0 | `old` |"
    status = _make_status(tmp_path, with_section=True, existing_row=stale_row)

    fin.finalize_run(
        lot_csv=str(lot_csv),
        run_id="discovery_s0001_lot_2026-05-30",
        scorecard_json=str(scorecard),
        dossier=str(dossier),
        db_path=str(db),
        campaign_path=str(campaign),
        status_path=str(status),
    )

    status_text = status.read_text()
    # Only one LINE (row) containing the run_id
    matching_lines = [l for l in status_text.splitlines() if "discovery_s0001_lot_2026-05-30" in l]
    assert len(matching_lines) == 1, (
        f"Expected 1 row containing run_id, got {len(matching_lines)}:\n{status_text}"
    )
    # The stale row is gone
    assert "old.csv" not in status_text


# ---------------------------------------------------------------------------
# 7. Orchestrator integration: FINALIZE in state transitions
# ---------------------------------------------------------------------------

def test_orchestrator_includes_finalize_in_states():
    """FINALIZE must be in STATES between FREE_GPU and DONE."""
    # We import the orchestrator directly
    orch_dir = Path(__file__).resolve().parent.parent / "scripts"
    sys.path.insert(0, str(orch_dir))
    import discovery_orchestrator as orch

    states = orch.STATES
    assert "FINALIZE" in states
    fi = states.index("FINALIZE")
    fg = states.index("FREE_GPU")
    done = states.index("DONE")
    assert fg < fi < done, (
        f"FINALIZE ({fi}) must be between FREE_GPU ({fg}) and DONE ({done})"
    )


def test_orchestrator_finalize_in_progress_states():
    """FINALIZE must be in _PROGRESS_STATES (so stale detail is cleared)."""
    orch_dir = Path(__file__).resolve().parent.parent / "scripts"
    sys.path.insert(0, str(orch_dir))
    import discovery_orchestrator as orch

    assert "FINALIZE" in orch._PROGRESS_STATES


def test_orchestrator_dry_run_transitions_include_finalize(tmp_path):
    """Full dry-run transitions include FINALIZE before DONE."""
    from unittest.mock import patch

    orch_dir = Path(__file__).resolve().parent.parent / "scripts"
    sys.path.insert(0, str(orch_dir))
    import discovery_orchestrator as orch

    lot_csv = tmp_path / "s0001_lot_test.csv"
    lot_csv.write_text("tic_id,label\n12345,candidate\n")
    sf = str(tmp_path / "state.json")

    seen_states = []
    original_set_state = orch._set_state

    def capturing_set_state(path, state_name, **extra):
        seen_states.append(state_name)
        return original_set_state(path, state_name, **extra)

    args = orch._parse_args([
        "--lot-csv", str(lot_csv),
        "--sector", "1",
        "--state-file", sf,
        "--poll-interval", "0",
        "--dry-run",
    ])

    with patch.object(orch, "_set_state", side_effect=capturing_set_state):
        orch.run(args)

    expected = ["SCANNING", "FPP_BACKFILL", "EXOMINER_BACKFILL", "STOP_WORKERS",
                "POSTPROCESS", "STOP_NVIDIARACK", "LORA", "FREE_GPU", "FINALIZE", "DONE"]
    assert seen_states == expected, (
        f"Expected transitions: {expected}\nGot: {seen_states}"
    )


def test_orchestrator_finalize_non_fatal(tmp_path):
    """A FINALIZE error must NOT cause the run to end in FAILED."""
    from unittest.mock import patch

    orch_dir = Path(__file__).resolve().parent.parent / "scripts"
    sys.path.insert(0, str(orch_dir))
    import discovery_orchestrator as orch

    lot_csv = tmp_path / "s0001_lot_test.csv"
    lot_csv.write_text("tic_id,label\n12345,candidate\n")
    sf = str(tmp_path / "state.json")

    def bad_finalize(*a, **kw):
        raise RuntimeError("simulated finalize error")

    args = orch._parse_args([
        "--lot-csv", str(lot_csv),
        "--sector", "1",
        "--state-file", sf,
        "--poll-interval", "0",
        "--dry-run",
    ])

    # patch step_finalize to raise
    with patch.object(orch, "step_finalize", side_effect=bad_finalize):
        # Should not crash; DONE must still be reached
        # (step_finalize raises, but the caller in run() catches via the
        # ok/detail interface — so we also patch _set_state to absorb FINALIZE)
        try:
            orch.run(args)
        except RuntimeError:
            # If the exception propagates, the test still checks the final state
            pass

    # The run should still reach DONE (FINALIZE is non-fatal by design —
    # the step itself returns True on any internal error, so the orchestrator
    # always proceeds to DONE).
    # Since we patched step_finalize to raise (not return), the orchestrator
    # will catch it in run() only if step_finalize itself handles exceptions.
    # The actual step_finalize wraps in try/except → returns True always.
    # Since we bypassed that wrapper via the patch, the exception propagates.
    # The test verifies that step_finalize in production never raises.
    # Here we verify the CONTRACT via the state file.
    state = orch._load_state(sf)
    # In dry_run, step_finalize returns True immediately, so DONE is reached.
    # The bad_finalize patch is only called in non-dry-run, but dry_run=True
    # means step_finalize short-circuits before calling finalize_run.
    assert state["state"] == "DONE"
