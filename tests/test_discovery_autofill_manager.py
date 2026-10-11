"""Tests for the Task-8 auto-fill manager: autofill_tick + run_autofill.

These tests exercise the safety logic (empty-lot guard, anomaly cap, completion,
happy path) using a temporary SQLite DB and mocked AutofillDeps callables.  They
do NOT need PostgreSQL and must ALWAYS run — no PG skipif here.

Contrast with tests/test_discovery_autofill.py, which tests the Task-7
get_autofill/set_autofill round-trip against the throwaway exonit_test PG DB
and is guarded by a pg_test_available skipif.
"""

import os
import sqlite3
import tempfile

import pytest


# ---------------------------------------------------------------------------
# SQLite helpers
# ---------------------------------------------------------------------------

def _make_sqlite_db() -> str:
    """Create a temporary SQLite DB with the discovery autofill + state tables."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    con = sqlite3.connect(path)
    con.executescript("""
        CREATE TABLE IF NOT EXISTS discovery_autofill (
          sector       INTEGER PRIMARY KEY,
          status       TEXT,
          current_lot  TEXT,
          lots_done    INTEGER DEFAULT 0,
          anomalies    INTEGER DEFAULT 0,
          last_alert   TEXT,
          updated_at   TEXT
        );
        CREATE TABLE IF NOT EXISTS discovery_sector_state (
          sector     INTEGER PRIMARY KEY,
          state      TEXT NOT NULL DEFAULT 'coming_soon',
          detail     TEXT,
          updated_at TEXT NOT NULL DEFAULT (datetime('now'))
        );
        CREATE TABLE IF NOT EXISTS discovery_no_data (
          tic_id     TEXT NOT NULL,
          sector     INTEGER NOT NULL,
          reason     TEXT NOT NULL DEFAULT 'no_fits',
          checked_at TEXT NOT NULL DEFAULT (datetime('now')),
          PRIMARY KEY (tic_id, sector)
        );
    """)
    con.commit()
    con.close()
    return path


@pytest.fixture
def sqlite_db():
    """Temporary SQLite DB with autofill + sector-state tables; cleaned up after test."""
    path = _make_sqlite_db()
    yield path
    try:
        os.unlink(path)
    except OSError:
        pass


def _deps(db_path, *, gap_fn, scan_fn, launch_fn=None, poll_fn=None, download_fn=None):
    """Build an AutofillDeps with the given callables (remaining ones are no-ops / DONE)."""
    from exonit.discovery_autofill import AutofillDeps
    return AutofillDeps(
        db_path=db_path,
        gap_stats=gap_fn,
        scan=scan_fn,
        launch=launch_fn if launch_fn is not None else (lambda lot, sec: None),
        poll=poll_fn if poll_fn is not None else (lambda: {"state": "DONE"}),
        download=download_fn,
    )


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

def test_autofill_tick_happy_path(sqlite_db):
    """gaps>0 → scan enqueues, orchestrator finishes DONE → lots_done advances."""
    from exonit.discovery_autofill import autofill_tick
    from exonit.discovery_state import get_autofill

    launched = []
    polled = []

    deps = _deps(
        sqlite_db,
        gap_fn=lambda sec: {"gaps_processable": 5},
        scan_fn=lambda sec: {"enqueued": 10, "lot_csv": "test_lot.csv"},
        launch_fn=lambda lot, sec: launched.append((lot, sec)),
        poll_fn=lambda: (polled.append(True) or {"state": "DONE"}),
    )

    result = autofill_tick(1, deps=deps)

    assert result["status"] == "running", f"unexpected status: {result}"
    assert result["lots_done"] == 1
    # Orchestrator was launched exactly once
    assert launched == [("test_lot.csv", 1)]
    # Poll was called exactly once
    assert len(polled) == 1
    # DB row reflects the new state
    row = get_autofill(1, db_path=sqlite_db)
    assert row is not None
    assert row["lots_done"] == 1
    assert int(row["anomalies"] or 0) == 0


# ---------------------------------------------------------------------------
# Empty-lot guard
# ---------------------------------------------------------------------------

def test_autofill_tick_empty_lot_guard(sqlite_db):
    """scan returns enqueued=0 while gaps>0 → status alert, orchestrator NOT launched."""
    from exonit.discovery_autofill import autofill_tick
    from exonit.discovery_state import get_autofill

    launched = []

    deps = _deps(
        sqlite_db,
        gap_fn=lambda sec: {"gaps_processable": 5},
        scan_fn=lambda sec: {"enqueued": 0, "lot_csv": ""},
        launch_fn=lambda lot, sec: launched.append((lot, sec)),
    )

    result = autofill_tick(1, deps=deps)

    assert result["status"] == "alert", f"expected alert, got: {result}"
    assert result.get("reason") == "empty_lot"
    # Orchestrator must NOT have been started (never chain empty lots)
    assert launched == [], "orchestrator must not be launched on empty lot"
    # DB reflects alert
    row = get_autofill(1, db_path=sqlite_db)
    assert row is not None
    assert row["status"] == "alert"
    # Empty-lot is a different condition from a FAILED orchestrator run:
    # the anomaly counter must NOT be incremented
    assert int(row["anomalies"] or 0) == 0, (
        "anomalies must not be incremented on empty-lot alert "
        "(empty-lot is not a consecutive-FAILED anomaly)"
    )


def test_run_autofill_empty_lot_stops_loop(sqlite_db):
    """run_autofill does not loop when scan is empty — stops on first empty result."""
    from exonit.discovery_autofill import run_autofill
    from exonit.discovery_state import get_autofill

    scan_calls = []

    deps = _deps(
        sqlite_db,
        gap_fn=lambda sec: {"gaps_processable": 5},
        scan_fn=lambda sec: (scan_calls.append(sec) or {"enqueued": 0, "lot_csv": ""}),
    )

    run_autofill(1, deps=deps)

    # Scan must have been called exactly once (no chaining of empty lots)
    assert len(scan_calls) == 1, f"expected 1 scan call, got {len(scan_calls)}"
    row = get_autofill(1, db_path=sqlite_db)
    assert row["status"] == "alert"


# ---------------------------------------------------------------------------
# Anomaly cap
# ---------------------------------------------------------------------------

def test_autofill_tick_anomaly_cap(sqlite_db):
    """The tick at MAX_ANOMALIES-1 anomalies + another FAILED → alert, stop."""
    from exonit.discovery_autofill import autofill_tick, MAX_ANOMALIES
    from exonit.discovery_state import get_autofill, set_autofill

    # Seed DB: one anomaly already recorded (one below the cap)
    set_autofill(1, db_path=sqlite_db, status="running",
                 anomalies=MAX_ANOMALIES - 1, lots_done=3)

    deps = _deps(
        sqlite_db,
        gap_fn=lambda sec: {"gaps_processable": 5},
        scan_fn=lambda sec: {"enqueued": 10, "lot_csv": "lot.csv"},
        poll_fn=lambda: {"state": "FAILED"},
    )

    result = autofill_tick(1, deps=deps)

    assert result["status"] == "alert", f"expected alert, got: {result}"
    assert result.get("reason") == "anomaly_cap"
    row = get_autofill(1, db_path=sqlite_db)
    assert row["status"] == "alert"
    assert int(row["anomalies"] or 0) == MAX_ANOMALIES


def test_run_autofill_anomaly_cap_stops_loop(sqlite_db):
    """run_autofill stops after MAX_ANOMALIES consecutive FAILED orchestrator runs."""
    from exonit.discovery_autofill import run_autofill, MAX_ANOMALIES
    from exonit.discovery_state import get_autofill

    poll_calls = []

    deps = _deps(
        sqlite_db,
        gap_fn=lambda sec: {"gaps_processable": 5},
        scan_fn=lambda sec: {"enqueued": 10, "lot_csv": "lot.csv"},
        poll_fn=lambda: (poll_calls.append(True) or {"state": "FAILED"}),
    )

    run_autofill(1, deps=deps)

    # Loop must have called poll exactly MAX_ANOMALIES times before stopping
    assert len(poll_calls) == MAX_ANOMALIES, (
        f"expected {MAX_ANOMALIES} poll calls (one per anomaly before cap), "
        f"got {len(poll_calls)}"
    )
    row = get_autofill(1, db_path=sqlite_db)
    assert row["status"] == "alert"
    assert int(row["anomalies"] or 0) == MAX_ANOMALIES


# ---------------------------------------------------------------------------
# Completion
# ---------------------------------------------------------------------------

def test_autofill_tick_completion(sqlite_db):
    """gaps_processable==0 → sector state 'complete', autofill status 'done'."""
    from exonit.discovery_autofill import autofill_tick
    from exonit.discovery_state import get_autofill, get_sector_state

    scan_calls = []

    def scan_fn(sec):
        scan_calls.append(sec)
        raise AssertionError("scan must not be called when gaps_processable==0")

    deps = _deps(
        sqlite_db,
        gap_fn=lambda sec: {"gaps_processable": 0},
        scan_fn=scan_fn,
    )

    result = autofill_tick(1, deps=deps)

    assert result["status"] == "done", f"expected done, got: {result}"
    assert scan_calls == [], "scan must not be called on completion"
    # Autofill DB row
    row = get_autofill(1, db_path=sqlite_db)
    assert row is not None
    assert row["status"] == "done"
    # Sector state
    state = get_sector_state(1, db_path=sqlite_db)
    assert state == "complete", f"expected sector state 'complete', got {state!r}"


# ---------------------------------------------------------------------------
# Task 9: Re-download + mark dataless TICs
# ---------------------------------------------------------------------------

def test_autofill_tick_download_triggered_when_missing(sqlite_db):
    """When gaps_processable==0 but missing_download_tics is non-empty,
    the download dep is called with those TIC IDs."""
    from exonit.discovery_autofill import autofill_tick

    download_calls = []
    call_count = [0]

    def gap_fn(sec):
        call_count[0] += 1
        if call_count[0] == 1:
            # First call: no processable gaps but TICs are awaiting download
            return {"gaps_processable": 0, "missing_download_tics": [1001, 1002]}
        else:
            # Second call (after download+mark): everything drained
            return {"gaps_processable": 0, "missing_download_tics": []}

    def download_fn(sec, tic_ids):
        download_calls.append((sec, list(tic_ids)))
        # Report all as still missing so they get inserted into no_data
        return set(tic_ids)

    def scan_fn(sec):
        raise AssertionError("scan must not be called when gaps_processable==0")

    deps = _deps(
        sqlite_db,
        gap_fn=gap_fn,
        scan_fn=scan_fn,
        download_fn=download_fn,
    )

    autofill_tick(1, deps=deps)

    # Download dep must have been called exactly once
    assert len(download_calls) == 1, f"expected 1 download call, got {download_calls}"
    # Called for sector 1 with the exact TIC IDs
    called_sec, called_tics = download_calls[0]
    assert called_sec == 1
    assert set(called_tics) == {1001, 1002}


def test_autofill_tick_marks_no_data_and_sector_completes(sqlite_db):
    """TICs still missing after the download retry are inserted into
    discovery_no_data, sector_gap_stats re-check shows 0 gaps, sector completes."""
    from exonit.discovery_autofill import autofill_tick
    from exonit.discovery_state import get_autofill, get_sector_state

    call_count = [0]

    def gap_fn(sec):
        call_count[0] += 1
        if call_count[0] == 1:
            return {"gaps_processable": 0, "missing_download_tics": [1001, 1002]}
        else:
            # After no_data inserts: no more processable gaps, no more missing
            return {"gaps_processable": 0, "missing_download_tics": []}

    def download_fn(sec, tic_ids):
        # Both TICs remain without data after the attempted re-download
        return set(tic_ids)

    def scan_fn(sec):
        raise AssertionError("scan must not be called when gaps_processable==0")

    deps = _deps(
        sqlite_db,
        gap_fn=gap_fn,
        scan_fn=scan_fn,
        download_fn=download_fn,
    )

    result = autofill_tick(1, deps=deps)

    # Manager should report completion
    assert result["status"] == "done", f"expected done, got: {result}"

    # discovery_no_data must contain both TICs
    con = sqlite3.connect(sqlite_db)
    rows = con.execute(
        "SELECT tic_id FROM discovery_no_data WHERE sector=1"
    ).fetchall()
    con.close()
    recorded_tics = {int(r[0]) for r in rows}
    assert 1001 in recorded_tics, f"1001 not in no_data rows: {recorded_tics}"
    assert 1002 in recorded_tics, f"1002 not in no_data rows: {recorded_tics}"

    # Autofill row must be 'done'
    af_row = get_autofill(1, db_path=sqlite_db)
    assert af_row is not None
    assert af_row["status"] == "done"

    # Sector state must be 'complete'
    state = get_sector_state(1, db_path=sqlite_db)
    assert state == "complete", f"expected sector state 'complete', got {state!r}"


def test_autofill_tick_download_success_returns_running(sqlite_db):
    """When some TICs are successfully downloaded (no longer missing),
    the tick returns 'running' so the next iteration can process them."""
    from exonit.discovery_autofill import autofill_tick

    call_count = [0]

    def gap_fn(sec):
        call_count[0] += 1
        if call_count[0] == 1:
            return {"gaps_processable": 0, "missing_download_tics": [1001, 1002]}
        else:
            # After download: 1001 now available, 1002 still missing but inserted as no_data
            # Suppose 1002 was marked no_data by download_fn; 1001 now adds a processable gap
            return {"gaps_processable": 1, "missing_download_tics": []}

    def download_fn(sec, tic_ids):
        # Only 1002 is still missing; 1001 was successfully downloaded
        return {1002}

    def scan_fn(sec):
        raise AssertionError("scan must not be called on the re-check within tick")

    deps = _deps(
        sqlite_db,
        gap_fn=gap_fn,
        scan_fn=scan_fn,
        download_fn=download_fn,
    )

    result = autofill_tick(1, deps=deps)

    # Next tick must be allowed — status 'running', not 'done'
    assert result["status"] == "running", f"expected running, got: {result}"

    # 1002 should be in no_data (still missing), 1001 should NOT be
    con = sqlite3.connect(sqlite_db)
    rows = con.execute(
        "SELECT tic_id FROM discovery_no_data WHERE sector=1"
    ).fetchall()
    con.close()
    recorded_tics = {int(r[0]) for r in rows}
    assert 1002 in recorded_tics, "1002 should be marked no_data (still missing)"
    assert 1001 not in recorded_tics, "1001 should not be no_data (was downloaded)"


def test_autofill_tick_no_download_dep_falls_back_to_done(sqlite_db):
    """When there is no download dep, gaps_processable==0 still marks complete,
    even if missing_download_tics is present — the dep is optional."""
    from exonit.discovery_autofill import autofill_tick
    from exonit.discovery_state import get_sector_state

    deps = _deps(
        sqlite_db,
        gap_fn=lambda sec: {"gaps_processable": 0, "missing_download_tics": [1001]},
        scan_fn=lambda sec: (_ for _ in ()).throw(AssertionError("scan must not be called")),
        download_fn=None,  # no download dep
    )

    result = autofill_tick(1, deps=deps)

    assert result["status"] == "done", f"expected done, got: {result}"
    state = get_sector_state(1, db_path=sqlite_db)
    assert state == "complete"


# ---------------------------------------------------------------------------
# terminal_state_is_fresh — the _poll staleness guard (race fix)
# ---------------------------------------------------------------------------

def test_fresh_when_no_launch_timestamp():
    """launched_at=None -> accept any terminal state (pre-guard fallback)."""
    from exonit.discovery_autofill import terminal_state_is_fresh

    assert terminal_state_is_fresh(0.0, None) is True
    assert terminal_state_is_fresh(1e12, None) is True


def test_stale_state_rejected():
    """State file written BEFORE launch is stale -> reject (the bug fixed)."""
    from exonit.discovery_autofill import terminal_state_is_fresh

    launched_at = 1000.0
    assert terminal_state_is_fresh(999.999, launched_at) is False


def test_fresh_state_accepted():
    """State file written AFTER launch is fresh -> accept."""
    from exonit.discovery_autofill import terminal_state_is_fresh

    launched_at = 1000.0
    assert terminal_state_is_fresh(1000.001, launched_at) is True


def test_boundary_equal_mtime_is_fresh():
    """mtime == launched_at counts as fresh (>= comparison)."""
    from exonit.discovery_autofill import terminal_state_is_fresh

    assert terminal_state_is_fresh(1000.0, 1000.0) is True


# ---------------------------------------------------------------------------
# External-stop reliability (the "stop button doesn't work" bug fix)
# ---------------------------------------------------------------------------

def test_external_stop_during_poll_is_honored(sqlite_db):
    """Stop received while poll() is running is honored: status stays 'stopped',
    lots_done is NOT incremented, autofill_tick returns {'status': 'stopped'}.

    Simulates the race: user presses Stop → endpoint writes status='stopped' to
    the DB → orchestrator finishes DONE in the same window → poll() returns
    {'state': 'DONE'}.  Without the post-poll re-check, the DONE handler would
    overwrite status back to 'running' and the loop would launch another lot,
    silently ignoring the user's stop.
    """
    from exonit.discovery_autofill import autofill_tick
    from exonit.discovery_state import get_autofill, set_autofill

    # Seed: autofill running with 2 lots already done
    set_autofill(1, db_path=sqlite_db, status="running", lots_done=2, anomalies=0)

    def _poll_with_stop():
        # Simulate the user pressing Stop during the poll.
        set_autofill(1, db_path=sqlite_db, status="stopped")
        # Orchestrator happened to finish DONE in the same window.
        return {"state": "DONE"}

    deps = _deps(
        sqlite_db,
        gap_fn=lambda sec: {"gaps_processable": 5},
        scan_fn=lambda sec: {"enqueued": 10, "lot_csv": "x.csv"},
        launch_fn=lambda lot, sec: None,
        poll_fn=_poll_with_stop,
    )

    result = autofill_tick(1, deps=deps)

    # tick must honor the stop, not the DONE
    assert result == {"status": "stopped"}, (
        f"expected {{'status': 'stopped'}}, got {result!r}"
    )

    # DB row must still be 'stopped' — NOT clobbered to 'running'
    row = get_autofill(1, db_path=sqlite_db)
    assert row is not None
    assert row["status"] == "stopped", (
        f"DB status was clobbered to {row['status']!r} (expected 'stopped')"
    )

    # lots_done must NOT have been incremented (stop = do not count this lot)
    assert int(row.get("lots_done") or 0) == 2, (
        f"lots_done must stay at 2, got {row.get('lots_done')!r}"
    )


def test_poll_stopped_sentinel_is_honored(sqlite_db):
    """When _poll returns {'state': 'STOPPED'} (Change B sentinel), autofill_tick
    returns {'status': 'stopped'} without incrementing anomalies or lots_done.

    This is the belt-and-braces path: _poll itself detected the stop (via the
    DB check in Change B) and returned early with the STOPPED sentinel.
    """
    from exonit.discovery_autofill import autofill_tick
    from exonit.discovery_state import get_autofill, set_autofill

    # Seed: autofill running with 1 anomaly already recorded
    set_autofill(1, db_path=sqlite_db, status="running", lots_done=3, anomalies=1)

    deps = _deps(
        sqlite_db,
        gap_fn=lambda sec: {"gaps_processable": 5},
        scan_fn=lambda sec: {"enqueued": 10, "lot_csv": "lot.csv"},
        launch_fn=lambda lot, sec: None,
        # poll returns the STOPPED sentinel directly (as Change B would)
        poll_fn=lambda: {"state": "STOPPED"},
    )

    result = autofill_tick(1, deps=deps)

    assert result == {"status": "stopped"}, (
        f"expected {{'status': 'stopped'}}, got {result!r}"
    )

    # lots_done and anomalies must NOT have been incremented
    row = get_autofill(1, db_path=sqlite_db)
    assert row is not None
    assert int(row.get("lots_done") or 0) == 3, (
        f"lots_done must stay at 3, got {row.get('lots_done')!r}"
    )
    assert int(row.get("anomalies") or 0) == 1, (
        f"anomalies must stay at 1, got {row.get('anomalies')!r}"
    )
