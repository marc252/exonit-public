"""Schema migration test: tasks.lc_load_recovered column exists and
defaults to 0. Idempotent across multiple ExoNitDB() instantiations."""
import os
import sqlite3
import tempfile

from exonit.db import ExoNitDB


def test_tasks_has_lc_load_recovered_column():
    with tempfile.TemporaryDirectory() as td:
        db_path = os.path.join(td, "exonit.db")
        ExoNitDB(db_path)  # __init__ runs migrations
        conn = sqlite3.connect(db_path)
        cols = [r[1] for r in conn.execute("PRAGMA table_info(tasks)").fetchall()]
        conn.close()
        assert "lc_load_recovered" in cols


def test_tasks_lc_load_recovered_defaults_to_zero():
    with tempfile.TemporaryDirectory() as td:
        db_path = os.path.join(td, "exonit.db")
        db = ExoNitDB(db_path)
        db._exec(
            "INSERT INTO tasks (task_id, tic_id, status) VALUES (?, ?, 'pending')",
            ("t1", "12345"))
        row = db.conn.execute(
            "SELECT lc_load_recovered FROM tasks WHERE task_id='t1'"
        ).fetchone()
        assert row[0] == 0


def test_migration_is_idempotent():
    with tempfile.TemporaryDirectory() as td:
        db_path = os.path.join(td, "exonit.db")
        ExoNitDB(db_path)  # first init creates the column
        ExoNitDB(db_path)  # second init must not raise
        conn = sqlite3.connect(db_path)
        cols = [r[1] for r in conn.execute("PRAGMA table_info(tasks)").fetchall()]
        conn.close()
        assert cols.count("lc_load_recovered") == 1


def test_complete_task_persists_lc_load_recovered_flag():
    with tempfile.TemporaryDirectory() as td:
        db_path = os.path.join(td, "exonit.db")
        db = ExoNitDB(db_path)
        db._exec(
            "INSERT INTO tasks (task_id, tic_id, status) "
            "VALUES (?, ?, 'assigned')",
            ("t-recovered", "99999"))
        db.complete_task(
            "t-recovered", processing_time=12.3, gpu_used=0,
            lc_load_recovered=True)
        row = db.conn.execute(
            "SELECT status, lc_load_recovered FROM tasks "
            "WHERE task_id='t-recovered'"
        ).fetchone()
        assert row[0] == "completed"
        assert row[1] == 1


def test_complete_task_default_lc_load_recovered_zero():
    with tempfile.TemporaryDirectory() as td:
        db_path = os.path.join(td, "exonit.db")
        db = ExoNitDB(db_path)
        db._exec(
            "INSERT INTO tasks (task_id, tic_id, status) "
            "VALUES (?, ?, 'assigned')",
            ("t-normal", "11111"))
        db.complete_task("t-normal", processing_time=5.0, gpu_used=1)
        row = db.conn.execute(
            "SELECT lc_load_recovered FROM tasks WHERE task_id='t-normal'"
        ).fetchone()
        assert row[0] == 0
