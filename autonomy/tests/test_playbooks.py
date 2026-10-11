import sys

from autonomy.types import PlaybookChoice
from autonomy.playbooks import Executor, validate, CATALOG, command_for
import autonomy.playbooks as playbooks_module


def _c(pb, params, conf=0.9):
    return PlaybookChoice(pb, params, conf, "test", "rules")


def test_catalog_complete():
    assert set(CATALOG) == {"rebuild_streams", "requeue_orphans", "restart_worker",
                            "kill_zombies", "clean_fits_cache", "wal_checkpoint",
                            "resume_run", "restart_server", "pause_and_alert"}


def test_validate_rejects_unknown_and_bad_params():
    assert validate(_c("format_disk", {})) is not None
    assert validate(_c("restart_worker", {"worker": "worker-bogus"})) is not None
    assert validate(_c("restart_worker", {})) is not None  # missing required
    assert validate(_c("rebuild_streams", {"stage": "fpp"})) is None


def test_dry_run_logs_but_never_executes(tmp_path):
    ex = Executor(tmp_path / "s.json", tmp_path / "a.log", dry_run=True)
    r = ex.execute(_c("rebuild_streams", {"stage": "fpp"}), trigger="test")
    assert r.status == "dry_run"
    lines = (tmp_path / "a.log").read_text().strip().splitlines()
    assert len(lines) == 1 and '"rebuild_streams"' in lines[0]


def test_limits_per_run(tmp_path):
    ex = Executor(tmp_path / "s.json", tmp_path / "a.log", dry_run=True)
    for _ in range(2):
        assert ex.execute(_c("rebuild_streams", {"stage": "fpp"}), "t").status == "dry_run"
    assert ex.execute(_c("rebuild_streams", {"stage": "fpp"}), "t").status == "refused_limit"
    ex.reset_scope("run")
    assert ex.execute(_c("rebuild_streams", {"stage": "fpp"}), "t").status == "dry_run"


def test_armed_guard_forces_dry_run(tmp_path, monkeypatch):
    monkeypatch.delenv("EXONIT_AUTONOMY_ARMED", raising=False)
    ex = Executor(tmp_path / "s.json", tmp_path / "a.log", dry_run=False)
    r = ex.execute(_c("wal_checkpoint", {}), "t")
    assert r.status == "dry_run"  # without the double key, never executes


def test_command_for_builds_real_commands():
    cmd = command_for(_c("resume_run", {"run_id": "lot123"}))
    assert "--resume" in cmd and "lot123" in cmd and "--force" in cmd
    assert command_for(_c("pause_and_alert", {"reason": "x"})) is None


# ---------------------------------------------------------------------------
# restart_server (incident 2026-08-15): no params, limit 2/24h, global
# (not scoped per host/worker).
# ---------------------------------------------------------------------------

def test_restart_server_command_for_builds_script_call():
    cmd = command_for(_c("restart_server", {}))
    assert cmd[-1] == "scripts/restart_exonit_server.py"


def test_restart_server_validate_accepts_no_params():
    assert validate(_c("restart_server", {})) is None


def test_restart_server_validate_rejects_unknown_param():
    assert validate(_c("restart_server", {"host": "worker-default"})) is not None


def test_restart_server_limit_2_per_24h(tmp_path):
    ex = Executor(tmp_path / "s.json", tmp_path / "a.log", dry_run=True)
    for _ in range(2):
        assert ex.execute(_c("restart_server", {}), "t").status == "dry_run"
    r = ex.execute(_c("restart_server", {}), "t")
    assert r.status == "refused_limit"


# ---------------------------------------------------------------------------
# resume_run (incident 2026-09-19, unattended run of 1-2 months):
#   (a) `detached=True` -- the `Executor` does NOT run it in the foreground with
#       `timeout=300` (it would kill the orchestrator that runs for hours).
#   (b) the 3/run limit is scoped per `run_id`, not global.
# ---------------------------------------------------------------------------

def test_resume_run_is_detached_others_are_not():
    assert CATALOG["resume_run"].detached is True
    assert CATALOG["restart_server"].detached is False
    assert CATALOG["requeue_orphans"].detached is False


def test_resume_run_limit_scoped_by_run_id_not_global(tmp_path):
    ex = Executor(tmp_path / "s.json", tmp_path / "a.log", dry_run=True)
    for _ in range(3):
        assert ex.execute(_c("resume_run", {"run_id": "run-A"}), "t").status == "dry_run"
    assert ex.execute(_c("resume_run", {"run_id": "run-A"}), "t").status == "refused_limit"

    # Before the fix the counter was global (`_count_key` without scope):
    # a DIFFERENT `run_id` already found it exhausted. Now run-B has its
    # own counter, intact.
    assert ex.execute(_c("resume_run", {"run_id": "run-B"}), "t").status == "dry_run"


def test_resume_run_reset_scope_run_clears_per_run_id_counter(tmp_path):
    ex = Executor(tmp_path / "s.json", tmp_path / "a.log", dry_run=True)
    for _ in range(3):
        ex.execute(_c("resume_run", {"run_id": "run-A"}), "t")
    assert ex.execute(_c("resume_run", {"run_id": "run-A"}), "t").status == "refused_limit"

    ex.reset_scope("run")

    assert ex.execute(_c("resume_run", {"run_id": "run-A"}), "t").status == "dry_run"


def test_run_detached_immediate_death_reports_dead_and_log_tail(tmp_path):
    ex = Executor(tmp_path / "s.json", tmp_path / "a.log", dry_run=True, detached_grace_s=2.0)
    cmd = [sys.executable, "-c", "import sys; print('boom'); sys.exit(3)"]

    result = ex._run_detached(cmd, "resume_run")

    assert result.status == "executed"
    assert "DEAD_IMMEDIATELY" in result.detail
    assert "rc=3" in result.detail
    assert "boom" in result.detail  # tail of the captured log
    logs = list((tmp_path / "playbook_logs").glob("resume_run_*.log"))
    assert len(logs) == 1


def test_run_detached_still_alive_after_grace_reports_pid(tmp_path):
    ex = Executor(tmp_path / "s.json", tmp_path / "a.log", dry_run=True, detached_grace_s=0.2)
    cmd = [sys.executable, "-c", "import time; time.sleep(0.6)"]

    result = ex._run_detached(cmd, "resume_run")

    assert result.status == "executed"
    assert "DEAD_IMMEDIATELY" not in result.detail
    assert "alive" in result.detail
    assert "pid=" in result.detail


def test_execute_resume_run_armed_dispatches_to_detached_path(tmp_path, monkeypatch):
    """`execute()` in armed mode (not dry_run) for a `detached` playbook does NOT
    go through `subprocess.run(..., timeout=300)` -- `command_for` is
    monkeypatched so the test never really launches
    `scripts/discovery_orchestrator.py` (see gotcha
    bug_discovery_scan_test_spawns_orphan_orchestrator.md)."""
    monkeypatch.setenv("EXONIT_AUTONOMY_ARMED", "1")
    monkeypatch.setattr(
        playbooks_module,
        "command_for",
        lambda choice: [sys.executable, "-c", "import sys; sys.exit(0)"],
    )
    ex = Executor(tmp_path / "s.json", tmp_path / "a.log", dry_run=False, detached_grace_s=2.0)

    result = ex.execute(_c("resume_run", {"run_id": "run-A"}), "t")

    assert result.status == "executed"
    assert "detached=True" in result.detail
    assert "rc=0" in result.detail


def test_load_fleet_warns_and_falls_back_when_file_missing(tmp_path, caplog):
    import logging
    from autonomy.playbooks import load_fleet
    with caplog.at_level(logging.WARNING, logger="autonomy.playbooks"):
        fleet = load_fleet(tmp_path / "nope.json")
    assert list(fleet) == ["worker-default"]
    assert any("fleet file" in r.message for r in caplog.records)


def test_armed_playbook_with_missing_script_fails_clearly_without_running(tmp_path, monkeypatch):
    """requeue_orphans/kill_zombies/clean_fits_cache/wal_checkpoint name scripts that do not exist in the
    repository: the armed executor must say so and never spawn a process."""
    monkeypatch.setenv("EXONIT_AUTONOMY_ARMED", "1")
    called = []
    monkeypatch.setattr(playbooks_module.subprocess, "run", lambda *a, **k: called.append(a) or None)
    ex = Executor(tmp_path / "s.json", tmp_path / "a.log", dry_run=False)
    r = ex.execute(_c("wal_checkpoint", {}), "t")
    assert r.status == "executed"
    assert "script not found: scripts/wal_checkpoint.py" in r.detail and "nothing was run" in r.detail
    assert called == []


def test_missing_script_helper_only_flags_nonexistent_scripts():
    from autonomy.playbooks import _missing_script
    assert _missing_script(["py", "scripts/restart_exonit_server.py"]) is None   # exists in the repo
    assert _missing_script(["py", "scripts/does_not_exist.py"]) == "scripts/does_not_exist.py"
    assert _missing_script(["py", "bin/farmctl", "restart", "x"]) is None
    assert _missing_script(None) is None
