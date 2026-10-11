"""Tests for scripts/discovery_orchestrator.py.

Constraints (non-negotiable per briefing):
- NEVER real SSH, farmctl, server poll, or subprocess network/GPU calls.
- NEVER touch the live DB data/exonit.db.
- All external I/O is mocked/monkeypatched.
- Assertable constant: LORA_SSH_CMD must match the canonical string exactly.
"""

import json
import os
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import pytest

# ---------------------------------------------------------------------------
# Import the module under test
# ---------------------------------------------------------------------------
# Ensure scripts/ is on the path so the module can be imported directly.
SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import discovery_orchestrator as orch  # noqa: E402

# Fleet-autoscale-by-phase (docs/superpowers/specs/fleet-autoscale-by-phase.md):
# captured at COLLECTION time, before the autouse `mock_scale_fleet` fixture
# below ever patches `orch._scale_fleet`. Tests that need the REAL
# implementation (its own subprocess/kill-switch/AWS/fail-soft logic) call
# this directly instead of going through `orch._scale_fleet` (which every
# other test in this module gets neutered by default — see mock_scale_fleet).
SCALE_FLEET_REAL = orch._scale_fleet


@pytest.fixture(autouse=True)
def mock_scale_fleet():
    """Neuter the fleet-autoscale hook for every test in this module.

    `_scale_fleet` shells out to `bin/farmctl start --phase ...`, which SSHes
    into REAL hardware (the fleet is stopped but real — see briefing). Every
    test that calls step_scanning/step_fpp_backfill must never trigger a real
    farmctl invocation as a side effect of an otherwise-unrelated assertion.

    Tests that verify the HOOK WIRING (which phase gets passed, dry_run
    passthrough) assert against this mock directly (request it as a fixture
    parameter). Tests that verify `_scale_fleet`'s OWN internals re-patch
    `orch._scale_fleet` to `SCALE_FLEET_REAL` for the duration of the test and
    drive `subprocess.run` themselves.
    """
    with patch.object(orch, "_scale_fleet", return_value=(True, "")) as m:
        yield m


@pytest.fixture(autouse=True)
def _redirect_compact_remote_fail_log_dir(tmp_path, monkeypatch):
    """Every remote-compaction failure path writes a full stdout+stderr
    dump to ``_COMPACT_REMOTE_FAIL_LOG_DIR`` (2026-07-21 fix). That defaults
    to the REAL ``data/discovery/`` — never let a test write there (a live
    discovery run may be using it concurrently, per this module's own
    constraints above): redirect it to a fresh tmp_path for every test."""
    monkeypatch.setattr(orch, "_COMPACT_REMOTE_FAIL_LOG_DIR",
                         tmp_path / "compact_remote_fail_logs")


# ---------------------------------------------------------------------------
# Canonical constant guard — must not drift
# ---------------------------------------------------------------------------

# The four site-specific parts come from EXONIT_GPUQ_*; pin the whole command
# for FIXED neutral values by loading a private copy of the module under them.
_GPUQ_ENV = {
    "EXONIT_GPUQ_SSH_TARGET": "tester@gpu.example",
    "EXONIT_GPUQ_BIN": "/opt/gpuq/bin/gpuq",
    "EXONIT_GPUQ_NVML_WRAPPER": "/opt/gpuq/nvml_compat.sh",
    "EXONIT_GPUQ_LORA_SCRIPT": "/opt/gpuq/lora_via_gpuq.sh",
}
CANONICAL_LORA_CMD = (
    "ssh -o BatchMode=yes -o ConnectTimeout=8 tester@gpu.example "
    "\"bash -lc '/opt/gpuq/nvml_compat.sh /opt/gpuq/bin/gpuq run --name exonit-lora "
    "--idle-mib 1000 --max-wait 0 bash /opt/gpuq/lora_via_gpuq.sh'\""
)
CANONICAL_STATUS_CMD = (
    "ssh -o BatchMode=yes -o ConnectTimeout=8 tester@gpu.example "
    '"/opt/gpuq/bin/gpuq status --limit 50"'
)
CANONICAL_CANCEL_TEMPLATE = (
    "ssh -o BatchMode=yes -o ConnectTimeout=8 tester@gpu.example "
    '"/opt/gpuq/bin/gpuq cancel {job_id} --kill"'
)


@pytest.fixture
def orch_gpuq(monkeypatch):
    import importlib.util
    for k, v in _GPUQ_ENV.items():
        monkeypatch.setenv(k, v)
    monkeypatch.delenv("EXONIT_LORA_CMD", raising=False)
    spec = importlib.util.spec_from_file_location(
        "discovery_orchestrator_gpuq_fixture", orch.__file__)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_lora_ssh_cmd_canonical(orch_gpuq):
    """LORA_SSH_CMD must equal the exact canonical string (non-negotiable) for
    fixed EXONIT_GPUQ_* values, NVML wrapper included."""
    assert orch_gpuq.LORA_SSH_CMD == CANONICAL_LORA_CMD
    assert orch_gpuq.LORA_CANCEL_STATUS_SSH_CMD == CANONICAL_STATUS_CMD
    assert orch_gpuq.LORA_CANCEL_SSH_CMD_TEMPLATE == CANONICAL_CANCEL_TEMPLATE


def test_lora_ssh_cmd_without_wrapper_has_no_double_space(monkeypatch, orch_gpuq):
    monkeypatch.setenv("EXONIT_GPUQ_NVML_WRAPPER", "")
    import importlib.util
    spec = importlib.util.spec_from_file_location("dorch_nowrap", orch.__file__)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert "bash -lc '/opt/gpuq/bin/gpuq run" in mod.LORA_SSH_CMD


# ---------------------------------------------------------------------------
# Phase 2 (2026-08-26): LoRA command override (EXONIT_LORA_CMD /
# data/lora_cmd.override) so the LORA step can run locally on <host>
# (ops/lora_via_llamacpp.sh) instead of ssh+gpuq+vLLM on <host>.
# Phase 3 (2026-09-26): the DEFAULT changed to the Macs driver (LORA_MACS_CMD)
# and the override accepts short aliases (<host>/llamacpp/macs).
# ---------------------------------------------------------------------------

def test_resolve_lora_cmd_defaults_to_macs(monkeypatch, tmp_path):
    """No env, no override file -> the Macs driver (Phase 3 default)."""
    monkeypatch.delenv("EXONIT_LORA_CMD", raising=False)
    monkeypatch.setattr(orch, "LORA_CMD_OVERRIDE_FILE",
                        str(tmp_path / "absent.override"))
    assert orch._resolve_lora_cmd() == orch.LORA_MACS_CMD


def test_resolve_lora_cmd_env_wins(monkeypatch, tmp_path):
    """EXONIT_LORA_CMD beats both the file and the default."""
    override = tmp_path / "lora_cmd.override"
    override.write_text("bash /somewhere/from_file.sh\n")
    monkeypatch.setattr(orch, "LORA_CMD_OVERRIDE_FILE", str(override))
    monkeypatch.setenv("EXONIT_LORA_CMD", "bash /somewhere/from_env.sh")
    assert orch._resolve_lora_cmd() == "bash /somewhere/from_env.sh"


def test_resolve_lora_cmd_file_skips_comments_and_blanks(monkeypatch, tmp_path):
    """The file's first non-empty, non-comment line is the command."""
    override = tmp_path / "lora_cmd.override"
    override.write_text(
        "# fase 2: LoRA local a <host>\n"
        "\n"
        "bash /opt/exonit/ops/lora_via_llamacpp.sh\n")
    monkeypatch.setattr(orch, "LORA_CMD_OVERRIDE_FILE", str(override))
    monkeypatch.delenv("EXONIT_LORA_CMD", raising=False)
    assert (orch._resolve_lora_cmd()
            == "bash /opt/exonit/ops/lora_via_llamacpp.sh")


def test_resolve_lora_cmd_comment_only_file_falls_back(monkeypatch, tmp_path):
    """A file with only comments/blank lines is NOT an override."""
    override = tmp_path / "lora_cmd.override"
    override.write_text("# desactivat\n\n")
    monkeypatch.setattr(orch, "LORA_CMD_OVERRIDE_FILE", str(override))
    monkeypatch.delenv("EXONIT_LORA_CMD", raising=False)
    assert orch._resolve_lora_cmd() == orch.LORA_MACS_CMD


# --- Phase 3: alias table (<host> / llamacpp / macs) ---------------------

def test_resolve_lora_cmd_alias_nvidiarack_via_env(monkeypatch, tmp_path):
    monkeypatch.setattr(orch, "LORA_CMD_OVERRIDE_FILE", str(tmp_path / "absent.override"))
    monkeypatch.setenv("EXONIT_LORA_CMD", "<host>")
    assert orch._resolve_lora_cmd() == orch.LORA_SSH_CMD


def test_resolve_lora_cmd_alias_llamacpp_via_env(monkeypatch, tmp_path):
    monkeypatch.setattr(orch, "LORA_CMD_OVERRIDE_FILE", str(tmp_path / "absent.override"))
    monkeypatch.setenv("EXONIT_LORA_CMD", "llamacpp")
    assert orch._resolve_lora_cmd() == orch.LORA_LLAMACPP_CMD


def test_resolve_lora_cmd_alias_macs_via_env(monkeypatch, tmp_path):
    monkeypatch.setattr(orch, "LORA_CMD_OVERRIDE_FILE", str(tmp_path / "absent.override"))
    monkeypatch.setenv("EXONIT_LORA_CMD", "macs")
    assert orch._resolve_lora_cmd() == orch.LORA_MACS_CMD


def test_resolve_lora_cmd_alias_case_insensitive(monkeypatch, tmp_path):
    monkeypatch.setattr(orch, "LORA_CMD_OVERRIDE_FILE", str(tmp_path / "absent.override"))
    monkeypatch.setenv("EXONIT_LORA_CMD", "NvidiaRack")
    assert orch._resolve_lora_cmd() == orch.LORA_SSH_CMD


def test_resolve_lora_cmd_alias_via_override_file(monkeypatch, tmp_path):
    override = tmp_path / "lora_cmd.override"
    override.write_text("<host>\n")
    monkeypatch.setattr(orch, "LORA_CMD_OVERRIDE_FILE", str(override))
    monkeypatch.delenv("EXONIT_LORA_CMD", raising=False)
    assert orch._resolve_lora_cmd() == orch.LORA_SSH_CMD


def test_resolve_lora_cmd_unknown_alias_is_used_verbatim(monkeypatch, tmp_path):
    """Anything not in LORA_CMD_ALIASES is a literal command, unchanged
    (Phase 2 behaviour) — only the 3 known aliases are special-cased."""
    monkeypatch.setattr(orch, "LORA_CMD_OVERRIDE_FILE", str(tmp_path / "absent.override"))
    monkeypatch.setenv("EXONIT_LORA_CMD", "bash /somewhere/custom.sh")
    assert orch._resolve_lora_cmd() == "bash /somewhere/custom.sh"


# --- Phase 3: _is_nvidiarack_cmd predicate -----------------------------------

def test_is_nvidiarack_cmd_true_for_ssh_cmd():
    assert orch._is_nvidiarack_cmd(orch.LORA_SSH_CMD)


def test_is_nvidiarack_cmd_true_after_sector_rewrite():
    rewritten = orch.LORA_SSH_CMD.replace("lora_via_gpuq.sh'", "lora_via_gpuq.sh 5 50000'")
    assert orch._is_nvidiarack_cmd(rewritten)


def test_is_nvidiarack_cmd_false_for_macs_and_llamacpp():
    assert not orch._is_nvidiarack_cmd(orch.LORA_MACS_CMD)
    assert not orch._is_nvidiarack_cmd(orch.LORA_LLAMACPP_CMD)
    assert not orch._is_nvidiarack_cmd("bash /somewhere/custom.sh")


def test_lora_cmd_for_sector_nvidiarack_alias_still_gets_positionals(monkeypatch):
    """The `<host>` alias resolves to LORA_SSH_CMD exactly, so it must
    keep getting the SECTOR/BACKLOG positional rewrite like the raw
    constant does (design §3: 'the "<host>" override would not receive the
    SECTOR/BACKLOG positionals')."""
    monkeypatch.setattr(orch, "LORA_CMD", orch.LORA_CMD_ALIASES["<host>"])
    cmd = orch._lora_cmd_for_sector(5, 50000)
    assert cmd == orch.LORA_SSH_CMD.replace(
        "lora_via_gpuq.sh'", "lora_via_gpuq.sh 5 50000'")


def test_step_lora_runs_the_resolved_override(monkeypatch):
    """step_lora executes LORA_CMD (the override), not LORA_SSH_CMD, and an
    override run that times out must NOT attempt the remote gpuq cancel
    (there is no remote job to cancel).

    2026-09-26 (Phase 3): the non-<host> branch uses subprocess.Popen
    (+ start_new_session, for a killpg-capable timeout — see step_lora's
    docstring), not subprocess.run — mock Popen, never subprocess.run, for
    this branch (a real subprocess.run mock would NOT be seen by a real
    Popen'd child process, since it runs in a separate interpreter)."""
    monkeypatch.setattr(orch, "LORA_CMD", "bash /local/lora_via_llamacpp.sh")
    ran = {}

    class _FakeProc:
        pid = 4242
        returncode = 0

        def communicate(self, timeout=None):
            return "", None

    def fake_popen(cmd, **kwargs):
        ran["cmd"] = cmd
        ran["kwargs"] = kwargs
        return _FakeProc()

    monkeypatch.setattr(orch.subprocess, "Popen", fake_popen)
    ok, msg = orch.step_lora("unused_state_file", dry_run=False)
    assert ok is True
    assert ran["cmd"] == "bash /local/lora_via_llamacpp.sh"
    assert ran["kwargs"].get("start_new_session") is True


def test_step_lora_override_timeout_skips_remote_cancel(monkeypatch):
    """On TimeoutExpired with an override active: no _cancel_remote_lora_job,
    the operator message points at the LOCAL processes to clear, and the
    LOCAL PROCESS GROUP gets SIGTERM via os.killpg (killpg path, Phase 3)."""
    monkeypatch.setattr(orch, "LORA_CMD", "bash /local/lora_via_llamacpp.sh")

    class _FakeProc:
        pid = 4242

        def communicate(self, timeout=None):
            raise orch.subprocess.TimeoutExpired(cmd="x", timeout=timeout)

        def wait(self, timeout=None):
            return 0

    def fake_popen(cmd, **kwargs):
        return _FakeProc()

    killed = []
    cancelled = {"called": False}
    monkeypatch.setattr(orch.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(orch.os, "getpgid", lambda pid: pid)
    monkeypatch.setattr(orch.os, "killpg", lambda pgid, sig: killed.append((pgid, sig)))
    monkeypatch.setattr(orch, "_cancel_remote_lora_job",
                        lambda: cancelled.__setitem__("called", True))
    ok, msg = orch.step_lora("unused_state_file", dry_run=False)
    assert ok is False
    assert cancelled["called"] is False
    assert "mlx_lm.server" in msg or "llama-server" in msg
    assert "gpuq" not in msg
    assert killed == [(4242, orch.signal.SIGTERM)]


def test_step_lora_default_timeout_still_cancels_remote(monkeypatch):
    """With the default <host> command, the timeout path keeps the
    Phase 3 Task 5 behaviour: best-effort remote gpuq cancel + gpuq guidance."""
    monkeypatch.setattr(orch, "LORA_CMD", orch.LORA_SSH_CMD)

    def fake_run(cmd, **kwargs):
        raise orch.subprocess.TimeoutExpired(cmd=cmd, timeout=1)

    cancelled = {"called": False}
    monkeypatch.setattr(orch.subprocess, "run", fake_run)
    monkeypatch.setattr(orch, "_cancel_remote_lora_job",
                        lambda: cancelled.__setitem__("called", True))
    ok, msg = orch.step_lora("unused_state_file", dry_run=False)
    assert ok is False
    assert cancelled["called"] is True
    assert "gpuq" in msg


@pytest.fixture(autouse=True)
def _reset_orchestrator_db_globals():
    """Isolate the module-level PG dual-write state between tests.

    Phase 3 Task 2: run() stashes its resolved db path / dry-run gate / cached
    ExoNitDB handle as module globals on ``discovery_orchestrator`` (it is a
    standalone script with no per-instance object, and _persist_state's
    signature is fixed by the Phase-3 plan with no db-handle parameter). A
    real (non-dry-run) orch.run() call in one test would otherwise leave a
    stale db path/handle for the NEXT test that calls _set_state or
    _poll_until_drained directly. Reset before AND after every test in this
    module so ordering never matters.
    """
    orch._db_path = None
    orch._pg_dual_write_enabled = False
    orch._db_instance = None
    yield
    orch._db_path = None
    orch._pg_dual_write_enabled = False
    orch._db_instance = None


# ---------------------------------------------------------------------------
# State-file helpers
# ---------------------------------------------------------------------------

def test_save_and_load_state(tmp_path):
    """State is written atomically and can be read back."""
    sf = str(tmp_path / "state.json")
    orch._save_state(sf, {"state": "SCANNING", "lot_csv": "/tmp/lot.csv"})
    loaded = orch._load_state(sf)
    assert loaded["state"] == "SCANNING"
    assert loaded["lot_csv"] == "/tmp/lot.csv"


def test_load_state_missing_file(tmp_path):
    """Missing state file returns empty dict without raising."""
    sf = str(tmp_path / "nonexistent.json")
    assert orch._load_state(sf) == {}


def test_load_state_malformed(tmp_path):
    """Malformed JSON returns empty dict without raising."""
    sf = str(tmp_path / "bad.json")
    Path(sf).write_text("{ NOT VALID JSON <<<")
    assert orch._load_state(sf) == {}


def test_set_state_transitions(tmp_path):
    """_set_state writes the new state and merges extra fields."""
    sf = str(tmp_path / "state.json")
    orch._set_state(sf, "SCANNING", lot_csv="x.csv")
    s = orch._load_state(sf)
    assert s["state"] == "SCANNING"
    assert s["lot_csv"] == "x.csv"
    assert "updated_at" in s

    orch._set_state(sf, "FPP_BACKFILL")
    s2 = orch._load_state(sf)
    assert s2["state"] == "FPP_BACKFILL"
    # Prior fields preserved
    assert s2["lot_csv"] == "x.csv"


def test_set_state_fresh_discards_previous_run_fields(tmp_path):
    """fresh=True starts a new run from an empty state, so result fields left
    over from a previous run (finished_at/run_id/scorecard/note) never linger."""
    sf = str(tmp_path / "state.json")
    # A completed previous run, plus a stray fossil 'note'.
    orch._save_state(sf, {
        "state": "DONE",
        "finished_at": "2026-05-30T22:00:00+00:00",
        "run_id": "discovery_s0001_lotX_2026-05-30",
        "scorecard_json": "data/scorecards/old.json",
        "dossier": "data/scorecards/old_dossier.md",
        "note": "LoRA completat (18057 expl, 0 fail)",
    })

    orch._set_state(sf, "SCANNING", fresh=True, lot_csv="new.csv")
    s = orch._load_state(sf)
    assert s["state"] == "SCANNING"
    assert s["lot_csv"] == "new.csv"
    for stale in ("finished_at", "run_id", "scorecard_json", "dossier", "note"):
        assert stale not in s, f"{stale} should not survive a fresh run"

    # Without fresh=, the default merge behaviour is unchanged (no regression).
    orch._save_state(sf, {"state": "DONE", "note": "keep-me"})
    orch._set_state(sf, "SCANNING", lot_csv="y.csv")
    assert orch._load_state(sf)["note"] == "keep-me"


def test_set_state_clears_detail_for_progress_states(tmp_path):
    """Transitioning to a non-FAILED state clears any leftover 'detail' field."""
    sf = str(tmp_path / "state.json")
    # Simulate a FAILED state with a detail
    orch._set_state(sf, "FAILED", detail="some error")
    s = orch._load_state(sf)
    assert s.get("detail") == "some error"

    # Now transition to a progress state — detail must be cleared
    orch._set_state(sf, "SCANNING")
    s2 = orch._load_state(sf)
    assert s2["state"] == "SCANNING"
    assert "detail" not in s2, "detail must be cleared when entering a progress state"


def test_set_state_preserves_detail_for_failed(tmp_path):
    """Transitioning to FAILED preserves the new detail from **extra."""
    sf = str(tmp_path / "state.json")
    orch._set_state(sf, "FAILED", detail="docker hook not running")
    s = orch._load_state(sf)
    assert s["state"] == "FAILED"
    assert s.get("detail") == "docker hook not running"


def test_set_state_clears_detail_for_all_progress_states(tmp_path):
    """All named progress states clear a pre-existing 'detail'."""
    sf = str(tmp_path / "state.json")
    for state_name in ["SCANNING", "FPP_BACKFILL", "EXOMINER_BACKFILL",
                       "POSTPROCESS", "STOP_NVIDIARACK", "LORA", "FREE_GPU", "IDLE"]:
        # Prime with a detail
        orch._save_state(sf, {"state": "FAILED", "detail": "leftover"})
        orch._set_state(sf, state_name)
        s = orch._load_state(sf)
        assert "detail" not in s, (
            f"detail not cleared when transitioning to {state_name}"
        )


# ---------------------------------------------------------------------------
# Phase 3 Task 2 — run_id minted at launch + dual-write to PG + heartbeat
# ---------------------------------------------------------------------------

def test_set_state_dual_writes_pg_same_state(tmp_path):
    """_set_state writes the JSON mirror AND best-effort dual-writes the same
    state to PG via db.update_discovery_run — same call, same state."""
    sf = str(tmp_path / "state.json")
    mock_db = MagicMock()
    with patch.object(orch, "_get_orchestrator_db", return_value=mock_db):
        result = orch._set_state(sf, "SCANNING", run_id="rid-1", lot_csv="x.csv")

    assert result["state"] == "SCANNING"
    assert result["run_id"] == "rid-1"
    s = orch._load_state(sf)
    assert s["state"] == "SCANNING"
    assert s["run_id"] == "rid-1"
    # F6 (final-review fix): finished=False since SCANNING is not terminal.
    mock_db.update_discovery_run.assert_called_once_with(
        "rid-1", state="SCANNING", detail=None, heartbeat=False, finished=False)


def test_set_state_run_id_inherited_from_state_file_on_later_calls(tmp_path):
    """Only the FIRST _set_state call of a run needs run_id= explicitly — it
    persists in the JSON file, so subsequent calls (fresh=False) still
    dual-write to the SAME run_id without re-passing it."""
    sf = str(tmp_path / "state.json")
    mock_db = MagicMock()
    with patch.object(orch, "_get_orchestrator_db", return_value=mock_db):
        orch._set_state(sf, "SCANNING", fresh=True, run_id="rid-2")
        orch._set_state(sf, "FPP_BACKFILL")

    # F6 (final-review fix): finished=False since FPP_BACKFILL is not terminal.
    assert mock_db.update_discovery_run.call_args_list[-1] == call(
        "rid-2", state="FPP_BACKFILL", detail=None, heartbeat=False, finished=False)


def test_merge_state_dual_writes_pg_state_unchanged(tmp_path):
    """_merge_state must NOT change the PG row's state column (state=None)."""
    sf = str(tmp_path / "state.json")
    orch._save_state(sf, {"state": "POSTPROCESS", "run_id": "rid-3"})
    mock_db = MagicMock()
    with patch.object(orch, "_get_orchestrator_db", return_value=mock_db):
        orch._merge_state(sf, scorecard_json="a.json")

    # F6 (final-review fix): finished=False since state=None is never terminal.
    mock_db.update_discovery_run.assert_called_once_with(
        "rid-3", state=None, detail=None, heartbeat=False, finished=False)
    assert orch._load_state(sf)["scorecard_json"] == "a.json"


def test_set_state_pg_exception_json_still_written(tmp_path):
    """PG down (update_discovery_run raises): JSON is still written and
    _set_state does not raise — PG being unreachable must never fail a step."""
    sf = str(tmp_path / "state.json")
    mock_db = MagicMock()
    mock_db.update_discovery_run.side_effect = Exception("connection refused")
    with patch.object(orch, "_get_orchestrator_db", return_value=mock_db):
        result = orch._set_state(sf, "SCANNING", run_id="rid-4")

    assert result["state"] == "SCANNING"
    s = orch._load_state(sf)
    assert s["state"] == "SCANNING"
    assert s["run_id"] == "rid-4"


def test_set_state_db_handle_build_fails_json_still_written(tmp_path):
    """PG totally unreachable (can't even build the handle): same guarantee."""
    sf = str(tmp_path / "state.json")
    with patch.object(orch, "_get_orchestrator_db", side_effect=Exception("no pg")):
        result = orch._set_state(sf, "SCANNING", run_id="rid-5")

    assert result["state"] == "SCANNING"
    assert orch._load_state(sf)["state"] == "SCANNING"


def test_run_id_arg_never_calls_derive_run_id(tmp_path):
    """A server-spawned run (--run-id passed) must NEVER call _derive_run_id —
    that re-derivation (today's date, hours into a run) is the midnight bug."""
    lot_csv = str(tmp_path / "s0001_lot4.csv")
    Path(lot_csv).write_text("tic_id,label\n12345,candidate\n")
    sf = str(tmp_path / "pipeline_state.json")

    args = orch._parse_args([
        "--lot-csv", lot_csv,
        "--sector", "1",
        "--state-file", sf,
        "--db", str(tmp_path / "exonit.db"),
        "--poll-interval", "0",
        "--dry-run",
        "--run-id", "discovery_s0001_lot4_2026-07-04",
    ])

    with patch.object(orch, "_derive_run_id") as mock_derive:
        orch.run(args)

    mock_derive.assert_not_called()
    state = orch._load_state(sf)
    assert state["run_id"] == "discovery_s0001_lot4_2026-07-04"


def test_run_without_run_id_derives_once_and_persists_in_state(tmp_path):
    """Manual CLI use (no --run-id): derived once via _derive_run_id, then
    present in the state file from the first SCANNING write onward."""
    lot_csv = str(tmp_path / "s0001_lot4.csv")
    Path(lot_csv).write_text("tic_id,label\n12345,candidate\n")
    sf = str(tmp_path / "pipeline_state.json")

    args = orch._parse_args([
        "--lot-csv", lot_csv,
        "--sector", "1",
        "--state-file", sf,
        "--db", str(tmp_path / "exonit.db"),
        "--poll-interval", "0",
        "--dry-run",
    ])

    orch.run(args)

    assert args.run_id == orch._derive_run_id(1, lot_csv)
    state = orch._load_state(sf)
    assert state["run_id"] == args.run_id


def test_run_sets_run_id_in_state_from_scanning_onward(tmp_path):
    """run_id must appear in the mirror JSON from the FIRST SCANNING write —
    not only at POSTPROCESS (today's behaviour before this task)."""
    lot_csv = str(tmp_path / "s0001_lot4.csv")
    Path(lot_csv).write_text("tic_id,label\n12345,candidate\n")
    sf = str(tmp_path / "pipeline_state.json")

    seen_at_scanning = []
    original_set_state = orch._set_state

    def capturing(path, state_name, **extra):
        result = original_set_state(path, state_name, **extra)
        if state_name == "SCANNING":
            seen_at_scanning.append(result.get("run_id"))
        return result

    args = orch._parse_args([
        "--lot-csv", lot_csv,
        "--sector", "1",
        "--state-file", sf,
        "--db", str(tmp_path / "exonit.db"),
        "--poll-interval", "0",
        "--dry-run",
        "--run-id", "discovery_s0001_lot4_2026-07-04",
    ])
    with patch.object(orch, "_set_state", side_effect=capturing):
        orch.run(args)

    assert seen_at_scanning == ["discovery_s0001_lot4_2026-07-04"]


def test_poll_heartbeats_pg_throttled(tmp_path):
    """_poll_until_drained heartbeats via db.update_discovery_run(heartbeat=True)
    — throttled so a burst of polls in a short real-time window only heartbeats
    once, not once per poll."""
    sf = str(tmp_path / "state.json")
    orch._save_state(sf, {"state": "SCANNING", "run_id": "rid-hb"})

    calls = [0]

    def fake_urlopen(url, timeout=None):
        calls[0] += 1
        if calls[0] < 5:
            return _FakeResp(_health_response(primary_pending=5, primary_assigned=0))
        return _FakeResp(_health_response(primary_pending=0, primary_assigned=0))

    mock_db = MagicMock()
    with patch("urllib.request.urlopen", side_effect=fake_urlopen), \
         patch("time.sleep"), \
         patch.object(orch, "_get_orchestrator_db", return_value=mock_db):
        ok, detail = orch.step_scanning("http://localhost:8400", sf, 1, dry_run=False)

    assert ok is True
    assert mock_db.update_discovery_run.call_count == 1, (
        "heartbeat must be throttled to at most once across a fast poll burst"
    )
    # F6 (final-review fix): finished=False since state=None is never terminal.
    mock_db.update_discovery_run.assert_called_with(
        "rid-hb", state=None, detail=None, heartbeat=True, finished=False)


def test_poll_no_heartbeat_without_run_id(tmp_path):
    """When the state file has no run_id (legacy/direct-call tests), no
    heartbeat/PG call is attempted at all."""
    sf = str(tmp_path / "state.json")
    orch._save_state(sf, {"state": "SCANNING"})

    resp = _FakeResp(_health_response(primary_pending=0, primary_assigned=0))
    mock_db = MagicMock()
    with patch("urllib.request.urlopen", return_value=resp), \
         patch.object(orch, "_get_orchestrator_db", return_value=mock_db):
        ok, _ = orch.step_scanning("http://localhost:8400", sf, 1, dry_run=False)

    assert ok is True
    mock_db.update_discovery_run.assert_not_called()


def test_step_postprocess_uses_given_run_id_not_derived(tmp_path):
    """step_postprocess must use the passed-in run_id for scorecard/dossier
    paths — NOT re-derive via _derive_run_id (the midnight bug this fixes)."""
    lot_csv = str(tmp_path / "s0001_lot4.csv")
    Path(lot_csv).write_text("tic_id,label\n12345,candidate\n")
    sf = str(tmp_path / "state.json")

    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = ""
    mock_result.stderr = ""

    given_run_id = "discovery_s0001_lot4_2026-01-01"  # deliberately NOT today

    with patch("subprocess.run", return_value=mock_result) as mock_run, \
         patch.object(orch, "_derive_run_id") as mock_derive:
        ok, detail = orch.step_postprocess(
            lot_csv=lot_csv, sector=1, db=str(tmp_path / "exonit.db"),
            label="candidate", state_file=sf, dry_run=False,
            run_id=given_run_id,
        )

    assert ok is True
    mock_derive.assert_not_called()
    called_cmds = [mock_run.call_args_list[i][0][0] for i in range(mock_run.call_count)]
    scorecard_cmd = next(
        c for c in called_cmds if "scorecard.py" in " ".join(str(a) for a in c)
    )
    assert given_run_id in scorecard_cmd
    state = orch._load_state(sf)
    assert state["run_id"] == given_run_id


def test_step_postprocess_without_run_id_falls_back_to_derive(tmp_path):
    """Backward compat: a direct caller that omits run_id still works exactly
    as before (legacy fallback path)."""
    lot_csv = str(tmp_path / "s0001_lot4.csv")
    Path(lot_csv).write_text("tic_id,label\n12345,candidate\n")
    sf = str(tmp_path / "state.json")

    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = ""
    mock_result.stderr = ""

    with patch("subprocess.run", return_value=mock_result):
        ok, detail = orch.step_postprocess(
            lot_csv=lot_csv, sector=1, db=str(tmp_path / "exonit.db"),
            label="candidate", state_file=sf, dry_run=False,
        )

    assert ok is True
    expected_run_id = orch._derive_run_id(1, lot_csv)
    state = orch._load_state(sf)
    assert state["run_id"] == expected_run_id


def test_ensure_discovery_run_row_creates_if_missing(tmp_path, monkeypatch):
    """Backward-compat manual CLI path: the discovery_runs row is created if
    it doesn't already exist (upsert-if-missing)."""
    db_path = str(tmp_path / "exonit.db")
    monkeypatch.setattr(orch, "_db_path", db_path)
    monkeypatch.setattr(orch, "_pg_dual_write_enabled", True)
    monkeypatch.setattr(orch, "_db_instance", None)

    orch._ensure_discovery_run_row(
        "discovery_s0001_lot4_2026-07-04", 1, "s0001_lot4.csv",
        False, False, "http://localhost:8400")

    from exonit.db import ExoNitDB
    db = ExoNitDB(db_path)
    try:
        row = db.get_discovery_run("discovery_s0001_lot4_2026-07-04")
        assert row is not None
        assert row["state"] == "ENQUEUEING"
        assert row["sector"] == 1
    finally:
        db.close()


def test_ensure_discovery_run_row_noop_if_already_exists(tmp_path, monkeypatch):
    """Calling it twice for the SAME run_id must not raise (no PK collision) —
    covers the normal server-spawned path where the endpoint already created
    the row before --run-id ever reaches the orchestrator."""
    db_path = str(tmp_path / "exonit.db")
    monkeypatch.setattr(orch, "_db_path", db_path)
    monkeypatch.setattr(orch, "_pg_dual_write_enabled", True)
    monkeypatch.setattr(orch, "_db_instance", None)

    run_id = "discovery_s0001_lot4_2026-07-04"
    orch._ensure_discovery_run_row(run_id, 1, "s0001_lot4.csv", False, False,
                                    "http://localhost:8400")
    orch._ensure_discovery_run_row(run_id, 1, "s0001_lot4.csv", False, False,
                                    "http://localhost:8400")  # must not raise


def test_ensure_discovery_run_row_noop_under_dry_run_gate(tmp_path, monkeypatch):
    """_pg_dual_write_enabled=False (the --dry-run gate): no DB file is ever
    created, even though _db_path points at a real (nonexistent) path."""
    db_path = str(tmp_path / "exonit.db")
    monkeypatch.setattr(orch, "_db_path", db_path)
    monkeypatch.setattr(orch, "_pg_dual_write_enabled", False)
    monkeypatch.setattr(orch, "_db_instance", None)

    orch._ensure_discovery_run_row(
        "discovery_s0001_lot4_2026-07-04", 1, "s0001_lot4.csv",
        True, False, "http://localhost:8400")

    assert not Path(db_path).exists(), "dry_run gate must prevent any DB file creation"


def test_ensure_discovery_run_row_db_handle_build_fails_non_fatal(monkeypatch):
    """PG totally unreachable at connection-build time must not raise out of
    _ensure_discovery_run_row (non-fatal, same guarantee as _persist_state)."""
    with patch.object(orch, "_get_orchestrator_db", side_effect=Exception("no pg")):
        orch._ensure_discovery_run_row(  # must not raise
            "discovery_s0001_lot4_2026-07-04", 1, "s0001_lot4.csv",
            False, False, "http://localhost:8400")


def test_run_manual_cli_creates_discovery_runs_row_end_to_end(tmp_path):
    """Full run() (dry_run=False, no --run-id): the discovery_runs row exists
    afterwards, keyed by the legacy-derived run_id — the actual backward-compat
    contract, exercised end-to-end rather than by calling the helper directly."""
    lot_csv = str(tmp_path / "s0001_lot4.csv")
    Path(lot_csv).write_text("tic_id,label\n12345,candidate\n")
    sf = str(tmp_path / "pipeline_state.json")
    db_path = str(tmp_path / "exonit.db")

    args = orch._parse_args([
        "--lot-csv", lot_csv,
        "--sector", "1",
        "--state-file", sf,
        "--db", db_path,
        "--poll-interval", "0",
        # NOT --dry-run: exercises the real (non-gated) ensure-row + dual-write
        # path. Every step is stubbed so no real subprocess/ssh/HTTP happens.
    ])

    with patch.object(orch, "step_scanning", return_value=(True, "")), \
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

    from exonit.db import ExoNitDB
    db = ExoNitDB(db_path)
    try:
        row = db.get_discovery_run(args.run_id)
        assert row is not None
        assert row["run_id"] == args.run_id
        # F1 (final-review fix): run() stamps its OWN pid + heartbeat as its
        # first act, unconditionally — not just when the caller (a spawning
        # endpoint) happens to write it. Without this a chained/autofill run
        # (spawned without --run-id, so no caller-side pid write ever
        # happens) would be invisible to the dead-run watchdog forever.
        assert row["orchestrator_pid"] == os.getpid()
        assert row["heartbeat_at"] is not None
        # F6 (final-review fix): the PG row's finished_at must be stamped on
        # this DONE transition too, not just the JSON mirror's finished_at
        # field — _persist_state now derives finished=True whenever state is
        # a terminal state, so no caller has to remember to pass it.
        assert row["finished_at"] is not None
    finally:
        db.close()

    state = orch._load_state(sf)
    assert state["state"] == "DONE"
    assert state["run_id"] == args.run_id


def test_run_stamps_own_pid_and_heartbeat_on_chained_style_row(tmp_path, monkeypatch):
    """F1 (final-review fix): a row created independently of run() (mirroring
    a chained/autofill lot's discovery_runs row — created via
    _ensure_discovery_run_row / create_discovery_run before run() is invoked,
    exactly like the AWS_FINALIZE/chained callers) gets its pid + heartbeat
    stamped once run() starts, even though nothing in run()'s own argv (no
    --run-id was passed by a caller here) triggered a pid write elsewhere."""
    from exonit.db import ExoNitDB

    lot_csv = str(tmp_path / "s0001_lot9.csv")
    Path(lot_csv).write_text("tic_id,label\n99999,candidate\n")
    sf = str(tmp_path / "pipeline_state.json")
    db_path = str(tmp_path / "exonit.db")

    # Pre-create the row exactly as a chained/autofill caller would, BEFORE
    # run() ever executes — simulating the "row already exists" branch of
    # _ensure_discovery_run_row.
    run_id = orch._derive_run_id(1, lot_csv)
    db = ExoNitDB(db_path)
    db.create_discovery_run(run_id, sector=1, lot_csv=lot_csv, dry_run=False,
                            chained=True, server_url="http://localhost:8400")
    db.close()

    args = orch._parse_args([
        "--lot-csv", lot_csv,
        "--sector", "1",
        "--state-file", sf,
        "--db", db_path,
        "--poll-interval", "0",
        "--run-id", run_id,
    ])

    with patch.object(orch, "step_scanning", return_value=(True, "")), \
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

    db = ExoNitDB(db_path)
    try:
        row = db.get_discovery_run(run_id)
        assert row["orchestrator_pid"] == os.getpid()
        assert row["heartbeat_at"] is not None
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Individual step unit tests (all network/subprocess mocked)
# ---------------------------------------------------------------------------

# Helper: build a fake urllib response
class _FakeResp:
    def __init__(self, data: dict):
        self._data = json.dumps(data).encode()

    def read(self):
        return self._data

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass


def _health_response(
    primary_pending=0,
    primary_assigned=0,
    fpp_actionable=0,
    exominer_pending=0,
    exominer_assigned=0,
):
    return {
        "primary_tasks": {"pending": primary_pending, "assigned": primary_assigned},
        "fpp_backfill": {"actionable": fpp_actionable},
        "exominer_backfill": {
            "pending": exominer_pending,
            "currently_assigned": exominer_assigned,
        },
    }


# --- step_scanning ---

def test_step_scanning_dry_run(tmp_path):
    """dry_run: scanning returns (True, "") without making any HTTP call."""
    sf = str(tmp_path / "state.json")
    orch._save_state(sf, {"state": "SCANNING"})
    ok, detail = orch.step_scanning("http://localhost:8400", sf, 1, dry_run=True)
    assert ok is True
    assert detail == ""


def test_step_scanning_drains_immediately(tmp_path):
    """When primary tasks are already zero, scanning returns on first poll."""
    sf = str(tmp_path / "state.json")
    resp = _FakeResp(_health_response(primary_pending=0, primary_assigned=0))
    with patch("urllib.request.urlopen", return_value=resp):
        ok, detail = orch.step_scanning("http://localhost:8400", sf, 0, dry_run=False)
    assert ok is True
    assert detail == ""


def test_step_scanning_waits_until_drained(tmp_path):
    """Scanning retries until primary tasks drain."""
    sf = str(tmp_path / "state.json")
    calls = [0]

    def fake_urlopen(url, timeout=None):
        calls[0] += 1
        # First two calls: still pending; third: drained
        if calls[0] < 3:
            return _FakeResp(_health_response(primary_pending=5, primary_assigned=2))
        return _FakeResp(_health_response(primary_pending=0, primary_assigned=0))

    with patch("urllib.request.urlopen", side_effect=fake_urlopen), \
         patch("time.sleep"):  # skip real sleeps
        ok, detail = orch.step_scanning("http://localhost:8400", sf, 1, dry_run=False)
    assert ok is True
    assert calls[0] == 3


def test_step_scanning_tolerates_http_errors(tmp_path):
    """Transient HTTP errors are retried, not propagated."""
    import urllib.error
    sf = str(tmp_path / "state.json")
    calls = [0]

    def fake_urlopen(url, timeout=None):
        calls[0] += 1
        if calls[0] < 3:
            raise urllib.error.URLError("timeout")
        return _FakeResp(_health_response(primary_pending=0, primary_assigned=0))

    with patch("urllib.request.urlopen", side_effect=fake_urlopen), \
         patch("time.sleep"):
        ok, detail = orch.step_scanning("http://localhost:8400", sf, 1, dry_run=False)
    assert ok is True


def test_step_scanning_bails_on_persistent_connection_errors(tmp_path):
    """After POLL_CONNECT_ERROR_LIMIT consecutive errors, scanning bails (no hang)."""
    import urllib.error
    sf = str(tmp_path / "state.json")

    def always_fail(url, timeout=None):
        raise urllib.error.URLError("connection refused")

    with patch("urllib.request.urlopen", side_effect=always_fail), \
         patch("time.sleep"):
        ok, detail = orch.step_scanning("http://localhost:8400", sf, 30, dry_run=False)
    assert ok is False
    assert "unreachable" in detail.lower()


def test_step_scanning_bails_on_no_progress(tmp_path):
    """A frozen pending+assigned (orphaned tasks) bails after SCANNING_STUCK_POLLS."""
    sf = str(tmp_path / "state.json")
    call_count = [0]

    def frozen(url, timeout=None):
        call_count[0] += 1
        # pending never reaches 0, assigned stuck → no net progress, ever
        return _FakeResp(_health_response(primary_pending=0, primary_assigned=7))

    with patch("urllib.request.urlopen", side_effect=frozen), \
         patch("time.sleep"):
        ok, detail = orch.step_scanning("http://localhost:8400", sf, 30, dry_run=False)
    assert ok is False
    assert "stuck" in detail.lower()
    # baseline poll + SCANNING_STUCK_POLLS non-decreasing polls
    assert call_count[0] == orch.SCANNING_STUCK_POLLS + 1


# --- step_fpp_backfill ---

def test_step_fpp_backfill_dry_run(tmp_path):
    sf = str(tmp_path / "state.json")
    ok, detail = orch.step_fpp_backfill("http://localhost:8400", sf, 1, dry_run=True)
    assert ok is True
    assert detail == ""


def test_step_fpp_backfill_drains(tmp_path):
    sf = str(tmp_path / "state.json")
    resp = _FakeResp(_health_response(fpp_actionable=0))
    with patch("urllib.request.urlopen", return_value=resp):
        ok, detail = orch.step_fpp_backfill("http://localhost:8400", sf, 0, dry_run=False)
    assert ok is True


def test_step_fpp_backfill_exominer_nonzero_does_not_block(tmp_path):
    """ExoMiner backfill being non-zero must NOT block past FPP_BACKFILL.

    (discovery-campaign.md step 12: trigger = primary + FPP backfill a 0;
    ExoMiner Docker is left running in the background.)
    """
    sf = str(tmp_path / "state.json")
    # fpp_actionable = 0 but exominer has a huge pending count — should still drain
    resp_data = {
        "primary_tasks": {"pending": 0, "assigned": 0},
        "fpp_backfill": {"actionable": 0},
        "exominer_backfill": {"pending": 9999, "currently_assigned": 5},
    }
    resp = _FakeResp(resp_data)
    with patch("urllib.request.urlopen", return_value=resp):
        ok, detail = orch.step_fpp_backfill("http://localhost:8400", sf, 0, dry_run=False)
    assert ok is True


def test_step_fpp_backfill_bails_on_no_progress(tmp_path):
    """A frozen FPP actionable count bails after SCANNING_STUCK_POLLS (no hang)."""
    sf = str(tmp_path / "state.json")

    def frozen(url, timeout=None):
        return _FakeResp(_health_response(fpp_actionable=12))

    with patch("urllib.request.urlopen", side_effect=frozen), \
         patch("time.sleep"):
        ok, detail = orch.step_fpp_backfill("http://localhost:8400", sf, 30, dry_run=False)
    assert ok is False
    assert "stuck" in detail.lower()


def test_step_fpp_backfill_reconciles_the_stream_while_it_waits(tmp_path):
    """A stalled backfill must retry the stream reconcile, not just wait it out.

    Regression, 2026-08-01: FPP sat at actionable=7 while the fpp stream held
    queued=0 with 11 items dead-lettered — nothing could ever be served, so the
    phase was going to burn its full 68-minute stuck budget and FAIL a run that
    had otherwise finished. Reconciling only on phase entry cannot see drift
    that appears later; a human with curl was the only cure.
    """
    sf = str(tmp_path / "state.json")
    reconciles = []

    def fake_urlopen(req, timeout=None):
        url = req if isinstance(req, str) else req.full_url
        if "reconcile_stream" in url:
            reconciles.append(url)
            return _FakeResp({"stage": "fpp", "enqueued": 0})
        return _FakeResp(_health_response(fpp_actionable=7))

    with patch("urllib.request.urlopen", side_effect=fake_urlopen), \
         patch("time.sleep"):
        ok, detail = orch.step_fpp_backfill(
            "http://localhost:8400", sf, 30, dry_run=False)

    assert ok is False
    # One on entry, then one every STALL_HEAL_POLLS stalled polls.
    expected = 1 + orch.SCANNING_STUCK_POLLS // orch.STALL_HEAL_POLLS
    assert len(reconciles) == expected
    assert all("stage=fpp" in u for u in reconciles)


def test_stall_heal_does_not_run_while_the_metric_is_moving(tmp_path):
    """Healing is for a stall — a draining queue must not be reconciled."""
    sf = str(tmp_path / "state.json")
    reconciles = []
    remaining = [200 - 10 * i for i in range(25)]

    def fake_urlopen(req, timeout=None):
        url = req if isinstance(req, str) else req.full_url
        if "reconcile_stream" in url:
            reconciles.append(url)
            return _FakeResp({"stage": "fpp", "enqueued": 0})
        return _FakeResp(_health_response(
            fpp_actionable=remaining.pop(0) if remaining else 0))

    with patch("urllib.request.urlopen", side_effect=fake_urlopen), \
         patch("time.sleep"):
        ok, detail = orch.step_fpp_backfill(
            "http://localhost:8400", sf, 30, dry_run=False)

    assert ok is True
    assert len(reconciles) == 1, "only the entry reconcile should have run"


# ---------------------------------------------------------------------------
# Fleet-autoscale-by-phase
# (docs/superpowers/specs/fleet-autoscale-by-phase.md)
# ---------------------------------------------------------------------------

class TestScaleFleetReal:
    """Unit tests of the REAL `_scale_fleet` helper (SCALE_FLEET_REAL is
    captured at collection time, before the module's autouse `mock_scale_fleet`
    fixture patches `orch._scale_fleet` for everything else in this file).
    """

    def test_calls_farmctl_start_with_phase(self, monkeypatch):
        monkeypatch.delenv("EXONIT_FLEET_AUTOSCALE", raising=False)
        monkeypatch.delenv("EXONIT_DEPLOYMENT", raising=False)
        mock_result = MagicMock(returncode=0, stdout="", stderr="")
        with patch("subprocess.run", return_value=mock_result) as mock_run:
            ok, detail = SCALE_FLEET_REAL("fpp", False)
        assert ok is True
        assert detail == ""
        cmd = mock_run.call_args[0][0]
        assert cmd == [orch.FARMCTL_BIN, "start", "--all", "--phase", "fpp"]

    def test_calls_farmctl_start_with_scan_phase(self, monkeypatch):
        monkeypatch.delenv("EXONIT_FLEET_AUTOSCALE", raising=False)
        monkeypatch.delenv("EXONIT_DEPLOYMENT", raising=False)
        mock_result = MagicMock(returncode=0, stdout="", stderr="")
        with patch("subprocess.run", return_value=mock_result) as mock_run:
            SCALE_FLEET_REAL("scan", False)
        cmd = mock_run.call_args[0][0]
        assert cmd == [orch.FARMCTL_BIN, "start", "--all", "--phase", "scan"]

    def test_dry_run_skips_subprocess(self, monkeypatch):
        monkeypatch.delenv("EXONIT_FLEET_AUTOSCALE", raising=False)
        monkeypatch.delenv("EXONIT_DEPLOYMENT", raising=False)
        with patch("subprocess.run") as mock_run:
            ok, detail = SCALE_FLEET_REAL("scan", True)
        assert ok is True
        mock_run.assert_not_called()

    def test_kill_switch_off_skips_farmctl(self, monkeypatch):
        """EXONIT_FLEET_AUTOSCALE=0 → no call to farmctl."""
        monkeypatch.setenv("EXONIT_FLEET_AUTOSCALE", "0")
        with patch("subprocess.run") as mock_run:
            ok, detail = SCALE_FLEET_REAL("scan", False)
        assert ok is True
        mock_run.assert_not_called()
        assert "EXONIT_FLEET_AUTOSCALE" in detail

    def test_kill_switch_default_is_on(self, monkeypatch):
        """Absent EXONIT_FLEET_AUTOSCALE (unset) must NOT skip — default ON."""
        monkeypatch.delenv("EXONIT_FLEET_AUTOSCALE", raising=False)
        monkeypatch.delenv("EXONIT_DEPLOYMENT", raising=False)
        mock_result = MagicMock(returncode=0, stdout="", stderr="")
        with patch("subprocess.run", return_value=mock_result) as mock_run:
            SCALE_FLEET_REAL("scan", False)
        mock_run.assert_called_once()

    def test_aws_deployment_skips_farmctl(self, monkeypatch):
        """Run AWS → no call to farmctl."""
        monkeypatch.delenv("EXONIT_FLEET_AUTOSCALE", raising=False)
        monkeypatch.setenv("EXONIT_DEPLOYMENT", "aws")
        with patch("subprocess.run") as mock_run:
            ok, detail = SCALE_FLEET_REAL("fpp", False)
        assert ok is True
        mock_run.assert_not_called()
        assert "aws" in detail.lower()

    def test_farmctl_nonzero_is_non_fatal(self, monkeypatch):
        """farmctl start with an error code → the step continues (fail-soft)."""
        monkeypatch.delenv("EXONIT_FLEET_AUTOSCALE", raising=False)
        monkeypatch.delenv("EXONIT_DEPLOYMENT", raising=False)
        mock_result = MagicMock(returncode=1, stdout="", stderr="worker-HP unreachable")
        with patch("subprocess.run", return_value=mock_result):
            ok, detail = SCALE_FLEET_REAL("scan", False)
        assert ok is True  # fail-soft: never fails the caller
        assert "(non-fatal)" in detail

    def test_farmctl_exception_is_non_fatal(self, monkeypatch):
        """A farmctl exception (e.g. timeout) must NOT fail the caller."""
        monkeypatch.delenv("EXONIT_FLEET_AUTOSCALE", raising=False)
        monkeypatch.delenv("EXONIT_DEPLOYMENT", raising=False)
        with patch("subprocess.run", side_effect=TimeoutError("farmctl timed out")):
            ok, detail = SCALE_FLEET_REAL("fpp", False)
        assert ok is True
        assert "(non-fatal)" in detail

    def test_uses_fleet_autoscale_timeout_constant(self, monkeypatch):
        monkeypatch.delenv("EXONIT_FLEET_AUTOSCALE", raising=False)
        monkeypatch.delenv("EXONIT_DEPLOYMENT", raising=False)
        mock_result = MagicMock(returncode=0, stdout="", stderr="")
        with patch("subprocess.run", return_value=mock_result) as mock_run:
            SCALE_FLEET_REAL("scan", False)
        assert mock_run.call_args.kwargs["timeout"] == orch.FLEET_AUTOSCALE_TIMEOUT_SECONDS

    def test_cmd_includes_all_flag(self, monkeypatch):
        """Post-incident regression (2026-07-13): the constructed argv MUST
        include --all. Without it, bin/farmctl's select_workers([], want_all=
        False) calls sys.exit("specify worker names or use --all") -> rc=1 ->
        the fail-soft path here silently swallows it as a "(non-fatal)"
        warning and the run proceeds with an EMPTY fleet at every phase. This
        assertion alone would NOT have caught the original bug (it only
        inspects the mocked argv, same blind spot as every other test in this
        class) — see TestFarmctlCallSitesAcceptedByRealParser below for the
        test that actually would have caught it (real argparse + select_workers,
        no mocked subprocess.run to hide behind).
        """
        monkeypatch.delenv("EXONIT_FLEET_AUTOSCALE", raising=False)
        monkeypatch.delenv("EXONIT_DEPLOYMENT", raising=False)
        mock_result = MagicMock(returncode=0, stdout="", stderr="")
        with patch("subprocess.run", return_value=mock_result) as mock_run:
            SCALE_FLEET_REAL("scan", False)
        cmd = mock_run.call_args[0][0]
        assert "--all" in cmd


class TestFarmctlCallSitesAcceptedByRealParser:
    """Regression for the 2026-07-13 incident: `_scale_fleet` built
    `farmctl start --phase <phase>` WITHOUT `--all`. Every unit test above
    mocks `subprocess.run` and only inspects the argv list it was called
    with — none of them actually fed that argv through farmctl's REAL
    argparse + `select_workers`, so the missing `--all` (which makes
    `select_workers` call `sys.exit("specify worker names or use --all")`,
    rc=1, silently turned into a "(non-fatal)" warning by the fail-soft path)
    was invisible to the whole suite until executed for real.

    These tests close that gap: they capture the EXACT argv each
    discovery_orchestrator farmctl call site constructs (subprocess.run
    mocked only at the point of capture), then feed that argv through
    farmctl's REAL `main()` (loaded via SourceFileLoader, exactly like
    tests/test_multiworker.py) with `agent_call` monkeypatched so NO SSH
    ever happens. If `--all` (or an explicit worker name) is ever dropped
    again, `select_workers` calls `sys.exit(...)`, `main()` propagates that
    SystemExit, and these tests fail loudly instead of silently no-op'ing in
    production.
    """

    def _load_farmctl(self):
        import importlib.util
        from importlib.machinery import SourceFileLoader
        farmctl_path = str(Path(__file__).resolve().parent.parent
                           / "bin" / "farmctl")
        loader = SourceFileLoader("farmctl_mod_realparser", farmctl_path)
        spec = importlib.util.spec_from_loader("farmctl_mod_realparser", loader)
        mod = importlib.util.module_from_spec(spec)
        loader.exec_module(mod)
        return mod

    def _capture_cmd(self, fn) -> list[str]:
        """Call `fn` (a zero-arg closure around the real orchestrator step)
        with subprocess.run mocked, and return the argv it constructed minus
        the interpreter/binary path (argv[0])."""
        mock_result = MagicMock(returncode=0, stdout="", stderr="")
        with patch("subprocess.run", return_value=mock_result) as mock_run:
            fn()
        cmd = mock_run.call_args[0][0]
        assert cmd[0] == orch.FARMCTL_BIN
        return cmd[1:]

    def _run_through_real_farmctl(self, argv: list[str]):
        """Run `argv` through farmctl's REAL main()/select_workers/cmd_*,
        against the LIVE ops/workers.json, with agent_call monkeypatched so
        no SSH ever happens. Returns (rc, called_worker_names).

        EXONIT_FARMCTL_ALLOW_START is set here because that is precisely what
        the caller under test does: _scale_fleet passes it in the subprocess
        env (bin/farmctl START_GUARD_ENV — the 2026-07-14 guard against agents
        starting the real fleet). Without it farmctl correctly refuses with
        rc=2, so simulating the real caller means simulating its env too. The
        guard itself is covered by TestFarmctlStartGuard in
        tests/test_multiworker.py, and the assertion below pins that
        _scale_fleet keeps opting in — if someone drops the env, THIS fails.

        EXONIT_FARMCTL_ALLOW_UNDER_PYTEST is set for the separate,
        pytest-only guard added 2026-09-26 (incident: a test elsewhere in
        this suite reached scripts/lora_macs_driver.py for real, which
        called a real `farmctl stop` on the production fleet). This class
        runs bin/farmctl's REAL `main()`/`cmd_start`/`cmd_stop` in-process —
        by design, that's the whole point (regression coverage for the
        argparse/select_workers wiring) — but with `agent_call` fully
        replaced by `_fake_agent_call` above, which is the ONLY place
        farmctl's start/stop path performs SSH (verified by reading
        bin/farmctl: `requests.get` only exists in `cmd_status`, never
        called here), so no real socket is ever opened despite running
        against the live `ops/workers.json` manifest."""
        fc = self._load_farmctl()
        called = []

        def _fake_agent_call(w, *a, **kw):
            called.append(w["name"])
            return (0, '{"status": "ok"}', "")

        with patch.dict(os.environ, {"EXONIT_FARMCTL_ALLOW_START": "1",
                                     "EXONIT_FARMCTL_ALLOW_UNDER_PYTEST": "1"}), \
                patch.object(fc, "agent_call", side_effect=_fake_agent_call):
            rc = fc.main(argv)
        return rc, called

    def test_scale_fleet_opts_into_the_start_guard(self, monkeypatch):
        """_scale_fleet must pass EXONIT_FARMCTL_ALLOW_START=1 to farmctl.

        Without it, farmctl refuses (rc=2) and the fail-soft path swallows it
        as a "(non-fatal)" warning — the run would proceed with an EMPTY fleet,
        exactly the silent no-op class of bug the guard's own regression test
        exists to prevent.

        monkeypatch.delenv (not a bare call): tests/conftest.py forces
        EXONIT_FLEET_AUTOSCALE=0 globally (incident 2026-08-01 fleet-start
        fix) so this test needs to explicitly opt back into the "kill-switch
        off" default-ON path it's actually exercising, same as every other
        test in this class."""
        monkeypatch.delenv("EXONIT_FLEET_AUTOSCALE", raising=False)
        monkeypatch.delenv("EXONIT_DEPLOYMENT", raising=False)
        mock_result = MagicMock(returncode=0, stdout="", stderr="")
        with patch("subprocess.run", return_value=mock_result) as mock_run:
            SCALE_FLEET_REAL("scan", False)
        env = mock_run.call_args.kwargs.get("env") or {}
        assert env.get("EXONIT_FARMCTL_ALLOW_START") == "1", (
            "_scale_fleet no longer opts into the farmctl start guard — "
            "farmctl will refuse and the fleet will never start")

    def _all_enabled_worker_names(self, fc) -> set:
        manifest = fc.load_manifest()
        return {w["name"] for w in manifest["workers"] if w.get("instances", 1) != 0}

    def test_scale_fleet_scan_argv_reaches_every_enabled_worker(self, monkeypatch):
        monkeypatch.delenv("EXONIT_FLEET_AUTOSCALE", raising=False)
        monkeypatch.delenv("EXONIT_DEPLOYMENT", raising=False)
        argv = self._capture_cmd(lambda: SCALE_FLEET_REAL("scan", False))
        rc, called = self._run_through_real_farmctl(argv)
        assert rc == 0
        expected = self._all_enabled_worker_names(self._load_farmctl())
        assert set(called) == expected, (
            f"select_workers did not reach every enabled worker — likely a "
            f"missing --all (or worker name) regression. called={called!r}"
        )

    def test_scale_fleet_fpp_argv_reaches_every_enabled_worker(self, monkeypatch):
        monkeypatch.delenv("EXONIT_FLEET_AUTOSCALE", raising=False)
        monkeypatch.delenv("EXONIT_DEPLOYMENT", raising=False)
        argv = self._capture_cmd(lambda: SCALE_FLEET_REAL("fpp", False))
        rc, called = self._run_through_real_farmctl(argv)
        assert rc == 0
        expected = self._all_enabled_worker_names(self._load_farmctl())
        assert set(called) == expected

    def test_step_stop_workers_argv_reaches_every_enabled_worker(self):
        argv = self._capture_cmd(
            lambda: orch.step_stop_workers(state_file="unused", dry_run=False))
        rc, called = self._run_through_real_farmctl(argv)
        assert rc == 0
        expected = self._all_enabled_worker_names(self._load_farmctl())
        assert set(called) == expected

    def test_step_stop_nvidiarack_argv_reaches_named_worker(self):
        """`agent_call`'s first positional arg is always the per-HOST dict
        (constant across every instance of that host — see cmd_stop), so
        `called` legitimately contains "worker-<host>" once per instance
        _iter_instances_all_phases expands to (19, the fpp-phase count) —
        the per-instance NAME expansion itself is covered separately by
        TestIterInstancesAllPhasesUnion in test_multiworker.py. What THIS
        test verifies is narrower and is exactly the incident's blast
        radius: select_workers(["worker-<host>"], False) must resolve to
        that ONE host and nothing else — not sys.exit, not every host."""
        argv = self._capture_cmd(
            lambda: orch.step_stop_nvidiarack(state_file="unused", dry_run=False))
        rc, called = self._run_through_real_farmctl(argv)
        assert rc == 0
        assert set(called) == {"worker-<host>"}

    def test_missing_all_flag_would_have_been_caught(self):
        """Sanity check that this harness actually detects the ORIGINAL bug
        shape (argv without --all) — proves these tests are not vacuously
        green. Must raise SystemExit (bin/farmctl's select_workers exits
        rc=1 with 'specify worker names or use --all')."""
        buggy_argv = ["start", "--phase", "scan"]  # the pre-fix argv, no --all
        with pytest.raises(SystemExit):
            self._run_through_real_farmctl(buggy_argv)


class TestFleetAutoscaleHooks:
    """step_scanning / step_fpp_backfill must call `_scale_fleet` with the
    right phase and pass their own `dry_run` through unchanged. Uses the
    module-level autoused `mock_scale_fleet` fixture (requested by name so
    the test can assert against it).
    """

    def test_step_scanning_calls_scale_fleet_scan_phase(self, tmp_path, mock_scale_fleet):
        sf = str(tmp_path / "state.json")
        orch.step_scanning("http://localhost:8400", sf, 1, dry_run=True)
        mock_scale_fleet.assert_called_once_with("scan", True)

    def test_step_fpp_backfill_calls_scale_fleet_fpp_phase(self, tmp_path, mock_scale_fleet):
        sf = str(tmp_path / "state.json")
        orch.step_fpp_backfill("http://localhost:8400", sf, 1, dry_run=True)
        mock_scale_fleet.assert_called_once_with("fpp", True)

    def test_step_scanning_dry_run_false_passthrough(self, tmp_path, mock_scale_fleet):
        sf = str(tmp_path / "state.json")
        resp = _FakeResp(_health_response(primary_pending=0, primary_assigned=0))
        with patch("urllib.request.urlopen", return_value=resp):
            orch.step_scanning("http://localhost:8400", sf, 0, dry_run=False)
        mock_scale_fleet.assert_called_once_with("scan", False)

    def test_step_fpp_backfill_dry_run_false_passthrough(self, tmp_path, mock_scale_fleet):
        sf = str(tmp_path / "state.json")
        resp = _FakeResp(_health_response(fpp_actionable=0))
        with patch("urllib.request.urlopen", return_value=resp):
            orch.step_fpp_backfill("http://localhost:8400", sf, 0, dry_run=False)
        mock_scale_fleet.assert_called_once_with("fpp", False)

    def test_step_scanning_scale_fleet_failure_is_non_fatal_and_recorded(self, tmp_path, monkeypatch):
        """Integration: a real farmctl failure inside the hook must NOT fail
        the step (fail-soft) and must be recorded into the state file (spec:
        'record it in the state')."""
        monkeypatch.delenv("EXONIT_FLEET_AUTOSCALE", raising=False)
        monkeypatch.delenv("EXONIT_DEPLOYMENT", raising=False)
        sf = str(tmp_path / "state.json")
        orch._save_state(sf, {"state": "SCANNING"})
        fail_result = MagicMock(returncode=1, stdout="", stderr="worker-HP unreachable")
        resp = _FakeResp(_health_response(primary_pending=0, primary_assigned=0))
        with patch.object(orch, "_scale_fleet", SCALE_FLEET_REAL), \
             patch("subprocess.run", return_value=fail_result), \
             patch("urllib.request.urlopen", return_value=resp):
            ok, detail = orch.step_scanning("http://localhost:8400", sf, 0, dry_run=False)
        assert ok is True  # scanning itself still succeeds — fail-soft
        state = json.loads(Path(sf).read_text())
        assert "(non-fatal)" in state.get("fleet_autoscale_warning", "")

    def test_step_scanning_scale_fleet_success_does_not_pollute_state(self, tmp_path, monkeypatch):
        """A successful (or skipped) fleet scale must NOT write a spurious
        fleet_autoscale_warning into the state file."""
        monkeypatch.delenv("EXONIT_FLEET_AUTOSCALE", raising=False)
        monkeypatch.delenv("EXONIT_DEPLOYMENT", raising=False)
        sf = str(tmp_path / "state.json")
        orch._save_state(sf, {"state": "SCANNING"})
        ok_result = MagicMock(returncode=0, stdout="", stderr="")
        resp = _FakeResp(_health_response(primary_pending=0, primary_assigned=0))
        with patch.object(orch, "_scale_fleet", SCALE_FLEET_REAL), \
             patch("subprocess.run", return_value=ok_result), \
             patch("urllib.request.urlopen", return_value=resp):
            orch.step_scanning("http://localhost:8400", sf, 0, dry_run=False)
        state = json.loads(Path(sf).read_text())
        assert "fleet_autoscale_warning" not in state


# --- step_stop_workers ---

def test_step_stop_workers_dry_run():
    ok, detail = orch.step_stop_workers(state_file="unused", dry_run=True)
    assert ok is True
    assert detail == ""


def test_step_stop_workers_success():
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = "stopped 6 workers"
    mock_result.stderr = ""
    with patch("subprocess.run", return_value=mock_result) as mock_run:
        ok, detail = orch.step_stop_workers(state_file="unused", dry_run=False)
    assert ok is True
    # Must invoke farmctl stop --all
    cmd = mock_run.call_args[0][0]
    assert cmd[-2:] == ["stop", "--all"]


def test_step_stop_workers_nonzero_is_non_fatal():
    """An offline worker (farmctl rc!=0) must NOT fail the run — returns True."""
    mock_result = MagicMock()
    mock_result.returncode = 1
    mock_result.stdout = ""
    mock_result.stderr = "worker-HP unreachable"
    with patch("subprocess.run", return_value=mock_result):
        ok, detail = orch.step_stop_workers(state_file="unused", dry_run=False)
    assert ok is True  # non-fatal
    assert "non-fatal" in detail


def test_step_stop_workers_exception_is_non_fatal():
    """A farmctl exception (e.g. timeout) must NOT fail the run."""
    with patch("subprocess.run", side_effect=TimeoutError("farmctl timed out")):
        ok, detail = orch.step_stop_workers(state_file="unused", dry_run=False)
    assert ok is True
    assert "non-fatal" in detail


# --- step_exominer_backfill ---

def test_step_exominer_backfill_dry_run(tmp_path):
    sf = str(tmp_path / "state.json")
    ok, detail = orch.step_exominer_backfill("http://localhost:8400", sf, 1, dry_run=True)
    assert ok is True
    assert detail == ""


def test_step_exominer_backfill_skipped_when_disabled(tmp_path, monkeypatch):
    """When EXONIT_EXOMINER_BACKFILL!=1, the phase is skipped without any polling.

    This is the expected state for home FFI-discovery runs started with the
    stopgap flag (EXONIT_EXOMINER_BACKFILL=0).  The queue will never drain
    (Docker hook is not running), so the orchestrator must short-circuit here
    and let postprocess/scorecard/LoRA proceed.  exominer_ffi_score (written
    always-on by the worker during scan) is the authoritative score, same
    rationale as the AWS path.
    """
    sf = str(tmp_path / "state.json")

    # Ensure the env var is absent/0 — either case must skip
    for val in (None, "0"):
        if val is None:
            monkeypatch.delenv("EXONIT_EXOMINER_BACKFILL", raising=False)
        else:
            monkeypatch.setenv("EXONIT_EXOMINER_BACKFILL", val)

        # urlopen must NOT be called — if it is, the test fails because there is
        # no real server.  We use a side_effect that raises to make that loud.
        def _should_not_poll(url, timeout=None):
            raise AssertionError(
                "step_exominer_backfill must not poll the server when backfill is disabled"
            )

        with patch("urllib.request.urlopen", side_effect=_should_not_poll):
            ok, detail = orch.step_exominer_backfill(
                "http://localhost:8400", sf, 30, dry_run=False
            )

        assert ok is True, f"Expected (True, ...) for EXONIT_EXOMINER_BACKFILL={val!r}, got ok={ok}"
        assert "skipped" in detail
        assert "EXONIT_EXOMINER_BACKFILL" in detail


def test_step_exominer_backfill_drains_immediately(tmp_path, monkeypatch):
    """When exominer_backfill.pending == 0 on first poll, returns success."""
    monkeypatch.setenv("EXONIT_EXOMINER_BACKFILL", "1")
    sf = str(tmp_path / "state.json")
    resp = _FakeResp(_health_response(exominer_pending=0))
    with patch("urllib.request.urlopen", return_value=resp):
        ok, detail = orch.step_exominer_backfill(
            "http://localhost:8400", sf, 0, dry_run=False
        )
    assert ok is True
    assert detail == ""


def test_step_exominer_backfill_drains_after_several_polls(tmp_path, monkeypatch):
    """ExoMiner backfill drains after pending decreases to 0."""
    monkeypatch.setenv("EXONIT_EXOMINER_BACKFILL", "1")
    sf = str(tmp_path / "state.json")
    calls = [0]

    def fake_urlopen(url, timeout=None):
        calls[0] += 1
        # First call: pending=10, second: pending=5, third: drained
        if calls[0] == 1:
            return _FakeResp(_health_response(exominer_pending=10))
        if calls[0] == 2:
            return _FakeResp(_health_response(exominer_pending=5))
        return _FakeResp(_health_response(exominer_pending=0))

    with patch("urllib.request.urlopen", side_effect=fake_urlopen), \
         patch("time.sleep"):
        ok, detail = orch.step_exominer_backfill(
            "http://localhost:8400", sf, 1, dry_run=False
        )
    assert ok is True
    assert detail == ""
    assert calls[0] == 3


def test_step_exominer_backfill_stuck_detection(tmp_path, monkeypatch):
    """After EXOMINER_STUCK_POLLS consecutive non-decreasing polls, returns (False, detail)."""
    monkeypatch.setenv("EXONIT_EXOMINER_BACKFILL", "1")
    sf = str(tmp_path / "state.json")

    # Always return pending=42 (never decreases → stuck)
    resp = _FakeResp(_health_response(exominer_pending=42))

    call_count = [0]
    def fake_urlopen(req, timeout=None):
        # Count POLLS only. The phase also POSTs /api/admin/reconcile_stream
        # once on entry (2026-07-14) — a different URL, and not a poll: it is
        # what stops the phase from waiting on work that is in the DB but in no
        # queue. Counting it here would make the poll budget below drift by one
        # every time a new pre-phase call is added.
        url = req if isinstance(req, str) else req.full_url
        if "health/work" in url:
            call_count[0] += 1
        return _FakeResp(_health_response(exominer_pending=42))

    with patch("urllib.request.urlopen", side_effect=fake_urlopen), \
         patch("time.sleep"):
        ok, detail = orch.step_exominer_backfill(
            "http://localhost:8400", sf, 30, dry_run=False
        )

    assert ok is False
    assert "EXOMINER_BACKFILL stuck" in detail
    assert "pending=42" in detail
    assert "docker ps" in detail.lower() or "docker" in detail
    # Must have fired after exactly EXOMINER_STUCK_POLLS non-decreasing polls
    # (first poll sets the baseline, then STUCK_POLLS more non-decreasing ones)
    assert call_count[0] == orch.EXOMINER_STUCK_POLLS + 1


def test_step_exominer_backfill_waits_on_actionable_not_pending(tmp_path, monkeypatch):
    """2026-07-14: the phase must drain on 'actionable', never on 'pending'.

    A candidate whose FITS metadata is not cached is held back by the serve
    loop on purpose (no serve, no ack), so it can NEVER leave 'pending'. A run
    that waits for pending==0 therefore waits forever — that is exactly how the
    2026-07-14 sector-14 run hung on 43 unscoreable sector-2 leftovers.
    """
    monkeypatch.setenv("EXONIT_EXOMINER_BACKFILL", "1")
    sf = str(tmp_path / "state.json")

    # Nothing left that a worker can be served (actionable=0), but 43 rows are
    # still 'pending' forever. The phase must consider itself drained.
    payload = _health_response(exominer_pending=43)
    payload["exominer_backfill"]["actionable"] = 0
    payload["exominer_backfill"]["unrecoverable"] = 43

    with patch("urllib.request.urlopen",
               side_effect=lambda url, timeout=None: _FakeResp(payload)), \
         patch("time.sleep"):
        ok, detail = orch.step_exominer_backfill(
            "http://localhost:8400", sf, 30, dry_run=False)

    assert ok is True
    assert detail == ""


def test_step_exominer_backfill_still_waits_when_work_is_servable(tmp_path, monkeypatch):
    """The converse: actionable>0 must NOT be mistaken for drained."""
    monkeypatch.setenv("EXONIT_EXOMINER_BACKFILL", "1")
    sf = str(tmp_path / "state.json")

    payload = _health_response(exominer_pending=5)
    payload["exominer_backfill"]["actionable"] = 5
    payload["exominer_backfill"]["unrecoverable"] = 0

    with patch("urllib.request.urlopen",
               side_effect=lambda url, timeout=None: _FakeResp(payload)), \
         patch("time.sleep"):
        ok, detail = orch.step_exominer_backfill(
            "http://localhost:8400", sf, 30, dry_run=False)

    assert ok is False                      # never drained -> stuck, not done
    assert "actionable=5" in detail


def test_health_work_url_carries_run_id(tmp_path):
    """The phase polls must ask for THEIR OWN lot's counters, not the DB's."""
    assert orch._health_work_url("http://s:8400", "run_A") == (
        "http://s:8400/api/health/work?nocache=1&run_id=run_A")
    # No run_id (manual/legacy invocation) -> the original global URL.
    assert orch._health_work_url("http://s:8400", None) == (
        "http://s:8400/api/health/work?nocache=1")


def test_step_exominer_backfill_resets_stuck_counter_on_decrease(tmp_path, monkeypatch):
    """Stuck counter resets when pending drops to a new minimum."""
    monkeypatch.setenv("EXONIT_EXOMINER_BACKFILL", "1")
    sf = str(tmp_path / "state.json")

    # Sequence: 10, 10 (stuck_count=1), 10 (stuck_count=2), then 5 (reset),
    # then 5 repeated until STUCK_POLLS, then 0 (drains).
    # We just verify that a decrease prevents premature stuck detection.
    call_count = [0]

    def fake_urlopen(url, timeout=None):
        call_count[0] += 1
        # Stay at 10 for the first (STUCK_POLLS - 1) polls, then drop to 5, then drain
        if call_count[0] <= orch.EXOMINER_STUCK_POLLS - 1:
            return _FakeResp(_health_response(exominer_pending=10))
        if call_count[0] == orch.EXOMINER_STUCK_POLLS:
            # Decrease resets counter — not yet stuck
            return _FakeResp(_health_response(exominer_pending=5))
        return _FakeResp(_health_response(exominer_pending=0))

    with patch("urllib.request.urlopen", side_effect=fake_urlopen), \
         patch("time.sleep"):
        ok, detail = orch.step_exominer_backfill(
            "http://localhost:8400", sf, 30, dry_run=False
        )

    assert ok is True
    assert detail == ""


def test_step_exominer_backfill_tolerates_http_errors(tmp_path, monkeypatch):
    """Transient HTTP errors during ExoMiner polling are retried, not fatal."""
    monkeypatch.setenv("EXONIT_EXOMINER_BACKFILL", "1")
    import urllib.error
    sf = str(tmp_path / "state.json")
    calls = [0]

    def fake_urlopen(url, timeout=None):
        calls[0] += 1
        if calls[0] < 3:
            raise urllib.error.URLError("timeout")
        return _FakeResp(_health_response(exominer_pending=0))

    with patch("urllib.request.urlopen", side_effect=fake_urlopen), \
         patch("time.sleep"):
        ok, detail = orch.step_exominer_backfill(
            "http://localhost:8400", sf, 1, dry_run=False
        )
    assert ok is True


# --- step_postprocess ---

def test_step_postprocess_dry_run(tmp_path):
    """dry_run: no subprocess is spawned."""
    lot_csv = str(tmp_path / "lot4.csv")
    Path(lot_csv).write_text("tic_id,label\n12345,candidate\n")
    sf = str(tmp_path / "state.json")
    ok, detail = orch.step_postprocess(
        lot_csv=lot_csv,
        sector=1,
        db=str(tmp_path / "exonit.db"),
        label="candidate",
        state_file=sf,
        dry_run=True,
    )
    assert ok is True
    assert detail == ""


def test_step_postprocess_subprocess_success(tmp_path):
    """Exactly two subprocesses are called (postprocess + scorecard); augment is NOT."""
    lot_csv = str(tmp_path / "s0001_lot4.csv")
    Path(lot_csv).write_text("tic_id,label\n12345,candidate\n")
    sf = str(tmp_path / "state.json")

    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = ""
    mock_result.stderr = ""

    with patch("subprocess.run", return_value=mock_result) as mock_run:
        ok, detail = orch.step_postprocess(
            lot_csv=lot_csv,
            sector=1,
            db=str(tmp_path / "exonit.db"),
            label="candidate",
            state_file=sf,
            dry_run=False,
        )

    assert ok is True
    assert detail == ""
    # Must have called exactly two subprocesses: discovery_postprocess + scorecard.
    # augment_scorecard_exominer.py is intentionally excluded (requires TOI ground-truth
    # columns absent from discovery lot CSVs — would raise KeyError: 'disposition').
    assert mock_run.call_count == 2
    called_cmds = [mock_run.call_args_list[i][0][0] for i in range(mock_run.call_count)]
    flat_cmds = [" ".join(str(a) for a in cmd) for cmd in called_cmds]
    assert any("discovery_postprocess.py" in c for c in flat_cmds), (
        "Expected discovery_postprocess.py to be called"
    )
    assert any("scorecard.py" in c for c in flat_cmds), (
        "Expected scorecard.py to be called"
    )
    assert not any("augment_scorecard_exominer.py" in c for c in flat_cmds), (
        "augment_scorecard_exominer.py must NOT be called from discovery orchestrator"
    )


def test_step_postprocess_subprocess_failure(tmp_path):
    """A non-zero returncode on any subprocess → (False, detail)."""
    lot_csv = str(tmp_path / "s0001_lot4.csv")
    Path(lot_csv).write_text("tic_id,label\n12345,candidate\n")
    sf = str(tmp_path / "state.json")

    fail = MagicMock()
    fail.returncode = 1
    fail.stdout = "some stdout"
    fail.stderr = "some error"

    with patch("subprocess.run", return_value=fail):
        ok, detail = orch.step_postprocess(
            lot_csv=lot_csv,
            sector=1,
            db=str(tmp_path / "exonit.db"),
            label="candidate",
            state_file=sf,
            dry_run=False,
        )

    assert ok is False
    assert "rc=1" in detail


# ---------------------------------------------------------------------------
# Phase 4a Task 4: _finalize_compact_completeness (the FINALIZE-step hook that
# drains completeness_staging -> Parquet+PG via scripts/compact_completeness.py)
# + step_wal_checkpoint's flag-ON no-op guard. Pure unit tests: subprocess.run
# is always mocked, no real PG/psycopg/network involved.
# ---------------------------------------------------------------------------

_FAKE_PG_DSN = "postgresql://fake@localhost/fake_exonit"


def test_finalize_compact_completeness_flag_off_does_not_call_subprocess(monkeypatch):
    monkeypatch.delenv("EXONIT_COMPLETENESS_PG", raising=False)
    with patch("subprocess.run") as mock_run:
        orch._finalize_compact_completeness(_FAKE_PG_DSN)
    mock_run.assert_not_called()


def test_finalize_compact_completeness_flag_on_but_sqlite_db_does_not_call_subprocess(
        monkeypatch, tmp_path):
    """The cutover flag is only meaningful with a PG main backend (see
    exonit/db.py's ExoNitDB.__init__ docstring) — a SQLite db_path must never
    spawn the compaction subprocess even with the flag ON."""
    monkeypatch.setenv("EXONIT_COMPLETENESS_PG", "1")
    with patch("subprocess.run") as mock_run:
        orch._finalize_compact_completeness(str(tmp_path / "exonit.db"))
    mock_run.assert_not_called()


def test_finalize_compact_completeness_flag_on_pg_calls_compact_script(monkeypatch):
    monkeypatch.setenv("EXONIT_COMPLETENESS_PG", "1")
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = '{"ok": true, "buckets_touched": [1, 2], "rows_merged": 10}'
    mock_result.stderr = ""
    with patch("subprocess.run", return_value=mock_result) as mock_run:
        orch._finalize_compact_completeness(_FAKE_PG_DSN)

    mock_run.assert_called_once()
    cmd = mock_run.call_args[0][0]
    assert cmd[0] == sys.executable
    assert "compact_completeness.py" in " ".join(str(c) for c in cmd)
    assert "--pg" in cmd
    assert _FAKE_PG_DSN in cmd
    assert mock_run.call_args.kwargs.get("timeout") == orch.COMPLETENESS_COMPACTION_TIMEOUT_SECONDS


def test_finalize_compact_completeness_nonzero_rc_is_non_fatal(monkeypatch, caplog):
    monkeypatch.setenv("EXONIT_COMPLETENESS_PG", "1")
    mock_result = MagicMock()
    mock_result.returncode = 1
    mock_result.stdout = ""
    mock_result.stderr = "boom"
    with patch("subprocess.run", return_value=mock_result):
        orch._finalize_compact_completeness(_FAKE_PG_DSN)  # must not raise


def test_finalize_compact_completeness_timeout_is_non_fatal(monkeypatch):
    monkeypatch.setenv("EXONIT_COMPLETENESS_PG", "1")
    import subprocess as _subprocess
    with patch("subprocess.run",
               side_effect=_subprocess.TimeoutExpired(cmd="x", timeout=1800)):
        orch._finalize_compact_completeness(_FAKE_PG_DSN)  # must not raise


def test_finalize_compact_completeness_exception_is_non_fatal(monkeypatch):
    monkeypatch.setenv("EXONIT_COMPLETENESS_PG", "1")
    with patch("subprocess.run", side_effect=OSError("no such file")):
        orch._finalize_compact_completeness(_FAKE_PG_DSN)  # must not raise


def test_finalize_compact_completeness_bad_json_stdout_is_non_fatal(monkeypatch):
    monkeypatch.setenv("EXONIT_COMPLETENESS_PG", "1")
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = "not json"
    mock_result.stderr = ""
    with patch("subprocess.run", return_value=mock_result):
        orch._finalize_compact_completeness(_FAKE_PG_DSN)  # must not raise


# ---------------------------------------------------------------------------
# Phase 4a Task 4 fast-path (2026-07-17): EXONIT_COMPACT_REMOTE_HOST -- SSH
# invocation on <host> + fallback to local execution on ANY failure.
# ---------------------------------------------------------------------------

_REMOTE_HOST = "user@storage-host"


def test_compact_remote_dsn_replaces_host_with_loopback():
    got = orch._compact_remote_dsn("postgresql://exonit:pw@pg-host.example:5432/exonit")
    assert got == "postgresql://exonit:pw@127.0.0.1:5432/exonit"


def test_compact_remote_dsn_no_userinfo_no_port():
    got = orch._compact_remote_dsn("postgresql://pg-host.example/exonit")
    assert got == "postgresql://127.0.0.1/exonit"


def test_compact_remote_dsn_env_override_wins(monkeypatch):
    monkeypatch.setenv("EXONIT_COMPACT_REMOTE_DSN",
                        "postgresql://someone@10.0.0.1:6543/other")
    got = orch._compact_remote_dsn("postgresql://exonit@pg-host.example:5432/exonit")
    assert got == "postgresql://someone@10.0.0.1:6543/other"


def test_compact_remote_cmd_matches_verified_invocation():
    """The builder must reproduce the operator-verified (dry-run tested)
    command EXACTLY -- see the implementer briefing / spec 2026-07-17."""
    pg_dsn = "postgresql://exonit@pg-host.example:5432/exonit"
    cmd = orch._compact_remote_cmd(_REMOTE_HOST, pg_dsn)
    assert cmd == [
        "ssh", "-o", "ConnectTimeout=15", _REMOTE_HOST,
        "set -a && . $HOME/exonit/.env && set +a && "
        "PYTHONPATH=$HOME/exonit $HOME/exonit/.venv/bin/python "
        "$HOME/exonit/scripts/compact_completeness.py "
        "--pg postgresql://exonit@127.0.0.1:5432/exonit",
    ]


def test_finalize_compact_completeness_remote_unset_never_calls_ssh(monkeypatch):
    """No EXONIT_COMPACT_REMOTE_HOST -> unchanged local-only behaviour, one
    subprocess.run call, argv[0] == sys.executable (never 'ssh')."""
    monkeypatch.setenv("EXONIT_COMPLETENESS_PG", "1")
    monkeypatch.delenv("EXONIT_COMPACT_REMOTE_HOST", raising=False)
    mock_result = MagicMock(returncode=0, stdout="{}", stderr="")
    with patch("subprocess.run", return_value=mock_result) as mock_run:
        orch._finalize_compact_completeness(_FAKE_PG_DSN)
    mock_run.assert_called_once()
    assert mock_run.call_args[0][0][0] == sys.executable


def test_finalize_compact_completeness_remote_success_skips_local(monkeypatch):
    monkeypatch.setenv("EXONIT_COMPLETENESS_PG", "1")
    monkeypatch.setenv("EXONIT_COMPACT_REMOTE_HOST", _REMOTE_HOST)
    mock_result = MagicMock(returncode=0,
                             stdout='{"ok": true, "buckets_touched": [1]}',
                             stderr="")
    with patch("subprocess.run", return_value=mock_result) as mock_run:
        orch._finalize_compact_completeness(_FAKE_PG_DSN)
    mock_run.assert_called_once()
    cmd = mock_run.call_args[0][0]
    assert cmd[0] == "ssh"
    assert _REMOTE_HOST in cmd
    assert mock_run.call_args.kwargs.get("timeout") == orch.EXONIT_COMPACT_REMOTE_TIMEOUT_SECONDS


def test_finalize_compact_completeness_remote_nonzero_rc_falls_back_to_local(monkeypatch):
    monkeypatch.setenv("EXONIT_COMPLETENESS_PG", "1")
    monkeypatch.setenv("EXONIT_COMPACT_REMOTE_HOST", _REMOTE_HOST)
    remote_fail = MagicMock(returncode=1, stdout="", stderr="ssh: connect refused")
    local_ok = MagicMock(returncode=0, stdout='{"ok": true}', stderr="")
    with patch("subprocess.run", side_effect=[remote_fail, local_ok]) as mock_run:
        orch._finalize_compact_completeness(_FAKE_PG_DSN)  # must not raise
    assert mock_run.call_count == 2
    first_cmd, second_cmd = (c.args[0] for c in mock_run.call_args_list)
    assert first_cmd[0] == "ssh"
    assert second_cmd[0] == sys.executable
    assert _FAKE_PG_DSN in second_cmd  # local fallback uses the ORIGINAL dsn, not the loopback one


def test_finalize_compact_completeness_remote_timeout_falls_back_to_local(monkeypatch):
    import subprocess as _subprocess
    monkeypatch.setenv("EXONIT_COMPLETENESS_PG", "1")
    monkeypatch.setenv("EXONIT_COMPACT_REMOTE_HOST", _REMOTE_HOST)
    local_ok = MagicMock(returncode=0, stdout='{"ok": true}', stderr="")
    with patch("subprocess.run", side_effect=[
            _subprocess.TimeoutExpired(cmd="ssh", timeout=2700), local_ok]) as mock_run:
        orch._finalize_compact_completeness(_FAKE_PG_DSN)  # must not raise
    assert mock_run.call_count == 2
    assert mock_run.call_args_list[1].args[0][0] == sys.executable


def test_finalize_compact_completeness_remote_unreachable_falls_back_to_local(monkeypatch):
    """ssh itself failing to spawn (e.g. no ssh binary) must also fall back,
    not just a completed-but-nonzero-rc ssh."""
    monkeypatch.setenv("EXONIT_COMPLETENESS_PG", "1")
    monkeypatch.setenv("EXONIT_COMPACT_REMOTE_HOST", _REMOTE_HOST)
    local_ok = MagicMock(returncode=0, stdout='{"ok": true}', stderr="")
    with patch("subprocess.run",
               side_effect=[OSError("no such file or directory: ssh"), local_ok]) as mock_run:
        orch._finalize_compact_completeness(_FAKE_PG_DSN)  # must not raise
    assert mock_run.call_count == 2


def test_finalize_compact_completeness_remote_bad_json_falls_back_to_local(monkeypatch):
    monkeypatch.setenv("EXONIT_COMPLETENESS_PG", "1")
    monkeypatch.setenv("EXONIT_COMPACT_REMOTE_HOST", _REMOTE_HOST)
    remote_bad_json = MagicMock(returncode=0, stdout="not json", stderr="")
    local_ok = MagicMock(returncode=0, stdout='{"ok": true}', stderr="")
    with patch("subprocess.run", side_effect=[remote_bad_json, local_ok]) as mock_run:
        orch._finalize_compact_completeness(_FAKE_PG_DSN)  # must not raise
    assert mock_run.call_count == 2


# ---------------------------------------------------------------------------
# Component 3 (2026-09-09, docs/superpowers/specs/2026-09-09-compactd-
# service-design.md): the exonit-compactd HTTP-service fast-path —
# precedence service -> SSH -> local. A real ``http.server`` in a background
# thread on an ephemeral port stands in for the (stdlib-only) service, same
# technique as tests/test_kill_and_resume.py's stub_server fixture; nothing
# here ever touches a real network host or <host>.
# ---------------------------------------------------------------------------

import contextlib  # noqa: E402
import http.server  # noqa: E402
import threading  # noqa: E402
import urllib.parse as _urlparse  # noqa: E402

_COMPACT_SERVICE_TOKEN = "test-compactd-token"  # nosec - test fixture only


class _CompactdState:
    def __init__(self):
        self.lock = threading.Lock()
        self.post_calls = 0
        self.get_calls: list[str] = []
        self.post_bodies: list[dict] = []


def _make_compactd_handler(state: "_CompactdState", script: dict):
    """``script`` = {"post": (status, body), "jobs": {job_id: [(status, body), ...]}}.

    Each ``GET /jobs/<id>`` call advances through that job's response list
    (the last entry repeats once exhausted) — lets a test model a job that
    is "running" for N polls before flipping to "done"/"error".
    """
    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a):  # silence — keep test output clean
            pass

        def _json(self, obj, code=200):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0) or 0)
            raw_body = self.rfile.read(length) if length else b""
            try:
                parsed_body = json.loads(raw_body) if raw_body else {}
            except (json.JSONDecodeError, TypeError):
                parsed_body = None
            with state.lock:
                state.post_calls += 1
                state.post_bodies.append(parsed_body)
            status, body = script["post"]
            self._json(body, status)

        def do_GET(self):
            parsed = _urlparse.urlparse(self.path)
            job_id = parsed.path.rsplit("/", 1)[-1]
            seq = script.get("jobs", {}).get(job_id, [(404, {"error": "not found"})])
            with state.lock:
                state.get_calls.append(parsed.path)
                idx = min(len(state.get_calls) - 1, len(seq) - 1)
            status, body = seq[idx]
            self._json(body, status)

    return Handler


@contextlib.contextmanager
def _fake_compactd(script: dict):
    state = _CompactdState()
    server = http.server.ThreadingHTTPServer(
        ("127.0.0.1", 0), _make_compactd_handler(state, script))
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{port}", state
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _closed_port_url() -> str:
    """A loopback URL nothing is listening on — guarantees ECONNREFUSED
    rather than depending on a hardcoded 'probably free' port number."""
    tmp = http.server.HTTPServer(("127.0.0.1", 0), http.server.BaseHTTPRequestHandler)
    port = tmp.server_address[1]
    tmp.server_close()
    return f"http://127.0.0.1:{port}"


# --- _run_compact_service unit tests (direct, no _finalize_ involvement) ---

def test_run_compact_service_ok_returns_summary_and_polls_until_done():
    script = {
        "post": (202, {"job_id": "job1"}),
        "jobs": {"job1": [
            (200, {"state": "running"}),
            (200, {"state": "done", "summary": {"ok": True, "rows_merged": 5}}),
        ]},
    }
    with _fake_compactd(script) as (url, state), patch("time.sleep") as mock_sleep:
        result = orch._run_compact_service(url, _COMPACT_SERVICE_TOKEN, run_id="run-1")
    assert result == {"ok": True, "rows_merged": 5}
    assert state.post_calls == 1
    assert state.get_calls == ["/jobs/job1", "/jobs/job1"]
    assert mock_sleep.call_count == 2  # one sleep per poll iteration


def test_run_compact_service_busy_409_follows_active_job_to_done():
    """A 409 on POST /compact means another caller already owns a job —
    following IT to completion is a SUCCESS for this call."""
    script = {
        "post": (409, {"error": "busy", "job_id": "job-active"}),
        "jobs": {"job-active": [(200, {"state": "done",
                                        "summary": {"ok": True, "path": "busy-follow"}})]},
    }
    with _fake_compactd(script) as (url, state), patch("time.sleep"):
        result = orch._run_compact_service(url, _COMPACT_SERVICE_TOKEN)
    assert result == {"ok": True, "path": "busy-follow"}
    assert state.post_calls == 1
    assert state.get_calls == ["/jobs/job-active"]


def test_run_compact_service_post_body_carries_progress_fields_with_run_id_and_url(monkeypatch):
    """2026-09-09 review Finding 2: parity with the SSH path's progress_env
    (_compact_remote_cmd) — run_id + a LAN-reachable progress URL must ride
    the POST /compact body so the service can export them as
    EXONIT_PROGRESS_* on the compaction subprocess."""
    monkeypatch.delenv("EXONIT_PROGRESS_URL", raising=False)
    monkeypatch.setattr(orch, "_SERVER_URL", "http://10.0.0.76:8400")
    script = {
        "post": (202, {"job_id": "job1"}),
        "jobs": {"job1": [(200, {"state": "done", "summary": {"ok": True}})]},
    }
    with _fake_compactd(script) as (url, state), patch("time.sleep"):
        result = orch._run_compact_service(url, _COMPACT_SERVICE_TOKEN, run_id="run-A")
    assert result == {"ok": True}
    assert len(state.post_bodies) == 1
    assert state.post_bodies[0] == {
        "run_id": "run-A",
        "phase": "FINALIZE",
        "progress_url": "http://10.0.0.76:8400",
    }


def test_run_compact_service_post_body_omits_progress_fields_without_run_id(monkeypatch):
    """No run_id (e.g. a hand-run compaction) -> the fields must be OMITTED
    entirely, never sent as empty strings — same posture as the SSH path's
    _compact_remote_cmd without a run_id."""
    monkeypatch.setattr(orch, "_SERVER_URL", "http://10.0.0.76:8400")
    script = {
        "post": (202, {"job_id": "job1"}),
        "jobs": {"job1": [(200, {"state": "done", "summary": {"ok": True}})]},
    }
    with _fake_compactd(script) as (url, state), patch("time.sleep"):
        result = orch._run_compact_service(url, _COMPACT_SERVICE_TOKEN)  # no run_id
    assert result == {"ok": True}
    assert state.post_bodies == [{}]


def test_run_compact_service_post_body_omits_progress_fields_without_usable_url(monkeypatch):
    """run_id present but no LAN-reachable progress URL (e.g. the server_url
    is localhost, unusable from a remote host per _progress_url's own
    contract) -> still omitted, never a URL pointing <host> at itself."""
    monkeypatch.delenv("EXONIT_PROGRESS_URL", raising=False)
    monkeypatch.setattr(orch, "_SERVER_URL", "http://localhost:8400")
    script = {
        "post": (202, {"job_id": "job1"}),
        "jobs": {"job1": [(200, {"state": "done", "summary": {"ok": True}})]},
    }
    with _fake_compactd(script) as (url, state), patch("time.sleep"):
        result = orch._run_compact_service(url, _COMPACT_SERVICE_TOKEN, run_id="run-A")
    assert result == {"ok": True}
    assert state.post_bodies == [{}]


def test_run_compact_service_job_error_writes_faillog_and_returns_none(monkeypatch):
    script = {
        "post": (202, {"job_id": "job1"}),
        "jobs": {"job1": [(200, {"state": "error", "rc": 1,
                                  "log_tail": "boom exploded traceback"})]},
    }
    with _fake_compactd(script) as (url, state), patch("time.sleep"):
        result = orch._run_compact_service(url, _COMPACT_SERVICE_TOKEN)
    assert result is None
    log_files = list(orch._COMPACT_REMOTE_FAIL_LOG_DIR.glob("compact_remote_fail_*.log"))
    assert len(log_files) == 1
    content = log_files[0].read_text()
    assert "reason: service_error" in content
    assert "boom exploded traceback" in content


def test_run_compact_service_job_404_writes_faillog_and_returns_none(monkeypatch):
    """The job vanished (e.g. the service restarted mid-job) — 404 on poll
    is treated the same as a job error: fail-log + None."""
    script = {"post": (202, {"job_id": "job-gone"}), "jobs": {}}
    with _fake_compactd(script) as (url, state), patch("time.sleep"):
        result = orch._run_compact_service(url, _COMPACT_SERVICE_TOKEN)
    assert result is None
    log_files = list(orch._COMPACT_REMOTE_FAIL_LOG_DIR.glob("compact_remote_fail_*.log"))
    assert len(log_files) == 1
    assert "reason: service_error" in log_files[0].read_text()


def test_run_compact_service_unreachable_writes_faillog_and_returns_none(monkeypatch):
    """The POST itself cannot even connect (service down / wrong port) —
    must degrade to None, never raise."""
    url = _closed_port_url()
    result = orch._run_compact_service(url, _COMPACT_SERVICE_TOKEN)
    assert result is None
    log_files = list(orch._COMPACT_REMOTE_FAIL_LOG_DIR.glob("compact_remote_fail_*.log"))
    assert len(log_files) == 1
    assert "reason: service_unreachable" in log_files[0].read_text()


def test_run_compact_service_total_timeout_writes_faillog_and_returns_none(monkeypatch):
    """The total elapsed time (POST + polls) crossing
    EXONIT_COMPACT_REMOTE_TIMEOUT_SECONDS must bail out, never poll forever."""
    script = {
        "post": (202, {"job_id": "job1"}),
        "jobs": {"job1": [(200, {"state": "running"})]},  # never reaches done
    }
    # Two time.monotonic() calls happen before the first timeout check can
    # possibly fire: one for `started`, one for the loop's own elapsed
    # check. Forcing the second call far past `started` triggers the
    # timeout branch on the very first iteration, without needing a real
    # 2700s wait or a patched constant.
    with _fake_compactd(script) as (url, state), \
         patch("time.sleep"), \
         patch("time.monotonic", side_effect=[0.0, 999999.0]):
        result = orch._run_compact_service(url, _COMPACT_SERVICE_TOKEN)
    assert result is None
    log_files = list(orch._COMPACT_REMOTE_FAIL_LOG_DIR.glob("compact_remote_fail_*.log"))
    assert len(log_files) == 1
    assert "reason: service_timeout" in log_files[0].read_text()
    # The timeout fired before any poll was even attempted.
    assert state.get_calls == []


# --- _finalize_compact_completeness precedence: service -> SSH -> local ---

def test_finalize_compact_completeness_service_success_skips_ssh_and_local(monkeypatch):
    """(a) service OK -> neither ssh nor local ever call subprocess.run,
    even with EXONIT_COMPACT_REMOTE_HOST also set (proves the service wins
    the precedence, not just that it's the only path configured)."""
    monkeypatch.setenv("EXONIT_COMPLETENESS_PG", "1")
    monkeypatch.setenv("EXONIT_COMPACT_REMOTE_HOST", _REMOTE_HOST)
    script = {
        "post": (202, {"job_id": "job1"}),
        "jobs": {"job1": [(200, {"state": "done", "summary": {"ok": True}})]},
    }
    with _fake_compactd(script) as (url, state):
        monkeypatch.setenv("EXONIT_COMPACT_SERVICE_URL", url)
        monkeypatch.setenv("EXONIT_COMPACT_SERVICE_TOKEN", _COMPACT_SERVICE_TOKEN)
        with patch("subprocess.run") as mock_run, patch("time.sleep"):
            orch._finalize_compact_completeness(_FAKE_PG_DSN)
    mock_run.assert_not_called()
    assert state.post_calls == 1


def test_finalize_compact_completeness_service_error_falls_back_to_ssh_and_writes_faillog(monkeypatch):
    """(b) service returns an error job -> falls back to the SSH path, and
    the service failure itself is fail-logged."""
    monkeypatch.setenv("EXONIT_COMPLETENESS_PG", "1")
    monkeypatch.setenv("EXONIT_COMPACT_REMOTE_HOST", _REMOTE_HOST)
    script = {
        "post": (202, {"job_id": "job1"}),
        "jobs": {"job1": [(200, {"state": "error", "rc": 1, "log_tail": "svc died"})]},
    }
    ssh_ok = MagicMock(returncode=0, stdout='{"ok": true}', stderr="")
    with _fake_compactd(script) as (url, state):
        monkeypatch.setenv("EXONIT_COMPACT_SERVICE_URL", url)
        monkeypatch.setenv("EXONIT_COMPACT_SERVICE_TOKEN", _COMPACT_SERVICE_TOKEN)
        with patch("subprocess.run", return_value=ssh_ok) as mock_run, patch("time.sleep"):
            orch._finalize_compact_completeness(_FAKE_PG_DSN)  # must not raise
    mock_run.assert_called_once()
    assert mock_run.call_args[0][0][0] == "ssh"
    log_files = list(orch._COMPACT_REMOTE_FAIL_LOG_DIR.glob("compact_remote_fail_*.log"))
    assert len(log_files) == 1
    assert "reason: service_error" in log_files[0].read_text()
    assert "svc died" in log_files[0].read_text()


def test_finalize_compact_completeness_service_unreachable_falls_back_net(monkeypatch):
    """(c) service inaccessible (connection refused) -> clean fallback to
    the SSH path (no exception, ssh still invoked exactly once)."""
    monkeypatch.setenv("EXONIT_COMPLETENESS_PG", "1")
    monkeypatch.setenv("EXONIT_COMPACT_REMOTE_HOST", _REMOTE_HOST)
    monkeypatch.setenv("EXONIT_COMPACT_SERVICE_URL", _closed_port_url())
    monkeypatch.setenv("EXONIT_COMPACT_SERVICE_TOKEN", _COMPACT_SERVICE_TOKEN)
    ssh_ok = MagicMock(returncode=0, stdout='{"ok": true}', stderr="")
    with patch("subprocess.run", return_value=ssh_ok) as mock_run:
        orch._finalize_compact_completeness(_FAKE_PG_DSN)  # must not raise
    mock_run.assert_called_once()
    assert mock_run.call_args[0][0][0] == "ssh"
    log_files = list(orch._COMPACT_REMOTE_FAIL_LOG_DIR.glob("compact_remote_fail_*.log"))
    assert len(log_files) == 1
    assert "reason: service_unreachable" in log_files[0].read_text()


def test_finalize_compact_completeness_service_error_no_remote_host_falls_back_to_local(monkeypatch):
    """service fails and EXONIT_COMPACT_REMOTE_HOST is unset -> falls all
    the way through to the local path (argv[0] == sys.executable)."""
    monkeypatch.setenv("EXONIT_COMPLETENESS_PG", "1")
    monkeypatch.delenv("EXONIT_COMPACT_REMOTE_HOST", raising=False)
    monkeypatch.setenv("EXONIT_COMPACT_SERVICE_URL", _closed_port_url())
    monkeypatch.setenv("EXONIT_COMPACT_SERVICE_TOKEN", _COMPACT_SERVICE_TOKEN)
    local_ok = MagicMock(returncode=0, stdout='{"ok": true}', stderr="")
    with patch("subprocess.run", return_value=local_ok) as mock_run:
        orch._finalize_compact_completeness(_FAKE_PG_DSN)  # must not raise
    mock_run.assert_called_once()
    assert mock_run.call_args[0][0][0] == sys.executable


def test_finalize_compact_completeness_busy_409_service_still_wins_over_ssh(monkeypatch):
    """(d) 409 busy -> the service still follows the active job to 'done'
    and that counts as the service path succeeding — ssh/local never run."""
    monkeypatch.setenv("EXONIT_COMPLETENESS_PG", "1")
    monkeypatch.setenv("EXONIT_COMPACT_REMOTE_HOST", _REMOTE_HOST)
    script = {
        "post": (409, {"error": "busy", "job_id": "job-active"}),
        "jobs": {"job-active": [(200, {"state": "done", "summary": {"ok": True}})]},
    }
    with _fake_compactd(script) as (url, state):
        monkeypatch.setenv("EXONIT_COMPACT_SERVICE_URL", url)
        monkeypatch.setenv("EXONIT_COMPACT_SERVICE_TOKEN", _COMPACT_SERVICE_TOKEN)
        with patch("subprocess.run") as mock_run, patch("time.sleep"):
            orch._finalize_compact_completeness(_FAKE_PG_DSN)
    mock_run.assert_not_called()
    assert state.post_calls == 1
    assert state.get_calls == ["/jobs/job-active"]


def test_finalize_compact_completeness_no_service_env_uses_ssh_path_unchanged(monkeypatch):
    """(e) neither EXONIT_COMPACT_SERVICE_URL nor _TOKEN set -> the service
    path is never attempted and the pre-existing SSH precedence is
    byte-for-byte unchanged."""
    monkeypatch.setenv("EXONIT_COMPLETENESS_PG", "1")
    monkeypatch.delenv("EXONIT_COMPACT_SERVICE_URL", raising=False)
    monkeypatch.delenv("EXONIT_COMPACT_SERVICE_TOKEN", raising=False)
    monkeypatch.setenv("EXONIT_COMPACT_REMOTE_HOST", _REMOTE_HOST)
    ssh_ok = MagicMock(returncode=0, stdout='{"ok": true}', stderr="")
    with patch.object(orch, "_run_compact_service") as mock_service, \
         patch("subprocess.run", return_value=ssh_ok) as mock_run:
        orch._finalize_compact_completeness(_FAKE_PG_DSN)
    mock_service.assert_not_called()
    mock_run.assert_called_once()
    assert mock_run.call_args[0][0][0] == "ssh"


def test_finalize_compact_completeness_only_url_set_does_not_trigger_service(monkeypatch):
    """Both env vars are required — EXONIT_COMPACT_SERVICE_URL alone (no
    token) must not attempt the service (the service always requires
    auth — see Component 2's spec)."""
    monkeypatch.setenv("EXONIT_COMPLETENESS_PG", "1")
    monkeypatch.setenv("EXONIT_COMPACT_SERVICE_URL", "http://127.0.0.1:8646")
    monkeypatch.delenv("EXONIT_COMPACT_SERVICE_TOKEN", raising=False)
    monkeypatch.setenv("EXONIT_COMPACT_REMOTE_HOST", _REMOTE_HOST)
    ssh_ok = MagicMock(returncode=0, stdout='{"ok": true}', stderr="")
    with patch.object(orch, "_run_compact_service") as mock_service, \
         patch("subprocess.run", return_value=ssh_ok) as mock_run:
        orch._finalize_compact_completeness(_FAKE_PG_DSN)
    mock_service.assert_not_called()
    mock_run.assert_called_once()


# ---------------------------------------------------------------------------
# 2026-07-21 fix: full remote stdout+stderr persisted on failure (the
# previous 500-char HEAD truncation lost every traceback — the incident
# that motivated this was a 28-minute remote compaction dying with rc=1 and
# leaving zero usable diagnosis behind).
# ---------------------------------------------------------------------------

_HEAD_PADDING_LINES = 300  # * ~22 chars/line ~= 6600 chars, well past the 2000-char tail window
_LONG_TRACEBACK = (
    "INFO some setup line\n" * _HEAD_PADDING_LINES
    + "Traceback (most recent call last):\n"
    + "  File \"compact_completeness.py\", line 999, in <module>\n"
    + "    raise RuntimeError(THIS-IS-THE-REAL-ERROR)\n"
    + "RuntimeError: THIS-IS-THE-REAL-ERROR\n"
)


def test_run_compact_remote_nonzero_rc_writes_full_log_and_returns_none(monkeypatch):
    """rc!=0: the full remote stdout+stderr is written to a
    compact_remote_fail_<ts>.log file (NOT just a 500-char head snippet),
    and _run_compact_remote still returns None (unchanged contract)."""
    remote_fail = MagicMock(returncode=1, stdout="", stderr=_LONG_TRACEBACK)
    with patch("subprocess.run", return_value=remote_fail):
        result = orch._run_compact_remote(_REMOTE_HOST, _FAKE_PG_DSN)

    assert result is None
    log_files = list(orch._COMPACT_REMOTE_FAIL_LOG_DIR.glob("compact_remote_fail_*.log"))
    assert len(log_files) == 1
    content = log_files[0].read_text()
    # The FULL stderr survives verbatim, not truncated to the first 500 chars.
    assert _LONG_TRACEBACK in content
    assert "THIS-IS-THE-REAL-ERROR" in content
    assert f"host: {_REMOTE_HOST}" in content
    assert "reason: nonzero_rc" in content
    assert "returncode: 1" in content


def test_run_compact_remote_nonzero_rc_logs_path_and_tail(monkeypatch, caplog):
    """The warning line references the log PATH and a TAIL excerpt from the
    end of stderr (where the traceback lives) — never the head."""
    remote_fail = MagicMock(returncode=1, stdout="", stderr=_LONG_TRACEBACK)
    with patch("subprocess.run", return_value=remote_fail):
        with caplog.at_level("WARNING"):
            orch._run_compact_remote(_REMOTE_HOST, _FAKE_PG_DSN)

    warning_text = "\n".join(r.message for r in caplog.records)
    assert "compact_remote_fail_" in warning_text
    assert "THIS-IS-THE-REAL-ERROR" in warning_text
    # The head-only INFO padding must NOT be what got surfaced — the tail
    # excerpt is what matters, not the first 500 chars of the head.
    assert warning_text.count("INFO some setup line") < _HEAD_PADDING_LINES


def test_run_compact_remote_timeout_writes_partial_captured_output(monkeypatch):
    """A TimeoutExpired still carries whatever stdout/stderr subprocess.run
    captured before the kill — that partial output must be persisted too,
    not discarded."""
    import subprocess as _subprocess

    timeout_exc = _subprocess.TimeoutExpired(
        cmd="ssh", timeout=2700, output="partial stdout",
        stderr="partial stderr before timeout")
    with patch("subprocess.run", side_effect=timeout_exc):
        result = orch._run_compact_remote(_REMOTE_HOST, _FAKE_PG_DSN)

    assert result is None
    log_files = list(orch._COMPACT_REMOTE_FAIL_LOG_DIR.glob("compact_remote_fail_*.log"))
    assert len(log_files) == 1
    content = log_files[0].read_text()
    assert "partial stdout" in content
    assert "partial stderr before timeout" in content
    assert "reason: timeout" in content


def test_run_compact_remote_bad_json_writes_full_log(monkeypatch):
    remote_bad_json = MagicMock(returncode=0, stdout="not json" + ("x" * 1000),
                                 stderr="")
    with patch("subprocess.run", return_value=remote_bad_json):
        result = orch._run_compact_remote(_REMOTE_HOST, _FAKE_PG_DSN)

    assert result is None
    log_files = list(orch._COMPACT_REMOTE_FAIL_LOG_DIR.glob("compact_remote_fail_*.log"))
    assert len(log_files) == 1
    content = log_files[0].read_text()
    assert "not json" + ("x" * 1000) in content
    assert "reason: bad_json_stdout" in content


def test_run_compact_remote_success_writes_no_fail_log(monkeypatch):
    """The happy path must never write a fail log."""
    remote_ok = MagicMock(returncode=0, stdout='{"ok": true}', stderr="")
    with patch("subprocess.run", return_value=remote_ok):
        result = orch._run_compact_remote(_REMOTE_HOST, _FAKE_PG_DSN)

    assert result == {"ok": True}
    assert not orch._COMPACT_REMOTE_FAIL_LOG_DIR.exists() or \
        list(orch._COMPACT_REMOTE_FAIL_LOG_DIR.glob("compact_remote_fail_*.log")) == []


def test_run_compact_remote_unwritable_log_dir_still_returns_none(monkeypatch, caplog):
    """Review regression (2026-07-21): the diagnostic log write itself must
    NEVER break _run_compact_remote's own 'never raises' contract. An
    unwritable log dir (disk full / permission error / a path that can't
    even be mkdir'd) must degrade to 'diagnosis lost' — a warning — never
    to an unhandled exception propagating out of here."""
    unwritable = orch.Path("/proc/nonexistent_readonly_dir/subdir")
    monkeypatch.setattr(orch, "_COMPACT_REMOTE_FAIL_LOG_DIR", unwritable)

    remote_fail = MagicMock(returncode=1, stdout="", stderr=_LONG_TRACEBACK)
    with patch("subprocess.run", return_value=remote_fail):
        with caplog.at_level("WARNING"):
            result = orch._run_compact_remote(_REMOTE_HOST, _FAKE_PG_DSN)  # must not raise

    assert result is None
    warning_text = "\n".join(r.message for r in caplog.records)
    # The write-failure itself was logged...
    assert "could not write compact_remote_fail log" in warning_text
    # ...AND the minimal diagnostic (tail excerpt) was still surfaced inline,
    # even though the full log could not be persisted.
    assert "THIS-IS-THE-REAL-ERROR" in warning_text


# --- step_finalize wiring: calls _finalize_compact_completeness ---

def _finalize_state_file(tmp_path):
    sf = str(tmp_path / "state.json")
    orch._save_state(sf, {
        "state": "FINALIZE",
        "run_id": "discovery_s0001_lot1_2026-07-05",
        "scorecard_json": "data/scorecards/discovery_s0001_lot1_2026-07-05.json",
        "dossier": "data/scorecards/discovery_s0001_lot1_2026-07-05_dossier.md",
    })
    return sf


def test_step_finalize_calls_compact_completeness_hook_on_success(tmp_path):
    sf = _finalize_state_file(tmp_path)
    fake_result = {"campaign_action": "updated", "n_lot": 1, "n_completed": 1,
                   "n_failed": 0, "n_other": 0, "n_strong": 0, "n_finalists": 0}
    with patch("discovery_finalize.finalize_run", return_value=fake_result), \
         patch.object(orch, "_finalize_compact_completeness") as mock_compact:
        ok, detail = orch.step_finalize(
            sf, lot_csv="unused.csv", sector=1, db=_FAKE_PG_DSN, dry_run=False)
    assert ok is True
    # 2026-07-27: the hook now also receives the run_id, so the FINALIZE
    # child can report its bucket progress under this run.
    mock_compact.assert_called_once_with(
        _FAKE_PG_DSN, "discovery_s0001_lot1_2026-07-05")


def test_step_finalize_calls_compact_completeness_hook_even_if_finalize_run_raises(tmp_path):
    """finalize_run (CAMPAIGN.md/STATUS.md) and the completeness compaction
    hook are independent concerns — one raising must not skip the other."""
    sf = _finalize_state_file(tmp_path)
    with patch("discovery_finalize.finalize_run", side_effect=RuntimeError("boom")), \
         patch.object(orch, "_finalize_compact_completeness") as mock_compact:
        ok, detail = orch.step_finalize(
            sf, lot_csv="unused.csv", sector=1, db=_FAKE_PG_DSN, dry_run=False)
    assert ok is True  # FINALIZE is non-fatal regardless
    # 2026-07-27: the hook now also receives the run_id, so the FINALIZE
    # child can report its bucket progress under this run.
    mock_compact.assert_called_once_with(
        _FAKE_PG_DSN, "discovery_s0001_lot1_2026-07-05")


def test_step_finalize_survives_compact_completeness_hook_raising(tmp_path):
    """Defense in depth: even if _finalize_compact_completeness somehow raises
    (it is designed not to), step_finalize must still return non-fatally."""
    sf = _finalize_state_file(tmp_path)
    fake_result = {"campaign_action": "updated", "n_lot": 1, "n_completed": 1,
                   "n_failed": 0, "n_other": 0, "n_strong": 0, "n_finalists": 0}
    with patch("discovery_finalize.finalize_run", return_value=fake_result), \
         patch.object(orch, "_finalize_compact_completeness",
                      side_effect=RuntimeError("unexpected")):
        ok, detail = orch.step_finalize(
            sf, lot_csv="unused.csv", sector=1, db=_FAKE_PG_DSN, dry_run=False)
    assert ok is True


def test_step_finalize_dry_run_never_calls_compact_completeness_hook(tmp_path):
    sf = _finalize_state_file(tmp_path)
    with patch.object(orch, "_finalize_compact_completeness") as mock_compact:
        ok, detail = orch.step_finalize(
            sf, lot_csv="unused.csv", sector=1, db=_FAKE_PG_DSN, dry_run=True)
    assert ok is True
    mock_compact.assert_not_called()


# --- step_finalize wiring: best-effort worker-recycle trigger (2026-07-17) ---
# docs/superpowers/specs/2026-07-17-worker-recycle-postrun.md

def test_step_finalize_calls_recycle_workers_hook_on_success(tmp_path):
    sf = _finalize_state_file(tmp_path)
    fake_result = {"campaign_action": "updated", "n_lot": 1, "n_completed": 1,
                   "n_failed": 0, "n_other": 0, "n_strong": 0, "n_finalists": 0}
    with patch("discovery_finalize.finalize_run", return_value=fake_result), \
         patch.object(orch, "_finalize_compact_completeness"), \
         patch.object(orch, "_recycle_workers") as mock_recycle:
        ok, detail = orch.step_finalize(
            sf, lot_csv="unused.csv", sector=1, db=_FAKE_PG_DSN, dry_run=False,
            server_url="http://localhost:8400")
    assert ok is True
    mock_recycle.assert_called_once_with("http://localhost:8400", False)


def test_step_finalize_recycle_hook_failure_does_not_fail_finalize(tmp_path):
    """A dead/unreachable server for the recycle POST is non-fatal — FINALIZE
    still reports ok=True (the science artifacts are already complete)."""
    sf = _finalize_state_file(tmp_path)
    fake_result = {"campaign_action": "updated", "n_lot": 1, "n_completed": 1,
                   "n_failed": 0, "n_other": 0, "n_strong": 0, "n_finalists": 0}
    with patch("discovery_finalize.finalize_run", return_value=fake_result), \
         patch.object(orch, "_finalize_compact_completeness"), \
         patch.object(orch, "_recycle_workers",
                      side_effect=RuntimeError("unexpected")):
        ok, detail = orch.step_finalize(
            sf, lot_csv="unused.csv", sector=1, db=_FAKE_PG_DSN, dry_run=False,
            server_url="http://localhost:8400")
    assert ok is True


def test_step_finalize_no_server_url_skips_recycle_call_silently(tmp_path):
    """The default (server_url="") — e.g. the AWS_FINALIZE call site, which
    never passes one — must not attempt any HTTP call."""
    sf = _finalize_state_file(tmp_path)
    fake_result = {"campaign_action": "updated", "n_lot": 1, "n_completed": 1,
                   "n_failed": 0, "n_other": 0, "n_strong": 0, "n_finalists": 0}
    with patch("discovery_finalize.finalize_run", return_value=fake_result), \
         patch.object(orch, "_finalize_compact_completeness"), \
         patch("urllib.request.urlopen",
               side_effect=AssertionError("must not call urlopen")):
        ok, detail = orch.step_finalize(
            sf, lot_csv="unused.csv", sector=1, db=_FAKE_PG_DSN, dry_run=False)
    assert ok is True


def test_recycle_workers_dry_run_skips_http_call():
    with patch("urllib.request.urlopen",
               side_effect=AssertionError("must not call urlopen")):
        orch._recycle_workers("http://localhost:8400", dry_run=True)  # no raise


def test_recycle_workers_posts_and_logs_generation():
    resp = _FakeResp({"generation": 3})
    with patch("urllib.request.urlopen", return_value=resp) as mock_urlopen:
        orch._recycle_workers("http://localhost:8400", dry_run=False)
    (req,), kwargs = mock_urlopen.call_args
    assert req.full_url == "http://localhost:8400/api/admin/recycle-workers"
    assert req.get_method() == "POST"
    assert kwargs.get("timeout") == 10


def test_recycle_workers_http_failure_is_non_fatal():
    with patch("urllib.request.urlopen", side_effect=Exception("connection refused")):
        orch._recycle_workers("http://localhost:8400", dry_run=False)  # must not raise


def test_recycle_workers_carries_worker_token_auth_header(monkeypatch):
    monkeypatch.setenv("EXONIT_WORKER_TOKEN", "secret-token")
    resp = _FakeResp({"generation": 1})
    with patch("urllib.request.urlopen", return_value=resp) as mock_urlopen:
        orch._recycle_workers("http://localhost:8400", dry_run=False)
    (req,), _kwargs = mock_urlopen.call_args
    assert req.get_header("Authorization") == "Bearer secret-token"


# --- step_wal_checkpoint: flag-ON no-op guard ---

def test_step_wal_checkpoint_flag_on_never_opens_sqlite(monkeypatch):
    monkeypatch.setenv("EXONIT_COMPLETENESS_PG", "1")
    with patch("sqlite3.connect", side_effect=AssertionError(
            "step_wal_checkpoint must not open sqlite when the completeness "
            "PG flag is ON")):
        orch.step_wal_checkpoint(db=None, dry_run=False)  # must not raise


def test_step_wal_checkpoint_flag_off_pg_db_is_still_a_noop(monkeypatch):
    """Flag OFF (unchanged behaviour): a PG DSN main backend is already a
    no-op via is_pg_url — must remain so."""
    monkeypatch.delenv("EXONIT_COMPLETENESS_PG", raising=False)
    with patch("sqlite3.connect", side_effect=AssertionError(
            "must not open sqlite for a PG db_path")):
        orch.step_wal_checkpoint(db=_FAKE_PG_DSN, dry_run=False)  # must not raise


def test_step_wal_checkpoint_flag_off_sqlite_db_unchanged(monkeypatch, tmp_path):
    """Flag OFF + a real SQLite path: behaviour is completely unchanged —
    the WAL TRUNCATE checkpoint still runs."""
    monkeypatch.delenv("EXONIT_COMPLETENESS_PG", raising=False)
    db_path = str(tmp_path / "exonit.db")
    import sqlite3
    con = sqlite3.connect(db_path)
    con.execute("CREATE TABLE t (x INTEGER)")
    con.commit()
    con.close()
    orch.step_wal_checkpoint(db=db_path, dry_run=False)  # must not raise


# --- step_stop_nvidiarack ---

def test_step_stop_nvidiarack_dry_run():
    ok, detail = orch.step_stop_nvidiarack(state_file="unused", dry_run=True)
    assert ok is True


def test_step_stop_nvidiarack_unconfigured_fails_on_gpuq_path(monkeypatch):
    """No GPU-host worker configured + gpuq LoRA path -> failure, no farmctl."""
    monkeypatch.setattr(orch, "NVIDIARACK_WORKER_NAME", "")
    monkeypatch.setattr(orch, "LORA_CMD", orch.LORA_SSH_CMD)
    with patch("subprocess.run") as mock_run:
        ok, detail = orch.step_stop_nvidiarack(state_file="unused", dry_run=False)
    assert ok is False
    assert "EXONIT_GPU_HOST_WORKER" in detail
    mock_run.assert_not_called()


def test_step_stop_nvidiarack_unconfigured_is_noop(monkeypatch):
    """EXONIT_GPU_HOST_WORKER unset -> nothing to stop, farmctl never called."""
    monkeypatch.setattr(orch, "NVIDIARACK_WORKER_NAME", "")
    monkeypatch.setattr(orch, "LORA_CMD", orch.LORA_MACS_CMD)  # default Mac-fleet path
    with patch("subprocess.run") as mock_run:
        ok, detail = orch.step_stop_nvidiarack(state_file="unused", dry_run=False)
    assert ok is True
    mock_run.assert_not_called()


def test_step_stop_nvidiarack_success(monkeypatch):
    monkeypatch.setattr(orch, "NVIDIARACK_WORKER_NAME", "worker-gpu-host")
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = "stopped"
    mock_result.stderr = ""
    with patch("subprocess.run", return_value=mock_result):
        ok, detail = orch.step_stop_nvidiarack(state_file="unused", dry_run=False)
    assert ok is True


def test_step_stop_nvidiarack_failure(monkeypatch):
    monkeypatch.setattr(orch, "NVIDIARACK_WORKER_NAME", "worker-gpu-host")
    mock_result = MagicMock()
    mock_result.returncode = 2
    mock_result.stdout = ""
    mock_result.stderr = "SSH failed"
    with patch("subprocess.run", return_value=mock_result):
        ok, detail = orch.step_stop_nvidiarack(state_file="unused", dry_run=False)
    assert ok is False
    assert "rc=2" in detail


# --- step_lora ---
#
# 2026-09-26 (Phase 3): the default LORA_CMD is now LORA_MACS_CMD (non-
# <host>) — step_lora's non-<host> branch spawns a REAL child
# process (subprocess.Popen, not subprocess.run) so it is immune to a
# `patch("subprocess.run", ...)` in THIS process (a Popen'd child is a
# separate interpreter). The <host>-specific tests below therefore
# explicitly monkeypatch LORA_CMD to LORA_SSH_CMD so they keep exercising
# (and mocking) the subprocess.run branch — never let a step_lora test in
# this file run under the Macs default without dry_run=True or a
# subprocess.Popen mock (see test_step_lora_runs_the_resolved_override and
# test_step_lora_override_timeout_skips_remote_cancel below, which exercise
# that Popen branch directly).

def test_step_lora_dry_run():
    ok, detail = orch.step_lora(state_file="unused", dry_run=True)
    assert ok is True


def test_step_lora_success(monkeypatch):
    monkeypatch.setattr(orch, "LORA_CMD", orch.LORA_SSH_CMD)
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = ""
    mock_result.stderr = ""
    with patch("subprocess.run", return_value=mock_result):
        ok, detail = orch.step_lora(state_file="unused", dry_run=False)
    assert ok is True


def test_step_lora_failure(monkeypatch):
    monkeypatch.setattr(orch, "LORA_CMD", orch.LORA_SSH_CMD)
    mock_result = MagicMock()
    mock_result.returncode = 3
    mock_result.stdout = "ENGINE FAILED"
    mock_result.stderr = ""
    with patch("subprocess.run", return_value=mock_result):
        ok, detail = orch.step_lora(state_file="unused", dry_run=False)
    assert ok is False
    assert "rc=3" in detail


def test_step_lora_uses_exact_canonical_command(monkeypatch):
    """subprocess.run must be called with LORA_SSH_CMD (shell=True, exact string)."""
    monkeypatch.setattr(orch, "LORA_CMD", orch.LORA_SSH_CMD)
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = ""
    mock_result.stderr = ""
    with patch("subprocess.run", return_value=mock_result) as mock_run:
        orch.step_lora(state_file="unused", dry_run=False)
    call_args = mock_run.call_args
    assert call_args[0][0] == orch.LORA_SSH_CMD, (
        "step_lora must call subprocess.run with the exact canonical LORA_SSH_CMD"
    )
    assert call_args[1].get("shell") is True


# ---------------------------------------------------------------------------
# Full happy-path integration test (all external I/O mocked)
# ---------------------------------------------------------------------------

def test_full_happy_path(tmp_path):
    """Full run: mocked health responses + mocked subprocesses → DONE."""
    lot_csv = str(tmp_path / "s0001_lot4.csv")
    Path(lot_csv).write_text("tic_id,label\n12345,candidate\n")
    sf = str(tmp_path / "pipeline_state.json")

    # Health endpoint always returns "drained"
    drain_resp = _FakeResp(_health_response(
        primary_pending=0, primary_assigned=0, fpp_actionable=0))

    # All subprocesses (postprocess×3, farmctl stop, lora ssh) return 0
    mock_proc = MagicMock()
    mock_proc.returncode = 0
    mock_proc.stdout = ""
    mock_proc.stderr = ""

    args = orch._parse_args([
        "--lot-csv", lot_csv,
        "--sector", "1",
        "--state-file", sf,
        "--db", str(tmp_path / "exonit.db"),
        "--poll-interval", "0",
        "--dry-run",   # skip all real network/subprocess calls
    ])

    orch.run(args)

    state = orch._load_state(sf)
    assert state["state"] == "DONE", f"Expected DONE, got: {state}"
    assert "finished_at" in state


def test_full_happy_path_state_transitions(tmp_path):
    """Verify every state transition is recorded in order (using dry-run)."""
    lot_csv = str(tmp_path / "s0001_lot4.csv")
    Path(lot_csv).write_text("tic_id,label\n12345,candidate\n")
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
    assert seen_states == expected_order, (
        f"Expected transitions: {expected_order}\nGot: {seen_states}"
    )


def test_failed_on_subprocess_failure(tmp_path):
    """A subprocess failure in POSTPROCESS → FAILED state with detail."""
    lot_csv = str(tmp_path / "s0001_lot4.csv")
    Path(lot_csv).write_text("tic_id,label\n12345,candidate\n")
    sf = str(tmp_path / "pipeline_state.json")

    fail = MagicMock()
    fail.returncode = 1
    fail.stdout = ""
    fail.stderr = "fatal error"

    args = orch._parse_args([
        "--lot-csv", lot_csv,
        "--sector", "1",
        "--state-file", sf,
        "--poll-interval", "0",
        "--dry-run",
    ])

    # Override only step_postprocess to simulate failure (everything else dry-run)
    original_postprocess = orch.step_postprocess

    def failing_postprocess(*a, **kw):
        # dry_run is passed through; simulate real failure even in dry_run context
        return False, "postprocess rc=1: fatal error"

    with patch.object(orch, "step_postprocess", side_effect=failing_postprocess):
        orch.run(args)

    state = orch._load_state(sf)
    assert state["state"] == "FAILED"
    assert "postprocess" in state.get("detail", "").lower()


def test_lora_failure_records_nvidiarack_stopped(tmp_path):
    """When LoRA fails (<host> already stopped), note that in state."""
    lot_csv = str(tmp_path / "s0001_lot4.csv")
    Path(lot_csv).write_text("tic_id,label\n12345,candidate\n")
    sf = str(tmp_path / "pipeline_state.json")

    args = orch._parse_args([
        "--lot-csv", lot_csv,
        "--sector", "1",
        "--state-file", sf,
        "--poll-interval", "0",
        "--dry-run",  # everything dry-run except lora which we override
    ])

    def failing_lora(*a, **kw):
        return False, "lora rc=3: ENGINE FAILED"

    with patch.object(orch, "step_lora", side_effect=failing_lora):
        orch.run(args)

    state = orch._load_state(sf)
    assert state["state"] == "FAILED"
    assert state.get("nvidiarack_worker_stopped") is True


# ---------------------------------------------------------------------------
# GET /api/discovery/pipeline endpoint tests (via FastAPI TestClient)
# ---------------------------------------------------------------------------

@pytest.fixture
def pipeline_client(tmp_path, monkeypatch):
    """FastAPI TestClient with isolated DB + temp state file."""
    db_file = tmp_path / "test.db"
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir(parents=True)

    monkeypatch.setenv("EXONIT_DB_PATH", str(db_file))
    monkeypatch.setenv("EXONIT_REPORTS_DIR", str(reports_dir))
    # Isolate the orchestrator state file too: the endpoint reads
    # EXONIT_PIPELINE_STATE_FILE (default data/discovery/pipeline_state.json).
    # Without this, tests read the LIVE state file and break when a real run
    # has left it at DONE. Point it inside tmp_path (absent by default → IDLE).
    monkeypatch.setenv(
        "EXONIT_PIPELINE_STATE_FILE",
        str(tmp_path / "discovery" / "pipeline_state.json"),
    )

    from exonit.server import create_app
    app = create_app()

    from fastapi.testclient import TestClient
    with TestClient(app) as client:
        yield client, tmp_path


def test_pipeline_endpoint_no_file(pipeline_client):
    """GET /api/discovery/pipeline returns IDLE when state file absent."""
    client, _ = pipeline_client
    r = client.get("/api/discovery/pipeline")
    assert r.status_code == 200
    data = r.json()
    assert data["state"] == "IDLE"
    assert data.get("never_run") is True


def test_pipeline_endpoint_with_state_file(pipeline_client):
    """GET /api/discovery/pipeline returns content from state file.

    The fixture points EXONIT_PIPELINE_STATE_FILE inside tmp_path, so we just
    write the state there — no need to touch (or guard against) live data.
    """
    client, tmp_path = pipeline_client

    state_content = {"state": "SCANNING", "lot_csv": "/tmp/lot4.csv"}
    state_path = tmp_path / "discovery" / "pipeline_state.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state_content))

    r = client.get("/api/discovery/pipeline")
    assert r.status_code == 200
    data = r.json()
    assert data["state"] == "SCANNING"
    assert data["lot_csv"] == "/tmp/lot4.csv"


def test_pipeline_endpoint_malformed_json(pipeline_client):
    """Malformed state file returns IDLE gracefully (no 500)."""
    client, tmp_path = pipeline_client

    state_path = tmp_path / "discovery" / "pipeline_state.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text("{ INVALID JSON <<<")

    r = client.get("/api/discovery/pipeline")
    assert r.status_code == 200
    data = r.json()
    assert data["state"] == "IDLE"
    assert data.get("parse_error") is True


# ---------------------------------------------------------------------------
# _lot_number_from_csv — naming fix (anti-collision for auto-generated lots)
# ---------------------------------------------------------------------------

def test_lot_number_auto_returns_timestamp():
    """Auto lot s0001_lot_20260531T082820.csv → '20260531T082820'."""
    result = orch._lot_number_from_csv("data/discovery/s0001_lot_20260531T082820.csv")
    assert result == "20260531T082820"


def test_lot_number_auto_two_same_day_no_collision():
    """Two auto lots generated the same day produce DIFFERENT identifiers.

    This is the core regression: before the fix both returned 'X', causing
    run_id and scorecard/dossier paths to collide.
    """
    id1 = orch._lot_number_from_csv("data/discovery/s0001_lot_20260531T082820.csv")
    id2 = orch._lot_number_from_csv("data/discovery/s0001_lot_20260531T235959.csv")
    assert id1 != id2, (
        f"Two auto lots on the same day must produce different identifiers, "
        f"got {id1!r} and {id2!r}"
    )


def test_lot_number_numbered_lot_unchanged():
    """Numbered lot s0001_lot4.csv → '4' (no regression)."""
    assert orch._lot_number_from_csv("data/discovery/s0001_lot4.csv") == "4"
    assert orch._lot_number_from_csv("/some/path/s0001_lot12.csv") == "12"


def test_lot_number_fallback():
    """Unrecognised filename → 'X'."""
    assert orch._lot_number_from_csv("weird.csv") == "X"
    assert orch._lot_number_from_csv("data/discovery/nolotnumber.csv") == "X"


def test_lot_number_run_id_unique_for_auto_lots():
    """run_id built from auto lots on the same day must be unique."""
    sector_str = "0001"
    today = "2026-05-31"
    id1 = orch._lot_number_from_csv("data/discovery/s0001_lot_20260531T082820.csv")
    id2 = orch._lot_number_from_csv("data/discovery/s0001_lot_20260531T195000.csv")
    run_id1 = f"discovery_s{sector_str}_lot{id1}_{today}"
    run_id2 = f"discovery_s{sector_str}_lot{id2}_{today}"
    assert run_id1 != run_id2, (
        f"run_ids must differ: {run_id1!r} vs {run_id2!r}"
    )


# ---------------------------------------------------------------------------
# FIX-1 (final-review, spec §4): distinctive deep_lot run_id for deep-rescan
# lots -- _lot_prefix_from_csv + deep-aware _derive_run_id.
# ---------------------------------------------------------------------------

def test_lot_number_unchanged_for_deep_lot_csv():
    """_lot_number_from_csv's bare ts extraction stays byte-identical even
    for deep-rescan csv names (the regex already matches the embedded
    'lot_<ts>' substring inside 's0014_deep_lot_<ts>.csv'); the deep/normal
    distinction lives in the separate _lot_prefix_from_csv helper."""
    assert (orch._lot_number_from_csv(
        "data/discovery/s0014_deep_lot_20260723T101010.csv")
        == "20260723T101010")


def test_lot_prefix_from_csv_deep():
    assert orch._lot_prefix_from_csv(
        "data/discovery/s0014_deep_lot_20260723T101010.csv") == "deep_lot"


def test_lot_prefix_from_csv_normal_unchanged():
    """Normal lots (timestamp or numbered) keep the 'lot' prefix."""
    assert orch._lot_prefix_from_csv(
        "data/discovery/s0001_lot_20260531T082820.csv") == "lot"
    assert orch._lot_prefix_from_csv("data/discovery/s0001_lot4.csv") == "lot"


def test_derive_run_id_deep_lot_is_distinctive():
    """Spec §4: deep-rescan lots mint discovery_s<sector>_deep_lot<ts>_<date>,
    separable from normal campaign lots at a glance."""
    run_id = orch._derive_run_id(
        14, "data/discovery/s0014_deep_lot_20260723T101010.csv")
    assert run_id.startswith("discovery_s0014_deep_lot20260723T101010_")


def test_derive_run_id_normal_lot_byte_identical_to_before():
    """Normal (non-deep) lots must derive the exact same run_id as before
    this fix -- no regression for existing campaign lots."""
    from datetime import datetime, timezone
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    run_id = orch._derive_run_id(
        14, "data/discovery/s0014_lot_20260723T101010.csv")
    assert run_id == f"discovery_s0014_lot20260723T101010_{today}"


class TestFinalizeProgressEnv:
    """The FINALIZE child must inherit the progress identity."""

    def test_remote_compact_command_carries_progress_env(self, monkeypatch):
        import scripts.discovery_orchestrator as ORCH

        # The URL comes from --server-url, published into _SERVER_URL at the
        # top of run(). It must be LAN-reachable for the remote child to
        # report at all — see TestProgressUrlResolution.
        monkeypatch.delenv("EXONIT_PROGRESS_URL", raising=False)
        monkeypatch.setattr(ORCH, "_SERVER_URL", "http://10.0.0.76:8400")
        cmd = ORCH._compact_remote_cmd("<host>", "postgresql://x/y",
                                       run_id="run-A")
        remote = cmd[-1]
        assert "EXONIT_PROGRESS_RUN_ID=run-A" in remote
        assert "EXONIT_PROGRESS_PHASE=FINALIZE" in remote
        assert "EXONIT_PROGRESS_URL=http://10.0.0.76:8400" in remote

    def test_remote_compact_command_unchanged_without_a_run_id(self):
        """No run_id (a hand-run compaction) must leave the command as it was."""
        import scripts.discovery_orchestrator as ORCH

        cmd = ORCH._compact_remote_cmd("<host>", "postgresql://x/y")
        assert "EXONIT_PROGRESS_" not in cmd[-1]

    def test_local_compact_fallback_carries_progress_env(self, monkeypatch):
        import exonit.completeness_store as CS
        import scripts.discovery_orchestrator as ORCH

        # Two preconditions this helper returns early on, both of which have
        # to hold for the local path to run at all: the completeness-on-PG
        # flag, and NO remote host (which would take the ssh fast path).
        monkeypatch.setattr(CS, "completeness_pg_enabled", lambda: True)
        monkeypatch.delenv("EXONIT_COMPACT_REMOTE_HOST", raising=False)

        captured = {}

        def fake_run(cmd, **kw):
            captured["env"] = kw.get("env") or {}
            import subprocess as sp
            return sp.CompletedProcess(cmd, 0, "{}", "")

        monkeypatch.setattr(ORCH.subprocess, "run", fake_run)
        ORCH._finalize_compact_completeness("postgresql://x/y", run_id="run-A")
        assert captured["env"]["EXONIT_PROGRESS_RUN_ID"] == "run-A"
        assert captured["env"]["EXONIT_PROGRESS_PHASE"] == "FINALIZE"
        # A full env, not a bare triplet — the child still needs PATH etc.
        assert "PATH" in captured["env"]



class TestProgressUrlResolution:
    """The URL a progress client posts to — the piece with no test coverage
    that silently disabled the whole feature (2026-07-27 review)."""

    def _reset(self, monkeypatch, server_url=""):
        import scripts.discovery_orchestrator as ORCH
        monkeypatch.delenv("EXONIT_PROGRESS_URL", raising=False)
        monkeypatch.delenv("EXONIT_SERVER_URL", raising=False)
        monkeypatch.setattr(ORCH, "_SERVER_URL", server_url)
        return ORCH

    def test_no_url_anywhere_means_do_not_report(self, monkeypatch):
        ORCH = self._reset(monkeypatch)
        assert ORCH._progress_url() == ""
        assert ORCH._progress_url(remote=True) == ""

    def test_local_child_may_use_a_localhost_server_url(self, monkeypatch):
        """The server really does pass http://localhost:8400, and for a child
        on this same host that is correct."""
        ORCH = self._reset(monkeypatch, "http://localhost:8400")
        assert ORCH._progress_url() == "http://localhost:8400"

    def test_remote_child_refuses_localhost(self, monkeypatch):
        """On <host>, localhost is <host> — the marks would vanish."""
        ORCH = self._reset(monkeypatch, "http://localhost:8400")
        assert ORCH._progress_url(remote=True) == ""

    def test_remote_child_accepts_a_lan_address(self, monkeypatch):
        ORCH = self._reset(monkeypatch, "http://10.0.0.76:8400")
        assert ORCH._progress_url(remote=True) == "http://10.0.0.76:8400"

    def test_explicit_override_wins_everywhere(self, monkeypatch):
        ORCH = self._reset(monkeypatch, "http://localhost:8400")
        monkeypatch.setenv("EXONIT_PROGRESS_URL", "http://10.0.0.76:8400")
        assert ORCH._progress_url(remote=True) == "http://10.0.0.76:8400"

    def test_remote_compact_command_omits_progress_when_url_unusable(self, monkeypatch):
        """No URL the remote host can reach => no env prefix at all, rather
        than a prefix pointing the remote host at itself."""
        ORCH = self._reset(monkeypatch, "http://localhost:8400")
        cmd = ORCH._compact_remote_cmd("<host>", "postgresql://x/y",
                                       run_id="run-A")
        assert "EXONIT_PROGRESS_" not in cmd[-1]


class TestPhaseReportsItself:
    def test_queue_phases_are_left_to_the_server(self):
        import scripts.discovery_orchestrator as ORCH
        for phase in ("SCANNING", "FPP_BACKFILL", "EXOMINER_BACKFILL", "LORA"):
            assert ORCH._phase_reports_itself(phase) is False

    def test_free_gpu_has_nothing_to_report(self):
        import scripts.discovery_orchestrator as ORCH
        assert ORCH._phase_reports_itself("FREE_GPU") is False

    def test_the_opaque_phases_do_report(self):
        import scripts.discovery_orchestrator as ORCH
        for phase in ("STOP_WORKERS", "POSTPROCESS", "STOP_NVIDIARACK",
                      "FINALIZE", "AWS_FINALIZE"):
            assert ORCH._phase_reports_itself(phase) is True

    def test_phase_reporter_is_none_without_a_run_id(self):
        import scripts.discovery_orchestrator as ORCH
        assert ORCH._phase_reporter("STOP_WORKERS", None) is None


# ---------------------------------------------------------------------------
# 2026-09-20: LoRA step scoped to the run's sector (s5 lot 1 incident —
# whole-DB generator, 222k rows, 52 h vs the 24 h step timeout).
# ---------------------------------------------------------------------------

def test_lora_cmd_for_sector_appends_positional_arg_on_default_path(monkeypatch):
    monkeypatch.setattr(orch, "LORA_CMD", orch.LORA_SSH_CMD)
    cmd = orch._lora_cmd_for_sector(5)
    assert cmd == orch.LORA_SSH_CMD.replace(
        "lora_via_gpuq.sh'", "lora_via_gpuq.sh 5'")
    # still a single, well-formed ssh + bash -lc '...' invocation
    assert cmd.endswith(f"bash {orch.GPUQ_LORA_SCRIPT} 5'\"")
    assert cmd.count("lora_via_gpuq.sh") == 1


def test_lora_cmd_for_sector_none_is_canonical(monkeypatch):
    monkeypatch.setattr(orch, "LORA_CMD", orch.LORA_SSH_CMD)
    assert orch._lora_cmd_for_sector(None) == orch.LORA_SSH_CMD


def test_lora_cmd_for_sector_leaves_override_untouched(monkeypatch):
    monkeypatch.setattr(orch, "LORA_CMD", "bash ops/lora_via_llamacpp.sh")
    assert orch._lora_cmd_for_sector(5) == "bash ops/lora_via_llamacpp.sh"


def test_lora_cmd_for_sector_int_casts(monkeypatch):
    monkeypatch.setattr(orch, "LORA_CMD", orch.LORA_SSH_CMD)
    assert orch._lora_cmd_for_sector("14").endswith("lora_via_gpuq.sh 14'\"")
    with pytest.raises(ValueError):
        orch._lora_cmd_for_sector("5; rm -rf /")


def test_step_lora_runs_sector_scoped_cmd_and_exports_env(monkeypatch):
    monkeypatch.setattr(orch, "LORA_CMD", orch.LORA_SSH_CMD)
    seen = {}

    def fake_run(cmd, **kw):
        seen["cmd"] = cmd
        seen["env"] = kw.get("env")
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(orch.subprocess, "run", fake_run)
    ok, msg = orch.step_lora("unused_state_file", dry_run=False, sector=5)
    assert ok and msg == ""
    assert seen["cmd"].endswith("lora_via_gpuq.sh 5'\"")
    assert seen["env"]["EXONIT_LORA_SECTOR"] == "5"


def test_step_lora_without_sector_keeps_legacy_whole_db_cmd(monkeypatch):
    monkeypatch.setattr(orch, "LORA_CMD", orch.LORA_SSH_CMD)
    seen = {}

    def fake_run(cmd, **kw):
        seen["cmd"] = cmd
        seen["env"] = kw.get("env")
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(orch.subprocess, "run", fake_run)
    ok, _ = orch.step_lora("unused_state_file", dry_run=False)
    assert ok
    assert seen["cmd"] == orch.LORA_SSH_CMD
    assert "EXONIT_LORA_SECTOR" not in seen["env"]


# ---------------------------------------------------------------------------
# 2026-09-20 (operator): backlog pass — LORA also drains up to N historic rows.
# ---------------------------------------------------------------------------

def test_lora_cmd_backlog_appends_second_positional_arg(monkeypatch):
    monkeypatch.setattr(orch, "LORA_CMD", orch.LORA_SSH_CMD)
    cmd = orch._lora_cmd_for_sector(5, 50000)
    assert cmd.endswith(f"bash {orch.GPUQ_LORA_SCRIPT} 5 50000'\"")
    assert cmd == orch.LORA_SSH_CMD.replace(
        "lora_via_gpuq.sh'", "lora_via_gpuq.sh 5 50000'")


def test_lora_cmd_backlog_zero_is_sector_only(monkeypatch):
    monkeypatch.setattr(orch, "LORA_CMD", orch.LORA_SSH_CMD)
    assert orch._lora_cmd_for_sector(5, 0) == orch._lora_cmd_for_sector(5)
    assert orch._lora_cmd_for_sector(5, 0).endswith("lora_via_gpuq.sh 5'\"")


def test_lora_cmd_backlog_without_sector_is_ignored(monkeypatch):
    # A whole-DB pass already covers the backlog: no args at all.
    monkeypatch.setattr(orch, "LORA_CMD", orch.LORA_SSH_CMD)
    assert orch._lora_cmd_for_sector(None, 50000) == orch.LORA_SSH_CMD


def test_lora_cmd_backlog_int_casts(monkeypatch):
    monkeypatch.setattr(orch, "LORA_CMD", orch.LORA_SSH_CMD)
    assert orch._lora_cmd_for_sector(5, "50000").endswith("5 50000'\"")
    with pytest.raises(ValueError):
        orch._lora_cmd_for_sector(5, "50000; rm -rf /")


def test_step_lora_uses_module_default_backlog_limit_and_env(monkeypatch):
    monkeypatch.setattr(orch, "LORA_CMD", orch.LORA_SSH_CMD)
    monkeypatch.setattr(orch, "LORA_BACKLOG_LIMIT", 50000)
    seen = {}

    def fake_run(cmd, **kw):
        seen["cmd"] = cmd
        seen["env"] = kw.get("env")
        return types.SimpleNamespace(
            returncode=0, stderr="",
            stdout="[job] x\n[job] generator done rc=0 (sector pass)\n"
                   "[job] backlog pass done rc=0 (non-fatal; job rc stays 0)\n")

    monkeypatch.setattr(orch.subprocess, "run", fake_run)
    ok, msg = orch.step_lora("unused", dry_run=False, sector=5)
    assert ok and msg == ""
    assert seen["cmd"].endswith("lora_via_gpuq.sh 5 50000'\"")
    assert seen["env"]["EXONIT_LORA_SECTOR"] == "5"
    assert seen["env"]["EXONIT_LORA_BACKLOG_LIMIT"] == "50000"


def test_step_lora_explicit_backlog_overrides_module_default(monkeypatch):
    monkeypatch.setattr(orch, "LORA_CMD", orch.LORA_SSH_CMD)
    monkeypatch.setattr(orch, "LORA_BACKLOG_LIMIT", 50000)
    seen = {}

    def fake_run(cmd, **kw):
        seen["cmd"] = cmd
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(orch.subprocess, "run", fake_run)
    ok, _ = orch.step_lora("unused", dry_run=False, sector=5, backlog_limit=0)
    assert ok
    assert seen["cmd"].endswith("lora_via_gpuq.sh 5'\"")


def test_lora_backlog_limit_env_parsing(monkeypatch):
    import importlib
    monkeypatch.setenv("EXONIT_LORA_BACKLOG_LIMIT", "50000")
    mod = importlib.reload(orch)
    try:
        assert mod.LORA_BACKLOG_LIMIT == 50000
        monkeypatch.setenv("EXONIT_LORA_BACKLOG_LIMIT", "")
        assert importlib.reload(orch).LORA_BACKLOG_LIMIT == 0
        monkeypatch.delenv("EXONIT_LORA_BACKLOG_LIMIT")
        assert importlib.reload(orch).LORA_BACKLOG_LIMIT == 0
    finally:
        importlib.reload(orch)
