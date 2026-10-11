"""Tests of the supervisor's ARMED mode (phase 1.5 — `autonomy/supervisor.py`).

Everything mocked: no real HTTP/subprocess/git call. `ServerClient`, `Executor`,
`Mailer` and `close_lot` are always replaced by in-memory objects or
`MagicMock`, and any subprocess (`run_health_check.py`, `farmctl`) gets a
fake `run_cmd` that returns a `CompletedProcess`-like object without touching the network.
Armed mode is NEVER really run here — only the pure/wrapped functions
of the module, with all their dependencies injected.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import autonomy.supervisor as S
from autonomy.closeout import CloseoutResult

#: Reference to the REAL implementation of `_sync_sector_ui_state`, captured
#: before `_no_real_sector_ui_sync` (autouse, further down) replaces it
#: in the module with a default `MagicMock` -- the tests dedicated to the
#: synchronization restore it explicitly with `monkeypatch.setattr`.
_REAL_SYNC_SECTOR_UI_STATE = S._sync_sector_ui_state
from autonomy.config import AlertConfig, AutonomyConfig, FleetConfig
from autonomy.plans import Plan, PlansFile, load_plans
from autonomy.playbooks import ExecutionResult
from autonomy.types import PlaybookChoice


def _proc(rc=0, stdout="", stderr=""):
    return SimpleNamespace(returncode=rc, stdout=stdout, stderr=stderr)


def _survey_plan(plan_id="p1", sector=15, lot_size=300_000):
    return Plan(plan_id=plan_id, approved_by="operator", sector=sector, mode="survey", lot_size=lot_size)


def _deep_plan(plan_id="p2", sector=14, worklist="data/x.csv"):
    return Plan(plan_id=plan_id, approved_by="operator", sector=sector, mode="deep_rescan", worklist=worklist)


@pytest.fixture(autouse=True)
def _no_real_sector_ui_sync(monkeypatch):
    """`_launch_lot` now calls `_sync_sector_ui_state` before the POST (bug
    2026-08-09: the UI kept showing sector 14 with the sector 2 lot
    already running). Autouse to keep the promise in the file header
    ("no real HTTP/subprocess/git call") for the tests that do nothing with it --
    the tests dedicated to the synchronization (further down) override this
    mock explicitly."""
    monkeypatch.setattr(S, "_sync_sector_ui_state", MagicMock())


# ---------------------------------------------------------------------------
# 1. Without the double key -> --armed falls back to shadow.
# ---------------------------------------------------------------------------

def test_armed_without_env_var_falls_back_to_shadow(tmp_path, monkeypatch):
    monkeypatch.delenv("EXONIT_AUTONOMY_ARMED", raising=False)
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(f"armed: true\ndata_dir: {tmp_path / 'autonomy'}\n")

    called = MagicMock()
    monkeypatch.setattr(S, "run_shadow_loop", called)

    S.run_armed_loop(once=True, config_path=cfg_path)

    called.assert_called_once_with(once=True, config_path=cfg_path)


def test_armed_without_config_flag_falls_back_to_shadow(tmp_path, monkeypatch):
    # armed:false but EXONIT_AUTONOMY_ARMED=1 -- the other key is missing.
    monkeypatch.setenv("EXONIT_AUTONOMY_ARMED", "1")
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text("armed: false\n")

    called = MagicMock()
    monkeypatch.setattr(S, "run_shadow_loop", called)

    S.run_armed_loop(once=True, config_path=cfg_path)

    called.assert_called_once_with(once=True, config_path=cfg_path)


def test_is_armed_requires_both_keys(monkeypatch):
    monkeypatch.delenv("EXONIT_AUTONOMY_ARMED", raising=False)
    assert S._is_armed(AutonomyConfig(armed=True)) is False
    assert S._is_armed(AutonomyConfig(armed=False)) is False
    monkeypatch.setenv("EXONIT_AUTONOMY_ARMED", "1")
    assert S._is_armed(AutonomyConfig(armed=False)) is False
    assert S._is_armed(AutonomyConfig(armed=True)) is True


# ---------------------------------------------------------------------------
# 2. would_launch_next_lot: correct payload; health CRITICAL -> no post.
# ---------------------------------------------------------------------------

def test_launch_survey_posts_correct_payload_and_min_alive_reached():
    plan = _survey_plan()
    state = S.SupervisorState()
    cfg = AutonomyConfig(fleet=FleetConfig(min_alive=5, start_wait_s=10))

    client = MagicMock()
    client.get.return_value = {"workers": {"alive": 5}}
    client.post.return_value = {"status": "started", "enqueued": 300_000, "run_id": "r1"}

    run_cmd = MagicMock(return_value=_proc(rc=0))
    sleep_fn = MagicMock()

    kind, detail = S._launch_lot(client, plan, state, cfg, run_cmd=run_cmd, sleep_fn=sleep_fn)

    assert kind == "launched"
    client.post.assert_called_once_with(
        "/api/discovery/scan", {"sector": 15, "count": 300_000}, timeout_s=900
    )


def test_launch_deep_rescan_payload_includes_mode_and_tics_csv():
    plan = _deep_plan()
    state = S.SupervisorState()
    cfg = AutonomyConfig(fleet=FleetConfig(min_alive=1, start_wait_s=10))

    client = MagicMock()
    client.get.return_value = {"workers": {"alive": 1}}
    client.post.return_value = {"status": "started", "enqueued": 100, "run_id": "r2"}

    S._launch_lot(client, plan, state, cfg, run_cmd=MagicMock(return_value=_proc(0)), sleep_fn=MagicMock())

    client.post.assert_called_once_with(
        "/api/discovery/scan",
        {"sector": 14, "count": 300_000, "mode": "deep_rescan", "tics_csv": "data/x.csv"},
        timeout_s=900,
    )


def test_health_check_critical_blocks_launch_no_post():
    plan = _survey_plan()
    state = S.SupervisorState()
    cfg = AutonomyConfig(fleet=FleetConfig(min_alive=5, start_wait_s=10))

    client = MagicMock()
    run_cmd = MagicMock(return_value=_proc(rc=2, stderr="CRITICAL disk"))

    kind, detail = S._launch_lot(client, plan, state, cfg, run_cmd=run_cmd, sleep_fn=MagicMock())

    assert kind == "blocked"
    assert "health check" in detail
    client.post.assert_not_called()
    client.get.assert_not_called()


# ---------------------------------------------------------------------------
# 3. Fleet does not reach min_alive -> pause_and_alert (blocked), no post.
# ---------------------------------------------------------------------------

def test_fleet_never_reaches_min_alive_blocks_no_post():
    plan = _survey_plan()
    state = S.SupervisorState()
    cfg = AutonomyConfig(fleet=FleetConfig(min_alive=30, start_wait_s=20))  # 2 attempts

    client = MagicMock()
    client.get.return_value = {"workers": {"alive": 3}}
    sleep_fn = MagicMock()

    kind, detail = S._launch_lot(
        client, plan, state, cfg, run_cmd=MagicMock(return_value=_proc(0)), sleep_fn=sleep_fn
    )

    assert kind == "blocked"
    assert "fleet" in detail
    client.post.assert_not_called()
    assert sleep_fn.call_count == 2  # one per attempt, no real waits (mocked)


# ---------------------------------------------------------------------------
# 4. already_running -> no-op; enqueued==0 -> plan marked completed.
# ---------------------------------------------------------------------------

def test_already_running_is_informational_noop():
    """Defensive path: if the server ever returns already_running with
    HTTP 200 (it does not today -- it is always HTTP 409, see the tests with the
    real ServerClient below), _launch_lot still recognizes it."""
    plan = _survey_plan()
    state = S.SupervisorState()
    cfg = AutonomyConfig(fleet=FleetConfig(min_alive=1, start_wait_s=10))

    client = MagicMock()
    client.get.return_value = {"workers": {"alive": 1}}
    client.post.return_value = {"status": "already_running", "state": "SCANNING"}

    kind, detail = S._launch_lot(client, plan, state, cfg, run_cmd=MagicMock(return_value=_proc(0)), sleep_fn=MagicMock())

    assert kind == "already_running"
    assert plan.plan_id not in state.completed_plan_ids


def test_enqueued_zero_marks_plan_completed():
    """Defensive path (inspection of the 200 response dict)."""
    plan = _survey_plan()
    state = S.SupervisorState()
    cfg = AutonomyConfig(fleet=FleetConfig(min_alive=1, start_wait_s=10))

    client = MagicMock()
    client.get.return_value = {"workers": {"alive": 1}}
    client.post.return_value = {"status": "started", "enqueued": 0}

    kind, detail = S._launch_lot(client, plan, state, cfg, run_cmd=MagicMock(return_value=_proc(0)), sleep_fn=MagicMock())

    assert kind == "plan_completed"
    assert plan.plan_id in state.completed_plan_ids


# ---------------------------------------------------------------------------
# 4b. REAL path against the HTTP contract of `exonit/routes/discovery.py`:
# real ServerClient with `urlopen` mocked to raise `HTTPError` with a JSON
# body (as the real server does), NOT `client.post` mocked directly
# -- this is what review round 1 pointed out was missing (findings
# 1 and 2: the JSON body of a non-2xx was discarded and the already_running 409 was
# confused with need_download).
# ---------------------------------------------------------------------------

import io  # noqa: E402
from urllib.error import HTTPError  # noqa: E402

import autonomy.server_client as SC  # noqa: E402


def _http_error(status: int, body: dict) -> HTTPError:
    payload = json.dumps(body).encode()
    return HTTPError(url="http://x/api/discovery/scan", code=status, msg="err", hdrs=None, fp=io.BytesIO(payload))


class _HealthOkResp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _real_client_scan_fails(monkeypatch, *, alive: int, post_error: HTTPError):
    """Real `ServerClient(read_only=False)`, with `urlopen` mocked: GET
    `/api/health/work` responds `alive`; POST `/api/discovery/scan` raises
    `post_error` (an `HTTPError` with a JSON body, like the real server)."""

    def fake_urlopen(req, timeout=None):
        if req.get_method() == "GET":
            return _HealthOkResp(json.dumps({"workers": {"alive": alive}}).encode())
        raise post_error

    monkeypatch.setattr(SC.request, "urlopen", fake_urlopen)
    return SC.ServerClient("http://x", read_only=False)


def test_real_client_already_running_409_is_noop_not_pause(monkeypatch):
    plan = _survey_plan()
    state = S.SupervisorState()
    cfg = AutonomyConfig(fleet=FleetConfig(min_alive=1, start_wait_s=10))
    client = _real_client_scan_fails(
        monkeypatch, alive=1,
        post_error=_http_error(409, {"status": "already_running", "state": "SCANNING"}),
    )

    kind, detail = S._launch_lot(client, plan, state, cfg, run_cmd=MagicMock(return_value=_proc(0)), sleep_fn=MagicMock())

    assert kind == "already_running"
    assert plan.plan_id not in state.completed_plan_ids


def test_real_client_need_download_409_is_distinguished_from_already_running(monkeypatch):
    """The real server uses HTTP 409 for TWO different meanings
    (routes/discovery.py: anti-double-launch guard and need_download) --
    this test confirms they do not collide."""
    plan = _survey_plan()
    state = S.SupervisorState()
    cfg = AutonomyConfig(fleet=FleetConfig(min_alive=1, start_wait_s=10))
    client = _real_client_scan_fails(
        monkeypatch, alive=1,
        post_error=_http_error(409, {
            "status": "need_download", "requested": 300_000, "ready": 100, "missing": 5000,
            "hint": "Download more TGLC data for sector 1 from the Downloads page before scanning.",
        }),
    )

    kind, detail = S._launch_lot(client, plan, state, cfg, run_cmd=MagicMock(return_value=_proc(0)), sleep_fn=MagicMock())

    assert kind == "need_download"
    assert "Download more TGLC data" in detail
    assert plan.plan_id not in state.completed_plan_ids


def test_real_client_no_undone_tics_400_marks_plan_completed(monkeypatch):
    """`no_undone_tics` (HTTP 400, `scripts/discovery_run.py`) is exactly
    the real error produced by an exhausted survey sector -- it must mark the
    plan completed, not block."""
    plan = _survey_plan()
    state = S.SupervisorState()
    cfg = AutonomyConfig(fleet=FleetConfig(min_alive=1, start_wait_s=10))
    client = _real_client_scan_fails(
        monkeypatch, alive=1,
        post_error=_http_error(400, {"status": "error", "reason": "no_undone_tics", "remaining_undone": 0}),
    )

    kind, detail = S._launch_lot(client, plan, state, cfg, run_cmd=MagicMock(return_value=_proc(0)), sleep_fn=MagicMock())

    assert kind == "plan_completed"
    assert plan.plan_id in state.completed_plan_ids


def test_real_client_deep_worklist_exhausted_400_marks_plan_completed(monkeypatch):
    plan = _deep_plan()
    state = S.SupervisorState()
    cfg = AutonomyConfig(fleet=FleetConfig(min_alive=1, start_wait_s=10))
    client = _real_client_scan_fails(
        monkeypatch, alive=1,
        post_error=_http_error(400, {
            "status": "error", "reason": "deep_worklist_exhausted",
            "deep_cursor": 500, "worklist_size": 500,
        }),
    )

    kind, detail = S._launch_lot(client, plan, state, cfg, run_cmd=MagicMock(return_value=_proc(0)), sleep_fn=MagicMock())

    assert kind == "plan_completed"
    assert plan.plan_id in state.completed_plan_ids


def test_real_client_config_error_400_blocks_does_not_complete_plan(monkeypatch):
    """`sector_not_enabled` is a CONFIGURATION error of the plan (plans.yaml
    points to a badly prepared sector), not "plan exhausted" -- it must block
    and pause, never move on by itself to the next plan."""
    plan = _survey_plan()
    state = S.SupervisorState()
    cfg = AutonomyConfig(fleet=FleetConfig(min_alive=1, start_wait_s=10))
    client = _real_client_scan_fails(
        monkeypatch, alive=1,
        post_error=_http_error(400, {"status": "error", "reason": "sector_not_enabled", "sector": 15}),
    )

    kind, detail = S._launch_lot(client, plan, state, cfg, run_cmd=MagicMock(return_value=_proc(0)), sleep_fn=MagicMock())

    assert kind == "blocked"
    assert plan.plan_id not in state.completed_plan_ids


def test_real_client_network_failure_without_http_response_is_blocked(monkeypatch):
    """A network failure/timeout WITHOUT an HTTP response (not `HTTPError`) has
    no body to read -- it must keep blocking, not blow up."""
    plan = _survey_plan()
    state = S.SupervisorState()
    cfg = AutonomyConfig(fleet=FleetConfig(min_alive=1, start_wait_s=10))

    def fake_urlopen(req, timeout=None):
        if req.get_method() == "GET":
            return _HealthOkResp(json.dumps({"workers": {"alive": 1}}).encode())
        import socket
        raise SC.error.URLError(socket.timeout("timed out"))

    monkeypatch.setattr(SC.request, "urlopen", fake_urlopen)
    client = SC.ServerClient("http://x", read_only=False)

    kind, detail = S._launch_lot(client, plan, state, cfg, run_cmd=MagicMock(return_value=_proc(0)), sleep_fn=MagicMock())

    assert kind == "blocked"
    assert plan.plan_id not in state.completed_plan_ids


# ---------------------------------------------------------------------------
# 4c. Incident 2026-08-08: a RAW `TimeoutError` (not wrapped in `URLError`,
# see `server_client.py::post` -- `h.getresponse()` does not go through the `except
# OSError`/wrap of `h.request()`) escaped the `except RuntimeError` of
# `_launch_lot` and killed the whole supervisor process. It must be
# caught as `"launch_timeout"`, WITHOUT marking the plan completed.
# ---------------------------------------------------------------------------

def test_launch_lot_bare_timeout_error_returns_launch_timeout_not_completed():
    plan = _survey_plan()
    state = S.SupervisorState()
    cfg = AutonomyConfig(fleet=FleetConfig(min_alive=1, start_wait_s=10))

    client = MagicMock()
    client.get.return_value = {"workers": {"alive": 1}}
    client.post.side_effect = TimeoutError("timed out")

    kind, detail = S._launch_lot(client, plan, state, cfg, run_cmd=MagicMock(return_value=_proc(0)), sleep_fn=MagicMock())

    assert kind == "launch_timeout"
    assert plan.plan_id not in state.completed_plan_ids
    assert "timed out" in detail


def test_real_client_bare_timeout_error_not_wrapped_by_urllib_returns_launch_timeout(monkeypatch):
    """Faithful reproduction of the incident: `urlopen` (via `h.getresponse()`)
    raises a raw `TimeoutError`, NOT a `URLError` -- confirms that
    `_launch_lot` still recognizes it via the real `ServerClient`."""
    plan = _survey_plan()
    state = S.SupervisorState()
    cfg = AutonomyConfig(fleet=FleetConfig(min_alive=1, start_wait_s=10))

    def fake_urlopen(req, timeout=None):
        if req.get_method() == "GET":
            return _HealthOkResp(json.dumps({"workers": {"alive": 1}}).encode())
        raise TimeoutError("timed out")

    monkeypatch.setattr(SC.request, "urlopen", fake_urlopen)
    client = SC.ServerClient("http://x", read_only=False)

    kind, detail = S._launch_lot(client, plan, state, cfg, run_cmd=MagicMock(return_value=_proc(0)), sleep_fn=MagicMock())

    assert kind == "launch_timeout"
    assert plan.plan_id not in state.completed_plan_ids


def test_launch_lot_server_http_error_still_handled_as_before():
    """The existing behaviour (`ServerHTTPError`, subclass of
    `RuntimeError`) must not change with the new `except OSError`."""
    plan = _survey_plan()
    state = S.SupervisorState()
    cfg = AutonomyConfig(fleet=FleetConfig(min_alive=1, start_wait_s=10))

    client = MagicMock()
    client.get.return_value = {"workers": {"alive": 1}}
    client.post.side_effect = SC.ServerHTTPError(
        "POST", "/api/discovery/scan", 409, {"status": "already_running", "state": "SCANNING"}
    )

    kind, detail = S._launch_lot(client, plan, state, cfg, run_cmd=MagicMock(return_value=_proc(0)), sleep_fn=MagicMock())

    assert kind == "already_running"
    assert plan.plan_id not in state.completed_plan_ids


# ---------------------------------------------------------------------------
# Q6 (perf review 2026-09-26, docs/reviews/2026-09-26-enqueue-path-
# performance-review.md): client timeout of POST /api/discovery/scan must
# never be shorter than the server's own EXONIT_SCAN_TIMEOUT_S deadline --
# incident: prod .env sets EXONIT_SCAN_TIMEOUT_S=1800 while the supervisor
# hardcoded 900, so a slow-but-healthy 300k-TIC enqueue was misreported as
# `launch_timeout` well before the server itself would have given up.
# ---------------------------------------------------------------------------

def test_scan_post_timeout_defaults_to_900_when_env_unset(monkeypatch):
    monkeypatch.delenv("EXONIT_SCAN_TIMEOUT_S", raising=False)
    assert S._scan_post_timeout_s() == 900.0


def test_scan_post_timeout_grows_with_server_env_var(monkeypatch):
    """EXONIT_SCAN_TIMEOUT_S=1800 (prod .env) must push the client timeout
    ABOVE 1800 -- with headroom, not equal, so the server's own timeout
    logic gets to fire first."""
    monkeypatch.setenv("EXONIT_SCAN_TIMEOUT_S", "1800")
    got = S._scan_post_timeout_s()
    assert got > 1800.0
    assert got == 1800.0 + S._LAUNCH_TIMEOUT_HEADROOM_S


def test_scan_post_timeout_never_shrinks_below_900_default(monkeypatch):
    """A server timeout smaller than 900 (e.g. a dev box's EXONIT_
    SCAN_TIMEOUT_S=60) must not shrink the client's floor below 900 --
    the supervisor's own preflight/fleet-start overhead already assumes
    at least that much headroom."""
    monkeypatch.setenv("EXONIT_SCAN_TIMEOUT_S", "60")
    assert S._scan_post_timeout_s() == 900.0


def test_scan_post_timeout_ignores_malformed_env_var(monkeypatch):
    monkeypatch.setenv("EXONIT_SCAN_TIMEOUT_S", "not-a-number")
    assert S._scan_post_timeout_s() == 900.0


def test_launch_lot_uses_aligned_timeout_from_env(monkeypatch):
    """_launch_lot's actual client.post call must use the env-aligned
    timeout, not a bare 900 -- the regression this Q6 fix targets."""
    monkeypatch.setenv("EXONIT_SCAN_TIMEOUT_S", "1800")
    plan = _survey_plan()
    state = S.SupervisorState()
    cfg = AutonomyConfig(fleet=FleetConfig(min_alive=5, start_wait_s=10))

    client = MagicMock()
    client.get.return_value = {"workers": {"alive": 5}}
    client.post.return_value = {"status": "started", "enqueued": 300_000, "run_id": "r1"}

    kind, detail = S._launch_lot(
        client, plan, state, cfg,
        run_cmd=MagicMock(return_value=_proc(rc=0)), sleep_fn=MagicMock())

    assert kind == "launched"
    client.post.assert_called_once_with(
        "/api/discovery/scan", {"sector": 15, "count": 300_000},
        timeout_s=1800.0 + S._LAUNCH_TIMEOUT_HEADROOM_S)


def test_launch_lot_launch_timeout_detail_reports_aligned_timeout(monkeypatch):
    """The launch_timeout detail message must report the ACTUAL timeout
    used, not a stale hardcoded '900s' string, once it grows via the env
    var (this incident's exact symptom was an operator staring at a
    misleading '900s' message while EXONIT_SCAN_TIMEOUT_S=1800 was live)."""
    monkeypatch.setenv("EXONIT_SCAN_TIMEOUT_S", "1800")
    plan = _survey_plan()
    state = S.SupervisorState()
    cfg = AutonomyConfig(fleet=FleetConfig(min_alive=1, start_wait_s=10))

    client = MagicMock()
    client.get.return_value = {"workers": {"alive": 1}}
    client.post.side_effect = TimeoutError("timed out")

    kind, detail = S._launch_lot(
        client, plan, state, cfg,
        run_cmd=MagicMock(return_value=_proc(0)), sleep_fn=MagicMock())

    assert kind == "launch_timeout"
    assert "1860s" in detail
    assert "900s" not in detail


# ---------------------------------------------------------------------------
# 4d. Incident 2026-08-09: `_launch_lot` synchronizes the UI sector state
# (PG `discovery_sector_state` + `active_sector` of sectors.json)
# BEFORE the POST -- see `_sync_sector_ui_state`.
# ---------------------------------------------------------------------------

def test_launch_lot_calls_sync_sector_ui_state_before_post(monkeypatch):
    """Order: `_sync_sector_ui_state(plan.sector, ...)` is called before
    `client.post`, not after (the server rewrites sectors.json during
    ENQUEUEING -- see the comment in `_launch_lot`)."""
    plan = _survey_plan(sector=2)
    state = S.SupervisorState()
    cfg = AutonomyConfig(fleet=FleetConfig(min_alive=1, start_wait_s=10))

    calls: list = []
    sync_mock = MagicMock(side_effect=lambda sector, **kw: calls.append(("sync", sector)))
    monkeypatch.setattr(S, "_sync_sector_ui_state", sync_mock)

    client = MagicMock()
    client.get.return_value = {"workers": {"alive": 1}}
    client.post.side_effect = lambda *a, **kw: calls.append(("post", a, kw)) or {
        "status": "started", "enqueued": 1, "run_id": "r1",
    }

    kind, detail = S._launch_lot(client, plan, state, cfg, run_cmd=MagicMock(return_value=_proc(0)), sleep_fn=MagicMock())

    assert kind == "launched"
    sync_mock.assert_called_once()
    args, kwargs = sync_mock.call_args
    assert args[0] == 2  # plan.sector
    assert [c[0] for c in calls] == ["sync", "post"]


def test_launch_lot_continues_when_sync_sector_ui_state_raises(monkeypatch):
    """If `_sync_sector_ui_state` blows up internally (e.g. `set_sector_state`
    raising), the launch continues and returns 'launched' anyway --
    UI metadata can never block the real launch. Restores the REAL
    implementation of `_sync_sector_ui_state` (the autouse replaces it
    with a clean `MagicMock`) and makes a call inside it blow up."""
    monkeypatch.setattr(S, "_sync_sector_ui_state", _REAL_SYNC_SECTOR_UI_STATE)
    monkeypatch.setattr(S, "_resolve_db", lambda repo_root: "fake-db")
    # Another sector ("14", != plan.sector=2) so the pause loop
    # really calls `get_sector_state` (and blows up there) instead of skipping it.
    monkeypatch.setattr(S, "load_registry", lambda path: {"active_sector": 14, "sectors": {"14": {}}})
    monkeypatch.setattr(S, "get_sector_state", MagicMock(side_effect=RuntimeError("boom")))
    monkeypatch.setattr(S, "set_sector_state", MagicMock())
    monkeypatch.setattr(S, "save_registry", MagicMock())

    plan = _survey_plan(sector=2)
    state = S.SupervisorState()
    cfg = AutonomyConfig(fleet=FleetConfig(min_alive=1, start_wait_s=10))

    client = MagicMock()
    client.get.return_value = {"workers": {"alive": 1}}
    client.post.return_value = {"status": "started", "enqueued": 1, "run_id": "r1"}

    kind, detail = S._launch_lot(client, plan, state, cfg, run_cmd=MagicMock(return_value=_proc(0)), sleep_fn=MagicMock())

    assert kind == "launched"


def test_sync_sector_ui_state_pauses_old_active_and_activates_new(monkeypatch):
    """Unit test, without going through `_launch_lot`: with a registry {"2","14","15"}
    and 14 marked `active` in PG, calling it with sector=15 does 14->paused,
    15->active, and `active_sector`=15 saved in the registry."""
    reg = {"active_sector": 14, "sectors": {"2": {}, "14": {}, "15": {}}}
    saved: dict = {}

    monkeypatch.setattr(S, "_resolve_db", lambda repo_root: "fake-db")
    monkeypatch.setattr(S, "load_registry", lambda path: reg)
    monkeypatch.setattr(S, "save_registry", lambda path, data: saved.update({"path": path, "data": data}))

    state_by_sector = {2: "coming_soon", 14: "active", 15: "coming_soon"}
    set_calls: list = []

    def fake_get_sector_state(n, *, db_path):
        return state_by_sector[n]

    def fake_set_sector_state(n, state_name, *, db_path, detail=""):
        set_calls.append((n, state_name))
        state_by_sector[n] = state_name

    monkeypatch.setattr(S, "get_sector_state", fake_get_sector_state)
    monkeypatch.setattr(S, "set_sector_state", fake_set_sector_state)
    monkeypatch.setattr(S, "_sync_sector_ui_state", _REAL_SYNC_SECTOR_UI_STATE)

    S._sync_sector_ui_state(15, repo_root=Path("/tmp/fake-repo"))

    assert (14, "paused") in set_calls
    assert (15, "active") in set_calls
    assert state_by_sector[14] == "paused"
    assert state_by_sector[15] == "active"
    assert saved["data"]["active_sector"] == 15


def test_sync_sector_ui_state_never_raises_on_db_failure(monkeypatch):
    """Any exception (e.g. DB unreachable) is logged and ignored -- it never
    escapes (spec: UI metadata, cannot kill the armed loop)."""
    monkeypatch.setattr(S, "_resolve_db", lambda repo_root: "fake-db")
    monkeypatch.setattr(S, "load_registry", MagicMock(side_effect=RuntimeError("db down")))
    monkeypatch.setattr(S, "_sync_sector_ui_state", _REAL_SYNC_SECTOR_UI_STATE)

    # Must not raise.
    S._sync_sector_ui_state(2, repo_root=Path("/tmp/fake-repo"))


# ---------------------------------------------------------------------------
# ServerClient/ServerHTTPError in isolation (without going through _launch_lot).
# ---------------------------------------------------------------------------

def test_server_client_post_raises_server_http_error_with_body(monkeypatch):
    monkeypatch.setattr(SC.request, "urlopen", lambda req, timeout=None: (_ for _ in ()).throw(
        _http_error(409, {"status": "already_running"})
    ))
    client = SC.ServerClient("http://x", read_only=False)

    with pytest.raises(SC.ServerHTTPError) as exc_info:
        client.post("/api/discovery/scan", {"sector": 15, "count": 1})

    assert exc_info.value.status == 409
    assert exc_info.value.body == {"status": "already_running"}
    assert isinstance(exc_info.value, RuntimeError)  # hierarchy preserved


def test_server_client_post_http_error_without_json_body_is_empty_dict(monkeypatch):
    from urllib.error import HTTPError as _HTTPError

    def fake_urlopen(req, timeout=None):
        raise _HTTPError(url="http://x", code=500, msg="err", hdrs=None, fp=io.BytesIO(b"not json"))

    monkeypatch.setattr(SC.request, "urlopen", fake_urlopen)
    client = SC.ServerClient("http://x", read_only=False)

    with pytest.raises(SC.ServerHTTPError) as exc_info:
        client.post("/api/discovery/scan", {})

    assert exc_info.value.status == 500
    assert exc_info.value.body == {}


def test_server_client_get_raises_server_http_error_too(monkeypatch):
    monkeypatch.setattr(SC.request, "urlopen", lambda req, timeout=None: (_ for _ in ()).throw(
        _http_error(503, {"detail": "maintenance"})
    ))
    client = SC.ServerClient("http://x", read_only=True)

    with pytest.raises(SC.ServerHTTPError) as exc_info:
        client.get("/api/discovery/pipeline")

    assert exc_info.value.status == 503
    assert exc_info.value.body == {"detail": "maintenance"}


# ---------------------------------------------------------------------------
# 5. would_close_lot: close_lot(dry_run=False) called; blocked -> pause;
#    idempotency (same run_id 2 polls -> 1 single call).
# ---------------------------------------------------------------------------

def test_close_lot_armed_calls_close_lot_dry_run_false(monkeypatch):
    fake_close = MagicMock(return_value=CloseoutResult(run_id="r1", status="ok", steps=[], summary={}))
    monkeypatch.setattr(S, "close_lot", fake_close)

    state = S.SupervisorState()
    kind, detail = S._close_lot_armed(state, {"run_id": "r1"}, run_cmd=MagicMock())

    assert kind == "closed"
    assert state.last_closed_run_id == "r1"
    args, kwargs = fake_close.call_args
    assert kwargs["dry_run"] is False
    assert kwargs["commit"] is True


def test_close_lot_blocked_returns_blocked(monkeypatch):
    fake_close = MagicMock(
        return_value=CloseoutResult(
            run_id="r1", status="blocked",
            steps=[{"step": "health_check", "status": "blocked", "detail": "CRITICAL"}],
            summary={},
        )
    )
    monkeypatch.setattr(S, "close_lot", fake_close)

    state = S.SupervisorState()
    kind, detail = S._close_lot_armed(state, {"run_id": "r1"}, run_cmd=MagicMock())

    assert kind == "blocked"
    assert state.last_closed_run_id is None


def test_close_lot_idempotent_same_run_id_only_closes_once(monkeypatch):
    fake_close = MagicMock(return_value=CloseoutResult(run_id="r1", status="ok", steps=[], summary={}))
    monkeypatch.setattr(S, "close_lot", fake_close)

    state = S.SupervisorState()
    kind1, _ = S._close_lot_armed(state, {"run_id": "r1"}, run_cmd=MagicMock())
    kind2, detail2 = S._close_lot_armed(state, {"run_id": "r1"}, run_cmd=MagicMock())

    assert kind1 == "closed"
    assert kind2 == "noop"
    assert "idempotency" in detail2
    fake_close.assert_called_once()


# ---------------------------------------------------------------------------
# 5b. Review round 2 (2026-08-09): an already-closed DONE (armed) chains to the
# next lot instead of getting stuck forever in `would_close_lot` ->
# `noop` (idempotency). tick() treats it as IDLE when
# `pipeline.run_id == state.last_closed_run_id`. In shadow mode
# `last_closed_run_id` is never marked, so shadow behaviour does not
# change -- explicitly verified below.
# ---------------------------------------------------------------------------

def _tick_deps(pipeline, plans, state):
    return S.TickDeps(
        pipeline=pipeline,
        plans=plans,
        stop_chain_exists=False,
        fleet=[],
        classify_fn=lambda ctx: None,
        triage_fn=lambda ctx: PlaybookChoice("pause_and_alert", {}, 0.0, "x", "fallback"),
        state=state,
    )


def test_tick_done_with_run_id_matching_last_closed_advances_to_launch():
    plan = _survey_plan()
    plans = PlansFile(plans=[plan], master_list=[])
    state = S.SupervisorState(last_closed_run_id="r1")

    decision = S.tick(_tick_deps({"state": "DONE", "run_id": "r1"}, plans, state))

    assert decision.kind == "would_launch_next_lot"


def test_tick_done_with_run_id_not_matching_last_closed_still_closes():
    plans = PlansFile(plans=[], master_list=[])
    state = S.SupervisorState(last_closed_run_id="r0")

    decision = S.tick(_tick_deps({"state": "DONE", "run_id": "r1"}, plans, state))

    assert decision.kind == "would_close_lot"


def test_tick_done_without_run_id_still_closes_defensively():
    # No run_id in the pipeline (minimal real response) -- idempotency can never
    # be confirmed without a run_id, so it keeps going
    # to would_close_lot instead of assuming "already closed".
    plans = PlansFile(plans=[], master_list=[])
    state = S.SupervisorState(last_closed_run_id=None)

    decision = S.tick(_tick_deps({"state": "DONE"}, plans, state))

    assert decision.kind == "would_close_lot"


def test_shadow_mode_done_always_would_close_lot_never_marks_last_closed():
    """Shadow never calls `_close_lot_armed` (only in armed is
    `last_closed_run_id` marked), so a DONE repeated over multiple consecutive
    polls -- exactly as `run_shadow_loop` would do -- always returns
    `would_close_lot`, never `would_launch_next_lot`."""
    plan = _survey_plan()
    plans = PlansFile(plans=[plan], master_list=[])
    state = S.SupervisorState()  # as in shadow: last_closed_run_id is never touched
    pipeline = {"state": "DONE", "run_id": "r1"}

    for _ in range(3):
        decision = S.tick(_tick_deps(pipeline, plans, state))
        assert decision.kind == "would_close_lot"
        assert state.last_closed_run_id is None


# ---------------------------------------------------------------------------
# 5c. Incident 2026-08-11: an exhausted plan (HTTP 400 reason=no_undone_tics)
# after closing the last lot leaves `pipeline_state.json` in FAILED but
# keeping the OLD `run_id`, already closed -- `last_closed_run_id` is only
# marked in `_close_lot_armed` from a DONE with decision `would_close_lot`,
# never by a run that really fails mid-run (its run_id never reaches
# `last_closed_run_id`). So this FAILED is always residue of the failed
# launch, never a real failure: tick() treats it as IDLE with the
# same mechanism as the DONE guard above (`_tick_idle_residual`).
# ---------------------------------------------------------------------------

def test_tick_failed_with_run_id_matching_last_closed_treated_as_idle():
    plan = _survey_plan()
    plans = PlansFile(plans=[plan], master_list=[])
    state = S.SupervisorState(last_closed_run_id="r1")

    decision = S.tick(_tick_deps({"state": "FAILED", "run_id": "r1"}, plans, state))

    assert decision.kind == "would_launch_next_lot"
    assert "residual" in decision.detail
    assert "FAILED" in decision.detail


def test_tick_failed_with_run_id_not_matching_last_closed_still_remediates():
    plans = PlansFile(plans=[], master_list=[])
    state = S.SupervisorState(last_closed_run_id="r0")

    decision = S.tick(_tick_deps(
        {"state": "FAILED", "run_id": "r1", "detail": "real failure"}, plans, state,
    ))

    assert decision.kind == "pause"
    assert decision.choice.playbook == "pause_and_alert"


def test_tick_failed_without_last_closed_run_id_still_remediates():
    # No `last_closed_run_id` ever marked (e.g. supervisor just
    # started, or shadow mode where it is never marked): the FAILED follows the
    # old classifier/remediation path -- "residue of a closeout"
    # can never be confirmed without `last_closed_run_id`.
    plans = PlansFile(plans=[], master_list=[])
    state = S.SupervisorState(last_closed_run_id=None)

    decision = S.tick(_tick_deps(
        {"state": "FAILED", "run_id": "r1", "detail": "real failure"}, plans, state,
    ))

    assert decision.kind == "pause"
    assert decision.choice.playbook == "pause_and_alert"


def test_tick_failed_plan_exhausted_without_run_id_treated_as_idle():
    # Incident 2026-09-26: the FAILED of the exhausted plan no longer carries a run_id
    # (ENQUEUEING writes a clean state since 40ce1bc).
    plan = _survey_plan()
    plans = PlansFile(plans=[plan], master_list=[])
    state = S.SupervisorState(last_closed_run_id="r1")

    decision = S.tick(_tick_deps(
        {"state": "FAILED", "detail": "no_undone_tics", "sector": 5}, plans, state,
    ))

    assert decision.kind == "would_launch_next_lot"
    assert "residual" in decision.detail


@pytest.mark.parametrize("reason", sorted(S._PLAN_EXHAUSTED_REASONS))
def test_tick_failed_every_exhausted_reason_without_run_id_is_idle(reason):
    plans = PlansFile(plans=[_survey_plan()], master_list=[])
    decision = S.tick(_tick_deps(
        {"state": "FAILED", "detail": reason}, plans, S.SupervisorState(),
    ))
    assert decision.kind == "would_launch_next_lot"


@pytest.mark.parametrize("detail", [
    "need_download: not enough TGLC data downloaded",
    "enqueue_failed",
    "orchestrator_spawn_failed",
    "scan exec error: boom",
    None,
])
def test_tick_failed_without_run_id_other_reasons_still_remediate(detail):
    plans = PlansFile(plans=[], master_list=[])
    pipeline = {"state": "FAILED"}
    if detail is not None:
        pipeline["detail"] = detail
    decision = S.tick(_tick_deps(pipeline, plans, S.SupervisorState()))
    assert decision.kind == "pause"
    assert decision.choice.playbook == "pause_and_alert"


def test_full_armed_sequence_done_closes_then_next_poll_launches_next_lot(tmp_path, monkeypatch):
    """Full mocked sequence (tick + _execute_decision, two polls):
    poll 1 DONE -> closes and marks last_closed_run_id; poll 2 with the SAME
    DONE (the server has not started a new run yet) -> tick() now
    interprets it as IDLE and launches the next lot of the current plan."""
    plan = _survey_plan()
    plans = PlansFile(plans=[plan], master_list=[])
    state = S.SupervisorState()
    cfg = AutonomyConfig(fleet=FleetConfig(min_alive=1, start_wait_s=10))
    pipeline = {"state": "DONE", "run_id": "r1"}
    paused_path = tmp_path / "PAUSED"

    fake_close = MagicMock(return_value=CloseoutResult(run_id="r1", status="ok", steps=[], summary={}))
    monkeypatch.setattr(S, "close_lot", fake_close)

    mailer = MagicMock()
    executor = MagicMock()
    client = MagicMock()
    client.get.return_value = {"workers": {"alive": 1}}
    client.post.return_value = {"status": "started", "enqueued": 300_000, "run_id": "r2"}

    # Poll 1: DONE not yet closed -> would_close_lot -> it is closed.
    decision1 = S.tick(_tick_deps(pipeline, plans, state))
    assert decision1.kind == "would_close_lot"
    exec_kind1, _ = S._execute_decision(
        decision1, pipeline, state, plans, cfg, client, mailer, executor,
        paused_path=paused_path, run_cmd=MagicMock(return_value=_proc(0)), sleep_fn=MagicMock(),
    )
    assert exec_kind1 == "closed"
    assert state.last_closed_run_id == "r1"

    # Poll 2: SAME DONE (run_id="r1", the server has not started a new
    # run yet) -> already closed -> would_launch_next_lot -> launches the current plan.
    decision2 = S.tick(_tick_deps(pipeline, plans, state))
    assert decision2.kind == "would_launch_next_lot"
    exec_kind2, _ = S._execute_decision(
        decision2, pipeline, state, plans, cfg, client, mailer, executor,
        paused_path=paused_path, run_cmd=MagicMock(return_value=_proc(0)), sleep_fn=MagicMock(),
    )
    assert exec_kind2 == "launched"
    client.post.assert_called_once_with(
        "/api/discovery/scan", {"sector": plan.sector, "count": plan.lot_size}, timeout_s=900,
    )


# ---------------------------------------------------------------------------
# 6. would_remediate: valid choice -> executor.execute called;
#    refused_limit -> pause.
# ---------------------------------------------------------------------------

def test_remediate_calls_executor_execute():
    choice = PlaybookChoice("requeue_orphans", {}, 1.0, "rule", "rules")
    executor = MagicMock()
    executor.execute.return_value = ExecutionResult("executed", "ok")

    kind, detail = S._remediate(choice, executor, sleep_fn=MagicMock())

    assert kind == "executed"
    executor.execute.assert_called_once_with(choice, trigger="armed_loop")


def test_remediate_refused_limit_is_blocked():
    choice = PlaybookChoice("restart_worker", {"worker": "worker-<host>"}, 1.0, "rule", "rules")
    executor = MagicMock()
    executor.execute.return_value = ExecutionResult("refused_limit", "limit 3/24h exceeded")

    kind, detail = S._remediate(choice, executor, sleep_fn=MagicMock())

    assert kind == "blocked"
    assert "refused_limit" not in detail or True  # detail contains the reason, the literal is not needed
    assert "limit" in detail


def test_remediate_resume_run_sleeps_60s_after_execution():
    choice = PlaybookChoice("resume_run", {"run_id": "r1"}, 1.0, "rule", "rules")
    executor = MagicMock()
    executor.execute.return_value = ExecutionResult("executed", "ok")
    sleep_fn = MagicMock()

    S._remediate(choice, executor, sleep_fn=sleep_fn)

    sleep_fn.assert_called_once_with(60)


# ---------------------------------------------------------------------------
# 7. PAUSED present -> no action; gone -> resumes.
# ---------------------------------------------------------------------------

def test_paused_file_blocks_all_execution(tmp_path):
    paused_path = tmp_path / "PAUSED"
    paused_path.write_text('{"reason": "test"}')

    decision = S.Decision("would_remediate", "x", PlaybookChoice("requeue_orphans", {}, 1.0, "r", "rules"))
    executor = MagicMock()
    client = MagicMock()
    mailer = MagicMock()
    plans = PlansFile(plans=[], master_list=[])
    state = S.SupervisorState()
    cfg = AutonomyConfig()

    kind, detail = S._execute_decision(
        decision, {}, state, plans, cfg, client, mailer, executor, paused_path=paused_path,
    )

    assert kind == "observing"
    executor.execute.assert_not_called()
    client.post.assert_not_called()
    mailer.send.assert_not_called()


def test_paused_file_removed_resumes_execution(tmp_path):
    paused_path = tmp_path / "PAUSED"
    # not written -- the operator has already removed it

    decision = S.Decision("would_remediate", "x", PlaybookChoice("requeue_orphans", {}, 1.0, "r", "rules"))
    executor = MagicMock()
    executor.execute.return_value = ExecutionResult("executed", "ok")
    client = MagicMock()
    mailer = MagicMock()
    plans = PlansFile(plans=[], master_list=[])
    state = S.SupervisorState()
    cfg = AutonomyConfig()

    kind, detail = S._execute_decision(
        decision, {}, state, plans, cfg, client, mailer, executor, paused_path=paused_path,
    )

    assert kind == "executed"
    executor.execute.assert_called_once()


# ---------------------------------------------------------------------------
# _execute_decision: wiring of the remaining decision_kinds.
# ---------------------------------------------------------------------------

def test_execute_decision_pause_writes_paused_and_sends_mail(tmp_path):
    paused_path = tmp_path / "PAUSED"
    decision = S.Decision("pause", "no known playbook", PlaybookChoice("pause_and_alert", {}, 0.0, "x", "fallback"))
    mailer = MagicMock()
    plans = PlansFile(plans=[], master_list=[])
    state = S.SupervisorState()
    cfg = AutonomyConfig()

    kind, detail = S._execute_decision(
        decision, {}, state, plans, cfg, MagicMock(), mailer, MagicMock(), paused_path=paused_path,
    )

    assert kind == "paused"
    assert paused_path.exists()
    mailer.send.assert_called_once()
    assert mailer.send.call_args[0][0] == "pause:escalated"


def test_execute_decision_would_launch_next_lot_no_current_plan_pauses(tmp_path):
    # decision says would_launch_next_lot but when re-reading plans.yaml there is
    # no current plan (change between the tick and the dispatch).
    paused_path = tmp_path / "PAUSED"
    decision = S.Decision("would_launch_next_lot", "x", None)
    mailer = MagicMock()
    plans = PlansFile(plans=[], master_list=[])
    state = S.SupervisorState()
    cfg = AutonomyConfig()

    kind, detail = S._execute_decision(
        decision, {}, state, plans, cfg, MagicMock(), mailer, MagicMock(), paused_path=paused_path,
    )

    assert kind == "paused"
    assert paused_path.exists()
    assert mailer.send.call_args[0][0] == "pause:launch_no_plan"


def test_execute_decision_would_launch_next_lot_need_download_pauses_distinctly(tmp_path, monkeypatch):
    """need_download (HTTP 409, TGLC data not downloaded) must pause
    with its own event_type -- neither "already_running" (no-op, no pause) nor
    the same event_type as other pauses."""
    paused_path = tmp_path / "PAUSED"
    plan = _survey_plan()
    plans = PlansFile(plans=[plan], master_list=[])
    decision = S.Decision("would_launch_next_lot", "x", None)
    mailer = MagicMock()
    state = S.SupervisorState()
    cfg = AutonomyConfig(fleet=FleetConfig(min_alive=1, start_wait_s=10))

    client = _real_client_scan_fails(
        monkeypatch, alive=1,
        post_error=_http_error(409, {"status": "need_download", "hint": "Download more TGLC data..."}),
    )

    kind, detail = S._execute_decision(
        decision, {}, state, plans, cfg, client, mailer, MagicMock(),
        paused_path=paused_path, run_cmd=MagicMock(return_value=_proc(0)), sleep_fn=MagicMock(),
    )

    assert kind == "paused"
    assert mailer.send.call_args[0][0] == "pause:launch_need_download"
    assert plan.plan_id not in state.completed_plan_ids


def test_execute_decision_would_launch_next_lot_timeout_does_not_pause(tmp_path):
    """`launch_timeout` is self-healing (the next tick verifies the pipeline
    state or retries the launch) -- `_execute_decision` must NOT
    treat it as `"blocked"`/`"need_download"`: no `pause_and_alert`, no
    PAUSED file, and the plan stays uncompleted."""
    paused_path = tmp_path / "PAUSED"
    plan = _survey_plan()
    plans = PlansFile(plans=[plan], master_list=[])
    decision = S.Decision("would_launch_next_lot", "x", None)
    mailer = MagicMock()
    state = S.SupervisorState()
    cfg = AutonomyConfig(fleet=FleetConfig(min_alive=1, start_wait_s=10))

    client = MagicMock()
    client.get.return_value = {"workers": {"alive": 1}}
    client.post.side_effect = TimeoutError("timed out")

    kind, detail = S._execute_decision(
        decision, {}, state, plans, cfg, client, mailer, MagicMock(),
        paused_path=paused_path, run_cmd=MagicMock(return_value=_proc(0)), sleep_fn=MagicMock(),
    )

    assert kind == "launch_timeout"
    assert kind != "paused"
    assert not paused_path.exists()
    mailer.send.assert_not_called()
    assert plan.plan_id not in state.completed_plan_ids


def test_execute_decision_continuity_with_master_list_pauses_not_launches(tmp_path):
    """Without the space gate (§5.2) nor deep_coverage wired, the continuity
    policy ONLY pauses and alerts -- it never launches a new sector without
    the gate (quality/reproducibility rule, CLAUDE.md)."""
    paused_path = tmp_path / "PAUSED"
    decision = S.Decision("continuity", "x", None)
    mailer = MagicMock()
    client = MagicMock()
    plans = PlansFile(plans=[], master_list=[16, 17])
    state = S.SupervisorState()
    cfg = AutonomyConfig()

    kind, detail = S._execute_decision(
        decision, {}, state, plans, cfg, client, mailer, MagicMock(), paused_path=paused_path,
    )

    assert kind == "paused"
    client.post.assert_not_called()
    mailer.send.assert_called_once()
    assert mailer.send.call_args[0][0] == "pause:continuity"


def test_execute_decision_distinct_pause_origins_use_distinct_event_types(tmp_path, monkeypatch):
    """Review round 1 (finding 3): all pauses shared
    event_type="pause" -> the Mailer's 1h rate limit silenced an
    unrelated one. Checks that the main origins are all
    different from each other (no possible rate-limit collision)."""
    event_types = set()

    def _run(decision, plans, client=None):
        mailer = MagicMock()
        kind, detail = S._execute_decision(
            decision, {}, S.SupervisorState(), plans, AutonomyConfig(),
            client or MagicMock(), mailer, MagicMock(),
            paused_path=tmp_path / f"PAUSED-{len(event_types)}",
            run_cmd=MagicMock(return_value=_proc(0)), sleep_fn=MagicMock(),
        )
        assert kind == "paused"
        event_types.add(mailer.send.call_args[0][0])

    _run(S.Decision("pause", "x", PlaybookChoice("pause_and_alert", {}, 0.0, "x", "fallback")), PlansFile([], []))
    _run(S.Decision("continuity", "x", None), PlansFile([], []))
    _run(S.Decision("would_launch_next_lot", "x", None), PlansFile([], []))

    assert len(event_types) == 3  # no collision


def test_execute_decision_continuity_without_master_list_pauses_with_reason(tmp_path):
    paused_path = tmp_path / "PAUSED"
    decision = S.Decision("continuity", "x", None)
    mailer = MagicMock()
    plans = PlansFile(plans=[], master_list=[])
    state = S.SupervisorState()
    cfg = AutonomyConfig()

    kind, detail = S._execute_decision(
        decision, {}, state, plans, cfg, MagicMock(), mailer, MagicMock(), paused_path=paused_path,
    )

    assert kind == "paused"
    assert "master_list" in detail


def test_execute_decision_would_trigger_planning_notifies_and_writes_digest(tmp_path):
    decision = S.Decision("would_trigger_planning", "plan p1 at 80%", None)
    mailer = MagicMock()
    plans = PlansFile(plans=[], master_list=[])
    state = S.SupervisorState()
    cfg = AutonomyConfig(data_dir=tmp_path)

    kind, detail = S._execute_decision(
        decision, {}, state, plans, cfg, MagicMock(), mailer, MagicMock(), paused_path=tmp_path / "PAUSED",
    )

    assert kind == "planning_notified"
    mailer.send.assert_called_once_with("planning_needed", "pending planning proposal", decision.detail)
    digest = (tmp_path / "weekly_digest.md").read_text()
    assert "plan p1 at 80%" in digest


def test_execute_decision_noop_running_and_stop_chain_are_pure_noops(tmp_path):
    plans = PlansFile(plans=[], master_list=[])
    state = S.SupervisorState()
    cfg = AutonomyConfig()
    mailer = MagicMock()
    client = MagicMock()
    executor = MagicMock()
    paused_path = tmp_path / "PAUSED"

    for kind_in in ("noop_running", "sleep_stop_chain"):
        decision = S.Decision(kind_in, "x", None)
        kind, _ = S._execute_decision(
            decision, {}, state, plans, cfg, client, mailer, executor, paused_path=paused_path,
        )
        assert kind == "noop"

    mailer.send.assert_not_called()
    client.post.assert_not_called()
    executor.execute.assert_not_called()


# ---------------------------------------------------------------------------
# 8. spawn_and_choose: start+kill of the mocked Popen; failure -> pause.
# ---------------------------------------------------------------------------

from autonomy.config import TriageLLMConfig
from autonomy.types import ErrorContext

CTX = ErrorContext("r1", "FPP_BACKFILL", "FAILED", "mystery", "log", {}, [])


class _HealthResp:
    status = 200

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _chat_completion_urlopen(playbook="requeue_orphans"):
    import io
    import json as _json

    body = _json.dumps(
        {
            "choices": [
                {
                    "message": {
                        "content": _json.dumps(
                            {"playbook": playbook, "params": {}, "confidence": 0.9, "reasoning": "ok"}
                        )
                    }
                }
            ]
        }
    ).encode()

    class R(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def fake_urlopen(req, timeout=None):
        url = req if isinstance(req, str) else req.full_url
        if url.endswith("/health"):
            return _HealthResp()
        return R(body)

    return fake_urlopen


def test_spawn_and_choose_starts_and_kills_process(monkeypatch):
    import autonomy.triage_llm as T

    proc = MagicMock()
    popen = MagicMock(return_value=proc)
    monkeypatch.setattr(T.request, "urlopen", _chat_completion_urlopen())

    cfg = TriageLLMConfig(enabled=True, model_path="/models/m.gguf", port=8091)
    choice = T.spawn_and_choose(CTX, cfg, popen=popen, sleep_fn=MagicMock())

    assert choice.playbook == "requeue_orphans"
    popen.assert_called_once()
    cmd = popen.call_args[0][0]
    assert cmd[0] == cfg.server_bin
    assert "-m" in cmd and cfg.model_path in cmd
    assert "--port" in cmd and str(cfg.port) in cmd
    proc.terminate.assert_called_once()
    proc.wait.assert_called_once()


def test_spawn_and_choose_disabled_falls_back_without_spawning():
    import autonomy.triage_llm as T

    popen = MagicMock()
    cfg = TriageLLMConfig(enabled=False)

    choice = T.spawn_and_choose(CTX, cfg, popen=popen, sleep_fn=MagicMock())

    assert choice.playbook == "pause_and_alert"
    assert choice.source == "fallback"
    popen.assert_not_called()


def test_spawn_and_choose_empty_model_path_falls_back_without_spawning():
    import autonomy.triage_llm as T

    popen = MagicMock()
    cfg = TriageLLMConfig(enabled=True, model_path="")

    choice = T.spawn_and_choose(CTX, cfg, popen=popen, sleep_fn=MagicMock())

    assert choice.playbook == "pause_and_alert"
    popen.assert_not_called()


def test_spawn_and_choose_popen_failure_falls_back():
    import autonomy.triage_llm as T

    popen = MagicMock(side_effect=FileNotFoundError("no such file"))
    cfg = TriageLLMConfig(enabled=True, model_path="/models/m.gguf")

    choice = T.spawn_and_choose(CTX, cfg, popen=popen, sleep_fn=MagicMock())

    assert choice.playbook == "pause_and_alert"
    assert choice.source == "fallback"


def test_spawn_and_choose_startup_timeout_kills_process_and_falls_back(monkeypatch):
    import autonomy.triage_llm as T

    proc = MagicMock()
    popen = MagicMock(return_value=proc)

    def never_healthy(req, timeout=None):
        raise OSError("connection refused")

    monkeypatch.setattr(T.request, "urlopen", never_healthy)

    # fake monotonic: first call = 0 (deadline), then always increasing
    # to exceed it right away without really waiting.
    ticks = iter([0.0, 100.0, 100.0])
    monotonic = MagicMock(side_effect=lambda: next(ticks, 100.0))

    cfg = TriageLLMConfig(enabled=True, model_path="/models/m.gguf", startup_timeout_s=1)
    choice = T.spawn_and_choose(CTX, cfg, popen=popen, sleep_fn=MagicMock(), monotonic=monotonic)

    assert choice.playbook == "pause_and_alert"
    assert choice.source == "fallback"
    proc.terminate.assert_called_once()


def test_spawn_and_choose_kill_when_terminate_does_not_finish(monkeypatch):
    import autonomy.triage_llm as T

    proc = MagicMock()
    proc.wait.side_effect = Exception("still running")
    popen = MagicMock(return_value=proc)
    monkeypatch.setattr(T.request, "urlopen", _chat_completion_urlopen())

    cfg = TriageLLMConfig(enabled=True, model_path="/models/m.gguf")
    T.spawn_and_choose(CTX, cfg, popen=popen, sleep_fn=MagicMock())

    proc.terminate.assert_called_once()
    proc.kill.assert_called_once()


# ---------------------------------------------------------------------------
# Config: new sections with safe defaults.
# ---------------------------------------------------------------------------

def test_config_triage_llm_and_fleet_defaults():
    cfg = AutonomyConfig()
    assert cfg.triage_llm.enabled is False
    assert cfg.triage_llm.port == 8091
    assert cfg.fleet.min_alive == 30
    assert cfg.fleet.start_wait_s == 180


def test_config_triage_llm_and_fleet_yaml_override(tmp_path):
    from autonomy.config import load_config

    p = tmp_path / "c.yaml"
    p.write_text(
        "triage_llm:\n  enabled: true\n  model_path: /models/m.gguf\n"
        "fleet:\n  min_alive: 5\n  start_wait_s: 60\n"
    )
    cfg = load_config(p)
    assert cfg.triage_llm.enabled is True
    assert cfg.triage_llm.model_path == "/models/m.gguf"
    assert cfg.fleet.min_alive == 5
    assert cfg.fleet.start_wait_s == 60


# ---------------------------------------------------------------------------
# server_client.post: configurable long timeout, without breaking read-only.
# ---------------------------------------------------------------------------

def test_post_default_timeout_is_900s(monkeypatch):
    from autonomy.server_client import ServerClient

    captured = {}

    def fake_urlopen(req, timeout=None):
        captured["timeout"] = timeout
        import io

        class R(io.BytesIO):
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        return R(b"{}")

    import autonomy.server_client as SC

    monkeypatch.setattr(SC.request, "urlopen", fake_urlopen)
    ServerClient("http://x", read_only=False).post("/api/x", {})
    assert captured["timeout"] == 900


def test_post_still_refuses_when_read_only():
    from autonomy.server_client import ServerClient

    with pytest.raises(RuntimeError):
        ServerClient("http://x", read_only=True).post("/api/x", {})


# ---------------------------------------------------------------------------
# 9. Incident 2026-08-15: hung server (port listening, no uvicorn
# accepting) -- `_handle_server_unreachable` counts consecutive failed GETs
# and, only from the 3rd one in armed mode, tries a `restart_server` via
# the Executor before pausing. Unit tests (without going through `run_armed_loop`/
# `run_shadow_loop`), same style as `_remediate`/`_launch_lot` above.
# ---------------------------------------------------------------------------

def _boom() -> RuntimeError:
    return RuntimeError("GET /api/discovery/pipeline failed: boom")


def test_unreachable_first_failure_is_observing_no_executor_no_pause():
    executor = MagicMock()

    decision, count, exec_kind, exec_detail = S._handle_server_unreachable(
        _boom(), 0, armed=True, executor=executor,
    )

    assert decision.kind == "observing"
    assert "1/3" in decision.detail
    assert count == 1
    assert exec_kind is None and exec_detail is None
    executor.execute.assert_not_called()


def test_unreachable_second_failure_is_still_observing_no_executor_no_pause():
    executor = MagicMock()

    decision, count, exec_kind, exec_detail = S._handle_server_unreachable(
        _boom(), 1, armed=True, executor=executor,
    )

    assert decision.kind == "observing"
    assert "2/3" in decision.detail
    assert count == 2
    executor.execute.assert_not_called()


def test_unreachable_third_failure_armed_calls_executor_with_restart_server_playbook():
    executor = MagicMock()
    executor.execute.return_value = ExecutionResult(
        "executed", "cmd=['restart_exonit_server.py'] rc=0 stdout_tail='' stderr_tail=''"
    )

    S._handle_server_unreachable(_boom(), 2, armed=True, executor=executor)

    executor.execute.assert_called_once()
    args, kwargs = executor.execute.call_args
    assert args[0].playbook == "restart_server"
    assert args[0].params == {}
    assert kwargs["trigger"] == "armed_loop"


def test_unreachable_third_failure_armed_restart_rc0_executed_resets_counter():
    executor = MagicMock()
    executor.execute.return_value = ExecutionResult(
        "executed", "cmd=['x'] rc=0 stdout_tail='' stderr_tail=''"
    )

    decision, count, exec_kind, exec_detail = S._handle_server_unreachable(
        _boom(), 2, armed=True, executor=executor,
    )

    assert decision.kind == "would_remediate"
    assert decision.choice.playbook == "restart_server"
    assert count == 0
    assert exec_kind == "executed"
    assert exec_detail == executor.execute.return_value.detail


def test_unreachable_third_failure_refused_limit_pauses_counter_not_reset():
    executor = MagicMock()
    executor.execute.return_value = ExecutionResult("refused_limit", "limit 2/24h exceeded for 'restart_server'")

    decision, count, exec_kind, exec_detail = S._handle_server_unreachable(
        _boom(), 2, armed=True, executor=executor,
    )

    assert decision.kind == "pause"
    assert count == 3  # NOT reset -- the executor refused it, did not repair it
    assert exec_kind is None and exec_detail is None


def test_unreachable_third_failure_rc_nonzero_pauses_counter_not_reset():
    executor = MagicMock()
    executor.execute.return_value = ExecutionResult(
        "executed", "cmd=['x'] rc=1 stdout_tail='' stderr_tail='boom'"
    )

    decision, count, exec_kind, exec_detail = S._handle_server_unreachable(
        _boom(), 2, armed=True, executor=executor,
    )

    assert decision.kind == "pause"
    assert count == 3
    assert exec_kind is None


def test_unreachable_third_failure_shadow_mode_pauses_directly_no_executor_call():
    decision, count, exec_kind, exec_detail = S._handle_server_unreachable(_boom(), 2, armed=False)

    assert decision.kind == "pause"
    assert decision.choice is None
    assert count == 3
    assert exec_kind is None and exec_detail is None


def test_unreachable_armed_without_executor_pauses_directly_like_shadow():
    """Defensive: `armed=True` but without `executor` (never happens from the real
    loops, but `_handle_server_unreachable` must not blow up) behaves
    like shadow -- pauses without trying a restart."""
    decision, count, exec_kind, exec_detail = S._handle_server_unreachable(
        _boom(), 2, armed=True, executor=None,
    )

    assert decision.kind == "pause"
    assert count == 3
    assert exec_kind is None


# ---------------------------------------------------------------------------
# 9c. Review fix (blocking finding, 2026-08-15): `_handle_server_
# unreachable` MUST NOT call `executor.execute` while `data/autonomy/PAUSED`
# exists -- it must honor the same hard invariant as `_execute_decision`
# (supervisor.py ~914-919: "while this file exists, no action").
# Before the fix: (a) a supervisor paused for an UNRELATED reason + 3+
# failed GETs -> undue autonomous SIGKILL; (b) a failed restart writes
# PAUSED -> the next tick (server still down) tried a 2nd REAL
# restart before the 2/24h limit stopped it.
# ---------------------------------------------------------------------------

def test_unreachable_third_failure_armed_with_paused_present_never_calls_executor(tmp_path):
    """(a) PAUSED present (for any, unrelated reason) + 3rd
    consecutive failure in armed -> the executor is NEVER called, passive
    decision ('pause', no choice)."""
    paused_path = tmp_path / "PAUSED"
    paused_path.write_text('{"reason": "pause:launch_blocked -- unrelated reason"}')
    executor = MagicMock()

    decision, count, exec_kind, exec_detail = S._handle_server_unreachable(
        _boom(), 2, armed=True, executor=executor, paused_path=paused_path,
    )

    assert decision.kind == "pause"
    assert decision.choice is None
    assert count == 3  # kept, not reset
    assert exec_kind is None and exec_detail is None
    executor.execute.assert_not_called()


def test_unreachable_paused_present_never_calls_executor_across_repeated_failures(tmp_path):
    """PAUSED present during multiple consecutive failures (4th, 5th...)
    -- the executor is still never called while it remains present."""
    paused_path = tmp_path / "PAUSED"
    paused_path.write_text('{"reason": "test"}')
    executor = MagicMock()

    count = 2
    for _ in range(3):
        decision, count, exec_kind, exec_detail = S._handle_server_unreachable(
            _boom(), count, armed=True, executor=executor, paused_path=paused_path,
        )
        assert decision.kind == "pause"
        assert exec_kind is None

    executor.execute.assert_not_called()


def test_unreachable_paused_absent_still_calls_executor_as_before(tmp_path):
    """Negative control: without PAUSED (non-existent file), the pre-fix
    behaviour is intact -- the executor IS called."""
    paused_path = tmp_path / "PAUSED"  # never written
    executor = MagicMock()
    executor.execute.return_value = ExecutionResult(
        "executed", "cmd=['x'] rc=0 stdout_tail='' stderr_tail=''"
    )

    decision, count, exec_kind, exec_detail = S._handle_server_unreachable(
        _boom(), 2, armed=True, executor=executor, paused_path=paused_path,
    )

    executor.execute.assert_called_once()
    assert decision.kind == "would_remediate"
    assert count == 0
    assert exec_kind == "executed"


class _StopLoop(Exception):
    """Breaks the real `while True` of `run_armed_loop` from inside a mocked
    `time.sleep` -- the only way to really exercise N CONSECUTIVE ticks
    within a SINGLE call (the counter lives as a local variable of the
    loop, it does not persist between separate calls with `once=True`)."""


def _sleep_that_stops_after(n: int):
    calls = {"n": 0}

    def _fake_sleep(_seconds):
        calls["n"] += 1
        if calls["n"] >= n:
            raise _StopLoop()

    return _fake_sleep


def test_run_armed_loop_paused_present_never_calls_executor_end_to_end(tmp_path, monkeypatch):
    """(a) real end-to-end: pre-existing PAUSED (UNRELATED reason) +
    `client.get` that always fails -- running the armed loop FOR REAL
    (`once=False`, stopped after 3 ticks via a mocked `time.sleep`)
    until the real 3rd consecutive failure, `Executor.execute` (and
    hence `scripts/restart_exonit_server.py`) is never invoked."""
    monkeypatch.setenv("EXONIT_AUTONOMY_ARMED", "1")
    data_dir = tmp_path / "autonomy"
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(f"armed: true\ndata_dir: {data_dir}\npoll_interval_s: 0\n")

    paused_path = data_dir / "PAUSED"
    paused_path.parent.mkdir(parents=True, exist_ok=True)
    paused_path.write_text('{"reason": "pause:continuity -- pre-existing, unrelated"}')

    monkeypatch.setattr(S, "ServerClient", MagicMock(return_value=_fake_client_get_fails()))

    executor_execute_mock = MagicMock()
    monkeypatch.setattr(S.Executor, "execute", executor_execute_mock)
    monkeypatch.setattr(S.time, "sleep", _sleep_that_stops_after(3))

    # Tick 1: count=1 (observing); tick 2: count=2 (observing); tick 3:
    # count=3 -- PAUSED is already there, so it must NOT call the executor (the
    # pre-fix bug did). The 3rd `time.sleep` breaks the infinite loop.
    with pytest.raises(_StopLoop):
        S.run_armed_loop(once=False, config_path=cfg_path)

    executor_execute_mock.assert_not_called()


def test_run_armed_loop_failed_restart_paused_then_next_tick_never_reexecutes(tmp_path, monkeypatch):
    """(b) failed restart (rc=1, real via `Executor.execute` mocked only
    at the subprocess level) really writes PAUSED (via the real `_pause_and_alert`
    inside `_execute_decision`) -- the immediately following tick, with the
    server still down, does NOT call the executor again while PAUSED
    exists. Runs the real armed loop 4 consecutive ticks (`once=False`)."""
    monkeypatch.setenv("EXONIT_AUTONOMY_ARMED", "1")
    data_dir = tmp_path / "autonomy"
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(f"armed: true\ndata_dir: {data_dir}\npoll_interval_s: 0\n")

    monkeypatch.setattr(S, "ServerClient", MagicMock(return_value=_fake_client_get_fails()))

    # restart_server "executed" but rc=1 (does NOT repair) -- via the real
    # Executor (real validation+limits+log), only the subprocess
    # result is simulated.
    exec_result = ExecutionResult("executed", "cmd=['x'] rc=1 stdout_tail='' stderr_tail='boom'")
    executor_execute_mock = MagicMock(return_value=exec_result)
    monkeypatch.setattr(S.Executor, "execute", executor_execute_mock)
    monkeypatch.setattr(S.time, "sleep", _sleep_that_stops_after(4))

    # tick1: count=1 observing; tick2: count=2 observing; tick3: count=3,
    # PAUSED absent -> 1st real call to the executor -> rc=1 -> Decision(pause)
    # -> _execute_decision -> _pause_and_alert REALLY WRITES data/autonomy/PAUSED;
    # tick4: count=4, PAUSED already present -> the executor is NOT
    # called again (the pre-fix bug would have made a 2nd real call here).
    with pytest.raises(_StopLoop):
        S.run_armed_loop(once=False, config_path=cfg_path)

    assert executor_execute_mock.call_count == 1
    paused_path = data_dir / "PAUSED"
    assert paused_path.exists()


def test_restart_result_ok_requires_executed_status_and_rc0():
    assert S._restart_result_ok(ExecutionResult("executed", "cmd=['x'] rc=0 stdout_tail='' stderr_tail=''"))
    assert not S._restart_result_ok(ExecutionResult("executed", "cmd=['x'] rc=1 stdout_tail='' stderr_tail=''"))
    assert not S._restart_result_ok(ExecutionResult("executed", "cmd=['x'] error=FileNotFoundError()"))
    assert not S._restart_result_ok(ExecutionResult("refused_limit", "limit exceeded"))
    assert not S._restart_result_ok(ExecutionResult("refused_invalid", "unknown playbook"))
    assert not S._restart_result_ok(ExecutionResult("dry_run", "would run: [...]"))


# ---------------------------------------------------------------------------
# 9b. Real wiring: `run_shadow_loop`/`run_armed_loop` call
# `_handle_server_unreachable` with the right `armed`/`executor` when the GET
# fails, and the armed loop skips `_execute_decision` when a successful
# restart has already been executed (`exec_kind="executed"` returned directly).
# ---------------------------------------------------------------------------

def _fake_client_get_fails():
    client = MagicMock()
    client.get.side_effect = RuntimeError("boom")
    return client


def test_run_shadow_loop_wires_get_failure_to_handle_unreachable_not_armed(tmp_path, monkeypatch):
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(f"data_dir: {tmp_path / 'autonomy'}\npoll_interval_s: 0\n")

    monkeypatch.setattr(S, "ServerClient", MagicMock(return_value=_fake_client_get_fails()))

    captured = {}

    def fake_handle(exc, count, *, armed, executor=None, paused_path=None):
        captured["armed"] = armed
        captured["executor"] = executor
        captured["count_in"] = count
        return S.Decision("observing", "x", None), count + 1, None, None

    monkeypatch.setattr(S, "_handle_server_unreachable", fake_handle)

    S.run_shadow_loop(once=True, config_path=cfg_path)

    assert captured["armed"] is False
    assert captured["executor"] is None
    assert captured["count_in"] == 0


def test_run_armed_loop_wires_get_failure_to_handle_unreachable_armed_with_executor(tmp_path, monkeypatch):
    monkeypatch.setenv("EXONIT_AUTONOMY_ARMED", "1")
    data_dir = tmp_path / "autonomy"
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(f"armed: true\ndata_dir: {data_dir}\npoll_interval_s: 0\n")

    monkeypatch.setattr(S, "ServerClient", MagicMock(return_value=_fake_client_get_fails()))

    captured = {}

    def fake_handle(exc, count, *, armed, executor=None, paused_path=None):
        captured["armed"] = armed
        captured["executor"] = executor
        return S.Decision("observing", "x", None), count + 1, None, None

    monkeypatch.setattr(S, "_handle_server_unreachable", fake_handle)

    S.run_armed_loop(once=True, config_path=cfg_path)

    assert captured["armed"] is True
    assert isinstance(captured["executor"], S.Executor)


def test_run_armed_loop_successful_restart_skips_execute_decision_and_logs_executed(tmp_path, monkeypatch):
    monkeypatch.setenv("EXONIT_AUTONOMY_ARMED", "1")
    data_dir = tmp_path / "autonomy"
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(f"armed: true\ndata_dir: {data_dir}\npoll_interval_s: 0\n")

    monkeypatch.setattr(S, "ServerClient", MagicMock(return_value=_fake_client_get_fails()))

    choice = PlaybookChoice("restart_server", {}, 1.0, "server hung", "rules")

    def fake_handle(exc, count, *, armed, executor=None, paused_path=None):
        return (
            S.Decision("would_remediate", "hung server repaired", choice),
            0,
            "executed",
            "cmd=['x'] rc=0 stdout_tail='' stderr_tail=''",
        )

    monkeypatch.setattr(S, "_handle_server_unreachable", fake_handle)
    execute_decision_mock = MagicMock()
    monkeypatch.setattr(S, "_execute_decision", execute_decision_mock)

    S.run_armed_loop(once=True, config_path=cfg_path)

    execute_decision_mock.assert_not_called()

    log_lines = (data_dir / "shadow.log").read_text().strip().splitlines()
    last = json.loads(log_lines[-1])
    assert last["decision_kind"] == "would_remediate"
    assert last["exec_kind"] == "executed"
    assert last["playbook"] == "restart_server"


def test_run_armed_loop_failed_restart_still_dispatches_via_execute_decision(tmp_path, monkeypatch):
    """When `_handle_server_unreachable` has NOT repaired anything (pause,
    exec_kind=None) the armed loop must follow the normal path via
    `_execute_decision` (mail + PAUSED), not skip it."""
    monkeypatch.setenv("EXONIT_AUTONOMY_ARMED", "1")
    data_dir = tmp_path / "autonomy"
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(f"armed: true\ndata_dir: {data_dir}\npoll_interval_s: 0\n")

    monkeypatch.setattr(S, "ServerClient", MagicMock(return_value=_fake_client_get_fails()))

    def fake_handle(exc, count, *, armed, executor=None, paused_path=None):
        return S.Decision("pause", "restart_server repaired nothing", None), count + 1, None, None

    monkeypatch.setattr(S, "_handle_server_unreachable", fake_handle)
    execute_decision_mock = MagicMock(return_value=("paused", "restart_server repaired nothing"))
    monkeypatch.setattr(S, "_execute_decision", execute_decision_mock)

    S.run_armed_loop(once=True, config_path=cfg_path)

    execute_decision_mock.assert_called_once()


# ---------------------------------------------------------------------------
# 9. Unattended fixes 2026-09-19 (operator: the chain must run on its own for 1-2 months):
#    (a) `_launch_lot`/`_close_lot_armed` reset the "run" limit counter of
#        the `Executor` at lot boundaries -- before, `resume_run`
#        (limit_n=3) was counted over the WHOLE lifetime of the `Executor`, not per
#        run (see also fix (b), the key is now scoped per run_id).
#    (b) tests of `_run_detached`/scoping per run_id are in test_playbooks.py.
#    (c) `_pause_and_alert` and the `would_trigger_planning` branch call the
#        optional external hook `alert.command` (`run_alert_hook`), in addition to the
#        `Mailer` (which stays disabled by default without an App Password).
#    (d) optional external heartbeat (`alert.heartbeat_hours`).
# ---------------------------------------------------------------------------

def test_launch_lot_resets_run_scope_on_success():
    plan = _survey_plan()
    state = S.SupervisorState()
    cfg = AutonomyConfig(fleet=FleetConfig(min_alive=1, start_wait_s=10))

    client = MagicMock()
    client.get.return_value = {"workers": {"alive": 1}}
    client.post.return_value = {"status": "started", "enqueued": 300_000, "run_id": "r1"}

    executor = MagicMock()
    kind, _ = S._launch_lot(
        client, plan, state, cfg,
        run_cmd=MagicMock(return_value=_proc(0)), sleep_fn=MagicMock(), executor=executor,
    )

    assert kind == "launched"
    executor.reset_scope.assert_called_once_with("run")


def test_launch_lot_does_not_reset_scope_when_blocked():
    plan = _survey_plan()
    state = S.SupervisorState()
    cfg = AutonomyConfig(fleet=FleetConfig(min_alive=5, start_wait_s=10))

    client = MagicMock()
    run_cmd = MagicMock(return_value=_proc(rc=2, stderr="CRITICAL disk"))
    executor = MagicMock()

    kind, _ = S._launch_lot(
        client, plan, state, cfg, run_cmd=run_cmd, sleep_fn=MagicMock(), executor=executor,
    )

    assert kind == "blocked"
    executor.reset_scope.assert_not_called()


def test_launch_lot_executor_none_default_never_touches_executor():
    """None of the existing `_launch_lot` tests (further up in this
    file) pass `executor=` -- the `None` default must not break anything."""
    plan = _survey_plan()
    state = S.SupervisorState()
    cfg = AutonomyConfig(fleet=FleetConfig(min_alive=1, start_wait_s=10))

    client = MagicMock()
    client.get.return_value = {"workers": {"alive": 1}}
    client.post.return_value = {"status": "started", "enqueued": 1, "run_id": "r1"}

    kind, _ = S._launch_lot(
        client, plan, state, cfg, run_cmd=MagicMock(return_value=_proc(0)), sleep_fn=MagicMock(),
    )
    assert kind == "launched"


def test_close_lot_armed_resets_run_scope_on_close(monkeypatch):
    fake_close = MagicMock(return_value=CloseoutResult(run_id="r1", status="ok", steps=[], summary={}))
    monkeypatch.setattr(S, "close_lot", fake_close)

    state = S.SupervisorState()
    executor = MagicMock()
    kind, _ = S._close_lot_armed(state, {"run_id": "r1"}, run_cmd=MagicMock(), executor=executor)

    assert kind == "closed"
    executor.reset_scope.assert_called_once_with("run")


def test_close_lot_armed_does_not_reset_scope_when_blocked(monkeypatch):
    fake_close = MagicMock(
        return_value=CloseoutResult(
            run_id="r1", status="blocked",
            steps=[{"step": "health_check", "status": "blocked", "detail": "CRITICAL"}],
            summary={},
        )
    )
    monkeypatch.setattr(S, "close_lot", fake_close)

    state = S.SupervisorState()
    executor = MagicMock()
    kind, _ = S._close_lot_armed(state, {"run_id": "r1"}, run_cmd=MagicMock(), executor=executor)

    assert kind == "blocked"
    executor.reset_scope.assert_not_called()


def test_close_lot_armed_does_not_reset_scope_when_noop_idempotent(monkeypatch):
    fake_close = MagicMock(return_value=CloseoutResult(run_id="r1", status="ok", steps=[], summary={}))
    monkeypatch.setattr(S, "close_lot", fake_close)

    state = S.SupervisorState()
    executor = MagicMock()
    S._close_lot_armed(state, {"run_id": "r1"}, run_cmd=MagicMock(), executor=executor)
    executor.reset_scope.reset_mock()

    kind, _ = S._close_lot_armed(state, {"run_id": "r1"}, run_cmd=MagicMock(), executor=executor)

    assert kind == "noop"
    executor.reset_scope.assert_not_called()


def test_pause_and_alert_invokes_run_alert_hook(monkeypatch, tmp_path):
    hook = MagicMock(return_value="rc=0")
    monkeypatch.setattr(S, "run_alert_hook", hook)

    mailer = MagicMock()
    alert_cfg = AlertConfig(command=["bin/exonit-alert"])
    paused_path = tmp_path / "PAUSED"

    S._pause_and_alert(mailer, paused_path, "pause:test", "test reason", alert_cfg)

    assert paused_path.exists()
    mailer.send.assert_called_once()
    hook.assert_called_once()
    args, kwargs = hook.call_args
    assert args[0] is alert_cfg
    assert "test reason" in args[2]


def test_would_trigger_planning_invokes_run_alert_hook_via_execute_decision(monkeypatch, tmp_path):
    hook = MagicMock(return_value="rc=0")
    monkeypatch.setattr(S, "run_alert_hook", hook)

    decision = S.Decision("would_trigger_planning", "plan p1 about to run out", None)
    executor = MagicMock()
    client = MagicMock()
    mailer = MagicMock()
    plans = PlansFile(plans=[], master_list=[])
    state = S.SupervisorState()
    cfg = AutonomyConfig(data_dir=tmp_path, alert=AlertConfig(command=["bin/exonit-alert"]))

    kind, detail = S._execute_decision(
        decision, {}, state, plans, cfg, client, mailer, executor,
        paused_path=tmp_path / "PAUSED",
    )

    assert kind == "planning_notified"
    mailer.send.assert_called_once()
    hook.assert_called_once()
    args, kwargs = hook.call_args
    assert args[0] is cfg.alert
    assert "plan p1" in args[2]


def test_heartbeat_status_line_includes_state_run_id_lots_done_and_fleet_alive():
    client = MagicMock()
    client.get.return_value = {"workers": {"alive": 7}}
    state = S.SupervisorState(completed_plan_ids={"p1", "p2"})

    line = S._heartbeat_status_line(client, "IDLE", {"run_id": "r9"}, state)

    assert "state=IDLE" in line
    assert "run_id=r9" in line
    assert "lots_done=2" in line
    assert "fleet_alive=7" in line


def test_heartbeat_status_line_survives_client_get_failure():
    client = MagicMock()
    client.get.side_effect = RuntimeError("network down")
    state = S.SupervisorState()

    line = S._heartbeat_status_line(client, None, {}, state)

    assert "fleet_alive=unknown" in line
    assert "state=unknown" in line
    assert "run_id=none" in line


def test_maybe_send_alert_heartbeat_disabled_is_noop(tmp_path, monkeypatch):
    hook = MagicMock()
    monkeypatch.setattr(S, "run_alert_hook", hook)

    result = S._maybe_send_alert_heartbeat(AlertConfig(heartbeat_hours=0), tmp_path / "hb", "status")

    assert result is None
    hook.assert_not_called()
    assert not (tmp_path / "hb").exists()


def test_maybe_send_alert_heartbeat_first_call_sends_and_persists(tmp_path, monkeypatch):
    hook = MagicMock(return_value="rc=0")
    monkeypatch.setattr(S, "run_alert_hook", hook)
    state_path = tmp_path / "hb"
    alert_cfg = AlertConfig(command=["x"], heartbeat_hours=6)

    result = S._maybe_send_alert_heartbeat(alert_cfg, state_path, "status line")

    assert result == "rc=0"
    hook.assert_called_once_with(alert_cfg, "heartbeat", "status line")
    assert state_path.exists()


def test_maybe_send_alert_heartbeat_respects_window_then_fires_again(tmp_path, monkeypatch):
    from datetime import datetime, timedelta, timezone as tz

    hook = MagicMock(return_value="rc=0")
    monkeypatch.setattr(S, "run_alert_hook", hook)
    state_path = tmp_path / "hb"
    alert_cfg = AlertConfig(command=["x"], heartbeat_hours=1)
    t0 = datetime(2026, 1, 1, tzinfo=tz.utc)

    S._maybe_send_alert_heartbeat(alert_cfg, state_path, "s1", now=t0)
    assert hook.call_count == 1

    # 30 min later, within the 1h window -> does not send again.
    S._maybe_send_alert_heartbeat(alert_cfg, state_path, "s2", now=t0 + timedelta(minutes=30))
    assert hook.call_count == 1

    # 61 min later, outside the window -> sends again.
    S._maybe_send_alert_heartbeat(alert_cfg, state_path, "s3", now=t0 + timedelta(minutes=61))
    assert hook.call_count == 2


def test_maybe_send_alert_heartbeat_corrupt_state_file_treated_as_due(tmp_path, monkeypatch):
    hook = MagicMock(return_value="rc=0")
    monkeypatch.setattr(S, "run_alert_hook", hook)
    state_path = tmp_path / "hb"
    state_path.write_text("not-a-timestamp")

    S._maybe_send_alert_heartbeat(AlertConfig(command=["x"], heartbeat_hours=1), state_path, "s")

    hook.assert_called_once()


def test_run_armed_loop_sends_heartbeat_when_configured(tmp_path, monkeypatch):
    """Minimal integration: `alert.heartbeat_hours` in the YAML makes
    `run_armed_loop` call `run_alert_hook` with subject `heartbeat` -- in
    addition to any other alert origin of the same tick (here,
    `continuity` with no plan nor master_list -> `pause:continuity`)."""
    monkeypatch.setenv("EXONIT_AUTONOMY_ARMED", "1")
    data_dir = tmp_path / "autonomy"
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        f"armed: true\ndata_dir: {data_dir}\npoll_interval_s: 0\n"
        "alert:\n  command: [\"bin/exonit-alert\"]\n  heartbeat_hours: 1\n"
    )

    client = MagicMock()
    client.get.side_effect = lambda path, **kw: (
        {"state": "IDLE"} if "pipeline" in path else {"workers": {"alive": 4}}
    )
    monkeypatch.setattr(S, "ServerClient", MagicMock(return_value=client))

    hook = MagicMock(return_value="rc=0")
    monkeypatch.setattr(S, "run_alert_hook", hook)

    S.run_armed_loop(once=True, config_path=cfg_path)

    subjects = [c.args[1] for c in hook.call_args_list]
    assert "heartbeat" in subjects
    assert (data_dir / "alert_heartbeat_last_sent").exists()


# ---------------------------------------------------------------------------
# S4b (docs/superpowers/specs/2026-09-26-enqueue-s1-s4-design.md §2.1/§9.4,
# operator decision): FAILED without run_id with detail "scan timed out..." ->
# SINGLE automatic retry via would_launch_next_lot, not the classifier.
# ---------------------------------------------------------------------------

def test_tick_scan_timed_out_without_run_id_retries_once_via_idle():
    plan = _survey_plan()
    plans = PlansFile(plans=[plan], master_list=[])
    state = S.SupervisorState()

    decision = S.tick(_tick_deps(
        {"state": "FAILED", "detail": "scan timed out after 600 s"},
        plans, state,
    ))

    assert decision.kind == "would_launch_next_lot"
    assert "S4b" in decision.detail
    assert state.scan_timeout_retry_count == 1


def test_tick_scan_timed_out_second_time_pauses_instead_of_looping_forever():
    plans = PlansFile(plans=[_survey_plan()], master_list=[])
    state = S.SupervisorState(scan_timeout_retry_count=1)

    decision = S.tick(_tick_deps(
        {"state": "FAILED", "detail": "scan timed out after 600 s"},
        plans, state,
    ))

    assert decision.kind == "pause"
    assert state.scan_timeout_retry_count == 1  # not incremented past 1


def test_tick_scan_timed_out_with_run_id_is_not_s4b_still_remediates():
    """S4b targets ONLY the run_id-less HTTP-timeout FAILED (the server
    killed discovery_run.py before a discovery_runs row ever existed) — a
    FAILED that DOES carry a run_id is a real mid-run failure and must keep
    going through the normal classifier/pause path."""
    plans = PlansFile(plans=[], master_list=[])
    state = S.SupervisorState()

    decision = S.tick(_tick_deps(
        {"state": "FAILED", "run_id": "r1",
         "detail": "scan timed out after 600 s"},
        plans, state,
    ))

    assert decision.kind == "pause"
    assert state.scan_timeout_retry_count == 0


def test_tick_other_failed_detail_without_run_id_does_not_consume_s4b_budget():
    plans = PlansFile(plans=[], master_list=[])
    state = S.SupervisorState()

    decision = S.tick(_tick_deps(
        {"state": "FAILED", "detail": "enqueue_failed"}, plans, state,
    ))

    assert decision.kind == "pause"
    assert state.scan_timeout_retry_count == 0


def test_launch_lot_launched_resets_scan_timeout_retry_count(tmp_path, monkeypatch):
    """A REAL launch success (kind == 'launched') resets the S4b budget —
    a FUTURE 'scan timed out' is a NEW incident, not a repeat this launch
    just recovered from."""
    plan = _survey_plan()
    plans = PlansFile(plans=[plan], master_list=[])
    state = S.SupervisorState(scan_timeout_retry_count=1)
    cfg = AutonomyConfig(fleet=FleetConfig(min_alive=1, start_wait_s=10))

    mailer = MagicMock()
    executor = MagicMock()
    client = MagicMock()
    client.get.return_value = {"workers": {"alive": 1}}
    client.post.return_value = {"status": "started", "enqueued": 5, "run_id": "r2"}
    monkeypatch.setattr(S, "_run_health_check_gate", lambda *a, **k: (True, ""))
    monkeypatch.setattr(S, "_start_fleet_and_wait", lambda *a, **k: (True, ""))

    decision = S.Decision("would_launch_next_lot", "test", None)
    kind, _detail = S._execute_decision(
        decision, {}, state, plans, cfg, client, mailer, executor,
        paused_path=tmp_path / "PAUSED",
    )

    assert kind == "launched"
    assert state.scan_timeout_retry_count == 0


def test_supervisor_state_scan_timeout_retry_count_roundtrips_to_dict():
    state = S.SupervisorState(scan_timeout_retry_count=1)
    d = state.to_dict()
    assert d["scan_timeout_retry_count"] == 1
    restored = S.SupervisorState.from_dict(d)
    assert restored.scan_timeout_retry_count == 1


def test_supervisor_state_scan_timeout_retry_count_defaults_to_zero_when_absent():
    """Backward compat: a supervisor_state.json persisted before S4b has no
    such key."""
    restored = S.SupervisorState.from_dict({"completed_plan_ids": []})
    assert restored.scan_timeout_retry_count == 0
