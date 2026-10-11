from autonomy.supervisor import tick, TickDeps, SupervisorState
from autonomy.plans import load_plans
from autonomy.types import PlaybookChoice


def _deps(tmp_path, pipeline, stop=False, classify=lambda c: None,
          triage=None, plans_yaml="master_list: [15]\nplans:\n  - plan_id: p1\n    approved_by: operator\n    sector: 15\n    mode: survey\n    lot_size: 300000\n"):
    p = tmp_path / "plans.yaml"; p.write_text(plans_yaml)
    triage = triage or (lambda c: PlaybookChoice("pause_and_alert", {"reason": "shadow"}, 0.0, "shadow", "fallback"))
    return TickDeps(pipeline=pipeline, plans=load_plans(p), stop_chain_exists=stop,
                    fleet=[], classify_fn=classify, triage_fn=triage,
                    state=SupervisorState())


def test_stop_chain_wins(tmp_path):
    d = tick(_deps(tmp_path, {"state": "SCANNING"}, stop=True))
    assert d.kind == "sleep_stop_chain"


def test_running_is_noop(tmp_path):
    assert tick(_deps(tmp_path, {"state": "FPP_BACKFILL"})).kind == "noop_running"


def test_done_closes_lot(tmp_path):
    assert tick(_deps(tmp_path, {"state": "DONE"})).kind == "would_close_lot"


def test_idle_launches_next_lot(tmp_path):
    assert tick(_deps(tmp_path, {"state": "IDLE"})).kind == "would_launch_next_lot"


def test_failed_uses_rules_first(tmp_path):
    rule = lambda ctx: PlaybookChoice("requeue_orphans", {}, 1.0, "rule", "rules")
    d = tick(_deps(tmp_path, {"state": "FAILED", "detail": "orphans"}, classify=rule))
    assert d.kind == "would_remediate" and d.choice.playbook == "requeue_orphans"


def test_failed_unknown_falls_to_triage_then_pause(tmp_path):
    d = tick(_deps(tmp_path, {"state": "FAILED", "detail": "mystery"}))
    assert d.kind == "pause" and d.choice.playbook == "pause_and_alert"


def test_resume_run_overrides_ia_run_id_with_context(tmp_path):
    # Incident 2026-08-11: the AI returned the literal placeholder instead
    # of the real run_id -> the system must always override it.
    triage = lambda ctx: PlaybookChoice(
        "resume_run", {"run_id": "<run_id from the context>"}, 0.8, "r", "llm"
    )
    d = tick(_deps(tmp_path, {"state": "FAILED", "run_id": "RUN_REAL", "detail": "orphan"}, triage=triage))
    assert d.kind == "would_remediate"
    assert d.choice.playbook == "resume_run"
    assert d.choice.params["run_id"] == "RUN_REAL"


def test_resume_run_overrides_ia_run_id_when_params_not_dict(tmp_path):
    triage = lambda ctx: PlaybookChoice("resume_run", None, 0.8, "r", "llm")
    d = tick(_deps(tmp_path, {"state": "FAILED", "run_id": "RUN_REAL", "detail": "orphan"}, triage=triage))
    assert d.kind == "would_remediate"
    assert d.choice.params == {"run_id": "RUN_REAL"}


def test_non_resume_playbook_not_modified(tmp_path):
    triage = lambda ctx: PlaybookChoice("restart_worker", {"worker": "x"}, 0.8, "r", "llm")
    d = tick(_deps(tmp_path, {"state": "FAILED", "run_id": "RUN_REAL", "detail": "orphan"}, triage=triage))
    assert d.kind == "would_remediate"
    assert d.choice.playbook == "restart_worker"
    assert d.choice.params == {"worker": "x"}


def test_read_only_client_refuses_post():
    from autonomy.server_client import ServerClient
    import pytest
    with pytest.raises(RuntimeError):
        ServerClient("http://x", read_only=True).post("/api/x", {})
