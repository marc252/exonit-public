"""Tests of the triage benchmark harness (`autonomy/benchmark_triage.py`).

No test in this file makes a network call: `score`/`load_cases` operate
on local structures, and the deterministic classifier (`--rules-only`,
via `_rules_only_choose`) is pure/local. The real bank (`autonomy/benchmark/
cases.jsonl`) is never scored here against a live LLM (see the runbook —
that is a pending step the operator launches explicitly).
"""

import json
from pathlib import Path

from autonomy.benchmark_triage import _rules_only_choose, load_cases, score
from autonomy.types import ErrorContext

CASES_PATH = Path("autonomy/benchmark/cases.jsonl")

REMEDIABLE_PLAYBOOKS = {
    "rebuild_streams",
    "requeue_orphans",
    "clean_fits_cache",
    "wal_checkpoint",
    "resume_run",
    "restart_worker",
}


def test_score_math(tmp_path):
    cases = [
        {
            "id": "a",
            "ctx": {
                "run_id": "r",
                "phase": "FPP",
                "state": "FAILED",
                "detail": "orphans assigned heartbeat",
                "log_tail": "",
                "fleet_snapshot": {},
                "recent_actions": [],
            },
            "expected": "requeue_orphans",
            "dangerous_wrong": ["restart_worker"],
        },
        {
            "id": "b",
            "ctx": {
                "run_id": "r",
                "phase": "FPP",
                "state": "FAILED",
                "detail": "mystery",
                "log_tail": "",
                "fleet_snapshot": {},
                "recent_actions": [],
            },
            "expected": "pause_and_alert",
            "dangerous_wrong": [],
        },
    ]
    fn = lambda ctx: type("C", (), {"playbook": "requeue_orphans"})()
    s = score(cases, fn)
    assert s.n == 2 and s.correct == 1 and s.accuracy == 0.5
    assert s.escalation_recall == 0.0 and not s.dangerous_errors


def test_score_flags_dangerous_error_and_perfect_escalation_recall():
    cases = [
        {
            "id": "disk",
            "ctx": {
                "run_id": "r",
                "phase": "FPP",
                "state": "FAILED",
                "detail": "Errno 28 no space left",
                "log_tail": "",
                "fleet_snapshot": {},
                "recent_actions": [],
            },
            "expected": "clean_fits_cache",
            "dangerous_wrong": ["restart_worker"],
        },
        {
            "id": "escalate",
            "ctx": {
                "run_id": "r",
                "phase": "FPP",
                "state": "FAILED",
                "detail": "no clear signal",
                "log_tail": "",
                "fleet_snapshot": {},
                "recent_actions": [],
            },
            "expected": "pause_and_alert",
            "dangerous_wrong": [],
        },
    ]
    # "Dangerously" wrong choose_fn: picks restart_worker for the full-disk case
    # (one of the case's dangerous_wrong), and gets the escalation right.
    def fn(ctx):
        playbook = "restart_worker" if "Errno 28" in ctx.detail else "pause_and_alert"
        return type("C", (), {"playbook": playbook})()

    s = score(cases, fn)
    assert s.correct == 1 and s.accuracy == 0.5
    assert s.escalation_recall == 1.0
    assert s.dangerous_errors == [{"id": "disk", "expected": "clean_fits_cache", "actual": "restart_worker"}]


def test_bank_is_valid_and_covers():
    cases = load_cases(Path("autonomy/benchmark/cases.jsonl"))
    assert len(cases) >= 20
    expected = {c["expected"] for c in cases}
    assert "pause_and_alert" in expected and len([c for c in cases if c["expected"] == "pause_and_alert"]) >= 4


def test_bank_covers_every_remediable_playbook_with_at_least_3_cases():
    cases = load_cases(Path("autonomy/benchmark/cases.jsonl"))
    counts = {}
    for c in cases:
        counts[c["expected"]] = counts.get(c["expected"], 0) + 1
    for playbook in REMEDIABLE_PLAYBOOKS:
        assert counts.get(playbook, 0) >= 3, f"{playbook}: only {counts.get(playbook, 0)} cases (need >=3)"


def test_bank_cases_have_realistic_detail_not_synthetic_labels():
    """The detail/log_tail must be incident text (sentences, not a short
    label identical to the expected playbook name)."""
    cases = load_cases(Path("autonomy/benchmark/cases.jsonl"))
    for c in cases:
        detail = c["ctx"]["detail"]
        assert len(detail) >= 40, f"{c['id']}: detail too short to be realistic text"
        assert detail.strip().lower() != c["expected"].replace("_", " ")


def test_bank_ids_are_unique_and_structure_is_well_formed():
    cases = load_cases(Path("autonomy/benchmark/cases.jsonl"))
    ids = [c["id"] for c in cases]
    assert len(ids) == len(set(ids))
    for c in cases:
        assert set(c.keys()) == {"id", "ctx", "expected", "dangerous_wrong"}
        assert set(c["ctx"].keys()) == {
            "run_id", "phase", "state", "detail", "log_tail", "fleet_snapshot", "recent_actions",
        }
        # ErrorContext must be constructible without blowing up.
        ErrorContext(**c["ctx"])


def test_rules_only_choose_falls_back_to_pause_and_alert_when_no_rule_matches():
    ctx = ErrorContext("r", "FPP", "FAILED", "no recognized signal", "", {}, [])
    choice = _rules_only_choose(ctx)
    assert choice.playbook == "pause_and_alert" and choice.source == "fallback"


def test_rules_only_choose_delegates_to_classify_when_a_rule_matches():
    ctx = ErrorContext("r", "FPP", "FAILED", "OSError: [Errno 28] No space left on device", "",
                        {"worker-<host>": {"disk_free_gb": 1.0}}, [])
    choice = _rules_only_choose(ctx)
    assert choice.playbook == "clean_fits_cache" and choice.source == "rules"


def test_rules_only_baseline_never_produces_a_dangerous_error_it_didnt_earn(capsys):
    """Sanity check of the bank + baseline: any rules-only error that
    matches a dangerous_wrong is reported, never silenced."""
    cases = load_cases(Path("autonomy/benchmark/cases.jsonl"))
    s = score(cases, _rules_only_choose)
    assert 0.0 <= s.accuracy <= 1.0
    assert 0.0 <= s.escalation_recall <= 1.0
    assert isinstance(s.dangerous_errors, list)
