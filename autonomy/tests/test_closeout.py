"""Tests for `autonomy/closeout.py` (spec §4.5, phase 1.5).

Everything mocked: no real subprocess (`run_cmd` is always a fake), no real
git, everything under `tmp_path`. `repo_root` always points to `tmp_path` — never to the
real checkout.
"""

import json

import pytest

from autonomy.closeout import close_lot, _resolve_db


# ---------------------------------------------------------------------------
# Injectable subprocess.run fake
# ---------------------------------------------------------------------------

class FakeProc:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def make_run_cmd(*, health_rc=0, augment_rc=0, diff_rc=1, commit_rc=0):
    """diff_rc: rc of `git diff --cached --quiet` (0=nothing staged, 1=there are changes)."""
    calls = []

    def fake(cmd, **kwargs):
        calls.append(list(cmd))
        joined = " ".join(str(c) for c in cmd)
        if "run_health_check.py" in joined:
            return FakeProc(health_rc, stdout='{"status": "x"}')
        if "augment_scorecard_exominer_from_db.py" in joined:
            return FakeProc(augment_rc, stdout="augmented ok")
        if cmd[:2] == ["git", "add"]:
            return FakeProc(0)
        if cmd[:3] == ["git", "diff", "--cached"]:
            return FakeProc(diff_rc)
        if cmd[:2] == ["git", "commit"]:
            return FakeProc(commit_rc)
        raise AssertionError(f"unexpected command: {cmd}")

    fake.calls = calls
    return fake


# ---------------------------------------------------------------------------
# Fake repo_root fixtures
# ---------------------------------------------------------------------------

RUN_ID = "discovery_s0014_lotX_2026-08-08"

DOSSIER_TEXT = (
    "# Dossier\n\n"
    "Strong candidates (v6>0.5, post-veto): 7.\n\n"
    # Literal format emitted by discovery_postprocess.py and parsed by discovery_finalize._parse_dossier.
    "**Finalists (pass all + FPP<0.5): 2**: TIC 1, TIC 2\n"
)


def _write_scorecard(repo_root, run_id=RUN_ID):
    p = repo_root / "data" / "scorecards" / f"{run_id}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"run_id": run_id, "metrics": {}}), encoding="utf-8")
    return p


def _write_dossier(repo_root, run_id=RUN_ID):
    p = repo_root / "data" / "scorecards" / f"{run_id}_dossier.md"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(DOSSIER_TEXT, encoding="utf-8")
    return p


def _write_lot_csv(repo_root, name="lot.csv", with_ground_truth=False):
    p = repo_root / "data" / "discovery" / name
    p.parent.mkdir(parents=True, exist_ok=True)
    if with_ground_truth:
        p.write_text("tic_id,disposition\n1,PC\n2,FP\n", encoding="utf-8")
    else:
        p.write_text("tic_id,label\n1,1\n2,0\n", encoding="utf-8")
    return p


def _write_campaign(repo_root, lot_csv_basename, done=True):
    p = repo_root / "data" / "discovery" / "CAMPAIGN.md"
    p.parent.mkdir(parents=True, exist_ok=True)
    status = "**DONE** (2026-08-08) — 2/2 completed" if done else "ENQUEUED"
    p.write_text(f"| lot | `{lot_csv_basename}` | {status} |\n", encoding="utf-8")
    return p


def _write_state(repo_root, *, state="DONE", lot_csv_name="lot.csv", run_id=RUN_ID):
    scorecard = _write_scorecard(repo_root, run_id)
    dossier = _write_dossier(repo_root, run_id)
    state_path = repo_root / "data" / "discovery" / "pipeline_state.json"
    payload = {
        "state": state,
        "run_id": run_id,
        "lot_csv": f"data/discovery/{lot_csv_name}",
        "scorecard_json": str(scorecard.relative_to(repo_root)),
        "dossier": str(dossier.relative_to(repo_root)),
        "sector": 14,
        "started_at": "2026-08-08T08:00:00+00:00",
        "finished_at": "2026-08-08T10:00:00+00:00",
    }
    state_path.write_text(json.dumps(payload), encoding="utf-8")
    return state_path


def _full_setup(tmp_path, *, with_ground_truth=False, campaign_done=True):
    """Build a fake repo_root with all DONE artifacts, valid."""
    lot_csv = _write_lot_csv(tmp_path, with_ground_truth=with_ground_truth)
    state_path = _write_state(tmp_path, lot_csv_name=lot_csv.name)
    _write_campaign(tmp_path, lot_csv.name, done=campaign_done)
    return state_path


# ---------------------------------------------------------------------------
# 1. state != DONE -> blocked, no further step
# ---------------------------------------------------------------------------

def test_state_not_done_blocks_immediately(tmp_path):
    state_path = tmp_path / "pipeline_state.json"
    state_path.write_text(json.dumps({"state": "SCANNING", "run_id": "x"}), encoding="utf-8")

    run_cmd = make_run_cmd()
    result = close_lot(state_path, repo_root=tmp_path, run_cmd=run_cmd, dry_run=True)

    assert result.status == "blocked"
    assert len(result.steps) == 1
    assert result.steps[0]["step"] == "read_state"
    assert run_cmd.calls == []  # no subprocess called


# ---------------------------------------------------------------------------
# 2. health rc=2 -> blocked; digest with the critical (execute); no commit
# ---------------------------------------------------------------------------

def test_health_critical_blocks_and_skips_commit(tmp_path):
    state_path = _full_setup(tmp_path)
    run_cmd = make_run_cmd(health_rc=2)

    result = close_lot(state_path, repo_root=tmp_path, run_cmd=run_cmd, dry_run=False)

    assert result.status == "blocked"
    hc = next(s for s in result.steps if s["step"] == "health_check")
    assert hc["status"] == "blocked"

    digest = (tmp_path / "data" / "autonomy" / "weekly_digest.md").read_text(encoding="utf-8")
    assert "critical" in digest

    commit_step = next(s for s in result.steps if s["step"] == "commit")
    assert commit_step["status"] == "skipped"
    assert not any(c[:2] == ["git", "add"] for c in run_cmd.calls)
    assert not any(c[:2] == ["git", "commit"] for c in run_cmd.calls)


# ---------------------------------------------------------------------------
# 3. health rc=1 -> ok_with_warnings, execution continues
# ---------------------------------------------------------------------------

def test_health_warning_continues_as_ok_with_warnings(tmp_path):
    state_path = _full_setup(tmp_path)
    run_cmd = make_run_cmd(health_rc=1)

    result = close_lot(state_path, repo_root=tmp_path, run_cmd=run_cmd, dry_run=True)

    assert result.status == "ok_with_warnings"
    hc = next(s for s in result.steps if s["step"] == "health_check")
    assert hc["status"] == "warning"
    # later steps run anyway
    assert any(s["step"] == "verify_artifacts" for s in result.steps)
    assert any(s["step"] == "digest" for s in result.steps)


# ---------------------------------------------------------------------------
# 4. conditional augment depending on the lot_csv header
# ---------------------------------------------------------------------------

def _cmd_contains(cmd, needle):
    return any(needle in str(part) for part in cmd)


def test_augment_skipped_for_deep_lot_without_ground_truth(tmp_path):
    state_path = _full_setup(tmp_path, with_ground_truth=False)
    run_cmd = make_run_cmd()

    result = close_lot(state_path, repo_root=tmp_path, run_cmd=run_cmd, dry_run=True)

    aug = next(s for s in result.steps if s["step"] == "augment_exominer")
    assert aug["status"] == "skipped"
    assert "ground truth" in aug["detail"]
    assert not any(_cmd_contains(c, "augment_scorecard_exominer_from_db.py") for c in run_cmd.calls)


def test_augment_executed_for_lot_with_disposition(tmp_path):
    """dry_run=False: the augment really runs via run_cmd (mocked)."""
    state_path = _full_setup(tmp_path, with_ground_truth=True)
    run_cmd = make_run_cmd()

    result = close_lot(state_path, repo_root=tmp_path, run_cmd=run_cmd, dry_run=False)

    aug = next(s for s in result.steps if s["step"] == "augment_exominer")
    assert aug["status"] == "ok"

    aug_cmd = next(c for c in run_cmd.calls if _cmd_contains(c, "augment_scorecard_exominer_from_db.py"))
    assert "--sample" in aug_cmd and "--db" in aug_cmd and "--out" in aug_cmd
    sample_idx = aug_cmd.index("--sample")
    assert aug_cmd[sample_idx + 1].endswith("lot.csv")
    out_idx = aug_cmd.index("--out")
    assert aug_cmd[out_idx + 1].endswith(f"{RUN_ID}.augmented")


def test_augment_never_executed_in_dry_run_even_with_disposition(tmp_path):
    """dry_run=True ONLY reads; never calls augment_scorecard_exominer_from_db.py
    (regression: it wrote 3 real files to data/scorecards/ in dry-run)."""
    state_path = _full_setup(tmp_path, with_ground_truth=True)
    run_cmd = make_run_cmd()

    result = close_lot(state_path, repo_root=tmp_path, run_cmd=run_cmd, dry_run=True)

    aug = next(s for s in result.steps if s["step"] == "augment_exominer")
    assert aug["status"] == "dry_run"
    assert "dry_run" in aug["detail"]
    assert "augment_scorecard_exominer_from_db.py" in aug["detail"]
    assert not any(_cmd_contains(c, "augment_scorecard_exominer_from_db.py") for c in run_cmd.calls)


# ---------------------------------------------------------------------------
# 5. missing artifact -> blocked
# ---------------------------------------------------------------------------

def test_missing_scorecard_blocks(tmp_path):
    lot_csv = _write_lot_csv(tmp_path)
    state_path = _write_state(tmp_path, lot_csv_name=lot_csv.name)
    _write_campaign(tmp_path, lot_csv.name, done=True)

    # delete the scorecard so it does not exist
    (tmp_path / "data" / "scorecards" / f"{RUN_ID}.json").unlink()

    run_cmd = make_run_cmd()
    result = close_lot(state_path, repo_root=tmp_path, run_cmd=run_cmd, dry_run=True)

    assert result.status == "blocked"
    art = next(s for s in result.steps if s["step"] == "verify_artifacts")
    assert art["status"] == "blocked"


# ---------------------------------------------------------------------------
# 6. CAMPAIGN without DONE -> warning, not blocked, file NOT modified
# ---------------------------------------------------------------------------

def test_campaign_not_done_is_warning_and_not_modified(tmp_path):
    state_path = _full_setup(tmp_path, campaign_done=False)
    campaign_path = tmp_path / "data" / "discovery" / "CAMPAIGN.md"
    before = campaign_path.read_text(encoding="utf-8")

    run_cmd = make_run_cmd()
    result = close_lot(state_path, repo_root=tmp_path, run_cmd=run_cmd, dry_run=True)

    camp = next(s for s in result.steps if s["step"] == "verify_campaign")
    assert camp["status"] == "warning"
    assert result.status == "ok_with_warnings"

    after = campaign_path.read_text(encoding="utf-8")
    assert before == after


# ---------------------------------------------------------------------------
# 7. digest: execute writes the block; dry_run writes nothing
# ---------------------------------------------------------------------------

def test_digest_written_on_execute_with_summary_fields(tmp_path):
    state_path = _full_setup(tmp_path)
    run_cmd = make_run_cmd()

    result = close_lot(state_path, repo_root=tmp_path, run_cmd=run_cmd, dry_run=False)

    digest_path = tmp_path / "data" / "autonomy" / "weekly_digest.md"
    assert digest_path.exists()
    text = digest_path.read_text(encoding="utf-8")
    assert text.startswith("# Weekly digest")
    assert RUN_ID in text
    assert "Strong: 7" in text
    assert "Finalists: 2" in text
    assert "sector 14" in text


def test_digest_not_written_on_dry_run(tmp_path):
    state_path = _full_setup(tmp_path)
    run_cmd = make_run_cmd()

    close_lot(state_path, repo_root=tmp_path, run_cmd=run_cmd, dry_run=True)

    digest_path = tmp_path / "data" / "autonomy" / "weekly_digest.md"
    assert not digest_path.exists()


# ---------------------------------------------------------------------------
# 8. commit: git add only with existing paths; dry_run never calls git; never push
# ---------------------------------------------------------------------------

def test_commit_adds_only_existing_paths_and_commits(tmp_path):
    state_path = _full_setup(tmp_path)
    # docs/STATUS.md does not exist -> must not be added
    run_cmd = make_run_cmd(diff_rc=1, commit_rc=0)

    result = close_lot(state_path, repo_root=tmp_path, run_cmd=run_cmd, dry_run=False)

    add_call = next(c for c in run_cmd.calls if c[:2] == ["git", "add"])
    added = add_call[2:]
    assert any(p.endswith(f"{RUN_ID}.json") for p in added)
    assert any(p.endswith(f"{RUN_ID}_dossier.md") for p in added)
    assert any(p.endswith("CAMPAIGN.md") for p in added)
    assert not any(p.endswith("STATUS.md") for p in added)

    commit_call = next(c for c in run_cmd.calls if c[:2] == ["git", "commit"])
    assert not any("push" in c for c in run_cmd.calls)

    # explicit pathspec in the commit: never takes along anything else already
    # staged in the live checkout (review round 1 finding).
    assert "--" in commit_call
    pathspec = commit_call[commit_call.index("--") + 1:]
    assert set(pathspec) == set(added)

    diff_call = next(c for c in run_cmd.calls if c[:3] == ["git", "diff", "--cached"])
    assert "--" in diff_call
    assert set(diff_call[diff_call.index("--") + 1:]) == set(added)

    commit_step = next(s for s in result.steps if s["step"] == "commit")
    assert commit_step["status"] == "ok"


def test_commit_skipped_when_nothing_staged(tmp_path):
    state_path = _full_setup(tmp_path)
    run_cmd = make_run_cmd(diff_rc=0)  # nothing staged

    result = close_lot(state_path, repo_root=tmp_path, run_cmd=run_cmd, dry_run=False)

    commit_step = next(s for s in result.steps if s["step"] == "commit")
    assert commit_step["status"] == "skipped"
    assert not any(c[:2] == ["git", "commit"] for c in run_cmd.calls)


def test_dry_run_no_git_calls(tmp_path):
    state_path = _full_setup(tmp_path)
    run_cmd = make_run_cmd()

    close_lot(state_path, repo_root=tmp_path, run_cmd=run_cmd, dry_run=True)

    assert not any(c[0] == "git" for c in run_cmd.calls)
    assert not any("push" in c for c in run_cmd.calls)


# ---------------------------------------------------------------------------
# 9. summary: _db_stats/_parse_dossier mocked
# ---------------------------------------------------------------------------

def test_summary_uses_db_stats_and_dossier(tmp_path, monkeypatch):
    state_path = _full_setup(tmp_path)
    run_cmd = make_run_cmd()

    calls = {}

    def fake_db_stats(db_path, tic_ids):
        calls["db_stats"] = (db_path, list(tic_ids))
        return {"n_lot": 240, "n_completed": 180, "n_failed": 0, "n_other": 60}

    def fake_parse_dossier(dossier_path):
        calls["parse_dossier"] = dossier_path
        return {"n_strong": 9, "n_finalists": 3}

    monkeypatch.setattr("autonomy.closeout._db_stats", fake_db_stats)
    monkeypatch.setattr("autonomy.closeout._parse_dossier", fake_parse_dossier)

    result = close_lot(state_path, repo_root=tmp_path, run_cmd=run_cmd, dry_run=True)

    assert result.summary["n_lot"] == 240
    assert result.summary["n_completed"] == 180
    assert result.summary["n_strong"] == 9
    assert result.summary["n_finalists"] == 3
    assert result.summary["duration_h"] == 2.0
    # 180 completed / 120 min -> 1.5 TICs/min
    assert result.summary["rate_tics_min"] == 1.5
    assert calls["db_stats"][0]  # db_path passed
    summary_step = next(s for s in result.steps if s["step"] == "summary")
    assert summary_step["status"] == "ok"


def test_summary_survives_db_stats_exception(tmp_path, monkeypatch):
    state_path = _full_setup(tmp_path)
    run_cmd = make_run_cmd()

    def raising_db_stats(db_path, tic_ids):
        raise RuntimeError("db unreachable")

    monkeypatch.setattr("autonomy.closeout._db_stats", raising_db_stats)

    # must not propagate the exception
    result = close_lot(state_path, repo_root=tmp_path, run_cmd=run_cmd, dry_run=True)

    assert result.summary["n_lot"] is None
    assert result.summary["n_completed"] is None
    summary_step = next(s for s in result.steps if s["step"] == "summary")
    assert summary_step["status"] == "warning"
    # the closeout does not blow up on an incomplete summary: the remaining steps continue
    assert any(s["step"] == "digest" for s in result.steps)


# ---------------------------------------------------------------------------
# 10. _resolve_db: EXONIT_DATABASE_URL > EXONIT_DB_PATH > default (review round 2)
# ---------------------------------------------------------------------------
# Real bug found while testing the 300k lot: production runs on PostgreSQL via
# EXONIT_DATABASE_URL, but resolution fell back to `data/exonit.db` (ARCHIVED
# SQLite) because it only looked at EXONIT_DB_PATH -> false n_completed=0 for a
# 100% completed lot.

DSN = "postgresql://user:pw@host:5432/exonit"


def test_resolve_db_defaults_to_archived_sqlite_when_no_env(tmp_path, monkeypatch):
    monkeypatch.delenv("EXONIT_DATABASE_URL", raising=False)
    monkeypatch.delenv("EXONIT_DB_PATH", raising=False)

    assert _resolve_db(tmp_path) == str(tmp_path / "data" / "exonit.db")


def test_resolve_db_prefers_exonit_db_path_over_default(tmp_path, monkeypatch):
    monkeypatch.delenv("EXONIT_DATABASE_URL", raising=False)
    monkeypatch.setenv("EXONIT_DB_PATH", "/custom/sqlite.db")

    assert _resolve_db(tmp_path) == "/custom/sqlite.db"


def test_resolve_db_prefers_database_url_over_everything(tmp_path, monkeypatch):
    monkeypatch.setenv("EXONIT_DATABASE_URL", DSN)
    monkeypatch.setenv("EXONIT_DB_PATH", "/custom/sqlite.db")

    assert _resolve_db(tmp_path) == DSN


def test_close_lot_summary_and_augment_use_database_url_when_present(tmp_path, monkeypatch):
    """Integration: both the summary (_db_stats) and the augment (--db) receive the
    PostgreSQL DSN when EXONIT_DATABASE_URL is present, even if
    EXONIT_DB_PATH is also set (strict priority)."""
    state_path = _full_setup(tmp_path, with_ground_truth=True)
    run_cmd = make_run_cmd()

    monkeypatch.setenv("EXONIT_DATABASE_URL", DSN)
    monkeypatch.setenv("EXONIT_DB_PATH", "/custom/sqlite.db")

    seen_db_paths = []

    def fake_db_stats(db_path, tic_ids):
        seen_db_paths.append(db_path)
        return {"n_lot": 2, "n_completed": 2, "n_failed": 0, "n_other": 0}

    monkeypatch.setattr("autonomy.closeout._db_stats", fake_db_stats)

    result = close_lot(state_path, repo_root=tmp_path, run_cmd=run_cmd, dry_run=False)

    assert seen_db_paths == [DSN]

    aug_cmd = next(c for c in run_cmd.calls if _cmd_contains(c, "augment_scorecard_exominer_from_db.py"))
    db_idx = aug_cmd.index("--db")
    assert aug_cmd[db_idx + 1] == DSN

    # the result does not depend on step order; sanity check that the lot
    # could be closed without blowing up
    assert result.status in ("ok", "ok_with_warnings", "blocked")
