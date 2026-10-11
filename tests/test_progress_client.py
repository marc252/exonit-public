"""The progress client must never cost the caller anything.

Every test here asserts a NON-event: no exception, no hang, no work when
disabled. This is the most important test file of the feature — a client that
raises would turn telemetry into an outage.
"""

import time

import pytest

from exonit import progress_client as PC


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for var in ("EXONIT_PROGRESS_RUN_ID", "EXONIT_PROGRESS_PHASE",
                "EXONIT_PROGRESS_URL", "EXONIT_SERVER_URL",
                "EXONIT_WORKER_TOKEN"):
        monkeypatch.delenv(var, raising=False)


def _enable(monkeypatch, url="http://127.0.0.1:9/"):
    monkeypatch.setenv("EXONIT_PROGRESS_RUN_ID", "run-A")
    monkeypatch.setenv("EXONIT_PROGRESS_PHASE", "FINALIZE")
    monkeypatch.setenv("EXONIT_PROGRESS_URL", url)
    monkeypatch.setenv("EXONIT_WORKER_TOKEN", "tok")


# --- disabled by default -------------------------------------------------

def test_disabled_without_env_vars():
    r = PC.ProgressReporter(unit="buckets", total=10)
    assert r.enabled is False
    r.report(1)          # must not raise
    r.finish()


def test_disabled_when_only_run_id_is_set(monkeypatch):
    monkeypatch.setenv("EXONIT_PROGRESS_RUN_ID", "run-A")
    assert PC.ProgressReporter().enabled is False


def test_disabled_reporter_never_opens_a_connection(monkeypatch):
    called = []
    monkeypatch.setattr(PC, "_post", lambda *a, **k: called.append(a))
    PC.ProgressReporter(unit="buckets").report(1)
    assert called == []


# --- resilience ----------------------------------------------------------

def test_report_does_not_raise_when_server_is_unreachable(monkeypatch):
    _enable(monkeypatch)                 # port 9 = discard, refuses
    r = PC.ProgressReporter(unit="buckets", total=3)
    r.report(1, force=True)              # must not raise


def test_report_does_not_raise_on_http_500(monkeypatch):
    _enable(monkeypatch)

    def boom(url, body, token, timeout):
        import urllib.error
        raise urllib.error.HTTPError(url, 500, "boom", {}, None)

    monkeypatch.setattr(PC, "_post", boom)
    PC.ProgressReporter(unit="buckets").report(1, force=True)


def test_report_does_not_raise_on_timeout(monkeypatch):
    _enable(monkeypatch)

    def slow(url, body, token, timeout):
        raise TimeoutError("timed out")

    monkeypatch.setattr(PC, "_post", slow)
    PC.ProgressReporter(unit="buckets").report(1, force=True)


def test_report_does_not_raise_on_any_exception(monkeypatch):
    _enable(monkeypatch)

    def weird(url, body, token, timeout):
        raise RuntimeError("something nobody predicted")

    monkeypatch.setattr(PC, "_post", weird)
    PC.ProgressReporter(unit="buckets").report(1, force=True)


def test_failures_are_logged_once_per_burst(monkeypatch, caplog):
    _enable(monkeypatch)
    monkeypatch.setattr(PC, "_post",
                        lambda *a, **k: (_ for _ in ()).throw(OSError("down")))
    r = PC.ProgressReporter(unit="buckets", interval_s=0)
    for _ in range(10):
        r.report(1, force=True)
    warnings = [rec for rec in caplog.records if rec.levelname == "WARNING"]
    assert len(warnings) == 1


# --- rate limiting -------------------------------------------------------

def test_marks_are_rate_limited_to_one_per_interval(monkeypatch):
    _enable(monkeypatch)
    sent = []
    monkeypatch.setattr(PC, "_post",
                        lambda url, body, token, timeout: sent.append(body))
    r = PC.ProgressReporter(unit="buckets", total=100, interval_s=60)
    r.report(1)          # first one always goes (start of phase)
    r.report(2)
    r.report(3)
    assert len(sent) == 1


def test_force_bypasses_the_rate_limit(monkeypatch):
    _enable(monkeypatch)
    sent = []
    monkeypatch.setattr(PC, "_post",
                        lambda url, body, token, timeout: sent.append(body))
    r = PC.ProgressReporter(unit="buckets", total=100, interval_s=60)
    r.report(1)
    r.report(2, force=True)
    assert len(sent) == 2


def test_finish_always_sends(monkeypatch):
    _enable(monkeypatch)
    sent = []
    monkeypatch.setattr(PC, "_post",
                        lambda url, body, token, timeout: sent.append(body))
    r = PC.ProgressReporter(unit="buckets", total=5, interval_s=3600)
    r.report(1)
    r.finish(message="done")
    assert len(sent) == 2
    assert sent[-1]["message"] == "done"
    assert sent[-1]["done"] == sent[-1]["total"] == 5


def test_interval_elapsed_lets_the_next_mark_through(monkeypatch):
    _enable(monkeypatch)
    sent = []
    monkeypatch.setattr(PC, "_post",
                        lambda url, body, token, timeout: sent.append(body))
    r = PC.ProgressReporter(unit="buckets", total=100, interval_s=0.05)
    r.report(1)
    time.sleep(0.06)
    r.report(2)
    assert len(sent) == 2


# --- payload -------------------------------------------------------------

def test_payload_carries_identity_and_interval(monkeypatch):
    _enable(monkeypatch)
    sent = []
    monkeypatch.setattr(PC, "_post",
                        lambda url, body, token, timeout: sent.append(body))
    r = PC.ProgressReporter(unit="buckets", total=56,
                            substep="compact_completeness", interval_s=10)
    r.report(12, message="compacting bucket 012")
    body = sent[0]
    assert body["run_id"] == "run-A"
    assert body["phase"] == "FINALIZE"
    assert body["substep"] == "compact_completeness"
    assert body["unit"] == "buckets"
    assert body["done"] == 12
    assert body["total"] == 56
    assert body["interval_s"] == 10
    assert body["pid"] > 0
    assert body["host"]


def test_substep_can_be_changed_per_report(monkeypatch):
    _enable(monkeypatch)
    sent = []
    monkeypatch.setattr(PC, "_post",
                        lambda url, body, token, timeout: sent.append(body))
    r = PC.ProgressReporter(unit="steps", total=2, interval_s=0)
    r.report(0, substep="postprocess")
    r.report(1, substep="scorecard")
    assert [b["substep"] for b in sent] == ["postprocess", "scorecard"]


def test_url_falls_back_to_server_url(monkeypatch):
    monkeypatch.setenv("EXONIT_PROGRESS_RUN_ID", "run-A")
    monkeypatch.setenv("EXONIT_PROGRESS_PHASE", "FINALIZE")
    monkeypatch.setenv("EXONIT_SERVER_URL", "http://example:8400")
    r = PC.ProgressReporter()
    assert r.enabled is True
    assert r.url == "http://example:8400/api/progress"


# --- explicit identity (used by the orchestrator, which spawns no child) ---

def test_explicit_run_id_and_phase_win_over_the_environment(monkeypatch):
    monkeypatch.setenv("EXONIT_PROGRESS_RUN_ID", "from-env")
    monkeypatch.setenv("EXONIT_PROGRESS_PHASE", "SCANNING")
    monkeypatch.setenv("EXONIT_SERVER_URL", "http://example:8400")
    r = PC.ProgressReporter(run_id="explicit", phase="FREE_GPU")
    assert (r.run_id, r.phase) == ("explicit", "FREE_GPU")
    assert r.enabled is True


def test_explicit_empty_run_id_disables_without_touching_the_environment(monkeypatch):
    monkeypatch.setenv("EXONIT_SERVER_URL", "http://example:8400")
    r = PC.ProgressReporter(run_id=None, phase="FREE_GPU")
    assert r.enabled is False
    assert "EXONIT_PROGRESS_RUN_ID" not in __import__("os").environ
