from autonomy.types import ErrorContext
from autonomy.classifier import classify


def _ctx(detail="", log="", state="FAILED", fleet=None, recent=None, run_id="lotX"):
    return ErrorContext(run_id, "FPP_BACKFILL", state, detail, log,
                        fleet or {}, recent or [])


def test_disk_full_mac():
    fleet = {"worker-mac-1": {"disk_free_gb": 1.2}, "worker-linux-gpu": {"disk_free_gb": 900}}
    c = classify(_ctx(detail="OSError: [Errno 28] No space left on device", fleet=fleet))
    assert c.playbook == "clean_fits_cache" and c.params["host"] == "worker-mac-1"


def test_disk_full_unknown_host_escalates():
    c = classify(_ctx(detail="Errno 28", fleet={"worker-linux-gpu": {"disk_free_gb": 900}}))
    assert c.playbook == "pause_and_alert"


def test_stream_drift_fpp():
    c = classify(_ctx(state="FPP_BACKFILL", detail="queue full but workers idle after the restart"))
    assert c.playbook == "rebuild_streams" and c.params["stage"] == "fpp"


def test_orphans():
    assert classify(_ctx(detail="12 tasks assigned without heartbeat")).playbook == "requeue_orphans"


def test_orphans_positive_mention_still_fires():
    c = classify(_ctx(detail="12 orphan tasks detected"))
    assert c is not None and c.playbook == "requeue_orphans"


def test_orphans_negated_zero_count_does_not_fire():
    # Benign log: "no orphans" and "0 stragglers" near the term must not
    # trigger requeue_orphans (review round 1 regression, commit 723b437).
    c = classify(_ctx(detail="checked orphan tasks: no orphans found, 0 stragglers"))
    assert c is None


def test_orphans_negation_and_positive_mention_coexist_still_fires():
    # Review round 2/3 regression: a negation in one sentence must not hide
    # a real positive mention in another sentence of the same text. The
    # positive sentence does NOT contain "assigned" or "heartbeat" (independent
    # OR branch), so this test only passes if the negation guard is really
    # localized to the sentence.
    c = classify(_ctx(
        detail="Pre-check: no orphans found. Post-restart: 12 orphan tasks "
        "require manual requeue."
    ))
    assert c is not None and c.playbook == "requeue_orphans"


def test_orphans_negation_in_detail_positive_in_log_tail_still_fires():
    # Review round 3 regression: `_text()` joins detail and log_tail with "\n";
    # the negation in `detail` must not "eat" across the line break
    # a real positive mention in `log_tail`. No "assigned"/"heartbeat".
    c = classify(_ctx(
        detail="Pre-check: no orphans found.",
        log="12 orphan tasks require manual requeue",
    ))
    assert c is not None and c.playbook == "requeue_orphans"


def test_transient_resumes_once():
    c = classify(_ctx(detail="ConnectionError: Connection refused", run_id="lot9"))
    assert c.playbook == "resume_run" and c.params["run_id"] == "lot9"
    c2 = classify(_ctx(detail="Connection refused", recent=["resume_run"]))
    assert c2 is None  # already attempted: let the AI decide or escalate


def test_unknown_returns_none():
    assert classify(_ctx(detail="something never seen before")) is None
