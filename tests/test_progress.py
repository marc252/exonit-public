"""Model + store for the pipeline progress records (exonit/progress.py).

Constraints: pure unit tests over a MemoryBackend — never the real Redis,
never the real pipeline_state.json (see tests/conftest.py autouse fixtures).
"""

import json

import pytest

from exonit import progress as P
from exonit.shared_state import MemoryBackend


@pytest.fixture
def state():
    return MemoryBackend()


def _mark(**over):
    base = {
        "run_id": "run-A",
        "phase": "FINALIZE",
        "substep": "compact_completeness",
        "unit": "buckets",
        "done": 0,
        "total": 56,
        "message": "start",
        "host": "<host>",
        "pid": 4242,
        "interval_s": 10,
    }
    base.update(over)
    return base


# --- validation ----------------------------------------------------------

def test_normalise_requires_run_id_and_phase():
    with pytest.raises(P.ProgressError):
        P.normalise_mark(_mark(run_id=""))
    with pytest.raises(P.ProgressError):
        P.normalise_mark(_mark(phase=""))


def test_normalise_rejects_negative_done():
    with pytest.raises(P.ProgressError):
        P.normalise_mark(_mark(done=-1))


def test_normalise_rejects_non_positive_interval():
    with pytest.raises(P.ProgressError):
        P.normalise_mark(_mark(interval_s=0))


def test_normalise_allows_null_total():
    out = P.normalise_mark(_mark(total=None))
    assert out["total"] is None


def test_normalise_truncates_message():
    out = P.normalise_mark(_mark(message="x" * 5000))
    assert len(out["message"]) <= 500


# --- storage round-trip --------------------------------------------------

def test_record_then_read_round_trip(state):
    P.record_mark(state, _mark(done=12), now=1000.0)
    got = P.read_progress(state, "run-A", "FINALIZE", now=1001.0)
    assert got["done"] == 12
    assert got["total"] == 56
    assert got["substep"] == "compact_completeness"
    assert got["stale"] is False


def test_read_returns_none_when_absent(state):
    assert P.read_progress(state, "nope", "FINALIZE", now=1.0) is None


def test_stored_payload_has_no_integer_keys(state):
    """Redis round-trips through JSON: integer dict keys become strings."""
    P.record_mark(state, _mark(done=1), now=1000.0)
    raw = json.loads(state.get(P.progress_key("run-A", "FINALIZE")))
    assert isinstance(raw["samples"], list)
    for sample in raw["samples"]:
        assert isinstance(sample, list) and len(sample) == 2


# --- monotonic guard -----------------------------------------------------

def test_done_never_goes_backwards_within_a_substep(state):
    P.record_mark(state, _mark(done=30), now=1000.0)
    P.record_mark(state, _mark(done=7), now=1010.0)
    assert P.read_progress(state, "run-A", "FINALIZE", now=1011.0)["done"] == 30


def test_substep_change_resets_the_counter(state):
    P.record_mark(state, _mark(done=30), now=1000.0)
    P.record_mark(state, _mark(substep="wal_checkpoint", done=0, total=3),
                  now=1010.0)
    got = P.read_progress(state, "run-A", "FINALIZE", now=1011.0)
    assert got["done"] == 0
    assert got["substep"] == "wal_checkpoint"
    assert got["total"] == 3


def test_total_change_resets_the_counter(state):
    # Deep-rescan s17 (2026-08-23): the first server-fed mark of the run was
    # derived from a stale health payload still carrying the PREVIOUS run's
    # baseline, so done=51769 arrived with total=85779; once the baseline was
    # re-seeded to 34010 the monotonic guard kept done frozen at 51769 and the
    # phase showed "51,769 of 34,010", remaining 0, for its whole life. A
    # total change is a baseline re-seed → new epoch, same as a substep change.
    P.record_mark(state, _mark(done=51769, total=85779), now=1000.0)
    P.record_mark(state, _mark(done=804, total=34010), now=1010.0)
    got = P.read_progress(state, "run-A", "FINALIZE", now=1011.0)
    assert got["done"] == 804
    assert got["total"] == 34010
    # New epoch also re-bases the since-start average: done_at_start must be
    # the new mark's done, not 0 (which would credit 804 tasks to 10 s).
    raw = json.loads(state.get(P.progress_key("run-A", "FINALIZE")))
    assert raw["done_at_start"] == 804


def test_missing_total_keeps_the_monotonic_guard(state):
    # Marks without a total on either side cannot prove an epoch change: the
    # out-of-order protection must survive.
    P.record_mark(state, _mark(done=30, total=None), now=1000.0)
    P.record_mark(state, _mark(done=7, total=None), now=1010.0)
    assert P.read_progress(state, "run-A", "FINALIZE", now=1011.0)["done"] == 30


# --- sample ring ---------------------------------------------------------

def test_ring_is_capped_by_sample_count(state):
    for i in range(P.RING_MAX_SAMPLES + 20):
        P.record_mark(state, _mark(done=i), now=1000.0 + i)
    raw = json.loads(state.get(P.progress_key("run-A", "FINALIZE")))
    assert len(raw["samples"]) == P.RING_MAX_SAMPLES


def test_ring_drops_samples_older_than_the_age_cap(state):
    P.record_mark(state, _mark(done=0), now=1000.0)
    P.record_mark(state, _mark(done=5), now=1000.0 + P.RING_MAX_AGE_S + 10)
    raw = json.loads(state.get(P.progress_key("run-A", "FINALIZE")))
    assert [s[1] for s in raw["samples"]] == [5]


# --- rate + ETA ----------------------------------------------------------

def test_rate_needs_a_minimum_window(state):
    P.record_mark(state, _mark(done=0), now=1000.0)
    P.record_mark(state, _mark(done=10), now=1030.0)   # 30 s < 120 s
    got = P.read_progress(state, "run-A", "FINALIZE", now=1030.0)
    assert got["rate_source"] == "avg"


def test_rate_from_window_once_wide_enough(state):
    P.record_mark(state, _mark(done=0), now=1000.0)
    P.record_mark(state, _mark(done=20), now=1200.0)   # 200 s, +20
    got = P.read_progress(state, "run-A", "FINALIZE", now=1200.0)
    assert got["rate_source"] == "window"
    assert got["rate"] == pytest.approx(0.1)
    # 56 - 20 = 36 buckets left at 0.1/s
    assert got["eta_phase_s"] == pytest.approx(360.0)


def test_eta_is_none_when_total_unknown(state):
    P.record_mark(state, _mark(total=None, done=0), now=1000.0)
    P.record_mark(state, _mark(total=None, done=20), now=1200.0)
    got = P.read_progress(state, "run-A", "FINALIZE", now=1200.0)
    assert got["eta_phase_s"] is None


def test_rate_is_none_when_nothing_moved(state):
    P.record_mark(state, _mark(done=0), now=1000.0)
    P.record_mark(state, _mark(done=0), now=1300.0)
    got = P.read_progress(state, "run-A", "FINALIZE", now=1300.0)
    assert got["rate"] is None
    assert got["eta_phase_s"] is None


# --- trailing rate window (2026-08-01) -----------------------------------

def test_rate_window_is_trailing_not_since_the_start_of_the_ring(state):
    """A burst long ago must not keep inflating the rate for the whole ring.

    The window used to anchor on the ring's OLDEST sample, so a phase that
    moved 1,000 items in its first minute and 10/min since still reported the
    average of the two — decaying slowly over 45 minutes instead of showing
    what is happening now.
    """
    t = 1000.0
    P.record_mark(state, _mark(total=None, done=0), now=t)
    P.record_mark(state, _mark(total=None, done=1000), now=t + 60.0)
    # Then a long, slow tail well past RATE_WINDOW_S: +1 per minute.
    done = 1000
    for i in range(1, 11):
        done += 1
        P.record_mark(state, _mark(total=None, done=done), now=t + 60.0 + i * 60.0)

    now = t + 660.0
    got = P.read_progress(state, "run-A", "FINALIZE", now=now)
    assert got["rate_source"] == "window"
    # The trailing 300 s hold only the +1/min tail => ~0.0167/s. Anchoring on
    # the ring's oldest sample would give ~1.53/s, ~90x higher.
    assert got["rate"] == pytest.approx(1.0 / 60.0, rel=0.2)


def test_rate_falls_back_to_the_full_ring_before_the_window_fills(state):
    """Early in a phase the trailing window is narrower than the minimum."""
    P.record_mark(state, _mark(done=0), now=1000.0)
    P.record_mark(state, _mark(done=20), now=1200.0)
    got = P.read_progress(state, "run-A", "FINALIZE", now=1200.0)
    assert got["rate_source"] == "window"
    assert got["rate"] == pytest.approx(0.1)


# --- rebuilt records (2026-08-01) ----------------------------------------

def test_avg_does_not_count_work_done_before_the_record_existed(state):
    """A record rebuilt mid-phase must not claim the whole phase's progress.

    Regression: the FPP record expired an hour into a phase and was rebuilt
    with done already at ~33,000. phase_started_ts reset to "now", so the
    since-start average read 33,000 items over a few minutes — thousands per
    minute for work that had taken over an hour.
    """
    # Rebuilt record: its very first mark already reports 30,000 done.
    P.record_mark(state, _mark(total=None, done=30_000), now=1000.0)
    P.record_mark(state, _mark(total=None, done=30_010), now=1060.0)
    got = P.read_progress(state, "run-A", "FINALIZE", now=1060.0)
    assert got["rate_source"] == "avg"
    # 10 items in 60 s, NOT 30,010 in 60 s.
    assert got["rate"] == pytest.approx(10.0 / 60.0)


def test_avg_is_unchanged_for_a_phase_that_genuinely_starts_at_zero(state):
    P.record_mark(state, _mark(total=None, done=0), now=1000.0)
    P.record_mark(state, _mark(total=None, done=30), now=1030.0)
    got = P.read_progress(state, "run-A", "FINALIZE", now=1030.0)
    assert got["rate_source"] == "avg"
    assert got["rate"] == pytest.approx(1.0)


# --- observation timestamps (2026-08-01) ---------------------------------

def test_sample_is_dated_by_observation_not_by_arrival(state):
    """A mark derived from a cached reading is dated when it was read.

    Otherwise the work represented by a stale reading is compressed into the
    gap since the previous mark and reported as a burst — the ~7,000 TIC/min
    the FPP gauge showed on 2026-08-01.
    """
    # Reading taken at t=1000 but only recorded at t=1300 (served from cache).
    P.record_mark(state, _mark(total=None, done=0), now=1300.0,
                  observed_ts=1000.0)
    P.record_mark(state, _mark(total=None, done=300), now=1310.0,
                  observed_ts=1310.0)
    got = P.read_progress(state, "run-A", "FINALIZE", now=1310.0)
    assert got["rate_source"] == "window"
    # 300 items over the REAL 310 s, not over the 10 s between arrivals.
    assert got["rate"] == pytest.approx(300.0 / 310.0)


def test_staleness_still_tracks_the_reporter_not_the_reading(state):
    """updated_ts stays on the wall clock: stale means nobody is reporting."""
    P.record_mark(state, _mark(done=0), now=1000.0, observed_ts=400.0)
    got = P.read_progress(state, "run-A", "FINALIZE", now=1005.0)
    assert got["age_s"] == pytest.approx(5.0)
    assert got["stale"] is False


def test_an_observation_older_than_the_ring_head_is_not_appended(state):
    """Out-of-order readings must not invert the window."""
    P.record_mark(state, _mark(total=None, done=10), now=1000.0,
                  observed_ts=1000.0)
    P.record_mark(state, _mark(total=None, done=20), now=1010.0,
                  observed_ts=500.0)
    raw = json.loads(state.get(P.progress_key("run-A", "FINALIZE")))
    assert [s[0] for s in raw["samples"]] == [1000.0]
    # The mark itself still landed — only the sample was dropped.
    assert raw["done"] == 20


def test_an_observation_from_the_future_is_clamped(state):
    P.record_mark(state, _mark(done=0), now=1000.0, observed_ts=9999.0)
    raw = json.loads(state.get(P.progress_key("run-A", "FINALIZE")))
    assert raw["samples"][0][0] == 1000.0


def test_feeder_dates_the_sample_by_the_payloads_computed_at(state):
    """feed_queue_phase must honour health["computed_at"] (routes/admin.py)."""
    health = {"primary_tasks": {"baseline": 100, "done": 0},
              "computed_at": 500.0}
    P.feed_queue_phase(state, run_id="run-A", phase="SCANNING", health=health,
                       now=900.0)
    raw = json.loads(state.get(P.progress_key("run-A", "SCANNING")))
    assert raw["samples"][0][0] == 500.0
    assert raw["updated_ts"] == 900.0


def test_feeder_without_computed_at_falls_back_to_now(state):
    health = {"primary_tasks": {"baseline": 100, "done": 0}}
    P.feed_queue_phase(state, run_id="run-A", phase="SCANNING", health=health,
                       now=900.0)
    raw = json.loads(state.get(P.progress_key("run-A", "SCANNING")))
    assert raw["samples"][0][0] == 900.0


# --- staleness -----------------------------------------------------------

def test_not_stale_within_three_intervals(state):
    P.record_mark(state, _mark(interval_s=10), now=1000.0)
    got = P.read_progress(state, "run-A", "FINALIZE", now=1029.0)
    assert got["stale"] is False


def test_stale_after_three_intervals(state):
    P.record_mark(state, _mark(interval_s=10), now=1000.0)
    got = P.read_progress(state, "run-A", "FINALIZE", now=1031.0)
    assert got["stale"] is True
    assert got["age_s"] == pytest.approx(31.0)


# --- TTL -----------------------------------------------------------------

def test_record_is_written_with_a_ttl(state):
    captured = {}
    original = state.set

    def spy(key, value, *, ttl=None):
        captured[key] = ttl
        return original(key, value, ttl=ttl)

    state.set = spy
    P.record_mark(state, _mark(), now=1000.0)
    assert captured[P.progress_key("run-A", "FINALIZE")] == P.PROGRESS_TTL_S


# --- current pipeline state ---------------------------------------------

def test_current_pipeline_state_missing_file(tmp_path, monkeypatch):
    monkeypatch.setenv("EXONIT_PIPELINE_STATE_FILE",
                       str(tmp_path / "absent.json"))
    assert P.current_pipeline_state() == {}


def test_current_pipeline_state_malformed_file(tmp_path, monkeypatch):
    bad = tmp_path / "pipeline_state.json"
    bad.write_text("{not json")
    monkeypatch.setenv("EXONIT_PIPELINE_STATE_FILE", str(bad))
    assert P.current_pipeline_state() == {}


def test_current_pipeline_state_reads_run_id_and_state(tmp_path, monkeypatch):
    good = tmp_path / "pipeline_state.json"
    good.write_text(json.dumps({"state": "FINALIZE", "run_id": "run-A"}))
    monkeypatch.setenv("EXONIT_PIPELINE_STATE_FILE", str(good))
    out = P.current_pipeline_state()
    assert out["state"] == "FINALIZE"
    assert out["run_id"] == "run-A"


# --- env handed to a spawned child --------------------------------------

def test_child_env_carries_the_triplet(monkeypatch):
    monkeypatch.setenv("EXONIT_SERVER_URL", "http://<server-ip>:8400")
    env = P.child_env("run-A", "POSTPROCESS", base={"PATH": "/usr/bin"})
    assert env["EXONIT_PROGRESS_RUN_ID"] == "run-A"
    assert env["EXONIT_PROGRESS_PHASE"] == "POSTPROCESS"
    assert env["EXONIT_PROGRESS_URL"] == "http://<server-ip>:8400"
    assert env["PATH"] == "/usr/bin"


def test_child_env_without_a_run_id_adds_nothing(monkeypatch):
    env = P.child_env(None, "POSTPROCESS", base={"PATH": "/usr/bin"})
    assert env == {"PATH": "/usr/bin"}


def test_child_env_does_not_mutate_its_base():
    base = {"PATH": "/usr/bin"}
    P.child_env("run-A", "FINALIZE", base=base)
    assert base == {"PATH": "/usr/bin"}


def test_child_env_keeps_an_explicit_progress_url(monkeypatch):
    monkeypatch.setenv("EXONIT_SERVER_URL", "http://ignored:1")
    env = P.child_env("run-A", "FINALIZE",
                      base={"EXONIT_PROGRESS_URL": "http://kept:8400"})
    assert env["EXONIT_PROGRESS_URL"] == "http://kept:8400"


# --- ENQUEUEING has no run_id of its own ---------------------------------

def test_enqueueing_reports_under_a_fixed_key():
    """Its run_id is derived from the lot CSV the child is still writing."""
    assert P.progress_lookup_run_id("ENQUEUEING", None) == P.ENQUEUEING_RUN_ID


def test_enqueueing_ignores_a_stale_run_id_in_the_state_file():
    """pipeline_state.json merges, so during ENQUEUEING it can still carry the
    PREVIOUS run's id — reading progress under it would show a finished run."""
    assert P.progress_lookup_run_id("ENQUEUEING", "run-anterior") == P.ENQUEUEING_RUN_ID


def test_every_other_phase_uses_its_own_run_id():
    assert P.progress_lookup_run_id("FINALIZE", "run-A") == "run-A"


# --- server-side feeder for the queue phases -----------------------------

class TestQueuePhaseFeeder:
    """The server derives queue-phase progress from counters it already has."""

    def test_scanning_uses_primary_task_counters(self, state):
        health = {"primary_tasks": {"done": 120, "baseline": 1000,
                                    "pending": 800, "assigned": 80}}
        P.feed_queue_phase(state, run_id="run-A", phase="SCANNING",
                           health=health, now=1000.0)
        got = P.read_progress(state, "run-A", "SCANNING", now=1000.0)
        assert got["done"] == 120
        assert got["total"] == 1000
        assert got["unit"] == "tasks"
        assert got["source"] == "server"

    def test_exominer_uses_its_own_counters(self, state):
        health = {"exominer_backfill": {"done": 5, "baseline": 40, "pending": 35}}
        P.feed_queue_phase(state, run_id="run-A", phase="EXOMINER_BACKFILL",
                           health=health, now=1000.0)
        got = P.read_progress(state, "run-A", "EXOMINER_BACKFILL", now=1000.0)
        assert (got["done"], got["total"]) == (5, 40)

    def test_fpp_baseline_is_captured_on_first_sight(self, state):
        """FPP has no server-side baseline: the queue depth at entry IS it."""
        P.feed_queue_phase(state, run_id="run-A", phase="FPP_BACKFILL",
                           health={"fpp_backfill": {"actionable": 500}},
                           now=1000.0)
        P.feed_queue_phase(state, run_id="run-A", phase="FPP_BACKFILL",
                           health={"fpp_backfill": {"actionable": 300}},
                           now=1100.0)
        got = P.read_progress(state, "run-A", "FPP_BACKFILL", now=1100.0)
        assert got["total"] == 500
        assert got["done"] == 200

    def test_fpp_baseline_survives_a_queue_that_grows(self, state):
        P.feed_queue_phase(state, run_id="run-A", phase="FPP_BACKFILL",
                           health={"fpp_backfill": {"actionable": 100}},
                           now=1000.0)
        P.feed_queue_phase(state, run_id="run-A", phase="FPP_BACKFILL",
                           health={"fpp_backfill": {"actionable": 400}},
                           now=1100.0)
        got = P.read_progress(state, "run-A", "FPP_BACKFILL", now=1100.0)
        assert got["done"] == 0          # monotonic guard, never negative

    def test_lora_uses_stale_plus_missing(self, state):
        health = {"lora_explanation": {"done": 3, "baseline": 10,
                                       "stale": 2, "missing": 5}}
        P.feed_queue_phase(state, run_id="run-A", phase="LORA",
                           health=health, now=1000.0)
        got = P.read_progress(state, "run-A", "LORA", now=1000.0)
        assert (got["done"], got["total"], got["remaining"]) == (3, 10, 7)

    def test_non_queue_phase_is_never_fed(self, state):
        assert P.feed_queue_phase(state, run_id="run-A", phase="FINALIZE",
                                  health={}, now=1000.0) is None

    def test_missing_counters_feed_nothing(self, state):
        assert P.feed_queue_phase(state, run_id="run-A", phase="SCANNING",
                                  health={}, now=1000.0) is None

    def test_a_post_sourced_record_is_not_overwritten(self, state):
        """LORA is fed by both the orchestrator and the counters; the
        orchestrator's own marks win, because they know about substeps."""
        P.record_mark(state, {"run_id": "run-A", "phase": "LORA", "done": 9,
                              "total": 10, "interval_s": 10}, now=1000.0)
        P.feed_queue_phase(state, run_id="run-A", phase="LORA",
                           health={"lora_explanation": {"done": 1, "baseline": 10,
                                                        "stale": 0, "missing": 9}},
                           now=1010.0)
        assert P.read_progress(state, "run-A", "LORA", now=1010.0)["done"] == 9


# --- total ETA -------------------------------------------------------------

PRIORS = {"phases": {
    "SCANNING": {"latest_per_1k_tics_s": 10.0, "scales_with_lot": True},
    "FPP_BACKFILL": {"latest_s": 100, "scales_with_lot": False},
    "EXOMINER_BACKFILL": {"latest_s": 200, "scales_with_lot": False},
    "STOP_WORKERS": {"latest_s": 60, "scales_with_lot": False},
    "POSTPROCESS": {"latest_per_1k_tics_s": 2.0, "scales_with_lot": True},
    "STOP_NVIDIARACK": {"latest_s": 30, "scales_with_lot": False},
    "LORA": {"latest_s": 300, "scales_with_lot": False},
    "FREE_GPU": {"latest_s": 20, "scales_with_lot": False},
    "FINALIZE": {"median_s": 900, "scales_with_lot": False},
}}


class TestEtaTotal:
    def test_stage_list_excludes_terminal_states(self):
        stages = P.build_stages("SCANNING")
        assert "DONE" not in stages and "FAILED" not in stages
        assert stages[0] == "SCANNING"

    def test_enqueueing_is_prepended_only_while_it_runs(self):
        assert P.build_stages("ENQUEUEING")[0] == "ENQUEUEING"
        assert "ENQUEUEING" not in P.build_stages("SCANNING")

    def test_prior_scales_with_lot_when_declared(self):
        assert P.prior_estimate_s(PRIORS, "POSTPROCESS", 300000) == 600.0

    def test_prior_falls_back_to_median(self):
        assert P.prior_estimate_s(PRIORS, "FINALIZE", 1000) == 900

    def test_prior_is_none_for_an_unknown_phase(self):
        assert P.prior_estimate_s(PRIORS, "NOPE", 1000) is None

    def test_total_is_current_eta_plus_later_priors(self):
        progress = {"eta_phase_s": 100.0}
        total = P.eta_total_s(state="STOP_WORKERS", progress=progress,
                              priors=PRIORS, lot=300000, now=0.0)
        # 100 (live) + POSTPROCESS 600 + STOP_NVIDIARACK 30 + LORA 300
        # + FREE_GPU 20 + FINALIZE 900
        assert total == pytest.approx(1950.0)

    def test_a_lot_sensitive_prior_without_a_lot_kills_the_total(self):
        """POSTPROCESS only has a per-1k figure: no lot, no estimate, no total."""
        progress = {"eta_phase_s": 100.0}
        assert P.eta_total_s(state="STOP_WORKERS", progress=progress,
                             priors=PRIORS, lot=None, now=0.0) is None

    def test_total_degrades_to_none_when_a_later_prior_is_missing(self):
        priors = {"phases": {"FINALIZE": {"latest_s": 10}}}
        progress = {"eta_phase_s": 5.0}
        assert P.eta_total_s(state="STOP_NVIDIARACK", progress=progress,
                             priors=priors, lot=None, now=0.0) is None

    def test_total_is_none_without_a_current_eta(self):
        assert P.eta_total_s(state="SCANNING", progress=None, priors=PRIORS,
                             lot=None, now=0.0) is None

    def test_load_priors_never_raises(self, tmp_path, monkeypatch):
        monkeypatch.setenv("EXONIT_PHASE_TIMINGS_FILE",
                           str(tmp_path / "absent.json"))
        assert P.load_priors() == {}
    assert P.progress_lookup_run_id("SCANNING", None) is None
