"""Tests for exonit.mast_http — the transport-level timeout + deadline
fix for the 2026-08-03 MAST-hang outage (docs: see exonit/mast_http.py
module docstring and the _run_spoc_check_background docstring in
exonit/server.py).

No test in this file makes a real network call — install_mast_session_
timeout is exercised against a fake astroquery.mast module tree (same
pattern as tests/test_systematic_veto.py's _install_fake_astroquery), and
call_with_deadline is exercised against plain Python callables.
"""
import sys
import threading
import time
import types

import pytest

from exonit.mast_http import call_with_deadline, install_mast_session_timeout


# --------------------------------------------------------------------------- #
# Fake astroquery.mast.Observations tree
# --------------------------------------------------------------------------- #

class _FakeSession:
    """Stand-in for requests.Session: records every call to .request()."""

    def __init__(self):
        self.calls = []

    def request(self, method, url, *args, **kwargs):
        self.calls.append((method, url, kwargs))
        return "fake-response"


class _FakePortalAPI:
    def __init__(self, session):
        self._session = session


def _install_fake_observations(monkeypatch, session=None):
    """Install a fake astroquery.mast.Observations with a real _FakeSession
    at Observations._portal_api_connection._session, mirroring the real
    astroquery 0.4.11 shape probed in exonit/mast_http.py's docstring."""
    session = session or _FakeSession()

    aq = types.ModuleType("astroquery")
    mast = types.ModuleType("astroquery.mast")

    class _FakeObservations:
        _portal_api_connection = _FakePortalAPI(session)

    mast.Observations = _FakeObservations
    aq.mast = mast
    monkeypatch.setitem(sys.modules, "astroquery", aq)
    monkeypatch.setitem(sys.modules, "astroquery.mast", mast)
    return session


# --------------------------------------------------------------------------- #
# 1-4. install_mast_session_timeout
# --------------------------------------------------------------------------- #

class TestInstallMastSessionTimeout:

    def test_injects_default_timeout_when_caller_omits_it(self, monkeypatch):
        session = _install_fake_observations(monkeypatch)
        monkeypatch.delenv("EXONIT_MAST_HTTP_TIMEOUT", raising=False)

        install_mast_session_timeout()
        session.request("GET", "http://example.invalid")

        assert session.calls[-1][2]["timeout"] == 60.0

    def test_respects_explicit_caller_timeout(self, monkeypatch):
        session = _install_fake_observations(monkeypatch)
        monkeypatch.delenv("EXONIT_MAST_HTTP_TIMEOUT", raising=False)

        install_mast_session_timeout()
        session.request("GET", "http://example.invalid", timeout=12.5)

        assert session.calls[-1][2]["timeout"] == 12.5

    def test_idempotent_double_install_does_not_double_wrap(self, monkeypatch):
        session = _install_fake_observations(monkeypatch)
        monkeypatch.delenv("EXONIT_MAST_HTTP_TIMEOUT", raising=False)

        install_mast_session_timeout()
        wrapped_once = session.request
        install_mast_session_timeout()
        wrapped_twice = session.request

        assert wrapped_once is wrapped_twice, (
            "a second install() must not re-wrap an already-wrapped session"
        )
        # Sanity: a single wrap still injects the timeout correctly.
        session.request("GET", "http://example.invalid")
        assert session.calls[-1][2]["timeout"] == 60.0

    def test_env_var_zero_disables_the_patch(self, monkeypatch):
        session = _install_fake_observations(monkeypatch)
        monkeypatch.setenv("EXONIT_MAST_HTTP_TIMEOUT", "0")

        install_mast_session_timeout()
        session.request("GET", "http://example.invalid")

        assert "timeout" not in session.calls[-1][2]
        assert not getattr(session, "_exonit_timeout_installed", False)

    def test_env_var_negative_disables_the_patch(self, monkeypatch):
        session = _install_fake_observations(monkeypatch)
        monkeypatch.setenv("EXONIT_MAST_HTTP_TIMEOUT", "-5")

        install_mast_session_timeout()
        session.request("GET", "http://example.invalid")

        assert "timeout" not in session.calls[-1][2]

    def test_env_var_garbage_disables_the_patch_and_does_not_raise(self, monkeypatch, caplog):
        session = _install_fake_observations(monkeypatch)
        monkeypatch.setenv("EXONIT_MAST_HTTP_TIMEOUT", "not-a-number")

        with caplog.at_level("WARNING", logger="exonit.mast_http"):
            install_mast_session_timeout()  # must not raise
        session.request("GET", "http://example.invalid")

        assert "timeout" not in session.calls[-1][2]
        assert any("not a valid float" in r.message for r in caplog.records)

    def test_explicit_timeout_kwarg_overrides_env_var(self, monkeypatch):
        session = _install_fake_observations(monkeypatch)
        monkeypatch.setenv("EXONIT_MAST_HTTP_TIMEOUT", "999")

        install_mast_session_timeout(timeout=5.0)
        session.request("GET", "http://example.invalid")

        assert session.calls[-1][2]["timeout"] == 5.0

    def test_missing_astroquery_is_fail_soft(self, monkeypatch, caplog):
        """If astroquery isn't importable, log a warning and return — never
        raise (fail-soft contract, mirrors _run_spoc_check_background)."""
        monkeypatch.setitem(sys.modules, "astroquery", None)
        monkeypatch.setitem(sys.modules, "astroquery.mast", None)

        with caplog.at_level("WARNING", logger="exonit.mast_http"):
            install_mast_session_timeout()  # must not raise
        assert any("not available" in r.message for r in caplog.records)


# --------------------------------------------------------------------------- #
# 5-6. call_with_deadline
# --------------------------------------------------------------------------- #

class TestCallWithDeadline:

    def test_returns_value_on_success(self):
        result = call_with_deadline(lambda a, b: a + b, 2, 3, deadline_s=5)
        assert result == 5

    def test_propagates_exception_from_fn(self):
        def _boom():
            raise ValueError("kaboom")

        with pytest.raises(ValueError, match="kaboom"):
            call_with_deadline(_boom, deadline_s=5)

    def test_stuck_call_times_out_and_thread_is_daemon(self):
        never_set = threading.Event()
        captured_thread = []

        def _blocker():
            captured_thread.append(threading.current_thread())
            never_set.wait()  # blocks forever (never set by this test)
            return "should never get here"

        t0 = time.monotonic()
        with pytest.raises(TimeoutError, match="deadline-test"):
            call_with_deadline(_blocker, deadline_s=0.2, name="deadline-test")
        elapsed = time.monotonic() - t0

        assert elapsed < 2.0, "must return promptly at the deadline, not block"
        # Give the runner thread a moment to actually start and register itself.
        deadline = time.monotonic() + 2.0
        while not captured_thread and time.monotonic() < deadline:
            time.sleep(0.01)
        assert captured_thread, "blocker thread never started"
        assert captured_thread[0].daemon is True, (
            "the abandoned thread must be daemon so it never blocks process exit"
        )

    def test_deadline_disabled_via_env_var_zero_runs_inline(self, monkeypatch):
        monkeypatch.setenv("EXONIT_SPOC_CHECK_DEADLINE", "0")
        calling_thread = threading.current_thread()
        seen = []

        def _fn():
            seen.append(threading.current_thread())
            return "ok"

        result = call_with_deadline(_fn)
        assert result == "ok"
        assert seen[0] is calling_thread, (
            "deadline<=0 must run fn directly on the caller's thread, no "
            "thread spawned"
        )

    def test_default_deadline_env_var_is_honoured(self, monkeypatch):
        monkeypatch.setenv("EXONIT_SPOC_CHECK_DEADLINE", "0.2")
        never_set = threading.Event()

        with pytest.raises(TimeoutError):
            call_with_deadline(lambda: never_set.wait(), name="env-deadline-test")


# --------------------------------------------------------------------------- #
# 7. Integration: _run_spoc_check_background stays fail-soft when the MAST
#    call hangs past the deadline (no thread, no DB write, no exception).
# --------------------------------------------------------------------------- #

class TestSpocCheckBackgroundDeadlineFailSoft:

    def test_hanging_is_ffi_only_at_mast_times_out_fail_soft(self, tmp_path, monkeypatch, caplog):
        from unittest.mock import patch

        from exonit.db import ExoNitDB
        from exonit.server import _run_spoc_check_background

        db = ExoNitDB(str(tmp_path / "test.db"))
        tic_id = "900"
        db.add_star(
            tic_id, ra=10.0, dec=-20.0, tmag=11.0,
            n_sectors=1, n_datapoints=1000, time_span_days=27.0,
            status="completed",
        )
        cid = db.add_candidate(
            tic_id, candidate_num=1, period_days=3.0, t0=0.0,
            duration_days=0.1, depth_ppm=1000, snr=12.0, cnn_score=0.9,
        )

        monkeypatch.setenv("EXONIT_SPOC_CHECK_DEADLINE", "0.2")

        never_set = threading.Event()

        def _hanging_is_ffi_only_at_mast(tic, cache_dir):
            never_set.wait()  # never returns within the test lifetime
            return True  # pragma: no cover

        with patch(
            "exonit.ffi_download.is_ffi_only_at_mast",
            side_effect=_hanging_is_ffi_only_at_mast,
        ), caplog.at_level("WARNING", logger="exonit.server"):
            _run_spoc_check_background(db, tic_id, db.get_candidates(tic_id))

        # Fail-soft: candidate untouched, has_spoc untouched, only a warning
        # logged — never an exception escaping the background function.
        cands = db.get_candidates(tic_id)
        assert cands[0]["exominer_score"] is None
        star = db._query_one("SELECT has_spoc FROM stars WHERE tic_id=?", (tic_id,))
        assert star["has_spoc"] is None
        assert any("SPOC-gate failed" in r.message for r in caplog.records)
        db.close()
