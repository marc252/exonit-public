"""Tests for the quiet-poll access-log filter.

The filter (_QuietPollFilter) is installed by _install_quiet_poll_filter()
inside create_app(), so it runs in every uvicorn worker process.

Test matrix:
  (a) 2xx on a polling endpoint  -> discarded (filter returns False)
  (b) 4xx/5xx on a polling endpoint -> kept (filter returns True)
  (c) 2xx on a non-polling endpoint -> kept (filter returns True)
  (d) EXONIT_QUIET_POLL_LOGS=0    -> filter not installed at all
  (e) Unexpected record.args format -> kept, no exception raised
"""

import logging
import os

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_access_record(method: str, path: str, status: int) -> logging.LogRecord:
    """Build a LogRecord that mimics uvicorn.access entries.

    uvicorn.access emits: record.args = (client_addr, method, full_path,
    http_version, status_code).  status_code is kept as int here (uvicorn
    actually passes it as an int too).
    """
    record = logging.LogRecord(
        name="uvicorn.access",
        level=logging.INFO,
        pathname="",
        lineno=0,
        msg='%s - "%s %s HTTP/%s" %d',
        args=("127.0.0.1:12345", method, path, "1.1", status),
        exc_info=None,
    )
    return record


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _clean_access_logger():
    """Remove any _QuietPollFilter instances before and after each test."""
    from exonit.server import _QuietPollFilter
    access_logger = logging.getLogger("uvicorn.access")

    def _strip():
        access_logger.filters = [
            f for f in access_logger.filters
            if not isinstance(f, _QuietPollFilter)
        ]

    _strip()
    yield
    _strip()


@pytest.fixture(autouse=True)
def _reset_quiet_env(monkeypatch):
    """Ensure EXONIT_QUIET_POLL_LOGS is set to "1" by default for most tests.

    Individual tests that need a different value override via monkeypatch.
    """
    monkeypatch.setenv("EXONIT_QUIET_POLL_LOGS", "1")


# ---------------------------------------------------------------------------
# Unit tests on the filter class directly
# ---------------------------------------------------------------------------

class TestQuietPollFilter:
    """Direct unit tests on _QuietPollFilter without invoking create_app."""

    @pytest.fixture(autouse=True)
    def _filter(self):
        from exonit.server import _QuietPollFilter
        self.f = _QuietPollFilter()

    # (a) 2xx on polling endpoints -> discarded
    @pytest.mark.parametrize("path", [
        "/api/worker/task",
        "/api/worker/fpp_task",
        "/api/worker/exominer_task",
        "/api/worker/heartbeat",
    ])
    @pytest.mark.parametrize("status", [200, 204])
    def test_discards_2xx_polling(self, path, status):
        record = _make_access_record("GET", path, status)
        assert self.f.filter(record) is False, (
            f"Expected False for {path} status={status}"
        )

    # (a) POST /api/worker/heartbeat 2xx -> also discarded
    def test_discards_post_heartbeat_2xx(self):
        record = _make_access_record("POST", "/api/worker/heartbeat", 200)
        assert self.f.filter(record) is False

    # (a) query-string stripped — path still matches
    def test_discards_2xx_polling_with_query_string(self):
        record = _make_access_record("GET", "/api/worker/task?worker_id=abc", 200)
        assert self.f.filter(record) is False

    # (b) 4xx/5xx on polling endpoint -> kept
    @pytest.mark.parametrize("status", [400, 401, 403, 404, 500, 503])
    def test_keeps_error_on_polling_endpoint(self, status):
        record = _make_access_record("GET", "/api/worker/task", status)
        assert self.f.filter(record) is True, (
            f"Expected True (keep) for polling endpoint status={status}"
        )

    # (c) 2xx on non-polling endpoints -> kept
    @pytest.mark.parametrize("path", [
        "/api/worker/result",
        "/api/worker/fpp_result",
        "/api/worker/register",
        "/api/stats",
        "/api/health/work",
        "/api/version",
        "/api/worker/list",
        "/api/worker/update",
    ])
    def test_keeps_2xx_non_polling(self, path):
        record = _make_access_record("POST", path, 200)
        assert self.f.filter(record) is True, (
            f"Expected True (keep) for non-polling path {path}"
        )

    # (e) Unexpected record.args format -> keeps record, no exception
    def test_keeps_record_on_unexpected_args_format(self):
        record = logging.LogRecord(
            name="uvicorn.access",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="something unexpected",
            args=("only_one_item",),  # wrong arity
            exc_info=None,
        )
        # Must not raise and must return True (keep)
        result = self.f.filter(record)
        assert result is True

    def test_keeps_record_on_none_args(self):
        record = logging.LogRecord(
            name="uvicorn.access",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="no args",
            args=None,
            exc_info=None,
        )
        result = self.f.filter(record)
        assert result is True

    def test_keeps_record_on_non_integer_status(self):
        """status_code that can't be converted to int -> keep, no crash."""
        record = logging.LogRecord(
            name="uvicorn.access",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="bad status",
            args=("127.0.0.1", "GET", "/api/worker/task", "1.1", "??"),
            exc_info=None,
        )
        result = self.f.filter(record)
        assert result is True


# ---------------------------------------------------------------------------
# Integration tests via _install_quiet_poll_filter
# ---------------------------------------------------------------------------

class TestInstallQuietPollFilter:
    """Test that _install_quiet_poll_filter() attaches to uvicorn.access."""

    def test_installs_filter_when_env_is_1(self):
        from exonit.server import _install_quiet_poll_filter, _QuietPollFilter
        _install_quiet_poll_filter()
        access_logger = logging.getLogger("uvicorn.access")
        assert any(isinstance(f, _QuietPollFilter) for f in access_logger.filters)

    def test_idempotent_double_install(self):
        from exonit.server import _install_quiet_poll_filter, _QuietPollFilter
        _install_quiet_poll_filter()
        _install_quiet_poll_filter()
        access_logger = logging.getLogger("uvicorn.access")
        count = sum(1 for f in access_logger.filters if isinstance(f, _QuietPollFilter))
        assert count == 1, "Filter should be installed exactly once"

    # (d) EXONIT_QUIET_POLL_LOGS=0 -> filter NOT installed
    def test_no_filter_when_env_is_0(self, monkeypatch):
        monkeypatch.setenv("EXONIT_QUIET_POLL_LOGS", "0")
        from exonit.server import _install_quiet_poll_filter, _QuietPollFilter
        _install_quiet_poll_filter()
        access_logger = logging.getLogger("uvicorn.access")
        assert not any(
            isinstance(f, _QuietPollFilter) for f in access_logger.filters
        )


# ---------------------------------------------------------------------------
# End-to-end: create_app installs the filter
# ---------------------------------------------------------------------------

class TestCreateAppInstallsFilter:
    """Verify that create_app() calls _install_quiet_poll_filter()."""

    def test_filter_present_after_create_app(self, tmp_path, monkeypatch):
        monkeypatch.setenv("EXONIT_DB_PATH", str(tmp_path / "test.db"))
        monkeypatch.setenv("EXONIT_REPORTS_DIR", str(tmp_path / "reports"))
        monkeypatch.setenv("EXONIT_QUIET_POLL_LOGS", "1")
        (tmp_path / "reports").mkdir()

        from exonit.server import create_app, _QuietPollFilter
        create_app()

        access_logger = logging.getLogger("uvicorn.access")
        assert any(isinstance(f, _QuietPollFilter) for f in access_logger.filters), (
            "create_app() must install _QuietPollFilter on uvicorn.access"
        )

    def test_filter_absent_when_disabled(self, tmp_path, monkeypatch):
        monkeypatch.setenv("EXONIT_DB_PATH", str(tmp_path / "test.db"))
        monkeypatch.setenv("EXONIT_REPORTS_DIR", str(tmp_path / "reports"))
        monkeypatch.setenv("EXONIT_QUIET_POLL_LOGS", "0")
        (tmp_path / "reports").mkdir()

        from exonit.server import create_app, _QuietPollFilter
        create_app()

        access_logger = logging.getLogger("uvicorn.access")
        assert not any(isinstance(f, _QuietPollFilter) for f in access_logger.filters), (
            "create_app() must NOT install filter when EXONIT_QUIET_POLL_LOGS=0"
        )
