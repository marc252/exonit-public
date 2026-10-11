"""Tests for the A3 multi-process server support.

Coverage:
1. Hard-assert in get_shared_state when EXONIT_SERVER_WORKERS > 1:
   - force_memory=True → RuntimeError (in-memory forbidden in multi-process mode)
   - EXONIT_SHARED_STATE_BACKEND=memory → RuntimeError
   - Bad Redis URL → RuntimeError (no silent fallback)
   - Redis available (db=15) → RedisBackend (happy path; skipped if Redis down)
   - EXONIT_SERVER_WORKERS=1 (or absent) + Redis down → MemoryBackend (pre-A3
     single-process degradation preserved)

2. server.main() launcher (uvicorn.run monkeypatched):
   - workers=1 → app instance passed directly; no EXONIT_SERVER_WORKERS set
   - workers=3 (Redis available) → import-string + factory=True + workers=3;
     EXONIT_SERVER_WORKERS env set to "3"
   - workers=3 (Redis down) → RuntimeError before uvicorn.run is called

3. run_nightly.py CLI parser:
   - `serve --workers 4` → args.workers == 4
   - `serve` (no flag) → args.workers == 1  (default)

No real processes are started; uvicorn.run is always monkeypatched.
"""

from __future__ import annotations

import os
import sys
import types
from unittest.mock import MagicMock, patch, call

import pytest

from exonit.shared_state import (
    MemoryBackend,
    RedisBackend,
    reset_shared_state,
    get_shared_state,
)

# ---------------------------------------------------------------------------
# Module-level isolation fixture
#
# The conftest autouse fixture skips this file (because it manages its own
# state).  We define a per-test fixture here that:
#   1. resets the shared-state singleton before and after every test, and
#   2. restores EXONIT_SERVER_WORKERS via monkeypatch so that server.main()
#      calls which write os.environ directly do not bleed into subsequent tests.
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_state(monkeypatch):
    """Reset shared-state singleton and clean env vars around every test."""
    # Save the current value so we can restore it even if the test sets it
    # via os.environ directly (not through monkeypatch).
    _original_workers = os.environ.get("EXONIT_SERVER_WORKERS")

    monkeypatch.delenv("EXONIT_SERVER_WORKERS", raising=False)
    reset_shared_state()

    yield

    # Unconditional cleanup: restore what was there before (usually absent).
    if _original_workers is None:
        os.environ.pop("EXONIT_SERVER_WORKERS", None)
    else:
        os.environ["EXONIT_SERVER_WORKERS"] = _original_workers

    reset_shared_state()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_REDIS_TEST_URL = "redis://localhost:6379/15"
_BAD_REDIS_URL = "redis://127.0.0.1:1/0"  # nothing listening here


def _redis_available() -> bool:
    """Return True if the Redis test DB (db=15) is reachable."""
    try:
        b = RedisBackend(_REDIS_TEST_URL)
        return b.ping()
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Section 1 — Hard-assert in get_shared_state
# ---------------------------------------------------------------------------


class TestGetSharedStateMultiProcess:
    """A3: Redis is mandatory when EXONIT_SERVER_WORKERS > 1."""

    def setup_method(self):
        reset_shared_state()

    def teardown_method(self):
        reset_shared_state()

    # -- in-memory forbidden -------------------------------------------------

    def test_force_memory_raises_when_multi_process(self, monkeypatch):
        monkeypatch.setenv("EXONIT_SERVER_WORKERS", "2")
        monkeypatch.delenv("EXONIT_SHARED_STATE_BACKEND", raising=False)
        with pytest.raises(RuntimeError, match="multi-process server"):
            get_shared_state(force_memory=True)

    def test_env_memory_raises_when_multi_process(self, monkeypatch):
        monkeypatch.setenv("EXONIT_SERVER_WORKERS", "2")
        monkeypatch.setenv("EXONIT_SHARED_STATE_BACKEND", "memory")
        with pytest.raises(RuntimeError, match="multi-process server"):
            get_shared_state()

    def test_runtime_error_mentions_worker_count(self, monkeypatch):
        monkeypatch.setenv("EXONIT_SERVER_WORKERS", "4")
        monkeypatch.delenv("EXONIT_SHARED_STATE_BACKEND", raising=False)
        with pytest.raises(RuntimeError, match="EXONIT_SERVER_WORKERS=4"):
            get_shared_state(force_memory=True)

    # -- bad Redis URL → RuntimeError, no MemoryBackend fallback -------------

    def test_bad_redis_url_raises_when_multi_process(self, monkeypatch):
        monkeypatch.setenv("EXONIT_SERVER_WORKERS", "2")
        monkeypatch.delenv("EXONIT_SHARED_STATE_BACKEND", raising=False)
        monkeypatch.delenv("EXONIT_REDIS_URL", raising=False)
        with pytest.raises(RuntimeError, match="multi-process server"):
            get_shared_state(url=_BAD_REDIS_URL)

    def test_bad_redis_url_error_mentions_url(self, monkeypatch):
        monkeypatch.setenv("EXONIT_SERVER_WORKERS", "2")
        monkeypatch.delenv("EXONIT_SHARED_STATE_BACKEND", raising=False)
        monkeypatch.delenv("EXONIT_REDIS_URL", raising=False)
        with pytest.raises(RuntimeError, match=r"127\.0\.0\.1:1"):
            get_shared_state(url=_BAD_REDIS_URL)

    # -- Redis available → RedisBackend --------------------------------------

    def test_redis_ok_returns_redis_backend_when_multi_process(self, monkeypatch):
        if not _redis_available():
            pytest.skip("Redis not reachable at " + _REDIS_TEST_URL)
        monkeypatch.setenv("EXONIT_SERVER_WORKERS", "2")
        monkeypatch.setenv("EXONIT_REDIS_URL", _REDIS_TEST_URL)
        monkeypatch.delenv("EXONIT_SHARED_STATE_BACKEND", raising=False)
        backend = get_shared_state()
        assert isinstance(backend, RedisBackend)

    def test_redis_backend_pings_ok(self, monkeypatch):
        if not _redis_available():
            pytest.skip("Redis not reachable at " + _REDIS_TEST_URL)
        monkeypatch.setenv("EXONIT_SERVER_WORKERS", "3")
        monkeypatch.setenv("EXONIT_REDIS_URL", _REDIS_TEST_URL)
        monkeypatch.delenv("EXONIT_SHARED_STATE_BACKEND", raising=False)
        backend = get_shared_state()
        assert backend.ping() is True

    # -- Single-process: pre-A3 degradation preserved -----------------------

    def test_single_process_redis_down_falls_back_to_memory(self, monkeypatch):
        """workers=1 (or absent): Redis failure → MemoryBackend (not an error)."""
        monkeypatch.delenv("EXONIT_SERVER_WORKERS", raising=False)
        monkeypatch.delenv("EXONIT_SHARED_STATE_BACKEND", raising=False)
        monkeypatch.delenv("EXONIT_REDIS_URL", raising=False)
        backend = get_shared_state(url=_BAD_REDIS_URL)
        assert isinstance(backend, MemoryBackend)

    def test_single_process_explicit_1_redis_down_falls_back(self, monkeypatch):
        monkeypatch.setenv("EXONIT_SERVER_WORKERS", "1")
        monkeypatch.delenv("EXONIT_SHARED_STATE_BACKEND", raising=False)
        monkeypatch.delenv("EXONIT_REDIS_URL", raising=False)
        backend = get_shared_state(url=_BAD_REDIS_URL)
        assert isinstance(backend, MemoryBackend)

    def test_single_process_invalid_workers_env_falls_back(self, monkeypatch):
        """Non-integer EXONIT_SERVER_WORKERS treated as 1 → no RuntimeError."""
        monkeypatch.setenv("EXONIT_SERVER_WORKERS", "not_a_number")
        monkeypatch.delenv("EXONIT_SHARED_STATE_BACKEND", raising=False)
        monkeypatch.delenv("EXONIT_REDIS_URL", raising=False)
        backend = get_shared_state(url=_BAD_REDIS_URL)
        assert isinstance(backend, MemoryBackend)


# ---------------------------------------------------------------------------
# Section 2 — server.main() launcher
# ---------------------------------------------------------------------------


class TestServerMainLauncher:
    """server.main() dispatches correctly between single- and multi-process paths."""

    def _make_uvicorn_mock(self):
        """Return a MagicMock suitable for patching uvicorn.run."""
        return MagicMock(return_value=None)

    def test_workers_1_passes_app_instance(self, monkeypatch):
        """Single-process: uvicorn.run receives an app object, not a string."""
        uvicorn_run = self._make_uvicorn_mock()

        with patch("uvicorn.run", uvicorn_run):
            from exonit.server import main as serve_main
            serve_main(workers=1)

        assert uvicorn_run.called
        positional_app = uvicorn_run.call_args[0][0] if uvicorn_run.call_args[0] else None
        # The first positional argument must NOT be a plain string.
        assert not isinstance(positional_app, str), (
            "workers=1 must pass an app instance, not an import-string"
        )

    def test_workers_1_does_not_set_workers_env(self, monkeypatch):
        """Single-process: EXONIT_SERVER_WORKERS must not be set in the environment."""
        monkeypatch.delenv("EXONIT_SERVER_WORKERS", raising=False)
        uvicorn_run = self._make_uvicorn_mock()

        with patch("uvicorn.run", uvicorn_run):
            from exonit.server import main as serve_main
            serve_main(workers=1)

        assert os.environ.get("EXONIT_SERVER_WORKERS") is None, (
            "workers=1 must not set EXONIT_SERVER_WORKERS"
        )

    def test_workers_3_sets_env_before_uvicorn(self, monkeypatch):
        """Multi-process: EXONIT_SERVER_WORKERS must be set to '3' before uvicorn.run."""
        if not _redis_available():
            pytest.skip("Redis not reachable — skipping multi-process launcher test")

        monkeypatch.setenv("EXONIT_REDIS_URL", _REDIS_TEST_URL)
        monkeypatch.delenv("EXONIT_SHARED_STATE_BACKEND", raising=False)
        monkeypatch.delenv("EXONIT_SERVER_WORKERS", raising=False)

        captured_env: dict[str, str | None] = {}
        uvicorn_run = self._make_uvicorn_mock()

        def capture_env(*args, **kwargs):
            captured_env["EXONIT_SERVER_WORKERS"] = os.environ.get("EXONIT_SERVER_WORKERS")

        uvicorn_run.side_effect = capture_env

        with patch("uvicorn.run", uvicorn_run):
            from exonit.server import main as serve_main
            serve_main(workers=3)

        assert captured_env.get("EXONIT_SERVER_WORKERS") == "3"

    def test_workers_3_passes_import_string_and_factory(self, monkeypatch):
        """Multi-process: uvicorn.run gets import-string, factory=True, workers=3."""
        if not _redis_available():
            pytest.skip("Redis not reachable — skipping multi-process launcher test")

        monkeypatch.setenv("EXONIT_REDIS_URL", _REDIS_TEST_URL)
        monkeypatch.delenv("EXONIT_SHARED_STATE_BACKEND", raising=False)
        monkeypatch.delenv("EXONIT_SERVER_WORKERS", raising=False)

        uvicorn_run = self._make_uvicorn_mock()

        with patch("uvicorn.run", uvicorn_run):
            from exonit.server import main as serve_main
            serve_main(workers=3)

        assert uvicorn_run.called
        kwargs = uvicorn_run.call_args.kwargs
        positional = uvicorn_run.call_args[0]
        first_arg = positional[0] if positional else kwargs.get("app")

        assert first_arg == "exonit.server:create_app", (
            f"Expected import-string 'exonit.server:create_app', got {first_arg!r}"
        )
        assert kwargs.get("factory") is True, "factory=True must be passed"
        assert kwargs.get("workers") == 3, "workers=3 must be forwarded to uvicorn"

    def test_workers_3_redis_down_raises_before_uvicorn(self, monkeypatch):
        """Multi-process: RuntimeError from preflight prevents uvicorn.run call.

        We point EXONIT_REDIS_URL at a port with nothing listening so the
        preflight ping fails regardless of whether local Redis is running.
        """
        monkeypatch.delenv("EXONIT_SHARED_STATE_BACKEND", raising=False)
        monkeypatch.setenv("EXONIT_REDIS_URL", _BAD_REDIS_URL)

        uvicorn_run = self._make_uvicorn_mock()

        with patch("uvicorn.run", uvicorn_run):
            from exonit.server import main as serve_main
            with pytest.raises(RuntimeError, match="multi-process server"):
                serve_main(workers=3)

        assert not uvicorn_run.called, "uvicorn.run must NOT be called if Redis is absent"


# ---------------------------------------------------------------------------
# Section 2b — _uvicorn_max_requests() parsing + limit_max_requests wiring
#
# Added 2026-07-31 (worker recycling, STATUS §0u OOM incident): uvicorn
# worker processes are recycled periodically via limit_max_requests on the
# workers>1 path only. See exonit/server.py main() for the full rationale.
# ---------------------------------------------------------------------------


class TestUvicornMaxRequestsParsing:
    """_uvicorn_max_requests(): env parsing semantics.

    unset -> 20000 (default); "0" -> None (explicit opt-out, silent);
    "N" -> N; invalid (non-int or negative) -> 20000 (fail-safe: an OOM
    safeguard must not be silently disabled by a config typo) + logger.warning.
    """

    def test_unset_defaults_to_20000(self, monkeypatch):
        monkeypatch.delenv("EXONIT_UVICORN_MAX_REQUESTS", raising=False)
        from exonit.server import _uvicorn_max_requests
        assert _uvicorn_max_requests() == 20000

    def test_zero_disables_silently(self, monkeypatch, caplog):
        monkeypatch.setenv("EXONIT_UVICORN_MAX_REQUESTS", "0")
        from exonit.server import _uvicorn_max_requests
        with caplog.at_level("WARNING"):
            result = _uvicorn_max_requests()
        assert result is None
        assert not caplog.records, "'0' is a deliberate opt-out, must not warn"

    def test_custom_value_parsed(self, monkeypatch):
        monkeypatch.setenv("EXONIT_UVICORN_MAX_REQUESTS", "5000")
        from exonit.server import _uvicorn_max_requests
        assert _uvicorn_max_requests() == 5000

    def test_non_integer_falls_back_to_default_and_warns(self, monkeypatch, caplog):
        monkeypatch.setenv("EXONIT_UVICORN_MAX_REQUESTS", "abc")
        from exonit.server import _uvicorn_max_requests
        with caplog.at_level("WARNING"):
            result = _uvicorn_max_requests()
        assert result == 20000, "invalid value must fail SAFE to the default, not disable"
        assert any("EXONIT_UVICORN_MAX_REQUESTS" in r.message for r in caplog.records)

    def test_negative_falls_back_to_default_and_warns(self, monkeypatch, caplog):
        monkeypatch.setenv("EXONIT_UVICORN_MAX_REQUESTS", "-3")
        from exonit.server import _uvicorn_max_requests
        with caplog.at_level("WARNING"):
            result = _uvicorn_max_requests()
        assert result == 20000, "invalid value must fail SAFE to the default, not disable"
        assert any("EXONIT_UVICORN_MAX_REQUESTS" in r.message for r in caplog.records)


class TestServerMainUvicornRecycle:
    """serve_main() wires limit_max_requests on workers>1 ONLY."""

    def _make_uvicorn_mock(self):
        return MagicMock(return_value=None)

    def test_workers_4_passes_default_limit_max_requests(self, monkeypatch):
        """Multi-process, no env override: limit_max_requests=20000 (default)."""
        if not _redis_available():
            pytest.skip("Redis not reachable — skipping multi-process launcher test")

        monkeypatch.setenv("EXONIT_REDIS_URL", _REDIS_TEST_URL)
        monkeypatch.delenv("EXONIT_SHARED_STATE_BACKEND", raising=False)
        monkeypatch.delenv("EXONIT_SERVER_WORKERS", raising=False)
        monkeypatch.delenv("EXONIT_UVICORN_MAX_REQUESTS", raising=False)

        uvicorn_run = self._make_uvicorn_mock()

        with patch("uvicorn.run", uvicorn_run):
            from exonit.server import main as serve_main
            serve_main(workers=4)

        assert uvicorn_run.called
        kwargs = uvicorn_run.call_args.kwargs
        assert kwargs.get("limit_max_requests") == 20000

    def test_workers_4_custom_env_value_forwarded(self, monkeypatch):
        """Multi-process, EXONIT_UVICORN_MAX_REQUESTS=5000 -> forwarded as-is."""
        if not _redis_available():
            pytest.skip("Redis not reachable — skipping multi-process launcher test")

        monkeypatch.setenv("EXONIT_REDIS_URL", _REDIS_TEST_URL)
        monkeypatch.delenv("EXONIT_SHARED_STATE_BACKEND", raising=False)
        monkeypatch.delenv("EXONIT_SERVER_WORKERS", raising=False)
        monkeypatch.setenv("EXONIT_UVICORN_MAX_REQUESTS", "5000")

        uvicorn_run = self._make_uvicorn_mock()

        with patch("uvicorn.run", uvicorn_run):
            from exonit.server import main as serve_main
            serve_main(workers=4)

        assert uvicorn_run.call_args.kwargs.get("limit_max_requests") == 5000

    def test_workers_4_zero_disables_recycling(self, monkeypatch):
        """Multi-process, EXONIT_UVICORN_MAX_REQUESTS=0 -> kwarg absent entirely."""
        if not _redis_available():
            pytest.skip("Redis not reachable — skipping multi-process launcher test")

        monkeypatch.setenv("EXONIT_REDIS_URL", _REDIS_TEST_URL)
        monkeypatch.delenv("EXONIT_SHARED_STATE_BACKEND", raising=False)
        monkeypatch.delenv("EXONIT_SERVER_WORKERS", raising=False)
        monkeypatch.setenv("EXONIT_UVICORN_MAX_REQUESTS", "0")

        uvicorn_run = self._make_uvicorn_mock()

        with patch("uvicorn.run", uvicorn_run):
            from exonit.server import main as serve_main
            serve_main(workers=4)

        assert "limit_max_requests" not in uvicorn_run.call_args.kwargs

    def test_workers_1_never_receives_limit_max_requests(self, monkeypatch):
        """Single-process: limit_max_requests must NEVER be passed, no supervisor
        exists to respawn a process that would exit at the limit."""
        monkeypatch.delenv("EXONIT_UVICORN_MAX_REQUESTS", raising=False)
        uvicorn_run = self._make_uvicorn_mock()

        with patch("uvicorn.run", uvicorn_run):
            from exonit.server import main as serve_main
            serve_main(workers=1)

        assert uvicorn_run.called
        assert "limit_max_requests" not in uvicorn_run.call_args.kwargs


# ---------------------------------------------------------------------------
# Section 3 — run_nightly.py CLI parser
# ---------------------------------------------------------------------------


class TestRunNightlyServeParser:
    """serve sub-command accepts --workers and defaults to 1.

    All tests use the *real* build_parser() from run_nightly.py so that
    removing or renaming the --workers argument there would break these tests.
    """

    def _get_parser(self):
        """Return the real ArgumentParser built by run_nightly.build_parser()."""
        from run_nightly import build_parser  # noqa: conftest adds repo root to sys.path
        return build_parser()

    def test_workers_flag_parsed_correctly(self):
        parser = self._get_parser()
        args = parser.parse_args(["serve", "--workers", "4"])
        assert args.workers == 4

    def test_workers_default_is_1(self):
        parser = self._get_parser()
        args = parser.parse_args(["serve"])
        assert args.workers == 1

    def test_run_nightly_serve_parser_has_workers_arg(self):
        """Explicit check: the real parser exposes --workers on the serve sub-command."""
        from run_nightly import build_parser
        args = build_parser().parse_args(["serve", "--workers", "4"])
        assert args.workers == 4

    def test_run_nightly_serve_default_workers_1(self):
        from run_nightly import build_parser
        args = build_parser().parse_args(["serve"])
        assert args.workers == 1
