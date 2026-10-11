"""Tests for the already_online detection fix in /api/workers/start.

Covers:
- _expected_instances: reads ops/workers.json; handles multi-instance workers,
  single-instance workers, absent workers, and read errors.
- _count_online_instances: matching logic for base name and suffixed names;
  heartbeat threshold; mixed alive/dead workers.
- Integration via HTTP client: multi-instance worker fully online → already_online;
  partially online → start called; single-instance (N=1) retrocompat.

No real processes, no network, no server restarts, no DB mutations.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from exonit.server import _expected_instances, _count_online_instances


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_OFFLINE_THRESHOLD = 600.0  # matches WORKER_OFFLINE_SECONDS


def _alive_worker(name: str, ago_seconds: float = 5.0) -> dict:
    """Build a workers_state entry that is alive (heartbeat < threshold ago)."""
    return {"name": name, "last_heartbeat": time.time() - ago_seconds}


def _dead_worker(name: str) -> dict:
    """Build a workers_state entry that is offline (heartbeat well past threshold)."""
    return {"name": name, "last_heartbeat": time.time() - 9999.0}


# ---------------------------------------------------------------------------
# 1. _expected_instances
# ---------------------------------------------------------------------------

class TestExpectedInstances:

    def test_live_manifest_parses_all_workers(self):
        """The live ops/workers.json parses without error for every known worker.

        Does NOT hardcode specific counts — avoids breakage when the operator adjusts
        fleet sizes.  Invariants that ARE stable:
        - absent workers always return 1 (the safe default)
        - every worker present in the manifest returns an integer >= 0
        """
        from exonit.server import _WORKERS_MANIFEST

        manifest = json.loads(_WORKERS_MANIFEST.read_text())
        known_names = [w["name"] for w in manifest.get("workers", [])]
        assert len(known_names) > 0, "manifest must list at least one worker"

        for name in known_names:
            result = _expected_instances(name)
            assert isinstance(result, int), f"{name}: expected int, got {type(result)}"
            assert result >= 0, f"{name}: expected >= 0, got {result}"

        # A worker absent from the manifest must return exactly 1 (safe default)
        assert _expected_instances("worker-does-not-exist") == 1

    def test_hp_returns_1(self):
        """worker-HP has no instances field → default 1."""
        assert _expected_instances("worker-HP") == 1

    def test_absent_worker_returns_1(self):
        """Name not in manifest → default 1."""
        assert _expected_instances("worker-does-not-exist") == 1

    def test_returns_1_when_manifest_missing(self, tmp_path):
        """Non-existent manifest path → default 1, no exception."""
        missing = tmp_path / "no_such_file.json"
        result = _expected_instances("worker-<host>", manifest_path=missing)
        assert result == 1

    def test_returns_1_when_manifest_invalid_json(self, tmp_path):
        """Malformed JSON → default 1, no exception."""
        bad = tmp_path / "workers.json"
        bad.write_text("THIS IS NOT JSON {{{")
        result = _expected_instances("worker-<host>", manifest_path=bad)
        assert result == 1

    def test_custom_manifest_returns_correct_value(self, tmp_path):
        """Reads the correct instances value from a custom manifest."""
        manifest = {
            "_schema_version": 1,
            "workers": [
                {"name": "worker-X", "instances": 5},
                {"name": "worker-Y"},
            ],
        }
        path = tmp_path / "workers.json"
        path.write_text(json.dumps(manifest))
        assert _expected_instances("worker-X", manifest_path=path) == 5

    def test_custom_manifest_absent_worker_returns_1(self, tmp_path):
        """Worker not in the manifest → default 1."""
        manifest = {"_schema_version": 1, "workers": [{"name": "worker-Z"}]}
        path = tmp_path / "workers.json"
        path.write_text(json.dumps(manifest))
        assert _expected_instances("worker-absent", manifest_path=path) == 1

    def test_worker_with_instances_1_explicit(self, tmp_path):
        """Explicit instances=1 → returns 1."""
        manifest = {
            "_schema_version": 1,
            "workers": [{"name": "worker-HP", "instances": 1}],
        }
        path = tmp_path / "workers.json"
        path.write_text(json.dumps(manifest))
        assert _expected_instances("worker-HP", manifest_path=path) == 1

    def test_custom_manifest_instances_zero(self, tmp_path):
        """Explicit instances=0 → returns 0 (worker intentionally disabled)."""
        manifest = {
            "_schema_version": 1,
            "workers": [{"name": "worker-default", "instances": 0}],
        }
        path = tmp_path / "workers.json"
        path.write_text(json.dumps(manifest))
        assert _expected_instances("worker-default", manifest_path=path) == 0


# ---------------------------------------------------------------------------
# 2. _count_online_instances
# ---------------------------------------------------------------------------

class TestCountOnlineInstances:

    # --- Base-name matching (N=1 retrocompat) ---

    def test_base_name_alive_counts_1(self):
        workers = [_alive_worker("worker-default")]
        assert _count_online_instances(
            "worker-default", workers, time.time(), _OFFLINE_THRESHOLD
        ) == 1

    def test_base_name_dead_counts_0(self):
        workers = [_dead_worker("worker-default")]
        assert _count_online_instances(
            "worker-default", workers, time.time(), _OFFLINE_THRESHOLD
        ) == 0

    def test_empty_state_counts_0(self):
        assert _count_online_instances(
            "worker-<host>", [], time.time(), _OFFLINE_THRESHOLD
        ) == 0

    # --- Suffixed-name matching (multi-instance) ---

    def test_three_suffixed_instances_all_alive(self):
        workers = [
            _alive_worker("worker-<host>-1"),
            _alive_worker("worker-<host>-2"),
            _alive_worker("worker-<host>-3"),
        ]
        assert _count_online_instances(
            "worker-<host>", workers, time.time(), _OFFLINE_THRESHOLD
        ) == 3

    def test_two_of_three_suffixed_alive(self):
        workers = [
            _alive_worker("worker-<host>-1"),
            _alive_worker("worker-<host>-2"),
            _dead_worker("worker-<host>-3"),
        ]
        assert _count_online_instances(
            "worker-<host>", workers, time.time(), _OFFLINE_THRESHOLD
        ) == 2

    def test_none_of_three_suffixed_alive(self):
        workers = [
            _dead_worker("worker-<host>-1"),
            _dead_worker("worker-<host>-2"),
            _dead_worker("worker-<host>-3"),
        ]
        assert _count_online_instances(
            "worker-<host>", workers, time.time(), _OFFLINE_THRESHOLD
        ) == 0

    # --- No false positives from unrelated workers ---

    def test_unrelated_workers_not_counted(self):
        workers = [
            _alive_worker("worker-default"),
            _alive_worker("worker-HP"),
            _alive_worker("worker-<host>-extra"),  # not a digit suffix
            _alive_worker("worker-nvidiarackX-1"),     # different base name
        ]
        assert _count_online_instances(
            "worker-<host>", workers, time.time(), _OFFLINE_THRESHOLD
        ) == 0

    def test_suffix_must_be_digits_only(self):
        """Suffixes like '-foo' or '-1a' must NOT match."""
        workers = [
            _alive_worker("worker-<host>-foo"),
            _alive_worker("worker-<host>-1a"),
            _alive_worker("worker-<host>-"),
        ]
        assert _count_online_instances(
            "worker-<host>", workers, time.time(), _OFFLINE_THRESHOLD
        ) == 0

    # --- Mixed base name + suffixed in same state ---

    def test_base_name_counted_for_n1(self):
        """N=1 worker may register with the base name → still counted."""
        workers = [_alive_worker("worker-HP")]
        assert _count_online_instances(
            "worker-HP", workers, time.time(), _OFFLINE_THRESHOLD
        ) == 1

    # --- Heartbeat threshold boundary ---

    def test_exactly_at_threshold_not_counted(self):
        """Worker whose heartbeat is exactly at threshold is considered offline."""
        now = time.time()
        # last_heartbeat = now - threshold → age == threshold → NOT alive
        workers = [{"name": "worker-X", "last_heartbeat": now - _OFFLINE_THRESHOLD}]
        assert _count_online_instances("worker-X", workers, now, _OFFLINE_THRESHOLD) == 0

    def test_just_below_threshold_counted(self):
        """Worker whose heartbeat is 1s before threshold is still online."""
        now = time.time()
        workers = [{"name": "worker-X", "last_heartbeat": now - (_OFFLINE_THRESHOLD - 1)}]
        assert _count_online_instances("worker-X", workers, now, _OFFLINE_THRESHOLD) == 1


# ---------------------------------------------------------------------------
# 3. Integration via HTTP client
# ---------------------------------------------------------------------------

@pytest.fixture()
def discovery_client():
    """Thin fixture: create_app + TestClient, mirroring test_discovery_api.py."""
    from fastapi.testclient import TestClient
    from exonit.server import create_app

    app = create_app()
    client = TestClient(app)
    return client, app


class TestWorkersStartMultiInstance:
    """Verify already_online logic via the real HTTP endpoint."""

    def test_nvidiarack_3_instances_all_online(self, discovery_client):
        """worker-<host> with N live instances → already_online.

        Uses monkeypatch on _expected_instances so the test is decoupled from
        the live ops/workers.json fleet count.  The fixed expected value is 3
        and we register exactly 3 live instances.
        """
        client, app = discovery_client
        ws = app.state.workers_state
        now = time.time()

        # Register 3 suffixed workers directly into workers_state (bypasses SSH)
        for i in range(1, 4):
            wid = f"<host>-test-{i}"
            ws[wid] = {
                "name": f"worker-<host>-{i}",
                "last_heartbeat": now - 10,
                "status": "idle",
            }

        import exonit.server as _srv

        def _patched_expected(name, manifest_path=None):
            if name == "worker-<host>":
                return 3
            return _expected_instances(name, manifest_path=manifest_path)

        with patch.object(_srv, "_expected_instances", side_effect=_patched_expected), \
             patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="ok", stderr="")
            resp = client.post("/api/workers/start")

        assert resp.status_code == 200
        results = {r["worker"]: r for r in resp.json()}
        assert results["worker-<host>"]["action"] == "already_online"
        # Detail must mention the count
        assert "3/3" in results["worker-<host>"]["detail"]

    def test_nvidiarack_2_of_3_online_triggers_start(self, discovery_client):
        """worker-<host> with only 2/3 live instances → farmctl start called."""
        client, app = discovery_client
        ws = app.state.workers_state
        now = time.time()

        # Only 2 of 3 instances online
        for i in range(1, 3):
            wid = f"<host>-partial-{i}"
            ws[wid] = {
                "name": f"worker-<host>-{i}",
                "last_heartbeat": now - 10,
                "status": "idle",
            }

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="started", stderr="")
            resp = client.post("/api/workers/start")

        assert resp.status_code == 200
        results = {r["worker"]: r for r in resp.json()}
        # Must NOT be already_online — must have called farmctl
        assert results["worker-<host>"]["action"] in ("started", "failed")

    def test_worker_default_single_instance_retrocompat(self, discovery_client):
        """worker-default (N=1) with base-name heartbeat → already_online."""
        client, app = discovery_client
        ws = app.state.workers_state
        now = time.time()

        # Register with the base name (no suffix), as N=1 workers do
        ws["default-test-1"] = {
            "name": "worker-default",
            "last_heartbeat": now - 10,
            "status": "idle",
        }

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="ok", stderr="")
            resp = client.post("/api/workers/start")

        assert resp.status_code == 200
        results = {r["worker"]: r for r in resp.json()}
        assert results["worker-default"]["action"] == "already_online"

    def test_all_offline_calls_farmctl_for_all(self, discovery_client):
        """No workers online → farmctl called for all campaign workers with instances>=1.

        Monkeypatches _expected_instances to return 1 for every worker so that
        the test is independent of the live ops/workers.json fleet counts (a
        worker with instances=0 like worker-default legitimately skips farmctl
        via already_online, which would make the assert flaky against the live
        config).
        """
        client, app = discovery_client
        # workers_state is empty (no registrations)

        import exonit.server as _srv

        with patch.object(_srv, "_expected_instances", return_value=1), \
             patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="ok", stderr="")
            resp = client.post("/api/workers/start")

        assert resp.status_code == 200
        results = resp.json()
        from exonit.server import _campaign_worker_names
        expected_names = _campaign_worker_names()  # live from ops/workers.json
        assert expected_names  # sanity: manifest readable, non-empty
        assert len(results) == len(expected_names)
        actions = {r["worker"]: r["action"] for r in results}
        for name in expected_names:
            assert name in actions
            assert actions[name] in ("started", "failed")

    def test_already_online_detail_format(self, discovery_client):
        """already_online detail must include online_count/expected."""
        client, app = discovery_client
        ws = app.state.workers_state
        now = time.time()

        # worker-HP has instances=1; register with base name
        ws["hp-test"] = {
            "name": "worker-HP",
            "last_heartbeat": now - 5,
            "status": "idle",
        }

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="ok", stderr="")
            resp = client.post("/api/workers/start")

        results = {r["worker"]: r for r in resp.json()}
        assert results["worker-HP"]["action"] == "already_online"
        assert "1/1" in results["worker-HP"]["detail"]
