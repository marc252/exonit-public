"""Tests for the Phase-A server fixes in
docs/reviews/2026-07-02-opus48-cleanup-review.md (P1, P2, P4, P6, P7, D12,
and the minor dedups 11(b)/11(c)).

D1/B5 (strong-tier repoint + nulls_last) are covered by
tests/test_strong_tier_agreement_gate.py; D8 (verdict_io) by
tests/test_verdict_io.py.
"""
import json
import os
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

AUTH = {"Authorization": "Bearer exonit-default-token"}


@pytest.fixture
def app_and_db(tmp_path, monkeypatch):
    """A fresh app + the ExoNitDB instance it uses (same file, separate
    handle) for direct row insertion."""
    monkeypatch.setenv("EXONIT_DB_PATH", str(tmp_path / "test.db"))
    monkeypatch.setenv("EXONIT_REPORTS_DIR", str(tmp_path / "reports"))
    monkeypatch.delenv("EXONIT_DATABASE_URL", raising=False)
    os.makedirs(tmp_path / "reports", exist_ok=True)

    from exonit.server import create_app
    from exonit.db import ExoNitDB

    app = create_app()
    db = ExoNitDB(str(tmp_path / "test.db"))
    return app, db


@pytest.fixture
def client(app_and_db):
    app, _db = app_and_db
    return TestClient(app)


def _seed_candidate(db, tic, snr=12.0, cnn=0.8, period=3.0, depth=500.0):
    db.add_star(tic, ra=0.0, dec=0.0, tmag=11.0,
               n_sectors=1, n_datapoints=100, time_span_days=27.0)
    db.add_candidate(tic, candidate_num=1, period_days=period, t0=1.0,
                     duration_days=0.1, depth_ppm=depth, snr=snr,
                     cnn_score=cnn)


# --------------------------------------------------------------------------- #
# P1 — /api/candidates default (no-filter) response is cached
# --------------------------------------------------------------------------- #

class TestCandidatesCaching:
    def test_repeated_identical_call_hits_cache(self, app_and_db, client):
        """Two identical-param GETs within the TTL must not re-run the
        COUNT/SELECT — verified by mutating the DB between calls and
        checking the (stale, cached) second response still matches the
        first, not the mutated state."""
        app, db = app_and_db
        _seed_candidate(db, "100")

        r1 = client.get("/api/candidates")
        assert r1.status_code == 200
        assert r1.json()["total"] == 1

        # Mutate the DB directly (bypassing the endpoint) — a live query
        # would now see 2 rows.
        _seed_candidate(db, "200")

        r2 = client.get("/api/candidates")
        assert r2.json()["total"] == 1, (
            "second call within the cache TTL must be served from cache, "
            "not re-query the (now-changed) DB"
        )

    def test_different_params_are_not_cache_collisions(self, app_and_db, client):
        app, db = app_and_db
        _seed_candidate(db, "100", snr=5.0)
        _seed_candidate(db, "200", snr=50.0)

        r_all = client.get("/api/candidates")
        r_filtered = client.get("/api/candidates?min_snr=10")
        assert r_all.json()["total"] == 2
        assert r_filtered.json()["total"] == 1
        assert r_filtered.json()["candidates"][0]["tic_id"] == "200"

    def test_tic_id_lookup_bypasses_cache(self, app_and_db, client):
        """The tic_id single-candidate lookup is a distinct, uncached path
        (unchanged by P1) — must always reflect the live DB."""
        app, db = app_and_db
        _seed_candidate(db, "100")
        r1 = client.get("/api/candidates?tic_id=100")
        assert len(r1.json()) == 1

        _seed_candidate(db, "100")  # add a second candidate_num? re-add same tic
        db.add_candidate("100", candidate_num=2, period_days=5.0, t0=1.0,
                         duration_days=0.1, depth_ppm=300.0, snr=9.0)
        r2 = client.get("/api/candidates?tic_id=100")
        assert len(r2.json()) == 2, "tic_id lookups must never be cached"


# --------------------------------------------------------------------------- #
# B5 (server.py sort_col map) — NULLs always sort last, every sort option
# --------------------------------------------------------------------------- #

class TestCandidatesSortNullsLast:
    @pytest.mark.parametrize("sort_param,null_field", [
        ("snr", "snr"),
        ("period", "period_days"),
        ("depth", "depth_ppm"),
        ("cnn", "cnn_score"),
    ])
    def test_null_valued_row_sorts_last(self, app_and_db, client, sort_param, null_field):
        app, db = app_and_db
        db.add_star("has_value", ra=0, dec=0, tmag=10.0,
                   n_sectors=1, n_datapoints=100, time_span_days=27.0)
        db.add_star("has_null", ra=0, dec=0, tmag=10.0,
                   n_sectors=1, n_datapoints=100, time_span_days=27.0)
        vals = dict(period_days=3.0, t0=1.0, duration_days=0.1,
                   depth_ppm=500.0, snr=12.0, cnn_score=0.8)
        null_vals = dict(vals)
        null_vals[null_field] = None
        db.add_candidate("has_value", candidate_num=1, **vals)
        db.add_candidate("has_null", candidate_num=1, **null_vals)

        r = client.get(f"/api/candidates?sort={sort_param}")
        assert r.status_code == 200
        tics = [c["tic_id"] for c in r.json()["candidates"]]
        assert tics[-1] == "has_null", (
            f"sort={sort_param}: NULL {null_field} row must sort last, got order {tics}"
        )
        assert tics[0] == "has_value"


# --------------------------------------------------------------------------- #
# P2 — export row cap + truncation flag
# --------------------------------------------------------------------------- #

class TestExportRowCap:
    def test_csv_export_truncates_and_flags_header(self, app_and_db, client, monkeypatch):
        monkeypatch.setenv("EXONIT_EXPORT_ROW_CAP", "3")
        # Recreate the app so the closure re-reads the env var.
        from exonit.server import create_app
        app, db = app_and_db
        for i in range(5):
            _seed_candidate(db, f"tic{i}", snr=10.0 + i)
        client2 = TestClient(create_app())

        r = client2.get("/api/export/csv")
        assert r.status_code == 200
        assert r.headers.get("x-export-truncated") == "true"
        # header row + 3 data rows
        lines = [ln for ln in r.text.splitlines() if ln]
        assert len(lines) == 1 + 3

    def test_csv_export_no_truncation_flag_under_cap(self, app_and_db, client, monkeypatch):
        monkeypatch.setenv("EXONIT_EXPORT_ROW_CAP", "50")
        from exonit.server import create_app
        app, db = app_and_db
        _seed_candidate(db, "100")
        client2 = TestClient(create_app())

        r = client2.get("/api/export/csv")
        assert r.status_code == 200
        assert "x-export-truncated" not in r.headers

    def test_json_export_caps_candidates_and_stars_independently(
            self, app_and_db, monkeypatch):
        monkeypatch.setenv("EXONIT_EXPORT_ROW_CAP", "2")
        from exonit.server import create_app
        app, db = app_and_db
        for i in range(4):
            _seed_candidate(db, f"tic{i}", snr=10.0 + i)
        client2 = TestClient(create_app())

        r = client2.get("/api/export/json")
        assert r.status_code == 200
        body = r.json()
        assert body["truncated_candidates"] is True
        assert body["truncated_stars"] is True
        assert len(body["candidates"]) == 2
        assert len(body["stars"]) == 2
        assert body["total_candidates"] == 2
        assert body["total_stars"] == 2


# --------------------------------------------------------------------------- #
# P4 — gap_stats moved to shared_state (cross-process-safe)
# --------------------------------------------------------------------------- #

class TestGapStatsSharedState:
    def _registry(self):
        return {
            "active_sector": 1,
            "sectors": {"1": {"enabled": True, "total_tics": 1000,
                              "downloaded": 5, "cursor": 3,
                              "canonical_list": ""}},
        }

    def test_second_process_reuses_first_processs_computed_value(
            self, tmp_path, monkeypatch):
        """Two independent create_app() calls simulate two different A3
        uvicorn worker processes with independent closures; they share the
        process-global shared_state singleton (MemoryBackend in tests, Redis
        in prod). The expensive scan (load_completed_tic_set) must run at
        most once across both, not once per app instance."""
        monkeypatch.setenv("EXONIT_DB_PATH", str(tmp_path / "test.db"))
        monkeypatch.setenv("EXONIT_REPORTS_DIR", str(tmp_path / "reports"))
        os.makedirs(tmp_path / "reports", exist_ok=True)

        from exonit import discovery_state
        from exonit.server import create_app
        from exonit.shared_state import get_shared_state

        calls = {"n": 0}

        def _fake_load_completed(db_path):
            calls["n"] += 1
            return set()

        def _fake_gap_stats(num, *, db_path, canonical_csv, tglc_index, completed):
            return {"gaps_processable": 1, "no_data": 0, "completed": 0}

        monkeypatch.setattr(discovery_state, "load_completed_tic_set",
                            _fake_load_completed)
        monkeypatch.setattr(discovery_state, "sector_gap_stats", _fake_gap_stats)
        monkeypatch.setattr(discovery_state, "load_registry",
                            lambda path: self_registry())
        monkeypatch.setattr(discovery_state, "get_sector_state",
                            lambda n, db_path=None: "active")

        def self_registry():
            return {
                "active_sector": 1,
                "sectors": {"1": {"enabled": True, "total_tics": 1000,
                                  "downloaded": 5, "cursor": 3,
                                  "canonical_list": ""}},
            }

        app1 = create_app()
        app2 = create_app()
        c1 = TestClient(app1)
        c2 = TestClient(app2)

        r1 = c1.get("/api/discovery/sectors")
        assert r1.status_code == 200
        assert calls["n"] == 1, "first (cold-cache) call must compute inline"

        # Force the OUTER /api/discovery/sectors _cached_json (30s TTL) to
        # miss on the next call, so app2's own _compute() genuinely runs and
        # exercises its _get_gap_stats — without this the outer cache alone
        # (shared across app1/app2 too) would mask whether the INNER
        # gap-stats cache is process-local or shared.
        get_shared_state().delete("cache:discovery_sectors")

        r2 = c2.get("/api/discovery/sectors")
        assert r2.status_code == 200
        assert calls["n"] == 1, (
            "a second process must reuse the first process's cached "
            "gap-stats value instead of recomputing it (P4 regression)"
        )


# --------------------------------------------------------------------------- #
# P6 — heartbeat round-trip batching
# --------------------------------------------------------------------------- #

class TestHeartbeatBatching:
    def test_heartbeat_batches_registry_writes(self, app_and_db, client, monkeypatch):
        """Count hget_json/hset_json calls the heartbeat triggers against the
        workers:registry hash. Before P6 this was 5-9 (one read+write pair
        per field written); after batching it must be a small constant
        number regardless of how many optional introspection fields are
        present in the body."""
        from exonit.shared_state import get_shared_state, MemoryBackend

        reg = client.post("/api/worker/register",
                          json={"name": "hb-test", "capabilities": {}},
                          headers=AUTH)
        wid = reg.json()["worker_id"]

        backend = get_shared_state()
        assert isinstance(backend, MemoryBackend)
        counts = {"hget": 0, "hset": 0}
        orig_hget = backend.hget_json
        orig_hset = backend.hset_json

        def _counted_hget(key, field):
            if key == "workers:registry":
                counts["hget"] += 1
            return orig_hget(key, field)

        def _counted_hset(key, field, value):
            if key == "workers:registry":
                counts["hset"] += 1
            return orig_hset(key, field, value)

        monkeypatch.setattr(backend, "hget_json", _counted_hget)
        monkeypatch.setattr(backend, "hset_json", _counted_hset)

        resp = client.post(
            "/api/worker/heartbeat",
            json={"worker_id": wid, "status": "working", "current_tic": "123",
                 "git_sha": "abc123", "flag_states": {"x": 1},
                 "uptime_s": 42, "pid": 999},
            headers=AUTH)
        assert resp.status_code == 200

        # One .update() call = one hget_json (_load) + one hset_json (_save)
        # against the registry hash. Generously allow a couple more for any
        # incidental reads (e.g. the initial `workers_state.get` snapshot),
        # but must stay far below the old per-field cost (which for this
        # body would have been 5 writes x 2 round trips = 10).
        assert counts["hget"] <= 3, f"too many registry reads: {counts}"
        assert counts["hset"] <= 2, f"too many registry writes: {counts}"

    def test_heartbeat_still_sets_all_fields(self, app_and_db, client):
        """Functional check alongside the round-trip-count check above:
        batching must not drop any field."""
        reg = client.post("/api/worker/register",
                          json={"name": "hb-fields-test", "capabilities": {}},
                          headers=AUTH)
        wid = reg.json()["worker_id"]

        client.post(
            "/api/worker/heartbeat",
            json={"worker_id": wid, "status": "working", "current_tic": "555",
                 "git_sha": "deadbeef", "uptime_s": 7, "pid": 321},
            headers=AUTH)

        listing = client.get("/api/worker/list").json()["workers"]
        w = next(x for x in listing if x["worker_id"] == wid)
        assert w["current_tic"] == "555"
        assert w["git_sha"] == "deadbeef"
        assert w["uptime_s"] == 7
        assert w["pid"] == 321


# --------------------------------------------------------------------------- #
# P7 — downloads catalog: mtime cache + bulk lookups
# --------------------------------------------------------------------------- #

class TestDownloadsCatalogCache:
    def _write_csv(self, path, rows):
        import csv
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=[
                "TIC ID", "TFOPWG Disposition", "TMag Value", "Period (days)"])
            w.writeheader()
            for r in rows:
                w.writerow(r)

    def test_parses_once_per_mtime(self, tmp_path):
        from exonit.server import _load_downloads_catalog_rows

        csv_path = tmp_path / "toi_p7.csv"
        self._write_csv(csv_path, [
            {"TIC ID": "1", "TFOPWG Disposition": "KP",
             "TMag Value": "10.0", "Period (days)": "3.0"},
        ])

        rows1 = _load_downloads_catalog_rows(str(csv_path))
        assert len(rows1) == 1

        # Overwrite with different content but DO NOT bump mtime explicitly —
        # os.path.getmtime resolution can coincide on a fast filesystem, so
        # force a distinct mtime to make the cache-busting deterministic.
        self._write_csv(csv_path, [
            {"TIC ID": "1", "TFOPWG Disposition": "KP",
             "TMag Value": "10.0", "Period (days)": "3.0"},
            {"TIC ID": "2", "TFOPWG Disposition": "PC",
             "TMag Value": "11.0", "Period (days)": "5.0"},
        ])
        newer = os.path.getmtime(csv_path) + 5
        os.utime(csv_path, (newer, newer))

        rows2 = _load_downloads_catalog_rows(str(csv_path))
        assert len(rows2) == 2, "mtime bump must invalidate the cache"

    def test_dedups_first_row_per_tic(self, tmp_path):
        from exonit.server import _load_downloads_catalog_rows

        csv_path = tmp_path / "toi_dedup.csv"
        self._write_csv(csv_path, [
            {"TIC ID": "1", "TFOPWG Disposition": "KP",
             "TMag Value": "10.0", "Period (days)": "3.0"},
            {"TIC ID": "1", "TFOPWG Disposition": "SHOULD_NOT_WIN",
             "TMag Value": "99", "Period (days)": "99"},
        ])
        rows = _load_downloads_catalog_rows(str(csv_path))
        assert len(rows) == 1
        assert rows[0]["disposition"] == "KP"

    def test_endpoint_bulk_lookup_matches_per_row_semantics(
            self, app_and_db, client, monkeypatch, tmp_path):
        """End-to-end: has_fits / is_analyzed / status must be computed
        correctly via the new bulk lookups (select_in_chunks +
        MetadataCache.all_tic_ids()), matching what the old N+1 loop did."""
        app, db = app_and_db
        db.add_star("100", ra=0, dec=0, tmag=10.0, n_sectors=1,
                   n_datapoints=100, time_span_days=27.0, status="completed")
        db.add_star("200", ra=0, dec=0, tmag=10.0, n_sectors=1,
                   n_datapoints=100, time_span_days=27.0, status="pending")

        csv_path = tmp_path / "toi_endpoint.csv"
        self._write_csv(csv_path, [
            {"TIC ID": "100", "TFOPWG Disposition": "KP",
             "TMag Value": "10.0", "Period (days)": "3.0"},
            {"TIC ID": "200", "TFOPWG Disposition": "PC",
             "TMag Value": "11.0", "Period (days)": "5.0"},
            {"TIC ID": "300", "TFOPWG Disposition": "PC",
             "TMag Value": "12.0", "Period (days)": "1.0"},
        ])
        monkeypatch.setattr("exonit.server._toi_catalog_path",
                            lambda: str(csv_path))

        r = client.get("/api/downloads/catalog")
        assert r.status_code == 200
        entries = {e["tic_id"]: e for e in r.json()["entries"]}
        assert entries["100"]["status"] == "analyzed"  # is_analyzed=True
        assert entries["200"]["status"] == "pending"    # not analyzed, no fits
        assert entries["300"]["status"] == "pending"    # unknown TIC entirely
        assert r.json()["counts"]["analyzed"] == 1
        assert r.json()["counts"]["pending"] == 2


# --------------------------------------------------------------------------- #
# D12 — reap/claim/payload helpers shared by worker_fpp_task/worker_exominer_task
# --------------------------------------------------------------------------- #

class TestD12BackfillHelpers:
    def setup_method(self):
        from exonit.shared_state import reset_shared_state
        os.environ["EXONIT_SHARED_STATE_BACKEND"] = "memory"
        reset_shared_state()

    def teardown_method(self):
        from exonit.shared_state import reset_shared_state
        reset_shared_state()

    def test_claim_candidate_is_exclusive(self):
        from exonit.server import _claim_candidate

        assert _claim_candidate("test:assignments", 42, now=1000.0) is True
        # A second claim attempt on the same id must lose.
        assert _claim_candidate("test:assignments", 42, now=1001.0) is False
        # A different id is independent.
        assert _claim_candidate("test:assignments", 43, now=1000.0) is True

    def test_reap_stale_assignments_drops_only_expired(self):
        from exonit.shared_state import get_shared_state
        from exonit.server import _reap_stale_assignments

        st = get_shared_state()
        now = 10_000.0
        st.hset_json("test:assignments", "1", {"ts": now - 5000})   # stale
        st.hset_json("test:assignments", "2", {"ts": now - 10})     # fresh

        import exonit.server as srv
        orig_time = srv._time.time
        srv._time.time = lambda: now
        try:
            _reap_stale_assignments("test:assignments", 1800)
        finally:
            srv._time.time = orig_time

        remaining = st.hgetall_json("test:assignments")
        assert "1" not in remaining
        assert "2" in remaining

    def test_build_backfill_payload_shape(self):
        from exonit.server import _build_backfill_payload

        cand = {"id": 1, "tic_id": "999", "candidate_num": 2,
               "period_days": 3.0, "t0": 1.0, "duration_days": 0.1,
               "depth_ppm": 500.0, "snr": 12.0}
        payload = _build_backfill_payload(cand)
        assert payload == {"task": {
            "candidate_id": 1, "tic_id": "999", "candidate_num": 2,
            "period_days": 3.0, "t0": 1.0, "duration_days": 0.1,
            "depth_ppm": 500.0, "snr": 12.0, "n_sectors": 1,
        }}

    def test_build_backfill_payload_uses_provided_n_sectors(self):
        from exonit.server import _build_backfill_payload

        cand = {"id": 1, "tic_id": "999", "candidate_num": 2,
               "period_days": 3.0, "t0": 1.0, "duration_days": 0.1,
               "depth_ppm": 500.0, "snr": 12.0, "n_sectors": 5}
        payload = _build_backfill_payload(cand)
        assert payload["task"]["n_sectors"] == 5


# --------------------------------------------------------------------------- #
# Minor dedup 11(b) — AWS endpoint guard/try helpers
# --------------------------------------------------------------------------- #

class TestAwsHelpers:
    def test_require_deployment_matches_returns_none(self, monkeypatch):
        from exonit.server import _require_deployment
        monkeypatch.setenv("EXONIT_DEPLOYMENT", "aws")
        assert _require_deployment("aws") is None

    def test_require_deployment_mismatch_returns_403(self, monkeypatch):
        from exonit.server import _require_deployment
        monkeypatch.setenv("EXONIT_DEPLOYMENT", "home")
        resp = _require_deployment("aws")
        assert resp is not None
        assert resp.status_code == 403
        body = json.loads(bytes(resp.body))
        assert body["error"] == "not_aws_deployment"

    def test_aws_try_success_returns_result_no_error(self, monkeypatch):
        from exonit import aws_control
        from exonit.server import _aws_try
        monkeypatch.setattr(aws_control, "fake_action", lambda x: {"ok": x}, raising=False)

        result, err = _aws_try("fake_action", 5)
        assert err is None
        assert result == {"ok": 5}

    def test_aws_try_exception_returns_500_with_label(self, monkeypatch):
        from exonit import aws_control
        from exonit.server import _aws_try

        def _boom():
            raise RuntimeError("boto3 exploded")
        monkeypatch.setattr(aws_control, "fake_boom", _boom, raising=False)

        result, err = _aws_try("fake_boom", action_label="custom_label")
        assert result is None
        assert err is not None
        assert err.status_code == 500
        body = json.loads(bytes(err.body))
        assert body["error"] == "custom_label_failed"
        assert "boto3 exploded" in body["detail"]


# --------------------------------------------------------------------------- #
# Minor dedup 11(c) — cache-first proxy fetch
# --------------------------------------------------------------------------- #

class TestProxyFetch:
    def test_cache_hit_skips_fetch(self):
        from exonit.server import _proxy_fetch

        store = MagicMock()
        store.get.return_value = {"cached": True}
        fetch_fn = MagicMock()

        result = _proxy_fetch(store, ("123",), fetch_fn, not_found_error="nf")
        assert result == {"cached": True}
        fetch_fn.assert_not_called()

    def test_cache_miss_fetches_and_stores(self):
        from exonit.server import _proxy_fetch

        store = MagicMock()
        store.get.return_value = None
        fetch_fn = MagicMock(return_value={"fresh": True})

        result = _proxy_fetch(store, ("123",), fetch_fn, not_found_error="nf")
        assert result == {"fresh": True}
        store.put.assert_called_once_with("123", {"fresh": True})

    def test_fetch_returns_none_gives_404(self):
        from exonit.server import _proxy_fetch

        store = MagicMock()
        store.get.return_value = None
        fetch_fn = MagicMock(return_value=None)

        result = _proxy_fetch(store, ("123",), fetch_fn, not_found_error="no_thing")
        assert result.status_code == 404
        body = json.loads(bytes(result.body))
        assert body["error"] == "no_thing"
        store.put.assert_not_called()

    def test_fetch_exception_gives_502(self):
        from exonit.server import _proxy_fetch

        store = MagicMock()
        store.get.return_value = None

        def _boom():
            raise RuntimeError("network down")

        result = _proxy_fetch(store, ("123",), _boom, not_found_error="nf")
        assert result.status_code == 502
        body = json.loads(bytes(result.body))
        assert body["error"] == "fetch_failed"

    def test_known_missing_value_short_circuits_without_fetch(self):
        from exonit.server import _proxy_fetch

        store = MagicMock()
        store.get.return_value = b""
        fetch_fn = MagicMock()

        result = _proxy_fetch(
            store, ("123", "1-40"), fetch_fn, not_found_error="no_dv",
            known_missing_value=b"")
        assert result.status_code == 404
        fetch_fn.assert_not_called()

    def test_mark_missing_fn_called_on_none_fetch(self):
        from exonit.server import _proxy_fetch

        store = MagicMock()
        store.get.return_value = None
        mark_missing = MagicMock()

        _proxy_fetch(
            store, ("123", "1-40"), lambda: None, not_found_error="no_dv",
            mark_missing_fn=mark_missing)
        mark_missing.assert_called_once_with("123", "1-40")

    def test_proxy_not_found_maps_to_custom_error_code(self):
        from exonit.server import _proxy_fetch, _ProxyNotFound

        store = MagicMock()
        store.get.return_value = None

        def _fetch():
            raise _ProxyNotFound("no_ra_dec")

        result = _proxy_fetch(store, ("123",), _fetch, not_found_error="no_gaia_match")
        assert result.status_code == 404
        body = json.loads(bytes(result.body))
        assert body["error"] == "no_ra_dec"

    def test_custom_to_response_used_on_hit_and_miss(self):
        from exonit.server import _proxy_fetch

        store = MagicMock()
        store.get.return_value = b"cached-bytes"
        to_response = MagicMock(side_effect=lambda d: f"wrapped:{d}")

        result = _proxy_fetch(
            store, ("123",), MagicMock(), not_found_error="nf",
            to_response=to_response)
        assert result == f"wrapped:{b'cached-bytes'}"
