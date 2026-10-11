"""Tests for exonit.enqueue — standalone enqueue logic."""

import unittest
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def _clear_metadata_v2_env(monkeypatch):
    """Isolate every test in this module from the metadata-bulk-enqueue flags
    (docs/superpowers/specs/2026-09-26-metadata-refresh-and-bulk-enqueue-
    design.md) — default OFF for all pre-existing tests, which pin the
    legacy (V2=0) behaviour byte-for-byte."""
    monkeypatch.delenv("EXONIT_ENQUEUE_METADATA_V2", raising=False)
    monkeypatch.delenv("EXONIT_METADATA_REFRESH", raising=False)


class TestEnqueueTargets(unittest.TestCase):
    """Test enqueue_targets() function."""

    def setUp(self):
        self.db = MagicMock()
        # completed_tic_ids() is the new bulk-set API used by the optimised
        # pre-filter.  Default: no TICs already completed.
        self.db.completed_tic_ids.return_value = set()
        self.server_url = "http://localhost:8400"

    @patch("exonit.enqueue.MetadataCache")
    @patch("exonit.enqueue.requests")
    def test_enqueue_skips_analyzed(self, mock_requests, mock_mc_cls):
        """Already-analyzed stars are skipped without checking metadata or posting."""
        from exonit.enqueue import enqueue_targets

        self.db.completed_tic_ids.return_value = {"123456789"}

        # MetadataCache mock: all_tic_ids returns empty (irrelevant, TIC already
        # in analyzed_set so we never reach the metadata check)
        mock_mc = MagicMock()
        mock_mc.all_tic_ids.return_value = set()
        mock_mc_cls.return_value = mock_mc

        targets = [{"tic_id": "123456789"}]
        result = enqueue_targets(targets, self.db, self.server_url,
                                 cache_path="/tmp/dummy.db")

        self.assertEqual(result["skipped"], 1)
        self.assertEqual(result["enqueued"], 0)
        mock_requests.post.assert_not_called()

    @patch("exonit.enqueue.MetadataCache")
    @patch("exonit.enqueue.requests")
    def test_enqueue_posts_to_server(self, mock_requests, mock_mc_cls):
        """Targets with metadata get bulk-posted to /api/worker/enqueue_bulk.

        The probe POST (empty list) returns 200, so the bulk path activates and
        all targets go in one chunked request — not one POST per target.
        """
        from exonit.enqueue import enqueue_targets

        # Neither TIC is analyzed.
        self.db.completed_tic_ids.return_value = set()

        # Both TICs are already in the metadata cache (no ensure_metadata needed).
        mock_mc = MagicMock()
        mock_mc.all_tic_ids.return_value = {"111", "222"}
        mock_mc_cls.return_value = mock_mc

        bulk_resp = MagicMock(status_code=200)
        bulk_resp.json.return_value = {
            "enqueued": 2, "skipped": 0, "failed": 0, "elapsed_seconds": 0.01,
        }
        mock_requests.post.return_value = bulk_resp

        targets = [
            {"tic_id": "111", "tmag": 10.5, "ra": 120.0, "dec": -30.0},
            {"tic_id": "222", "tmag": 11.0, "ra": 130.0, "dec": -25.0},
        ]
        result = enqueue_targets(targets, self.db, self.server_url,
                                 cache_path="/tmp/dummy.db")

        self.assertEqual(result["enqueued"], 2)
        self.assertEqual(result["skipped"], 0)
        # 1 probe + 1 bulk POST = 2 total (was N+1 per-target POSTs in old flow)
        self.assertEqual(mock_requests.post.call_count, 2)

        # Find the bulk POST (the second call; the first is the probe).
        bulk_call = mock_requests.post.call_args_list[1]
        self.assertIn("/api/worker/enqueue_bulk", bulk_call[0][0])
        payload = bulk_call[1]["json"]
        self.assertIn("targets", payload)
        self.assertEqual(len(payload["targets"]), 2)
        # Auth header on the bulk call
        self.assertIn("Authorization", bulk_call[1]["headers"])
        self.assertTrue(
            bulk_call[1]["headers"]["Authorization"].startswith("Bearer "))

    @patch("exonit.enqueue.MetadataCache")
    @patch("exonit.enqueue.requests")
    def test_enqueue_skips_no_metadata(self, mock_requests, mock_mc_cls):
        """Stars without metadata cache entry are skipped and marked no_fits."""
        from exonit.enqueue import enqueue_targets

        self.db.completed_tic_ids.return_value = set()

        # TIC 999 is NOT in the metadata set → goes to need_index.
        mock_mc = MagicMock()
        mock_mc.all_tic_ids.return_value = set()
        mock_mc_cls.return_value = mock_mc

        with patch("exonit.enqueue.ensure_metadata", return_value=False):
            targets = [{"tic_id": "999"}]
            result = enqueue_targets(targets, self.db, self.server_url,
                                     cache_path="/tmp/dummy.db")

        self.assertEqual(result["skipped"], 1)
        self.assertEqual(result["enqueued"], 0)
        mock_requests.post.assert_not_called()
        # Should mark the star as no_fits in the DB — Q9 (perf review
        # 2026-09-26): one batched add_stars_bulk() call, not one add_star()
        # per TIC.
        self.db.add_stars_bulk.assert_called_once()
        call_args, call_kwargs = self.db.add_stars_bulk.call_args
        self.assertEqual(list(call_args[0]), ["999"])
        self.assertEqual(call_kwargs["status"], "no_fits")


class TestLegacySummaryMetadataKeysShape(unittest.TestCase):
    """2026-09-26 review fix #2: the legacy path (EXONIT_ENQUEUE_METADATA_V2
    unset/0, the default) must return the SAME zero-valued metadata_* keys
    the V2 path uses, not a bare ``{}`` — the summary shape must be uniform
    across both paths regardless of outcome."""

    _EXPECTED_ZERO_KEYS = {
        "metadata_created": 0, "metadata_refreshed": 0,
        "metadata_unchanged": 0, "metadata_preserved_non_tglc": 0,
    }

    def setUp(self):
        self.db = MagicMock()
        self.server_url = "http://localhost:8400"

    @patch("exonit.enqueue.MetadataCache")
    @patch("exonit.enqueue.requests")
    def test_early_return_zero_ready_tics_has_metadata_keys(
            self, mock_requests, mock_mc_cls):
        """The '0 ready_tics' early-return branch (everything skipped)."""
        from exonit.enqueue import enqueue_targets

        self.db.completed_tic_ids.return_value = {"123456789"}
        mock_mc = MagicMock()
        mock_mc.all_tic_ids.return_value = set()
        mock_mc_cls.return_value = mock_mc

        result = enqueue_targets(
            [{"tic_id": "123456789"}], self.db, self.server_url,
            cache_path="/tmp/dummy.db")

        for key, expected in self._EXPECTED_ZERO_KEYS.items():
            self.assertIn(key, result, f"{key} missing from legacy summary")
            self.assertEqual(result[key], expected)

    @patch("exonit.enqueue.MetadataCache")
    @patch("exonit.enqueue.requests")
    def test_normal_post_path_has_metadata_keys(
            self, mock_requests, mock_mc_cls):
        """The normal 'ready_tics posted to the server' branch."""
        from exonit.enqueue import enqueue_targets

        self.db.completed_tic_ids.return_value = set()
        mock_mc = MagicMock()
        mock_mc.all_tic_ids.return_value = {"111"}
        mock_mc_cls.return_value = mock_mc

        bulk_resp = MagicMock(status_code=200)
        bulk_resp.json.return_value = {
            "enqueued": 1, "skipped": 0, "failed": 0,
        }
        mock_requests.post.return_value = bulk_resp

        result = enqueue_targets(
            [{"tic_id": "111"}], self.db, self.server_url,
            cache_path="/tmp/dummy.db")

        for key, expected in self._EXPECTED_ZERO_KEYS.items():
            self.assertIn(key, result, f"{key} missing from legacy summary")
            self.assertEqual(result[key], expected)


class TestHasMetadata(unittest.TestCase):
    """Test _has_metadata() helper (retained for backwards compat tests)."""

    @patch("exonit.enqueue.MetadataCache")
    def test_returns_true_when_cached(self, mock_cache_cls):
        from exonit.enqueue import _has_metadata

        mock_cache = MagicMock()
        mock_cache.get_entry.return_value = {"tic_id": "123", "fits_files": ["a.fits"]}
        mock_cache_cls.return_value = mock_cache

        self.assertTrue(_has_metadata("123", cache_path="/tmp/test.db"))

    @patch("exonit.enqueue.MetadataCache")
    def test_returns_false_when_not_cached(self, mock_cache_cls):
        from exonit.enqueue import _has_metadata

        mock_cache = MagicMock()
        mock_cache.get_entry.return_value = None
        mock_cache_cls.return_value = mock_cache

        self.assertFalse(_has_metadata("999", cache_path="/tmp/test.db"))


class TestEnqueueTargetsBulkPath(unittest.TestCase):
    """Verify that enqueue_targets uses the bulk endpoint for large batches.

    We monkeypatch requests.post so we can count calls and assert that
    5000 synthetic targets produce exactly ceil(5000 / _BULK_CHUNK_SIZE)
    bulk calls — NOT 5000 individual calls.
    """

    def setUp(self):
        self.db = MagicMock()
        self.db.completed_tic_ids.return_value = set()
        self.server_url = "http://localhost:8400"

    @patch("exonit.enqueue.ensure_metadata", return_value=True)
    @patch("exonit.enqueue.MetadataCache")
    @patch("exonit.enqueue.requests")
    def test_5000_targets_use_bulk_not_per_target(self, mock_requests, mock_mc_cls, _ens):
        """5000 targets → ceil(5000/5000) = 1 bulk call, not 5000 single calls."""
        import math
        from exonit.enqueue import enqueue_targets, _BULK_CHUNK_SIZE

        n_targets = 5000
        # All TICs are already in the metadata set (no ensure_metadata needed).
        mock_mc = MagicMock()
        mock_mc.all_tic_ids.return_value = {str(i) for i in range(1, n_targets + 1)}
        mock_mc_cls.return_value = mock_mc

        def _fake_post(url, **kwargs):
            if "/enqueue_bulk" in url:
                n = len(kwargs.get("json", {}).get("targets", []))
                r = MagicMock()
                r.status_code = 200
                r.json.return_value = {"enqueued": n, "skipped": 0, "failed": 0}
                r.raise_for_status = MagicMock()
                return r
            # probe call (empty targets list)
            r = MagicMock()
            r.status_code = 200
            r.raise_for_status = MagicMock()
            return r

        mock_requests.post.side_effect = _fake_post

        targets = [{"tic_id": str(i)} for i in range(1, n_targets + 1)]
        result = enqueue_targets(targets, self.db, self.server_url,
                                 cache_path="/tmp/dummy.db")

        # Number of POST calls: 1 probe + ceil(n/chunk) bulk chunks
        expected_chunks = math.ceil(n_targets / _BULK_CHUNK_SIZE)
        total_calls = mock_requests.post.call_count
        # At minimum: not n_targets individual calls
        self.assertLess(
            total_calls, n_targets,
            f"Expected <{n_targets} calls (bulk path), got {total_calls}",
        )
        # Exactly 1 probe + expected_chunks bulk calls
        self.assertEqual(
            total_calls, 1 + expected_chunks,
            f"Expected {1 + expected_chunks} calls (probe + {expected_chunks} bulk chunks), "
            f"got {total_calls}",
        )
        self.assertEqual(result["enqueued"], n_targets)
        self.assertEqual(result["failed"], 0)

    @patch("exonit.enqueue.ensure_metadata", return_value=True)
    @patch("exonit.enqueue.MetadataCache")
    @patch("exonit.enqueue.requests")
    def test_fallback_to_per_target_when_bulk_404(self, mock_requests, mock_mc_cls, _ens):
        """When /enqueue_bulk returns 404, falls back to per-target POSTs."""
        from exonit.enqueue import enqueue_targets

        n_targets = 3
        mock_mc = MagicMock()
        mock_mc.all_tic_ids.return_value = {str(i) for i in range(1, n_targets + 1)}
        mock_mc_cls.return_value = mock_mc

        def _fake_post(url, **kwargs):
            if "/enqueue_bulk" in url:
                r = MagicMock()
                r.status_code = 404
                r.raise_for_status = MagicMock()
                return r
            # per-target /enqueue call
            r = MagicMock()
            r.status_code = 200
            r.raise_for_status = MagicMock()
            return r

        mock_requests.post.side_effect = _fake_post

        targets = [{"tic_id": str(i)} for i in range(1, n_targets + 1)]
        result = enqueue_targets(targets, self.db, self.server_url,
                                 cache_path="/tmp/dummy.db")

        # 1 probe + n_targets per-target calls
        total_calls = mock_requests.post.call_count
        self.assertEqual(total_calls, 1 + n_targets)
        self.assertEqual(result["enqueued"], n_targets)


class TestEnqueuePrefilterBulkSets(unittest.TestCase):
    """Tests that verify the O(1) pre-filter behaviour using real SQLite fixtures.

    All DBs are created under tmp_path — never data/exonit.db or
    data/metadata_cache.db.  Tests verify:
      - already-analyzed TIC → skipped (analyzed_set lookup)
      - TIC in metadata but not analyzed → ready (metadata_set lookup)
      - TIC not in metadata, indexable → ready (ensure_metadata called once)
      - TIC not in metadata, not indexable → no_fits + skipped
      - MetadataCache is constructed a FIXED number of times (not O(N))
    """

    def setUp(self):
        self.db = MagicMock()
        self.server_url = "http://localhost:8400"

    def _make_mc_with_tics(self, tmp_path, tic_ids):
        """Create a real MetadataCache under tmp_path with given TIC IDs."""
        import json
        from exonit.metadata_cache import MetadataCache
        db_path = str(tmp_path / "metadata_cache.db")
        mc = MetadataCache(db_path)
        for tic in tic_ids:
            mc.add_entry(
                tic_id=str(tic),
                sectors=[1],
                exptime=120.0,
                fits_files=["dummy.fits"],
            )
        return mc, db_path

    def test_already_analyzed_skipped(self, tmp_path=None):
        """TIC in analyzed_set → skipped immediately, no metadata lookup."""
        import tempfile, os
        tmp = tempfile.mkdtemp()

        from exonit.enqueue import enqueue_targets

        # Mark TIC 100 as already completed.
        self.db.completed_tic_ids.return_value = {"100"}

        with patch("exonit.enqueue.MetadataCache") as mock_mc_cls, \
             patch("exonit.enqueue.requests") as mock_req:
            mock_mc = MagicMock()
            mock_mc.all_tic_ids.return_value = set()
            mock_mc_cls.return_value = mock_mc

            result = enqueue_targets(
                [{"tic_id": "100"}], self.db, self.server_url,
                cache_path=os.path.join(tmp, "mc.db"),
            )

        self.assertEqual(result["skipped"], 1)
        self.assertEqual(result["enqueued"], 0)
        # metadata_set was built (1 MC open), but no ensure_metadata call
        mock_mc.all_tic_ids.assert_called_once()

    def test_has_metadata_not_analyzed_ready(self):
        """TIC in metadata_set but not in analyzed_set → goes to ready_tics."""
        import tempfile, os
        tmp = tempfile.mkdtemp()

        from exonit.enqueue import enqueue_targets

        self.db.completed_tic_ids.return_value = set()  # not analyzed

        with patch("exonit.enqueue.MetadataCache") as mock_mc_cls, \
             patch("exonit.enqueue.requests") as mock_req:
            mock_mc = MagicMock()
            mock_mc.all_tic_ids.return_value = {"200"}  # TIC 200 in metadata
            mock_mc_cls.return_value = mock_mc

            # Probe returns 200, bulk returns 1 enqueued
            def _fake_post(url, **kwargs):
                r = MagicMock()
                r.status_code = 200
                r.json.return_value = {"enqueued": 1, "skipped": 0, "failed": 0}
                r.raise_for_status = MagicMock()
                return r
            mock_req.post.side_effect = _fake_post

            result = enqueue_targets(
                [{"tic_id": "200"}], self.db, self.server_url,
                cache_path=os.path.join(tmp, "mc.db"),
            )

        self.assertEqual(result["enqueued"], 1)
        self.assertEqual(result["skipped"], 0)

    def test_no_metadata_indexable_ready(self):
        """TIC not in metadata_set but ensure_metadata succeeds → ready_tics."""
        import tempfile, os
        tmp = tempfile.mkdtemp()

        from exonit.enqueue import enqueue_targets

        self.db.completed_tic_ids.return_value = set()

        with patch("exonit.enqueue.MetadataCache") as mock_mc_cls, \
             patch("exonit.enqueue.ensure_metadata", return_value=True) as mock_ens, \
             patch("exonit.enqueue.requests") as mock_req:
            mock_mc = MagicMock()
            mock_mc.all_tic_ids.return_value = set()  # TIC 300 NOT in metadata set
            mock_mc_cls.return_value = mock_mc

            def _fake_post(url, **kwargs):
                r = MagicMock()
                r.status_code = 200
                r.json.return_value = {"enqueued": 1, "skipped": 0, "failed": 0}
                r.raise_for_status = MagicMock()
                return r
            mock_req.post.side_effect = _fake_post

            result = enqueue_targets(
                [{"tic_id": "300"}], self.db, self.server_url,
                cache_path=os.path.join(tmp, "mc.db"),
            )

        self.assertEqual(result["enqueued"], 1)
        self.assertEqual(result["skipped"], 0)
        # ensure_metadata was called for this TIC
        mock_ens.assert_called_once()
        call_kwargs = mock_ens.call_args
        self.assertEqual(str(call_kwargs[0][0]), "300")

    def test_no_metadata_not_indexable_skipped(self):
        """TIC not in metadata and ensure_metadata returns False → no_fits + skipped."""
        import tempfile, os
        tmp = tempfile.mkdtemp()

        from exonit.enqueue import enqueue_targets

        self.db.completed_tic_ids.return_value = set()

        with patch("exonit.enqueue.MetadataCache") as mock_mc_cls, \
             patch("exonit.enqueue.ensure_metadata", return_value=False), \
             patch("exonit.enqueue.requests") as mock_req:
            mock_mc = MagicMock()
            mock_mc.all_tic_ids.return_value = set()
            mock_mc_cls.return_value = mock_mc

            result = enqueue_targets(
                [{"tic_id": "400"}], self.db, self.server_url,
                cache_path=os.path.join(tmp, "mc.db"),
            )

        self.assertEqual(result["skipped"], 1)
        self.assertEqual(result["enqueued"], 0)
        mock_req.post.assert_not_called()
        # DB must be updated with no_fits — Q9 (perf review 2026-09-26): one
        # batched add_stars_bulk() call, not one add_star() per TIC.
        self.db.add_stars_bulk.assert_called_once()
        call_args, kwargs = self.db.add_stars_bulk.call_args
        self.assertEqual(list(call_args[0]), ["400"])
        self.assertEqual(kwargs.get("status"), "no_fits")

    def test_metacache_constructed_fixed_times_not_o_n(self):
        """MetadataCache is instantiated exactly ONCE regardless of target count.

        This is the core performance invariant: the old code did 2 opens/target,
        the new code does 1 open total.
        """
        import tempfile, os
        tmp = tempfile.mkdtemp()
        n = 50  # 50 targets, all with metadata already

        from exonit.enqueue import enqueue_targets

        self.db.completed_tic_ids.return_value = set()
        tic_ids = [str(i) for i in range(1, n + 1)]

        mc_open_count = [0]
        original_init_called = []

        with patch("exonit.enqueue.MetadataCache") as mock_mc_cls, \
             patch("exonit.enqueue.requests") as mock_req:
            def _mc_factory(path):
                mc_open_count[0] += 1
                m = MagicMock()
                m.all_tic_ids.return_value = set(tic_ids)
                return m
            mock_mc_cls.side_effect = _mc_factory

            def _fake_post(url, **kwargs):
                r = MagicMock()
                r.status_code = 200
                r.json.return_value = {
                    "enqueued": len(kwargs.get("json", {}).get("targets", [])),
                    "skipped": 0, "failed": 0,
                }
                r.raise_for_status = MagicMock()
                return r
            mock_req.post.side_effect = _fake_post

            targets = [{"tic_id": t} for t in tic_ids]
            result = enqueue_targets(targets, self.db, self.server_url,
                                     cache_path=os.path.join(tmp, "mc.db"))

        # MetadataCache must be opened exactly 1 time (not N=50 times)
        self.assertEqual(
            mc_open_count[0], 1,
            f"MetadataCache was opened {mc_open_count[0]} times for {n} targets "
            f"(expected 1 — must not be O(N))",
        )
        self.assertEqual(result["enqueued"], n)


class TestEnqueueTargetsV2:
    """``EXONIT_ENQUEUE_METADATA_V2=1`` path (docs/superpowers/specs/
    2026-09-26-metadata-refresh-and-bulk-enqueue-design.md). Real SQLite
    MetadataCache fixtures (never ``data/metadata_cache.db``), mocked
    ``db``/``_post_bulk`` (never a real ExoNitDB/HTTP call needed here)."""

    @staticmethod
    def _seed_tglc(mc_path, rows):
        from exonit.metadata_cache import MetadataCache
        mc = MetadataCache(mc_path)
        mc.ensure_tglc_table()
        mc.tglc_add_rows_bulk(rows)
        mc.close()

    def test_v2_off_by_default_routes_to_legacy(self, tmp_path, monkeypatch):
        from exonit import enqueue as enq

        calls = []
        monkeypatch.setattr(
            enq, "_enqueue_targets_v2",
            lambda *a, **k: calls.append("v2") or ([], [], 0, {}))
        monkeypatch.setattr(
            enq, "_enqueue_targets_legacy",
            lambda *a, **k: calls.append("legacy") or ([], [], 0, {}))
        enq.enqueue_targets([], MagicMock(), cache_path=str(tmp_path / "mc.db"))
        assert calls == ["legacy"]

    def test_v2_flag_on_routes_to_v2(self, tmp_path, monkeypatch):
        from exonit import enqueue as enq

        monkeypatch.setenv("EXONIT_ENQUEUE_METADATA_V2", "1")
        calls = []
        monkeypatch.setattr(
            enq, "_enqueue_targets_v2",
            lambda *a, **k: calls.append("v2") or ([], [], 0, {}))
        monkeypatch.setattr(
            enq, "_enqueue_targets_legacy",
            lambda *a, **k: calls.append("legacy") or ([], [], 0, {}))
        enq.enqueue_targets([], MagicMock(), cache_path=str(tmp_path / "mc.db"))
        assert calls == ["v2"]

    def test_v2_uses_scoped_completed_tic_ids_in_not_full_set(
            self, tmp_path, monkeypatch):
        """The V2 path must call db.completed_tic_ids_in (scoped to the lot),
        never db.completed_tic_ids() (the ~27M-row full load this spec
        removes from the hot path)."""
        from exonit import enqueue as enq

        monkeypatch.setenv("EXONIT_ENQUEUE_METADATA_V2", "1")
        mc_path = str(tmp_path / "mc.db")
        self._seed_tglc(mc_path, [
            ("111", 1, "/x/hlsp_tglc-a-s0001-1.fits", 1800.0, 9.0, 1.0, 2.0),
        ])
        db = MagicMock()
        db.completed_tic_ids_in.return_value = set()
        posted = []
        monkeypatch.setattr(
            enq, "_post_bulk",
            lambda tics, *a, **k: posted.append(list(tics)) or
            {"enqueued": len(tics), "skipped": 0, "failed": 0})

        result = enq.enqueue_targets(
            [{"tic_id": "111"}], db, cache_path=mc_path)

        db.completed_tic_ids_in.assert_called_once()
        db.completed_tic_ids.assert_not_called()
        assert result["enqueued"] == 1
        assert result["metadata_created"] == 1
        assert posted == [["111"]]

    def test_v2_refresh_off_by_default_leaves_stale_row(
            self, tmp_path, monkeypatch):
        from exonit import enqueue as enq

        monkeypatch.setenv("EXONIT_ENQUEUE_METADATA_V2", "1")
        mc_path = str(tmp_path / "mc.db")
        self._seed_tglc(mc_path, [
            ("222", 1, "/x/hlsp_tglc-a-s0001-1.fits", 1800.0, 9.0, 1.0, 2.0),
        ])
        db = MagicMock()
        db.completed_tic_ids_in.return_value = set()
        monkeypatch.setattr(
            enq, "_post_bulk",
            lambda tics, *a, **k: {"enqueued": len(tics), "skipped": 0, "failed": 0})

        result1 = enq.enqueue_targets(
            [{"tic_id": "222"}], db, cache_path=mc_path)
        assert result1["metadata_created"] == 1

        # A new sector appears in tglc, but EXONIT_METADATA_REFRESH is off.
        self._seed_tglc(mc_path, [
            ("222", 2, "/x/hlsp_tglc-a-s0002-1.fits", 1800.0, 9.0, 1.0, 2.0),
        ])
        result2 = enq.enqueue_targets(
            [{"tic_id": "222"}], db, cache_path=mc_path)
        assert result2["metadata_refreshed"] == 0
        assert result2["metadata_unchanged"] == 1

    def test_v2_refresh_on_refreshes_stale_row(self, tmp_path, monkeypatch):
        from exonit import enqueue as enq

        monkeypatch.setenv("EXONIT_ENQUEUE_METADATA_V2", "1")
        monkeypatch.setenv("EXONIT_METADATA_REFRESH", "1")
        mc_path = str(tmp_path / "mc.db")
        self._seed_tglc(mc_path, [
            ("333", 1, "/x/hlsp_tglc-a-s0001-1.fits", 1800.0, 9.0, 1.0, 2.0),
        ])
        db = MagicMock()
        db.completed_tic_ids_in.return_value = set()
        monkeypatch.setattr(
            enq, "_post_bulk",
            lambda tics, *a, **k: {"enqueued": len(tics), "skipped": 0, "failed": 0})

        enq.enqueue_targets([{"tic_id": "333"}], db, cache_path=mc_path)
        self._seed_tglc(mc_path, [
            ("333", 2, "/x/hlsp_tglc-a-s0002-1.fits", 1800.0, 9.0, 1.0, 2.0),
        ])
        result2 = enq.enqueue_targets(
            [{"tic_id": "333"}], db, cache_path=mc_path)
        assert result2["metadata_refreshed"] == 1

    def test_v2_toi_row_never_refreshed(self, tmp_path, monkeypatch):
        from exonit import enqueue as enq
        from exonit.metadata_cache import MetadataCache

        monkeypatch.setenv("EXONIT_ENQUEUE_METADATA_V2", "1")
        monkeypatch.setenv("EXONIT_METADATA_REFRESH", "1")
        mc_path = str(tmp_path / "mc.db")
        mc = MetadataCache(mc_path)
        mc.add_entry(tic_id="444", sectors=[9], exptime=120.0,
                    fits_files=["tess2020-s0009-0000000000000444-0123-s_lc.fits"])
        mc.close()
        self._seed_tglc(mc_path, [
            ("444", 9, "/x/hlsp_tglc-a-s0009-1.fits", 1800.0, 9.0, 1.0, 2.0),
        ])
        db = MagicMock()
        db.completed_tic_ids_in.return_value = set()
        monkeypatch.setattr(
            enq, "_post_bulk",
            lambda tics, *a, **k: {"enqueued": len(tics), "skipped": 0, "failed": 0})

        result = enq.enqueue_targets(
            [{"tic_id": "444"}], db, cache_path=mc_path)
        assert result["metadata_preserved_non_tglc"] == 1
        assert result["metadata_refreshed"] == 0

    def test_v2_force_rescan_skips_analyzed_and_still_refreshes(
            self, tmp_path, monkeypatch):
        from exonit import enqueue as enq
        from exonit.metadata_cache import MetadataCache

        monkeypatch.setenv("EXONIT_ENQUEUE_METADATA_V2", "1")
        monkeypatch.setenv("EXONIT_METADATA_REFRESH", "1")
        mc_path = str(tmp_path / "mc.db")
        self._seed_tglc(mc_path, [
            ("555", 1, "/x/hlsp_tglc-a-s0001-1.fits", 1800.0, 9.0, 1.0, 2.0),
            ("555", 2, "/x/hlsp_tglc-a-s0002-1.fits", 1800.0, 9.0, 1.0, 2.0),
        ])
        mc = MetadataCache(mc_path)
        mc.add_entry(tic_id="555", sectors=[1], exptime=1800.0,
                    fits_files=["hlsp_tglc-a-s0001-1.fits"],
                    tmag=9.0, ra=1.0, dec=2.0)
        mc.close()

        db = MagicMock()
        db.completed_tic_ids_in.return_value = {"555"}  # already analyzed
        captured = {}

        def fake_bulk(tics, *a, **k):
            captured["tics"] = list(tics)
            return {"enqueued": len(tics), "skipped": 0, "failed": 0}
        monkeypatch.setattr(enq, "_post_bulk", fake_bulk)

        result = enq.enqueue_targets(
            [{"tic_id": "555"}], db, cache_path=mc_path, force_rescan=True)
        assert captured["tics"] == ["555"]
        assert result["metadata_refreshed"] == 1

    def test_v2_progress_reporter_reports_ensure_metadata_substep(
            self, tmp_path, monkeypatch):
        """ProgressReporter.report is invoked (>= 1 chunk) while
        ensure_metadata_bulk runs under the V2 path, with the
        'ensure_metadata' substep."""
        from exonit import enqueue as enq
        from exonit.progress_client import ProgressReporter

        monkeypatch.setenv("EXONIT_ENQUEUE_METADATA_V2", "1")
        mc_path = str(tmp_path / "mc.db")
        self._seed_tglc(mc_path, [
            ("666", 1, "/x/hlsp_tglc-a-s0001-1.fits", 1800.0, 9.0, 1.0, 2.0),
        ])
        db = MagicMock()
        db.completed_tic_ids_in.return_value = set()
        monkeypatch.setattr(
            enq, "_post_bulk",
            lambda tics, *a, **k: {"enqueued": len(tics), "skipped": 0, "failed": 0})

        reports = []

        def _spy_report(self, done, *, total=None, message=None,
                        substep=None, force=False):
            reports.append((done, substep or self.substep))
        monkeypatch.setattr(ProgressReporter, "report", _spy_report)

        enq.enqueue_targets([{"tic_id": "666"}], db, cache_path=mc_path)

        assert reports, "ProgressReporter.report was never called"
        assert any(s == "ensure_metadata" for _, s in reports)

    def test_v2_and_legacy_agree_on_counts_same_fixture(
            self, tmp_path, monkeypatch):
        """§4 risk #5: a survey lot with V2 on and refresh off must produce
        the same enqueued/skipped/failed counts as the legacy path — only
        faster, never different."""
        from exonit import enqueue as enq
        from exonit.metadata_cache import MetadataCache

        def _fresh_mc(path):
            mc = MetadataCache(path)
            mc.ensure_tglc_table()
            mc.tglc_add_rows_bulk([
                ("777", 1, "/x/hlsp_tglc-a-s0001-1.fits", 1800.0, 9.0, 1.0, 2.0),
            ])
            mc.close()

        def _fresh_db():
            db = MagicMock()
            db.completed_tic_ids.return_value = set()
            db.completed_tic_ids_in.return_value = set()
            return db

        mc_path_legacy = str(tmp_path / "legacy.db")
        mc_path_v2 = str(tmp_path / "v2.db")
        _fresh_mc(mc_path_legacy)
        _fresh_mc(mc_path_v2)

        monkeypatch.setattr(
            enq, "_post_bulk",
            lambda tics, *a, **k: {"enqueued": len(tics), "skipped": 0, "failed": 0})

        monkeypatch.delenv("EXONIT_ENQUEUE_METADATA_V2", raising=False)
        legacy_result = enq.enqueue_targets(
            [{"tic_id": "777"}], _fresh_db(), cache_path=mc_path_legacy)

        monkeypatch.setenv("EXONIT_ENQUEUE_METADATA_V2", "1")
        v2_result = enq.enqueue_targets(
            [{"tic_id": "777"}], _fresh_db(), cache_path=mc_path_v2)

        for key in ("enqueued", "skipped", "failed"):
            assert legacy_result[key] == v2_result[key], key


# --------------------------------------------------------------------------- #
# Q5 (perf review 2026-09-26): _post_bulk chunk-size + retry-with-backoff.
# --------------------------------------------------------------------------- #

class TestPostBulkRetry(unittest.TestCase):
    """Direct tests of exonit.enqueue._post_bulk (not via enqueue_targets)."""

    def setUp(self):
        self.server_url = "http://localhost:8400"
        self.token = "tok"

    def _probe_response(self):
        r = MagicMock()
        r.status_code = 200
        return r

    @staticmethod
    def _is_probe_call(kwargs) -> bool:
        """_probe_bulk_endpoint POSTs to the SAME URL as a real chunk, with
        an empty targets list -- distinguish by payload, not URL."""
        return len(kwargs.get("json", {}).get("targets", [])) == 0

    @patch("exonit.enqueue.time.sleep")
    @patch("exonit.enqueue.requests")
    def test_chunk_retries_and_succeeds_on_second_attempt(
        self, mock_requests, mock_sleep
    ):
        """A chunk that fails once then succeeds must be counted as
        enqueued, not failed — the whole point of Q5's retry-with-backoff."""
        from exonit.enqueue import _post_bulk

        ok = MagicMock()
        ok.status_code = 200
        ok.raise_for_status = MagicMock()
        ok.json.return_value = {"enqueued": 3, "skipped": 0, "failed": 0}

        call_log = []

        def _fake_post(url, **kwargs):
            if self._is_probe_call(kwargs):
                return self._probe_response()
            call_log.append(1)
            if len(call_log) == 1:
                raise ConnectionError("transient network blip")
            return ok

        mock_requests.post.side_effect = _fake_post

        result = _post_bulk(["1", "2", "3"], self.server_url, self.token)

        self.assertEqual(result, {"enqueued": 3, "skipped": 0, "failed": 0, "adopted": 0})
        self.assertEqual(len(call_log), 2, "expected exactly one retry")
        mock_sleep.assert_called_once()  # backoff slept once before the retry

    @patch("exonit.enqueue.time.sleep")
    @patch("exonit.enqueue.requests")
    def test_chunk_gives_up_after_max_retries(self, mock_requests, mock_sleep):
        """After exhausting retries, the chunk counts as failed (not lost
        silently, not retried forever)."""
        from exonit.enqueue import _post_bulk, _BULK_POST_MAX_RETRIES

        def _fake_post(url, **kwargs):
            if self._is_probe_call(kwargs):
                return self._probe_response()
            raise ConnectionError("server down")

        mock_requests.post.side_effect = _fake_post

        result = _post_bulk(["1", "2", "3"], self.server_url, self.token)

        self.assertEqual(result, {"enqueued": 0, "skipped": 0, "failed": 3, "adopted": 0})
        # 1 initial attempt + _BULK_POST_MAX_RETRIES retries, all failing.
        # (Probe and real chunk calls share the same URL -- distinguish by
        # payload, same as _is_probe_call.)
        bulk_calls = [c for c in mock_requests.post.call_args_list
                     if not self._is_probe_call(c.kwargs)]
        self.assertEqual(len(bulk_calls), 1 + _BULK_POST_MAX_RETRIES)

    @patch("exonit.enqueue.time.sleep")
    @patch("exonit.enqueue.requests")
    def test_chunk_succeeds_first_try_no_retry_no_sleep(
        self, mock_requests, mock_sleep
    ):
        """The common case (no failures) must not sleep or retry at all."""
        from exonit.enqueue import _post_bulk

        ok = MagicMock()
        ok.status_code = 200
        ok.raise_for_status = MagicMock()
        ok.json.return_value = {"enqueued": 2, "skipped": 0, "failed": 0}

        def _fake_post(url, **kwargs):
            if self._is_probe_call(kwargs):
                return self._probe_response()
            return ok

        mock_requests.post.side_effect = _fake_post

        result = _post_bulk(["1", "2"], self.server_url, self.token)

        self.assertEqual(result, {"enqueued": 2, "skipped": 0, "failed": 0, "adopted": 0})
        mock_sleep.assert_not_called()

    def test_bulk_chunk_size_is_20000(self):
        """Locks in the Q5 chunk-size bump (5000 -> 20000): a 300k-TIC lot
        now needs ~16 bulk POSTs instead of ~61."""
        from exonit.enqueue import _BULK_CHUNK_SIZE
        self.assertEqual(_BULK_CHUNK_SIZE, 20000)


if __name__ == "__main__":
    unittest.main()
