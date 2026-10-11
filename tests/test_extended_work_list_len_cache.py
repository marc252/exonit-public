"""Tests for exonit.server._extended_work_list_len's shared-state cache.

Q7 (perf review 2026-09-26, docs/reviews/2026-09-26-enqueue-path-
performance-review.md): the TTL+row-count cache moved from a plain
per-process dict to shared_state (Redis in prod, MemoryBackend in tests —
see tests/conftest.py's _isolate_shared_state autouse fixture) so all 4
uvicorn worker processes (A3) share one cached value instead of each
re-validating independently. These tests exercise the module function
directly, with count_extended_canonical / the FS/row-count signals mocked,
so no real canonical CSV or TGLC index is needed.
"""

from unittest.mock import patch

from exonit import server as srv


def _clear_cache():
    from exonit.shared_state import get_shared_state
    get_shared_state().hdel(
        srv._WORK_LIST_SIZE_HKEY,
        srv._work_list_cache_field("canon.csv", "tglc.db", 1))


class TestExtendedWorkListLenCache:
    def setup_method(self):
        _clear_cache()

    def test_cold_cache_computes_and_caches(self):
        with patch.object(srv, "_canonical_sig", return_value=(10, 100)), \
             patch.object(srv, "_tglc_sector_rowcount", return_value=5), \
             patch(
                 "exonit.discovery_state.count_extended_canonical",
                 return_value=42) as m_count:
            size = srv._extended_work_list_len("canon.csv", "tglc.db", sector=1)
        assert size == 42
        m_count.assert_called_once()

    def test_warm_cache_within_ttl_skips_recompute(self):
        with patch.object(srv, "_canonical_sig", return_value=(10, 100)), \
             patch.object(srv, "_tglc_sector_rowcount", return_value=5), \
             patch(
                 "exonit.discovery_state.count_extended_canonical",
                 return_value=42) as m_count:
            srv._extended_work_list_len("canon.csv", "tglc.db", sector=1)
            size2 = srv._extended_work_list_len("canon.csv", "tglc.db", sector=1)
        assert size2 == 42
        m_count.assert_called_once()  # second call must NOT recompute

    def test_ttl_expired_unchanged_signals_skips_full_recompute(self):
        """Past the TTL, if the canonical CSV signature and row count are
        both unchanged, the cache must refresh its TTL WITHOUT rebuilding
        the work list -- the whole point of the row-count re-validation."""
        with patch.object(srv, "_canonical_sig", return_value=(10, 100)), \
             patch.object(srv, "_tglc_sector_rowcount", return_value=5), \
             patch(
                 "exonit.discovery_state.count_extended_canonical",
                 return_value=42) as m_count, \
             patch("time.time", side_effect=[1000.0, 1000.0 + srv._WORK_LIST_TTL_S + 1]):
            size1 = srv._extended_work_list_len("canon.csv", "tglc.db", sector=1)
            size2 = srv._extended_work_list_len("canon.csv", "tglc.db", sector=1)
        assert size1 == size2 == 42
        m_count.assert_called_once()  # only the first (cold) call recomputed

    def test_ttl_expired_row_count_changed_triggers_recompute(self):
        """A genuine row-count change past the TTL must trigger a full
        recompute -- the cache must never serve a stale size.

        Rowcount is checked twice on the second call: once for the cheap
        TTL-revalidation comparison (returns 9, differs from the cached 5),
        and again inside the "miss or genuine change" full-recompute branch
        below it (same pre-existing double-check as the old per-process
        dict version of this cache) -- 3 rowcount calls total across the
        two _extended_work_list_len invocations.
        """
        with patch.object(srv, "_canonical_sig", return_value=(10, 100)), \
             patch.object(srv, "_tglc_sector_rowcount",
                          side_effect=[5, 9, 9]), \
             patch(
                 "exonit.discovery_state.count_extended_canonical",
                 side_effect=[42, 99]) as m_count, \
             patch("time.time", side_effect=[1000.0, 1000.0 + srv._WORK_LIST_TTL_S + 1]):
            size1 = srv._extended_work_list_len("canon.csv", "tglc.db", sector=1)
            size2 = srv._extended_work_list_len("canon.csv", "tglc.db", sector=1)
        assert size1 == 42
        assert size2 == 99
        assert m_count.call_count == 2

    def test_shared_across_get_shared_state_calls_not_per_process_dict(self):
        """The whole point of Q7: two independent lookups of get_shared_state()
        (standing in for two different uvicorn worker processes, which each
        call get_shared_state() to reach the SAME Redis in prod) must see the
        same cached value -- a per-process dict could never do this."""
        from exonit.shared_state import get_shared_state

        with patch.object(srv, "_canonical_sig", return_value=(10, 100)), \
             patch.object(srv, "_tglc_sector_rowcount", return_value=5), \
             patch(
                 "exonit.discovery_state.count_extended_canonical",
                 return_value=7):
            srv._extended_work_list_len("canon.csv", "tglc.db", sector=1)

        field = srv._work_list_cache_field("canon.csv", "tglc.db", 1)
        cached = get_shared_state().hget_json(srv._WORK_LIST_SIZE_HKEY, field)
        assert cached is not None
        assert cached["size"] == 7

    def test_shared_state_read_failure_falls_back_to_recompute(self):
        """A shared_state error (e.g. Redis down) must fail OPEN to a full
        recompute, never raise and never silently serve a wrong size."""
        from exonit.shared_state import get_shared_state

        with patch.object(get_shared_state(), "hget_json",
                          side_effect=RuntimeError("redis down")), \
             patch.object(srv, "_canonical_sig", return_value=(10, 100)), \
             patch.object(srv, "_tglc_sector_rowcount", return_value=5), \
             patch(
                 "exonit.discovery_state.count_extended_canonical",
                 return_value=13) as m_count:
            size = srv._extended_work_list_len("canon.csv", "tglc.db", sector=1)
        assert size == 13
        m_count.assert_called_once()

    def test_shared_state_write_failure_does_not_raise(self):
        """A cache-write failure is an optimisation loss, not a 500."""
        from exonit.shared_state import get_shared_state

        with patch.object(get_shared_state(), "hset_json",
                          side_effect=RuntimeError("redis down")), \
             patch.object(srv, "_canonical_sig", return_value=(10, 100)), \
             patch.object(srv, "_tglc_sector_rowcount", return_value=5), \
             patch(
                 "exonit.discovery_state.count_extended_canonical",
                 return_value=21):
            size = srv._extended_work_list_len("canon.csv", "tglc.db", sector=1)
        assert size == 21

    def test_different_sectors_do_not_collide(self):
        with patch.object(srv, "_canonical_sig", return_value=(10, 100)), \
             patch.object(srv, "_tglc_sector_rowcount", return_value=5), \
             patch(
                 "exonit.discovery_state.count_extended_canonical",
                 side_effect=[11, 22]):
            size_s1 = srv._extended_work_list_len("canon.csv", "tglc.db", sector=1)
            size_s2 = srv._extended_work_list_len("canon.csv", "tglc.db", sector=2)
        assert size_s1 == 11
        assert size_s2 == 22
