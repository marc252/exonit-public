"""Parity tests for the columnar population variant of find_systematic_periods
(2026-08-01, postprocess-streaming spec F2,
docs/superpowers/specs/2026-08-01-postprocess-streaming.md §3.B/§3.D).

These tests are purely additive — no existing test in test_systematic_veto.py
or test_systematic_families_io.py is modified. The contract under test:

  find_systematic_periods_columnar(ColumnarPopulation.from_rows(rows), **kw)
      == find_systematic_periods(rows, **kw)

for the SAME `rows` list, in the SAME order, with `==` on the full list of
dicts (order included) — a REQUISIT DUR per the spec, not a "close enough"
check. Also covers `ColumnarPopulation.from_rows` (casting, error path,
order) and `find_deep_eclipse_hosts` accepting an arbitrary iterable (§3.D).
"""
import math

import numpy as np
import pytest

from exonit.systematic_veto import (
    ColumnarPopulation,
    _COMMON_PERIOD_REL_TOL,
    find_deep_eclipse_hosts,
    find_systematic_periods,
    find_systematic_periods_columnar,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_cand(tic_id, period, t0):
    """Same shape find_systematic_periods expects. tic_id MUST be numeric
    here (unlike test_systematic_veto.py's helper) — ColumnarPopulation
    requires int-castable tic_id (the DB reality this module targets)."""
    return {"tic_id": str(tic_id), "period_days": period, "t0": t0}


def _assert_parity(rows, **kwargs):
    """Run both implementations on *rows* and assert bit-identical output,
    order included."""
    pop = ColumnarPopulation.from_rows(rows)
    old = find_systematic_periods(rows, **kwargs)
    new = find_systematic_periods_columnar(pop, **kwargs)
    assert new == old, f"columnar parity failure\n old={old}\n new={new}"
    return old


# ---------------------------------------------------------------------------
# 1. Epoch-clustered path parity
# ---------------------------------------------------------------------------

class TestEpochClusteredParity:
    def test_qualified_bin_clustered_epoch(self):
        """A period with clustered epoch across >= min_stars TICs: qualifies."""
        period = 0.71
        epoch = 0.3
        rows = [_make_cand(i, period, epoch + (i % 3) * 0.005) for i in range(50)]
        old = _assert_parity(rows, min_stars=10, min_concentration=0.5)
        assert len(old) >= 1

    def test_qualified_bin_but_dispersed_epoch_not_flagged(self):
        """>= min_stars TICs in the bin but uniform epoch (R < min_concentration):
        the bin is visited (grouping happens) but no entry is emitted — this
        exercises the "qualified count, failed concentration" branch."""
        period = 2.5
        rows = [_make_cand(i, period, (i / 50.0) * period) for i in range(50)]
        old = _assert_parity(rows, min_stars=10, min_concentration=0.5)
        assert old == []

    def test_bin_below_min_stars_not_flagged(self):
        """Fewer than min_stars distinct TICs in the bin: skipped before the
        epoch/concentration computation is ever reached."""
        rows = [_make_cand(i, 0.71, 0.3) for i in range(8)]
        old = _assert_parity(rows, min_stars=10, min_concentration=0.5)
        assert old == []

    def test_multiple_bins_mixed_qualified_unqualified(self):
        """A mix of a qualifying bin, a below-threshold bin, and a
        low-concentration bin in the SAME population — exercises by_bin
        iteration/insertion order across several distinct keys."""
        rows = (
            [_make_cand(i, 0.71, 0.3 + 0.002 * i) for i in range(40)]              # qualifies
            + [_make_cand(1000 + i, 5.0, 0.1) for i in range(3)]                   # too few TICs
            + [_make_cand(2000 + i, 12.3, (i / 30.0) * 12.3) for i in range(30)]   # dispersed epoch
        )
        old = _assert_parity(rows, min_stars=10, min_concentration=0.5)
        assert len(old) == 1
        assert abs(old[0]["period"] - 0.71) < 0.05

    def test_bin_boundary_tie_values(self):
        """Periods sitting exactly at half-bin boundaries (banker's-rounding
        tie cases for round(round(p/period_bin)*period_bin, 3)) — the whole
        point of the columnar path's per-row Python round() fallback."""
        rows = []
        # 0.125 is exactly half of period_bin=0.25 -> round(0.5) ties to 0
        # under Python banker's rounding (round-half-to-even).
        boundary_periods = [0.125, 0.375, 0.625, 0.875, 1.125, 1.375]
        for g, bp in enumerate(boundary_periods):
            for i in range(15):
                rows.append(_make_cand(g * 100 + i, bp, 0.05 * i))
        old = _assert_parity(rows, min_stars=10, min_concentration=0.0)


# ---------------------------------------------------------------------------
# 2. Common-period path (P < 2 d) parity
# ---------------------------------------------------------------------------

class TestCommonPeriodParity:
    def _dispersed(self, period, n_tics, t0_spread=0.5):
        return [_make_cand(i, period, i * (t0_spread / n_tics)) for i in range(n_tics)]

    def test_common_period_detects_dispersed_short_period(self):
        rows = self._dispersed(0.71, 20)
        old = _assert_parity(rows, min_stars=10, min_concentration=0.5,
                              common_period_min_tics=10)
        assert any(abs(s["period"] - 0.71) < 0.05 for s in old)

    def test_common_period_below_threshold(self):
        rows = self._dispersed(0.71, 9)
        old = _assert_parity(rows, min_stars=10, min_concentration=0.5,
                              common_period_min_tics=10)
        assert old == []

    def test_common_period_two_distinct_short_clusters(self):
        rows = self._dispersed(0.71, 58) + self._dispersed(0.75, 12, t0_spread=0.8)
        old = _assert_parity(rows, min_stars=10, min_concentration=0.5,
                              common_period_min_tics=10)
        periods = [s["period"] for s in old]
        assert any(abs(p - 0.71) < 0.05 for p in periods)
        assert any(abs(p - 0.75) < 0.05 for p in periods)

    def test_common_period_dedup_with_epoch_clustered(self):
        """Same period detected by both paths -> single entry, epoch-clustered
        wins (epoch_agnostic=False) — exercises the `already` dedup check."""
        period = 0.71
        epoch = 0.3
        rows = [_make_cand(i, period, epoch + 0.001 * i) for i in range(30)]
        old = _assert_parity(rows, min_stars=10, min_concentration=0.5,
                              common_period_min_tics=10)
        hits = [s for s in old if abs(s["period"] - period) < 0.05]
        assert len(hits) == 1
        assert hits[0]["epoch_agnostic"] is False

    def test_common_period_ge_max_period_excluded(self):
        """A dispersed-epoch period >= common_period_max_period (default 2.0)
        must NOT be flagged by the common-period path (only epoch-clustered
        can catch it, and it won't here since epoch is uniform). t0_spread
        must be a large fraction of the period for the epoch to actually be
        dispersed (unlike the short-period tests above, spread=0.5 would be
        too small relative to P=2.5 and would accidentally epoch-cluster)."""
        rows = self._dispersed(2.5, 20, t0_spread=2.5)
        old = _assert_parity(rows, min_stars=10, min_concentration=0.5,
                              common_period_min_tics=10)
        assert old == []


class TestCommonPeriodBoundary:
    """The pre-filter margin (common_period_max_period * (1 + rel_tol)) used
    by find_systematic_periods_columnar, instead of a naive
    `< common_period_max_period`, is exercised here: a cluster whose SEED is
    < 2.0 but whose tail member sits just above 2.0 must still count that
    tail member in n_stars — a naive filter would silently drop it and
    change the result (or make it miss the min_tics threshold entirely)."""

    def test_cluster_straddling_the_2d_boundary(self):
        """Regression-fixed 2026-08-01 (review CHANGES_REQUESTED, Finding 1):
        the original fixture used t0 = 0.1*i, which happened to be clustered
        enough (R=0.568 >= the default min_concentration=0.5) that the
        EPOCH-CLUSTERED path alone produced this test's expected result
        (epoch_agnostic=False) — the common-period pre-filter margin under
        test was never actually exercised; the test passed identically even
        with the margin reverted to the naive (unsafe) `< common_period_max_
        period` filter. Fixed by dispersing t0 relative to each row's own
        period (`(i/len(periods))*p`, spanning close to a full period across
        the group) and raising min_concentration to 0.9 so the epoch-
        clustered arm cannot independently qualify the same bin — isolating
        the common-period path (epoch_agnostic=True) as the only path that
        can produce a hit here.

        Verified by mutation (per review instructions): temporarily reverted
        the implementation's margin to the naive `common_period_max_period`
        (no `*(1+rel_tol)`) and re-ran this exact fixture directly against
        both find_systematic_periods and find_systematic_periods_columnar —
        old stayed `[{'period': 1.985, 'n_stars': 11, 'epoch_agnostic':
        True}]`, new (naive/mutated) became `[]`: the test genuinely fails
        under the naive filter. Restored the fix and re-confirmed the test
        passes. See progress/impl-f2-systematic-veto-columnar.md for the
        transcript.
        """
        rows = []
        # seed 1.970 ... climbs to 1.997, then a straddling member at 2.005
        # (all within 2% of the 1.970 seed: (2.005-1.970)/1.970 = 0.0178 < 0.02)
        periods = [1.970 + 0.003 * k for k in range(10)] + [2.005]
        assert len(periods) == 11
        n = len(periods)
        for i, p in enumerate(periods):
            # Dispersed epoch (spans ~the full period across the group) so
            # the epoch-clustered path cannot also independently qualify.
            rows.append(_make_cand(i, p, (i / n) * p))
        old = _assert_parity(rows, min_stars=10, min_concentration=0.9,
                              common_period_min_tics=11)
        hits = [s for s in old if abs(s["period"] - 1.985) < 0.05]
        assert len(hits) == 1, f"Expected the straddling cluster to qualify; got {old}"
        assert hits[0]["n_stars"] == 11, (
            f"n_stars must include the straddling member at P=2.005; got {hits[0]}"
        )
        assert hits[0]["epoch_agnostic"] is True, (
            "This test must be isolating the common-period path, not the "
            f"epoch-clustered one; got {hits[0]}"
        )

    def test_cluster_entirely_past_threshold_excluded(self):
        """Sanity: a cluster whose seed is already >= common_period_max_period
        is correctly excluded (never even considered a false-positive risk)."""
        rows = [_make_cand(i, 2.01 + 0.001 * i, 0.1 * i) for i in range(15)]
        old = _assert_parity(rows, min_stars=10, min_concentration=0.5,
                              common_period_min_tics=10)
        assert old == []


class TestManyDuplicatePeriodsStableTiebreak:
    """F2.2 (2026-08-01): the common-period path was rewritten to cluster
    the short-period subset with numpy (stable argsort + linear scan) instead
    of projecting it to a list of dicts and calling _cluster_by_relative_bin.
    Exact/near-exact period ties are the #1 risk of that rewrite (stable-sort
    tie-break must reproduce the same insertion/cluster order as Python's
    stable list.sort on the dict-projection path) — these tests stress that
    directly, at a scale (thousands of rows, exact float duplicates) the
    other fixtures in this file don't exercise."""

    def test_large_cluster_all_identical_short_period(self):
        """Thousands of EXACT duplicate periods -> a single cluster; stresses
        the stable argsort with a huge run of tied sort keys."""
        rows = [_make_cand(i, 0.71, 0.001 * (i % 100)) for i in range(3000)]
        old = _assert_parity(rows, min_stars=1000, min_concentration=1.0,
                              common_period_min_tics=50)
        hits = [s for s in old if abs(s["period"] - 0.71) < 1e-9]
        assert len(hits) == 1
        assert hits[0]["n_stars"] == 3000

    def test_two_equal_nstars_short_clusters_tiebreak_order(self):
        """Two distinct short-period clusters with the SAME n_stars,
        interleaved in population order -> exercises the final
        ``out.sort(key=lambda s: -s['n_stars'])`` stable tie-break, which
        depends on the order clusters were formed/appended in (ascending
        seed/period order for the common-period path)."""
        rows = []
        for i in range(20):
            rows.append(_make_cand(i, 0.51, 0.001 * i))          # cluster A
            rows.append(_make_cand(1000 + i, 0.90, 0.002 * i))   # cluster B
        old = _assert_parity(rows, min_stars=1000, min_concentration=1.0,
                              common_period_min_tics=20)
        periods_out = [s["period"] for s in old]
        assert len(periods_out) == 2
        assert all(s["n_stars"] == 20 for s in old)
        # A (period 0.51, smaller) precedes B (0.90) in ascending-seed
        # cluster-formation order -> stable tie-break keeps A first.
        assert abs(periods_out[0] - 0.51) < 0.01
        assert abs(periods_out[1] - 0.90) < 0.01

    def test_duplicate_periods_straddling_cluster_boundary(self):
        """A block of exact-duplicate periods sitting right at the ±2%
        relative-tolerance boundary of another block of exact duplicates —
        many ties immediately adjacent to the one comparison
        ((p - cl_seed) / cl_seed <= rel_tol) that decides cluster membership."""
        seed_p = 1.00
        boundary_p = seed_p * (1.0 + _COMMON_PERIOD_REL_TOL)  # exactly at tol
        rows = [_make_cand(i, seed_p, 0.1) for i in range(15)]
        rows += [_make_cand(100 + i, boundary_p, 0.1) for i in range(15)]
        old = _assert_parity(rows, min_stars=1000, min_concentration=1.0,
                              common_period_min_tics=10)
        # Whatever the boundary comparison resolves to (inclusive or not,
        # a pure floating-point question independent of this rewrite), the
        # two implementations MUST agree bit-for-bit -- _assert_parity above
        # is the actual gate; this just pins the concrete outcome so a
        # future silent divergence in the boundary arithmetic is caught too.
        assert sum(s["n_stars"] for s in old) == 30

    def test_many_duplicate_periods_two_clusters_dict_key_no_collision(self):
        """Many exact duplicates in TWO adjacent-but-distinct clusters (seeds
        >2% apart) whose rounded medians must NOT collide in the
        key_to_range dict -- both clusters must survive as separate output
        entries."""
        rows = [_make_cand(i, 0.71, 0.001 * i) for i in range(200)]
        rows += [_make_cand(1000 + i, 1.68, 0.002 * i) for i in range(200)]
        old = _assert_parity(rows, min_stars=1000, min_concentration=1.0,
                              common_period_min_tics=50)
        periods_out = sorted(s["period"] for s in old)
        assert len(periods_out) == 2
        assert abs(periods_out[0] - 0.71) < 0.01
        assert abs(periods_out[1] - 1.68) < 0.01


# ---------------------------------------------------------------------------
# 3. t0 None/NaN handling
# ---------------------------------------------------------------------------

class TestT0NoneParity:
    def test_mixed_none_and_present_t0(self):
        """Some rows have t0=None (encoded as NaN columnar-side) mixed with
        rows that have a real t0 in the SAME bin."""
        period = 0.71
        epoch = 0.3
        rows = [_make_cand(i, period, epoch + 0.002 * i) for i in range(10)]
        rows += [_make_cand(100 + i, period, None) for i in range(10)]
        old = _assert_parity(rows, min_stars=10, min_concentration=0.0)

    def test_all_t0_none_bin_never_reaches_min_stars_phase_check(self):
        """If every t0 in a qualifying bin is None, len(phases) < min_stars
        and the bin is skipped (phases list ends up empty)."""
        rows = [_make_cand(i, 0.71, None) for i in range(20)]
        old = _assert_parity(rows, min_stars=10, min_concentration=0.5)
        assert old == []


# ---------------------------------------------------------------------------
# 4. Empty / degenerate populations
# ---------------------------------------------------------------------------

class TestDegenerateParity:
    def test_empty_population(self):
        old = _assert_parity([], min_stars=10, min_concentration=0.5,
                              common_period_min_tics=10)
        assert old == []

    def test_no_bin_qualifies(self):
        """Every TIC has a distinct/unique period -> no bin ever reaches
        min_stars."""
        rows = [_make_cand(i, 1.0 + i * 0.5, 0.1) for i in range(30)]
        old = _assert_parity(rows, min_stars=10, min_concentration=0.5,
                              common_period_min_tics=10)
        assert old == []

    def test_invalid_periods_skipped(self):
        """None-period, negative, zero and NaN rows are all silently
        skipped by both implementations."""
        rows = [
            _make_cand(1, None, 0.1),
            _make_cand(2, -1.0, 0.1),
            _make_cand(3, 0.0, 0.1),
            _make_cand(4, float("nan"), 0.1),
        ] + [_make_cand(100 + i, 0.71, 0.3 + 0.001 * i) for i in range(15)]
        old = _assert_parity(rows, min_stars=10, min_concentration=0.5)
        assert len(old) == 1


# ---------------------------------------------------------------------------
# 5. Repeated TICs at different periods
# ---------------------------------------------------------------------------

class TestRepeatedTicsParity:
    def test_same_tic_different_periods_counted_once_per_bin(self):
        """The same TIC appearing at two clearly different periods
        contributes to TWO distinct bins' distinct-TIC counts, each counted
        once — mirrors a real multi-signal star."""
        rows = []
        for i in range(20):
            rows.append(_make_cand(i, 0.71, 0.3 + 0.001 * i))   # bin A
            rows.append(_make_cand(i, 5.13, 0.2 + 0.001 * i))   # bin B (same TICs)
        old = _assert_parity(rows, min_stars=10, min_concentration=0.5)
        assert len(old) == 2

    def test_same_tic_same_bin_does_not_double_count(self):
        """The same TIC appearing TWICE in the SAME bin (two signals at
        near-identical period) must still count as ONE distinct TIC — this
        is why {c["tic_id"] for c in grp} (a set) is used, not len(grp).
        16 rows, TIC 0 duplicated -> 15 distinct TICs (0..14), not 16."""
        rows = [_make_cand(0, 0.71, 0.30), _make_cand(0, 0.712, 0.301)]
        rows += [_make_cand(i, 0.71, 0.3 + 0.001 * i) for i in range(1, 15)]
        old = _assert_parity(rows, min_stars=10, min_concentration=0.0)
        assert len(old) == 1
        assert old[0]["n_stars"] == 15  # 15 distinct TICs, not 16 rows


# ---------------------------------------------------------------------------
# 6. Large random population — catch-all for rounding edge cases
# ---------------------------------------------------------------------------

class TestLargeRandomParity:
    def test_50k_random_population_bit_identical(self):
        rng = np.random.default_rng(42)
        n = 50_000
        # Mix of short (<2d, feeds common-period path) and long periods,
        # plus a chunk of None t0 to exercise the NaN encoding at scale.
        periods = np.concatenate([
            rng.uniform(0.05, 2.0, n // 2),
            rng.uniform(2.0, 60.0, n - n // 2),
        ])
        rng.shuffle(periods)
        t0_raw = rng.uniform(0.0, 100.0, n)
        none_mask = rng.random(n) < 0.05

        rows = []
        for i in range(n):
            tic = i % 5000  # force repeats so bins can plausibly qualify
            t0 = None if none_mask[i] else float(t0_raw[i])
            rows.append(_make_cand(tic, float(periods[i]), t0))

        old = find_systematic_periods(
            rows, min_stars=10, min_concentration=0.5,
            common_period_min_tics=25, common_period_max_period=2.0,
        )
        pop = ColumnarPopulation.from_rows(rows)
        new = find_systematic_periods_columnar(
            pop, min_stars=10, min_concentration=0.5,
            common_period_min_tics=25, common_period_max_period=2.0,
        )
        assert new == old


# ---------------------------------------------------------------------------
# 7. ColumnarPopulation.from_rows
# ---------------------------------------------------------------------------

class TestFromRows:
    def test_casts_and_order_preserved(self):
        rows = [
            {"tic_id": "300", "period_days": 1.5, "t0": 0.25},
            {"tic_id": "100", "period_days": 2.5, "t0": None},
            {"tic_id": "200", "period_days": 3.5, "t0": 0.75},
        ]
        pop = ColumnarPopulation.from_rows(rows)
        assert len(pop) == 3
        assert pop.tic.tolist() == [300, 100, 200]
        assert pop.period.tolist() == [1.5, 2.5, 3.5]
        assert pop.tic.dtype == np.int64
        assert pop.period.dtype == np.float64
        assert pop.t0.dtype == np.float64
        assert math.isnan(pop.t0[1])
        assert pop.t0[0] == 0.25 and pop.t0[2] == 0.75

    def test_int_tic_id_also_accepted(self):
        rows = [{"tic_id": 42, "period_days": 1.0, "t0": 0.5}]
        pop = ColumnarPopulation.from_rows(rows)
        assert pop.tic.tolist() == [42]

    def test_non_numeric_tic_id_raises_clear_error(self):
        rows = [
            {"tic_id": "12345", "period_days": 1.0, "t0": 0.5},
            {"tic_id": "not-a-tic", "period_days": 2.0, "t0": 0.5},
        ]
        with pytest.raises(ValueError, match="non-numeric tic_id"):
            ColumnarPopulation.from_rows(rows)

    def test_empty_iterable(self):
        pop = ColumnarPopulation.from_rows([])
        assert len(pop) == 0
        assert pop.tic.dtype == np.int64
        assert pop.period.dtype == np.float64
        assert pop.t0.dtype == np.float64

    def test_batch_boundary_does_not_lose_or_reorder_rows(self):
        """Force multiple internal flush()es (batch_size small) and check the
        concatenated result still matches input order exactly."""
        rows = [{"tic_id": str(i), "period_days": float(i), "t0": float(i) / 2}
                for i in range(37)]
        pop = ColumnarPopulation.from_rows(rows, batch_size=5)
        assert pop.tic.tolist() == list(range(37))
        assert pop.period.tolist() == [float(i) for i in range(37)]

    def test_period_none_encoded_as_nan(self):
        rows = [{"tic_id": "1", "period_days": None, "t0": 0.1}]
        pop = ColumnarPopulation.from_rows(rows)
        assert math.isnan(pop.period[0])


# ---------------------------------------------------------------------------
# 8. find_deep_eclipse_hosts accepts any iterable (§3.D)
# ---------------------------------------------------------------------------

class TestFindDeepEclipseHostsIterable:
    def _rows(self):
        return [
            {"tic_id": "1", "period_days": 25.0, "t0": 9.4, "depth_ppm": 45_000.0},
            {"tic_id": "2", "period_days": 1.5, "t0": 0.5, "depth_ppm": 1_000.0},
            {"tic_id": "3", "period_days": 25.0, "t0": 0.0, "depth_ppm": 50_000.0},
        ]

    def _sysp(self):
        return [{"period": 25.0, "epoch_phase": 9.4, "n_stars": 76,
                  "concentration": 0.9, "epoch_agnostic": False}]

    def test_generator_matches_list(self):
        rows = self._rows()
        sysp = self._sysp()
        from_list = find_deep_eclipse_hosts(rows, sysp, stellar_radii={})
        from_gen = find_deep_eclipse_hosts((r for r in rows), sysp, stellar_radii={})
        assert from_list == from_gen
        assert from_list == {"1"}

    def test_chained_chunks_matches_single_list(self):
        """Simulates a chunked/streamed source (§3.D's actual motivation):
        itertools.chain of several small lists must give the same result as
        one flat list."""
        import itertools
        rows = self._rows()
        chunks = [rows[:1], rows[1:2], rows[2:]]
        sysp = self._sysp()
        from_list = find_deep_eclipse_hosts(rows, sysp, stellar_radii={})
        from_chain = find_deep_eclipse_hosts(itertools.chain.from_iterable(chunks),
                                              sysp, stellar_radii={})
        assert from_list == from_chain
