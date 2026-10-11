"""Tests for the shared-signal cross-check between finalists (Step 5e).

Why this check exists (and why these numbers, not invented ones): in lot
``discovery_s0014_lot20260722T073004`` four of the eleven finalists were a
single signal counted four times. Every other finalist gate judges one
candidate in isolation, so none of them could see it. The canonical case below
is that real group, copied from the database on 2026-07-25.

Pure functions only — no DB, no MAST, no filesystem.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from exonit.ffi_finalist_checks import (  # noqa: E402
    angular_separation_deg,
    shared_signal_groups,
)


# The real group — TIC, P, t0, depth, snr, ra, dec as stored.
REAL_GROUP = [
    dict(tic_id="202948052", period_days=10.3437, t0=1693.8075, duration_days=0.15,
         depth_ppm=1714, snr=16.7, ra=304.6052, dec=30.3675),
    dict(tic_id="416877345", period_days=10.3437, t0=1693.8075, duration_days=0.15,
         depth_ppm=2055, snr=16.8, ra=304.2789, dec=30.1941),
    dict(tic_id="202806474", period_days=10.3468, t0=1693.8106, duration_days=0.15,
         depth_ppm=2351, snr=18.7, ra=304.2370, dec=30.2046),
    dict(tic_id="202944495", period_days=10.3609, t0=1693.7727, duration_days=0.15,
         depth_ppm=2057, snr=17.6, ra=304.5038, dec=30.1435),
]

# A genuinely unrelated finalist from the same lot: same patch of sky, but a
# period twice as long and a different epoch.
UNRELATED = dict(tic_id="202945491", period_days=21.6604, t0=1700.0,
                 duration_days=0.15, depth_ppm=1848, snr=14.2,
                 ra=304.5038, dec=30.2)


class TestAngularSeparation:

    def test_known_pair(self):
        """The two closest members of the real group are 2.3 arcmin apart."""
        sep = angular_separation_deg(304.2370, 30.2046, 304.2789, 30.1941)
        assert sep == pytest.approx(0.0377, abs=0.0005)

    def test_identical_position_is_zero(self):
        assert angular_separation_deg(10.0, -20.0, 10.0, -20.0) == pytest.approx(0.0)

    def test_symmetric(self):
        a = angular_separation_deg(12.3, 45.6, 78.9, -12.3)
        b = angular_separation_deg(78.9, -12.3, 12.3, 45.6)
        assert a == pytest.approx(b)

    def test_pole_and_ra_wrap_do_not_explode(self):
        """RA 359.9 and 0.1 are 0.2° apart, not 359.8°."""
        sep = angular_separation_deg(359.9, 0.0, 0.1, 0.0)
        assert sep == pytest.approx(0.2, abs=1e-6)


class TestSharedSignalGroups:

    def test_finds_the_real_group(self):
        groups = shared_signal_groups(REAL_GROUP + [UNRELATED])
        assert len(groups) == 1
        assert {c["tic_id"] for c in groups[0]} == {c["tic_id"] for c in REAL_GROUP}

    def test_keeps_the_strongest_by_snr(self):
        """The kept member must be the highest-SNR one — the pipeline's own
        strength measure. It is NOT a claim that this one is the real source."""
        groups = shared_signal_groups(REAL_GROUP)
        assert groups[0][0]["tic_id"] == "202806474"   # snr 18.7, the highest

    def test_unrelated_finalist_is_not_grouped(self):
        groups = shared_signal_groups(REAL_GROUP + [UNRELATED])
        assert all(UNRELATED["tic_id"] not in {c["tic_id"] for c in g} for g in groups)

    def test_result_is_order_independent(self):
        a = shared_signal_groups(REAL_GROUP)
        b = shared_signal_groups(list(reversed(REAL_GROUP)))
        assert [c["tic_id"] for c in a[0]] == [c["tic_id"] for c in b[0]]

    def test_singletons_are_never_returned(self):
        assert shared_signal_groups([UNRELATED]) == []
        assert shared_signal_groups([]) == []

    def test_same_period_far_apart_is_not_a_group(self):
        """Two stars on opposite sides of the sky can share a period by
        coincidence — or share a *global* systematic — but this check is about
        one signal bleeding into neighbours, so distance must veto it."""
        far = dict(REAL_GROUP[0], tic_id="999", ra=100.0, dec=-40.0)
        assert shared_signal_groups([REAL_GROUP[0], far]) == []

    def test_close_together_but_different_period_is_not_a_group(self):
        other = dict(REAL_GROUP[0], tic_id="999", period_days=5.0, t0=1693.8075)
        assert shared_signal_groups([REAL_GROUP[0], other]) == []

    def test_close_and_same_period_but_antiphase_is_not_a_group(self):
        """Same period, same patch of sky, but transits half a period apart —
        that is two different events, not one shared one."""
        anti = dict(REAL_GROUP[0], tic_id="999",
                    t0=REAL_GROUP[0]["t0"] + REAL_GROUP[0]["period_days"] / 2)
        assert shared_signal_groups([REAL_GROUP[0], anti]) == []

    def test_epoch_match_across_many_cycles(self):
        """t0 values many periods apart still describe the same transit train;
        the comparison must fold, not subtract."""
        later = dict(REAL_GROUP[0], tic_id="999",
                     t0=REAL_GROUP[0]["t0"] + 40 * REAL_GROUP[0]["period_days"],
                     ra=304.2789, dec=30.1941)
        groups = shared_signal_groups([REAL_GROUP[0], later])
        assert len(groups) == 1

    def test_grouping_is_transitive(self):
        """A~B and B~C must yield ONE group of three, which is what a
        systematic shared across a CCD region looks like."""
        groups = shared_signal_groups(REAL_GROUP[:3])
        assert len(groups) == 1 and len(groups[0]) == 3

    def test_missing_coordinates_never_drops_a_finalist(self):
        """Conservative by construction: a candidate with no ra/dec cannot be
        cross-checked, so it must survive rather than be grouped on a guess."""
        blind = dict(REAL_GROUP[1], tic_id="999", ra=None, dec=None)
        groups = shared_signal_groups([REAL_GROUP[0], blind])
        assert groups == []

    def test_missing_period_or_epoch_never_drops_a_finalist(self):
        no_p = dict(REAL_GROUP[1], tic_id="998", period_days=None)
        no_t0 = dict(REAL_GROUP[1], tic_id="997", t0=None)
        assert shared_signal_groups([REAL_GROUP[0], no_p, no_t0]) == []

    def test_missing_duration_falls_back_and_still_groups(self):
        pair = [dict(c, duration_days=None) for c in REAL_GROUP[:2]]
        assert len(shared_signal_groups(pair)) == 1

    def test_tolerances_are_honoured(self):
        """A tighter separation threshold must break the real group up — the
        knobs have to actually do something, or they are decoration."""
        assert shared_signal_groups(REAL_GROUP, max_sep_deg=0.01) == []
        assert shared_signal_groups(REAL_GROUP, period_rtol=1e-9) != REAL_GROUP
