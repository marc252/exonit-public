"""Tests for exonit.systematic_families_io (dump_families / load_families).

Deliverable 2 of docs/superpowers/specs/2026-07-11-injection-recovery-e2e.md:
freeze the systematic-families structure ``find_systematic_periods`` returns
to Parquet, and load it back byte-identical (same veto decisions) so the
injection-recovery e2e harness can veto synthetic candidates against a frozen
snapshot of prod's families instead of recomputing them live over a
synthetic-contaminated population.
"""
from __future__ import annotations

import math

import pytest

from exonit.systematic_families_io import dump_families, load_families
from exonit.systematic_veto import (
    find_deep_eclipse_hosts,
    find_systematic_periods,
    is_vetoed,
)


def _make_cand(tic_id, period, t0, verdict="planet_candidate"):
    return {"tic_id": str(tic_id), "period_days": period, "t0": t0,
            "claude_verdict": verdict}


class TestRoundTrip:
    def test_empty_list_round_trips(self, tmp_path):
        path = str(tmp_path / "empty.parquet")
        dump_families([], path)
        assert load_families(path) == []

    def test_epoch_clustered_entry_round_trips_exactly(self, tmp_path):
        """A find_systematic_periods()-shaped epoch-clustered entry survives
        dump->load with identical keys/values (dtypes preserved)."""
        sysp = find_systematic_periods(
            [_make_cand(i, 17.0 + (i % 3) * 0.001, 9.4 + (i % 3) * 0.01)
             for i in range(20)],
            min_stars=10, min_concentration=0.4,
        )
        assert sysp, "fixture must actually produce a systematic entry"
        assert sysp[0]["epoch_agnostic"] is False
        assert "epoch_phase" in sysp[0] and "concentration" in sysp[0]

        path = str(tmp_path / "clustered.parquet")
        dump_families(sysp, path)
        loaded = load_families(path)

        assert len(loaded) == len(sysp)
        for orig, got in zip(sysp, loaded):
            assert got["period"] == pytest.approx(orig["period"])
            assert got["n_stars"] == orig["n_stars"]
            assert got["epoch_agnostic"] == orig["epoch_agnostic"]
            assert got["epoch_phase"] == pytest.approx(orig["epoch_phase"])
            assert got["concentration"] == pytest.approx(orig["concentration"])

    def test_epoch_agnostic_entry_omits_epoch_phase_and_concentration(self, tmp_path):
        """Common-period (epoch_agnostic=True) entries have NO epoch_phase/
        concentration keys in find_systematic_periods's own output — the
        round trip must preserve that absence, not fabricate 0.0/NaN values
        that is_vetoed could misinterpret."""
        cands = [_make_cand(i, 0.71 + (i % 5) * 0.0001, (i * 37) % 100 * 0.001)
                 for i in range(15)]
        sysp = find_systematic_periods(
            cands, min_stars=999,  # disable epoch-clustered path
            min_concentration=0.4,
            common_period_min_tics=10, common_period_max_period=2.0,
        )
        assert sysp and sysp[0]["epoch_agnostic"] is True
        assert "epoch_phase" not in sysp[0]
        assert "concentration" not in sysp[0]

        path = str(tmp_path / "agnostic.parquet")
        dump_families(sysp, path)
        loaded = load_families(path)

        assert len(loaded) == 1
        assert loaded[0]["epoch_agnostic"] is True
        assert "epoch_phase" not in loaded[0]
        assert "concentration" not in loaded[0]
        assert loaded[0]["period"] == pytest.approx(sysp[0]["period"])
        assert loaded[0]["n_stars"] == sysp[0]["n_stars"]

    def test_mixed_clustered_and_agnostic_round_trip_preserves_order(self, tmp_path):
        """is_vetoed returns the FIRST matching systematic in list order, so
        a dump/load round trip must preserve relative order exactly."""
        sysp = [
            {"period": 17.0, "n_stars": 76, "epoch_phase": 9.4,
             "concentration": 0.9, "epoch_agnostic": False},
            {"period": 0.71, "n_stars": 50, "epoch_agnostic": True},
            {"period": 25.0, "n_stars": 30, "epoch_phase": 12.5,
             "concentration": 0.85, "epoch_agnostic": False},
        ]
        path = str(tmp_path / "mixed.parquet")
        dump_families(sysp, path)
        loaded = load_families(path)
        assert [s["period"] for s in loaded] == [s["period"] for s in sysp]
        assert loaded == sysp


class TestRoundTripPreservesVetoDecisions:
    """The actual gate: dump -> load -> identical is_vetoed decisions."""

    def test_epoch_clustered_veto_decision_identical(self, tmp_path):
        sysp = [{"period": 17.0, "n_stars": 76, "epoch_phase": 9.4,
                 "concentration": 0.9, "epoch_agnostic": False}]
        path = str(tmp_path / "sysp.parquet")
        dump_families(sysp, path)
        loaded = load_families(path)

        cases = [
            (17.02, 9.41),   # inside window -> vetoed
            (17.02, 3.0),    # period ok, epoch far -> NOT vetoed
            (5.0, 9.4),      # period far -> NOT vetoed
        ]
        for period, t0 in cases:
            before = is_vetoed(period, t0, sysp)
            after = is_vetoed(period, t0, loaded)
            assert (before is None) == (after is None), (period, t0)
            if before is not None:
                assert before["period"] == after["period"]

    def test_epoch_agnostic_veto_decision_identical(self, tmp_path):
        sysp = [{"period": 0.71, "n_stars": 50, "epoch_agnostic": True}]
        path = str(tmp_path / "sysp.parquet")
        dump_families(sysp, path)
        loaded = load_families(path)

        cases = [
            (0.7105, 99.0),  # within 2% rel tol, arbitrary t0 -> vetoed
            (0.90, 99.0),    # outside tol -> NOT vetoed
        ]
        for period, t0 in cases:
            before = is_vetoed(period, t0, sysp)
            after = is_vetoed(period, t0, loaded)
            assert (before is None) == (after is None), (period, t0)

    def test_find_deep_eclipse_hosts_decision_identical(self, tmp_path):
        """find_deep_eclipse_hosts is the OTHER consumer of the sysp shape
        (Step 1c) — verify its primary (commonality) arm agrees pre/post
        round trip, not just is_vetoed directly."""
        sysp = [{"period": 25.0, "n_stars": 40, "epoch_phase": 12.5,
                 "concentration": 0.88, "epoch_agnostic": False}]
        path = str(tmp_path / "sysp.parquet")
        dump_families(sysp, path)
        loaded = load_families(path)

        sigs = [
            {"tic_id": "1", "depth_ppm": 55000, "period_days": 25.05, "t0": 12.51},
            {"tic_id": "2", "depth_ppm": 55000, "period_days": 5.0, "t0": 1.0},
        ]
        before = find_deep_eclipse_hosts(sigs, sysp, stellar_radii={})
        after = find_deep_eclipse_hosts(sigs, loaded, stellar_radii={})
        assert before == after
        assert before == {"1"}
