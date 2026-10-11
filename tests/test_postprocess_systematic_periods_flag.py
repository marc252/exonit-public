"""Tests for scripts/discovery_postprocess.py's --systematic-periods flag.

Deliverable 2 of docs/superpowers/specs/2026-07-11-injection-recovery-e2e.md:
Steps 1/1c must SKIP the live find_systematic_periods computation and instead
LOAD a frozen family set when --systematic-periods is given; without the
flag, production behaviour is UNCHANGED (still calls find_systematic_periods).

The gate for "unchanged" is this test file, not a scorecard pair (no
scorecard-before/after is planned for this deliverable -- state explicit in
the handoff): _resolve_systematic_periods is a pure branch, unit-testable by
asserting which of the two functions is consulted, with zero DB/network
dependency.

Constraints (matching tests/test_anti_toi_override.py /
tests/test_orchestrator_postprocess_cleanup_2026_07_02.py): never touches the
live DB or production PostgreSQL -- these tests never open any DB connection
at all, they call _resolve_systematic_periods directly with an in-memory
population list and a tmp_path Parquet file.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import discovery_postprocess as pp  # noqa: E402

from exonit.systematic_families_io import dump_families  # noqa: E402


class _Args:
    """Minimal stand-in for the argparse.Namespace fields _resolve_systematic_periods reads."""

    def __init__(self, systematic_periods=None, common_period_max_period=2.0):
        self.systematic_periods = systematic_periods
        self.common_period_max_period = common_period_max_period


# ---------------------------------------------------------------------------
# 1. Flag OFF -> inertness (default path unchanged)
# ---------------------------------------------------------------------------

class TestFlagOffInertness:
    def test_default_calls_find_systematic_periods_not_loader(self):
        """Without --systematic-periods, _resolve_systematic_periods MUST call
        find_systematic_periods with the exact same arguments as before this
        change, and MUST NOT touch load_families at all."""
        pop = [{"tic_id": "1", "period_days": 5.0, "t0": 1.0}]
        args = _Args(systematic_periods=None)
        with patch.object(pp, "find_systematic_periods") as mock_find, \
             patch.object(pp, "load_families") as mock_load:
            mock_find.return_value = [{"period": 5.0, "n_stars": 1, "epoch_agnostic": True}]
            result = pp._resolve_systematic_periods(pop, args, common_period_min_tics=10)

        mock_find.assert_called_once_with(
            pop, min_stars=10, min_concentration=0.4,
            common_period_min_tics=10, common_period_max_period=2.0,
        )
        mock_load.assert_not_called()
        assert result == mock_find.return_value

    def test_default_forwards_common_period_max_period_from_args(self):
        """A non-default --common-period-max-period must still reach
        find_systematic_periods unchanged (regression guard: the refactor
        that extracted _resolve_systematic_periods must not hardcode it)."""
        pop = []
        args = _Args(systematic_periods=None, common_period_max_period=3.5)
        with patch.object(pp, "find_systematic_periods") as mock_find:
            mock_find.return_value = []
            pp._resolve_systematic_periods(pop, args, common_period_min_tics=None)
        mock_find.assert_called_once_with(
            pop, min_stars=10, min_concentration=0.4,
            common_period_min_tics=None, common_period_max_period=3.5,
        )

    def test_empty_string_flag_also_falls_back(self):
        """An empty-string --systematic-periods (falsy) behaves like unset --
        guards the falsy-check (``if args.systematic_periods:``) against a
        future refactor that defaults the CLI value to ''."""
        args = _Args(systematic_periods="")
        with patch.object(pp, "find_systematic_periods") as mock_find, \
             patch.object(pp, "load_families") as mock_load:
            mock_find.return_value = []
            pp._resolve_systematic_periods([], args, common_period_min_tics=None)
        mock_find.assert_called_once()
        mock_load.assert_not_called()


# ---------------------------------------------------------------------------
# 2. Flag ON -> loads frozen families, live computation skipped
# ---------------------------------------------------------------------------

class TestFlagOnUsesLoadedFamilies:
    def test_loader_called_and_find_systematic_periods_skipped(self, tmp_path):
        sysp = [{"period": 17.0, "n_stars": 76, "epoch_phase": 9.4,
                 "concentration": 0.9, "epoch_agnostic": False}]
        path = str(tmp_path / "frozen.parquet")
        dump_families(sysp, path)
        args = _Args(systematic_periods=path)

        with patch.object(pp, "find_systematic_periods") as mock_find:
            result = pp._resolve_systematic_periods([], args, common_period_min_tics=10)

        mock_find.assert_not_called()
        assert result == sysp

    def test_loaded_family_vetoes_synthetic_candidate_on_it(self, tmp_path):
        """A family exported to parquet (dump_families) and loaded back via
        --systematic-periods must veto a synthetic candidate sitting on it,
        and NOT veto one that isn't -- exactly like a live-computed family."""
        sysp = [{"period": 17.0, "n_stars": 76, "epoch_phase": 9.4,
                 "concentration": 0.9, "epoch_agnostic": False}]
        path = str(tmp_path / "frozen.parquet")
        dump_families(sysp, path)
        args = _Args(systematic_periods=path)

        loaded = pp._resolve_systematic_periods([], args, common_period_min_tics=10)

        on_systematic = {"period_days": 17.01, "t0": 9.41}
        off_systematic = {"period_days": 3.5, "t0": 1.0}
        assert pp.is_vetoed(on_systematic["period_days"], on_systematic["t0"], loaded) is not None
        assert pp.is_vetoed(off_systematic["period_days"], off_systematic["t0"], loaded) is None

    def test_loaded_epoch_agnostic_family_vetoes_synthetic_candidate(self, tmp_path):
        """Same as above but for a common-period (epoch_agnostic) family, the
        branch missing epoch_phase/concentration keys entirely -- the more
        fragile shape to round-trip correctly."""
        sysp = [{"period": 0.71, "n_stars": 50, "epoch_agnostic": True}]
        path = str(tmp_path / "frozen.parquet")
        dump_families(sysp, path)
        args = _Args(systematic_periods=path)

        loaded = pp._resolve_systematic_periods([], args, common_period_min_tics=10)

        assert pp.is_vetoed(0.7105, 12345.6, loaded) is not None  # t0 irrelevant, agnostic path
        assert pp.is_vetoed(0.90, 12345.6, loaded) is None
