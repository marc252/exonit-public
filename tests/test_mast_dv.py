"""Tests for exonit.mast_dv — SPOC DV XML run discovery + cache."""
from unittest.mock import patch, MagicMock

import pytest

from exonit.mast_dv import (
    MastDVCache,
    dv_runs_cover_sector,
    has_dv_for_sector,
    parse_dv_filename,
    pick_best_run,
)


# -------- parse_dv_filename --------

def test_parse_dv_filename_multi_sector():
    assert parse_dv_filename(
        "tess2018206190142-s0001-s0039-0000000382437043-00493_dvr.xml"
    ) == (1, 39)


def test_parse_dv_filename_single_sector_dvr():
    assert parse_dv_filename(
        "tess2018206190142-s0014-s0014-0000000382437043-00493_dvr.xml"
    ) == (14, 14)


def test_parse_dv_filename_ignores_non_dv():
    assert parse_dv_filename("tess2018206045859-s0001-0000000382437043-0120-s_lc.fits") is None


def test_parse_dv_filename_ignores_dvt():
    # dvt.xml is the time-series DV file, we only match on dvr.xml (report)
    assert parse_dv_filename("tess-s0001-s0039-000382437043-00493_dvt.xml") is None


# -------- pick_best_run --------

def test_pick_best_run_prefers_longest_multi_sector():
    runs = [(1, 13), (14, 26), (1, 39), (27, 39)]
    assert pick_best_run(runs) == "1-39"


def test_pick_best_run_falls_back_to_latest_single_sector():
    runs = [(44, 44), (46, 46), (71, 71)]
    assert pick_best_run(runs) == "71-71"


def test_pick_best_run_empty():
    assert pick_best_run([]) is None


def test_pick_best_run_mixed_prefers_multi():
    runs = [(14, 14), (1, 36), (25, 25)]
    assert pick_best_run(runs) == "1-36"


# -------- MastDVCache --------

def test_cache_empty_on_creation(tmp_path):
    cache = MastDVCache(db_path=str(tmp_path / "dv.db"))
    assert cache.get_runs("123") is None
    cache.close()


def test_cache_stores_and_retrieves(tmp_path):
    cache = MastDVCache(db_path=str(tmp_path / "dv.db"))
    cache.put_runs("123", [(1, 13), (14, 26)])
    runs = cache.get_runs("123")
    cache.close()
    assert runs == [(1, 13), (14, 26)]


def test_cache_stores_empty_runs_as_negative(tmp_path):
    """TICs without DV XML are cached as empty list to avoid re-querying."""
    cache = MastDVCache(db_path=str(tmp_path / "dv.db"))
    cache.put_runs("456", [])
    runs = cache.get_runs("456")
    cache.close()
    assert runs == []


def test_cache_distinguishes_cache_miss_from_empty(tmp_path):
    cache = MastDVCache(db_path=str(tmp_path / "dv.db"))
    # not yet queried
    assert cache.get_runs("999") is None
    # queried, found nothing
    cache.put_runs("999", [])
    assert cache.get_runs("999") == []
    cache.close()


def test_cache_overwrites_existing(tmp_path):
    cache = MastDVCache(db_path=str(tmp_path / "dv.db"))
    cache.put_runs("1", [(1, 13)])
    cache.put_runs("1", [(1, 13), (14, 26)])
    assert cache.get_runs("1") == [(1, 13), (14, 26)]
    cache.close()


# -------- fetch_dv_runs (with MAST mock) --------

def test_fetch_dv_runs_parses_products_and_caches(tmp_path):
    from exonit.mast_dv import fetch_dv_runs

    cache = MastDVCache(db_path=str(tmp_path / "dv.db"))

    # Mock astroquery response
    mock_obs = MagicMock()
    mock_obs.__len__ = MagicMock(return_value=1)

    mock_products = {
        "productFilename": [
            "tess2018206045859-s0001-0000000382437043-0120-s_lc.fits",
            "tess2018206190142-s0001-s0039-0000000382437043-00493_dvr.xml",
            "tess-s0014-s0014-000382437043-00111_dvr.xml",
        ]
    }

    with patch("exonit.mast_dv.Observations") as mobs:
        mobs.query_criteria.return_value = mock_obs
        mobs.get_product_list.return_value = mock_products

        runs = fetch_dv_runs("382437043", cache=cache)

    assert set(runs) == {(1, 39), (14, 14)}
    # Second call should hit cache, no MAST call
    with patch("exonit.mast_dv.Observations") as mobs2:
        mobs2.query_criteria.side_effect = AssertionError("should not be called")
        runs2 = fetch_dv_runs("382437043", cache=cache)
    assert set(runs2) == {(1, 39), (14, 14)}
    cache.close()


def test_fetch_dv_runs_caches_empty_for_missing_tic(tmp_path):
    from exonit.mast_dv import fetch_dv_runs

    cache = MastDVCache(db_path=str(tmp_path / "dv.db"))
    mock_obs = MagicMock()
    mock_obs.__len__ = MagicMock(return_value=0)

    with patch("exonit.mast_dv.Observations") as mobs:
        mobs.query_criteria.return_value = mock_obs
        runs = fetch_dv_runs("nonexistent", cache=cache)

    assert runs == []
    assert cache.get_runs("nonexistent") == []
    cache.close()


# -------- dv_runs_cover_sector (docs/superpowers/specs/                 --
# -------- exominer-dv-gate-sector.md — the range-coverage predicate)    --

def test_dv_runs_cover_sector_single_sector_no_match():
    """(12, 12) does NOT cover sector 14 — the exact premise error this
    fix closes: SPOC DV for sector 12 says nothing about sector 14."""
    assert dv_runs_cover_sector([(12, 12)], 14) is False


def test_dv_runs_cover_sector_multi_sector_no_match():
    """(1, 13) does NOT cover sector 14 — a multi-sector run that ends
    right before the candidate's sector is still no coverage."""
    assert dv_runs_cover_sector([(1, 13)], 14) is False


def test_dv_runs_cover_sector_multi_sector_match():
    """(13, 14) DOES cover sector 14 — multi-sector run whose range
    includes it."""
    assert dv_runs_cover_sector([(13, 14)], 14) is True


def test_dv_runs_cover_sector_single_sector_match():
    """(14, 14) DOES cover sector 14 — exact single-sector match."""
    assert dv_runs_cover_sector([(14, 14)], 14) is True


def test_dv_runs_cover_sector_empty_runs():
    assert dv_runs_cover_sector([], 14) is False


def test_dv_runs_cover_sector_only_one_of_several_runs_covers():
    """A TIC can have several DV reports; coverage is TRUE if ANY of them
    covers the sector, not just the first/best one."""
    assert dv_runs_cover_sector([(1, 12), (20, 25)], 14) is False
    assert dv_runs_cover_sector([(1, 12), (13, 20)], 14) is True


# -------- has_dv_for_sector (I/O-aware wrapper, shares fetch_dv_runs' cache) --

def test_has_dv_for_sector_uses_cache_no_mast_call(tmp_path):
    """A cache hit must never fall through to a real MAST query."""
    cache = MastDVCache(db_path=str(tmp_path / "dv.db"))
    cache.put_runs("500", [(1, 13)])

    with patch("exonit.mast_dv.Observations") as mobs:
        mobs.query_criteria.side_effect = AssertionError(
            "should not hit MAST — cache hit")
        covered = has_dv_for_sector("500", 14, cache=cache)

    assert covered is False  # (1, 13) does not cover 14
    cache.close()


def test_has_dv_for_sector_true_when_covered(tmp_path):
    cache = MastDVCache(db_path=str(tmp_path / "dv.db"))
    cache.put_runs("501", [(13, 15)])
    assert has_dv_for_sector("501", 14, cache=cache) is True
    cache.close()


def test_has_dv_for_sector_false_when_no_runs_at_all(tmp_path):
    cache = MastDVCache(db_path=str(tmp_path / "dv.db"))
    cache.put_runs("502", [])
    assert has_dv_for_sector("502", 14, cache=cache) is False
    cache.close()
