"""ExoMiner++ 2.0 conditional-gate tests.

The gate decides, per TIC, whether to score with ExoMiner++ 2.0 FFI (DV
available for a sector >= 36) or to skip and fall back to the existing
ExoMiner-FFI flux model (no DV / DV only below s36). These tests pin the
routing logic using injected DV runs (no MAST round-trip).
"""
from exonit.exominer_pp.gate import (
    SPOC_FFI_MIN_SECTOR,
    GateDecision,
    decide_gate,
    has_spoc_dv,
)


def test_no_dv_skips_pp():
    d = decide_gate("123", dv_runs=[])
    assert isinstance(d, GateDecision)
    assert d.use_exominer_pp is False
    assert d.reason == "no_dv"
    assert d.sector_run is None
    assert has_spoc_dv("123", dv_runs=[]) is False


def test_dv_below_s36_skips_pp():
    # 2-min era single + multi-sector runs all < 36 → no FFI DV.
    d = decide_gate("456", dv_runs=[(1, 1), (14, 14), (1, 13)])
    assert d.use_exominer_pp is False
    assert d.reason == "dv_below_s36"
    assert d.sector_run is None


def test_single_sector_s36_routes_to_pp():
    d = decide_gate("789", dv_runs=[(67, 67)])
    assert d.use_exominer_pp is True
    assert d.reason == "dv_s>=36"
    assert d.sector_run == "67-67"


def test_multi_sector_ffi_prefers_longest_run():
    d = decide_gate("111", dv_runs=[(56, 69), (40, 40), (50, 55)])
    assert d.use_exominer_pp is True
    # longest span (56-69 spans 13 vs 50-55 spans 5) wins
    assert d.sector_run == "56-69"


def test_mixed_old_and_ffi_uses_only_ffi_runs():
    # Has both a sub-36 multi run and an FFI single run → FFI single chosen.
    d = decide_gate("222", dv_runs=[(1, 13), (60, 60)])
    assert d.use_exominer_pp is True
    assert d.sector_run == "60-60"


def test_boundary_sector_36_is_included():
    d = decide_gate("333", dv_runs=[(SPOC_FFI_MIN_SECTOR, SPOC_FFI_MIN_SECTOR)])
    assert d.use_exominer_pp is True
    assert d.sector_run == f"{SPOC_FFI_MIN_SECTOR}-{SPOC_FFI_MIN_SECTOR}"


def test_sector_35_is_excluded():
    d = decide_gate("334", dv_runs=[(35, 35)])
    assert d.use_exominer_pp is False
    assert d.reason == "dv_below_s36"


def test_dv_runs_preserved_in_decision():
    d = decide_gate("335", dv_runs=[(35, 35), (67, 67)])
    assert d.dv_runs == ((35, 35), (67, 67))
    assert d.use_exominer_pp is True


def test_no_regression_no_dv_path_is_pure():
    """The no-DV path must not require MAST or any model — it just returns
    use_exominer_pp=False so the caller keeps the existing discovery flow."""
    d = decide_gate("999", dv_runs=[])
    assert d.use_exominer_pp is False
    # caller's contract: a falsey decision means "do nothing new".
    assert not d.use_exominer_pp
