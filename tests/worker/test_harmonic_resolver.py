"""Unit tests for the symmetric harmonic resolver in the multi-planet loop.

Audit §76-85: the original asymmetric resolver only checked subharmonics
(P/2 .. P/5) and broke at the first match, so when BLS landed on a
subharmonic of the true transit, the candidate was reported with the
wrong (shorter) period and downstream stages (CNN, FPP, scorecard
matching) saw 4-5 duplicate candidates per TIC.

Symmetric resolver: also check superharmonics (2P .. 5P), prefer the
LONGEST period whose SNR>=5 and whose depth is compatible with the
seed (depth_ratio in (0.5, 2.0)).

These tests use a mocked BLS callable so the focus is on the resolver
logic, not BLS itself.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "worker") not in sys.path:
    sys.path.append(str(ROOT / "worker"))


def _bls_mock(period_to_snr_depth: dict, default=None):
    """Build a fake bls_fn that returns canned SNR/depth based on the
    midpoint of the requested period window. Period_to_snr_depth keys
    must match the resolver's exact arithmetic (base * N or base / N)."""
    def fake(time, flux, *, period_min, period_max, n_periods):
        center = (period_min + period_max) / 2.0
        for p, payload in period_to_snr_depth.items():
            if abs(center - p) / p < 0.05:
                return {
                    "period": p,
                    "snr": payload["snr"],
                    "depth": payload["depth"],
                    "depth_ppm": payload["depth"] * 1e6,
                    "duration": payload.get("duration", 0.1),
                    "t0": payload.get("t0", 0.5),
                    "power": payload.get("power", 1.0),
                }
        if default is not None:
            return default
        return {"period": center, "snr": 0.0, "depth": 0.0,
                "depth_ppm": 0.0, "duration": 0.1, "t0": 0.0, "power": 0.0}
    return fake


# ─────────────────────────────────────────────────────────────────────
# Test 1 (GOLDEN bug case): BLS landed on subharmonic, resolver promotes
# ─────────────────────────────────────────────────────────────────────
def test_promotes_seed_to_real_longer_period():
    """TRAPPIST-1-b style: real transit at P=4.0d, BLS spectrum landed
    on P=2.0d (the P/2 subharmonic). Resolver must check 2x=4.0d, find
    a strong compatible peak there, and return P=4.0d."""
    from worker.worker import resolve_harmonic

    seed = {"period": 2.0, "snr": 18.0, "depth": 0.0050,
            "depth_ppm": 5000, "duration": 0.10, "t0": 0.5, "power": 1.0}
    bls = _bls_mock({
        # 2x of seed = 4.0 — true transit, should be picked
        4.0: {"snr": 22.0, "depth": 0.0048, "duration": 0.10},
        # other harmonics: weak or incompatible
        6.0: {"snr": 3.0, "depth": 0.0040, "duration": 0.10},
        8.0: {"snr": 2.0, "depth": 0.0030, "duration": 0.10},
        10.0: {"snr": 2.0, "depth": 0.0030, "duration": 0.10},
        1.0: {"snr": 12.0, "depth": 0.0050, "duration": 0.10},  # P/2 subhar
    })

    out = resolve_harmonic(
        time=None, flux=None, result=seed,
        period_min=0.5, period_max=50.0, bls_fn=bls,
    )
    assert abs(out["period"] - 4.0) < 0.01, \
        f"Expected promotion to P=4.0 (real), got P={out['period']}"


# ─────────────────────────────────────────────────────────────────────
# Test 2: superharmonics limited to 2x (3x/4x/5x intentionally NOT probed)
# ─────────────────────────────────────────────────────────────────────
def test_superharmonics_limited_to_2x():
    """For periodic signals, BLS at 3x/4x/5x of the true period also
    reports comparable SNR (multiple transits stack per fold), so
    promoting beyond 2x is over-aggressive and creates spurious upward
    jumps. Even if 5x reports good SNR/depth, the resolver must NOT
    pick it — only 2x is a safe superharmonic promotion."""
    from worker.worker import resolve_harmonic

    seed = {"period": 2.0, "snr": 15.0, "depth": 0.0050,
            "depth_ppm": 5000, "duration": 0.10, "t0": 0.5, "power": 1.0}
    bls = _bls_mock({
        4.0:  {"snr": 16.0, "depth": 0.0050, "duration": 0.10},  # 2x ok — pick this
        6.0:  {"snr": 18.0, "depth": 0.0050, "duration": 0.10},  # 3x looks good but skipped
        10.0: {"snr": 20.0, "depth": 0.0050, "duration": 0.10},  # 5x looks great but skipped
    })

    out = resolve_harmonic(
        time=None, flux=None, result=seed,
        period_min=0.5, period_max=50.0, bls_fn=bls,
    )
    assert abs(out["period"] - 4.0) < 0.01, \
        f"Expected 2x promotion to P=4.0 (3x/4x/5x not probed), got P={out['period']}"


# ─────────────────────────────────────────────────────────────────────
# Test 3: no compatible harmonic → keep seed
# ─────────────────────────────────────────────────────────────────────
def test_keeps_seed_when_no_compatible_harmonic():
    """Seed is the real transit, all other harmonics have weak SNR or
    incompatible depth. Resolver must return the seed unchanged."""
    from worker.worker import resolve_harmonic

    seed = {"period": 5.0, "snr": 25.0, "depth": 0.0080,
            "depth_ppm": 8000, "duration": 0.15, "t0": 1.2, "power": 2.0}
    bls = _bls_mock({}, default={
        "period": 0.0, "snr": 1.0, "depth": 0.001,
        "depth_ppm": 1000, "duration": 0.1, "t0": 0.0, "power": 0.0,
    })

    out = resolve_harmonic(
        time=None, flux=None, result=seed,
        period_min=0.5, period_max=50.0, bls_fn=bls,
    )
    assert out is seed, "Should return seed unchanged when no harmonic qualifies"


# ─────────────────────────────────────────────────────────────────────
# Test 4: subharmonic chosen if it's the only compatible AND it's longer-than-cutoff
# (existing behavior preservation: subharmonic resolution is still possible
#  when no superharmonic works and the seed is itself a higher harmonic)
# ─────────────────────────────────────────────────────────────────────
def test_subharmonic_still_picked_when_only_option():
    """Seed at P=10d (BLS picked a 2x harmonic of the true 5d transit).
    No superharmonic (20d, 30d, 40d, 50d) qualifies. P/2=5d does qualify.
    Resolver should fall back to subharmonic and return P=5d."""
    from worker.worker import resolve_harmonic

    seed = {"period": 10.0, "snr": 14.0, "depth": 0.0050,
            "depth_ppm": 5000, "duration": 0.10, "t0": 0.5, "power": 1.0}
    bls = _bls_mock({
        # superharmonics: all weak
        20.0: {"snr": 2.0, "depth": 0.0020, "duration": 0.10},
        30.0: {"snr": 2.0, "depth": 0.0020, "duration": 0.10},
        40.0: {"snr": 1.5, "depth": 0.0010, "duration": 0.10},
        50.0: {"snr": 1.5, "depth": 0.0010, "duration": 0.10},
        # subharmonic P/2 = 5d, strong
        5.0:  {"snr": 18.0, "depth": 0.0048, "duration": 0.10},
    })

    out = resolve_harmonic(
        time=None, flux=None, result=seed,
        period_min=0.5, period_max=50.0, bls_fn=bls,
    )
    assert abs(out["period"] - 5.0) < 0.01, \
        f"Expected fallback to P=5.0 subharmonic, got P={out['period']}"


# ─────────────────────────────────────────────────────────────────────
# Test 5: candidates outside [period_min, period_max] are skipped
# ─────────────────────────────────────────────────────────────────────
def test_respects_period_bounds():
    """Seed at P=15d, period_max=20d. Superharmonics 30d/45d/60d/75d are
    all out of range. P/3=5d is in range and strong. Resolver should
    pick 5d (only valid candidate)."""
    from worker.worker import resolve_harmonic

    seed = {"period": 15.0, "snr": 12.0, "depth": 0.0050,
            "depth_ppm": 5000, "duration": 0.10, "t0": 0.5, "power": 1.0}
    bls = _bls_mock({
        30.0: {"snr": 20.0, "depth": 0.0050, "duration": 0.10},  # OOB
        45.0: {"snr": 20.0, "depth": 0.0050, "duration": 0.10},  # OOB
        5.0:  {"snr": 16.0, "depth": 0.0048, "duration": 0.10},  # P/3 ok
    })

    out = resolve_harmonic(
        time=None, flux=None, result=seed,
        period_min=0.5, period_max=20.0, bls_fn=bls,
    )
    assert abs(out["period"] - 5.0) < 0.01, \
        f"Expected P=5.0 (only in-range candidate), got P={out['period']}"


# ─────────────────────────────────────────────────────────────────────
# Test 6: depth-incompatible harmonic skipped + weaker subharmonic skipped
# ─────────────────────────────────────────────────────────────────────
def test_skips_depth_incompatible_and_weaker_harmonic():
    """Superharmonic 2x=10d has high SNR but depth ratio = 0.1 (way out
    of [0.5, 2.0]) — skipped. Subharmonic P/2=2.5d has compatible depth
    but SNR=15 < seed's SNR=18 — skipped (weaker than seed → seed is
    more likely the true period). Resolver returns the seed unchanged."""
    from worker.worker import resolve_harmonic

    seed = {"period": 5.0, "snr": 18.0, "depth": 0.0050,
            "depth_ppm": 5000, "duration": 0.10, "t0": 0.5, "power": 1.0}
    bls = _bls_mock({
        10.0: {"snr": 30.0, "depth": 0.0005, "duration": 0.10},  # depth_ratio=0.1, skip
        2.5:  {"snr": 15.0, "depth": 0.0050, "duration": 0.10},  # weaker than seed, skip
    })

    out = resolve_harmonic(
        time=None, flux=None, result=seed,
        period_min=0.5, period_max=50.0, bls_fn=bls,
    )
    assert out is seed, \
        f"Expected seed unchanged (no qualifying harmonic), got P={out['period']}"


# ─────────────────────────────────────────────────────────────────────
# FFI regime (data_source='ffi'): BLS locks onto P/3..P/5 subharmonics
# far more often; the resolver must climb past 2x and tolerate a noisier
# (lower-SNR) fundamental. SPOC behaviour must stay unchanged.
# Root cause: 2026-05-20 audit of big-run-v3 FFI-only KP/CP.
# ─────────────────────────────────────────────────────────────────────
def test_ffi_promotes_third_subharmonic():
    """FFI: real transit at P=6.0d, BLS landed on P=2.0d (the P/3 alias).
    SPOC mode only probes 2x and cannot recover 6.0; FFI mode probes 3x
    and recovers it."""
    from worker.worker import resolve_harmonic

    seed = {"period": 2.0, "snr": 18.0, "depth": 0.0050,
            "depth_ppm": 5000, "duration": 0.10, "t0": 0.5, "power": 1.0}
    bls = _bls_mock({
        6.0:  {"snr": 20.0, "depth": 0.0048, "duration": 0.10},  # 3x = true
        4.0:  {"snr": 6.0,  "depth": 0.0010, "duration": 0.10},  # 2x weak/incompat
        8.0:  {"snr": 2.0,  "depth": 0.0030, "duration": 0.10},
        10.0: {"snr": 2.0,  "depth": 0.0030, "duration": 0.10},
    })

    # SPOC: cannot reach 3x → stays at the seed alias.
    out_spoc = resolve_harmonic(
        time=None, flux=None, result=seed,
        period_min=0.5, period_max=50.0, bls_fn=bls, data_source="spoc",
    )
    assert abs(out_spoc["period"] - 2.0) < 0.01, \
        f"SPOC must NOT promote past 2x; got P={out_spoc['period']}"

    # FFI: recovers the true 3x fundamental.
    out_ffi = resolve_harmonic(
        time=None, flux=None, result=seed,
        period_min=0.5, period_max=50.0, bls_fn=bls, data_source="ffi",
    )
    assert abs(out_ffi["period"] - 6.0) < 0.01, \
        f"FFI must recover P=6.0 (P/3 alias); got P={out_ffi['period']}"


def test_ffi_relaxed_snr_allows_noisier_fundamental():
    """FFI: true 2x fundamental (4.0d) is genuinely noisier (SNR=12) than
    the spurious P/2 alias seed (SNR=18). SPOC rejects it (12 < 18); FFI
    accepts it (12 >= 0.6*18 = 10.8)."""
    from worker.worker import resolve_harmonic

    seed = {"period": 2.0, "snr": 18.0, "depth": 0.0050,
            "depth_ppm": 5000, "duration": 0.10, "t0": 0.5, "power": 1.0}
    bls = _bls_mock({
        4.0: {"snr": 12.0, "depth": 0.0050, "duration": 0.10},  # true, noisier
    })

    out_spoc = resolve_harmonic(
        time=None, flux=None, result=seed,
        period_min=0.5, period_max=50.0, bls_fn=bls, data_source="spoc",
    )
    assert abs(out_spoc["period"] - 2.0) < 0.01, \
        f"SPOC must reject weaker harmonic; got P={out_spoc['period']}"

    out_ffi = resolve_harmonic(
        time=None, flux=None, result=seed,
        period_min=0.5, period_max=50.0, bls_fn=bls, data_source="ffi",
    )
    assert abs(out_ffi["period"] - 4.0) < 0.01, \
        f"FFI must accept noisier true fundamental; got P={out_ffi['period']}"


def test_ffi_transit_count_guard_blocks_over_promotion():
    """FFI: the extended 3x/4x/5x reach must not over-promote to a period
    so long that fewer than two transits fit the baseline. Baseline = 20d.
    5x = 12.0d (only 1.67 transits) is blocked despite great SNR; the 2x
    fundamental at 4.8d (4.17 transits) is picked instead."""
    import numpy as np
    from worker.worker import resolve_harmonic

    time = np.linspace(0.0, 20.0, 300)
    seed = {"period": 2.4, "snr": 15.0, "depth": 0.0050,
            "depth_ppm": 5000, "duration": 0.10, "t0": 0.5, "power": 1.0}
    bls = _bls_mock({
        12.0: {"snr": 25.0, "depth": 0.0050, "duration": 0.10},  # 5x but <2 transits
        9.6:  {"snr": 3.0,  "depth": 0.0050, "duration": 0.10},  # 4x weak
        7.2:  {"snr": 3.0,  "depth": 0.0050, "duration": 0.10},  # 3x weak
        4.8:  {"snr": 16.0, "depth": 0.0050, "duration": 0.10},  # 2x, supported
    })

    out = resolve_harmonic(
        time=time, flux=time, result=seed,
        period_min=0.5, period_max=50.0, bls_fn=bls, data_source="ffi",
    )
    assert abs(out["period"] - 4.8) < 0.01, \
        f"Transit-count guard must block 12.0d over-promotion; got P={out['period']}"


# ─────────────────────────────────────────────────────────────────────
# SPOC long-period recovery (config-gated, default off): BLS locks onto a
# rational sub-harmonic (e.g. 2P/3) of a long-period transit whose alias
# outshines the noisier true fundamental. With long_period_recovery on, the
# resolver promotes via the inverse factor (3/2) into the long-period regime.
# Validated 2026-05-22 on big-run-v4 (scripts/diag_bin_alias.py).
# ─────────────────────────────────────────────────────────────────────
def test_spoc_long_period_recovery_rational_alias():
    """SPOC: real transit at P=20.0d, BLS landed on the 2P/3 alias at
    13.333d (SNR 80). The true 20d fundamental is genuinely noisier
    (SNR 50). Flag OFF → SPOC stays at the alias (only 2x probed, and the
    fundamental is weaker). Flag ON → promotes via 3/2 to 20.0d."""
    import numpy as np
    from worker.worker import resolve_harmonic

    time = np.linspace(0.0, 200.0, 2000)  # 10 transits of a 20d planet
    seed = {"period": 13.3333, "snr": 80.0, "depth": 0.0050,
            "depth_ppm": 5000, "duration": 0.10, "t0": 0.5, "power": 1.0}
    bls = _bls_mock({
        20.0:   {"snr": 50.0, "depth": 0.0050, "duration": 0.10},  # 3/2 -> true
        26.6666: {"snr": 6.0, "depth": 0.0050, "duration": 0.10},  # 2x weak
    }, default={"period": 0.0, "snr": 2.0, "depth": 0.001,
                "depth_ppm": 1000, "duration": 0.1, "t0": 0.0, "power": 0.0})

    out_off = resolve_harmonic(
        time=time, flux=time, result=seed,
        period_min=0.5, period_max=50.0, bls_fn=bls, data_source="spoc",
    )
    assert abs(out_off["period"] - 13.3333) < 0.01, \
        f"Flag OFF must keep the alias; got P={out_off['period']}"

    out_on = resolve_harmonic(
        time=time, flux=time, result=seed,
        period_min=0.5, period_max=50.0, bls_fn=bls, data_source="spoc",
        long_period_recovery=True,
    )
    assert abs(out_on["period"] - 20.0) < 0.01, \
        f"Flag ON must promote 2P/3 alias to 20.0d; got P={out_on['period']}"


def test_spoc_long_recovery_respects_min_days():
    """Short-period SPOC detections must be unaffected even with the flag on.
    Seed at P=4.0d; the 3/2 promotion (6.0d) is below long_period_min_days
    (10d) so it is not probed, and the strict 2x path finds nothing strong.
    Resolver keeps the seed."""
    import numpy as np
    from worker.worker import resolve_harmonic

    time = np.linspace(0.0, 80.0, 1000)
    seed = {"period": 4.0, "snr": 30.0, "depth": 0.0050,
            "depth_ppm": 5000, "duration": 0.10, "t0": 0.5, "power": 1.0}
    bls = _bls_mock({
        6.0: {"snr": 35.0, "depth": 0.0050, "duration": 0.10},  # 3/2 but <10d
        8.0: {"snr": 5.0,  "depth": 0.0050, "duration": 0.10},  # 2x weak
    }, default={"period": 0.0, "snr": 2.0, "depth": 0.001,
                "depth_ppm": 1000, "duration": 0.1, "t0": 0.0, "power": 0.0})

    out = resolve_harmonic(
        time=time, flux=time, result=seed,
        period_min=0.5, period_max=50.0, bls_fn=bls, data_source="spoc",
        long_period_recovery=True, long_period_min_days=10.0,
    )
    assert abs(out["period"] - 4.0) < 0.01, \
        f"Promotions below min_days must be skipped; got P={out['period']}"


def test_spoc_long_recovery_transit_guard_blocks():
    """The promoted long period must still fit >=2 transits in the baseline.
    Baseline 35d; seed P=13.333d alias, 3/2 promotion = 20d but only
    1.75 transits fit 35d → blocked despite a strong peak. Resolver keeps
    the seed."""
    import numpy as np
    from worker.worker import resolve_harmonic

    time = np.linspace(0.0, 35.0, 600)
    seed = {"period": 13.3333, "snr": 80.0, "depth": 0.0050,
            "depth_ppm": 5000, "duration": 0.10, "t0": 0.5, "power": 1.0}
    bls = _bls_mock({
        20.0: {"snr": 60.0, "depth": 0.0050, "duration": 0.10},  # 3/2, <2 transits
    }, default={"period": 0.0, "snr": 2.0, "depth": 0.001,
                "depth_ppm": 1000, "duration": 0.1, "t0": 0.0, "power": 0.0})

    out = resolve_harmonic(
        time=time, flux=time, result=seed,
        period_min=0.5, period_max=50.0, bls_fn=bls, data_source="spoc",
        long_period_recovery=True,
    )
    assert abs(out["period"] - 13.3333) < 0.01, \
        f"Transit-count guard must block under-supported promotion; got P={out['period']}"
