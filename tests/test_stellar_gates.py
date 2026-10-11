"""The Rp-gate: a transit too deep, for its star, to be a planet.

depth = (Rp/R*)². The CNN never sees R*, so an 0.5% eclipse on a 15 R☉ red
giant scores exactly like a Jupiter on a dwarf — while implying a companion the
size of the Sun. Measured on lot 20260714T175216: 91% of the candidates that
reached TRICERATOPS implied a companion above 2 R_Jup (median 7.1).

The threshold is calibrated against REAL planets, not chosen: over the 1.301
KP/CP of the TOI catalogue carrying both R* and depth, the implied radius has a
median of 0.89 R_Jup and a MAXIMUM of 2.20. A 2.0 cut would throw away 14 real
planets; 3.0 throws away none and still leaves 83% of the queue gated. The
margin is deliberate — our BLS is known to inflate depth, and an inflated depth
inflates the implied radius with it.
"""
from __future__ import annotations

import csv
import math
from pathlib import Path

import pytest

from exonit import stellar_gates as sg


class TestImpliedRadius:

    def test_jupiter_on_a_sun(self):
        # A Jupiter (0.1005 R☉) on a 1 R☉ star: depth = (1/9.95)² ≈ 10100 ppm.
        rp = sg.implied_companion_rjup(1.0, 10100)
        assert rp == pytest.approx(1.0, rel=0.02)

    def test_same_depth_on_a_giant_is_a_star(self):
        """The whole point: identical light curve, ten times the star."""
        rp = sg.implied_companion_rjup(10.0, 10100)
        assert rp == pytest.approx(10.0, rel=0.02)     # 10 R_Jup ≈ 1 R☉
        assert sg.implies_stellar_companion(10.0, 10100) is True
        assert sg.implies_stellar_companion(1.0, 10100) is False

    @pytest.mark.parametrize("r_star,depth", [
        (None, 10000), (1.0, None), (0, 10000), (1.0, 0),
        (float("nan"), 10000), (1.0, float("nan")), ("x", 10000),
    ])
    def test_fail_open_on_unusable_inputs(self, r_star, depth):
        """A missing catalogue entry is not evidence of a binary. Unknown must
        never gate — that would convert missing data into a verdict."""
        assert sg.implied_companion_rjup(r_star, depth) is None
        assert sg.implies_stellar_companion(r_star, depth) is False

    def test_depth_bound_is_the_exact_inverse(self):
        """max_planetary_depth_ppm is what the eager gate pushes into SQL, so it
        must agree with the per-row predicate to the last decimal — otherwise
        the batch UPDATE and the serve-time check would disagree about the same
        candidate."""
        for r_star in (0.4, 1.0, 2.7, 15.3):
            bound = sg.max_planetary_depth_ppm(r_star, max_rjup=3.0)
            assert not sg.implies_stellar_companion(r_star, bound * 0.999,
                                                    max_rjup=3.0)
            assert sg.implies_stellar_companion(r_star, bound * 1.001,
                                                max_rjup=3.0)

    def test_threshold_is_env_overridable(self, monkeypatch):
        monkeypatch.setenv("EXONIT_FPP_RP_MAX_RJUP", "2.5")
        assert sg.max_companion_rjup() == 2.5
        monkeypatch.setenv("EXONIT_FPP_RP_MAX_RJUP", "nonsense")
        assert sg.max_companion_rjup() == sg.DEFAULT_MAX_COMPANION_RJUP


class TestThresholdKeepsRealPlanets:
    """THE test. A throughput gate that costs recall is not a gate, it is a bug.
    Run against the real TOI catalogue in the repo."""

    @pytest.fixture(scope="class")
    def known_planets(self):
        cat = Path(__file__).resolve().parent.parent / "data" / "toi_catalog.csv"
        if not cat.exists():
            pytest.skip("TOI catalogue not present")
        out = []
        for row in csv.DictReader(open(cat)):
            if (row.get("TFOPWG Disposition") or "").strip() not in ("KP", "CP"):
                continue
            try:
                depth = float(row["Depth (ppm)"])
                r_star = float(row["Stellar Radius (R_Sun)"])
            except (KeyError, TypeError, ValueError):
                continue
            if depth > 0 and r_star > 0:
                out.append((row.get("TIC ID"), r_star, depth))
        if len(out) < 500:
            pytest.skip("too few usable KP/CP rows to calibrate against")
        return out

    def test_default_threshold_loses_no_known_planet(self, known_planets):
        lost = [(tic, sg.implied_companion_rjup(r, d))
                for tic, r, d in known_planets
                if sg.implies_stellar_companion(r, d)]
        assert lost == [], (
            f"the Rp-gate would discard {len(lost)} CONFIRMED planets at "
            f"{sg.max_companion_rjup()} R_Jup — recall is not negotiable. "
            f"Worst: {sorted(lost, key=lambda x: -x[1])[:3]}")

    def test_a_2_rjup_cut_would_have_lost_real_planets(self, known_planets):
        """Pins WHY the default is 3.0 and not the 2.0 that ffi_finalist_checks
        uses for its (advisory, end-of-pipeline) flag. If someone lowers the
        default, this test tells them exactly what it costs."""
        lost = [1 for _tic, r, d in known_planets
                if sg.implies_stellar_companion(r, d, max_rjup=2.0)]
        assert len(lost) >= 10, (
            "expected the 2.0 cut to visibly cost real planets (14 as measured "
            "2026-07-14); if this no longer holds, re-derive the threshold")

    def test_real_planets_sit_far_below_the_cut(self, known_planets):
        """Median implied radius of a real planet ≈ 0.9 R_Jup — the gate is not
        cutting close to the population it must keep."""
        rps = sorted(sg.implied_companion_rjup(r, d)
                     for _t, r, d in known_planets)
        median = rps[len(rps) // 2]
        assert median < 1.5
        assert max(rps) < sg.max_companion_rjup()
