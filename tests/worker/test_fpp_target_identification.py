"""Regression tests for FPP target star identification.

Bug discovered 2026-04-20: ``_query_tic_stars`` blindly took ``stars[0]``
as the target, but the MAST ``Catalogs.query_region`` result is NOT
sorted by distance from the query center. Roughly 40% of the baseline-100
sample (39/98 TICs) had a random faint neighbour at index 0 with
missing catalog data, and FPP skipped with "missing stellar params"
even though the real target — with complete M/R/Teff — was further
down in the returned list.

Evidence captured live against worker-default on 2026-04-20:
  - TIC 391958006 → target was at stars[143] of 171 (!)
  - TIC 49705089  → target was at stars[4]   of  21
In both cases stars[0] was a ~Tmag 18 blob with NaN params.

The fix promotes the entry whose ``ID`` matches the queried ``tic_id``
to position 0 before returning.
"""
from __future__ import annotations

from unittest.mock import patch, MagicMock

import numpy as np


def _fake_nearby_response(stars, gal_b=15.0, source="mast"):
    """Build a requests.Response-ish MagicMock for /api/nearby."""
    m = MagicMock()
    m.status_code = 200
    m.json.return_value = {
        "tic_id": "391958006",
        "stars": stars,
        "gal_b": gal_b,
        "source": source,
    }
    return m


def test_target_promoted_when_not_at_index_0():
    """Regression: when the real target is buried deep in the list, it
    must be promoted to ``stars[0]`` so downstream FPP math uses the
    correct host star."""
    import fpp_native

    decoys_before = [
        # Faint neighbours that Catalogs.query_region happens to list first.
        {"ID": "1897299026", "ra": 291.77, "dec": 1.38,
         "mass": None, "rad": 1.43, "Teff": 3970.0,
         "Tmag": 18.95, "plx": -0.27},
        {"ID": "1897299152", "ra": 291.77, "dec": 1.38,
         "mass": None, "rad": None, "Teff": None,
         "Tmag": 18.60, "plx": None},
    ]
    target = {
        "ID": "391958006", "ra": 291.777, "dec": 1.384,
        "mass": 0.972, "rad": 0.937, "Teff": 5529.0,
        "Tmag": 11.67, "plx": 1.5,
    }
    decoys_after = [
        {"ID": "1897299155", "ra": 291.77, "dec": 1.38,
         "mass": 0.91, "rad": 0.73, "Teff": 5283.0,
         "Tmag": 18.33, "plx": None},
    ]
    nearby = decoys_before + [target] + decoys_after

    with patch("requests.get", return_value=_fake_nearby_response(nearby)):
        result = fpp_native._query_tic_stars(
            "391958006", server_url="http://fake:8400", token="tok")

    assert result is not None
    stars, gal_b = result
    assert stars[0]["ID"] == "391958006"
    assert stars[0]["mass"] == 0.972
    assert stars[0]["rad"] == 0.937
    assert stars[0]["Teff"] == 5529.0
    # All of the original entries must still be present — we only reorder.
    ids = {str(s["ID"]) for s in stars}
    assert ids == {"391958006", "1897299026", "1897299152", "1897299155"}


def test_target_already_at_index_0_is_noop():
    """Happy path: when MAST already returns the target first (the
    ~60% case), the list must pass through unchanged."""
    import fpp_native

    target = {
        "ID": "123456789", "ra": 10.0, "dec": -5.0,
        "mass": 1.0, "rad": 1.0, "Teff": 5777.0,
        "Tmag": 10.5, "plx": 5.0,
    }
    neighbour = {
        "ID": "987654321", "ra": 10.01, "dec": -5.0,
        "mass": 0.8, "rad": 0.85, "Teff": 5200.0,
        "Tmag": 15.0, "plx": None,
    }

    with patch("requests.get",
               return_value=_fake_nearby_response([target, neighbour])):
        stars, _ = fpp_native._query_tic_stars(
            "123456789", server_url="http://fake:8400", token="tok")

    assert stars[0]["ID"] == "123456789"
    assert stars[1]["ID"] == "987654321"


def test_target_missing_falls_back_to_stars0_with_warning(caplog):
    """Edge case: target TIC is not in its own region query result.
    This shouldn't happen in practice but the code must not crash —
    we keep the old stars[0] behaviour and emit a warning so the
    incident is visible in the log."""
    import logging
    import fpp_native

    only_neighbours = [
        {"ID": "111", "ra": 1.0, "dec": 1.0, "mass": 0.5, "rad": 0.5,
         "Teff": 4000.0, "Tmag": 14.0, "plx": 1.0},
        {"ID": "222", "ra": 1.0, "dec": 1.0, "mass": None, "rad": None,
         "Teff": None, "Tmag": 18.0, "plx": None},
    ]

    with caplog.at_level(logging.WARNING, logger="exonit"):
        with patch("requests.get",
                   return_value=_fake_nearby_response(only_neighbours)):
            stars, _ = fpp_native._query_tic_stars(
                "999999", server_url="http://fake:8400", token="tok")

    # Fallback: first star stays first — but we flagged the problem.
    assert stars[0]["ID"] == "111"
    assert any("not found in its own" in rec.message for rec in caplog.records)


def test_nan_conversion_still_applied_after_promotion():
    """The null→NaN conversion pass must still touch the promoted
    target entry (so downstream np.isnan(M_s) works)."""
    import fpp_native

    stars = [
        {"ID": "decoy", "ra": 0.0, "dec": 0.0, "mass": None, "rad": None,
         "Teff": None, "Tmag": 20.0, "plx": None},
        {"ID": "42", "ra": 0.0, "dec": 0.0, "mass": 1.1, "rad": 1.05,
         "Teff": 5800.0, "Tmag": 11.0, "plx": None},
    ]

    with patch("requests.get", return_value=_fake_nearby_response(stars)):
        result, _ = fpp_native._query_tic_stars(
            "42", server_url="http://fake:8400", token="tok")

    target = result[0]
    assert target["ID"] == "42"
    # plx was None → must be NaN, not None (FPP does np.isnan on it)
    assert isinstance(target["plx"], float) and np.isnan(target["plx"])
    # Decoy also NaN-converted
    decoy = result[1]
    assert np.isnan(decoy["mass"])
    assert not np.isnan(decoy["Tmag"])
    assert decoy["Tmag"] == 20.0
