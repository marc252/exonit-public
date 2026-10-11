"""Tests for scripts/precompute_sky_background.py (sky map Fase 1 + Fase 2).

See docs/superpowers/specs/2026-07-18-sky-map-phase1.md and -phase2.md.
Covers the deterministic rowid-stride sampling (ceiling + reproducibility),
the Fase 1 3-element JSON output shape, and the Fase 2 optional 4th
``dist_pc`` element (present only when a ``gaia_dist.db`` fixture exists,
null-safe otherwise) — against small temporary SQLite fixtures, never the
real data/metadata_cache.db or data/discovery/tglc_cache.
"""

from __future__ import annotations

import json
import os
import sqlite3

from precompute_sky_background import generate_background, stride_for


def _make_index(path, rows):
    """rows: list of (tic_id, sector, ra, dec, tmag) tuples."""
    con = sqlite3.connect(path)
    con.execute(
        "CREATE TABLE tglc (tic_id TEXT, sector INTEGER, path TEXT, "
        "exptime REAL, tmag REAL, ra REAL, dec REAL)")
    con.executemany(
        "INSERT INTO tglc (tic_id, sector, path, exptime, tmag, ra, dec) "
        "VALUES (?, ?, '', 1800.0, ?, ?, ?)",
        [(tic, sector, tmag, ra, dec) for tic, sector, ra, dec, tmag in rows])
    con.commit()
    con.close()


class TestStrideFor:
    def test_under_cap_no_downsampling(self):
        assert stride_for(100, 30000) == 1

    def test_exactly_at_cap(self):
        assert stride_for(30000, 30000) == 1

    def test_over_cap_downsamples(self):
        # 100000 rows, cap 30000 -> stride = ceil(100000/30000) = 4
        assert stride_for(100_000, 30_000) == 4


class TestGenerateBackground:
    def test_writes_expected_shape_under_cap(self, tmp_path):
        index_path = str(tmp_path / "metadata_cache.db")
        rows = [(str(1000 + i), 14, 10.0 + i * 0.01, 20.0 + i * 0.01, 9.0 + i * 0.1)
                for i in range(50)]
        _make_index(index_path, rows)

        out_path = generate_background(index_path, 14, str(tmp_path / "sky_bg"))
        with open(out_path) as fh:
            payload = json.load(fh)

        assert payload["sector"] == 14
        assert payload["count"] == 50
        assert len(payload["points"]) == 50
        # 3-decimal rounding.
        assert payload["points"][0] == [10.0, 20.0, 9.0]
        assert payload["points"][1][0] == round(10.01, 3)

    def test_other_sectors_excluded(self, tmp_path):
        index_path = str(tmp_path / "metadata_cache.db")
        rows = [("1", 14, 10.0, 20.0, 9.0), ("2", 15, 11.0, 21.0, 9.5)]
        _make_index(index_path, rows)

        out_path = generate_background(index_path, 14, str(tmp_path / "sky_bg"))
        with open(out_path) as fh:
            payload = json.load(fh)
        assert payload["count"] == 1

    def test_ceiling_and_determinism_of_sampling(self, tmp_path):
        """Over the cap: sampling is bounded and reproducible across runs."""
        index_path = str(tmp_path / "metadata_cache.db")
        rows = [(str(i), 14, float(i % 360), float((i % 180) - 90), 10.0)
                for i in range(1000)]
        _make_index(index_path, rows)

        out1 = generate_background(index_path, 14, str(tmp_path / "run1"), cap=100)
        out2 = generate_background(index_path, 14, str(tmp_path / "run2"), cap=100)

        with open(out1) as fh:
            payload1 = json.load(fh)
        with open(out2) as fh:
            payload2 = json.load(fh)

        # Ceiling respected (stride = ceil(1000/100) = 10 -> ~100 rows, never
        # more than the full 1000).
        assert payload1["count"] <= 1000
        assert payload1["count"] < 1000
        # Deterministic: identical inputs -> byte-identical sample, not a
        # fresh random draw each run.
        assert payload1 == payload2

    def test_missing_index_raises(self, tmp_path):
        import pytest
        with pytest.raises(FileNotFoundError):
            generate_background(str(tmp_path / "nope.db"), 14, str(tmp_path / "out"))


class TestGenerateBackgroundWithGaiaDist:
    """Fase 2: dist_pc as a 4th point element when gaia_dist.db exists."""

    def _make_gaia_dist(self, path, sector, rows):
        """rows: list of (tic_id, dist_pc|None) tuples."""
        con = sqlite3.connect(path)
        con.execute(
            "CREATE TABLE gaia_dist (tic_id TEXT, sector INTEGER, gaia_id TEXT, "
            "parallax REAL, parallax_error REAL, dist_pc REAL, "
            "PRIMARY KEY(tic_id, sector))")
        con.executemany(
            "INSERT INTO gaia_dist (tic_id, sector, dist_pc) VALUES (?, ?, ?)",
            [(tic, sector, dist) for tic, dist in rows])
        con.commit()
        con.close()

    def test_without_gaia_dist_stays_3_element_shape(self, tmp_path):
        """No gaia_dist.db at all -> Fase 1 shape untouched (regression guard)."""
        index_path = str(tmp_path / "metadata_cache.db")
        rows = [(str(1000 + i), 14, 10.0 + i * 0.01, 20.0 + i * 0.01, 9.0 + i * 0.1)
                for i in range(5)]
        _make_index(index_path, rows)

        out_dir = str(tmp_path / "sky_bg")
        out_path = generate_background(index_path, 14, out_dir)
        with open(out_path) as fh:
            payload = json.load(fh)
        assert all(len(p) == 3 for p in payload["points"])

    def test_with_gaia_dist_adds_4th_element(self, tmp_path):
        index_path = str(tmp_path / "metadata_cache.db")
        rows = [("1000", 14, 10.0, 20.0, 9.0), ("1001", 14, 11.0, 21.0, 9.5)]
        _make_index(index_path, rows)

        out_dir = str(tmp_path / "sky_bg")
        os.makedirs(out_dir, exist_ok=True)
        self._make_gaia_dist(
            os.path.join(out_dir, "gaia_dist.db"), 14,
            [("1000", 123.4), ("1001", None)])

        out_path = generate_background(index_path, 14, out_dir)
        with open(out_path) as fh:
            payload = json.load(fh)

        by_radec = {(p[0], p[1]): p for p in payload["points"]}
        assert by_radec[(10.0, 20.0)] == [10.0, 20.0, 9.0, 123.4]
        # dist_pc NULL in the DB -> null in the JSON, never invented.
        assert by_radec[(11.0, 21.0)] == [11.0, 21.0, 9.5, None]

    def test_tic_absent_from_gaia_dist_gets_null(self, tmp_path):
        """A star with no matching gaia_dist row at all -> null, not an error."""
        index_path = str(tmp_path / "metadata_cache.db")
        _make_index(index_path, [("2000", 14, 12.0, 22.0, 10.0)])

        out_dir = str(tmp_path / "sky_bg")
        os.makedirs(out_dir, exist_ok=True)
        self._make_gaia_dist(os.path.join(out_dir, "gaia_dist.db"), 14, [])

        out_path = generate_background(index_path, 14, out_dir)
        with open(out_path) as fh:
            payload = json.load(fh)
        assert payload["points"][0] == [12.0, 22.0, 10.0, None]

    def test_explicit_gaia_dist_path_override(self, tmp_path):
        """--gaia-dist can point elsewhere than <out>/gaia_dist.db."""
        index_path = str(tmp_path / "metadata_cache.db")
        _make_index(index_path, [("3000", 14, 13.0, 23.0, 11.0)])

        out_dir = str(tmp_path / "sky_bg")
        gaia_dist_path = str(tmp_path / "elsewhere" / "custom.db")
        os.makedirs(os.path.dirname(gaia_dist_path), exist_ok=True)
        self._make_gaia_dist(gaia_dist_path, 14, [("3000", 55.5)])

        out_path = generate_background(index_path, 14, out_dir,
                                        gaia_dist_path=gaia_dist_path)
        with open(out_path) as fh:
            payload = json.load(fh)
        assert payload["points"][0] == [13.0, 23.0, 11.0, 55.5]
