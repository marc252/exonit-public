"""Tests for target catalog management."""

import csv
import pytest
from exonit.targets import parse_toi_catalog, get_next_targets
from exonit.db import ExoNitDB


@pytest.fixture
def sample_csv(tmp_path):
    """Sample CSV using ExoFOP format (real column names)."""
    csv_path = str(tmp_path / "toi_catalog.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["TIC ID", "TOI", "RA", "Dec", "TESS Mag",
                          "TFOPWG Disposition", "Stellar Radius (R_Sun)",
                          "Period (days)"])
        writer.writerow(["150428135", "700.01", "97.65", "-65.58", "13.1",
                          "CP", "0.42", "16.05"])
        writer.writerow(["231670397", "175.01", "120.5", "-40.2", "9.8",
                          "KP", "1.1", "3.24"])
        writer.writerow(["999999999", "9999.01", "50.0", "10.0", "8.5",
                          "", "", ""])
    return csv_path


@pytest.fixture
def db(tmp_path):
    db_path = str(tmp_path / "test.db")
    database = ExoNitDB(db_path)
    yield database
    database.close()


class TestTargets:
    def test_parse_toi_catalog(self, sample_csv):
        targets = parse_toi_catalog(sample_csv)
        assert len(targets) == 3
        assert targets[0]["tic_id"] == "150428135"
        assert targets[0]["tmag"] == 13.1

    def test_get_next_targets_skips_analyzed(self, sample_csv, db):
        targets = parse_toi_catalog(sample_csv)
        db.add_star("150428135", ra=0, dec=0, tmag=10.0,
                    n_sectors=1, n_datapoints=1000, time_span_days=30.0)
        next_targets = get_next_targets(targets, db, max_targets=10)
        tic_ids = [t["tic_id"] for t in next_targets]
        assert "150428135" not in tic_ids
        assert "231670397" in tic_ids

    def test_get_next_targets_respects_limit(self, sample_csv, db):
        targets = parse_toi_catalog(sample_csv)
        next_targets = get_next_targets(targets, db, max_targets=1)
        assert len(next_targets) == 1

    def test_get_next_targets_sorts_by_brightness(self, sample_csv, db):
        targets = parse_toi_catalog(sample_csv)
        next_targets = get_next_targets(targets, db, max_targets=3)
        tmags = [t["tmag"] for t in next_targets]
        assert tmags == sorted(tmags)
