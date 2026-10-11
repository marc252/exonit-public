"""Tests for scripts/export_run_bundle.py.

Covers:
- --all-sectors exports candidates across mixed sector values (Fix B regression guard)
- --sector N still filters to N only (back-compat)
- error if neither --sector nor --all-sectors is given
- error if both --sector and --all-sectors are given (mutually exclusive)
"""

import json
import os
import sqlite3
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import export_run_bundle as erb  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_src_db(path: str) -> None:
    """Create a minimal source DB with candidates across sector=1 and sector=5."""
    con = sqlite3.connect(path)
    con.execute(
        "CREATE TABLE candidates "
        "(id INTEGER PRIMARY KEY, tic_id INTEGER, sector INTEGER, snr REAL)"
    )
    con.execute("CREATE TABLE stars (id INTEGER PRIMARY KEY, tic_id INTEGER)")
    # Three candidates: two in sector=1, one in sector=5.
    con.execute("INSERT INTO candidates VALUES (1, 100, 1, 15.0)")
    con.execute("INSERT INTO candidates VALUES (2, 200, 1, 12.0)")
    con.execute("INSERT INTO candidates VALUES (3, 300, 5,  8.0)")
    con.execute("INSERT INTO stars VALUES (1, 100)")
    con.execute("INSERT INTO stars VALUES (2, 200)")
    con.execute("INSERT INTO stars VALUES (3, 300)")
    con.commit()
    con.close()


# ---------------------------------------------------------------------------
# --all-sectors: exports ALL candidates regardless of sector value
# ---------------------------------------------------------------------------

def test_all_sectors_exports_candidates_from_multiple_sectors(tmp_path):
    """--all-sectors must include candidates from sector=1 AND sector=5."""
    src = str(tmp_path / "src.db")
    out = str(tmp_path / "bundle.db")
    _make_src_db(src)

    stats = erb.export_bundle(src, out, all_sectors=True)

    assert stats["candidates"] == 3, (
        f"Expected 3 candidates (all sectors), got {stats['candidates']}"
    )
    assert stats["stars"] == 3, (
        f"Expected 3 stars, got {stats['stars']}"
    )
    assert stats["sector"] == "all", (
        f"Expected sector='all' in stats, got {stats['sector']!r}"
    )

    # Verify the bundle actually contains both sectors
    con = sqlite3.connect(out)
    sectors = {row[0] for row in con.execute("SELECT DISTINCT sector FROM candidates")}
    con.close()
    assert 1 in sectors and 5 in sectors, (
        f"bundle must contain sector=1 and sector=5, got: {sectors}"
    )


def test_all_sectors_via_main(tmp_path):
    """main() with --all-sectors exports all candidates and prints valid JSON stats."""
    src = str(tmp_path / "src.db")
    out = str(tmp_path / "bundle.db")
    _make_src_db(src)

    import io
    from unittest.mock import patch

    captured = []

    def fake_print(s):
        captured.append(s)

    with patch("builtins.print", side_effect=fake_print):
        rc = erb.main([
            "--src-db", src,
            "--out", out,
            "--all-sectors",
        ])

    assert rc == 0
    assert len(captured) == 1
    stats = json.loads(captured[0])
    assert stats["candidates"] == 3
    assert stats["sector"] == "all"


# ---------------------------------------------------------------------------
# --sector N: still filters (back-compat)
# ---------------------------------------------------------------------------

def test_sector_filter_keeps_only_matching_sector(tmp_path):
    """--sector 1 must export only the 2 candidates with sector=1, not sector=5."""
    src = str(tmp_path / "src.db")
    out = str(tmp_path / "bundle.db")
    _make_src_db(src)

    stats = erb.export_bundle(src, out, sector=1)

    assert stats["candidates"] == 2, (
        f"Expected 2 candidates for sector=1, got {stats['candidates']}"
    )
    assert stats["stars"] == 2, (
        f"Expected 2 stars for sector=1, got {stats['stars']}"
    )
    assert stats["sector"] == 1

    # The sector=5 candidate must NOT be present
    con = sqlite3.connect(out)
    sectors = {row[0] for row in con.execute("SELECT DISTINCT sector FROM candidates")}
    con.close()
    assert 5 not in sectors, f"sector=5 must not appear in bundle, got: {sectors}"


def test_sector_filter_via_main(tmp_path):
    """main() with --sector 5 exports only the 1 candidate with sector=5."""
    src = str(tmp_path / "src.db")
    out = str(tmp_path / "bundle.db")
    _make_src_db(src)

    import io
    from unittest.mock import patch

    captured = []

    def fake_print(s):
        captured.append(s)

    with patch("builtins.print", side_effect=fake_print):
        rc = erb.main([
            "--src-db", src,
            "--out", out,
            "--sector", "5",
        ])

    assert rc == 0
    stats = json.loads(captured[0])
    assert stats["candidates"] == 1
    assert stats["sector"] == 5


# ---------------------------------------------------------------------------
# Error cases: neither or both --sector/--all-sectors
# ---------------------------------------------------------------------------

def test_main_error_if_neither_sector_nor_all_sectors_given(tmp_path, capsys):
    """main() must exit with error if neither --sector nor --all-sectors is given."""
    src = str(tmp_path / "src.db")
    out = str(tmp_path / "bundle.db")
    _make_src_db(src)

    with pytest.raises(SystemExit) as exc_info:
        erb.main(["--src-db", src, "--out", out])

    assert exc_info.value.code != 0, (
        "Must exit with non-zero if neither --sector nor --all-sectors is given"
    )


def test_main_error_if_both_sector_and_all_sectors_given(tmp_path, capsys):
    """main() must exit with error if both --sector and --all-sectors are given."""
    src = str(tmp_path / "src.db")
    out = str(tmp_path / "bundle.db")
    _make_src_db(src)

    with pytest.raises(SystemExit) as exc_info:
        erb.main(["--src-db", src, "--out", out, "--sector", "1", "--all-sectors"])

    assert exc_info.value.code != 0, (
        "Must exit with non-zero if both --sector and --all-sectors are given"
    )


def test_export_bundle_api_error_if_neither(tmp_path):
    """export_bundle() raises ValueError if neither sector nor all_sectors given."""
    src = str(tmp_path / "src.db")
    out = str(tmp_path / "bundle.db")
    _make_src_db(src)

    with pytest.raises(ValueError, match="sector"):
        erb.export_bundle(src, out)  # both None/False


def test_export_bundle_api_error_if_both(tmp_path):
    """export_bundle() raises ValueError if both sector and all_sectors given."""
    src = str(tmp_path / "src.db")
    out = str(tmp_path / "bundle.db")
    _make_src_db(src)

    with pytest.raises(ValueError, match="sector"):
        erb.export_bundle(src, out, sector=1, all_sectors=True)
