"""Postprocess Step 1d wiring (skip / non-fatal failure / dossier line)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import discovery_postprocess as pp  # noqa: E402
from exonit.db import ExoNitDB  # noqa: E402
from exonit.db_backend import op_session  # noqa: E402


def _seed(tmp_path):
    p = str(tmp_path / "t.db")
    db = ExoNitDB(p, completeness_path=str(tmp_path / "c.json"))
    db.add_star("5", 0.0, 0.0, 12.0, 1, 100, 27.0, status="completed")
    db.add_candidate("5", 1, 3.0, 1.0, 0.1, 1000.0, 9.0, sector=6)
    db.add_candidate("5", 1, 6.01, 1.0, 0.1, 1000.0, 9.0, sector=7)
    return p


def test_skipped_without_sector(tmp_path, capsys):
    p = _seed(tmp_path)
    con = op_session(p)
    n, hits = pp._run_cross_sector_step(con, {"5"}, None, str(tmp_path / "x_dossier.md"),
                                        "lot.csv", None, 0.0)
    con.close()
    assert (n, hits) == (None, {})
    assert "--sector not given: step skipped" in capsys.readouterr().out
    assert pp._cross_sector_dossier_lines(n, hits) == []


def test_writes_files_and_dossier_line(tmp_path):
    p = _seed(tmp_path)
    con = op_session(p)
    n, hits = pp._run_cross_sector_step(con, {"5"}, 6, str(tmp_path / "x_dossier.md"),
                                        "lot.csv", None, 0.0)
    con.close()
    assert n == 1
    assert (tmp_path / "x_cross_sector.md").exists()
    assert (tmp_path / "x_cross_sector.json").exists()
    assert "Cross-sector review queue: 1 TIC(s)." in pp._cross_sector_dossier_lines(n, hits)


def test_failure_is_non_fatal(tmp_path, capsys, monkeypatch):
    p = _seed(tmp_path)
    import exonit.cross_sector as cs

    def boom(*a, **k):
        raise RuntimeError("boom")
    monkeypatch.setattr(cs, "find_cross_sector_hits", boom)
    con = op_session(p)
    n, hits = pp._run_cross_sector_step(con, {"5"}, 6, str(tmp_path / "x_dossier.md"),
                                        "lot.csv", None, 0.0)
    con.close()
    assert (n, hits) == (None, {})
    assert "WARNING step failed (non-fatal)" in capsys.readouterr().out
