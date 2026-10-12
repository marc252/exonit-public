"""Cross-sector recurrence helper (pure) + SQLite integration + renderers."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from exonit.cross_sector import (  # noqa: E402
    find_cross_sector_hits, render_review_queue_md, review_queue_json)


def _r(i, tic, sec, p, snr=10.0, verdict="planet_candidate"):
    return {"id": i, "tic_id": tic, "sector": sec, "period_days": p, "t0": 1.0,
            "snr": snr, "claude_verdict": verdict, "exominer_ffi_score": 0.7}


def _hit(p_other, **kw):
    rows = [_r(1, "10", 6, 3.0), _r(2, "10", 7, p_other, **kw)]
    return find_cross_sector_hits(rows, 6)


def test_direct_harmonic_subharmonic_and_miss():
    assert _hit(3.01)["10"]["matches"][0]["kind"] == "direct"
    assert _hit(6.0)["10"]["cross_sector_hits"] == 1       # 2x
    assert _hit(1.5)["10"]["cross_sector_hits"] == 1       # 0.5x
    assert _hit(3.15) == {}                                # 5 % off


def test_ignores_same_sector_and_other_tic():
    rows = [_r(1, "10", 6, 3.0), _r(2, "10", 6, 3.0), _r(3, "11", 7, 3.0)]
    assert find_cross_sector_hits(rows, 6) == {}


def test_sector1_flagged_not_excluded():
    rows = [_r(1, "10", 6, 3.0), _r(2, "10", 1, 3.0)]
    m = find_cross_sector_hits(rows, 6)["10"]["matches"][0]
    assert m["suspect_sector1"] is True
    assert _hit(3.0)["10"]["matches"][0]["suspect_sector1"] is False


def test_min_snr_and_best_and_distinct_sectors():
    rows = [_r(1, "10", 6, 3.0, snr=5), _r(2, "10", 6, 3.0, snr=20),
            _r(3, "10", 7, 3.0), _r(4, "10", 7, 6.0), _r(5, "10", 8, 3.0)]
    h = find_cross_sector_hits(rows, 6)["10"]
    assert h["cross_sector_hits"] == 2 and h["best"]["id"] == 2
    assert find_cross_sector_hits(rows, 6, min_snr=30) == {}


def test_renderers_include_likely_false_positive_tic_and_sort():
    rows = [_r(1, "10", 6, 3.0, snr=9, verdict="likely_false_positive"),
            _r(2, "10", 7, 3.0, verdict="likely_false_positive"),
            _r(3, "20", 6, 5.0, snr=30), _r(4, "20", 7, 5.0), _r(5, "20", 8, 5.0)]
    hits = find_cross_sector_hits(rows, 6)
    md = render_review_queue_md(hits, "lot", 6)
    assert "| 10 |" in md and "likely_false_positive" in md
    assert md.index("| 20 |") < md.index("| 10 |")
    js = review_queue_json(hits, "lot", 6)
    assert [t["tic_id"] for t in json.loads(json.dumps(js))["tics"]] == ["20", "10"]


def test_sqlite_integration(tmp_path):
    import discovery_postprocess as pp
    from exonit.db import ExoNitDB
    from exonit.db_backend import op_session
    db_path = str(tmp_path / "t.db")
    db = ExoNitDB(db_path, completeness_path=str(tmp_path / "c.json"))
    db.add_star("5", 0.0, 0.0, 12.0, 1, 100, 27.0, status="completed")
    db.add_candidate("5", 1, 3.00, 1500.0, 0.08, 1000.0, 9.0, sector=6,
                     claude_verdict="likely_false_positive")
    db.add_candidate("5", 1, 6.01, 1500.0, 0.08, 1000.0, 9.0, sector=7)
    con = op_session(db_path)
    try:
        rows = []
        for chunk in pp._select_in_chunks_iter(
                con, "SELECT id,tic_id,sector,period_days,t0,snr,claude_verdict,"
                "exominer_ffi_score FROM candidates WHERE tic_id IN", ["5"],
                sql_after="AND period_days IS NOT NULL"):
            rows.extend(dict(r) for r in chunk)
    finally:
        con.close()
    hits = find_cross_sector_hits(rows, 6)
    assert list(hits) == ["5"] and hits["5"]["matches"][0]["other_sector"] == 7
