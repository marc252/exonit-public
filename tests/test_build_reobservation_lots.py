import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import build_reobservation_lots as b  # noqa: E402


def _c(tic, sec=6, snr=15, em=0.8, fpp=None, v="likely_false_positive"):
    return {"tic_id": tic, "sector": sec, "snr": snr, "exominer_ffi_score": em,
            "triceratops_fpp": fpp, "claude_verdict": v}


def test_selection_thresholds_independent_of_verdict():
    rows = [_c("1"), _c("2", em=0.5), _c("3", snr=5), _c("4", fpp=0.02),
            _c("5", fpp=0.001), _c("6", sec=7), _c("7", fpp=-1.0)]
    assert b.select_source_tics(rows, 6, require_fpp=False) == ["1", "5"]
    st = {}
    assert b.select_source_tics(rows, 6, stats=st) == ["5"]
    assert st["null_fpp_excluded"] == 1
    assert b.select_source_tics(rows, 6, include_verdicts={"planet_candidate"}) == []


def test_grouping_excludes_done_and_source():
    sectors = {"1": {6, 32, 98}, "2": {6, 32}}
    g = b.group_by_target_sector(["1", "2"], sectors, {("2", 32)}, 6)
    assert g == {32: ["1"], 98: ["1"]}


def test_write_outputs(tmp_path):
    w = b.write_outputs({32: ["1", "2"]}, 6, tmp_path, "20261011T000000", {"a": 1})
    csv_p = w[0][1]
    assert csv_p.name == "reobs_s0032_from_s0006_20261011T000000.csv"
    rows = list(csv.reader(open(csv_p)))
    assert rows == [["tic_id", "label"], ["1", "candidate"], ["2", "candidate"]]
    man = json.loads(w[0][2].read_text())
    assert man["n_tics"] == 2 and man["target_sector"] == 32
    assert "approved_by:" in b.plan_snippet(32, 6, csv_p, 2)


def test_dry_run_writes_nothing(tmp_path, monkeypatch, capsys):
    from exonit.db import ExoNitDB
    dbp = str(tmp_path / "t.db")
    db = ExoNitDB(dbp, completeness_path=str(tmp_path / "c.json"))
    db.add_star("5", 0.0, 0.0, 12.0, 1, 100, 27.0, status="completed")
    db.add_candidate("5", 1, 3.0, 1.0, 0.08, 1000.0, 20.0, sector=6,
                     exominer_ffi_score=0.9)
    lot = tmp_path / "lot.csv"
    lot.write_text("tic_id,label\n5,candidate\n")

    class _MC:
        def __init__(self, *_a, **_k):
            pass

        def tglc_lookup_bulk(self, tics):
            return {t: [{"sector": 6}, {"sector": 32}] for t in tics}

    import exonit.metadata_cache as mcmod
    monkeypatch.setattr(mcmod, "MetadataCache", _MC)
    monkeypatch.delenv("EXONIT_METADATA_DB_URL", raising=False)
    out = tmp_path / "out"
    args = ["--sample", str(lot), "--sector", "6", "--db", dbp, "--out-dir", str(out),
            "--allow-null-fpp"]
    assert b.main(args) == 0
    assert "target s32: 1 TIC" in capsys.readouterr().out
    assert not out.exists()
    assert b.main(args + ["--write"]) == 0
    assert len(list(out.glob("reobs_s0032_from_s0006_*.csv"))) == 1


def test_fetch_done_pairs_reraises_non_missing_table_errors():
    class Con:
        def execute(self, *a, **k):
            raise RuntimeError("connection reset")
    import pytest
    with pytest.raises(RuntimeError):
        b._fetch_done_pairs(Con(), ["1"])


def test_fetch_done_pairs_tolerates_missing_table():
    class Con:
        def execute(self, *a, **k):
            raise RuntimeError('relation "tic_sector_state" does not exist')
    assert b._fetch_done_pairs(Con(), ["1"]) == set()
