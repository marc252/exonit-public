"""Tests for worker.exominer_client — HTTP + subprocess plumbing, no TF."""
import json
import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "worker"))

from exominer_client import (
    _match_period,
    fetch_tfrecord,
    run_inference_cli,
    score_candidates,
)


# ---------- _match_period ----------

def test_match_period_direct():
    tces = [{"tce_num": 1, "period": 14.64, "score": 0.9}]
    m = _match_period(tces, 14.64)
    assert m["tce_num"] == 1


def test_match_period_half_alias():
    tces = [{"tce_num": 1, "period": 10.0, "score": 0.8}]
    assert _match_period(tces, 5.0)["tce_num"] == 1


def test_match_period_double_alias():
    tces = [{"tce_num": 1, "period": 5.0, "score": 0.8}]
    assert _match_period(tces, 10.0)["tce_num"] == 1


def test_match_period_no_match():
    tces = [{"tce_num": 1, "period": 100.0, "score": 0.9}]
    assert _match_period(tces, 5.0) is None


def test_match_period_empty_list():
    assert _match_period([], 5.0) is None


def test_match_period_3x_alias():
    """ExoMiner TCE at 4.817d, BLS detected the same transit at 1.6057d
    (3× alias). Real-world regression from TIC 300871545 cand 2 — without
    the 3× factor the backfill marked it -1 and lost a 5965 ppm SNR=785
    candidate. tol=2%, 1.6057*3=4.8171, relerr ≈ 5e-5 → matches."""
    tces = [{"tce_num": 1, "period": 4.8173, "score": 0.5}]
    m = _match_period(tces, 1.6057)
    assert m is not None
    assert m["tce_num"] == 1


def test_match_period_1_over_3_alias():
    tces = [{"tce_num": 1, "period": 1.5, "score": 0.3}]
    m = _match_period(tces, 4.5)
    assert m is not None
    assert m["tce_num"] == 1


def test_match_period_4x_alias():
    """ExoMiner TCE at 2.4998d, BLS at 9.9994d (4× alias). Real-world
    regression from TIC 388104525 (all 4 candidates)."""
    tces = [{"tce_num": 1, "period": 2.4998, "score": 0.5}]
    m = _match_period(tces, 9.9994)
    assert m is not None
    assert m["tce_num"] == 1


def test_match_period_1_over_4_alias():
    tces = [{"tce_num": 1, "period": 8.0, "score": 0.5}]
    m = _match_period(tces, 2.0)
    assert m is not None
    assert m["tce_num"] == 1


def test_match_period_picks_minimum_error_when_multiple_factors_apply():
    """Two TCEs, one at exact match and one at a far alias — the exact
    match must win so the score we attribute is the right one."""
    tces = [
        {"tce_num": 1, "period": 10.0, "score": 0.1},   # 1× — exact
        {"tce_num": 2, "period": 30.001, "score": 0.9}, # 1/3× of 10
    ]
    m = _match_period(tces, 10.0)
    assert m is not None
    assert m["tce_num"] == 1, "exact match must beat the 3× alias"


# ---------- fetch_tfrecord ----------

def test_fetch_tfrecord_returns_path_on_200(tmp_path):
    fake_resp = MagicMock()
    fake_resp.status_code = 200
    fake_resp.headers = {"x-sector-run": "1-69"}
    fake_resp.iter_content.return_value = [b"\x00\x01\x02"]

    with patch("exominer_client.requests.get", return_value=fake_resp):
        got = fetch_tfrecord(
            server_url="http://server:8400",
            tic_id="382437043",
            token="tok",
            dest_dir=str(tmp_path),
        )
    assert got is not None
    assert got["sector_run"] == "1-69"
    assert os.path.exists(got["path"])
    assert open(got["path"], "rb").read() == b"\x00\x01\x02"


def test_fetch_tfrecord_returns_none_on_202(tmp_path):
    fake_resp = MagicMock()
    fake_resp.status_code = 202
    with patch("exominer_client.requests.get", return_value=fake_resp):
        got = fetch_tfrecord("http://s", "42", "tok", str(tmp_path))
    assert got is None


def test_fetch_tfrecord_returns_none_on_404(tmp_path):
    fake_resp = MagicMock()
    fake_resp.status_code = 404
    with patch("exominer_client.requests.get", return_value=fake_resp):
        got = fetch_tfrecord("http://s", "42", "tok", str(tmp_path))
    assert got is None


def test_fetch_tfrecord_returns_none_on_network_error(tmp_path):
    import requests

    with patch("exominer_client.requests.get",
               side_effect=requests.ConnectionError("down")):
        got = fetch_tfrecord("http://s", "42", "tok", str(tmp_path))
    assert got is None


# ---------- run_inference_cli ----------

def test_run_inference_cli_parses_json_tces():
    fake = MagicMock()
    fake.returncode = 0
    fake.stdout = json.dumps({
        "tic_id": "1",
        "tces": [{"tce_num": 1, "period": 14.64, "score": 0.87}],
    })
    fake.stderr = ""
    with patch("exominer_client.subprocess.run", return_value=fake):
        tces = run_inference_cli("/py", "/cli.py", "/tfr", "1")
    assert tces and tces[0]["score"] == pytest.approx(0.87)


def test_run_inference_cli_returns_none_on_nonzero_exit():
    fake = MagicMock(returncode=1, stdout="", stderr="boom")
    with patch("exominer_client.subprocess.run", return_value=fake):
        assert run_inference_cli("/py", "/cli.py", "/tfr", "1") is None


def test_run_inference_cli_returns_none_on_timeout():
    import subprocess as sp
    with patch("exominer_client.subprocess.run",
               side_effect=sp.TimeoutExpired(cmd=["x"], timeout=5)):
        assert run_inference_cli("/py", "/cli.py", "/tfr", "1", timeout=5) is None


# ---------- score_candidates ----------

def test_score_candidates_decorates_matched_candidate(tmp_path):
    candidates = [
        {"candidate_num": 1, "period_days": 14.64, "cnn_score": 0.99},
        {"candidate_num": 2, "period_days": 500.0, "cnn_score": 0.02},
    ]
    fake_resp = MagicMock()
    fake_resp.status_code = 200
    fake_resp.headers = {"x-sector-run": "1-69"}
    fake_resp.iter_content.return_value = [b"\x00\x01"]
    fake_run = MagicMock(returncode=0,
                         stdout=json.dumps({"tces": [{"tce_num": 1, "period": 14.64, "score": 0.87}]}),
                         stderr="")

    with patch("exominer_client.requests.get", return_value=fake_resp), \
         patch("exominer_client.subprocess.run", return_value=fake_run):
        summary = score_candidates(
            tic_id="382437043",
            candidates=candidates,
            server_url="http://s",
            token="t",
            venv_python="/py",
            cli_path="/cli.py",
            work_dir=str(tmp_path),
        )

    assert summary["n_matched"] == 1
    assert summary["n_tces"] == 1
    assert candidates[0]["exominer_score"] == pytest.approx(0.87)
    assert candidates[0]["exominer_tce_period"] == pytest.approx(14.64)
    assert candidates[0]["exominer_sector_run"] == "1-69"
    # Unmatched candidate stays untouched
    assert "exominer_score" not in candidates[1]


def test_score_candidates_skips_on_server_miss(tmp_path):
    candidates = [{"candidate_num": 1, "period_days": 5.0}]
    fake_resp = MagicMock(status_code=404)
    with patch("exominer_client.requests.get", return_value=fake_resp):
        summary = score_candidates(
            tic_id="42",
            candidates=candidates,
            server_url="http://s", token="t",
            venv_python="/py", cli_path="/cli.py",
            work_dir=str(tmp_path),
        )
    assert summary is None
    assert "exominer_score" not in candidates[0]


def test_score_candidates_empty_candidates_no_op():
    assert score_candidates(
        tic_id="1", candidates=[],
        server_url="http://s", token="t",
        venv_python="/py", cli_path="/cli.py",
    ) is None
