"""Tests for worker.exominer_infer_cli — smoke-level, no TF required."""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / "worker" / "exominer_infer_cli.py"


def test_cli_help_runs():
    r = subprocess.run([sys.executable, str(CLI), "--help"], capture_output=True, text=True)
    assert r.returncode == 0
    assert "--tfrecord" in r.stdout
    assert "--model" in r.stdout


def test_cli_missing_tfrecord_exits_2(tmp_path):
    r = subprocess.run(
        [sys.executable, str(CLI), "--tfrecord", str(tmp_path / "nope.tfrecord")],
        capture_output=True, text=True,
    )
    assert r.returncode == 2
    err = json.loads(r.stderr.splitlines()[-1])
    assert "tfrecord not found" in err["error"]


def test_cli_tfrecord_exists_but_no_tf(tmp_path, monkeypatch):
    """When TF is unavailable, the CLI still exits cleanly (code 2)."""
    fake_tfrecord = tmp_path / "fake.tfrecord"
    fake_tfrecord.write_bytes(b"\x00")
    fake_model = tmp_path / "fake.keras"
    fake_model.write_bytes(b"\x00")
    # Force the "no TF" path by pointing PYTHONPATH at a dir that has no tensorflow
    env = {"PYTHONPATH": str(tmp_path), "PATH": "/usr/bin"}
    r = subprocess.run(
        [sys.executable, str(CLI),
         "--tfrecord", str(fake_tfrecord),
         "--model", str(fake_model)],
        capture_output=True, text=True, env=env,
    )
    # If TF happens to be available locally, exit would be 3 (inference fails on fake bytes).
    # Accept both outcomes; a 0 would mean we scored garbage, which we don't want.
    assert r.returncode in (2, 3)
