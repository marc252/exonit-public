"""Tests for scripts/bench/lora_macs_verify_manifest.py's pure core."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts" / "bench"
sys.path.insert(0, str(SCRIPTS_DIR))

import lora_macs_verify_manifest as vm  # noqa: E402


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def test_verify_manifest_all_ok(tmp_path):
    data = b"hello world"
    (tmp_path / "a.bin").write_bytes(data)
    manifest = {"files": {"a.bin": {"size": len(data), "sha256": _sha(data)}}}
    assert vm.verify_manifest(manifest, tmp_path) == []


def test_verify_manifest_missing_file(tmp_path):
    manifest = {"files": {"missing.bin": {"size": 10}}}
    problems = vm.verify_manifest(manifest, tmp_path)
    assert len(problems) == 1
    assert "missing" in problems[0]


def test_verify_manifest_size_mismatch(tmp_path):
    (tmp_path / "a.bin").write_bytes(b"short")
    manifest = {"files": {"a.bin": {"size": 999}}}
    problems = vm.verify_manifest(manifest, tmp_path)
    assert "size mismatch" in problems[0]


def test_verify_manifest_sha_mismatch(tmp_path):
    data = b"hello world"
    (tmp_path / "a.bin").write_bytes(data)
    manifest = {"files": {"a.bin": {"size": len(data), "sha256": "deadbeef" * 8}}}
    problems = vm.verify_manifest(manifest, tmp_path)
    assert "sha256 mismatch" in problems[0]


def test_verify_manifest_size_mismatch_skips_sha_check(tmp_path):
    """A wrong size is already fatal — don't also compute a (guaranteed
    mismatching) sha256 of a truncated file; one problem, not two."""
    (tmp_path / "a.bin").write_bytes(b"short")
    manifest = {"files": {"a.bin": {"size": 999, "sha256": "irrelevant"}}}
    problems = vm.verify_manifest(manifest, tmp_path)
    assert len(problems) == 1
    assert "size mismatch" in problems[0]


def test_verify_manifest_no_expectations_just_existence(tmp_path):
    (tmp_path / "a.bin").write_bytes(b"anything")
    manifest = {"files": {"a.bin": {}}}
    assert vm.verify_manifest(manifest, tmp_path) == []


def test_main_ok(tmp_path, capsys):
    data = b"hello"
    (tmp_path / "a.bin").write_bytes(data)
    manifest_path = tmp_path / "MANIFEST.json"
    manifest_path.write_text(json.dumps({"files": {"a.bin": {"size": len(data), "sha256": _sha(data)}}}))
    rc = vm.main([str(manifest_path)])
    assert rc == 0
    assert "INTEGRITY OK" in capsys.readouterr().out


def test_main_failure(tmp_path, capsys):
    manifest_path = tmp_path / "MANIFEST.json"
    manifest_path.write_text(json.dumps({"files": {"missing.bin": {"size": 1}}}))
    rc = vm.main([str(manifest_path)])
    assert rc == 1
    assert "INTEGRITY FAILED" in capsys.readouterr().out
