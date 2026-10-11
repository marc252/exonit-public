"""Tests for EXONIT_EXOMINER_DOCKER_ROOT opt-in flag.

Verifies that:
  - When EXONIT_EXOMINER_DOCKER_ROOT=1: the docker command has NO --user and
    includes -e HOME=/tmp.
  - When EXONIT_EXOMINER_DOCKER_ROOT is unset (or any value other than "1"):
    the docker command has --user uid:gid and no -e HOME=/tmp.

These tests call _build_docker_cmd directly so no docker daemon is needed.
"""
import os

import pytest

from exonit.exominer_service import ExoMinerService


@pytest.fixture
def svc(tmp_path):
    """ExoMinerService with temp dirs, no real docker or MAST needed."""
    s = ExoMinerService(
        cache_db=str(tmp_path / "cache.db"),
        work_dir=str(tmp_path / "work"),
    )
    yield s
    s.close()


def _call_build(svc, monkeypatch, docker_root_value):
    """Set (or unset) EXONIT_EXOMINER_DOCKER_ROOT and call _build_docker_cmd."""
    if docker_root_value is None:
        monkeypatch.delenv("EXONIT_EXOMINER_DOCKER_ROOT", raising=False)
    else:
        monkeypatch.setenv("EXONIT_EXOMINER_DOCKER_ROOT", docker_root_value)

    return svc._build_docker_cmd(
        tics_csv="/tmp/tics_tbl.csv",
        out_dir="/tmp/output",
        model="exominer++_single",
        resource_flags=[],
    )


# ---------------------------------------------------------------------------
# Root mode: EXONIT_EXOMINER_DOCKER_ROOT=1
# ---------------------------------------------------------------------------

def test_root_mode_no_user_flag(svc, monkeypatch):
    cmd = _call_build(svc, monkeypatch, "1")
    assert "--user" not in cmd, "root mode must NOT include --user"


def test_root_mode_has_home_tmp(svc, monkeypatch):
    cmd = _call_build(svc, monkeypatch, "1")
    # -e HOME=/tmp must appear as consecutive tokens
    pairs = list(zip(cmd, cmd[1:]))
    assert ("-e", "HOME=/tmp") in pairs, "root mode must include -e HOME=/tmp"


def test_root_mode_uid_not_in_cmd(svc, monkeypatch):
    """Extra check: the uid:gid string must not appear anywhere."""
    cmd = _call_build(svc, monkeypatch, "1")
    uid_gid = f"{os.getuid()}:{os.getgid()}"
    assert uid_gid not in cmd, f"root mode must not embed uid:gid ({uid_gid})"


# ---------------------------------------------------------------------------
# Default mode: env var unset
# ---------------------------------------------------------------------------

def test_default_mode_has_user_flag(svc, monkeypatch):
    cmd = _call_build(svc, monkeypatch, None)
    assert "--user" in cmd, "default mode must include --user"


def test_default_mode_user_value_is_uid_gid(svc, monkeypatch):
    cmd = _call_build(svc, monkeypatch, None)
    idx = cmd.index("--user")
    expected = f"{os.getuid()}:{os.getgid()}"
    assert cmd[idx + 1] == expected, f"expected --user {expected}, got {cmd[idx+1]}"


def test_default_mode_no_home_tmp(svc, monkeypatch):
    cmd = _call_build(svc, monkeypatch, None)
    assert "HOME=/tmp" not in cmd, "default mode must NOT include HOME=/tmp"


# ---------------------------------------------------------------------------
# Non-"1" values are treated as default (not root)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("val", ["0", "false", "true", "yes", "", " "])
def test_non_one_values_treated_as_default(svc, monkeypatch, val):
    cmd = _call_build(svc, monkeypatch, val)
    assert "--user" in cmd, f"EXONIT_EXOMINER_DOCKER_ROOT={val!r} should behave as default (--user present)"
    assert "HOME=/tmp" not in cmd


# ---------------------------------------------------------------------------
# Structural: command still has required ExoMiner pipeline flags in both modes
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("root_val", [None, "1"])
def test_pipeline_flags_present_in_both_modes(svc, monkeypatch, root_val):
    cmd = _call_build(svc, monkeypatch, root_val)
    cmd_str = " ".join(cmd)
    for flag in [
        "--tic_ids_fp=/tics_tbl.csv",
        "--output_dir=/outputs",
        "--data_collection_mode=2min",
        "--download_spoc_data_products=false",
        "--stellar_parameters_source=ticv8",
        "--ruwe_source=gaiadr2",
        "--exominer_model=exominer++_single",
    ]:
        assert flag in cmd_str, f"flag {flag!r} missing from command in root_val={root_val!r}"
