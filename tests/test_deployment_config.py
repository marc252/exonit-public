"""Tests for deployment/sector configuration (Component 1).

These cover the single-codebase home/AWS split: deployment role, the active
sector derived from the registry (with env override), and the worker token —
all read live from the environment so a process can be home or AWS by config.
"""
import json
import os

import pytest

from exonit import config as cfg
from exonit import discovery_state as ds


# ----------------------------- deployment() ----------------------------- #

def test_deployment_defaults_home(monkeypatch):
    monkeypatch.delenv("EXONIT_DEPLOYMENT", raising=False)
    assert cfg.deployment() == "home"


def test_deployment_aws(monkeypatch):
    monkeypatch.setenv("EXONIT_DEPLOYMENT", "AWS")  # case-insensitive
    assert cfg.deployment() == "aws"


# ----------------------------- worker_token() --------------------------- #

def test_worker_token_default(monkeypatch):
    monkeypatch.delenv("EXONIT_WORKER_TOKEN", raising=False)
    assert cfg.worker_token() == "exonit-default-token"


def test_worker_token_from_env(monkeypatch):
    monkeypatch.setenv("EXONIT_WORKER_TOKEN", "exonit-bigrun-abc")
    assert cfg.worker_token() == "exonit-bigrun-abc"


# ----------------------------- active_sector() -------------------------- #

def test_active_sector_from_registry(tmp_path, monkeypatch):
    monkeypatch.delenv("EXONIT_SECTOR", raising=False)
    reg = tmp_path / "sectors.json"
    reg.write_text(json.dumps({"active_sector": 2, "sectors": {"2": {"enabled": True}}}))
    assert ds.active_sector(str(reg)) == 2


def test_active_sector_env_override(tmp_path, monkeypatch):
    reg = tmp_path / "sectors.json"
    reg.write_text(json.dumps({"active_sector": 2, "sectors": {}}))
    monkeypatch.setenv("EXONIT_SECTOR", "5")
    # env override wins over registry
    assert ds.active_sector(str(reg)) == 5


def test_active_sector_missing_registry_defaults_1(tmp_path, monkeypatch):
    monkeypatch.delenv("EXONIT_SECTOR", raising=False)
    missing = tmp_path / "does_not_exist.json"
    assert ds.active_sector(str(missing)) == 1


def test_active_sector_explicit_override_arg(tmp_path, monkeypatch):
    monkeypatch.delenv("EXONIT_SECTOR", raising=False)
    reg = tmp_path / "sectors.json"
    reg.write_text(json.dumps({"active_sector": 1, "sectors": {}}))
    assert ds.active_sector(str(reg), env_override="3") == 3
