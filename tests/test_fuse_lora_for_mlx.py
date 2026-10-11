"""Unit tests for the rope_theta config-flattening helper in
scripts/bench/fuse_lora_for_mlx.py (2026-09-26 MLX benchmark follow-up).

Only tests the pure JSON-patching logic (no torch/peft — those are
heavy ML deps this test suite shouldn't need to import).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "bench"))

from fuse_lora_for_mlx import patch_rope_theta_in_config  # noqa: E402


def test_flattens_default_rope_type():
    cfg = {"rope_parameters": {"rope_theta": 5000000, "rope_type": "default"}}
    out, warning = patch_rope_theta_in_config(cfg)
    assert out["rope_theta"] == 5000000
    assert warning is None


def test_noop_when_already_flat():
    cfg = {"rope_theta": 123, "rope_parameters": {"rope_theta": 456, "rope_type": "default"}}
    out, warning = patch_rope_theta_in_config(cfg)
    assert out["rope_theta"] == 123  # untouched
    assert warning is None


def test_noop_when_no_rope_parameters():
    cfg = {"hidden_size": 2560}
    out, warning = patch_rope_theta_in_config(cfg)
    assert "rope_theta" not in out
    assert warning is None


def test_warns_on_non_default_rope_type():
    cfg = {"rope_parameters": {"rope_theta": 5000000, "rope_type": "yarn", "factor": 4.0}}
    out, warning = patch_rope_theta_in_config(cfg)
    assert out["rope_theta"] == 5000000
    assert warning is not None
    assert "yarn" in warning
