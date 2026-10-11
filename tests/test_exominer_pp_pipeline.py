"""ExoMiner++ FFI pipeline-driver + norm-stats config tests (pure-Python).

These lock the FFI new-architecture config overrides (the deltas from the
stock 2-min Docker pipeline) and the bundled norm-stats completeness, without
running Docker or TensorFlow.
"""
import os
from pathlib import Path

import pytest

from exonit.exominer_pp.pipeline import (
    FfiPipelineConfig,
    build_docker_cmd,
    build_ffi_config_overrides,
    image_size,
    write_tics_csv,
)


def test_ffi_image_size_is_55():
    assert image_size(FfiPipelineConfig()) == 55  # 11 * 5


def test_ffi_overrides_enable_snr_and_neighbors_and_55px():
    ov = build_ffi_config_overrides(FfiPipelineConfig())
    assert ov["config_diff_img_preprocessing.yaml"]["upscale_f"] == {"x": 5, "y": 5}
    fields = ov["config_add_diff_img_tfrecords.yaml"]["imgs_fields"]
    assert "snr_imgs" in fields and "neighbors_imgs" in fields
    assert ov["config_normalize_data.yaml"]["diff_img_data_shape"] == [5, 55, 55]


def test_docker_cmd_uses_ffi_mode_and_custom_model(tmp_path):
    tics = write_tics_csv(tmp_path / "t.csv", "123", "56-69")
    cmd = build_docker_cmd(tics, tmp_path, model="/custom_model.keras")
    assert "--data_collection_mode=ffi" in cmd
    assert "--exominer_model=/custom_model.keras" in cmd
    assert any(c.endswith(":/tics_tbl.csv:Z") for c in cmd)


def test_docker_cmd_root_mode_drops_user(tmp_path):
    tics = write_tics_csv(tmp_path / "t.csv", "123", "56-69")
    cmd = build_docker_cmd(tics, tmp_path, model="m", run_as_root=True)
    assert "--user" not in cmd
    assert "HOME=/tmp" in cmd


def test_write_tics_csv_header(tmp_path):
    p = write_tics_csv(tmp_path / "t.csv", "999", "67-67")
    lines = Path(p).read_text().splitlines()
    assert lines[0] == "tic_id,sector_run"
    assert lines[1] == "999,67-67"


@pytest.mark.skipif(
    not (Path(__file__).resolve().parents[1]
         / "data" / "exominer_pp" / "norm_stats"
         / "train_scalarparam_norm_stats.npy").exists(),
    reason="bundled norm-stats not present",
)
def test_norm_stats_load_and_completeness():
    from exonit.exominer_pp.norm_stats import check_completeness, load_norm_stats

    stats = load_norm_stats()
    # scalar stats cover all 33 ExoMiner++ scalar inputs.
    from exonit.exominer_pp.contract import SCALAR_INPUTS

    for n in SCALAR_INPUTS:
        base = n[:-5] if n.endswith("_norm") else n
        assert base in stats["scalar_params"], base
    comp = check_completeness(stats)
    # diff/oot present; snr is the documented gap (derivable).
    assert comp["diff_imgs_tc"] is True
    assert comp["oot_imgs_tc"] is True
    assert comp["snr_imgs_tc"] is False
