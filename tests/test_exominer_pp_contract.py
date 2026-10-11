"""ExoMiner++ 2.0 FFI input-contract tests.

The 56-input contract was extracted from the released CV-ensemble model and
verified (programmatically) to exactly match the model's InputLayer names +
shapes. These tests lock that contract and exercise the verifier.
"""
import numpy as np
import pytest

from exonit.exominer_pp.contract import (
    EXOMINER_PP_INPUTS,
    IMAGE_BRANCHES,
    SCALAR_INPUTS,
    ContractError,
    input_names,
    verify_feature_dict,
)


def _valid_feature_dict():
    """A contract-valid (zeros) feature dict, one example (unbatched)."""
    return {n: np.zeros(s, dtype=np.float32) for n, s in EXOMINER_PP_INPUTS.items()}


def test_contract_has_56_inputs():
    assert len(EXOMINER_PP_INPUTS) == 56
    assert len(input_names()) == 56
    assert len(set(input_names())) == 56  # all unique


def test_four_image_branches_are_55x55x5():
    assert IMAGE_BRANCHES == (
        "diff_imgs_tc_std_trainset",
        "oot_imgs_tc_std_trainset",
        "snr_imgs_tc_std_trainset",
        "neighbors_imgs_tc_fixed_min_max_norm",
    )
    for b in IMAGE_BRANCHES:
        assert EXOMINER_PP_INPUTS[b] == (5, 55, 55)


def test_scalar_inputs_count():
    # 33 of the 56 inputs are (1,) scalar diagnostics.
    assert len(SCALAR_INPUTS) == 33
    for n in SCALAR_INPUTS:
        assert EXOMINER_PP_INPUTS[n] == (1,)


def test_verify_accepts_valid_unbatched():
    verify_feature_dict(_valid_feature_dict())  # no raise


def test_verify_accepts_valid_batched():
    feats = {n: np.zeros((4,) + s, dtype=np.float32)
             for n, s in EXOMINER_PP_INPUTS.items()}
    verify_feature_dict(feats)  # leading batch dim allowed


def test_verify_rejects_missing_input():
    feats = _valid_feature_dict()
    del feats["snr_imgs_tc_std_trainset"]
    with pytest.raises(ContractError, match="missing"):
        verify_feature_dict(feats)


def test_verify_rejects_wrong_shape():
    feats = _valid_feature_dict()
    feats["diff_imgs_tc_std_trainset"] = np.zeros((5, 33, 33), dtype=np.float32)
    with pytest.raises(ContractError, match="shape"):
        verify_feature_dict(feats)


def test_verify_rejects_non_array():
    feats = _valid_feature_dict()
    feats["ruwe_norm"] = 0.0  # python float, no .shape
    with pytest.raises(ContractError, match="not array-like"):
        verify_feature_dict(feats)


def test_verify_extra_allowed_by_default_but_rejectable():
    feats = _valid_feature_dict()
    feats["some_auxiliary_tfrecord_field"] = np.zeros((1,), dtype=np.float32)
    verify_feature_dict(feats)  # allow_extra=True default
    with pytest.raises(ContractError, match="unexpected"):
        verify_feature_dict(feats, allow_extra=False)


@pytest.mark.skipif(
    not __import__("pathlib").Path(
        "~/exonit-extern/_probe_result.json"
    ).exists(),
    reason="model probe result not available",
)
def test_contract_matches_released_model_exactly():
    """If the model-probe artifact is present, the contract must match it 1:1."""
    import json
    from pathlib import Path

    probe = json.loads(
        Path("~/exonit-extern/_probe_result.json").read_text()
    )
    model_inputs = {
        k: tuple(d for d in v[1:]) for k, v in probe["inputs"].items()
    }
    contract = {k: tuple(v) for k, v in EXOMINER_PP_INPUTS.items()}
    assert set(model_inputs) == set(contract)
    for k in contract:
        assert contract[k] == model_inputs[k], k
