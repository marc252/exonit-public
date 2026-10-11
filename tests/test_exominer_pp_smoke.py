"""ExoMiner++ 2.0 smoke inference test.

Loads the released FFI CV ensemble and runs a contract-valid forward pass,
asserting a single sigmoid score in [0, 1] per fold and a sensible CV mean.

This requires TensorFlow 2.13 + keras 2.13 (the model's pinned stack) and the
downloaded CV ensemble, so it is SKIPPED automatically when either is absent
(e.g. the default CI interpreter has no TF). Run it in the dedicated venv:

    EXONIT_EXOMINER_REPO=~/exonit-extern/ExoMiner \
    EXONIT_EXOMINER_PP_MODELS=~/exonit-extern/exominerpp_zenodo/models_cv \
    ~/exonit-extern/exominerpp_venv311/bin/python -m pytest \
        tests/test_exominer_pp_smoke.py -v
"""
import os
from pathlib import Path

import numpy as np
import pytest

from exonit.exominer_pp.contract import EXOMINER_PP_INPUTS
from exonit.exominer_pp.inference import DEFAULT_ENSEMBLE_DIR, fold_paths

tf = pytest.importorskip("tensorflow", reason="ExoMiner++ needs TF 2.13")

_HAVE_MODELS = all(p.exists() for p in fold_paths())
pytestmark = pytest.mark.skipif(
    not _HAVE_MODELS,
    reason=f"CV ensemble not found under {DEFAULT_ENSEMBLE_DIR}",
)


def _valid_batch(batch_size: int = 2):
    return {
        n: np.zeros((batch_size,) + s, dtype=np.float32)
        for n, s in EXOMINER_PP_INPUTS.items()
    }


def test_ensemble_loads_and_scores_zero_batch():
    from exonit.exominer_pp.inference import ExoMinerPPEnsemble

    ens = ExoMinerPPEnsemble().load()
    # all 5 folds loaded, each with the 56-input contract
    assert len(ens._models) == 5
    assert set(ens.input_names) == set(EXOMINER_PP_INPUTS)

    mean, folds = ens.predict(_valid_batch(2), per_fold=True)
    assert mean.shape == (2,)
    assert folds.shape == (5, 2)
    assert np.all((folds >= 0.0) & (folds <= 1.0))
    assert np.all((mean >= 0.0) & (mean <= 1.0))


def test_contract_enforced_before_inference():
    from exonit.exominer_pp.contract import ContractError
    from exonit.exominer_pp.inference import ExoMinerPPEnsemble

    ens = ExoMinerPPEnsemble().load()
    bad = _valid_batch(1)
    del bad["snr_imgs_tc_std_trainset"]
    with pytest.raises(ContractError):
        ens.predict(bad)
