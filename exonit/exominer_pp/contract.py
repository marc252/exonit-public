"""The exact ExoMiner++ 2.0 FFI new-architecture input contract.

These 56 model inputs were extracted from the released CV-ensemble model's
``config.json`` (``models_cv/cv_iter_0/ensemble_model/ensemble_avg_model.keras``)
and cross-checked against ``tfrecord_feature_schema.txt`` and
``models/exominer_plusplus_ffi.yaml`` in NASA's repo. Each entry is the
*normalized* feature name the model's ``InputLayer`` expects, with its
(batch-free) shape.

The model concatenates every branch into a shared fully-connected head, so
the contract is all-or-nothing: feeding the model a feature dict that is
missing inputs, or has wrong shapes, is a hard error — there is no
"drop a branch for free" (see ``progress/exominerpp_setup.md`` §2).
"""
from __future__ import annotations

from typing import Dict, Mapping, Sequence, Tuple

__all__ = [
    "EXOMINER_PP_INPUTS",
    "IMAGE_BRANCHES",
    "SCALAR_INPUTS",
    "TIMESERIES_INPUTS",
    "ContractError",
    "verify_feature_dict",
    "input_names",
]


class ContractError(ValueError):
    """Raised when a feature dict violates the ExoMiner++ input contract."""


# name -> (shape excluding batch dim).
# 4 image branches (5, 55, 55); time-series views; scalar diagnostics.
EXOMINER_PP_INPUTS: Dict[str, Tuple[int, ...]] = {
    # --- 4 image branches (the ExoMiner++ pixel edge) ---
    "diff_imgs_tc_std_trainset": (5, 55, 55),
    "oot_imgs_tc_std_trainset": (5, 55, 55),
    "snr_imgs_tc_std_trainset": (5, 55, 55),
    "neighbors_imgs_tc_fixed_min_max_norm": (5, 55, 55),
    "quality": (5,),
    # --- unfolded flux ---
    "unfolded_local_flux_view_fluxnorm": (20, 31),
    "unfolded_local_flux_view_fluxnorm_var": (20, 31),
    "tce_num_transits_norm": (1,),
    "tce_num_transits_obs_norm": (1,),
    # --- global / local flux ---
    "global_flux_view_fluxnorm": (301, 1),
    "global_flux_view_fluxnorm_var": (301, 1),
    "flux_global_stat_abs_min_norm": (1,),
    "local_flux_view_fluxnorm": (31, 1),
    "local_flux_view_fluxnorm_var": (31, 1),
    "flux_local_stat_abs_min": (1,),
    # --- odd / even ---
    "local_flux_odd_view_fluxnorm": (31, 1),
    "local_flux_even_view_fluxnorm": (31, 1),
    "local_flux_odd_view_fluxnorm_var": (31, 1),
    "local_flux_even_view_fluxnorm_var": (31, 1),
    "flux_odd_local_stat_abs_min_norm": (1,),
    "flux_even_local_stat_abs_min_norm": (1,),
    # --- centroid ---
    "local_centr_view_std_noclip": (31, 1),
    "local_centr_view_std_noclip_var": (31, 1),
    # --- weak secondary ---
    "local_weak_secondary_view_selfnorm": (31, 1),
    "local_weak_secondary_view_selfnorm_var": (31, 1),
    "flux_weak_secondary_local_stat_abs_min_norm": (1,),
    "tce_maxmes_norm": (1,),
    "tce_albedo_stat_norm": (1,),
    "tce_ptemp_stat_norm": (1,),
    # --- diagnostics ---
    "tce_max_mult_ev_norm": (1,),
    "tce_max_sngle_ev_norm": (1,),
    "tce_robstat_norm": (1,),
    "tce_model_chisq_norm": (1,),
    "boot_fap_norm": (1,),
    "tce_cap_stat_norm": (1,),
    "tce_hap_stat_norm": (1,),
    # --- stellar ---
    "tce_sdens_norm": (1,),
    "tce_steff_norm": (1,),
    "tce_smet_norm": (1,),
    "tce_slogg_norm": (1,),
    "tce_smass_norm": (1,),
    "tce_sradius_norm": (1,),
    "ruwe_norm": (1,),
    "mag_shift_norm": (1,),
    # --- tce params ---
    "tce_prad_norm": (1,),
    "tce_period_norm": (1,),
    # --- difference-image sky metrics ---
    "tce_dikco_msky_norm": (1,),
    "tce_dikco_msky_err_norm": (1,),
    # --- flux trend ---
    "flux_trend_global_norm": (301, 1),
    "flux_trend_global_norm_var": (301, 1),
    "flux_trend_global_stat_min_norm": (1,),
    "flux_trend_global_stat_max_norm": (1,),
    # --- periodograms ---
    "pgram_smooth_norm": (674, 1),
    "pgram_tpm_smooth_norm": (674, 1),
    "pgram_smooth_max_power_norm": (1,),
    "pgram_tpm_smooth_max_power_norm": (1,),
}

# The 4 pixel image branches (each (5, 55, 55)) that make ExoMiner++ distinct
# from a flux-only vetter. Producible only from SPOC DV difference images +
# TIC-8 neighbor search — i.e. only for sectors WITH SPOC DV (s >= 36).
IMAGE_BRANCHES: Tuple[str, ...] = (
    "diff_imgs_tc_std_trainset",
    "oot_imgs_tc_std_trainset",
    "snr_imgs_tc_std_trainset",
    "neighbors_imgs_tc_fixed_min_max_norm",
)

SCALAR_INPUTS: Tuple[str, ...] = tuple(
    n for n, s in EXOMINER_PP_INPUTS.items() if s == (1,)
)
TIMESERIES_INPUTS: Tuple[str, ...] = tuple(
    n
    for n, s in EXOMINER_PP_INPUTS.items()
    if n not in IMAGE_BRANCHES and n != "quality" and s != (1,)
)


def input_names() -> Tuple[str, ...]:
    """The 56 model input names, in contract order."""
    return tuple(EXOMINER_PP_INPUTS.keys())


def verify_feature_dict(
    features: Mapping[str, object],
    *,
    allow_extra: bool = True,
    expected: Mapping[str, Sequence[int]] = EXOMINER_PP_INPUTS,
) -> None:
    """Validate that ``features`` satisfies the ExoMiner++ input contract.

    Each value must be array-like with a ``.shape`` (numpy array / TF tensor)
    whose trailing dims equal the contract shape. A leading batch dim is
    allowed (and ignored). Raises :class:`ContractError` on the first
    violation; returns ``None`` on success.

    Args:
        features: name -> array-like mapping (e.g. one TFRecord example
            decoded to numpy, or a batched model input dict).
        allow_extra: if False, keys not in the contract are an error.
            (TFRecords carry many auxiliary fields, so the default permits
            extras.)
        expected: the contract to check against (overridable for tests).
    """
    missing = [n for n in expected if n not in features]
    if missing:
        raise ContractError(
            f"ExoMiner++ contract: {len(missing)} required input(s) missing: "
            f"{sorted(missing)}"
        )

    if not allow_extra:
        extra = [n for n in features if n not in expected]
        if extra:
            raise ContractError(
                f"ExoMiner++ contract: unexpected input(s): {sorted(extra)}"
            )

    for name, want in expected.items():
        val = features[name]
        shape = getattr(val, "shape", None)
        if shape is None:
            raise ContractError(
                f"ExoMiner++ contract: input {name!r} is not array-like "
                f"(no .shape): {type(val).__name__}"
            )
        got = tuple(int(d) for d in shape)
        want_t = tuple(int(d) for d in want)
        # Accept either exact match, or a leading batch dim then exact match.
        if got == want_t:
            continue
        if len(got) == len(want_t) + 1 and tuple(got[1:]) == want_t:
            continue
        raise ContractError(
            f"ExoMiner++ contract: input {name!r} has shape {got}, "
            f"expected {want_t} (optionally batched)."
        )
