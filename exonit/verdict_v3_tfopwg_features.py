"""Shared feature builder for verdict v3_tfopwg (Phase 3, block 2, Phase A).

Single source of truth for the v3_tfopwg feature vector, used by BOTH the
training side (``scripts/build_verdict_v3_tfopwg_dataset.py``,
``scripts/train_verdict_classifier_v3_tfopwg.py``) and the serving side
(``exonit/verdict_classifier_v3_tfopwg.py``) — so train and serve can never
drift apart on feature engineering (a past source of prod bugs in this repo).

Per spec (``docs/superpowers/specs/2026-09-19-verdict-v3-tfopwg-design.md``
§4), it does NOT reimplement the feature math: it delegates to the
already-deployed v2/v2.1/v2.2 preprocessing pipeline
(``exonit.verdict_classifier_v2_2.preprocess_v2_2``, which itself chains
``preprocess_v2`` → ``preprocess_v2_1`` → ``preprocess_v2_2``) and simply
selects + reorders the Phase-A subset of columns it produces. The 14 base
numeric/derived features (v2 lineage) + NFPP (v2.1) + ``depth_x_rprs``
(v2.2) are all already columns of ``preprocess_v2_2``'s output — nothing
needs to be recomputed by hand here.

Explicitly EXCLUDED vs. what v2.2 computes (spec §4 "Fora"):
  - sector one-hot columns (``sec_*``)           -> memorisation risk at n=870
  - ``exominer_score`` / ``exominer_tce_period_log`` (SPOC)  -> dead in FFI
  - ``triceratops_prob_planet``                  -> redundant w/ FPP, not in
    the spec's named Phase-A list
  - ``ffi_valid``                                -> not in the spec's named
    list either (the raw ``exominer_ffi_score`` sentinel is still cleaned to
    NaN by ``preprocess_v2_2``, just without a companion validity flag)
  - ``berger_mass``/``berger_rad``/``berger_teff_log``/``berger_logg``/
    ``berger_feh`` (Berger+2023 stellar params, v2.1) — **REMOVED 2026-09-19**
    after training surfaced label leakage: Berger+2023 GKTH is a catalog of
    KNOWN PLANET HOST stars, so cross-match PRESENCE alone (independent of
    the actual values) scored AUC 0.85 against the KP/CP-vs-FP/FA label —
    it is effectively encoding "is this TIC already a confirmed host",
    which is close to the label itself for a TOI-selected training sample.
    In discovery (non-TOI field stars) those columns are always NaN, so they
    would carry ZERO signal there and only inflate TOI-benchmark numbers.
    This module therefore never imports or calls
    ``exonit.berger_catalog`` — the underlying candidates/stars raw
    columns (``berger_mass`` etc.) are still accepted as harmless
    NaN-defaulted input (``preprocess_v2_2``/``preprocess_v2_1`` read them
    unconditionally) but are never selected into the output feature matrix.
    See ``progress/verdict_v3_tfopwg_report.md`` "Iteration 1" section for
    the leaked numbers this replaced.

FPP/NFPP ``fpp_policy`` (2026-09-19, coordinator-directed after
``progress/review-verdict-v3-tfopwg-train.md`` found the Berger-style
missingness exploit repeats on ``triceratops_fpp``: FP-promoted was 45.9% on
the FPP-present subgroup, >2x the spec ceiling, because LightGBM's default
missing-value split direction on ``triceratops_fpp``/``triceratops_nfpp``
themselves — not just the ``tri_valid``/``nfpp_valid`` flags — absorbs the
disposition-correlated missingness). The feature LIST is now a function of
an explicit ``fpp_policy`` (see :func:`feature_names_for_policy`) so the
three candidate fixes can be trained and compared side by side:

  - ``"all"``: the original Phase-A convention (FPP/NFPP values + validity
    flags). Kept only for the REJECTED baseline model
    (``models/verdict_classifier_v3_tfopwg_2026-09-19.pkl``) — do not use
    for new training.
  - ``"none"``: drop ``triceratops_fpp``/``triceratops_nfpp``/``tri_valid``/
    ``nfpp_valid`` entirely (Variant A) — same reasoning as the Berger
    removal: if presence alone carries most of the signal, using the raw
    column can't be trusted not to do the same via missing-split routing.
  - ``"present_only"``: keep FPP/NFPP as plain values, no validity flags
    (Variant B) — the CALLER is responsible for only training on
    FPP-present rows and for gating serve-time predictions on FPP presence
    (see ``exonit/verdict_classifier_v3_tfopwg.py``'s ``(no_fpp)`` path);
    this module only controls which columns exist in the feature matrix.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from exonit.verdict_classifier_v2_2 import preprocess_v2_2

# Phase A base feature list (spec §4, minus Berger — see module docstring
# "REMOVED 2026-09-19" note), in a fixed, documented order:
#   14 base numeric/derived (v2 lineage)
# + depth_x_rprs interaction (v2.2)
# NO sector one-hot, NO exominer_score (SPOC, dead in FFI), NO TOI-catalog
# fields, NO berger_* (label leakage — Berger+2023 is a known-planet-host
# catalog; cross-match presence alone was AUC 0.85 against the label).
# FPP/NFPP columns are appended per `fpp_policy` — see feature_names_for_policy.
_BASE_FEATURE_NAMES: tuple[str, ...] = (
    "period_log", "depth_log1p", "duration_days", "snr_log1p", "cnn_score",
    "exominer_ffi_score",
    "n_sectors", "time_span_days_log1p",
    "duration_per_period", "expected_transits", "depth_over_snr",
    "log_depth_over_dur", "n_datapoints",
    "depth_x_rprs",
)

FPP_POLICIES: tuple[str, ...] = ("all", "none", "present_only")

# What each fpp_policy appends after _BASE_FEATURE_NAMES. See module
# docstring "FPP/NFPP fpp_policy" section for the rationale of each.
_FPP_FEATURES_BY_POLICY: dict[str, tuple[str, ...]] = {
    "all": ("triceratops_fpp", "tri_valid", "triceratops_nfpp", "nfpp_valid"),
    "none": (),
    "present_only": ("triceratops_fpp", "triceratops_nfpp"),
}


def feature_names_for_policy(fpp_policy: str = "all") -> tuple[str, ...]:
    """Ordered feature-column tuple for a given ``fpp_policy``.

    Raises ``ValueError`` on an unknown policy — same "fail loud, never
    silently fall through" convention as
    ``exonit.verdict_classifier.predict_verdict``'s version-string handling.
    """
    if fpp_policy not in FPP_POLICIES:
        raise ValueError(
            f"fpp_policy={fpp_policy!r} is not one of {FPP_POLICIES}")
    return _BASE_FEATURE_NAMES + _FPP_FEATURES_BY_POLICY[fpp_policy]


# Backward-compatible alias: this is what "the v3_tfopwg feature list" has
# always meant before fpp_policy existed (policy "all") — the REJECTED
# baseline model was trained on exactly this. New training should go through
# feature_names_for_policy("none"/"present_only") instead; this constant is
# kept so the old model payload's ``feature_names`` list and any code that
# imported it directly keep working unchanged.
V3_TFOPWG_FEATURE_NAMES: tuple[str, ...] = feature_names_for_policy("all")

# Columns preprocess_v2_2 (transitively preprocess_v2 / preprocess_v2_1)
# unconditionally reads from its input df, even for columns we then drop
# (e.g. exominer_score) or never select (berger_* — see module docstring).
# Must be present; NaN/None is a safe default. The berger_* entries here are
# NOT a Berger-catalog dependency: they are just column names
# preprocess_v2_1 happens to read off the input df and would KeyError
# without — this module never looks them up (no berger_catalog import/call)
# and always leaves them NaN-filled, so they carry no information into
# V3_TFOPWG_FEATURE_NAMES.
_V2_2_REQUIRED_INPUT_COLS = (
    "period_days", "duration_days", "depth_ppm", "snr", "cnn_score",
    "triceratops_fpp", "exominer_score", "exominer_tce_period",
    "n_sectors", "time_span_days", "n_datapoints",
    "triceratops_nfpp", "triceratops_prob_planet",
    "berger_mass", "berger_rad", "berger_teff", "berger_logg", "berger_feh",
    "exominer_ffi_score",
)


def build_v3_tfopwg_features(df: pd.DataFrame,
                              fpp_force_nan: bool = False,
                              fpp_policy: str = "all") -> pd.DataFrame:
    """Build the v3_tfopwg feature matrix from a candidates+stars-joined df.

    ``df`` must be row-aligned to "one candidate" and have (at least) the
    raw columns candidates/stars carry in the DB:
    period_days, duration_days, depth_ppm, snr, cnn_score, triceratops_fpp,
    triceratops_nfpp, exominer_ffi_score, n_sectors, time_span_days,
    n_datapoints. Missing optional columns (including any berger_* columns
    that happen to be present — they are read but never selected into the
    output, see module docstring) default to NaN.

    Delegates to ``preprocess_v2_2`` with an empty ``top_sectors`` (no sector
    one-hot wanted; the resulting single "sec___OTHER__" dummy column is
    simply not selected), then reindexes to
    ``feature_names_for_policy(fpp_policy)``.

    ``fpp_force_nan``: ablation switch (spec §4 "ablation: FPP forced to NaN")
    — when True, ``triceratops_fpp``/``triceratops_nfpp`` are blanked
    *before* preprocessing, so the model sees them exactly as it would a
    genuinely-missing reading (value NaN, validity flag 0). Used only to
    measure how much the promotion decision depends on FPP/NFPP being
    present at all — never to change how a real candidate is scored. A
    no-op under ``fpp_policy="none"`` (no FPP columns are selected either way).

    ``fpp_policy``: which FPP/NFPP columns to include — see
    :func:`feature_names_for_policy` and the module docstring's "FPP/NFPP
    fpp_policy" section.
    """
    work = df.copy()
    for col in _V2_2_REQUIRED_INPUT_COLS:
        if col not in work.columns:
            work[col] = np.nan
    if fpp_force_nan:
        work["triceratops_fpp"] = np.nan
        work["triceratops_nfpp"] = np.nan

    full = preprocess_v2_2(work, top_sectors=())
    return full[list(feature_names_for_policy(fpp_policy))].reset_index(drop=True)


def candidate_row_dict(candidate: dict, star_info: dict | None = None) -> dict:
    """Map a (candidate, star_info) pair of dicts onto preprocess_v2_2's raw
    input column names — the single place serving-time key lookups happen,
    shared by :func:`build_v3_tfopwg_features_single` and any caller that
    wants the raw row before feature-building (e.g. dataset construction
    from a matched DB row already shaped this way).

    Mirrors ``exonit.verdict_classifier_v2_2._predict_single_v2_2_stages``'s
    row assembly so serving and training never diverge on how candidate/
    star_info dict keys map onto ``preprocess_v2_2``'s expected input.
    """
    si = star_info or {}
    cand = candidate or {}
    return {
        "period_days": cand.get("period_days"),
        "depth_ppm": cand.get("depth_ppm"),
        "duration_days": cand.get("duration_days"),
        "snr": cand.get("snr"),
        "cnn_score": cand.get("cnn_score"),
        "triceratops_fpp": cand.get("triceratops_fpp", -1),
        "triceratops_nfpp": cand.get("triceratops_nfpp", -1),
        "triceratops_prob_planet": cand.get("triceratops_prob_planet"),
        "exominer_score": None,
        "exominer_ffi_score": cand.get("exominer_ffi_score"),
        "exominer_tce_period": None,
        "n_sectors": si.get("n_sectors"),
        "time_span_days": si.get("time_span_days"),
        "n_datapoints": si.get("n_datapoints") or cand.get("n_datapoints"),
        # No berger_* keys here on purpose (2026-09-19 leakage removal, see
        # module docstring) — build_v3_tfopwg_features NaN-defaults them via
        # _V2_2_REQUIRED_INPUT_COLS, and they are never selected into the
        # output regardless.
    }


def build_v3_tfopwg_features_single(candidate: dict, star_info: dict | None = None,
                                     fpp_force_nan: bool = False,
                                     fpp_policy: str = "all") -> pd.DataFrame:
    """Build a 1-row feature DataFrame for a single candidate (serving path)."""
    row = candidate_row_dict(candidate, star_info)
    return build_v3_tfopwg_features(
        pd.DataFrame([row]), fpp_force_nan=fpp_force_nan, fpp_policy=fpp_policy)


__all__ = [
    "V3_TFOPWG_FEATURE_NAMES",
    "FPP_POLICIES",
    "feature_names_for_policy",
    "build_v3_tfopwg_features",
    "build_v3_tfopwg_features_single",
    "candidate_row_dict",
]
