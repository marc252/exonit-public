"""Pure dict-construction helpers for the verdict-classifier call.

D8 (docs/reviews/2026-07-02-opus48-cleanup-review.md): the ``candidate`` /
``star_info`` dicts that ``exonit.verdict_classifier.predict_verdict``
expects were built by hand in three places that must stay byte-identical:

- ``server.py::_run_verdict_background`` (the live path, runs on every
  strong-candidate FPP/ExoMiner result)
- ``scripts/recompute_verdict.py`` (offline recovery: recompute verdicts for
  a directed set of (tic_id, candidate_num) straight from PG, no fleet)
- ``scripts/regate_missed_spoc.py`` (offline recovery, SPOC-gate replay —
  does not currently build a candidate dict, but is named alongside the
  above in the review as an example of the same "hand-mirrored recovery
  script" pattern)

Only the *pure* dict-shaping logic lives here — no DB I/O, no threading, no
network calls, matching the review's explicit scoping.  Callers fetch the
raw row(s) themselves (server.py via ``db._query``, the scripts via
``psycopg``) and pass them in.

This module intentionally has no dependency on ``exonit.db`` /
``exonit.server`` so it can be imported from both the live server and
standalone recovery scripts without pulling in FastAPI or the DB layer.

NOTE: this module is imported by server.py today (``build_star_info``,
replacing the inline construction in ``_run_verdict_background``).
``scripts/recompute_verdict.py`` and ``scripts/regate_missed_spoc.py`` are
NOT edited by this change (a follow-up owns them) — ``build_candidate_dict``
is provided ahead of that follow-up so it can delete
``recompute_verdict.py``'s local ``_build_candidate_dict``/``_build_star_info``
and import these instead, with zero shape changes (verified against the
current hand-mirrored copy field-for-field).
"""

from __future__ import annotations

from typing import Mapping, Optional

# Fields pulled from the `stars` row. Order matters only for readability;
# `predict_verdict` / the v2.1 feature builder reads by key, not position.
# Mirrors the SELECT in server.py::_run_verdict_background and
# scripts/recompute_verdict.py's STAR_COLS (minus "tic_id", which is always
# injected separately below since it's the dict's join key, not a value
# read off the row).
STAR_INFO_FIELDS = (
    "tmag", "ra", "dec", "n_sectors", "time_span_days", "n_datapoints",
)

# Fields pulled from the `candidates` row for the verdict classifier.
# Mirrors scripts/recompute_verdict.py's CAND_COLS minus "claude_verdict"
# (recompute_verdict.py only SELECTs that column for its own transition-matrix
# bookkeeping, not because predict_verdict reads it).
CANDIDATE_FIELDS = (
    "period_days", "depth_ppm", "duration_days", "snr", "cnn_score",
    "triceratops_fpp", "triceratops_nfpp", "triceratops_prob_planet",
    "exominer_score", "exominer_ffi_score", "exominer_tce_period",
    "exominer_sector_run",
)


def build_star_info(tic_id, star_row: Optional[Mapping]) -> dict:
    """Build the ``star_info`` dict ``predict_verdict`` expects.

    ``tic_id`` is always present in the result (needed by the verdict
    classifier's Berger-catalog lookup even when no ``stars`` row exists
    yet). ``star_row`` may be ``None`` (no matching row) or any mapping with
    (a subset of) ``STAR_INFO_FIELDS`` — missing keys are simply omitted,
    matching the original inline behaviour of both call sites this was
    extracted from (server.py's ``star_row[0]`` indexing pattern and
    scripts/recompute_verdict.py's ``_build_star_info``).
    """
    star_info: dict = {"tic_id": str(tic_id)}
    if star_row:
        star_info.update({k: star_row[k] for k in STAR_INFO_FIELDS if k in star_row})
    return star_info


def build_candidate_dict(row: Mapping) -> dict:
    """Build the ``candidate`` dict ``predict_verdict`` expects from a raw
    ``candidates`` row.

    Passes values through as-is, INCLUDING the ``exominer_score == -1.0``
    no-DV cache-negative sentinel (written by the has_spoc / no-metadata /
    bulk-FFI-only gates from 2026-05-03 onward). That sentinel is NOT
    something the deployed classifiers were trained to expect — v2 (trained
    2026-04-29) and v2.1 (trained 2026-05-01) both predate every -1 writer,
    so -1 never appeared in their training data (the "-1 is a trained-on
    representation" claim in earlier commit messages was false; see
    progress/investigation-exominer-minus1-2026-07-02.md). This function
    stays a pure, unopinionated passthrough on purpose — the -1 -> "absent"
    normalization (falling back to exominer_ffi_score, matching how a real
    NULL is already handled) lives in ONE place, the classifier's own
    feature-build (``exonit.verdict_classifier_v2_1._predict_single_v2_1`` /
    ``exonit.verdict_classifier_v2._predict_single_v2``), so every caller of
    ``predict_verdict`` gets the fix for free without duplicating the -1
    check here too. See
    scripts/recompute_verdict.py::_build_candidate_dict, whose exact shape
    this mirrors so a future change can delete that copy and import this.

    Not currently called by server.py: its two live call sites already have
    a ready-made candidate dict by the time ``_run_verdict_background`` runs
    (the worker's own POST body for the FPP-result path, or a locally-shaped
    dict for the ExoMiner-result path that additionally carries ``t0`` —
    unused by the classifier but left untouched to guarantee zero behaviour
    change). Provided here so the offline recovery scripts (whose column set
    matches this exactly) can adopt it without server.py needing to change.
    """
    return {
        "tic_id": str(row["tic_id"]),
        "candidate_num": int(row["candidate_num"]),
        **{k: row[k] for k in CANDIDATE_FIELDS if k in row},
    }


def default_vetting_result() -> dict:
    """The constant "passed" vetting stub every verdict-recompute path uses.

    The verdict classifier takes a ``vetting_result`` argument but the
    backfill/recompute paths (server.py background gates, recompute_verdict.py)
    never re-run BLS vetting — they always pass ``{"passed": True}``. Named
    here so that constant has one definition instead of being retyped at
    each call site.
    """
    return {"passed": True}
