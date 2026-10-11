"""Detect and veto systematic (instrumental) periods in a batch of candidates.

A genuine transiting planet appears around ONE star with an arbitrary epoch.
An instrumental systematic (scattered light, momentum dumps, a common gap
pattern in the photometry) shows up at the SAME period in many unrelated stars,
and — crucially — with a CLUSTERED epoch (T0 mod P), because it is the same
physical event seen by all of them at once.

This module flags periods that are over-represented across unrelated stars and
whose epoch is tightly clustered. It is batch-level: it needs the whole search,
not a single star, so it runs as a post-search step (not in the per-star worker).

Discovered empirically in Discovery Run v1 (2026-05-26): P≈17.0 d across 76
unrelated stars with T0 mod 17 ≈ 9.4 d. See docs/discovery-v1-vetting-dossier.md.

Changes 2026-06-02:
  - is_vetoed: phase_tol is now a FRACTION of the period (relative), not absolute
    days. Default 0.1 means ±10% of the period. This fixes the degenerate case
    where phase_tol=4 days > period (e.g. P=0.71 d), which made the epoch check
    vacuous for short periods.
  - apply_depth_gate: new function to flag candidates whose implied planet radius
    exceeds a sub-stellar threshold (Rp > MAX_PLANET_RADIUS_RJUP). Uses stellar
    radius from the metadata cache. Conservative: missing R* → never flagged.

Changes 2026-06-02 (Q3 fix):
  - is_vetoed: period_tol is now also RELATIVE to the candidate's period for
    short periods.  For P ≥ 3 d the behaviour is unchanged (tol = 0.3 d absolute
    cap).  For P < 3 d the effective tolerance is max(0.05, 0.1 * P), which
    prevents accidental cross-period vetoes such as a P=0.51 d candidate being
    killed because it lands within 0.22 d of the P=0.729 d systematic (43% of
    its own period — clearly a different physical period, not the same one).
    The parameter ``period_tol`` is kept as an *upper cap*; callers can still
    tighten it but not widen it past the relative scale for short periods.

    Effective tolerance table (period_tol cap = 0.3):
      P = 0.50 d  →  tol = 0.050 d   (old: 0.300 d)
      P = 0.51 d  →  tol = 0.051 d   (old: 0.300 d)
      P = 0.71 d  →  tol = 0.071 d   (old: 0.300 d)
      P = 1.68 d  →  tol = 0.168 d   (old: 0.300 d)
      P = 3.00 d  →  tol = 0.300 d   (equal to cap — unchanged)
      P = 25.0 d  →  tol = 0.300 d   (equal to cap — unchanged)

    This does NOT lose legitimate vetoes because short-period systematics
    (P~0.51 d, ~0.57 d) are detected as their OWN distinct bins by
    find_systematic_periods; a candidate at those periods will still be vetoed
    by ITS OWN systematic bin + phase check (the correct mechanism).

Changes 2026-06-06 (deep-systematic veto, v2 redesign):
  - find_deep_eclipse_hosts: new function that flags TICs whose candidate set
    contains at least one deep eclipse (≥ deep_eclipse_ppm) confirmed by either
    is_vetoed (period sits on a campaign-wide systematic — the primary arm) or
    Rp > max_rp_rjup (when R* is available — the secondary/geometry arm).
    Works per-TIC across all signals (any verdict), so a deep systematic signal
    vetoes a shallow planet_candidate from the same star.  The old FPP and
    TRICERATOPS verdict arms are removed (v1): they were noisy (TIC 211353239
    had fpp=0.199 / verdict=candidate yet was a false finalist) and superseded
    by commonality (is_vetoed catches all 5 false finalists including 211353239
    because their deep signals sit on the P≈25 d epoch-clustered systematic).
    Conservative: missing depth or empty systematic_periods → no-op.  See
    docs/superpowers/specs/2026-06-06-tic-deep-eclipse-veto.md (v2).

Changes 2026-06-05 (common-period veto):
  - find_systematic_periods: new optional parameters ``common_period_min_tics``
    and ``common_period_max_period``.  When ``common_period_min_tics`` is set, any
    period bin with ≥ common_period_min_tics *distinct* TICs is flagged as
    systematic regardless of epoch clustering — but ONLY for periods strictly
    below ``common_period_max_period`` (default 2.0 d).  This catches the known
    short-period artefacts (P=0.71 d, etc.) while avoiding false-positive vetoes
    of genuine planets at longer periods (the epoch-clustered path already covers
    systematics at P>2 d when their epoch is concentrated).  Flagged entries carry
    ``"epoch_agnostic": True``; binning uses ±2 % relative tolerance
    (``_cluster_by_relative_bin``).
  - is_vetoed: ``epoch_agnostic`` systematics skip the epoch/phase check entirely;
    only the period match is required.  The matching tolerance for epoch_agnostic
    entries is a *relative* 2 % of the systematic period (symmetric with the
    cluster construction tolerance), NOT the absolute-capped
    ``_effective_period_tol``.  The epoch-clustered path is unchanged.
  - API backward-compatible: ``common_period_min_tics=None`` preserves existing
    behaviour exactly.

Changes 2026-07-03 (anti-TOI override):
  - load_toi_protection / is_toi_protected / filter_anti_toi_protected /
    partition_deep_veto_targets: SYSTEMATIC_VETO and DEEP_SYSTEMATIC_VETO must
    never demote a CONFIRMED planet (TOI TFOPWG Disposition KP/CP) — a
    confirmed planet's real period can coincide with an instrumental
    systematic purely by chance. Known victims (2026-06-30 miss-analyst, s1):
    TIC 176956893 (CP, P=6.907d), TIC 336732544 (KP, P=3.277d), TIC 364395234
    (CP, P=14.255d). See gotcha_systematic_veto_demotes_confirmed_planets and
    docs/superpowers/specs/period_systematic_veto_research.md (whose
    acceptance principle "0 detected KP/CP flagged" is exactly what this
    override enforces on the production veto). Consumed by
    scripts/discovery_postprocess.py (Steps 1 and 1c) and
    scripts/apply_systematic_veto.py.

Changes 2026-07-10 (harmonic/sub-harmonic anti-TOI protection):
  - is_toi_protected: extended from a direct-period-match-only check to also
    protect harmonic/sub-harmonic aliases of a KP/CP host's catalog period,
    using the SAME alias set and tolerance convention as
    ``exonit.scorecard.match_period`` (``HARMONIC_FACTORS_SUPER=(2,3,4,5)``,
    ``HARMONIC_FACTORS_SUB=(0.5,1/3,0.25,0.2)``, reused via import — not
    duplicated). Motivated by
    docs/reviews/meta_analysis_discovery_yield_2026-07-10.md (§Forat A + §5
    addendum): 27 % of recovered KP/CP detections in the sector-1 validation
    run (56/211) are harmonic-matched, not direct-matched, and got NO
    protection under the direct-only rule — with a real production victim,
    TIC 260004324 (catalog P=3.8143 d, BLS detected P=0.7629 d = P/5,
    demoted to ``likely_false_positive``/``SYSTEMATIC_VETO``). The
    direct-only design note below is superseded by this change; kept for
    historical context. Scope is otherwise unchanged: still KP/CP hosts
    only, still never overrides the DEPTH_GATE (a different, independent
    veto — see ``scripts/discovery_postprocess.py``'s
    ``_strong_candidate_passes``, which checks ``depth_gate_ids`` and
    returns before ``is_toi_protected`` is ever consulted).

Changes 2026-08-01 (F2, columnar population — postprocess streaming spec,
docs/superpowers/specs/2026-08-01-postprocess-streaming.md §3.B):
  - ColumnarPopulation / find_systematic_periods_columnar: additive variant
    of the campaign-wide population used by find_systematic_periods. The
    22 M-row `allc` list-of-dicts built by scripts/discovery_postprocess.py
    (H1/H7 in the spec's hot-spot inventory) costs ~13 GB just to hold
    {tic_id, period_days, t0} as Python dicts; the columnar form holds the
    same 3 values as int64/float64 numpy arrays (~24 B/row, ~530 MB at 22 M
    rows). find_systematic_periods_columnar is REQUIRED to return output
    that is bit-identical to find_systematic_periods on the same population
    (same list of dicts, same order) — see the function's own docstring for
    how each sub-path guarantees this. find_systematic_periods itself is
    UNCHANGED (not a single line touched): it remains the list-of-dicts API
    used by the existing test suite and the `--systematic-periods` frozen-
    families harness. Wiring the columnar loader into
    scripts/discovery_postprocess.py is F3, not this change — F2 lands the
    computation with parity tests only, nobody calls it yet.
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

# Harmonic/sub-harmonic alias factors for is_toi_protected (2026-07-10).
# Reused from exonit.scorecard (the single source of truth for the alias set
# used across the codebase to compare a detected period against a known/
# catalog period) rather than duplicated here. No import cycle:
# exonit/scorecard.py does not import exonit.systematic_veto (verified
# 2026-07-10), and systematic_veto is only ever imported server-side
# (scripts/*.py, tests/*) — never from worker/ — so scorecard's heavier
# import chain (pandas, exonit.db) is already paid for in every caller.
from exonit.scorecard import match_period  # noqa: E402


# ---------------------------------------------------------------------------
# Depth gate constants
# ---------------------------------------------------------------------------

# Maximum plausible planet radius in Jupiter radii. Above this the object is
# almost certainly a stellar companion (EB) or a brown dwarf, not a planet.
# 2.5 R_Jup is a conservative sub-stellar boundary (Brown+Burrows 2001; the
# empirical upper envelope for confirmed planets is ~2 R_Jup, but we leave
# margin so as not to kill hot Jupiters with noisy depth estimates).
MAX_PLANET_RADIUS_RJUP: float = 2.5

# Conversion: 1 R_sun = 9.7313 R_Jup (IAU 2015 nominal solar/Jupiter radii)
RSUN_TO_RJUP: float = 9.7313


def _circular_stats(phases: List[float], period: float) -> tuple[float, float]:
    """Return (mean_phase, R) where R is the circular concentration in [0,1].

    R≈1 means tightly clustered epoch (instrumental: same event in all stars);
    R≈0 means uniform (independent stars / noise). R is dimensionless, so it is
    comparable across periods — unlike an absolute std in days, which is
    trivially small for short periods.
    """
    if not phases:
        return 0.0, 0.0
    ang = [2 * math.pi * (p / period) for p in phases]
    c = sum(math.cos(a) for a in ang) / len(ang)
    s = sum(math.sin(a) for a in ang) / len(ang)
    r = math.hypot(c, s)
    mean_ang = math.atan2(s, c) % (2 * math.pi)
    mean_phase = mean_ang / (2 * math.pi) * period
    return mean_phase, r


def _cluster_by_relative_bin(
    candidates: List[Dict[str, Any]],
    rel_tol: float = 0.02,
) -> Dict[float, List[Dict[str, Any]]]:
    """Group candidates into period clusters using relative ±rel_tol tolerance.

    Each cluster is seeded by the first unseen period and grows by absorbing any
    subsequent period within rel_tol (fraction) of the cluster's *seed* period.
    Because candidates are processed in sorted order, a candidate belongs to the
    current (most recent) cluster when its period is within rel_tol of that
    cluster's seed period.  A new cluster is started only when the gap exceeds
    the threshold.  This guarantees O(n log n) complexity (sort + one linear
    scan) and correctly separates nearby but distinct periods (e.g. P=0.71 d and
    P=0.75 d are 5.6 % apart, safely above the 2 % threshold).

    Note: using the *seed* (first period of each cluster) rather than the
    running median avoids recomputing the median on every insertion.  For the
    small fractional tolerance used here (2 %) this is equivalent: a cluster
    whose members all lie within 2 % of the seed also have a median within 2 %
    of the seed, so no cluster is split or merged differently.

    Returns a dict mapping the cluster's median period (rounded) to the list of
    candidate dicts in that cluster.
    """
    valid = [(c, c["period_days"]) for c in candidates
             if c.get("period_days") and math.isfinite(c["period_days"]) and c["period_days"] > 0]
    if not valid:
        return {}
    # Sort ascending by period; the single linear scan below is correct because
    # a new candidate p is always ≥ seed of the current cluster.
    valid.sort(key=lambda x: x[1])

    # Each cluster: [seed_period, list_of_cands]
    cl_seed: float = valid[0][1]
    cl_cands: List[Dict[str, Any]] = [valid[0][0]]
    all_clusters: List[tuple] = []  # (seed, cands)

    for cand, p in valid[1:]:
        if (p - cl_seed) / cl_seed <= rel_tol:
            cl_cands.append(cand)
        else:
            all_clusters.append((cl_seed, cl_cands))
            cl_seed = p
            cl_cands = [cand]
    all_clusters.append((cl_seed, cl_cands))

    out: Dict[float, List[Dict[str, Any]]] = {}
    for _seed, cands in all_clusters:
        periods = sorted(c["period_days"] for c in cands)
        med = periods[len(periods) // 2]
        key = round(med, 4)
        out[key] = cands
    return out


def find_systematic_periods(
    candidates: List[Dict[str, Any]],
    min_stars: int = 10,
    period_bin: float = 0.25,
    min_concentration: float = 0.5,
    common_period_min_tics: Optional[int] = None,
    common_period_max_period: float = 2.0,
) -> List[Dict[str, Any]]:
    """Return systematic periods over-represented across unrelated stars.

    ``candidates``: dicts with keys ``tic_id``, ``period_days``, ``t0``.

    **Epoch-clustered path (existing behaviour):**
    A period bin is flagged if it contains ≥ ``min_stars`` distinct TICs AND the
    epoch (T0 mod median-period) is clustered with circular concentration
    R ≥ ``min_concentration`` (R≈0 uniform, R≈1 a single common event).
    These entries have ``"epoch_agnostic": False``.

    **Common-period path (new, opt-in):**
    When ``common_period_min_tics`` is not None, any period bin whose count of
    *distinct* TICs reaches or exceeds ``common_period_min_tics`` is **also**
    flagged, independently of epoch clustering — but ONLY for periods strictly
    below ``common_period_max_period`` (default 2.0 d).  This catches the known
    short-period artefacts (P=0.71 d, 1.68 d, etc.) whose epochs are dispersed
    and therefore escape the concentration filter.

    The 2.0 d cap prevents false-positive vetoes of genuine planets at longer
    periods: at P>2 d, K=10 barely exceeds the statistical noise floor (e.g.
    P=49 d, n=13 out of 134 k TICs is ~10 ppm) and the epoch-clustered path
    already handles systematics at those periods when the epoch is concentrated.

    Binning for the common-period path uses ±2 % relative tolerance (via
    ``_cluster_by_relative_bin``) instead of the coarser absolute ``period_bin``
    used by the epoch path, so that nearby but physically distinct short periods
    (e.g. P=0.71 d and P=0.75 d) are kept in separate clusters.

    A period detected by BOTH paths is reported once with the epoch-clustered
    entry taking precedence (``"epoch_agnostic": False``).

    These entries have ``"epoch_agnostic": True`` and no ``epoch_phase``/
    ``concentration`` keys so callers can tell them apart.

    API backward-compatible: with ``common_period_min_tics=None`` the function
    returns exactly the same results as before.
    """
    # --- epoch-clustered path (unchanged logic) ---
    by_bin: Dict[float, List[Dict[str, Any]]] = defaultdict(list)
    for c in candidates:
        p = c.get("period_days")
        if p is None or not math.isfinite(p) or p <= 0:
            continue
        key = round(round(p / period_bin) * period_bin, 3)
        by_bin[key].append(c)

    out: List[Dict[str, Any]] = []
    epoch_clustered_periods: List[float] = []  # track for dedup with common-period path
    for _key, grp in by_bin.items():
        tics = {c["tic_id"] for c in grp}
        if len(tics) < min_stars:
            continue
        periods = [c["period_days"] for c in grp]
        med_p = sorted(periods)[len(periods) // 2]
        phases = [c["t0"] % med_p for c in grp if c.get("t0") is not None]
        if len(phases) < min_stars:
            continue
        mean_phase, conc = _circular_stats(phases, med_p)
        if conc >= min_concentration:
            entry: Dict[str, Any] = {
                "period": round(med_p, 4),
                "n_stars": len(tics),
                "epoch_phase": round(mean_phase, 3),
                "concentration": round(conc, 3),
                "epoch_agnostic": False,
            }
            out.append(entry)
            epoch_clustered_periods.append(round(med_p, 4))

    # --- common-period path (new, epoch-agnostic) ---
    if common_period_min_tics is not None:
        rel_clusters = _cluster_by_relative_bin(candidates, rel_tol=0.02)
        for med_p, grp in rel_clusters.items():
            # Only flag periods below the max-period gate. At P >= common_period_max_period
            # the statistical noise floor is too close to the K threshold and genuine
            # planets are common; the epoch-clustered path handles longer systematics.
            if med_p >= common_period_max_period:
                continue
            tics = {c["tic_id"] for c in grp}
            if len(tics) < common_period_min_tics:
                continue
            # Skip if already covered by the epoch-clustered path (same period
            # within the relative 2 % tolerance used for is_vetoed matching).
            already = any(
                abs(med_p - ep) / med_p <= 0.02
                for ep in epoch_clustered_periods
            )
            if already:
                continue
            out.append({
                "period": round(med_p, 4),
                "n_stars": len(tics),
                "epoch_agnostic": True,
            })

    out.sort(key=lambda s: -s["n_stars"])
    return out


# ---------------------------------------------------------------------------
# Columnar population (2026-08-01, postprocess-streaming F2) — additive.
#
# Kills the ~13 GB list-of-dicts (`allc`) the campaign-wide population costs
# at 22 M rows (see the module changelog above). `find_systematic_periods`
# itself is NOT touched; `find_systematic_periods_columnar` below is a
# bit-identical variant that consumes a `ColumnarPopulation` instead.
# ---------------------------------------------------------------------------

# Relative tolerance used by _cluster_by_relative_bin's caller inside
# find_systematic_periods (hard-coded there, not a parameter) — mirrored here
# so the columnar common-period pre-filter can compute a provably-safe margin
# (see find_systematic_periods_columnar's docstring, common-period path).
_COMMON_PERIOD_REL_TOL: float = 0.02


@dataclass
class ColumnarPopulation:
    """Columnar {tic_id, period_days, t0} view of a candidate population.

    Three parallel numpy arrays instead of a list of dicts — ~24 B/row vs
    ~600 B/row for the equivalent Python dict, so a 22 M-row campaign
    population costs ~530 MB instead of ~13 GB (the H1/H7 hot spots in
    docs/superpowers/specs/2026-08-01-postprocess-streaming.md §1.3).

    ``tic``: int64. ``tic_id`` arrives as TEXT from the DB but is always
    numeric in production (same assumption ``_load_toi_catalog`` makes with
    its ``int(t)`` cast) — enforced at construction time in ``from_rows``.
    ``period``: float64, one-to-one with ``tic``/``t0`` by position.
    ``t0``: float64, ``NaN`` where the source row's ``t0`` was ``None``/NULL
    (mirrors ``math.isfinite(None)`` raising in the list-of-dicts code path:
    the original always short-circuits on ``is None``/``.get(...) is not
    None`` BEFORE touching the value arithmetically, so NaN — which is
    simply not finite / always fails equality / propagates through `%` as
    NaN — reproduces the same "skip this row" behaviour everywhere the
    columnar functions below inspect ``t0``).
    """

    tic: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int64))
    period: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float64))
    t0: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float64))

    def __len__(self) -> int:
        return len(self.tic)

    @classmethod
    def from_rows(
        cls,
        rows: Iterable[Dict[str, Any]],
        batch_size: int = 50_000,
    ) -> "ColumnarPopulation":
        """Build a ColumnarPopulation from a flat iterable of row dicts.

        ``rows``: an iterable of dicts with keys ``tic_id``, ``period_days``,
        ``t0`` — the exact shape ``find_systematic_periods`` accepts today
        (e.g. one row per ``SELECT tic_id, period_days, t0 FROM
        candidates``). Deliberately a FLAT iterable, not an iterable of
        batches: a caller sitting on top of a batched source (e.g.
        ``execute_stream``) chains its batches first
        (``itertools.chain.from_iterable(...)``) — keeping this constructor's
        contract simple was preferred over an ambiguous "rows-or-batches"
        signature (feedback: simplify, don't complicate).

        Accumulates into fixed-size Python-list buffers and flushes each to a
        numpy array block every ``batch_size`` rows, then ``np.concatenate``s
        all blocks once at the end — this bounds the peak to O(batch_size)
        Python objects + O(n) numpy bytes, never O(n) Python objects (the
        thing this whole change exists to avoid).

        Raises ``ValueError`` with the offending row index and value if any
        ``tic_id`` is not castable to ``int`` — the DB's TIC IDs are TEXT but
        always numeric in production (same assumption ``_load_toi_catalog``
        makes); a non-numeric value here means the caller handed this
        function the wrong column or a corrupt row, and continuing silently
        would silently drop/miscount TICs in the systematic-veto population.
        """
        tic_blocks: List[np.ndarray] = []
        period_blocks: List[np.ndarray] = []
        t0_blocks: List[np.ndarray] = []

        buf_tic: List[int] = []
        buf_period: List[float] = []
        buf_t0: List[float] = []

        def _flush() -> None:
            if not buf_tic:
                return
            tic_blocks.append(np.array(buf_tic, dtype=np.int64))
            period_blocks.append(np.array(buf_period, dtype=np.float64))
            t0_blocks.append(np.array(buf_t0, dtype=np.float64))
            buf_tic.clear()
            buf_period.clear()
            buf_t0.clear()

        for i, row in enumerate(rows):
            raw_tic = row["tic_id"]
            try:
                tic_int = int(raw_tic)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"ColumnarPopulation.from_rows: non-numeric tic_id "
                    f"{raw_tic!r} at row {i} — the campaign population is "
                    "assumed all-numeric TIC IDs (same assumption "
                    "_load_toi_catalog's int(t) cast makes)."
                ) from exc
            buf_tic.append(tic_int)
            p = row.get("period_days")
            buf_period.append(float(p) if p is not None else math.nan)
            t0 = row.get("t0")
            buf_t0.append(float(t0) if t0 is not None else math.nan)
            if len(buf_tic) >= batch_size:
                _flush()
        _flush()

        if not tic_blocks:
            return cls()
        return cls(
            tic=np.concatenate(tic_blocks),
            period=np.concatenate(period_blocks),
            t0=np.concatenate(t0_blocks),
        )


def find_systematic_periods_columnar(
    pop: ColumnarPopulation,
    min_stars: int = 10,
    period_bin: float = 0.25,
    min_concentration: float = 0.5,
    common_period_min_tics: Optional[int] = None,
    common_period_max_period: float = 2.0,
) -> List[Dict[str, Any]]:
    """Columnar-population variant of ``find_systematic_periods``.

    Same parameters, same defaults, and — REQUIRED, not best-effort —
    bit-identical output to ``find_systematic_periods(rows, ...)`` when
    ``pop`` is ``ColumnarPopulation.from_rows(rows)`` built from the same
    ``rows`` in the same order. See
    ``tests/test_systematic_veto_columnar.py`` for the parity suite this is
    verified against (synthetic edge cases + a 50 k-row seeded-random catch-
    all). ``find_systematic_periods`` itself is untouched by this change.

    **Epoch-clustered path**: only the bin key
    (``round(round(p / period_bin) * period_bin, 3)``) is computed per-row;
    everything downstream of a qualifying bin (median, phases, circular
    stats) reuses the exact same Python arithmetic as
    ``find_systematic_periods`` on the values extracted from that bin — no
    per-bin numpy reimplementation, so no risk of numpy/Python float
    divergence in the parts that actually decide the science.

    The bin key itself is computed with Python's builtin ``round()``, NOT
    ``np.round`` — empirically the two agree on this pipeline's arithmetic
    (division by ``period_bin`` then round-to-int, times ``period_bin``,
    round to 3 decimals: no case found across ~2 M random periods plus
    targeted exact-halfway values at every multiple of ``period_bin/2``), but
    no correctness proof exists for arbitrary future ``period_bin`` values —
    ``np.round`` uses a multiply/rint/divide approximation while Python's
    ``round()`` computes the true decimal value of the double before
    rounding, and the two are documented to disagree on some inputs for
    ``ndigits > 0``. Given the "bit-identical, not approximately" requirement
    here, the per-row Python ``round()`` is kept rather than risk a silent
    divergence for a case the test suite didn't happen to hit. This is the
    ONLY per-row Python-level work in this function; everything else operates
    on numpy index arrays. (F2.1, 2026-08-01: the round() output now lands in
    a pre-allocated numpy float64 array — one bounded-size Python list per
    500 k-row chunk, never a whole-population Python container — instead of
    directly building a ``Dict[float, List[int]]``, which measured a ~3 GB
    transient RSS spike on the real 23.16 M-row campaign population. See
    below for the grouping mechanism this replaced the dict with.)

    Bin *iteration* order matters for the final output: ``out.sort(key=...)``
    is a stable sort, so ties in ``n_stars`` are broken by insertion order,
    which must match ``find_systematic_periods``'s ``defaultdict`` (= order
    of first appearance of each bin key while scanning ``candidates``). The
    grouping below reproduces that WITHOUT a dict: a stable ``np.argsort`` of
    the bin-key array puts rows with an equal (bit-identical) key into one
    contiguous run, and — because the sort is stable — a run's rows stay in
    ascending original-index order, i.e. exactly the order they'd have been
    ``.append()``-ed to a ``defaultdict`` list. The run's first element is
    therefore also that bin's first-appearance position, so sorting the runs
    by that value reproduces ``by_bin.items()``'s "order bins were first
    seen" iteration order — all with numpy index arrays, no dict of Python
    lists at population scale.

    **Common-period path** (``common_period_min_tics`` not None): rows are
    pre-filtered to ``period < common_period_max_period * (1 +
    _COMMON_PERIOD_REL_TOL)`` — same provably-safe margin as before (a
    naive ``< common_period_max_period`` pre-filter can silently DROP
    members of a qualifying cluster whose median sits just under the
    threshold but whose tail crosses it, changing ``n_stars``; see
    ``tests/test_systematic_veto_columnar.py::TestCommonPeriodBoundary``).

    F2.2 (2026-08-01, memory follow-up to F2.1): the pre-filtered subset used
    to be projected into a Python list of ``{"tic_id", "period_days"}`` dicts
    and handed to ``_cluster_by_relative_bin`` unchanged. On the real 23.16
    M-row campaign population, 38 % of rows sit below the 2.04 d margin —
    8.7 M dicts, ~2.5 GB of transient Python objects, THE dominant remaining
    hot spot after F2.1. This function now reimplements
    ``_cluster_by_relative_bin``'s sort+scan directly on numpy index arrays
    for the short-period subset, never materialising more than O(n_clusters)
    Python objects (a few thousand on real data, not 8.7 M). The helper
    itself (``_cluster_by_relative_bin``) is untouched and still used by
    ``find_systematic_periods`` and its own test suite; this is a parallel,
    numpy-native reimplementation of the SAME algorithm, required to be
    bit-identical:

      1. Ordering / tie-break: ``_cluster_by_relative_bin`` builds its input
         list in population-scan order, then does a stable
         ``list.sort(key=lambda x: x[1])`` — ties (equal periods) keep their
         scan-order relative position. ``short_idx = np.nonzero(short_mask)``
         is itself in ascending original-row order (population-scan order),
         so ``np.argsort(periods[short_idx], kind="stable")`` reproduces the
         EXACT same tie-break: a stable sort on the same values in the same
         input order is the same permutation regardless of whether Python's
         Timsort or numpy's stable (mergesort-based) argsort is used — both
         are stable sorts, and stability is defined purely by input order,
         not by algorithm.
      2. Cluster-membership condition: identical floats. ``short_idx[i]``'s
         period is read via ``float(sorted_periods_arr[i])`` — a numpy
         float64 array element converted with Python's ``float()`` builtin
         yields the exact same double as the ``float(periods[i])`` cast used
         when ``ColumnarPopulation`` (and the old dict-projection path) built
         its values in the first place, so ``(p - cl_seed) / cl_seed <=
         rel_tol`` is evaluated on bit-identical Python floats, same as the
         original scan.
      3. Median / dict key: a cluster's members are a contiguous run of the
         globally-sorted array, i.e. already sorted — so
         ``sorted_periods_arr[start + (end-start)//2]`` equals
         ``sorted(c["period_days"] for c in cands)[len(cands)//2]`` exactly
         (the original re-sort of an already-sorted list is a no-op).
         ``round(median, 4)`` is Python's builtin ``round()``, same as the
         original's dict key.
      4. Dict overwrite semantics: ``_cluster_by_relative_bin`` does
         ``out[key] = cands`` — if two *different* clusters happen to round
         to the same 4-decimal median (only possible if their medians are
         within 0.0001 d, which given rel_tol=2% would already have merged
         them into one cluster for any realistic seed, but is not provably
         impossible for a pathological input), the LATER cluster's members
         win but the dict keeps the EARLIER cluster's insertion position.
         The ``key_to_range: Dict[float, tuple]`` built below is a plain
         Python dict assigned in the same ascending-cluster-seed order, so it
         has identical overwrite/insertion-order semantics.

    Peak *incremental* memory for this path is O(n_short) numpy arrays (raw
    int64/float64 doubles, ~8 B/element, no per-element Python object
    overhead) plus O(n_clusters) small Python objects — see the module's
    micro-benchmark in ``tests/test_systematic_veto_columnar.py`` /
    ``progress/impl-f22-common-period-memory.md`` for measured numbers on a
    synthetic 10 M-row population.
    """
    periods = pop.period
    tics = pop.tic
    t0s = pop.t0

    # --- epoch-clustered path (same rule as find_systematic_periods: p is
    # None/not finite/<=0 -> skip; NaN encodes None so isfinite already
    # excludes it, matching the short-circuited `p is None or not
    # math.isfinite(p)` check without ever calling isfinite(None)) ---
    valid = np.isfinite(periods) & (periods > 0)
    valid_idx = np.nonzero(valid)[0]
    n_valid = int(valid_idx.shape[0])

    # F2.1 (2026-08-01, postprocess-streaming memory follow-up): the bin key
    # (round(round(p/period_bin)*period_bin, 3)) still needs Python's
    # builtin round() per row for bit-identity (see the docstring above —
    # np.round is NOT provably equivalent for arbitrary period_bin). What
    # changed is WHERE the result lands: previously every row grew a
    # `Dict[float, List[int]]` (23.16 M Python int/float objects live at
    # once on the real campaign population — measured ~3 GB transient RSS
    # spike, the dominant cost of this whole function). Now round() writes
    # straight into a pre-allocated float64 numpy array, one bounded
    # (<=chunk_size) Python list at a time — never a population-scale
    # Python container.
    chunk_size = 500_000
    keys = np.empty(n_valid, dtype=np.float64)
    for start in range(0, n_valid, chunk_size):
        end = min(start + chunk_size, n_valid)
        chunk_periods = periods[valid_idx[start:end]]
        keys[start:end] = [
            round(round(p / period_bin) * period_bin, 3)
            for p in chunk_periods.tolist()
        ]

    # Group rows by EXACT bin-key equality using numpy, not a dict of Python
    # lists: a stable sort of `keys` puts equal keys in contiguous runs and,
    # because ties are broken by original position, each run is already in
    # ascending original-row order (== the row's position while scanning the
    # population). This reproduces both invariants a `defaultdict` gives for
    # free: (a) exact-float-equality bin membership (bit-identical keys, same
    # values as the dict path would hash/compare equal on), and (b) member
    # order within a bin == first-appearance/scan order.
    if n_valid > 0:
        sort_order = np.argsort(keys, kind="stable")
        sorted_keys = keys[sort_order]
        del keys
        boundaries = np.nonzero(np.diff(sorted_keys) != 0)[0] + 1
        starts = np.concatenate(([0], boundaries))
        ends = np.concatenate((boundaries, [n_valid]))
        del sorted_keys, boundaries
        # The FIRST element of each contiguous run is that bin's earliest
        # scan position (stable sort keeps runs in ascending original-index
        # order) — so `sort_order[starts]` IS the bin's first-appearance
        # position, no extra scan/min() needed. Sorting bins by this value
        # reproduces the defaultdict's "order of first appearance" iteration
        # order, which the final `out.sort()` (stable) relies on for
        # deterministic tie-breaking by n_stars.
        first_pos = sort_order[starts]
        appearance_order = np.argsort(first_pos, kind="stable")
        del first_pos
    else:
        sort_order = valid_idx
        starts = np.empty(0, dtype=np.int64)
        ends = np.empty(0, dtype=np.int64)
        appearance_order = np.empty(0, dtype=np.int64)

    out: List[Dict[str, Any]] = []
    epoch_clustered_periods: List[float] = []
    for g in appearance_order.tolist():
        s, e = int(starts[g]), int(ends[g])
        idxs_arr = valid_idx[sort_order[s:e]]
        tic_set = set(tics[idxs_arr].tolist())
        if len(tic_set) < min_stars:
            continue
        grp_periods = periods[idxs_arr].tolist()
        med_p = sorted(grp_periods)[len(grp_periods) // 2]
        grp_t0 = t0s[idxs_arr]
        phases = [float(v) % med_p for v in grp_t0.tolist() if not math.isnan(v)]
        if len(phases) < min_stars:
            continue
        mean_phase, conc = _circular_stats(phases, med_p)
        if conc >= min_concentration:
            entry: Dict[str, Any] = {
                "period": round(med_p, 4),
                "n_stars": len(tic_set),
                "epoch_phase": round(mean_phase, 3),
                "concentration": round(conc, 3),
                "epoch_agnostic": False,
            }
            out.append(entry)
            epoch_clustered_periods.append(round(med_p, 4))

    del sort_order, valid_idx, starts, ends, appearance_order

    # --- common-period path (new, epoch-agnostic) ---
    if common_period_min_tics is not None:
        margin = common_period_max_period * (1.0 + _COMMON_PERIOD_REL_TOL)
        short_mask = valid & (periods < margin)
        short_idx = np.nonzero(short_mask)[0]
        n_short = int(short_idx.shape[0])

        # key -> (start, end) into the SORTED short-subset arrays below.
        # Built even when n_short == 0 (stays empty) so the consumer loop
        # further down doesn't need a separate empty-case branch.
        key_to_range: Dict[float, tuple] = {}
        sort_order = np.empty(0, dtype=np.int64)

        if n_short > 0:
            # Reproduce _cluster_by_relative_bin's sort+scan on this subset
            # WITHOUT ever materialising it as a list of Python dicts (up to
            # 8.7 M rows on the real campaign population — see docstring
            # point 1 for why `short_idx`'s natural (ascending) order lets a
            # stable argsort reproduce the exact same tie-break).
            short_periods = periods[short_idx]
            sort_order = np.argsort(short_periods, kind="stable")
            sorted_periods_arr = short_periods[sort_order]
            del short_periods

            # Single linear scan over the sorted subset (docstring point 2):
            # only cluster BOUNDARIES (a handful of ints per real
            # short-period systematic, not O(n_short) objects) are retained.
            # `float(sorted_periods_arr[i])` at each step is the only
            # per-row Python object created, and it is NOT retained past the
            # next iteration (only `cl_seed`, a single scalar, survives).
            cluster_starts: List[int] = [0]
            cl_seed = float(sorted_periods_arr[0])
            for i in range(1, n_short):
                p = float(sorted_periods_arr[i])
                if (p - cl_seed) / cl_seed <= _COMMON_PERIOD_REL_TOL:
                    continue
                cluster_starts.append(i)
                cl_seed = p
            cluster_starts.append(n_short)

            # Median + dict key per cluster (docstring points 3-4): a
            # cluster's members are a contiguous run of the already-sorted
            # array, so its median is a direct index, no re-sort needed.
            for ci in range(len(cluster_starts) - 1):
                start, end = cluster_starts[ci], cluster_starts[ci + 1]
                mid = start + (end - start) // 2
                med = float(sorted_periods_arr[mid])
                key_to_range[round(med, 4)] = (start, end)
            del sorted_periods_arr, cluster_starts

        for med_p, (start, end) in key_to_range.items():
            if med_p >= common_period_max_period:
                continue
            member_pos = short_idx[sort_order[start:end]]
            tics_grp = set(tics[member_pos].tolist())
            if len(tics_grp) < common_period_min_tics:
                continue
            already = any(
                abs(med_p - ep) / med_p <= 0.02
                for ep in epoch_clustered_periods
            )
            if already:
                continue
            out.append({
                "period": round(med_p, 4),
                "n_stars": len(tics_grp),
                "epoch_agnostic": True,
            })

    out.sort(key=lambda s: -s["n_stars"])
    return out


def _effective_period_tol(period: float, period_tol_cap: float = 0.3) -> float:
    """Compute the effective period-matching tolerance for a candidate at *period*.

    For long periods (P ≥ 3 d) the result equals ``period_tol_cap`` (default
    0.3 d — identical to the old hard-coded value, so no behaviour change).
    For short periods the tolerance scales as 0.1 * P, clamped to a minimum of
    0.05 d so that a handful of extremely-short-period candidates (P < 0.5 d)
    are not left with a vanishingly small window.

    The formula is:  tol = max(0.05, min(period_tol_cap, 0.1 * period))

    This prevents accidental cross-period vetoes: a P=0.51 d candidate would
    otherwise fall within the 0.3 d window of the P=0.729 d systematic purely
    by accident (the two periods are 0.22 d apart, i.e. 43% of the candidate
    period — clearly different physical periods).
    """
    return max(0.05, min(period_tol_cap, 0.1 * period))


def is_vetoed(
    period: float,
    t0: Optional[float],
    systematics: List[Dict[str, Any]],
    period_tol: float = 0.3,
    phase_tol_frac: float = 0.1,
) -> Optional[Dict[str, Any]]:
    """Return the matching systematic if (period, t0) is vetoed, else None.

    Matches on period proximity AND epoch proximity (circular). If ``t0`` is None,
    a period match alone vetoes (conservative).

    ``period_tol`` is now treated as an *upper cap* on the period-matching
    window.  The effective window is ``_effective_period_tol(period, period_tol)``
    — relative (0.1 * P) for short periods, capped at ``period_tol`` for long
    ones.  For P ≥ 3 d the cap is always active, so behaviour is identical to
    the previous hard-coded 0.3.

    ``phase_tol_frac`` is a FRACTION of the systematic period (default 0.1 = ±10%
    of the period). This replaced the old absolute ``phase_tol=4.0`` days, which
    was larger than the period itself for P < 4 d, making the epoch check vacuous
    for short periods (any T0 would pass, so every star at that period got vetoed
    regardless of whether its epoch was actually clustered with the systematic).

    **epoch_agnostic systematics** (``s["epoch_agnostic"] is True``):
    These are common-period artefacts detected without epoch clustering. For these
    the phase check is skipped entirely: a period match alone is sufficient to
    veto, regardless of T0.  The matching tolerance uses a *relative* 2 % of the
    systematic period — symmetric with the ±2 % tolerance used when building the
    cluster — not the absolute-capped ``_effective_period_tol``.  Using the
    absolute cap (up to 0.3 d) for epoch-agnostic systematics would create an
    asymmetric 8–10 % window for P~3–4 d artefacts, silently vetoing genuine
    planets whose periods are merely 3–10 % away (no epoch guard compensates for
    epoch-agnostic entries).

    Effective period tolerance (period_tol cap = 0.3, epoch-clustered path):
      P = 0.51 d  →  period_tol = 0.051 d   (old: 0.300 d)
      P = 0.71 d  →  period_tol = 0.071 d   (old: 0.300 d)
      P = 1.68 d  →  period_tol = 0.168 d   (old: 0.300 d)
      P = 3.00 d  →  period_tol = 0.300 d   (unchanged)
      P = 25.0 d  →  period_tol = 0.300 d   (unchanged)

    Effective period tolerance (epoch-agnostic path, fixed 2 % relative):
      P = 0.71 d  →  tol = 0.0142 d   (vs 0.071 d for epoch-clustered)
      P = 1.68 d  →  tol = 0.0336 d
      P = 3.35 d  →  tol = 0.0670 d   (old absolute cap would give 0.300 d, 4.5×)

    Phase window (epoch-clustered systematics only):
      P = 0.71 d  →  ±0.071 d   (old absolute: ±4 d = entire period × 5)
      P = 25 d    →  ±2.5 d     (old: ±4 d — similar)
    """
    eff_period_tol = _effective_period_tol(period, period_tol)
    for s in systematics:
        # epoch_agnostic: use relative 2% tolerance (matches cluster construction);
        # do NOT use _effective_period_tol which can be up to 0.3 d absolute.
        if s.get("epoch_agnostic"):
            if abs(period - s["period"]) / s["period"] > 0.02:
                continue
            return s
        if abs(period - s["period"]) > eff_period_tol:
            continue
        if t0 is None:
            return s
        ph = t0 % s["period"]
        d = abs(ph - s["epoch_phase"])
        d = min(d, s["period"] - d)  # circular distance
        phase_tol = phase_tol_frac * s["period"]
        if d <= phase_tol:
            return s
    return None


def implied_planet_radius_rjup(
    depth_ppm: float,
    stellar_radius_rsun: float,
) -> float:
    """Return implied planet radius in Jupiter radii given transit depth and R*.

    Rp = sqrt(depth_ppm / 1e6) * R* [R_sun] * RSUN_TO_RJUP
    """
    rp_rsun = math.sqrt(depth_ppm / 1_000_000.0) * stellar_radius_rsun
    return rp_rsun * RSUN_TO_RJUP


def apply_depth_gate(
    candidates: List[Dict[str, Any]],
    stellar_radii: Dict[str, float],
    max_rp_rjup: float = MAX_PLANET_RADIUS_RJUP,
) -> List[Dict[str, Any]]:
    """Return candidates flagged as EB/FP because their implied Rp > max_rp_rjup.

    ``candidates``: dicts with keys ``tic_id``, ``depth_ppm``.
    ``stellar_radii``: mapping tic_id → R* in R_sun from the metadata cache.

    Conservative rules:
      - If R* is missing for a TIC → NOT flagged (planet survives).
      - If depth_ppm is missing or ≤ 0 → NOT flagged.

    Returns a list of candidate dicts that exceed the radius threshold.
    Each returned dict has an added key ``depth_gate_rp_rjup`` (float).
    """
    flagged: List[Dict[str, Any]] = []
    for c in candidates:
        depth = c.get("depth_ppm")
        if not depth or depth <= 0:
            continue
        rstar = stellar_radii.get(str(c["tic_id"]))
        if rstar is None or rstar <= 0:
            # Conservative: missing R* → do not flag
            continue
        rp = implied_planet_radius_rjup(depth, rstar)
        if rp > max_rp_rjup:
            flagged.append({**c, "depth_gate_rp_rjup": round(rp, 3)})
    return flagged


def find_deep_eclipse_hosts(
    candidates: Iterable[Dict[str, Any]],
    systematic_periods: List[Dict[str, Any]],
    stellar_radii: Dict[str, float],
    deep_eclipse_ppm: float = 40_000.0,
    max_rp_rjup: float = MAX_PLANET_RADIUS_RJUP,
) -> set:
    """Return the set of TIC IDs that are deep-systematic or deep-EB host stars.

    ``candidates`` accepts any ITERABLE of row dicts, not just a list — a
    single forward pass, no ``len()``/indexing anywhere in this function
    (2026-08-01, postprocess-streaming F2, §3.D: this lets a chunked/
    streamed source such as ``select_in_chunks_iter`` feed this function
    chunk-by-chunk without ever materialising the full ``all_sigs`` list —
    see docs/superpowers/specs/2026-08-01-postprocess-streaming.md). A plain
    ``list`` is of course still a valid iterable, so this is purely additive:
    every existing caller (passing a list) is unaffected.

    Works per-TIC across ALL signals regardless of verdict, so a deep systematic
    signal can veto a shallow ``planet_candidate`` signal from the same star —
    the gap exploited by the common-period veto operating at signal level only.

    A TIC is flagged as an artefact/EB host iff it has at least one signal ``s``
    with:

        s.depth_ppm >= deep_eclipse_ppm
        AND (
              is_vetoed(s.period_days, s.t0, systematic_periods)            # primary: commonality
           OR (R* available AND implied_planet_radius_rjup(s.depth, R*) > max_rp_rjup)  # secondary: geometry
        )

    **Primary arm — commonality** (``is_vetoed``): the deep signal sits on a
    campaign-wide systematic period (epoch-clustered or common-period, e.g.
    P≈25 d).  Catches all 5 false finalists in lot 20260605T194912, including
    TIC 211353239 whose TRICERATOPS verdict was ``candidate`` (fpp=0.199) — the
    failure mode of the v1 FPP-based design.

    **Secondary arm — geometry** (``Rp > max_rp_rjup``, only when R* is cached):
    catches a genuine, idiosyncratic deep EB (deep eclipse on a non-systematic
    period) when stellar radius is available.

    **Dropped from v1:** the ``triceratops_fpp`` and ``triceratops_verdict`` arms
    — noisy and superseded by commonality (see module changelog above).

    Conservative on missing data:
      - ``depth_ppm`` None/≤0 → skip the signal (not flagged).
      - ``t0`` None on a systematic period → ``is_vetoed`` returns the match
        conservatively (period match alone is sufficient for epoch-clustered
        systematics).  Acceptable behaviour.
      - missing R* → secondary geometry arm simply does not fire (not flagged
        by geometry alone).
      - empty ``systematic_periods`` → primary arm is all-False for every signal
        (no-op).

    Philosophy mirrors the double-period check and depth gate: "when in doubt,
    do not veto."  A real planet does not share its T0 with 33 785 unrelated
    stars, and a 1 R_Jup planet on an idiosyncratic period is never
    ``is_vetoed``.

    Parameters
    ----------
    candidates : iterable of dicts
        ALL candidate rows for the TICs under test (any verdict, any signal).
        Required keys: ``tic_id``, ``period_days``, ``t0``, ``depth_ppm``.
        Any iterable — a list, a generator, or a chain of streamed chunks —
        consumed in a single forward pass.
    systematic_periods : list of dicts
        Output of ``find_systematic_periods`` — the already-built campaign-wide
        systematic periods list.  Pass the same ``sysp`` used by the
        common-period veto step; do NOT recompute.
    stellar_radii : dict
        Mapping ``tic_id`` (str) → R* in R_sun from the metadata cache.
        Missing TICs: secondary geometry arm is skipped.
    deep_eclipse_ppm : float
        Absolute depth threshold in ppm (default 40 000 = 4 %).  Only signals
        with depth ≥ this value are examined.
    max_rp_rjup : float
        Sub-stellar radius boundary in R_Jup (default MAX_PLANET_RADIUS_RJUP = 2.5).
        Used only when R* is available.

    Returns
    -------
    set of str
        TIC IDs (as str) flagged as deep-systematic / artefact / EB hosts.
    """
    eb_hosts: set = set()

    for c in candidates:
        depth = c.get("depth_ppm")
        if depth is None or depth <= 0:
            # Missing or non-positive depth → skip (conservative)
            continue
        if depth < deep_eclipse_ppm:
            # Below absolute threshold — not a stellar-scale eclipse
            continue
        tic = str(c["tic_id"])
        period = c.get("period_days")
        t0 = c.get("t0")

        # Primary arm: commonality — deep signal on a campaign-wide systematic period.
        # is_vetoed returns the matching systematic dict (truthy) or None (falsy).
        # Conservative with t0=None: period match alone is sufficient for
        # epoch-clustered systematics (is_vetoed handles this internally).
        if period is not None and is_vetoed(period, t0, systematic_periods):
            eb_hosts.add(tic)
            continue

        # Secondary arm: geometry — Rp > max_rp_rjup (only when R* is available).
        # This catches genuine, idiosyncratic deep EBs on non-systematic periods.
        rstar = stellar_radii.get(tic)
        if rstar is not None and rstar > 0 and depth is not None:
            rp = implied_planet_radius_rjup(depth, rstar)
            if rp > max_rp_rjup:
                eb_hosts.add(tic)

    return eb_hosts


# ---------------------------------------------------------------------------
# Anti-TOI override (2026-07-03) — never demote a confirmed planet (KP/CP)
# ---------------------------------------------------------------------------

def load_toi_protection(catalog_path: str) -> Dict[int, List[Optional[float]]]:
    """Load TIC-level anti-TOI protection from the TOI catalog CSV.

    Returns ``{tic_id: [period_days_or_None, ...]}`` for every TIC that hosts
    at least one CONFIRMED TOI (``TFOPWG Disposition`` in {"KP", "CP"}). A TIC
    can host several TOIs, hence the list. ``period_days`` is ``None`` when
    the catalog's ``Period (days)`` cell is missing/unparseable for that
    row — ``is_toi_protected`` treats a ``None`` entry as TIC-level
    protection regardless of the candidate's own period (conservative: never
    demote a confirmed planet's host star just because its catalog period
    wasn't parseable here).

    Non-fatal on a missing/malformed catalog (mirrors the graceful pattern of
    ``_load_toi_catalog`` in ``scripts/discovery_postprocess.py``): returns
    ``{}`` with a warning, which disables anti-TOI protection for this run
    rather than crashing the veto step.
    """
    import pandas as pd

    try:
        df = pd.read_csv(catalog_path, dtype=str, low_memory=False)
    except (FileNotFoundError, OSError):
        import warnings as _warnings
        _warnings.warn(
            f"[anti-toi] TOI catalog not found at {catalog_path!r}; "
            "SYSTEMATIC_VETO/DEEP_SYSTEMATIC_VETO anti-TOI protection disabled "
            "for this run.",
            RuntimeWarning,
            stacklevel=2,
        )
        return {}

    required = {"TIC ID", "TOI", "TFOPWG Disposition", "Period (days)"}
    missing = required - set(df.columns)
    if missing:
        import warnings as _warnings
        _warnings.warn(
            f"[anti-toi] TOI catalog at {catalog_path!r} is missing column(s) "
            f"{sorted(missing)}; anti-TOI protection disabled for this run.",
            RuntimeWarning,
            stacklevel=2,
        )
        return {}

    tic_num = pd.to_numeric(df["TIC ID"], errors="coerce")
    disp = df["TFOPWG Disposition"].fillna("").str.strip()
    period_num = pd.to_numeric(df["Period (days)"], errors="coerce")
    mask = tic_num.notna() & disp.isin(["KP", "CP"])

    protection: Dict[int, List[Optional[float]]] = {}
    for tic, p in zip(tic_num[mask], period_num[mask]):
        tic_int = int(tic)
        p_val = float(p) if pd.notna(p) else None
        protection.setdefault(tic_int, []).append(p_val)
    return protection


def is_toi_protected(
    tic_id: Any,
    period_days: Optional[float],
    protection: Dict[int, List[Optional[float]]],
    rel_tol: float = 0.01,
) -> bool:
    """Return True if (tic_id, period_days) is a confirmed (KP/CP) TOI.

    ``tic_id`` may arrive as ``str`` or ``int`` — both are used by the callers
    in this codebase — and is normalised to ``int`` here.

    True when ``tic_id`` is a key of ``protection`` AND either:
      - any of its TOI periods is ``None`` (unparseable in the catalog — TIC-
        level fallback protection, conservative), OR
      - ``period_days`` is not None/finite/positive and it matches (within
        ``rel_tol`` relative tolerance) at least one of the TIC's TOI periods
        — a DIRECT match, OR a harmonic/sub-harmonic ALIAS match (the
        candidate's period equals the catalog period times one of
        ``exonit.scorecard.HARMONIC_FACTORS_SUPER`` (2, 3, 4, 5) or
        ``HARMONIC_FACTORS_SUB`` (1/2, 1/3, 1/4, 1/5), same ``rel_tol``).
        Delegates to ``exonit.scorecard.match_period`` — the single source of
        truth for this alias set, also used by ``scripts/scorecard.py``'s
        recall accounting — rather than re-implementing it.

    Extended to harmonics/sub-harmonics 2026-07-10 (previously direct-match
    only — see the module changelog above and
    docs/reviews/meta_analysis_discovery_yield_2026-07-10.md). The 3 original
    known victims (TIC 176956893, 336732544, 364395234) are all direct
    matches and remain covered; TIC 260004324 (P/5 sub-harmonic) is the
    motivating harmonic case this extension closes.
    """
    try:
        tic_int = int(tic_id)
    except (TypeError, ValueError):
        return False
    periods = protection.get(tic_int)
    if periods is None:
        return False
    if any(p is None for p in periods):
        return True
    if period_days is None or not math.isfinite(period_days) or period_days <= 0:
        return False
    for p in periods:
        if p is None or p <= 0:
            continue
        # inclusive=True preserves the original direct-match semantics
        # (``<= rel_tol``, not scorecard's default strict ``< rel_tol``) —
        # a regression-safety detail so the 3 known-victim direct matches
        # keep matching at exactly the boundary.
        matched, _kind = match_period(period_days, p, tolerance=rel_tol, inclusive=True)
        if matched:
            return True
    return False


def filter_anti_toi_protected(
    candidates: List[Dict[str, Any]],
    protection: Dict[int, List[Optional[float]]],
    veto_label: str,
) -> "tuple[List[Dict[str, Any]], int]":
    """Split *candidates* into (survivors, n_protected) w.r.t. anti-TOI protection.

    Prints one ``"[anti-toi] PROTECTED from {veto_label}: TIC ... P=... "``
    line per dropped row (each *candidates* dict needs ``tic_id`` and
    ``period_days`` keys). Shared by ``scripts/discovery_postprocess.py``
    (Step 1, ``veto_label="SYSTEMATIC_VETO"``) and
    ``scripts/apply_systematic_veto.py`` so the matching logic (via
    ``is_toi_protected``) and print format are defined exactly once.
    """
    survivors: List[Dict[str, Any]] = []
    n_protected = 0
    for c in candidates:
        if is_toi_protected(c["tic_id"], c.get("period_days"), protection):
            n_protected += 1
            p = c.get("period_days")
            p_str = f"{p:.4f}" if p is not None else "?"
            print(f"[anti-toi] PROTECTED from {veto_label}: TIC {c['tic_id']} "
                  f"P={p_str} (matches KP/CP TOI)")
        else:
            survivors.append(c)
    return survivors, n_protected


def partition_deep_veto_targets(
    to_update: set,
    planet_candidate_rows: List[Dict[str, Any]],
    protection: Dict[int, List[Optional[float]]],
) -> "tuple[set, Dict[str, set]]":
    """Row-level anti-TOI split for the DEEP_SYSTEMATIC_VETO UPDATE targets.

    ``to_update``: TIC ids about to have ALL their ``planet_candidate`` rows
    reclassified to ``likely_false_positive``/``DEEP_SYSTEMATIC_VETO``.
    ``planet_candidate_rows``: candidate dicts (``id``, ``tic_id``,
    ``period_days``, ``claude_verdict``) — rows whose ``tic_id`` is not in
    ``to_update`` or whose ``claude_verdict`` isn't ``planet_candidate`` are
    ignored, so callers may pass a broader set (e.g. every signal for the
    sample's TICs) without pre-filtering.

    Returns ``(to_update_minus_fully_protected, protected_ids_by_tic)`` where
    ``protected_ids_by_tic`` maps a TIC (still present in the returned set) to
    the subset of its candidate ids that must be EXCLUDED from the UPDATE
    (they matched a confirmed KP/CP TOI period). A TIC whose EVERY
    planet_candidate row is protected is dropped from the returned set
    entirely — no UPDATE at all for it. Prints one
    ``"[anti-toi] PROTECTED from DEEP_SYSTEMATIC_VETO: ..."`` line per
    protected row.
    """
    if not to_update or not protection:
        return to_update, {}

    pc_rows_by_tic: Dict[str, List[Dict[str, Any]]] = {}
    for c in planet_candidate_rows:
        if c["tic_id"] in to_update and c.get("claude_verdict") == "planet_candidate":
            pc_rows_by_tic.setdefault(c["tic_id"], []).append(c)

    protected_ids_by_tic: Dict[str, set] = {}
    fully_protected: set = set()
    for tic, rows in pc_rows_by_tic.items():
        protected_ids = {r["id"] for r in rows
                         if is_toi_protected(tic, r.get("period_days"), protection)}
        if not protected_ids:
            continue
        for r in rows:
            if r["id"] in protected_ids:
                print(f"[anti-toi] PROTECTED from DEEP_SYSTEMATIC_VETO: "
                      f"TIC {tic} P={r.get('period_days')} (matches KP/CP TOI)")
        if protected_ids == {r["id"] for r in rows}:
            fully_protected.add(tic)
        else:
            protected_ids_by_tic[tic] = protected_ids

    return to_update - fully_protected, protected_ids_by_tic
