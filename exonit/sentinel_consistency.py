"""Shared logic for the ExoMiner-SPOC ``-1`` sentinel consistency invariant.

The invariant (docs/superpowers/specs/exominer-dv-gate.md, "Step 1: the
SPOC-gate must decide by DV, not by SPOC" — corrected by
docs/superpowers/specs/exominer-dv-gate-sector.md, 2026-07-14):

    No candidate may have ``stars.has_spoc = 1`` AND
    ``candidates.exominer_score = -1`` AND a real DV report covering
    THAT CANDIDATE'S OWN SECTOR at MAST.

Since 2026-07-13 the SPOC-gate (``exonit.server._run_spoc_check_background``)
legitimately writes ``-1`` for has-SPOC TICs/candidates that SPOC never
published a DV report covering their sector for (see that function's
docstring). A candidate matching ``has_spoc=1 AND exominer_score=-1`` is
therefore only a *suspect*, not a confirmed violation: telling the two apart
requires an actual MAST DV lookup, which is why the two functions below are
split — finding suspects is a cheap DB query, confirming a violation costs a
MAST round-trip per TIC.

2026-07-14 fix (docs/superpowers/specs/exominer-dv-gate-sector.md): the
original version of ``verify_candidates_against_mast`` asked "does this TIC
have ANY DV report?" — the exact premise error the gate itself was fixed
for on the SAME day. A DV report only covers the sector RANGE it was run
over (``exonit.mast_dv.parse_dv_filename``); "has DV somewhere" is not "has
DV for THIS candidate's sector". ``find_suspicious_sentinel_candidates`` now
returns each row's ``sector`` too, and ``verify_candidates_against_mast``
checks coverage of THAT sector (``exonit.mast_dv.dv_runs_cover_sector``) —
same criterion the live gate uses, so the two never diverge again.

Two consumers share this module (extracted 2026-07-13 rather than
duplicated, per the spec above):
  - ``scripts/fix_exominer_spurious_sentinel.py`` — finds ALL suspects and,
    with ``--apply``, fixes every one confirmed to have DV (full scan).
  - ``scripts/check_exominer_sentinel_consistency.py`` — health check: reports
    the raw suspect count and, with ``--verify-dv N``, confirms a sample
    against MAST without writing anything.

2026-07-14 (docs/superpowers/specs/exominer-status-column.md — "Pas 2"):
``count_status_score_violations`` adds a SECOND, unrelated invariant to the
same health-check script — ``exominer_status`` (the new state column) never
contradicting ``exominer_score``. It needs no MAST lookup (pure DB query),
so it always runs, unlike the DV-verified suspect count above.
"""
from __future__ import annotations

from typing import Callable, List, Optional, Tuple

from exonit.db import ExoNitDB
from exonit.mast_dv import MastDVCache, dv_runs_cover_sector, fetch_dv_runs


def find_suspicious_sentinel_candidates(
    db: ExoNitDB,
    min_cnn: float = 0.7,
    sector: Optional[int] = None,
    limit: Optional[int] = None,
) -> list:
    """Return candidate rows (``id``, ``tic_id``, ``sector``) matching the
    suspect predicate: ``has_spoc=1 AND exominer_score=-1 AND cnn_score >=
    min_cnn`` on a completed star.

    This mirrors the ExoMiner backfill's own selection predicate (minus the
    ``exominer_score IS NULL`` half, since here we're looking at the *other*
    branch — rows the backfill will now never see because they're stuck at
    -1) so "suspects" means exactly "candidates that WOULD be backfill-
    eligible if their -1 turns out to be spurious".

    ``sector`` is included in the returned rows (docs/superpowers/specs/
    exominer-dv-gate-sector.md) so ``verify_candidates_against_mast`` can
    check DV coverage of the CANDIDATE's own sector, not just "any DV for
    this TIC" — note this is unrelated to the ``sector`` FILTER parameter
    below (which scopes the query to one campaign sector, e.g. for a
    per-run health check).
    """
    where_sector = "AND c.sector = ?" if sector is not None else ""
    params: list = [min_cnn] + ([sector] if sector is not None else [])
    rows = db._query(
        f"""SELECT c.id, c.tic_id, c.sector
            FROM candidates c
            JOIN stars s ON c.tic_id = s.tic_id
            WHERE c.exominer_score = -1
              AND c.cnn_score >= ?
              AND s.has_spoc = 1
              AND s.status = 'completed'
              {where_sector}
            ORDER BY c.id""",
        tuple(params))
    if limit:
        rows = rows[:limit]
    return rows


def count_status_score_violations(db: ExoNitDB) -> dict:
    """Count rows where ``candidates.exominer_status`` contradicts
    ``candidates.exominer_score`` (docs/superpowers/specs/exominer-status-
    column.md — "Consistency").

    Invariant (mirrors ``exonit.db.derive_exominer_status``'s corrected rule
    — the -1 sentinel is the only special-cased score value, so 'scored'
    only requires "not NULL and not -1", NOT "score >= 0": a real ExoMiner
    probability can round to a tiny negative float32 artifact like
    -5.960464477539063e-08 and is still a genuine score, not the sentinel):

        pending           <=> score IS NULL
        no_spoc / no_dv   <=> score = -1
        scored            <=> score IS NOT NULL AND score != -1
        failed            => score IS NULL OR score = -1 (never a real score)

    Rows with ``exominer_status IS NULL`` (not yet migrated/resolved) are
    excluded — this checks COHERENCE of the two columns where both are set,
    not migration completeness (that is scripts/migrate_exominer_status.py's
    own final-count responsibility).

    Returns a dict of per-bucket violation counts (0 if healthy) plus
    'total' — the health check script exits non-zero when 'total' > 0.
    """
    row = db._query_one(
        """SELECT
             SUM(CASE WHEN exominer_status = 'pending'
                      AND exominer_score IS NOT NULL
                 THEN 1 ELSE 0 END) AS bad_pending,
             SUM(CASE WHEN exominer_status IN ('no_spoc', 'no_dv')
                      AND (exominer_score IS NULL OR exominer_score != -1)
                 THEN 1 ELSE 0 END) AS bad_no_score,
             SUM(CASE WHEN exominer_status = 'scored'
                      AND (exominer_score IS NULL OR exominer_score = -1)
                 THEN 1 ELSE 0 END) AS bad_scored,
             SUM(CASE WHEN exominer_status = 'failed'
                      AND exominer_score IS NOT NULL AND exominer_score != -1
                 THEN 1 ELSE 0 END) AS bad_failed
           FROM candidates
           WHERE exominer_status IS NOT NULL""")
    counts = {k: (v or 0) for k, v in (row or {}).items()}
    counts["total"] = sum(counts.values())
    return counts


def verify_candidates_against_mast(
    rows,
    cache: Optional[MastDVCache] = None,
    progress_every: int = 50,
    log: Callable[[str], None] = print,
) -> Tuple[List[int], int, int]:
    """Verify a list of suspect rows (``id``, ``tic_id``, ``sector``) against
    MAST DV reports, one ``fetch_dv_runs`` call per row (the on-disk
    MastDVCache makes a second call for the same TIC free, so this is
    deliberately not pre-deduplicated in Python — that would also change
    the per-row error count callers rely on).

    A row is only classified "with DV" if a report COVERS its own ``sector``
    (docs/superpowers/specs/exominer-dv-gate-sector.md) — the same criterion
    the live gate (``exonit.server._run_spoc_check_background``) uses, so
    this health check and the gate can never silently diverge again.

    Returns ``(amb_dv, sense_dv, errors)``:
      amb_dv   -- candidate ids CONFIRMED to have a DV report covering their
                  OWN sector — real violations of the invariant (has_spoc=1,
                  exominer_score=-1, DV exists for that sector): the -1 is
                  spurious and should be retracted.
      sense_dv -- count with no DV report covering their sector (either no
                  DV at all, or DV for a DIFFERENT sector) — the -1 is
                  correct, leave as-is.
      errors   -- count of rows where the MAST lookup itself raised; not
                  touched either way (unresolved, safe to retry later).
    """
    cache = cache if cache is not None else MastDVCache()
    amb_dv: List[int] = []
    sense_dv = 0
    errors = 0
    total = len(rows)

    for i, row in enumerate(rows, 1):
        tic = str(row["tic_id"])
        sector = row["sector"]
        try:
            runs = fetch_dv_runs(tic, cache=cache)
        except Exception as exc:
            errors += 1
            log(f"  [{i}/{total}] TIC {tic}: ERROR {type(exc).__name__}: {exc}")
            continue
        if dv_runs_cover_sector(runs, sector):
            amb_dv.append(row["id"])
        else:
            sense_dv += 1
        if progress_every and i % progress_every == 0:
            log(f"  … {i}/{total} — with DV: {len(amb_dv)}, "
                f"without DV: {sense_dv}, errors: {errors}")

    return amb_dv, sense_dv, errors
