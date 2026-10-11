#!/usr/bin/env python3
"""Detect systematic (instrumental) periods in a search and optionally veto them.

Batch-level post-search step: a period present in many unrelated stars with a
clustered epoch is an instrumental systematic (scattered light, momentum dumps,
common gaps), not a planet. See exonit.systematic_veto and
docs/discovery-v1-vetting-dossier.md (Discovery Run v1, P~17d artifact).

By default reports only (dry-run). With --apply, reclassifies the matching
candidates to 'likely_false_positive' (keeping a note in cross_source_name).

Anti-TOI override (2026-07-03): the veto never touches a CONFIRMED planet
(TOI TFOPWG Disposition KP/CP) — see exonit.systematic_veto.is_toi_protected
and gotcha_systematic_veto_demotes_confirmed_planets. Also provides a one-off,
generic (not hardcoded to any TIC) repair mode for rows already wrongly
demoted before this override existed: --repair-anti-toi (report only) and
--apply-anti-toi-repair (restore them).
"""
import argparse
from pathlib import Path

from exonit.config import ExoNitConfig
from exonit.db_backend import op_session
from exonit.systematic_veto import (
    find_systematic_periods, is_vetoed, is_toi_protected,
    load_toi_protection, filter_anti_toi_protected,
)

ROOT = Path(__file__).resolve().parent.parent


def _repair_anti_toi(con, toi_catalog_path, apply_repair):
    """One-off, generic repair for rows demoted before the anti-TOI override.

    Scans the WHOLE candidates table for claude_verdict='likely_false_positive'
    rows with cross_source_name in ('SYSTEMATIC_VETO', 'DEEP_SYSTEMATIC_VETO')
    whose (tic_id, period_days) matches a confirmed (KP/CP) TOI per
    is_toi_protected — i.e. rows the anti-TOI override would have protected
    had it existed when they were demoted. NOT hardcoded to any TIC.

    Report only (default): prints each row that WOULD be restored, no DB
    write. With *apply_repair*: additionally writes
    claude_verdict='planet_candidate', cross_source_name='ANTI_TOI_RESTORED'
    (audit trail kept — never nulled).
    """
    toi_protection = load_toi_protection(toi_catalog_path)
    if not toi_protection:
        print("[repair-anti-toi] TOI catalog empty/missing — nothing to repair.")
        return 0

    demoted = [dict(r) for r in con.execute(
        "SELECT id, tic_id, period_days, t0, cross_source_name FROM candidates "
        "WHERE claude_verdict='likely_false_positive' "
        "AND cross_source_name IN ('SYSTEMATIC_VETO', 'DEEP_SYSTEMATIC_VETO')")]

    to_restore = [c for c in demoted
                 if is_toi_protected(c["tic_id"], c["period_days"], toi_protection)]

    print(f"[repair-anti-toi] {len(demoted)} row(s) demoted by SYSTEMATIC_VETO/"
          f"DEEP_SYSTEMATIC_VETO; {len(to_restore)} match a confirmed (KP/CP) TOI")
    for c in to_restore:
        print(f"  RESTORE: TIC {c['tic_id']} id={c['id']} P={c['period_days']} "
              f"t0={c['t0']} (was {c['cross_source_name']})")

    if not apply_repair:
        print("[repair-anti-toi] report-only — pass --apply-anti-toi-repair to write.")
        return 0

    for c in to_restore:
        con.execute(
            "UPDATE candidates SET claude_verdict='planet_candidate', "
            "cross_source_name='ANTI_TOI_RESTORED' WHERE id=?", (c["id"],))
    con.commit()
    print(f"[repair-anti-toi] restored {len(to_restore)} row(s) to planet_candidate "
          "(cross_source_name='ANTI_TOI_RESTORED')")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=None)
    ap.add_argument("--min-stars", type=int, default=10)
    ap.add_argument("--min-concentration", type=float, default=0.6)
    ap.add_argument("--apply", action="store_true", help="reclassify vetoed candidates (default: dry-run)")
    ap.add_argument("--toi-catalog", default=None,
                    help="path to TOI catalog CSV for the anti-TOI override "
                         "(default: data/toi_catalog.csv relative to repo root)")
    ap.add_argument("--repair-anti-toi", action="store_true",
                    help="REPORT ONLY: list candidates already reclassified to "
                         "likely_false_positive with cross_source_name in "
                         "('SYSTEMATIC_VETO', 'DEEP_SYSTEMATIC_VETO') whose "
                         "tic_id+period is a confirmed (KP/CP) TOI. Does NOT "
                         "modify the DB. Mutually exclusive with the veto scan.")
    ap.add_argument("--apply-anti-toi-repair", action="store_true",
                    help="together with --repair-anti-toi, actually restore the "
                         "listed rows (claude_verdict='planet_candidate', "
                         "cross_source_name='ANTI_TOI_RESTORED' — audit trail "
                         "kept, not nulled).")
    args = ap.parse_args()

    db = args.db or ExoNitConfig().db_path
    # Operational DB access goes through the backend seam so a postgresql://
    # DSN works exactly like a SQLite path (matches discovery_postprocess.py).
    con = op_session(db)
    toi_catalog_path = args.toi_catalog or str(ROOT / "data" / "toi_catalog.csv")

    if args.repair_anti_toi:
        rc = _repair_anti_toi(con, toi_catalog_path, args.apply_anti_toi_repair)
        con.close()
        return rc

    allc = [dict(r) for r in con.execute(
        "SELECT id,tic_id,period_days,t0,claude_verdict FROM candidates "
        "WHERE period_days IS NOT NULL AND t0 IS NOT NULL")]

    sysp = find_systematic_periods(allc, min_stars=args.min_stars,
                                   min_concentration=args.min_concentration)
    print(f"systematic periods (n_stars>={args.min_stars}, R>={args.min_concentration}): {len(sysp)}")
    for s in sysp:
        print(f"  P={s['period']:8.3f}d  n_stars={s['n_stars']:3d}  epoch={s['epoch_phase']:.2f}  R={s['concentration']}")

    vetoed_all = [c for c in allc if is_vetoed(c["period_days"], c["t0"], sysp)]
    # Anti-TOI override: never veto a confirmed (KP/CP) planet.
    toi_protection = load_toi_protection(toi_catalog_path)
    vetoed, n_protected = filter_anti_toi_protected(vetoed_all, toi_protection, "SYSTEMATIC_VETO")
    pc_vetoed = [c for c in vetoed if c["claude_verdict"] == "planet_candidate"]
    print(f"\ncandidates vetoed: {len(vetoed)} ({len(pc_vetoed)} were planet_candidate)"
          + (f"; {n_protected} anti-TOI protected" if n_protected else ""))

    if not args.apply:
        print("[dry-run] no DB changes. Use --apply to reclassify.")
        con.close()
        return 0

    n = 0
    for c in vetoed:
        if c["claude_verdict"] == "planet_candidate":
            con.execute("UPDATE candidates SET claude_verdict='likely_false_positive', "
                        "cross_source_name='SYSTEMATIC_VETO' WHERE id=?", (c["id"],))
            n += 1
    con.commit()
    print(f"reclassified {n} planet_candidate -> likely_false_positive (SYSTEMATIC_VETO)")
    con.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
