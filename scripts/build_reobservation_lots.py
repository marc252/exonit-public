#!/usr/bin/env python3
"""Build re-observation lots for cross-sector recurrence (P1, part B).

Takes the TICs of a finished discovery lot, selects the ones with a strong
candidate in the source sector (thresholds on exominer_ffi_score / FPP / SNR,
independent of claude_verdict unless --include-verdicts is given), looks up
which OTHER sectors have a TGLC light curve (metadata PG ``tglc`` table:
tic_id, sector, path, exptime, tmag, ra, dec) and groups the TICs per target
sector, excluding (tic, sector) pairs already completed in
``tic_sector_state`` or already present in ``candidates``.

Dry-run by default (counts only). ``--write`` emits per target sector:
  data/discovery/reobs_s<target:04d>_from_s<src:04d>_<ts>.csv   (tic_id,label)
  data/discovery/reobs_s<target:04d>_from_s<src:04d>_<ts>.json  (manifest)
and prints a plans.yaml snippet (approved_by EMPTY: the operator approves).
Read-only on every DB.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def select_source_tics(cand_rows, src_sector, min_em_ffi=0.6, max_fpp=0.015,
                       min_snr=12.0, include_verdicts=None, require_fpp=True,
                       stats=None):
    """cand_rows: dicts {tic_id, sector, snr, exominer_ffi_score,
    triceratops_fpp, claude_verdict}. -> sorted list of selected tic_ids.

    FPP semantics: NULL FPP = never reached the FPP stage = weaker candidate,
    so it is EXCLUDED by default (require_fpp=True; --allow-null-fpp opts
    out); negative values (-1 sentinel = EB/failed) never pass. ``stats``
    (dict) receives ``null_fpp_excluded`` = TICs otherwise-passing dropped
    only for NULL FPP.
    """
    null_only = set()
    keep = set()
    for r in cand_rows:
        if r.get("sector") != src_sector:
            continue
        if include_verdicts and r.get("claude_verdict") not in include_verdicts:
            continue
        em, snr, fpp = r.get("exominer_ffi_score"), r.get("snr"), r.get("triceratops_fpp")
        if em is None or em < min_em_ffi:
            continue
        if snr is None or snr < min_snr:
            continue
        if fpp is None:
            if require_fpp:
                null_only.add(str(r["tic_id"]))
                continue
        elif fpp < 0 or fpp > max_fpp:
            continue
        keep.add(str(r["tic_id"]))
    if stats is not None:
        stats["null_fpp_excluded"] = len(null_only - keep)
    return sorted(keep)


def group_by_target_sector(tics, tglc_sectors_by_tic, excluded_pairs, src_sector):
    """-> {target_sector: sorted [tic_id]}; skips src sector and excluded
    (tic, sector) pairs."""
    out = defaultdict(list)
    for tic in tics:
        for sec in sorted({int(s) for s in tglc_sectors_by_tic.get(tic, ())}):
            if sec == src_sector or (tic, sec) in excluded_pairs:
                continue
            out[sec].append(tic)
    return {s: sorted(v) for s, v in sorted(out.items())}


def plan_snippet(target, src, csv_path, n):
    return (
        f"  - plan_id: reobs-s{target:04d}-from-s{src:04d}\n"
        f"    approved_by:             # EMPTY: operator approves before the supervisor runs it\n"
        f"    sector: {target}\n"
        f"    mode: deep_rescan\n"
        f"    worklist: {csv_path}\n"
        f"    lot_size: {n}\n"
        f"    max_lots: 1\n"
        f"    fleet_profile: full\n"
        f"    notes: \"cross-sector re-observation of {n} TICs strong in s{src}\"\n"
    )


def write_outputs(groups, src, out_dir, ts, params, label="candidate"):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for target, tics in groups.items():
        stem = f"reobs_s{target:04d}_from_s{src:04d}_{ts}"
        csv_p, man_p = out_dir / f"{stem}.csv", out_dir / f"{stem}.json"
        with open(csv_p, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["tic_id", "label"])
            for t in tics:
                w.writerow([t, label])
        man_p.write_text(json.dumps({
            "source_sector": src, "target_sector": target, "n_tics": len(tics),
            "created": ts, "params": params, "csv": str(csv_p), "tics": tics,
        }, indent=2))
        written.append((target, csv_p, man_p, len(tics)))
    return written


# ---------------------------------------------------------------- DB access

def _load_lot_tics(sample, label):
    with open(sample) as fh:
        return sorted({r["tic_id"].strip() for r in csv.DictReader(fh)
                       if r["label"] == label})


def _fetch(con, tics):
    from exonit.db_backend import select_in_chunks
    cands = select_in_chunks(
        con.execute,
        "SELECT tic_id,sector,snr,exominer_ffi_score,triceratops_fpp,claude_verdict "
        "FROM candidates WHERE tic_id IN", tics)
    return [dict(r) for r in cands]


def _fetch_done_pairs(con, tics):
    from exonit.db_backend import select_in_chunks
    done = set()
    try:
        for r in select_in_chunks(
                con.execute, "SELECT tic_id,sector FROM tic_sector_state WHERE tic_id IN",
                tics, sql_after="AND status='completed'"):
            done.add((str(r["tic_id"]), int(r["sector"])))
    except Exception as exc:
        # Only the "table absent" case (legacy SQLite) is tolerated; any other
        # error (e.g. real PG failure) must not silently drop the exclusions.
        msg = str(exc).lower()
        if "no such table" in msg or "does not exist" in msg:
            print(f"[reobs] WARNING tic_sector_state absent: {exc!r}", file=sys.stderr)
        else:
            raise
    return done


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--sample", required=True, help="lot CSV (tic_id,label)")
    ap.add_argument("--label", default="candidate")
    ap.add_argument("--sector", type=int, required=True, help="source sector")
    ap.add_argument("--min-em-ffi", type=float, default=0.6)
    ap.add_argument("--max-fpp", type=float, default=0.015)
    ap.add_argument("--min-snr", type=float, default=12.0)
    ap.add_argument("--allow-null-fpp", action="store_true",
                    help="also accept candidates with NULL triceratops_fpp "
                         "(excluded by default)")
    ap.add_argument("--include-verdicts", nargs="*", default=None,
                    help="restrict source candidates to these claude_verdicts")
    ap.add_argument("--db", default=None, help="operational DB (default: config)")
    ap.add_argument("--meta-db", default=None, help="metadata DB path/DSN")
    ap.add_argument("--out-dir", default=str(ROOT / "data" / "discovery"))
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args(argv)

    from exonit.config import ExoNitConfig
    from exonit.db_backend import op_session
    from exonit.metadata_backend import is_pg_url
    from exonit.metadata_cache import MetadataCache

    tics = _load_lot_tics(args.sample, args.label)
    con = op_session(args.db or ExoNitConfig().db_path)
    try:
        cands = _fetch(con, tics)
        stats = {}
        selected = select_source_tics(
            cands, args.sector, args.min_em_ffi, args.max_fpp, args.min_snr,
            set(args.include_verdicts) if args.include_verdicts else None,
            not args.allow_null_fpp, stats)
        done = _fetch_done_pairs(con, selected) if selected else set()
    finally:
        con.close()
    # present in candidates for that sector also counts as already observed
    selected_set = set(selected)
    excluded = done | {(str(c["tic_id"]), int(c["sector"])) for c in cands
                       if str(c["tic_id"]) in selected_set}

    meta_target = args.meta_db or (os.environ.get("EXONIT_METADATA_DB_URL")
                                   if is_pg_url(os.environ.get("EXONIT_METADATA_DB_URL"))
                                   else "data/metadata_cache.db")
    mc = MetadataCache(meta_target)
    tglc = mc.tglc_lookup_bulk(selected) if selected else {}
    sectors_by_tic = {t: {r["sector"] for r in rows} for t, rows in tglc.items()}
    groups = group_by_target_sector(selected, sectors_by_tic, excluded, args.sector)

    print("[reobs] metadata source: table tglc (tic_id, sector, path, exptime, "
          f"tmag, ra, dec) via MetadataCache.tglc_lookup_bulk on "
          f"{'PG' if is_pg_url(meta_target) else meta_target}; "
          "exclusions: tic_sector_state(status='completed') + candidates(tic_id,sector)")
    print(f"[reobs] lot TICs={len(tics)} selected in s{args.sector}={len(selected)} "
          f"(NULL-FPP excluded: {stats.get('null_fpp_excluded', 0)})")
    for target, ts_ in groups.items():
        print(f"[reobs] target s{target}: {len(ts_)} TIC(s)")
    if not groups:
        print("[reobs] no re-observation targets")
    if not args.write:
        print("[reobs] dry-run: nothing written (use --write)")
        return 0

    ts = datetime.now().strftime("%Y%m%dT%H%M%S")
    params = {"min_em_ffi": args.min_em_ffi, "max_fpp": args.max_fpp,
              "min_snr": args.min_snr, "allow_null_fpp": args.allow_null_fpp,
              "include_verdicts": args.include_verdicts, "sample": args.sample}
    written = write_outputs(groups, args.sector, args.out_dir, ts, params, args.label)
    print("\n# --- plans.yaml snippet (approved_by left EMPTY) ---\nplans:")
    for target, csv_p, _man, n in written:
        print(plan_snippet(target, args.sector, csv_p, n))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
