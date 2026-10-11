#!/usr/bin/env python3
"""Export a minimal run bundle from a source ExoNit DB.

Run on the AWS side at the end of a run (FINALIZE). Produces a small SQLite
``bundle.db`` containing candidates and the ``stars`` they reference — NOT the
full DB and NOT the FITS. Optionally uploads the bundle (and any scorecard files
passed) to an S3 prefix so the home UI can repatriate it (see
``repatriate_and_lora.sh``).

Two selection modes (exactly one required):

  --sector N        Export only candidates where candidates.sector = N.
                    Use for home runs where the campaign sector matches the
                    observation sector stored in the DB.

  --all-sectors     Export ALL candidates regardless of sector.
                    Use for AWS per-run DBs: candidates.sector stores the TESS
                    *observation* sector (e.g. 1), not the campaign lot sector
                    (e.g. 2), so filtering by campaign sector yields an empty
                    bundle.  The AWS DB is fresh/per-run, so all candidates
                    belong to this run.

Kept deliberately minimal: the home side re-keys candidate ids and merges by the
natural key, so we only need the candidate + star rows, not internal ids.
"""
import argparse
import os
import sqlite3
import subprocess
import sys


def export_bundle(src_db: str, out_bundle: str,
                  sector: int | None = None, all_sectors: bool = False) -> dict:
    """Write a bundle.db with candidates + their stars. Returns stats dict.

    Exactly one of ``sector`` (int) or ``all_sectors=True`` must be provided.
    """
    if (sector is None) == (not all_sectors):
        # both None/False, or both set — ambiguous
        if sector is None and not all_sectors:
            raise ValueError("Provide either sector=<int> or all_sectors=True")
        # sector set AND all_sectors set — also ambiguous
        raise ValueError("Provide either sector=<int> or all_sectors=True, not both")

    if not os.path.exists(src_db):
        raise FileNotFoundError(src_db)
    if os.path.exists(out_bundle):
        os.remove(out_bundle)
    con = sqlite3.connect(out_bundle, timeout=30)
    try:
        con.execute("ATTACH DATABASE ? AS src", (src_db,))
        if all_sectors:
            con.execute("CREATE TABLE candidates AS SELECT * FROM src.candidates")
            con.execute(
                "CREATE TABLE stars AS SELECT * FROM src.stars WHERE tic_id IN "
                "(SELECT DISTINCT tic_id FROM src.candidates)")
        else:
            con.execute(
                "CREATE TABLE candidates AS "
                "SELECT * FROM src.candidates WHERE sector = ?", (sector,))
            con.execute(
                "CREATE TABLE stars AS SELECT * FROM src.stars WHERE tic_id IN "
                "(SELECT DISTINCT tic_id FROM src.candidates WHERE sector = ?)",
                (sector,))
        n_cand = con.execute("SELECT COUNT(*) FROM candidates").fetchone()[0]
        n_stars = con.execute("SELECT COUNT(*) FROM stars").fetchone()[0]
        con.commit()
    finally:
        con.close()
    sector_label: int | str = sector if sector is not None else "all"
    return {"sector": sector_label, "candidates": n_cand, "stars": n_stars,
            "bundle": out_bundle}


def upload(paths: list, s3_prefix: str) -> None:
    """aws s3 cp each path to s3_prefix (must end with /)."""
    if not s3_prefix.endswith("/"):
        s3_prefix += "/"
    for p in paths:
        subprocess.run(["aws", "s3", "cp", p, s3_prefix + os.path.basename(p)],
                       check=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src-db", required=True, help="Source exonit.db (AWS)")
    ap.add_argument("--out", default="bundle.db", help="Output bundle path")
    # Mutually exclusive selection modes — exactly one required.
    sel = ap.add_mutually_exclusive_group(required=True)
    sel.add_argument("--sector", type=int, default=None,
                     help="Export candidates where sector = N (home/back-compat)")
    sel.add_argument("--all-sectors", action="store_true", default=False,
                     help="Export ALL candidates (AWS per-run DB, observation sector"
                          " != campaign sector)")
    ap.add_argument("--s3-prefix", default=None,
                    help="If set, aws s3 cp the bundle (+ --scorecard files) here")
    ap.add_argument("--scorecard", action="append", default=[],
                    help="Extra scorecard file(s) to upload alongside the bundle")
    args = ap.parse_args(argv)
    stats = export_bundle(args.src_db, args.out,
                          sector=args.sector, all_sectors=args.all_sectors)
    if args.s3_prefix:
        upload([args.out] + args.scorecard, args.s3_prefix)
        stats["uploaded_to"] = args.s3_prefix
    import json
    print(json.dumps(stats))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
