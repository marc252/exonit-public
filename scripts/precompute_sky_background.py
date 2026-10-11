#!/usr/bin/env python3
"""Precompute the static star-field background for the sky map (Fase 1).

Reads the TGLC index (``data/metadata_cache.db``, table ``tglc``) for a given
sector — NOT the ``data/discovery/tglc_cache`` FITS tree (22M+ files, never
touch that directly) — and writes a small JSON file of sampled (ra, dec,
tmag) points that ``www/sky.html`` paints once as a static backdrop.

Read-only, high busy_timeout: this index is written concurrently by
scripts/index_tglc_batch.py (and read by the live server) during a discovery
run, so we must coexist rather than fail with "database is locked" (same
rationale as index_tglc_batch.py:109).

Sampling: deterministic stride over ``rowid`` (NOT ``ORDER BY random()``,
which would force a full-table sort over millions of rows) capped at
``--max-points`` (~30,000 by default — enough to paint a convincing field,
small enough for the browser to render every poll).

Fase 2 (docs/superpowers/specs/2026-07-18-sky-map-phase2.md): if
``gaia_dist.db`` (scripts/crossmatch_gaia_distances.py's output) exists next
to the background JSON, each point gets a 4th element ``dist_pc`` (float or
null). Without it, points stay exactly the 3-element ``[ra, dec, tmag]``
shape from Fase 1 — the 2D map never breaks, distance is a pure addition.

Usage:
  python3 scripts/precompute_sky_background.py --sector 14 \
      --index data/metadata_cache.db --out data/discovery/sky_bg/
"""
import argparse
import json
import math
import os
import sqlite3


def stride_for(count: int, cap: int) -> int:
    """Return the deterministic ``rowid`` stride to sample ~cap rows out of count.

    stride=1 (no downsampling) when count already fits under the cap.
    """
    if cap <= 0:
        raise ValueError("cap must be positive")
    if count <= cap:
        return 1
    return math.ceil(count / cap)


def _open_readonly(index_path: str) -> sqlite3.Connection:
    """Open *index_path* read-only with a high busy_timeout.

    Uses the SQLite URI ``mode=ro`` so this script can never accidentally
    write to the index, and a 300s busy_timeout so it politely waits its
    turn instead of erroring out while a discovery run or the indexer holds
    the file (same convention as scripts/index_tglc_batch.py:109).
    """
    con = sqlite3.connect(f"file:{index_path}?mode=ro", uri=True, timeout=300.0)
    con.execute("PRAGMA busy_timeout = 300000")
    return con


def _load_dist_pc_lookup(gaia_dist_path: str, sector: int,
                          tic_ids) -> dict:
    """Return {tic_id: dist_pc} for *tic_ids* in *sector*, chunked IN queries.

    Only looks up the (small, capped) set of sampled tic_ids — never the
    whole (potentially millions-of-rows) gaia_dist table — so this stays
    cheap regardless of sector size. Missing/NULL entries are simply absent
    from the returned dict; the caller treats that as "no distance".
    """
    con = _open_readonly(gaia_dist_path)
    try:
        lookup = {}
        ids = [str(t) for t in tic_ids]
        chunk_size = 900  # well under SQLite's bound-parameter limit
        for i in range(0, len(ids), chunk_size):
            batch = ids[i:i + chunk_size]
            placeholders = ",".join("?" * len(batch))
            rows = con.execute(
                f"SELECT tic_id, dist_pc FROM gaia_dist WHERE sector = ? "
                f"AND tic_id IN ({placeholders}) AND dist_pc IS NOT NULL",
                (sector, *batch),
            ).fetchall()
            for tic_id, dist_pc in rows:
                lookup[str(tic_id)] = dist_pc
        return lookup
    finally:
        con.close()


def generate_background(index_path: str, sector: int, out_dir: str,
                         cap: int = 30000,
                         gaia_dist_path: str = None) -> str:
    """Sample the ``tglc`` index for *sector* and write the background JSON.

    Returns the path written. Raises FileNotFoundError if *index_path*
    doesn't exist (caller decides how to report that).

    gaia_dist_path: override for the Fase 2 distance DB; defaults to
    ``gaia_dist.db`` alongside *out_dir* (scripts/crossmatch_gaia_distances.py's
    default output location). If it doesn't exist, points stay the Fase 1
    3-element shape — dist_pc is a pure addition, never a hard dependency.
    """
    if not os.path.exists(index_path):
        raise FileNotFoundError(index_path)

    if gaia_dist_path is None:
        gaia_dist_path = os.path.join(out_dir, "gaia_dist.db")
    has_dist = os.path.exists(gaia_dist_path)

    con = _open_readonly(index_path)
    try:
        count = con.execute(
            "SELECT COUNT(*) FROM tglc WHERE sector = ?", (sector,)
        ).fetchone()[0]
        stride = stride_for(count, cap)
        rows = con.execute(
            """SELECT ra, dec, tmag, tic_id FROM tglc
               WHERE sector = ? AND (rowid % ?) = 0""",
            (sector, stride),
        ).fetchall()
    finally:
        con.close()

    dist_lookup = {}
    if has_dist:
        dist_lookup = _load_dist_pc_lookup(
            gaia_dist_path, sector, [r[3] for r in rows])

    points = []
    for ra, dec, tmag, tic_id in rows:
        if ra is None or dec is None:
            continue
        point = [
            round(float(ra), 3),
            round(float(dec), 3),
            round(float(tmag), 3) if tmag is not None else None,
        ]
        if has_dist:
            point.append(dist_lookup.get(str(tic_id)))
        points.append(point)

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"sector_{sector}.json")
    payload = {"sector": sector, "count": len(points), "points": points}
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh)
    return out_path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sector", type=int, required=True)
    ap.add_argument("--index", default="data/metadata_cache.db")
    ap.add_argument("--out", default="data/discovery/sky_bg/")
    ap.add_argument("--max-points", type=int, default=30000,
                    help="Sample ceiling (default: 30000)")
    ap.add_argument("--gaia-dist", default=None,
                    help="Fase 2 distance DB (default: gaia_dist.db under --out; "
                         "skipped silently if absent)")
    args = ap.parse_args()

    out_path = generate_background(
        args.index, args.sector, args.out, cap=args.max_points,
        gaia_dist_path=args.gaia_dist)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
