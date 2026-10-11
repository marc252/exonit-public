#!/usr/bin/env python3
"""Prefetch the SPOC-gate's bulk index — the preflight step for
``EXONIT_SPOC_GATE_OFFLINE=1`` (docs/superpowers/specs/2026-08-03-spoc-gate-
prefetch-offline-design.md, §2/§4).

Downloads, per sector, the two static MAST bulk-download scripts:

    https://archive.stsci.edu/missions/tess/download_scripts/sector/
        tesscurl_sector_{N}_lc.sh   (every SPOC 2-min *_lc.fits product)
        tesscurl_sector_{N}_dv.sh   (every DV product — some sectors have
                                      none: SPOC only runs DV where its own
                                      pipeline found a TCE)

and hands their content straight to ``exonit.spoc_bulk.ingest_sector`` (no
intermediate file is kept — the parsed rows are what matters, not the
script text). Idempotent: a sector already recorded in ``spoc_bulk_meta``
is skipped unless ``--refresh``.

COMPLETE DV coverage additionally requires a SEPARATE class of script:
multi-sector DV runs (a TIC observed across more than one sector) are NOT
included in any per-sector ``tesscurl_sector_{N}_dv.sh`` — verified against
the real archive (2026-08-03): ``tesscurl_sector_14_dv.sh`` is 15,618
products, 100% single-sector ``s0014-s0014`` rows, zero multi-sector. MAST
instead publishes each ``(start, end)`` range as its OWN script,
``tesscurl_multisector_s{START}-s{END}_dv.sh``, at the SAME directory
listing (observed ranges include ``s0001-s0002`` through ``s0001-s0069``;
ranges do not all start at ``s0001``). This script discovers those by
scraping the directory listing (regex over the ``href``s, no assumption
about which ranges exist) and ingests every one it finds via
``exonit.spoc_bulk.ingest_multisector_dv`` — by default, in EVERY run of
this script (not just ``--all-published``), since a multisector DV report
can apply to a TIC in any requested sector regardless of which per-sector
range was asked for. Pass ``--skip-multisector-dv`` to opt out (e.g. a
fast dev iteration that only cares about per-sector coverage).

    Full DV coverage = per-sector scripts + multisector scripts, both
    ingested. Omitting either one silently under-reports DV.

Usage:
    python3 scripts/prefetch_spoc_bulk.py --sectors 14,15,17
    python3 scripts/prefetch_spoc_bulk.py --sectors 1-96 --refresh
    python3 scripts/prefetch_spoc_bulk.py --all-published
    python3 scripts/prefetch_spoc_bulk.py --sectors 14,15,17 --dry-run
    python3 scripts/prefetch_spoc_bulk.py --sectors 14 --skip-multisector-dv

Run this before starting a discovery run whose sectors will be scanned with
``EXONIT_SPOC_GATE_OFFLINE=1`` — that flag makes the gate refuse to call
MAST at all, so an un-ingested sector there means an unresolved (NULL)
has_spoc/exominer_score for every TIC in it, not a fallback.
"""
import argparse
import re
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # allow `python3 scripts/prefetch_spoc_bulk.py` from any cwd

from exonit.spoc_bulk import bulk_coverage, ingest_multisector_dv, ingest_sector  # noqa: E402

BASE_URL = "https://archive.stsci.edu/missions/tess/download_scripts/sector"
_HTTP_TIMEOUT = 30.0
_MAX_RETRIES = 3
_RETRY_BACKOFF_S = 2.0
# Sector 1 is the first with SPOC 2-min data; stop probing --all-published
# once this many CONSECUTIVE sectors 404 (a single missing sector between
# real ones would not trip this — TESS sectors are contiguous in practice,
# but a couple of retries of slack costs nothing against a false stop).
_ALL_PUBLISHED_CONSECUTIVE_404_STOP = 3
_ALL_PUBLISHED_PROBE_CAP = 200  # hard ceiling so a bug here can never loop forever

# Matches the multisector DV script filename anywhere in the directory
# listing's markup (an Apache-style autoindex page, or any other listing
# format — deliberately not anchored to a specific `href="..."` shape, just
# the filename substring, which is all that is needed either way).
_MULTISECTOR_DV_RE = re.compile(r"tesscurl_multisector_s(\d{4})-s(\d{4})_dv\.sh")


def _sector_url(sector: int, kind: str) -> str:
    return f"{BASE_URL}/tesscurl_sector_{sector}_{kind}.sh"


def _fetch(url: str, session: requests.Session):
    """GET *url* with a timeout + gentle retry. Returns (status_code, text)
    — status_code is None on a fully-exhausted connection failure (no HTTP
    response at all, e.g. DNS/connect timeout), distinct from a real 404."""
    last_exc = None
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            resp = session.get(url, timeout=_HTTP_TIMEOUT)
            return resp.status_code, resp.text
        except requests.RequestException as exc:
            last_exc = exc
            if attempt < _MAX_RETRIES:
                time.sleep(_RETRY_BACKOFF_S * attempt)
    print(f"  ERROR: {url}: {last_exc} (giving up after {_MAX_RETRIES} attempts)",
          file=sys.stderr)
    return None, None


def _ingest_one(sector: int, kind: str, session: requests.Session,
                 dry_run: bool) -> str:
    """Fetch + ingest one (sector, kind) pair. Returns a one-word outcome:
    'ingested', 'not_found' (404 — normal for a sector with no DV), or
    'error' (connection failure or a non-200/404 status)."""
    url = _sector_url(sector, kind)
    status, text = _fetch(url, session)
    if status is None:
        return "error"
    if status == 404:
        print(f"  s{sector:04d} {kind}: 404 (not published) — skipping")
        return "not_found"
    if status != 200:
        print(f"  ERROR: s{sector:04d} {kind}: HTTP {status} at {url}",
              file=sys.stderr)
        return "error"
    if dry_run:
        print(f"  s{sector:04d} {kind}: would ingest ({len(text)} bytes downloaded)")
        return "ingested"
    n = ingest_sector(sector, kind, text)
    print(f"  s{sector:04d} {kind}: ingested {n:,} row(s)")
    return "ingested"


def _discover_multisector_dv_scripts(session: requests.Session) -> list:
    """Scrape the sector directory listing for multisector DV bulk scripts
    (``tesscurl_multisector_s{START}-s{END}_dv.sh``) — see the module
    docstring for why these are a separate class of script from any
    per-sector ``tesscurl_sector_{N}_dv.sh``. Returns a sorted, deduplicated
    list of ``(start, end)`` tuples.

    Unlike a 404 on one sector's own LC/DV script (normal, expected — many
    sectors have no DV), failing to even fetch the listing itself is a hard
    error: DV coverage would be silently, systematically incomplete without
    it, and there is no per-range fallback to fall back to.
    """
    status, text = _fetch(f"{BASE_URL}/", session)
    if status != 200 or text is None:
        raise RuntimeError(
            f"could not fetch the sector directory listing ({BASE_URL}/) "
            f"to discover multisector DV scripts (status={status}) — "
            "DV coverage would be silently incomplete without this; "
            "re-run once the archive directory listing is reachable, or "
            "pass --skip-multisector-dv to proceed with per-sector DV only "
            "(NOT recommended before EXONIT_SPOC_GATE_OFFLINE=1).")
    return sorted({
        (int(m.group(1)), int(m.group(2)))
        for m in _MULTISECTOR_DV_RE.finditer(text)
    })


def _ingest_one_multisector(range_start: int, range_end: int,
                             session: requests.Session, dry_run: bool) -> str:
    """Fetch + ingest one multisector DV script. Same outcome contract as
    ``_ingest_one`` ('ingested' / 'not_found' / 'error')."""
    label = f"s{range_start:04d}-s{range_end:04d}"
    url = f"{BASE_URL}/tesscurl_multisector_{label}_dv.sh"
    status, text = _fetch(url, session)
    if status is None:
        return "error"
    if status == 404:
        # Should not normally happen (the URL came straight from the
        # listing itself) but tolerate a race — script renamed/removed
        # between the listing fetch and this one — as a warning, not a
        # hard error, same as a per-sector 404.
        print(f"  WARNING: multisector {label} dv: 404 after being listed "
              "— skipping", file=sys.stderr)
        return "not_found"
    if status != 200:
        print(f"  ERROR: multisector {label} dv: HTTP {status} at {url}",
              file=sys.stderr)
        return "error"
    if dry_run:
        print(f"  multisector {label} dv: would ingest "
              f"({len(text)} bytes downloaded)")
        return "ingested"
    n = ingest_multisector_dv(range_start, range_end, text)
    print(f"  multisector {label} dv: ingested {n:,} row(s)")
    return "ingested"


def _parse_sectors_arg(spec: str) -> list:
    """Parse ``--sectors``: comma-separated ints and/or ``A-B`` ranges,
    e.g. ``"1-3,14,15,17"`` -> ``[1, 2, 3, 14, 15, 17]`` (deduped, sorted)."""
    out = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            out.update(range(int(lo), int(hi) + 1))
        else:
            out.add(int(part))
    return sorted(out)


def _discover_published_sectors(session: requests.Session, cap: int) -> list:
    """Probe sector LC scripts starting at 1 until N consecutive 404s.

    The LC script (not DV — DV is legitimately absent for many sectors) is
    the presence signal: every published sector has an LC bulk script.
    """
    sectors = []
    consecutive_404 = 0
    sector = 1
    while sector <= cap and consecutive_404 < _ALL_PUBLISHED_CONSECUTIVE_404_STOP:
        url = _sector_url(sector, "lc")
        status, _ = _fetch(url, session)
        if status == 200:
            sectors.append(sector)
            consecutive_404 = 0
        elif status == 404:
            consecutive_404 += 1
        else:
            # A connection error mid-probe must not be mistaken for "no more
            # sectors" — stop discovering rather than silently truncating.
            print(f"  ERROR: could not probe sector {sector} (status={status}) "
                  "— stopping discovery here; pass --sectors explicitly to "
                  "cover any sector at/after this one.", file=sys.stderr)
            break
        sector += 1
    return sectors


def run(sectors: list, refresh: bool, dry_run: bool,
        include_multisector: bool = True) -> int:
    """Ingest --sectors (both per-sector kinds) plus, by default, every
    multisector DV script discovered on the directory listing (see the
    module docstring — pass ``include_multisector=False`` /
    ``--skip-multisector-dv`` to opt out). Returns process exit code."""
    session = requests.Session()
    already = bulk_coverage() if not refresh else {"lc": [], "dv": [], "dv_multi": []}
    n_errors = 0

    print(f"Prefetching SPOC bulk index for {len(sectors)} sector(s) "
          f"{'(dry-run)' if dry_run else ''}".rstrip())
    for sector in sectors:
        for kind in ("lc", "dv"):
            if not refresh and sector in already.get(kind, []):
                print(f"  s{sector:04d} {kind}: already ingested — skipping "
                      "(pass --refresh to re-ingest)")
                continue
            outcome = _ingest_one(sector, kind, session, dry_run)
            if outcome == "error":
                n_errors += 1

    if include_multisector:
        print("\nDiscovering multisector DV scripts (directory listing scrape)...")
        try:
            ranges = _discover_multisector_dv_scripts(session)
        except RuntimeError as exc:
            print(f"  ERROR: {exc}", file=sys.stderr)
            n_errors += 1
            ranges = []
        else:
            print(f"  found {len(ranges)} multisector DV script(s): "
                  f"{[f's{s:04d}-s{e:04d}' for s, e in ranges]}")
        already_multi = set(already.get("dv_multi", [])) if not refresh else set()
        for start, end in ranges:
            label = f"s{start:04d}-s{end:04d}"
            if not refresh and label in already_multi:
                print(f"  multisector {label} dv: already ingested — "
                      "skipping (pass --refresh to re-ingest)")
                continue
            outcome = _ingest_one_multisector(start, end, session, dry_run)
            if outcome == "error":
                n_errors += 1
    else:
        print("\n--skip-multisector-dv passed: NOT ingesting multisector DV "
              "scripts — DV coverage will be incomplete for any TIC whose "
              "only DV report spans more than one sector.", file=sys.stderr)

    coverage = bulk_coverage() if not dry_run else already
    print("\nCoverage after this run:")
    print(f"  lc sectors ({len(coverage.get('lc', []))}): "
          f"{coverage.get('lc', [])}")
    print(f"  dv sectors ({len(coverage.get('dv', []))}): "
          f"{coverage.get('dv', [])}")
    print(f"  dv multisector ranges ({len(coverage.get('dv_multi', []))}): "
          f"{coverage.get('dv_multi', [])}")
    if n_errors:
        print(f"\n{n_errors} download error(s) — see stderr above. "
              "Coverage is incomplete; do NOT start a run with "
              "EXONIT_SPOC_GATE_OFFLINE=1 until this is resolved.",
              file=sys.stderr)
    return 1 if n_errors else 0


def main():
    parser = argparse.ArgumentParser(
        description="Prefetch the SPOC-gate bulk index (LC + DV) per sector, "
                    "offline-first preflight for EXONIT_SPOC_GATE_OFFLINE=1. "
                    "Complete DV coverage = per-sector "
                    "tesscurl_sector_{N}_dv.sh scripts + multisector "
                    "tesscurl_multisector_s{START}-s{END}_dv.sh scripts "
                    "(ingested by default in every run, see "
                    "--skip-multisector-dv).")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--sectors",
        help="Sectors to ingest, e.g. '14,15,17' or '1-96' or a mix.")
    group.add_argument(
        "--all-published", action="store_true",
        help="Discover every published sector by probing sequentially "
             f"until {_ALL_PUBLISHED_CONSECUTIVE_404_STOP} consecutive "
             "404s (capped at sector "
             f"{_ALL_PUBLISHED_PROBE_CAP}). Multisector DV scripts are "
             "always included in this mode (see --skip-multisector-dv).")
    parser.add_argument(
        "--refresh", action="store_true", default=False,
        help="Re-ingest sectors already recorded in spoc_bulk_meta "
             "(default: skip them).")
    parser.add_argument(
        "--dry-run", action="store_true", default=False,
        help="Download and report, but do not write to the index.")
    parser.add_argument(
        "--skip-multisector-dv", action="store_true", default=False,
        help="Do NOT discover/ingest tesscurl_multisector_s{START}-"
             "s{END}_dv.sh scripts (default: always ingest them, in both "
             "--sectors and --all-published modes — a multisector DV "
             "report can apply to a TIC in any requested sector). Leaves "
             "DV coverage incomplete for any TIC whose only DV report "
             "spans more than one sector; only use for a fast dev "
             "iteration, never before setting EXONIT_SPOC_GATE_OFFLINE=1.")
    args = parser.parse_args()

    session = requests.Session()
    if args.all_published:
        sectors = _discover_published_sectors(session, _ALL_PUBLISHED_PROBE_CAP)
        print(f"Discovered {len(sectors)} published sector(s): {sectors}")
    else:
        sectors = _parse_sectors_arg(args.sectors)

    if not sectors:
        print("No sectors to ingest.", file=sys.stderr)
        sys.exit(1)

    sys.exit(run(sectors, refresh=args.refresh, dry_run=args.dry_run,
                  include_multisector=not args.skip_multisector_dv))


if __name__ == "__main__":
    main()
