"""Target catalog management for ExoNit.

Downloads and parses the TOI (TESS Objects of Interest) catalog,
manages target selection and prioritization.
"""

import csv
import os
import requests

TOI_CSV_URL = ("https://exofop.ipac.caltech.edu/tess/"
               "download_toi.php?sort=toi&output=csv")


def download_toi_catalog(output_path="data/toi_catalog.csv"):
    """Download the latest TOI catalog from ExoFOP."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    print(f"Downloading TOI catalog from ExoFOP...")
    resp = requests.get(TOI_CSV_URL, timeout=120)
    resp.raise_for_status()
    with open(output_path, "w") as f:
        f.write(resp.text)
    n_lines = resp.text.count("\n") - 1
    print(f"  Saved {n_lines} TOIs to {output_path}")
    return output_path


def parse_toi_catalog(csv_path):
    """Parse a TOI catalog CSV into a list of target dicts."""
    targets = []
    seen_tics = set()
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # ExoFOP CSV uses "TIC ID", older formats use "tid"
            tic_id = (row.get("TIC ID") or row.get("tid") or "").strip()
            if not tic_id or tic_id in seen_tics:
                continue
            seen_tics.add(tic_id)
            try:
                tmag = float(row.get("TESS Mag") or row.get("tmag") or 99)
            except (ValueError, TypeError):
                tmag = 99.0
            try:
                ra = float(row.get("RA") or row.get("ra") or 0)
                dec = float(row.get("Dec") or row.get("dec") or 0)
            except (ValueError, TypeError):
                ra, dec = 0.0, 0.0

            def _safe_float(val, default=0):
                try:
                    return float(val)
                except (ValueError, TypeError):
                    return default

            targets.append({
                "tic_id": tic_id,
                "toi": row.get("TOI") or row.get("toi") or "",
                "ra": ra, "dec": dec, "tmag": tmag,
                "disposition": (row.get("TFOPWG Disposition")
                                or row.get("tfopwg_disp") or ""),
                "stellar_radius": _safe_float(
                    row.get("Stellar Radius (R_Sun)"), None),
                "period_known": _safe_float(
                    row.get("Period (days)"), None),
            })
    return targets


def get_next_targets(targets, db, max_targets=10):
    """Select the next batch of targets to analyze.
    Skips already-analyzed AND failed stars, sorts by brightness."""
    all_in_db = set(r['tic_id'] for r in db._query('SELECT tic_id FROM stars'))
    pending = [t for t in targets if t["tic_id"] not in all_in_db]
    pending.sort(key=lambda t: t["tmag"])
    return pending[:max_targets]
