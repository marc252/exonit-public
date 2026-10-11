# scripts/release/fetch_toi_catalog.py
"""Download the TESS Objects of Interest (TOI) catalogue from ExoFOP as CSV."""
from __future__ import annotations
import argparse
import csv
import io
import os
import sys
import urllib.error
import urllib.request

USER_AGENT = "exonit-fetch-toi/1.0"
DEFAULT_URL = "https://exofop.ipac.caltech.edu/tess/download_toi.php?sort=toi&output=csv"
# Everything exonit/scorecard.py::load_toi_catalog reads with a hard df[...] lookup; validating all of them here
# makes a bad download fail at the "toi catalog" step instead of at the end of a long run.
REQUIRED_COLUMNS = ("TIC ID", "TFOPWG Disposition", "Period (days)", "Depth (ppm)", "Duration (hours)",
                    "Time Series Observations")


def fetch(url: str, timeout: int) -> bytes:
    """Return the raw response body for a TOI catalogue CSV download."""
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def _data_lines(text: str) -> list[str]:
    """Drop leading comment/blank lines so csv.reader sees the real header."""
    lines = text.splitlines()
    i = 0
    while i < len(lines) and (lines[i].startswith("#") or not lines[i].strip()):
        i += 1
    return lines[i:]


def clean(csv_bytes: bytes) -> bytes:
    """Strip leading '#' comment / blank lines (pd.read_csv does not skip them); the result starts at the header."""
    lines = _data_lines(csv_bytes.decode("utf-8", errors="replace"))
    return ("\n".join(lines) + "\n").encode("utf-8") if lines else b""


def validate(csv_bytes: bytes) -> int:
    """Check the CSV header and return the number of data rows."""
    lines = _data_lines(csv_bytes.decode("utf-8", errors="replace"))
    if not lines:
        raise ValueError("downloaded file is not a CSV: no header line found")
    reader = csv.reader(io.StringIO("\n".join(lines)))
    header = next(reader, None)
    if header is None:
        raise ValueError("downloaded file is not a CSV: no header line found")
    columns = [c.strip() for c in header]
    missing = [c for c in REQUIRED_COLUMNS if c not in columns]
    if missing:
        raise ValueError(f"downloaded CSV is missing required column(s): {', '.join(missing)}")
    rows = sum(1 for row in reader if any(cell.strip() for cell in row))
    if rows < 1:
        raise ValueError("downloaded CSV has a valid header but no data rows")
    return rows


def write_atomic(data: bytes, out: str) -> None:
    """Write data to <out>.tmp in the same directory, then rename onto <out>."""
    path = os.path.abspath(out)
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "wb") as fh:
        fh.write(data)
    os.replace(tmp, path)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Download the ExoFOP TOI catalogue as CSV.")
    ap.add_argument("--out", required=True, metavar="PATH", help="output CSV file")
    ap.add_argument("--url", default=DEFAULT_URL, metavar="URL", help="catalogue download URL")
    ap.add_argument("--timeout", type=int, default=120, metavar="SECONDS", help="HTTP timeout")
    a = ap.parse_args(argv)
    try:
        data = clean(fetch(a.url, a.timeout))
        n = validate(data)
    except ValueError as e:
        print(f"invalid TOI catalogue: {e}", file=sys.stderr)
        return 1
    except urllib.error.URLError as e:
        print(f"cannot download TOI catalogue: {e}", file=sys.stderr)
        return 1
    write_atomic(data, a.out)
    print(f"wrote {a.out} ({n} TOIs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
