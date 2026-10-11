#!/usr/bin/env python3
"""Dump a real TESS light-curve FITS file (TGLC/QLP/SPOC) into a compact JSON
reference for the Rust `fits.rs` golden test, using astropy directly (no
lightkurve, no quality masking) so it reflects the RAW column contents that
`FITSReader` itself must decode byte-for-byte.

Usage:
    ~/exonit/.venv/bin/python3 tests/tools/dump_fits.py <fits_path> <out_json>

Read-only: never writes outside <out_json>. Source FITS files come from
~/exonit/worker/fits_cache*/ (read-only, NOT part of this repo).
"""
import json
import sys

import numpy as np
from astropy.io import fits


def dump(path: str) -> dict:
    with fits.open(path) as hdul:
        hdu = None
        for candidate in hdul:
            if candidate.header.get("XTENSION", "").strip() == "BINTABLE":
                hdu = candidate
                break
        if hdu is None:
            raise SystemExit("no BINTABLE HDU found")
        header = hdu.header
        names = [header.get(f"TTYPE{i}") for i in range(1, header.get("TFIELDS", 0) + 1)]
        lower = {n.lower(): n for n in names if n}
        data = hdu.data

        def col(name):
            real = lower.get(name.lower())
            if real is None:
                return None
            arr = np.asarray(data[real])
            return arr

        time_col = col("time")
        out = {
            "path": path,
            "naxis1": int(header.get("NAXIS1", 0)),
            "naxis2": int(header.get("NAXIS2", 0)),
            "columns": names,
            "row_count": int(len(time_col)) if time_col is not None else 0,
            "time": [float(x) for x in np.asarray(time_col, dtype=np.float64)] if time_col is not None else None,
        }

        for candidate_name in ("cal_aper_flux", "pdcsap_flux", "sap_flux"):
            arr = col(candidate_name)
            if arr is not None:
                out["flux_column"] = candidate_name
                out["flux"] = [float(x) for x in np.asarray(arr, dtype=np.float64)]
                break

        for flag_name in ("tess_flags", "tglc_flags"):
            arr = col(flag_name)
            if arr is not None:
                out[flag_name] = [int(x) for x in np.asarray(arr)]

        quality = col("quality")
        if quality is not None:
            out["quality"] = [int(x) for x in np.asarray(quality)]

        cadence = col("cadenceno")
        if cadence is not None:
            out["cadenceno"] = [int(x) for x in np.asarray(cadence)]

        return out


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit(f"usage: {sys.argv[0]} <fits_path> <out_json>")
    payload = dump(sys.argv[1])
    with open(sys.argv[2], "w") as fh:
        json.dump(payload, fh)
    print(f"wrote {sys.argv[2]} ({payload['row_count']} rows, flux_column={payload.get('flux_column')})")


if __name__ == "__main__":
    main()
