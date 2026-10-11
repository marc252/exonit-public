"""Shared QLP/TGLC download + cache indexing module.

Factored out of scripts/build_ffi_dataset.py (Phase 1) so that Phase 2.5
admin scripts (index_phase1_cache.py, predownload_ffi_toi.py) and the
existing dataset builder share a single source of truth.
"""
from __future__ import annotations

import datetime as _dt
import json
import re
import time
from pathlib import Path
from typing import Iterator


_QLP_RE = re.compile(r"^hlsp_qlp_tess_ffi_s(\d{4})-(\d{16})_tess_v\d+_llc\.fits$")
_TGLC_RE = re.compile(r"^hlsp_tglc_tess_ffi_gaiaid-(\d+)-s(\d{4})-cam\d-ccd\d_tess_v\d+_llc\.fits$")


def canonical_filename(path: Path | str) -> str:
    """Return the FITS filename a worker would request for this file.

    Workers call /api/fits/<filename>, where <filename> is the basename of
    the local FITS file. Always strip leading directories.
    """
    return Path(path).name


def _hlsp_dir(cache_dir: Path) -> Path:
    return Path(cache_dir) / "mastDownload" / "HLSP"


def list_qlp_files(cache_dir: Path | str) -> Iterator[tuple[int, int, Path]]:
    """Yield (tic, sector, absolute_path) for every QLP FITS in cache_dir.

    Filename pattern: hlsp_qlp_tess_ffi_s<NNNN>-<TIC10>_tess_v01_llc.fits
    """
    root = _hlsp_dir(cache_dir)
    if not root.exists():
        return
    for fits_path in root.rglob("hlsp_qlp_tess_ffi_*.fits"):
        m = _QLP_RE.match(fits_path.name)
        if not m:
            continue
        sector = int(m.group(1))
        tic = int(m.group(2))
        yield tic, sector, fits_path.resolve()


def _build_tic_lookup_index(cache_dir: Path | str) -> dict[str, str]:
    """Walk .tic_lookup/ sidecars and return {filename: tic_str}."""
    index: dict[str, str] = {}
    base = Path(cache_dir) / ".tic_lookup"
    if not base.exists():
        return index
    for sidecar in base.glob("*.json"):
        try:
            d = json.loads(sidecar.read_text())
        except Exception:
            continue
        tic = d.get("tic")
        for fn in d.get("filenames", []):
            if tic and fn:
                index[fn] = str(tic)
    return index


def list_tglc_files(cache_dir: Path | str) -> Iterator[tuple[int, int, Path]]:
    """Yield (tic, sector, absolute_path) for every TGLC FITS in cache_dir.

    TGLC filenames are keyed by Gaia DR3 source id. We resolve TIC via the
    sidecar .tic_lookup/<tic>__tglc.json files. Files with no sidecar
    mapping are skipped.
    """
    cache_dir = Path(cache_dir)
    root = _hlsp_dir(cache_dir)
    if not root.exists():
        return
    sidecar_index = _build_tic_lookup_index(cache_dir)
    for fits_path in root.rglob("hlsp_tglc_tess_ffi_*.fits"):
        m = _TGLC_RE.match(fits_path.name)
        if not m:
            continue
        sector = int(m.group(2))
        tic_str = sidecar_index.get(fits_path.name)
        if not tic_str:
            continue
        try:
            tic = int(tic_str)
        except ValueError:
            continue
        yield tic, sector, fits_path.resolve()


def index_files(entries, metadata_cache) -> int:
    """Insert (filename, full_path) rows into metadata_cache.fits_paths.

    `entries` is an iterable of (tic, sector, path) tuples produced by
    list_qlp_files / list_tglc_files. Returns the number of rows written.

    Uses INSERT OR REPLACE semantics (idempotent).
    """
    rows = [(Path(p).name, str(Path(p).resolve())) for _tic, _sector, p in entries]
    if not rows:
        return 0
    metadata_cache.add_fits_paths_bulk(rows)
    return len(rows)


def _search_lightcurve(target: str, author: str, exptime: int | None = None):
    """Indirection seam for tests to patch. Returns a SearchResult."""
    import lightkurve as lk
    if exptime is not None:
        return lk.search_lightcurve(target, author=author, exptime=exptime)
    return lk.search_lightcurve(target, author=author)


def _write_sidecar(cache_dir: Path, tic: int, source: str, filenames: list[str]) -> None:
    p = cache_dir / ".tic_lookup" / f"{tic}__{source.lower()}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({
        "tic": str(tic), "source": source.upper(),
        "filenames": filenames, "ts": time.time(),
    }))


def _read_sidecar_filenames(cache_dir: Path, tic: int, source: str) -> list[str]:
    p = cache_dir / ".tic_lookup" / f"{tic}__{source.lower()}.json"
    if not p.exists():
        return []
    try:
        return list(json.loads(p.read_text()).get("filenames", []))
    except Exception:
        return []


def _resolve_local_path(cache_dir: Path, filename: str) -> Path | None:
    candidate = cache_dir / "mastDownload" / "HLSP" / filename.replace(".fits", "") / filename
    return candidate if candidate.exists() else None


def _is_qlp_cached(cache_dir: Path, tic: int) -> list[Path]:
    """Return existing QLP files for tic, [] if none cached."""
    out: list[Path] = []
    for fn in _read_sidecar_filenames(cache_dir, tic, "qlp"):
        p = _resolve_local_path(cache_dir, fn)
        if p:
            out.append(p)
    return out


def _is_tglc_cached(cache_dir: Path, tic: int) -> list[Path]:
    """TGLC sentinel-based check."""
    sentinel = cache_dir / ".tglc_done" / f"{tic}.json"
    if not sentinel.exists():
        return []
    out: list[Path] = []
    for fn in _read_sidecar_filenames(cache_dir, tic, "tglc"):
        p = _resolve_local_path(cache_dir, fn)
        if p:
            out.append(p)
    return out


def _download_one(tic: int, source: str, cache_dir: Path) -> list[Path]:
    """Run the lightkurve search + download. Writes sidecar. Returns paths."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = f"TIC {tic}"
    sr = _search_lightcurve(target, author=source.upper())
    paths: list[Path] = []
    filenames: list[str] = []
    n = len(sr) if sr is not None else 0
    if n == 0:
        _write_sidecar(cache_dir, tic, source, [])
        return []
    for product in sr:
        try:
            lc = product.download(download_dir=str(cache_dir))
            fname = ""
            try:
                fname = lc.meta.get("FILENAME") or lc.meta.get("filename") or ""
            except Exception:
                pass
            if not fname:
                continue
            fname = Path(str(fname)).name
            filenames.append(fname)
            local = _resolve_local_path(cache_dir, fname)
            if local:
                paths.append(local)
        except Exception:
            continue
    _write_sidecar(cache_dir, tic, source, filenames)
    return paths


def _qlp_was_searched(cache_dir: Path, tic: int) -> bool:
    """QLP search marker: presence of the .tic_lookup sidecar."""
    return (cache_dir / ".tic_lookup" / f"{tic}__qlp.json").exists()


def _tglc_was_searched(cache_dir: Path, tic: int) -> bool:
    """TGLC search marker: presence of the .tglc_done sentinel."""
    return (cache_dir / ".tglc_done" / f"{tic}.json").exists()


def download_qlp(tic: int, cache_dir: Path | str) -> list[Path]:
    """Download all QLP sectors for tic. Idempotent.

    A previous search is recorded by the sidecar `.tic_lookup/<tic>__qlp.json`.
    If it exists we trust its filenames list, even if empty, and skip MAST.
    """
    cache_dir = Path(cache_dir)
    if _qlp_was_searched(cache_dir, tic):
        return _is_qlp_cached(cache_dir, tic)
    return _download_one(tic, "QLP", cache_dir)


def download_tglc(tic: int, cache_dir: Path | str) -> list[Path]:
    """Download all TGLC sectors for tic. Idempotent.

    A previous search is recorded by the sentinel `.tglc_done/<tic>.json`.
    If it exists we trust the sidecar's filenames list, even if empty.
    """
    cache_dir = Path(cache_dir)
    if _tglc_was_searched(cache_dir, tic):
        return _is_tglc_cached(cache_dir, tic)
    paths = _download_one(tic, "TGLC", cache_dir)
    sentinel = cache_dir / ".tglc_done" / f"{tic}.json"
    sentinel.parent.mkdir(parents=True, exist_ok=True)
    sentinel.write_text(json.dumps({"tic": str(tic), "n": len(paths),
                                    "ts": time.time()}))
    return paths


def is_ffi_only_at_mast(tic: int, cache_dir: Path | str) -> bool:
    """Return True if no 2-minute SPOC light curve exists at MAST for TIC.

    Caches the verdict in cache_dir/.spoc_check/<tic>.json so subsequent calls
    do not re-query MAST.
    """
    cache_dir = Path(cache_dir)
    cache_path = cache_dir / ".spoc_check" / f"{tic}.json"
    if cache_path.exists():
        try:
            d = json.loads(cache_path.read_text())
            return d.get("has_spoc") is False
        except Exception:
            pass
    sr = _search_lightcurve(f"TIC {tic}", author="SPOC", exptime=120)
    has_spoc = bool(sr) and len(sr) > 0
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps({
        "has_spoc": has_spoc,
        "checked_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
    }))
    return not has_spoc
