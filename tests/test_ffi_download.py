"""Unit tests for exonit.ffi_download."""
from __future__ import annotations

import json
from pathlib import Path

import pytest


# =============================================================================
# canonical_filename
# =============================================================================

def test_canonical_filename_returns_basename():
    from exonit.ffi_download import canonical_filename

    p = Path(
        "/some/dir/HLSP/hlsp_qlp_tess_ffi_s0001-0000000149603524_tess_v01_llc/"
        "hlsp_qlp_tess_ffi_s0001-0000000149603524_tess_v01_llc.fits"
    )
    assert canonical_filename(p) == \
        "hlsp_qlp_tess_ffi_s0001-0000000149603524_tess_v01_llc.fits"


def test_canonical_filename_for_tglc():
    from exonit.ffi_download import canonical_filename

    p = Path(
        "/x/HLSP/hlsp_tglc_tess_ffi_gaiaid-12345-s0009-cam2-ccd3_tess_v1_llc/"
        "hlsp_tglc_tess_ffi_gaiaid-12345-s0009-cam2-ccd3_tess_v1_llc.fits"
    )
    assert canonical_filename(p) == \
        "hlsp_tglc_tess_ffi_gaiaid-12345-s0009-cam2-ccd3_tess_v1_llc.fits"


# =============================================================================
# list_qlp_files / list_tglc_files
# =============================================================================

def test_list_qlp_files_walks_cache(tmp_path):
    from exonit.ffi_download import list_qlp_files

    base = tmp_path / "mastDownload" / "HLSP"
    d1 = base / "hlsp_qlp_tess_ffi_s0001-0000000149603524_tess_v01_llc"
    d1.mkdir(parents=True)
    f1 = d1 / "hlsp_qlp_tess_ffi_s0001-0000000149603524_tess_v01_llc.fits"
    f1.write_bytes(b"fake fits")

    d2 = base / "hlsp_qlp_tess_ffi_s0058-0000000207463446_tess_v01_llc"
    d2.mkdir(parents=True)
    f2 = d2 / "hlsp_qlp_tess_ffi_s0058-0000000207463446_tess_v01_llc.fits"
    f2.write_bytes(b"fake fits")

    out = sorted(list_qlp_files(tmp_path))
    assert len(out) == 2
    assert out[0] == (149603524, 1, f1.resolve())
    assert out[1] == (207463446, 58, f2.resolve())


def test_list_tglc_files_resolves_via_sidecar(tmp_path):
    from exonit.ffi_download import list_tglc_files

    base = tmp_path / "mastDownload" / "HLSP"
    d1 = base / "hlsp_tglc_tess_ffi_gaiaid-5396580575830873728-s0009-cam2-ccd3_tess_v1_llc"
    d1.mkdir(parents=True)
    f1 = d1 / "hlsp_tglc_tess_ffi_gaiaid-5396580575830873728-s0009-cam2-ccd3_tess_v1_llc.fits"
    f1.write_bytes(b"fake fits")

    sidecar = tmp_path / ".tic_lookup" / "434100688__tglc.json"
    sidecar.parent.mkdir(parents=True)
    sidecar.write_text(json.dumps({
        "tic": "434100688",
        "source": "TGLC",
        "filenames": [f1.name],
        "ts": 1.0,
    }))

    out = list(list_tglc_files(tmp_path))
    assert len(out) == 1
    assert out[0] == (434100688, 9, f1.resolve())


def test_list_tglc_files_skips_unknown_sidecar(tmp_path):
    from exonit.ffi_download import list_tglc_files

    base = tmp_path / "mastDownload" / "HLSP"
    d1 = base / "hlsp_tglc_tess_ffi_gaiaid-9999999-s0001-cam1-ccd1_tess_v1_llc"
    d1.mkdir(parents=True)
    (d1 / "hlsp_tglc_tess_ffi_gaiaid-9999999-s0001-cam1-ccd1_tess_v1_llc.fits"
     ).write_bytes(b"x")

    out = list(list_tglc_files(tmp_path))
    assert out == []


# =============================================================================
# download_qlp / download_tglc — mocked
# =============================================================================

def _make_fake_search(filenames: list[str], tmp_cache: Path):
    """Return a fake SearchResult that 'downloads' files into the cache."""

    class FakeLC:
        def __init__(self, name):
            self.meta = {"FILENAME": name}

    class FakeProduct:
        def __init__(self, name):
            self._name = name

        def download(self, download_dir):
            d = Path(download_dir) / "mastDownload" / "HLSP" / \
                self._name.replace(".fits", "")
            d.mkdir(parents=True, exist_ok=True)
            (d / self._name).write_bytes(b"fake")
            return FakeLC(self._name)

    class FakeSearch:
        def __init__(self, products):
            self._products = products

        def __len__(self):
            return len(self._products)

        def __iter__(self):
            return iter(self._products)

    return FakeSearch([FakeProduct(n) for n in filenames])


def test_download_qlp_writes_fits_and_sidecar(tmp_path, monkeypatch):
    from exonit import ffi_download

    fake_search = _make_fake_search([
        "hlsp_qlp_tess_ffi_s0001-0000000149603524_tess_v01_llc.fits",
        "hlsp_qlp_tess_ffi_s0027-0000000149603524_tess_v01_llc.fits",
    ], tmp_path)

    def fake_search_lightcurve(target, author, exptime=None):
        assert target == "TIC 149603524"
        assert author == "QLP"
        return fake_search

    monkeypatch.setattr(ffi_download, "_search_lightcurve", fake_search_lightcurve)

    paths = ffi_download.download_qlp(149603524, tmp_path)
    assert len(paths) == 2
    assert all(p.exists() for p in paths)

    sidecar = tmp_path / ".tic_lookup" / "149603524__qlp.json"
    assert sidecar.exists()
    data = json.loads(sidecar.read_text())
    assert data["tic"] == "149603524"
    assert len(data["filenames"]) == 2


def test_download_qlp_idempotent_skips_existing(tmp_path, monkeypatch):
    from exonit import ffi_download

    name = "hlsp_qlp_tess_ffi_s0001-0000000149603524_tess_v01_llc.fits"
    d = tmp_path / "mastDownload" / "HLSP" / name.replace(".fits", "")
    d.mkdir(parents=True)
    (d / name).write_bytes(b"already here")

    sidecar = tmp_path / ".tic_lookup" / "149603524__qlp.json"
    sidecar.parent.mkdir(parents=True)
    sidecar.write_text(json.dumps({
        "tic": "149603524", "source": "QLP",
        "filenames": [name], "ts": 1.0,
    }))

    called = []

    def should_not_be_called(*args, **kwargs):
        called.append(args)
        raise AssertionError("download should not be called")

    monkeypatch.setattr(ffi_download, "_search_lightcurve", should_not_be_called)

    paths = ffi_download.download_qlp(149603524, tmp_path)
    assert len(paths) == 1
    assert called == []


def test_download_tglc_uses_tglc_done_sentinel(tmp_path, monkeypatch):
    from exonit import ffi_download

    sentinel = tmp_path / ".tglc_done" / "434100688.json"
    sentinel.parent.mkdir(parents=True)
    sentinel.write_text(json.dumps({"tic": "434100688", "n": 2, "ts": 1.0}))

    name = "hlsp_tglc_tess_ffi_gaiaid-X-s0009-cam1-ccd1_tess_v1_llc.fits"
    sidecar = tmp_path / ".tic_lookup" / "434100688__tglc.json"
    sidecar.parent.mkdir(parents=True)
    sidecar.write_text(json.dumps({
        "tic": "434100688", "source": "TGLC",
        "filenames": [name], "ts": 1.0,
    }))

    base = tmp_path / "mastDownload" / "HLSP" / name.replace(".fits", "")
    base.mkdir(parents=True)
    f = base / name
    f.write_bytes(b"fake")

    def should_not_be_called(*args, **kwargs):
        raise AssertionError("download should not be called")

    monkeypatch.setattr(ffi_download, "_search_lightcurve", should_not_be_called)

    paths = ffi_download.download_tglc(434100688, tmp_path)
    assert len(paths) == 1
    assert paths[0].resolve() == f.resolve()


# =============================================================================
# is_ffi_only_at_mast
# =============================================================================

def test_is_ffi_only_when_spoc_returns_empty(tmp_path, monkeypatch):
    from exonit import ffi_download

    class FakeEmptySearch:
        def __len__(self): return 0
        def __iter__(self): return iter([])

    monkeypatch.setattr(
        ffi_download, "_search_lightcurve",
        lambda target, author, exptime=None: FakeEmptySearch(),
    )
    assert ffi_download.is_ffi_only_at_mast(434100688, tmp_path) is True

    cached = tmp_path / ".spoc_check" / "434100688.json"
    assert cached.exists()
    assert json.loads(cached.read_text())["has_spoc"] is False


def test_is_ffi_only_uses_cached_check(tmp_path, monkeypatch):
    from exonit import ffi_download

    cache_path = tmp_path / ".spoc_check" / "111.json"
    cache_path.parent.mkdir(parents=True)
    cache_path.write_text(json.dumps(
        {"has_spoc": True, "checked_at": "2026-05-05"}
    ))

    def should_not_call(*a, **kw):
        raise AssertionError("MAST should not be queried")

    monkeypatch.setattr(ffi_download, "_search_lightcurve", should_not_call)
    assert ffi_download.is_ffi_only_at_mast(111, tmp_path) is False


def test_is_ffi_only_when_spoc_has_results(tmp_path, monkeypatch):
    from exonit import ffi_download

    class FakeNonEmpty:
        def __len__(self): return 3
        def __iter__(self): return iter([])

    monkeypatch.setattr(
        ffi_download, "_search_lightcurve",
        lambda target, author, exptime=None: FakeNonEmpty(),
    )
    assert ffi_download.is_ffi_only_at_mast(149603524, tmp_path) is False
