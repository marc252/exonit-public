"""Tests for FITS packing (exonit/fits_pack.py) and the fits_blobs index."""

import io
import tarfile

from exonit.fits_pack import build_container, extract_member


def test_build_container_offsets_round_trip():
    entries = [
        ("hlsp_tglc_tess_ffi_gaiaid-5499922952171241728-s0001-cam4-ccd4_tess_v1_llc.fits",
         b"A" * 55000),
        ("hlsp_tglc_tess_ffi_gaiaid-2311885890760886400-s0002-cam1-ccd1_tess_v1_llc.fits",
         b"B" * 54321),
        ("small.fits", b"C" * 7),
        ("empty.fits", b""),
    ]
    container, locs = build_container(entries)
    orig = dict(entries)
    assert len(locs) == len(entries)
    for name, off, length in locs:
        assert extract_member(container, off, length) == orig[name], name


def test_build_container_is_valid_tar():
    entries = [("x.fits", b"hello"), ("y.fits", b"world!!")]
    container, _ = build_container(entries)
    with tarfile.open(fileobj=io.BytesIO(container), mode="r") as tar:
        names = [m.name for m in tar.getmembers()]
        assert names == ["x.fits", "y.fits"]
        # data extractable via tarfile too (recoverable without the index)
        assert tar.extractfile("y.fits").read() == b"world!!"


def test_build_container_deterministic():
    entries = [("a.fits", b"123"), ("b.fits", b"4567")]
    c1, l1 = build_container(entries)
    c2, l2 = build_container(entries)
    assert c1 == c2 and l1 == l2


def test_fits_blobs_index_crud(tmp_path):
    from exonit.metadata_cache import MetadataCache

    mc = MetadataCache(str(tmp_path / "mc.db"))
    rows = [
        ("a.fits", "containers/s2/c1.tar", 512, 55000),
        ("b.fits", "containers/s2/c1.tar", 56320, 40000),
    ]
    mc.add_fits_blobs_bulk(rows)
    assert mc.get_fits_blob("a.fits") == ("containers/s2/c1.tar", 512, 55000)
    assert mc.get_fits_blob("b.fits") == ("containers/s2/c1.tar", 56320, 40000)
    assert mc.get_fits_blob("missing.fits") is None
    # upsert: re-pack moves the file to a new container
    mc.add_fits_blobs_bulk([("a.fits", "containers/s2/c2.tar", 1024, 55000)])
    assert mc.get_fits_blob("a.fits") == ("containers/s2/c2.tar", 1024, 55000)
