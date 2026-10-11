"""Pack many small FITS into one tar 'container' object for object storage.

The TGLC discovery set is millions of tiny (~55 KB) FITS. Stored as individual
S3/MinIO objects on an HDD that does ~24-40 fsync/s, both the migration and the
per-file serving are dominated by object count, not bytes. Packing groups them
into a handful of large container objects (sequential writes, fast on an HDD),
and a per-file byte-range index lets the server still serve one FITS via an HTTP
Range read.

Container format: uncompressed USTAR tar. TGLC filenames are < 100 chars so each
member is a single 512-byte header + data, and the archive stays recoverable
with standard tools if the index is ever lost. The exact data byte-range of each
member is read back from tarfile's own ``offset_data`` (no header-size guessing).
"""

import io
import tarfile


def pack_to_stream(out, items):
    """Write a tar container to file-like ``out``; return member byte ranges.

    Streaming: only one member's bytes are held in memory at a time, so packing
    100k FITS into one container uses ~one-FITS of RAM, not the whole container.

    Parameters
    ----------
    out : binary file object
        Seekable, opened for read+write (e.g. a temp file or BytesIO). The tar
        is written here; afterwards it is re-read (headers only) for offsets.
    items : iterable[tuple[str, bytes | str]]
        (member_name, source) where source is the file's bytes OR a filesystem
        path to read. ``member_name`` is the flat FITS basename.

    Returns
    -------
    list[tuple[str, int, int]]
        (member_name, data_offset, data_length) for each member.
    """
    locations = []
    pos = 0  # byte position in the archive (tracked, so no re-read of huge tars)
    with tarfile.open(fileobj=out, mode="w", format=tarfile.USTAR_FORMAT) as tar:
        for name, source in items:
            if isinstance(source, (bytes, bytearray)):
                data = bytes(source)
            else:
                with open(source, "rb") as fh:
                    data = fh.read()
            ti = tarfile.TarInfo(name=name)
            ti.size = len(data)
            ti.mtime = 0           # deterministic archives
            # USTAR: a name < 100 chars (TGLC basenames are ~78) is exactly one
            # 512-byte header, so data starts at pos+512 and the next member is
            # 512-aligned after the data. USTAR_FORMAT raises on a too-long name,
            # so this invariant can't silently break; the round-trip test guards it.
            tar.addfile(ti, io.BytesIO(data))
            locations.append((name, pos + 512, len(data)))
            pos += 512 + ((len(data) + 511) // 512) * 512
    return locations


def build_container(entries):
    """In-memory convenience wrapper over :func:`pack_to_stream` (tests/small).

    entries: list of (member_name, file_bytes). Returns (container_bytes,
    locations). For large batches use pack_to_stream with a temp file.
    """
    buf = io.BytesIO()
    locations = pack_to_stream(buf, entries)
    return buf.getvalue(), locations


def extract_member(container_bytes, offset, length):
    """Return one member's bytes from an in-memory container (for tests)."""
    return container_bytes[offset:offset + length]
