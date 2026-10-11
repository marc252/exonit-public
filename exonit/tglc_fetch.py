"""Shared MinIO/TGLC FITS-fetch helpers (D11 dedup, 2026-07-02).

Canonical home for the fetch helpers that used to be byte-identical (modulo
typing style) copies in ``scripts/build_ffi_dataset_v2.py`` and
``scripts/build_exominer_ffi_dataset_v2.py``:

  * ``fetch_fits_bytes``    — MinIO packed-container range read, per-object
    fallback.
  * ``sector_from_filename`` — parse the TESS sector out of an HLSP/TGLC
    basename.

NOTE: the two scripts' TGLC-bytes-to-arrays readers
(``_read_tglc_from_bytes`` in build_ffi_dataset_v2.py vs
``_read_tglc_full_from_bytes`` in build_exominer_ffi_dataset_v2.py) are
**not** consolidated here — the latter additionally extracts the X/Y
centroid columns for the ExoMiner centroid branch, so the two are not
behavior-identical (see docs/reviews/2026-07-02-opus48-cleanup-review.md
item D11 and progress/impl-training-consolidation-2026-07-02.md). Each
script keeps its own reader.
"""
from __future__ import annotations

import logging
import re

LOG = logging.getLogger("exonit.tglc_fetch")


def fetch_fits_bytes(store, cache, filename: str) -> bytes | None:
    """Fetch one FITS from MinIO: packed-container range read, else per-object."""
    try:
        blob = cache.get_fits_blob(filename)
    except Exception:
        blob = None
    if blob is not None:
        container_key, offset, length = blob
        try:
            return store.get_range(container_key, offset, length)
        except Exception as e:
            LOG.debug("range read failed for %s: %s; trying per-object", filename, e)
    try:
        return store.get_bytes(filename)
    except Exception as e:
        LOG.debug("per-object get failed for %s: %s", filename, e)
        return None


def sector_from_filename(filename: str) -> int | None:
    m = re.search(r"-s(\d{4})-", filename)
    return int(m.group(1)) if m else None


# Backwards-compatible underscore aliases: the importing scripts call these
# as ``_fetch_fits_bytes`` / ``_sector_from_filename`` throughout their
# bodies, so importers can bind the underscore names to these implementations
# without touching every call site.
_fetch_fits_bytes = fetch_fits_bytes
_sector_from_filename = sector_from_filename
