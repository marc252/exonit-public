"""Shared guard — the RESERVED TOI hold-out must never enter model training.

The reserved hold-out (`data/eval/toi_holdout_reserved_v1.txt`, built by
`scripts/eval/build_toi_holdout_reserved.py`) is the model-unseen test set for
honest specificity measurement (see docs/VALIDATION_REPORT.md). It is only
honest if NO model ever trains on it, so every split assigner / dataset builder
must route these TICs OUT of the ``train`` split.

Fails LOUDLY, never silently: a missing file or an absent TIC column prints a
stderr WARNING and is a no-op, so a mis-wiring is visible rather than quietly
poisoning the hold-out.
"""
from __future__ import annotations

import sys
from pathlib import Path

RESERVED_PATH = (Path(__file__).resolve().parents[1]
                 / "data" / "eval" / "toi_holdout_reserved_v1.txt")

_TIC_COL_CANDIDATES = ("tic_id", "tic", "TIC ID", "ticid", "TIC")


def normalize_tic(x) -> str:
    """Canonical TIC key: int-string when possible ('12345'), else stripped."""
    try:
        return str(int(x))
    except (ValueError, TypeError):
        return str(x).strip()


def reserved_holdout_tics(path=RESERVED_PATH) -> set[str]:
    """Reserved TIC ids as normalized int-strings; empty + WARNING if missing."""
    p = Path(path)
    if not p.exists():
        print(f"[reserved-holdout] WARNING: {p} missing — NO TICs excluded",
              file=sys.stderr)
        return set()
    return {normalize_tic(ln) for ln in p.read_text().splitlines() if ln.strip()}


def exclude_reserved(df, tic_col=None, *, path=RESERVED_PATH, label=""):
    """Drop rows whose TIC is reserved; return the filtered df.

    Use only where dropping rows is safe (simple tic-keyed pools). For datasets
    with row-aligned side artefacts (e.g. H5 views), reassign the split instead
    (see ``reassign_reserved_split``).
    """
    reserved = reserved_holdout_tics(path)
    if not reserved or df is None or len(df) == 0:
        return df
    if tic_col is None or tic_col not in getattr(df, "columns", []):
        tic_col = next((c for c in _TIC_COL_CANDIDATES if c in df.columns), None)
        if tic_col is None:
            print(f"[reserved-holdout] {label}: WARNING no TIC column in "
                  f"{list(df.columns)} — NO exclusion", file=sys.stderr)
            return df
    mask = df[tic_col].map(normalize_tic).isin(reserved)
    n = int(mask.sum())
    print(f"[reserved-holdout] {label}: excluded {n} rows / "
          f"{df.loc[mask, tic_col].map(normalize_tic).nunique() if n else 0} "
          f"reserved TICs (col '{tic_col}')", file=sys.stderr)
    return df[~mask].copy()


def reassign_reserved_split(df, tic_col="tic_id", split_col="split",
                            reserved_label="reserved", *, path=RESERVED_PATH,
                            label=""):
    """Force reserved TICs' ``split`` to ``reserved_label`` (out of 'train').

    Safe for row-aligned datasets: it never drops rows, only relabels the split
    so trainers that read ``split == 'train'`` skip them. Returns df (mutated).
    """
    reserved = reserved_holdout_tics(path)
    if not reserved or df is None or len(df) == 0 or tic_col not in df.columns:
        if tic_col not in getattr(df, "columns", []):
            print(f"[reserved-holdout] {label}: WARNING no '{tic_col}' column — "
                  "NO split reassignment", file=sys.stderr)
        return df
    mask = df[tic_col].map(normalize_tic).isin(reserved)
    n_tics = df.loc[mask, tic_col].map(normalize_tic).nunique() if mask.any() else 0
    df.loc[mask, split_col] = reserved_label
    print(f"[reserved-holdout] {label}: routed {n_tics} reserved TICs "
          f"({int(mask.sum())} rows) to split='{reserved_label}'", file=sys.stderr)
    return df
