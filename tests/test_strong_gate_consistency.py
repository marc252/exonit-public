"""Drift tripwire for the strong-candidate / FPP gate thresholds.

The predicate ``snr >= 10 AND cnn_score >= 0.7`` is intentionally embedded
in many SQL strings (queue queries, UI counts, partial-index definitions in
``exonit/db.py`` / ``exonit/server.py``) — SQLite's partial-index matching
needs the literal text, so we do NOT build it dynamically. The price is
drift risk: change one copy and the worker queue and the UI silently
disagree (this family of bug produced the 2026-06-08 ETA error).

This test greps every threshold occurrence and fails if any copy diverges
from the canonical values in ``worker/fpp_thresholds.py``.
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "worker"))

from fpp_thresholds import SNR_THRESHOLD, CNN_THRESHOLD  # noqa: E402

SOURCES = [ROOT / "exonit" / "db.py", ROOT / "exonit" / "server.py"]

SNR_RE = re.compile(r"\bsnr\s*>=\s*(\d+(?:\.\d+)?)")
CNN_RE = re.compile(r"\bcnn_score\s*>=\s*(\d+(?:\.\d+)?)")


def _occurrences(regex):
    out = []
    for src in SOURCES:
        text = src.read_text()
        for m in regex.finditer(text):
            line_no = text[: m.start()].count("\n") + 1
            out.append((src.name, line_no, float(m.group(1))))
    return out


def test_snr_gate_consistent_everywhere():
    occ = _occurrences(SNR_RE)
    assert occ, "expected snr>= predicates in db.py/server.py"
    bad = [o for o in occ if o[2] != SNR_THRESHOLD]
    assert not bad, (
        f"snr gate drift vs worker/fpp_thresholds.SNR_THRESHOLD={SNR_THRESHOLD}: {bad}"
    )


def test_cnn_gate_consistent_everywhere():
    occ = _occurrences(CNN_RE)
    assert occ, "expected cnn_score>= predicates in db.py/server.py"
    bad = [o for o in occ if o[2] != CNN_THRESHOLD]
    assert not bad, (
        f"cnn gate drift vs worker/fpp_thresholds.CNN_THRESHOLD={CNN_THRESHOLD}: {bad}"
    )
