"""Single source of truth for the FPP gate thresholds and verdict cutoffs.

Both FPP backends (``fpp_native``, the production path, and the legacy
``triceratops_fpp`` fallback) import from here so the scientific thresholds
can never drift between them. The server-side eager FPP-gate
(``exonit/server.py``, EXONIT_FPP_GATE) mirrors ``should_run_triceratops``
in SQL — keep them bit-identical if you ever change these values.
"""

# Gate: only candidates worth the Monte Carlo cost get an FPP.
SNR_THRESHOLD = 10.0
CNN_THRESHOLD = 0.7

# Verdict cutoffs (Giacalone+ 2020): FPP<0.015 AND NFPP<0.001 → validated.
FPP_VALIDATED = 0.015
FPP_CANDIDATE = 0.5
NFPP_VALIDATED = 0.001


def should_run_triceratops(candidate, vetting_result, cnn_score=None,
                           snr_threshold=SNR_THRESHOLD,
                           cnn_threshold=CNN_THRESHOLD):
    """Decide whether to run FPP on a candidate (shared by both backends)."""
    if not vetting_result.get("passed", False):
        return False
    if candidate.get("snr", 0) < snr_threshold:
        return False
    if cnn_score is not None and cnn_score < cnn_threshold:
        return False
    return True
