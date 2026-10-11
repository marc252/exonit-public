"""Tests for the EXONIT_LORA_STRONG_ONLY flag on the server's LoRA counter.

Checks that:
- Flag OFF: behaviour identical to the original (counts every cnn>=0.7 without expl).
- Flag ON: counts only planet_candidate.
- Key case: 0 unexplained planet_candidate + N>0 unexplained non-candidates
  → ON=0, OFF=N.

The _lora_workload_counts() function is exercised directly against an
in-memory sqlite DB through ExoNitDB, without bringing up the whole server.
"""

import os
import pytest
import sqlite3

from exonit.db import ExoNitDB
from exonit.server import _lora_workload_counts


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def mem_db(tmp_path):
    """ExoNitDB in-memory isolated per test."""
    db_path = str(tmp_path / "test_lora.db")
    db = ExoNitDB(db_path)
    yield db
    db.conn.close()


def _insert_star(db, tic_id, status="completed"):
    db.conn.execute(
        "INSERT OR REPLACE INTO stars (tic_id, status) VALUES (?, ?)",
        (tic_id, status),
    )
    db.conn.commit()


def _insert_candidate(db, tic_id, cnn_score, claude_verdict, claude_analysis=None,
                      exominer_ffi_score=None):
    """Insert a minimal candidate with the fields the queries use."""
    db.conn.execute(
        """INSERT INTO candidates
           (tic_id, period_days, t0, duration_days, snr, depth_ppm, cnn_score,
            claude_verdict, claude_analysis, exominer_ffi_score)
           VALUES (?, 1.0, 0.0, 0.1, 10.0, 1000, ?, ?, ?, ?)""",
        (tic_id, cnn_score, claude_verdict, claude_analysis, exominer_ffi_score),
    )
    db.conn.commit()


# ---------------------------------------------------------------------------
# Base behaviour tests (flag OFF)
# ---------------------------------------------------------------------------

class TestFlagOff:
    """With the flag OFF the behaviour must be identical to the original code."""

    def test_missing_counts_all_cnn_ge_07_without_analysis(self, mem_db):
        """Flag OFF: lora_missing counts every cnn>=0.7 without analysis."""
        _insert_star(mem_db, "TIC1")
        _insert_star(mem_db, "TIC2")
        _insert_star(mem_db, "TIC3")
        # cnn>=0.7, no analysis → must count
        _insert_candidate(mem_db, "TIC1", 0.8, "planet_candidate", claude_analysis=None)
        _insert_candidate(mem_db, "TIC2", 0.9, "false_positive", claude_analysis=None)
        # cnn<0.7 → must not count
        _insert_candidate(mem_db, "TIC3", 0.5, "false_positive", claude_analysis=None)

        stale, missing = _lora_workload_counts(mem_db, strong_only=False)
        assert missing == 2, f"Expected 2, got {missing}"

    def test_stale_counts_all_verdicts(self, mem_db):
        """Flag OFF: lora_stale counts any verdict with a non-lora-v2 analysis."""
        _insert_star(mem_db, "TIC1")
        _insert_star(mem_db, "TIC2")
        _insert_candidate(mem_db, "TIC1", 0.8, "planet_candidate",
                          claude_analysis="Long Claude analysis without lora-v2 prefix " + "x" * 50)
        _insert_candidate(mem_db, "TIC2", 0.8, "false_positive",
                          claude_analysis="Long Claude analysis without lora-v2 prefix " + "x" * 50)

        stale, _ = _lora_workload_counts(mem_db, strong_only=False)
        assert stale == 2, f"Expected 2, got {stale}"

    def test_lora_v2_analysis_not_counted_as_stale(self, mem_db):
        """An analysis with the [lora-v2] prefix must not count as stale."""
        _insert_star(mem_db, "TIC1")
        _insert_candidate(mem_db, "TIC1", 0.8, "planet_candidate",
                          claude_analysis="[lora-v2] Modern, quite long explanation " + "x" * 50)

        stale, _ = _lora_workload_counts(mem_db, strong_only=False)
        assert stale == 0

    def test_empty_db_returns_zeros(self, mem_db):
        stale, missing = _lora_workload_counts(mem_db, strong_only=False)
        assert stale == 0
        assert missing == 0


# ---------------------------------------------------------------------------
# Gate behaviour tests (flag ON)
# ---------------------------------------------------------------------------

class TestFlagOn:
    """With the flag ON only planet_candidate is counted."""

    def test_missing_only_planet_candidate(self, mem_db):
        """Flag ON: lora_missing ignores non-planet_candidate without analysis."""
        _insert_star(mem_db, "TIC1")
        _insert_star(mem_db, "TIC2")
        _insert_star(mem_db, "TIC3")
        _insert_candidate(mem_db, "TIC1", 0.8, "planet_candidate", claude_analysis=None)
        _insert_candidate(mem_db, "TIC2", 0.9, "false_positive", claude_analysis=None)
        _insert_candidate(mem_db, "TIC3", 0.9, "eclipsing_binary", claude_analysis=None)

        stale, missing = _lora_workload_counts(mem_db, strong_only=True)
        assert missing == 1, f"Expected 1 (planet_candidate), got {missing}"

    def test_stale_only_planet_candidate(self, mem_db):
        """Flag ON: lora_stale ignores non-planet_candidate with a legacy analysis."""
        _insert_star(mem_db, "TIC1")
        _insert_star(mem_db, "TIC2")
        long_legacy = "Legacy Claude analysis without prefix " + "x" * 50
        _insert_candidate(mem_db, "TIC1", 0.8, "planet_candidate",
                          claude_analysis=long_legacy)
        _insert_candidate(mem_db, "TIC2", 0.8, "false_positive",
                          claude_analysis=long_legacy)

        stale, _ = _lora_workload_counts(mem_db, strong_only=True)
        assert stale == 1, f"Expected 1 (planet_candidate), got {stale}"

    def test_key_case_zero_planet_candidates_pending(self, mem_db):
        """Key case: 0 planet_candidate without expl + N>0 non-candidates without expl.

        Flag ON must give 0 (the bar must not show 0/N).
        Flag OFF must give N.
        """
        N = 5
        for i in range(N):
            tic = f"TIC{i}"
            _insert_star(mem_db, tic)
            _insert_candidate(mem_db, tic, 0.8, "false_positive", claude_analysis=None)

        stale_on, missing_on = _lora_workload_counts(mem_db, strong_only=True)
        stale_off, missing_off = _lora_workload_counts(mem_db, strong_only=False)

        assert missing_on == 0, f"Flag ON: expected 0, got {missing_on}"
        assert missing_off == N, f"Flag OFF: expected {N}, got {missing_off}"
        assert stale_on == 0
        assert stale_off == 0

    def test_mixed_verdicts_with_some_planet_candidates(self, mem_db):
        """Mix of verdicts: ON counts only the planet_candidate rows."""
        _insert_star(mem_db, "TIC_PC1")
        _insert_star(mem_db, "TIC_PC2")
        _insert_star(mem_db, "TIC_FP1")
        _insert_star(mem_db, "TIC_FP2")
        _insert_star(mem_db, "TIC_FP3")
        long_legacy = "Legacy analysis text that is longer than eighty characters " + "x" * 30

        # 2 planet_candidate: 1 without expl (missing), 1 legacy (stale)
        _insert_candidate(mem_db, "TIC_PC1", 0.9, "planet_candidate", claude_analysis=None)
        _insert_candidate(mem_db, "TIC_PC2", 0.9, "planet_candidate",
                          claude_analysis=long_legacy)
        # 3 false_positive: mix of missing/stale/[lora-v2]
        _insert_candidate(mem_db, "TIC_FP1", 0.8, "false_positive", claude_analysis=None)
        _insert_candidate(mem_db, "TIC_FP2", 0.8, "false_positive",
                          claude_analysis=long_legacy)
        _insert_candidate(mem_db, "TIC_FP3", 0.8, "false_positive",
                          claude_analysis="[lora-v2] Modern analysis " + "x" * 60)

        stale_on, missing_on = _lora_workload_counts(mem_db, strong_only=True)
        stale_off, missing_off = _lora_workload_counts(mem_db, strong_only=False)

        # ON: 1 stale (TIC_PC2 legacy) + 1 missing (TIC_PC1)
        assert stale_on == 1, f"stale ON: expected 1, got {stale_on}"
        assert missing_on == 1, f"missing ON: expected 1, got {missing_on}"
        # OFF: 2 stale (TIC_PC2 + TIC_FP2 legacy) + 2 missing (TIC_PC1 + TIC_FP1)
        assert stale_off == 2, f"stale OFF: expected 2, got {stale_off}"
        assert missing_off == 2, f"missing OFF: expected 2, got {missing_off}"

    def test_completed_star_required_for_missing(self, mem_db):
        """lora_missing requires star.status='completed'; non-completed stars are ignored."""
        _insert_star(mem_db, "TIC_PEND", status="pending")
        _insert_candidate(mem_db, "TIC_PEND", 0.9, "planet_candidate", claude_analysis=None)

        stale, missing = _lora_workload_counts(mem_db, strong_only=True)
        assert missing == 0, "Must not count candidates of non-completed stars"


# ---------------------------------------------------------------------------
# OR-gate (2026-07-11, docs/reviews/injection_e2e_results_2026-07-11.md +
# docs/reviews/ffi_cnn_investigation_2026-07-11.md §5): lora_missing must
# stay in lockstep with the population generate_missing_explanations.py
# --strong-only actually explains (claude_verdict='planet_candidate',
# regardless of cnn) — a cnn<0.7 em_ffi-rescued planet_candidate must still
# be counted, or the progress bar undercounts real pending work.
# ---------------------------------------------------------------------------

class TestOrGateEmFfiRescue:
    def test_missing_counts_em_ffi_rescued_low_cnn_candidate(self, mem_db):
        _insert_star(mem_db, "TIC_RESCUE")
        # cnn<0.7 alone would have failed the old (pre-OR-gate) predicate.
        _insert_candidate(mem_db, "TIC_RESCUE", cnn_score=0.4,
                          claude_verdict="planet_candidate",
                          claude_analysis=None, exominer_ffi_score=0.6)

        stale, missing = _lora_workload_counts(mem_db, strong_only=False)
        assert missing == 1, f"em_ffi-rescued row must count; got {missing}"

    def test_missing_excludes_low_cnn_low_em_ffi_candidate(self, mem_db):
        _insert_star(mem_db, "TIC_NO_RESCUE")
        _insert_candidate(mem_db, "TIC_NO_RESCUE", cnn_score=0.4,
                          claude_verdict="planet_candidate",
                          claude_analysis=None, exominer_ffi_score=0.3)

        _, missing = _lora_workload_counts(mem_db, strong_only=False)
        assert missing == 0, (
            "neither cnn>=0.7 nor exominer_ffi_score>=0.5 -> must not count"
        )

    def test_missing_excludes_low_cnn_null_em_ffi_candidate(self, mem_db):
        _insert_star(mem_db, "TIC_NULL_FFI")
        _insert_candidate(mem_db, "TIC_NULL_FFI", cnn_score=0.4,
                          claude_verdict="planet_candidate",
                          claude_analysis=None, exominer_ffi_score=None)

        _, missing = _lora_workload_counts(mem_db, strong_only=False)
        assert missing == 0, "NULL exominer_ffi_score must never crash or count"

    def test_missing_counts_em_ffi_rescue_with_strong_only(self, mem_db):
        """Flag ON (strong_only) must ALSO see the em_ffi-rescued row —
        the whole point is keeping this count in lockstep with the actual
        --strong-only explanation workload."""
        _insert_star(mem_db, "TIC_RESCUE_STRONG")
        _insert_candidate(mem_db, "TIC_RESCUE_STRONG", cnn_score=0.4,
                          claude_verdict="planet_candidate",
                          claude_analysis=None, exominer_ffi_score=0.55)

        _, missing = _lora_workload_counts(mem_db, strong_only=True)
        assert missing == 1
