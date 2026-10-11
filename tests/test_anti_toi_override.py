"""Tests for the anti-TOI override: SYSTEMATIC_VETO / DEEP_SYSTEMATIC_VETO must
never demote a CONFIRMED planet (TOI TFOPWG Disposition KP/CP).

See exonit.systematic_veto (load_toi_protection, is_toi_protected,
filter_anti_toi_protected, partition_deep_veto_targets), the row-level use in
scripts/discovery_postprocess.py (Steps 1 and 1c), and the standalone-tool use
+ repair mode in scripts/apply_systematic_veto.py.

Known victims (2026-06-30 miss-analyst, s1) this closes:
  TIC 176956893 (CP, P=6.907d), TIC 336732544 (KP, P=3.277d),
  TIC 364395234 (CP, P=14.255d).

Constraints (matching tests/test_orchestrator_postprocess_cleanup_2026_07_02.py):
- NEVER touch the live DB data/exonit.db or production PostgreSQL — the
  _never_touch_prod_db autouse fixture in tests/conftest.py already guards
  this; here we additionally only ever use tmp_path SQLite DBs via op_session.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from exonit.db_backend import op_session  # noqa: E402
from exonit.systematic_veto import (  # noqa: E402
    filter_anti_toi_protected,
    is_toi_protected,
    load_toi_protection,
    partition_deep_veto_targets,
)

import discovery_postprocess as pp  # noqa: E402

import apply_systematic_veto as veto_tool  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_catalog(tmp_path, rows):
    """Write a minimal TOI catalog CSV with the 4 required columns.

    ``rows``: list of (tic_id, toi, disposition, period_days_or_empty).
    """
    path = tmp_path / "toi_catalog.csv"
    lines = ["TIC ID,TOI,TFOPWG Disposition,Period (days)"]
    for tic, toi, disp, period in rows:
        lines.append(f"{tic},{toi},{disp},{period}")
    path.write_text("\n".join(lines) + "\n")
    return str(path)


# ---------------------------------------------------------------------------
# 1. load_toi_protection
# ---------------------------------------------------------------------------

class TestLoadToiProtection:
    def test_keeps_only_kp_cp(self, tmp_path):
        path = _write_catalog(tmp_path, [
            ("176956893", "2184.01", "CP", "6.906786841151312"),
            ("336732544", "7486.01", "KP", "3.2771117876140976"),
            ("111111111", "1.01", "PC", "5.0"),   # not confirmed — excluded
            ("222222222", "2.01", "FP", "5.0"),   # not confirmed — excluded
            ("333333333", "3.01", "APC", "5.0"),  # not confirmed — excluded
        ])
        protection = load_toi_protection(path)
        assert set(protection) == {176956893, 336732544}
        assert protection[176956893] == pytest.approx([6.906786841151312])
        assert protection[336732544] == pytest.approx([3.2771117876140976])

    def test_multiple_tois_per_tic(self, tmp_path):
        """A TIC hosting two confirmed TOIs gets both periods in its list."""
        path = _write_catalog(tmp_path, [
            ("176956893", "2184.01", "CP", "6.9"),
            ("176956893", "2184.02", "CP", "14.0"),
        ])
        protection = load_toi_protection(path)
        assert set(protection[176956893]) == {6.9, 14.0}

    def test_missing_file_returns_empty_and_warns(self, tmp_path):
        missing_path = str(tmp_path / "does_not_exist.csv")
        with pytest.warns(RuntimeWarning, match="not found"):
            protection = load_toi_protection(missing_path)
        assert protection == {}

    def test_missing_required_columns_returns_empty_and_warns(self, tmp_path):
        path = tmp_path / "bad_catalog.csv"
        path.write_text("TIC ID,TOI\n176956893,2184.01\n")
        with pytest.warns(RuntimeWarning, match="missing column"):
            protection = load_toi_protection(str(path))
        assert protection == {}

    def test_unparseable_period_becomes_none(self, tmp_path):
        """A malformed 'Period (days)' cell yields a None entry (TIC-level
        fallback protection), never crashes."""
        path = _write_catalog(tmp_path, [
            ("176956893", "2184.01", "CP", "not_a_number"),
        ])
        protection = load_toi_protection(path)
        assert protection[176956893] == [None]

    def test_non_numeric_tic_id_skipped(self, tmp_path):
        path = _write_catalog(tmp_path, [
            ("not_a_tic", "2184.01", "CP", "6.9"),
            ("176956893", "2184.02", "KP", "3.3"),
        ])
        protection = load_toi_protection(path)
        assert set(protection) == {176956893}


# ---------------------------------------------------------------------------
# 2. is_toi_protected
# ---------------------------------------------------------------------------

class TestIsToiProtected:
    PROTECTION = {
        176956893: [6.906786841151312],
        336732544: [3.2771117876140976],
        364395234: [14.2545623191155],
        999999999: [None],  # unparseable catalog period -> TIC-level fallback
    }

    def test_exact_period_match(self):
        assert is_toi_protected(176956893, 6.906786841151312, self.PROTECTION)

    def test_within_tolerance_boundary(self):
        """A period 0.5 % off (well inside the default 1 % rel_tol) is protected."""
        p = 3.2771117876140976
        assert is_toi_protected(336732544, p * 1.005, self.PROTECTION, rel_tol=0.01)

    def test_outside_tolerance_boundary(self):
        """A period 2 % off (outside the default 1 % rel_tol) is NOT protected."""
        p = 14.2545623191155
        assert not is_toi_protected(364395234, p * 1.02, self.PROTECTION, rel_tol=0.01)

    def test_none_period_fallback_protects_regardless_of_candidate_period(self):
        """A TIC whose catalog period was unparseable (None) is protected at
        the TIC level — any candidate period, including a very different one."""
        assert is_toi_protected(999999999, 0.71, self.PROTECTION)
        assert is_toi_protected(999999999, None, self.PROTECTION)

    def test_non_protected_tic_returns_false(self):
        assert not is_toi_protected(111111111, 5.0, self.PROTECTION)

    def test_protected_tic_but_candidate_period_none_returns_false(self):
        """Protected TIC (with a real catalog period), candidate period is
        None -> cannot confirm the match -> not protected (only the None-
        catalog-period fallback protects unconditionally)."""
        assert not is_toi_protected(176956893, None, self.PROTECTION)

    def test_str_tic_id_normalized(self):
        """tic_id may arrive as str (as in DB rows) — normalises to int."""
        assert is_toi_protected("176956893", 6.906786841151312, self.PROTECTION)

    def test_int_tic_id_also_works(self):
        assert is_toi_protected(176956893, 6.906786841151312, self.PROTECTION)

    def test_non_numeric_tic_id_returns_false(self):
        assert not is_toi_protected("not-a-tic", 5.0, self.PROTECTION)

    def test_empty_protection_dict_returns_false(self):
        assert not is_toi_protected(176956893, 6.9, {})


# ---------------------------------------------------------------------------
# 2b. is_toi_protected — harmonic/sub-harmonic alias matches (2026-07-10)
# ---------------------------------------------------------------------------

class TestIsToiProtectedHarmonics:
    """TIC 260004324: catalog P=3.8143 d, real production victim demoted at
    the P/5 sub-harmonic (SYSTEMATIC_VETO). See
    docs/reviews/meta_analysis_discovery_yield_2026-07-10.md §Forat A."""

    PROTECTION = {260004324: [3.8143]}

    def test_direct_match_still_protected(self):
        """Regression: the direct-match case (unchanged behaviour) must keep
        working exactly as before the harmonic extension."""
        assert is_toi_protected(260004324, 3.8143, self.PROTECTION)

    def test_superharmonic_x2_protected(self):
        """Detected period = 2x the catalog period (HARMONIC_FACTORS_SUPER)."""
        assert is_toi_protected(260004324, 3.8143 * 2, self.PROTECTION)

    def test_subharmonic_div3_protected(self):
        """Detected period = catalog period / 3 (HARMONIC_FACTORS_SUB)."""
        assert is_toi_protected(260004324, 3.8143 / 3, self.PROTECTION)

    def test_subharmonic_div5_protected_real_victim(self):
        """The exact real-world victim: TIC 260004324 detected at P/5."""
        assert is_toi_protected(260004324, 3.8143 / 5, self.PROTECTION)

    def test_harmonic_off_by_more_than_tolerance_not_protected(self):
        """A period 2 % off a harmonic alias (outside the 1 % rel_tol) must
        NOT be protected."""
        assert not is_toi_protected(
            260004324, (3.8143 / 5) * 1.02, self.PROTECTION, rel_tol=0.01)

    def test_non_kp_cp_tic_not_protected_even_at_harmonic_period(self):
        """A TIC absent from the protection dict (not a confirmed KP/CP host)
        gets no protection, harmonic or otherwise."""
        assert not is_toi_protected(999999999, 3.8143 / 5, self.PROTECTION)


# ---------------------------------------------------------------------------
# 3. filter_anti_toi_protected — Step 1 (SYSTEMATIC_VETO) filtering logic
# ---------------------------------------------------------------------------

class TestFilterAntiToiProtected:
    def _cand(self, tic, period, cid=None):
        return {"id": cid, "tic_id": tic, "period_days": period,
                "claude_verdict": "planet_candidate"}

    def test_protected_candidate_excluded_from_veto(self, capsys):
        """Core acceptance case: a confirmed CP whose period coincides with a
        systematic must survive the veto filter (TIC 176956893 analogue)."""
        protection = {176956893: [6.906786841151312]}
        vetoed_raw = [
            self._cand("176956893", 6.906786841151312, cid=1),  # protected
            self._cand("999999999", 6.91, cid=2),                # not protected
        ]
        survivors, n_protected = filter_anti_toi_protected(
            vetoed_raw, protection, "SYSTEMATIC_VETO")
        assert n_protected == 1
        assert [c["id"] for c in survivors] == [2]
        out = capsys.readouterr().out
        assert "[anti-toi] PROTECTED from SYSTEMATIC_VETO: TIC 176956893" in out

    def test_no_protection_no_change(self):
        vetoed_raw = [self._cand("111", 5.0, cid=1), self._cand("222", 7.0, cid=2)]
        survivors, n_protected = filter_anti_toi_protected(vetoed_raw, {}, "SYSTEMATIC_VETO")
        assert n_protected == 0
        assert len(survivors) == 2

    def test_all_three_known_victims_protected(self):
        """The 3 known-victim TICs, each vetoed at its own real period, all
        survive the filter."""
        protection = {
            176956893: [6.906786841151312],
            336732544: [3.2771117876140976],
            364395234: [14.2545623191155],
        }
        vetoed_raw = [
            self._cand("176956893", 6.907, cid=1),
            self._cand("336732544", 3.277, cid=2),
            self._cand("364395234", 14.255, cid=3),
        ]
        survivors, n_protected = filter_anti_toi_protected(
            vetoed_raw, protection, "SYSTEMATIC_VETO")
        assert n_protected == 3
        assert survivors == []


# ---------------------------------------------------------------------------
# 4. partition_deep_veto_targets — Step 1c (DEEP_SYSTEMATIC_VETO) row-level logic
# ---------------------------------------------------------------------------

class TestPartitionDeepVetoTargets:
    def test_tic_with_one_protected_and_one_unprotected_row(self, capsys):
        """A TIC with two planet_candidate signals: one matches a KP/CP TOI
        (protected), the other doesn't (still vetoed). The TIC survives in
        to_update, but with the protected row's id excluded."""
        protection = {176956893: [6.906786841151312]}
        to_update = {"176956893"}
        pc_rows = [
            {"id": 10, "tic_id": "176956893", "period_days": 6.906786841151312,
             "claude_verdict": "planet_candidate"},   # protected
            {"id": 11, "tic_id": "176956893", "period_days": 25.0,
             "claude_verdict": "planet_candidate"},   # not protected
        ]
        new_to_update, protected_ids_by_tic = partition_deep_veto_targets(
            to_update, pc_rows, protection)
        assert new_to_update == {"176956893"}
        assert protected_ids_by_tic == {"176956893": {10}}
        out = capsys.readouterr().out
        assert "[anti-toi] PROTECTED from DEEP_SYSTEMATIC_VETO: TIC 176956893" in out

    def test_tic_with_every_row_protected_drops_out_entirely(self):
        """A TIC whose SOLE planet_candidate row is protected drops out of
        to_update — no UPDATE at all for it."""
        protection = {336732544: [3.2771117876140976]}
        to_update = {"336732544"}
        pc_rows = [
            {"id": 20, "tic_id": "336732544", "period_days": 3.2771117876140976,
             "claude_verdict": "planet_candidate"},
        ]
        new_to_update, protected_ids_by_tic = partition_deep_veto_targets(
            to_update, pc_rows, protection)
        assert new_to_update == set()
        assert protected_ids_by_tic == {}

    def test_unrelated_tic_unaffected(self):
        protection = {176956893: [6.9]}
        to_update = {"555555555"}
        pc_rows = [
            {"id": 30, "tic_id": "555555555", "period_days": 25.0,
             "claude_verdict": "planet_candidate"},
        ]
        new_to_update, protected_ids_by_tic = partition_deep_veto_targets(
            to_update, pc_rows, protection)
        assert new_to_update == {"555555555"}
        assert protected_ids_by_tic == {}

    def test_empty_protection_is_a_no_op(self):
        to_update = {"1", "2", "3"}
        new_to_update, protected_ids_by_tic = partition_deep_veto_targets(
            to_update, [], {})
        assert new_to_update == to_update
        assert protected_ids_by_tic == {}

    def test_rows_outside_to_update_are_ignored(self):
        """A planet_candidate row for a TIC not in to_update must not affect
        the result (mirrors real usage where the caller passes a broader
        sample_cands list)."""
        protection = {1: [5.0]}
        to_update = {"2"}
        pc_rows = [
            {"id": 1, "tic_id": "1", "period_days": 5.0,
             "claude_verdict": "planet_candidate"},
            {"id": 2, "tic_id": "2", "period_days": 7.0,
             "claude_verdict": "planet_candidate"},
        ]
        new_to_update, protected_ids_by_tic = partition_deep_veto_targets(
            to_update, pc_rows, protection)
        assert new_to_update == {"2"}
        assert protected_ids_by_tic == {}


# ---------------------------------------------------------------------------
# 5. Integration: Step 1c's actual DB UPDATE, protected row(s) excluded
# ---------------------------------------------------------------------------

class TestStep1cUpdateExcludesProtectedRows:
    def test_update_with_not_in_leaves_protected_row_untouched(self, tmp_path):
        """Reproduces exactly the UPDATE discovery_postprocess.py's Step 1c
        issues for a TIC with a partially-protected row set: the protected
        id must keep claude_verdict='planet_candidate', the other must flip
        to likely_false_positive/DEEP_SYSTEMATIC_VETO."""
        db_path = str(tmp_path / "t.db")
        con = op_session(db_path)
        try:
            con.execute(
                "CREATE TABLE candidates (id INTEGER PRIMARY KEY, tic_id TEXT, "
                "period_days REAL, claude_verdict TEXT, cross_source_name TEXT)"
            )
            con.execute("INSERT INTO candidates (id, tic_id, period_days, claude_verdict) "
                        "VALUES (10, '176956893', 6.906786841151312, 'planet_candidate')")
            con.execute("INSERT INTO candidates (id, tic_id, period_days, claude_verdict) "
                        "VALUES (11, '176956893', 25.0, 'planet_candidate')")
            con.commit()

            protection = {176956893: [6.906786841151312]}
            to_update = {"176956893"}
            pc_rows = [dict(r) for r in con.execute(
                "SELECT id, tic_id, period_days, claude_verdict FROM candidates")]
            to_update, protected_ids_by_tic = partition_deep_veto_targets(
                to_update, pc_rows, protection)

            for tic in to_update:
                protected_ids = protected_ids_by_tic.get(tic)
                placeholders = ",".join("?" * len(protected_ids))
                con.execute(
                    "UPDATE candidates SET claude_verdict='likely_false_positive', "
                    "cross_source_name='DEEP_SYSTEMATIC_VETO' "
                    "WHERE tic_id=? AND claude_verdict='planet_candidate' "
                    f"AND id NOT IN ({placeholders})",
                    (tic, *protected_ids),
                )
            con.commit()

            row10 = dict(con.execute(
                "SELECT claude_verdict, cross_source_name FROM candidates WHERE id=10"
            ).fetchone())
            row11 = dict(con.execute(
                "SELECT claude_verdict, cross_source_name FROM candidates WHERE id=11"
            ).fetchone())
        finally:
            con.close()

        assert row10["claude_verdict"] == "planet_candidate", (
            "protected row (matches KP/CP TOI period) must NOT be demoted"
        )
        assert row11["claude_verdict"] == "likely_false_positive"
        assert row11["cross_source_name"] == "DEEP_SYSTEMATIC_VETO"


# ---------------------------------------------------------------------------
# 6. Repair mode — scripts/apply_systematic_veto.py --repair-anti-toi
# ---------------------------------------------------------------------------

class TestRepairAntiToi:
    def _make_db(self, tmp_path):
        db_path = str(tmp_path / "t.db")
        con = op_session(db_path)
        con.execute(
            "CREATE TABLE candidates (id INTEGER PRIMARY KEY, tic_id TEXT, "
            "period_days REAL, t0 REAL, claude_verdict TEXT, cross_source_name TEXT)"
        )
        con.commit()
        return con

    def test_report_mode_finds_fabricated_demoted_row_no_db_write(self, tmp_path, capsys):
        """A fabricated row that was wrongly demoted (SYSTEMATIC_VETO,
        tic_id+period matches a confirmed KP/CP TOI) is picked up by the
        report-mode SELECT predicate; the DB is NOT modified."""
        con = self._make_db(tmp_path)
        con.execute(
            "INSERT INTO candidates (id, tic_id, period_days, t0, claude_verdict, "
            "cross_source_name) VALUES (1, '176956893', 6.906786841151312, 1400.0, "
            "'likely_false_positive', 'SYSTEMATIC_VETO')"
        )
        # An unrelated demoted row that must NOT be touched (not TOI-protected).
        con.execute(
            "INSERT INTO candidates (id, tic_id, period_days, t0, claude_verdict, "
            "cross_source_name) VALUES (2, '111111111', 0.71, 1300.0, "
            "'likely_false_positive', 'SYSTEMATIC_VETO')"
        )
        con.commit()

        catalog_path = _write_catalog(tmp_path, [
            ("176956893", "2184.01", "CP", "6.906786841151312"),
        ])

        rc = veto_tool._repair_anti_toi(con, catalog_path, apply_repair=False)
        out = capsys.readouterr().out
        con.close()

        assert rc == 0
        assert "RESTORE: TIC 176956893" in out
        assert "111111111" not in out
        assert "report-only" in out

        # Re-open and confirm nothing changed.
        con2 = op_session(str(tmp_path / "t.db"))
        row1 = dict(con2.execute(
            "SELECT claude_verdict, cross_source_name FROM candidates WHERE id=1"
        ).fetchone())
        con2.close()
        assert row1["claude_verdict"] == "likely_false_positive"
        assert row1["cross_source_name"] == "SYSTEMATIC_VETO"

    def test_apply_repair_restores_row(self, tmp_path):
        """--apply-anti-toi-repair actually writes the restore, keeping an
        audit trail via cross_source_name='ANTI_TOI_RESTORED' (not nulled)."""
        con = self._make_db(tmp_path)
        con.execute(
            "INSERT INTO candidates (id, tic_id, period_days, t0, claude_verdict, "
            "cross_source_name) VALUES (1, '336732544', 3.2771117876140976, 1400.0, "
            "'likely_false_positive', 'DEEP_SYSTEMATIC_VETO')"
        )
        con.commit()

        catalog_path = _write_catalog(tmp_path, [
            ("336732544", "7486.01", "KP", "3.2771117876140976"),
        ])

        veto_tool._repair_anti_toi(con, catalog_path, apply_repair=True)
        row1 = dict(con.execute(
            "SELECT claude_verdict, cross_source_name FROM candidates WHERE id=1"
        ).fetchone())
        con.close()

        assert row1["claude_verdict"] == "planet_candidate"
        assert row1["cross_source_name"] == "ANTI_TOI_RESTORED", (
            "audit trail must be kept (not nulled)"
        )

    def test_no_demoted_rows_is_a_no_op(self, tmp_path, capsys):
        con = self._make_db(tmp_path)
        catalog_path = _write_catalog(tmp_path, [
            ("176956893", "2184.01", "CP", "6.9"),
        ])
        rc = veto_tool._repair_anti_toi(con, catalog_path, apply_repair=False)
        con.close()
        out = capsys.readouterr().out
        assert rc == 0
        assert "0 row(s) demoted" in out

    def test_missing_catalog_disables_repair_gracefully(self, tmp_path):
        con = self._make_db(tmp_path)
        with pytest.warns(RuntimeWarning):
            rc = veto_tool._repair_anti_toi(
                con, str(tmp_path / "missing.csv"), apply_repair=False)
        con.close()
        assert rc == 0


# ---------------------------------------------------------------------------
# 7. Review round 2 (2026-07-03) — Step 2+3 survivor-selection re-filter
#    (discovery_postprocess.py's `_strong_candidate_passes`) must ALSO honour
#    anti-TOI protection, or a protected row keeps claude_verdict=
#    'planet_candidate' in the DB yet is silently excluded from
#    survivors/finalists forever.
# ---------------------------------------------------------------------------

class TestStrongCandidatePasses:
    def _row(self, cid, tic, period, t0=1300.0):
        return {"id": cid, "tic_id": tic, "period_days": period, "t0": t0}

    def test_protected_row_survives_systematic_veto(self):
        """A confirmed CP whose period sits on a systematic (is_vetoed=True)
        must still pass — the anti-TOI override applies here exactly as it
        does at the Step 1 DB-write level."""
        sysp = [{"period": 6.906786841151312, "epoch_phase": 0.3, "n_stars": 40,
                 "concentration": 0.8, "epoch_agnostic": False}]
        protection = {176956893: [6.906786841151312]}
        row = self._row(1, "176956893", 6.906786841151312, t0=1000 * 6.906786841151312 + 0.3)
        assert pp._strong_candidate_passes(
            row, sysp, eb_hosts=set(), depth_gate_ids=set(), toi_protection=protection)

    def test_unprotected_row_still_excluded_by_systematic_veto(self):
        """Same systematic, but a DIFFERENT (non-TOI) TIC at the same period
        must still be excluded — the override is per-row/per-TOI, not global."""
        sysp = [{"period": 6.906786841151312, "epoch_phase": 0.3, "n_stars": 40,
                 "concentration": 0.8, "epoch_agnostic": False}]
        protection = {176956893: [6.906786841151312]}
        row = self._row(2, "999999999", 6.906786841151312, t0=1000 * 6.906786841151312 + 0.3)
        assert not pp._strong_candidate_passes(
            row, sysp, eb_hosts=set(), depth_gate_ids=set(), toi_protection=protection)

    def test_protected_row_survives_eb_hosts_membership(self):
        """A confirmed KP whose TIC is a deep-systematic/EB host (eb_hosts)
        must still pass."""
        protection = {336732544: [3.2771117876140976]}
        row = self._row(3, "336732544", 3.2771117876140976)
        assert pp._strong_candidate_passes(
            row, sysp=[], eb_hosts={"336732544"}, depth_gate_ids=set(),
            toi_protection=protection)

    def test_same_tic_protected_and_unprotected_rows_both_present(self):
        """Core acceptance case from the review: a protected row passes while
        an UNPROTECTED artefact row on the SAME TIC (different period,
        matches the eb_hosts/systematic membership) is still excluded."""
        protection = {176956893: [6.906786841151312]}
        eb_hosts = {"176956893"}  # TIC-level: both rows share this membership
        protected_row = self._row(10, "176956893", 6.906786841151312)
        artefact_row = self._row(11, "176956893", 25.0)
        assert pp._strong_candidate_passes(
            protected_row, sysp=[], eb_hosts=eb_hosts, depth_gate_ids=set(),
            toi_protection=protection), "protected row must survive"
        assert not pp._strong_candidate_passes(
            artefact_row, sysp=[], eb_hosts=eb_hosts, depth_gate_ids=set(),
            toi_protection=protection), "unprotected sibling row must still be excluded"

    def test_depth_gate_never_overridden_by_anti_toi(self):
        """DEPTH_GATE is a DIFFERENT veto — anti-TOI protection must NOT
        override it (explicitly out of scope per the review)."""
        protection = {176956893: [6.906786841151312]}
        row = self._row(4, "176956893", 6.906786841151312)
        assert not pp._strong_candidate_passes(
            row, sysp=[], eb_hosts=set(), depth_gate_ids={4},
            toi_protection=protection)

    def test_depth_gate_unaffected_by_harmonic_protection(self):
        """2026-07-10 extension check: even though the row's period is now a
        harmonic-match protected period (not just a direct match), DEPTH_GATE
        membership still short-circuits before ``is_toi_protected`` is
        consulted — the harmonic extension does not create a new leak in the
        depth-gate seam."""
        protection = {260004324: [3.8143]}
        row = self._row(6, "260004324", 3.8143 / 5)  # sub-harmonic, protected
        assert not pp._strong_candidate_passes(
            row, sysp=[], eb_hosts=set(), depth_gate_ids={6},
            toi_protection=protection)

    def test_unprotected_normal_candidate_unaffected(self):
        """No systematic, no eb_host, no depth-gate, no protection -> passes
        (no behaviour change for the common case)."""
        row = self._row(5, "555555555", 7.3)
        assert pp._strong_candidate_passes(
            row, sysp=[], eb_hosts=set(), depth_gate_ids=set(), toi_protection={})


class TestDossierDeepSystematicVetoConsistency:
    """Mirrors the exact set arithmetic discovery_postprocess.py's Step 6
    dossier section uses for the "Deep-systematic veto" ... "excluded from
    finalists" line, so a protected+surviving TIC is never listed there
    (which would contradict the "[anti-toi] PROTECTED ..." stdout line from
    Step 1c for the very same TIC and run)."""

    def test_protected_survivor_excluded_from_the_excluded_list(self):
        eb_hosts = {"176956893", "999999999"}
        planet_candidate_tics = {"176956893", "999999999"}
        # 176956893's protected row survived into `survivors` (Step 2+3);
        # 999999999 has no protection and was correctly excluded.
        eb_hosts_protected_survivors = {"176956893"}

        eb_with_pc = eb_hosts & planet_candidate_tics
        eb_with_pc_excluded = eb_with_pc - eb_hosts_protected_survivors

        assert "176956893" not in eb_with_pc_excluded, (
            "a protected TIC whose row survived must NOT be listed as "
            "excluded from finalists in the dossier"
        )
        assert "999999999" in eb_with_pc_excluded, (
            "a non-protected eb-host TIC must still be listed as excluded"
        )

    def test_no_protected_survivors_unchanged_behaviour(self):
        """With no anti-TOI overrides in play, the excluded list is exactly
        eb_with_pc — no behaviour change for the common case."""
        eb_hosts = {"111", "222"}
        planet_candidate_tics = {"111", "222"}
        eb_hosts_protected_survivors = set()

        eb_with_pc = eb_hosts & planet_candidate_tics
        eb_with_pc_excluded = eb_with_pc - eb_hosts_protected_survivors

        assert eb_with_pc_excluded == eb_with_pc == {"111", "222"}
