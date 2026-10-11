"""Parity and determinism tests for scripts/recompute_verdict.py.

Integration-level: requires PostgreSQL and the verdict_classifier_v2_1_*.pkl model.
Skip by default (opt-in only — see below).

Hard gate: tool verdict == direct predict_verdict for >= 20 sampled candidates.

**Opt-in only, no default DSN**: set ``EXONIT_PARITY_TEST_DSN`` to a
throwaway/non-production database (e.g. a local ``exonit_test``
schema/database) to actually run this module's PG-backed tests. A bare
``pytest`` invocation (no env var set) always skips, regardless of what
happens to be reachable on the network — reachability alone is NOT a safe
gate (2026-07-02 review finding: this file previously defaulted to
``postgresql://exonit@<postgres-host>:5432/exonit`` — production PG — with a
reachability-only skip-guard, so a bare test run from a machine with network
access to <postgres-host> silently reconnected to prod and executed every test
in this module against it). **NEVER point ``EXONIT_PARITY_TEST_DSN`` at
production PG (<postgres-host>).**

D8 (docs/reviews/2026-07-02-opus48-cleanup-review.md, 2026-07-02 follow-up):
originally this module only compared the script's own two internal call
paths against EACH OTHER (``rv._compute_verdict`` vs a second direct
``predict_verdict`` call) — proving internal self-consistency, but nothing
about whether the script's hand-mirrored dict-building actually matched
server.py's. ``scripts/recompute_verdict.py``'s ``_build_candidate_dict`` /
``_build_star_info`` are now thin wrappers around the SAME shared module
server.py's ``_run_verdict_background`` imports (``exonit.verdict_io`` —
see that module's own docstring). ``test_build_candidate_dict_matches_verdict_io``
and ``test_build_star_info_matches_verdict_io`` below assert the script's
wrapper output against ``exonit.verdict_io`` directly, so "parity" is now a
STRUCTURAL guarantee (the wrapper literally calls the shared function) rather
than something a separate hand-maintained comparison test has to keep
verifying by chance. These two tests exist as a regression tripwire: if a
future change re-introduces a local hand-rolled copy in either
recompute_verdict.py or server.py instead of importing verdict_io, this is
the test that will catch the drift.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
# scripts/ is on sys.path via conftest; add explicitly here too for safety.
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

# Set env before any import of verdict_classifier.
os.environ.setdefault("EXONIT_VERDICT_VERSION", "v2.1")
os.environ.setdefault("EXONIT_USE_CLAUDE_FALLBACK", "0")

# No default: DB_URL is None unless the operator explicitly opts in. This is
# the actual fix for the blocking review finding — do not reintroduce a
# fallback DSN here, prod or otherwise.
DB_URL = os.environ.get("EXONIT_PARITY_TEST_DSN")

# Use sector 1: has classified candidates (claude_verdict IS NOT NULL).
PARITY_SECTOR = 1
# Number of candidates to sample for parity; brief requires >= 20.
N_PARITY = 30


# ---------------------------------------------------------------------------
# Availability guards
# ---------------------------------------------------------------------------

def _pg_available() -> bool:
    """True only when EXONIT_PARITY_TEST_DSN is explicitly set AND reachable.

    Checking DB_URL truthiness FIRST (before attempting any connection) is
    the actual opt-in gate: with no env var set, DB_URL is None and this
    returns False immediately — no socket is ever opened, so network
    reachability to any host (prod PG included) cannot silently un-skip
    this module.
    """
    if not DB_URL:
        return False
    try:
        import psycopg
        with psycopg.connect(DB_URL, connect_timeout=5) as conn:
            conn.cursor().execute("SELECT 1")
        return True
    except Exception:
        return False


def _model_available() -> bool:
    models_dir = Path(os.environ.get(
        "EXONIT_MODELS_DIR", str(ROOT / "models")))
    return bool(list(models_dir.glob("verdict_classifier_v2_1_*.pkl")))


@pytest.fixture(scope="module")
def infra_available():
    """Skip entire module when EXONIT_PARITY_TEST_DSN is unset/unreachable,
    or when the model file is not present."""
    if not _pg_available():
        pytest.skip(
            "EXONIT_PARITY_TEST_DSN not set (or unreachable) — this module's "
            "tests are opt-in only, no default DSN. Set EXONIT_PARITY_TEST_DSN "
            "to a throwaway/non-production database to run them; NEVER point "
            "it at production PG (<postgres-host>)."
        )
    if not _model_available():
        pytest.skip("verdict_classifier_v2_1_*.pkl not found in models/")


@pytest.fixture(scope="module")
def model_only_available():
    """Skip only when the v2.1 model file is missing — deliberately does NOT
    require EXONIT_PARITY_TEST_DSN / PG connectivity.

    Review follow-up (2026-07-11, progress/review-orgate.md item 2): the
    module-level ``infra_available`` fixture bundles a PG-reachability check
    that makes EVERY test in this file (including the pure-Python gate-logic
    test below, which never touches a real DB — see
    ``test_sub_threshold_gate_skips_candidates``/``test_sub_threshold_gate_
    or_gate_em_ffi_rescue``, both mock ``_fetch_candidates``/``_fetch_stars``)
    all-skip whenever ``EXONIT_PARITY_TEST_DSN`` is unset, which is the
    common/default case. That left the OR-gate predicate in ``run_main``
    (mirroring ``exonit.work_producers.verdict_eligible``) with zero test
    coverage in a bare ``pytest`` run — exactly the gap that let the
    ``cnn_score>=0.7``-only gate drift silently out of sync with production
    in the first place. This fixture only checks for the model file (already
    committed under ``models/`` for other non-PG tests in this repo), so
    tests gated on it run in the default/no-DSN environment too.
    """
    if not _model_available():
        pytest.skip("verdict_classifier_v2_1_*.pkl not found in models/")


# ---------------------------------------------------------------------------
# Shared data fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def parity_data(infra_available):
    """Return sampled candidate rows + star map for PARITY_SECTOR.

    Shape: {"cand_rows": [...], "star_map": {...}}
    """
    import psycopg
    import recompute_verdict as rv

    cols_sql = ", ".join(f"c.{c}" for c in rv.CAND_COLS)
    with psycopg.connect(DB_URL) as conn:
        with conn.cursor() as cur:
            cur.execute(f"""
                SELECT {cols_sql}
                FROM candidates c
                WHERE c.sector = %s
                  AND c.cnn_score IS NOT NULL
                ORDER BY c.cnn_score DESC
                LIMIT %s
            """, (PARITY_SECTOR, N_PARITY))
            rows = cur.fetchall()

    if not rows:
        pytest.skip(f"No candidates found for sector {PARITY_SECTOR}")

    cand_rows = [dict(zip(rv.CAND_COLS, r)) for r in rows]
    tic_ids = list({str(r["tic_id"]) for r in cand_rows})
    star_map = rv._fetch_stars(DB_URL, tic_ids)

    return {"cand_rows": cand_rows, "star_map": star_map}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_minimum_candidates_sampled(parity_data):
    """At least 20 candidates must be fetched from PG (connectivity smoke-check)."""
    n = len(parity_data["cand_rows"])
    assert n >= 20, (
        f"Only {n} candidates returned — check sector {PARITY_SECTOR} data or "
        f"PG connectivity."
    )


def test_parity_tool_vs_direct(parity_data, infra_available):
    """HARD GATE: tool path verdict == direct predict_verdict for every candidate.

    The tool path uses _compute_verdict (our helpers + env routing).
    The direct path calls predict_verdict with copied dicts independently.
    Mismatch means _build_candidate_dict or _build_star_info are wrong.
    """
    import recompute_verdict as rv
    from exonit.verdict_classifier import predict_verdict

    cand_rows = parity_data["cand_rows"]
    star_map = parity_data["star_map"]

    failures = []
    for row in cand_rows:
        tic_id = str(row["tic_id"])
        cnum = int(row["candidate_num"])

        candidate = rv._build_candidate_dict(row)
        star_info = rv._build_star_info(star_map.get(tic_id), tic_id)

        # Tool path
        result_tool = rv._compute_verdict(candidate, star_info)

        # Direct independent call (same dicts, copied to avoid any shared state)
        result_direct = predict_verdict(
            dict(candidate), {"passed": True}, star_info=dict(star_info))

        if result_tool["verdict"] != result_direct["verdict"]:
            failures.append({
                "tic_id": tic_id,
                "candidate_num": cnum,
                "tool": result_tool["verdict"],
                "direct": result_direct["verdict"],
            })

    n_checked = len(cand_rows)
    if failures:
        for f in failures[:5]:
            print(f"FAIL tic={f['tic_id']} cnum={f['candidate_num']} "
                  f"tool={f['tool']} direct={f['direct']}")

    assert len(failures) == 0, (
        f"PARITY FAILED: {len(failures)}/{n_checked} candidates have different verdicts "
        f"between tool path and direct predict_verdict. "
        f"dict construction or env routing is wrong."
    )


def test_determinism_same_candidate_twice(parity_data, infra_available):
    """Calling _compute_verdict twice on the same candidate returns the same verdict.

    Verifies the classifier is stateless (deterministic predictions).
    Tests every sampled candidate, not just one.
    """
    import recompute_verdict as rv

    cand_rows = parity_data["cand_rows"]
    star_map = parity_data["star_map"]

    non_deterministic = []
    for row in cand_rows:
        tic_id = str(row["tic_id"])

        candidate = rv._build_candidate_dict(row)
        star_info = rv._build_star_info(star_map.get(tic_id), tic_id)

        r1 = rv._compute_verdict(candidate, star_info)
        r2 = rv._compute_verdict(dict(candidate), dict(star_info))

        if r1["verdict"] != r2["verdict"]:
            non_deterministic.append({
                "tic_id": tic_id,
                "candidate_num": row["candidate_num"],
                "call1": r1["verdict"],
                "call2": r2["verdict"],
            })

    assert len(non_deterministic) == 0, (
        f"Non-determinism detected in {len(non_deterministic)} candidates: "
        f"{non_deterministic[:3]}"
    )


def test_summary_serializable_to_json(parity_data, infra_available, tmp_path):
    """Summary dict must serialise to valid JSON (no numpy scalars, etc.)."""
    import recompute_verdict as rv

    cand_rows = parity_data["cand_rows"]
    star_map = parity_data["star_map"]

    transitions = []
    for row in cand_rows:
        tic_id = str(row["tic_id"])
        candidate = rv._build_candidate_dict(row)
        star_info = rv._build_star_info(star_map.get(tic_id), tic_id)
        result = rv._compute_verdict(candidate, star_info)
        old = row.get("claude_verdict")
        transitions.append((old, result["verdict"]))

    summary_path = tmp_path / "recompute_verdict_s1_summary.json"

    class _FakeArgs:
        sector = PARITY_SECTOR

    rv._write_summary(summary_path, PARITY_SECTOR, transitions,
                      n_no_cand=0, n_no_star=0, dry_run=True)

    assert summary_path.exists(), "Summary file was not created"
    loaded = json.loads(summary_path.read_text())

    assert "sector" in loaded
    assert "n_processed" in loaded
    assert "transition_matrix" in loaded
    assert "n_sub_threshold" in loaded, "n_sub_threshold must be present in summary"
    assert isinstance(loaded["n_new_planet_candidate"], int)
    assert isinstance(loaded["n_lost_planet_candidate"], int)
    assert isinstance(loaded["n_unchanged"], int)
    assert isinstance(loaded["n_sub_threshold"], int)
    assert loaded["n_processed"] == len(transitions)


def test_build_candidate_dict_keys(parity_data, infra_available):
    """_build_candidate_dict must include all keys the classifier reads."""
    import recompute_verdict as rv

    row = parity_data["cand_rows"][0]
    d = rv._build_candidate_dict(row)

    required_keys = {
        "tic_id", "candidate_num", "period_days", "depth_ppm", "duration_days",
        "snr", "cnn_score", "triceratops_fpp", "triceratops_nfpp",
        "triceratops_prob_planet", "exominer_score", "exominer_ffi_score",
        "exominer_tce_period", "exominer_sector_run",
    }
    missing = required_keys - set(d.keys())
    assert not missing, f"Missing keys in candidate dict: {missing}"


def test_build_star_info_keys(parity_data, infra_available):
    """_build_star_info must include all keys the server query returns."""
    import recompute_verdict as rv

    row = parity_data["cand_rows"][0]
    tic_id = str(row["tic_id"])
    star_row = parity_data["star_map"].get(tic_id)

    si = rv._build_star_info(star_row, tic_id)
    assert "tic_id" in si, "star_info must contain tic_id for Berger lookup"
    if star_row is not None:
        for k in ("tmag", "ra", "dec", "n_sectors", "time_span_days", "n_datapoints"):
            assert k in si, f"star_info missing key: {k}"


def test_build_candidate_dict_matches_verdict_io(parity_data, infra_available):
    """rv._build_candidate_dict must be structurally identical to
    exonit.verdict_io.build_candidate_dict — not just internally consistent.

    D8: recompute_verdict.py's own dict-building is now a thin wrapper around
    exonit.verdict_io (the same module server.py's _run_verdict_background
    imports), so this assertion is trivially true by construction — that IS
    the point: with both the script and the server importing the same shared
    module, "parity" is a structural guarantee rather than something a
    separate hand-maintained comparison test has to keep verifying by
    chance. This test is a regression tripwire: if a future change
    re-introduces a local hand-rolled copy instead of importing verdict_io,
    this is the test that will catch the drift.
    """
    import recompute_verdict as rv
    from exonit.verdict_io import build_candidate_dict

    row = parity_data["cand_rows"][0]
    assert rv._build_candidate_dict(row) == build_candidate_dict(row)


def test_build_star_info_matches_verdict_io(parity_data, infra_available):
    """rv._build_star_info must be structurally identical to
    exonit.verdict_io.build_star_info.

    Mind the swapped argument order: the script's wrapper keeps its
    pre-existing (star_row, tic_id) signature for backward compatibility;
    verdict_io's own signature is (tic_id, star_row). See D8 note above.
    """
    import recompute_verdict as rv
    from exonit.verdict_io import build_star_info

    row = parity_data["cand_rows"][0]
    tic_id = str(row["tic_id"])
    star_row = parity_data["star_map"].get(tic_id)

    assert rv._build_star_info(star_row, tic_id) == build_star_info(tic_id, star_row)


def test_parity_env_is_v2_1(infra_available):
    """EXONIT_VERDICT_VERSION must be 'v2.1' so predict_verdict routes correctly."""
    ver = os.environ.get("EXONIT_VERDICT_VERSION")
    assert ver == "v2.1", (
        f"EXONIT_VERDICT_VERSION is '{ver}', expected 'v2.1'. "
        f"Script must set os.environ.setdefault before importing the classifier."
    )
    fallback = os.environ.get("EXONIT_USE_CLAUDE_FALLBACK", "0")
    assert fallback == "0", (
        f"EXONIT_USE_CLAUDE_FALLBACK is '{fallback}', expected '0'."
    )


def test_sub_threshold_gate_skips_candidates(tmp_path, infra_available):
    """Strong-tier gate must skip candidates with cnn_score < 0.7 or snr < 10.0.

    Uses synthetic candidate rows to isolate the gate logic from real DB data.
    Verifies:
      - n_sub_threshold == 2 (one gated on cnn, one on snr)
      - n_processed == 1 (only the strong-tier candidate reaches the classifier)
    _compute_verdict is mocked so the test does not depend on model output for
    synthetic rows; the classifier IS loaded (via infra_available) but never called
    for the gated rows.
    """
    from unittest.mock import patch

    import recompute_verdict as rv

    # Three synthetic pairs: two sub-threshold, one strong-tier.
    csv_path = tmp_path / "pairs.csv"
    csv_path.write_text("tic_id,candidate_num\n100001,1\n100002,1\n100003,1\n")

    def _make_cand(tic_id: str, cnn: float, snr: float) -> dict:
        return {
            "tic_id": tic_id, "candidate_num": 1,
            "period_days": 3.0, "depth_ppm": 1000.0, "duration_days": 0.1,
            "snr": snr, "cnn_score": cnn,
            "triceratops_fpp": None, "triceratops_nfpp": None,
            "triceratops_prob_planet": None,
            "exominer_score": None, "exominer_ffi_score": None,
            "exominer_tce_period": None, "exominer_sector_run": None,
            "claude_verdict": None,
        }

    cand_map = {
        ("100001", 1): _make_cand("100001", 0.5, 15.0),   # cnn=0.5 < 0.7  → gated
        ("100002", 1): _make_cand("100002", 0.8, 5.0),    # snr=5.0 < 10.0 → gated
        ("100003", 1): _make_cand("100003", 0.8, 15.0),   # strong-tier     → classified
    }

    captured: dict = {}

    def _fake_write_summary(path, sector, transitions, n_no_cand, n_no_star,
                             dry_run, n_sub_threshold=0):
        captured["n_sub_threshold"] = n_sub_threshold
        captured["n_processed"] = len(transitions)
        # Write minimal JSON so run_main can log the path without crashing.
        path.write_text(json.dumps({"n_sub_threshold": n_sub_threshold}))

    class _Args:
        from_csv = str(csv_path)
        sector = 1
        dry_run = True
        backup = None
        db = "postgresql://fake/fake"

    with patch.object(rv, "_fetch_candidates", return_value=cand_map), \
         patch.object(rv, "_fetch_stars", return_value={}), \
         patch.object(rv, "_compute_verdict",
                      return_value={"verdict": "likely_false_positive"}), \
         patch.object(rv, "_write_summary", side_effect=_fake_write_summary):
        rv.run_main(_Args())

    assert captured.get("n_sub_threshold") == 2, (
        f"Expected 2 sub-threshold candidates gated out (cnn<0.7 or snr<10.0), "
        f"got {captured.get('n_sub_threshold')}. "
        f"The strong-tier gate in run_main is missing or incorrect."
    )
    assert captured.get("n_processed") == 1, (
        f"Expected 1 strong-tier candidate to reach the classifier, "
        f"got {captured.get('n_processed')}."
    )


# ---------------------------------------------------------------------------
# OR-gate (2026-07-11, docs/reviews/injection_e2e_results_2026-07-11.md +
# docs/reviews/ffi_cnn_investigation_2026-07-11.md §5): run_main's gate must
# mirror exonit.work_producers.verdict_eligible exactly — snr>=10 AND
# (cnn_score>=0.7 OR exominer_ffi_score>=0.5) — not cnn_score>=0.7 alone.
#
# Gated on model_only_available (NOT infra_available): no PG connection
# needed, _fetch_candidates/_fetch_stars/_compute_verdict are all mocked, so
# this test runs in the default/no-DSN environment and cannot silently skip
# the way test_sub_threshold_gate_skips_candidates above does when
# EXONIT_PARITY_TEST_DSN is unset (progress/review-orgate.md item 2).
# ---------------------------------------------------------------------------

def test_sub_threshold_gate_or_gate_em_ffi_rescue(tmp_path, model_only_available):
    """cnn<0.7 alone must NOT reach the classifier UNLESS exominer_ffi_score
    >=0.5 rescues it (OR-gate) — and the pre-existing cnn>=0.7-only path
    must keep working (monotonicity)."""
    from unittest.mock import patch

    import recompute_verdict as rv

    csv_path = tmp_path / "pairs.csv"
    csv_path.write_text(
        "tic_id,candidate_num\n"
        "200001,1\n200002,1\n200003,1\n200004,1\n")

    def _make_cand(tic_id: str, cnn, snr, em_ffi) -> dict:
        return {
            "tic_id": tic_id, "candidate_num": 1,
            "period_days": 3.0, "depth_ppm": 1000.0, "duration_days": 0.1,
            "snr": snr, "cnn_score": cnn,
            "triceratops_fpp": None, "triceratops_nfpp": None,
            "triceratops_prob_planet": None,
            "exominer_score": None, "exominer_ffi_score": em_ffi,
            "exominer_tce_period": None, "exominer_sector_run": None,
            "claude_verdict": None,
        }

    cand_map = {
        # cnn<0.7, em_ffi>=0.5 -> RESCUED, must reach the classifier.
        ("200001", 1): _make_cand("200001", 0.4, 12.0, 0.6),
        # cnn<0.7, em_ffi<0.5 -> still gated (OR-gate, neither side fires).
        ("200002", 1): _make_cand("200002", 0.4, 12.0, 0.3),
        # cnn<0.7, em_ffi NULL -> still gated, and must never raise.
        ("200003", 1): _make_cand("200003", 0.4, 12.0, None),
        # cnn>=0.7, em_ffi NULL -> pre-existing path, must still pass
        # (monotonicity: the OR-gate never removes an eligible row).
        ("200004", 1): _make_cand("200004", 0.8, 12.0, None),
    }

    captured: dict = {}

    def _fake_write_summary(path, sector, transitions, n_no_cand, n_no_star,
                             dry_run, n_sub_threshold=0):
        captured["n_sub_threshold"] = n_sub_threshold
        captured["n_processed"] = len(transitions)
        path.write_text(json.dumps({"n_sub_threshold": n_sub_threshold}))

    class _Args:
        from_csv = str(csv_path)
        sector = 1
        dry_run = True
        backup = None
        db = "postgresql://fake/fake"

    with patch.object(rv, "_fetch_candidates", return_value=cand_map), \
         patch.object(rv, "_fetch_stars", return_value={}), \
         patch.object(rv, "_compute_verdict",
                      return_value={"verdict": "likely_false_positive"}), \
         patch.object(rv, "_write_summary", side_effect=_fake_write_summary):
        rv.run_main(_Args())

    assert captured.get("n_sub_threshold") == 2, (
        f"Expected 2 gated (cnn<0.7 AND em_ffi<0.5-or-NULL), got "
        f"{captured.get('n_sub_threshold')}. The OR-gate in run_main is "
        f"missing exominer_ffi_score or incorrect."
    )
    assert captured.get("n_processed") == 2, (
        f"Expected 2 candidates to reach the classifier (1 em_ffi-rescued + "
        f"1 pre-existing cnn>=0.7 path), got {captured.get('n_processed')}."
    )


def test_verdict_eligible_used_directly_by_recompute_verdict(model_only_available):
    """Structural guard: run_main's gate must literally BE
    exonit.work_producers.verdict_eligible (imported, not re-implemented) so
    this script cannot drift from the server-side predicate again — mirrors
    the D8 structural-guarantee pattern used for _build_candidate_dict /
    _build_star_info parity above."""
    import recompute_verdict as rv
    from exonit.work_producers import verdict_eligible

    assert rv.verdict_eligible is verdict_eligible
