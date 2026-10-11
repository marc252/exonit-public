"""Tests for scripts/generate_missing_explanations.py

Tests are offline (sqlite in-memory, no explainer, no network).
They target:
  - build_where_clause(include_stale, strong_only)
  - select_candidates(con, include_stale, strong_only, limit)
  - flush_batch retry-backoff logic (mocked requests + time.sleep)
  - the --endpoints-json / --max-seconds CLI contract (mutual exclusion,
    endpoints-json parsing, dead-endpoint-dropped-at-start, deadline_hit) —
    part of the "LoRA on 5 Mac minis" design, part 1.
"""
from __future__ import annotations

import json
import sqlite3
import time
import types
from unittest.mock import MagicMock, call, patch

import pytest

# ---------------------------------------------------------------------------
# Import helpers under test
# ---------------------------------------------------------------------------
import importlib.util
import sys
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "generate_missing_explanations.py"
_spec = importlib.util.spec_from_file_location("generate_missing_explanations", _SCRIPT)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

build_where_clause = _mod.build_where_clause
select_candidates = _mod.select_candidates
flush_batch = _mod.flush_batch
FLUSH_MAX_RETRIES = _mod.FLUSH_MAX_RETRIES
FLUSH_BACKOFF_BASE = _mod.FLUSH_BACKOFF_BASE
_clean = _mod._clean
_STOP_TOKENS = _mod._STOP_TOKENS
build_arg_parser = _mod.build_arg_parser
load_endpoints_json = _mod.load_endpoints_json
health_check_endpoints = _mod.health_check_endpoints


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_db() -> sqlite3.Connection:
    """Create an in-memory SQLite DB with the minimal schema and test data."""
    con = sqlite3.connect(":memory:")
    con.row_factory = sqlite3.Row
    con.executescript("""
        CREATE TABLE stars (
            tic_id INTEGER PRIMARY KEY,
            tmag REAL,
            n_sectors INTEGER,
            time_span_days REAL,
            n_datapoints INTEGER
        );
        CREATE TABLE candidates (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            tic_id INTEGER,
            candidate_num INTEGER DEFAULT 1,
            period_days REAL,
            depth_ppm REAL,
            duration_days REAL,
            snr REAL,
            cnn_score REAL,
            triceratops_fpp REAL,
            triceratops_nfpp REAL,
            triceratops_verdict TEXT,
            exominer_score REAL,
            claude_verdict TEXT,
            claude_analysis TEXT,
            sector INTEGER NOT NULL DEFAULT 1
        );
    """)

    # Insert stars
    for tic in range(1, 8):
        con.execute(
            "INSERT INTO stars VALUES (?,?,?,?,?)",
            (tic, 10.0 + tic, 1, 27.0, 1000),
        )

    # Candidate layout (id auto):
    # 1: planet_candidate, no analysis, cnn_score OK     → strong+normal match
    # 2: planet_candidate, analysis NULL, cnn_score NULL  → never selected (no cnn)
    # 3: planet_candidate, analysis long+lora-v2 tag      → never selected (already done)
    # 4: false_positive,   no analysis, cnn_score OK      → normal match, NOT strong
    # 5: likely_fp,        no analysis, cnn_score OK      → normal match, NOT strong
    # 6: planet_candidate, analysis short (<=80), cnn_score OK → strong+normal match
    # 7: planet_candidate, analysis non-lora legacy,      → only selected with --include-stale
    #    cnn_score OK
    rows = [
        # tic, verdict, cnn_score, analysis
        (1, "planet_candidate", 0.9,  None),
        (2, "planet_candidate", None, None),
        (3, "planet_candidate", 0.8,  "[lora-v2] A long explanation that is definitely more than eighty characters long ok."),
        (4, "false_positive",   0.2,  None),
        (5, "likely_fp",        0.3,  None),
        (6, "planet_candidate", 0.7,  "short"),
        (7, "planet_candidate", 0.6,  "A legacy analysis without the lora tag but longer than eighty characters for sure yes."),
    ]
    # sector: tics 1 and 6 belong to sector 5 (the "current lot"), the rest
    # to sector 14 (historic backlog) — see TestSectorScope.
    for tic, verdict, cnn, analysis in rows:
        con.execute(
            "INSERT INTO candidates "
            "(tic_id, candidate_num, period_days, depth_ppm, duration_days, snr,"
            " cnn_score, triceratops_fpp, triceratops_nfpp, triceratops_verdict,"
            " exominer_score, claude_verdict, claude_analysis, sector) "
            "VALUES (?,1,1.0,1000.0,0.1,10.0,?,0.01,0.99,'fp',0.5,?,?,?)",
            (tic, cnn, verdict, analysis, 5 if tic in (1, 6) else 14),
        )
    con.commit()
    return con


# ---------------------------------------------------------------------------
# _clean / _STOP_TOKENS — must match the production (Qwen3) explainer
# ---------------------------------------------------------------------------

class TestCleanStopTokens:
    """2026-07-02: the explainer migrated Phi-3.5 -> Qwen3 on 2026-06-25
    (commit 1a0379b1) but this script's client-side cleanup kept the old
    Phi stop-token set, a silent no-op against real Qwen3 output."""

    def test_stop_tokens_match_qwen3_explainer(self):
        assert _STOP_TOKENS == ("<|im_end|>", "<|endoftext|>")

    def test_stop_tokens_no_longer_phi35(self):
        for phi_tok in ("<|end|>", "<|user|>", "<|assistant|>"):
            assert phi_tok not in _STOP_TOKENS

    def test_clean_truncates_at_im_end(self):
        raw = "This is a real transit signal.<|im_end|>\nsome trailing junk"
        assert _clean(raw) == "This is a real transit signal."

    def test_clean_truncates_at_endoftext(self):
        raw = "Explanation text here.<|endoftext|>garbage"
        assert _clean(raw) == "Explanation text here."

    def test_clean_no_stop_token_returns_stripped_text(self):
        assert _clean("  Clean explanation, no artifacts.  ") == \
            "Clean explanation, no artifacts."

    def test_clean_strips_unk_and_zero_width_space(self):
        raw = "Good text<unk> with artifacts​ here.<|im_end|>"
        out = _clean(raw)
        assert "<unk>" not in out
        assert "​" not in out

    def test_clean_empty_string_passthrough(self):
        assert _clean("") == ""


# ---------------------------------------------------------------------------
# build_where_clause — structural tests (no DB)
# ---------------------------------------------------------------------------

class TestBuildWhereClause:

    def test_default_no_stale_no_strong(self):
        w = build_where_clause(include_stale=False, strong_only=False)
        assert "claude_verdict" not in w, (
            "strong_only=False must NOT filter by claude_verdict"
        )
        assert "NOT LIKE '[lora-v2]%'" not in w, (
            "include_stale=False must NOT include stale clause"
        )
        assert "cnn_score IS NOT NULL" in w

    def test_include_stale_adds_not_like(self):
        w = build_where_clause(include_stale=True, strong_only=False)
        assert "NOT LIKE '[lora-v2]%'" in w
        assert "claude_verdict" not in w

    def test_strong_only_adds_verdict_filter(self):
        w = build_where_clause(include_stale=False, strong_only=True)
        assert "claude_verdict = 'planet_candidate'" in w
        assert "NOT LIKE '[lora-v2]%'" not in w

    def test_strong_only_and_include_stale(self):
        w = build_where_clause(include_stale=True, strong_only=True)
        assert "claude_verdict = 'planet_candidate'" in w
        assert "NOT LIKE '[lora-v2]%'" in w

    def test_default_clause_byte_identical_to_original(self):
        """The clause produced with both flags False must match the original hard-coded string."""
        expected = (
            "((c.claude_analysis IS NULL"
            " OR length(coalesce(c.claude_analysis,'')) <= 80)"
            " AND c.cnn_score IS NOT NULL)"
        )
        assert build_where_clause(False, False) == expected


# ---------------------------------------------------------------------------
# select_candidates — integration tests against in-memory DB
# ---------------------------------------------------------------------------

class TestSelectCandidates:

    @pytest.fixture
    def con(self):
        db = _make_db()
        yield db
        db.close()

    def test_flag_off_returns_all_missing(self, con):
        """Without strong_only, all candidates with missing analysis AND cnn_score are selected."""
        rows = select_candidates(con, include_stale=False, strong_only=False)
        ids = [r["tic_id"] for r in rows]
        # Expected: tic 1 (planet_cand, no analysis), 4 (fp, no analysis),
        #           5 (likely_fp, no analysis), 6 (planet_cand, short analysis)
        # Not expected: tic 2 (no cnn_score), 3 (long lora-v2 analysis), 7 (legacy long)
        assert set(ids) == {1, 4, 5, 6}, f"Got {ids}"

    def test_flag_off_does_not_filter_by_verdict(self, con):
        """Flag off: false_positive and likely_fp must be included."""
        rows = select_candidates(con, include_stale=False, strong_only=False)
        verdicts = {r["claude_verdict"] for r in rows}
        assert "false_positive" in verdicts
        assert "likely_fp" in verdicts

    def test_strong_only_returns_only_planet_candidate(self, con):
        """With strong_only=True, only planet_candidate rows with missing analysis are returned."""
        rows = select_candidates(con, include_stale=False, strong_only=True)
        ids = [r["tic_id"] for r in rows]
        # Expected: tic 1 and 6 (planet_cand, missing/short, cnn OK)
        # Not expected: tic 4 (fp), 5 (likely_fp), 2 (no cnn), 3 (done), 7 (legacy long)
        assert set(ids) == {1, 6}, f"Got {ids}"
        for r in rows:
            assert r["claude_verdict"] == "planet_candidate"

    def test_strong_only_excludes_false_positives(self, con):
        rows = select_candidates(con, include_stale=False, strong_only=True)
        verdicts = [r["claude_verdict"] for r in rows]
        assert "false_positive" not in verdicts
        assert "likely_fp" not in verdicts

    def test_strong_only_excludes_no_cnn_score(self, con):
        """tic 2 has planet_candidate verdict but NULL cnn_score — must be excluded."""
        rows = select_candidates(con, include_stale=False, strong_only=True)
        ids = [r["tic_id"] for r in rows]
        assert 2 not in ids

    def test_strong_only_excludes_already_explained(self, con):
        """tic 3 has a complete [lora-v2] explanation — must be excluded even with strong_only."""
        rows = select_candidates(con, include_stale=False, strong_only=True)
        ids = [r["tic_id"] for r in rows]
        assert 3 not in ids

    def test_include_stale_without_strong(self, con):
        """include_stale adds legacy long-text rows regardless of strong_only."""
        rows = select_candidates(con, include_stale=True, strong_only=False)
        ids = [r["tic_id"] for r in rows]
        # tic 7 (legacy long analysis, no lora-v2 tag) should now appear
        assert 7 in ids
        # tic 3 (already has lora-v2 tag, long) must remain excluded
        assert 3 not in ids

    def test_include_stale_with_strong_only(self, con):
        """include_stale + strong_only: adds stale planet_candidate rows only."""
        rows = select_candidates(con, include_stale=True, strong_only=True)
        ids = [r["tic_id"] for r in rows]
        # tic 7 is planet_candidate + legacy long → now selected
        assert 7 in ids
        # tic 3 (lora-v2 done) must remain excluded
        assert 3 not in ids
        # Non-planet verdicts never selected
        for r in rows:
            assert r["claude_verdict"] == "planet_candidate"

    def test_limit_respected(self, con):
        rows = select_candidates(con, include_stale=False, strong_only=False, limit=2)
        assert len(rows) == 2

    def test_no_limit_zero(self, con):
        rows = select_candidates(con, include_stale=False, strong_only=False, limit=0)
        # limit=0 means no limit; expect all 4 qualifying rows
        assert len(rows) == 4

    def test_strong_only_flag_off_is_superset_of_flag_on(self, con):
        """Results with strong_only=True must be a subset of strong_only=False."""
        all_rows = select_candidates(con, include_stale=False, strong_only=False)
        strong_rows = select_candidates(con, include_stale=False, strong_only=True)
        all_ids = {r["tic_id"] for r in all_rows}
        strong_ids = {r["tic_id"] for r in strong_rows}
        assert strong_ids.issubset(all_ids)
        # And strict subset (there are non-planet rows in the normal result)
        assert strong_ids != all_ids


# ---------------------------------------------------------------------------
# --sector / --order (2026-09-20: s5 LoRA step explained the whole DB)
# ---------------------------------------------------------------------------

class TestSectorScope:

    @pytest.fixture
    def con(self):
        db = _make_db()
        yield db
        db.close()

    def test_where_clause_without_sector_is_unchanged(self):
        assert build_where_clause(True, True) == build_where_clause(True, True, None)
        assert "sector" not in build_where_clause(True, True)

    def test_where_clause_with_sector_appends_equality(self):
        w = build_where_clause(True, True, 5)
        assert w.endswith(" AND c.sector = 5")
        assert w.startswith(build_where_clause(True, True))

    def test_sector_is_int_cast(self):
        # A non-numeric value must never reach the SQL string.
        with pytest.raises((ValueError, TypeError)):
            build_where_clause(True, True, "5 OR 1=1")
        assert build_where_clause(True, True, "5").endswith("c.sector = 5")

    def test_select_scoped_to_sector_returns_only_that_sector(self, con):
        rows = select_candidates(con, include_stale=True, strong_only=True, sector=5)
        assert {r["tic_id"] for r in rows} == {1, 6}

    def test_select_other_sector_excludes_current_lot(self, con):
        rows = select_candidates(con, include_stale=True, strong_only=True, sector=14)
        assert {r["tic_id"] for r in rows} == {7}

    def test_select_no_sector_is_whole_db(self, con):
        rows = select_candidates(con, include_stale=True, strong_only=True)
        assert {r["tic_id"] for r in rows} == {1, 6, 7}

    def test_select_empty_sector_returns_nothing(self, con):
        assert select_candidates(con, True, True, sector=99) == []


class TestOrderStrong:

    @pytest.fixture
    def con(self):
        db = _make_db()
        # tic 1: snr 10 / cnn 0.9 -> passes the strong gate; make tic 4 and 6
        # fail it (snr below 10 / cnn below 0.7) and tic 5 pass with a higher id.
        db.execute("UPDATE candidates SET snr = 5.0 WHERE tic_id = 4")
        db.execute("UPDATE candidates SET cnn_score = 0.65 WHERE tic_id = 6")
        db.execute("UPDATE candidates SET cnn_score = 0.75 WHERE tic_id = 5")
        db.commit()
        yield db
        db.close()

    def test_order_choices(self):
        assert set(_mod.ORDER_BY) == {"id", "strong"}

    def test_default_order_is_id_ascending(self, con):
        rows = select_candidates(con, include_stale=False, strong_only=False)
        ids = [r["id"] for r in rows]
        assert ids == sorted(ids)

    def test_order_strong_puts_gate_passers_first_then_id(self, con):
        rows = select_candidates(con, include_stale=False, strong_only=False,
                                 order="strong")
        tics = [r["tic_id"] for r in rows]
        # gate passers (snr>=10 & cnn>=0.7): tic 1, tic 5 (by id); then 4, 6.
        assert tics == [1, 5, 4, 6]

    def test_order_strong_with_limit_takes_gate_passers(self, con):
        rows = select_candidates(con, False, False, limit=2, order="strong")
        assert [r["tic_id"] for r in rows] == [1, 5]

    def test_unknown_order_rejected(self, con):
        with pytest.raises(KeyError):
            select_candidates(con, False, False, order="random")


# ---------------------------------------------------------------------------
# flush_batch — retry / backoff tests (no network, no sleep delay)
# ---------------------------------------------------------------------------

_TEST_URL = "http://testserver/api/candidate/update_analysis"
_TEST_HEADERS = {"Authorization": "Bearer test-token"}
_TEST_TIMEOUT = 30.0


def _make_response(status_code: int, text: str = "") -> MagicMock:
    """Build a minimal mock that looks like a requests.Response."""
    r = MagicMock()
    r.status_code = status_code
    r.text = text
    return r


class TestFlushBatchRetry:
    """Test the real module-level flush_batch via the injectable _post param.

    All tests call the actual production function directly — no replica or
    helper closure.  time.sleep is patched at the module level so tests do
    not wait for real backoff delays.
    """

    SAMPLE_BATCH = [{"candidate_id": 1, "analysis": "[lora-v2] test"}]

    @patch("time.sleep")
    def test_500_then_200_returns_true(self, mock_sleep):
        """500, 500, 200 → True; exactly 2 retries (2 sleeps), no flush_err."""
        responses = [
            _make_response(500),
            _make_response(500),
            _make_response(200),
        ]
        post_mock = MagicMock(side_effect=responses)

        result = flush_batch(self.SAMPLE_BATCH, _TEST_URL, _TEST_HEADERS,
                             _TEST_TIMEOUT, _post=post_mock)

        assert result is True, "Expected True after eventual 200"
        assert post_mock.call_count == 3, (
            f"Expected 3 POST calls (2 failures + 1 success), got {post_mock.call_count}"
        )
        assert mock_sleep.call_count == 2, (
            f"Expected 2 sleeps (after each 500), got {mock_sleep.call_count}"
        )

    @patch("time.sleep")
    def test_persistent_500_exhausts_retries_returns_false(self, mock_sleep):
        """All calls return 500 → False after FLUSH_MAX_RETRIES+1 attempts."""
        post_mock = MagicMock(return_value=_make_response(500))

        result = flush_batch(self.SAMPLE_BATCH, _TEST_URL, _TEST_HEADERS,
                             _TEST_TIMEOUT, _post=post_mock)

        assert result is False
        assert post_mock.call_count == FLUSH_MAX_RETRIES + 1, (
            f"Expected {FLUSH_MAX_RETRIES + 1} calls, got {post_mock.call_count}"
        )
        # Sleep called FLUSH_MAX_RETRIES times (not after the last attempt).
        assert mock_sleep.call_count == FLUSH_MAX_RETRIES

    @patch("time.sleep")
    def test_400_returns_false_with_single_attempt(self, mock_sleep):
        """4xx client error → False immediately, no retry, no sleep."""
        post_mock = MagicMock(return_value=_make_response(400))

        result = flush_batch(self.SAMPLE_BATCH, _TEST_URL, _TEST_HEADERS,
                             _TEST_TIMEOUT, _post=post_mock)

        assert result is False
        assert post_mock.call_count == 1, (
            f"Expected exactly 1 call for 4xx (no retry), got {post_mock.call_count}"
        )
        assert mock_sleep.call_count == 0, "Expected no sleep for deterministic 4xx"

    @patch("time.sleep")
    def test_connection_error_then_200_returns_true(self, mock_sleep):
        """ConnectionError followed by 200 → True after 1 retry."""
        import requests as _req

        responses = [
            _req.exceptions.ConnectionError("simulated timeout"),
            _make_response(200),
        ]
        post_mock = MagicMock(side_effect=responses)

        result = flush_batch(self.SAMPLE_BATCH, _TEST_URL, _TEST_HEADERS,
                             _TEST_TIMEOUT, _post=post_mock)

        assert result is True
        assert post_mock.call_count == 2
        assert mock_sleep.call_count == 1, "Expected 1 sleep after the ConnectionError"

    @patch("time.sleep")
    def test_empty_batch_returns_true_without_post(self, mock_sleep):
        """Empty batch short-circuits to True with zero network calls."""
        post_mock = MagicMock()

        result = flush_batch([], _TEST_URL, _TEST_HEADERS, _TEST_TIMEOUT,
                             _post=post_mock)

        assert result is True
        post_mock.assert_not_called()
        mock_sleep.assert_not_called()

    @patch("time.sleep")
    def test_backoff_values_are_exponential(self, mock_sleep):
        """Verify the sleep durations follow the expected exponential sequence."""
        post_mock = MagicMock(return_value=_make_response(500))

        flush_batch(self.SAMPLE_BATCH, _TEST_URL, _TEST_HEADERS, _TEST_TIMEOUT,
                    _post=post_mock)

        # Expected sleeps: base*2^0, base*2^1, base*2^2, base*2^3
        expected = [FLUSH_BACKOFF_BASE * (2 ** i) for i in range(FLUSH_MAX_RETRIES)]
        actual = [c.args[0] for c in mock_sleep.call_args_list]
        assert actual == expected, f"Backoff sequence wrong: {actual} != {expected}"


# ---------------------------------------------------------------------------
# --endpoints-json / --max-seconds CLI contract (LoRA-on-Macs design, part 1)
# ---------------------------------------------------------------------------

class TestArgParserMutualExclusion:

    def test_explainer_url_and_endpoints_json_are_mutually_exclusive(self):
        ap = build_arg_parser()
        with pytest.raises(SystemExit) as exc_info:
            ap.parse_args(["--explainer-url", "http://a/api/explain",
                           "--endpoints-json", "endpoints.json"])
        assert exc_info.value.code == 2

    def test_explainer_url_alone_is_fine(self):
        ap = build_arg_parser()
        args = ap.parse_args(["--explainer-url", "http://a/api/explain"])
        assert args.explainer_url == "http://a/api/explain"
        assert args.endpoints_json is None

    def test_endpoints_json_alone_is_fine(self):
        ap = build_arg_parser()
        args = ap.parse_args(["--endpoints-json", "endpoints.json"])
        assert args.endpoints_json == "endpoints.json"
        assert args.explainer_url is None

    def test_neither_flag_leaves_both_none_at_parse_time(self):
        """The historical default URL is applied in main() after parsing
        (only when neither flag was given), not baked into the parser —
        otherwise the mutually-exclusive group could never tell "user gave
        --endpoints-json" apart from "user didn't say anything"."""
        ap = build_arg_parser()
        args = ap.parse_args([])
        assert args.explainer_url is None
        assert args.endpoints_json is None

    def test_max_seconds_default_none(self):
        ap = build_arg_parser()
        args = ap.parse_args([])
        assert args.max_seconds is None


class TestLoadEndpointsJson:

    def test_valid_file_parses_with_defaults(self, tmp_path):
        p = tmp_path / "endpoints.json"
        p.write_text(json.dumps([
            {"url": "http://m4b:8500/api/explain", "concurrency": 8, "name": "m4b"},
            {"url": "http://m4c:8500/api/explain"},  # concurrency/name defaulted
        ]))
        endpoints = load_endpoints_json(str(p))
        assert endpoints[0] == {"url": "http://m4b:8500/api/explain",
                               "concurrency": 8, "name": "m4b"}
        assert endpoints[1]["concurrency"] == 1
        assert endpoints[1]["name"] == "http://m4c:8500/api/explain"

    def test_missing_url_key_raises(self, tmp_path):
        p = tmp_path / "endpoints.json"
        p.write_text(json.dumps([{"concurrency": 4}]))
        with pytest.raises(ValueError, match="url"):
            load_endpoints_json(str(p))

    def test_non_list_json_raises(self, tmp_path):
        p = tmp_path / "endpoints.json"
        p.write_text(json.dumps({"url": "http://a/api/explain"}))
        with pytest.raises(ValueError):
            load_endpoints_json(str(p))

    def test_empty_list_raises(self, tmp_path):
        p = tmp_path / "endpoints.json"
        p.write_text(json.dumps([]))
        with pytest.raises(ValueError):
            load_endpoints_json(str(p))

    def test_missing_file_raises_oserror(self, tmp_path):
        with pytest.raises(OSError):
            load_endpoints_json(str(tmp_path / "does-not-exist.json"))


class TestHealthCheckEndpoints:
    """Covers the 'dead endpoint dropped at start' contract: main() requires
    >=1 alive endpoint after this filter (rc=2 otherwise)."""

    def test_all_alive_all_returned(self):
        endpoints = [{"url": "http://a/api/explain", "concurrency": 1, "name": "a"},
                    {"url": "http://b/api/explain", "concurrency": 1, "name": "b"}]
        fake_get = MagicMock(return_value=MagicMock(
            json=MagicMock(return_value={"upstream": "ok"})))
        alive = health_check_endpoints(endpoints, _get=fake_get)
        assert alive == endpoints

    def test_dead_endpoint_dropped_survivor_kept(self):
        endpoints = [{"url": "http://dead/api/explain", "concurrency": 1, "name": "dead"},
                    {"url": "http://alive/api/explain", "concurrency": 1, "name": "alive"}]

        def fake_get(url, timeout):
            if "dead" in url:
                raise ConnectionError("no route to host")
            return MagicMock(json=MagicMock(return_value={"upstream": "ok"}))

        alive = health_check_endpoints(endpoints, _get=fake_get)
        assert [e["name"] for e in alive] == ["alive"]

    def test_all_dead_returns_empty_list(self):
        endpoints = [{"url": "http://a/api/explain", "concurrency": 1, "name": "a"}]

        def fake_get(url, timeout):
            raise ConnectionError("down")

        assert health_check_endpoints(endpoints, _get=fake_get) == []

    def test_healthz_derived_same_way_as_explainer_url(self):
        """/healthz must be derived from the /api/explain URL exactly like
        the single-endpoint --explainer-url path always has (consistency
        required by the design)."""
        seen_urls = []

        def fake_get(url, timeout):
            seen_urls.append(url)
            return MagicMock(json=MagicMock(return_value={"upstream": "ok"}))

        health_check_endpoints(
            [{"url": "http://host:8500/api/explain", "concurrency": 1, "name": "h"}],
            _get=fake_get,
        )
        assert seen_urls == ["http://host:8500/healthz"]


# ---------------------------------------------------------------------------
# Full main() CLI integration: --endpoints-json / --max-seconds contract.
#
# call_explainer / flush_batch / health_check_endpoints are monkeypatched on
# the module (bare-name global lookups inside main(), resolved dynamically
# at call time — unlike flush_batch's own injectable ``_post`` default,
# which is bound at function-definition time and therefore NOT reachable by
# patching the ``requests`` module after the fact). No real network I/O.
# ---------------------------------------------------------------------------

def _make_file_db(path, n=10):
    """A tiny on-disk sqlite DB: n candidates, all with claude_verdict
    already set (so compute_verdict/predict_fn — which needs a trained
    model artifact — is never invoked; out of scope for this test)."""
    con = sqlite3.connect(str(path))
    con.executescript("""
        CREATE TABLE stars (
            tic_id INTEGER PRIMARY KEY, tmag REAL, n_sectors INTEGER,
            time_span_days REAL, n_datapoints INTEGER
        );
        CREATE TABLE candidates (
            id INTEGER PRIMARY KEY AUTOINCREMENT, tic_id INTEGER,
            candidate_num INTEGER DEFAULT 1, period_days REAL, depth_ppm REAL,
            duration_days REAL, snr REAL, cnn_score REAL,
            triceratops_fpp REAL, triceratops_nfpp REAL,
            triceratops_verdict TEXT, exominer_score REAL,
            claude_verdict TEXT, claude_analysis TEXT,
            sector INTEGER NOT NULL DEFAULT 1
        );
    """)
    for tic in range(1, n + 1):
        con.execute("INSERT INTO stars VALUES (?,?,?,?,?)", (tic, 10.0, 1, 27.0, 1000))
        con.execute(
            "INSERT INTO candidates (tic_id, candidate_num, period_days, depth_ppm,"
            " duration_days, snr, cnn_score, triceratops_fpp, triceratops_nfpp,"
            " triceratops_verdict, exominer_score, claude_verdict, claude_analysis,"
            " sector) VALUES (?,1,1.0,1000.0,0.1,10.0,0.9,0.01,0.99,'fp',0.5,"
            "'planet_candidate',NULL,1)",
            (tic,),
        )
    con.commit()
    con.close()


def _run_main(monkeypatch, argv):
    """Run ``_mod.main()`` with *argv*, pinning ``--server-url`` to a
    non-production placeholder unless the caller already passed one
    explicitly (2026-09-26 review follow-up: ``main()``'s new pytest guard
    refuses under PYTEST_CURRENT_TEST when ``--server-url`` resolves to the
    real production ExoNit server — the script's own ``--server-url``
    default is ``http://127.0.0.1:8400`` (the local server), so every test here would
    otherwise be refused before ever reaching the ``flush_batch``/
    ``health_check_endpoints`` mocks it actually wants to exercise)."""
    if "--server-url" not in argv:
        argv = [*argv, "--server-url", "http://127.0.0.1:1/unused-test-server-url"]
    monkeypatch.setattr("sys.argv", ["generate_missing_explanations.py"] + argv)
    return _mod.main()


class TestEndpointsJsonEndToEnd:

    def test_dead_endpoint_dropped_survivor_still_completes(self, tmp_path, monkeypatch, capsys):
        db_path = tmp_path / "test.db"
        _make_file_db(db_path, n=6)
        endpoints_path = tmp_path / "endpoints.json"
        endpoints_path.write_text(json.dumps([
            {"url": "http://dead-mac/api/explain", "concurrency": 2, "name": "dead-mac"},
            {"url": "http://good-mac/api/explain", "concurrency": 2, "name": "good-mac"},
        ]))

        monkeypatch.setattr(_mod, "health_check_endpoints",
                            lambda endpoints, **kw: [e for e in endpoints
                                                     if e["name"] != "dead-mac"])
        monkeypatch.setattr(_mod, "call_explainer",
                            lambda url, payload, timeout: f"exp-{payload['tic_id']}")
        flushed = []
        monkeypatch.setattr(_mod, "flush_batch",
                            lambda batch, url, headers, timeout: (
                                flushed.extend(batch) or True))

        rc = _run_main(monkeypatch, [
            "--db", str(db_path), "--endpoints-json", str(endpoints_path),
        ])
        out = capsys.readouterr().out

        assert rc == 0
        assert "good-mac" in out
        assert "dead-mac" not in [ln for ln in out.splitlines() if "[plan] using" in ln][0]
        assert len(flushed) == 6
        assert "deadline_hit=0" in out

    def test_all_endpoints_dead_exits_rc2(self, tmp_path, monkeypatch, capsys):
        db_path = tmp_path / "test.db"
        _make_file_db(db_path, n=3)
        endpoints_path = tmp_path / "endpoints.json"
        endpoints_path.write_text(json.dumps([
            {"url": "http://dead/api/explain", "concurrency": 1, "name": "dead"},
        ]))
        monkeypatch.setattr(_mod, "health_check_endpoints", lambda endpoints, **kw: [])

        rc = _run_main(monkeypatch, [
            "--db", str(db_path), "--endpoints-json", str(endpoints_path),
        ])
        err = capsys.readouterr().err
        assert rc == 2
        assert "no endpoint responded" in err.lower()

    def test_malformed_endpoints_json_exits_rc2(self, tmp_path, monkeypatch, capsys):
        db_path = tmp_path / "test.db"
        _make_file_db(db_path, n=1)
        endpoints_path = tmp_path / "endpoints.json"
        endpoints_path.write_text("not json")

        rc = _run_main(monkeypatch, [
            "--db", str(db_path), "--endpoints-json", str(endpoints_path),
        ])
        assert rc == 2


class TestMaxSecondsDeadlineEndToEnd:

    def test_deadline_hit_reported_and_rc_zero_when_no_real_failures(
            self, tmp_path, monkeypatch, capsys):
        db_path = tmp_path / "test.db"
        _make_file_db(db_path, n=25)

        def slow_explainer(url, payload, timeout):
            time.sleep(0.03)
            return f"exp-{payload['tic_id']}"

        monkeypatch.setattr(_mod, "health_check_endpoints", lambda endpoints, **kw: endpoints)
        monkeypatch.setattr(_mod, "call_explainer", slow_explainer)
        monkeypatch.setattr(_mod, "flush_batch",
                            lambda batch, url, headers, timeout: True)

        rc = _run_main(monkeypatch, [
            "--db", str(db_path), "--explainer-url", "http://one-mac/api/explain",
            "--workers", "1", "--max-seconds", "0.05",
        ])
        out = capsys.readouterr().out

        assert rc == 0, out
        assert "deadline_hit=1" in out
        done_line = [ln for ln in out.splitlines() if ln.startswith("[done]")][0]
        assert "fail=0" in done_line

    def test_no_max_seconds_reports_deadline_hit_zero(self, tmp_path, monkeypatch, capsys):
        db_path = tmp_path / "test.db"
        _make_file_db(db_path, n=3)
        monkeypatch.setattr(_mod, "health_check_endpoints", lambda endpoints, **kw: endpoints)
        monkeypatch.setattr(_mod, "call_explainer",
                            lambda url, payload, timeout: f"exp-{payload['tic_id']}")
        monkeypatch.setattr(_mod, "flush_batch",
                            lambda batch, url, headers, timeout: True)

        rc = _run_main(monkeypatch, [
            "--db", str(db_path), "--endpoints-json",
            str(_write_single_endpoint_json(tmp_path)),
        ])
        out = capsys.readouterr().out
        assert rc == 0
        assert "deadline_hit=0" in out


def _write_single_endpoint_json(tmp_path):
    p = tmp_path / "single_endpoint.json"
    p.write_text(json.dumps([{"url": "http://one/api/explain", "concurrency": 2,
                             "name": "one"}]))
    return p


class TestOldPathUnchangedWhenNoNewFlags:
    """--explainer-url alone (no --endpoints-json, no --max-seconds) must
    take the historical ThreadPoolExecutor branch: no [pool] line, no
    deadline_hit field in [done] — output stays byte-identical to before
    this feature existed."""

    def test_no_pool_line_and_no_deadline_hit_field(self, tmp_path, monkeypatch, capsys):
        db_path = tmp_path / "test.db"
        _make_file_db(db_path, n=3)
        # The historical single-endpoint path does its own inline health
        # check via a bare `requests.get(...)` call (resolved dynamically at
        # call time, unlike flush_batch's/health_check_endpoints's own
        # injectable defaults which are bound at definition time) — so
        # patching the shared `requests` module's attribute here does reach
        # it.
        monkeypatch.setattr(_mod.requests, "get",
                            lambda url, timeout: types.SimpleNamespace(
                                json=lambda: {"upstream": "ok"}))
        monkeypatch.setattr(_mod, "call_explainer",
                            lambda url, payload, timeout: f"exp-{payload['tic_id']}")
        monkeypatch.setattr(_mod, "flush_batch",
                            lambda batch, url, headers, timeout: True)

        rc = _run_main(monkeypatch, [
            "--db", str(db_path), "--explainer-url", "http://legacy/api/explain",
            "--workers", "2",
        ])
        out = capsys.readouterr().out

        assert rc == 0
        assert "[pool]" not in out
        done_line = [ln for ln in out.splitlines() if ln.startswith("[done]")][0]
        assert "deadline_hit" not in done_line
        assert "skipped" not in done_line


# ---------------------------------------------------------------------------
# Pytest prod guard (2026-09-26 incident; tightened 2026-09-26 review
# follow-up to also check --server-url, not just --db)
# ---------------------------------------------------------------------------

class TestIsProdDbDsn:

    def test_prod_host_non_test_dbname_is_prod(self):
        assert _mod._is_prod_db_dsn("postgresql://exonit@<postgres-host>:5432/exonit") is True

    def test_prod_host_test_suffix_dbname_is_not_prod(self):
        assert _mod._is_prod_db_dsn("postgresql://exonit@<postgres-host>:5432/exonit_test") is False

    def test_any_host_non_test_dbname_is_prod(self):
        # A local PostgreSQL is the real database on a single-host install, and the LAN host may be any address.
        assert _mod._is_prod_db_dsn("postgresql://exonit@<lan-ip>:5432/exonit") is True
        assert _mod._is_prod_db_dsn("postgresql://exonit@localhost:5432/exonit") is True
        assert _mod._is_prod_db_dsn("postgres://exonit@db.example/exonit?sslmode=require") is True

    def test_test_suffix_dbname_on_any_host_is_not_prod(self):
        assert _mod._is_prod_db_dsn("postgresql://exonit@localhost:5432/exonit_test") is False

    def test_plain_sqlite_path_is_not_prod(self):
        assert _mod._is_prod_db_dsn("data/exonit.db") is False


class TestIsProdServerUrl:

    def test_prod_lan_host_is_prod(self):
        assert _mod._is_prod_server_url("http://<server-ip>:8400") is True

    def test_any_192_168_30_x_host_is_prod(self):
        """Not just .76 (the default) — the whole ExoNit LAN is real
        infrastructure, e.g. the Mac fleet or <host> itself."""
        assert _mod._is_prod_server_url("http://<postgres-host>:8500") is True

    def test_loopback_on_the_production_port_is_prod(self):
        # The default --server-url: this script runs on the server host, so 8400 on loopback IS the server.
        assert _mod._is_prod_server_url("http://localhost:8400") is True
        assert _mod._is_prod_server_url("http://127.0.0.1:8400") is True

    def test_localhost_on_another_port_is_not_prod(self):
        assert _mod._is_prod_server_url("http://localhost:8123") is False

    def test_127_0_0_1_is_not_prod(self):
        assert _mod._is_prod_server_url("http://127.0.0.1:54321/x") is False

    def test_any_private_lan_address_is_prod(self):
        assert _mod._is_prod_server_url("http://10.0.0.5:8400") is True
        assert _mod._is_prod_server_url("http://172.16.4.2:9999") is True

    def test_public_address_and_hostname_are_not_judged(self):
        assert _mod._is_prod_server_url("http://8.8.8.8:8400") is False
        assert _mod._is_prod_server_url("http://some-host:8400") is False

    def test_malformed_url_is_not_prod(self):
        assert _mod._is_prod_server_url("not a url::::") is False


class TestPytestProdGuardEndToEnd:
    """Direct sys.argv + _mod.main() invocations — NOT via `_run_main`
    (which pins a safe --server-url for every other test in this file) —
    so these exercise the guard's actual refusal path end to end."""

    def _run(self, monkeypatch, argv):
        monkeypatch.setattr("sys.argv", ["generate_missing_explanations.py"] + argv)
        return _mod.main()

    def test_prod_db_refused_under_pytest(self, monkeypatch, capsys):
        rc = self._run(monkeypatch, [
            "--db", "postgresql://exonit@<postgres-host>:5432/exonit",
            "--server-url", "http://127.0.0.1:1/unused",
        ])
        err = capsys.readouterr().err
        assert rc == 2
        assert "REFUSED" in err
        assert "<postgres-host>" in err
        assert "PYTEST_CURRENT_TEST" in err or "pytest" in err.lower()

    def test_prod_server_url_refused_under_pytest_even_with_safe_db(self, monkeypatch, capsys, tmp_path):
        """The exact gap the review flagged: a harmless --db must NOT be
        enough to bypass the guard if --server-url is real infrastructure."""
        db_path = tmp_path / "harmless.db"
        rc = self._run(monkeypatch, [
            "--db", str(db_path),
            # --server-url omitted -> default http://127.0.0.1:8400 = the local production server.
        ])
        err = capsys.readouterr().err
        assert rc == 2
        assert "REFUSED" in err
        assert "127.0.0.1:8400" in err

    def test_both_prod_db_and_prod_server_reports_both_reasons(self, monkeypatch, capsys):
        rc = self._run(monkeypatch, [
            "--db", "postgresql://exonit@<postgres-host>:5432/exonit",
        ])
        err = capsys.readouterr().err
        assert rc == 2
        assert "<postgres-host>" in err
        assert "127.0.0.1:8400" in err

    def test_non_prod_db_and_server_url_not_refused(self, monkeypatch, capsys, tmp_path):
        """A fully local --db + --server-url must sail through the guard
        (though it will fail later trying to reach the fake --explainer-url's
        /healthz — that's the historical health-check failure path, not the
        guard; rc=1 there, not our rc=2). --explainer-url is ALSO pinned to
        localhost here: its own historical default is
        http://<postgres-host>:8500/api/explain (real <host>) and is out
        of scope for THIS guard, but a real GET to it must never happen as a
        side effect of testing this one."""
        db_path = tmp_path / "harmless.db"
        rc = self._run(monkeypatch, [
            "--db", str(db_path), "--server-url", "http://127.0.0.1:1/unused",
            "--explainer-url", "http://127.0.0.1:1/unused-explainer",
        ])
        err = capsys.readouterr().err
        assert "REFUSED" not in err
        assert rc != 2 or "REFUSED" not in err

    def test_allow_var_bypasses_prod_db_guard(self, monkeypatch, capsys):
        monkeypatch.setenv("EXONIT_GEN_EXPLANATIONS_ALLOW_UNDER_PYTEST", "1")
        rc = self._run(monkeypatch, [
            "--db", "postgresql://exonit@<postgres-host>:5432/exonit",
            "--server-url", "http://127.0.0.1:1/unused",
            "--explainer-url", "http://127.0.0.1:1/unused-explainer",
        ])
        err = capsys.readouterr().err
        assert "REFUSED" not in err
        assert rc != 2 or "REFUSED" not in err

    def test_allow_var_bypasses_prod_server_url_guard(self, monkeypatch, capsys, tmp_path):
        monkeypatch.setenv("EXONIT_GEN_EXPLANATIONS_ALLOW_UNDER_PYTEST", "1")
        db_path = tmp_path / "harmless.db"
        rc = self._run(monkeypatch, [
            "--db", str(db_path),
            "--explainer-url", "http://127.0.0.1:1/unused-explainer",
        ])
        err = capsys.readouterr().err
        assert "REFUSED" not in err
