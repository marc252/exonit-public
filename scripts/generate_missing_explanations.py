#!/usr/bin/env python3
"""Generate LoRA explanations for candidates that lack one.

Complements regenerate_explanations.py: that script re-explains rows with
existing analyses, while this one targets the 13k+ candidates whose
claude_analysis is empty/short. For rows without claude_verdict, runs the
local v2.1 classifier first to assign a verdict.

Usage:
    python scripts/generate_missing_explanations.py --workers 1 --limit 0
"""
from __future__ import annotations

import argparse
import ipaddress
import json
import os
import sqlite3
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlsplit

import requests


# Retry policy for flush_batch: applies to 5xx and transient network errors.
# Backoff sequence (seconds): 1.5, 3.0, 6.0, 12.0  (total ≤ ~22.5 s).
FLUSH_MAX_RETRIES = 4
FLUSH_BACKOFF_BASE = 1.5

# Must match the production explainer's actual stop tokens. The explainer is
# Qwen3 (scripts/explainer_service_vllm_qwen3.py; migrated from Phi-3.5 on
# 2026-06-25, commit 1a0379b1) — its chat template emits <|im_end|>, not
# Phi's <|end|>/<|user|>/<|assistant|>, so the old set here was a silent
# no-op (never matched anything the server actually returns).
_STOP_TOKENS = ("<|im_end|>", "<|endoftext|>")


def flush_batch(
    batch: list[dict],
    url: str,
    headers: dict,
    timeout: float,
    *,
    _post=requests.post,
) -> bool:
    """POST the accumulated batch to the server.  Returns True on success.

    Retries up to FLUSH_MAX_RETRIES times with exponential backoff on
    transient failures (5xx responses or network errors).  Client errors
    (4xx) are not retried — they signal a deterministic problem that a
    retry cannot fix.  The endpoint is idempotent (UPDATE by candidate_id),
    so retrying the same batch is safe.

    Parameters
    ----------
    batch:   list of update dicts (``{"candidate_id": ..., "analysis": ...}``)
    url:     full POST URL (e.g. ``http://host:8400/api/candidate/update_analysis``)
    headers: auth headers (``{"Authorization": "Bearer <token>"}``)
    timeout: per-request timeout in seconds
    _post:   injectable HTTP POST callable (default: ``requests.post``).
             Override in tests to avoid real network I/O.
    """
    if not batch:
        return True
    last_exc: Exception | None = None
    for attempt in range(FLUSH_MAX_RETRIES + 1):
        try:
            r = _post(
                url,
                json={"updates": batch},
                headers=headers,
                timeout=timeout,
            )
            if r.status_code == 200:
                return True
            if r.status_code < 500:
                # 4xx — deterministic client error, no retry.
                print(f"[flush] server error {r.status_code}: {r.text[:200]}",
                      file=sys.stderr)
                return False
            # 5xx — transient; schedule a retry.
            backoff = FLUSH_BACKOFF_BASE * (2 ** attempt)
            if attempt < FLUSH_MAX_RETRIES:
                print(
                    f"[flush] transient {r.status_code}, "
                    f"retry {attempt + 1}/{FLUSH_MAX_RETRIES} in {backoff:.1f}s",
                    file=sys.stderr,
                )
                time.sleep(backoff)
            else:
                print(f"[flush] server error {r.status_code}: {r.text[:200]}",
                      file=sys.stderr)
        except requests.exceptions.RequestException as exc:
            last_exc = exc
            backoff = FLUSH_BACKOFF_BASE * (2 ** attempt)
            if attempt < FLUSH_MAX_RETRIES:
                print(
                    f"[flush] transient {exc!r}, "
                    f"retry {attempt + 1}/{FLUSH_MAX_RETRIES} in {backoff:.1f}s",
                    file=sys.stderr,
                )
                time.sleep(backoff)
            else:
                print(f"[flush] POST failed: {exc}", file=sys.stderr)
    return False


def _clean(text: str) -> str:
    if not text:
        return text
    cut = len(text)
    for tok in _STOP_TOKENS:
        i = text.find(tok)
        if 0 <= i < cut:
            cut = i
    return text[:cut].replace("<unk>", "").replace("​", "").strip()


def build_payload(row, verdict: str | None) -> dict:
    return {
        "tic_id": str(row["tic_id"]),
        "candidate_num": row["candidate_num"],
        "period_days": row["period_days"],
        "depth_ppm": row["depth_ppm"],
        "duration_days": row["duration_days"],
        "snr": row["snr"],
        "cnn_score": row["cnn_score"],
        "triceratops_fpp": row["triceratops_fpp"],
        "triceratops_nfpp": row["triceratops_nfpp"],
        "triceratops_verdict": row["triceratops_verdict"],
        "exominer_score": row["exominer_score"],
        "final_verdict": verdict or "unknown",
        "tmag": row["tmag"],
        "n_sectors": row["n_sectors"],
        "time_span_days": row["time_span_days"],
    }


def call_explainer(url: str, payload: dict, timeout: float) -> str | None:
    try:
        r = requests.post(url, json=payload, timeout=timeout)
        if r.status_code == 200:
            raw = r.json().get("explanation")
            return _clean(raw) if raw else None
    except Exception:
        return None
    return None


def load_endpoints_json(path: str) -> list[dict]:
    """Parse the ``--endpoints-json`` file: a list of
    ``{"url": ..., "concurrency": N, "name": ...}`` dicts, written by the
    LoRA-on-Macs driver. ``url`` is the endpoint's ``/api/explain`` URL —
    same convention as ``--explainer-url`` (the healthz check below derives
    ``/healthz`` from it the same way the single-endpoint path always has).
    ``concurrency`` defaults to 1, ``name`` defaults to the url.
    """
    with open(path) as f:
        data = json.load(f)
    if not isinstance(data, list) or not data:
        raise ValueError(f"{path}: expected a non-empty JSON list of endpoints")
    endpoints = []
    for i, e in enumerate(data):
        if not isinstance(e, dict) or "url" not in e:
            raise ValueError(f"{path}: endpoint #{i} missing required 'url' field")
        endpoints.append({
            "url": e["url"],
            "concurrency": int(e.get("concurrency", 1)),
            "name": e.get("name") or e["url"],
        })
    return endpoints


def health_check_endpoints(endpoints: list[dict], *, timeout: float = 5.0,
                           _get=requests.get) -> list[dict]:
    """Probe every endpoint's ``/healthz`` and return the ones that respond.

    An endpoint counts as alive if the GET succeeds and returns parseable
    JSON — content is not gated (matches the historical single-endpoint
    health check below, which never inspected the body either, only
    whether the request itself raised).
    """
    alive = []
    for ep in endpoints:
        health_url = ep["url"].replace("/api/explain", "/healthz")
        name = ep.get("name") or ep["url"]
        try:
            h = _get(health_url, timeout=timeout).json()
            print(f"[health] {name}: {h}", flush=True)
            alive.append(ep)
        except Exception as e:
            print(f"[health] {name}: FAIL: {e}", file=sys.stderr, flush=True)
    return alive


def compute_verdict(row, predict_fn):
    """Run v2.1 classifier on the row's features. Returns verdict string or None."""
    candidate = {
        "candidate_num": row["candidate_num"],
        "period_days": row["period_days"],
        "depth_ppm": row["depth_ppm"],
        "duration_days": row["duration_days"],
        "snr": row["snr"],
        "cnn_score": row["cnn_score"],
        "triceratops_fpp": row["triceratops_fpp"],
        "triceratops_nfpp": row["triceratops_nfpp"],
        "exominer_score": row["exominer_score"],
        "tic_id": str(row["tic_id"]),
    }
    star_info = {
        "tic_id": str(row["tic_id"]),
        "tmag": row["tmag"],
        "n_sectors": row["n_sectors"],
        "time_span_days": row["time_span_days"],
        "n_datapoints": row["n_datapoints"],
    }
    try:
        result = predict_fn(candidate, {"passed": True}, star_info=star_info)
        return result.get("verdict") if result else None
    except Exception:
        return None


def build_where_clause(include_stale: bool, strong_only: bool,
                       sector: int | None = None) -> str:
    """Return the WHERE clause string for candidate selection.

    Parameters
    ----------
    include_stale:
        When True, also include rows that have a non-empty claude_analysis
        lacking the [lora-v2] prefix (legacy Claude text).
    strong_only:
        When True, restrict to candidates with claude_verdict = 'planet_candidate'.
        Intended for discovery runs where ~30k non-planet candidates must be skipped.
    sector:
        When given, restrict to ``c.sector = sector``. 2026-09-20: without it a
        discovery run's LoRA step explained EVERY unexplained planet_candidate
        in the DB (222k after the v3_tfopwg verdict recalculation promoted
        221k historic candidates), not the ~400 of its own lot — 52 h of
        vLLM against a 24 h step timeout. The historic backlog is now a
        separate, manual job (see ``--order strong``).
    """
    missing_analysis = (
        "(c.claude_analysis IS NULL"
        " OR length(coalesce(c.claude_analysis,'')) <= 80"
        + (" OR c.claude_analysis NOT LIKE '[lora-v2]%'" if include_stale else "")
        + ")"
    )
    clause = f"({missing_analysis} AND c.cnn_score IS NOT NULL)"
    if strong_only:
        clause += " AND c.claude_verdict = 'planet_candidate'"
    if sector is not None:
        clause += f" AND c.sector = {int(sector)}"
    return clause


#: ``--order`` choices. ``id``: historical default (oldest candidate first).
#: ``strong``: strong-tier gate first (``snr >= 10 AND cnn_score >= 0.7``,
#: the same gate as server.py's strong-tier / recompute_verdict), then id —
#: for manual backfills of the historic backlog, so the candidates that can
#: actually surface in the UI as strong get their explanation before the
#: long tail of v3 promotions that never will.
ORDER_BY = {
    "id": "c.id ASC",
    "strong": ("(CASE WHEN c.snr >= 10 AND c.cnn_score >= 0.7 THEN 0 ELSE 1 END),"
               " c.id ASC"),
}


def select_candidates(
    con: sqlite3.Connection,
    include_stale: bool,
    strong_only: bool,
    limit: int = 0,
    sector: int | None = None,
    order: str = "id",
) -> list:
    """Fetch the candidate rows to explain from *con*.

    Extracted from main() so tests can drive selection logic without a live
    explainer or network access.
    """
    where_clause = build_where_clause(include_stale, strong_only, sector)
    order_by = ORDER_BY[order]
    return con.execute(
        f"""SELECT c.id, c.tic_id, c.candidate_num, c.period_days, c.depth_ppm,
                   c.duration_days, c.snr, c.cnn_score, c.triceratops_fpp,
                   c.triceratops_nfpp, c.triceratops_verdict, c.exominer_score,
                   c.claude_verdict,
                   s.tmag, s.n_sectors, s.time_span_days, s.n_datapoints
            FROM candidates c LEFT JOIN stars s ON c.tic_id = s.tic_id
            WHERE {where_clause}
            ORDER BY {order_by}
            {f"LIMIT {int(limit)}" if limit > 0 else ""}
        """,
    ).fetchall()


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI parser. Extracted from main() so tests can exercise
    argument parsing (mutual exclusion, defaults) without running the rest
    of main() (DB access, network calls)."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--db",
                    default=os.environ.get("EXONIT_DATABASE_URL")
                            or "data/exonit.db",
                    help="Operational DB: a postgresql:// URL (candidates live "
                         "in PG since the 2026-06-24 migration) or a SQLite path. "
                         "Defaults to $EXONIT_DATABASE_URL, else data/exonit.db.")
    url_group = ap.add_mutually_exclusive_group()
    url_group.add_argument("--explainer-url",
                    # <host> vLLM explainer (canonical since the LoRA
                    # moved there; .60 was the old <host> experiment).
                    # default=None here (not the URL above) so the mutually
                    # exclusive group can tell "not given" apart from "given
                    # explicitly" — the historical default is applied below,
                    # after parsing, only when neither flag was passed.
                    default=None)
    url_group.add_argument("--endpoints-json",
                    help="Path to a JSON file with a list of "
                         "{\"url\": ..., \"concurrency\": N, \"name\": ...} "
                         "explainer endpoints (written by the LoRA-on-Macs "
                         "driver). Work is distributed across all live "
                         "endpoints via a shared queue "
                         "(exonit.explainer_pool.EndpointPool) instead of a "
                         "single URL + --workers threads. Mutually exclusive "
                         "with --explainer-url.")
    ap.add_argument("--max-seconds", type=float, default=None,
                    help="Deadline in seconds. Once elapsed, no new explain "
                         "call is dispatched (in-flight ones still drain); "
                         "candidates still queued are left unexplained for "
                         "the next run (not counted as failures). Reported "
                         "as deadline_hit=0/1 in the [done] line. Forces the "
                         "pool code path even with a single --explainer-url.")
    ap.add_argument("--timeout", type=float, default=90.0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--mark-tag", default="[lora-v2]")
    ap.add_argument("--include-stale", action="store_true",
                    help="also regenerate rows with non-empty claude_analysis "
                         "lacking the [lora-v2] prefix (legacy Claude text).")
    ap.add_argument("--strong-only", action="store_true",
                    help="only explain candidates with claude_verdict='planet_candidate'. "
                         "Use for discovery runs to skip the ~30k non-planet candidates.")
    ap.add_argument("--sector", type=int, default=None,
                    help="only explain candidates with candidates.sector = N. "
                         "The discovery orchestrator passes its run's sector so a "
                         "lot's LoRA step never re-explains the whole DB "
                         "(2026-09-20 s5 incident: 222k rows, 52h vs 24h timeout).")
    ap.add_argument("--order", choices=sorted(ORDER_BY), default="id",
                    help="'id' (default, oldest first) or 'strong' (strong-tier gate "
                         "snr>=10 & cnn>=0.7 first, then id) — use with --limit for "
                         "manual backfills of the historic backlog.")
    ap.add_argument("--server-url",
                    default=os.environ.get("EXONIT_SERVER_URL") or "http://127.0.0.1:8400",
                    help="Base URL of the ExoNit server (used for the "
                         "batch-write endpoint). Defaults to $EXONIT_SERVER_URL, "
                         "else the local server (this script runs on the server host).")
    ap.add_argument("--batch-size", type=int, default=25,
                    help="Number of results to accumulate before flushing to "
                         "the server via POST /api/candidate/update_analysis.")
    ap.add_argument("--worker-token",
                    default=os.environ.get("EXONIT_WORKER_TOKEN",
                                           "exonit-default-token"),
                    help="Bearer token for the ExoNit server API.")
    return ap


#: Hard guard (2026-09-26 incident, tightened 2026-09-26 review follow-up):
#: a pytest run reached this script for real (via an unmocked
#: discovery_orchestrator.step_lora -> lora_macs_driver.py ->
#: generate_missing_explanations.py chain) with the default --db, and wrote
#: 1,650 real LoRA explanations to the production Postgres (a private-LAN
#: address). pytest sets PYTEST_CURRENT_TEST in os.environ for the
#: duration of every test (subprocesses inherit it), so this is a reliable
#: signal we are running under pytest.
#:
#: Review follow-up: the --db check alone keys off the wrong signal for the
#: GENERAL case — the actual dangerous mutation this script performs is the
#: POST {--server-url}/api/candidate/update_analysis call, and --server-url
#: defaults to the real ExoNit server (the local one, or $EXONIT_SERVER_URL)
#: independently of --db. A hypothetical future caller with a harmless local
#: --db (e.g. an empty tmp SQLite file) but the default --server-url would
#: sail past a --db-only check and still POST any matching rows to prod. So
#: this guard now fires on EITHER signal: --db being any PostgreSQL DSN whose
#: database name does not end in `_test`, OR --server-url resolving to a
#: private-LAN address or to the local production port (8400) — a test's own
#: local HTTP server/TestClient binds a random port, never 8400.
#:
#: Every test in this repo's own suite (tests/test_generate_missing_explanations.py)
#: already points --db at a throwaway tmp_path SQLite file and --server-url
#: at a non-prod placeholder (see that file's _run_main helper), so this
#: never fires for them; a test that genuinely needs to hit a real Postgres/
#: server end-to-end must point both at non-prod targets, or set
#: EXONIT_GEN_EXPLANATIONS_ALLOW_UNDER_PYTEST=1 explicitly.
GEN_EXPLANATIONS_PYTEST_GUARD_ENV = "EXONIT_GEN_EXPLANATIONS_ALLOW_UNDER_PYTEST"


#: The production server's port. A test's own local server uses a random port, so
#: loopback + 8400 is the real server (this script usually runs on the server host).
PROD_SERVER_PORT = 8400


def _is_prod_db_dsn(db: str) -> bool:
    """True for ANY PostgreSQL DSN whose dbname does NOT end in `_test` — e.g.
    ``postgresql://exonit@10.0.0.5:5432/exonit`` or ``postgresql://exonit@localhost/exonit``
    (a local PostgreSQL is the real database on a single-host install). A plain SQLite
    path is never considered "prod" by this check."""
    if not db.startswith(("postgresql://", "postgres://")):
        return False
    dbname = db.rsplit("/", 1)[-1].split("?", 1)[0]
    return not dbname.endswith("_test")


def _is_prod_server_url(server_url: str) -> bool:
    """True if *server_url* is real infrastructure: a private-LAN address (10/8, 172.16/12,
    192.168/16), or loopback on the production port 8400. Loopback on any other port
    (what a test's own local HTTP server or FastAPI TestClient binds to) and
    hostnames are never considered prod."""
    try:
        parts = urlsplit(server_url)
        host, port = parts.hostname or "", parts.port
    except ValueError:
        return False
    if host in ("localhost", "127.0.0.1", "::1"):
        return port == PROD_SERVER_PORT
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return False
    return ip.is_private and not ip.is_loopback and not ip.is_unspecified


def main() -> int:
    args = build_arg_parser().parse_args()

    prod_db = _is_prod_db_dsn(args.db)
    prod_server = _is_prod_server_url(args.server_url)
    if ("PYTEST_CURRENT_TEST" in os.environ and (prod_db or prod_server)
            and os.environ.get(GEN_EXPLANATIONS_PYTEST_GUARD_ENV) != "1"):
        reasons = []
        if prod_db:
            reasons.append(f"--db points at a LAN PostgreSQL server ({args.db!r})")
        if prod_server:
            reasons.append(f"--server-url points at a private-LAN host ({args.server_url!r})")
        print(
            f"REFUSED: generate_missing_explanations.py detected "
            f"PYTEST_CURRENT_TEST in the environment (running under pytest) "
            f"and {' and '.join(reasons)} — refusing to touch real candidate "
            f"data (2026-09-26 incident: an unmocked test run wrote 1,650 "
            f"real LoRA explanations to prod). Point --db at a database "
            f"whose name ends in `_test` and --server-url at a "
            f"localhost/127.0.0.1 test server, or set "
            f"{GEN_EXPLANATIONS_PYTEST_GUARD_ENV}=1 if you have mocked every "
            f"remote/DB call yourself.",
            file=sys.stderr,
        )
        return 2

    # Historical default, applied only when neither --explainer-url nor
    # --endpoints-json was given explicitly (see the mutually-exclusive
    # group above).
    if args.endpoints_json is None and args.explainer_url is None:
        args.explainer_url = (os.environ.get("EXONIT_EXPLAINER_URL")
                              or "http://127.0.0.1:8500/api/explain")

    # use_pool selects the new exonit.explainer_pool.EndpointPool code path.
    # A single --explainer-url + --workers N with no --max-seconds keeps the
    # historical ThreadPoolExecutor path below completely untouched, so its
    # output ([health]/[plan]/progress/[done] lines) and exit codes stay
    # byte-identical to today.
    use_pool = args.endpoints_json is not None or args.max_seconds is not None
    endpoints_cfg: list[dict] | None = None

    if not use_pool:
        # Historical single-endpoint health check — untouched.
        health_url = args.explainer_url.replace("/api/explain", "/healthz")
        try:
            h = requests.get(health_url, timeout=5).json()
            print(f"[health] {h}")
        except Exception as e:
            print(f"[health] FAIL: {e}", file=sys.stderr)
            return 1
    else:
        if args.endpoints_json:
            try:
                endpoints_cfg = load_endpoints_json(args.endpoints_json)
            except (OSError, ValueError) as e:
                print(f"[health] FAIL: cannot load --endpoints-json "
                      f"{args.endpoints_json}: {e}", file=sys.stderr, flush=True)
                return 2
        else:
            endpoints_cfg = [{"url": args.explainer_url,
                             "concurrency": max(1, args.workers),
                             "name": "default"}]
        endpoints_cfg = health_check_endpoints(endpoints_cfg)
        if not endpoints_cfg:
            print("[health] FAIL: no endpoint responded (need >=1)",
                  file=sys.stderr, flush=True)
            return 2

    # Load classifier (for rows with no verdict)
    from exonit.verdict_classifier import predict_verdict as predict_fn
    print("[classifier] loaded predict_verdict")

    # Read candidates from the operational DB. Since the 2026-06-24 migration the
    # candidates table lives in PostgreSQL (data/exonit.db is completeness-only),
    # so a SQLite read here raises "no such table: candidates". Writes still go
    # through the server API below, so we only need a read connection. The
    # select_candidates SQL is dialect-portable (length/coalesce/IS NULL/NOT LIKE).
    from exonit.db_backend import is_pg_url
    if is_pg_url(args.db):
        import psycopg
        from psycopg.rows import dict_row
        con = psycopg.connect(args.db, row_factory=dict_row)
        try:
            rows = select_candidates(con, args.include_stale, args.strong_only,
                                     args.limit, sector=args.sector,
                                     order=args.order)
        finally:
            con.close()
    else:
        con = sqlite3.connect(args.db, timeout=30.0)
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA busy_timeout=10000")
        rows = select_candidates(con, args.include_stale, args.strong_only,
                                 args.limit, sector=args.sector, order=args.order)
        con.close()

    scope = f" sector={args.sector}" if args.sector is not None else " (ALL sectors)"
    if args.strong_only:
        print(f"[plan] strong-only{scope} order={args.order}: "
              f"{len(rows)} planet_candidate to explain")
    else:
        print(f"[plan] {len(rows)} candidates to explain")
    if not use_pool:
        print(f"[plan] using {args.explainer_url} workers={args.workers}")
    else:
        ep_desc = ", ".join(f"{e['name']}(c={e['concurrency']})" for e in endpoints_cfg)
        total_c = sum(e["concurrency"] for e in endpoints_cfg)
        deadline_desc = f" max_seconds={args.max_seconds}" if args.max_seconds else ""
        print(f"[plan] using {len(endpoints_cfg)} endpoint(s): {ep_desc} "
              f"total_concurrency={total_c}{deadline_desc}", flush=True)
    if args.dry_run:
        print("[dry-run] not writing to DB")

    n_ok = n_fail = n_flush_err = n_skipped = 0
    n_verdict_set = 0
    deadline_hit = 0
    t0 = time.time()

    # Write path: POST batches to the server instead of opening a direct
    # SQLite write connection.  This avoids write-lock contention with the
    # server's WAL connection while a run is active.
    update_url = args.server_url.rstrip("/") + "/api/candidate/update_analysis"
    auth_headers = {"Authorization": f"Bearer {args.worker_token}"}
    pending_batch: list[dict] = []

    if not use_pool:
        # Historical single-endpoint path — untouched, byte-identical output.
        def process(row):
            verdict = row["claude_verdict"]
            verdict_assigned = False
            if not verdict:
                v = compute_verdict(row, predict_fn)
                if v:
                    verdict = v
                    verdict_assigned = True
            payload = build_payload(row, verdict)
            text = call_explainer(args.explainer_url, payload, args.timeout)
            return row["id"], verdict, verdict_assigned, text

        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            for i, (cid, verdict, verdict_assigned, text) in enumerate(
                    ex.map(process, rows), 1):
                if text is None:
                    n_fail += 1
                else:
                    tag = args.mark_tag.strip()
                    final = f"{tag} {text}" if tag else text
                    if not args.dry_run:
                        entry = {"candidate_id": cid, "analysis": final}
                        if verdict_assigned and verdict:
                            # Only include "verdict" when we actually assigned one.
                            # Absent key → server/db will NOT touch claude_verdict,
                            # preserving any existing value for this candidate.
                            entry["verdict"] = verdict
                            n_verdict_set += 1
                        pending_batch.append(entry)
                        # Flush when batch is full.
                        if len(pending_batch) >= args.batch_size:
                            if not flush_batch(pending_batch, update_url, auth_headers, 30.0):
                                n_flush_err += 1
                            pending_batch.clear()
                    n_ok += 1
                if i % 25 == 0 or i == len(rows):
                    elapsed = time.time() - t0
                    rate = i / max(1e-6, elapsed)
                    eta = (len(rows) - i) / max(1e-6, rate)
                    print(f"  {i}/{len(rows)} ok={n_ok} fail={n_fail} "
                          f"verdict_set={n_verdict_set} flush_err={n_flush_err} "
                          f"rate={rate:.2f}/s eta={eta/60:.1f}min", flush=True)
    else:
        # Multi-endpoint (or single-endpoint + --max-seconds) path: work is
        # distributed across all live endpoints through a shared queue
        # (exonit.explainer_pool.EndpointPool). The CPU-local part (verdict
        # assignment via the local classifier + payload building) is done up
        # front, sequentially — it's classifier inference, not network I/O,
        # so it does not need to be spread across endpoint threads.
        from exonit.explainer_pool import Endpoint, EndpointPool

        def _pool_call(ep: Endpoint, item: tuple) -> str:
            # item is (cid, verdict, verdict_assigned, explain_payload) — see
            # `items` below; only the explain payload goes over the wire.
            _cid, _verdict, _verdict_assigned, payload = item
            text = call_explainer(ep.url, payload, args.timeout)
            if text is None:
                raise RuntimeError(f"explain call failed via {ep.name}")
            return text

        items = []
        for row in rows:
            verdict = row["claude_verdict"]
            verdict_assigned = False
            if not verdict:
                v = compute_verdict(row, predict_fn)
                if v:
                    verdict = v
                    verdict_assigned = True
            payload = build_payload(row, verdict)
            items.append((row["id"], verdict, verdict_assigned, payload))

        pool = EndpointPool(endpoints_cfg, call_fn=_pool_call, timeout=args.timeout,
                            max_seconds=args.max_seconds)
        for i, result in enumerate(pool.run(items), 1):
            cid, verdict, verdict_assigned, _payload = result.payload
            if result.skipped:
                n_skipped += 1
            elif not result.ok:
                n_fail += 1
            else:
                text = result.value
                tag = args.mark_tag.strip()
                final = f"{tag} {text}" if tag else text
                if not args.dry_run:
                    entry = {"candidate_id": cid, "analysis": final}
                    if verdict_assigned and verdict:
                        entry["verdict"] = verdict
                        n_verdict_set += 1
                    pending_batch.append(entry)
                    if len(pending_batch) >= args.batch_size:
                        if not flush_batch(pending_batch, update_url, auth_headers, 30.0):
                            n_flush_err += 1
                        pending_batch.clear()
                n_ok += 1
            if i % 25 == 0 or i == len(items):
                elapsed = time.time() - t0
                rate = i / max(1e-6, elapsed)
                eta = (len(items) - i) / max(1e-6, rate)
                print(f"  {i}/{len(items)} ok={n_ok} fail={n_fail} "
                      f"verdict_set={n_verdict_set} flush_err={n_flush_err} "
                      f"skipped={n_skipped} rate={rate:.2f}/s eta={eta/60:.1f}min",
                      flush=True)

        deadline_hit = int(pool.deadline_hit)
        print(f"[pool] endpoint stats: {pool.stats()}", flush=True)

    # Flush any remaining items.
    if not args.dry_run and pending_batch:
        if not flush_batch(pending_batch, update_url, auth_headers, 30.0):
            n_flush_err += 1
        pending_batch.clear()

    if not use_pool:
        print(f"[done] ok={n_ok} fail={n_fail} verdict_set={n_verdict_set} "
              f"flush_err={n_flush_err} elapsed={(time.time()-t0)/60:.1f}min")
    else:
        print(f"[done] ok={n_ok} fail={n_fail} verdict_set={n_verdict_set} "
              f"flush_err={n_flush_err} skipped={n_skipped} "
              f"elapsed={(time.time()-t0)/60:.1f}min deadline_hit={deadline_hit}",
              flush=True)
    return 0 if (n_fail == 0 and n_flush_err == 0) else 2


if __name__ == "__main__":
    sys.exit(main())
