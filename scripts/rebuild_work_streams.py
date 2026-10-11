#!/usr/bin/env python3
"""Rebuild a work stream from the operational DB (source of record).

Usage: python scripts/rebuild_work_streams.py --stage fpp [--dry-run]

Clears the target stream and re-derives it using the SAME WHERE clauses as
the legacy scan path (the two arms of GET /api/worker/fpp_task, copied
verbatim from exonit/routes/workers.py:805-824 — see exonit/work_producers.py
for the predicate form of the same clauses), ORDER BY snr DESC so the queue
drains most-promising-first, with NO LIMIT (the live handler caps each arm
at 200 for the hot poll path; a rebuild wants every eligible row once). Safe
to run any time the fleet is idle for that stage — a candidate currently
being computed would just be re-enqueued and recomputed (wasteful, not
incorrect: the consumer re-checks eligibility against PG before serving).

This is also the reconciliation path for writers that bypass the
work-stream producer hooks (exonit.work_producers) entirely —
scripts/rescore_cnn_ffi.py updates candidates.cnn_score via raw SQL and
never calls maybe_enqueue_fpp, so any candidate it newly makes eligible
only enters the queue on the next rebuild.

Respects EXONIT_DATABASE_URL like every other script in scripts/, via
exonit.config.ExoNitConfig (db_path favours EXONIT_DATABASE_URL over
EXONIT_DB_PATH / the default SQLite file). Prints which DB it resolved to,
because outside systemd (which sources .env into the process environment)
EXONIT_DATABASE_URL is unset and the config silently falls back to the
local SQLite file — a startup message here prevents rebuilding against the
wrong DB unnoticed.

--stage exominer re-derives the "exominer" stream using the SAME WHERE as
the legacy GET /api/worker/exominer_task scan (has_spoc=1, cnn_score>=0.7,
exominer_score IS NULL, star completed — see exonit/work_producers.py's
exominer_eligible for the predicate form). It deliberately does NOT filter
on (or -1-mark for) missing metadata-cache entries: has_spoc=1 means
MAST-confirmed SPOC data exists, so a missing entry is an indexing gap,
not an eligibility fact. The legacy handler's serve-time -1-marking was a
2026-06-10 queue-drain mechanism (commit 2ccb7888) that the stream
replaces structurally — a metadata-less item is held back at claim time
(no ack, no write) and lands in the DLQ after max_deliveries, visible and
re-enqueueable by a rebuild once the metadata is indexed. Writing -1 at
rebuild time would permanently and silently discard a computable ExoMiner
score. The -1 sentinel is only written by the SPOC-gate producer paths
(has_spoc=0 per-TIC, has_spoc=NULL bulk), whose rows this WHERE already
excludes.

--stage primary re-derives the "primary" (BLS) task stream using the SAME
WHERE + ORDER as the legacy get_and_assign_task query (status='pending',
ORDER BY priority ASC, created_at ASC — see exonit/work_producers.py's
primary_eligible for the predicate form). No joined-table filter: a task's
own status column is its only eligibility fact. Every writer in this repo
hardcodes priority=2, so this ordering is currently equivalent to plain
created_at ASC, but the explicit ORDER BY keeps the rebuild correct even if
that ever changes.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _describe_db(db_path: str) -> str:
    """Human-readable, credential-redacted description of the resolved DB."""
    if db_path.startswith("postgresql://") or db_path.startswith("postgres://"):
        return re.sub(r"://([^:/@]+):[^@]*@", r"://\1:***@", db_path)
    return f"sqlite:{db_path}"


# The derivation SQL moved to exonit/work_producers.py (2026-07-14) so this
# script and the orchestrator's automatic reconciliation cannot drift apart —
# they must select the same rows by construction, not by two copies staying in
# sync by hand. This script keeps its own semantics: CLEAR + rebuild (see
# main()), vs reconcile_stream()'s non-destructive top-up.
from exonit.work_producers import (          # noqa: E402
    _derive_exominer,                        # noqa: F401  (re-exported: this
    _derive_fpp,                             # noqa: F401   module was their home
    _derive_primary,                         # noqa: F401   until 2026-07-14 and
    derive_eligible,                         # noqa: F401   tests import them here)
)


def _derive(db, stage: str):
    return derive_eligible(db, stage)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["fpp", "exominer", "primary"])
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    from exonit.config import ExoNitConfig
    from exonit.db import ExoNitDB
    from exonit.work_streams import get_stream

    cfg = ExoNitConfig()
    print(f"[rebuild_work_streams] db = {_describe_db(cfg.db_path)}",
          file=sys.stderr)
    db = ExoNitDB(cfg.db_path)

    rows = _derive(db, args.stage)
    print(json.dumps({"stage": args.stage, "eligible": len(rows),
                      "dry_run": args.dry_run}))
    if args.dry_run:
        return 0
    ws = get_stream(args.stage)
    ws.clear()
    # G3 (review package, 2026-07-05): enqueue_many() batches the SADD/XADD
    # round trips (chunked pipelines on Redis) instead of one enqueue() call
    # per row — a rebuild can carry tens of thousands of rows (see the
    # module docstring), and the per-row loop this replaced made that many
    # synchronous round trips in this single-threaded script.
    n = ws.enqueue_many(rows)
    print(json.dumps({"stage": args.stage, "enqueued": n}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
