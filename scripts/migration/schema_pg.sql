-- ============================================================
-- ExoNit PostgreSQL 18 schema
-- Target: <postgres-host>:5432, database "exonit", role "exonit"
-- Source: SQLite data/exonit.db
-- Generated: 2026-06-20
-- ============================================================
--
-- TYPE MAPPING DECISIONS (SQLite → PostgreSQL)
-- ─────────────────────────────────────────────
-- TEXT                → TEXT            (unchanged; PG TEXT is uncapped, same as SQLite)
-- REAL                → DOUBLE PRECISION (SQLite REAL is 8-byte IEEE 754, same as PG DOUBLE PRECISION)
-- INTEGER (pk/counter)→ BIGINT          (future-proof; SQLite INTEGER can be 1-8 bytes, PG INTEGER is 4-byte)
-- INTEGER (0/1 bool)  → BOOLEAN         (ctoi_submitted, gpu_used, lc_load_recovered, has_spoc,
--                                         debug_enabled, cross_source_agreement; these are always 0/1)
-- TEXT (datetime)     → TIMESTAMPTZ     (all datetime columns store ISO-8601 UTC strings;
--                                         AT TIME ZONE 'UTC' is applied during COPY load)
-- sqlite_sequence     → (omitted; PG uses SEQUENCE objects, reset post-load)
-- completeness        → (EXCLUDED: huge table, goes to Parquet)
--
-- PARTITIONING STRATEGY
-- ─────────────────────
-- candidates: PARTITION BY LIST (sector)  – sector 1 (~99%) + sector 2 + default
-- stars:      no sector column directly; partitioned by HASH(tic_id) for write parallelism
--             (the tic_id column is a text TIC number; tic_id % N is not meaningful in PG,
--              HASH partitioning keeps PK distribution even across future shards)
--             Alternatively kept unpartitioned since JOINs are by tic_id (FK in candidates/tasks).
--             DECISION: stars kept unpartitioned (FK targets cannot be partitioned parents in PG <17
--             unless parent has partition key; FK in candidates references stars.tic_id directly).
--             If needed, partition stars by tic_id hash later once FK constraints are dropped.
-- tasks:      NOT partitioned (Change 1, 2026-06-20).
--             Partitioning by status forced PRIMARY KEY (task_id, status), allowing
--             the same task_id to appear with multiple status values — a silent
--             correctness hazard because db.py looks up tasks by task_id alone
--             (≥9 call sites do WHERE task_id=?). At 1.9M rows the table is well
--             within SQLite-era norms; a plain (status, priority) index is sufficient.
--             Restore single-column PRIMARY KEY (task_id) and unpartitioned table.
-- completeness_agg: tiny (72 rows), no partitioning needed.
-- worker_telemetry:  PARTITION BY RANGE (ts) monthly, but given current scale (280k rows)
--                    start with a single partition + default and add monthly later.
-- ============================================================

-- Ensure we're in the right database (checked by caller, not enforced in SQL)

-- ─────────────────────────────────────────────────────────────
-- EXTENSION (for future UUID / stats use)
-- ─────────────────────────────────────────────────────────────
CREATE EXTENSION IF NOT EXISTS pg_stat_statements;

-- ─────────────────────────────────────────────────────────────
-- TABLE: stars
-- ─────────────────────────────────────────────────────────────
-- Not partitioned: serves as FK parent for candidates and tasks.
-- Adding partitioning to the parent of a FK constraint requires
-- careful re-evaluation; defer until data volume warrants it.
CREATE TABLE stars (
    tic_id           TEXT         NOT NULL,
    ra               DOUBLE PRECISION,
    dec              DOUBLE PRECISION,
    tmag             DOUBLE PRECISION,
    analyzed_at      TIMESTAMPTZ,
    n_sectors        BIGINT,
    n_datapoints     BIGINT,
    time_span_days   DOUBLE PRECISION,
    status           TEXT         NOT NULL DEFAULT 'completed',
    sectors_observed TEXT,
    has_spoc         BOOLEAN,
    PRIMARY KEY (tic_id)
);

-- Partial index for /api/stars: lets PG satisfy the WHERE status='completed'
-- + ORDER BY analyzed_at DESC + LIMIT via an index range scan on the completed
-- subset only, avoiding a sequential scan + sort over all ~3M star rows.
CREATE INDEX IF NOT EXISTS idx_stars_completed_analyzed ON stars (analyzed_at DESC) WHERE status = 'completed';

-- P3 (docs/reviews/2026-07-02-opus48-cleanup-review.md): stars.has_spoc had NO
-- index anywhere, despite being the hot filter behind the 2026-06-20 PG
-- meltdown (planner mis-estimated a has_spoc-filtered join at 1.45M rows vs
-- an actual 161 and picked a nested-loop scan; the current mitigation is a
-- 20s TTL cache band-aid, not an index). Partial on has_spoc=TRUE (the
-- selective, "has 2-min SPOC" side of the flag) so it stays cheap: most rows
-- are FFI-only (FALSE) or unchecked (NULL) and never enter the index. Mirrors
-- exonit/db.py's idx_stars_has_spoc (SQLite side) — added there and here
-- together so the index set stays reproducible from source on both backends.
CREATE INDEX IF NOT EXISTS idx_stars_has_spoc ON stars (tic_id) WHERE has_spoc = TRUE;
CREATE INDEX IF NOT EXISTS idx_stars_spoc_completed_tic ON stars (tic_id)
    WHERE has_spoc AND status = 'completed';  -- 2026-10-05, health/work em_by_tic

-- ─────────────────────────────────────────────────────────────
-- TABLE: candidates  (partitioned by sector)
-- ─────────────────────────────────────────────────────────────
-- SQLite uses AUTOINCREMENT (sequence-backed); PG uses BIGSERIAL.
-- The "id" surrogate key is carried over; unique constraint
-- (sector, tic_id, candidate_num) is enforced per partition.
CREATE TABLE candidates (
    id                        BIGINT         NOT NULL,  -- GENERATED ALWAYS added per-partition
    -- NOT NULL + FK is an intentional tightening vs SQLite (SQLite allowed NULL tic_id
    -- on candidates).  Dry-run on the live DB confirmed 0 NULL tic_id rows, so migration
    -- is safe.  Application code going forward must insert the star row before the
    -- candidate row (the add_candidate race fix 1161f230 already enforces this).
    tic_id                    TEXT           NOT NULL REFERENCES stars(tic_id),
    candidate_num             BIGINT,
    period_days               DOUBLE PRECISION,
    t0                        DOUBLE PRECISION,
    duration_days             DOUBLE PRECISION,
    depth_ppm                 DOUBLE PRECISION,
    snr                       DOUBLE PRECISION,
    cnn_score                 DOUBLE PRECISION,
    triceratops_fpp           DOUBLE PRECISION,
    claude_verdict            TEXT,
    claude_analysis           TEXT,
    ctoi_submitted            BOOLEAN        NOT NULL DEFAULT FALSE,
    exominer_score            DOUBLE PRECISION,
    exominer_tce_period       DOUBLE PRECISION,
    exominer_sector_run       TEXT,
    -- exominer_status (docs/superpowers/specs/exominer-status-column.md,
    -- "Step 2", 2026-07-14): additive state column alongside exominer_score
    -- (unchanged, -1 sentinel and all). On the already-migrated production
    -- DB this column is added + backfilled by
    -- scripts/migrate_exominer_status.py, NOT by re-running this file (see
    -- the DEPLOY NOTE above idx_cand_fpp_todo below) — it is included here
    -- only so a FRESH bootstrap from this schema starts with the column
    -- already present.
    exominer_status            TEXT,
    triceratops_nfpp          DOUBLE PRECISION,
    triceratops_prob_planet   DOUBLE PRECISION,
    triceratops_verdict       TEXT,
    -- fpp_skip_reason (2026-07-14): WHY a row carries the fpp=-1 sentinel.
    -- 'no_stellar_mass'   — the catalogue has no mass for the host (TIC v8
    --                       only derives masses on the main sequence, so these
    --                       are overwhelmingly red giants): TRICERATOPS cannot
    --                       run at all.
    -- 'stellar_companion' — depth + R* imply a companion far too large to be a
    --                       planet (exonit/stellar_gates.py): an eclipsing
    --                       binary or a blend. TRICERATOPS SHOULD not run.
    -- NULL = a real TRICERATOPS result, or never gated. On the already-migrated
    -- production DB this column is added by ALTER TABLE, not by re-running this
    -- file (same deploy note as exominer_status above).
    fpp_skip_reason           TEXT,
    -- verdict_proba (P3): raw classifier P(planet), NULL = not computed. On the
    -- already-migrated production DB this column is added manually (as
    -- postgres): ALTER TABLE candidates ADD COLUMN IF NOT EXISTS
    -- verdict_proba DOUBLE PRECISION; -- BEFORE deploying the server code.
    verdict_proba             DOUBLE PRECISION,
    exominer_ffi_score        DOUBLE PRECISION,
    exominer_ffi_model_version TEXT,
    cross_source_name         TEXT,
    cross_source_snr          DOUBLE PRECISION,
    cross_source_depth_ppm    DOUBLE PRECISION,
    cross_source_depth_ratio  DOUBLE PRECISION,
    cross_source_agreement    BOOLEAN,
    sector                    BIGINT         NOT NULL DEFAULT 1,
    PRIMARY KEY (id, sector)
) PARTITION BY LIST (sector);

-- Partition: sector 1 (≈99% of current data, ~4.98M rows)
CREATE TABLE candidates_sector1 PARTITION OF candidates
    FOR VALUES IN (1);

-- Partition: sector 2 (~14k rows currently)
CREATE TABLE candidates_sector2 PARTITION OF candidates
    FOR VALUES IN (2);

-- Partition: sector 23 (s23 discovery campaign, ~679k rows)
CREATE TABLE candidates_sector23 PARTITION OF candidates
    FOR VALUES IN (23);

-- Default partition for future sectors (3, 4, ...)
CREATE TABLE candidates_default PARTITION OF candidates
    DEFAULT;

-- Unique constraint per partition (equivalent to SQLite idx_candidates_unique)
CREATE UNIQUE INDEX idx_cand_s1_unique  ON candidates_sector1 (sector, tic_id, candidate_num);
CREATE UNIQUE INDEX idx_cand_s2_unique  ON candidates_sector2 (sector, tic_id, candidate_num);
CREATE UNIQUE INDEX idx_cand_s23_unique ON candidates_sector23 (sector, tic_id, candidate_num);
CREATE UNIQUE INDEX idx_cand_def_unique ON candidates_default  (sector, tic_id, candidate_num);

-- Operational indexes (mirroring SQLite indexes)
CREATE INDEX idx_candidates_snr        ON candidates (snr DESC);
CREATE INDEX idx_candidates_verdict_snr ON candidates (claude_verdict, snr DESC);
CREATE INDEX idx_candidates_tic        ON candidates (tic_id);
CREATE INDEX idx_candidates_cnn        ON candidates (cnn_score);

-- Partial indexes (WHERE clause equivalents — PG supports these natively)
--
-- idx_cand_fpp_todo / idx_cand_nfpp_todo / idx_cand_strong predicates below
-- carry the 2026-07-11 OR-gate (docs/reviews/injection_e2e_results_2026-07-
-- 11.md + docs/reviews/ffi_cnn_investigation_2026-07-11.md §5):
-- "cnn_score >= 0.7 OR exominer_ffi_score >= 0.5" replaces the old
-- cnn_score>=0.7-only eligibility everywhere in the verdict/FPP/strong
-- chain (exonit/work_producers.verdict_eligible is the Python-side source
-- of truth; these predicates are its SQL-side mirror — see
-- progress/impl-orgate.md). idx_cand_exominer_todo (Docker-SPOC ExoMiner
-- backfill, EXONIT_ENABLE_EXOMINER_AUTO=0 in prod) is DELIBERATELY left
-- unchanged — out of scope, item 6 of progress/impl-orgate.md.
--
-- DEPLOY NOTE (this file is applied once, not idempotently, per the
-- pre-existing idx_cand_strong NULLIF precedent below): on an
-- already-migrated production DB these three indexes must be rebuilt by
-- hand — run, against live PG, during a deploy window (NOT while a
-- discovery run is live; candidates is a 10.9M+-row partitioned table so a
-- plain (non-CONCURRENTLY) DROP+CREATE briefly locks writers):
--   DROP INDEX IF EXISTS idx_cand_fpp_todo;
--   CREATE INDEX idx_cand_fpp_todo ON candidates (snr DESC)
--       WHERE triceratops_fpp IS NULL AND snr >= 10.0
--         AND (cnn_score >= 0.7 OR exominer_ffi_score >= 0.5);
--   DROP INDEX IF EXISTS idx_cand_nfpp_todo;
--   CREATE INDEX idx_cand_nfpp_todo ON candidates (snr DESC)
--       WHERE triceratops_nfpp IS NULL
--         AND triceratops_fpp >= 0 AND triceratops_fpp < 0.5
--         AND snr >= 10.0 AND (cnn_score >= 0.7 OR exominer_ffi_score >= 0.5);
--   DROP INDEX IF EXISTS idx_cand_strong;
--   CREATE INDEX idx_cand_strong ON candidates (snr DESC)
--       WHERE (cnn_score >= 0.7 OR exominer_ffi_score >= 0.5)
--         AND COALESCE(NULLIF(exominer_score, -1), exominer_ffi_score) >= 0.7;
-- Until that manual step is done, PG silently falls back to a full scan on
-- these three query families (correct results, just slower — same
-- fail-safe as the pre-existing idx_cand_strong NULLIF migration of
-- 2026-07-02, see progress/impl-minus1-fix-2026-07-02.md).
CREATE INDEX idx_cand_fpp_todo ON candidates (snr DESC)
    WHERE triceratops_fpp IS NULL AND snr >= 10.0
      AND (cnn_score >= 0.7 OR exominer_ffi_score >= 0.5);

CREATE INDEX idx_cand_nfpp_todo ON candidates (snr DESC)
    WHERE triceratops_nfpp IS NULL
      AND triceratops_fpp >= 0 AND triceratops_fpp < 0.5
      AND snr >= 10.0 AND (cnn_score >= 0.7 OR exominer_ffi_score >= 0.5);

CREATE INDEX idx_cand_exominer_todo ON candidates (snr DESC)
    WHERE exominer_score IS NULL AND cnn_score >= 0.7;

-- Backlog-query tic_id indexes (2026-10-05, see
-- scripts/migration/2026-10-05_backlog_partial_indexes.sql): drive
-- /api/health/work's em_by_tic join as index-only scans.
CREATE INDEX idx_cand_em_todo_tic ON candidates (tic_id)
    WHERE exominer_score IS NULL AND cnn_score >= 0.7;

-- idx_cand_exominer_status_todo (docs/superpowers/specs/exominer-status-
-- column.md, "Step 2"): same shape as idx_cand_exominer_todo above, keyed on
-- the new status column. Not read by any production query yet in this
-- iteration — see progress/impl-exominer-status.md. On the live production
-- DB this index is created by scripts/migrate_exominer_status.py (after its
-- backfill), not by re-running this file.
CREATE INDEX idx_cand_exominer_status_todo ON candidates (snr DESC)
    WHERE exominer_status = 'pending' AND cnn_score >= 0.7;

-- Strong-tier index (COALESCE translated to GREATEST with NULL handling).
-- NULLIF(exominer_score, -1): exominer_score carries a -1 no-DV cache-
-- negative sentinel (see exonit/db.py's _strong_where docstring and
-- progress/investigation-exominer-minus1-2026-07-02.md) that must read as
-- "absent", not "score -1", so it falls back to exominer_ffi_score like a
-- real NULL would. The predicate here MUST stay textually in sync with
-- _strong_where's SQL (neither SQLite nor PG proves predicate implication
-- across a NULLIF rewrite, so a mismatched index silently stops being used
-- by the planner -> full scan). On an already-migrated production DB this
-- index must be rebuilt by hand (this file is applied once, not
-- idempotently) -- see progress/impl-minus1-fix-2026-07-02.md for the
-- DROP INDEX + CREATE INDEX deploy step (and the OR-gate deploy note
-- above for the 2026-07-11 predicate update).
CREATE INDEX idx_cand_strong ON candidates (snr DESC)
    WHERE (cnn_score >= 0.7 OR exominer_ffi_score >= 0.5)
      AND COALESCE(NULLIF(exominer_score, -1), exominer_ffi_score) >= 0.7;

CREATE INDEX idx_cand_nfpp_pending ON candidates (tic_id)
    WHERE triceratops_fpp >= 0 AND triceratops_fpp < 0.5
      AND triceratops_nfpp IS NULL;

CREATE INDEX idx_cand_verdict_notnull ON candidates (triceratops_verdict)
    WHERE triceratops_verdict IS NOT NULL;

-- Sequence for id (reset post-load to MAX(id)+1)
CREATE SEQUENCE candidates_id_seq;
-- Bind the sequence as the column default so application INSERTs that omit "id"
-- (ExoNitDB.add_candidate via execute_returning_id) auto-generate it.  The COPY
-- bulk-load still supplies explicit ids; reset_sequences() bumps the sequence
-- past MAX(id) afterwards.  The default on the partitioned parent propagates to
-- every partition.
ALTER TABLE candidates ALTER COLUMN id SET DEFAULT nextval('candidates_id_seq');
ALTER SEQUENCE candidates_id_seq OWNED BY candidates.id;

-- ─────────────────────────────────────────────────────────────
-- TABLE: tasks  (NOT partitioned — see PARTITIONING STRATEGY note above)
-- ─────────────────────────────────────────────────────────────
-- Unpartitioned so that PRIMARY KEY (task_id) is a single-column global
-- unique constraint, matching SQLite semantics.  db.py addresses tasks
-- exclusively by task_id (≥9 WHERE task_id=? call sites); a compound PK
-- forced by partitioning would allow duplicate task_ids across status values
-- and silently corrupt UPDATE/SELECT operations from unmodified application code.
-- At 1.9M rows the table is small enough that a plain (status, priority) index
-- provides all the operational query performance needed.
-- sector (done-state design, docs/superpowers/specs/2026-09-26-tic-sector-
-- done-state-design.md §2.2): the LOT's TESS sector at enqueue time. NULL =
-- unattributed (enqueued outside a --sector-aware path); NULLable, no
-- rewrite cost on an already-migrated table. On the already-migrated
-- production DB this column is added by ALTER TABLE, not by re-running this
-- file (same deploy note as exominer_status/fpp_skip_reason above):
--   ALTER TABLE tasks ADD COLUMN sector BIGINT;
-- run_id (S3, docs/superpowers/specs/2026-09-26-enqueue-s1-s4-design.md
-- §3.2): the LOT's run identity at enqueue time — replaces
-- discovery_run_tics as the scoping source for a run's TIC set (see that
-- table's own comment below; NOT dropped by this migration — operator
-- decision §9.2). NULL = unattributed (enqueued outside a --run-id-aware
-- path, e.g. a manual enqueue_from_csv.py without --run-id — adoptable by
-- S4, out of scope for any run). On the already-migrated production DB this
-- column + its partial index are added by hand (CONCURRENTLY — see
-- scripts/migration/2026-09-26_tasks_run_id.sql), not by re-running this file:
--   ALTER TABLE tasks ADD COLUMN run_id TEXT;
--   CREATE INDEX CONCURRENTLY idx_tasks_run_id
--     ON tasks (run_id, status) INCLUDE (tic_id) WHERE run_id IS NOT NULL;
CREATE TABLE tasks (
    task_id          TEXT         NOT NULL,
    tic_id           TEXT,
    status           TEXT         NOT NULL DEFAULT 'pending',
    priority         BIGINT       NOT NULL DEFAULT 2,
    assigned_to      TEXT,
    assigned_at      TIMESTAMPTZ,
    completed_at     TIMESTAMPTZ,
    processing_time  DOUBLE PRECISION,
    gpu_used         BOOLEAN      NOT NULL DEFAULT FALSE,
    attempts         BIGINT       NOT NULL DEFAULT 0,
    last_error       TEXT,
    created_at       TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    lc_load_recovered BOOLEAN     NOT NULL DEFAULT FALSE,
    sector           BIGINT,
    run_id           TEXT,
    PRIMARY KEY (task_id)
);

-- Operational index (equiv to SQLite idx_tasks_status_priority)
CREATE INDEX idx_tasks_status_priority ON tasks (status, priority, created_at);
-- Fast pending-task index for worker assignment queries
CREATE INDEX idx_tasks_pending_priority ON tasks (priority, created_at)
    WHERE status = 'pending';
-- Run-scope index (S3): serves GROUP BY status per run_id, DISTINCT tic_id
-- per run_id, and EXISTS(tasks WHERE run_id=?) all index-only. Partial
-- (WHERE run_id IS NOT NULL) so it costs nothing for the (overwhelming
-- majority, pre-S3) rows that never carry a run_id.
CREATE INDEX idx_tasks_run_id ON tasks (run_id, status) INCLUDE (tic_id)
    WHERE run_id IS NOT NULL;

-- ─────────────────────────────────────────────────────────────
-- TABLE: workers_registry
-- ─────────────────────────────────────────────────────────────
CREATE TABLE workers_registry (
    worker_id              TEXT         NOT NULL,
    name                   TEXT         NOT NULL,
    first_seen             TIMESTAMPTZ,
    last_seen              TIMESTAMPTZ,
    tasks_completed        BIGINT       NOT NULL DEFAULT 0,
    tasks_failed           BIGINT       NOT NULL DEFAULT 0,
    total_processing_time  DOUBLE PRECISION NOT NULL DEFAULT 0,
    PRIMARY KEY (worker_id),
    UNIQUE (name)
);

-- ─────────────────────────────────────────────────────────────
-- TABLE: worker_telemetry  (partitioned by month)
-- ─────────────────────────────────────────────────────────────
-- ts mirrors the SQLite "timestamp" TEXT column (renamed to ts to avoid
-- clashing with the PostgreSQL reserved word TIMESTAMP).
-- At current scale (280k rows), a single partition + default is sufficient.
-- Add monthly partitions as data grows.
CREATE TABLE worker_telemetry (
    id                 BIGINT         NOT NULL,
    worker_id          TEXT           NOT NULL REFERENCES workers_registry(worker_id),
    ts                 TIMESTAMPTZ    NOT NULL,
    status             TEXT,
    current_tic        BIGINT,
    phase              TEXT,
    phase_progress     DOUBLE PRECISION,
    star_elapsed_s     DOUBLE PRECISION,
    stars_done         BIGINT,
    avg_time_per_star  DOUBLE PRECISION,
    last_error         TEXT,
    cpu_percent        DOUBLE PRECISION,
    ram_used_gb        DOUBLE PRECISION,
    ram_total_gb       DOUBLE PRECISION,
    cpu_temp_c         DOUBLE PRECISION,
    pid_rss_mb         DOUBLE PRECISION,
    recent_timings     TEXT,
    swap_used_gb       DOUBLE PRECISION,
    PRIMARY KEY (id, ts)
) PARTITION BY RANGE (ts);

-- Partition covering all data to date (2026 and earlier)
CREATE TABLE worker_telemetry_2026 PARTITION OF worker_telemetry
    FOR VALUES FROM ('2026-01-01') TO ('2027-01-01');

-- Default for future years
CREATE TABLE worker_telemetry_default PARTITION OF worker_telemetry
    DEFAULT;

-- Operational index (equiv to SQLite idx_telemetry_worker)
CREATE INDEX idx_telemetry_worker ON worker_telemetry (worker_id, id DESC);

-- Sequence for id (reset post-load)
CREATE SEQUENCE worker_telemetry_id_seq;
-- Bind the sequence as the column default so application INSERTs that omit "id"
-- (ExoNitDB.insert_telemetry) auto-generate it.  Propagates to every partition.
ALTER TABLE worker_telemetry ALTER COLUMN id SET DEFAULT nextval('worker_telemetry_id_seq');
ALTER SEQUENCE worker_telemetry_id_seq OWNED BY worker_telemetry.id;

-- ─────────────────────────────────────────────────────────────
-- TABLE: worker_debug_mode
-- ─────────────────────────────────────────────────────────────
CREATE TABLE worker_debug_mode (
    worker_id      TEXT        NOT NULL,
    debug_enabled  BOOLEAN     NOT NULL DEFAULT FALSE,
    enabled_at     TIMESTAMPTZ,
    enabled_by     TEXT,
    PRIMARY KEY (worker_id)
);

-- ─────────────────────────────────────────────────────────────
-- TABLE: completeness_agg
-- ─────────────────────────────────────────────────────────────
-- Only 72 rows: no partitioning needed.
--
-- TRIGGER MAINTENANCE GAP (Change 2, 2026-06-20) — OPEN DECISION REQUIRED:
--
-- SQLite has three triggers (trg_completeness_agg_insert, _update, _delete) that
-- keep this table in sync with the raw `completeness` table automatically.
-- Those triggers are NOT migrated to PG because the raw `completeness` table
-- is excluded from PG (going to Parquet instead).
--
-- On migration day this table is loaded from a one-time snapshot.  After
-- cutover, new discovery runs will insert rows into the raw completeness source
-- (SQLite or Parquet), but PG's completeness_agg will NOT be updated — it will
-- go stale.  The dashboard and scorecard read from completeness_agg (72 rows,
-- detection probability summaries).
--
-- REQUIRED before pointing the application at PG:
--   Option A: continue reading completeness_agg from SQLite (PG copy is a
--             read-only historical snapshot); wire the DB routing accordingly.
--   Option B: add PG triggers or an application-layer hook to write
--             completeness_agg directly whenever completeness data changes.
--
-- This is a blocking open decision for the application cutover; it is NOT
-- silently resolved by this schema.
CREATE TABLE completeness_agg (
    radius_earth  DOUBLE PRECISION NOT NULL,
    period_days   DOUBLE PRECISION NOT NULL,
    sum_prob      DOUBLE PRECISION NOT NULL,
    n             BIGINT           NOT NULL,
    PRIMARY KEY (radius_earth, period_days)
);

-- ─────────────────────────────────────────────────────────────
-- TABLE: metadata
-- ─────────────────────────────────────────────────────────────
-- Cache table for TESS target metadata (currently 0 rows in SQLite).
-- Included for schema completeness; can be omitted from initial data load.
CREATE TABLE metadata (
    tic_id          TEXT             NOT NULL,
    sectors         TEXT             NOT NULL,
    exptime         DOUBLE PRECISION NOT NULL,
    fits_files      TEXT             NOT NULL,
    stellar_radius  DOUBLE PRECISION,
    teff            DOUBLE PRECISION,
    logg            DOUBLE PRECISION,
    tmag            DOUBLE PRECISION,
    ra              DOUBLE PRECISION,
    dec             DOUBLE PRECISION,
    updated_at      TIMESTAMPTZ      NOT NULL DEFAULT NOW(),
    PRIMARY KEY (tic_id)
);

-- ─────────────────────────────────────────────────────────────
-- TABLE: fits_paths
-- ─────────────────────────────────────────────────────────────
-- Local filesystem cache index (~10k rows).
CREATE TABLE fits_paths (
    filename   TEXT NOT NULL,
    full_path  TEXT NOT NULL,
    PRIMARY KEY (filename)
);

-- ─────────────────────────────────────────────────────────────
-- TABLE: nearby_stars
-- ─────────────────────────────────────────────────────────────
-- Cache for neighbour star data (currently 0 rows in SQLite).
CREATE TABLE nearby_stars (
    tic_id      TEXT             NOT NULL,
    neighbor_id TEXT             NOT NULL,
    ra          DOUBLE PRECISION,
    dec         DOUBLE PRECISION,
    mass        DOUBLE PRECISION,
    rad         DOUBLE PRECISION,
    teff        DOUBLE PRECISION,
    tmag        DOUBLE PRECISION,
    plx         DOUBLE PRECISION,
    gal_b       DOUBLE PRECISION,
    updated_at  TIMESTAMPTZ      NOT NULL DEFAULT NOW(),
    PRIMARY KEY (tic_id, neighbor_id)
);

-- ─────────────────────────────────────────────────────────────
-- TABLE: discovery_sector_state
-- ─────────────────────────────────────────────────────────────
-- UI-controlled state for each discovery sector.
-- state values: coming_soon | active | paused | complete
-- The UI is the source of truth; this table drives sector labels,
-- auto-fill eligibility, and campaign progress reporting.
CREATE TABLE IF NOT EXISTS discovery_sector_state (
  sector     INTEGER PRIMARY KEY,
  state      TEXT NOT NULL DEFAULT 'coming_soon',
  detail     TEXT,
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ─────────────────────────────────────────────────────────────
-- TABLE: discovery_no_data
-- ─────────────────────────────────────────────────────────────
-- TICs in a sector work list for which no light-curve data exists
-- (no FITS file available anywhere after retry).  These count toward
-- "sector complete" (pct_complete) but are excluded from gaps_processable.
CREATE TABLE IF NOT EXISTS discovery_no_data (
  tic_id     TEXT NOT NULL,
  sector     INTEGER NOT NULL,
  reason     TEXT NOT NULL DEFAULT 'no_fits',
  checked_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  PRIMARY KEY (tic_id, sector)
);
CREATE INDEX IF NOT EXISTS idx_no_data_sector ON discovery_no_data(sector);

-- ─────────────────────────────────────────────────────────────
-- TABLE: tic_sector_state
-- ─────────────────────────────────────────────────────────────
-- Canonical (TIC, sector) done-state (docs/superpowers/specs/2026-09-26-
-- tic-sector-done-state-design.md §2.2). See exonit/done_state.py for the
-- read predicates (done(t)/exhausted(t,S)) this backs and the write path
-- (mark_tic_done/mark_tic_failed, called from routes/workers.py's
-- worker_result on every task result — dual-write, always on).
--
-- status: 'completed' | 'failed'. sector = 0 is reserved for legacy rows
-- (backfilled from tasks/stars with no reliable per-run sector attribution,
-- §2.3) and for any write whose task has sector IS NULL.
-- attempts: LOT-level TERMINAL failures for this (tic, sector) — NOT the
-- same counter as tasks.attempts (worker-level retries within one lot,
-- capped at db.MAX_TASK_ATTEMPTS=3). exhausted(t,S) is
-- attempts >= done_state.MAX_LOTS_PER_SECTOR (2, operator decision §7.1).
--
-- On the already-migrated production DB this table is created by hand
-- (idempotent — CREATE TABLE IF NOT EXISTS), not by re-running this whole
-- file:
--   CREATE TABLE IF NOT EXISTS tic_sector_state ( ... );  -- (this block)
--   CREATE INDEX IF NOT EXISTS idx_tss_done ...;
--   CREATE INDEX IF NOT EXISTS idx_tss_sector_failed ...;
-- then backfilled once via scripts/backfill_tic_sector_state.py (§2.3).
CREATE TABLE IF NOT EXISTS tic_sector_state (
  tic_id       TEXT    NOT NULL,
  sector       INTEGER NOT NULL,
  status       TEXT    NOT NULL,
  attempts     INTEGER NOT NULL DEFAULT 0,
  last_task_id TEXT,
  last_error   TEXT,
  updated_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
  PRIMARY KEY (tic_id, sector)
);
-- done(t) scan (load_done_tic_set / sector-scale probe fallback): index-only
-- over the (typically much smaller than the full table) completed subset.
CREATE INDEX IF NOT EXISTS idx_tss_done
    ON tic_sector_state (tic_id) WHERE status = 'completed';
-- exhausted(S) is small (~27k rows globally at the 2026-09-26 measurement) —
-- a per-sector index keeps sector_gap_stats' failed_terminal cheap even at
-- campaign scale.
CREATE INDEX IF NOT EXISTS idx_tss_sector_failed
    ON tic_sector_state (sector) WHERE status = 'failed';

-- ─────────────────────────────────────────────────────────────
-- TABLE: discovery_autofill
-- ─────────────────────────────────────────────────────────────
-- Durable state for the UI-driven per-sector auto-fill manager (Phase 2).
-- The auto-fill manager reads/writes this table each tick so it can resume
-- safely across server restarts.
--
-- status values: running | stopped | alert | done
-- current_lot:   lot-id string currently being processed (NULL when idle)
-- lots_done:     count of lots completed in this autofill run
-- anomalies:     consecutive anomaly count (empty-lot, FAILED); resets on ok tick
-- last_alert:    human-readable reason string when status='alert'
-- updated_at:    always refreshed by set_autofill() so staleness is detectable
CREATE TABLE IF NOT EXISTS discovery_autofill (
  sector       INTEGER PRIMARY KEY,
  status       TEXT,
  current_lot  TEXT,
  lots_done    INT DEFAULT 0,
  anomalies    INT DEFAULT 0,
  last_alert   TEXT,
  updated_at   TIMESTAMPTZ DEFAULT now()
);

-- ─────────────────────────────────────────────────────────────
-- TABLE: discovery_runs
-- ─────────────────────────────────────────────────────────────
-- Durable run record for the discovery orchestrator (Phase-3 durable-
-- orchestration plan, Task 1). Replaces pipeline_state.json as the source of
-- truth for in-flight run state, so a restart can resume from the last
-- checkpoint instead of losing progress.
--
-- state machine: ENQUEUEING -> SCANNING -> ... -> DONE | FAILED (terminal).
-- At most one non-terminal (state NOT IN ('DONE','FAILED')) row at a time —
-- an application-level invariant (get_active_discovery_run()), not a DB
-- constraint.
--
-- checkpoints stays TEXT (serialized JSON), not JSONB: exonit/db.py does
-- json.loads/json.dumps in Python on both backends, so the two dialects
-- share identical application-level handling instead of PG getting native
-- JSON semantics the SQLite side cannot mirror.
--
-- dry_run/chained/orchestrator_pid/checkpoints are additive columns on a
-- brand-new table (no pre-existing SQLite data to migrate), so — unlike the
-- tables above — they use PG-idiomatic BOOLEAN/BIGINT types directly rather
-- than the INTEGER 0/1 SQLite started with.
CREATE TABLE IF NOT EXISTS discovery_runs (
    run_id           TEXT         NOT NULL,
    sector           BIGINT       NOT NULL,
    lot_csv          TEXT         NOT NULL,
    state            TEXT         NOT NULL DEFAULT 'ENQUEUEING',
    detail           TEXT,
    dry_run          BOOLEAN      NOT NULL DEFAULT FALSE,
    chained          BOOLEAN      NOT NULL DEFAULT FALSE,
    server_url       TEXT,
    orchestrator_pid BIGINT,
    checkpoints      TEXT         NOT NULL DEFAULT '{}',
    started_at       TIMESTAMPTZ  NOT NULL,
    updated_at       TIMESTAMPTZ  NOT NULL,
    heartbeat_at     TIMESTAMPTZ,
    finished_at      TIMESTAMPTZ,
    PRIMARY KEY (run_id)
);

-- ─────────────────────────────────────────────────────────────
-- TABLES: completeness_staging / completeness_tics / completeness_agg_bucket
-- ─────────────────────────────────────────────────────────────
-- Phase 4a (2026-07-05): raw completeness (183.95M rows, SQLite-only) ->
-- bucketed Parquet on MinIO; staging + exact aggregates here. Gated by
-- EXONIT_COMPLETENESS_PG=1 (docs/superpowers/plans/
-- 2026-07-05-phase4-completeness-parquet.md). With the flag OFF (default) the
-- SQLite sidecar (completeness/completeness_agg, this file line ~307)
-- stays intact; these 3 tables remain empty and unused.
--
-- completeness_staging: append-only write path (POST /api/worker/result ->
-- ExoNitDB.add_completeness_many). NO on-conflict/unique — the same
-- (tic_id, radius_earth, period_days) may appear more than once between
-- compactions; last-write-wins by written_at/id is the compaction's job
-- (Task 3), not this write path's.
--
-- completeness_tics: compacted index of already-processed TICs (tic_id -> bucket
-- crc32(tic_id)%256, see exonit/completeness_store.bucket_of), for an exact
-- distinct-count without scanning Parquet (ExoNitDB.
-- count_completeness_tics, combined with any tic still only in
-- staging).
--
-- completeness_agg_bucket: aggregate per (bucket, radius_earth, period_days) —
-- ALWAYS recomputed whole per bucket at compaction (never deltas), because
-- compaction does last-write-wins by PK and fragile deltas would not
-- capture that correctly. The reader (ExoNitDB.query_completeness_agg) does
-- SUM(sum_prob)/SUM(n) grouped by cell, summing across buckets — the
-- bucket partition is a write-side detail invisible to readers.
--
-- ExoNitDB._ensure_completeness_pg_schema() creates these 3 tables
-- in code (CREATE TABLE IF NOT EXISTS) in addition to here, so that a fresh PG DB
-- (e.g. exonit_test) has them without a separate provisioning step.
CREATE TABLE IF NOT EXISTS completeness_staging (
    id             BIGSERIAL        PRIMARY KEY,
    tic_id         TEXT             NOT NULL,
    radius_earth   DOUBLE PRECISION NOT NULL,
    period_days    DOUBLE PRECISION NOT NULL,
    detection_prob DOUBLE PRECISION NOT NULL,
    written_at     TIMESTAMPTZ      NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_cmpl_staging_tic ON completeness_staging(tic_id);

CREATE TABLE IF NOT EXISTS completeness_tics (
    tic_id TEXT     PRIMARY KEY,
    bucket SMALLINT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_cmpl_tics_bucket ON completeness_tics(bucket);

CREATE TABLE IF NOT EXISTS completeness_agg_bucket (
    bucket       SMALLINT         NOT NULL,
    radius_earth DOUBLE PRECISION NOT NULL,
    period_days  DOUBLE PRECISION NOT NULL,
    sum_prob     DOUBLE PRECISION NOT NULL,
    n            BIGINT           NOT NULL,
    PRIMARY KEY (bucket, radius_earth, period_days)
);

-- ─────────────────────────────────────────────────────────────
-- TABLE: discovery_run_tics
-- ─────────────────────────────────────────────────────────────
-- The TIC set of a run's lot, materialised so the server can scope a
-- backfill counter to the run that is actually waiting on it.
--
-- Why this exists (2026-07-14): the orchestrator's phase-exit conditions read
-- /api/health/work, whose fpp/exominer counters are GLOBAL COUNT(*)s over the
-- whole candidates table. A run therefore waited on candidates belonging to
-- *other sectors and other lots*. On 2026-07-14 a 50k sector-14 run hung at
-- EXOMINER_BACKFILL on 43 leftover sector-2 candidates that can never be
-- scored (FFI/TGLC targets with no SPOC DV report): the global counter never
-- reached 0, so the exit condition was unreachable and EVERY future run would
-- have hung on the same 43 rows. Scoping the counter to the run's own lot is
-- the fix; the work STREAMS stay global on purpose (a worker may still pick up
-- another lot's eligible candidate — that is useful work, it just must not
-- gate this run's exit).
--
-- lot_csv on discovery_runs is only a path, and the file may not exist on the
-- host serving the API (AWS), so the TIC set is materialised here instead of
-- re-read from disk on every poll.
--
-- Rows are inserted once per run, at ENQUEUEING, by the orchestrator. They are
-- kept after the run ends (cheap: ~50k short rows/run) so a finished run's
-- scope stays reconstructible for post-hoc analysis.
--
-- S3 (docs/superpowers/specs/2026-09-26-enqueue-s1-s4-design.md §3): being
-- REPLACED by tasks.run_id as the scoping source (db.run_scope_kind,
-- EXONIT_RUN_SCOPE) — this table is NOT dropped by S3 itself. Operator
-- decision §9.2: drop only after (i) the done-state backfill is complete,
-- (ii) >=2 lots have run with EXONIT_RUN_SCOPE=tasks AND
-- EXONIT_RUN_TICS_WRITE=0, and (iii) a `pg_dump -t discovery_run_tics`
-- export exists. Until then it stays read-only history — EXONIT_RUN_TICS_
-- WRITE (default 1) controls whether the orchestrator keeps writing to it.
CREATE TABLE IF NOT EXISTS discovery_run_tics (
    run_id TEXT NOT NULL,
    tic_id TEXT NOT NULL,
    PRIMARY KEY (run_id, tic_id)
);

-- The health/work counters JOIN candidates/tasks -> discovery_run_tics on
-- tic_id for a single run_id, so the (run_id, tic_id) PK already serves them.
-- This second index covers the reverse direction (given a tic, which runs
-- covered it) used by post-hoc analysis.
CREATE INDEX IF NOT EXISTS idx_run_tics_tic ON discovery_run_tics (tic_id);

-- ─────────────────────────────────────────────────────────────
-- GRANT all objects to role exonit
-- ─────────────────────────────────────────────────────────────
GRANT ALL ON ALL TABLES    IN SCHEMA public TO exonit;
GRANT ALL ON ALL SEQUENCES IN SCHEMA public TO exonit;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON TABLES    TO exonit;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON SEQUENCES TO exonit;
