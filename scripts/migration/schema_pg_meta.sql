-- ============================================================
-- ExoNit PostgreSQL 18 schema — TGLC metadata index (exonit_meta DB)
-- Target: <postgres-host>:5432, database "exonit_meta", role "exonit"
-- Source: SQLite data/metadata_cache.db (tables tglc, fits_blobs, metadata,
--         fits_paths, nearby_stars)
-- Generated: 2026-07-20
-- Spec: docs/superpowers/specs/2026-07-20-tglc-index-to-pg.md (D1/D4)
-- ============================================================
--
-- WHY A SEPARATE DATABASE (D1)
-- ─────────────────────────────
-- exonit_meta is a SEPARATE PostgreSQL database from the operational
-- "exonit" DB (scripts/migration/schema_pg.sql), same instance/role:
--   (a) the index is regenerable (from bulk scripts / MinIO tars) — it must
--       be excludable from the operational DB's backup policy;
--   (b) no SQL joins between the index and operational tables exist today
--       BY CONSTRUCTION (they were always separate SQLite files), so nothing
--       is lost by keeping them separate PG databases;
--   (c) vacuum/stats/bloat isolation from the (already 16+ GB) operational
--       DB — tglc alone is ~27.7M+ rows and growing every sector.
--
-- TYPE MAPPING DECISIONS (SQLite → PostgreSQL) — same convention as
-- schema_pg.sql: TEXT→TEXT, REAL→DOUBLE PRECISION, INTEGER(pk)→BIGINT,
-- TEXT(datetime)→TIMESTAMPTZ. tglc/fits_blobs have no boolean columns (the
-- discovery report confirmed this — see item 7 of its "Risks & open
-- questions" section), so no 0/1→BOOLEAN conversion is needed for them.
--
-- PARTITIONING DECISION: tglc IS partitioned, PARTITION BY LIST (sector)
-- ────────────────────────────────────────────────────────────────────
-- The task briefing authorized the implementer to decide (F1) between
-- partitioning tglc or using a flat table with the same indexes, noting
-- either is acceptable if documented. DECISION: partition, because:
--   (1) It mirrors the established convention for exactly this kind of
--       table (candidates: PARTITION BY LIST (sector) in schema_pg.sql).
--   (2) The DOMINANT access pattern across every consumer identified in the
--       discovery report (§3/§5) is "WHERE sector = ?" — enqueue's
--       load_extended_canonical, server.py's _tglc_sector_rowcount,
--       discovery_download.py's _indexed_basenames, index_tglc_batch.py's
--       skip-set (once fixed per D5) — partition pruning turns every one of
--       these into a single-partition scan instead of an index range scan
--       over a table that is tens of millions of rows and growing every
--       sector.
--   (3) ensure_tglc_partition(sector) (exonit/metadata_cache.py) is a real,
--       useful operation under this design (create the sector's partition
--       before the indexer's first bulk insert) — it would be a permanent
--       no-op under the flat-table alternative, which is a signal the
--       partitioned design better matches the spec's own D3 helper set.
-- Unlike candidates (which only partitions the 3 sectors with real data
-- volume + a DEFAULT catch-all), tglc gets ONE partition PER SECTOR — every
-- sector the campaign touches carries a large, roughly comparable row count
-- (there is no small/negligible-sector case here the way there was for
-- candidates' sector 2), so a DEFAULT-only fallback would just accumulate
-- most of the data in the unpartitioned default and defeat the purpose.
-- Sector count stays in the tens over the campaign's lifetime (comparable to
-- worker_telemetry's monthly partitions) — not a partition-count concern.
-- A DEFAULT partition is still included as a safety net for any sector
-- inserted before ensure_tglc_partition() runs (F3), or a sector this file
-- doesn't yet enumerate.
--
-- `seq` — ordering column (no PG "rowid")
-- ────────────────────────────────────────
-- MetadataCache.tglc_tic_ids() (exonit/metadata_cache.py) needs a stable,
-- insertion-order column on PG (SQLite's implicit `rowid` has no PG
-- equivalent) — discovery_state.load_extended_canonical's "extension TICs
-- appended in roughly the order they were indexed" semantics depend on it.
-- `seq` is a plain BIGINT column, NOT a native `GENERATED ALWAYS AS
-- IDENTITY` — PG's per-partition identity sequences are NOT guaranteed to
-- share one counter across LIST partitions the way a single manually-bound
-- sequence is (the exact gotcha schema_pg.sql's `candidates.id` comment
-- documents: "The default on the partitioned parent propagates to every
-- partition" — achieved there via CREATE SEQUENCE + ALTER TABLE ... SET
-- DEFAULT, not IDENTITY). This file follows the identical pattern.
-- ============================================================

-- Ensure we're in the right database (checked by caller, not enforced in SQL)

-- ─────────────────────────────────────────────────────────────
-- TABLE: tglc  (partitioned by sector)
-- ─────────────────────────────────────────────────────────────
CREATE TABLE tglc (
    tic_id  TEXT             NOT NULL,
    sector  BIGINT           NOT NULL,
    path    TEXT             NOT NULL,
    exptime DOUBLE PRECISION,
    tmag    DOUBLE PRECISION,
    ra      DOUBLE PRECISION,
    dec     DOUBLE PRECISION,
    seq     BIGINT           NOT NULL,
    PRIMARY KEY (tic_id, sector)
) PARTITION BY LIST (sector);

-- Sequence for `seq` (insertion-order surrogate — see the design note above).
-- ONE sequence shared across every partition, bound as the column default on
-- the partitioned parent (propagates to every partition it has today AND
-- every partition created after this ALTER — see the CREATE TABLE ...
-- PARTITION OF statements below and exonit/metadata_cache.py's
-- ensure_tglc_partition, which creates FUTURE per-sector partitions the same
-- way, inheriting this default automatically).
CREATE SEQUENCE tglc_seq_seq;
ALTER TABLE tglc ALTER COLUMN seq SET DEFAULT nextval('tglc_seq_seq');
ALTER SEQUENCE tglc_seq_seq OWNED BY tglc.seq;

-- Per-sector partitions. This list is NOT exhaustive of every sector the
-- live campaign has touched (F1 does not migrate data — that's F2); it is a
-- reasonable starting set matching the sectors named in the spec's
-- "Problem" section (the in-flight s7 download + the queued hold-out
-- sectors) plus sector 1 (the original discovery sector). F2's migration
-- tooling MUST call ensure_tglc_partition(sector) (or run the equivalent
-- CREATE TABLE ... PARTITION OF) for every sector actually present in the
-- SQLite source before COPYing rows for it — do not assume this list is
-- complete.
CREATE TABLE tglc_s0001 PARTITION OF tglc FOR VALUES IN (1);
CREATE TABLE tglc_s0002 PARTITION OF tglc FOR VALUES IN (2);
CREATE TABLE tglc_s0005 PARTITION OF tglc FOR VALUES IN (5);
CREATE TABLE tglc_s0006 PARTITION OF tglc FOR VALUES IN (6);
CREATE TABLE tglc_s0007 PARTITION OF tglc FOR VALUES IN (7);
CREATE TABLE tglc_s0011 PARTITION OF tglc FOR VALUES IN (11);
CREATE TABLE tglc_s0014 PARTITION OF tglc FOR VALUES IN (14);
CREATE TABLE tglc_s0015 PARTITION OF tglc FOR VALUES IN (15);
CREATE TABLE tglc_s0017 PARTITION OF tglc FOR VALUES IN (17);
CREATE TABLE tglc_s0018 PARTITION OF tglc FOR VALUES IN (18);
CREATE TABLE tglc_s0023 PARTITION OF tglc FOR VALUES IN (23);
CREATE TABLE tglc_s0026 PARTITION OF tglc FOR VALUES IN (26);
CREATE TABLE tglc_s0036 PARTITION OF tglc FOR VALUES IN (36);
CREATE TABLE tglc_s0045 PARTITION OF tglc FOR VALUES IN (45);

-- Safety net for any sector not enumerated above (or inserted before
-- ensure_tglc_partition() runs for it).
CREATE TABLE tglc_default PARTITION OF tglc DEFAULT;

-- ix_tglc_tic/ix_tglc_sector mirror the SQLite indexes exactly (names kept
-- identical for cross-reference with the discovery report / spec).
-- ix_tglc_path is NEW (D5 of the spec): the chunk-scoped skip-set query
-- (index_tglc_batch.py's `WHERE path = ANY(%s)`, F3) needs it — the old
-- unscoped `SELECT path FROM tglc` full-table read never used an index on
-- `path` at all, so this index did not exist on the SQLite side either.
CREATE INDEX ix_tglc_tic    ON tglc (tic_id);
CREATE INDEX ix_tglc_sector ON tglc (sector);
CREATE INDEX ix_tglc_path   ON tglc (path);
-- ix_tglc_seq (S1, docs/superpowers/specs/2026-09-26-enqueue-s1-s4-
-- design.md §4.2): serves exonit.metadata_cache.MetadataCache.
-- tglc_tic_ids_page's "WHERE sector=? AND seq>? ORDER BY seq LIMIT ?" paged
-- selector query index-only (partition pruning on sector + an index range
-- scan on seq, instead of the Parallel Seq Scan + Sort the spec's §1
-- evidence measured without this index). A plain CREATE INDEX on the
-- partitioned parent (not ON ONLY) both builds it on every EXISTING
-- partition and makes it a "partitioned index" that any FUTURE partition
-- (ensure_tglc_partition) automatically inherits at creation time — see
-- scripts/migration/2026-09-26_tglc_seq_index.sql for the CONCURRENTLY,
-- per-partition equivalent used against the already-populated production
-- table (a plain CREATE INDEX here is fine only because schema_pg_meta.sql
-- is applied to an EMPTY database).
CREATE INDEX ix_tglc_seq    ON tglc (seq);

-- ─────────────────────────────────────────────────────────────
-- TABLE: fits_blobs
-- ─────────────────────────────────────────────────────────────
-- Not partitioned: PK is a filename (no natural sector-partition key —
-- filenames span containers packed across a whole download run, not
-- necessarily one sector), and it is looked up almost exclusively by exact
-- filename (the /api/fits hot path — routes/proxies.py::serve_fits), for
-- which a plain PK index is already optimal; the one sector-scoped query
-- against it (_indexed_basenames' `WHERE filename LIKE '%-sNNNN-%'`) is a
-- bounded-frequency full scan on SQLite too (see MetadataCache.
-- indexed_basenames's docstring) — partitioning would not speed that up
-- without a LIKE-friendly partition key, which sector-in-filename is not.
CREATE TABLE fits_blobs (
    filename      TEXT   PRIMARY KEY,
    container_key TEXT   NOT NULL,
    data_offset   BIGINT NOT NULL,
    data_length   BIGINT NOT NULL
);

-- ─────────────────────────────────────────────────────────────
-- TABLE: metadata
-- ─────────────────────────────────────────────────────────────
-- Same DDL as schema_pg.sql's placeholder metadata table (the exonit DB's
-- copy stays orphaned per D1 of the spec — this is the real migration
-- target), updated_at -> TIMESTAMPTZ DEFAULT now() per that file's
-- established convention.
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
CREATE TABLE fits_paths (
    filename   TEXT NOT NULL,
    full_path  TEXT NOT NULL,
    PRIMARY KEY (filename)
);

-- ─────────────────────────────────────────────────────────────
-- TABLE: nearby_stars
-- ─────────────────────────────────────────────────────────────
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
-- GRANT all objects to role exonit
-- ─────────────────────────────────────────────────────────────
-- This file is applied as a role with CREATE privilege on exonit_meta (e.g.
-- postgres) — see commit 5b258c54 ("the app user is not the owner of the
-- table in PG"): the app's `exonit` role does NOT necessarily own tables it
-- didn't CREATE itself, and PG requires ownership for ALTER TABLE even as a
-- no-op. GRANT ALL (not ALTER ... OWNER TO) is deliberate here, matching
-- schema_pg.sql's existing convention exactly — it gives `exonit` full
-- read/write/DDL-via-privilege without requiring it to be the table owner.
GRANT ALL ON ALL TABLES    IN SCHEMA public TO exonit;
GRANT ALL ON ALL SEQUENCES IN SCHEMA public TO exonit;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON TABLES    TO exonit;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON SEQUENCES TO exonit;
