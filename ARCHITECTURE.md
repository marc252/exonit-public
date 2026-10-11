# Architecture

A map of where each mechanism lives, so you can find the code for a part of the system without reading the whole
repository. Anchors are module and function names (line numbers drift). If this document and the code disagree,
the code wins.

```
exonit/      server, database layer, scoring helpers (FastAPI app, routes, stores)
worker/      the Python worker: pipeline, GPU backends, models, installer, agent
www/         dashboard (static HTML and ES modules)
autonomy/    optional unattended supervisor
scripts/     campaign tooling: download, enqueue, orchestrate, post-process, scorecard
bin/farmctl  fleet control over SSH
ops/         systemd units, PostgreSQL tuning example, worker manifest example
native/      Swift and Rust worker ports (separate build systems)
```

## Server (`exonit/server.py`, `exonit/routes/`)

A FastAPI application served by uvicorn in several processes (`run_nightly.py serve --workers N`). Cross-process
state (counters, caches, locks, baselines) goes through `exonit/shared_state.py`, which uses Redis in production
and an in-memory backend in tests and single-process development.

- `serve_dashboard` returns `www/index.html` fresh from disk on every request, so a UI change needs only a browser
  refresh, not a restart.
- `exonit/routes/workers.py`: the worker API: register, heartbeat, task claim (`/api/worker/task`,
  `/api/worker/fpp_task`, `/api/worker/exominer_task`), result posting, data serving (`/api/fits/...`,
  `/api/metadata/...`, `/api/ffi/...`, `/api/nearby/...`), and the update endpoints.
- `exonit/routes/discovery.py`: the Discovery panel backend: sector registry, scan and download launch, the
  orchestrator spawn and its liveness checks.
- `exonit/routes/admin.py`, `candidates.py`, `progress.py`, `sky.py`: statistics, candidate views and exports,
  per-phase progress, sky maps; `proxies.py` forwards a few calls to optional sidecars.
- Authentication is a bearer token (`EXONIT_WORKER_TOKEN`) on the worker and admin endpoints.

### Work streams (Redis)

Tasks are handed out through Redis streams (`EXONIT_WORK_STREAMS=1`) so many workers can claim concurrently
without polling SQL. PostgreSQL remains the system of record; the streams are a delivery layer built from it.
They can drift from the database after a server restart during a live run, or after manual row deletion.
`scripts/rebuild_work_streams.py` rebuilds them from the database. Orphaned claims are released after a timeout.

## Database layer (`exonit/db_backend.py`, `exonit/db.py`)

`db_backend.py` abstracts SQLite and PostgreSQL: it translates `?` placeholders to `%s`, SQLite date functions to
their PostgreSQL equivalents, and provides pooled connections and `FOR UPDATE SKIP LOCKED` for atomic task claims.
`db.py` (`ExoNitDB`) is the operational API: stars, candidates, tasks, statistics. SQLite is for development and
tests; production uses PostgreSQL (`EXONIT_DATABASE_URL`).

`exonit/done_state.py` tracks completion per (TIC, sector) in `tic_sector_state`, so "done" has a single definition
shared by the lot selector, the enqueuer and the progress bars.

## Data stores

| What | Where | Notes |
|---|---|---|
| Operational data (stars, candidates, tasks, telemetry) | PostgreSQL `exonit` | `candidates` is partitioned by sector; schema in `scripts/migration/schema_pg.sql` |
| Completeness (injection-recovery results) | PostgreSQL, compacted by `exonit-compactd` | the largest table by row count |
| Light-curve index (`tglc`) and FITS byte ranges (`fits_blobs`) | PostgreSQL `exonit_meta` | schema in `schema_pg_meta.sql`; accessed via `exonit/metadata_cache.py` |
| FITS light curves | local files or S3-compatible store | `exonit/object_store.py`; `exonit/fits_pack.py` packs many FITS into tar containers with a byte-range index |
| Sector registry | `data/discovery/sectors.json` | active sector, per-sector downloaded count and cursor (`exonit/discovery_state.py`) |
| Run state | `data/discovery/pipeline_state.json` | the orchestrator's state, fresh per run |
| Scorecards and dossiers | `data/scorecards/` | committed artefacts, one set per lot |

## Worker (`worker/worker.py`)

One process per instance; many instances per host. Per task it runs the pipeline:

1. Fetch metadata and the light curve from the server; read SPOC, QLP or TGLC (`cal_aper_flux` plus quality flags
   for TGLC).
2. Detrend (wotan biweight, iterative), then BLS with hole-filling for multiple planets and a harmonic resolver
   (`bls.py` plus GPU variants `bls_cuda.py`, `bls_opencl.py`, `bls_metal.py`).
3. Vetting (`vetting.py`): odd/even, V-shape, secondary eclipse, known systematic periods.
4. Classification: CNN-FFI (`cnn_ffi.py`, ONNX) and ExoMiner-FFI (`exominer_ffi_inference.py`, ONNX ensemble with
   a calibration curve) are always run for each candidate with a loaded light curve.
5. Injection-recovery (`injection.py`) to measure completeness for that star.
6. A separate task type, **FPP**, runs TRICERATOPS (`triceratops_fpp.py`, with CPU, CUDA, OpenCL and Metal
   kernels in `fpp_*.py`).
7. The result is posted to the server; the server computes the verdict.

Other mechanisms in the worker:

- `agent.py` is the per-host controller `farmctl` calls over SSH (`start`, `stop`, `status`, `logs`): instance
  naming, PID files, config derivation per instance, CPU budgeting.
- **Auto-update** (`check_for_updates`): at start-up and when idle the worker compares sha256 hashes against the
  server's manifest (`/api/worker/update`), downloads what changed, and re-executes itself.
- **CUDA self-restart**: three CUDA failures in a row trigger `os.execv` to get a clean context.
- GPU backend selection is by availability and the `EXONIT_*` backend variables; see `worker/onnx_providers.py`
  and the per-backend modules.

## Verdict and post-processing

The verdict classifier (`models/verdict_classifier_*.pkl`, gradient-boosted trees) turns the per-candidate
features (SNR, vetting results, CNN and ExoMiner-FFI scores, FPP, stellar parameters) into a verdict and a tier
(strong / weak / reject). Two models are shipped: **v2.1** for stars with SPOC data and **v3 TFOPWG** for the FFI
path. Configuration: `EXONIT_VERDICT_VERSION`, `EXONIT_VERDICT_FFI_VERSION`. Post-processing
(`scripts/discovery_postprocess.py`, `discovery_finalize.py`, `apply_systematic_veto.py`) applies the
systematic-family veto, cross-sector checks, writes the scorecard and the dossier, and commits the artefacts.

## Orchestrator (`scripts/discovery_orchestrator.py`)

A state machine, one run per lot: scan, FPP backfill, post-process, optional explanation step, finalize. It polls
the server for drain status, re-sizes the fleet per phase (`bin/farmctl start --phase`), and has watchdogs so it
cannot hang forever on a dead fleet. It is spawned by the server and survives dashboard sessions.

## Supervisor (`autonomy/`)

Optional unattended chaining of lots from an approved plan queue (`plans.py`), with health checks
(`fleet_health.py`), scripted playbooks (`playbooks.py`), alerting, and a shadow mode that observes without acting.
See [docs/running-a-campaign.md](docs/running-a-campaign.md).

## Models

Weights live in `worker/models/` (shipped to workers through the update endpoint) and `models/` (server side).
The full list with training data and licences is in [MODELS.md](MODELS.md).

## Native ports (`native/swift`, `native/rust`)

Independent implementations of the worker in Swift (Apple Silicon, Metal) and Rust (CUDA, Linux and Windows).
They speak the same HTTP protocol and are validated against golden fixtures produced by the Python worker, so the
server cannot tell them apart. See [docs/workers/swift.md](docs/workers/swift.md) and
[docs/workers/rust.md](docs/workers/rust.md).

## Tests

`tests/` holds the unit and functional tests. Tests that touch PostgreSQL run against a dedicated test database
(`EXONIT_TEST_DATABASE_URL`) and refuse to run against a database named like a production one. See
[CONTRIBUTING.md](CONTRIBUTING.md) for how to run them safely.
