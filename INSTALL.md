# Installing the ExoNit server

The server hosts the dashboard and the worker API, hands out tasks through Redis work streams, and stores
results in PostgreSQL. Linux is the recommended platform (the reference deployment runs on Ubuntu with systemd).
Workers are installed separately: see [docs/install-workers.md](docs/install-workers.md).

Read the measured requirements in the [README](README.md#what-it-takes) first. A full all-sky campaign needs
terabytes of disk; a 100-star trial needs very little.

## 1. Prerequisites

| Component | Version | Used for |
|---|---|---|
| Python | 3.12 (3.10+ is the code floor) | server, scripts |
| PostgreSQL | 16 or newer | operational database and metadata database |
| Redis | 7 or newer | shared state across server processes, work streams |
| MinIO (optional) or local disk | at least 1 TB for a few sectors | FITS light curves |
| Linux with systemd | any current LTS | service management |

Install them with your distribution's package manager. Nothing here needs Docker.

## 2. Get the code and the Python environment

```bash
git clone <this-repository> /opt/exonit
cd /opt/exonit
python3 install_server.py --update
```

`install_server.py --update` creates `venv/` and installs `requirements.txt` without touching configuration.
Running it without `--update` (or with `--unattended --port 8400 --fits-dir <dir> --worker-token <token>`)
additionally writes a minimal `.env` for a single-machine trial. For the production layout below use the
`.env.example` file instead and adjust the `ExecStart=` lines of the systemd units to the interpreter you use
(`/opt/exonit/venv/bin/python` if you used the installer).

## 3. PostgreSQL

Create a role and the two databases, then apply the schemas:

```bash
sudo -u postgres psql -c "CREATE ROLE exonit LOGIN PASSWORD '<choose-a-password>'"
sudo -u postgres createdb -O exonit exonit
sudo -u postgres createdb -O exonit exonit_meta
sudo -u postgres psql -d exonit -c "CREATE EXTENSION IF NOT EXISTS pg_stat_statements"
psql postgresql://exonit@localhost/exonit      -f scripts/migration/schema_pg.sql
psql postgresql://exonit@localhost/exonit_meta -f scripts/migration/schema_pg_meta.sql
```

`exonit` is the operational database (tasks, candidates, stars, verdicts, telemetry; 61 GB at 15 sectors).
`exonit_meta` is the light-curve metadata index (198 GB at 15 sectors) and can live on a different host.

Tuning: copy `ops/postgres/10-exonit-tuning.conf.example` to PostgreSQL's `conf.d` directory (for example
`/etc/postgresql/<version>/main/conf.d/10-exonit-tuning.conf`) and restart PostgreSQL. The values are a starting
point sized for a 32 GB host; measure with `pg_stat_statements` before changing them further.

## 4. Redis

Any default Redis 7 works (the measured footprint is 33 MB). Bind it to localhost or to a private interface
only. The server uses it for cross-process shared state and for the task work streams.

## 5. Configuration (`.env`)

```bash
cp .env.example .env
chmod 600 .env
```

Edit `.env` and replace every `CHANGE_ME` and `<placeholder>`. The variables you must set:

| Variable | Meaning |
|---|---|
| `EXONIT_DATABASE_URL` | operational PostgreSQL DSN |
| `EXONIT_METADATA_DB_URL` | metadata PostgreSQL DSN |
| `EXONIT_REDIS_URL` | Redis URL |
| `EXONIT_WORKER_TOKEN` | shared secret workers present to the server; generate a long random string |
| `EXONIT_SERVER_URL` | URL of the server as seen by workers, for example `http://<server-ip>:8400` |
| `EXONIT_NFS_CACHE` (alias `EXONIT_TESS_CACHE_DIR`) | where the SPOC FITS cache lives; must point at an existing local directory on a single machine |

The verdict models shipped in this repository are the v2.1 SPOC-path classifier and the v3 TFOPWG FFI-path
classifier (the older v1 model is not shipped). The public server must therefore run with:

```
EXONIT_VERDICT_VERSION=v2.1
EXONIT_VERDICT_FFI_VERSION=v3_tfopwg
```

Both are already set in `.env.example`. The remaining "policy" values in that file are the settings of the
reference deployment; every optional variable is commented with its meaning.

The TGLC light-curve store has no environment variable: its location is the `--dest` directory you give to
`scripts/discovery_download.py` (see [docs/preparing-a-sector.md](docs/preparing-a-sector.md)); the metadata index
records the absolute path of each file.

## 6. Start the server

For a first test, run it in the foreground:

```bash
. venv/bin/activate
python3 run_nightly.py serve --workers 4
```

For a permanent install, use the systemd units in `ops/systemd/`:

```bash
sudo useradd --system --create-home --home-dir /opt/exonit exonit     # once; or chown the checkout
sudo cp ops/systemd/exonit.service /etc/systemd/system/
sudo systemctl daemon-reload && sudo systemctl enable --now exonit.service
```

`exonit.service` runs `run_nightly.py serve --workers 4` with `EnvironmentFile=/opt/exonit/.env` and a soft
memory ceiling (`MemoryHigh=32G`, sized for a 64 GB host; scale it with your RAM).

### What each service is for

- **`exonit.service`** is the server: dashboard, worker API (`/api/worker/...`), task hand-out, the discovery
  endpoints that launch the orchestrator, and result ingestion. The orchestrator of a running lot lives in its
  cgroup, so do not restart it while a lot is running (see [docs/running-a-campaign.md](docs/running-a-campaign.md)).
- **`exonit-supervisor.service`** (optional, do not install before reading the campaign guide) chains lots
  automatically and handles failures. It starts in shadow mode (read-only) unless two independent keys arm it.
- **`exonit-compactd.service`** (optional) is a small HTTP sidecar that performs the heavy compaction of the
  completeness store on a host with spare RAM, so the main server never has to. It needs its own token in
  `/etc/default/exonit-compactd` (root-owned, mode 0600).

## 7. Health check and first login

```bash
curl http://<server-ip>:8400/api/status
```

A JSON document means the server is up. Open `http://<server-ip>:8400/` in a browser for the dashboard. Worker
and admin endpoints require the bearer token (`EXONIT_WORKER_TOKEN`). The dashboard sends the value stored in the
browser's `localStorage` key `exonit_token` (format `Bearer <token>`) and falls back to the built-in default token
when the key is absent, so after changing the token open the browser console once and run
`localStorage.exonit_token = 'Bearer <your token>'`. Do not expose port 8400 to the internet (see [SECURITY.md](SECURITY.md)).

## 8. Next steps

1. Install one worker: [docs/install-workers.md](docs/install-workers.md).
2. Prepare a sector: [docs/preparing-a-sector.md](docs/preparing-a-sector.md).
3. Run the 100-star trial or a first lot: [docs/running-a-campaign.md](docs/running-a-campaign.md).
