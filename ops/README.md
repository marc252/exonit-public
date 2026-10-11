# ops/

Deployment templates. Nothing here is executed by the application itself.

| Path | What it is |
|---|---|
| `systemd/exonit.service` | The API server (`run_nightly.py serve`). Reads `EnvironmentFile=` from the checkout's `.env`. |
| `systemd/exonit-supervisor.service` | The autonomy supervisor that chains discovery lots (see `docs/running-a-campaign.md`). |
| `systemd/exonit-compactd.service` | Optional completeness-compaction HTTP service (`scripts/exonit_compactd.py`). |
| `postgres/10-exonit-tuning.conf.example` | PostgreSQL drop-in with the settings used for the measured requirements in `INSTALL.md`. |
| `workers.example.json` | Worker inventory used by `bin/farmctl` for multi-host fleets. |
| `autonomy_fleet.example.json` | Fleet map for the autonomy supervisor (copy to `ops/autonomy_fleet.json`). |

Replace every `/opt/exonit`, `User=` and path in the units with your own before installing them.
