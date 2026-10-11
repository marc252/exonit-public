# Running a campaign

A campaign is a sequence of **lots**: batches (typically 300,000 stars; one lot takes about a day on 30 Swift
instances across 5 Mac mini M4) of stars from one sector that the farm scans, vets and post-processes together.
This guide covers a short trial, a normal lot, the artefacts, and the optional unattended supervisor.

## 0. The short trial: 100 stars

Before anything long, run the canonical 100-star sample (`data/samples/stratified_100_seed42.csv`, stratified over
the TOI catalogue, seed 42) and check the scorecard. This validates the whole chain: server, worker, models.

**Stage the light curves first.** The sample's stars are mostly SPOC 2-minute targets. The server only serves files
that are already in the SPOC cache (`EXONIT_NFS_CACHE`) and workers never fetch from MAST on demand, so an empty cache
means every task fails with 404. Download the sample's light curves once, then index them:

```bash
python3 -c "import csv;print(*[r['tic_id'] for r in csv.DictReader(open('data/samples/stratified_100_seed42.csv'))],sep='\n')" > /tmp/sample_tics.txt
python3 scripts/predownload_tic_list.py /tmp/sample_tics.txt       # needs aria2c and lightkurve; fills EXONIT_NFS_CACHE
python3 run_nightly.py build-cache                                  # indexes the cache into the metadata database
```

If you only have TGLC light curves of one sector (see [preparing-a-sector.md](preparing-a-sector.md)), use their TOI
hosts instead: intersect `data/toi_catalog.csv` (column `TIC ID`) with the TICs you indexed, write them as a CSV with a
`tic_id` column, and use that file in place of the sample below (and `--sample-size <rows>`).

```bash
python3 scripts/enqueue_from_csv.py --csv data/samples/stratified_100_seed42.csv --server http://<server-ip>:8400
```

Then start your workers ([fleet.md](fleet.md)). When the queue drains, build the scorecard:

```bash
python3 scripts/scorecard.py --db data/exonit.db --toi data/toi_catalog.csv --out data/scorecards/trial.json \
    --run-id trial --sample-size 100 --pipeline-version <git-sha> --sample data/samples/stratified_100_seed42.csv
```

A scorecard compares what ExoNit recovered against the TOI catalogue (recall on confirmed and known planets,
false-positive promotion, per-bin breakdown). Run-to-run noise on the 100-star scorecard is about two to four
percentage points, so only differences larger than that mean anything. Regenerate a sample with
`scripts/sample_stratified.py --catalog data/toi_catalog.csv --n 100 --seed 42`; the catalogue is the public TOI table
from ExoFOP, saved as `data/toi_catalog.csv`.

## 1. Lots

A lot is a CSV with one star per line. The required column is `tic_id`; an optional `label` column is carried
along (the orchestrator's `--label` sets the default, `candidate`):

```
tic_id,label
123456789,candidate
```

`scripts/enqueue_from_csv.py` posts the stars to the server queue. Enqueue **before** starting the workers when you
want a clean zero-to-N progress bar, and enqueue with the fleet stopped if you are resetting counters.

## 2. The orchestrator

`scripts/discovery_orchestrator.py` is a state machine that drives a lot to completion. It is normally launched by
the server from the dashboard (Discovery panel) and survives the browser, but you can run it directly:

```bash
python3 scripts/discovery_orchestrator.py --lot-csv <lot.csv> --sector 6 --server-url http://<server-ip>:8400 --dry-run
```

Drop `--dry-run` to run for real. A resumable run is continued with `--resume RUN_ID`. Phases:

| Phase | What happens |
|---|---|
| `SCANNING` | workers pull scan tasks: detrend, BLS, vetting, CNN-FFI, ExoMiner-FFI, injection-recovery |
| `FPP_BACKFILL` | strong candidates get a TRICERATOPS false-positive probability (separate FPP tasks, the fleet is re-sized for it) |
| `POSTPROCESS` | systematic-family veto, final vetting, verdict, dossier, scorecard |
| `LORA` (optional, off by default) | writes natural-language explanations for strong candidates with a local LoRA-tuned language model; no external API |
| `FINALIZE` | commits the scorecard and closes the lot |

Exact state names and the optional ExoMiner-Docker phase are in the orchestrator's module docstring.

## 3. Artefacts

Per lot, under `data/scorecards/`:

- `<run_id>.json` - machine-readable scorecard (per-star results, counts).
- `<run_id>_dossier.md` - human-readable dossier: the strong candidates with their diagnostics and verdicts.
- `<run_id>_cross_sector.md` - candidates whose period and epoch recur across sectors (a review queue; a genuine
  planet should recur, an instrumental systematic usually repeats at a fixed instrumental period).

Treat every strong candidate as a lead for human review, not a result. FPP is not a hard gate; the verdict and the
dossier are advisory (see the [Confidence Ledger](../CONFIDENCE_LEDGER.md)).

## 4. Review queue and re-observation lots

The cross-sector recurrence step produces a review queue. `scripts/build_reobservation_lots.py` turns the
recurring candidates into small lots that re-process those stars in the other sectors that observed them, so a
signal can be confirmed or rejected across independent data. Run it with `--help` for its options.

## 5. The optional supervisor

`autonomy/` contains an unattended supervisor that chains lots according to an approved plan queue and reacts to
failures with scripted playbooks. It is **opt-in** and starts in shadow mode (read-only).

- Plans: copy `autonomy/plans.example.yaml` to `data/autonomy/plans.yaml`. A plan without `approved_by` is loaded
  but never executed.
- Configuration: copy `autonomy/config.example.yaml` to `data/autonomy/config.yaml`.
- Arming needs two independent keys: `armed: true` in the config **and** `EXONIT_AUTONOMY_ARMED=1` in the
  environment (the unit `ops/systemd/exonit-supervisor.service` sets the latter). With only one key it stays in
  shadow mode.
- Brakes: create `data/STOP_CHAIN` to stop it from starting new lots; create `data/autonomy/PAUSED` to stop it from
  executing anything. Delete the file to release the brake.

Read the shadow-mode logs for several lots before arming it, and keep the fleet start guard in mind: `farmctl`
refuses to start workers unless `EXONIT_FARMCTL_ALLOW_START=1` is set in that shell or unit.

## 6. Operational gotchas

- **Restart the server only between lots.** The orchestrator of a running lot lives in the server's systemd cgroup;
  restarting `exonit.service` kills it and the lot ends as failed. Wait for the lot to reach `DONE`.
- **After a server restart with a live run**, the Redis work streams can drift from the database and workers sit
  idle with a full queue. Rebuild them with `scripts/rebuild_work_streams.py --stage <phase>` (use `--dry-run`
  first).
- **Never delete tasks by hand** with `DELETE FROM tasks`; stale entries stay in the Redis streams. Use the rebuild
  script afterwards.
- **Do not edit `worker/` while a lot is running**: every worker auto-updates and restarts mid-task.
- **Closing a sector** is a database state change (`set_sector_state`), not just a registry edit.
- **A reset is not destructive**: the dashboard's "Reset stats" button only resets in-memory baselines and worker
  counters; it deletes no data. A new run otherwise inherits the previous run's baseline.
- **Measure by wall clock**: the "stars per minute" rate drops when a lot contains multi-sector stars; compare the
  cost per light-curve point, not raw stars per minute.
