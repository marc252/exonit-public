# Preparing a sector

ExoNit processes TESS data one sector at a time. The main data source for the all-sky search is TGLC (TESS-Gaia
Light Curves, a high-level science product on MAST that covers every star in the full-frame images with a Gaia
cross-match). Before a lot can run, the sector's light curves must be on disk (or in the object store) and indexed
so the server can resolve TIC to file.

## What a sector costs

| Item | Measured |
|---|---|
| TGLC light-curve store | 74 - 923 GB per sector (sector 6: 237 GB, sector 7: 370 GB); about 0.3 MB per light curve |
| Operational PostgreSQL DB | about 4 GB per sector (61 GB at 15 sectors) |
| Metadata PostgreSQL DB | 198 GB at 15 sectors |
| Download time | limited by your link to MAST; the downloader is resumable and polite (32 parallel connections by default) |

Plan for the largest sectors, not the average. A sector holds from several hundred thousand to over 2 million
stars; the target list for sector 1 alone is about 2.46 million TICs.

## Step 1 - download

The high-level downloader fetches the sector's MAST bulk script, picks the next `--count` light curves that are not
yet on disk, downloads them, indexes them, and updates the sector registry (`data/discovery/sectors.json`).
Use `--dry-run` first to see what it would do.

```bash
python3 scripts/discovery_download.py --sector 6 --count 100000 --dry-run
python3 scripts/discovery_download.py --sector 6 --count 100000 --dest <tglc-store>
```

Output is one JSON line on stdout; the subprocess logs go to `<dest>/download_<timestamp>.log`. It is idempotent:
re-running skips files already present. Repeat until the sector is complete.

The low-level building blocks, if you want to drive them yourself:

```bash
python3 scripts/download_tglc_batch.py --script <mast-bulk-script.sh> --dest <tglc-store> --log <log-file>
python3 scripts/index_tglc_batch.py --cache <tglc-store> --index <index-db>
```

`download_tglc_batch.py` reads a MAST bulk `curl` script (one `curl ... --output PATH URL` line per light curve),
retries transient failures, and names each unrecoverable file in a `[failed]` log line.
`index_tglc_batch.py` reads the FITS headers (TGLC files are named by Gaia DR3 id; the TIC is in the header) and
writes the TIC-to-file index. It is incremental. With `EXONIT_METADATA_DB_URL` set to a PostgreSQL DSN the index is
written to the metadata database; without it a SQLite file is used.

## Step 2 - object store or local disk

Two layouts are supported:

- **Local files.** The store location is the `--dest <tglc-store>` directory you pass to the download scripts (there is no
  environment variable for it); the index stores the absolute path of every file, so the server finds the light
  curves wherever you put them. Simple, and enough for a single machine.
- **MinIO / S3.** Light curves are packed into tar containers (`containers/s<NNNN>/dl_<timestamp>.tar`) in the
  bucket `EXONIT_S3_FITS_BUCKET`, byte ranges are recorded in the metadata database, and the local copies are deleted
  after upload. Set `EXONIT_S3_ENDPOINT`, `EXONIT_S3_ACCESS_KEY`, `EXONIT_S3_SECRET_KEY` (note: `_SECRET_KEY`).
  Workers fetch through the server (`/api/fits/...`), so they never need S3 credentials.

## Step 3 - refresh the metadata of a lot (optional preview)

Per-star metadata (Tmag, coordinates, stellar parameters) is filled lazily when a lot is enqueued
(`EXONIT_METADATA_REFRESH=1`). To preview or backfill by hand:

```bash
python3 scripts/refresh_metadata_bulk.py --csv <lot.csv>                    # dry run, writes nothing
python3 scripts/refresh_metadata_bulk.py --csv <lot.csv> --refresh --apply  # write
```

Even the dry run opens a real connection to the metadata database, so point it at the right one.

## Step 4 - SPOC bulk index (for the SPOC gate)

Some stars have an official SPOC light curve and Data Validation (DV) product; the pipeline checks this to decide
which verdict model applies. To avoid live MAST calls during a run (`EXONIT_SPOC_GATE_OFFLINE=1`), prefetch the
static bulk indexes once:

```bash
python3 scripts/prefetch_spoc_bulk.py --all-published
```

ExoMiner eligibility note: having a DV run for a TIC in *some* sector does not make it eligible for the published
NASA ExoMiner in *this* sector. Eligibility needs a DV product and a TCE from the sector being processed. Stars
discovered in FFI data are normally ineligible, which is why ExoNit uses its own FFI ensemble for them
([MODELS.md](../MODELS.md)).

## Time estimates

| Step | Order of magnitude |
|---|---|
| Download a 100k-star batch | hours (network bound) |
| Index a batch | minutes (header reads, parallel) |
| Full sector download | days for the largest sectors |
| SPOC bulk prefetch | minutes per sector |

When the sector is downloaded and indexed, continue with [running-a-campaign.md](running-a-campaign.md).
