# Advanced: server-side NASA ExoMiner (Docker)

This page is the only place in the documentation where Docker appears. It is optional, **off by default**, and
never needed by workers. Skip it unless you specifically want NASA's published ExoMiner scores in addition to
ExoNit's own models.

## What it is

NASA publishes ExoMiner as a container image, `ghcr.io/nasa/exominer:latest`. It scores Threshold Crossing
Events (TCEs) from the SPOC pipeline's Data Validation (DV) products. ExoNit can run that container on the server
and store its score next to each candidate (`exominer_score`).

## Eligibility: SPOC only

The published model needs a SPOC DV product and a TCE **from the sector being processed**. Stars found in
full-frame-image light curves (TGLC, QLP) almost never have one, so the NASA container is structurally unable to
score the typical ExoNit discovery candidate. Having a DV run for the star in some other sector does not make it
eligible. For FFI-only stars ExoNit uses its own ONNX ensemble, ExoMiner-FFI, on the workers (see
[MODELS.md](../../MODELS.md)); that path is always on and needs no Docker.

## Why it is off by default

- In FFI discovery it is redundant: the stars it can score are a small minority, and the worker-side FFI model
  already scores everything.
- It is slow and resource-hungry (the container is limited to a few CPUs and GB of RAM by default).
- The upstream image has changed its file permissions between releases (see below), which can turn it into a noisy
  failure source.

The reference deployment therefore runs with `EXONIT_ENABLE_EXOMINER_AUTO=0` and `EXONIT_EXOMINER_BACKFILL=0`.
The verdict uses the worker-side FFI score when the NASA score is absent.

## Enabling it

Requirements: a Docker engine on the server host, access to pull the image, and a checkout of the NASA ExoMiner
repository. Then in `.env`:

```
EXONIT_ENABLE_EXOMINER_AUTO=1
EXONIT_EXOMINER_REPO=/opt/exonit-extern/ExoMiner
# EXONIT_EXOMINER_CPU_LIMIT=2
# EXONIT_EXOMINER_MEMORY_LIMIT=4g
```

and restart the server between lots. With the flag on, scored candidates get `exominer_score`; if the worker-side
ExoMiner-FFI model also produced a score, the strong-tier display gate uses the minimum of the two.
`EXONIT_EXOMINER_BACKFILL=1` additionally scores candidates after the scan; it only makes sense together with the
Docker path.

If both the worker-side ONNX model and the server-side container are active, the server-side result is a
redundant cache hit: harmless, but wasted work.

## The permission regression (`Errno 13`)

By default ExoNit runs the container as the calling user (`--user uid:gid`) so output files are owned by the server
account and can be cleaned up. Some builds of the `latest` image keep the pipeline code readable by root only;
running as a non-root user then fails immediately with `[Errno 13] Permission denied`. The workaround is
`EXONIT_EXOMINER_DOCKER_ROOT=1`, which runs the container as root with `HOME=/tmp`. Output files are then
root-owned and cleanup is best effort. Pin the image to a digest you have tested if you depend on this path.

## Related

[MODELS.md](../../MODELS.md), [running-a-campaign.md](../running-a-campaign.md) (the optional ExoMiner backfill
phase of the orchestrator).
