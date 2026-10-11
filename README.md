# ExoNit

![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)
![License MIT](https://img.shields.io/badge/license-MIT-green)

**A distributed, GPU-accelerated transit-search pipeline for NASA TESS full-frame-image light curves.**

> **Experimental — personal research project (Confidence Level L0). Results are not validated by an astrophysicist.**
> Do NOT submit candidates to ExoFOP and do not make scientific claims based on this output.
> See the [Confidence Ledger](CONFIDENCE_LEDGER.md) for the explicit criteria to move up a level.

ExoNit searches TESS light curves for transiting exoplanets, sector by sector, across every star that has a
light curve. A server hands out work to a farm of worker machines; each worker runs the full per-star pipeline
on its own GPU and reports the result. The project was built with substantial AI assistance (Claude, Anthropic)
and without an astrophysicist on the team, which is exactly why the Confidence Ledger exists.

## How it works

For every star the worker runs:

1. **Light curve load and detrending** (TGLC, SPOC or QLP; iterative biweight detrending).
2. **BLS transit search** over tens of thousands of trial periods, multi-planet with hole filling and
   harmonic resolution (P/2 .. P/5).
3. **Vetting** (odd/even depth, V-shape, secondary eclipse, systematic-period veto).
4. **CNN-FFI classifier** (1D dual-view network, ONNX, under 1 ms per candidate).
5. **ExoMiner-FFI** (in-house ONNX ensemble trained on TESS FFI light curves, see [MODELS.md](MODELS.md)).
6. **TRICERATOPS false-positive probability (FPP)** for strong candidates (CPU, CUDA, OpenCL or Metal kernels).
7. **Verdict** (gradient-boosted classifier "v3 TFOPWG" on the FFI path, "v2.1" on the SPOC path), then
   **post-processing**: systematic-family veto, cross-sector recurrence, scorecard and dossier generation.

Completeness is measured by injection-recovery (synthetic transits injected into the real light curve).
Details: [ARCHITECTURE.md](ARCHITECTURE.md).

## What it takes

These are measured numbers from the reference deployment, not minimums to try the software on a handful of stars.

| Item | Measured |
|---|---|
| Operational PostgreSQL database | 61 GB for 15 sectors (about 4 GB per sector) |
| Metadata PostgreSQL database (light-curve index) | 198 GB |
| TGLC light-curve store | 74 - 923 GB per sector (sector 6: 237 GB, sector 7: 370 GB) |
| Redis | 33 MB |
| Server | 64 GB RAM recommended (`MemoryHigh=32G` in the systemd unit) |
| Throughput | one 300k-TIC lot takes about 1 day on 30 Swift worker instances across 5 Mac mini M4 |

A small test (the 100-star sample, see [docs/running-a-campaign.md](docs/running-a-campaign.md)) needs only the
server, one worker and a few GB of disk.

## Platforms, GPUs and worker ports

| Worker | OS | GPU backend | Notes |
|---|---|---|---|
| Python (reference) | Windows 11 (native), Linux, macOS | CUDA (NVIDIA), Metal via MLX (Apple Silicon), OpenCL (Intel/AMD iGPU), CPU | See [docs/workers/python.md](docs/workers/python.md) |
| Swift | macOS, Apple Silicon | Metal | Fastest on M-series Macs; see [docs/workers/swift.md](docs/workers/swift.md) |
| Rust | Linux, Windows | CUDA (OpenCL / SYCL / ISPC optional) | See [docs/workers/rust.md](docs/workers/rust.md) |

Docker is not required (and not used) for workers. WSL2 is not a supported way to run a worker on Windows.
A worker without a working GPU backend is a misconfigured worker: the project rule is that production workers
use their GPU, and CPU is only for development and tests.

## Quick start

```bash
git clone <this-repository> ~/exonit && cd ~/exonit
python3 install_server.py --update           # creates venv/ and installs the server dependencies
. venv/bin/activate
cp .env.example .env                         # fill every CHANGE_ME (PostgreSQL and Redis: see INSTALL.md)
python3 run_nightly.py serve --workers 4     # dashboard on http://<server-ip>:8400
```

Then add workers ([docs/install-workers.md](docs/install-workers.md)) and enqueue a sample
([docs/running-a-campaign.md](docs/running-a-campaign.md)). The full server setup is in [INSTALL.md](INSTALL.md).

## Documentation

- [INSTALL.md](INSTALL.md) - server installation (PostgreSQL, Redis, systemd).
- [docs/install-workers.md](docs/install-workers.md) - worker installation per platform.
- [docs/preparing-a-sector.md](docs/preparing-a-sector.md) - downloading and indexing a sector.
- [docs/running-a-campaign.md](docs/running-a-campaign.md) - lots, orchestrator, scorecards, supervisor.
- [docs/fleet.md](docs/fleet.md) - managing a farm of workers with `farmctl`.
- [ARCHITECTURE.md](ARCHITECTURE.md), [MODELS.md](MODELS.md), [CONFIDENCE_LEDGER.md](CONFIDENCE_LEDGER.md),
  [CONTRIBUTING.md](CONTRIBUTING.md), [SECURITY.md](SECURITY.md).

## Regression case

TIC 4206066 (a reported 3.18-day small-planet candidate) is kept as a recall regression case. Its ephemeris comes
from Rabtsevich 2026, doi:[10.5281/zenodo.22967456](https://doi.org/10.5281/zenodo.22967456), licensed CC BY 4.0.

## License

MIT, for the code and for the model weights (see [MODELS.md](MODELS.md)).

## Acknowledgements

NASA TESS and the MAST archive; the TESS-Gaia Light Curves (TGLC) project; lightkurve; wotan; TRICERATOPS;
NASA ExoMiner (architecture reference for the in-house FFI ensemble); the TESS Follow-up Observing Program
Working Group (TFOPWG) dispositions used as training labels.
