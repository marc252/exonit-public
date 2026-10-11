# Contributing

Thank you for looking at ExoNit. Please read the [Confidence Ledger](CONFIDENCE_LEDGER.md) first: the project is a
personal, experimental pipeline whose scientific output is unvalidated, and the most valuable contribution is
careful scrutiny of the science (vetting thresholds, BLS settings, training sets, calibration) by people who know
TESS data.

## How syncs work

This repository is a **generated export** of an internal development repository. Consequently:

- Pull requests and issues against this repository are welcome, but accepted changes are **re-applied upstream** in
  the internal repository and then reach this one with the next export. A merged PR here may be rewritten, squashed
  or replaced by an equivalent upstream commit.
- `PUBLIC_SYNC.json` at the repository root records which internal commit this tree was exported from.
- Fixes to files that are written only for the public tree (the documents and templates in this repository's root
  and `docs/`) are applied directly.

## Running tests

Run **targeted** tests, never the whole suite blind. Some tests exercise code that can touch databases,
PostgreSQL parity checks, or fleet-control tools, and a blind run on a machine that holds real data or a real
farm can cause real actions.

```bash
python3 -m pytest tests/test_<module>.py -q           # one file
python3 -m pytest tests/worker -q                      # worker unit tests
python3 -m pytest tests/scripts -q                     # script tests
python3 -m pytest autonomy/tests -q                    # supervisor tests
```

Rules for a safe run:

- Work in a clone that has no production `.env`; unset `EXONIT_*` variables from your shell, or point
  `EXONIT_DATABASE_URL` / `EXONIT_DB_PATH` at a throwaway location.
- PostgreSQL parity tests use `EXONIT_TEST_DATABASE_URL`, truncate their tables, and refuse a database named like a
  production one. Point it only at a disposable test database, or leave it unset so they skip.
- GPU parity tests (CUDA, Metal, OpenCL) skip automatically when the hardware is absent.
- The farm-control tool refuses to start workers during tests. Do not work around that guard.

## Coding conventions

- Match the surrounding code: naming, comment density, idioms. Read a file before editing it.
- **One change per iteration.** Do not mix refactors with behaviour changes.
- **Science-path changes need evidence.** Anything that changes what is detected or how it is scored (detrending,
  BLS, vetting thresholds, models, verdict) must come with a scorecard before and after on the same sample
  (`scripts/scorecard.py`, the 100-star sample in `data/samples/`), reported honestly including regressions. The
  scorecard noise band is about 2-4 points; smaller differences are noise.
- **Reproducibility over speed.** Seeds are explicit; a change that makes results machine-dependent will be
  rejected.
- **No CPU fallback in production paths.** A GPU worker whose GPU backend fails must stop loudly, not silently
  continue on the CPU.
- **Keep 25% of the CPUs free** in any code that sizes thread pools.
- Add or update tests for what you change; regression tests for fixed bugs go in `tests/test_functional*.py`-style
  files next to the module's other tests.
- Comments and documentation are in English.
- P2 refinement (`duration_refine`, `merge_same_period`) stays opt-in (false) in the Python worker until the Swift
  and Rust ports have it; the ports are validated bit-for-bit against the Python worker, so a Python-only change to
  the numerical path breaks that parity.

## Reporting bugs

Open an issue with: what you ran, the TIC and sector if it concerns a star, the log excerpt, and the output of
`bin/farmctl status` if it concerns the fleet. For security issues follow [SECURITY.md](SECURITY.md).
