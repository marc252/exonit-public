# Swift worker (macOS, Apple Silicon)

A native port of the worker for Apple Silicon Macs, in `native/swift`. It implements the whole scan pipeline in Swift
with Metal kernels and Core ML models, in a single multi-purpose binary, and it plugs into the same `farmctl`
manifest and the same HTTP protocol as the Python worker.

## Why use it

Measured on a Mac mini M4, the Swift worker's CPU BLS runs about 1.7 - 1.9 times faster than Python with numba
(Float64, same grid and tie-breaking), FITS reading about 8 times faster, and the full QLP plus detrending
pipeline about 2.5 times faster. In production, three Swift instances processed more stars per minute than four
Python instances. A fleet of 5 Mac mini M4 running 30 Swift instances processes a 300k-star lot in about a day.
Detailed benchmark methods are not published yet (the English write-up of the Swift worker docs is pending).

## Requirements

- Apple Silicon Mac (M1 or newer; developed and tuned on M4), macOS with a recent Xcode or Swift toolchain
  (Swift 6).
- Core ML model packages for the CNN-FFI and ExoMiner-FFI models. They are derived from the shipped ONNX files
  (`worker/models/`) with `coremltools` (conversion deterministic, packages identified by SHA-256) and verified
  against `Models/coreml-manifest.json`, which records the hash of every source ONNX file and converted package. The
  converted packages themselves are not stored in this repository; the converter script is not part of the export
  either, so convert them on a Mac from the ONNX sources and compare the hashes in the manifest.
- Python science sidecar (`Sidecars/exonit_science_sidecar.py`): TRICERATOPS FPP and a few Python-only science
  components are reached through a small JSON bridge to a Python environment on the same Mac (the worker
  directory of the Python worker plus its venv).

## Architecture

```
ExoNitSwiftWorker      the universal binary (agent-compatible CLI + worker runtime)
ExoNitCore             library: FITS, detrending, BLS, vetting, CNN, injection, HTTP runtime
Metal                  fused BLS kernel and injection kernel
Core ML                CNN-FFI and ExoMiner-FFI inference
Sidecars/              Python bridge for Python-only components
```

The binary understands two ways of being called: directly (`--config`, `--worker-name`, `--server-url`) and as a
drop-in for the Python interpreter in the manifest, where `farmctl` calls `<binary> <worker_dir>/agent.py start|stop|status|logs`.
So switching a host to Swift is a change of the `python` field in `ops/workers.json`. Details are in
the sources under `native/swift/Sources`.

## Build and run

```bash
cd native/swift
swift build -c release                       # builds exonit-swift-worker and the helper tools
swift run -c release exonit-contract-tests   # HTTP contract tests, no server needed
```

Releases are packaged and installed with signed bundles so that a fleet can be updated atomically with rollback:

```bash
native/swift/scripts/package_release.sh --version <v> --binary <path> --cnn <dir> --exominer <dir> \
    --sidecar <file> --signing-key <key> --signer <identity> --output <dir>
native/swift/scripts/install_release.sh --bundle <dir> --root <install-root> --allowed-signers <file> --signer <identity>
native/swift/scripts/install_release.sh --rollback --root <install-root>
```

`install_release.sh` keeps `current` and `previous` symlinks under `<install-root>/releases/`.

Point the manifest at the installed binary:

```json
{ "name": "worker-mac-m4", "ssh_host": "user@<host>", "platform": "macos",
  "worker_dir": "~/exonit-worker", "python": "~/exonit-swift-worker/current/bin/exonit-swift-worker",
  "instances": 5, "instances_fpp": 9 }
```

Keep the machine awake (`caffeinate`, or disable sleep): a sleeping Mac looks like a crashed worker.

## Parity

Validated bit-exact against a fresh Python worker on the same host for BLS, detrending, vetting, CNN and injection
(offline gate: 864 of 864 candidates matched), using the golden fixtures in `native/swift/Fixtures/` (regenerate with
`scripts/generate_*_golden.py`). A float64 critical path is deliberate: float32 normalisation of TGLC light curves
flipped aliases in testing.

## Known limits

- **P2 refinement (`duration_refine`, `merge_same_period`) is not ported yet** and stays opt-in (off) in the Python
  worker so that the fleet behaves identically.
- FPP and other Python-only components run through the sidecar, so a Python environment must exist on the host.
- macOS on Apple Silicon only; no Intel Macs, no Linux, no Windows.
