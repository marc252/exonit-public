# Rust worker (Linux and Windows, CUDA)

A native port of the worker in `native/rust`, built as a single binary (`exonit-rust-worker`) with no language
runtime. It targets machines where the Python worker is CPU-bound: it uses less CPU and RAM per star, so more
instances fit inside the 75% CPU budget, and it runs the CUDA BLS and injection kernels. Linux and Windows 11
(native) are both first-class. It uses the same HTTP protocol and `farmctl` manifest as the Python worker.

## Requirements

- Rust toolchain pinned in `native/rust/rust-toolchain.toml` (installed with `rustup`).
- Linux: a C toolchain. Windows: MSVC build tools.
- For GPU: an NVIDIA driver and CUDA toolkit (feature `cuda`). Optional backends for non-NVIDIA hardware:
  `opencl` (Intel/AMD GPUs), `sycl`, and `ispc` (a vectorised CPU path), each needing its own external toolchain.
- The ONNX models from the server's `worker/models/` and the ONNX Runtime library for your platform.

## Build

```bash
cd native/rust
cargo build --release -p exonit-worker --features cuda          # NVIDIA
cargo build --release -p exonit-worker --features "cuda opencl" # both backends in one binary
cargo test --all                                                 # unit, golden and contract tests
```

The binary is `target/release/exonit-rust-worker`. On Windows call `cargo` by its full path in non-interactive
SSH sessions (it is usually not on `PATH`), and stop any running instance before rebuilding: the linker cannot
overwrite a running executable.

## Run

Like the Swift worker, the binary also implements the `agent.py` command set, so `farmctl` drives it directly:
set the manifest's `python` field to the binary and `worker_dir` to the directory with its `config.json`.

```json
{ "name": "worker-linux-gpu", "ssh_host": "user@<host>", "platform": "linux",
  "worker_dir": "~/exonit/worker", "python": "~/exonit-rust/target/release/exonit-rust-worker",
  "instances": 8, "instances_fpp": 0 }
```

Direct invocation (for debugging): `exonit-rust-worker --config config.json --worker-name <name> --server-url <url>`.
On Linux, set `LD_LIBRARY_PATH` so the CUDA libraries are found (the `env` block of the manifest entry is the
place); on Windows, the CUDA runtime DLLs must be on `PATH`.

## GPU backends

| Feature | Hardware | Notes |
|---|---|---|
| `cuda` | NVIDIA | BLS and injection kernels, CUDA execution provider for ONNX |
| `opencl` | Intel/AMD GPUs | batch BLS and injection; selection order is CUDA, then OpenCL, then CPU |
| `sycl` | Intel iGPU | uses an external DPC++ toolchain |
| `ispc` | CPU (SIMD) | bit-identical to the autovectorised CPU path |

As with every worker, a GPU host must run a GPU backend; the CPU path exists for development and tests.

## FPP

TRICERATOPS false-positive probability is the one stage that the default Rust runtime does not run itself. A
Rust worker refuses to start with an FPP pool unless the Rust FPP port is enabled with `EXONIT_RUST_FPP=1`
(a CPU port, validated bit-exact against the Python FPP; keep it a canary until you have compared results). The
supported way to cover FPP is a Python sidecar: configure `python_sidecar` in the host's `config.json`
(`python`, `worker_dir`, optional `env`) and start the agent with `--fpp-instances M`, which makes the Rust agent
manage M Python worker instances for the FPP phase.

## Parity

Translated from the validated Swift port and checked against the same golden fixtures: BLS, detrending, vetting,
CNN and injection are **bit-exact against a fresh Python worker on the same host** (some stages agree to about
1e-15 in power and depth, with identical masks and final results). Run `cargo test --all` after any change.

## Known limits

- **P2 refinement (`duration_refine`, `merge_same_period`) is not ported yet** and stays opt-in (off) in the Python
  worker.
- FPP needs either the Python sidecar or the opt-in Rust FPP port (see above).
- Large test fixtures (over 1 MB) are not included in this repository; they are listed in `native/rust/FIXTURES.md`.
  Tests that need them are skipped or must be regenerated from the Python worker.
- Not for macOS; use the Swift worker there.
