# Python worker (reference implementation)

The Python worker in `worker/` is the reference implementation of the ExoNit pipeline. The Swift and Rust ports
are validated against it, and it is the only worker that runs every task type on every platform. Start here unless
you have a specific reason to use a native port ([install-workers.md](../install-workers.md#when-to-prefer-a-native-port)).

## Build and install

There is nothing to compile. The installer creates `worker/venv/`, installs `worker/requirements.txt` (numpy,
scipy, numba, onnxruntime, lightkurve, wotan, pydl, psutil, and others) and the GPU package for your platform:

```bash
python3 worker/install.py --server-url http://<server-ip>:8400 --token <token> --worker-name <name>
```

Run it from a Python 3.10 or newer. `--unattended` skips the questions; `--update` only refreshes dependencies.
Per-platform prerequisites and `config.json` keys are in [install-workers.md](../install-workers.md).

## Run

```bash
cd worker && venv/bin/python worker.py --config config.json     # Windows: venv\Scripts\python.exe
```

`--worker-name` and `--server-url` on the command line override `config.json`; `farmctl` uses this so the host's
identity never lives in a file that updates can overwrite. For several instances per host use the agent, which
derives a config, cache directory, log and CPU budget per instance:

```bash
bin/farmctl start <worker-name>
```

## GPU backends

| Backend | Hardware | Used for | Install |
|---|---|---|---|
| CUDA | NVIDIA GPUs | BLS (`bls_cuda.py`), FPP (`fpp_cuda.py`), ONNX models via `onnxruntime-gpu` | `cupy-cuda12x`, `onnxruntime-gpu`, CUDA toolkit, numba |
| Metal | Apple Silicon | BLS and FPP kernels (`bls_metal*.py`, `fpp_metal*.py`), injection | `mlx` |
| OpenCL | Intel and AMD integrated GPUs, some discrete cards | BLS and FPP (`bls_opencl.py`, `fpp_opencl.py`), injection | `pyopencl` plus the OpenCL runtime |
| CPU (numba/numpy) | any | reference kernels; **development and tests only** | none |

The backend is auto-detected at start-up; force one with `EXONIT_BLS_BACKEND` (and the other `EXONIT_*` backend
variables in `.env.example`, set per host through the manifest's `env` block). A worker configured for a GPU must
not fall back to the CPU in production: check the log line at start-up for the active provider (for example
`ONNX-CUDA`) and treat `CPU` on a GPU host as a fault. Integrated GPUs saturate with a single instance; add CPU
instances instead of GPU instances on such hosts.

## Parity

The Python worker is the numerical reference. The Swift and Rust workers are validated to be **bit-exact against a
fresh Python run on the same host** for the stages they implement (detrending, BLS, vetting, CNN, injection,
FPP where ported), using golden fixtures generated from this code. Different hosts can differ in the last digits
(different BLAS, libm and GPU drivers), so compare like with like. Any change to the numerical path of the Python
worker must be mirrored in the native ports or it breaks that guarantee.

## Known limits

- **P2 refinement (`duration_refine`, `merge_same_period`) is opt-in and off by default.** It is not yet ported
  to Swift and Rust, so enabling it on Python workers would make the fleet inconsistent. Leave it off in a mixed
  fleet.
- Python is the slowest and the most CPU- and RAM-hungry worker; the same GPU does more work through a native
  port.
- Heavy BLS or injection phases can starve the heartbeat thread (Python's GIL), so a busy worker can show as
  offline for a while. Check the host's processes before restarting it.
- numba kernels compile on first use; the first task on a fresh host is slow.
- On Windows run natively; WSL2 is not supported.
