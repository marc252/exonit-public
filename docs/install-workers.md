# Installing workers

A worker is a process that polls the server for tasks (scan, FPP, and optional ExoMiner tasks), downloads the
light curve, runs the pipeline on its GPU and posts the result. You can run several instances per machine.

There are three implementations: the reference **Python** worker (this guide), and two native ports (**Swift**
for Apple Silicon, **Rust** for Linux/Windows with CUDA). See [When to prefer a native port](#when-to-prefer-a-native-port).

Project rules that apply to every worker:

- **The GPU is mandatory in production.** A worker that silently falls back to CPU is a misconfiguration, not a
  supported mode. CPU is for development and tests.
- **Never use 100% of a machine's CPUs.** Leave about 25% free so the host stays responsive (see `cpu_threads`
  and `fpp_cpu_cores` below).
- **Workers do not need internet access** beyond the server: light curves, metadata and models are served by the
  server (`/api/fits`, `/api/metadata`, `/api/worker/update`).
- **No Docker.** Workers are plain Python (or native binaries).

## Prerequisites per platform

| Platform | Python | GPU stack |
|---|---|---|
| Windows 11 (**native**) | 3.10+ from python.org | NVIDIA: current driver plus CUDA toolkit; `cupy-cuda12x` and `onnxruntime-gpu`. Intel/AMD iGPU: OpenCL runtime plus `pyopencl` |
| Linux | 3.10+ | NVIDIA: driver plus CUDA toolkit; `cupy-cuda12x`, `onnxruntime-gpu`. Intel/AMD: `pyopencl` plus the OpenCL ICD (`libOpenCL.so`) |
| macOS (Apple Silicon) | 3.10+ | Metal through `mlx`; nothing else to install |

**Windows: WSL2 is not supported.** Run the worker natively in PowerShell. WSL2 adds a GPU-passthrough layer
that proved unreliable for numba CUDA and for long-running detached processes.

`numba` is used for the CPU kernels and, on NVIDIA, for CUDA kernels; it needs the CUDA toolkit libraries on the
loader path.

## Install

The installer creates a virtual environment next to `worker/`, installs the base dependencies, detects your GPU
and installs the matching acceleration package, then writes `config.json`.

```bash
python3 worker/install.py --server-url http://<server-ip>:8400 --token <token> --worker-name <name>
```

Use `--unattended` for a non-interactive run and `--update` to refresh dependencies only. The worker name must
be **unique per worker** (two workers with the same name share one identity on the server and overwrite each
other).

Alternatively download the deployment bundle from the server (`/api/worker/download`), unzip it on the worker
machine and run the same installer from inside it.

Start the worker:

```bash
cd worker && venv/bin/python worker.py --config config.json
```

On first start it registers with the server, then keeps a heartbeat every 30 seconds.

## `config.json`

`worker/config.json` is per host and is never overwritten by auto-update. Keys you will touch:

| Key | Default | Meaning |
|---|---|---|
| `server_url` | `http://localhost:8400` | server URL (normally injected on the command line by `farmctl`) |
| `token` | `exonit-default-token` | shared secret; must equal the server's `EXONIT_WORKER_TOKEN` |
| `worker_name` | `worker-default` | unique worker identity |
| `cpu_threads` | `6` | CPU threads for BLS and detrending; keep at most 75% of the machine's cores |
| `fpp_cpu_cores` | `6` | CPU cores for TRICERATOPS FPP |
| `gpu_enabled` | `false` | must be `true` on any machine with a supported GPU (the installer sets it) |
| `poll_interval_seconds` / `heartbeat_interval_seconds` | `5` / `30` | polling cadence |
| `fits_cache_dir` | `fits_cache` | local cache of downloaded light curves |
| `exominer_mode` | `native` | ExoMiner path: `native` runs the ONNX model on the worker |
| `ffi_fallback_on_load_fail` | `true` | recover from a light-curve load failure through the FFI path |
| `gui_port`, `log_file`, `log_level` | `8402`, `worker.log`, `INFO` | local status page and logging |

For GPU backend selection and per-host overrides see the `EXONIT_*` backend variables in `.env.example`
(for example `EXONIT_BLS_BACKEND`, `EXONIT_INJECTION_METAL`); with `farmctl` they go in the `env` block of the
worker entry ([fleet.md](fleet.md)).

## Autostart

The installer offers a user service on Linux (systemd) and macOS (launchd). For a single machine you can also set it
up yourself:

- **Windows:** create a Task Scheduler task "At log on" that runs
  `<worker-dir>\venv\Scripts\python.exe <worker-dir>\worker.py --config <worker-dir>\config.json`, set to run only
  when the user is logged on (a non-interactive service session may not see the GPU). Starting detached workers
  through plain `Start-Process` over SSH is unreliable; prefer Task Scheduler or `farmctl`.
- **Linux:** the installer writes `~/.config/systemd/user/exonit-worker.service`; enable it with
  `systemctl --user enable --now exonit-worker.service` and `loginctl enable-linger <user>` so it survives logout.
- **macOS:** the installer writes a launchd agent. A Mac that sleeps stops its worker, so wrap the process with
  `caffeinate -i` (or a launchd job running `caffeinate`) and disable sleep for always-on machines. A sleeping
  Mac looks like a crashed worker: the heartbeat simply stops.

For more than a handful of machines use `farmctl` instead of per-host services ([fleet.md](fleet.md)).

## Auto-update

On start-up, and again whenever it is idle, the worker asks the server for `/api/worker/update`, which returns a
manifest of `{filename, sha256}` for the whole `worker/` tree and all models. Changed files are downloaded, and
the worker restarts itself (`os.execv`). Starting a worker therefore always brings its code and models in sync
with the server. `config.json` is excluded. Editing `worker/` on the server while a run is live restarts every
worker mid-run, so make code changes between lots.

## When to prefer a native port

- **Apple Silicon Macs:** use the [Swift worker](workers/swift.md). Measured on a Mac mini M4, three Swift
  instances processed more stars per minute than four Python instances.
- **Linux or Windows with an NVIDIA GPU:** the [Rust worker](workers/rust.md) uses less CPU and RAM per star than
  Python and so fits more instances inside the 75% CPU budget.
- **Anything else (Intel/AMD iGPU, mixed fleets, first trial):** the [Python worker](workers/python.md).

All three speak the same HTTP protocol and are managed by the same `farmctl` manifest.
