# Managing a worker fleet

`bin/farmctl` is the command-line tool that starts, stops, updates and inspects a farm of workers over SSH. It
runs on the server host (or any machine that can reach the workers by SSH and the server over HTTP). Use it as
soon as you have more than one or two machines.

Requirements: SSH access to every remote worker with **public-key authentication** (no passwords), and the worker
installed on each host ([install-workers.md](install-workers.md)).

## The manifest

Copy `ops/workers.example.json` to `ops/workers.json` and edit it. `farmctl` reads it for the server URL and the
list of workers.

```json
{
  "server_url": "http://<server-ip>:8400",
  "workers": [
    {
      "name": "worker-linux-gpu",
      "ssh_host": "user@<host>",
      "platform": "linux",
      "worker_dir": "~/exonit/worker",
      "python": "~/exonit/worker/venv/bin/python3",
      "instances": 2,
      "instances_fpp": 2,
      "config": { "gpu_enabled": true },
      "env": { "EXONIT_INJECTION_METAL": "1" }
    }
  ]
}
```

| Field | Meaning |
|---|---|
| `name` | unique worker name; instances become `<name>-1` .. `<name>-N`. Two entries with the same name collide on the server |
| `ssh_host` | `user@host`, or `null` for the machine that runs `farmctl` itself |
| `platform` | `linux`, `macos` or `windows` |
| `worker_dir` | directory holding `worker.py` and `agent.py` on that host |
| `python` | interpreter used to run them; for a native Swift/Rust worker this is the binary, which also implements the `agent.py` commands |
| `instances` | number of worker processes in the **scan** phase; `0` disables the entry in every phase |
| `instances_fpp` | number in the **FPP** phase; `0` means the host sits that phase out |
| `config` | optional, applied on top of the host's `config.json` at every start |
| `env` | optional, exported to every worker process of that host |
| `instance_overrides` | optional, keys are 1-based instance indices (as strings), tune single instances |

`worker_name` and `server_url` are injected on the command line by `farmctl`, not read from `config.json`, so a
code update never overwrites a host's identity.

## Commands

```bash
bin/farmctl status                 # online/offline, code version and flags per worker (from the server)
bin/farmctl status --detail        # additionally runs agent.py status over SSH on each host
bin/farmctl start worker-linux-gpu # start named workers, sized for the scan phase
bin/farmctl start --all --phase fpp
bin/farmctl stop --all
bin/farmctl restart --all
bin/farmctl deploy --all --branch main
bin/farmctl logs worker-linux-gpu -n 100
bin/farmctl exec worker-linux-gpu "df -h"
```

- `start` is idempotent and sizes the fleet **both ways**: instances that are running but not sized for the
  requested phase are stopped first (use `--no-prune` to only add). Giving every host `instances` for scan and
  `instances_fpp` for FPP lets the orchestrator re-size the fleet between phases.
- `stop` always works and covers the instance names of every phase.
- `deploy` runs `git pull` on git-tracked hosts and syncs the `worker/` tree on the others. It can overwrite a
  host's `config.json` (for example resetting `gpu_enabled` to `false`); re-check GPU settings after a deploy, or
  rely on the `config` block of the manifest, which is re-applied at every start. Normal code distribution does
  not need `deploy`: workers auto-update from the server on start.
- Safety guard: `start` refuses to run unless `EXONIT_FARMCTL_ALLOW_START=1` is set in that shell. It exists so
  that tests, scripts and automated agents cannot start a real fleet by accident. Set it yourself, in the
  terminal where you intend to start workers.

## Status is not ground truth

`farmctl status` reads heartbeats from the server. It can show a worker as online for a while after it died, and
as offline while it is alive but busy (a long GPU kernel can starve the heartbeat thread), and a sleeping Mac
simply stops reporting. Before concluding that something is running or stopped, **check the processes on the
host** (`ps`/`pgrep` on Linux and macOS, `Get-Process` on Windows). On macOS `pgrep` has no `-c` flag, and on Linux
`pgrep` needs `-f` for command lines longer than 15 characters; count with `ps ax | grep '[w]orker.py' | wc -l`.
Use `bin/farmctl status --detail` for a ground-truth check through the host's own agent.

## Sizing: instances per phase

- Scan phase is GPU-bound on Apple Silicon and CPU plus GPU on NVIDIA hosts; adding instances beyond the point
  where the GPU is saturated does not add throughput. Integrated GPUs (Intel, AMD, small Apple chips) saturate
  with a single instance.
- FPP is mostly CPU. Hosts with fast CPUs can run more FPP instances than scan instances (the reference config has 5
  scan and 9 FPP instances on an M4); slow hosts can sit the FPP tail out with `instances_fpp: 0`.
- **Leave about 25% of each host's CPUs free.** A machine pushed to 100% stops answering SSH, stalls its heartbeat
  and takes its own tasks hostage until they are reassigned (30 minutes by default). Budget `cpu_threads` times
  instances per host accordingly.
- On Windows, GPU memory (WDDM) limits how many FPP instances fit; add them one at a time and watch for driver resets.
- A machine that hosts the server or a database should either run no worker or a deliberately small one.

## Stuck tasks

If a run is finishing and a few tasks stay `assigned` to a worker that is no longer processing them (heartbeat
blocked, worker restarted), they are reassigned automatically after the timeout. To finish sooner, stop the
worker, requeue its tasks, and start it again.
