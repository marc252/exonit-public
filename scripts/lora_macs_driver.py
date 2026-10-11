#!/usr/bin/env python3
"""LoRA explanation step driver — the 5 Mac mini M4 fleet (Phase, 2026-09-26).

Runs on <host> with ``.venv-server`` (same host/venv as the generator and the
old ``ops/lora_via_llamacpp.sh``/``ops/lora_via_gpuq.sh`` — see
docs/superpowers/specs/2026-09-26-lora-macs-design.md for the full design). It is
the drop-in replacement for the <host>/vLLM path invoked by
``discovery_orchestrator.step_lora``: preflight the 5 Macs, start
``mlx_lm.server`` + the ``/api/explain`` shim on every host that passes, run
``scripts/generate_missing_explanations.py`` once against a shared
``--endpoints-json`` pool (sector-scoped pass, then a non-fatal backlog
pass), and ALWAYS tear every host back down — on success, on failure, and on
a signal (SIGTERM/SIGINT/SIGHUP).

Python (not bash) because there are 5 hosts to preflight/start/tear down in
parallel, a backlog budget to compute, and RAM/disk rules to apply — all of
that needs to be unit-testable as pure functions, which a shell script
cannot give us (see the "Pure functions" section below; the impure I/O layer
is a thin wrapper around them so tests can inject a fake SSH runner and never
touch real hardware).

Exit codes:
  0  sector pass ok (fail=0; backlog pass result is non-fatal either way)
  2  sector pass failed (generator returned fail>0 or rc!=0)
  3  zero healthy hosts after preflight (< min_hosts)
  4  another driver instance already holds the lock
  5  refused: running under pytest without EXONIT_LORA_MACS_ALLOW_UNDER_PYTEST=1

Usage:
    scripts/lora_macs_driver.py                       # normal step, reads
                                                        # EXONIT_LORA_SECTOR /
                                                        # EXONIT_LORA_BACKLOG_LIMIT
    scripts/lora_macs_driver.py --hosts m4b,m4c        # only these hosts
    scripts/lora_macs_driver.py --exclude m4           # every enabled host but m4
    scripts/lora_macs_driver.py --preflight-only       # preflight, report, exit
    scripts/lora_macs_driver.py --smoke                # start+readiness+10 reqs
                                                        # (no DB writes)+teardown
"""
from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import fcntl
import json
import logging
import math
import os
import re
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = ROOT / "ops" / "lora_macs.json"
LOCK_PATH = ROOT / "data" / "lora_macs.lock"
LOG_ROOT = ROOT / "data" / "lora_macs"
GEN_SCRIPT = ROOT / "scripts" / "generate_missing_explanations.py"
MACWAKE_BIN = ROOT / "bin" / "macwake"
FARMCTL_BIN = ROOT / "bin" / "farmctl"
BENCH_CLIENT = ROOT / "scripts" / "bench" / "m4_lora_bench_client.py"

logger = logging.getLogger("lora_macs_driver")

# Idempotent cleanup of a previous driver run's leftovers (design §2.1/§2.8):
# both mlx_lm.server and the uvicorn shim live under <remote_dir>/.venv/bin/,
# so one pattern kills either regardless of which one is still up.
ORPHAN_PKILL_PATTERN = r"exonit-lora/\.venv/bin/(mlx_lm\.server|uvicorn)"


# ---------------------------------------------------------------------------
# Pure functions — no I/O, no subprocess, no network. Every one of these is
# exercised directly by tests/test_lora_macs_driver.py with plain strings/
# dicts in, plain values out.
# ---------------------------------------------------------------------------

#: Apple Silicon (M-series) VM page size, bytes — vm_stat reports page COUNTS,
#: not bytes; this is the multiplier the design's RAM rule uses (§2.2).
MAC_PAGE_SIZE_BYTES = 16384

_VM_STAT_LINE_RE = re.compile(r"^(?P<label>[^:]+):\s+(?P<value>\d+)\.?\s*$")


def parse_vm_stat(text: str) -> dict[str, int]:
    """Parse macOS ``vm_stat`` output into ``{normalized_label: page_count}``.

    Labels are lowercased and non-alnum runs collapsed to a single
    underscore, e.g. ``"Pages free"`` -> ``"pages_free"``, ``"Pages
    speculative"`` -> ``"pages_speculative"``. The header line ("Mach
    Virtual Memory Statistics: (page size of 16384 bytes)") never matches
    (no trailing digit run) and is silently skipped.
    """
    out: dict[str, int] = {}
    for line in text.splitlines():
        m = _VM_STAT_LINE_RE.match(line.strip())
        if not m:
            continue
        label = re.sub(r"[^a-z0-9]+", "_", m.group("label").strip().lower()).strip("_")
        out[label] = int(m.group("value"))
    return out


def reclaimable_ram_gb(vm_stat_text: str, page_size: int = MAC_PAGE_SIZE_BYTES) -> float:
    """"Recuperable" RAM per the design (§0/§2.2): free + inactive +
    speculative + purgeable pages, in GiB."""
    pages = parse_vm_stat(vm_stat_text)
    keys = ("pages_free", "pages_inactive", "pages_speculative", "pages_purgeable")
    total_pages = sum(pages.get(k, 0) for k in keys)
    return (total_pages * page_size) / (1024 ** 3)


_SWAPUSAGE_USED_RE = re.compile(r"used\s*=\s*([\d.]+)\s*([MG])", re.IGNORECASE)


def parse_swap_used_gb(swapusage_text: str) -> float:
    """Parse ``used = 386.00M`` (or ``...G``) out of ``sysctl
    vm.swapusage`` output. No match (e.g. swap fully off) -> 0.0."""
    m = _SWAPUSAGE_USED_RE.search(swapusage_text)
    if not m:
        return 0.0
    value = float(m.group(1))
    return value / 1024.0 if m.group(2).upper() == "M" else value


def parse_df_avail_gb(df_g_output: str) -> float:
    """Parse the ``Avail`` column (GB) out of ``df -g <path>`` output.

    Uses the header row to locate the column when present; falls back to
    the fixed macOS ``df -g`` layout (``Filesystem 1G-blocks Used Avail
    Capacity ...``, column index 3) if the header can't be matched (e.g. a
    truncated/mocked capture in a test).
    """
    lines = [ln for ln in df_g_output.splitlines() if ln.strip()]
    if len(lines) < 2:
        raise ValueError(f"unexpected `df -g` output (need header+data): {df_g_output!r}")
    header = lines[0].split()
    data = lines[-1].split()
    try:
        idx = header.index("Avail")
    except ValueError:
        idx = 3
    return float(data[idx])


_WORKER_PS_RE = re.compile(r"exonit-swift-worker|exonit-worker/worker\.py")


def count_exonit_worker_processes(ps_output: str) -> int:
    """Count ExoNit worker processes in a ``ps ax -o command=`` capture.

    Same two patterns the design specifies for the remote
    ``grep -c '[e]xonit-swift-worker\\|exonit-worker/[w]orker.py'`` pipeline,
    reimplemented in Python: the SSH round trip returns the raw ``ps``
    listing and this matches against it locally, so there is no `ps ax`
    line for grep's own argv to self-match (the ``[e]xonit...`` bracket
    trick in the design exists only for the remote-shell version of this
    check; not needed here).
    """
    return sum(1 for line in ps_output.splitlines() if _WORKER_PS_RE.search(line))


#: Design §2.2's host-specific safeguard, generalized to any host
#: (config-driven per host in ops/lora_macs.json, defaults applied
#: fleet-wide): ".60 in use by the operator" — a resident model in their own oMLX app,
#: or any other stray mlx_lm/llama.cpp process NOT started by us, competes
#: for the same RAM/Metal the RAM rule above just cleared the host for. The
#: generic reclaimable-RAM rule (apply_ram_rule) may often catch this
#: indirectly (a resident model eats into free/inactive/purgeable pages),
#: but the design explicitly calls this an independent check because that
#: coverage is unknown ("unknown whether oMLX keeps the model resident when
#: idle") — hence a second, direct guard rather than relying on the first.
DEFAULT_FOREIGN_LLM_PROCESS_PATTERNS = [
    r"omlx",              # the operator's oMLX.app (personal, seen on .60)
    r"mlx[_-]lm",         # any mlx_lm.server/CLI NOT under our own remote_dir
    r"llama[-_.]?server", # llama.cpp server (Phase 2 override artifact, or the operator's own)
    r"llama\.cpp",
]
#: Fleet-wide default RSS ceiling (GB) for a match above — design's ".60 ...
#: RSS > 2 GB" number, applied everywhere unless a host overrides it.
DEFAULT_FOREIGN_LLM_RSS_GB_MAX = 2.0

_PS_RSS_LINE_RE = re.compile(r"^\s*(\d+)\s+(.*)$")


def find_foreign_llm_processes(ps_output: str, patterns: list[str],
                                own_dir: str, rss_gb_max: float,
                                ) -> list[tuple[str, float]]:
    """Parse a ``ps ax -o rss=,command=`` capture (RSS in KB, then the full
    command line) and return every ``(command, rss_gb)`` pair that:

    - matches at least one of *patterns* (regex, case-insensitive) — e.g.
      the operator's oMLX app, a stray ``mlx_lm.server``/CLI, a llama.cpp server;
    - is **not one of our own processes**: *own_dir* (the host's
      ``~/exonit-lora`` ``remote_dir``) does not appear in the command line.
      ``cleanup_orphans()`` already kills any leftover process of ours
      before preflight runs, but this is a second, independent guard — it
      must never exclude a host just because the ``mlx_lm.server`` WE are
      about to start would itself match the generic ``mlx`` pattern;
    - exceeds *rss_gb_max* GB resident (RSS).

    Empty *patterns* is a per-host opt-out -> always returns ``[]``.
    Malformed/unmatched ``ps`` lines (e.g. the header, if present) are
    silently skipped, same convention as ``parse_vm_stat``.
    """
    if not patterns:
        return []
    compiled = [re.compile(p, re.IGNORECASE) for p in patterns]
    hits: list[tuple[str, float]] = []
    for line in ps_output.splitlines():
        m = _PS_RSS_LINE_RE.match(line)
        if not m:
            continue
        rss_kb, command = int(m.group(1)), m.group(2)
        if own_dir and own_dir in command:
            continue
        if not any(p.search(command) for p in compiled):
            continue
        rss_gb = rss_kb / (1024 ** 2)
        if rss_gb > rss_gb_max:
            hits.append((command, rss_gb))
    return hits


@dataclass(frozen=True)
class RamVerdict:
    """Preflight RAM rule outcome for one host."""
    concurrency: int | None   # None = host EXCLUDED
    reason: str


def apply_ram_rule(configured_concurrency: int, reclaimable_gb_value: float,
                    swap_used_gb_value: float) -> RamVerdict:
    """Design §2.2's RAM rule:
      - swap used > 1.5 GB           -> EXCLUDED (regardless of free RAM)
      - reclaimable < 6 GB           -> EXCLUDED
      - 6 GB <= reclaimable < 8 GB   -> concurrency downgraded to min(cfg, 4)
      - reclaimable >= 8 GB          -> configured concurrency, unchanged
    Downgrade only ever lowers concurrency, never raises it (a host
    configured at c4 stays c4 even with 20 GB free).
    """
    if swap_used_gb_value > 1.5:
        return RamVerdict(None, f"swap used {swap_used_gb_value:.2f} GB > 1.5 GB cap")
    if reclaimable_gb_value < 6.0:
        return RamVerdict(None, f"reclaimable RAM {reclaimable_gb_value:.2f} GB < 6 GB floor")
    if reclaimable_gb_value < 8.0:
        downgraded = min(int(configured_concurrency), 4)
        return RamVerdict(
            downgraded,
            f"reclaimable RAM {reclaimable_gb_value:.2f} GB in [6,8) GB -> "
            f"concurrency downgraded {configured_concurrency}->{downgraded}",
        )
    return RamVerdict(
        int(configured_concurrency),
        f"reclaimable RAM {reclaimable_gb_value:.2f} GB >= 8 GB -> "
        f"concurrency {configured_concurrency} kept",
    )


def compute_backlog_budget(rates_per_min: list[float], budget_hours: float,
                            backlog_limit_cap: int) -> int:
    """Design §2.5: ``L = min(EXONIT_LORA_BACKLOG_LIMIT, floor(Σ rate_i * 60
    * budget_h))``. ``backlog_limit_cap <= 0`` means the backlog pass is off
    entirely (same semantics as the pre-existing ``LORA_BACKLOG_LIMIT``
    knob) -> 0, independent of fleet capacity.
    """
    if backlog_limit_cap <= 0:
        return 0
    by_capacity = math.floor(sum(rates_per_min) * 60.0 * float(budget_hours))
    return max(0, min(int(backlog_limit_cap), by_capacity))


def build_endpoints(healthy_hosts: list[dict[str, Any]], port: int) -> list[dict[str, Any]]:
    """Build the ``--endpoints-json`` payload: ``[{url, concurrency, name}]``.

    *healthy_hosts* items must carry ``ip``, ``name`` and a resolved
    ``concurrency`` (post RAM-rule) — the config's per-host ``port`` is
    only overridden if the host dict sets its own (multi-tenant future
    proofing; today every host shares the fleet-wide *port*).
    """
    out = []
    for h in healthy_hosts:
        out.append({
            "url": f"http://{h['ip']}:{h.get('port', port)}/api/explain",
            "concurrency": int(h["concurrency"]),
            "name": h["name"],
        })
    return out


def resolve_hosts(all_hosts: list[dict[str, Any]], only: list[str] | None,
                   exclude: list[str] | None) -> list[dict[str, Any]]:
    """Apply ``--hosts``/``--exclude`` (by ``name`` alias, matching
    bin/macwake's vocabulary) on top of each host's own ``enabled`` flag.

    ``--hosts`` is an explicit ask and overrides ``enabled: false`` (e.g. to
    test the personal .60 Mac on demand); ``--exclude`` always wins even
    over an explicit ``--hosts`` entry (a way to say "all except this one,
    no matter what").
    """
    exclude_set = set(exclude or [])
    if only:
        only_set = set(only)
        chosen = [h for h in all_hosts if h["name"] in only_set]
    else:
        chosen = [h for h in all_hosts if h.get("enabled", True)]
    return [h for h in chosen if h["name"] not in exclude_set]


def check_manifest_integrity(manifest: dict[str, Any],
                              remote_sizes: dict[str, int]) -> tuple[bool, str]:
    """Cheap per-pass integrity check (design §2.2): compare each
    ``manifest['files'][path]['size']`` (bytes) against what the host
    reports for that path (``remote_sizes``, e.g. via ``stat -f%z``). Full
    sha256 verification is the provisioning script's job (§4) — this only
    needs to catch a truncated/missing file cheaply, not re-hash 4 GB on
    every LoRA step.
    """
    missing, mismatched = [], []
    for rel, meta in manifest.get("files", {}).items():
        expected = meta.get("size")
        actual = remote_sizes.get(rel)
        if actual is None:
            missing.append(rel)
        elif expected is not None and int(actual) != int(expected):
            mismatched.append(rel)
    if missing or mismatched:
        return False, f"missing={missing} mismatched={mismatched}"
    return True, ""


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class HostConfig:
    name: str
    ssh: str
    ip: str
    concurrency: int
    rate_per_min: float
    workers: list[str] = field(default_factory=list)
    enabled: bool = True
    note: str = ""
    #: Per-host override of the fleet-wide foreign-LLM-process guard
    #: (FleetConfig.foreign_llm_process_patterns/_rss_gb_max below). None
    #: means "use the fleet default"; an explicit `[]` opts this host out.
    foreign_llm_process_patterns: list[str] | None = None
    foreign_llm_rss_gb_max: float | None = None


@dataclass
class FleetConfig:
    hosts: list[HostConfig]
    port: int = 8500
    mlx_port: int = 8503
    remote_dir: str = "exonit-lora"  # relative to the remote $HOME
    min_hosts: int = 1
    min_free_disk_gb: float = 10.0
    budget_hours: float = 14.0
    deadline_hours: float = 20.0
    model: str = "qwen3-4b-lora-fused-q8"
    #: Fleet-wide default for the foreign-LLM-process preflight guard (see
    #: find_foreign_llm_processes); a host's own
    #: foreign_llm_process_patterns/_rss_gb_max override these when set.
    foreign_llm_process_patterns: list[str] = field(
        default_factory=lambda: list(DEFAULT_FOREIGN_LLM_PROCESS_PATTERNS))
    foreign_llm_rss_gb_max: float = DEFAULT_FOREIGN_LLM_RSS_GB_MAX

    @classmethod
    def load(cls, path: Path | str = DEFAULT_CONFIG_PATH) -> "FleetConfig":
        with open(path, encoding="utf-8") as fh:
            raw = json.load(fh)
        hosts = [
            HostConfig(
                name=h["name"], ssh=h["ssh"], ip=h["ip"],
                concurrency=int(h["concurrency"]), rate_per_min=float(h["rate_per_min"]),
                workers=list(h.get("workers", [])), enabled=bool(h.get("enabled", True)),
                note=h.get("note", ""),
                foreign_llm_process_patterns=(
                    list(h["foreign_llm_process_patterns"])
                    if "foreign_llm_process_patterns" in h else None),
                foreign_llm_rss_gb_max=(
                    float(h["foreign_llm_rss_gb_max"])
                    if "foreign_llm_rss_gb_max" in h else None),
            )
            for h in raw["hosts"]
        ]
        return cls(
            hosts=hosts,
            port=int(raw.get("port", 8500)),
            mlx_port=int(raw.get("mlx_port", 8503)),
            remote_dir=raw.get("remote_dir", "exonit-lora"),
            min_hosts=int(raw.get("min_hosts", 1)),
            min_free_disk_gb=float(raw.get("min_free_disk_gb", 10.0)),
            budget_hours=float(raw.get("budget_hours", 14.0)),
            deadline_hours=float(raw.get("deadline_hours", 20.0)),
            model=raw.get("model", "qwen3-4b-lora-fused-q8"),
            foreign_llm_process_patterns=list(raw.get(
                "foreign_llm_process_patterns", DEFAULT_FOREIGN_LLM_PROCESS_PATTERNS)),
            foreign_llm_rss_gb_max=float(raw.get(
                "foreign_llm_rss_gb_max", DEFAULT_FOREIGN_LLM_RSS_GB_MAX)),
        )


@dataclass
class HostResult:
    """Preflight/lifecycle outcome for one host, threaded through the run."""
    config: HostConfig
    healthy: bool = False
    excluded_reason: str = ""
    concurrency: int | None = None     # resolved (post RAM-rule) concurrency
    ip: str = ""
    port: int = 0
    started: bool = False
    ready: bool = False
    clean: bool | None = None          # teardown verification result


# ---------------------------------------------------------------------------
# SSH runner — the ONE seam between the pure logic above and real hardware.
# Every impure method below takes this as a parameter so tests can supply a
# fake that never opens a socket.
# ---------------------------------------------------------------------------

@dataclass
class SSHResult:
    returncode: int
    stdout: str
    stderr: str


SSHRunner = Callable[[str, str, float], SSHResult]


def real_ssh_runner(ssh_host: str, remote_cmd: str, timeout: float = 20.0) -> SSHResult:
    """Default SSHRunner: real `ssh`, BatchMode (never prompts), ConnectTimeout
    8s (matches macwake/farmctl's own convention)."""
    try:
        proc = subprocess.run(
            ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", ssh_host, remote_cmd],
            capture_output=True, text=True, timeout=timeout,
        )
        return SSHResult(proc.returncode, proc.stdout, proc.stderr)
    except subprocess.TimeoutExpired as exc:
        return SSHResult(255, exc.stdout or "", (exc.stderr or "") + " (ssh timeout)")
    except Exception as exc:  # host unreachable, etc.
        return SSHResult(255, "", str(exc))


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

class Driver:
    """Owns one LoRA-macs-fleet run: preflight -> start -> readiness ->
    generator pass(es) -> teardown (always). All hardware I/O goes through
    the injectable callables below so `--smoke`/production code and the
    test suite share the exact same control flow.
    """

    def __init__(self, config: FleetConfig, *,
                 ssh: SSHRunner = real_ssh_runner,
                 http_get: Callable[[str, float], tuple[int, str]] | None = None,
                 http_post: Callable[[str, dict, float], tuple[int, str]] | None = None,
                 macwake: Callable[[str], bool] | None = None,
                 farmctl_stop: Callable[[list[str]], bool] | None = None,
                 popen: Callable[..., subprocess.Popen] = subprocess.Popen,
                 log_dir: Path | None = None):
        self.config = config
        self.ssh = ssh
        self.http_get = http_get or self._default_http_get
        self.http_post = http_post or self._default_http_post
        self.macwake = macwake or self._default_macwake
        self.farmctl_stop = farmctl_stop or self._default_farmctl_stop
        self.popen = popen
        self.log_dir = log_dir or (LOG_ROOT / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
        self.results: dict[str, HostResult] = {}
        self._shutting_down = False

    # -- default impure helpers (overridable for tests) --------------------

    @staticmethod
    def _default_http_get(url: str, timeout: float) -> tuple[int, str]:
        import urllib.request
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r:
                return r.status, r.read().decode("utf-8", "replace")
        except Exception as exc:
            return 0, str(exc)

    @staticmethod
    def _default_http_post(url: str, payload: dict, timeout: float) -> tuple[int, str]:
        import urllib.request
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.status, r.read().decode("utf-8", "replace")
        except Exception as exc:
            return 0, str(exc)

    def _default_macwake(self, alias: str) -> bool:
        try:
            proc = subprocess.run([str(MACWAKE_BIN), alias, "--wait"],
                                  capture_output=True, text=True, timeout=130)
            return proc.returncode == 0
        except Exception:
            return False

    def _default_farmctl_stop(self, worker_names: list[str]) -> bool:
        if not worker_names:
            return True
        try:
            proc = subprocess.run([str(FARMCTL_BIN), "stop", *worker_names, "--force"],
                                  capture_output=True, text=True, timeout=60)
            return proc.returncode == 0
        except Exception:
            return False

    # -- logging -------------------------------------------------------

    def _job(self, msg: str) -> None:
        line = f"[job] {msg}"
        print(line, flush=True)
        logger.info(msg)

    # -- lifecycle: preflight -------------------------------------------

    def preflight_host(self, host: HostConfig) -> HostResult:
        """One host's full preflight (design §2.2), run synchronously —
        callers parallelize via a ThreadPoolExecutor across hosts."""
        res = HostResult(config=host, ip=host.ip, port=self.config.port)

        # 1) Reachability, with a wake-on-LAN fallback.
        probe = self.ssh(host.ssh, "true", 8.0)
        if probe.returncode != 0:
            self._job(f"host={host.name} unreachable, trying macwake --wait")
            if not self.macwake(host.name):
                res.excluded_reason = "unreachable (macwake --wait timed out)"
                self._job(f"host={host.name} EXCLUDED: {res.excluded_reason}")
                return res
            probe = self.ssh(host.ssh, "true", 8.0)
            if probe.returncode != 0:
                res.excluded_reason = "unreachable after macwake"
                self._job(f"host={host.name} EXCLUDED: {res.excluded_reason}")
                return res

        # 2) No ExoNit worker still running.
        ps = self.ssh(host.ssh, "ps ax -o command=", 15.0)
        if ps.returncode == 0 and count_exonit_worker_processes(ps.stdout) > 0:
            self._job(f"host={host.name} ExoNit worker(s) still running, "
                      f"attempting farmctl stop {host.workers}")
            self.farmctl_stop(host.workers)
            ps2 = self.ssh(host.ssh, "ps ax -o command=", 15.0)
            if ps2.returncode != 0 or count_exonit_worker_processes(ps2.stdout) > 0:
                res.excluded_reason = "ExoNit worker still running after farmctl stop"
                self._job(f"host={host.name} EXCLUDED: {res.excluded_reason}")
                return res

        # 3) Disk.
        df = self.ssh(host.ssh, f"df -g {self.config.remote_dir}", 15.0)
        if df.returncode != 0:
            res.excluded_reason = f"df failed: {(df.stderr or df.stdout)[:200]}"
            self._job(f"host={host.name} EXCLUDED: {res.excluded_reason}")
            return res
        try:
            avail_gb = parse_df_avail_gb(df.stdout)
        except ValueError as exc:
            res.excluded_reason = f"unparseable df output: {exc}"
            self._job(f"host={host.name} EXCLUDED: {res.excluded_reason}")
            return res
        if avail_gb < self.config.min_free_disk_gb:
            res.excluded_reason = (f"disk {avail_gb:.1f} GB < {self.config.min_free_disk_gb} GB "
                                   f"floor — clean up fits_cache_*")
            self._job(f"host={host.name} EXCLUDED: {res.excluded_reason}")
            return res

        # 4) RAM.
        vm = self.ssh(host.ssh, "vm_stat", 15.0)
        swap = self.ssh(host.ssh, "sysctl vm.swapusage", 15.0)
        if vm.returncode != 0 or swap.returncode != 0:
            res.excluded_reason = "vm_stat/sysctl vm.swapusage failed"
            self._job(f"host={host.name} EXCLUDED: {res.excluded_reason}")
            return res
        reclaimable = reclaimable_ram_gb(vm.stdout)
        swap_used = parse_swap_used_gb(swap.stdout)
        verdict = apply_ram_rule(host.concurrency, reclaimable, swap_used)
        self._job(f"host={host.name} RAM: {verdict.reason}")
        if verdict.concurrency is None:
            res.excluded_reason = verdict.reason
            self._job(f"host={host.name} EXCLUDED: {res.excluded_reason}")
            return res

        # 4b) Foreign LLM process guard (design §2.2's ".60 in use by the operator"
        # check, generalized/config-driven to every host — see
        # find_foreign_llm_processes). A resident oMLX/mlx_lm/llama.cpp
        # process not started by us competes for the same RAM/Metal the RAM
        # rule just cleared this host for; the generic reclaimable-RAM rule
        # above may catch this indirectly but the design calls out this as
        # an independent, direct check because that coverage is unknown.
        patterns = (host.foreign_llm_process_patterns
                   if host.foreign_llm_process_patterns is not None
                   else self.config.foreign_llm_process_patterns)
        rss_gb_max = (host.foreign_llm_rss_gb_max
                     if host.foreign_llm_rss_gb_max is not None
                     else self.config.foreign_llm_rss_gb_max)
        if patterns:
            ps_rss = self.ssh(host.ssh, "ps ax -o rss=,command=", 15.0)
            if ps_rss.returncode != 0:
                res.excluded_reason = "ps rss= scan failed"
                self._job(f"host={host.name} EXCLUDED: {res.excluded_reason}")
                return res
            hits = find_foreign_llm_processes(
                ps_rss.stdout, patterns, self.config.remote_dir, rss_gb_max)
            if hits:
                cmd, rss_gb = max(hits, key=lambda h: h[1])
                res.excluded_reason = f"foreign_llm_loaded rss={rss_gb:.1f} GB ({cmd[:80]!r})"
                self._job(f"host={host.name} EXCLUDED: {res.excluded_reason}")
                return res

        # 5) Integrity (best-effort; missing manifest excludes — a host that
        # was never provisioned cannot serve).
        manifest_cmd = f"cat {self.config.remote_dir}/MANIFEST.json"
        manifest_raw = self.ssh(host.ssh, manifest_cmd, 15.0)
        if manifest_raw.returncode != 0:
            res.excluded_reason = "MANIFEST.json missing — host not provisioned"
            self._job(f"host={host.name} EXCLUDED: {res.excluded_reason}")
            return res
        try:
            manifest = json.loads(manifest_raw.stdout)
        except json.JSONDecodeError:
            res.excluded_reason = "MANIFEST.json unparseable"
            self._job(f"host={host.name} EXCLUDED: {res.excluded_reason}")
            return res
        remote_sizes = self._stat_manifest_files(host, manifest)
        ok, detail = check_manifest_integrity(manifest, remote_sizes)
        if not ok:
            res.excluded_reason = f"integrity check failed: {detail}"
            self._job(f"host={host.name} EXCLUDED: {res.excluded_reason}")
            return res

        res.healthy = True
        res.concurrency = verdict.concurrency
        self._job(f"host={host.name} PASSED preflight, concurrency={verdict.concurrency}")
        return res

    def _stat_manifest_files(self, host: HostConfig, manifest: dict) -> dict[str, int]:
        files = list(manifest.get("files", {}).keys())
        if not files:
            return {}
        remote_dir = self.config.remote_dir
        cmd = " ; ".join(f"stat -f'%N %z' {remote_dir}/{f} 2>/dev/null" for f in files)
        result = self.ssh(host.ssh, cmd, 30.0)
        sizes: dict[str, int] = {}
        for line in result.stdout.splitlines():
            parts = line.rsplit(" ", 1)
            if len(parts) != 2:
                continue
            path, size = parts
            for f in files:
                if path.endswith(f):
                    with contextlib.suppress(ValueError):
                        sizes[f] = int(size)
        return sizes

    def preflight_all(self, hosts: list[HostConfig],
                      max_workers: int = 8) -> list[HostResult]:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
            results = list(pool.map(self.preflight_host, hosts))
        for r in results:
            self.results[r.config.name] = r
        return results

    # -- lifecycle: orphan cleanup ----------------------------------------

    def cleanup_orphans(self, hosts: list[HostConfig]) -> None:
        """Idempotent best-effort pkill of a previous run's leftovers on
        every host we're about to touch (design §2.1) — never fatal."""
        def _one(h: HostConfig) -> None:
            self.ssh(h.ssh, f"pkill -f '{ORPHAN_PKILL_PATTERN}' 2>/dev/null; true", 10.0)
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(_one, hosts))

    # -- lifecycle: start ---------------------------------------------------

    def start_host(self, result: HostResult) -> None:
        host = result.config
        remote_dir = self.config.remote_dir
        c = result.concurrency
        mlx_cmd = (
            f"cd {remote_dir} && mkdir -p run logs && "
            f"nohup .venv/bin/mlx_lm.server --model models/{self.config.model} "
            f"--host 127.0.0.1 --port {self.config.mlx_port} "
            f"--decode-concurrency {c} --prompt-concurrency {c} "
            f"--max-tokens 640 --prompt-cache-bytes 1073741824 "
            f"--log-level WARNING < /dev/null > logs/mlx.log 2>&1 & "
            f"echo $! > run/mlx.pid"
        )
        shim_cmd = (
            f"cd {remote_dir} && "
            f"PYTHONPATH=$PWD/shim "
            f"EXPLAINER_LLAMA_URL=http://127.0.0.1:{self.config.mlx_port} "
            f"EXPLAINER_MAX_TOKENS=640 EXPLAINER_REQUEST_TIMEOUT=150 "
            f"EXPLAINER_BACKEND=mlx "
            f"nohup .venv/bin/uvicorn --app-dir $PWD/shim "
            f"explainer_service_llamacpp:app --host 0.0.0.0 --port {self.config.port} "
            f"< /dev/null > logs/shim.log 2>&1 & "
            f"echo $! > run/shim.pid"
        )
        r1 = self.ssh(host.ssh, mlx_cmd, 15.0)
        r2 = self.ssh(host.ssh, shim_cmd, 15.0)
        result.started = (r1.returncode == 0 and r2.returncode == 0)
        if not result.started:
            self._job(f"host={host.name} START FAILED mlx_rc={r1.returncode} shim_rc={r2.returncode}")

    def start_all(self, results: list[HostResult]) -> None:
        healthy = [r for r in results if r.healthy]
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(self.start_host, healthy))

    # -- lifecycle: readiness ------------------------------------------

    def wait_ready(self, result: HostResult, timeout_s: float = 120.0) -> None:
        """Readiness = shim /healthz reports upstream ok AND one real
        /api/explain request succeeds (design §2.4 — a bare /healthz can be
        200 before Metal has generated a single token)."""
        host = result.config
        deadline = time.time() + timeout_s
        healthz_url = f"http://{result.ip}:{result.port}/healthz"
        upstream_ok = False
        while True:
            code, body = self.http_get(healthz_url, 5.0)
            if code == 200:
                try:
                    upstream_ok = json.loads(body).get("upstream") == "ok"
                except json.JSONDecodeError:
                    upstream_ok = False
            if upstream_ok or time.time() >= deadline:
                break
            time.sleep(3.0)
        if not upstream_ok:
            result.ready = False
            self._job(f"host={host.name} NOT READY (shim/mlx never served)")
            return

        # Fields per explainer_service_vllm_qwen3.ExplainRequest (shared
        # contract, imported by the llamacpp shim — see design §1). Only
        # `tic_id` is required; a synthetic value is enough to exercise a
        # real generation round trip without touching real candidate data.
        explain_url = f"http://{result.ip}:{result.port}/api/explain"
        probe_payload = {
            "tic_id": "0", "candidate_num": 1, "period_days": 3.14,
            "depth_ppm": 500.0, "duration_days": 0.1, "snr": 12.0,
            "final_verdict": "PC",
        }
        code, _ = self.http_post(explain_url, probe_payload, 60.0)
        result.ready = code == 200
        self._job(f"host={host.name} readiness probe http={code} ready={result.ready}")

    def wait_ready_all(self, results: list[HostResult]) -> None:
        started = [r for r in results if r.started]
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(self.wait_ready, started))
        for r in started:
            if not r.ready:
                r.healthy = False
                r.excluded_reason = r.excluded_reason or "failed readiness probe"

    # -- lifecycle: teardown ----------------------------------------------

    def teardown_host(self, result: HostResult) -> None:
        host = result.config
        remote_dir = self.config.remote_dir
        kill_cmd = (
            f"cd {remote_dir} && "
            f"for f in run/mlx.pid run/shim.pid; do "
            f"[ -f \"$f\" ] && kill $(cat \"$f\") 2>/dev/null; done; "
            f"sleep 5; "
            f"pkill -9 -f '{ORPHAN_PKILL_PATTERN}' 2>/dev/null; "
            f"true"
        )
        self.ssh(host.ssh, kill_cmd, 20.0)
        verify = self.ssh(host.ssh, "ps ax -o command=", 15.0)
        remaining = 0
        if verify.returncode == 0:
            remaining = sum(1 for line in verify.stdout.splitlines()
                            if re.search(ORPHAN_PKILL_PATTERN, line))
        result.clean = (remaining == 0)
        self._job(f"host={host.name} clean={'yes' if result.clean else 'no'}")

    def teardown_all(self) -> None:
        touched = [r for r in self.results.values() if r.started or r.healthy]
        if not touched:
            return
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(self.teardown_host, touched))


# ---------------------------------------------------------------------------
# Lock file
# ---------------------------------------------------------------------------

class LockHeldError(RuntimeError):
    pass


@contextlib.contextmanager
def acquire_lock(path: Path = LOCK_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(path, "w")
    try:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        fh.close()
        raise LockHeldError(f"another lora_macs_driver holds {path}")
    try:
        fh.write(str(os.getpid()))
        fh.flush()
        yield
    finally:
        with contextlib.suppress(OSError):
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
        fh.close()


# ---------------------------------------------------------------------------
# Generator invocation (local, on <host> — see design §2.6/§2.7)
# ---------------------------------------------------------------------------

def run_generator(*, db: str, endpoints_json_path: Path, timeout: float,
                  sector: int | None, order: str | None, limit: int | None,
                  max_seconds: float | None, log_path: Path,
                  popen: Callable[..., subprocess.Popen] = subprocess.Popen) -> int:
    cmd = [
        sys.executable, str(GEN_SCRIPT),
        "--db", db,
        "--endpoints-json", str(endpoints_json_path),
        "--timeout", str(timeout),
        "--strong-only", "--include-stale",
    ]
    if sector is not None:
        cmd += ["--sector", str(int(sector))]
    if order:
        cmd += ["--order", order]
    if limit:
        cmd += ["--limit", str(int(limit))]
    if max_seconds:
        cmd += ["--max-seconds", str(int(max_seconds))]
    env = dict(os.environ)
    env["CLAUDE_FALLBACK"] = "0"
    with open(log_path, "w") as log_fh:
        proc = popen(cmd, stdout=log_fh, stderr=subprocess.STDOUT, env=env,
                    cwd=str(ROOT), start_new_session=True)
        return proc.wait()


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def _parse_hostlist(value: str | None) -> list[str] | None:
    if not value:
        return None
    return [v.strip() for v in value.split(",") if v.strip()]


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    p.add_argument("--hosts", default=None, help="comma-separated alias list (overrides enabled:false)")
    p.add_argument("--exclude", default=None, help="comma-separated alias list to drop")
    p.add_argument("--preflight-only", action="store_true")
    p.add_argument("--smoke", action="store_true",
                   help="start+readiness+N no-DB-write requests+teardown, no generator")
    p.add_argument("--smoke-n", type=int, default=10)
    p.add_argument("--db", default=os.environ.get(
        "EXONIT_LORA_DB", os.environ.get(
            "EXONIT_DATABASE_URL", "postgresql://exonit@localhost:5432/exonit")))
    p.add_argument("--gen-timeout", type=float, default=170.0,
                   help="--timeout passed to the generator (per-request deadline)")
    return p


#: Hard guard (2026-09-26 incident): a test in this repo's pytest suite
#: reached this script for real — via discovery_orchestrator.step_lora's
#: unmocked default — and it ran to completion against production: `farmctl
#: stop` on all 5 M4 Swift workers mid discovery-run, mlx_lm.server started
#: on every host, and 1,650 real LoRA explanations written to prod Postgres.
#: pytest sets PYTEST_CURRENT_TEST in os.environ for the duration of every
#: test, and subprocesses inherit it — so this is a reliable signal that we
#: are running under pytest, however we were invoked (direct call, real
#: subprocess spawned by an orchestrator test, etc.). Unit tests that need to
#: exercise this driver for real must set EXONIT_LORA_MACS_ALLOW_UNDER_PYTEST=1
#: AND mock every ssh/http/farmctl/Popen call themselves (see
#: tests/test_lora_macs_driver.py, which injects fakes into Driver() and
#: never calls main()).
LORA_MACS_PYTEST_GUARD_ENV = "EXONIT_LORA_MACS_ALLOW_UNDER_PYTEST"

#: Distinct exit code for the guard above, so a caller can tell "refused
#: under pytest" apart from every other failure mode (0/2/3/4 already mean
#: something — see the module docstring's "Exit codes" section).
EXIT_REFUSED_UNDER_PYTEST = 5


def _running_under_pytest_without_allow() -> bool:
    return ("PYTEST_CURRENT_TEST" in os.environ
            and os.environ.get(LORA_MACS_PYTEST_GUARD_ENV) != "1")


def main(argv: list[str] | None = None) -> int:
    # Before ANY ssh/farmctl/HTTP/DB action — see the guard's docstring above.
    if _running_under_pytest_without_allow():
        print(
            "REFUSED: lora_macs_driver.py detected PYTEST_CURRENT_TEST in "
            "the environment (this process is running under pytest) — "
            "refusing to touch real SSH/farmctl/HTTP/DB. This guard exists "
            "because a test caused this exact script to run for real on "
            "2026-09-26, stopping the production Swift fleet and writing "
            "1,650 real LoRA explanations to prod Postgres. If you are "
            f"deliberately testing this driver end-to-end, set "
            f"{LORA_MACS_PYTEST_GUARD_ENV}=1 and mock every remote action "
            "yourself.",
            file=sys.stderr,
        )
        return EXIT_REFUSED_UNDER_PYTEST
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_arg_parser().parse_args(argv)
    config = FleetConfig.load(args.config)
    hosts = resolve_hosts([h.__dict__ for h in config.hosts],
                          _parse_hostlist(args.hosts), _parse_hostlist(args.exclude))
    host_configs = [h for h in config.hosts if h.name in {x["name"] for x in hosts}]

    sector_env = os.environ.get("EXONIT_LORA_SECTOR", "").strip()
    sector = int(sector_env) if sector_env else None
    backlog_cap = int(os.environ.get("EXONIT_LORA_BACKLOG_LIMIT", "0") or 0)

    driver = Driver(config)
    driver.log_dir.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(driver.log_dir / "driver.log")
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(file_handler)

    deadline = time.time() + config.deadline_hours * 3600.0

    def _signal_handler(signum, _frame):
        driver._job(f"received signal {signum} — tearing down")
        driver.teardown_all()
        sys.exit(128 + signum)

    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, _signal_handler)

    try:
        with acquire_lock():
            driver.cleanup_orphans(host_configs)
            results = driver.preflight_all(host_configs)
            healthy = [r for r in results if r.healthy]
            if args.preflight_only:
                for r in results:
                    status = "HEALTHY" if r.healthy else f"EXCLUDED ({r.excluded_reason})"
                    driver._job(f"preflight host={r.config.name}: {status}")
                return 0 if len(healthy) >= config.min_hosts else 3

            if len(healthy) < config.min_hosts:
                driver._job(f"only {len(healthy)}/{config.min_hosts} min_hosts healthy — aborting")
                return 3

            driver.start_all(healthy)
            driver.wait_ready_all(healthy)
            healthy = [r for r in healthy if r.ready]
            if len(healthy) < config.min_hosts:
                driver._job(f"only {len(healthy)}/{config.min_hosts} min_hosts ready after start — aborting")
                return 3

            endpoints = build_endpoints(
                [{"ip": r.ip, "port": r.port, "concurrency": r.concurrency, "name": r.config.name}
                 for r in healthy],
                config.port,
            )
            endpoints_path = driver.log_dir / "endpoints.json"
            endpoints_path.write_text(json.dumps(endpoints, indent=2))
            driver._job(f"endpoints: {endpoints}")

            if args.smoke:
                smoke_cmd = [
                    sys.executable, str(BENCH_CLIENT),
                    "--explainer-url", endpoints[0]["url"],
                    "--pool", "quality", "--n", str(args.smoke_n),
                ]
                proc = subprocess.run(smoke_cmd, cwd=str(ROOT))
                return proc.returncode

            # Rate is the per-host CONFIGURED value (measured at that host's
            # configured concurrency, see ops/lora_macs.json) — matches the
            # design's own arithmetic (§2.5: "4x12,9+10,4 = 62 expl/min").
            # KNOWN LIMITATION: a host downgraded by the RAM rule (c8->c4)
            # keeps its c8-measured rate here; no c4 rate is on file for
            # those hosts. Optimistic in that rare case, not attempted.
            rates = [next(h.rate_per_min for h in config.hosts if h.name == r.config.name)
                    for r in healthy]
            budget_l = compute_backlog_budget(rates, config.budget_hours, backlog_cap)
            driver._job(f"backlog budget L={budget_l} (cap={backlog_cap}, "
                       f"budget_hours={config.budget_hours}, hosts={[r.config.name for r in healthy]})")

            sector_log = driver.log_dir / "lora_gen.log"
            t0 = time.time()
            rc = run_generator(db=args.db, endpoints_json_path=endpoints_path,
                              timeout=args.gen_timeout, sector=sector, order=None,
                              limit=None, max_seconds=None, log_path=sector_log)
            driver._job(f"sector pass rc={rc} elapsed={(time.time() - t0) / 60:.1f}min")

            if rc == 0 and sector is not None and budget_l > 0:
                remaining = max(60.0, deadline - time.time())
                backlog_log = driver.log_dir / "lora_gen_backlog.log"
                rc_bl = run_generator(
                    db=args.db, endpoints_json_path=endpoints_path,
                    timeout=args.gen_timeout, sector=None, order="strong",
                    limit=budget_l, max_seconds=remaining, log_path=backlog_log,
                )
                driver._job(f"backlog pass rc={rc_bl} (non-fatal; job rc stays {rc})")

            return rc
    except LockHeldError as exc:
        driver._job(str(exc))
        return 4
    finally:
        driver.teardown_all()


if __name__ == "__main__":
    sys.exit(main())
