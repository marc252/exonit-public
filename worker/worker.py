#!/usr/bin/env python3
"""ExoNit Worker v2 — headless daemon for distributed transit detection.

Autonomous worker: fetches metadata + FITS from the server and processes
locally. (A legacy base64 time/flux task mode existed until 2026-06-09 but
no production server ever used it.)
"""

# ---- CPU thread cap (CLAUDE.md rule 2: leave at least 25% free) -----------
# Env vars must be set BEFORE numpy / scipy / numba are imported so BLAS and
# OpenMP pools pick them up. Config is read here with only os + json so we
# can honor `cpu_threads` before any heavy import runs.
import os as _os_boot
import json as _json_boot
import sys as _sys_boot


def _boot_apply_cpu_cap():
    cfg_path = "config.json"
    # honor --config even though we haven't set up argparse yet
    for i, a in enumerate(_sys_boot.argv):
        if a == "--config" and i + 1 < len(_sys_boot.argv):
            cfg_path = _sys_boot.argv[i + 1]
            break
        if a.startswith("--config="):
            cfg_path = a.split("=", 1)[1]
            break
    try:
        with open(cfg_path, "r") as fh:
            _cfg = _json_boot.load(fh)
    except Exception:
        return
    n = _cfg.get("cpu_threads")
    if not isinstance(n, int) or n <= 0:
        return
    ns = str(n)
    for var in ("NUMBA_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
                "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS"):
        _os_boot.environ.setdefault(var, ns)


_boot_apply_cpu_cap()


def _boot_enable_rusticl():
    """Expose the AMD iGPU to OpenCL on hosts using Mesa rusticl.

    Mesa's rusticl driver is *off by default* and only surfaces devices
    when ``RUSTICL_ENABLE=<driver>`` is set before pyopencl queries
    platforms. Without this, ``cl.get_platforms()`` returns a rusticl
    platform with **zero devices** and the worker silently falls back
    to CPU for injection / FPP — which is exactly what happened to
    worker-default after a plain ``nohup`` restart 2026-04-20.

    ``setdefault`` respects an explicit user override (any non-empty
    value, including a different driver like ``iris`` for Intel), and
    is a harmless no-op on hosts without rusticl.
    """
    _os_boot.environ.setdefault("RUSTICL_ENABLE", "radeonsi")


_boot_enable_rusticl()


def _boot_start_caffeinate_if_mac():
    """On macOS, spawn a ``caffeinate`` child that prevents idle and
    system sleep while this worker lives.

    The child is bound to our PID via ``-w``, so it exits automatically
    when the worker dies. If the worker is replaced by ``os.execv``
    during autoupdate, the PID is preserved so the caffeinate child
    keeps ticking and a second one is spawned here — harmless duplicate
    with zero idle CPU.

    Detected 2026-04-21: worker-<host> (MacBook Air) stopped polling
    mid-run because the system went to sleep — the worker binary itself
    does not create any power assertion. Adding this at boot ensures
    the protection is a property of the worker, not of the launch
    script.

    No-op on Linux / Windows, and on macOS hosts where ``caffeinate``
    is not on ``PATH`` for some reason.
    """
    if _sys_boot.platform != "darwin":
        return
    try:
        import subprocess
        subprocess.Popen(
            ["caffeinate", "-i", "-s", "-w", str(_os_boot.getpid())],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    except (FileNotFoundError, OSError):
        pass


_boot_start_caffeinate_if_mac()


def _boot_promote_qos_if_mac():
    """On Apple Silicon, clear the Darwin *background* flag so numba
    threads can land on P-cores.

    macOS tags anything launched via ``nohup`` / SSH-bg / launchd as a
    *background* QoS task and the scheduler gates it to the efficiency
    cluster. At the <host> we saw the injection GPU pinned at 100%
    while the P-cores stayed idle and numba FPP inched along on the
    E-cores — killing the benefit of ``cpu_threads=7``. Running
    ``taskpolicy -B -p <pid>`` on a live <host> worker lifted the
    worker from ~3% CPU to ~530% immediately.

    Note on ``-c``: ``taskpolicy -c <class>`` is rejected with
    ``"Could not parse as a QoS clamp"`` when combined with ``-p`` on
    an existing process; the class clamp only applies when taskpolicy
    *launches* the child. The launch wrapper therefore has to live in
    the ``start.sh`` for full QoS-class promotion; here we do the
    subset that works on a running PID.

    No-op on Linux / Windows and on macOS hosts where ``taskpolicy`` is
    missing. Safe to run twice after ``os.execv``.
    """
    if _sys_boot.platform != "darwin":
        return
    try:
        import subprocess
        subprocess.run(
            ["taskpolicy", "-B", "-p", str(_os_boot.getpid())],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        pass


_boot_promote_qos_if_mac()
# ---------------------------------------------------------------------------

import warnings
warnings.filterwarnings("ignore", message=".*urllib3.*OpenSSL.*")
warnings.filterwarnings("ignore", message=".*NotOpenSSLWarning.*")

import argparse
import ctypes
import gc
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
import threading
import queue
import numpy as np
import requests
import logging
from collections import deque
from pathlib import Path

from fits_cache_helper import ensure_fits_present
from wotan_detrend import detrend_with_wotan
# Shared with agent.py's PID-file check (2026-07-02 cleanup review, D13) —
# see procutil.py for the Windows os.execv SystemError edge-case docstring.
from procutil import pid_alive as _pid_alive


def physical_t14_max(period_days, stellar_radius, stellar_mass):
    """Upper physical bound on transit duration T14 for a circular,
    edge-on orbit with a Jupiter-sized planet.

    BLS's ``duration_max_frac=0.15`` treats T14 as a constant fraction of
    the period, which is grossly unphysical at long periods: T14/P scales
    as P^(-2/3), so at P=50d T14 should be ~0.38d for a Sun-like host,
    not 7.5d. Worker uses 3× this bound as a reject threshold — any
    candidate with T14 > 3 · T14_max is noise (2026-04-21 production
    DB showed 15+ candidates at P∈[30,50]d with T14 15-20× physical).

    ``stellar_radius`` and ``stellar_mass`` fall back to solar values when
    the TIC v8 entry is missing them — conservative: overestimates T14_max
    for M-dwarfs so we don't accidentally reject their real transits.
    """
    from injection import transit_duration_days
    return transit_duration_days(
        period_days,
        stellar_radius_solar=stellar_radius if stellar_radius else 1.0,
        stellar_mass_solar=stellar_mass,
        planet_radius_earth=11.2,  # R_Jupiter in R_earth
        impact_parameter=0.0,
    )


def resolve_harmonic(time, flux, result, period_min, period_max, bls_fn,
                     snr_min: float = 5.0,
                     depth_ratio_lo: float = 0.5,
                     depth_ratio_hi: float = 2.0,
                     data_source: str = "spoc",
                     long_period_recovery: bool = False,
                     long_period_min_days: float = 10.0,
                     long_snr_factor: float = 0.5):
    """Symmetric harmonic resolver (audit §76-85).

    Given a seed BLS ``result`` at period P, probe the harmonic
    candidates {5P, 4P, 3P, 2P, P/2, P/3, P/4, P/5} (in this order:
    longest first) and return the LONGEST candidate whose SNR >= snr_min
    and whose depth is within (depth_ratio_lo, depth_ratio_hi) × seed
    depth. If none qualifies, return the seed unchanged.

    The previous resolver only checked subharmonics and broke at the
    first match — when BLS landed on a low harmonic of the true period
    (a known artifact of the freq-grid #9), it pinned the candidate at
    P/N instead of N·P, producing duplicate-period candidates per TIC
    in the multi-planet loop.

    Parameters
    ----------
    time, flux : np.ndarray
        Forwarded to ``bls_fn``; the resolver itself is array-agnostic.
    result : dict
        Seed BLS result with keys 'period', 'snr', 'depth' at minimum.
    period_min, period_max : float
        Hard bounds on candidate periods (inclusive).
    bls_fn : callable
        ``bls_fn(time, flux, period_min=..., period_max=..., n_periods=...)
        -> dict`` — same shape as ``run_bls``.
    snr_min : float
        Minimum SNR for a candidate to qualify.
    depth_ratio_lo, depth_ratio_hi : float
        Open interval (lo, hi) for the candidate-to-seed depth ratio.
    """
    base_period = result["period"]
    base_depth = result.get("depth", 0.0)

    # SPOC: superharmonics restricted to 2x. For periodic signals BLS at
    # N·P (N>=3) tends to find peaks of comparable SNR purely from fold
    # geometry (multiple transits stack per fold), making 3x/4x/5x
    # promotions over-aggressive. The 2x case captures the dominant bug
    # (BLS landing on P/2 of the true transit, e.g. TRAPPIST-1 b) without
    # inviting spurious upward jumps, and the harmonic must be at least as
    # strong as the seed.
    #
    # FFI (30-min cadence, few/noisy transits): BLS locks onto a *sub*-
    # harmonic of the true period far more often, and frequently at
    # P/3..P/5 — beyond the 2x climb SPOC allows. The 2026-05-20 audit of
    # big-run-v3 FFI-only KP/CP found 35/165 detections pinned at a clean
    # P/2..P/5 alias. So for data_source=='ffi' we (a) probe superharmonics
    # up to 5x, (b) relax the "fundamental must be >= alias SNR" rule (the
    # true longer period is genuinely noisier on FFI), guarded by (c) a
    # transit-count floor so the extra reach can't over-promote to an
    # unsupported long period. SPOC behaviour is left bit-identical.
    is_ffi = (data_source == "ffi")
    base_snr = result.get("snr", 0.0)

    # Baseline span for the transit-count guard (skipped when no time array
    # is provided, e.g. unit tests with a mocked bls_fn). Only consulted by
    # probes whose ``guard`` flag is set, so flag-off SPOC is unaffected.
    span = 0.0
    try:
        span = float(time[-1] - time[0])
    except Exception:
        span = 0.0

    # Build the probe list. Each probe = (period, label, snr_factor, guard):
    # snr_factor is the per-probe "candidate must be >= snr_factor*seed_snr"
    # threshold; guard toggles the >=2-transits-in-baseline check. Probes are
    # evaluated in list order and the first qualifying one wins.
    probes = []
    if is_ffi:
        # FFI: BLS locks onto P/2..P/5 sub-harmonics often; probe
        # superharmonics up to 5x with a relaxed SNR rule + transit guard.
        for n in (2, 3, 4, 5):
            cand = base_period * n
            if cand <= period_max:
                probes.append((cand, f"x{n}", 0.6, True))
    else:
        # SPOC: only the 2x superharmonic, at least as strong as the seed
        # (the dominant P/2-lock). Bit-identical to bf24c188 when the
        # long-period recovery below is disabled.
        cand = base_period * 2
        if cand <= period_max:
            probes.append((cand, "x2", 1.0, False))
        # SPOC long-period sub-harmonic recovery (config-gated, default off).
        # At long periods BLS locks onto integer AND rational sub-harmonics
        # (P/3, 2P/3, 2P/5, 3P/5, 3P/4) whose alias outshines the noisier true
        # fundamental — the same mechanism bf24c188 fixed for FFI, but SPOC
        # was left strict. Promote via the inverse factor, with a relaxed SNR
        # rule and the transit-count guard, but ONLY into the long-period
        # regime (>= long_period_min_days) so short-period SPOC detections
        # keep the strict 2x-only behaviour and cannot be over-promoted.
        # Validated 2026-05-22 on big-run-v4 (see scripts/diag_bin_alias.py).
        if long_period_recovery:
            from fractions import Fraction as _Fraction
            # inverse-of-alias factors for the RATIONAL aliases only:
            # 2P/3->3/2, 3P/5->5/3, 3P/4->4/3, 2P/5->5/2. Integer factors
            # (x3, x4, x5) are deliberately EXCLUDED: a real planet's clean
            # integer superharmonic carries comparable fold-geometry SNR, so
            # promoting via x3 over-promotes genuine detections (validated
            # 2026-05-22: x3 caused regressions, e.g. TIC 425997655's P/5
            # alias promoted to 3P/5, with zero offsetting recoveries). The
            # rational factors are not clean superharmonics of a real period,
            # so they only fire on genuine sub-harmonic locks.
            for fac in (_Fraction(3, 2), _Fraction(5, 3), _Fraction(4, 3),
                        _Fraction(5, 2)):
                cand = base_period * float(fac)
                if long_period_min_days <= cand <= period_max:
                    probes.append((cand, f"x{fac}", long_snr_factor, True))
    # Sub-harmonics (demotions): unchanged from bf24c188.
    sub_factor = 0.6 if is_ffi else 1.0
    for n in (2, 3, 4, 5):
        cand = base_period / n
        if cand >= period_min:
            probes.append((cand, f"/{n}", sub_factor, False))

    for cand_period, label, probe_snr_factor, guard in probes:
        try:
            sub = bls_fn(time, flux,
                period_min=cand_period * 0.95,
                period_max=cand_period * 1.05,
                n_periods=500)
        except Exception:
            continue
        cand_snr = sub.get("snr", 0.0)
        if cand_snr < snr_min:
            continue
        # Reject promotions to a weaker peak (per-probe threshold).
        if cand_snr < base_snr * probe_snr_factor:
            continue
        if base_depth > 0:
            depth_ratio = sub.get("depth", 0.0) / base_depth
            if not (depth_ratio_lo < depth_ratio < depth_ratio_hi):
                continue
        # Transit-count guard: a promoted (longer) period must still fit
        # >= 2 transits in the baseline, else the reach is over-promoting.
        if guard and span > 0 and sub.get("period", 0.0) > 0 \
                and span / sub["period"] < 2.0:
            continue
        # First qualifying probe wins.
        return sub
    return result


def _setup_logging(config):
    """Configure logging from worker config.

    When log_file is set, logs go ONLY to file (not terminal).
    When log_file is not set, logs go to stderr as fallback.
    """
    log_file = config.get("log_file", None)
    log_level = getattr(logging, config.get("log_level", "INFO").upper(),
                        logging.INFO)
    handlers = []
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    else:
        handlers.append(logging.StreamHandler(sys.stderr))
    # force=True replaces any handlers that a third-party import (astropy,
    # lightkurve) attached to the root logger before we got here, otherwise
    # basicConfig is a no-op and records still leak to stderr / terminal.
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
        force=True,
    )
    # Suppress noisy third-party loggers
    for name in ("lightkurve", "astroquery", "numexpr", "urllib3"):
        logging.getLogger(name).setLevel(logging.WARNING)

    # lightkurve.utils emits a WARNING-level record on every file read
    # ("Warning: NN% of the cadences will be ignored ...") that isn't
    # actionable — the quality mask is intentional. Previous attempts used
    # (a) a root-handler filter by message pattern, (b) stdout redirect
    # around lk.read(), and (c) warnings.filterwarnings. None of them
    # caught it on Windows where the Rich UI wraps handlers in ways that
    # break substring-filter propagation. Kill it at the source logger:
    # above-CRITICAL level + propagate=False so the record never leaves
    # lightkurve.utils.
    for name in ("lightkurve.utils",):
        lg = logging.getLogger(name)
        lg.setLevel(logging.CRITICAL + 1)
        lg.propagate = False

    return logging.getLogger("exonit")


# --- Algorithm imports (all local to worker/) ---
_worker_dir = os.path.dirname(os.path.abspath(__file__))
if _worker_dir not in sys.path:
    sys.path.insert(0, _worker_dir)

from bls import run_bls, get_backend_name as _bls_backend, get_gpu_info as _bls_gpu_info, is_gpu as _bls_is_gpu, get_hardware_summary as _bls_hardware_summary, get_hardware_info as _bls_hardware_info
from bls_fast import bls_fast, bls_two_pass, HAS_NUMBA as _has_numba
from vetting import vet_candidate
from cnn import TransitClassifier
try:
    from cnn_ffi import FFITransitClassifier
except Exception:
    FFITransitClassifier = None  # FFI model optional

try:
    from worker.server_ffi_client import fetch_ffi
except ImportError:
    from server_ffi_client import fetch_ffi


def _detect_data_source(local_paths) -> str:
    """Return 'spoc' or 'ffi' based on FITS filename patterns.

    SPOC files end with `_lc.fits` and don't have `hlsp_` prefix.
    QLP/TGLC files start with `hlsp_qlp` or `hlsp_tglc`.
    """
    if not local_paths:
        return "spoc"
    paths_lower = [str(p).lower() for p in local_paths]
    has_ffi = any("hlsp_qlp" in p or "hlsp_tglc" in p for p in paths_lower)
    has_spoc = any(p.endswith("_lc.fits") and "hlsp" not in p for p in paths_lower)
    if has_ffi and not has_spoc:
        return "ffi"
    return "spoc"


def _read_lightcurve(path):
    """Read a TESS light curve into a lightkurve object.

    TGLC FFI light curves store the calibrated flux in ``cal_aper_flux`` (with
    quality columns ``TESS_flags``/``TGLC_flags``), which lightkurve cannot
    auto-detect. Read those by hand; fall back to ``lk.read`` for SPOC/QLP.
    """
    import lightkurve as lk
    from astropy.io import fits as _fits
    try:
        with _fits.open(path, memmap=False) as hdul:
            cols = getattr(hdul[1], "columns", None) if len(hdul) > 1 else None
            names = {c.name.lower() for c in cols} if cols is not None else set()
            if "cal_aper_flux" in names:
                d = hdul[1].data
                t = np.asarray(d["time"], dtype=np.float64)
                f = np.asarray(d["cal_aper_flux"], dtype=np.float64)
                qa = (np.asarray(d["TESS_flags"], dtype=np.int64)
                      if "tess_flags" in names else np.zeros(len(t), np.int64))
                qb = (np.asarray(d["TGLC_flags"], dtype=np.int64)
                      if "tglc_flags" in names else np.zeros(len(t), np.int64))
                good = ((qa == 0) & (qb == 0)
                        & np.isfinite(t) & np.isfinite(f) & (f > 0))
                return lk.LightCurve(time=t[good], flux=f[good])
    except Exception:
        pass
    return lk.read(path)


from injection import run_injection_recovery

# FPP: prefer fpp_native (no triceratops/pytransit dependency)
_triceratops_available = False

class FPPAborted(Exception):
    """Fallback definition if fpp_native not available."""
    pass

try:
    from fpp_native import calc_fpp_native as calc_fpp, should_run_triceratops
    from fpp_native import FPPAborted  # override fallback with real class
    _triceratops_available = True
except ImportError as _fpp_e:
    logging.getLogger("exonit").warning(
        "fpp_native unavailable (%s) — trying legacy triceratops_fpp", _fpp_e)
    try:
        from triceratops_fpp import calc_fpp, should_run_triceratops
        import triceratops
        _triceratops_available = True
    except ImportError as _tri_e:
        # Never silent: without FPP every candidate ships without false-positive
        # probability — a scientific degradation that must scream in the log.
        logging.getLogger("exonit").error(
            "NO FPP BACKEND AVAILABLE (fpp_native: %s; triceratops_fpp: %s) — "
            "worker will run WITHOUT TRICERATOPS/FPP", _fpp_e, _tri_e)

# FPP GPU detection
_fpp_gpu_backend = None
try:
    from fpp_native import get_fpp_backend_info
    _fpp_gpu_backend, _ = get_fpp_backend_info()
except Exception:
    pass


_WORKER_GUI_HTML = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>ExoNit Worker</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{background:#0a0e1a;color:#e0e0e0;font-family:system-ui,-apple-system,sans-serif;padding:20px}
h1{color:#4fc3f7;font-size:1.5em;margin-bottom:15px}
.card{background:#141b2d;border-radius:8px;padding:16px;margin-bottom:12px;border:1px solid #1e2a45}
.stat{display:inline-block;margin-right:24px;margin-bottom:8px}
.stat .label{color:#888;font-size:0.8em;text-transform:uppercase}
.stat .value{font-size:1.4em;font-weight:bold;color:#4fc3f7}
.status-working{color:#ffa726}.status-idle{color:#66bb6a}
table{width:100%;border-collapse:collapse;margin-top:8px}
th,td{padding:6px 10px;text-align:left;border-bottom:1px solid #1e2a45}
th{color:#888;font-size:0.8em;text-transform:uppercase}
#current{font-size:1.2em;color:#ffa726}
</style></head><body>
<h1>ExoNit Worker</h1>
<div class="card" id="info"></div>
<div class="card"><h3 style="margin-bottom:8px">Recent Candidates</h3><table id="candidates">
<tr><th>TIC ID</th><th>Period (d)</th><th>SNR</th><th>Depth (ppm)</th></tr></table></div>
<script>
async function update(){
  try{
    const r=await fetch('/api/status');const d=await r.json();
    document.getElementById('info').innerHTML=`
      <div class="stat"><div class="label">Worker</div><div class="value">${d.name}</div></div>
      <div class="stat"><div class="label">Status</div><div class="value status-${d.status}">${d.status}</div></div>
      <div class="stat"><div class="label">Current</div><div class="value" id="current">${d.current_tic||'-'}</div></div>
      <div class="stat"><div class="label">Stars Done</div><div class="value">${d.stars_completed}</div></div>
      <div class="stat"><div class="label">Avg Time</div><div class="value">${d.avg_time_per_star}s</div></div>
      <div class="stat"><div class="label">GPU</div><div class="value">${d.gpu_enabled?'ON':'OFF'}</div></div>
    `;
    const tb=document.getElementById('candidates');
    let rows='<tr><th>TIC ID</th><th>Period (d)</th><th>SNR</th><th>Depth (ppm)</th></tr>';
    (d.last_candidates||[]).reverse().forEach(c=>{
      rows+=`<tr><td>${c.tic_id}</td><td>${c.period?.toFixed(4)||'-'}</td><td>${c.snr?.toFixed(1)||'-'}</td><td>${c.depth_ppm?.toFixed(0)||'-'}</td></tr>`;
    });
    tb.innerHTML=rows;
  }catch(e){}
}
update();setInterval(update,3000);
</script></body></html>"""


def _resolve_git_sha() -> str:
    """Best-effort git SHA of the worker checkout. Returns "unknown" when the
    worker is running from a non-git deploy (rsync) or the SHA cannot be read.
    Memoised on the function object to avoid spawning a subprocess per heartbeat.
    """
    cached = getattr(_resolve_git_sha, "_cached", None)
    if cached is not None:
        return cached
    sha = "unknown"
    try:
        worker_dir = os.path.dirname(os.path.abspath(__file__))
        result = subprocess.run(
            ["git", "rev-parse", "--short=8", "HEAD"],
            cwd=worker_dir, capture_output=True, text=True, timeout=2)
        if result.returncode == 0:
            sha = result.stdout.strip() or "unknown"
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass
    _resolve_git_sha._cached = sha
    return sha


def _resolve_flag_states(config: dict) -> dict:
    """Snapshot of feature-flag values the heartbeat should publish so farmctl
    status can show what each worker has enabled. Add new flags here as they
    are introduced — keep the dict small (these go in every heartbeat)."""
    keys = (
        "ffi_fallback_on_load_fail",
        "exominer_enabled",
        "exominer_mode",
        "gpu_enabled",
        "run_injection",
        "run_fpp",
    )
    return {k: config.get(k) for k in keys if k in config}


# ---------------------------------------------------------------------------
# GPU telemetry helpers — module-level so they work as plain functions
# regardless of what class `self` is (existing tests build a MinimalWorker
# stand-in for ExoNitWorker that only rebinds a handful of named methods;
# a helper that needed to be looked up as `self.<method>` would AttributeError
# on those instances unless every test was updated too).
# ---------------------------------------------------------------------------

def _friendly_gpu_name():
    """Human-meaningful GPU name for the fleet table (Status page/register()).

    Before this fix, every CPU-only host (worker-HP, most macOS/Linux boxes
    without a usable accelerator) registered its bare BLS backend word as
    the "GPU name" (`_bls_gpu_info() or _bls_backend()` -> "numba" when
    there's no accelerator) -- the dashboard then showed "numba (numba)",
    which reads as a GPU model, not "there is no GPU here". If BLS (or one
    of its backends' own device-name lookups) found a real, named
    accelerator, use that; otherwise say plainly that this is a CPU-only
    host and which BLS backend is doing the work.

    Called once, from register() at worker startup -- NOT per-heartbeat --
    so no caching is needed here beyond that natural call frequency."""
    info = _bls_gpu_info()
    if info:
        return info
    backend = _bls_backend() or "none"
    if backend in ("numba", "numpy"):
        return f"CPU only ({backend})"
    return backend


def _warn_telemetry_once(worker, key, msg, level="warning"):
    """Log *msg* the first time this *key* is seen on *worker*'s process,
    then stay silent for the rest of the run — a degraded telemetry probe
    must be VISIBLE (project rule 2's spirit: a GPU host quietly not
    reporting is the same failure mode as quietly running on CPU) without
    spamming a line into the log every ~30s heartbeat forever. Never raises:
    a broken warn-once call must not be the thing that breaks the heartbeat
    it exists to protect, so every step here is best-effort."""
    try:
        warned = getattr(worker, "_telemetry_warned", None)
        if warned is None:
            warned = set()
            try:
                worker._telemetry_warned = warned
            except Exception:
                pass
        if key in warned:
            return
        warned.add(key)
        logger = getattr(worker, "logger", None)
        if logger is not None:
            getattr(logger, level, logger.warning)(f"[telemetry] {msg}")
    except Exception:
        pass


def _safe_read_sysfs(path):
    """Read a sysfs attribute file, or None if missing/unreadable for any
    reason. Generic "file might not exist on this driver/vendor" reads go
    through this; the one place that needs to tell a missing file apart
    from a PermissionError (RAPL's energy_uj, see
    ExoNitWorker._linux_rapl_power_w) reads directly instead, on purpose."""
    try:
        with open(path) as f:
            return f.read().strip()
    except Exception:
        return None


_PCI_VENDOR_AMD = "0x1002"
_PCI_VENDOR_INTEL = "0x8086"

# macmon (https://github.com/vladkens/macmon) ships as a single static
# binary via `brew install macmon` or `cargo install macmon` — but neither
# Homebrew nor cargo can be assumed present (confirmed 2026-07-25: `which
# brew` finds nothing on worker-<host> or the M1 <host>, and installing
# Homebrew itself needs sudo to create /opt/homebrew — not something to ask
# of a machine that isn't ours). The realistic remedy on those hosts is
# dropping a prebuilt static binary into a user-writable directory, so this
# lookup list includes the no-admin-required spots, not just the
# package-manager ones.
_MACMON_SEARCH_DIRS = (
    "~/bin",
    "~/.local/bin",
    "/opt/homebrew/bin",
    "/usr/local/bin",
    "~/.cargo/bin",
)


# --- Windows PDH (Performance Data Helper) — pure parsing, no ctypes ---
# Split out from the ctypes-calling code below so the aggregation logic is
# testable on any OS (these run in CI on Linux) even though the actual PDH
# API calls can only be exercised for real on Windows.

def _pdh_engine_type(instance_name):
    """Extract the engine type suffix from a `\\GPU Engine(*)` PDH instance
    name, e.g. 'pid_1234_luid_0x0_0x1_phys_0_eng_0_engtype_3D' -> '3D'.
    Pure string parsing, no ctypes."""
    if not instance_name:
        return None
    marker = "engtype_"
    idx = instance_name.rfind(marker)
    if idx == -1:
        return None
    return instance_name[idx + len(marker):]


def _pdh_instance_from_path(counter_path):
    """Pull the instance name out of a concrete (post wildcard-expansion)
    PDH counter path, e.g. '\\GPU Engine(pid_123_..._engtype_3D)\\Utilization
    Percentage' -> 'pid_123_..._engtype_3D'."""
    start = counter_path.find("(")
    end = counter_path.rfind(")")
    if start == -1 or end == -1 or end <= start:
        return counter_path
    return counter_path[start + 1:end]


def _pdh_aggregate_engine_util(samples):
    """samples: {instance_name: percent_float}. Task Manager's headline GPU%
    is, per engine type, the SUM of every process's share of that engine
    (each `\\GPU Engine(*)` instance is one (pid, engine) pair) — the number
    actually shown is then the MAX across engine types (3D is normally
    busiest on this fleet's BLS/injection workload, but that isn't
    hardcoded — a kernel could get classified under Compute instead
    depending on the backend Windows picks). None values (a counter that
    failed to format) are skipped rather than treated as 0. Returns None
    for an empty/all-invalid sample set — that means "no data yet", not
    "0% busy"."""
    if not samples:
        return None
    by_type = {}
    for name, value in samples.items():
        if value is None:
            continue
        etype = _pdh_engine_type(name)
        if etype is None:
            continue
        by_type[etype] = by_type.get(etype, 0.0) + value
    if not by_type:
        return None
    return min(100.0, max(by_type.values()))


def _pdh_aggregate_adapter_memory_gb(samples):
    """samples: {instance_name: bytes_float} from `\\GPU Adapter Memory(*)
    \\Dedicated Usage`. Sums every adapter instance (normally one, for a
    single-iGPU laptop) and converts to GB. None for an empty sample set —
    distinct from a real 0 B reading."""
    if not samples:
        return None
    values = [v for v in samples.values() if v is not None]
    if not values:
        return None
    return sum(values) / (1024 ** 3)


# --- Windows PDH — the actual ctypes-calling layer ---
# Kept separate from the pure functions above (and from ExoNitWorker's
# instance methods, which own the persistent query/counter handles) so a
# test can mock exactly this boundary: `ctypes.windll` doesn't exist as an
# attribute on non-Windows platforms, so tests patch it in with
# `patch("ctypes.windll", create=True)` and drive these two functions
# directly.

_PDH_FMT_DOUBLE = 0x00000200
_PDH_CSTATUS_VALID_DATA = 0x00000000
_PDH_CSTATUS_NEW_DATA = 0x00000001


class _PdhFmtCounterValue(ctypes.Structure):
    """Minimal PDH_FMT_COUNTERVALUE — only the CStatus + doubleValue arms
    of the real union are modelled since PdhGetFormattedCounterValue is
    always called with dwFormat=PDH_FMT_DOUBLE below."""
    _fields_ = [("CStatus", ctypes.c_ulong), ("doubleValue", ctypes.c_double)]


def _pdh_expand_wildcard(pdh, wildcard_path):
    """Expand a wildcard PDH counter path (e.g. '\\GPU Engine(*)\\...') into
    the list of concrete per-instance paths currently active. Two-call PDH
    idiom: first call with a null buffer to learn the required size, second
    call to actually fill it. Returns [] on any failure — the caller
    treats "found nothing" as "nothing to add", not an error."""
    size = ctypes.c_ulong(0)
    pdh.PdhExpandWildCardPathW(None, wildcard_path, None, ctypes.pointer(size), 0)
    if size.value == 0:
        return []
    buf = ctypes.create_unicode_buffer(size.value)
    rc = pdh.PdhExpandWildCardPathW(None, wildcard_path, buf, ctypes.pointer(size), 0)
    if rc != 0:
        return []
    # buf holds a MULTI_SZ: concrete paths separated by, and doubly
    # terminated by, NUL.
    raw = ctypes.wstring_at(ctypes.addressof(buf), size.value)
    return [p for p in raw.split("\x00") if p]


def _pdh_read_counter_double(pdh, counter_handle):
    """Format one already-collected counter handle as a double. Returns
    None if PDH reports the call failed OR the value isn't valid yet (the
    rate-counter warm-up sample, see _pdh_ensure_query) — never a bogus
    0.0 standing in for "no data"."""
    value = _PdhFmtCounterValue()
    rc = pdh.PdhGetFormattedCounterValue(counter_handle, _PDH_FMT_DOUBLE, None, ctypes.pointer(value))
    if rc != 0:
        return None
    if value.CStatus not in (_PDH_CSTATUS_VALID_DATA, _PDH_CSTATUS_NEW_DATA):
        return None
    return value.doubleValue


# ---------------------------------------------------------------------------
# CUDA context-poison classifier
# ---------------------------------------------------------------------------

# Signatures that indicate the CUDA context is irrecoverably poisoned and the
# worker process must be restarted to recover.  OOM errors are deliberately
# excluded: ``cudaErrorMemoryAllocation`` / "out of memory" are transient
# (the allocation can be retried or a smaller model used) and do NOT require
# a restart — they must NOT appear here.
_CUDA_POISON_SIGNATURES = (
    "cudaerrorlaunchfailure",
    "unspecified launch failure",
    "cudaerrorillegaladdress",
    "an illegal memory access",
    "cuda_error_launch_failed",
    "cuda_error_illegal_address",
    "cudaerroreccuncorrectable",
)


def _is_cuda_context_error(exc: BaseException) -> bool:
    """Return True when *exc* signals an irrecoverable CUDA context error.

    The check is case-insensitive substring match on ``str(exc)`` so it
    catches errors wrapped at any layer (numba, PyTorch, CuPy, pycuda…).

    Transient errors such as ``cudaErrorMemoryAllocation`` and
    "out of memory" are intentionally **not** listed — those do not
    poison the context and do not warrant a process restart.
    """
    msg = str(exc).lower()
    return any(sig in msg for sig in _CUDA_POISON_SIGNATURES)


# ---------------------------------------------------------------------------
# Single-flight prefetch (D3, 2026-07-02 cleanup review)
# ---------------------------------------------------------------------------

class _PrefetchSlot:
    """Generic single-flight, maxsize=1 prefetch slot.

    Extracted from the FPP backfill (``_fetch_and_load_fpp`` /
    ``_enqueue_next_fpp`` / ``_try_fpp_backfill``) and ExoMiner backfill
    (``_fetch_and_prepare_exominer`` / ``_enqueue_next_exominer`` /
    ``_try_exominer_backfill``) pipelines, which were the same ~80-line
    pattern copy-pasted: reserve the next task at the server and pre-load
    its heavy inputs (light curve / DV+FITS) on a background thread while
    the main loop is still working on the previous item, so the cold-cache
    I/O overlaps with compute instead of stalling it (profiled 2026-05-16:
    ~2x wallclock reduction in steady state).

    Single-flight via a non-blocking lock: if a producer thread is still
    running, a second call to ``enqueue_next_bg`` is a no-op rather than
    racing for a duplicate server-side reservation — a real 2026-05-16
    incident inflated in-flight FPP/ExoMiner assignments to ~2.7x the
    worker count because two threads each reserved+downloaded for the
    same slot and the second ``put`` silently lost the race, orphaning a
    reservation until the 30-min stale reaper.

    This class is deliberately behavior-neutral: it holds only the
    fetch/queue/lock plumbing that was byte-identical between the two
    pipelines. Anything that differs between FPP and ExoMiner (the extra
    ``_exominer_backfill_available()`` gate, and what "release this
    reservation" means — FPP reports failed, ExoMiner also removes the
    prefetched tmp_root before reporting skipped) stays at the call site,
    passed in as ``fetch_fn`` / ``on_skip``.
    """

    def __init__(self, fetch_fn, on_skip, maxsize=1):
        """
        Parameters
        ----------
        fetch_fn : callable() -> dict | None
            Reserves the next task at the server and pre-loads its heavy
            inputs. Called both synchronously (on a queue miss in
            ``take()``) and from the background producer thread.
        on_skip : callable(prepared) -> None
            Invoked when a background-prefetched item can't be enqueued
            because the queue is already full (main loop grabbed one via
            ``take()`` in the meantime) — must release/report the
            reservation server-side so it doesn't sit stale. Exceptions
            are swallowed (best-effort cleanup, mirrors the pre-extraction
            behaviour).
        """
        self._fetch_fn = fetch_fn
        self._on_skip = on_skip
        self._q = queue.Queue(maxsize=maxsize)
        self._lock = threading.Lock()

    def _enqueue_next(self):
        """Background producer. Single-flight via tryacquire(blocking=False)."""
        if not self._lock.acquire(blocking=False):
            return
        try:
            if self._q.full():
                return
            prepared = self._fetch_fn()
            if prepared is None:
                return
            try:
                self._q.put_nowait(prepared)
            except queue.Full:
                # Main loop grabbed another task in the meantime — release
                # our reservation so it doesn't sit stale.
                try:
                    self._on_skip(prepared)
                except Exception:
                    pass
        finally:
            self._lock.release()

    def take(self):
        """Drain the queue (non-blocking); on a miss, fetch synchronously.

        Returns the prepared dict, or None if nothing is available. The
        caller is responsible for calling ``enqueue_next_bg()`` once it
        has the item in hand, before starting the (slow) processing step
        — mirrors the pre-extraction ``_try_*_backfill`` structure exactly.
        """
        try:
            return self._q.get_nowait()
        except queue.Empty:
            return self._fetch_fn()

    def enqueue_next_bg(self):
        """Spawn a daemon thread to prefetch the next item."""
        threading.Thread(target=self._enqueue_next, daemon=True).start()


class ExoNitWorker:
    # Class-level default so partially-built instances (tests construct via
    # ExoNitWorker.__new__) don't AttributeError in process_task; __init__
    # overrides it after probing the FPP Metal kernel.
    _fpp_prefers_gpu = False

    def __init__(self, config_path="config.json", worker_name_override=None,
                 server_url_override=None):
        with open(config_path) as f:
            self.config = json.load(f)
        # CLI --worker-name / --server-url take precedence over config.json.
        # This decouples per-host identity from the config file so that
        # rsync/git pull never clobbers it (a recurring problem in farm-wide
        # deploys).
        if worker_name_override:
            self.config["worker_name"] = worker_name_override
        if server_url_override:
            self.config["server_url"] = server_url_override
        self.logger = _setup_logging(self.config)
        # PID lock — refuse to start if another worker with the same
        # worker_name is already running on this host. Without this,
        # repeated `nohup python worker.py` or schtasks /Run create
        # zombie processes that share the worker_name and fight for
        # tasks/CPU (observed <host> 3-process zombie on 2026-04-21:
        # each TIC took 3× longer due to contention; see CLAUDE.md
        # rule 2 "leave 25% CPU free" — zombies burn 300% instead).
        self._acquire_pid_lock(config_path)
        self.started_at = time.time()
        self.server = self.config["server_url"].rstrip("/")
        self.token = self.config["token"]
        self.name = self.config.get("worker_name", "worker")
        self.poll_interval = self.config.get("poll_interval_seconds", 5)
        self.hb_interval = self.config.get("heartbeat_interval_seconds", 30)
        self.gpu_enabled = self.config.get("gpu_enabled", False)
        self.fpp_cpu_cores = self.config.get("fpp_cpu_cores", None)
        # Apply global CPU-thread cap (post-import backstop). Env vars set in
        # the boot block at module top handle BLAS/OpenMP; numba exposes its
        # thread count via set_num_threads and must be called after import.
        cpu_cap = self.config.get("cpu_threads")
        if isinstance(cpu_cap, int) and cpu_cap > 0:
            try:
                import numba
                numba.set_num_threads(cpu_cap)
                self.logger.info(f"CPU threading cap applied: {cpu_cap} threads (numba)")
            except Exception as e:
                self.logger.warning(f"numba.set_num_threads failed: {e}")
            # HARD RULE (project rule 2): warn (don't abort — an explicit
            # config value is authoritative) when the configured cap itself
            # exceeds 75% of the host's CPUs. Mirrors _check_gpu_consistency
            # below: one loud message rather than a silent violation (B7,
            # 2026-07-02 cleanup review — the *default* was already fixed
            # to floor(0.75*total) in fpp_native._configure_fpp_cpu_cores;
            # this catches an explicit override that re-introduces the
            # same violation).
            _cpu_budget = max(1, math.floor(0.75 * (os.cpu_count() or 4)))
            if cpu_cap > _cpu_budget:
                self.logger.warning(
                    "CPU CONSISTENCY: configured cpu_threads=%d exceeds 75%% "
                    "of this host's %d CPUs (budget=%d) — project rule 2 "
                    "requires >=25%% free; lower cpu_threads in config.json",
                    cpu_cap, os.cpu_count() or 4, _cpu_budget)
        # Always configure FPP cores — default leaves 25% of cores free for
        # the system (floor(0.75*total), same formula as agent.py).
        # Note: cpu_threads sets NUMBA_NUM_THREADS at boot (hard cap). If the
        # user configures fpp_cpu_cores above cpu_threads, _configure_fpp_cpu_cores
        # silently clamps and we surface the clamp here so it isn't mysterious.
        try:
            from fpp_native import _configure_fpp_cpu_cores
            requested = self.fpp_cpu_cores or max(1, math.floor(0.75 * (os.cpu_count() or 4)))
            effective = _configure_fpp_cpu_cores(self.fpp_cpu_cores)
            if effective is None:
                self.logger.info("FPP CPU cores: numba unavailable")
            elif effective < requested:
                self.logger.info(
                    f"FPP CPU cores: {effective} "
                    f"(requested {requested}, capped by cpu_threads={cpu_cap})")
            else:
                self.logger.info(f"FPP CPU cores: {effective}")
        except Exception as e:
            self.logger.warning(f"FPP CPU core config failed: {e}")

        # When GPU is disabled in config (the worker-default case for the
        # AMD Vega iGPU host, where long OpenCL kernels trigger amdgpu TDR
        # and take down the process), ensure injection and FPP also honour
        # the flag — both modules auto-detect OpenCL at import time and
        # would otherwise ignore the config.
        # Which compute unit runs FPP.
        #   - Apple Silicon / metal-kernel: FPP on GPU (~5x faster than numba).
        #     Injection: on hosts where the fused Metal kernel
        #     (bls_metal_batch, validated 2026-07-11 on M1/M3/M4 — see
        #     docs/superpowers/specs/2026-07-11-gpu-bls-f32-unified-kernel.md)
        #     is active, injection ALSO runs on the GPU concurrently with FPP
        #     (measured overlap 1.09-1.16x, parity exact, 4/4 contention
        #     rounds clean on every SoC bench). Otherwise (legacy bls_metal
        #     per-flux path, SoC not allowlisted, or EXONIT_INJECTION_METAL=0)
        #     injection runs on CPU concurrently with FPP, as before.
        #   - CUDA (<host>, <host>): FPP on GPU (fpp_cuda, 21-31x faster than
        #     CPU-2-threads per 2026-06-10 bench) AND injection on GPU concurrently —
        #     both fit the CUDA GPU; injection.disable_gpu() is NOT called so CUDA
        #     injection keeps its GPU path.
        #   - Any other / no GPU backend: legacy split (injection GPU, FPP CPU).
        # Kill-switch: EXONIT_FPP_GPU_BACKENDS=none (disables FPP GPU backend in
        # fpp_native._init_fpp_gpu) → get_fpp_backend_info() returns "cpu" →
        # _fpp_prefers_gpu stays False → behaviour identical to the no-GPU case.
        self._fpp_prefers_gpu = False
        if not self.gpu_enabled:
            try:
                import injection
                injection.disable_gpu()
                self.logger.info("injection GPU disabled by config (gpu_enabled=false)")
            except Exception as e:
                self.logger.warning(f"injection.disable_gpu failed: {e}")
            try:
                from fpp_native import disable_gpu as _fpp_disable_gpu
                _fpp_disable_gpu()
                self.logger.info("fpp_native GPU disabled by config (gpu_enabled=false)")
            except Exception as e:
                self.logger.warning(f"fpp_native.disable_gpu failed: {e}")
        else:
            try:
                from fpp_native import get_fpp_backend_info
                _fpp_backend_name = get_fpp_backend_info()[0]
                if _fpp_backend_name == "metal-kernel":
                    # Apple Silicon: FPP always gets the GPU. Injection stays on
                    # the GPU too iff the fused Metal kernel resolved as the
                    # active injection backend (bls_metal_batch imports first in
                    # injection.py's auto-detect and self-gates on SoC + the
                    # EXONIT_INJECTION_METAL kill-switch) — the 2026-07-11 bench
                    # validated FPP(Metal)||injection(Metal) contention as clean
                    # (4/4 rounds, parity exact) on every Apple Silicon SoC in
                    # the fleet. Otherwise fall back to the pre-existing split
                    # (injection.disable_gpu() -> CPU) unchanged.
                    import injection
                    self._fpp_prefers_gpu = True
                    if injection.is_fused_metal_active():
                        self.logger.info(
                            "Apple Silicon (Metal-4): FPP on Metal kernel (GPU) + "
                            "injection on Metal fused kernel (GPU, concurrent) — "
                            "validated 1.16x overlap, parity exact")
                    else:
                        injection.disable_gpu()
                        self.logger.info(
                            "Apple Silicon: FPP on Metal kernel (GPU), injection on CPU "
                            "(concurrent) — gpu_enabled=true")
                elif _fpp_backend_name == "cuda":
                    # CUDA: both FPP and injection share the GPU — injection.disable_gpu()
                    # is intentionally NOT called so CUDA injection keeps its GPU path.
                    self._fpp_prefers_gpu = True
                    self.logger.info(
                        "CUDA: FPP on GPU (fpp_cuda) + injection on GPU — gpu_enabled=true")
                elif _fpp_backend_name == "opencl":
                    # OpenCL (<host>, Intel Arc iGPU, spec 2026-07-24): mirrors the
                    # CUDA branch above, not the Metal one — OpenCL only has ONE
                    # injection design (the fused kernel in bls_opencl.py, no
                    # legacy per-flux fallback to distinguish), so there is no
                    # is_fused_metal_active()-style check to make: FPP and
                    # injection just share the GPU, and injection.disable_gpu()
                    # is intentionally NOT called. Before this branch existed,
                    # gpu_enabled=true + FPP resolving to "opencl" left this
                    # elif/else chain silently falling through with no log line
                    # and no explicit policy — injection kept whatever backend
                    # injection.py's own auto-detect chain had already picked
                    # (also OpenCL, independently), so the two GPU submitters
                    # (FPP on the main thread, injection on a background thread)
                    # could race on PyOpenCL's shared context/queue exactly like
                    # the 2026-07-12 MLX/Metal incident. The fix is NOT here —
                    # it is opencl_serial.opencl_guard(), the single process-wide
                    # RLock that bls_opencl.py and fpp_opencl.py both submit
                    # through — this branch just makes the shared-GPU policy
                    # explicit and logged, the same way the CUDA branch does.
                    self._fpp_prefers_gpu = True
                    self.logger.info(
                        "OpenCL (Intel Arc iGPU): FPP on GPU (fpp_opencl) + "
                        "injection on GPU (concurrent, opencl_serial.opencl_guard() "
                        "serialized) — gpu_enabled=true")
            except Exception as e:
                self.logger.warning(f"FPP/injection GPU assignment failed: {e}")
        self.logger.info(f"exominer_mode={self.config.get('exominer_mode', 'legacy-tfrecord')}")
        self.fits_cache_dir = self.config.get("fits_cache_dir", "fits_cache")
        self.worker_id = None
        self.status = "idle"
        self.current_tic = None
        self.running = True
        self.paused = False
        self.stars_completed = 0
        self.total_processing_time = 0
        self.last_candidates = []
        # Telemetry
        self.current_phase = None
        self.phase_progress = None
        self._star_start_time = None
        self.last_error = None
        self.debug_mode = False
        self.recent_timings = deque(maxlen=5)
        self.gui_port = self.config.get("gui_port", 8401)
        # Snapshot of the sha256 (line-ending-normalized) of every file in
        # worker_dir AS LOADED at process start. Used by check_for_updates()
        # to detect the "on-disk current but loaded code stale" race: when
        # multiple worker instances share a worker_dir, the first instance
        # that sees a new manifest downloads it and execv-restarts; the
        # OTHER instances then see disk already matching the remote hash
        # (nothing left `to_update`) and would otherwise never restart,
        # silently running stale in-memory code forever (confirmed
        # 2026-07-10 on worker-<host>: instances 1-2 restarted,
        # instances 3-4 stuck on old code). See check_for_updates().
        self._loaded_hashes = self._snapshot_worker_dir_hashes()
        # CUDA context-poison auto-restart.  Three consecutive tasks that fail
        # with an irrecoverable CUDA error (see _is_cuda_context_error) trigger
        # os.execv self-restart.  Non-CUDA errors leave the streak unchanged —
        # a mix of CUDA and unrelated failures still eventually hits the threshold.
        # A restart-lineage counter is passed via EXONIT_CUDA_RESTART_COUNT so we
        # stop trying after 3 restarts (the GPU/driver is truly dead; we stay up
        # and visible in the server rather than looping endlessly).
        self._cuda_error_streak = 0
        try:
            self._cuda_restart_count = int(
                os.environ.get("EXONIT_CUDA_RESTART_COUNT", "0")
            )
        except (ValueError, TypeError):
            self._cuda_restart_count = 0
        # Anti-loop guard for the "loaded code stale" self-restart in
        # check_for_updates() (see EXONIT_CUDA_RESTART_COUNT above for the
        # established pattern). If a manifest/snapshot key mismatch (or any
        # other bug) makes loaded_code_stale=True forever with nothing left
        # to download, execv would restart the process every poll interval
        # indefinitely (2026-07-10 incident on worker-<host>). Track how
        # many consecutive "stale, nothing to update" restarts this process
        # lineage has performed; stop restarting past the threshold. A real
        # download (to_update non-empty) resets this counter to 0 before its
        # own execv, since that is a legitimate code change, not a loop.
        try:
            self._update_restart_count = int(
                os.environ.get("EXONIT_UPDATE_RESTART_COUNT", "0")
            )
        except (ValueError, TypeError):
            self._update_restart_count = 0
        # FPP + ExoMiner backfill prefetch pipelines (D3, 2026-07-02
        # cleanup review — see _PrefetchSlot docstring for the full
        # rationale). Maxsize=1 so a worker never sits on more than one
        # reservation ahead (otherwise sibling workers starve); the
        # single-flight lock prevents the 2026-05-16 incident where two
        # producer threads double-reserved the same slot and inflated
        # in-flight assignments to ~2.7x the worker count.
        #
        # FPP: on a lost race, just report the reservation as failed so it
        # frees server-side. fetch_fn/on_skip are wrapped in lambdas (rather
        # than passing the bound methods directly) so attribute lookup on
        # ``self`` stays late-bound — tests patch e.g.
        # ``patch.object(w, "_fetch_and_load_fpp")`` after construction,
        # which only works if the slot re-resolves the attribute on every
        # call instead of freezing the bound method captured here.
        self._fpp_prefetch_slot = _PrefetchSlot(
            fetch_fn=lambda: self._fetch_and_load_fpp(),
            on_skip=lambda prepared: self._report_fpp_failed(prepared["task"]),
        )
        # ExoMiner: same shape, but a lost race also has a local tmp dir
        # (downloaded DV XML + TICv8 + Gaia + FITS) that must be cleaned up
        # before reporting skipped — see _on_exominer_prefetch_skip.
        self._em_prefetch_slot = _PrefetchSlot(
            fetch_fn=lambda: self._fetch_and_prepare_exominer(),
            on_skip=lambda prepared: self._on_exominer_prefetch_skip(prepared),
        )

        # Optional pool whitelist: restrict which task queues this worker
        # polls. Default (key absent / None) preserves the historical
        # behaviour of polling all three pools.
        #   "primary"            — /api/worker/task (BLS + CNN + FPP + ...)
        #   "fpp_backfill"       — /api/worker/fpp_task
        #   "exominer_backfill"  — /api/worker/exominer_task
        # Use case: cheap CPU-only Spot workers should NOT poll the primary
        # queue because BLS+TRICERATOPS on 1-vCPU c6g.medium runs 10-20×
        # slower than on a g4dn T4 — they'd starve the GPU fleet of work.
        # Set ``"task_pools": ["fpp_backfill", "exominer_backfill"]`` in
        # their config so the GPU pool keeps the primary run to itself.
        raw_pools = self.config.get("task_pools")
        if raw_pools is None:
            self.task_pools = None
        else:
            valid = {"primary", "fpp_backfill", "exominer_backfill"}
            pools = {p for p in raw_pools if p in valid}
            unknown = set(raw_pools) - valid
            if unknown:
                self.logger.warning(
                    f"task_pools: ignoring unknown pool(s) {sorted(unknown)}; "
                    f"valid values are {sorted(valid)}")
            if not pools:
                self.logger.warning(
                    "task_pools resolved to empty set after filtering; "
                    "worker will not poll any queue. Did you mean to omit "
                    "the key entirely?")
            self.task_pools = pools
            self.logger.info(f"task_pools restricted to {sorted(pools)}")

        # Create FITS cache directory
        os.makedirs(self.fits_cache_dir, exist_ok=True)

        try:
            self.classifier = TransitClassifier(use_gpu=self.gpu_enabled)
            self.logger.info(f"CNN classifier loaded (GPU: {self.gpu_enabled})")
        except Exception as e:
            self.logger.warning(f"CNN not available: {e}")
            self.classifier = None

        # Optional FFI classifier (cnn_ffi_v1.onnx) — used when data_source=='ffi'.
        self.classifier_ffi = None
        if FFITransitClassifier is not None:
            try:
                self.classifier_ffi = FFITransitClassifier(gpu_enabled=self.gpu_enabled)
                self.logger.info(f"CNN-FFI classifier loaded ({self.classifier_ffi.backend})")
            except FileNotFoundError:
                self.logger.info("CNN-FFI model not present — FFI rows will use SPOC classifier (fallback)")
            except Exception as e:
                self.logger.warning(f"CNN-FFI load failed: {e}")

        try:
            from worker_ui import WorkerUI
            import platform as _plat
            cpu_str = f"{os.cpu_count()} cores"
            try:
                cpu_str = f"{_plat.processor()} - {os.cpu_count()} cores"
                if self.fpp_cpu_cores:
                    cpu_str += f" ({self.fpp_cpu_cores} for FPP)"
            except Exception:
                pass
            gpu_str = "disabled"
            if self.gpu_enabled and _bls_is_gpu():
                gpu_str = _bls_gpu_info() or _bls_backend()
            elif self.gpu_enabled and _has_numba:
                # gpu_enabled=true but BLS resolved to a CPU backend — could be
                # by-design (host without GPU BLS) or a broken GPU init; the
                # startup GPU consistency check below logs the verdict.
                gpu_str = "CPU (numba) — gpu_enabled=true, no GPU BLS backend"
            self.ui = WorkerUI(
                self.name, gpu_str, cpu_str,
                _fpp_gpu_backend or "cpu",
                _bls_backend() or "cpu")
        except Exception as e:
            self.logger.warning(f"UI init failed, using plain output: {e}")
            self.ui = None

        self._check_gpu_consistency()

    def _check_gpu_consistency(self):
        """HARD RULE (project rule 2): gpu_enabled=true must mean a real GPU.

        One loud ERROR per degraded subsystem at startup — the silent GPU→CPU
        fallback has bitten the fleet repeatedly (libcublas not found, NVML
        contention, stale venv). The init excepts in bls/injection/onnx log
        the cause; this check makes the *outcome* impossible to miss.
        """
        if not self.gpu_enabled:
            return
        problems = []
        if not _bls_is_gpu():
            problems.append(f"BLS on CPU backend '{_bls_backend() or 'none'}'")
        try:
            import injection as _inj
            if _inj._bls_gpu is None:
                problems.append("injection-recovery on CPU (no GPU batch backend)")
        except Exception:
            pass
        if _fpp_gpu_backend is None:
            problems.append("FPP Monte Carlo on CPU (no CUDA/Metal/OpenCL kernel)")
        for clf_name, clf in (("CNN", self.classifier),
                              ("CNN-FFI", self.classifier_ffi)):
            backend = getattr(clf, "backend", None)
            if backend and "CPU" in backend:
                problems.append(f"{clf_name} inference on {backend}")
        if problems:
            for prob in problems:
                self.logger.error(f"GPU CONSISTENCY: gpu_enabled=true but {prob}")
            self.logger.error(
                "GPU CONSISTENCY: %d subsystem(s) degraded to CPU — check the "
                "import warnings above (driver/libcublas/LD_LIBRARY_PATH/cupy/"
                "mlx) before launching a lot", len(problems))
        else:
            self.logger.info(
                "GPU consistency OK: all GPU-capable subsystems on GPU backends")

    def _acquire_pid_lock(self, config_path: str) -> None:
        """Write {worker_dir}/.worker_{name}.pid after checking no live
        peer holds it. Stale locks (process gone) are silently replaced.
        Auto-releases at interpreter shutdown.
        """
        import atexit
        worker_name = self.config.get("worker_name", "worker")
        lock_dir = os.path.dirname(os.path.abspath(config_path)) or "."
        self._pidfile = os.path.join(lock_dir, f".worker_{worker_name}.pid")
        my_pid = os.getpid()
        if os.path.exists(self._pidfile):
            try:
                with open(self._pidfile) as f:
                    old_pid = int(f.read().strip())
                if old_pid == my_pid:
                    # os.execv auto-update — same PID, same lock, no-op
                    pass
                elif _pid_alive(old_pid):
                    raise SystemExit(
                        f"Another worker with pid={old_pid} already holds "
                        f"{self._pidfile}. Stop it first or wait for it to "
                        "exit. (ExoNit refuses to start a second process "
                        "with worker_name={worker_name!r}; multiple processes "
                        "fight for the same task queue.)"
                    )
                else:
                    self.logger.info(
                        f"Removing stale PID lock {self._pidfile} "
                        f"(pid {old_pid} not running)"
                    )
            except (ValueError, OSError):
                pass  # unreadable lockfile — overwrite
        try:
            with open(self._pidfile, "w") as f:
                f.write(str(my_pid))
        except OSError as e:
            self.logger.warning(
                f"Could not write PID lock {self._pidfile}: {e} — "
                "continuing without it (other workers may race)"
            )
            return
        atexit.register(self._release_pid_lock)

    def _release_pid_lock(self) -> None:
        try:
            if os.path.exists(self._pidfile):
                with open(self._pidfile) as f:
                    pid_in_file = int(f.read().strip())
                if pid_in_file == os.getpid():
                    os.remove(self._pidfile)
        except Exception:
            pass

    def _pool_enabled(self, pool: str) -> bool:
        """Return True if this worker is allowed to poll *pool*.

        When ``task_pools`` is unset (the common case), every pool is
        enabled — this preserves the historical behaviour of every worker
        polling every queue.
        """
        return self.task_pools is None or pool in self.task_pools

    def _headers(self):
        h = {"Authorization": f"Bearer {self.token}", "Content-Type": "application/json"}
        if self.worker_id:
            h["X-Worker-ID"] = self.worker_id
        return h

    def _star_elapsed(self):
        if self._star_start_time is None:
            return 0.0
        return round(time.time() - self._star_start_time, 1)

    # ── GPU telemetry for the Status fleet map (Task 2 of ui-status-fleet-map) ──
    # 2026-07-25 telemetry-gaps pass: every probe below now (a) tries to find
    # its data source in more than the one place the previous version looked,
    # (b) warns ONCE (never spams, never raises — see _warn_telemetry_once)
    # when it can't, instead of the previous bare `except Exception: return
    # None`, and (c) never fabricates a number when the real source is
    # unreadable (missing binary, permission-denied sysfs node, ...).

    def _get_gpu_telemetry(self):
        """Live GPU numbers for the heartbeat. NEVER raises — all-None on any failure.
        Called once per heartbeat (~30s) from _build_heartbeat_payload, so a short
        subprocess is fine. Each backend helper also swallows its own errors."""
        none = {"util_pct": None, "mem_used_gb": None, "mem_total_gb": None,
                "temp_c": None, "power_w": None}
        try:
            import platform
            hw = _bls_hardware_info() or {}
            if hw.get("cuda"):
                telem = self._gpu_telem_nvidia() or none
            elif hw.get("metal"):
                telem = self._gpu_telem_macmon() or none
            elif hw.get("opencl"):
                if platform.system() == "Linux":
                    telem = self._gpu_telem_amd_sysfs() or none
                else:
                    telem = self._gpu_telem_windows_pdh() or none
            elif platform.system() == "Linux":
                # BLS's hw flags reflect what BLS/injection could IMPORT in
                # THIS process's venv (e.g. cupy/pyopencl availability), not
                # "is there a GPU on this box" — confirmed on <host>
                # 2026-07-25: the Arc iGPU is real (`clinfo` sees it, driver
                # `xe` is bound) but hw["opencl"] was False purely because
                # pyopencl isn't installed in the server venv, so the branch
                # above was dead code for that host despite real hardware
                # being present. Telemetry answers the hardware question
                # directly instead of trusting BLS's own capability probe.
                telem = self._gpu_telem_amd_sysfs() or none
            elif platform.system() == "Darwin":
                # Same reasoning: a Mac forced onto a non-metal BLS backend
                # (EXONIT_BLS_BACKEND override) still has a real Apple GPU
                # that macmon can read independently of what BLS picked.
                telem = self._gpu_telem_macmon() or none
            else:
                telem = none
            return self._round_gpu_telemetry(telem)
        except Exception as exc:
            _warn_telemetry_once(
                self, "gpu_telemetry_dispatch",
                f"GPU telemetry dispatch failed ({type(exc).__name__}: "
                f"{exc}) — reporting all-None for this heartbeat.")
            return none

    @staticmethod
    def _round_gpu_telemetry(telem):
        """Round every numeric value in a gpu telemetry dict to 2 decimals.
        Raw backends (nvidia-smi float parsing, macmon JSON, sysfs division)
        return long floats like 0.8000000715255737 that clutter the DB and
        the Status fleet map UI — normalize once here for all backends."""
        if not telem:
            return telem
        out = {}
        for k, v in telem.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                out[k] = round(v, 2)
            else:
                out[k] = v
        return out

    def _gpu_telem_nvidia(self):
        """nvidia-smi subprocess (Linux or Windows/WSL). No in-process NVML."""
        if not shutil.which("nvidia-smi"):
            return None

        def _num(x):
            x = x.strip()
            if x in ("N/A", "[N/A]", "[Not Supported]", "[Insufficient Permissions]", ""):
                return None
            try:
                return float(x)
            except (ValueError, TypeError):
                return None

        try:
            out = subprocess.run(
                ["nvidia-smi",
                 "--query-gpu=utilization.gpu,temperature.gpu,memory.used,memory.total,power.draw",
                 "--format=csv,noheader,nounits", "-i", "0"],
                capture_output=True, text=True, timeout=4).stdout.strip()
            fields = [c for c in out.splitlines()[0].split(",")]
            u, t, mu, mt, p = fields[0], fields[1], fields[2], fields[3], fields[4]
            return {"util_pct": _num(u), "temp_c": _num(t),
                    "mem_used_gb": (_num(mu) or 0) / 1024,
                    "mem_total_gb": (_num(mt) or 0) / 1024,
                    "power_w": _num(p)}
        except Exception as exc:
            _warn_telemetry_once(
                self, "nvidia_smi_failed",
                f"nvidia-smi probe failed ({type(exc).__name__}: {exc}) — "
                "GPU telemetry disabled on this host until it starts "
                "working again.")
            return None

    def _find_macmon(self):
        """Locate the macmon binary: PATH first (respects an operator's own
        install), then the no-admin-friendly and package-manager locations
        in _MACMON_SEARCH_DIRS, in that order."""
        exe = shutil.which("macmon")
        if exe:
            return exe
        for d in _MACMON_SEARCH_DIRS:
            candidate = os.path.join(os.path.expanduser(d), "macmon")
            if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
                return candidate
        return None

    def _gpu_telem_macmon(self):
        """macmon subprocess for Apple Silicon (Metal, M1/M3/M4).
        macmon pipe -s 1 prints ONE JSON line then exits. Tolerate absence."""
        import json as _json
        exe = self._find_macmon()
        if exe is None:
            searched = ", ".join(_MACMON_SEARCH_DIRS)
            _warn_telemetry_once(
                self, "macmon_missing",
                "macmon not found on PATH or in any of the usual install "
                f"locations ({searched}). No admin rights or package "
                "manager needed to fix this: a single static `macmon` "
                "binary dropped into ~/bin is enough — the worker will "
                "pick it up on the next heartbeat with no restart or code "
                "change. GPU telemetry stays all-None on this host until "
                "then.")
            return None
        try:
            out = subprocess.run([exe, "pipe", "-s", "1"],
                                 capture_output=True, text=True, timeout=5).stdout.strip()
            d = _json.loads(out.splitlines()[0])
            mem = d.get("memory", {})
            ram_total = mem.get("ram_total")
            ram_used = mem.get("ram_usage")
            gpu = d.get("gpu_usage", [None, None])  # [freq_MHz, ratio]
            temp = d.get("temp", {})
            gpu_temp = temp.get("gpu_temp_avg")
            # macmon reports a bogus ~0.8C reading when the GPU sensor is idle
            # (physically impossible die temp) -> treat as "no reading".
            if gpu_temp is None or gpu_temp < 5.0:
                gpu_temp = None
            return {
                "util_pct": (gpu[1] * 100) if gpu and gpu[1] is not None else None,
                "mem_used_gb": (ram_used / (1024**3)) if ram_used else None,
                "mem_total_gb": (ram_total / (1024**3)) if ram_total else None,
                "temp_c": gpu_temp,
                # GPU power (not whole-system sys_power) -- 0.0 is a valid
                # idle reading and must not be coerced to None.
                "power_w": d.get("gpu_power"),
            }
        except Exception as exc:
            _warn_telemetry_once(
                self, "macmon_probe_failed",
                f"macmon probe failed ({type(exc).__name__}: {exc}) — GPU "
                "telemetry disabled on this host until macmon starts "
                "working again (found it at "
                f"{exe!r}, so this is a runtime failure, not a missing "
                "binary).")
            return None

    def _hwmon_power_w(self, base):
        """Shared hwmon power probe for a `/sys/class/drm/card*/device`
        node. power1_average is amdgpu's node; power1_input/power2_input
        cover other drivers that DO expose hwmon power. Intel's i915/xe
        drivers expose NO hwmon directory at all on an integrated part
        (confirmed on <host>'s Arc iGPU 2026-07-25) — this returns None
        there and the caller (_gpu_telem_amd_sysfs) falls through to the
        RAPL probe below, which is the only power source Intel Linux
        systems reliably have."""
        import glob
        for fname in ("power1_average", "power1_input", "power2_input"):
            for h in glob.glob(base + f"/hwmon/hwmon*/{fname}"):
                v = _safe_read_sysfs(h)
                if v:
                    return float(v) / 1_000_000.0
        return None

    def _warn_intel_util_unavailable(self):
        """One-time explanation for why util_pct stays None on an Intel
        iGPU, instead of a silently-blank dashboard cell forever. Reading
        xe/i915 GPU utilization needs perf_event_open() on the driver's PMU
        (`/sys/devices/xe_*/events/engine-active-ticks` etc — those files
        are PMU event *descriptors* for building a perf_event_attr, not
        readable counters by themselves). Confirmed on <host> 2026-07-25:
        `/proc/sys/kernel/perf_event_paranoid` is 4, which blocks
        perf_event_open for an unprivileged process unconditionally.
        Deliberately NOT implemented: the raw perf_event_open + mmap
        ring-buffer ctypes plumbing is a lot of fragile low-level code for
        a syscall that is guaranteed to EPERM under the current hardening —
        worth revisiting if an operator lowers perf_event_paranoid or
        grants CAP_PERFMON, not before."""
        import glob
        pmu_dirs = glob.glob("/sys/devices/xe_*/events") or glob.glob("/sys/devices/i915_*/events")
        if not pmu_dirs:
            return  # No PMU on this host either — nothing more to say.
        paranoid_raw = _safe_read_sysfs("/proc/sys/kernel/perf_event_paranoid")
        try:
            paranoid = int(paranoid_raw)
        except (TypeError, ValueError):
            paranoid = None
        if paranoid is None or paranoid > 2:
            _warn_telemetry_once(
                self, "intel_gpu_util_perf_paranoid",
                "Intel GPU utilization requires perf_event_open() on the "
                f"xe/i915 PMU ({pmu_dirs[0]}), gated by kernel."
                f"perf_event_paranoid (currently {paranoid_raw!r}; needs "
                "<= 2) or CAP_PERFMON — both are host-owner decisions the "
                "worker cannot grant itself, so this was not implemented. "
                "gpu_util_pct stays None on this host until the owner "
                "relaxes perf_event_paranoid or grants CAP_PERFMON to the "
                "worker process.")

    def _linux_rapl_power_w(self):
        """Intel RAPL (Running Average Power Limit) energy-delta power
        reading — the only power source Intel's Linux iGPU drivers
        reliably expose (no hwmon power node on i915/xe integrated parts,
        confirmed on <host> 2026-07-25). RAPL reports a monotonic
        microjoule energy counter per power domain under
        /sys/class/powercap/intel-rapl:*, not an instantaneous watt
        figure — so power is derived from the energy delta between two
        heartbeats (~30s apart, tracked in self._rapl_prev_sample). The
        FIRST call after worker start has no previous sample to diff
        against and must return None, not a bogus 0.0 or a wraparound
        spike (same idiom as psutil.cpu_percent(interval=None) above).

        Prefers the "uncore" child domain (intel-rapl:0:1 on this fleet's
        Intel host) — that is the iGPU + memory-controller power plane on
        recent Intel platforms. Falls back to the top-level "package"
        domain if no such child exists, which is coarser (whole SoC:
        cores + uncore + iGPU, NOT the GPU die alone) and is labelled as
        such in the warning if it's the domain actually used.

        Every read of energy_uj is Permission Denied by default on a
        stock install (standard post-PLATYPUS hardening, confirmed
        2026-07-25) — that is treated as an expected, well-explained
        outcome, not a bug to retry around: granting read access is a
        host-owner decision (a udev rule or group grant on
        /sys/class/powercap/intel-rapl:*/energy_uj), not something this
        worker process should or could escalate itself into.
        """
        import glob
        import time as _t

        parents = sorted(d for d in glob.glob("/sys/class/powercap/intel-rapl:*")
                          if d.count(":") == 1)
        if not parents:
            return None

        domain = None
        domain_label = None
        for parent in parents:
            for child in sorted(glob.glob(parent + ":*")):
                if _safe_read_sysfs(child + "/name") == "uncore":
                    domain, domain_label = child, "uncore (iGPU + memory-controller power plane)"
                    break
            if domain:
                break
        if domain is None:
            domain, domain_label = parents[0], "package (whole SoC — not GPU-die-only)"

        energy_path = domain + "/energy_uj"
        try:
            with open(energy_path) as f:
                energy_uj = int(f.read().strip())
        except PermissionError:
            _warn_telemetry_once(
                self, "rapl_permission_denied",
                f"RAPL energy counter at {energy_path} ({domain_label}) "
                "exists but is not readable by this process (Permission "
                "Denied — standard kernel hardening against the PLATYPUS "
                "side-channel on recent kernels). GPU/SoC power will stay "
                "None on this host until the owner grants read access "
                "(e.g. a udev rule for /sys/class/powercap/intel-rapl:*/"
                "energy_uj) — a host-owner decision; this worker process "
                "must not attempt to escalate its own privileges.")
            return None
        except FileNotFoundError:
            return None
        except Exception as exc:
            _warn_telemetry_once(
                self, "rapl_read_failed",
                f"RAPL energy counter read failed ({type(exc).__name__}: "
                f"{exc}).")
            return None

        now = _t.time()
        prev = getattr(self, "_rapl_prev_sample", None)
        self._rapl_prev_sample = (energy_uj, now, domain)
        if prev is None or prev[2] != domain:
            # First sample ever, or the domain changed under us (shouldn't
            # happen, but a delta across two different counters would be
            # meaningless) — nothing to diff against yet.
            return None
        prev_energy, prev_time, _ = prev
        dt = now - prev_time
        if dt <= 0:
            return None
        delta_uj = energy_uj - prev_energy
        if delta_uj < 0:
            # Counter wrapped (some platforms use a 32-bit energy_uj that
            # rolls over) — skip this sample rather than guess the wrap
            # amount; resyncs cleanly on the next heartbeat.
            return None
        return (delta_uj / 1_000_000.0) / dt

    def _gpu_telem_amd_sysfs(self):
        """Linux GPU telemetry via sysfs — vendor-detected from the PCI
        `vendor` id rather than assumed, so both AMD (amdgpu: busy%, VRAM,
        hwmon temp/power — worker-<host>'s Vega iGPU, unchanged from
        before) and Intel (i915/xe: no busy%, no VRAM, no hwmon at all on
        an integrated part — RAPL-only power, see _linux_rapl_power_w) get
        a real probe instead of the Intel branch silently finding nothing
        because the AMD-shaped file list doesn't exist on its driver."""
        import glob

        try:
            base = None
            vendor = None
            for c in sorted(glob.glob("/sys/class/drm/card*/device")):
                v = _safe_read_sysfs(c + "/vendor")
                if v in (_PCI_VENDOR_AMD, _PCI_VENDOR_INTEL):
                    base, vendor = c, v
                    break
            if not base:
                return None

            # AMD-only files (mem_info_vram_*, gpu_busy_percent are amdgpu
            # sysfs nodes); read unconditionally — they're simply absent
            # (None) on Intel, no vendor branch needed for these three.
            util = _safe_read_sysfs(base + "/gpu_busy_percent")
            mu = _safe_read_sysfs(base + "/mem_info_vram_used")
            mt = _safe_read_sysfs(base + "/mem_info_vram_total")

            temp = None
            for h in glob.glob(base + "/hwmon/hwmon*/temp1_input"):
                v = _safe_read_sysfs(h)
                if v:
                    temp = float(v) / 1000.0
                    break

            pw = self._hwmon_power_w(base)
            if pw is None and vendor == _PCI_VENDOR_INTEL:
                pw = self._linux_rapl_power_w()
                if util is None:
                    self._warn_intel_util_unavailable()

            return {"util_pct": float(util) if util else None,
                    "mem_used_gb": (int(mu) / (1024**3)) if mu else None,
                    "mem_total_gb": (int(mt) / (1024**3)) if mt else None,
                    "temp_c": temp, "power_w": pw}
        except Exception as exc:
            _warn_telemetry_once(
                self, "linux_sysfs_failed",
                f"Linux GPU sysfs probe failed ({type(exc).__name__}: "
                f"{exc}).")
            return None

    # --- Windows PDH (Performance Data Helper) — no admin, no WSL2 -------
    # Same counters Task Manager's own GPU tab reads: \GPU Engine(*)
    # \Utilization Percentage and \GPU Adapter Memory(*)\Dedicated Usage,
    # via ctypes + the OS-shipped pdh.dll (no new dependency). Uses
    # ctypes.pointer() rather than ctypes.byref() for every out-parameter
    # specifically so this boundary can be exercised with a mocked
    # ctypes.windll in tests (a byref() proxy can't be written through from
    # a MagicMock side_effect; a real pointer's .contents can).
    #
    # temp_c and power_w are NOT implemented: Intel iGPUs don't expose
    # either through any public no-admin API (only vendor overlay tools
    # like Intel Graphics Command Center do, and those aren't scriptable
    # headlessly) — returning None here is honest, a guess would not be.

    def _pdh_ensure_query(self):
        """Open the PDH query and add every counter instance ONCE,
        persisting the handles on self for the process's lifetime. GPU
        Engine Utilization is a rate counter — PDH reports
        PDH_CSTATUS_INVALID_DATA (not a number) on the very first
        PdhCollectQueryData after a counter is added, and only becomes
        meaningful on the SECOND collect. Keeping the query open across
        heartbeats (~30s apart) means one real PdhCollectQueryData call
        per heartbeat is enough — no need for two back-to-back samples
        inside a single call (which would need a ~1s sleep and blow the
        telemetry function's <100ms budget). Same idiom as
        psutil.cpu_percent(interval=None) elsewhere in this file, and the
        RAPL energy-delta probe above: the first heartbeat after setup
        returns None, every one after that is real."""
        if getattr(self, "_pdh_setup_failed", False):
            return False
        if getattr(self, "_pdh_query", None) is not None:
            return True
        import ctypes
        try:
            pdh = ctypes.windll.pdh
            hQuery = ctypes.c_void_p()
            if pdh.PdhOpenQueryW(None, 0, ctypes.pointer(hQuery)) != 0:
                raise RuntimeError("PdhOpenQueryW failed")
            counters = {}
            for wildcard, kind in (
                (r"\GPU Engine(*)\Utilization Percentage", "util"),
                (r"\GPU Adapter Memory(*)\Dedicated Usage", "mem"),
            ):
                for path in _pdh_expand_wildcard(pdh, wildcard):
                    hCounter = ctypes.c_void_p()
                    rc = pdh.PdhAddEnglishCounterW(hQuery, path, 0, ctypes.pointer(hCounter))
                    if rc == 0:
                        counters[path] = (hCounter, kind, _pdh_instance_from_path(path))
            if not counters:
                raise RuntimeError("no GPU Engine/Adapter Memory counter instances found")
            self._pdh_query = hQuery
            self._pdh_counters = counters
            return True
        except Exception as exc:
            self._pdh_setup_failed = True
            _warn_telemetry_once(
                self, "windows_pdh_setup_failed",
                f"Windows PDH GPU query setup failed ({type(exc).__name__}: "
                f"{exc}) — GPU telemetry disabled on this host for the "
                "rest of this process.")
            return False

    def _gpu_telem_windows_pdh(self):
        """Windows-native, no-admin GPU utilization + VRAM via PDH. See the
        block comment above this method for why, and _pdh_ensure_query for
        the rate-counter warm-up. Strictly isolated behind
        platform.system() == 'Windows': ctypes.windll doesn't exist as an
        attribute on Linux/macOS, so this must never run there."""
        import platform
        if platform.system() != "Windows":
            return None
        try:
            import ctypes
            if not self._pdh_ensure_query():
                return None
            pdh = ctypes.windll.pdh
            if pdh.PdhCollectQueryData(self._pdh_query) != 0:
                return None
            util_samples = {}
            mem_samples = {}
            for hCounter, kind, instance in self._pdh_counters.values():
                v = _pdh_read_counter_double(pdh, hCounter)
                (util_samples if kind == "util" else mem_samples)[instance] = v
            return {
                "util_pct": _pdh_aggregate_engine_util(util_samples),
                "mem_used_gb": _pdh_aggregate_adapter_memory_gb(mem_samples),
                # No no-admin PDH counter reports TOTAL adapter capacity for
                # an arbitrary adapter — an Intel iGPU's memory budget is
                # dynamic (shared system RAM), not a fixed VRAM size, and
                # querying it precisely needs the DXGI COM API, not PDH.
                "mem_total_gb": None,
                "temp_c": None,
                "power_w": None,
            }
        except Exception as exc:
            _warn_telemetry_once(
                self, "windows_pdh_failed",
                f"Windows PDH GPU probe failed ({type(exc).__name__}: "
                f"{exc}) — GPU telemetry disabled on this host.")
            return None

    def _build_heartbeat_payload(self):
        payload = {
            "worker_id": self.worker_id,
            "status": self.status,
            "current_tic": self.current_tic,
            "phase": self.current_phase,
            "phase_progress": self.phase_progress,
            "star_elapsed_s": self._star_elapsed(),
            "stars_done": self.stars_completed,
            "avg_time_per_star": round(
                self.total_processing_time / max(1, self.stars_completed), 1),
            "last_error": self.last_error,
            "uptime_s": round(time.time() - self.started_at, 1),
            "pid": os.getpid(),
            "git_sha": _resolve_git_sha(),
            "flag_states": _resolve_flag_states(self.config),
        }
        # GPU->CPU fallback counters (2026-07-14). A worker quietly finishing a
        # run on the CPU is the state project rule 2 forbids, and the only
        # way the rule stays enforceable is if the server can SEE it: the log
        # line lives on the worker's own disk, this reaches the dashboard.
        # "disabled": true means the circuit breaker retired that worker's GPU
        # for the rest of the process (restart it to retry the GPU).
        try:
            from fpp_native import gpu_fallback_stats
            stats = gpu_fallback_stats()
            if stats["total"] or stats["disabled"]:
                payload["gpu_fallback"] = stats
        except Exception:
            pass    # never let telemetry break a heartbeat
        # System metrics — collected on EVERY heartbeat (cheap psutil calls) so
        # host RAM/swap pressure is visible during a run without enabling debug
        # mode. swap_used_gb + pid_rss_mb are the signals behind the worker's
        # memory guard (see _decide_injection_background). The heavier bits
        # (CPU temp sensors, recent_timings) stay debug-only.
        try:
            import psutil
            proc = psutil.Process()
            vm = psutil.virtual_memory()
            sw = psutil.swap_memory()
            sysinfo = {
                "cpu_percent": psutil.cpu_percent(interval=None),
                "ram_used_gb": round(vm.used / (1024**3), 2),
                "ram_total_gb": round(vm.total / (1024**3), 2),
                "swap_used_gb": round(sw.used / (1024**3), 2),
                "pid_rss_mb": round(proc.memory_info().rss / (1024**2), 1),
            }
            if self.debug_mode:
                temps = (psutil.sensors_temperatures()
                         if hasattr(psutil, 'sensors_temperatures') else {})
                cpu_temp = None
                if temps:
                    for entries in temps.values():
                        if entries:
                            cpu_temp = entries[0].current
                            break
                sysinfo["cpu_temp_c"] = cpu_temp
            payload["system"] = sysinfo
        except ImportError:
            if not getattr(self, "_psutil_warned", False):
                self.logger.warning("psutil not installed; system metrics disabled. "
                                    "Run: pip install psutil")
                self._psutil_warned = True
            payload["system"] = {"error": "psutil_missing"}
        # GPU live telemetry (Status fleet map) — always ON, degrades to all-None
        payload["gpu"] = self._get_gpu_telemetry()
        if self.debug_mode:
            payload["recent_timings"] = list(self.recent_timings)
        return payload

    def register(self):
        gpu_info = _friendly_gpu_name()
        hw_summary = _bls_hardware_summary()
        hw_info = _bls_hardware_info()
        data = {
            "name": self.name,
            "capabilities": {
                "gpu": hw_summary,
                "bls_backend": _bls_backend() or "none",
                "bls_gpu_info": gpu_info,
                "hardware": hw_info,
                "fpp_backend": _fpp_gpu_backend or "cpu",
                "cpu_cores": os.cpu_count() or 1,
                "fpp_cpu_cores": self.fpp_cpu_cores or max(1, (os.cpu_count() or 3) - 2),
            }
        }
        # Retry loop so a worker started while the server is momentarily
        # down doesn't die on boot — it just waits and keeps trying.
        delay = 5
        while True:
            try:
                resp = requests.post(
                    f"{self.server}/api/worker/register",
                    json=data, headers=self._headers(), timeout=10)
                resp.raise_for_status()
                self.worker_id = resp.json()["worker_id"]
                self.logger.info(f"Registered as {self.worker_id}")
                return
            except (requests.ConnectionError, requests.Timeout,
                    requests.HTTPError) as exc:
                self.logger.warning(
                    f"register() failed ({type(exc).__name__}: {exc}); "
                    f"retrying in {delay}s")
                time.sleep(delay)
                delay = min(delay * 2, 60)

    @staticmethod
    def _file_sha256_normalized(path):
        """sha256 of *path* with CRLF normalized to LF, or None if unreadable."""
        try:
            with open(path, "rb") as f:
                content = f.read().replace(b"\r\n", b"\n")
                return hashlib.sha256(content).hexdigest()
        except Exception:
            return None

    def _snapshot_worker_dir_hashes(self):
        """Hash every file in worker_dir at process start (see _loaded_hashes).

        Key layout MUST mirror the server manifest (see
        exonit/routes/workers.py:worker_update_manifest / ``_syncable``):
        top-level files keyed by their bare filename, plus ``worker/models/*``
        files keyed as ``models/<name>`` (always "/"-separated, even on
        Windows). A mismatch here makes ``models/*`` entries permanently
        "stale" (self._loaded_hashes.get("models/x") is always None), which
        makes check_for_updates() think loaded code is stale on every poll
        with nothing left to download — an infinite execv restart loop
        (2026-07-10 incident on worker-<host>).
        """
        worker_dir = os.path.dirname(os.path.abspath(__file__))
        hashes = {}
        try:
            for filename in os.listdir(worker_dir):
                path = os.path.join(worker_dir, filename)
                if os.path.isfile(path):
                    h = self._file_sha256_normalized(path)
                    if h is not None:
                        hashes[filename] = h
            models_dir = os.path.join(worker_dir, "models")
            if os.path.isdir(models_dir):
                for filename in os.listdir(models_dir):
                    path = os.path.join(models_dir, filename)
                    if os.path.isfile(path):
                        h = self._file_sha256_normalized(path)
                        if h is not None:
                            hashes[f"models/{filename}"] = h
        except Exception:
            pass
        return hashes

    def check_for_updates(self):
        """Check server for file updates and apply them."""
        try:
            resp = requests.get(f"{self.server}/api/worker/update",
                                headers=self._headers(), timeout=10)
            manifest = resp.json()
            files = manifest.get("files", [])
            if not files:
                return False

            # First pass: identify all files that need updating, and
            # separately whether any manifest file's remote hash differs
            # from what THIS PROCESS loaded at start (self._loaded_hashes).
            # The two can diverge: when multiple worker instances share a
            # worker_dir, another instance may have already downloaded the
            # new file to disk before this check runs — so `local_hash ==
            # remote_hash` here, yet this process is still running the old
            # code in memory. See _loaded_hashes docstring at __init__.
            worker_dir = os.path.dirname(os.path.abspath(__file__))
            to_update = []
            loaded_code_stale = False
            for entry in files:
                filename = entry["filename"]
                remote_hash = entry["sha256"]
                local_path = os.path.join(worker_dir, filename)

                local_hash = ""
                if os.path.exists(local_path):
                    local_hash = self._file_sha256_normalized(local_path) or ""

                if local_hash != remote_hash:
                    to_update.append(filename)

                if remote_hash != self._loaded_hashes.get(filename):
                    loaded_code_stale = True

            if not to_update and not loaded_code_stale:
                return False

            # Second pass: download ALL changed files before restarting
            updated = False
            for filename in to_update:
                self.logger.info(f"[UPDATE] Downloading {filename}...")
                resp = requests.get(f"{self.server}/api/worker/update/{filename}",
                                    headers=self._headers(), timeout=30)
                if resp.status_code == 200:
                    local_path = os.path.join(worker_dir, filename)
                    # Write in binary to preserve line endings (avoid \r\n on Windows)
                    with open(local_path, "wb") as f:
                        f.write(resp.content)
                    self.logger.info(f"[UPDATE] {filename} updated")
                    updated = True

            if updated:
                # A real code change was just downloaded — legitimate restart,
                # not a stale-detection loop. Reset the anti-loop counter so
                # a subsequent stray "loaded stale" restart still gets its
                # full allowance.
                os.environ["EXONIT_UPDATE_RESTART_COUNT"] = "0"
                self.logger.info("[UPDATE] Files updated — restarting worker...")
                os.execv(sys.executable, [sys.executable] + sys.argv)
            elif loaded_code_stale:
                # Multi-instance race: another instance already refreshed the
                # files on disk (nothing left in to_update), but THIS process
                # loaded the old code into memory at start and never reloads
                # a module on its own. Restart unconditionally — UNLESS this
                # process lineage has already restarted for this exact reason
                # at least twice, which means the restart is not fixing
                # anything (manifest/snapshot key mismatch or similar bug)
                # and we'd otherwise execv-loop forever. See
                # _update_restart_count docstring at __init__.
                if self._update_restart_count >= 2:
                    self.logger.warning(
                        "[UPDATE] stale-restart loop detected — skipping restart")
                    return False
                next_count = self._update_restart_count + 1
                os.environ["EXONIT_UPDATE_RESTART_COUNT"] = str(next_count)
                self.logger.info(
                    "[UPDATE] On-disk current but loaded code stale — "
                    f"restarting worker (restart {next_count}/2)...")
                os.execv(sys.executable, [sys.executable] + sys.argv)
                updated = True
            return updated
        except Exception as e:
            # WARNING (not silent): a persistently failing auto-update means
            # this worker drifts from the fleet's code indefinitely without
            # any visible signal (2026-07-02 cleanup review, item 4).
            self.logger.warning(f"[UPDATE] check_for_updates failed: {e}")
            return False

    # ------------------------------------------------------------------
    # CUDA context-poison auto-restart
    # ------------------------------------------------------------------

    def _note_cuda_failure(self, exc: BaseException) -> None:
        """Record a CUDA context error and restart via os.execv if the streak
        reaches the threshold (3 consecutive CUDA-poisoned tasks).

        This must be called AFTER submit_result() has reported the failed task
        so the server can reassign it immediately.

        Non-CUDA exceptions are ignored (streak is not incremented nor reset)
        so that a mix of CUDA and unrelated failures eventually hits the
        threshold without a transient non-CUDA error hiding the problem.

        The restart-lineage counter EXONIT_CUDA_RESTART_COUNT is bumped in
        the child environment; if it already reached 3 we do NOT restart —
        the GPU/driver is truly dead and we stay up so the server sees
        continuous task failures rather than a silent crash-loop.
        """
        if not _is_cuda_context_error(exc):
            return

        self._cuda_error_streak += 1
        self.logger.error(
            f"[CUDA-POISONED] CUDA context error #{self._cuda_error_streak} "
            f"(streak={self._cuda_error_streak}/3): {exc}"
        )

        if self._cuda_error_streak < 3:
            return

        # Streak hit threshold — decide whether to restart.
        if self._cuda_restart_count >= 3:
            self.logger.critical(
                f"[CUDA-POISONED] streak=3 but restart_count={self._cuda_restart_count} "
                "has reached the lineage limit (3). GPU/driver appears truly dead. "
                "Worker continues running; tasks will keep failing until the "
                "operator restarts with a fresh GPU context."
            )
            # Reset streak so this CRITICAL is not spammed every task.
            self._cuda_error_streak = 0
            return

        next_count = self._cuda_restart_count + 1
        self.logger.critical(
            f"[CUDA-POISONED] CUDA context irrecoverably poisoned after 3 "
            f"consecutive failures. Performing self-restart "
            f"(restart {next_count}/3). Last error: {exc}"
        )
        os.environ["EXONIT_CUDA_RESTART_COUNT"] = str(next_count)
        os.execv(sys.executable, [sys.executable] + sys.argv)

    def _reset_cuda_streak(self) -> None:
        """Reset the CUDA error streak and lineage counter after a successful task.

        A restart that actually cured the context will eventually process a
        task successfully, at which point the guard should be disarmed so that
        a future genuine poison event gets its full 3-restart budget.
        """
        if self._cuda_error_streak != 0 or self._cuda_restart_count != 0:
            self._cuda_error_streak = 0
            self._cuda_restart_count = 0
            os.environ.pop("EXONIT_CUDA_RESTART_COUNT", None)

    def heartbeat_loop(self):
        hb_count = 0
        while self.running:
            try:
                resp = requests.post(f"{self.server}/api/worker/heartbeat",
                    json=self._build_heartbeat_payload(),
                    headers=self._headers(), timeout=10)
                cmd = resp.json().get("command", "continue")
                # B1 (2026-07-02 cleanup review): this used to be a separate
                # `if cmd == "re-register": ...; continue` ahead of the
                # elif-chain below. The `continue` skipped the hb_count
                # bump, the check_for_updates poll, AND time.sleep() at the
                # bottom of the loop — while the worker is "working" (long
                # task) the server keeps replying re-register every
                # heartbeat, so the loop spun with zero delay, hammering
                # /api/worker/heartbeat. Folding it into the elif-chain
                # (mutually exclusive on `cmd` already) means every branch
                # falls through to the same sleep at the end.
                if cmd == "re-register":
                    # Defer re-registration until worker is idle
                    # Metal GPU can crash if HTTP calls happen during GPU operations
                    self._needs_reregister = True
                    if self.status != "working":
                        self.logger.info("Server lost our registration, re-registering...")
                        self.register()
                        self._needs_reregister = False
                elif cmd == "pause":
                    self.paused = True
                    self.logger.info("Paused by server")
                elif cmd == "resume":
                    self.paused = False
                    self.logger.info("Resumed by server")
                elif cmd == "shutdown":
                    self.logger.info("Shutdown requested by server")
                    self.running = False
                elif cmd == "debug_on":
                    if not self.debug_mode:
                        self.debug_mode = True
                        self.logger.info("Debug mode enabled by server")
                elif cmd == "debug_off":
                    if self.debug_mode:
                        self.debug_mode = False
                        self.logger.info("Debug mode disabled by server")
            except Exception as e:
                # Fires often on flaky wifi (mac workers) — debug level so
                # it doesn't spam INFO/WARNING logs on a normal blip.
                self.logger.debug("heartbeat failed: %s", e)

            # Check for updates every 10 heartbeats (only when idle)
            hb_count += 1
            if hb_count % 10 == 0 and self.status == "idle":
                self.check_for_updates()

            time.sleep(self.hb_interval)

    def get_task(self):
        try:
            resp = requests.get(f"{self.server}/api/worker/task", headers=self._headers(), timeout=30)
            data = resp.json()
            return data if data.get("task_id") else None
        except Exception as e:
            self.logger.warning(f"Error getting task: {e}")
            return None

    def _load_star(self, tic_id):
        """Fetch metadata + FITS from server, read with lightkurve, return arrays.

        Returns
        -------
        (time_arr, flat_flux_arr, raw_flux_arr, n_sectors, meta)
            ``raw_flux_arr`` is the flux after stitch+fill_gaps but BEFORE
            the wotan detrending — needed by the multi-planet loop to
            re-detrend with an accumulated transit mask (Hippke+ 2019 §5).
            On failure: (None, None, None, 0, {}).
        """
        # 1. Fetch metadata
        try:
            resp = requests.get(f"{self.server}/api/metadata/{tic_id}",
                                headers=self._headers(), timeout=30)
            if resp.status_code != 200:
                self.logger.warning(f"No metadata for TIC {tic_id}: {resp.status_code}")
                fallback = self._try_ffi_fallback(tic_id, reason="metadata_http")
                if fallback is not None:
                    return fallback
                return None, None, None, 0, {}
            meta = resp.json()
        except Exception as e:
            self.logger.warning(f"Error fetching metadata for TIC {tic_id}: {e}")
            fallback = self._try_ffi_fallback(tic_id, reason="metadata_exception")
            if fallback is not None:
                return fallback
            return None, None, None, 0, {}

        fits_files = meta.get("fits_files", [])
        sectors = meta.get("sectors", [])
        if not fits_files:
            self.logger.warning(f"No FITS files for TIC {tic_id}")
            fallback = self._try_ffi_fallback(tic_id, reason="no_fits_files")
            if fallback is not None:
                return fallback
            return None, None, None, 0, {}

        # 2. Download each FITS file to local cache (post-download validated
        #    via ensure_fits_present so a truncated MAST response can't poison
        #    the cache with a file that will fail every subsequent attempt).
        local_paths = []
        for fname in fits_files:
            local_path = Path(self.fits_cache_dir) / fname

            def _fetch(dst, _fname=fname):
                self.logger.info(f"Downloading {_fname}...")
                resp = requests.get(
                    f"{self.server}/api/fits/{_fname}",
                    headers=self._headers(), timeout=120, stream=True)
                if resp.status_code != 200:
                    raise RuntimeError(f"server returned HTTP {resp.status_code}")
                with open(dst, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=65536):
                        f.write(chunk)

            ok = ensure_fits_present(
                local_path, _fetch, max_retries=3, logger=self.logger)
            if ok:
                local_paths.append(str(local_path))

        if not local_paths:
            self.logger.warning(f"No FITS files downloaded for TIC {tic_id}")
            fallback = self._try_ffi_fallback(tic_id, reason="fits_download_fail")
            if fallback is not None:
                return fallback
            return None, None, None, 0, {}

        # 3. Read with lightkurve, stitch, clean, normalize
        try:
            import warnings
            warnings.filterwarnings("ignore", message=".*tpfmodel.*oktopus.*")
            warnings.filterwarnings("ignore", message=".*cadences will be ignored due to the quality mask.*")
            import lightkurve as lk
            lc_list = []
            for path in local_paths:
                # B2 (2026-07-02 cleanup review): this loop used to swap the
                # process-global sys.stdout to os.devnull around each read to
                # suppress the lightkurve "N% of the cadences will be
                # ignored due to the quality mask" noise. _load_star runs on
                # both the main thread AND the FPP-prefetch background
                # thread (see _fetch_and_load_fpp), so two concurrent
                # readers could race and leave sys.stdout pointing at a
                # closed devnull handle.
                #
                # Investigated instead of just locking it: that quality-mask
                # message is emitted via `lightkurve.utils`'s Python
                # `logging` logger (utils.py: `log.warning(...)` /
                # `log.info(...)`), not via print()/stdout — verified
                # empirically against real cached FITS on 2026-07-02
                # (lk.read() on SPOC/QLP files produced zero stdout output
                # with or without the swap; the "cadences ignored" text only
                # appeared on stderr via the logging handlers). It is
                # already suppressed at the source, thread-safely, by
                # _setup_logging's `lightkurve.utils` CRITICAL+1 +
                # propagate=False filter above. The stdout swap suppressed
                # nothing and is removed rather than lock-protected.
                lc = _read_lightcurve(path)
                lc_list.append(lc)

            if len(lc_list) == 1:
                lc = lc_list[0]
            else:
                collection = lk.LightCurveCollection(lc_list)
                lc = collection.stitch()

            # Clean: remove NaNs, remove only POSITIVE outliers (preserve transits)
            lc = lc.remove_nans().remove_outliers(
                sigma_upper=5, sigma_lower=float('inf'))
            # Fill small gaps (within sectors) to prevent detrend edge effects
            # Do NOT fill large gaps (between sectors separated by months/years)
            # as this creates millions of filler points that dilute transit signals
            try:
                max_gap_days = 5  # only fill gaps < 5 days
                time_vals = lc.time.value
                if len(time_vals) > 1:
                    gaps = np.diff(time_vals)
                    max_actual_gap = np.max(gaps)
                    if max_actual_gap < max_gap_days:
                        lc = lc.fill_gaps()
                    else:
                        self.logger.info(f"Skipping fill_gaps: max gap={max_actual_gap:.0f}d > {max_gap_days}d")
            except (ValueError, Exception):
                pass

            # First pass: wotan biweight detrend (replaces lightkurve's
            # Savitzky-Golay flatten — the S-G suppressed ~15% of transit
            # depth per Hippke+ 2019, driving `no_bls_candidate` misses).
            t_raw = np.array(lc.time.value, dtype=np.float64)
            f_raw = np.array(lc.flux.value, dtype=np.float64)
            # Snapshot raw flux (post-stitch, post-fill_gaps, pre-wotan) so the
            # multi-planet BLS loop can re-detrend with an accumulated transit mask
            # (Hippke+ 2019 §5).
            raw_flux_arr = f_raw.copy()
            finite_mask = np.isfinite(t_raw) & np.isfinite(f_raw)
            flat_flux = np.full_like(f_raw, np.nan)
            flat_flux[finite_mask] = detrend_with_wotan(
                t_raw[finite_mask], f_raw[finite_mask])

            # Iterative refinement: quick BLS, mask detected transits,
            # re-detrend with those points flagged so the trend fit ignores
            # them (wotan preserves full depth with explicit mask).
            try:
                m = np.isfinite(t_raw) & np.isfinite(flat_flux)
                quick_result = bls_fast(t_raw[m], flat_flux[m],
                    period_min=0.5, period_max=50.0, n_periods=2000,
                    n_bins=100, n_dur_steps=5)
                if quick_result["snr"] >= 7.0:
                    phase = ((t_raw - quick_result["t0"]) % quick_result["period"]) / quick_result["period"]
                    half_dur = (quick_result["duration"] / quick_result["period"])  # 2× duration
                    transit_mask = (phase < half_dur) | (phase > 1.0 - half_dur)
                    # Re-detrend with transit points explicitly masked
                    flat_flux[finite_mask] = detrend_with_wotan(
                        t_raw[finite_mask], f_raw[finite_mask],
                        transit_mask=transit_mask[finite_mask])
                    self.logger.info(f"Iterative detrend: masked P={quick_result['period']:.4f}d SNR={quick_result['snr']:.1f}")
            except Exception:
                pass  # keep first-pass flatten if iterative fails

            time_arr = t_raw
            flux_arr = flat_flux
            # PDCSAP is already ~1.0; only re-normalize if way off
            median = np.nanmedian(flux_arr)
            if median > 1.5 or median < 0.5:
                flux_arr = flux_arr / median

            # Sort by time (stitch can return unsorted sectors)
            sort_idx = np.argsort(time_arr)
            time_arr = time_arr[sort_idx]
            flux_arr = flux_arr[sort_idx]
            raw_flux_arr = raw_flux_arr[sort_idx]

            # Remove NaN/inf across all three arrays so downstream callers
            # (BLS, multi-planet re-detrend) get fully-finite, index-aligned data.
            mask = (np.isfinite(time_arr)
                    & np.isfinite(flux_arr)
                    & np.isfinite(raw_flux_arr))
            time_arr = time_arr[mask]
            flux_arr = flux_arr[mask]
            raw_flux_arr = raw_flux_arr[mask]

            # Tag data source (spoc vs ffi) so the CNN routing can pick the right model.
            meta["data_source"] = _detect_data_source(local_paths)
            self.logger.info(f"Loaded TIC {tic_id}: {len(time_arr)} points, {len(sectors)} sectors, "
                  f"{time_arr[-1]-time_arr[0]:.1f} days, source={meta['data_source']}")
            return time_arr, flux_arr, raw_flux_arr, len(sectors), meta

        except Exception as e:
            self.logger.warning(f"Error reading FITS for TIC {tic_id}: {e}")
            self.last_error = f"{time.strftime('%H:%M:%S')} {type(e).__name__}: {str(e)[:200]}"
            fallback = self._try_ffi_fallback(tic_id, reason="lightkurve_exception")
            if fallback is not None:
                return fallback
            return None, None, None, 0, {}

    def _try_ffi_fallback(self, tic_id, reason):
        """Last-resort: fetch FFI from server's Phase 2.5 cache.

        See spec: docs/superpowers/specs/2026-05-08-ffi-fallback-on-lc-load-failure-design.md
        """
        if not self.config.get("ffi_fallback_on_load_fail", False):
            return None
        try:
            res = fetch_ffi(self.server, str(tic_id), token=self.token,
                            timeout=self.config.get("ffi_fetch_timeout", 30.0))
        except Exception as exc:
            self.logger.warning(
                "TIC %s: FFI fallback fetch raised %s", tic_id, exc)
            return None
        if res is None:
            self.logger.info(
                "TIC %s: SPOC load failed (%s) -> FFI fallback also failed "
                "(server 404)", tic_id, reason)
            return None
        time_arr, flux_arr, _q = res
        # Validate: drop NaN, require >= 1 sector and >= 100 finite points
        finite = np.isfinite(time_arr) & np.isfinite(flux_arr)
        time_arr = time_arr[finite]
        flux_arr = flux_arr[finite]
        if len(time_arr) < 100:
            self.logger.info(
                "TIC %s: FFI fallback rejected — only %d finite points",
                tic_id, len(time_arr))
            return None
        # Best-effort metadata refetch (5 s timeout). RA/Dec/Tmag may end up
        # None; downstream code tolerates this (UI formatter, FPP skip path).
        meta = {"tic_id": str(tic_id), "data_source": "ffi",
                "lc_load_recovered": True,
                "ra": None, "dec": None, "tmag": None, "sectors": []}
        try:
            resp = requests.get(
                f"{self.server}/api/metadata/{tic_id}",
                headers=self._headers(), timeout=5)
            if resp.status_code == 200:
                m = resp.json()
                for k in ("ra", "dec", "tmag", "sectors"):
                    if k in m:
                        meta[k] = m[k]
        except Exception:
            pass  # metadata best-effort; absence does not block fallback
        n_sectors = len(meta.get("sectors") or []) or 1
        raw_flux_arr = flux_arr.copy()
        self.logger.info(
            "TIC %s: SPOC load failed (%s) -> FFI fallback OK "
            "(%d sectors, %d points, source=ffi)",
            tic_id, reason, n_sectors, len(time_arr))
        return time_arr, flux_arr, raw_flux_arr, n_sectors, meta

    def _base_result(self, task):
        """Common /api/worker/result skeleton shared by the success path
        (end of process_task), the exception-path fallback (main loop),
        and _aborted_result.

        Before this (2026-07-02 cleanup review, item 7), each of those
        three built its own bespoke dict with a slightly different key
        set. exonit/server.py's worker_result handler always reads via
        ``body.get(key, default)`` so missing keys were silently
        tolerated, but the drift made it hard to tell which keys a given
        code path actually populates — this makes the full contract
        explicit in one place (absent values are None, not just missing).
        Additive only: every key server.py's handler reads is present
        here; nothing is renamed or removed, so the server contract is
        unchanged. Callers ``dict.update()`` their specifics on top.
        """
        return {
            "task_id": task.get("task_id"),
            "tic_id": task.get("tic_id"),
            "candidates": [],
            "vetting_results": [],
            "cnn_scores": [],
            "completeness": None,
            "processing_time_seconds": 0,
            "gpu_used": False,
            "n_sectors": 0,
            "n_points": 0,
            "time_span_days": None,
            "ra": None,
            "dec": None,
            "tmag": None,
            "timings": None,
            "lc_load_recovered": False,
            "status": "completed",
        }

    def _aborted_result(self, task, t_start, timings, lc_load_recovered=False):
        """Return a partial result when stop is requested mid-task."""
        elapsed = time.time() - t_start
        self.logger.info(f"TIC {task['tic_id']}: aborted by user after {elapsed:.1f}s")
        result = self._base_result(task)
        result.update({
            "processing_time_seconds": round(elapsed, 2),
            "status": "aborted",
            "timings": timings,
            "lc_load_recovered": lc_load_recovered,
        })
        return result

    def process_task(self, task):
        tic_id = task["tic_id"]
        config = task.get("config", {})
        # Local worker config can override task-level knobs. Used e.g. to
        # disable run_injection on hosts where Numba's workqueue layer is
        # not thread-safe (M1 macOS crashes when Injection CPU runs in a
        # background thread alongside FPP CPU on main thread).
        for key in ("run_injection", "medium_period_protected_slot",
                    "medium_slot_min_snr", "medium_slot_min_days",
                    "medium_slot_max_days", "medium_slot_ranges",
                    "cross_source_enabled"):
            if key in self.config:
                config[key] = self.config[key]
        self.status = "working"
        self.current_tic = tic_id
        self.logger.info(f"Processing TIC {tic_id}...")
        t_start = time.time()
        timings = {}  # Phase timing telemetry
        self._star_start_time = time.time()

        # Autonomous mode only: fetch metadata + FITS from the server.
        # (The legacy base64 time/flux task mode was removed 2026-06-09:
        # no production server ever sent it — its only emitter was the dead
        # exonit/coordinator.py — and it left raw_flux_arr undefined.)
        self.current_phase = "load"
        t_phase = time.time()
        # raw_flux_arr: consumed by multi-planet re-detrend loop (Task 3).
        time_arr, flux_arr, raw_flux_arr, n_sectors, star_meta = self._load_star(tic_id)
        if time_arr is None:
            elapsed = time.time() - t_start
            self.logger.warning(f"TIC {tic_id}: failed to load data ({elapsed:.1f}s)")
            return {
                "task_id": task["task_id"], "tic_id": tic_id,
                "candidates": [], "vetting_results": [],
                "cnn_scores": [], "completeness": None,
                "processing_time_seconds": round(elapsed, 2),
                "gpu_used": False, "n_sectors": 0, "n_points": 0,
                "status": "failed",
                "error": "load_failed",
                "lc_load_recovered": False,
            }
        lc_load_recovered = bool(
            star_meta.get("lc_load_recovered") if isinstance(star_meta, dict) else False)
        timings["load"] = round(time.time() - t_phase, 2)

        # Show star panel with real metadata
        if self.ui:
            def _fmt_ra(deg):
                if not deg: return "?"
                h = deg / 15.0
                m = (h % 1) * 60
                s = (m % 1) * 60
                return f"{int(h)}h{int(m)}m{s:.0f}s"
            def _fmt_dec(deg):
                if not deg: return "?"
                sign = "+" if deg >= 0 else "-"
                d = abs(deg)
                m = (d % 1) * 60
                s = (m % 1) * 60
                return f"{sign}{int(d)}d{int(m)}'{s:.0f}\""
            self.ui.star_start(
                tic_id,
                _fmt_ra(star_meta.get("ra")),
                _fmt_dec(star_meta.get("dec")),
                star_meta.get("tmag", 0),
                star_meta.get("sectors", []),
                "?",
                0,
                star_meta.get("stellar_radius", 0))

        # BLS multi-planet search with hole-filling
        # BLS dispatch: run_bls picks best backend automatically
        # (numba two-pass > CUDA > Metal > OpenCL > numpy)
        bls_fn = run_bls
        use_gpu = _bls_is_gpu()
        self.logger.info(f"Using {_bls_backend()} BLS")

        self.current_phase = "bls"
        self.phase_progress = None
        if self.ui: self.ui.phase_start("BLS search")
        t_phase = time.time()
        candidates = []
        current_flux = flux_arr.copy()
        accumulated_mask = np.zeros(len(time_arr), dtype=bool)
        n_wotan_fails = 0
        period_min = config.get("period_min", 0.5)
        # Adaptive period_max: legacy default 50d misses 100% of TOIs with
        # period >100d and 59% of 30-100d. For TICs with long baseline
        # (e.g. 12+ sectors ~ 350d), extend the search up to time_span/divisor
        # so BLS can still find ≥`period_max_min_transits` transits. Capped
        # at `period_max_cap` to keep the BLS grid bounded. TICs with short
        # baseline keep the legacy default — no extra cost.
        time_span_days_local = (float(time_arr[-1] - time_arr[0])
                                if len(time_arr) > 1 else 0.0)
        period_max_legacy = float(config.get("period_max", 50.0))
        period_max_cap = float(config.get("period_max_cap", 200.0))
        period_max_divisor = float(config.get(
            "period_max_min_transits", 2.5))  # need ≥2.5 transits in span
        if time_span_days_local > period_max_legacy * period_max_divisor:
            period_max_dynamic = min(
                time_span_days_local / period_max_divisor, period_max_cap)
        else:
            period_max_dynamic = period_max_legacy
        if period_max_dynamic > period_max_legacy:
            self.logger.info(
                f"Long-period sweep extends BLS to {period_max_dynamic:.0f}d "
                f"(main loop stays at {period_max_legacy:.0f}d, "
                f"time_span={time_span_days_local:.0f}d)"
            )

        for i in range(config.get("max_planets", 5)):
            # Re-detrend with accumulated mask starting at i>=1 (Hippke+ 2019 §5).
            # Defensive fallbacks: skip if mask coverage is too high, fall back
            # to hole-fill if wotan raises, abort after 2 consecutive failures.
            if i >= 1 and accumulated_mask.any():
                coverage = accumulated_mask.sum() / len(time_arr)
                if coverage > 0.30:
                    self.logger.warning(
                        f"Multi-planet mask coverage {coverage:.1%}, "
                        "skipping re-detrend"
                    )
                else:
                    try:
                        redetrended = detrend_with_wotan(
                            time_arr, raw_flux_arr,
                            transit_mask=accumulated_mask,
                        )
                        # wotan's flat_flux keeps in-transit samples (the mask
                        # only steers the trend fit). Without suppressing prior
                        # transits BLS re-finds the same signal every iteration
                        # — production 2026-04-21 showed ~85% of multi-planet
                        # TICs returning the same period ×5. Fill both NaN
                        # (edge_cutoff) and accumulated_mask with the OOT
                        # median to force BLS onto a new signal.
                        fill_mask = ~np.isfinite(redetrended) | accumulated_mask
                        good = np.isfinite(redetrended) & ~accumulated_mask
                        if good.any():
                            fill_val = np.nanmedian(redetrended[good])
                        else:
                            fill_val = 1.0
                            self.logger.warning(
                                f"Multi-planet re-detrend i={i}: no finite "
                                "out-of-transit points, filling with 1.0"
                            )
                        redetrended = redetrended.copy()
                        redetrended[fill_mask] = fill_val
                        current_flux = redetrended
                        # Reset the failure counter on success so the "2
                        # consecutive failures" abort only fires on a true
                        # consecutive streak (spec wording in plan §Task 3).
                        n_wotan_fails = 0
                        self.logger.info(
                            f"Multi-planet re-detrend i={i}: mask coverage "
                            f"{coverage:.1%} ({len(candidates)} candidates)"
                        )
                    except Exception as e:
                        n_wotan_fails += 1
                        self.logger.warning(
                            f"Multi-planet re-detrend failed ({e}); "
                            "falling back to hole-fill"
                        )
                        if n_wotan_fails >= 2:
                            self.logger.error(
                                "Multi-planet re-detrend failed twice in a row; "
                                "aborting further iterations"
                            )
                            break

            # Main multi-planet loop uses the LEGACY period_max (50d) to
            # avoid contamination from long-period stellar variability
            # peaks. The 2026-05-10 stratified-100 re-run showed that a
            # period_max=200d in the main loop lets BLS lock onto
            # P=180+d unphysical peaks, masks 45 % of the lightcurve to
            # filter them, and then never finds short-period real
            # signals (TIC 116483514 went from 4 candidates → 0). The
            # adaptive `period_max_dynamic` is reserved for the
            # long-period sweep below, which uses the wider wotan window
            # to actually preserve long-period transits.
            result = bls_fn(time_arr, current_flux,
                period_min=period_min,
                period_max=period_max_legacy,
                n_periods=config.get("n_periods", 50000))

            # Symmetric harmonic resolution (audit §76-85).
            seed_period = result["period"]
            result = resolve_harmonic(
                time_arr, current_flux, result,
                period_min=period_min,
                period_max=period_max_legacy,
                bls_fn=bls_fn,
                data_source=star_meta.get("data_source", "spoc"),
                long_period_recovery=config.get(
                    "harmonic_long_period_recovery", False),
                long_period_min_days=config.get(
                    "harmonic_long_period_min_days", 10.0),
                long_snr_factor=config.get(
                    "harmonic_long_snr_factor", 0.5),
            )
            if abs(result["period"] - seed_period) / max(seed_period, 1e-9) > 0.01:
                self.logger.info(
                    f"Harmonic: P={seed_period:.4f}d -> "
                    f"P={result['period']:.4f}d"
                )

            if result["snr"] < config.get("snr_threshold", 7.0):
                break

            # Physical T14 gate: BLS allows durations up to 15% of period
            # which is ~20× the physical bound at long periods. Reject
            # candidates with T14 > 3 × T14_max_jup — these are systematics
            # or noise, never real planets. 2026-04-21 production: 15+
            # candidates at P>30d with dur_frac=0.15 were all spurious.
            t14_max = physical_t14_max(
                result["period"],
                star_meta.get("stellar_radius"),
                star_meta.get("stellar_mass"),
            )
            if t14_max > 0 and result["duration"] > 3.0 * t14_max:
                # Mask the rejected peak's transits so the next iteration's
                # BLS cannot re-find it, then continue. Breaking here (as
                # the original fix did) killed the whole multi-planet search
                # whenever the first BLS peak was a long-period systematic —
                # baseline-100 2026-04-21 C lost 8 TICs (7 CP + 1 KP) this
                # way, including a P=1.45d planet at ExoMiner score 0.9993
                # hidden behind a P=35.8d dur=5.4d systematic.
                self.logger.info(
                    f"Rejecting unphysical candidate P={result['period']:.2f}d "
                    f"duration={result['duration']:.3f}d "
                    f"> 3×T14_max={3*t14_max:.3f}d — masking and continuing"
                )
                phase = ((time_arr - result["t0"]) % result["period"]) / result["period"]
                half_dur = (result["duration"] / result["period"]) / 2
                in_transit = (phase < half_dur) | (phase > 1.0 - half_dur)
                accumulated_mask |= in_transit
                median_flux = np.nanmedian(current_flux)
                current_flux = current_flux.copy()
                current_flux[in_transit] = median_flux
                continue
            candidates.append({
                "candidate_num": i + 1,
                "period_days": round(result["period"], 4),
                "t0": round(result["t0"], 4),
                "duration_days": round(result["duration"], 4),
                "depth_ppm": round(result["depth_ppm"], 1),
                "snr": round(result["snr"], 1),
                "bls_power": round(result["power"], 4),
                "n_transits_expected": int((time_arr[-1] - time_arr[0]) / result["period"]),
            })

            # Accumulate transit mask (for re-detrend at next iteration)
            phase = ((time_arr - result["t0"]) % result["period"]) / result["period"]
            half_dur = (result["duration"] / result["period"]) / 2
            in_transit = (phase < half_dur) | (phase > 1.0 - half_dur)
            accumulated_mask |= in_transit

            # Defensive fallback: if re-detrend fails next iteration, keep a
            # hole-filled current_flux as a backup so we never fall below the
            # previous (median-fill) quality bar.
            median_flux = np.nanmedian(current_flux)
            current_flux = current_flux.copy()
            current_flux[in_transit] = median_flux

        # Medium-period protected slot (config-gated, default off). The main
        # loop fills its candidate slots with the strongest BLS peaks by SNR;
        # on stars with strong short-period signals a real 10-50d transit is
        # found by BLS but outranked and never gets a slot (slot starvation).
        # Big-run-v4 audit (2026-05-22, scripts/diag_buried_signal.py): 30/40
        # missed 10-30d KP/CP had a >=7-SNR peak at the true period that lost
        # the ranking. Run ONE restricted-range BLS over [med_min, med_max] on
        # the standard (already transit-masked) flux and add the strongest
        # qualifying peak to a protected slot if it is not already represented.
        # The restricted range ignores the dominant short-period peaks; the
        # accumulated mask prevents re-finding candidates already captured. The
        # >=2-transits guard + physical-T14 gate keep the extra slot honest.
        if config.get("medium_period_protected_slot", True):
            # The protected slot uses a HIGHER SNR floor than the main loop.
            # Big-run-v4 audit: slot-starved real planets keep a strong peak
            # (median SNR ~16.7) because they only lost the RANKING to stronger
            # short-period signals, while red-noise / systematic peaks cluster
            # at SNR ~9. A floor of 12 keeps the recoveries while cutting
            # spurious adds ~78%.
            snr_thr = float(config.get("medium_slot_min_snr", 12.0))
            # Search ranges (days). Default is a single [min,max] from the
            # legacy keys (back-compat with the validated [10,50] slot); an
            # explicit list can also cover short periods, e.g.
            # [[3,10],[10,50]]. Each range adds AT MOST one protected
            # candidate, so disjoint ranges never steal slots from each other.
            # Big-run-v4 root-cause (scripts/diag_detectable_misses.py): 50/195
            # detectable misses were outranked real signals (SNR>=7 at the true
            # period), spread across 3-10d (19) and 10-30d (22). Pre-producció
            # validation (scripts/compare_slot310_validation.py, 2026-05-23):
            # extending to [[3,10],[10,50]] recovered +22 (3-10d) and +19
            # (10-30d) KP/CP over the v4 baseline with zero specificity cost
            # (EM-FFI delta +0.000, CNN -0.021) -> adopted as the default.
            if "medium_slot_ranges" in config:
                slot_ranges = config["medium_slot_ranges"]
            elif ("medium_slot_min_days" in config
                  or "medium_slot_max_days" in config):
                # Legacy single-range keys (back-compat).
                slot_ranges = [[float(config.get("medium_slot_min_days", 10.0)),
                                float(config.get("medium_slot_max_days", 50.0))]]
            else:
                slot_ranges = [[3.0, 10.0], [10.0, 50.0]]
            for _rng in slot_ranges:
                try:
                    med_min = float(_rng[0])
                    med_max = min(float(_rng[1]), period_max_legacy)
                    if not (med_max > med_min
                            and time_span_days_local >= 2.0 * med_min):
                        continue
                    rm = bls_fn(time_arr, current_flux,
                                period_min=med_min, period_max=med_max,
                                n_periods=config.get("n_periods", 50000))
                    rm_p = rm.get("period", 0.0)
                    already = any(
                        rm_p > 0 and abs(rm_p - c["period_days"])
                        / max(c["period_days"], 1e-9) < 0.02
                        for c in candidates)
                    t14m = physical_t14_max(
                        rm_p, star_meta.get("stellar_radius"),
                        star_meta.get("stellar_mass"))
                    unphysical = (t14m > 0 and rm["duration"] > 3.0 * t14m)
                    enough_transits = (rm_p > 0
                                       and time_span_days_local / rm_p >= 2.0)
                    if (rm.get("snr", 0.0) >= snr_thr and not already
                            and not unphysical and enough_transits):
                        candidates.append({
                            "candidate_num": len(candidates) + 1,
                            "period_days": round(rm["period"], 4),
                            "t0": round(rm["t0"], 4),
                            "duration_days": round(rm["duration"], 4),
                            "depth_ppm": round(rm["depth_ppm"], 1),
                            "snr": round(rm["snr"], 1),
                            "bls_power": round(rm.get("power", 0.0), 4),
                            "n_transits_expected": int(
                                time_span_days_local / rm["period"]),
                        })
                        self.logger.info(
                            f"Protected slot [{med_min:.0f},{med_max:.0f}]d: "
                            f"P={rm['period']:.4f}d SNR={rm['snr']:.1f} "
                            f"added (slot {len(candidates)})")
                except Exception as e:
                    self.logger.warning(
                        f"Protected slot [{_rng}] failed: {e}")

        # Long-period sweep with a wider wotan window. Hot Jupiters and
        # other long-period planets have transit durations >>0.5d, so the
        # default biweight window (3 × T₁₄ ~0.5d) absorbs the signal and
        # the multi-planet loop above finds nothing. Big-run-v3 evidence
        # (2026-05-09): 31 KP/CPs with depth 5-12k ppm produced 0
        # candidates despite long baselines. This sweep re-detrends with
        # a wider window (≈3 × T₁₄ for a 12-24 h transit) and runs BLS
        # over the long-period regime only.
        #
        # Always runs when time_span supports long-period detection.
        # If the multi-planet loop already filled all candidate slots,
        # the sweep substitutes the weakest existing candidate when its
        # SNR is exceeded by a long-period signal — that catches multi-
        # candidate stars where the long-period transit was masked out.
        if time_span_days_local > config.get("wotan_long_threshold_days", 100.0):
            try:
                # Try multiple windows: shorter for crisp transits, longer
                # for very wide ones. First match (≥SNR threshold) wins.
                # Windows are tried in order. Big-run-v3 5-TIC re-test
                # (2026-05-10) showed that 2.0d catches transits T₁₄≈12h
                # and 4.0d catches T₁₄≈24h. The 8.0d window was dropped
                # in the v3-rerun optimisation — it added a third wotan
                # pass per TIC for diminishing returns (only TIC
                # 113921235 saw a marginal SNR gain from 6.7→6.9, still
                # below threshold).
                long_windows = config.get("wotan_long_windows_days",
                                          [2.0, 4.0])
                long_period_min = float(config.get(
                    "wotan_long_period_min_days", 30.0))
                # Big-run-v3 rerun optimisation (2026-05-10): legacy
                # n_periods (50k) was bumped to 200k for finer resolution
                # at long periods, but the resulting BLS pass cost 30-50s
                # per long-baseline TIC (5× slower than the main loop)
                # and dropped throughput from ~232 TIC/h to ~140 TIC/h.
                # Empirically the recoveries we did get (e.g. TIC
                # 402898317 P=50d SNR=10) work fine at 50k. Keep the
                # legacy budget — the BLS frequency grid still scales
                # with period_max_dynamic so wider sweeps don't lose
                # resolution further.
                long_n_periods = int(config.get(
                    "wotan_long_n_periods",
                    max(50000, int(50000 * period_max_dynamic / 50.0))))
                # Iter 5 (2026-05-10): keep the snr_threshold at the same
                # 7.0 baseline as the main loop. The earlier 0.5-below
                # tweak (iter 3) added two false-positive long candidates
                # for TICs 273234825 and 209464063 that CNN then filtered
                # to score 0; harmless but polluted the DB with no
                # real-recovery gain.
                snr_thresh = float(config.get(
                    "wotan_long_snr_threshold",
                    config.get("snr_threshold", 7.0)))
                # Iter 4 (2026-05-10): always try all windows and keep the
                # highest-SNR result. The earlier early-exit ("first
                # window crossing threshold wins") was wasting wider
                # windows that found a better signal — saw 273234825 stop
                # at window=2.0d SNR=6.7 when window=8.0d would have
                # given 6.9.
                best_long = None
                best_window = None
                for long_window in long_windows:
                    long_window = float(long_window)
                    flat_long = detrend_with_wotan(
                        time_arr, raw_flux_arr,
                        window_length=long_window,
                        transit_mask=(accumulated_mask
                                      if accumulated_mask.any() else None),
                    )
                    bad = ~np.isfinite(flat_long) | accumulated_mask
                    good = ~bad
                    fill_val = (float(np.nanmedian(flat_long[good]))
                                if good.any() else 1.0)
                    flat_long_clean = flat_long.copy()
                    flat_long_clean[bad] = fill_val
                    rl = bls_fn(
                        time_arr, flat_long_clean,
                        period_min=long_period_min,
                        period_max=period_max_dynamic,
                        n_periods=long_n_periods)
                    if (best_long is None
                            or rl["snr"] > best_long["snr"]):
                        best_long = rl
                        best_window = long_window
                if best_long is None:
                    pass  # detrend exception path below if needed
                elif best_long["snr"] < snr_thresh:
                    self.logger.info(
                        f"Long-period sweep: no signal "
                        f"(best SNR={best_long['snr']:.1f} "
                        f"< {snr_thresh}, tried windows {long_windows})"
                    )
                else:
                    t14_max_long = physical_t14_max(
                        best_long["period"],
                        star_meta.get("stellar_radius"),
                        star_meta.get("stellar_mass"),
                    )
                    unphysical = (t14_max_long > 0
                                  and best_long["duration"] > 3.0 * t14_max_long)
                    if unphysical:
                        self.logger.info(
                            f"Long-period sweep: rejected unphysical "
                            f"P={best_long['period']:.2f}d "
                            f"duration={best_long['duration']:.3f}d "
                            f"> 3×T14_max={3 * t14_max_long:.3f}d"
                        )
                    else:
                        new_cand = {
                            "period_days": round(best_long["period"], 4),
                            "t0": round(best_long["t0"], 4),
                            "duration_days": round(best_long["duration"], 4),
                            "depth_ppm": round(best_long["depth_ppm"], 1),
                            "snr": round(best_long["snr"], 1),
                            "bls_power": round(best_long["power"], 4),
                            "n_transits_expected": int(
                                time_span_days_local / best_long["period"]),
                        }
                        max_planets = config.get("max_planets", 5)
                        if len(candidates) < max_planets:
                            new_cand["candidate_num"] = len(candidates) + 1
                            candidates.append(new_cand)
                            self.logger.info(
                                f"Long-period sweep "
                                f"(window={best_window}d): "
                                f"P={best_long['period']:.4f}d "
                                f"SNR={best_long['snr']:.1f} accepted (slot {new_cand['candidate_num']})"
                            )
                        else:
                            # Replace the weakest existing candidate if the
                            # long-period one beats it on SNR.
                            weakest_idx = min(
                                range(len(candidates)),
                                key=lambda i: candidates[i].get("snr", 0))
                            weakest = candidates[weakest_idx]
                            if best_long["snr"] > weakest.get("snr", 0):
                                new_cand["candidate_num"] = weakest["candidate_num"]
                                candidates[weakest_idx] = new_cand
                                self.logger.info(
                                    f"Long-period sweep "
                                    f"(window={best_window}d): "
                                    f"P={best_long['period']:.4f}d "
                                    f"SNR={best_long['snr']:.1f} replaces "
                                    f"slot {weakest['candidate_num']} "
                                    f"(was P={weakest['period_days']}d "
                                    f"SNR={weakest.get('snr', 0)})"
                                )
                            else:
                                self.logger.info(
                                    f"Long-period sweep "
                                    f"(window={best_window}d): "
                                    f"P={best_long['period']:.4f}d "
                                    f"SNR={best_long['snr']:.1f} found but "
                                    f"weaker than all existing candidates"
                                )
            except Exception as e:
                self.logger.warning(f"Long-period sweep failed: {e}")
        timings["bls"] = round(time.time() - t_phase, 2)
        if self.ui: self.ui.phase_done(timings["bls"])

        if not self.running:
            return self._aborted_result(task, t_start, timings, lc_load_recovered=lc_load_recovered)

        self.current_phase = "vetting"
        self.phase_progress = None
        if self.ui: self.ui.phase_start("Vetting checks")
        t_phase = time.time()
        vetting_results = [vet_candidate(c, candidates, time_arr, flux_arr) for c in candidates]
        timings["vetting"] = round(time.time() - t_phase, 2)
        if self.ui: self.ui.phase_done(timings["vetting"])

        if not self.running:
            return self._aborted_result(task, t_start, timings, lc_load_recovered=lc_load_recovered)

        self.current_phase = "cnn"
        self.phase_progress = None
        if self.ui: self.ui.phase_start("CNN classification")
        t_phase = time.time()
        cnn_scores = []
        # Route to FFI classifier when data_source=='ffi' AND we have an FFI model.
        data_source = star_meta.get("data_source", "spoc") if isinstance(star_meta, dict) else "spoc"
        active_classifier = self.classifier_ffi if (data_source == "ffi" and self.classifier_ffi) else self.classifier
        if active_classifier and candidates:
            self.logger.info(f"CNN routing: data_source={data_source} -> {type(active_classifier).__name__}")
            for c in candidates:
                score = active_classifier.classify(time_arr, flux_arr, c["period_days"], c["t0"], c["duration_days"])
                cnn_scores.append(score)
        else:
            cnn_scores = [None] * len(candidates)
        timings["cnn"] = round(time.time() - t_phase, 2)
        if self.ui: self.ui.phase_done(timings["cnn"])

        if not self.running:
            return self._aborted_result(task, t_start, timings, lc_load_recovered=lc_load_recovered)

        # TRICERATOPS FPP + Injection-recovery
        # Two paths:
        #   - background (GPU injection):  injection on GPU thread while
        #     FPP runs on CPU main thread. Maximises throughput.
        #   - inline (CPU injection):      injection runs sequentially
        #     AFTER FPP. Two concurrent numba parallel regions (FPP + inj)
        #     on a single CPU are unsafe with Numba's workqueue layer
        #     (Apple M1 + macOS SIGABRTs under multi-planet re-detrend).
        # Selection: config.injection_mode in {"auto", "inline", "background"}.
        # "auto" (default): background iff injection backend is GPU.
        self.current_phase = "triceratops"
        self.phase_progress = None
        t_phase = time.time()
        fpp_results = [None] * len(candidates)
        completeness = None
        run_injection = config.get("run_injection", True)

        fpp_eligible = []
        if _triceratops_available and candidates:
            for i, c in enumerate(candidates):
                cnn = cnn_scores[i] if i < len(cnn_scores) else None
                vet = vetting_results[i] if i < len(vetting_results) else {}
                if should_run_triceratops(c, vet, cnn_score=cnn):
                    fpp_eligible.append(i)

        try:
            from injection import get_backend_name as _inj_backend
            inj_label = _inj_backend()
        except Exception:
            inj_label = "?"
        injection_mode = config.get("injection_mode", "auto")
        if injection_mode == "auto":
            # Run injection in the background when a *different* compute unit is
            # free to overlap it: either injection itself is on the GPU (legacy),
            # or FPP holds the GPU and injection runs on the CPU concurrently.
            inj_background = ("CPU" not in inj_label) or self._fpp_prefers_gpu
        else:
            inj_background = (injection_mode == "background")

        # Memory guard (Phase 3 #8): backgrounding injection overlaps it with FPP
        # — two heavy phases at once. Across many instances per host that overlap
        # is what drives a box into swap. Under memory pressure, force injection
        # INLINE (sequential after FPP): slower for this star but it caps peak
        # memory and avoids swap death. Tunable per host in workers.json
        # (max_swap_gb / min_avail_ram_gb); fails open if psutil is unavailable.
        # An explicit injection_mode=background is respected as the operator's
        # choice and not overridden here (only the "auto" path is guarded).
        if inj_background and injection_mode == "auto":
            try:
                import psutil
                swap_used_gb = psutil.swap_memory().used / (1024**3)
                avail_gb = psutil.virtual_memory().available / (1024**3)
                max_swap = float(config.get("max_swap_gb", 4.0))
                min_avail = float(config.get("min_avail_ram_gb", 1.5))
                if swap_used_gb > max_swap or avail_gb < min_avail:
                    inj_background = False
                    self.logger.warning(
                        "memory pressure (swap=%.1f GB, avail=%.1f GB) -> running "
                        "injection inline instead of overlapping FPP",
                        swap_used_gb, avail_gb)
            except Exception:
                pass  # psutil missing / probe failed -> keep the auto decision

        # injection_max_seconds: wall-clock deadline so a giant multi-sector
        # FFI target (e.g. TIC 243278113, 173k points / 43 sectors) can't
        # monopolise a worker for 30-45 min on CPU injection-recovery.
        # Default 5 min never fires on normal targets (they finish in
        # seconds); configurable per host via config.json.
        injection_kwargs = {
            "stellar_radius": star_meta.get("stellar_radius") or 1.0,
            "stellar_mass": star_meta.get("stellar_mass"),
            "snr_threshold": config.get("snr_threshold", 7.0),
            "n_injections_per_cell": int(config.get("n_injections_per_cell", 10)),
            "bls_n_periods": int(config.get("injection_n_periods", 5000)),
            "bls_n_bins": int(config.get("injection_n_bins", 150)),
            "bls_n_dur_steps": int(config.get("injection_n_dur_steps", 10)),
            "max_seconds": config.get("injection_max_seconds", 300.0),
        }

        inj_result_holder = [None, None, None]  # [completeness, error, elapsed]
        inj_thread = None
        if fpp_eligible and run_injection and self.running and inj_background:
            def _run_injection():
                t_bg_inj = time.time()
                try:
                    inj_result_holder[0] = run_injection_recovery(
                        time_arr, flux_arr, **injection_kwargs)
                except Exception as e:
                    inj_result_holder[1] = e
                finally:
                    inj_result_holder[2] = round(time.time() - t_bg_inj, 2)

            inj_thread = threading.Thread(target=_run_injection, daemon=True)
            inj_thread.start()
            try:
                from fpp_native import get_fpp_backend_info as _get_fpp_bi
                _fpp_bn = _get_fpp_bi()[0]
            except Exception:
                _fpp_bn = "unknown"
            _fpp_unit = f"GPU ({_fpp_bn})" if self._fpp_prefers_gpu else "CPU"
            self.logger.info(
                f"Injection started in background ({inj_label}), FPP on {_fpp_unit}")

        # FPP on main thread. Forced to numba CPU only when injection holds the
        # GPU; when FPP itself prefers the GPU (Apple Silicon fused kernel) it
        # keeps the GPU and injection runs on the CPU concurrently.
        fpp_force_cpu = (inj_thread is not None) and not self._fpp_prefers_gpu
        if fpp_eligible:
            if self.ui: self.ui.phase_start("FPP Monte Carlo")
            for done_count, i in enumerate(fpp_eligible):
                if not self.running:
                    break
                c = candidates[i]
                if self.ui:
                    self.ui.phase_update(done_count / len(fpp_eligible),
                        f"FPP #{c['candidate_num']}/{len(fpp_eligible)}".ljust(20))
                try:
                    sectors_arr = np.arange(1, n_sectors + 1)
                    fpp = calc_fpp(
                        tic_id, c["period_days"], c["t0"],
                        c["duration_days"], c["depth_ppm"],
                        sectors_arr, time_arr, flux_arr,
                        workdir=config.get("cache_dir", "."),
                        server_url=self.server, token=self.token,
                        abort_check=lambda: not self.running,
                        force_cpu=fpp_force_cpu)
                    if fpp:
                        fpp_results[i] = fpp
                        c["triceratops_fpp"] = fpp["fpp"]
                        c["triceratops_nfpp"] = fpp.get("nfpp")
                        c["triceratops_prob_planet"] = fpp.get("prob_planet")
                        c["triceratops_verdict"] = fpp["verdict"]
                        self.logger.info(f"TRICERATOPS #{c['candidate_num']}: FPP={fpp['fpp']:.4f} NFPP={fpp.get('nfpp', -1):.4f} ({fpp['verdict']})")
                        if self.ui: self.ui.fpp_scenario_done(f"#{c['candidate_num']}")
                except FPPAborted:
                    self.logger.info(f"FPP aborted by user")
                    self.last_error = f"{time.strftime('%H:%M:%S')} FPPAborted: aborted by user"
                    break
                except Exception as e:
                    self.logger.warning(f"TRICERATOPS #{c['candidate_num']} failed: {e}")
                if self.ui:
                    self.ui.phase_update((done_count + 1) / len(fpp_eligible))
            if self.ui: self.ui.phase_done(round(time.time() - t_phase, 2))
        timings["triceratops"] = round(time.time() - t_phase, 2)

        if not self.running:
            if inj_thread:
                inj_thread.join(timeout=5)
            return self._aborted_result(task, t_start, timings, lc_load_recovered=lc_load_recovered)

        # Wait for parallel injection or run it now if it wasn't started
        self.current_phase = "injection"
        self.phase_progress = None
        t_inj = time.time()
        injection_elapsed = None
        if inj_thread:
            # Bounded join: `injection_kwargs["max_seconds"]` already caps
            # the work *inside* run_injection_recovery; the +30s grace here
            # just lets a near-deadline batch/BLS call in flight return
            # cleanly instead of the join firing right on top of it. If the
            # thread is still alive after that, injection is genuinely
            # hung/overrunning -- log it and move on treating injection as
            # unavailable for this star (no retry: retrying the same giant
            # target in series would just hang again) rather than blocking
            # the worker on this one TIC indefinitely.
            _inj_max_s = float(config.get("injection_max_seconds", 300.0))
            inj_thread.join(timeout=_inj_max_s + 30)
            if inj_thread.is_alive():
                self.logger.warning(
                    f"Injection overran max_seconds={_inj_max_s}s and did "
                    f"not return within the join grace period -> treating "
                    f"as unavailable for this star (no retry)")
                injection_elapsed = round(time.time() - t_inj, 2)
            elif inj_result_holder[0] is not None:
                injection_elapsed = inj_result_holder[2]
                completeness = inj_result_holder[0]
                self.logger.info(f"Injection completed (parallel with FPP)")
            elif inj_result_holder[1] is not None:
                injection_elapsed = inj_result_holder[2]
                # Injection failed in parallel — retry in series
                self.logger.warning(f"Injection failed in parallel ({inj_result_holder[1]}), retrying...")
                try:
                    retry_t = time.time()
                    completeness = run_injection_recovery(
                        time_arr, flux_arr, **injection_kwargs)
                    injection_elapsed = round(time.time() - retry_t, 2)
                except Exception as e:
                    self.logger.warning(f"Injection-recovery failed: {e}")
                    self.last_error = f"{time.strftime('%H:%M:%S')} {type(e).__name__}: {str(e)[:200]}"
        elif run_injection and self.running:
            # Inline injection: either no FPP was needed, or injection runs
            # on the same CPU as FPP and was deferred to avoid numba
            # workqueue races.
            if self.ui: self.ui.phase_start("Injection recovery")
            self.logger.info(f"Injection inline ({inj_label}) after FPP")
            try:
                completeness = run_injection_recovery(
                    time_arr, flux_arr, **injection_kwargs)
            except Exception as e:
                self.logger.warning(f"Injection-recovery failed: {e}")
                self.last_error = f"{time.strftime('%H:%M:%S')} {type(e).__name__}: {str(e)[:200]}"
            injection_elapsed = round(time.time() - t_inj, 2)
            if self.ui: self.ui.phase_done(injection_elapsed)
        timings["injection_wait"] = round(time.time() - t_inj, 2)
        timings["injection"] = injection_elapsed if injection_elapsed is not None else timings["injection_wait"]

        elapsed = time.time() - t_start
        timings["total"] = round(elapsed, 2)
        timings["n_points"] = len(time_arr)
        timings["n_sectors"] = n_sectors
        # Per-stage backends (each stage has its own accelerator path).
        # Reporting them separately avoids the previous confusion where the
        # single `gpu_backend` field only reflected BLS — making observability
        # graphs look CPU-bound while injection-recovery (and potentially CNN /
        # ExoMiner) were actually using the GPU.
        bls_backend_name = _bls_backend() if use_gpu else "CPU"
        timings["bls_backend"] = bls_backend_name
        try:
            from injection import get_backend_name as _inj_backend_name
            timings["injection_backend"] = _inj_backend_name() or "CPU"
        except Exception:
            timings["injection_backend"] = "unknown"
        # Legacy field kept for backward compat with older server
        # `/api/telemetry` summary code that hasn't been redeployed yet.
        # Prefer `bls_backend` / `injection_backend` going forward.
        timings["gpu_backend"] = bls_backend_name
        timings["n_candidates"] = len(candidates)
        self.recent_timings.append({
            "tic": tic_id,
            "total_s": timings.get("total", 0),
            "phases": {k: timings.get(k, 0) for k in
                       ["load", "bls", "vetting", "cnn", "triceratops", "injection"]},
        })
        self.current_phase = None
        self.phase_progress = None
        self._star_start_time = None

        # Print timing breakdown
        self.logger.info(f"TIC {tic_id}: {len(candidates)} candidates, {elapsed:.1f}s "
              f"[load={timings['load']}s bls={timings['bls']}s vet={timings['vetting']}s "
              f"cnn={timings['cnn']}s tri={timings.get('triceratops', 0)}s inj={timings['injection']}s]")
        for i, c in enumerate(candidates):
            v = vetting_results[i] if i < len(vetting_results) else {}
            cnn = cnn_scores[i] if i < len(cnn_scores) else None
            icon = "PASS" if v.get("passed") else "FAIL"
            cnn_str = f", CNN={cnn:.2f}" if cnn is not None else ""
            self.logger.info(f"  [{icon}] #{c['candidate_num']}: P={c['period_days']:.4f}d, SNR={c['snr']:.1f}{cnn_str}")

        if self.ui and candidates:
            c0 = candidates[0]
            fpp0 = fpp_results[0] if fpp_results[0] else {}
            self.ui.star_result(
                len(candidates),
                c0.get("period_days", 0),
                c0.get("depth_ppm", 0),
                fpp0.get("fpp"),
                fpp0.get("nfpp"),
                fpp0.get("verdict", "?"),
                elapsed)

        # ExoMiner scoring — three routes:
        #   data_source == "ffi": ExoMinerFFI ONNX (bypasses DV XML / SPOC pipeline)
        #   exominer_mode == "native": SPOC ExoMiner++ via worker.exominer_native.hook
        #   default: legacy-tfrecord (server-side Docker preprocess + venv_tf)
        worker_cfg = self.config  # per-worker config, not the per-task config
        exominer_mode = worker_cfg.get("exominer_mode", "legacy-tfrecord")
        exominer_on = worker_cfg.get("exominer_enabled") or exominer_mode == "native"
        data_source = (
            star_meta.get("data_source", "spoc")
            if isinstance(star_meta, dict)
            else "spoc"
        )
        # Always-on Phase 3 (writes exominer_ffi_score). Outside the exominer_on
        # gate because Phase 3 is independent of the SPOC ExoMiner routes.
        try:
            from worker.phase3_runner import run_phase3_always_on
        except ImportError:
            from phase3_runner import run_phase3_always_on
        run_phase3_always_on(
            tic_id=tic_id, candidates=candidates,
            loaded_time=time_arr, loaded_flux=flux_arr,
            data_source=data_source, star_meta=star_meta,
            config=worker_cfg, logger=self.logger,
        )

        # Cross-source photometric confirmation: confirm SPOC candidates against
        # the FFI light curve at their fixed ephemeris (independent evidence vs
        # model-score agreement). Feature only — never a gate. Config-gated.
        if worker_cfg.get("cross_source_enabled", False):
            try:
                from worker.cross_source import run_cross_source_check
            except ImportError:
                from cross_source import run_cross_source_check
            run_cross_source_check(
                tic_id=tic_id, candidates=candidates,
                data_source=data_source, config=worker_cfg,
                logger=self.logger,
            )

        if exominer_on and candidates and tic_id:
            try:
                if data_source == "ffi":
                    # FFI-only: SPOC ExoMiner has no DV XML — skip.
                    # Phase 3 above already wrote exominer_ffi_score.
                    pass
                elif exominer_mode == "native":
                    # Import via `exominer_native` (relative to worker/ on sys.path)
                    # — when the worker runs from its own directory, `worker.`
                    # prefix isn't importable.
                    try:
                        from worker.exominer_native.hook import (
                            score_candidates_native as _exo_score_native,
                        )
                    except ImportError:
                        from exominer_native.hook import (
                            score_candidates_native as _exo_score_native,
                        )
                    onnx_path = worker_cfg.get(
                        "exominer_onnx_path",
                        os.path.join(os.path.dirname(__file__),
                                     "models", "exominer_plusplus_single.onnx"),
                    )
                    summary = _exo_score_native(
                        tic_id=str(tic_id),
                        candidates=candidates,
                        server_url=worker_cfg["server_url"],
                        token=worker_cfg.get("worker_token")
                              or worker_cfg.get("token", "exonit-default-token"),
                        onnx_path=onnx_path,
                        config=worker_cfg,
                    )
                    if summary:
                        self.logger.info(
                            "  ExoMiner (native) TIC %s: %d/%d TCEs matched",
                            tic_id, summary.get("n_matched", 0),
                            summary.get("n_tces", 0),
                        )
                    else:
                        self.logger.info("  ExoMiner (native) TIC %s: no scores (skip)", tic_id)
                else:
                    from exominer_client import score_candidates as _exo_score
                    summary = _exo_score(
                        tic_id=str(tic_id),
                        candidates=candidates,
                        server_url=worker_cfg["server_url"],
                        token=worker_cfg.get("worker_token")
                              or worker_cfg.get("token", "exonit-default-token"),
                        venv_python=worker_cfg["exominer_venv_python"],
                        cli_path=worker_cfg.get(
                            "exominer_cli_path",
                            os.path.join(os.path.dirname(__file__),
                                         "exominer_infer_cli.py"),
                        ),
                        timeout=worker_cfg.get("exominer_timeout_sec", 180),
                    )
                    if summary:
                        self.logger.info(
                            "  ExoMiner TIC %s: %d/%d TCEs matched (run %s)",
                            tic_id, summary["n_matched"], summary["n_tces"],
                            summary["sector_run"],
                        )
                    else:
                        self.logger.info("  ExoMiner TIC %s: no scores (server miss/error)", tic_id)
            except Exception as e:  # noqa: BLE001 — optional feature, don't block pipeline
                self.logger.warning("ExoMiner failed TIC %s: %s", tic_id, e)

        result = self._base_result(task)
        result.update({
            "tic_id": tic_id,
            "candidates": candidates,
            "vetting_results": vetting_results,
            "cnn_scores": cnn_scores,
            "completeness": completeness,
            "processing_time_seconds": round(elapsed, 2),
            "gpu_used": self.gpu_enabled,
            "n_sectors": n_sectors,
            "n_points": len(time_arr),
            "time_span_days": float(time_arr[-1] - time_arr[0]) if len(time_arr) > 1 else 0,
            "ra": star_meta.get("ra"),
            "dec": star_meta.get("dec"),
            "tmag": star_meta.get("tmag"),
            "timings": timings,
            "lc_load_recovered": lc_load_recovered,
        })
        return result

    def _make_serializable(self, obj):
        """Convert numpy types to Python native for JSON serialization."""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._make_serializable(v) for v in obj]
        if hasattr(obj, 'item'):
            return obj.item()
        return obj

    def _start_gui_server(self):
        """Start a minimal HTTP server for the worker GUI on localhost."""
        from http.server import HTTPServer, BaseHTTPRequestHandler
        import urllib.parse

        worker = self

        class GUIHandler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path == "/api/status":
                    self._json({
                        "name": worker.name,
                        "worker_id": worker.worker_id,
                        "status": worker.status,
                        "current_tic": worker.current_tic,
                        "stars_completed": worker.stars_completed,
                        "total_processing_time": round(worker.total_processing_time, 1),
                        "avg_time_per_star": round(worker.total_processing_time / max(1, worker.stars_completed), 1),
                        "gpu_enabled": worker.gpu_enabled,
                        "server": worker.server,
                        "last_candidates": worker.last_candidates[-10:],
                    })
                elif self.path == "/":
                    self._html(_WORKER_GUI_HTML)
                else:
                    self.send_error(404)

            def _json(self, data):
                body = json.dumps(data).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(body)

            def _html(self, content):
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.end_headers()
                self.wfile.write(content.encode())

            def log_message(self, *args):
                pass  # suppress access logs

        try:
            httpd = HTTPServer(("0.0.0.0", worker.gui_port), GUIHandler)
            worker.logger.info(f"Worker GUI: http://localhost:{worker.gui_port}")
            httpd.serve_forever()
        except OSError:
            worker.logger.warning(f"Worker GUI: port {worker.gui_port} in use, skipping")

    def submit_result(self, result):
        result = self._make_serializable(result)
        try:
            resp = requests.post(f"{self.server}/api/worker/result", json=result, headers=self._headers(), timeout=30)
            return resp.json()
        except Exception as e:
            self.logger.warning(f"Error submitting result: {e}")
            return None

    def _report_fpp_failed(self, task):
        """Mark a candidate as FPP-failed so server doesn't reassign it."""
        try:
            requests.post(f"{self.server}/api/worker/fpp_result",
                json={"candidate_id": task["candidate_id"], "tic_id": task["tic_id"],
                      "fpp": -1.0, "nfpp": -1.0, "verdict": "failed",
                      "candidate": task},
                headers=self._headers(), timeout=30)
        except Exception:
            pass
        self.status = "idle"
        self.current_tic = None

    def _fetch_and_load_fpp(self):
        """Reserve the next FPP task at the server and pre-load its light
        curve. Returns a ``prepared`` dict (or ``None`` on no-task / failure).

        Extracted from the old _try_fpp_backfill so the same prep work can
        run in a background thread while the main loop is still processing
        the previous task — see _enqueue_next_fpp.
        """
        try:
            resp = requests.get(f"{self.server}/api/worker/fpp_task",
                                headers=self._headers(), timeout=10)
            data = resp.json()
            task = data.get("task")
        except Exception as e:
            msg = str(e)
            if "No JSON" not in msg and "Connection" not in msg:
                self.logger.warning(f"[FPP-prefetch] fpp_task fetch failed: {e}")
            return None
        if not task:
            return None
        tic_id = task["tic_id"]
        try:
            time_arr, flux_arr, _raw_flux_arr, n_sectors, _ = self._load_star(tic_id)
        except Exception as e:
            self.logger.warning(f"[FPP-prefetch] _load_star {tic_id} failed: {e}")
            try:
                self._report_fpp_failed(task)
            except Exception:
                pass
            return None
        if time_arr is None:
            self.logger.warning(
                f"[FPP-prefetch] TIC {tic_id}: no light curve data, releasing")
            try:
                self._report_fpp_failed(task)
            except Exception:
                pass
            return None
        return {
            "task": task,
            "time_arr": time_arr,
            "flux_arr": flux_arr,
            "n_sectors": n_sectors,
        }

    def _process_prepared_fpp(self, prepared):
        """Run TRICERATOPS on a pre-loaded LC and post the result.

        Receives the dict returned by _fetch_and_load_fpp so the main loop
        doesn't pay the cold-cache FITS download time before scoring can
        start. Mirrors the result-posting branch of the old _try_fpp_backfill.
        """
        task = prepared["task"]
        tic_id = task["tic_id"]
        time_arr = prepared["time_arr"]
        flux_arr = prepared["flux_arr"]
        n_sectors = prepared["n_sectors"]
        self.status = "backfill"
        self.current_tic = tic_id
        self.logger.info(
            f"[FPP] TIC {tic_id} #{task['candidate_num']} SNR={task['snr']:.1f}")
        t0 = time.time()
        try:
            sectors_arr = np.arange(1, max(n_sectors, task.get("n_sectors", 1)) + 1)
            result = calc_fpp(
                tic_id, task["period_days"], task["t0"],
                task["duration_days"], task["depth_ppm"],
                sectors_arr, time_arr, flux_arr,
                workdir=self.fits_cache_dir,
                server_url=self.server, token=self.token,
                abort_check=lambda: not self.running)
            elapsed = time.time() - t0
            if result:
                self.logger.info(
                    f"[FPP] TIC {tic_id}: FPP={result['fpp']:.4f} "
                    f"({result['verdict']}) {elapsed:.1f}s")
                requests.post(f"{self.server}/api/worker/fpp_result",
                    json={
                        "candidate_id": task["candidate_id"],
                        "tic_id": tic_id,
                        "fpp": result["fpp"],
                        "nfpp": result["nfpp"],
                        "prob_planet": result.get("prob_planet"),
                        "verdict": result.get("verdict"),
                        "candidate": task,
                    },
                    headers=self._headers(), timeout=30)
            else:
                self.logger.warning(
                    f"[FPP] TIC {tic_id}: TRICERATOPS failed ({elapsed:.1f}s)")
                requests.post(f"{self.server}/api/worker/fpp_result",
                    json={
                        "candidate_id": task["candidate_id"],
                        "tic_id": tic_id,
                        "fpp": -1.0,
                        "nfpp": -1.0,
                        "verdict": "failed",
                        "candidate": task,
                    },
                    headers=self._headers(), timeout=30)
        except FPPAborted:
            self.logger.info(f"[FPP] Aborted by user")
        except Exception as e:
            msg = str(e)
            transient = "No JSON" in msg or "Connection" in msg
            if not transient:
                self.logger.warning(f"[FPP] Error: {e}")
                try:
                    self._report_fpp_failed(task)
                except Exception:
                    pass
        finally:
            self.status = "idle"
            self.current_tic = None

    def _try_fpp_backfill(self):
        """Run one FPP backfill iteration with async prefetch.

        Without prefetch each task pays ~10s of cold-cache FITS download
        before TRICERATOPS can start (measured at <host> 2026-05-16:
        median 12s gap between tasks vs ~5s actual compute). The prefetch
        thread fetches+loads the *next* task while we run TRICERATOPS on
        the current one, halving wallclock per task in the steady state.
        """
        prepared = self._fpp_prefetch_slot.take()
        if prepared is None:
            return
        # Kick off prefetch for the NEXT task before we start computing.
        self._fpp_prefetch_slot.enqueue_next_bg()
        self._process_prepared_fpp(prepared)

    def _exominer_backfill_available(self):
        """Cached check: does this worker have the ExoMiner native hook and
        the ONNX model installed locally?

        Workers without the model would raise FileNotFoundError on every
        task and mark each candidate as -1 (destroying any real score in
        the column). Probe once at first call, cache the result, and skip
        polling forever if missing. Pairs with the server-side
        EXONIT_EXOMINER_WORKERS allowlist as defence in depth.
        """
        if hasattr(self, "_em_avail_cached"):
            return self._em_avail_cached
        from pathlib import Path as _P
        ok = False
        reason = ""
        try:
            try:
                from worker.exominer_native.hook import score_candidates_native  # noqa: F401
            except ImportError:
                from exominer_native.hook import score_candidates_native  # noqa: F401
        except Exception as e:  # pylint: disable=broad-except
            reason = f"hook import failed: {e}"
        else:
            onnx_path = self.config.get(
                "exominer_onnx_path",
                str(_P(__file__).resolve().parent
                    / "models" / "exominer_plusplus_single.onnx"))
            if _P(onnx_path).exists():
                ok = True
            else:
                reason = f"model not found at {onnx_path}"
        if not ok:
            self.logger.info(
                f"[ExoMiner] backfill disabled on this worker: {reason}")
        self._em_avail_cached = ok
        return ok

    def _fetch_and_prepare_exominer(self):
        """Reserve the next ExoMiner backfill task and pre-download its DV
        XML + TICv8 + Gaia + FITS files. Returns a ``prepared_full`` dict
        (or ``None``) — the network-heavy work is done before scoring
        starts so the prefetch thread can run it in parallel with main.
        """
        try:
            resp = requests.get(f"{self.server}/api/worker/exominer_task",
                                headers=self._headers(), timeout=10)
            data = resp.json()
            task = data.get("task")
        except Exception as e:
            msg = str(e)
            if "No JSON" not in msg and "Connection" not in msg:
                self.logger.warning(
                    f"[ExoMiner-prefetch] task fetch failed: {e}")
            return None
        if not task:
            return None
        tic_id = task["tic_id"]
        cand_id = task["candidate_id"]
        try:
            try:
                from worker.exominer_native.hook import prepare_exominer_inputs
            except ImportError:
                from exominer_native.hook import prepare_exominer_inputs
        except Exception as e:  # pylint: disable=broad-except
            self.logger.warning(f"[ExoMiner-prefetch] hook unavailable: {e}")
            self._report_exominer_skipped(cand_id, tic_id)
            return None
        cand = {
            "candidate_num": task["candidate_num"],
            "period_days": task["period_days"],
            "t0": task["t0"],
            "duration_days": task["duration_days"],
            "depth_ppm": task["depth_ppm"],
        }
        try:
            prepared_inputs = prepare_exominer_inputs(
                str(tic_id), [cand],
                server_url=self.server, token=self.token)
        except Exception as e:  # pylint: disable=broad-except
            self.logger.warning(f"[ExoMiner-prefetch] prepare {tic_id}: {e}")
            self._report_exominer_skipped(cand_id, tic_id)
            return None
        if prepared_inputs is None:
            # No DV / no LC — legitimate skip; mark -1 so we don't retry.
            self._report_exominer_skipped(cand_id, tic_id)
            return None
        return {
            "task": task,
            "cand": cand,
            "prepared_inputs": prepared_inputs,
        }

    def _on_exominer_prefetch_skip(self, prepared_full):
        """Release a prefetched-but-unused ExoMiner reservation.

        Called by ``self._em_prefetch_slot`` (a ``_PrefetchSlot``) when a
        background-prefetched item loses the race for the queue slot to
        ``take()``. Unlike the FPP slot's on_skip, this one also has a
        local tmp dir (downloaded DV XML + TICv8 + Gaia + FITS) that must
        be cleaned up before reporting skipped — same asymmetry that
        existed pre-extraction between ``_enqueue_next_fpp`` and
        ``_enqueue_next_exominer``, preserved here as a call-site
        difference rather than folded into the generic slot.
        """
        try:
            import shutil as _sh
            _sh.rmtree(
                prepared_full["prepared_inputs"]["tmp_root"],
                ignore_errors=True)
        except Exception:
            pass
        self._report_exominer_skipped(
            prepared_full["task"]["candidate_id"],
            prepared_full["task"]["tic_id"])

    def _process_prepared_exominer(self, prepared_full):
        """Run preprocess + ONNX + match on already-downloaded inputs and
        post the result. Mirrors the result branch of the old
        ``_try_exominer_backfill``.
        """
        task = prepared_full["task"]
        cand = prepared_full["cand"]
        prepared_inputs = prepared_full["prepared_inputs"]
        tic_id = task["tic_id"]
        cand_id = task["candidate_id"]
        self.status = "backfill"
        self.current_tic = tic_id
        self.logger.info(
            f"[ExoMiner] TIC {tic_id} #{task['candidate_num']} "
            f"SNR={task['snr']:.1f}")
        t0 = time.time()
        try:
            try:
                from worker.exominer_native.hook import score_prepared_exominer
            except ImportError:
                from exominer_native.hook import score_prepared_exominer
            onnx_path = self.config.get(
                "exominer_onnx_path",
                str(Path(__file__).resolve().parent
                    / "models" / "exominer_plusplus_single.onnx"))
            try:
                score_prepared_exominer(
                    [cand], prepared_inputs, onnx_path=str(onnx_path))
            except Exception as e:  # pylint: disable=broad-except
                self.logger.warning(f"[ExoMiner] scoring failed: {e}")
                self._report_exominer_skipped(cand_id, tic_id)
                return
            elapsed = time.time() - t0
            score = cand.get("exominer_score")
            if score is None:
                self.logger.info(
                    f"[ExoMiner] TIC {tic_id}: no TCE matched ({elapsed:.1f}s)")
                self._report_exominer_skipped(cand_id, tic_id)
                return
            self.logger.info(
                f"[ExoMiner] TIC {tic_id}: score={score:.3f} ({elapsed:.1f}s)")
            try:
                requests.post(
                    f"{self.server}/api/worker/exominer_result",
                    json={
                        "candidate_id": cand_id,
                        "tic_id": tic_id,
                        "score": score,
                        "tce_period": cand.get("exominer_tce_period"),
                        "sector_run": cand.get("exominer_sector_run"),
                    },
                    headers=self._headers(), timeout=30)
            except Exception as e:  # pylint: disable=broad-except
                self.logger.warning(f"[ExoMiner] result post failed: {e}")
        except Exception as e:  # pylint: disable=broad-except
            msg = str(e)
            if "No JSON" not in msg and "Connection" not in msg:
                self.logger.warning(f"[ExoMiner] Error: {e}")
        finally:
            self.status = "idle"
            self.current_tic = None

    def _try_exominer_backfill(self):
        """Run one ExoMiner backfill iteration with async prefetch.

        Same pattern as ``_try_fpp_backfill``: drain the queue first (a
        background thread has already fetched the task and downloaded DV
        XML + FITS + TICv8 + Gaia), and only fall back to a synchronous
        prepare on a miss. Then spawn the prefetch for the *next* task
        before we start preprocess+ONNX so the cold-cache download
        overlaps with compute on this one.
        """
        if not self._exominer_backfill_available():
            return
        prepared_full = self._em_prefetch_slot.take()
        if prepared_full is None:
            return
        self._em_prefetch_slot.enqueue_next_bg()
        self._process_prepared_exominer(prepared_full)

    def _report_exominer_skipped(self, candidate_id, tic_id):
        """Tell the server we can't score this candidate so the slot frees.

        Sends score=-1 so the DB row marks it tried-but-failed; the next
        time worker_exominer_task lists candidates it'll skip those with
        a non-NULL exominer_score.
        """
        try:
            requests.post(
                f"{self.server}/api/worker/exominer_result",
                json={"candidate_id": candidate_id, "tic_id": tic_id,
                      "score": -1.0},
                headers=self._headers(), timeout=10)
        except Exception:
            pass

    def run(self):
        if self.ui:
            self.ui._on_stop = lambda: setattr(self, 'running', False)
            self.ui.show_banner()
            self.ui.start_keypress_listener()
        else:
            print(f"\nExoNit Worker v2 — {self.name}")
            print(f"Server: {self.server}")
            print(f"GPU: {'enabled' if self.gpu_enabled else 'disabled (CPU only)'}")
            fpp_info = _fpp_gpu_backend or "CPU (numba)"
            print(f"FPP backend: {fpp_info}")
            print(f"FITS cache: {self.fits_cache_dir}")
            print("=" * 50)
        # Sync worker code from the server BEFORE registering or pulling any
        # task, so the worker always runs the same commit the server serves.
        # This removes the need to re-bake AMIs for code changes and closes the
        # server<->worker version-skew gap that caused the 2026-05-20 big-run
        # regression. The periodic heartbeat update check only fires while the
        # worker is idle, so during a busy run it never updates — this startup
        # call covers that. check_for_updates() os.execv-restarts on change, so
        # after one sync the process re-enters run() already up to date.
        #
        # On AWS concurrent deployments the bootstrap runs a single
        # exonit-codesync.service oneshot (as ubuntu, with write perms) before
        # launching N worker units. Each worker unit sets
        # EXONIT_SKIP_STARTUP_CODESYNC=1 so they skip this block and avoid the
        # PermissionError + race that caused all tasks to fail as load_failed.
        if os.environ.get("EXONIT_SKIP_STARTUP_CODESYNC") == "1":
            self.logger.info(
                "Startup code-sync: skipped (EXONIT_SKIP_STARTUP_CODESYNC=1; "
                "oneshot codesync already ran)")
        elif self.config.get("startup_code_sync", True):
            try:
                self.logger.info("Startup code-sync: checking server for updates...")
                self.check_for_updates()
            except Exception as e:
                self.logger.warning(f"Startup code-sync skipped ({e}); "
                                    "continuing with local code")
        self.register()
        hb_thread = threading.Thread(target=self.heartbeat_loop, daemon=True)
        hb_thread.start()
        # gui_enabled defaults to True for retrocompat (single-worker and
        # primary-instance workers keep the GUI).  Secondary instances in a
        # multi-worker setup have gui_enabled=False written to their derived
        # config by agent.py to avoid OSError on port-already-in-use.
        if self.config.get("gui_enabled", True):
            gui_thread = threading.Thread(target=self._start_gui_server, daemon=True)
            gui_thread.start()
        if self.ui:
            self.ui.waiting()
        else:
            print("  Waiting for tasks...\n")
        while self.running:
            if self.ui and self.ui.stop_requested:
                self.running = False
                break
            if self.paused:
                if self.ui:
                    self.ui._stop_event.wait(self.poll_interval)
                else:
                    time.sleep(self.poll_interval)
                continue
            task = self.get_task() if self._pool_enabled("primary") else None
            if task is None:
                # No regular tasks — fall through backfill queues. FPP first
                # because it's cheaper (~5s) than ExoMiner (~30-60s); then
                # ExoMiner backfill for candidates that never got a score.
                if self._pool_enabled("fpp_backfill") and _triceratops_available:
                    self._try_fpp_backfill()
                if self._pool_enabled("exominer_backfill"):
                    self._try_exominer_backfill()
                if self.ui:
                    self.ui._stop_event.wait(self.poll_interval)
                else:
                    time.sleep(self.poll_interval)
                continue
            try:
                result = self.process_task(task)
                self.submit_result(result)
                # Update local stats
                self.stars_completed += 1
                self.total_processing_time += result.get("processing_time_seconds", 0)
                if self.ui:
                    avg = self.total_processing_time / max(1, self.stars_completed)
                    self.ui.update_footer(self.stars_completed, self.total_processing_time, avg)
                if result.get("candidates"):
                    for c in result["candidates"]:
                        self.last_candidates.append({
                            "tic_id": result["tic_id"],
                            "period": c.get("period_days"),
                            "snr": c.get("snr"),
                            "depth_ppm": c.get("depth_ppm"),
                        })
                # Successful task: disarm the CUDA context-poison guard so a
                # future genuine poison event gets its full 3-restart budget.
                self._reset_cuda_streak()
            except Exception as e:
                self.logger.warning(f"Error processing task {task.get('tic_id', '?')}: {e}")
                # Report failure so server can reassign
                try:
                    failed_result = self._base_result(task)
                    failed_result["status"] = "failed"
                    self.submit_result(failed_result)
                except Exception:
                    pass
                # CUDA context-poison check: must run AFTER submit_result so
                # the server can immediately reassign the failed task.
                self._note_cuda_failure(e)
            self.status = "idle"
            self.current_tic = None
            # Free memory between stars
            gc.collect()
            # Limit FITS cache to 500MB
            try:
                import glob as _glob
                cache_dir = self.fits_cache_dir
                fits_files = sorted(_glob.glob(os.path.join(cache_dir, "*.fits")),
                                    key=os.path.getmtime)
                total = sum(os.path.getsize(f) for f in fits_files)
                while total > 500_000_000 and fits_files:
                    old = fits_files.pop(0)
                    total -= os.path.getsize(old)
                    os.remove(old)
            except Exception:
                pass
            # Deferred re-registration (after GPU operations are done)
            if getattr(self, '_needs_reregister', False):
                self.logger.info("Re-registering now (deferred)...")
                try:
                    self.register()
                except Exception:
                    pass
                self._needs_reregister = False
        if self.ui:
            self.ui.cleanup()
        print("\n  Worker stopped.")


if __name__ == "__main__":
    import signal
    parser = argparse.ArgumentParser(description="ExoNit Worker v2")
    parser.add_argument("--config", default="config.json")
    parser.add_argument(
        "--worker-name",
        help="Override worker_name from config.json. Used by farmctl/agent so "
             "that rsync/git pull of config.json never clobbers per-host identity.",
    )
    parser.add_argument(
        "--server-url",
        help="Override server_url from config.json. The committed default is "
             "http://localhost:8400 (correct for worker-default); remote "
             "hosts must use the server's network address. Same rationale as "
             "--worker-name: host-specific identity stays out of config.json.",
    )
    args = parser.parse_args()
    worker = ExoNitWorker(
        args.config,
        worker_name_override=args.worker_name,
        server_url_override=args.server_url,
    )

    _shutdown_count = 0

    def _shutdown(sig=None, frame=None):
        global _shutdown_count
        _shutdown_count += 1
        if _shutdown_count >= 2:
            # Second Ctrl+C: force exit immediately (bypasses numba/GIL blocks)
            print("\n  Force exit!")
            os._exit(1)
        worker.running = False
        if worker.ui:
            worker.ui.request_stop()
        else:
            print("\n  Stopping... (press Ctrl+C again to force quit)")

    signal.signal(signal.SIGINT, _shutdown)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _shutdown)
    worker.run()
