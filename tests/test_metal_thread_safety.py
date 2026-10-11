"""Regression tests for worker/metal_serial.py — the MLX/Metal thread-safety fix.

Incident (see worker/metal_serial.py's docstring for the full account): MLX
is not thread-safe. FPP Monte Carlo (``fpp_metal_kernel`` / legacy
``fpp_metal``) runs on worker.py's main thread while injection-recovery
(``bls_metal_batch`` / legacy ``bls_metal``) runs on a background daemon
thread concurrently. Once both submitted Metal work through MLX (the fused
kernel, deployed 2026-07-11), that race corrupted MLX's shared command-
buffer state and aborted the whole worker process, hours into production:

    -[_MTLCommandBuffer addCompletedHandler:]:1011: failed assertion
      `Completed handler provided after commit call'
    failed assertion _status < MTLCommandBufferStatusCommitted at
      -[IOGPUMetalCommandBuffer setCurrentCommandEncoder:]

The pre-deploy bench that "passed" ran only 4 rounds (~8s of GPU work) --
exactly the kind of undersized concurrency bench CLAUDE.md warns against.
This suite:

  1. Pins the "single shared lock, reentrant" contract that makes
     ``metal_guard()`` actually prevent the cross-module race. Platform-
     independent: ``fpp_metal_kernel``/``bls_metal_batch``/``bls_metal``/
     ``fpp_metal`` all import cleanly without real MLX installed (the
     ``import mlx.core`` is wrapped in ``try/except ImportError`` in every
     one of them), so this runs on Linux CI too.
  2. Runs FPP-Metal and injection-Metal concurrently from two threads for a
     wall-clock duration and asserts every call's output matches a serial
     reference computed with identical inputs. Skipped cleanly when the
     real fused Metal kernels are unavailable (this host, CI); it is meant
     to run on the Macs.

     READ THIS BEFORE TRUSTING (OR "FIXING") THIS TEST -- 2026-07-12,
     corrected same-day after an over-claim:

     The race this fix addresses is a HARDWARE-RACE-CONDITION, and it is
     NON-DETERMINISTIC. Measured on real M4 hardware, running the UNGUARDED
     (pre-fix) code concurrently for multiple minutes:
       - production-sized workloads (N=20000 FPP draws/T=600; 90 fluxes x
         5000 periods x 150 bins injection), ~30 submissions/s: 1 crash in
         5 runs. (An earlier report of "crashed within seconds, every time"
         was a single lucky observation on the first run -- 3 further runs
         at the same size survived the full 60s each. Production-sized
         kernels are actually WORSE at surfacing the race: each kernel
         occupies the GPU long enough that the other thread rarely
         interleaves at a submission boundary.)
       - small kernels hammered in a tight loop, ~456 submissions/s, 40s:
         1 crash in 3 runs.
       - small kernels, ~1733 submissions/s, 180s (312k submissions!):
         1 crash in 2 runs -- it still SURVIVED one full 180s run.
     Two distinct assertions were reproduced this way, both matching the
     production incident:
       ``-[_MTLCommandBuffer addCompletedHandler:]:1011: failed assertion
         'Completed handler provided after commit call'``
       ``-[AGXG16GFamilyCommandBuffer
         tryCoalescingPreviousComputeCommandEncoderWithConfig:...]:1094:
         failed assertion 'A command encoder is already encoding to this
         command buffer'``

     CONSEQUENCE: this test is a SMOKE TEST, not a deterministic
     falsification gate. A green run (no crash, no mismatch) does NOT prove
     ``metal_guard()`` works -- with ~30-50% crash odds per multi-minute
     unguarded run, a short guarded run passing is unsurprising whether or
     not the guard does anything at all. Do not treat a passing run of this
     test as validation. The guard's actual justification is STRUCTURAL
     (serializing submission removes the necessary condition for the race
     to occur at all -- there is no interleaving left for the assertion to
     catch), not statistical, and the authoritative validation is a
     PRODUCTION CANARY: one Mac worker host with the kernel re-enabled and
     watched for hours (unguarded workers previously died in production
     within 8-47 minutes; see docs/dossiers/mac-gpu.md, "2026-07-12 --
     INCIDENT: MLX is not thread-safe", "Time to crash varied per host"),
     not a
     test run. What this test IS good for: (a) the serial-reference parity
     assertion below is sound and deterministic (verifies the guard doesn't
     silently corrupt results when it does serialize), and (b) it is tuned
     to the highest-submission-rate configuration measured to have the best
     (still not reliable) chance of catching an outright regression (e.g.
     someone deleting a ``with metal_guard():`` line) during manual/CI runs
     on real hardware. See
     ``scripts/archive/mac_gpu_bench/repro_metal_thread_race.py``
     (``--rate-mode highrate``) for the standalone driver used to gather
     the frequencies above, and to run a much longer soak than a pytest
     test is appropriate for.
"""
from __future__ import annotations

import math
import os
import threading
import time as _time

import numpy as np
import pytest

import metal_serial
import fpp_metal_kernel
import bls_metal_batch
import bls_metal
import fpp_metal


# ================================================================== #
# 1. Shared-lock contract (platform-independent -- no MLX required)
# ================================================================== #

class TestMetalGuardIsSingleSharedReentrantLock:
    def test_metal_guard_returns_the_same_object_every_call(self):
        assert metal_serial.metal_guard() is metal_serial.metal_guard()

    def test_fpp_metal_kernel_and_bls_metal_batch_share_the_same_guard(self):
        """The whole point of metal_guard(): the FPP fused-kernel module and
        the injection fused-kernel module must import and call the SAME
        function (hence serialize against the SAME lock) -- two independent
        locks would not prevent the cross-thread race at all."""
        assert fpp_metal_kernel.metal_guard is metal_serial.metal_guard
        assert bls_metal_batch.metal_guard is metal_serial.metal_guard
        assert (fpp_metal_kernel.metal_guard()
                is bls_metal_batch.metal_guard()
                is metal_serial.metal_guard())

    def test_legacy_fallback_modules_share_the_same_guard_too(self):
        """bls_metal / fpp_metal are still-reachable fallback paths (see
        worker/injection.py's Metal/MLX-legacy branch and
        worker/fpp_native.py's EXONIT_FPP_FORCE_METAL / no-numba branch) --
        they must serialize against the same lock as the primary fused-
        kernel modules, not an independent one."""
        assert bls_metal.metal_guard is metal_serial.metal_guard
        assert fpp_metal.metal_guard is metal_serial.metal_guard
        assert (bls_metal.metal_guard() is fpp_metal.metal_guard()
                is metal_serial.metal_guard())

    def test_guard_is_reentrant(self):
        """A single thread must be able to re-enter without deadlocking
        (e.g. a guarded region calling another guarded helper -- ``_build()``
        is called from inside the ``with metal_guard():`` block in
        batch_transit_lnL, for instance)."""
        guard = metal_serial.metal_guard()
        entered = []
        with guard:
            entered.append(1)
            with guard:  # re-entry, same thread
                entered.append(2)
            entered.append(3)
        assert entered == [1, 2, 3]

    def test_guard_actually_blocks_a_different_thread(self):
        """Sanity check that this is a REAL lock, not an accidental no-op
        (e.g. contextlib.nullcontext) -- a second thread must block until
        the first thread releases it."""
        guard = metal_serial.metal_guard()
        order = []
        release_first = threading.Event()
        second_acquired = threading.Event()

        def _first():
            with guard:
                order.append("first-acquired")
                release_first.wait(timeout=5)
            order.append("first-released")

        def _second():
            with guard:
                second_acquired.set()
                order.append("second-acquired")

        t1 = threading.Thread(target=_first)
        t2 = threading.Thread(target=_second)
        t1.start()
        _time.sleep(0.1)  # let t1 acquire first
        t2.start()
        _time.sleep(0.1)
        assert not second_acquired.is_set(), (
            "second thread acquired metal_guard() while the first thread "
            "still held it -- it is not actually serializing")
        release_first.set()
        t1.join(timeout=5)
        t2.join(timeout=5)
        assert not t1.is_alive() and not t2.is_alive()
        assert order == ["first-acquired", "first-released", "second-acquired"]


# ================================================================== #
# 2. Concurrent FPP-Metal || injection-Metal smoke test (NOT a
#    deterministic falsification gate -- see module docstring)
# ================================================================== #

_MLX_REAL = False
try:
    import mlx.core as _mx_probe
    _MLX_REAL = hasattr(_mx_probe, "fast") and hasattr(_mx_probe.fast, "metal_kernel")
except ImportError:
    pass

# Bypass the production kill-switch / SoC-allowlist policy on purpose: this
# test validates raw hardware thread-safety, not worker.py's routing policy
# (EXONIT_INJECTION_METAL / _SUPPORTED_SOC_PREFIXES gate WHICH backend gets
# used in production, they are orthogonal to whether concurrent submission
# through MLX is safe on this piece of hardware).
_FPP_METAL_REAL = False
_INJ_METAL_REAL = False
if _MLX_REAL:
    try:
        _FPP_METAL_REAL = fpp_metal_kernel.is_available()
    except Exception:
        _FPP_METAL_REAL = False
    try:
        if bls_metal_batch.MLX_AVAILABLE:
            bls_metal_batch.bls_gpu_batch_v2(
                np.linspace(0.0, 1.0, 50), [np.ones(50)],
                n_periods=20, n_bins=20, n_dur_steps=3)
            _INJ_METAL_REAL = True
    except Exception:
        _INJ_METAL_REAL = False

_metal_stress_only = pytest.mark.skipif(
    not (_FPP_METAL_REAL and _INJ_METAL_REAL),
    reason="real MLX/Metal fused kernels unavailable on this host")


class TestConcurrentFppAndInjectionMetalDoNotRace:
    """Concurrent FPP-Metal || injection-Metal smoke test -- SEE THE MODULE
    DOCSTRING FIRST. The race is probabilistic (measured ~30-50% crash odds
    per multi-minute unguarded run on real M4 hardware); this test therefore
    CANNOT reliably detect a regression and a green run does NOT prove
    ``metal_guard()`` works. Its guarded serial-reference parity assertion
    IS a sound, deterministic check that the guard doesn't corrupt results.

    Tuned for the HIGHEST submission rate measured to matter for surfacing
    the race -- small kernels hammered in a tight loop, not production
    sizes. (Production-sized kernels submit far less often per second
    because each one occupies the GPU longer, which empirically made the
    race LESS likely to be caught, not more -- the opposite of the original
    assumption behind this test.) See
    ``scripts/archive/mac_gpu_bench/repro_metal_thread_race.py --rate-mode
    highrate`` for a longer, standalone soak using the same configuration.
    """

    DEFAULT_SECONDS = 30.0

    @_metal_stress_only
    def test_concurrent_stress_no_crash_and_matches_serial_reference(self):
        seconds = float(os.environ.get("EXONIT_METAL_STRESS_SECONDS",
                                       self.DEFAULT_SECONDS))
        rng = np.random.default_rng(3)

        # --- FPP-side SMALL inputs, tuned for submission RATE (not size) --
        # empirically the variable that matters for surfacing the race (see
        # module docstring): small kernels hammered in a tight loop reach
        # 400-1700+ submissions/s, vs ~30/s for production-sized kernels.
        T = 60
        N = 256
        fpp_kwargs = dict(
            time=np.sort(rng.uniform(-0.25, 0.25, T)).astype(np.float64),
            flux=(1.0 + rng.normal(0, 0.001, T)).astype(np.float64),
            sigma=0.001,
            ks=rng.uniform(0.02, 0.15, N), P_orbs=np.full(N, 5.0),
            incs_rad=rng.uniform(1.45, 1.5708, N),
            a_Rs_arr=rng.uniform(8, 20, N),
            eccs=rng.uniform(0, 0.1, N),
            ws_rad=rng.uniform(0, 2 * math.pi, N),
            u1=0.4, u2=0.3,
            companion_fluxratio=np.zeros(N), companion_is_host=False)

        def _fpp_call():
            return fpp_metal_kernel.batch_transit_lnL(**fpp_kwargs)

        # --- injection-side SMALL inputs, same rationale ---
        n_points = 200
        time_inj = np.linspace(0.0, 26.0, n_points) + 2458000.0
        base = (1.0 + 3e-4 * np.sin(2 * np.pi * time_inj / 8.7)
                + 1e-3 * rng.normal(size=n_points))
        flux_inj = [base + 1e-4 * rng.normal(size=n_points) for _ in range(4)]
        inj_kwargs = dict(period_min=2.5, period_max=10.0, n_periods=120,
                          n_bins=40, n_dur_steps=5)

        def _inj_call():
            return bls_metal_batch.bls_gpu_batch_v2(
                time_inj, flux_inj, **inj_kwargs)

        # Serial references, computed BEFORE any concurrency starts. Both
        # functions are deterministic (same inputs -> same outputs)
        # regardless of thread interleaving, PROVIDED submissions are
        # correctly serialized -- any divergence below means metal_guard()
        # failed to prevent cross-thread corruption. This half of the test
        # IS sound and deterministic, unlike the crash-detection half (see
        # module/class docstrings). Also warms up the kernel-compile cache
        # for both, so the timed loop below measures steady-state rate.
        ref_fpp = _fpp_call()
        ref_inj = _inj_call()

        stop = threading.Event()
        errors = []
        mismatches = []
        counts = {"fpp": 0, "inj": 0}

        def _fpp_worker():
            try:
                while not stop.is_set():
                    r = _fpp_call()
                    counts["fpp"] += 1
                    if not np.array_equal(r, ref_fpp):
                        mismatches.append(("fpp", counts["fpp"]))
            except Exception as e:  # pragma: no cover - only on a real regression
                errors.append(("fpp", repr(e)))
                stop.set()

        def _inj_worker():
            try:
                while not stop.is_set():
                    r = _inj_call()
                    counts["inj"] += 1
                    for fi in range(len(r)):
                        if (r[fi]["period"] != ref_inj[fi]["period"]
                                or r[fi]["power"] != ref_inj[fi]["power"]
                                or r[fi]["depth"] != ref_inj[fi]["depth"]):
                            mismatches.append(("inj", counts["inj"], fi))
            except Exception as e:  # pragma: no cover - only on a real regression
                errors.append(("inj", repr(e)))
                stop.set()

        # Background daemon thread for injection, "main" thread for FPP --
        # mirrors worker.py's actual architecture (~line 2550).
        t_inj = threading.Thread(target=_inj_worker, daemon=True)
        t_fpp = threading.Thread(target=_fpp_worker, daemon=True)
        t_inj.start()
        t_fpp.start()

        t0 = _time.monotonic()
        while _time.monotonic() - t0 < seconds and not stop.is_set():
            _time.sleep(0.5)
        stop.set()
        t_fpp.join(timeout=60)
        t_inj.join(timeout=60)

        elapsed = _time.monotonic() - t0
        assert not t_fpp.is_alive() and not t_inj.is_alive(), (
            "concurrent Metal stress threads did not finish -- possible "
            "deadlock in metal_guard()")
        assert errors == [], (
            f"exception(s) during concurrent Metal stress after "
            f"{elapsed:.0f}s (fpp_calls={counts['fpp']} "
            f"inj_cells={counts['inj']}): {errors} -- NOTE: passing this "
            "test does not prove the guard works (the race is "
            "probabilistic, see module docstring); but this failing does "
            "indicate a real problem")
        assert mismatches == [], (
            f"result(s) diverged from the serial reference under "
            f"concurrency -- metal_guard() failed to prevent corruption: "
            f"{mismatches[:5]}{'...' if len(mismatches) > 5 else ''}")
        # If neither thread completed even one call, the workload/duration
        # combination on this host isn't reaching a meaningful submission
        # rate at all -- treat as a hard failure, not a pass, so a
        # misconfigured/overloaded host can't silently look "clean".
        assert counts["fpp"] > 0 and counts["inj"] > 0, (
            f"no calls completed on one or both threads in {elapsed:.0f}s "
            f"(fpp_calls={counts['fpp']} inj_cells={counts['inj']}) -- "
            "raise EXONIT_METAL_STRESS_SECONDS for this host")
