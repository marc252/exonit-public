"""Golden/behaviour tests for fpp_native._dispatch_lnL (D7 extraction).

Context (2026-07-02 cleanup review, item D7): fpp_native.py duplicated the
same GPU-try/CPU-fallback try/except/``numba_parallel_guard()`` shape 3x
across ``_lnZ_TTP`` (transit) and ``_lnZ_TEB`` (EB primary + EB twin). This
suite pins the pre-extraction behaviour of all 3 call sites (transcribed
verbatim below as ``_old_*_dispatch``, matching worker/fpp_native.py at the
worktree base commit 7a635313) and asserts the post-extraction
``fpp_native._dispatch_lnL`` helper (used by the refactored ``_lnZ_TTP`` /
``_lnZ_TEB``) reproduces it EXACTLY:

  (a) CPU-only path (no GPU backend detected)      -> bit-identical output
  (b) GPU-success path (fake gpu_fn, no exception)  -> bit-identical output
  (c) GPU-failure path (fake gpu_fn raises)         -> bit-identical output
      AND identical logging.getLogger("exonit").debug(...) message text
      AND identical numba_parallel_guard() call count (CPU fallback only)

No real GPU is available on this host (<host>, AMD Vega iGPU — FPP GPU
backend detection returns None, see fpp_native._init_fpp_gpu / OpenCL probe
log). The GPU-success and GPU-failure paths are therefore exercised with
fake ``gpu_fn`` callables rather than a real CUDA/Metal kernel. This is a
deliberate, low-risk simplification: ``_dispatch_lnL``'s GPU branch is a
single passthrough line (``return gpu_fn(*args)``) — trivial enough for
review-by-reading — so a fake callable of the right shape covers the
control-flow contract (try/except/guard/logging) exhaustively; only the
*numerical identity of a real CUDA/Metal kernel's own output* is untested
here (unchanged by this refactor — same kernel functions, same call sites).
See progress/impl-c3-fppdispatch-2026-07-02.md for the exact command to run
this suite on worker-<host> (real CUDA) as a supplementary smoke test.
"""
from __future__ import annotations

import logging

import numpy as np
import pytest

import fpp_native
from numba_serial import numba_parallel_guard  # noqa: F401  (imported for parity with fpp_native's import)


# ================================================================== #
# OLD code — verbatim pre-extraction dispatch blocks
# (worker/fpp_native.py:722-855 at worktree base commit 7a635313)
# ================================================================== #

def _old_transit_dispatch(gpu_fn, cpu_fn, N, mask, time, flux, sigma, ks,
                           P_arr, incs_rad, a_Rs, eccs, ws_rad, u1, u2,
                           comp_fr, companion_is_host):
    """Verbatim transcription of the old _lnZ_TTP dispatch block."""
    lnL = np.full(N, -np.inf)
    if gpu_fn is not None:
        try:
            lnL[mask] = gpu_fn(
                time, flux, sigma,
                ks[mask], P_arr[mask], incs_rad[mask], a_Rs[mask],
                eccs[mask], ws_rad[mask], u1, u2,
                comp_fr, companion_is_host)
        except Exception as e:
            logging.getLogger("exonit").debug(f"GPU transit failed ({e}), falling back to CPU")
            with fpp_native.numba_parallel_guard():
                lnL[mask] = cpu_fn(
                    time, flux, sigma,
                    ks[mask], P_arr[mask], incs_rad[mask], a_Rs[mask],
                    eccs[mask], ws_rad[mask], u1, u2,
                    comp_fr, companion_is_host)
    else:
        with fpp_native.numba_parallel_guard():
            lnL[mask] = cpu_fn(
                time, flux, sigma,
                ks[mask], P_arr[mask], incs_rad[mask], a_Rs[mask],
                eccs[mask], ws_rad[mask], u1, u2,
                comp_fr, companion_is_host)
    return lnL


def _old_eb_dispatch(gpu_fn, cpu_fn, N, mask_eb, time, flux, sigma, ks,
                      fluxratios_eb, P_arr, incs_rad, a_Rs, eccs, ws_rad,
                      u1, u2, comp_arr, companion_is_host):
    """Verbatim transcription of the old _lnZ_TEB primary-EB dispatch block."""
    lnL_eb = np.full(N, -np.inf)
    if np.any(mask_eb):
        n_eb = np.sum(mask_eb)  # noqa: F841 (dead in the original too — preserved)
        if gpu_fn is not None:
            try:
                lnL_eb[mask_eb] = gpu_fn(
                    time, flux, sigma,
                    ks[mask_eb], fluxratios_eb[mask_eb], P_arr[mask_eb],
                    incs_rad[mask_eb], a_Rs[mask_eb], eccs[mask_eb], ws_rad[mask_eb],
                    u1, u2, comp_arr[mask_eb], companion_is_host)
            except Exception as e:
                logging.getLogger("exonit").debug(f"GPU EB failed ({e}), falling back to CPU")
                with fpp_native.numba_parallel_guard():
                    lnL_eb[mask_eb] = cpu_fn(
                        time, flux, sigma,
                        ks[mask_eb], fluxratios_eb[mask_eb], P_arr[mask_eb],
                        incs_rad[mask_eb], a_Rs[mask_eb], eccs[mask_eb], ws_rad[mask_eb],
                        u1, u2, comp_arr[mask_eb], companion_is_host)
        else:
            with fpp_native.numba_parallel_guard():
                lnL_eb[mask_eb] = cpu_fn(
                    time, flux, sigma,
                    ks[mask_eb], fluxratios_eb[mask_eb], P_arr[mask_eb],
                    incs_rad[mask_eb], a_Rs[mask_eb], eccs[mask_eb], ws_rad[mask_eb],
                    u1, u2, comp_arr[mask_eb], companion_is_host)
    return lnL_eb


def _old_twin_eb_dispatch(gpu_fn, cpu_fn, N, mask_twin, time, flux, sigma, ks,
                           fluxratios_eb, P_arr, incs_rad, a_twin_Rs, eccs,
                           ws_rad, u1, u2, comp_arr, companion_is_host):
    """Verbatim transcription of the old _lnZ_TEB twin-EB dispatch block."""
    lnL_twin = np.full(N, -np.inf)
    if np.any(mask_twin):
        if gpu_fn is not None:
            try:
                lnL_twin[mask_twin] = gpu_fn(
                    time, flux, sigma,
                    ks[mask_twin], fluxratios_eb[mask_twin], 2 * P_arr[mask_twin],
                    incs_rad[mask_twin], a_twin_Rs[mask_twin],
                    eccs[mask_twin], ws_rad[mask_twin],
                    u1, u2, comp_arr[mask_twin], companion_is_host)
            except Exception as e:
                logging.getLogger("exonit").debug(f"GPU twin EB failed ({e}), falling back to CPU")
                with fpp_native.numba_parallel_guard():
                    lnL_twin[mask_twin] = cpu_fn(
                        time, flux, sigma,
                        ks[mask_twin], fluxratios_eb[mask_twin], 2 * P_arr[mask_twin],
                        incs_rad[mask_twin], a_twin_Rs[mask_twin],
                        eccs[mask_twin], ws_rad[mask_twin],
                        u1, u2, comp_arr[mask_twin], companion_is_host)
        else:
            with fpp_native.numba_parallel_guard():
                lnL_twin[mask_twin] = cpu_fn(
                    time, flux, sigma,
                    ks[mask_twin], fluxratios_eb[mask_twin], 2 * P_arr[mask_twin],
                    incs_rad[mask_twin], a_twin_Rs[mask_twin],
                    eccs[mask_twin], ws_rad[mask_twin],
                    u1, u2, comp_arr[mask_twin], companion_is_host)
    return lnL_twin


# ================================================================== #
# NEW code — thin wrappers matching the refactored _lnZ_TTP / _lnZ_TEB
# call sites exactly (worker/fpp_native.py post-extraction).
# ================================================================== #

def _new_transit_dispatch(gpu_fn, cpu_fn, N, mask, time, flux, sigma, ks,
                           P_arr, incs_rad, a_Rs, eccs, ws_rad, u1, u2,
                           comp_fr, companion_is_host):
    lnL = np.full(N, -np.inf)
    lnL[mask] = fpp_native._dispatch_lnL(
        gpu_fn, cpu_fn,
        (time, flux, sigma,
         ks[mask], P_arr[mask], incs_rad[mask], a_Rs[mask],
         eccs[mask], ws_rad[mask], u1, u2,
         comp_fr, companion_is_host),
        "transit")
    return lnL


def _new_eb_dispatch(gpu_fn, cpu_fn, N, mask_eb, time, flux, sigma, ks,
                      fluxratios_eb, P_arr, incs_rad, a_Rs, eccs, ws_rad,
                      u1, u2, comp_arr, companion_is_host):
    lnL_eb = np.full(N, -np.inf)
    if np.any(mask_eb):
        lnL_eb[mask_eb] = fpp_native._dispatch_lnL(
            gpu_fn, cpu_fn,
            (time, flux, sigma,
             ks[mask_eb], fluxratios_eb[mask_eb], P_arr[mask_eb],
             incs_rad[mask_eb], a_Rs[mask_eb], eccs[mask_eb], ws_rad[mask_eb],
             u1, u2, comp_arr[mask_eb], companion_is_host),
            "EB")
    return lnL_eb


def _new_twin_eb_dispatch(gpu_fn, cpu_fn, N, mask_twin, time, flux, sigma, ks,
                           fluxratios_eb, P_arr, incs_rad, a_twin_Rs, eccs,
                           ws_rad, u1, u2, comp_arr, companion_is_host):
    lnL_twin = np.full(N, -np.inf)
    if np.any(mask_twin):
        lnL_twin[mask_twin] = fpp_native._dispatch_lnL(
            gpu_fn, cpu_fn,
            (time, flux, sigma,
             ks[mask_twin], fluxratios_eb[mask_twin], 2 * P_arr[mask_twin],
             incs_rad[mask_twin], a_twin_Rs[mask_twin],
             eccs[mask_twin], ws_rad[mask_twin],
             u1, u2, comp_arr[mask_twin], companion_is_host),
            "twin EB")
    return lnL_twin


# ================================================================== #
# Shared fixtures
# ================================================================== #

@pytest.fixture
def mask_partial(transit_params):
    """Deterministic ~2/3-True boolean mask (mimics a real transit-probability
    filter that never selects 100% of the draws)."""
    N = len(transit_params["ks"])
    return (np.arange(N) % 3) != 0


def _fake_gpu_ok(*args):
    """Deterministic stand-in GPU function: returns a distinctive array of the
    right shape (args[3] is always the masked ``ks`` slice, same length for
    all 3 call-site flavors — see docstring in the tests below)."""
    n = len(args[3])
    return np.linspace(-1.0, -2.0, n)


class _SimulatedGPUFailure(RuntimeError):
    """Stand-in for a real CUDA/Metal/OpenCL exception (bare `except
    Exception` in the dispatch means the concrete type is irrelevant)."""


def _fake_gpu_raises(*args):
    raise _SimulatedGPUFailure("simulated GPU failure")


# ================================================================== #
# (a) CPU-only path (gpu_fn=None) — bit-identical output, all 3 flavors
# ================================================================== #

class TestCPUOnlyPathIdentical:
    def test_transit_no_gpu_backend(self, transit_params, mask_partial):
        p = transit_params
        N = len(p["ks"])
        old = _old_transit_dispatch(
            None, fpp_native._batch_transit_lnL, N, mask_partial,
            p["time"], p["flux"], p["sigma"], p["ks"], p["P_orbs"],
            p["incs_rad"], p["a_Rs_arr"], p["eccs"], p["ws_rad"],
            p["u1"], p["u2"], np.full(mask_partial.sum(), 0.0), False)
        new = _new_transit_dispatch(
            None, fpp_native._batch_transit_lnL, N, mask_partial,
            p["time"], p["flux"], p["sigma"], p["ks"], p["P_orbs"],
            p["incs_rad"], p["a_Rs_arr"], p["eccs"], p["ws_rad"],
            p["u1"], p["u2"], np.full(mask_partial.sum(), 0.0), False)
        np.testing.assert_array_equal(old, new)

    def test_eb_no_gpu_backend(self, eb_params, mask_partial):
        p = eb_params
        N = len(p["ks"])
        comp_arr = np.zeros(N)
        old = _old_eb_dispatch(
            None, fpp_native._batch_eb_transit_lnL, N, mask_partial,
            p["time"], p["flux"], p["sigma"], p["ks"], p["eb_fluxratios"],
            p["P_orbs"], p["incs_rad"], p["a_Rs_arr"], p["eccs"], p["ws_rad"],
            p["u1"], p["u2"], comp_arr, False)
        new = _new_eb_dispatch(
            None, fpp_native._batch_eb_transit_lnL, N, mask_partial,
            p["time"], p["flux"], p["sigma"], p["ks"], p["eb_fluxratios"],
            p["P_orbs"], p["incs_rad"], p["a_Rs_arr"], p["eccs"], p["ws_rad"],
            p["u1"], p["u2"], comp_arr, False)
        np.testing.assert_array_equal(old, new)

    def test_twin_eb_no_gpu_backend(self, eb_params, mask_partial):
        p = eb_params
        N = len(p["ks"])
        comp_arr = np.zeros(N)
        old = _old_twin_eb_dispatch(
            None, fpp_native._batch_eb_transit_lnL, N, mask_partial,
            p["time"], p["flux"], p["sigma"], p["ks"], p["eb_fluxratios"],
            p["P_orbs"], p["incs_rad"], p["a_Rs_arr"], p["eccs"], p["ws_rad"],
            p["u1"], p["u2"], comp_arr, False)
        new = _new_twin_eb_dispatch(
            None, fpp_native._batch_eb_transit_lnL, N, mask_partial,
            p["time"], p["flux"], p["sigma"], p["ks"], p["eb_fluxratios"],
            p["P_orbs"], p["incs_rad"], p["a_Rs_arr"], p["eccs"], p["ws_rad"],
            p["u1"], p["u2"], comp_arr, False)
        np.testing.assert_array_equal(old, new)

    def test_mask_all_false_regions_stay_neginf(self, transit_params, mask_partial):
        """Positions outside the mask must remain -inf in both old and new
        (this is the surrounding np.full(N, -inf) + masked-assign contract,
        not something _dispatch_lnL itself controls — verified here to make
        sure the extraction didn't disturb it)."""
        p = transit_params
        N = len(p["ks"])
        new = _new_transit_dispatch(
            None, fpp_native._batch_transit_lnL, N, mask_partial,
            p["time"], p["flux"], p["sigma"], p["ks"], p["P_orbs"],
            p["incs_rad"], p["a_Rs_arr"], p["eccs"], p["ws_rad"],
            p["u1"], p["u2"], np.full(mask_partial.sum(), 0.0), False)
        assert np.all(new[~mask_partial] == -np.inf)


# ================================================================== #
# (b) GPU-success path (fake gpu_fn, no exception) — bit-identical output
# ================================================================== #

class TestGPUSuccessPathIdentical:
    """Real GPU hardware is unavailable on this host (see module docstring).
    ``_dispatch_lnL``'s GPU branch is ``return gpu_fn(*args)`` — a single
    passthrough line — so a fake callable of the right shape exhaustively
    covers the control-flow contract; only a real kernel's own numerics are
    untested here (unchanged: same kernel functions/call sites as before)."""

    def test_transit_gpu_success(self, transit_params, mask_partial):
        p = transit_params
        N = len(p["ks"])
        old = _old_transit_dispatch(
            _fake_gpu_ok, fpp_native._batch_transit_lnL, N, mask_partial,
            p["time"], p["flux"], p["sigma"], p["ks"], p["P_orbs"],
            p["incs_rad"], p["a_Rs_arr"], p["eccs"], p["ws_rad"],
            p["u1"], p["u2"], np.full(mask_partial.sum(), 0.0), False)
        new = _new_transit_dispatch(
            _fake_gpu_ok, fpp_native._batch_transit_lnL, N, mask_partial,
            p["time"], p["flux"], p["sigma"], p["ks"], p["P_orbs"],
            p["incs_rad"], p["a_Rs_arr"], p["eccs"], p["ws_rad"],
            p["u1"], p["u2"], np.full(mask_partial.sum(), 0.0), False)
        np.testing.assert_array_equal(old, new)
        # GPU path must actually have been used (not silently fell back)
        assert np.all(new[mask_partial] == np.linspace(-1.0, -2.0, mask_partial.sum()))

    def test_eb_gpu_success(self, eb_params, mask_partial):
        p = eb_params
        N = len(p["ks"])
        comp_arr = np.zeros(N)
        old = _old_eb_dispatch(
            _fake_gpu_ok, fpp_native._batch_eb_transit_lnL, N, mask_partial,
            p["time"], p["flux"], p["sigma"], p["ks"], p["eb_fluxratios"],
            p["P_orbs"], p["incs_rad"], p["a_Rs_arr"], p["eccs"], p["ws_rad"],
            p["u1"], p["u2"], comp_arr, False)
        new = _new_eb_dispatch(
            _fake_gpu_ok, fpp_native._batch_eb_transit_lnL, N, mask_partial,
            p["time"], p["flux"], p["sigma"], p["ks"], p["eb_fluxratios"],
            p["P_orbs"], p["incs_rad"], p["a_Rs_arr"], p["eccs"], p["ws_rad"],
            p["u1"], p["u2"], comp_arr, False)
        np.testing.assert_array_equal(old, new)

    def test_twin_eb_gpu_success(self, eb_params, mask_partial):
        p = eb_params
        N = len(p["ks"])
        comp_arr = np.zeros(N)
        old = _old_twin_eb_dispatch(
            _fake_gpu_ok, fpp_native._batch_eb_transit_lnL, N, mask_partial,
            p["time"], p["flux"], p["sigma"], p["ks"], p["eb_fluxratios"],
            p["P_orbs"], p["incs_rad"], p["a_Rs_arr"], p["eccs"], p["ws_rad"],
            p["u1"], p["u2"], comp_arr, False)
        new = _new_twin_eb_dispatch(
            _fake_gpu_ok, fpp_native._batch_eb_transit_lnL, N, mask_partial,
            p["time"], p["flux"], p["sigma"], p["ks"], p["eb_fluxratios"],
            p["P_orbs"], p["incs_rad"], p["a_Rs_arr"], p["eccs"], p["ws_rad"],
            p["u1"], p["u2"], comp_arr, False)
        np.testing.assert_array_equal(old, new)


# ================================================================== #
# (c) GPU-failure -> CPU fallback path — bit-identical output AND
# identical log message AND identical numba_parallel_guard() usage
# ================================================================== #

@pytest.fixture(autouse=True)
def _reset_gpu_fallback_state():
    """Reset the GPU-fallback counters between tests (2026-07-14).

    _dispatch_lnL's circuit breaker counts CONSECUTIVE failures at MODULE
    scope — which is exactly right in production (one worker process, one GPU)
    and exactly wrong across tests: three tests that each simulate one GPU
    failure would trip the breaker on the third and make it emit an extra
    "GPU DISABLED" warning, failing an assertion that has nothing to do with
    the breaker. Each test here stands for a FRESH worker process.
    """
    saved = (fpp_native._gpu_batch_transit, fpp_native._gpu_batch_eb_transit)
    fpp_native._GPU_FALLBACK_STATS.update(
        consecutive=0, total=0, disabled=False, last_error=None)
    yield
    fpp_native._gpu_batch_transit, fpp_native._gpu_batch_eb_transit = saved
    fpp_native._GPU_FALLBACK_STATS.update(
        consecutive=0, total=0, disabled=False, last_error=None)


class TestGPUFallbackPathIdentical:
    def test_transit_gpu_raises_falls_back_to_cpu(
        self, transit_params, mask_partial, caplog, monkeypatch
    ):
        p = transit_params
        N = len(p["ks"])
        args = (
            p["time"], p["flux"], p["sigma"], p["ks"], p["P_orbs"],
            p["incs_rad"], p["a_Rs_arr"], p["eccs"], p["ws_rad"],
            p["u1"], p["u2"], np.full(mask_partial.sum(), 0.0), False,
        )

        guard_calls = []
        real_guard = fpp_native.numba_parallel_guard

        def _counting_guard():
            guard_calls.append(1)
            return real_guard()

        monkeypatch.setattr(fpp_native, "numba_parallel_guard", _counting_guard)

        caplog.set_level(logging.DEBUG, logger="exonit")
        caplog.clear()
        old = _old_transit_dispatch(
            _fake_gpu_raises, fpp_native._batch_transit_lnL, N, mask_partial, *args)
        old_messages = [r.getMessage() for r in caplog.records]
        old_guard_calls = len(guard_calls)

        guard_calls.clear()
        caplog.clear()
        new = _new_transit_dispatch(
            _fake_gpu_raises, fpp_native._batch_transit_lnL, N, mask_partial, *args)
        new_messages = [r.getMessage() for r in caplog.records]
        new_guard_calls = len(guard_calls)

        np.testing.assert_array_equal(old, new)
        # 2026-07-14: the fallback message deliberately CHANGED (DEBUG ->
        # WARNING, richer text) — a silent CPU fallback is the state CLAUDE.md
        # project rule 2 forbids, so it is no longer allowed to be byte-identical to
        # the old one. What this parity test actually guards — identical
        # numbers, and the numba guard taken exactly once — is unchanged and
        # still asserted above. See tests/test_fpp_gpu_fallback_loud.py for the
        # new contract.
        assert len(new_messages) == 1
        assert "FAILED" in new_messages[0] and "CPU" in new_messages[0]
        assert old_guard_calls == new_guard_calls == 1

    def test_eb_gpu_raises_falls_back_to_cpu(
        self, eb_params, mask_partial, caplog, monkeypatch
    ):
        p = eb_params
        N = len(p["ks"])
        comp_arr = np.zeros(N)
        args = (
            p["time"], p["flux"], p["sigma"], p["ks"], p["eb_fluxratios"],
            p["P_orbs"], p["incs_rad"], p["a_Rs_arr"], p["eccs"], p["ws_rad"],
            p["u1"], p["u2"], comp_arr, False,
        )

        guard_calls = []
        real_guard = fpp_native.numba_parallel_guard

        def _counting_guard():
            guard_calls.append(1)
            return real_guard()

        monkeypatch.setattr(fpp_native, "numba_parallel_guard", _counting_guard)

        caplog.set_level(logging.DEBUG, logger="exonit")
        caplog.clear()
        old = _old_eb_dispatch(
            _fake_gpu_raises, fpp_native._batch_eb_transit_lnL, N, mask_partial, *args)
        old_messages = [r.getMessage() for r in caplog.records]
        old_guard_calls = len(guard_calls)

        guard_calls.clear()
        caplog.clear()
        new = _new_eb_dispatch(
            _fake_gpu_raises, fpp_native._batch_eb_transit_lnL, N, mask_partial, *args)
        new_messages = [r.getMessage() for r in caplog.records]
        new_guard_calls = len(guard_calls)

        np.testing.assert_array_equal(old, new)
        # 2026-07-14: the fallback message deliberately CHANGED (DEBUG ->
        # WARNING, richer text) — a silent CPU fallback is the state CLAUDE.md
        # project rule 2 forbids, so it is no longer allowed to be byte-identical to
        # the old one. What this parity test actually guards — identical
        # numbers, and the numba guard taken exactly once — is unchanged and
        # still asserted above. See tests/test_fpp_gpu_fallback_loud.py for the
        # new contract.
        assert len(new_messages) == 1
        assert "FAILED" in new_messages[0] and "CPU" in new_messages[0]
        assert old_guard_calls == new_guard_calls == 1

    def test_twin_eb_gpu_raises_falls_back_to_cpu(
        self, eb_params, mask_partial, caplog, monkeypatch
    ):
        p = eb_params
        N = len(p["ks"])
        comp_arr = np.zeros(N)
        args = (
            p["time"], p["flux"], p["sigma"], p["ks"], p["eb_fluxratios"],
            p["P_orbs"], p["incs_rad"], p["a_Rs_arr"], p["eccs"], p["ws_rad"],
            p["u1"], p["u2"], comp_arr, False,
        )

        guard_calls = []
        real_guard = fpp_native.numba_parallel_guard

        def _counting_guard():
            guard_calls.append(1)
            return real_guard()

        monkeypatch.setattr(fpp_native, "numba_parallel_guard", _counting_guard)

        caplog.set_level(logging.DEBUG, logger="exonit")
        caplog.clear()
        old = _old_twin_eb_dispatch(
            _fake_gpu_raises, fpp_native._batch_eb_transit_lnL, N, mask_partial, *args)
        old_messages = [r.getMessage() for r in caplog.records]
        old_guard_calls = len(guard_calls)

        guard_calls.clear()
        caplog.clear()
        new = _new_twin_eb_dispatch(
            _fake_gpu_raises, fpp_native._batch_eb_transit_lnL, N, mask_partial, *args)
        new_messages = [r.getMessage() for r in caplog.records]
        new_guard_calls = len(guard_calls)

        np.testing.assert_array_equal(old, new)
        # 2026-07-14: the fallback message deliberately CHANGED (DEBUG ->
        # WARNING, richer text) — a silent CPU fallback is the state CLAUDE.md
        # project rule 2 forbids, so it is no longer allowed to be byte-identical to
        # the old one. What this parity test actually guards — identical
        # numbers, and the numba guard taken exactly once — is unchanged and
        # still asserted above. See tests/test_fpp_gpu_fallback_loud.py for the
        # new contract.
        assert len(new_messages) == 1
        assert "FAILED" in new_messages[0] and "CPU" in new_messages[0]
        assert old_guard_calls == new_guard_calls == 1

    def test_cpu_only_path_also_takes_guard_once(
        self, transit_params, mask_partial, monkeypatch
    ):
        """gpu_fn=None (no GPU backend detected) must still enter the guard
        exactly once, matching the old ``else:`` branch."""
        p = transit_params
        N = len(p["ks"])
        guard_calls = []
        real_guard = fpp_native.numba_parallel_guard

        def _counting_guard():
            guard_calls.append(1)
            return real_guard()

        monkeypatch.setattr(fpp_native, "numba_parallel_guard", _counting_guard)
        _new_transit_dispatch(
            None, fpp_native._batch_transit_lnL, N, mask_partial,
            p["time"], p["flux"], p["sigma"], p["ks"], p["P_orbs"],
            p["incs_rad"], p["a_Rs_arr"], p["eccs"], p["ws_rad"],
            p["u1"], p["u2"], np.full(mask_partial.sum(), 0.0), False)
        assert len(guard_calls) == 1


# ================================================================== #
# Real integration smoke: _lnZ_TTP / _lnZ_TEB still run end-to-end
# through the extracted helper without raising (no GPU backend on this
# host, so this exercises the gpu_fn=None branch of the real functions).
# ================================================================== #

class TestRealLnZFunctionsSmoke:
    def test_lnZ_TTP_runs_via_dispatch_lnL(self):
        assert fpp_native._gpu_batch_transit is None, (
            "test assumes no GPU FPP backend on this host; if this ever "
            "trips, the GPU-success branch is exercised for real here too")
        np.random.seed(42)
        time = np.linspace(-0.2, 0.2, 200)
        flux = np.ones(200) + np.random.normal(0, 5e-4, 200)
        sigma = float(np.std(flux))
        lnZ = fpp_native._lnZ_TTP(
            time, flux, sigma, P_orb=3.5, M_s=1.0, R_s=1.0, Teff=5800,
            N=500, u1=0.4, u2=0.26)
        assert np.isfinite(lnZ) or lnZ == -np.inf

    def test_lnZ_TEB_runs_via_dispatch_lnL(self):
        assert fpp_native._gpu_batch_eb_transit is None
        np.random.seed(42)
        time = np.linspace(-0.2, 0.2, 200)
        flux = np.ones(200) + np.random.normal(0, 5e-4, 200)
        sigma = float(np.std(flux))
        lnZ_eb, lnZ_twin = fpp_native._lnZ_TEB(
            time, flux, sigma, P_orb=3.5, M_s=1.0, R_s=1.0, Teff=5800,
            N=500, u1=0.4, u2=0.26)
        assert np.isfinite(lnZ_eb) or lnZ_eb == -np.inf
        assert np.isfinite(lnZ_twin) or lnZ_twin == -np.inf
