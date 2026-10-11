"""Deterministic per-candidate FPP seed (DeepSeekReview_18092026.md, item C1).

worker/fpp_native.py used the process-global numpy RNG unseeded, so FPP/NFPP
for the same candidate differed between two runs. This suite pins the
normative seed scheme shared with the Rust (`fpp/rng.rs::tic_seed`) and
Swift (`FPPRNG.ticSeed`) ports:

    key      = f"{tic_id}:{period:.10e}:{t0:.10e}"      # ASCII
    seed_u32 = fnv1a64(key) & 0xFFFFFFFF

and verifies:
  (a) the FNV-1a 64 hash matches published test vectors and the Rust/Swift
      parity fixture (exonit-rust-worker-fpp/fixtures/fpp/rng_golden.json
      "tic_seeds" section — a handful of key->seed pairs are hardcoded
      below so this suite has no dependency on the sibling repo);
  (b) `np.random.RandomState(seed).rand(5)` reproduces the golden RNG
      stream pinned in the same fixture's "rng" section (seed=42, rand16);
  (c) `calc_fpp_native` is bit-exact across two calls on the same tiny
      synthetic candidate, and a different period (-> different seed)
      changes the result.

Threading finding (see progress/impl-fpp-seed.md): worker.py's FPP-prefetch
background thread only downloads/loads the light curve (`_load_star`); the
actual Monte Carlo (`calc_fpp_native`, which owns the RNG) always runs
serially on the main thread — one candidate at a time, never concurrently.
The `rng` parameter is still threaded explicitly through every `_sample_*`/
`_lnZ_*` helper (rather than relying on a mutated `np.random.seed()` global)
per the review's plumbing requirement, so this stays true even if a future
change parallelizes FPP across candidates.
"""
from __future__ import annotations

import numpy as np
import pytest

import fpp_native


# ============================================================
# (a) FNV-1a 64 + tic_seed
# ============================================================

def test_fnv1a64_published_test_vectors():
    assert fpp_native._fnv1a64(b"") == 0xCBF29CE484222325
    assert fpp_native._fnv1a64(b"a") == 0xAF63DC4C8601EC8C


# Hand-picked (key -> seed) pairs pinned in
# exonit-rust-worker-fpp/fixtures/fpp/rng_golden.json's "tic_seeds" section
# (generated from real frozen FPP inputs). tic_id is a string there because
# the Rust/Swift parity harness always feeds tic_seed() a string tic_id.
TIC_SEED_GOLDEN = [
    ("10756814", 0.6331, 1684.639, 518588343),
    ("10756829", 1.9989, 1684.2977, 3768691296),
    ("10756921", 0.9383, 1684.4157, 4108009443),
    ("10757100", 23.2789, 1695.8395, 396087499),
    ("10757120", 13.9883, 1694.8087, 553060415),
    ("10757308", 13.975, 1695.986, 3022563049),
]


@pytest.mark.parametrize("tic_id,period,t0,expected_seed", TIC_SEED_GOLDEN)
def test_tic_seed_matches_rust_swift_parity_fixture(tic_id, period, t0, expected_seed):
    assert fpp_native.tic_seed(tic_id, period, t0) == expected_seed


def test_tic_seed_key_format_is_python_e10():
    # f"{x:.10e}" -> mantissa with 10 decimals, exponent zero-padded to >=2
    # digits with an explicit sign -- normative per parity/fpp/README.md.
    # Reconstruct the key the same way tic_seed() does and check its shape
    # rather than reaching into the function body.
    built = f"10756814:{0.6331:.10e}:{1684.639:.10e}"
    assert built == "10756814:6.3310000000e-01:1.6846390000e+03"
    assert fpp_native._fnv1a64(built.encode("ascii")) & 0xFFFFFFFF == 518588343


def test_tic_seed_accepts_int_tic_id_identically_to_str():
    # calc_fpp_native's callers pass tic_id as whatever the task JSON gave
    # them (usually int); f"{tic_id}" renders the same digits either way.
    assert fpp_native.tic_seed(10756814, 0.6331, 1684.639) == \
        fpp_native.tic_seed("10756814", 0.6331, 1684.639)


def test_tic_seed_differs_for_different_period():
    s1 = fpp_native.tic_seed("12345", 3.5, 1200.0)
    s2 = fpp_native.tic_seed("12345", 3.500001, 1200.0)
    assert s1 != s2


# ============================================================
# (b) RandomState(seed) reproduces the golden numpy-legacy stream
# ============================================================

# fixtures/fpp/rng_golden.json rng["42"]["rand16"][:5], decoded from the
# little-endian f64 hex the parity harness stores floats as.
_GOLDEN_RAND5_SEED42 = [
    0.3745401188473625,
    0.9507143064099162,
    0.7319939418114051,
    0.5986584841970366,
    0.15601864044243652,
]


def test_randomstate_rand_matches_golden_stream_seed42():
    rng = np.random.RandomState(42)
    assert list(rng.rand(5)) == _GOLDEN_RAND5_SEED42


def test_randomstate_equivalent_to_global_seed():
    np.random.seed(553060415)
    global_stream = list(np.random.rand(8))
    rs_stream = list(np.random.RandomState(553060415).rand(8))
    assert global_stream == rs_stream


# ============================================================
# (c) calc_fpp_native determinism (tiny synthetic candidate, CPU-only)
# ============================================================

def _mock_target_star():
    """Single target star, no neighbors -- keeps the MC tiny and skips the
    nearby-star (NTP/NEB) loop entirely."""
    return ([{
        "ID": "999999999", "ra": 100.0, "dec": -20.0, "Tmag": 10.0,
        "mass": 1.0, "rad": 1.0, "Teff": 5777.0, "plx": 5.0,
    }], 15.0)


def _synthetic_lc(period, t0, seed=7, n=600):
    """Tiny synthetic phase-folded-friendly light curve with a shallow dip
    at t0 -- just enough structure for the MC scenarios to have something
    to weigh, not meant to be astrophysically realistic."""
    rng = np.random.RandomState(seed)
    time = np.linspace(0.0, 27.0, n)
    flux = np.ones(n) + rng.normal(0, 0.0005, n)
    phase = ((time - t0 + period / 2) % period) - period / 2
    in_transit = np.abs(phase) < 0.05
    flux[in_transit] -= 0.003
    return np.ascontiguousarray(time, dtype=np.float64), \
        np.ascontiguousarray(flux, dtype=np.float64)


@pytest.fixture
def mocked_fpp_env(monkeypatch):
    """No network: _query_tic_stars / _load_trilegal_stars are the only FPP
    inputs that hit the server/filesystem; force_cpu=True at the call site
    covers the GPU side (EXONIT_FPP_GPU_BACKENDS is not needed)."""
    monkeypatch.setattr(fpp_native, "_query_tic_stars",
                         lambda *a, **k: _mock_target_star())
    monkeypatch.setattr(fpp_native, "_load_trilegal_stars",
                         lambda *a, **k: [])


def _run_fpp(period, t0, N=200, tic_id=99999999):
    time, flux = _synthetic_lc(period, t0)
    return fpp_native.calc_fpp_native(
        tic_id, period, t0, duration=0.1, depth_ppm=2000.0,
        sectors=np.array([1]), time=time, flux=flux, N=N, force_cpu=True)


def test_calc_fpp_native_is_bit_exact_across_repeated_calls(mocked_fpp_env):
    r1 = _run_fpp(period=3.5, t0=1.2)
    r2 = _run_fpp(period=3.5, t0=1.2)
    assert r1 is not None and r2 is not None
    assert r1["fpp"] == r2["fpp"]
    assert r1["nfpp"] == r2["nfpp"]
    assert r1["verdict"] == r2["verdict"]
    assert r1["fpp_seed"] == r2["fpp_seed"]


def test_calc_fpp_native_seed_matches_tic_seed(mocked_fpp_env):
    r = _run_fpp(period=3.5, t0=1.2, tic_id=99999999)
    assert r is not None
    assert r["fpp_seed"] == fpp_native.tic_seed(99999999, 3.5, 1.2)


def test_calc_fpp_native_different_period_changes_seed_and_result(mocked_fpp_env):
    r1 = _run_fpp(period=3.5, t0=1.2)
    r2 = _run_fpp(period=5.7, t0=1.2)
    assert r1 is not None and r2 is not None
    assert r1["fpp_seed"] != r2["fpp_seed"]
    # Different draw stream -> not required to differ numerically in
    # principle, but with a real (non-degenerate) MC it does in practice;
    # this guards against seeding accidentally being ignored.
    assert (r1["fpp"], r1["nfpp"]) != (r2["fpp"], r2["nfpp"])


def test_calc_fpp_native_does_not_mutate_global_numpy_random_state(mocked_fpp_env):
    """The whole point of plumbing an explicit RandomState is that FPP must
    not perturb (or depend on) the process-global np.random stream."""
    np.random.seed(1234)
    before = list(np.random.rand(3))

    np.random.seed(1234)
    _run_fpp(period=3.5, t0=1.2)
    after = list(np.random.rand(3))

    assert before == after
