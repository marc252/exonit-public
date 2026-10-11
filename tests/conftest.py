"""Shared pytest fixtures for FPP GPU test suite.

Provides pre-computed CPU reference values and parameter sets that
GPU backends (CUDA, Metal, OpenCL) will validate against.
"""

import json
import math
import numpy as np
import pytest
import sys
import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Global fleet-autoscale kill-switch (incident 2026-08-01): a pytest run of
# tests/test_orchestrator_postprocess_cleanup_2026_07_02.py::
# TestPollUntilDrainedSharedHelper::test_all_three_steps_call_the_same_helper
# called the REAL step_scanning/step_fpp_backfill/step_exominer_backfill from
# scripts/discovery_orchestrator.py. Those call _scale_fleet(phase, dry_run),
# which shells out to `bin/farmctl start --all --phase <phase>` (with
# EXONIT_FARMCTL_ALLOW_START=1) — SSHing into all 10 real worker hosts and
# starting the fleet for real. The autouse `mock_scale_fleet` fixture that
# neuters `_scale_fleet` lives only inside tests/test_discovery_orchestrator.py
# (module-scoped autouse fixture, not global) and did not cover this other
# test file. A second mechanism (tests/test_discovery_api.py spawning a real
# orchestrator subprocess) can hit the same code path by inheriting the
# process environment.
#
# `_scale_fleet` itself already respects EXONIT_FLEET_AUTOSCALE=0 as a
# no-subprocess kill-switch (see scripts/discovery_orchestrator.py, ~line
# 871). Setting it here — at collection time, unconditionally, not via
# os.environ.setdefault — makes it structurally impossible for ANY test in
# the whole pytest process (and any subprocess it spawns, since subprocesses
# inherit os.environ) to start the real fleet, even from a dev shell that
# happens to already have EXONIT_FLEET_AUTOSCALE=1 exported. Tests that
# exercise the kill-switch itself (tests/test_discovery_orchestrator.py::
# TestScaleFleetReal) use monkeypatch.setenv/delenv per-test, which shadows
# and then restores this module-level value — they are unaffected.
#
# See EXONIT_LORA_CMD below for the same pattern applied to the 2026-09-26
# LoRA incident.
os.environ["EXONIT_FLEET_AUTOSCALE"] = "0"

# ---------------------------------------------------------------------------
# Global LoRA-command kill-switch (incident 2026-09-26): a pytest run of
# tests/test_orchestrator_postprocess_cleanup_2026_07_02.py's step_lora tests
# (only `subprocess.run` mocked, assuming the pre-Fase-3 <host> default)
# and tests/test_kill_and_resume.py (a PATH-shadowed fake `ssh`, assuming the
# same) both silently stopped intercepting anything once Fase 3 (2026-09-26)
# changed discovery_orchestrator's DEFAULT LORA_CMD to
# scripts/lora_macs_driver.py, invoked directly (no `ssh` in argv, and via
# subprocess.Popen not subprocess.run). Running the suite for real ran the
# REAL driver: it called `farmctl stop` on all 5 production M4 Swift workers
# mid discovery-run and wrote 1,650 real LoRA explanations to prod Postgres.
# Both test files have been fixed to pin their own EXONIT_LORA_CMD/LORA_CMD
# explicitly (see their own comments), and scripts/lora_macs_driver.py +
# bin/farmctl + scripts/generate_missing_explanations.py now each refuse to
# touch real infrastructure when PYTEST_CURRENT_TEST is set. This is the
# LAST line of defense: discovery_orchestrator.LORA_CMD is a MODULE-LEVEL
# constant resolved once, at import time, from EXONIT_LORA_CMD / the
# data/lora_cmd.override file (see _resolve_lora_cmd) — so the env var must
# be in place BEFORE any test module does `import discovery_orchestrator`.
# conftest.py's own top-level code (this module) is guaranteed by pytest to
# run before any test module is collected/imported, so setting it here,
# unconditionally (same reasoning as EXONIT_FLEET_AUTOSCALE above — a
# setdefault would let an already-exported real override survive), makes it
# structurally impossible for LORA_CMD to ever resolve to a real command by
# accident. `true` is a harmless no-op (exit 0, touches nothing) available on
# every dev machine and worker host this suite runs on.
#
# Tests that need to assert on the REAL default (e.g.
# test_discovery_orchestrator.py::test_resolve_lora_cmd_defaults_to_macs) call
# `orch._resolve_lora_cmd()` directly with `monkeypatch.delenv("EXONIT_LORA_CMD",
# raising=False)` first — monkeypatch shadows this module-level value for the
# duration of that one test and restores it afterwards, so they are
# unaffected. Tests that assert on `orch.LORA_CMD` itself always
# `monkeypatch.setattr(orch, "LORA_CMD", ...)` explicitly first (see
# tests/test_discovery_orchestrator.py's header comment), which overrides
# the import-time value directly and is likewise unaffected.
os.environ["EXONIT_LORA_CMD"] = "true"

# ---------------------------------------------------------------------------
# Shared-state isolation: force MemoryBackend + fresh instance per test.
#
# This autouse fixture ensures every test starts with a clean, isolated
# WorkerRegistry (MemoryBackend).  Without it, the singleton returned by
# get_shared_state() would bleed state between tests.
#
# test_shared_state.py is explicitly excluded: it constructs its own backends
# and calls reset_shared_state() in setup_method/teardown_method, so adding
# another autouse reset on top would be harmless but redundant — we still
# exclude it by checking the module name to avoid confusion.
# ---------------------------------------------------------------------------

# Repo root must come first so `import worker` resolves to the package
# (worker/__init__.py) and submodules like `worker.exominer_native` work.
# The flat worker/ directory is appended (not prepended) so legacy imports
# like `from injection import ...` and `import fpp_native` still resolve,
# without shadowing the `worker` package by the same-named worker.py file.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _REPO_ROOT)
if os.path.join(_REPO_ROOT, "worker") not in sys.path:
    sys.path.append(os.path.join(_REPO_ROOT, "worker"))
# scripts/ must be importable for tests that exercise scripts/backfill_*.py
if os.path.join(_REPO_ROOT, "scripts") not in sys.path:
    sys.path.append(os.path.join(_REPO_ROOT, "scripts"))

# Optional imports: some test environments (e.g. the lightweight worker venv
# used by tests/worker/**) don't ship fastapi or the compiled fpp_native
# extension. Fixtures that need these imports must reference them inside the
# fixture body so collection still succeeds when they're absent.
try:
    from fastapi.testclient import TestClient
except ImportError:  # pragma: no cover - exercised only in minimal venvs
    TestClient = None  # type: ignore[assignment]

try:
    import fpp_native
except ImportError:  # pragma: no cover - exercised only in minimal venvs
    fpp_native = None  # type: ignore[assignment]


@pytest.fixture(autouse=True)
def _never_touch_prod_db(monkeypatch):
    """Anti-prod guard (2026-07-02): NO exclusions, runs for EVERY test.

    Root cause of two prod-contamination incidents (June PG-parity truncation
    class): ``run_nightly.py`` calls ``load_dotenv()`` at import time and
    python-dotenv searches upward, finding the real ``~/exonit/.env``
    even from nested worktrees — polluting ``os.environ`` with the production
    ``EXONIT_DATABASE_URL`` for the rest of the pytest process. ``_get_db_path``
    prioritizes that DSN over per-test ``EXONIT_DB_PATH``, so app-building tests
    silently issued writes (``purge_telemetry``, worker registrations) against
    live production Postgres.

    Two layers: hide any already-polluted DSN, and neuter ``load_dotenv`` so a
    lazy ``import run_nightly`` inside a test can't re-pollute. PG-parity tests
    are unaffected — they use ``EXONIT_TEST_DATABASE_URL`` (see _pg_test_db.py).
    monkeypatch teardown restores the pre-test state either way.

    ``EXONIT_METADATA_DB_URL`` (2026-07-20 TGLC->PG migration, F1) is the same
    incident class: ``MetadataCache.__init__`` reads it UNCONDITIONALLY and
    overrides any explicit ``db_path``/``tmp_path`` a test passes, as soon as
    it's a valid ``postgresql://`` DSN — even more aggressive than
    ``EXONIT_DATABASE_URL``'s "only consulted when the caller passes None"
    rule. At F4 cutover this var lands in the real ``.env`` pointing at
    production ``exonit_meta``; without clearing it here, a polluted shell
    would silently redirect every ``MetadataCache(tmp_path)`` in the suite to
    prod. See docs/superpowers/specs/2026-07-20-tglc-index-to-pg.md D2 and
    progress/f1-metadata-backend-review.md item 1.
    """
    monkeypatch.delenv("EXONIT_DATABASE_URL", raising=False)
    monkeypatch.delenv("EXONIT_METADATA_DB_URL", raising=False)
    try:
        import dotenv
        monkeypatch.setattr(dotenv, "load_dotenv", lambda *a, **k: False)
    except ImportError:  # pragma: no cover - minimal worker venvs
        pass
    yield


@pytest.fixture(autouse=True)
def _isolate_shared_state(request, monkeypatch):
    """Force MemoryBackend and reset the shared-state singleton for every test.

    Skipped for tests in ``test_shared_state.py`` and ``test_server_workers.py``
    which manage the singleton (and multi-process env vars) themselves.

    Also ensures ``EXONIT_SERVER_WORKERS`` is absent before each test so that
    server.main(workers=N) side-effects from launcher tests do not bleed into
    subsequent tests.  (server.main writes to os.environ directly, bypassing
    monkeypatch scope — the cleanup here is the safe net.)
    """
    _self_managing = {"test_shared_state", "test_server_workers"}
    if any(name in request.fspath.basename for name in _self_managing):
        yield
        return

    # Force in-memory backend for this test (cheap, no Redis dependency).
    monkeypatch.setenv("EXONIT_SHARED_STATE_BACKEND", "memory")

    # Ensure EXONIT_SERVER_WORKERS is absent so the single-process path is used.
    monkeypatch.delenv("EXONIT_SERVER_WORKERS", raising=False)

    # Reset before the test so there is no leftover state from a previous run.
    from exonit.shared_state import reset_shared_state
    reset_shared_state()

    yield

    # Drain the endpoints' fire-and-forget refresh threads BEFORE resetting.
    # /api/discovery/sectors starts daemon threads ("downloaded-counts",
    # "gap-stats") that finish by writing to get_shared_state(). That call
    # resolves the singleton LAZILY, so a thread outliving its own test wakes
    # up inside the NEXT test and clobbers the state that test just seeded —
    # which is why TestDiscoverySectorsDownloadedCounts passed alone and in
    # small subsets but failed in a full-file run, where the timing lines up
    # (2026-07-27). Joining here makes the outcome independent of how fast
    # the endpoint happens to be.
    import threading as _threading
    for _t in _threading.enumerate():
        if _t is _threading.current_thread() or not _t.is_alive():
            continue
        if _t.name in ("downloaded-counts", "gap-stats"):
            _t.join(timeout=30)

    # Reset after the test to prevent leaking into the next one.
    reset_shared_state()


# ------------------------------------------------------------------ #
# pytest hooks: --run-parity flag + parity_gate marker
# ------------------------------------------------------------------ #
def pytest_addoption(parser):
    parser.addoption(
        "--run-parity",
        action="store_true",
        default=False,
        help="run slow Docker-based parity gate tests",
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "parity_gate: requires Docker + NASA ExoMiner image",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-parity"):
        return
    skip = pytest.mark.skip(
        reason="need --run-parity to execute parity gate tests"
    )
    for item in items:
        if "parity_gate" in item.keywords:
            item.add_marker(skip)


# ------------------------------------------------------------------ #
# Constants for realistic hot Jupiter scenario
# ------------------------------------------------------------------ #
N_DRAWS = 1000          # Monte Carlo draws
T_POINTS = 800          # Time samples
PERIOD_DAYS = 3.42      # Typical hot Jupiter period (days)
SIGMA_FLUX = 5e-4       # Photometric noise (500 ppm, TESS-like)
U1, U2 = 0.40, 0.26    # Quadratic limb darkening (solar-like, TESS band)
RNG_SEED = 42


# ------------------------------------------------------------------ #
# Helper: build a time array centered on transit
# ------------------------------------------------------------------ #
def _make_time(T, period):
    """Return T evenly-spaced times spanning ±0.12*period around t=0."""
    half = 0.12 * period
    return np.linspace(-half, half, T)


# ------------------------------------------------------------------ #
# Fixtures
# ------------------------------------------------------------------ #

@pytest.fixture(scope="session")
def transit_params():
    """Parameter arrays for N=1000 transit draws (hot Jupiter defaults).

    Returns a dict with:
        time, flux, sigma,
        ks, P_orbs, incs_rad, a_Rs_arr, eccs, ws_rad,
        companion_fluxratio, companion_is_host, u1, u2
    """
    rng = np.random.default_rng(RNG_SEED)

    time = _make_time(T_POINTS, PERIOD_DAYS)

    # Realistic hot Jupiter parameter distributions
    ks            = rng.uniform(0.08, 0.16, N_DRAWS)           # Rp/Rs  (8-16 %)
    P_orbs        = rng.uniform(3.0, 3.9, N_DRAWS)             # period scatter (days)
    incs_deg      = rng.uniform(82.0, 90.0, N_DRAWS)           # inclination (deg)
    incs_rad      = np.deg2rad(incs_deg)
    a_Rs_arr      = rng.uniform(7.0, 12.0, N_DRAWS)            # a/Rs (hot Jup range)
    eccs          = rng.uniform(0.0, 0.05, N_DRAWS)            # near-circular
    ws_deg        = rng.uniform(0.0, 360.0, N_DRAWS)           # arg of periastron
    ws_rad        = np.deg2rad(ws_deg)

    # No companion dilution for baseline transit scenario
    companion_fluxratio = np.zeros(N_DRAWS)
    companion_is_host   = False

    # Synthetic observed flux: flat (no signal) + Gaussian noise
    flux = 1.0 + rng.normal(0.0, SIGMA_FLUX, T_POINTS)

    return dict(
        time=time,
        flux=flux,
        sigma=SIGMA_FLUX,
        ks=ks,
        P_orbs=P_orbs,
        incs_rad=incs_rad,
        a_Rs_arr=a_Rs_arr,
        eccs=eccs,
        ws_rad=ws_rad,
        companion_fluxratio=companion_fluxratio,
        companion_is_host=companion_is_host,
        u1=U1,
        u2=U2,
    )


@pytest.fixture(scope="session")
def eb_params(transit_params):
    """Extends transit_params with eb_fluxratios for eclipsing-binary scenario."""
    rng = np.random.default_rng(RNG_SEED + 1)
    # EB flux ratio: secondary contributes 1-20 % of total system flux
    eb_fluxratios = rng.uniform(0.01, 0.20, N_DRAWS)
    return {**transit_params, "eb_fluxratios": eb_fluxratios}


@pytest.fixture(scope="session")
def cpu_reference_lnL(transit_params):
    """Pre-computed lnL array from fpp_native._batch_transit_lnL (CPU baseline)."""
    p = transit_params
    lnL = fpp_native._batch_transit_lnL(
        p["time"], p["flux"], p["sigma"],
        p["ks"], p["P_orbs"], p["incs_rad"],
        p["a_Rs_arr"], p["eccs"], p["ws_rad"],
        p["u1"], p["u2"],
        p["companion_fluxratio"], p["companion_is_host"],
    )
    return lnL


@pytest.fixture(scope="session")
def cpu_reference_eb_lnL(eb_params):
    """Pre-computed lnL array from fpp_native._batch_eb_transit_lnL (CPU baseline)."""
    p = eb_params
    lnL = fpp_native._batch_eb_transit_lnL(
        p["time"], p["flux"], p["sigma"],
        p["ks"], p["eb_fluxratios"], p["P_orbs"],
        p["incs_rad"], p["a_Rs_arr"], p["eccs"], p["ws_rad"],
        p["u1"], p["u2"],
        p["companion_fluxratio"], p["companion_is_host"],
    )
    return lnL


# ------------------------------------------------------------------ #
# Functional test fixtures
# ------------------------------------------------------------------ #

_FIXTURES_DIR = Path(__file__).parent / "fixtures"


def load_golden(name: str) -> dict:
    """Load a golden fixture JSON file from tests/fixtures/."""
    path = _FIXTURES_DIR / name
    return json.loads(path.read_text())


@pytest.fixture
def functional_client(tmp_path):
    """FastAPI TestClient with a registered worker and isolated DB.

    Yields (client, worker_id, headers, tmp_path) where headers
    includes both auth token and X-Worker-ID.
    """
    os.environ["EXONIT_DB_PATH"] = str(tmp_path / "test.db")
    os.environ["EXONIT_REPORTS_DIR"] = str(tmp_path / "reports")
    os.makedirs(tmp_path / "reports", exist_ok=True)

    from exonit.server import create_app
    app = create_app()
    client = TestClient(app)

    auth = {"Authorization": "Bearer exonit-default-token"}
    reg = client.post("/api/worker/register",
                      json={"name": "test-functional-worker",
                            "capabilities": {"gpu": "none"}},
                      headers=auth)
    worker_id = reg.json()["worker_id"]
    headers = {**auth, "X-Worker-ID": worker_id}

    yield client, worker_id, headers, tmp_path

    del os.environ["EXONIT_DB_PATH"]
    del os.environ["EXONIT_REPORTS_DIR"]
