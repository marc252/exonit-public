"""Test that discovery_orchestrator.py is importable when run as a standalone
script from scripts/ directory (production condition on AWS).

Production invocation on AWS:
    python /opt/exonit/scripts/discovery_orchestrator.py ...

In that case sys.path[0] = '/opt/exonit/scripts', and the exonit/ package at
project-root is NOT on sys.path unless the script adds it explicitly.

Before the fix, step_stop_workers and step_exominer_backfill did:
    from exonit.config import deployment as _deployment
inside the function body, which fired during EXOMINER_BACKFILL poll loops when
the Docker queue appeared stuck.  Because exonit/ was not on sys.path (scripts/
only), this raised ModuleNotFoundError, which the outer `except Exception` caught
and turned into "poll error (will retry)" — causing an infinite loop with 12 GPU
instances burning money.

This test reproduces the exact production condition via subprocess, verifies the
fix (sys.path.insert at module level) makes the import succeed, and exercises
the specific code path (step_stop_workers with EXONIT_DEPLOYMENT=aws, dry_run)
that triggered the bug in production.
"""

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"


def _run_isolated(code: str, extra_env: dict | None = None, cwd: str | None = None) -> subprocess.CompletedProcess:
    """Run a Python snippet in a subprocess whose sys.path[0] is scripts/
    (mirroring `python /opt/exonit/scripts/discovery_orchestrator.py ...`).

    The environment intentionally does NOT include the project root in
    PYTHONPATH, so the test is bit-identical to the production failure mode.
    """
    env = {
        # Minimal environment: only inherit PATH and HOME so Python finds its
        # standard library, but strip PYTHONPATH to avoid accidentally
        # inheriting the current test environment's project-root entry.
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": os.environ.get("HOME", "/root"),
    }
    if extra_env:
        env.update(extra_env)

    # Use -c to execute a snippet with sys.argv[0] = '' and sys.path[0] = ''.
    # We prepend scripts/ to sys.path the same way CPython does when you run
    # `python /path/to/scripts/foo.py` (it inserts the script's directory).
    preamble = textwrap.dedent(f"""\
        import sys
        # Simulate production: scripts/ at the front, project-root NOT present.
        sys.path.insert(0, {str(SCRIPTS_DIR)!r})
        # Remove any accidental project-root entry inherited from the test runner.
        _root = {str(REPO_ROOT)!r}
        sys.path = [p for p in sys.path if p != _root]
    """)

    full_code = preamble + "\n" + textwrap.dedent(code)
    return subprocess.run(
        [sys.executable, "-c", full_code],
        capture_output=True,
        text=True,
        cwd=cwd or str(REPO_ROOT),
        env=env,
        timeout=30,
    )


# ---------------------------------------------------------------------------
# Core regression test
# ---------------------------------------------------------------------------

def test_step_stop_workers_aws_dry_run_importable():
    """step_stop_workers(dry_run=True) with EXONIT_DEPLOYMENT=aws must succeed.

    This directly exercises the `from exonit.config import deployment` call that
    caused ModuleNotFoundError in production (scripts/ on sys.path but project-
    root absent).  With the fix (2026-07-03: `exonit` is pip-installed editable
    for the interpreter that runs this script — see pyproject.toml — so it
    resolves via the interpreter's meta path finder regardless of sys.path[0]),
    the import succeeds and step_stop_workers returns the expected dry-run tuple.
    """
    code = """\
        import os
        os.environ["EXONIT_DEPLOYMENT"] = "aws"
        import discovery_orchestrator as orch
        ok, detail = orch.step_stop_workers(state_file="unused", dry_run=True)
        assert ok is True, f"expected ok=True, got {ok!r}"
        expected = "dry_run: would terminate_fleet (aws)"
        assert detail == expected, f"got detail={detail!r}"
        print("PASS")
    """
    result = _run_isolated(code, extra_env={"EXONIT_DEPLOYMENT": "aws"})
    assert result.returncode == 0, (
        f"subprocess failed (rc={result.returncode}).\n"
        f"stdout: {result.stdout}\n"
        f"stderr: {result.stderr}"
    )
    assert "PASS" in result.stdout, (
        f"Expected 'PASS' in stdout.\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )


def test_step_exominer_backfill_aws_stuck_proceeds():
    """When EXOMINER_BACKFILL is stuck and EXONIT_DEPLOYMENT=aws, it proceeds.

    This is the second `from exonit.config import deployment` call site
    (line ~377) that also fired during the infinite-loop incident.  With the fix
    it resolves correctly and the AWS branch returns (True, <aws_msg>).
    """
    code = """\
        import os, json
        from unittest.mock import patch

        os.environ["EXONIT_DEPLOYMENT"] = "aws"
        # Reach the AWS-stuck branch: the phase short-circuits at the top when
        # EXONIT_EXOMINER_BACKFILL!=1 (disabled-skip from cb4b2934).
        os.environ["EXONIT_EXOMINER_BACKFILL"] = "1"
        import discovery_orchestrator as orch

        # Simulate: pending=42, never decreases → stuck after EXOMINER_STUCK_POLLS polls.
        call_count = [0]
        def fake_urlopen(url, timeout=None):
            call_count[0] += 1
            class _R:
                def read(self):
                    return json.dumps({
                        "exominer_backfill": {"pending": 42, "currently_assigned": 0}
                    }).encode()
                def __enter__(self): return self
                def __exit__(self, *_): pass
            return _R()

        with patch("urllib.request.urlopen", side_effect=fake_urlopen), \\
             patch("time.sleep"):
            ok, detail = orch.step_exominer_backfill(
                "http://localhost:8400", state_file="unused",
                poll_interval=1, dry_run=False
            )

        # On AWS: stuck → proceed (not FAILED)
        assert ok is True, f"expected ok=True on AWS stuck, got {ok!r}"
        assert "proceeding" in detail.lower() or "aws" in detail.lower(), \\
            f"expected AWS proceed message, got detail={detail!r}"
        print("PASS")
    """
    result = _run_isolated(code, extra_env={"EXONIT_DEPLOYMENT": "aws"})
    assert result.returncode == 0, (
        f"subprocess failed (rc={result.returncode}).\n"
        f"stdout: {result.stdout}\n"
        f"stderr: {result.stderr}"
    )
    assert "PASS" in result.stdout, (
        f"Expected 'PASS' in stdout.\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )


def test_orchestrator_importable_with_scripts_only_in_path():
    """Module-level import of discovery_orchestrator succeeds with scripts/ only.

    Verifies the fix (editable install of `exonit`, see pyproject.toml) covers
    module-load-time imports (not deferred), so no other code path can trigger
    ModuleNotFoundError later.
    """
    code = """\
        import discovery_orchestrator as orch
        # Verify the exonit package is now importable as a side-effect of the import
        import exonit.config
        print("PASS")
    """
    result = _run_isolated(code)
    assert result.returncode == 0, (
        f"subprocess failed (rc={result.returncode}).\n"
        f"stdout: {result.stdout}\n"
        f"stderr: {result.stderr}"
    )
    assert "PASS" in result.stdout


# ---------------------------------------------------------------------------
# Verify the fix mechanism is present (belt-and-suspenders check)
# ---------------------------------------------------------------------------

def test_exonit_importable_without_root_on_syspath(tmp_path):
    """`exonit` must resolve even with scripts/ (not project-root) on sys.path.

    2026-07-03: replaces the old static check that grepped the source for a
    `sys.path.insert(0, str(ROOT))` line. That textual hack was removed from
    discovery_orchestrator.py (and the rest of scripts/) in favour of an
    editable install of the `exonit` package (see pyproject.toml,
    `pip install --user --break-system-packages -e . --no-deps`), which makes
    `exonit` resolvable via the interpreter's meta path finder regardless of
    sys.path[0]/cwd. This test guards that mechanism directly (in the same
    stripped-sys.path subprocess as the tests above) instead of grepping for
    a line of code that no longer needs to exist.

    2026-07-03 review round 2 (confidence 80): `python -c` implicitly seeds
    sys.path[0] = '' (cwd), and _run_isolated()'s default cwd is REPO_ROOT
    itself — which contains the real exonit/ package directory. So the
    naive version of this test passed even with the editable install
    completely absent (verified with `python3 -S`, which never loads the
    site/meta-path finder machinery). Pinned cwd=tmp_path (a pytest temp
    dir outside the repo, with no exonit/ subdirectory) so the ONLY way
    `import exonit` can resolve here is via the installed package.

    Deployment note: any OTHER machine that runs scripts/*.py standalone
    (e.g. a future AWS box) needs this same editable install for its own
    interpreter — this test only proves the packaging/mechanism works, not
    that every possible target machine has run the install step.
    """
    code = """\
        import exonit
        assert exonit.__file__, "exonit resolved but has no __file__ (namespace pkg collision?)"
        print("PASS", exonit.__file__)
    """
    result = _run_isolated(code, cwd=str(tmp_path))
    assert result.returncode == 0, (
        f"subprocess failed (rc={result.returncode}).\n"
        f"stdout: {result.stdout}\n"
        f"stderr: {result.stderr}\n"
        "Is `exonit` pip-installed (editable) for this interpreter? Run "
        "`pip install --user --break-system-packages -e . --no-deps` from the repo root."
    )
    assert "PASS" in result.stdout, (
        f"Expected 'PASS' in stdout.\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )
    assert str(REPO_ROOT) in result.stdout, (
        f"`exonit` resolved from an unexpected location (expected under "
        f"{REPO_ROOT}): {result.stdout}"
    )
