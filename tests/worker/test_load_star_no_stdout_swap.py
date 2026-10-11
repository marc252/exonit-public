"""B2 regression test (2026-07-02 cleanup review): ``_load_star`` must
never reassign the process-global ``sys.stdout``.

``_load_star`` runs on both the main thread and the FPP-prefetch
background thread (``_fetch_and_load_fpp``, spawned via
``_PrefetchSlot``/``_try_fpp_backfill``). A per-call ``sys.stdout =
open(os.devnull, 'w')`` swap without a lock can race between the two
threads and leave ``sys.stdout`` pointing at a closed file handle.

Investigated 2026-07-02: the swap suppressed nothing in the first place
(the lightkurve "N% of the cadences will be ignored" noise it targeted
is emitted via the ``lightkurve.utils`` Python ``logging`` logger, not
via stdout/print — verified empirically against real cached FITS files,
and already thread-safely silenced at the source by ``_setup_logging``'s
``lightkurve.utils`` CRITICAL+1 + propagate=False filter). The swap was
removed rather than lock-protected. This test is a static/AST guard so a
future "helpful" re-add of the swap gets caught without needing a live
FITS-reading integration test.
"""
from __future__ import annotations

import ast
from pathlib import Path

_WORKER_SRC = Path(__file__).resolve().parents[2] / "worker" / "worker.py"


def _find_fn(tree: ast.AST, name: str) -> ast.FunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"Function {name!r} not found in {_WORKER_SRC}")


def _assigns_sys_stdout(node: ast.AST) -> bool:
    """True if *node* is an assignment whose target is ``sys.stdout``."""
    if not isinstance(node, ast.Assign):
        return False
    for target in node.targets:
        if (isinstance(target, ast.Attribute) and target.attr == "stdout"
                and isinstance(target.value, ast.Name) and target.value.id == "sys"):
            return True
    return False


def test_load_star_never_reassigns_sys_stdout():
    tree = ast.parse(_WORKER_SRC.read_text())
    fn = _find_fn(tree, "_load_star")

    offenders = [node.lineno for node in ast.walk(fn) if _assigns_sys_stdout(node)]

    assert not offenders, (
        "_load_star must not reassign sys.stdout (B2, 2026-07-02 cleanup "
        "review) — it runs on both the main thread and the FPP-prefetch "
        "background thread and a per-call swap races. Found assignment(s) "
        "at line(s): " + ", ".join(str(ln) for ln in offenders)
    )
