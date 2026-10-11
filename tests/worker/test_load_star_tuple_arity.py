"""Regression test: Worker._load_star must always return a 5-tuple.

Contract (as of 2026-04-20): ``_load_star`` returns
``(time, flux, raw_flux, n_sectors, meta)``.

``raw_flux`` was added on 2026-04-20 to enable the multi-planet
re-detrend loop (Hippke+ 2019 §5, audit #6): the BLS iterative masking
needs the flux *after* stitch+fill_gaps but *before* wotan detrending
so it can re-run detrend with an accumulated transit mask on each
iteration. See docs/superpowers/specs/2026-04-20-multi-planet-redetrend-design.md.

Historical context (kept — the guard this test enforces is still useful):
A bug observed 2026-04-20 on TIC 434100688 had the ``No FITS files downloaded``
branch at worker/worker.py:541 returning ``None, None, 0`` (3 elements) when
the contract was a 4-tuple, raising ``ValueError: not enough values to unpack``
and marking the task failed with a misleading message. Fix in commit
``b994c0f``. The same failure mode was flagged in the 2026-04-17 audit (FPP
infinite-loop overnight) and commit ``a956b54`` claimed to fix it — but that
specific return path slipped through. This structural test prevents any
future branch of ``_load_star`` from regressing the contract.
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


def test_load_star_every_return_has_five_elements():
    tree = ast.parse(_WORKER_SRC.read_text())
    fn = _find_fn(tree, "_load_star")

    offenders = []
    for node in ast.walk(fn):
        if isinstance(node, ast.Return) and isinstance(node.value, ast.Tuple):
            n = len(node.value.elts)
            if n != 5:
                offenders.append((node.lineno, n))

    assert not offenders, (
        "Worker._load_star must return a 5-tuple "
        "(time, flux, raw_flux, n_sectors, meta) on every branch. Found: "
        + ", ".join(f"line {ln} returns {k} elements" for ln, k in offenders)
    )


def test_callers_still_unpack_five_values():
    """Belt-and-braces: verify the call sites haven't silently moved to
    a different arity. If a future refactor wants to change the contract,
    this test forces updating both sides together."""
    tree = ast.parse(_WORKER_SRC.read_text())

    call_sites = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Assign)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Attribute)
            and node.value.func.attr == "_load_star"
        ):
            for target in node.targets:
                if isinstance(target, ast.Tuple):
                    call_sites.append((node.lineno, len(target.elts)))

    assert call_sites, "No _load_star call sites found — test is stale"
    for lineno, arity in call_sites:
        assert arity == 5, (
            f"Call site at line {lineno} unpacks {arity} values "
            "but _load_star returns a 5-tuple."
        )
