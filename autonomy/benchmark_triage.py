"""Error-triage benchmark harness (phase 1 — shadow mode).

`score(cases, choose_fn)` evaluates any function `ErrorContext -> object with
.playbook` (`classify()` from `autonomy/classifier.py`, the `.choose()` of a
`TriageClient`, or a test stub) against a bank of labelled cases.

`cases.jsonl` (`autonomy/benchmark/`, ≥20 cases) is the real bank: each line is
`{"id", "ctx": {ErrorContext fields}, "expected": "<playbook>",
"dangerous_wrong": [...]}`. The `detail`/`log_tail` of each case is an
adaptation of real incident text (`docs/STATUS.md` entries 0h/0r/0u + the
"Known patterns" section of `CLAUDE.md`), never a synthetic label — see the
generator (out of the repo's scope) and the task 10 report for the
case-by-case provenance.

CLI (`python -m autonomy.benchmark_triage`):

    --rules-only            evaluates ONLY `classifier.classify()` (deterministic,
                             no AI, no network call) — the local baseline.
    --base-url/--model      evaluates `TriageClient(base_url, model).choose` —
                             **makes a real HTTP call per case** to the indicated
                             OpenAI-compatible server. DO NOT run against the
                             production local qwen on <host> unless the operator
                             launches it themselves (contention with their
                             personal use — see `docs/runbooks/` for the pending step).

With `--rules-only`, a `classify()` that returns `None` (no deterministic rule
matches) is treated as the safe fallback `pause_and_alert` — the same criterion
that `autonomy/supervisor.py::_tick_error` uses today when the AI is not
consulted (shadow mode).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from autonomy.classifier import classify
from autonomy.types import ErrorContext, PlaybookChoice

DEFAULT_CASES_PATH = Path(__file__).parent / "benchmark" / "cases.jsonl"


@dataclass
class Score:
    """Result of evaluating a `choose_fn` against a bank of cases."""

    n: int
    correct: int
    accuracy: float
    dangerous_errors: list = field(default_factory=list)
    escalation_recall: float = 0.0


def load_cases(path: Path) -> list[dict]:
    """Read the case bank (`cases.jsonl`, one JSON per line)."""
    cases: list[dict] = []
    with Path(path).open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            cases.append(json.loads(line))
    return cases


def score(cases: list[dict], choose_fn: Callable[[ErrorContext], object]) -> Score:
    """Evaluate `choose_fn` against `cases`.

    `choose_fn` only needs to return an object with `.playbook` (duck
    typing: both `PlaybookChoice` and a test stub work). A case is
    "correct" if `choose_fn(ctx).playbook == case["expected"]`.

    `dangerous_errors` collects the cases where the answer is one of the case's
    `dangerous_wrong` (active, plausible but wrong actions,
    e.g. `restart_worker` when the real cause is a full disk). `escalation_recall`
    is the fraction of `expected == "pause_and_alert"` cases got right — it measures
    whether the system knows NOT to act when it should escalate.
    """
    n = len(cases)
    correct = 0
    dangerous_errors: list = []
    escalation_total = 0
    escalation_correct = 0

    for case in cases:
        ctx = ErrorContext(**case["ctx"])
        choice = choose_fn(ctx)
        actual = choice.playbook
        expected = case["expected"]
        is_correct = actual == expected

        if is_correct:
            correct += 1

        if expected == "pause_and_alert":
            escalation_total += 1
            if is_correct:
                escalation_correct += 1

        if actual in (case.get("dangerous_wrong") or []):
            dangerous_errors.append({"id": case["id"], "expected": expected, "actual": actual})

    accuracy = correct / n if n else 0.0
    escalation_recall = escalation_correct / escalation_total if escalation_total else 0.0

    return Score(n, correct, accuracy, dangerous_errors, escalation_recall)


def _rules_only_choose(ctx: ErrorContext) -> PlaybookChoice:
    """Local baseline: only `classify()`. `None` (no rule) -> `pause_and_alert`,
    the same safe-fallback criterion the supervisor uses in shadow
    mode when it does not consult the AI (never act on ambiguous evidence)."""
    choice = classify(ctx)
    if choice is None:
        return PlaybookChoice(
            "pause_and_alert",
            {"reason": "no deterministic rule matches"},
            0.0,
            "classify() returned None; in production this case would go to the local AI",
            "fallback",
        )
    return choice


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark of ExoNit error triage against a bank of labelled cases."
    )
    parser.add_argument("--base-url", default=None, help="base URL of the OpenAI-compatible server (TriageClient).")
    parser.add_argument("--model", default=None, help="model name (TriageClient).")
    parser.add_argument(
        "--rules-only",
        action="store_true",
        help="evaluate only the deterministic classifier (classify()) — no network call.",
    )
    parser.add_argument(
        "--cases",
        default=str(DEFAULT_CASES_PATH),
        help=f"path to the JSONL case bank (default: {DEFAULT_CASES_PATH}).",
    )
    return parser


def main(argv: list | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if not args.rules_only and (not args.base_url or not args.model):
        parser.error("need --rules-only, or else --base-url and --model for TriageClient.")

    cases = load_cases(Path(args.cases))

    if args.rules_only:
        choose_fn = _rules_only_choose
        mode = "rules-only (deterministic classify(), no network call)"
    else:
        # Deferred import: `urllib`/network is only needed if this mode is used.
        from autonomy.triage_llm import TriageClient

        client = TriageClient(args.base_url, args.model)
        choose_fn = client.choose
        mode = f"llm ({args.base_url}, model={args.model}) — REAL HTTP CALLS"

    s = score(cases, choose_fn)

    print(json.dumps(
        {
            "mode": mode,
            "cases_path": str(args.cases),
            "n": s.n,
            "correct": s.correct,
            "accuracy": round(s.accuracy, 4),
            "escalation_recall": round(s.escalation_recall, 4),
            "dangerous_errors": s.dangerous_errors,
        },
        indent=2,
        ensure_ascii=False,
    ))


if __name__ == "__main__":
    main()
