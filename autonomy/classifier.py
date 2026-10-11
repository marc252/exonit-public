"""Deterministic error classifier (phase 1 — v1 rules).

Evaluates an `ErrorContext` against an ordered list of rules (`RULES`);
the first one that matches determines the returned `PlaybookChoice`, always
with `source="rules"` and `confidence=1.0`. If no rule matches, returns
`None` and the supervisor will pass the context to the local AI (out of scope
for this task).

The playbooks referenced here are those of the closed CATALOG in
`autonomy/playbooks.py`; the `PlaybookChoice`s returned must
pass `playbooks.validate()`.
"""

from __future__ import annotations

import re
from typing import Callable, Optional

from autonomy.types import ErrorContext, PlaybookChoice

_TRANSIENT_NETWORK_TRIGGERS = (
    "connection refused",
    "connection reset",
    "timed out",
    "temporarily unavailable",
)

# Negation guard for rule 3: removes the WHOLE sentence (up to the
# nearest delimiter, in both directions) that contains "no orphan(s)" or
# "0 orphan(s)" before checking whether a positive mention remains.
# Scoped to the sentence (not global to the whole text) so that a negation
# in one sentence does not hide a real positive mention in another. The
# sentence delimiters include ".", ";" and "\n" — the latter is
# essential because `_text()` joins `detail` and `log_tail` with "\n": without
# excluding it, a negation in `detail` would also "eat" a positive mention
# in `log_tail` (or vice versa) across the line break.
_ORPHAN_NEGATION_RE = re.compile(r"[^.\n;]*\b(no|0)\s+orphans?\b[^.\n;]*[.\n;]?")


def _text(ctx: ErrorContext) -> str:
    return f"{ctx.detail}\n{ctx.log_tail}".lower()


def _choice(playbook: str, params: dict, reasoning: str) -> PlaybookChoice:
    return PlaybookChoice(playbook, params, 1.0, reasoning, "rules")


def _is_mac_host(name: str) -> bool:
    from autonomy.playbooks import is_mac_host
    return is_mac_host(name)


def _rule_disk_full_mac(ctx: ErrorContext) -> Optional[PlaybookChoice]:
    """Rule 1: disk full on a Mac (Errno 28 / no space left on device)."""
    text = _text(ctx)
    if "errno 28" not in text and "no space left on device" not in text:
        return None

    for host, health in (ctx.fleet_snapshot or {}).items():
        if not _is_mac_host(host):
            continue
        disk_free = health.get("disk_free_gb") if isinstance(health, dict) else None
        if disk_free is not None and disk_free < 5:
            return _choice(
                "clean_fits_cache",
                {"host": host},
                f"disk full (Errno 28) and {host} has disk_free_gb={disk_free} < 5",
            )

    return _choice(
        "pause_and_alert",
        {"reason": "disk full but host undetermined"},
        "Errno 28 / no space left detected but no identifiable mac host with critical disk",
    )


def _rule_stream_drift(ctx: ErrorContext) -> Optional[PlaybookChoice]:
    """Rule 2: stream drift in SCANNING or FPP_BACKFILL."""
    if ctx.state not in ("SCANNING", "FPP_BACKFILL"):
        return None

    text = _text(ctx)
    matched = (
        "workers idle" in text
        or ("pending" in text and "rate" in text and "0/min" in text)
        or "stream drift" in text
    )
    if not matched:
        return None

    stage = "primary" if ctx.state == "SCANNING" else "fpp"
    return _choice(
        "rebuild_streams",
        {"stage": stage},
        f"stream drift detected in state {ctx.state}",
    )


def _rule_orphans(ctx: ErrorContext) -> Optional[PlaybookChoice]:
    """Rule 3: orphaned tasks (assigned without heartbeat).

    The "orphan" term is negation-aware and localized: first the sentences
    containing "no orphan(s)" or "0 orphan(s)" are removed from the text (e.g.
    a check log reporting there are none); only then is it
    checked whether a real positive mention remains. A negation in one
    sentence does not suppress a positive mention in another sentence of the same
    text. The independent "assigned"+"heartbeat" pattern is evaluated separately.
    """
    text = _text(ctx)
    cleaned = _ORPHAN_NEGATION_RE.sub("", text)
    orphan_signal = "orphan" in cleaned
    assigned_heartbeat_signal = "assigned" in text and "heartbeat" in text
    if orphan_signal or assigned_heartbeat_signal:
        return _choice(
            "requeue_orphans", {}, "orphaned tasks detected (assigned without heartbeat)"
        )
    return None


def _rule_wal(ctx: ErrorContext) -> Optional[PlaybookChoice]:
    """Rule 4: WAL bloat / locked / pending checkpoint."""
    text = _text(ctx)
    if "wal" in text and ("bloat" in text or "locked" in text or "checkpoint" in text):
        return _choice("wal_checkpoint", {}, "WAL bloat/locked/checkpoint detected")
    return None


def _rule_transient_network(ctx: ErrorContext) -> Optional[PlaybookChoice]:
    """Rule 5: transient network/API error — anti-loop guard via recent_actions."""
    text = _text(ctx)
    if not any(trigger in text for trigger in _TRANSIENT_NETWORK_TRIGGERS):
        return None
    if "resume_run" in (ctx.recent_actions or []):
        return None
    return _choice(
        "resume_run",
        {"run_id": ctx.run_id},
        "transient network/API error detected, resume_run not yet attempted",
    )


def _rule_ram_critical(ctx: ErrorContext) -> Optional[PlaybookChoice]:
    """Rule 6: critical RAM on one host, rest of the fleet healthy."""
    critical = [
        (host, health.get("ram_avail_gb"))
        for host, health in (ctx.fleet_snapshot or {}).items()
        if isinstance(health, dict)
        and health.get("ram_avail_gb") is not None
        and health["ram_avail_gb"] < 1.0
    ]
    if len(critical) != 1:
        return None

    host, ram_avail = critical[0]
    return _choice(
        "restart_worker",
        {"worker": host},
        f"critical RAM on {host} (ram_avail_gb={ram_avail} < 1.0), rest of the fleet healthy",
    )


RULES: list[tuple[str, Callable[[ErrorContext], Optional[PlaybookChoice]]]] = [
    ("disk_full_mac", _rule_disk_full_mac),
    ("stream_drift", _rule_stream_drift),
    ("orphans", _rule_orphans),
    ("wal", _rule_wal),
    ("transient_network", _rule_transient_network),
    ("ram_critical", _rule_ram_critical),
]


def classify(ctx: ErrorContext) -> Optional[PlaybookChoice]:
    """Apply the v1 rules in order; return the first match or None."""
    for _name, rule in RULES:
        choice = rule(ctx)
        if choice is not None:
            return choice
    return None
