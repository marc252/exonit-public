"""Plan queue and continuity policy (phase 1 — shadow mode).

`plans.yaml` is the human source of truth of the autonomous loop: an
ordered queue of plans, not a single plan (spec §4.1). This module only reads
and interprets the queue — it touches neither the infra nor the server.

A plan without `approved_by` is loaded anyway (no error is raised) but
is never returned as executable from `current_plan`: the supervisor does not
execute anything the operator has not explicitly approved.

When the queue runs out with no pending approved plan, `continuity_decision`
applies the escalation approved in §4.6 so the system never stops to wait for
a decision.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import yaml


@dataclass
class Plan:
    plan_id: str
    approved_by: Optional[str]
    sector: int
    mode: str  # survey | deep_rescan
    worklist: Optional[str] = None
    lot_size: int = 300_000
    max_lots: Optional[int] = None
    fleet_profile: str = "full"
    notes: str = ""
    total_tics: Optional[int] = None  # filled in by the supervisor once known

    @property
    def approved(self) -> bool:
        return bool(self.approved_by)


@dataclass
class PlansFile:
    plans: list
    master_list: list


@dataclass
class ContinuityDecision:
    kind: str  # survey | deep_rescan | pause
    sector: Optional[int]
    reason: str


def load_plans(path) -> PlansFile:
    """Read `plans.yaml`. Plans without `approved_by` are loaded anyway
    (Plan.approved stays False) — they never raise an error on load."""
    path = Path(path)
    with path.open("r") as f:
        raw = yaml.safe_load(f) or {}

    master_list = list(raw.get("master_list") or [])
    plans = []
    for entry in raw.get("plans") or []:
        plans.append(
            Plan(
                plan_id=entry["plan_id"],
                approved_by=entry.get("approved_by"),
                sector=entry["sector"],
                mode=entry["mode"],
                worklist=entry.get("worklist"),
                lot_size=entry.get("lot_size", 300_000),
                max_lots=entry.get("max_lots"),
                fleet_profile=entry.get("fleet_profile", "full"),
                notes=entry.get("notes", ""),
                total_tics=entry.get("total_tics"),
            )
        )
    return PlansFile(plans=plans, master_list=master_list)


def current_plan(pf: PlansFile, completed_plan_ids: set) -> Optional[Plan]:
    """First approved plan not yet completed.

    The queue is sequential (§4.1): already-completed plans are skipped, but if
    the first non-completed plan is not yet approved, the queue is
    blocked here (a later, already-approved plan is not jumped to out of
    order) and None is returned.
    """
    for plan in pf.plans:
        if plan.plan_id in completed_plan_ids:
            continue
        return plan if plan.approved else None
    return None


def plan_progress(plan: Plan, tics_done: int, lots_done: int) -> float:
    """Progress 0..1 of the current plan: prefers tics_done/total_tics; if
    total_tics is unknown, falls back to lots_done/max_lots; otherwise 0.0."""
    if plan.total_tics:
        return tics_done / plan.total_tics
    if plan.max_lots:
        return lots_done / plan.max_lots
    return 0.0


def needs_planning(pf: PlansFile, plan: Plan, progress: float, lots_left: Optional[int]) -> bool:
    """True if the current plan reaches ~75% (or <=2 lots remain) and the queue
    has NO approved plan after the current one (early-planning
    trigger, §4.1)."""
    trigger = progress >= 0.75 or (lots_left is not None and lots_left <= 2)
    if not trigger:
        return False

    idx = next(i for i, p in enumerate(pf.plans) if p.plan_id == plan.plan_id)
    has_approved_successor = any(p.approved for p in pf.plans[idx + 1:])
    return not has_approved_successor


def continuity_decision(
    master_list: list,
    completed_sectors: set,
    space_ok: Callable[[int], bool],
    sectors_with_data: list,
    deep_coverage: dict,
) -> ContinuityDecision:
    """Continuity escalation when the plan queue runs out with no approved
    successor (spec §4.6) — the system never stops to wait for a decision:

    1. First non-completed sector of `master_list` that passes the space
       gate (`space_ok`).
    2. If none: `deep_rescan` of the sector in `sectors_with_data` with the lowest
       deep coverage (0.0 by default if absent from `deep_coverage`).
    3. If that is not possible either: `pause_and_alert`.
    """
    for sector in master_list:
        if sector in completed_sectors:
            continue
        if space_ok(sector):
            return ContinuityDecision(
                kind="survey",
                sector=sector,
                reason=f"next master_list sector with available space: {sector}",
            )

    if sectors_with_data:
        sector = min(sectors_with_data, key=lambda s: deep_coverage.get(s, 0.0))
        return ContinuityDecision(
            kind="deep_rescan",
            sector=sector,
            reason=(
                f"no new sector with space; deep rescan of the sector with the lowest "
                f"deep coverage: {sector} ({deep_coverage.get(sector, 0.0)})"
            ),
        )

    return ContinuityDecision(
        kind="pause",
        sector=None,
        reason="no new sector with space nor a sector with data for deep rescan",
    )
