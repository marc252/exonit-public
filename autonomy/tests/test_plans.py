import textwrap
from autonomy.plans import load_plans, current_plan, needs_planning, continuity_decision, plan_progress

YAML = textwrap.dedent("""
master_list: [15, 16, 17]
plans:
  - plan_id: p1
    approved_by: operator
    sector: 15
    mode: survey
    lot_size: 300000
  - plan_id: p2
    sector: 16
    mode: survey
    lot_size: 300000
""")

def _pf(tmp_path, text=YAML):
    p = tmp_path / "plans.yaml"; p.write_text(text)
    return load_plans(p)

def test_unapproved_never_executable(tmp_path):
    pf = _pf(tmp_path)
    assert current_plan(pf, set()).plan_id == "p1"
    assert current_plan(pf, {"p1"}) is None  # p2 not approved: NOT returned

def test_needs_planning_at_75pct_without_successor(tmp_path):
    pf = _pf(tmp_path)
    plan = current_plan(pf, set())
    assert needs_planning(pf, plan, progress=0.80, lots_left=None) is True
    assert needs_planning(pf, plan, progress=0.50, lots_left=None) is False
    assert needs_planning(pf, plan, progress=0.50, lots_left=1) is True

def test_needs_planning_false_with_approved_successor(tmp_path):
    pf = _pf(tmp_path, YAML.replace("  - plan_id: p2", "  - plan_id: p2\n    approved_by: operator"))
    plan = current_plan(pf, set())
    assert needs_planning(pf, plan, progress=0.90, lots_left=0) is False

def test_continuity_ladder():
    d = continuity_decision([15, 16], {15}, lambda s: True, [14], {14: 0.6})
    assert d.kind == "survey" and d.sector == 16
    d = continuity_decision([15], {15}, lambda s: False, [14, 13], {14: 0.6})
    assert d.kind == "deep_rescan" and d.sector == 13  # 13 has no deep coverage (0.0)
    d = continuity_decision([], set(), lambda s: False, [], {})
    assert d.kind == "pause"

def test_progress_prefers_tics():
    from autonomy.plans import Plan
    p = Plan(plan_id="x", approved_by="operator", sector=15, mode="survey", worklist=None,
             lot_size=100, max_lots=10, fleet_profile="full", notes="", total_tics=1000)
    assert plan_progress(p, tics_done=750, lots_done=1) == 0.75
