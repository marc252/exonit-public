#!/usr/bin/env python3
"""ExoNit operations health check.

Single-shot CLI that hits ``/api/health/work`` on the local server and
prints a human-readable summary. Exit code:

    0  every queue is empty / no warnings / no criticals
    1  warnings only (e.g. residual backfill, low disk)
    2  critical issues (duplicate TICs, disk near full, workers stuck)

Use it before declaring a long-running batch "done":

    python scripts/run_health_check.py

The numbers are the same shown on the dashboard's Operations Health card.
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request


def _color(s: str, code: str) -> str:
    if not sys.stdout.isatty():
        return s
    return f"\033[{code}m{s}\033[0m"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default="http://localhost:8400")
    ap.add_argument("--json", action="store_true",
                    help="Emit raw JSON instead of human-readable output")
    ap.add_argument("--quiet", action="store_true",
                    help="Only print problems (warnings/critical)")
    args = ap.parse_args()

    url = f"{args.server}/api/health/work"
    try:
        with urllib.request.urlopen(url, timeout=10) as r:
            data = json.loads(r.read().decode())
    except Exception as e:
        print(f"[health] FAILED to reach {url}: {e}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(data, indent=2))
    else:
        status = data["status"]
        status_lbl = {"ok": _color("OK", "32"),
                      "warning": _color("WARNING", "33"),
                      "critical": _color("CRITICAL", "31")}.get(status, status)
        if not args.quiet or status != "ok":
            print(f"=== ExoNit health: {status_lbl} ===")
            t = data["primary_tasks"]
            print(f"  primary tasks: {t['pending']} pending  "
                  f"{t['assigned']} assigned  "
                  f"{t['completed']} completed  {t['failed']} failed")
            f = data["fpp_backfill"]
            print(f"  FPP backfill : {f['actionable']} actionable  "
                  f"{f['unrecoverable']} unrecoverable  "
                  f"{f['currently_assigned']} currently assigned")
            ex = data["exominer_backfill"]
            print(f"  ExoMiner backfill: {ex['pending']} pending  "
                  f"{ex['currently_assigned']} currently assigned")
            l = data["lora_explanation"]
            print(f"  LoRA explanations: {l['missing']} missing  "
                  f"{l['stale']} stale (legacy text)")
            w = data["workers"]
            print(f"  workers      : {w['alive']} alive  "
                  f"{len(w['stuck'])} stuck  "
                  f"{len(w['duplicate_tics'])} duplicate-TIC")
            d = data["disk"]
            print(f"  disk         : {d['free_gb']} GB free "
                  f"({d['pct_used']}% used)")
        if data.get("critical"):
            print(_color("CRITICAL:", "31") + " " + ", ".join(data["critical"]))
        if data.get("warnings"):
            print(_color("WARNING:", "33") + " " + ", ".join(data["warnings"]))
        for s in data["workers"].get("stuck", []):
            print(_color("STUCK:", "33")
                  + f" {s['name']} on TIC {s['tic_id']} for "
                  f"{s['stuck_seconds']//60}m")
        for tic, names in data["workers"].get("duplicate_tics", {}).items():
            print(_color("DUPLICATE:", "31")
                  + f" TIC {tic} processed by {' + '.join(names)}")

    return {"ok": 0, "warning": 1, "critical": 2}.get(data["status"], 0)


if __name__ == "__main__":
    sys.exit(main())
