"""AWS control-plane endpoints (fleet start/stop, server start/stop, repatriate) (Fase D split).

Extracted verbatim from exonit.server.create_app() — see
docs/reviews/2026-07-02-opus48-cleanup-review.md and
progress/impl-createapp-split-2026-07-02.md for the module map. Endpoint
bodies below are byte-identical to the pre-split code (git move-detection
should match them); only this file's import/register() preamble is new.

Covers: /api/deployment, /api/aws/workers/start(+terminate),
/api/aws/fleet/status, /api/aws/server/start(+stop), /api/aws/status,
/api/aws/repatriate(+status).
"""

import os
import subprocess
import time as _time
from pathlib import Path

from fastapi import Request
from fastapi.responses import JSONResponse

from exonit.procutil import reap_in_background as _reap_in_background
from exonit.server import (
    WORKER_ALIVE_WINDOW_SECONDS,
    _PROJECT_ROOT,
    _aws_try,
    _require_deployment,
)
from exonit.routes.deps import ServerDeps
from exonit.shared_state import get_shared_state

# Discovery registry path — duplicated as a plain literal; see discovery.py's
# identical comment (not currently patchable via exonit.server, so a local
# literal here is behavior-identical to the pre-split create_app()-local).
_DISCOVERY_REGISTRY = str(_PROJECT_ROOT / "data" / "discovery" / "sectors.json")


def register(app, deps: ServerDeps) -> None:
    cfg = deps.cfg
    workers_state = deps.workers_state
    _check_auth = deps.check_auth

    # Redis / shared-state key for repatriate process metadata.
    _REPATRIATE_STATE_KEY = "aws:repatriate:state"
    _REPATRIATE_FIELD = "current"

    @app.get("/api/deployment")
    async def api_deployment():
        """Expose deployment role + active sector so the UI can gate panels.

        Home shows the AWS control panel; an AWS deployment shows the fleet/scan
        controls. Single codebase — the only difference is config.
        """
        from exonit.config import deployment as _deployment
        from exonit.discovery_state import active_sector
        return {
            "deployment": _deployment(),
            "active_sector": active_sector(_DISCOVERY_REGISTRY),
        }

    # --- AWS control plane (Component 3) ----------------------------- #
    # Single codebase: each endpoint acts only in its matching deployment.
    # AWS-side: raise/stop the GPU worker fleet (boto3 CreateFleet). Home-side:
    # manage the AWS *server* instance lifecycle (start/stop, never terminate —
    # it holds the DB/FITS volume). Mutating endpoints require the Bearer token.
    def _aws_registered_workers() -> int:
        now = _time.time()
        return sum(1 for w in workers_state.values()
                   if now - w.get("last_heartbeat", 0) < WORKER_ALIVE_WINDOW_SECONDS)

    @app.post("/api/aws/workers/start")
    async def api_aws_workers_start(request: Request):
        guard = _require_deployment("aws")
        if guard is not None:
            return guard
        if not _check_auth(request):
            return JSONResponse({"error": "unauthorized"}, status_code=401)
        try:
            body = await request.json()
        except Exception:
            body = {}
        n = int(body.get("count") or os.environ.get("EXONIT_AWS_WORKER_COUNT", "12"))
        resp, err = _aws_try("create_fleet", n)
        if err is not None:
            return err
        return {"status": "starting", "requested": n,
                "fleet_id": resp.get("FleetId")}

    @app.post("/api/aws/workers/terminate")
    async def api_aws_workers_terminate(request: Request):
        guard = _require_deployment("aws")
        if guard is not None:
            return guard
        if not _check_auth(request):
            return JSONResponse({"error": "unauthorized"}, status_code=401)
        result, err = _aws_try("terminate_fleet", action_label="terminate")
        if err is not None:
            return err
        return result

    @app.get("/api/aws/fleet/status")
    async def api_aws_fleet_status():
        guard = _require_deployment("aws")
        if guard is not None:
            return guard
        from exonit import aws_control
        try:
            fs = aws_control.fleet_status()
        except Exception as exc:
            fs = {"running": 0, "pending": 0, "instances": [], "error": str(exc)}
        fs["registered"] = _aws_registered_workers()
        return fs

    @app.post("/api/aws/server/start")
    async def api_aws_server_start(request: Request):
        guard = _require_deployment("home")
        if guard is not None:
            return guard
        result, err = _aws_try("server_start")
        if err is not None:
            return err
        return {"status": "starting"}

    @app.post("/api/aws/server/stop")
    async def api_aws_server_stop(request: Request):
        guard = _require_deployment("home")
        if guard is not None:
            return guard
        result, err = _aws_try("server_stop")
        if err is not None:
            return err
        return {"status": "stopping"}

    @app.get("/api/aws/status")
    async def api_aws_status():
        guard = _require_deployment("home")
        if guard is not None:
            return guard
        from exonit import aws_control
        out = {"server": {"state": "unknown", "public_ip": None},
               "fleet": {"running": 0, "pending": 0}, "aws_ui_url": None}
        try:
            out["server"] = aws_control.server_status()
        except Exception as exc:
            out["server"] = {"state": "error", "public_ip": None, "detail": str(exc)}
        try:
            out["fleet"] = aws_control.fleet_status()
        except Exception as exc:
            out["fleet"]["error"] = str(exc)
        srv_running = 1 if out["server"].get("state") == "running" else 0
        out["machines_running"] = (srv_running + out["fleet"].get("running", 0)
                                   + out["fleet"].get("pending", 0))
        ip = out["server"].get("public_ip")
        if out["server"].get("state") == "running" and ip:
            # Inject the AWS deployment's token so the operator doesn't type it.
            aws_token = os.environ.get("EXONIT_AWS_WORKER_TOKEN", cfg.worker_token)
            # 8400 = the AWS server's port (fixed in its systemd unit), not
            # necessarily this (home) server's cfg.port.
            out["aws_ui_url"] = f"http://{ip}:8400/?token={aws_token}"
        return out

    @app.post("/api/aws/repatriate")
    async def api_aws_repatriate(request: Request):
        """Home-only: fetch a sector-N run bundle from S3, merge it, run LoRA.

        Fire-and-forget: launches scripts/repatriate_and_lora.sh detached and
        returns immediately. The script does s3 sync -> import -> scorecards ->
        LoRA on <host> via gpuq.
        """
        from exonit.config import deployment
        if deployment() != "home":
            return JSONResponse({"error": "not_home_deployment"}, status_code=403)
        try:
            body = await request.json()
        except Exception:
            body = {}
        s3_uri = body.get("s3_uri") or os.environ.get("EXONIT_AWS_S3_PREFIX")
        sector = int(body.get("sector") or 2)
        if not s3_uri:
            return JSONResponse({"error": "no_s3_uri",
                "detail": "No S3 URI provided and EXONIT_AWS_S3_PREFIX is not set on the home "
                          "server. Paste the run's s3:// URI in the field, or set the env var."},
                status_code=400)
        if not str(s3_uri).startswith("s3://"):
            return JSONResponse({"error": "bad_s3_uri",
                "detail": f"S3 URI must start with s3:// (got: {s3_uri!r})"}, status_code=400)
        script = str(_PROJECT_ROOT / "scripts" / "repatriate_and_lora.sh")
        log_path = _PROJECT_ROOT / "data" / "discovery" / "repatriate.log"
        try:
            # Open log file (truncate per launch); fall back to DEVNULL on error
            # so a bad filesystem state can never 500 the endpoint.
            try:
                log_fh = open(log_path, "w")
            except Exception:
                log_fh = subprocess.DEVNULL
            proc = subprocess.Popen(
                ["bash", script, s3_uri, str(sector)],
                stdout=log_fh, stderr=log_fh,
                start_new_session=True, close_fds=True)
            # Zombie-leak fix (2026-07-10 review F3): nothing else here ever
            # poll()s/wait()s this detached child — without this, a finished
            # repatriate run lingers as a <defunct> until the server
            # restarts, same shape as the discovery orchestrator fix
            # (2026-07-07). See exonit/procutil.py's docstring.
            _reap_in_background(proc)
            log_key = str(log_path) if log_fh is not subprocess.DEVNULL else None
            # Persist cross-process metadata so all uvicorn workers can report
            # status via GET /api/aws/repatriate/status (A3 multi-process pattern,
            # same as download-status, commit 57896366).
            get_shared_state().hset_json(_REPATRIATE_STATE_KEY, _REPATRIATE_FIELD, {
                "pid": proc.pid,
                "log_path": log_key,
                "started_at": _time.time(),
            })
            return {"status": "started", "pid": proc.pid,
                    "s3_uri": s3_uri, "sector": sector,
                    "log": log_key}
        except Exception as exc:
            return JSONResponse(
                {"error": "repatriate_failed", "detail": str(exc)}, status_code=500)

    @app.get("/api/aws/repatriate/status")
    def api_aws_repatriate_status():
        """Report status of the background repatriate process.

        Reads cross-process metadata (pid, log_path, started_at) from
        shared_state so all uvicorn workers return the correct state.

        Returns:
            running  — pid is alive (best-effort; uses psutil or os.kill(0))
            done     — log contains "DONE:" marker
            warn     — log contains "WARN" marker
            started_at — unix timestamp when last repatriation was launched
            log_tail   — last ~60 lines of the repatriate.log, or null
        """
        _st = get_shared_state()
        meta = _st.hget_json(_REPATRIATE_STATE_KEY, _REPATRIATE_FIELD)

        if meta is None:
            return {"running": False, "never_run": True,
                    "done": False, "warn": False,
                    "started_at": None, "log_tail": None}

        pid = meta.get("pid")
        log_path = meta.get("log_path")
        started_at = meta.get("started_at")

        # Derive running from whether the pid is alive.
        running = False
        if pid is not None:
            try:
                import psutil as _psutil
                running = _psutil.pid_exists(pid)
                if running:
                    # Best-effort PID-recycling guard: the repatriate script is a
                    # bash wrapper, so we cannot check cmdline reliably on all
                    # platforms.  Accept pid_exists as sufficient here.
                    pass
            except ImportError:
                try:
                    os.kill(pid, 0)
                    running = True
                except (OSError, ProcessLookupError):
                    running = False

        # Parse log tail for done/warn markers and last lines.
        log_tail = None
        done = False
        warn = False
        if log_path:
            try:
                text = Path(log_path).read_text(errors="replace")
                lines = text.splitlines()
                log_tail = "\n".join(lines[-60:]) if lines else ""
                done = "DONE:" in text
                warn = "WARN" in text
            except Exception:
                pass

        return {
            "running": running,
            "never_run": False,
            "done": done,
            "warn": warn,
            "started_at": started_at,
            "log_tail": log_tail,
        }

