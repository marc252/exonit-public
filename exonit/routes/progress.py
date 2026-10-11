"""POST /api/progress — progress marks from long-running pipeline scripts.

Kept out of routes/discovery.py on purpose: that module is already ~2.5k lines
and this endpoint has nothing to do with the discovery-campaign logic. It is
the write half of exonit/progress.py; the read half is served by
GET /api/discovery/pipeline.

Auth: the same worker bearer token every other machine-to-machine endpoint
uses, so a script on <host> authenticates exactly like a worker does.
"""

import logging

from fastapi import Request
from fastapi.responses import JSONResponse

from exonit import progress as progress_model
from exonit.routes.deps import ServerDeps
from exonit.shared_state import get_shared_state

logger = logging.getLogger(__name__)


def register(app, deps: ServerDeps) -> None:
    _check_auth = deps.check_auth

    @app.post("/api/progress")
    async def api_progress(request: Request):
        """Record one progress mark. Cheap, idempotent, best-effort.

        Never lets a storage problem become the caller's problem: a mark that
        cannot be stored still answers 200, because the emitter must not slow
        down or fail over telemetry. Only a malformed payload is a 400.
        """
        if not _check_auth(request):
            return JSONResponse({"error": "unauthorized"}, status_code=401)
        try:
            payload = await request.json()
        except Exception:
            return JSONResponse({"error": "invalid json"}, status_code=400)

        try:
            progress_model.record_mark(get_shared_state(), payload)
        except progress_model.ProgressError as exc:
            logger.info("rejected progress mark: %s", exc)
            return JSONResponse({"error": str(exc)}, status_code=400)
        except Exception as exc:
            logger.warning("progress mark not stored: %s", exc)
        return {"status": "ok"}
