"""HTTP client for the ExoNit server (phase 1/1.5 — shadow READ-ONLY, armed with POST).

`ServerClient` is the supervisor's only entry point to the real server.
`run_shadow_loop` (phase 1, UNTOUCHED) ALWAYS builds the client with
`read_only=True` — no POST ever leaves in shadow mode, `.post()` raises
`RuntimeError` before touching the network. `run_armed_loop` (phase 1.5, see
`autonomy/supervisor.py`) only builds a `read_only=False` client
after validating the double arming key (`_is_armed`).

Review round 1 (2026-08-09): a real non-2xx HTTP response (e.g.
`/api/discovery/scan` returning 400/409/500 with a structured JSON body,
see `exonit/routes/discovery.py`) is now a `ServerHTTPError` — a subclass of
`RuntimeError` that keeps `status` and `body` (the parsed JSON,
best-effort) — instead of a bare `RuntimeError` that discarded the body.
Any caller that only did `except RuntimeError` keeps working
as before (same hierarchy); those that need the real detail (e.g.
`autonomy/supervisor.py::_interpret_post_error`, to tell
`already_running` from `need_download`, both HTTP 409) now have
access to it. A network failure/timeout (no HTTP response, a `URLError` that
is NOT an `HTTPError`) is still a bare `RuntimeError` — there is no body to
keep.

Stdlib only (`urllib`) — no new dependencies.
"""

from __future__ import annotations

import json
from typing import Optional
from urllib import error, request

_DEFAULT_TIMEOUT_S = 30
_DEFAULT_POST_TIMEOUT_S = 900  # the scan POST can take minutes (synchronous 300k enqueue)


class ServerHTTPError(RuntimeError):
    """Non-2xx HTTP response with the JSON body preserved (best-effort).

    `status` is the real HTTP code (e.g. 409, 400, 500); `body` is the JSON
    parsed from the error response if there was one and it was valid, else `{}`
    (the ExoNit server always answers errors with structured JSON —
    `exonit/routes/discovery.py` — but this client never assumes anything about the
    remote format without checking)."""

    def __init__(self, method: str, path: str, status: int, body: Optional[dict]):
        self.status = status
        self.body: dict = body or {}
        super().__init__(f"{method} {path} failed: HTTP {status} {self.body}".rstrip())


class ServerClient:
    """Minimal client (GET/POST JSON) against `base_url`, with a read-only guard."""

    def __init__(self, base_url: str, read_only: bool = True, timeout_s: int = _DEFAULT_TIMEOUT_S):
        self.base_url = base_url.rstrip("/")
        self.read_only = read_only
        self.timeout_s = timeout_s

    @staticmethod
    def _to_http_error(method: str, path: str, exc: "error.HTTPError") -> ServerHTTPError:
        body: Optional[dict] = None
        try:
            raw = exc.read()
            if raw:
                parsed = json.loads(raw)
                if isinstance(parsed, dict):
                    body = parsed
        except Exception:  # noqa: BLE001 - body missing/non-JSON: stays {} (best-effort)
            body = None
        return ServerHTTPError(method, path, getattr(exc, "code", 0) or 0, body)

    def get(self, path: str) -> dict:
        """GET `path` and parse JSON. Raises (network/HTTP/JSON) — the caller decides
        what to do with the failure (shadow mode: `Decision("pause", ...)`, see supervisor.py).
        A non-2xx response raises `ServerHTTPError` (subclass of `RuntimeError`,
        body preserved); a network failure/timeout, a bare `RuntimeError`."""
        url = f"{self.base_url}{path}"
        req = request.Request(url, method="GET")
        try:
            with request.urlopen(req, timeout=self.timeout_s) as resp:
                raw = resp.read()
        except error.HTTPError as exc:
            raise self._to_http_error("GET", path, exc) from exc
        except error.URLError as exc:
            raise RuntimeError(f"GET {path} failed: {exc}") from exc
        return json.loads(raw)

    def post(self, path: str, payload: dict, timeout_s: int = _DEFAULT_POST_TIMEOUT_S) -> dict:
        """POST `path` with a JSON `payload`. **Raises `RuntimeError` without touching the
        network if `read_only=True`** — shadow mode always builds the client that way.

        `timeout_s` (default 900s) is independent of the constructor's
        `timeout_s` (which still governs `get()`): the synchronous enqueue of a
        300k-TIC lot at `/api/discovery/scan` can take minutes, much more
        than any pipeline read GET.

        A non-2xx response raises `ServerHTTPError` (subclass of
        `RuntimeError`, `status`/`body` preserved — see the module
        docstring); a network failure/timeout with no HTTP response,
        a bare `RuntimeError` as before."""
        if self.read_only:
            raise RuntimeError("client read-only")

        url = f"{self.base_url}{path}"
        body = json.dumps(payload).encode()
        req = request.Request(
            url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=timeout_s) as resp:
                raw = resp.read()
        except error.HTTPError as exc:
            raise self._to_http_error("POST", path, exc) from exc
        except error.URLError as exc:
            raise RuntimeError(f"POST {path} failed: {exc}") from exc
        return json.loads(raw)
