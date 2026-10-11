"""Shared per-app-instance objects passed to each routes/*.register(app, deps).

``create_app()`` builds exactly one ``ServerDeps`` and passes the SAME
instance to every domain module's ``register(app, deps)`` call. This keeps
create_app() the single place that decides what is per-process (a fresh
object built on every ``create_app()`` call — correct under uvicorn
``--workers N``, A3 multi-process: each forked process calls create_app()
independently and gets its own ``db``/``workers_state``/etc.) versus what is
cross-process shared (``workers_state`` wraps ``get_shared_state()``, which is
Redis-backed in production — the WorkerRegistry object itself is still
per-process, but the data it reads/writes is shared).

Only fields referenced by more than one domain module live here (``db``,
``cfg``, ``berger``, ``workers_state``, plus the two auth/cache closures).
Fields used by exactly one domain (e.g. the pipeline Popen-handle dict, or
per-domain Redis-key constants) are defined locally inside that domain
module's ``register()`` instead — see
progress/impl-createapp-split-2026-07-02.md for the full rationale.
"""

from dataclasses import dataclass
from typing import Callable

from fastapi import Request

from exonit.berger_catalog import BergerCatalog
from exonit.config import ExoNitConfig
from exonit.db import ExoNitDB
from exonit.worker_registry import WorkerRegistry


@dataclass
class ServerDeps:
    """Shared objects built once per ``create_app()`` call."""

    db: ExoNitDB
    db_path: str
    cfg: ExoNitConfig
    reports_dir: str
    berger: BergerCatalog
    workers_state: WorkerRegistry
    # Bearer-token guard (closes over cfg.worker_token). Used by every
    # domain that serves worker-authenticated endpoints.
    check_auth: Callable[[Request], bool]
    # Redis-backed (shared_state) response cache — see server._cached_json's
    # own docstring for the key/TTL contract. Used by admin/candidates/
    # workers/discovery for expensive aggregate GETs.
    cached_json: Callable[[str, float, Callable[[], dict]], dict]
