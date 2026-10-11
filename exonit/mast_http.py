"""Bound the two failure modes that made the SPOC-gate MAST calls hang
forever (diagnosed 2026-08-03 during a real MAST outage):

  1. astroquery's ``conf.timeout`` (default 600s) is only checked BETWEEN
     pages of its long-polling loop — a single ``requests`` socket read that
     never returns (MAST accepts the TCP connection but never sends bytes)
     blocks under that timeout forever. ``install_mast_session_timeout``
     patches the underlying ``requests.Session`` so every HTTP call carries
     a real socket-level timeout, closing that hole at the transport layer.

  2. Even with (1) fixed, a caller still has to actually WAIT for that
     timeout inline in a background thread that a bounded ThreadPoolExecutor
     owns — and _shutdown_spoc_executor_fast (server.py, 2026-08-01) already
     proved that concurrent.futures' atexit hook joins that pool's threads
     no matter what, keeping a "shut down" uvicorn worker alive so long as
     even one thread is still running. ``call_with_deadline`` runs the MAST
     call in its own daemon thread instead, so a stuck call can never be the
     thing keeping the process alive — the caller gets an upper bound on how
     long it waits, and the abandoned thread dies with the process instead
     of blocking its exit.

Both are fail-soft by construction: neither function raises for a
misconfigured environment variable, only for the underlying call actually
failing/timing out — mirroring the fail-soft contract documented on
_run_spoc_check_background (server.py).
"""
from __future__ import annotations

import logging
import os
import threading
from typing import Any, Callable, TypeVar

logger = logging.getLogger("exonit.mast_http")

T = TypeVar("T")

_DEFAULT_MAST_HTTP_TIMEOUT = 60.0
_DEFAULT_SPOC_CHECK_DEADLINE = 180.0


def _read_float_env(name: str, default: float) -> float | None:
    """Parse a float env var; ``None`` means "do not install/wrap".

    A value that parses to <= 0 is treated the same as "disabled" — an
    explicit way to opt out without unsetting the variable. A value that
    fails to parse is logged as a warning and treated as disabled too
    (fail-soft: a typo in an env var must never crash the server, just
    leave the old unbounded behaviour in place for that one knob).
    """
    raw = os.environ.get(name)
    if raw is None:
        value = default
    else:
        try:
            value = float(raw)
        except ValueError:
            logger.warning(
                "%s=%r is not a valid float — ignoring (feature disabled)",
                name, raw)
            return None
    if value <= 0:
        return None
    return value


def install_mast_session_timeout(timeout: float | None = None) -> None:
    """Patch astroquery's shared MAST session so every request carries a
    real socket timeout, even calls astroquery itself never passes one for.

    ``timeout`` overrides ``EXONIT_MAST_HTTP_TIMEOUT`` (seconds, default 60).
    Either resolving to <= 0 (or unparsable) disables the patch entirely —
    the caller then gets astroquery's original, timeout-less behaviour.

    Wraps ``Observations._portal_api_connection._session.request`` at the
    INSTANCE level (not the class), so only this one process-wide singleton
    session is affected — never every requests.Session in the interpreter.
    The wrapper only fills in ``timeout`` when the caller didn't already
    pass one, so any future astroquery internal call that legitimately sets
    its own timeout is left alone.

    Idempotent: marks the session with ``_exonit_timeout_installed`` and
    returns immediately on a second call, so callers (e.g. every worker
    process's lazy ``_get_spoc_executor()`` init) can call this
    unconditionally without stacking wrappers.

    The wrapped session is shared beyond the SPOC-gate: it is the same
    singleton ``Observations`` session used by ``download_products`` (see
    exonit/exominer_native/fetch_mast_dv.py, reached via routes/proxies.py).
    That is safe because requests' ``timeout`` is per-socket-read, not a
    total-duration limit — a slow but steadily streaming DV download is
    never cut off, only a connection that has genuinely gone silent.

    Import of astroquery is lazy and the whole function is fail-soft: if
    astroquery is missing, or a future astroquery release changes this
    internal structure, this logs a warning and returns — it must never be
    the reason the server fails to start or a request handler raises.
    """
    resolved = (
        timeout if timeout is not None
        else _read_float_env("EXONIT_MAST_HTTP_TIMEOUT", _DEFAULT_MAST_HTTP_TIMEOUT)
    )
    if resolved is None:
        return

    try:
        from astroquery.mast import Observations
        session = Observations._portal_api_connection._session
    except Exception as exc:
        logger.warning(
            "install_mast_session_timeout: astroquery internals not "
            "available (%s: %s) — MAST calls remain unbounded at the "
            "socket layer", type(exc).__name__, exc)
        return

    if getattr(session, "_exonit_timeout_installed", False):
        return

    original_request = session.request

    def _request_with_timeout(method, url, *args, **kwargs):
        if kwargs.get("timeout") is None:
            kwargs["timeout"] = resolved
        return original_request(method, url, *args, **kwargs)

    session.request = _request_with_timeout
    session._exonit_timeout_installed = True
    logger.info(
        "install_mast_session_timeout: astroquery MAST session now has a "
        "%.0fs default socket timeout", resolved)


def call_with_deadline(
    fn: Callable[..., T],
    *args: Any,
    deadline_s: float | None = None,
    name: str = "mast-call",
    **kwargs: Any,
) -> T:
    """Run ``fn(*args, **kwargs)`` on a daemon thread, bounded by a deadline.

    ``deadline_s`` overrides ``EXONIT_SPOC_CHECK_DEADLINE`` (seconds,
    default 180). A resolved deadline <= 0 disables the wrapper entirely —
    ``fn`` runs directly on the caller's thread, no thread is spawned.

    Deliberately NOT a ThreadPoolExecutor: that pool's non-daemon threads
    are exactly what made _shutdown_spoc_executor_fast necessary in the
    first place (see server.py, 2026-08-01) — concurrent.futures registers
    an atexit hook that joins every live worker thread, so one stuck call
    keeps the whole process from exiting no matter how the pool is shut
    down. A bare ``threading.Thread(daemon=True)`` has no such hook: if the
    deadline is hit, this function raises ``TimeoutError`` and returns
    control to the caller immediately, and the abandoned thread — still
    blocked in whatever it was doing — dies silently when the process
    exits, never blocking it.

    In practice the leak is bounded, not just "ignored": install_mast_
    session_timeout (this module) gives the socket underneath a real
    timeout, so an abandoned MAST-call thread here almost always unblocks
    itself and exits on its own well before process shutdown — this
    deadline is the outer safety net for the deadline in the transport
    layer, not the only thing standing between a hung socket and a wedged
    server.

    On success, returns ``fn``'s return value. On failure, re-raises
    ``fn``'s own exception (not wrapped) so existing ``except Exception``
    fail-soft handlers around the call site keep working unchanged.
    """
    resolved = (
        deadline_s if deadline_s is not None
        else _read_float_env("EXONIT_SPOC_CHECK_DEADLINE", _DEFAULT_SPOC_CHECK_DEADLINE)
    )
    if resolved is None:
        return fn(*args, **kwargs)

    result: list[T] = []
    error: list[BaseException] = []

    def _runner():
        try:
            result.append(fn(*args, **kwargs))
        except BaseException as exc:  # re-raised on the caller's thread below
            error.append(exc)

    thread = threading.Thread(target=_runner, name=f"deadline-{name}", daemon=True)
    thread.start()
    thread.join(resolved)

    if thread.is_alive():
        raise TimeoutError(
            f"{name} did not complete within {resolved:.0f}s deadline "
            "(EXONIT_SPOC_CHECK_DEADLINE) — abandoning the call; the "
            "daemon thread will not block process shutdown")

    if error:
        raise error[0]
    return result[0]
