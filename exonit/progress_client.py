"""Fire-and-forget progress reporting for long-running pipeline scripts.

Usage inside a script:

    from exonit.progress_client import ProgressReporter

    rep = ProgressReporter(unit="buckets", total=len(buckets),
                           substep="compact_completeness", interval_s=10)
    for i, bucket in enumerate(buckets, 1):
        compact(bucket)
        rep.report(i, message=f"bucket {bucket}")
    rep.finish()

THE ONE RULE: this must never cost the caller anything. It cannot raise, it
cannot block for more than the 2 s timeout, and it does nothing at all unless
the process was launched as part of a run (EXONIT_PROGRESS_RUN_ID and
EXONIT_PROGRESS_PHASE set by whoever spawned it — the orchestrator for most
phases, the server for ENQUEUEING). Run the same script by hand from a
terminal and it is a silent no-op: no flag to remember, and no way to
contaminate a live run's progress.

stdlib only, deliberately: this runs on <host>'s venv and inside the
scripts' venvs, which do not all have `requests`.
"""

from __future__ import annotations

import json
import logging
import os
import socket
import time
import urllib.request

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT_S = 2.0
DEFAULT_INTERVAL_S = 10.0


def _post(url: str, body: dict, token: str, timeout: float) -> None:
    """One HTTP POST. Raises on any failure — the caller swallows it."""
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, method="POST",
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {token}"})
    with urllib.request.urlopen(req, timeout=timeout):
        pass


class ProgressReporter:
    """Reports progress for the phase this process was launched to run."""

    def __init__(self, *, run_id: str | None = None, phase: str | None = None,
                 unit: str | None = None, total: int | None = None,
                 substep: str | None = None,
                 interval_s: float = DEFAULT_INTERVAL_S,
                 timeout_s: float = DEFAULT_TIMEOUT_S) -> None:
        # Identity normally comes from the environment (a spawned child
        # inherits it for free). The explicit arguments exist for the ONE
        # caller that has no child to spawn: the orchestrator reporting on a
        # phase it executes itself. Passing them beats mutating os.environ,
        # which would leak across a long-lived process and across tests.
        self.run_id = (run_id or os.environ.get("EXONIT_PROGRESS_RUN_ID", "")).strip()
        self.phase = (phase or os.environ.get("EXONIT_PROGRESS_PHASE", "")).strip()
        base = (os.environ.get("EXONIT_PROGRESS_URL")
                or os.environ.get("EXONIT_SERVER_URL") or "").strip()
        self.url = f"{base.rstrip('/')}/api/progress" if base else ""
        if self.url and not self.url.startswith(("http://", "https://")):
            self.url = f"http://{self.url}"
        self.token = os.environ.get("EXONIT_WORKER_TOKEN", "exonit-default-token")

        self.unit = unit
        self.total = total
        self.substep = substep
        # interval_s is a promise to the server: "expect a mark this often".
        # The server declares the record stale at 3x this value, so it is the
        # single knob that tells "slow" from "dead" for this phase.
        self.interval_s = float(interval_s)
        self.timeout_s = float(timeout_s)

        self._last_sent = 0.0
        self._first_sent = False
        self._failing = False

        self.enabled = bool(self.run_id and self.phase and self.url)

    # -- internals --------------------------------------------------------

    def _payload(self, done: int, total: int | None, message: str | None,
                 substep: str | None) -> dict:
        return {
            "run_id": self.run_id,
            "phase": self.phase,
            "substep": substep if substep is not None else self.substep,
            "unit": self.unit,
            "done": int(done),
            "total": total if total is not None else self.total,
            "message": message,
            "host": socket.gethostname(),
            "pid": os.getpid(),
            "interval_s": self.interval_s,
        }

    def _send(self, body: dict) -> None:
        try:
            _post(self.url, body, self.token, self.timeout_s)
            self._failing = False
        except Exception as exc:
            # One warning per burst: a server that is down for an hour must
            # not fill this script's log with one line per mark.
            if not self._failing:
                self._failing = True
                logger.warning("progress reporting unavailable (%s); "
                               "continuing without it", exc)

    # -- public API -------------------------------------------------------

    def report(self, done: int, *, total: int | None = None,
               message: str | None = None, substep: str | None = None,
               force: bool = False) -> None:
        """Report *done* items. Rate-limited to one mark per interval_s.

        The first mark of the reporter's life always goes through, as does any
        mark whose substep differs from the last one sent — a substep change
        is a phase-internal transition the dashboard must see immediately.
        """
        if not self.enabled:
            return
        try:
            now = time.monotonic()
            effective_substep = substep if substep is not None else self.substep
            changed_substep = effective_substep != self.substep
            due = (force or not self._first_sent or changed_substep
                   or (now - self._last_sent) >= self.interval_s)
            if not due:
                return
            if substep is not None:
                self.substep = substep
            self._last_sent = now
            self._first_sent = True
            self._send(self._payload(done, total, message, effective_substep))
        except Exception as exc:
            logger.debug("progress report skipped: %s", exc)

    def finish(self, message: str | None = None) -> None:
        """Final mark of the phase: done == total. Always sent."""
        if not self.enabled:
            return
        try:
            total = self.total
            done = total if total is not None else 0
            self._last_sent = time.monotonic()
            self._first_sent = True
            self._send(self._payload(done, total, message, self.substep))
        except Exception as exc:
            logger.debug("progress finish skipped: %s", exc)


def reporter(**kwargs) -> ProgressReporter:
    """Convenience constructor so scripts need one import, not two."""
    return ProgressReporter(**kwargs)
