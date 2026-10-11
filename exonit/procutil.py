"""Shared subprocess-reaping helper.

Extracted 2026-07-10 (code-review F3 on the 2026-07-07 zombie-orchestrator
fix): ``_reap_in_background`` was originally a private helper local to
``exonit/routes/discovery.py``, applied only to the discovery orchestrator
and download ``subprocess.Popen`` sites. The exact same zombie-leak shape
(a long-lived uvicorn worker Popen's a detached child and never
poll()s/wait()s it) also exists at ``exonit/routes/aws.py`` (repatriate
script — never waited on at all) and ``exonit/routes/admin.py`` (pipeline
run — reaped by its own stdout-reader thread in the common case, but that
thread's ``proc.wait()`` is skipped if it raises before reaching EOF, e.g.
a ``shared_state`` write failure mid-loop). A single shared helper keeps
all of these consistent instead of re-deriving the same one-liner per
call site.
"""

import subprocess
import threading


def reap_in_background(proc: subprocess.Popen) -> None:
    """Reap *proc* the moment it exits, without blocking the caller.

    A Popen'd child that nothing ever poll()s/wait()s on becomes a
    <defunct> zombie the instant it exits — the kernel keeps its exit
    status around until this (still-running) parent process collects it,
    which for a long-lived server means "until the server restarts". A
    daemon thread doing a blocking ``proc.wait()`` reaps it as soon as it
    exits, without blocking the request handler that spawned it.

    Safe to call more than once on the same ``proc`` from different
    threads: ``subprocess.Popen.wait()`` is internally lock-protected in
    CPython, so concurrent callers simply serialise on the same result.
    """
    threading.Thread(target=proc.wait, daemon=True).start()
