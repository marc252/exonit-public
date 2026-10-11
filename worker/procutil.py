"""Cross-platform PID-liveness probe shared by ``worker.py`` and ``agent.py``.

Extracted 2026-07-02 cleanup review (D13): ``_pid_alive`` was duplicated
byte-similar in both files, *including* the Windows ``os.execv`` SystemError
edge case documented below — a genuine correctness detail (not boilerplate)
that must not be allowed to drift between two hand-maintained copies.
"""
from __future__ import annotations

import os


def pid_alive(pid) -> bool:
    """Return True if ``pid`` corresponds to a currently-running process.

    ``os.kill(pid, 0)`` raises ``ProcessLookupError`` on Unix when the PID
    is gone, ``PermissionError`` when the process exists but is owned by
    someone else, and ``OSError`` (ESRCH) on Windows (Python 3.3+).

    Windows quirk (CPython bug): after ``os.execv`` the new process has a
    *different* PID (``execv`` is emulated via spawn on Windows). Probing
    the old, now-dead PID via ``OpenProcess`` can raise ``SystemError``
    with the message "<class 'OSError'> returned a result with an
    exception set" instead of the expected ``OSError``. We treat this as
    "process not alive" — see memory gotcha
    ``gotcha_windows_execv_pid_systemerror.md`` (fix 755b5210).

    Also defensively treats a malformed ``pid`` (``None``, non-``int``,
    ``<= 0``) as "not alive" rather than letting ``os.kill`` raise
    ``TypeError``/``OverflowError`` — both callers can hand this a value
    read straight off disk (a PID-lock file / an agent PID file) that may
    be empty, stale, or corrupted.
    """
    if pid is None or not isinstance(pid, int) or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except (OSError, SystemError, ValueError):
        return False
    return True
