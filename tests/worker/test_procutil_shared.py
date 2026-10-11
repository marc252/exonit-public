"""D13 (2026-07-02 cleanup review): ``_pid_alive`` was duplicated
byte-similar in agent.py and worker.py, including the Windows
``os.execv`` SystemError edge case. Both now import the single
implementation from ``worker/procutil.py`` — this test guards against
the duplication silently creeping back.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "worker") not in sys.path:
    sys.path.append(str(ROOT / "worker"))

import agent  # noqa: E402
import worker.worker as worker_mod  # noqa: E402
import procutil  # noqa: E402


def test_agent_and_worker_share_the_same_pid_alive_function():
    assert agent._pid_alive is procutil.pid_alive
    assert worker_mod._pid_alive is procutil.pid_alive
    assert agent._pid_alive is worker_mod._pid_alive


def test_pid_alive_windows_systemerror_edge_case(monkeypatch):
    """SystemError (Windows os.execv/OpenProcess quirk after a stale PID)
    must be treated as 'not alive', not propagated — the one behavioural
    detail that made this worth extracting carefully rather than just
    picking one copy."""
    import os

    def _fake_kill(pid, sig):
        raise SystemError("<class 'OSError'> returned a result with an exception set")

    monkeypatch.setattr(os, "kill", _fake_kill)
    assert procutil.pid_alive(12345) is False


def test_pid_alive_rejects_malformed_pid():
    assert procutil.pid_alive(None) is False
    assert procutil.pid_alive(0) is False
    assert procutil.pid_alive(-1) is False
    assert procutil.pid_alive("123") is False
