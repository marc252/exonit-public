"""ExoMiner allowlist loader: env var (legacy) + hot-reload file (new).

The hot-reload path lets us add/remove workers without restarting the
server — critical when cloud autoscaling churns instance IDs.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from exonit.server import _load_exominer_allowlist


@pytest.fixture(autouse=True)
def _reset_cache_and_env(monkeypatch):
    from exonit import server as srv
    srv._exominer_allowlist_cache.update(
        {"mtime": 0.0, "names": None, "src": ""}
    )
    monkeypatch.delenv("EXONIT_EXOMINER_WORKERS_FILE", raising=False)
    monkeypatch.delenv("EXONIT_EXOMINER_WORKERS", raising=False)
    yield


def test_no_config_returns_none_meaning_allow_all():
    assert _load_exominer_allowlist() is None


def test_env_var_only(monkeypatch):
    monkeypatch.setenv("EXONIT_EXOMINER_WORKERS", "a,b,c")
    assert _load_exominer_allowlist() == {"a", "b", "c"}


def test_file_overrides_env_var(monkeypatch, tmp_path):
    p = tmp_path / "list.txt"
    p.write_text("worker-1\nworker-2\n")
    monkeypatch.setenv("EXONIT_EXOMINER_WORKERS", "ignored")
    monkeypatch.setenv("EXONIT_EXOMINER_WORKERS_FILE", str(p))
    assert _load_exominer_allowlist() == {"worker-1", "worker-2"}


def test_file_ignores_blanks_and_comments(monkeypatch, tmp_path):
    p = tmp_path / "list.txt"
    p.write_text("# header\nworker-1\n\n  # inline-ish\nworker-2\n   \n")
    monkeypatch.setenv("EXONIT_EXOMINER_WORKERS_FILE", str(p))
    assert _load_exominer_allowlist() == {"worker-1", "worker-2"}


def test_file_hot_reload_on_mtime_change(monkeypatch, tmp_path):
    p = tmp_path / "list.txt"
    p.write_text("worker-1\n")
    monkeypatch.setenv("EXONIT_EXOMINER_WORKERS_FILE", str(p))
    assert _load_exominer_allowlist() == {"worker-1"}
    # Advance mtime by 2s — same-second writes can land on the cached mtime
    # value on filesystems with 1s resolution and silently bypass the reload.
    p.write_text("worker-2\nworker-3\n")
    new_mtime = p.stat().st_mtime + 2
    os.utime(p, (new_mtime, new_mtime))
    assert _load_exominer_allowlist() == {"worker-2", "worker-3"}


def test_missing_file_is_fail_closed_not_fall_through(monkeypatch, tmp_path):
    """If the operator configures a file path and the file disappears,
    deny all rather than silently allowing every worker (which would
    re-open the period-alias damage window). Workers without the model
    must NOT be served tasks."""
    monkeypatch.setenv("EXONIT_EXOMINER_WORKERS", "would-allow-this")
    monkeypatch.setenv(
        "EXONIT_EXOMINER_WORKERS_FILE", str(tmp_path / "does-not-exist.txt")
    )
    assert _load_exominer_allowlist() == set()
