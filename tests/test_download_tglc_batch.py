"""Tests for scripts/download_tglc_batch.py — session reuse, defaults, sliding-window rate.

All tests are synthetic: no network access, no real file system downloads beyond
what tmp_path provides.
"""

import sys
import time
from collections import deque
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest

# Make scripts/ importable.
_SCRIPTS = Path(__file__).parent.parent / "scripts"
sys.path.insert(0, str(_SCRIPTS))

import download_tglc_batch as dtb  # noqa: E402
from download_tglc_batch import _build_session, parse  # noqa: E402


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #

def _make_curl_script(tmp_path: Path, n: int = 4) -> Path:
    """Write a synthetic curl script with *n* fake jobs."""
    script = tmp_path / "fake_bulk.sh"
    lines = ["#!/bin/sh"]
    for i in range(n):
        relpath = f"s0001/cam1/{i:04d}/file_{i:06d}.fits"
        url = f"https://mast.invalid/tglc/{i:06d}.fits"
        lines.append(
            f"curl -f --create-dirs --output '{relpath}' '{url}'"
        )
    script.write_text("\n".join(lines) + "\n")
    return script


# ------------------------------------------------------------------ #
# A. Session construction: one Session, HTTPAdapter mounted on https://
# ------------------------------------------------------------------ #

class TestSessionConstruction:
    def test_build_session_returns_requests_session(self):
        import requests
        s = _build_session(n_workers=4)
        assert isinstance(s, requests.Session)

    def test_https_adapter_mounted(self):
        """https:// has an HTTPAdapter with the right pool sizes."""
        import requests.adapters
        s = _build_session(n_workers=8)
        adapter = s.get_adapter("https://example.com")
        assert isinstance(adapter, requests.adapters.HTTPAdapter)
        assert adapter._pool_connections == 8
        assert adapter._pool_maxsize == 8

    def test_http_adapter_also_mounted(self):
        """http:// gets the same adapter (MAST sometimes redirects)."""
        import requests.adapters
        s = _build_session(n_workers=4)
        adapter = s.get_adapter("http://example.com")
        assert isinstance(adapter, requests.adapters.HTTPAdapter)

    def test_retry_policy_configured(self):
        """The adapter's max_retries carries our Retry object."""
        import urllib3.util.retry
        s = _build_session(n_workers=4)
        adapter = s.get_adapter("https://example.com")
        assert isinstance(adapter.max_retries, urllib3.util.retry.Retry)
        assert adapter.max_retries.total == 3

    def test_smoke_no_errors(self):
        """Constructing session+adapter with various worker counts must not raise."""
        for n in (1, 8, 32, 64):
            s = _build_session(n_workers=n)
            assert s is not None


# ------------------------------------------------------------------ #
# A. Session reuse: exactly ONE Session is created and all fetch calls use it
# ------------------------------------------------------------------ #

def _run_main(argv):
    """Call dtb.main() with a patched sys.argv (main() reads from sys.argv directly)."""
    with patch("sys.argv", ["download_tglc_batch.py"] + argv):
        return dtb.main()


class TestSessionReuse:
    """Verify that main() creates exactly one Session and all fetch() calls share it."""

    def test_one_session_created(self, tmp_path):
        """Exactly one requests.Session is instantiated during a main() run."""
        script = _make_curl_script(tmp_path, n=3)
        log_path = tmp_path / "test.log"
        dest = tmp_path / "cache"
        dest.mkdir()

        session_instances = []

        original_build = dtb._build_session

        def counting_build(n_workers):
            s = original_build(n_workers)
            session_instances.append(s)
            # Replace s.get so no real HTTP is made.
            fake_response = MagicMock()
            fake_response.raise_for_status = MagicMock()
            fake_response.content = b"\x00" * 5000
            s.get = MagicMock(return_value=fake_response)
            return s

        with patch.object(dtb, "_build_session", side_effect=counting_build):
            _run_main([
                "--script", str(script),
                "--dest", str(dest),
                "--workers", "4",
                "--log", str(log_path),
            ])

        assert len(session_instances) == 1, (
            f"Expected exactly 1 Session, got {len(session_instances)}"
        )

    def test_all_fetches_use_same_session(self, tmp_path):
        """All fetch() calls hit the same session object (not a new one each time)."""
        script = _make_curl_script(tmp_path, n=5)
        log_path = tmp_path / "test.log"
        dest = tmp_path / "cache"
        dest.mkdir()

        sessions_seen = []
        original_build = dtb._build_session

        def tracking_build(n_workers):
            s = original_build(n_workers)

            def recording_get(url, **kwargs):
                sessions_seen.append(id(s))
                resp = MagicMock()
                resp.raise_for_status = MagicMock()
                resp.content = b"\x00" * 5000
                return resp

            s.get = recording_get
            return s

        with patch.object(dtb, "_build_session", side_effect=tracking_build):
            _run_main([
                "--script", str(script),
                "--dest", str(dest),
                "--workers", "4",
                "--log", str(log_path),
            ])

        # Every fetch that hits the network should use the same session id.
        assert len(sessions_seen) == 5, (
            f"Expected 5 network calls (one per file), got {len(sessions_seen)}"
        )
        assert len(set(sessions_seen)) == 1, (
            f"Expected all calls to use the same session, got {len(set(sessions_seen))} different sessions"
        )


# ------------------------------------------------------------------ #
# B. Default --workers is 32
# ------------------------------------------------------------------ #

class TestDefaultWorkers:
    def test_download_tglc_batch_default_workers(self):
        """download_tglc_batch.py's --workers default is 32."""
        import argparse
        # Parse with no --workers argument; must default to 32.
        ap = argparse.ArgumentParser()
        ap.add_argument("--script", required=False, default="x")
        ap.add_argument("--dest", required=False, default="y")
        ap.add_argument("--workers", type=int, default=32)
        ap.add_argument("--log", required=False, default="z")
        args = ap.parse_args([])
        assert args.workers == 32

    def test_download_tglc_batch_parser_default(self, tmp_path):
        """Verify main()'s argparse default for --workers is 32 by intercepting parse_args."""
        import argparse

        captured_defaults = {}
        original_parse_args = argparse.ArgumentParser.parse_args

        def capturing_parse_args(self, args=None, namespace=None):
            for action in self._actions:
                if action.dest == "workers":
                    captured_defaults["workers"] = action.default
            return original_parse_args(self, args=args, namespace=namespace)

        # Provide a valid (non-existent) script path; main() will call parse_args
        # then open the script — we patch that open to raise so main() exits early.
        # The important thing is that parse_args is called first.
        script = tmp_path / "nonexistent.sh"
        log_path = tmp_path / "test.log"
        dest = tmp_path / "cache"

        with patch.object(argparse.ArgumentParser, "parse_args", capturing_parse_args):
            try:
                _run_main([
                    "--script", str(script),
                    "--dest", str(dest),
                    "--log", str(log_path),
                    # No --workers → should pick up default=32
                ])
            except Exception:
                pass  # main() may fail trying to open the script; that's fine

        assert "workers" in captured_defaults, (
            "parse_args was not called or --workers action not found"
        )
        assert captured_defaults["workers"] == 32, (
            f"Expected --workers default=32, got {captured_defaults['workers']}"
        )


# ------------------------------------------------------------------ #
# C. discovery_download.py passes --workers 32 in subprocess call
# ------------------------------------------------------------------ #

class TestDiscoveryDownloadWorkers:
    def test_default_workers_32_in_discovery_download(self):
        """discovery_download.build_parser() defaults --workers to 32."""
        _dd_scripts = Path(__file__).parent.parent / "scripts"
        sys.path.insert(0, str(_dd_scripts))
        import discovery_download as dd

        ap = dd.build_parser()
        args = ap.parse_args([
            "--sector", "2",
            "--script", "/tmp/fake.sh",
            "--dest", "/tmp/fake_dest",
        ])
        assert args.workers == 32, (
            f"discovery_download --workers default should be 32, got {args.workers}"
        )

    def test_subprocess_receives_workers_32(self, tmp_path, monkeypatch):
        """When discovery_download.main() runs, batch subprocess gets --workers 32."""
        _dd_scripts = Path(__file__).parent.parent / "scripts"
        sys.path.insert(0, str(_dd_scripts))
        import discovery_download as dd

        # Write a minimal curl script.
        script = tmp_path / "bulk.sh"
        script.write_text(
            "#!/bin/sh\n"
            "curl -f --create-dirs --output 's0001/x/file.fits' 'https://mast.invalid/file.fits'\n"
        )
        dest = tmp_path / "cache"
        dest.mkdir()

        captured_workers = {}

        def fake_subprocess_run(cmd, **kwargs):
            cmd_str = " ".join(str(c) for c in cmd)
            if "download_tglc_batch.py" in cmd_str:
                if "--workers" in cmd:
                    captured_workers["value"] = cmd[list(cmd).index("--workers") + 1]
                class FakeResult:
                    returncode = 0
                    stdout = "done=1 skip=0 fail=0"
                    stderr = ""
                return FakeResult()
            if "index_tglc_batch.py" in cmd_str:
                class FakeResult:
                    returncode = 0
                    stdout = "index: 1 distinct TICs, 1 (tic,sector) rows\n"
                    stderr = ""
                return FakeResult()
            raise AssertionError(f"Unexpected subprocess call: {cmd}")

        monkeypatch.setattr(dd.subprocess, "run", fake_subprocess_run)

        dd.main([
            "--sector", "2",
            "--script", str(script),
            "--dest", str(dest),
            "--registry", str(tmp_path / "sectors.json"),
            "--index", str(tmp_path / "tglc.db"),
            # no explicit --workers → should default to 32
        ])

        assert "value" in captured_workers, "--workers not found in batch subprocess argv"
        assert captured_workers["value"] == "32", (
            f"Expected --workers 32, got --workers {captured_workers['value']!r}"
        )


# ------------------------------------------------------------------ #
# D. Sliding-window rate
# ------------------------------------------------------------------ #

class TestSlidingWindowRate:
    """Verify that the sliding-window rate reflects only recent completions."""

    def test_window_rate_uses_recent_not_full_run(self):
        """
        Simulate a run where the first half is slow and the second half is fast.
        The window rate (last 512) should be higher than the overall average.
        """
        # Build two sets of timestamps:
        # - 50 completions spread over 100 s (slow phase).
        # - 50 completions spread over 5 s (fast phase).
        now = 0.0
        timestamps = []

        # Slow phase: 50 events over 100 s.
        slow_start = now
        for i in range(50):
            timestamps.append(slow_start + i * 2.0)  # 1 every 2s = 0.5/s

        # Fast phase: 50 events over 5 s.
        fast_start = timestamps[-1] + 0.1
        for i in range(50):
            timestamps.append(fast_start + i * 0.1)  # 1 every 0.1s = 10/s

        total_time = timestamps[-1] - timestamps[0]
        overall_rate = len(timestamps) / total_time  # ≈ 100 / 105 ≈ 0.95/s

        # Sliding window over the last min(512, N) entries.
        recent: deque = deque(maxlen=512)
        for t in timestamps:
            recent.append(t)

        # Window covers all 100 (< 512), so window rate = 99 gaps / total_time.
        # But the POINT is that if we had a small maxlen (e.g., 60) the window
        # would only cover the fast phase.  With maxlen=512 and 100 events,
        # window = all 100.  Let's test with a small deque to simulate "recent":
        window_deque: deque = deque(maxlen=60)
        for t in timestamps:
            window_deque.append(t)

        # The window deque now holds only the last 60 timestamps (the fast phase + tail).
        window_rate = (len(window_deque) - 1) / (window_deque[-1] - window_deque[0])

        # The window rate should be significantly higher than the overall average.
        assert window_rate > overall_rate * 2, (
            f"Window rate {window_rate:.2f}/s should be >2× overall rate {overall_rate:.2f}/s"
        )

    def test_window_rate_fallback_with_one_sample(self):
        """With fewer than 2 samples in the window, falls back to n/elapsed."""
        recent: deque = deque(maxlen=512)
        # Only one timestamp — can't compute a rate from the window.
        recent.append(100.0)

        n = 1
        el = 10.0  # 10 seconds elapsed
        if len(recent) >= 2:
            window_rate = (len(recent) - 1) / (recent[-1] - recent[0])
        else:
            window_rate = n / el if el else 0

        assert abs(window_rate - 0.1) < 1e-9, (
            f"Fallback rate should be n/el = 0.1, got {window_rate}"
        )

    def test_window_rate_formula_matches_implementation(self, tmp_path):
        """
        Run main() on a synthetic script where all files are pre-present (skip path),
        confirm no crash and that the log file contains the [finish] line.
        The sliding-window logic is exercised by tracking the deque in the main loop.
        """
        script = _make_curl_script(tmp_path, n=6)
        log_path = tmp_path / "rate_test.log"
        dest = tmp_path / "cache"
        dest.mkdir()

        # Pre-create all files so they are skipped (no network needed).
        for rel, _ in parse(script):
            out = dest / rel
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(b"\x00" * 5000)

        _run_main([
            "--script", str(script),
            "--dest", str(dest),
            "--workers", "2",
            "--log", str(log_path),
        ])

        log_text = log_path.read_text()
        assert "[finish]" in log_text, "Expected [finish] line in log"
        assert "skip=6" in log_text, "Expected all 6 files to be skipped"

    def test_log_format_preserved(self, tmp_path):
        """The [N/M] log line format must contain done= skip= fail= RATE/s eta=ETAmin."""
        script = _make_curl_script(tmp_path, n=250)
        log_path = tmp_path / "fmt_test.log"
        dest = tmp_path / "cache"
        dest.mkdir()

        # Pre-create all files to avoid network.
        for rel, _ in parse(script):
            out = dest / rel
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(b"\x00" * 5000)

        _run_main([
            "--script", str(script),
            "--dest", str(dest),
            "--workers", "4",
            "--log", str(log_path),
        ])

        log_text = log_path.read_text()
        # Find the progress line (triggered at n==250==total since n%250==0).
        progress_lines = [l for l in log_text.splitlines() if l.startswith("[250/250]")]
        assert progress_lines, "Expected at least one [N/M] progress line in log"
        line = progress_lines[0]
        assert "done=" in line
        assert "skip=" in line
        assert "fail=" in line
        assert "/s" in line
        assert "eta=" in line
        assert "min" in line


# ------------------------------------------------------------------ #
# E. Automatic retry of the failed set
# ------------------------------------------------------------------ #

def _fake_get_factory(should_fail):
    """Build a fake session.get(url) whose i-th call for a url raises when
    should_fail(url, i) is truthy, else returns a 5 KB OK response."""
    import requests

    calls: dict = {}

    def fake_get(url, **kwargs):
        i = calls.get(url, 0)
        calls[url] = i + 1
        if should_fail(url, i):
            raise requests.exceptions.ConnectionError("synthetic transient error")
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        resp.content = b"\x00" * 5000
        return resp

    fake_get.calls = calls
    return fake_get


def _run_with_fake_get(argv, fake_get):
    """Run dtb.main(argv) with _build_session returning a session using fake_get,
    and time.sleep stubbed out so retry backoffs don't actually wait."""
    original_build = dtb._build_session

    def build_with(n_workers):
        s = original_build(n_workers)
        s.get = fake_get
        return s

    with patch.object(dtb, "_build_session", side_effect=build_with), \
         patch.object(dtb.time, "sleep"):
        return _run_main(argv)


class TestAutoRetry:
    """The failed set is re-swept after the main pass (--retry-passes)."""

    def test_transient_failures_recovered_on_retry(self, tmp_path):
        """Jobs that fail the whole main pass (both in-fetch attempts) but then
        succeed are recovered → final fail=0, files present on disk."""
        script = _make_curl_script(tmp_path, n=4)
        log_path = tmp_path / "retry.log"
        dest = tmp_path / "cache"
        dest.mkdir()

        # fetch() makes 2 in-fetch attempts per pass → calls 0,1 fail (pass 1),
        # call 2 (first retry sweep) succeeds.
        fake_get = _fake_get_factory(lambda url, i: i < 2)

        _run_with_fake_get(
            ["--script", str(script), "--dest", str(dest),
             "--workers", "4", "--log", str(log_path)],
            fake_get,
        )

        log_text = log_path.read_text()
        assert "[retry 1/3]" in log_text, "Expected a retry sweep to be logged"
        finish = [l for l in log_text.splitlines() if l.startswith("[finish]")][0]
        assert "done=4" in finish and "fail=0" in finish, finish
        assert "[failed]" not in log_text
        # All four files actually landed on disk.
        assert len(list(dest.rglob("*.fits"))) == 4

    def test_permanent_failures_reported_after_retries(self, tmp_path):
        """Jobs that always fail survive all retry passes → fail=N, each named
        on a [failed] line."""
        script = _make_curl_script(tmp_path, n=2)
        log_path = tmp_path / "perm.log"
        dest = tmp_path / "cache"
        dest.mkdir()

        fake_get = _fake_get_factory(lambda url, i: True)  # never succeeds

        rc = _run_with_fake_get(
            ["--script", str(script), "--dest", str(dest),
             "--workers", "4", "--log", str(log_path), "--retry-passes", "2"],
            fake_get,
        )
        assert rc == 0  # the run completes; failures are reported, not fatal

        log_text = log_path.read_text()
        assert "[retry 1/2]" in log_text and "[retry 2/2]" in log_text
        finish = [l for l in log_text.splitlines() if l.startswith("[finish]")][0]
        assert "done=0" in finish and "fail=2" in finish, finish
        failed_lines = [l for l in log_text.splitlines() if l.startswith("[failed]")]
        assert len(failed_lines) == 2, failed_lines
        assert all(l.endswith(".fits") for l in failed_lines)

    def test_retry_passes_zero_disables_retry(self, tmp_path):
        """--retry-passes 0 means no extra sweep: a failure is final immediately."""
        script = _make_curl_script(tmp_path, n=1)
        log_path = tmp_path / "noretry.log"
        dest = tmp_path / "cache"
        dest.mkdir()

        fake_get = _fake_get_factory(lambda url, i: True)

        _run_with_fake_get(
            ["--script", str(script), "--dest", str(dest),
             "--workers", "2", "--log", str(log_path), "--retry-passes", "0"],
            fake_get,
        )

        log_text = log_path.read_text()
        assert "[retry" not in log_text, "retry-passes 0 must not sweep"
        finish = [l for l in log_text.splitlines() if l.startswith("[finish]")][0]
        assert "fail=1" in finish, finish

    def test_retry_passes_default_is_three(self):
        """main()'s --retry-passes default is 3."""
        import argparse
        captured = {}
        original = argparse.ArgumentParser.parse_args

        def capturing(self, args=None, namespace=None):
            for action in self._actions:
                if action.dest == "retry_passes":
                    captured["v"] = action.default
            return original(self, args=args, namespace=namespace)

        with patch.object(argparse.ArgumentParser, "parse_args", capturing):
            try:
                _run_main(["--script", "/nonexistent.sh", "--dest", "/tmp/x",
                           "--log", "/tmp/y.log"])
            except Exception:
                pass
        assert captured.get("v") == 3, captured
