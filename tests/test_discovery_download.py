"""Tests for scripts/discovery_download.py.

All tests are synthetic:
  - No network access (MAST or otherwise).
  - No access to real DB (exonit.db).
  - No subprocess calls to download/index scripts (they are stubbed or
    their underlying pure-logic functions are tested directly).
"""

import json
import os
import re
import sys
import textwrap
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Make the scripts/ directory importable.
_SCRIPTS = Path(__file__).parent.parent / "scripts"
sys.path.insert(0, str(_SCRIPTS))

from discovery_download import (  # noqa: E402
    select_next_jobs,
    _parse_count,
    _parse_indexed,
    _parse_files_total,
    _indexed_basenames,
    main,
)
from download_tglc_batch import parse_iter as _parse_iter  # noqa: E402


def _cmd_chunk(kind: str, cmd: list) -> int:
    """Extract the chunk number (…_c001, _c002, …) from a fake subprocess cmd."""
    flag = "--script" if kind == "download" else "--files-from"
    path = cmd[cmd.index(flag) + 1]
    m = re.search(r"_c(\d+)\.", path)
    assert m, f"could not find chunk number in {path!r}"
    return int(m.group(1))


# ------------------------------------------------------------------ #
# Fixtures
# ------------------------------------------------------------------ #

def _make_curl_script(tmp_path: Path, n: int = 6) -> Path:
    """Write a synthetic curl script with *n* fake jobs."""
    script = tmp_path / "fake_bulk.sh"
    lines = ["#!/bin/sh"]
    for i in range(n):
        relpath = f"s0001/cam1-ccd1/{i:04d}/hlsp_fake_{i:06d}_s0001.fits"
        url = f"https://example.invalid/tglc/{i:06d}.fits"
        lines.append(
            f"curl -f --create-dirs --output '{relpath}' '{url}'"
        )
    script.write_text("\n".join(lines) + "\n")
    return script


def _pre_create_files(dest: Path, relpaths: list[str], size: int = 5000) -> None:
    """Create fake FITS files in *dest* so they look already-downloaded."""
    for rel in relpaths:
        out = dest / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"\x00" * size)


def _make_index(tmp_path: Path, relpaths: list[str], sector: int = 2) -> Path:
    """Build a tglc index marking *relpaths* as already downloaded+indexed.

    "Already downloaded" means INDEXED, not "a file exists on disk" (2026-07-14).
    The old rule — on disk ⇒ done — is what made an interrupted download
    unrecoverable: a killed run left 24,747 files on disk in no index, and every
    later run skipped them (they existed) so they were never indexed, never
    packed, and never servable. Presence is now the tglc ∪ fits_blobs union, so
    a downloaded-but-unindexed file is correctly re-selected and healed.
    """
    import sqlite3
    idx = tmp_path / "index.db"
    con = sqlite3.connect(idx)
    con.execute("CREATE TABLE IF NOT EXISTS tglc (tic_id TEXT, sector INTEGER, "
                "path TEXT, exptime REAL, tmag REAL, ra REAL, dec REAL)")
    con.execute("CREATE TABLE IF NOT EXISTS fits_blobs (filename TEXT PRIMARY KEY, "
                "container_key TEXT, data_offset INTEGER, data_length INTEGER)")
    for i, rel in enumerate(relpaths):
        con.execute("INSERT INTO tglc(tic_id,sector,path) VALUES (?,?,?)",
                    (str(1000 + i), sector, rel))
    con.commit()
    con.close()
    return idx


# ------------------------------------------------------------------ #
# select_next_jobs (pure logic)
# ------------------------------------------------------------------ #

class TestSelectNextJobs:
    def _make_jobs(self, n=6):
        return [
            (f"s0001/cam{i}.fits", f"https://x.invalid/{i}.fits")
            for i in range(n)
        ]

    def test_all_absent_returns_up_to_count(self, tmp_path):
        jobs = self._make_jobs(6)
        already, batch = select_next_jobs(jobs, tmp_path, count=4)
        assert already == 0
        assert len(batch) == 4
        assert batch == jobs[:4]

    def test_respects_already_present(self, tmp_path):
        jobs = self._make_jobs(6)
        # Pre-create the first 2.
        for rel, _ in jobs[:2]:
            p = tmp_path / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b"\x00" * 5000)

        already, batch = select_next_jobs(jobs, tmp_path, count=10)
        assert already == 2
        assert len(batch) == 4
        # Batch should be jobs[2:6].
        assert batch == jobs[2:]

    def test_count_1_picks_first_absent(self, tmp_path):
        jobs = self._make_jobs(6)
        # Pre-create jobs[0] and jobs[1].
        for rel, _ in jobs[:2]:
            p = tmp_path / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b"\x00" * 5000)

        already, batch = select_next_jobs(jobs, tmp_path, count=1)
        assert already == 2
        assert len(batch) == 1
        assert batch[0] == jobs[2]

    def test_all_present_returns_empty_batch(self, tmp_path):
        jobs = self._make_jobs(4)
        for rel, _ in jobs:
            p = tmp_path / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b"\x00" * 5000)

        already, batch = select_next_jobs(jobs, tmp_path, count=100)
        assert already == 4
        assert batch == []

    def test_tiny_file_not_counted_as_downloaded(self, tmp_path):
        jobs = self._make_jobs(3)
        # Create a tiny file (≤1000 bytes) — should NOT count as downloaded.
        rel, _ = jobs[0]
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"\x00" * 100)

        already, batch = select_next_jobs(jobs, tmp_path, count=5)
        assert already == 0
        # The tiny file must be included in the batch.
        assert batch[0] == jobs[0]

    def test_empty_jobs_list(self, tmp_path):
        already, batch = select_next_jobs([], tmp_path, count=10)
        assert already == 0
        assert batch == []

    def test_ordering_preserved(self, tmp_path):
        # Pre-create jobs[1] and jobs[3] only.
        jobs = self._make_jobs(6)
        for rel, _ in [jobs[1], jobs[3]]:
            p = tmp_path / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b"\x00" * 5000)

        already, batch = select_next_jobs(jobs, tmp_path, count=10)
        assert already == 2
        # Absent: 0, 2, 4, 5 — in that order.
        assert [r for r, _ in batch] == [
            jobs[0][0], jobs[2][0], jobs[4][0], jobs[5][0]
        ]


# ------------------------------------------------------------------ #
# _parse_count / _parse_indexed
# ------------------------------------------------------------------ #

class TestParseHelpers:
    def test_parse_count_done(self):
        assert _parse_count("done=12 skip=3 fail=1", "done") == 12

    def test_parse_count_fail(self):
        assert _parse_count("done=5 skip=0 fail=2", "fail") == 2

    def test_parse_count_missing_key(self):
        assert _parse_count("done=5 skip=0", "fail") == 0

    def test_parse_indexed_normal(self):
        text = "index: 43 (tic,sector) rows\n"
        assert _parse_indexed(text) == 43

    def test_parse_indexed_missing(self):
        assert _parse_indexed("no match here") == 0

    def test_parse_files_total_normal(self):
        text = "scanning 5 TGLC fits in cache, 2 new\nfiles_total=5\n"
        assert _parse_files_total(text) == 5

    def test_parse_files_total_missing(self):
        assert _parse_files_total("no match here") == 0


# ------------------------------------------------------------------ #
# main() CLI — dry-run tests (no network, no subprocess)
# ------------------------------------------------------------------ #

class TestMainDryRun:
    def _run(self, argv, capsys):
        """Run main() and return (returncode, parsed_json_output).

        main() now also emits a {"status": "select_start"} marker before the
        selection scan (which runs in dry-run mode too), so stdout can have
        more than one JSON line; parse the last non-empty one — same pattern
        used elsewhere in this file for the index_start/skip_set markers.
        """
        rc = main(argv)
        captured = capsys.readouterr()
        raw = captured.out.strip()
        out = json.loads([l for l in raw.splitlines() if l.strip()][-1])
        return rc, out

    def test_dry_run_already_downloaded_2_to_download_4(self, tmp_path, capsys):
        script = _make_curl_script(tmp_path, n=6)
        dest = tmp_path / "cache"
        # Pre-create files for jobs 0 and 1 (from the script).
        # The fake script writes relpaths like s0001/cam1-ccd1/0000/hlsp_fake_000000_s0001.fits
        from download_tglc_batch import parse as _parse
        jobs = _parse(script)
        _pre_create_files(dest, [jobs[0][0], jobs[1][0]])
        # ...and INDEX them: on-disk alone no longer means "done" (see
        # _make_index). Without this they would be re-selected for healing.
        idx = _make_index(tmp_path, [jobs[0][0], jobs[1][0]])

        rc, out = self._run([
            "--sector", "2",
            "--script", str(script),
            "--dest", str(dest),
            "--index", str(idx),
            "--registry", str(tmp_path / "sectors.json"),
            "--dry-run",
        ], capsys)

        assert rc == 0
        assert out["status"] == "dry_run"
        assert out["already_downloaded"] == 2
        assert out["to_download"] == 4

    def test_dry_run_count_cap(self, tmp_path, capsys):
        script = _make_curl_script(tmp_path, n=6)
        dest = tmp_path / "cache"
        # No files pre-created; count=2 should cap.
        rc, out = self._run([
            "--sector", "2",
            "--script", str(script),
            "--dest", str(dest),
            "--registry", str(tmp_path / "sectors.json"),
            "--dry-run",
            "--count", "2",
        ], capsys)

        assert rc == 0
        assert out["status"] == "dry_run"
        assert out["already_downloaded"] == 0
        assert out["to_download"] == 2
        assert out["requested"] == 2

    def test_dry_run_count_1_picks_first_absent(self, tmp_path, capsys):
        script = _make_curl_script(tmp_path, n=6)
        dest = tmp_path / "cache"
        from download_tglc_batch import parse as _parse
        jobs = _parse(script)
        _pre_create_files(dest, [jobs[0][0]])  # first job present...
        idx = _make_index(tmp_path, [jobs[0][0]])   # ...and indexed

        rc, out = self._run([
            "--sector", "2",
            "--script", str(script),
            "--dest", str(dest),
            "--index", str(idx),
            "--registry", str(tmp_path / "sectors.json"),
            "--dry-run",
            "--count", "1",
        ], capsys)

        assert rc == 0
        assert out["to_download"] == 1
        assert out["already_downloaded"] == 1

    def test_dry_run_all_present_to_download_0(self, tmp_path, capsys):
        script = _make_curl_script(tmp_path, n=3)
        dest = tmp_path / "cache"
        from download_tglc_batch import parse as _parse
        jobs = _parse(script)
        _pre_create_files(dest, [r for r, _ in jobs])
        idx = _make_index(tmp_path, [r for r, _ in jobs])

        rc, out = self._run([
            "--sector", "2",
            "--script", str(script),
            "--dest", str(dest),
            "--index", str(idx),
            "--registry", str(tmp_path / "sectors.json"),
            "--dry-run",
        ], capsys)

        assert rc == 0
        assert out["already_downloaded"] == 3
        assert out["to_download"] == 0

    def test_on_disk_but_unindexed_is_re_selected_for_healing(
            self, tmp_path, capsys):
        """THE 2026-07-14 REGRESSION TEST.

        A server restart killed a download after it had fetched 24,747 files but
        before its index+pack step. The files were on disk, in no index — and
        the old rule (on disk ⇒ already downloaded) made every later run SKIP
        them. They were never indexed, never packed, and /api/fits/<name> 404s
        for a file the metadata cache has never heard of: 24,747 invisible,
        unusable files that no re-run could recover.

        Presence must therefore mean INDEXED. Such a file is re-selected, and
        healing is nearly free: download_tglc_batch.py skips the HTTP fetch for
        anything already on disk, so the file is simply indexed and packed.
        """
        script = _make_curl_script(tmp_path, n=4)
        dest = tmp_path / "cache"
        from download_tglc_batch import parse as _parse
        jobs = _parse(script)

        # On disk, but the index knows nothing about them — the exact state a
        # killed download leaves behind.
        _pre_create_files(dest, [jobs[0][0], jobs[1][0]])
        idx = _make_index(tmp_path, [])

        rc, out = self._run([
            "--sector", "2",
            "--script", str(script),
            "--dest", str(dest),
            "--index", str(idx),
            "--registry", str(tmp_path / "sectors.json"),
            "--dry-run",
        ], capsys)

        assert rc == 0
        assert out["already_downloaded"] == 0, (
            "a file on disk but absent from the index is NOT done — treating it "
            "as done is what stranded 24,747 files")
        assert out["to_download"] == 4      # all four, orphans included


# ------------------------------------------------------------------ #
# Sector validation
# ------------------------------------------------------------------ #

class TestSectorValidation:
    def test_unknown_sector_exits_2(self, tmp_path, capsys):
        # A sector with no per-sector bulk script is rejected at the gate.
        rc = main(["--sector", "99", "--registry",
                   str(tmp_path / "sectors.json"), "--dry-run"])
        assert rc == 2
        out = json.loads(capsys.readouterr().out.strip())
        assert out["reason"] == "sector_not_enabled"
        assert out["sector"] == 99

    def test_disabled_sector_exits_2(self, tmp_path, capsys):
        # A known sector explicitly disabled in the registry is rejected.
        import json as _json
        reg = tmp_path / "sectors.json"
        reg.write_text(_json.dumps(
            {"active_sector": 2, "sectors": {"2": {"enabled": False}}}))
        rc = main(["--sector", "2", "--registry", str(reg), "--dry-run"])
        assert rc == 2
        out = json.loads(capsys.readouterr().out.strip())
        assert out["reason"] == "sector_not_enabled"


# ------------------------------------------------------------------ #
# --no-fetch-script with missing --script
# ------------------------------------------------------------------ #

class TestNoFetchScript:
    def test_errors_cleanly_when_script_absent(self, tmp_path, capsys):
        missing = tmp_path / "nonexistent.sh"
        rc = main([
            "--sector", "2",
            "--script", str(missing),
            "--no-fetch-script",
            "--dest", str(tmp_path / "cache"),
            "--registry", str(tmp_path / "sectors.json"),
        ])
        assert rc == 2
        captured = capsys.readouterr()
        out = json.loads(captured.out.strip())
        assert out["status"] == "error"
        assert out["reason"] == "script_absent"

    def test_no_network_called(self, tmp_path, monkeypatch, capsys):
        """Verify _fetch_bulk_script is never called when --no-fetch-script."""
        import discovery_download as dd

        def _never_called(*a, **kw):
            raise AssertionError("_fetch_bulk_script must not be called")

        monkeypatch.setattr(dd, "_fetch_bulk_script", _never_called)
        missing = tmp_path / "nonexistent.sh"
        rc = main([
            "--sector", "2",
            "--script", str(missing),
            "--no-fetch-script",
            "--dest", str(tmp_path / "cache"),
            "--registry", str(tmp_path / "sectors.json"),
        ])
        assert rc == 2


# ------------------------------------------------------------------ #
# _head_content_length — best-effort HEAD probe (fail-open)
# ------------------------------------------------------------------ #

class TestHeadContentLength:
    def test_returns_int_from_content_length_header(self, monkeypatch):
        import discovery_download as dd

        class _FakeResp:
            headers = {"Content-Length": "12345"}

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        monkeypatch.setattr(
            dd.urllib.request, "urlopen",
            lambda req, timeout=20.0: _FakeResp())
        assert dd._head_content_length("https://example.com/x.sh") == 12345

    def test_fails_open_to_none_on_network_error(self, monkeypatch):
        """Any HEAD failure (network, timeout, bad response) must not raise —
        the bulk-script download must never be blocked by this probe."""
        import discovery_download as dd

        def _raise(*a, **kw):
            raise OSError("network unreachable")

        monkeypatch.setattr(dd.urllib.request, "urlopen", _raise)
        assert dd._head_content_length("https://example.com/x.sh") is None

    def test_fails_open_to_none_when_header_missing(self, monkeypatch):
        import discovery_download as dd

        class _FakeResp:
            headers = {}

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        monkeypatch.setattr(
            dd.urllib.request, "urlopen",
            lambda req, timeout=20.0: _FakeResp())
        assert dd._head_content_length("https://example.com/x.sh") is None


# ------------------------------------------------------------------ #
# _fetch_bulk_script — curl'd with a live progress monitor (no more silent
# subprocess.run(capture_output=True) for tens of minutes on a big sector).
# ------------------------------------------------------------------ #

class TestFetchBulkScript:
    def test_success_emits_progress_and_renames(self, tmp_path, monkeypatch, capsys):
        import discovery_download as dd

        dest = tmp_path / "s0014_full.sh"
        part = dest.with_suffix(dest.suffix + ".part")

        monkeypatch.setattr(dd, "_head_content_length",
                             lambda url, timeout=20.0: 300)
        # Speed up the test — the real code polls every 5s.
        monkeypatch.setattr(dd, "_BULK_FETCH_POLL_INTERVAL_S", 0.0)

        class _FakeCurlProc:
            """Stands in for the real curl subprocess: each poll() call grows
            the .part file a bit further, then reports rc=0 on the 3rd call."""

            def __init__(self, cmd, stdout=None, stderr=None):
                self._calls = 0
                self.returncode = None

            def poll(self):
                self._calls += 1
                if self._calls == 1:
                    part.write_bytes(b"x" * 100)
                    return None
                if self._calls == 2:
                    part.write_bytes(b"x" * 300)
                    return None
                self.returncode = 0
                return 0

        monkeypatch.setattr(dd.subprocess, "Popen", _FakeCurlProc)

        dd._fetch_bulk_script("https://example.com/s0014.sh", dest)

        # Final file lands at dest_path (not .part) with the fetched bytes.
        assert dest.exists()
        assert not part.exists()
        assert dest.read_bytes() == b"x" * 300

        raw = capsys.readouterr().out.strip()
        emits = [json.loads(l) for l in raw.splitlines() if l.strip()]
        statuses = [e["status"] for e in emits]

        assert statuses[0] == "bulk_fetch_start"
        assert emits[0]["bytes_total"] == 300
        assert statuses[-1] == "bulk_fetch_done"
        assert emits[-1]["bytes_done"] == 300
        # Progress emits track the growing .part size, in order.
        progress = [e for e in emits if e["status"] == "bulk_fetch"]
        assert [p["bytes_done"] for p in progress] == [100, 300]
        assert all(p["bytes_total"] == 300 for p in progress)

    def test_failure_raises_runtime_error_with_stderr_text(
        self, tmp_path, monkeypatch
    ):
        import discovery_download as dd

        dest = tmp_path / "s0014_full.sh"
        monkeypatch.setattr(dd, "_head_content_length",
                             lambda url, timeout=20.0: None)
        monkeypatch.setattr(dd, "_BULK_FETCH_POLL_INTERVAL_S", 0.0)

        class _FakeFailingCurlProc:
            """Writes curl's error to the stderr file handle it was given
            (mirrors curl writing to its inherited stderr fd) then exits 22."""

            def __init__(self, cmd, stdout=None, stderr=None):
                self._stderr = stderr
                self.returncode = None

            def poll(self):
                self._stderr.write(
                    b"curl: (22) The requested URL returned error: 404")
                self._stderr.flush()
                self.returncode = 22
                return 22

        monkeypatch.setattr(dd.subprocess, "Popen", _FakeFailingCurlProc)

        with pytest.raises(RuntimeError, match="404"):
            dd._fetch_bulk_script("https://example.com/missing.sh", dest)

        assert not dest.exists()


# ------------------------------------------------------------------ #
# Non-dry-run: sliced script written correctly (no real download)
# ------------------------------------------------------------------ #

class TestSlicedScriptContent:
    """Verify that the sliced .sh file gets the right curl lines.

    We stub the subprocess calls so no real download or indexing happens.
    The test creates a synthetic bulk script, pre-creates some files so only
    a subset needs downloading, then checks the slice file content.
    """

    def test_sliced_script_contains_only_absent_jobs(
        self, tmp_path, monkeypatch, capsys
    ):
        import discovery_download as dd

        script = _make_curl_script(tmp_path, n=5)
        dest = tmp_path / "cache"
        dest.mkdir()

        # Pre-create the first 2 jobs AND index them — since 2026-07-14 only an
        # INDEXED file counts as downloaded (an on-disk-but-unindexed file is
        # deliberately re-selected so it can be healed).
        from download_tglc_batch import parse as _parse
        jobs = _parse(script)
        _pre_create_files(dest, [jobs[0][0], jobs[1][0]])
        idx = _make_index(tmp_path, [jobs[0][0], jobs[1][0]])

        captured_slice = {}

        def fake_subprocess_run(cmd, **kwargs):
            # Capture the slice path from the download call.
            if "download_tglc_batch.py" in " ".join(cmd):
                script_arg = cmd[cmd.index("--script") + 1]
                captured_slice["path"] = script_arg
                captured_slice["content"] = Path(script_arg).read_text()
                # Return success; mimic stdout of download_tglc_batch.
                class FakeResult:
                    returncode = 0
                    stdout = "done=3 skip=0 fail=0"
                    stderr = ""
                return FakeResult()
            if "index_tglc_batch.py" in " ".join(cmd):
                # Return success; mimic stdout of index_tglc_batch.
                class FakeResult:
                    returncode = 0
                    stdout = "index: 3 (tic,sector) rows\nfiles_total=3\n"
                    stderr = ""
                return FakeResult()
            raise AssertionError(f"Unexpected subprocess call: {cmd}")

        monkeypatch.setattr(dd.subprocess, "run", fake_subprocess_run)

        rc = dd.main([
            "--sector", "2",
            "--script", str(script),
            "--dest", str(dest),
            "--registry", str(tmp_path / "sectors.json"),
            "--index", str(idx),
            "--workers", "4",
        ])

        assert rc == 0, f"main() returned {rc}"
        assert "path" in captured_slice, "download_tglc_batch.py was not called"

        # The sliced script should contain exactly 3 curl lines (jobs 2,3,4).
        lines = [
            l for l in captured_slice["content"].splitlines()
            if l.startswith("curl")
        ]
        assert len(lines) == 3
        # URLs/relpaths for jobs 2, 3, 4 must be in the slice.
        for rel, url in jobs[2:]:
            assert url in captured_slice["content"]
            assert rel in captured_slice["content"]
        # Jobs 0 and 1 (already downloaded) must NOT be in the slice.
        for rel, url in jobs[:2]:
            assert url not in captured_slice["content"]

    def test_download_failure_returns_error_json(
        self, tmp_path, monkeypatch, capsys
    ):
        import discovery_download as dd

        script = _make_curl_script(tmp_path, n=3)
        dest = tmp_path / "cache"
        dest.mkdir()

        def fake_subprocess_run(cmd, **kwargs):
            if "download_tglc_batch.py" in " ".join(cmd):
                class FakeResult:
                    returncode = 1
                    stdout = ""
                    stderr = "connection refused"
                return FakeResult()
            raise AssertionError(f"Should not reach index: {cmd}")

        monkeypatch.setattr(dd.subprocess, "run", fake_subprocess_run)

        rc = dd.main([
            "--sector", "2",
            "--script", str(script),
            "--dest", str(dest),
            "--registry", str(tmp_path / "sectors.json"),
            "--index", str(tmp_path / "tglc.db"),
        ])
        assert rc == 1
        # Multiple JSON lines now (skip_set marker + result): parse the last.
        raw = capsys.readouterr().out.strip()
        out = json.loads([l for l in raw.splitlines() if l.strip()][-1])
        assert out["status"] == "download_failed"

    def test_index_failure_returns_error_json(
        self, tmp_path, monkeypatch, capsys
    ):
        import discovery_download as dd

        script = _make_curl_script(tmp_path, n=3)
        dest = tmp_path / "cache"
        dest.mkdir()

        def fake_subprocess_run(cmd, **kwargs):
            if "download_tglc_batch.py" in " ".join(cmd):
                class FakeResult:
                    returncode = 0
                    stdout = "done=3 skip=0 fail=0"
                    stderr = ""
                return FakeResult()
            if "index_tglc_batch.py" in " ".join(cmd):
                class FakeResult:
                    returncode = 1
                    stdout = ""
                    stderr = "astropy not found"
                return FakeResult()
            raise AssertionError(f"Unexpected: {cmd}")

        monkeypatch.setattr(dd.subprocess, "run", fake_subprocess_run)

        rc = dd.main([
            "--sector", "2",
            "--script", str(script),
            "--dest", str(dest),
            "--registry", str(tmp_path / "sectors.json"),
            "--index", str(tmp_path / "tglc.db"),
        ])
        assert rc == 1
        # discovery_download.py now emits multiple JSON lines (index_start marker
        # + the final result).  Parse the last non-empty JSON line.
        raw = capsys.readouterr().out.strip()
        out = json.loads([l for l in raw.splitlines() if l.strip()][-1])
        assert out["status"] == "index_failed"

    def test_done_output_json(self, tmp_path, monkeypatch, capsys):
        """Happy path: status=done with correct counts."""
        import discovery_download as dd

        script = _make_curl_script(tmp_path, n=4)
        dest = tmp_path / "cache"
        dest.mkdir()

        def fake_subprocess_run(cmd, **kwargs):
            if "download_tglc_batch.py" in " ".join(cmd):
                class FakeResult:
                    returncode = 0
                    stdout = "done=4 skip=0 fail=0"
                    stderr = ""
                return FakeResult()
            if "index_tglc_batch.py" in " ".join(cmd):
                class FakeResult:
                    returncode = 0
                    stdout = "index: 4 (tic,sector) rows\nfiles_total=4\n"
                    stderr = ""
                return FakeResult()
            raise AssertionError(f"Unexpected: {cmd}")

        monkeypatch.setattr(dd.subprocess, "run", fake_subprocess_run)

        rc = dd.main([
            "--sector", "2",
            "--script", str(script),
            "--dest", str(dest),
            "--registry", str(tmp_path / "sectors.json"),
            "--index", str(tmp_path / "tglc.db"),
        ])
        assert rc == 0
        # discovery_download.py now emits multiple JSON lines (index_start marker
        # + the final result).  Parse the last non-empty JSON line.
        raw = capsys.readouterr().out.strip()
        out = json.loads([l for l in raw.splitlines() if l.strip()][-1])
        assert out["status"] == "done"
        assert out["downloaded"] == 4
        assert out["failed"] == 0
        assert out["indexed"] == 4

    def test_registry_uses_union_of_tglc_and_fits_blobs(
        self, tmp_path, monkeypatch, capsys
    ):
        """Registry 'downloaded' = size of tglc ∪ fits_blobs basename union.

        When fits_blobs is absent (older deployment) the union degrades to the
        bare tglc count — still correct.  When fits_blobs contains extra
        basenames (Gaia-ID duplicates packed before the tglc REPLACE discarded
        them) those are counted too, eliminating the phantom pending residual.

        This test uses a DB without fits_blobs so downloaded == tglc count (7),
        proving the tglc fallback works and sector-scoping is preserved
        (a sector-1 row must not inflate the sector-2 count).
        """
        import json as _json
        import sqlite3
        import discovery_download as dd

        script = _make_curl_script(tmp_path, n=2)
        dest = tmp_path / "cache"
        dest.mkdir()
        registry = tmp_path / "sectors.json"
        index = tmp_path / "tglc.db"

        # Pre-populate the tglc index with 7 sector-2 rows (and a sector-1 row to
        # prove the count is sector-scoped, not a total).
        con = sqlite3.connect(index)
        con.execute("CREATE TABLE tglc (tic_id TEXT, sector INTEGER, path TEXT, "
                    "exptime REAL, tmag REAL, ra REAL, dec REAL, "
                    "PRIMARY KEY(tic_id,sector))")
        con.executemany("INSERT INTO tglc(tic_id,sector,path) VALUES(?,?,?)",
                        [(str(i), 2, f"/x/s2_{i}.fits") for i in range(7)]
                        + [("999", 1, "/x/s1.fits")])
        con.commit()
        con.close()

        def fake_subprocess_run(cmd, **kwargs):
            if "download_tglc_batch.py" in " ".join(cmd):
                class FakeResult:
                    returncode = 0
                    stdout = "done=2 skip=0 fail=0"
                    stderr = ""
                return FakeResult()
            if "index_tglc_batch.py" in " ".join(cmd):
                # The batch is indexed via --files-from (not a full-cache glob).
                assert "--files-from" in cmd, (
                    "index must be invoked with --files-from for the batch"
                )
                class FakeResult:
                    returncode = 0
                    stdout = "index: 7 (tic,sector) rows\nfiles_total=99\n"
                    stderr = ""
                return FakeResult()
            raise AssertionError(f"Unexpected: {cmd}")

        monkeypatch.setattr(dd.subprocess, "run", fake_subprocess_run)

        rc = dd.main([
            "--sector", "2",
            "--script", str(script),
            "--dest", str(dest),
            "--registry", str(registry),
            "--index", str(index),
        ])
        assert rc == 0

        # downloaded = union size = 7 tglc rows (fits_blobs absent → guard
        # falls through; files_total=99 from the index subprocess is ignored).
        reg = _json.loads(registry.read_text())
        assert reg["sectors"]["2"]["downloaded"] == 7, (
            f"Expected downloaded=7 (tglc ∪ fits_blobs union, no blobs table), got: {reg}"
        )

    def test_nothing_to_download_returns_done(self, tmp_path, monkeypatch, capsys):
        """If all jobs are already present, return done without subprocess calls."""
        import discovery_download as dd

        script = _make_curl_script(tmp_path, n=3)
        dest = tmp_path / "cache"
        dest.mkdir()

        # Pre-create all files AND index them: "already present" means indexed.
        from download_tglc_batch import parse as _parse
        jobs = _parse(script)
        _pre_create_files(dest, [r for r, _ in jobs])
        idx = _make_index(tmp_path, [r for r, _ in jobs])

        def should_not_be_called(*a, **kw):
            raise AssertionError("subprocess.run must not be called")

        monkeypatch.setattr(dd.subprocess, "run", should_not_be_called)

        rc = dd.main([
            "--sector", "2",
            "--script", str(script),
            "--dest", str(dest),
            "--registry", str(tmp_path / "sectors.json"),
            "--index", str(idx),
        ])
        assert rc == 0
        # Multiple JSON lines now (skip_set marker + result): parse the last.
        raw = capsys.readouterr().out.strip()
        out = json.loads([l for l in raw.splitlines() if l.strip()][-1])
        assert out["status"] == "done"
        assert out["downloaded"] == 0


# ------------------------------------------------------------------ #
# Fix 1: --progress-log flag routes per-file progress into the server log
# ------------------------------------------------------------------ #

class TestProgressLogFlag:
    """Verify that --progress-log is forwarded as --log to download_tglc_batch.py."""

    def test_progress_log_passed_to_batch_subprocess(self, tmp_path, monkeypatch, capsys):
        """When --progress-log is given, the batch subprocess receives --log <path>."""
        import discovery_download as dd

        script = _make_curl_script(tmp_path, n=3)
        dest = tmp_path / "cache"
        dest.mkdir()
        progress_log = tmp_path / "server_managed.log"

        captured_log_arg = {}

        def fake_subprocess_run(cmd, **kwargs):
            if "download_tglc_batch.py" in " ".join(str(x) for x in cmd):
                # Find --log argument
                if "--log" in cmd:
                    captured_log_arg["path"] = cmd[cmd.index("--log") + 1]
                class FakeResult:
                    returncode = 0
                    stdout = "done=3 skip=0 fail=0"
                    stderr = ""
                return FakeResult()
            if "index_tglc_batch.py" in " ".join(str(x) for x in cmd):
                class FakeResult:
                    returncode = 0
                    stdout = "index: 3 (tic,sector) rows\nfiles_total=3\n"
                    stderr = ""
                return FakeResult()
            raise AssertionError(f"Unexpected subprocess call: {cmd}")

        monkeypatch.setattr(dd.subprocess, "run", fake_subprocess_run)

        rc = dd.main([
            "--sector", "2",
            "--script", str(script),
            "--dest", str(dest),
            "--registry", str(tmp_path / "sectors.json"),
            "--index", str(tmp_path / "tglc.db"),
            "--progress-log", str(progress_log),
        ])

        assert rc == 0
        assert "path" in captured_log_arg, "--log was not found in batch subprocess argv"
        assert captured_log_arg["path"] == str(progress_log), (
            f"Expected --log {progress_log}, got {captured_log_arg['path']!r}"
        )

    def test_no_progress_log_uses_auto_generated_path(self, tmp_path, monkeypatch, capsys):
        """When --progress-log is absent, --log is an auto-generated path inside dest."""
        import discovery_download as dd

        script = _make_curl_script(tmp_path, n=3)
        dest = tmp_path / "cache"
        dest.mkdir()

        captured_log_arg = {}

        def fake_subprocess_run(cmd, **kwargs):
            if "download_tglc_batch.py" in " ".join(str(x) for x in cmd):
                if "--log" in cmd:
                    captured_log_arg["path"] = cmd[cmd.index("--log") + 1]
                class FakeResult:
                    returncode = 0
                    stdout = "done=3 skip=0 fail=0"
                    stderr = ""
                return FakeResult()
            if "index_tglc_batch.py" in " ".join(str(x) for x in cmd):
                class FakeResult:
                    returncode = 0
                    stdout = "index: 3 (tic,sector) rows\nfiles_total=3\n"
                    stderr = ""
                return FakeResult()
            raise AssertionError(f"Unexpected subprocess call: {cmd}")

        monkeypatch.setattr(dd.subprocess, "run", fake_subprocess_run)

        rc = dd.main([
            "--sector", "2",
            "--script", str(script),
            "--dest", str(dest),
            "--registry", str(tmp_path / "sectors.json"),
            "--index", str(tmp_path / "tglc.db"),
            # no --progress-log
        ])

        assert rc == 0
        assert "path" in captured_log_arg
        log_path = Path(captured_log_arg["path"])
        # Auto-generated log must be inside dest and match download_<TS>.log pattern.
        assert log_path.parent == dest, (
            f"Auto log expected inside {dest}, got {log_path.parent}"
        )
        assert log_path.name.startswith("download_"), (
            f"Auto log name should start with 'download_', got {log_path.name!r}"
        )


# ------------------------------------------------------------------ #
# Fix 2: parse_iter — streaming generator, no full materialisation
# ------------------------------------------------------------------ #

class TestParseIter:
    """Verify parse_iter yields correct tuples without loading the whole file."""

    def test_parse_iter_yields_correct_tuples(self, tmp_path):
        """Small script → parse_iter yields same (relpath, url) as parse()."""
        from download_tglc_batch import parse as _parse_full

        script = _make_curl_script(tmp_path, n=10)
        expected = _parse_full(script)

        result = list(_parse_iter(script))
        assert result == expected

    def test_parse_iter_is_a_generator(self, tmp_path):
        """parse_iter returns a generator (not a list)."""
        import types
        script = _make_curl_script(tmp_path, n=5)
        gen = _parse_iter(script)
        assert isinstance(gen, types.GeneratorType), (
            f"parse_iter must return a generator, got {type(gen)}"
        )

    def test_parse_iter_can_be_stopped_early(self, tmp_path):
        """Iterating only N items from parse_iter does not require reading the whole file."""
        script = _make_curl_script(tmp_path, n=10)
        gen = _parse_iter(script)
        # Take only 3
        first_three = [next(gen) for _ in range(3)]
        assert len(first_three) == 3
        # Each is a 2-tuple
        for relpath, url in first_three:
            assert relpath.endswith(".fits")
            assert url.startswith("https://")

    def test_parse_iter_skips_non_curl_lines(self, tmp_path):
        """Non-curl lines (shebang, comments, blank) are skipped."""
        script = tmp_path / "mixed.sh"
        script.write_text(
            "#!/bin/sh\n"
            "# comment line\n"
            "\n"
            "curl -f --create-dirs --output 'a/b.fits' 'https://x.invalid/b.fits'\n"
            "echo done\n"
            "curl -f --create-dirs --output 'c/d.fits' 'https://x.invalid/d.fits'\n"
        )
        result = list(_parse_iter(script))
        assert len(result) == 2
        assert result[0] == ("a/b.fits", "https://x.invalid/b.fits")
        assert result[1] == ("c/d.fits", "https://x.invalid/d.fits")


# ------------------------------------------------------------------ #
# Fix 2: select_next_jobs early-exit — only scans as far as needed
# ------------------------------------------------------------------ #

class TestSelectNextJobsEarlyExit:
    """Verify select_next_jobs stops scanning as soon as count batch items found.

    With the parallel-stat implementation the generator is consumed in chunks
    (_chunk items at a time).  Early-exit is chunk-level: once count absent
    jobs are found within a chunk the remaining results of that chunk are
    discarded and no further chunks are pulled from the generator.  Tests use
    _chunk=5 so the chunk boundary is visible with small inputs.
    """

    def _make_counting_generator(self, n: int):
        """Return (generator, counter_list) where counter[0] tracks how many items yielded."""
        counter = [0]

        def _gen():
            for i in range(n):
                counter[0] += 1
                yield (f"s0001/cam{i}.fits", f"https://x.invalid/{i}.fits")

        return _gen(), counter

    def test_early_exit_when_count_reached(self, tmp_path):
        """With count=3, chunk=5 and 1000 total absent jobs, only one chunk is scanned.

        The implementation processes chunks of _chunk items in parallel.  When
        count absent jobs are found within the first chunk the generator is not
        advanced further.  With _chunk=5 and count=3 (all absent), exactly 5
        items are consumed from the generator (the first chunk) — no second
        chunk is pulled.
        """
        gen, counter = self._make_counting_generator(1000)
        already, batch = select_next_jobs(gen, tmp_path, count=3, _chunk=5)
        assert len(batch) == 3
        assert already == 0
        # Exactly one chunk of 5 items is consumed — no second chunk needed.
        assert counter[0] == 5, (
            f"Expected 5 items consumed (one chunk of _chunk=5), got {counter[0]}"
        )

    def test_early_exit_with_some_already_present(self, tmp_path):
        """Already-present files are counted; scan stops at the chunk boundary.

        With _chunk=5 and count=1: the first chunk (items 0-4) is pulled.
        Items 0 and 1 are pre-created (present), item 2 is absent → batch
        full → early-exit within the first chunk.  The generator yields exactly
        5 items (one full chunk), no second chunk is consumed.
        """
        relpaths = [f"s0001/cam{i}.fits" for i in range(10)]
        for rel in relpaths[:2]:
            p = tmp_path / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b"\x00" * 5000)

        gen, counter = self._make_counting_generator(1000)
        already, batch = select_next_jobs(gen, tmp_path, count=1, _chunk=5)
        assert already == 2
        assert len(batch) == 1
        # One chunk of 5 items consumed; early-exit fired inside the chunk.
        assert counter[0] == 5, (
            f"Expected 5 items consumed (one chunk of _chunk=5), got {counter[0]}"
        )

    def test_accepts_list_as_well_as_generator(self, tmp_path):
        """select_next_jobs accepts a plain list (existing callers must keep working)."""
        jobs = [(f"s0001/cam{i}.fits", f"https://x.invalid/{i}.fits") for i in range(6)]
        already, batch = select_next_jobs(jobs, tmp_path, count=3)
        assert len(batch) == 3
        assert already == 0

    def test_no_early_exit_when_fewer_absent_than_count(self, tmp_path):
        """If fewer absent items exist than count, scans all (no premature stop)."""
        gen, counter = self._make_counting_generator(4)
        already, batch = select_next_jobs(gen, tmp_path, count=10)
        # All 4 items are absent, count=10 → no early-exit, scans all 4.
        assert len(batch) == 4
        assert counter[0] == 4


# ------------------------------------------------------------------ #
# New tests: parallel-stat correctness (chunk boundary, MIN_FILE_BYTES)
# ------------------------------------------------------------------ #

class TestSelectNextJobsParallelStat:
    """Tests covering the parallel-stat implementation specifics.

    These tests use _chunk and _workers parameters to control chunk size and
    thread-pool size, making it easy to reason about chunk-boundary behaviour
    with small inputs.
    """

    def _make_jobs(self, n: int):
        return [
            (f"s0001/cam{i}.fits", f"https://x.invalid/{i}.fits")
            for i in range(n)
        ]

    def test_early_exit_mid_chunk_returns_same_as_sequential(self, tmp_path):
        """Early-exit within a chunk returns bit-identical (already, batch) to sequential.

        Reference sequential scan and parallel scan with _chunk=4 (so the
        early-exit fires within the first chunk for count=2 with some present).
        """
        jobs = self._make_jobs(20)

        # Pre-create jobs[0] and jobs[2] (present); jobs[1], jobs[3] absent.
        for idx in (0, 2):
            p = tmp_path / jobs[idx][0]
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b"\x00" * 5000)

        # Reference: sequential logic (in-process, no thread pool).
        from discovery_download import MIN_FILE_BYTES
        already_ref = 0
        batch_ref = []
        for relpath, url in jobs:
            out = tmp_path / relpath
            if out.exists() and out.stat().st_size > MIN_FILE_BYTES - 1:
                already_ref += 1
            else:
                batch_ref.append((relpath, url))
                if len(batch_ref) == 2:
                    break

        # Parallel with _chunk=4, _workers=2.
        already_par, batch_par = select_next_jobs(
            jobs, tmp_path, count=2, _chunk=4, _workers=2
        )

        assert already_par == already_ref, (
            f"already mismatch: parallel={already_par}, sequential={already_ref}"
        )
        assert batch_par == batch_ref, (
            f"batch mismatch: parallel={batch_par}, sequential={batch_ref}"
        )

    def test_cross_chunk_boundary_preserves_order(self, tmp_path):
        """Batch items from across chunk boundaries keep the original order.

        With _chunk=3 and 9 total jobs, the generator yields 3 chunks.  Making
        job[2] present (last of chunk 0) and job[5] present (last of chunk 1)
        means absent items span both chunks.  Order must be identical to a
        sequential scan.
        """
        jobs = self._make_jobs(9)

        # Make jobs[2] and jobs[5] present.
        for idx in (2, 5):
            p = tmp_path / jobs[idx][0]
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b"\x00" * 5000)

        already, batch = select_next_jobs(
            jobs, tmp_path, count=100, _chunk=3, _workers=2
        )

        # Expected: jobs[2] and jobs[5] present, the other 7 absent in order.
        expected_batch = [j for j in jobs if j not in (jobs[2], jobs[5])]
        assert already == 2
        assert batch == expected_batch, (
            f"Order mismatch across chunk boundary.\n"
            f"Expected: {expected_batch}\nGot:      {batch}"
        )

    def test_min_file_bytes_threshold_respected_parallel(self, tmp_path):
        """Files at or below 1000 bytes are treated as absent (parallel mirrors sequential).

        MIN_FILE_BYTES = 1001, so a file of exactly 1000 bytes is NOT counted
        as downloaded.  This must hold in the parallel path as well.
        """
        jobs = self._make_jobs(4)

        # job[0]: 1000 bytes — NOT present (below threshold).
        p0 = tmp_path / jobs[0][0]
        p0.parent.mkdir(parents=True, exist_ok=True)
        p0.write_bytes(b"\x00" * 1000)

        # job[1]: 1001 bytes — IS present (at or above threshold).
        p1 = tmp_path / jobs[1][0]
        p1.parent.mkdir(parents=True, exist_ok=True)
        p1.write_bytes(b"\x00" * 1001)

        already, batch = select_next_jobs(
            jobs, tmp_path, count=10, _chunk=4, _workers=2
        )

        # job[0] is tiny → absent (in batch), job[1] is large enough → already.
        assert already == 1, f"Expected 1 already-present, got {already}"
        # batch should contain jobs[0], jobs[2], jobs[3] (not jobs[1]).
        batch_relpaths = [r for r, _ in batch]
        assert jobs[0][0] in batch_relpaths, "Tiny file must appear in batch (not present)"
        assert jobs[1][0] not in batch_relpaths, "1001-byte file must NOT appear in batch"
        assert jobs[2][0] in batch_relpaths
        assert jobs[3][0] in batch_relpaths

    def test_second_chunk_not_consumed_after_early_exit(self, tmp_path):
        """Generator is not advanced past the chunk that satisfies count.

        With _chunk=5, count=3, and all jobs absent, the first chunk (5 items)
        yields 3 absent jobs → early-exit.  The second chunk (5 more items)
        must NOT be pulled from the generator.
        """
        counter = [0]

        def _counting_gen():
            for i in range(20):
                counter[0] += 1
                yield (f"s0001/cam{i}.fits", f"https://x.invalid/{i}.fits")

        already, batch = select_next_jobs(
            _counting_gen(), tmp_path, count=3, _chunk=5, _workers=2
        )

        assert len(batch) == 3
        assert already == 0
        # Only the first chunk (5 items) was consumed; the second was not needed.
        assert counter[0] == 5, (
            f"Expected 5 items consumed (one chunk), got {counter[0]}"
        )


# ------------------------------------------------------------------ #
# _indexed_basenames: tglc ∪ fits_blobs union
# ------------------------------------------------------------------ #

def _make_tglc_db(tmp_path: Path, sector: int, tic_paths: list) -> Path:
    """Create a minimal tglc index DB with the given (tic_id, path) rows."""
    import sqlite3
    db = tmp_path / "tglc.db"
    con = sqlite3.connect(db)
    con.execute(
        "CREATE TABLE tglc (tic_id TEXT, sector INTEGER, path TEXT, "
        "exptime REAL, tmag REAL, ra REAL, dec REAL, "
        "PRIMARY KEY(tic_id,sector))"
    )
    con.executemany(
        "INSERT INTO tglc(tic_id,sector,path) VALUES(?,?,?)",
        [(str(tic), sector, path) for tic, path in tic_paths],
    )
    con.commit()
    con.close()
    return db


class TestIndexedBasenames:
    """Unit tests for _indexed_basenames (tglc ∪ fits_blobs union logic)."""

    def test_returns_tglc_basenames_without_fits_blobs_table(self, tmp_path):
        """Without a fits_blobs table, returns only the tglc basenames (guard)."""
        db = _make_tglc_db(tmp_path, 23, [
            (1, "/cache/s0023/a/hlsp_tglc_tess_ffi_gaiaid-100-s0023-cam1-ccd1_tess_v1_llc.fits"),
            (2, "/cache/s0023/b/hlsp_tglc_tess_ffi_gaiaid-200-s0023-cam1-ccd1_tess_v1_llc.fits"),
        ])
        names = _indexed_basenames(str(db), 23)
        assert len(names) == 2
        assert "hlsp_tglc_tess_ffi_gaiaid-100-s0023-cam1-ccd1_tess_v1_llc.fits" in names
        assert "hlsp_tglc_tess_ffi_gaiaid-200-s0023-cam1-ccd1_tess_v1_llc.fits" in names

    def test_union_includes_fits_blobs_not_in_tglc(self, tmp_path):
        """Basenames in fits_blobs but absent from tglc are included in the union.

        This covers the duplicate-Gaia-ID collapse: when two Gaia IDs map to
        the same TIC, tglc's INSERT OR REPLACE keeps only one basename, but
        fits_blobs retains both.  The union must include the discarded one.
        """
        import sqlite3

        # TIC 42 has two Gaia IDs → tglc only survived gaiaid-WINNER (INSERT OR
        # REPLACE) but fits_blobs still has both basenames.
        winner_name = "hlsp_tglc_tess_ffi_gaiaid-winner-s0023-cam1-ccd1_tess_v1_llc.fits"
        loser_name = "hlsp_tglc_tess_ffi_gaiaid-loser-s0023-cam1-ccd1_tess_v1_llc.fits"
        other_name = "hlsp_tglc_tess_ffi_gaiaid-solo-s0023-cam2-ccd2_tess_v1_llc.fits"

        db = _make_tglc_db(tmp_path, 23, [
            (42, f"/cache/s0023/a/{winner_name}"),   # winner survived
            (99, f"/cache/s0023/b/{other_name}"),    # unrelated TIC
        ])
        # fits_blobs has both: the winner AND the loser (and the unrelated).
        con = sqlite3.connect(db)
        con.execute(
            "CREATE TABLE fits_blobs "
            "(filename TEXT PRIMARY KEY, container_key TEXT NOT NULL, "
            "data_offset INTEGER NOT NULL, data_length INTEGER NOT NULL)"
        )
        con.executemany(
            "INSERT INTO fits_blobs(filename,container_key,data_offset,data_length) "
            "VALUES(?,?,?,?)",
            [
                (winner_name, "containers/s0023/dl_1.tar", 0, 10000),
                (loser_name, "containers/s0023/dl_1.tar", 10000, 10000),
                (other_name, "containers/s0023/dl_1.tar", 20000, 10000),
            ],
        )
        con.commit()
        con.close()

        names = _indexed_basenames(str(db), 23)
        # Union must contain all three: winner (from both), loser (blobs only),
        # and the unrelated TIC (both).
        assert winner_name in names, "tglc winner must be in the union"
        assert loser_name in names, "fits_blobs loser must be in the union"
        assert other_name in names, "unrelated TIC must be in the union"
        assert len(names) == 3

    def test_sector_scoped_blob_pattern(self, tmp_path):
        """fits_blobs LIKE filter only picks basenames for the requested sector."""
        import sqlite3

        s23_name = "hlsp_tglc_tess_ffi_gaiaid-A-s0023-cam1-ccd1_tess_v1_llc.fits"
        s01_name = "hlsp_tglc_tess_ffi_gaiaid-B-s0001-cam1-ccd1_tess_v1_llc.fits"

        # tglc is empty for sector 23 (so the result must come from fits_blobs).
        db = _make_tglc_db(tmp_path, 99, [])  # seed an unrelated sector row
        con = sqlite3.connect(db)
        con.execute(
            "CREATE TABLE fits_blobs "
            "(filename TEXT PRIMARY KEY, container_key TEXT NOT NULL, "
            "data_offset INTEGER NOT NULL, data_length INTEGER NOT NULL)"
        )
        con.executemany(
            "INSERT INTO fits_blobs(filename,container_key,data_offset,data_length) "
            "VALUES(?,?,?,?)",
            [
                (s23_name, "containers/s0023/dl_1.tar", 0, 5000),
                (s01_name, "containers/s0001/dl_1.tar", 0, 5000),
            ],
        )
        con.commit()
        con.close()

        names23 = _indexed_basenames(str(db), 23)
        assert s23_name in names23, "sector-23 blob must appear when querying sector 23"
        assert s01_name not in names23, "sector-1 blob must NOT appear when querying sector 23"

        names1 = _indexed_basenames(str(db), 1)
        assert s01_name in names1, "sector-1 blob must appear when querying sector 1"
        assert s23_name not in names1, "sector-23 blob must NOT appear when querying sector 1"

    def test_nonexistent_db_returns_empty_set(self, tmp_path):
        """Returns empty set gracefully when the DB does not exist."""
        names = _indexed_basenames(str(tmp_path / "nonexistent.db"), 23)
        assert names == set()


# ------------------------------------------------------------------ #
# EXONIT_METADATA_DB_URL respected by discovery_download.py's write path
# (F3 item b — MetadataCache(args.index), used both by _index_and_pack_chunk's
# add_fits_blobs_bulk call and by _indexed_basenames, must honour the flag).
# ------------------------------------------------------------------ #

class TestMetadataDbUrlRespectedInWritePath:
    """EXONIT_METADATA_DB_URL, when set, wins over the ``--index`` SQLite path
    string for every MetadataCache construction on discovery_download.py's
    write path — mirrors F1's own env-var-override guarantee
    (MetadataCache.__init__), verified here specifically for THIS script's
    call sites: ``_indexed_basenames()`` (the skip-set / registry-recompute
    read) and the ``add_fits_blobs_bulk`` construction inside
    ``_index_and_pack_chunk`` (F2 already proved the seam works in general;
    this proves discovery_download.py actually goes through it).
    """

    def test_metadatacache_construction_ignores_sqlite_path_when_url_set(
        self, monkeypatch
    ):
        """Construction alone (no query) picks the PG backend regardless of
        the sqlite-looking path string — PgBackend does not validate
        connectivity synchronously (pool connects lazily), so this needs no
        reachable PG server."""
        from exonit.metadata_cache import MetadataCache

        monkeypatch.setenv(
            "EXONIT_METADATA_DB_URL",
            "postgresql://exonit@192.0.2.1:5432/exonit_meta",  # TEST-NET-1, unreachable by design
        )
        cache = MetadataCache("data/metadata_cache.db")
        try:
            assert cache._is_pg
            assert cache._backend.dialect.name == "postgres"
        finally:
            cache.close()

    def test_indexed_basenames_ignores_sqlite_path_when_url_set_real_pg(
        self, tmp_path, monkeypatch
    ):
        """End-to-end against a real throwaway PG: _indexed_basenames() reads
        from PG, not from the (nonexistent) SQLite path it was given."""
        import psycopg

        from tests._metadata_pg_test_db import pg_meta_test_available, resolve_test_meta_dsn

        dsn = resolve_test_meta_dsn()
        if not pg_meta_test_available(dsn):
            pytest.skip("EXONIT_TEST_META_PG_URL (throwaway PG) unset or not reachable")

        with psycopg.connect(dsn, connect_timeout=5, autocommit=True) as c:
            c.execute("DROP TABLE IF EXISTS tglc CASCADE")
            c.execute("DROP TABLE IF EXISTS fits_blobs CASCADE")
            c.execute(
                "CREATE TABLE tglc (tic_id TEXT NOT NULL, sector BIGINT NOT NULL, "
                "path TEXT NOT NULL, exptime DOUBLE PRECISION, tmag DOUBLE PRECISION, "
                "ra DOUBLE PRECISION, dec DOUBLE PRECISION, "
                "PRIMARY KEY (tic_id, sector))"
            )
            c.execute(
                "CREATE TABLE fits_blobs (filename TEXT PRIMARY KEY, "
                "container_key TEXT NOT NULL, data_offset BIGINT NOT NULL, "
                "data_length BIGINT NOT NULL)"
            )

        monkeypatch.setenv("EXONIT_METADATA_DB_URL", dsn)
        try:
            from exonit.metadata_cache import MetadataCache

            # The write path's exact call shape (_index_and_pack_chunk):
            # MetadataCache(args.index).add_fits_blobs_bulk([...]).
            cache = MetadataCache(str(tmp_path / "unused-path-pg-wins.db"))
            try:
                cache.add_fits_blobs_bulk([
                    ("hlsp_tglc_tess_ffi_gaiaid-1-s0023-cam1-ccd1_tess_v1_llc.fits",
                     "containers/s0023/dl_1.tar", 0, 10000),
                ])
            finally:
                cache.close()

            # The read side (skip-set / --recompute-registry) must see it.
            names = _indexed_basenames(
                str(tmp_path / "unused-path-pg-wins.db"), 23)
            assert "hlsp_tglc_tess_ffi_gaiaid-1-s0023-cam1-ccd1_tess_v1_llc.fits" in names
        finally:
            with psycopg.connect(dsn, connect_timeout=5, autocommit=True) as c:
                c.execute("DROP TABLE IF EXISTS tglc CASCADE")
                c.execute("DROP TABLE IF EXISTS fits_blobs CASCADE")


# ------------------------------------------------------------------ #
# --recompute-registry flag
# ------------------------------------------------------------------ #

class TestRecomputeRegistry:
    """Tests for the --recompute-registry early-exit path."""

    def _make_full_db(self, tmp_path: Path, sector: int,
                      tglc_basenames: list, blob_extra: list) -> Path:
        """Create a DB with tglc rows and extra fits_blobs entries."""
        import sqlite3

        db = tmp_path / "meta.db"
        con = sqlite3.connect(db)
        con.execute(
            "CREATE TABLE tglc (tic_id TEXT, sector INTEGER, path TEXT, "
            "exptime REAL, tmag REAL, ra REAL, dec REAL, "
            "PRIMARY KEY(tic_id,sector))"
        )
        # tglc: one row per basename, path = /x/<basename>
        con.executemany(
            "INSERT INTO tglc(tic_id,sector,path) VALUES(?,?,?)",
            [(str(i), sector, f"/x/{bn}") for i, bn in enumerate(tglc_basenames)],
        )
        con.execute(
            "CREATE TABLE fits_blobs "
            "(filename TEXT PRIMARY KEY, container_key TEXT NOT NULL, "
            "data_offset INTEGER NOT NULL, data_length INTEGER NOT NULL)"
        )
        # fits_blobs: the tglc ones PLUS the extra (duplicate) ones
        all_blobs = tglc_basenames + blob_extra
        con.executemany(
            "INSERT INTO fits_blobs(filename,container_key,data_offset,data_length) "
            "VALUES(?,?,?,?)",
            [(bn, "containers/test.tar", i * 1000, 1000) for i, bn in enumerate(all_blobs)],
        )
        con.commit()
        con.close()
        return db

    def test_recompute_single_sector_updates_registry(self, tmp_path, capsys):
        """--recompute-registry updates downloaded for the given sector."""
        import json as _json

        sector = 23
        tglc_names = [
            f"hlsp_tglc_tess_ffi_gaiaid-{i}-s0023-cam1-ccd1_tess_v1_llc.fits"
            for i in range(5)
        ]
        extra_names = [
            "hlsp_tglc_tess_ffi_gaiaid-dup-s0023-cam1-ccd1_tess_v1_llc.fits"
        ]
        db = self._make_full_db(tmp_path, sector, tglc_names, extra_names)

        registry = tmp_path / "sectors.json"
        # Prime the registry with a stale downloaded value.
        registry.write_text(_json.dumps(
            {"active_sector": 23, "sectors": {"23": {"downloaded": 5, "total_tics": 100}}}
        ))

        rc = main([
            "--sector", "23",
            "--index", str(db),
            "--registry", str(registry),
            "--recompute-registry",
        ])
        assert rc == 0

        raw = capsys.readouterr().out.strip()
        out = _json.loads(raw)
        assert out["status"] == "recompute_done"
        # JSON serialises integer dict keys as strings.
        assert out["sectors"]["23"]["before"] == 5
        assert out["sectors"]["23"]["after"] == 6  # 5 tglc + 1 extra blob

        reg = _json.loads(registry.read_text())
        assert reg["sectors"]["23"]["downloaded"] == 6

    def test_recompute_all_sectors_when_no_sector_given(self, tmp_path, capsys):
        """Without --sector, recomputes all sectors in SECTOR_BULK."""
        import json as _json
        from discovery_download import SECTOR_BULK

        # Empty DB — all sectors return 0.
        db = tmp_path / "empty.db"
        import sqlite3
        con = sqlite3.connect(db)
        con.execute(
            "CREATE TABLE tglc (tic_id TEXT, sector INTEGER, path TEXT, "
            "exptime REAL, tmag REAL, ra REAL, dec REAL, "
            "PRIMARY KEY(tic_id,sector))"
        )
        con.commit()
        con.close()

        registry = tmp_path / "sectors.json"
        rc = main([
            "--index", str(db),
            "--registry", str(registry),
            "--recompute-registry",
        ])
        assert rc == 0

        raw = capsys.readouterr().out.strip()
        out = _json.loads(raw)
        assert out["status"] == "recompute_done"
        # All known sectors must be present in the result.
        reported = {int(k) for k in out["sectors"]}
        assert reported == set(SECTOR_BULK.keys()), (
            f"Expected all SECTOR_BULK sectors, got {reported}"
        )
        for sec_key, vals in out["sectors"].items():
            assert vals["after"] == 0

    def test_recompute_does_not_download_anything(self, tmp_path, monkeypatch, capsys):
        """--recompute-registry must never call subprocess.run."""
        import json as _json
        import discovery_download as dd

        db = tmp_path / "empty.db"
        import sqlite3
        con = sqlite3.connect(db)
        con.execute(
            "CREATE TABLE tglc (tic_id TEXT, sector INTEGER, path TEXT, "
            "exptime REAL, tmag REAL, ra REAL, dec REAL, "
            "PRIMARY KEY(tic_id,sector))"
        )
        con.commit()
        con.close()

        def should_not_be_called(*a, **kw):
            raise AssertionError("subprocess.run must not be called with --recompute-registry")

        monkeypatch.setattr(dd.subprocess, "run", should_not_be_called)

        rc = dd.main([
            "--sector", "1",
            "--index", str(db),
            "--registry", str(tmp_path / "sectors.json"),
            "--recompute-registry",
        ])
        assert rc == 0


# ------------------------------------------------------------------ #
# --recompute-registry change-detection gate (2026-09-25, fix C): skip the
# per-sector indexed_basenames() scan when a cheap PG fingerprint shows
# neither tglc nor fits_blobs changed since the last real recompute.
# ------------------------------------------------------------------ #

class TestRecomputeRegistryFingerprintGate:
    """Unit tests for the fingerprint-based skip in the --recompute-registry
    early exit. The fingerprint itself (MetadataCache.download_registry_
    fingerprint) is monkeypatched here — it is only ever non-None against a
    real PostgreSQL backend, and its own PG-specific behaviour is covered in
    tests/test_metadata_cache_pg_parity.py; these tests exercise the gate
    logic in discovery_download.py's main() in isolation."""

    def _empty_db(self, tmp_path: Path) -> Path:
        db = tmp_path / "empty.db"
        _make_empty_tglc_db(db)
        return db

    def test_non_pg_backend_always_recomputes_even_with_matching_state(
        self, tmp_path, monkeypatch, capsys
    ):
        """download_registry_fingerprint() returning None (SQLite/tests) must
        always recompute, regardless of what a stale state file says — the
        documented fallback."""
        import discovery_download as dd

        monkeypatch.setattr(
            "exonit.metadata_cache.MetadataCache.download_registry_fingerprint",
            lambda self: None)
        state_file = tmp_path / "fp_state.json"
        monkeypatch.setenv("EXONIT_DOWNLOAD_REGISTRY_FINGERPRINT_FILE", str(state_file))
        state_file.write_text(json.dumps(
            {"fingerprint": {"tglc": {"n_live_tup": 1}}, "last_full_recompute": time.time()}))

        rc = dd.main([
            "--sector", "1", "--index", str(self._empty_db(tmp_path)),
            "--registry", str(tmp_path / "sectors.json"),
            "--recompute-registry",
        ])
        assert rc == 0
        out = json.loads(capsys.readouterr().out.strip())
        assert out["status"] == "recompute_done"

    def test_unchanged_fingerprint_within_window_skips_recompute(
        self, tmp_path, monkeypatch, capsys
    ):
        """A matching fingerprint, within the safety-net window, must skip
        the scan entirely (status recompute_skipped) and leave the registry
        untouched."""
        import discovery_download as dd

        fp = {"tglc": {"n_live_tup": 10}, "fits_blobs": {"n_live_tup": 20}}
        monkeypatch.setattr(
            "exonit.metadata_cache.MetadataCache.download_registry_fingerprint",
            lambda self: fp)
        state_file = tmp_path / "fp_state.json"
        monkeypatch.setenv("EXONIT_DOWNLOAD_REGISTRY_FINGERPRINT_FILE", str(state_file))
        state_file.write_text(json.dumps(
            {"fingerprint": fp, "last_full_recompute": time.time()}))

        registry = tmp_path / "sectors.json"
        registry.write_text(json.dumps(
            {"active_sector": 1, "sectors": {"1": {"downloaded": 42, "total_tics": 100}}}))

        rc = dd.main([
            "--sector", "1", "--index", str(self._empty_db(tmp_path)),
            "--registry", str(registry),
            "--recompute-registry",
        ])
        assert rc == 0
        out = json.loads(capsys.readouterr().out.strip())
        assert out["status"] == "recompute_skipped"
        assert out["reason"] == "fingerprint_unchanged"

        # Registry value from before the (skipped) recompute is untouched.
        reg = json.loads(registry.read_text())
        assert reg["sectors"]["1"]["downloaded"] == 42

    def test_changed_fingerprint_forces_recompute(self, tmp_path, monkeypatch, capsys):
        """A fingerprint that differs from the stored one means the tables
        changed — must recompute, not skip."""
        import discovery_download as dd

        monkeypatch.setattr(
            "exonit.metadata_cache.MetadataCache.download_registry_fingerprint",
            lambda self: {"tglc": {"n_live_tup": 999}})
        state_file = tmp_path / "fp_state.json"
        monkeypatch.setenv("EXONIT_DOWNLOAD_REGISTRY_FINGERPRINT_FILE", str(state_file))
        state_file.write_text(json.dumps(
            {"fingerprint": {"tglc": {"n_live_tup": 10}},
             "last_full_recompute": time.time()}))

        rc = dd.main([
            "--sector", "1", "--index", str(self._empty_db(tmp_path)),
            "--registry", str(tmp_path / "sectors.json"),
            "--recompute-registry",
        ])
        assert rc == 0
        out = json.loads(capsys.readouterr().out.strip())
        assert out["status"] == "recompute_done"

        # The new fingerprint is persisted for the next invocation.
        saved = json.loads(state_file.read_text())
        assert saved["fingerprint"] == {"tglc": {"n_live_tup": 999}}

    def test_unchanged_fingerprint_past_max_stale_forces_recompute(
        self, tmp_path, monkeypatch, capsys
    ):
        """Safety net: even a fingerprint match must not skip forever — past
        EXONIT_DOWNLOAD_REGISTRY_MAX_STALE_HOURS a real recompute still
        runs, in case the fingerprint itself is wrong for some reason this
        gate didn't anticipate."""
        import discovery_download as dd

        fp = {"tglc": {"n_live_tup": 10}}
        monkeypatch.setattr(
            "exonit.metadata_cache.MetadataCache.download_registry_fingerprint",
            lambda self: fp)
        monkeypatch.setenv("EXONIT_DOWNLOAD_REGISTRY_MAX_STALE_HOURS", "1")
        state_file = tmp_path / "fp_state.json"
        monkeypatch.setenv("EXONIT_DOWNLOAD_REGISTRY_FINGERPRINT_FILE", str(state_file))
        # Last real recompute 2h ago — past the 1h safety-net ceiling.
        state_file.write_text(json.dumps(
            {"fingerprint": fp, "last_full_recompute": time.time() - 7200.0}))

        rc = dd.main([
            "--sector", "1", "--index", str(self._empty_db(tmp_path)),
            "--registry", str(tmp_path / "sectors.json"),
            "--recompute-registry",
        ])
        assert rc == 0
        out = json.loads(capsys.readouterr().out.strip())
        assert out["status"] == "recompute_done"

    def test_no_state_file_recomputes_and_creates_one(self, tmp_path, monkeypatch, capsys):
        """Cold start (no prior state file): recompute, and persist a state
        file for the next invocation to compare against."""
        import discovery_download as dd

        fp = {"tglc": {"n_live_tup": 5}}
        monkeypatch.setattr(
            "exonit.metadata_cache.MetadataCache.download_registry_fingerprint",
            lambda self: fp)
        state_file = tmp_path / "fp_state.json"
        monkeypatch.setenv("EXONIT_DOWNLOAD_REGISTRY_FINGERPRINT_FILE", str(state_file))
        assert not state_file.exists()

        rc = dd.main([
            "--sector", "1", "--index", str(self._empty_db(tmp_path)),
            "--registry", str(tmp_path / "sectors.json"),
            "--recompute-registry",
        ])
        assert rc == 0
        out = json.loads(capsys.readouterr().out.strip())
        assert out["status"] == "recompute_done"
        assert json.loads(state_file.read_text())["fingerprint"] == fp


# ------------------------------------------------------------------ #
# --registry cross-host write guard (2026-07-25 sector 18 incident):
# --registry defaulted to a bare relative path resolved against the
# process cwd. The download ran on <host> from ~/exonit (its
# own separate clone, at the same path as the server's), so the default
# silently resolved to a real, existing -- but WRONG HOST's -- sectors.json.
# Two independent fixes, both covered here:
#   1. the default is now anchored to this script's own repo root, never cwd.
#   2. writing at that default is refused (loudly) unless this host looks
#      like the production server (exonit.service active) -- an explicit
#      --registry always proceeds, unaffected.
# ------------------------------------------------------------------ #

def _make_empty_tglc_db(path: Path) -> None:
    import sqlite3
    con = sqlite3.connect(path)
    con.execute(
        "CREATE TABLE tglc (tic_id TEXT, sector INTEGER, path TEXT, "
        "exptime REAL, tmag REAL, ra REAL, dec REAL, "
        "PRIMARY KEY(tic_id,sector))"
    )
    con.commit()
    con.close()


class TestIsServerHost:
    """Unit tests for _is_server_host()'s systemctl-based detection."""

    def test_true_when_unit_active(self, monkeypatch):
        import discovery_download as dd

        mock_result = MagicMock()
        mock_result.returncode = 0  # systemctl is-active --quiet: active
        monkeypatch.setattr(dd.subprocess, "run", lambda *a, **k: mock_result)
        assert dd._is_server_host() is True

    def test_false_when_unit_inactive(self, monkeypatch):
        import discovery_download as dd

        mock_result = MagicMock()
        mock_result.returncode = 3  # systemctl is-active --quiet: inactive
        monkeypatch.setattr(dd.subprocess, "run", lambda *a, **k: mock_result)
        assert dd._is_server_host() is False

    def test_false_when_systemctl_absent(self, monkeypatch):
        """Non-Linux worker / sandboxed env without systemd: inconclusive,
        fails toward 'not the server' (the safe direction -- see the
        function's docstring)."""
        import discovery_download as dd

        def _raise(*a, **k):
            raise FileNotFoundError("systemctl not found")
        monkeypatch.setattr(dd.subprocess, "run", _raise)
        assert dd._is_server_host() is False

    def test_false_when_systemctl_times_out(self, monkeypatch):
        import discovery_download as dd
        import subprocess as _sp

        def _timeout(*a, **k):
            raise _sp.TimeoutExpired(cmd="systemctl", timeout=5)
        monkeypatch.setattr(dd.subprocess, "run", _timeout)
        assert dd._is_server_host() is False


class TestGuardRegistryWrite:
    """Unit tests for _guard_registry_write() in isolation."""

    def test_explicit_path_never_consults_host_detection(self, tmp_path, monkeypatch):
        """registry_is_default=False short-circuits before _is_server_host()
        is even called -- every existing caller (tests, the server's own
        subprocess spawns, which always pass --registry explicitly) is
        completely unaffected by this guard."""
        import discovery_download as dd

        def _must_not_be_called():
            raise AssertionError("_is_server_host must not run for an explicit --registry")
        monkeypatch.setattr(dd, "_is_server_host", _must_not_be_called)

        assert dd._guard_registry_write(str(tmp_path / "sectors.json"), False) is True

    def test_default_path_on_server_host_proceeds(self, tmp_path, monkeypatch):
        import discovery_download as dd
        monkeypatch.setattr(dd, "_is_server_host", lambda: True)
        assert dd._guard_registry_write(str(tmp_path / "sectors.json"), True) is True

    def test_default_path_off_server_host_refused_loudly(self, tmp_path, monkeypatch, capsys):
        import discovery_download as dd
        monkeypatch.setattr(dd, "_is_server_host", lambda: False)

        ok = dd._guard_registry_write(str(tmp_path / "sectors.json"), True)
        assert ok is False

        captured = capsys.readouterr()
        assert "refusing to write" in captured.err
        status = json.loads(captured.out.strip())
        assert status["status"] == "registry_write_skipped_wrong_host"
        assert status["registry"] == str(tmp_path / "sectors.json")


class TestRegistryDefaultResolution:
    """Integration (through main()) of the two fixes together."""

    def test_default_anchored_to_repo_root_not_cwd(self, tmp_path, monkeypatch, capsys):
        """--registry left unset resolves against THIS script's own repo
        root (_PROJECT_ROOT), never the process cwd -- running from an
        unrelated directory must not silently target a different
        sectors.json."""
        import discovery_download as dd

        mock_result = MagicMock()
        mock_result.returncode = 0  # "this host is the server"
        monkeypatch.setattr(dd.subprocess, "run", lambda *a, **k: mock_result)

        fake_root = tmp_path / "fake_repo"
        (fake_root / "data" / "discovery").mkdir(parents=True)
        monkeypatch.setattr(dd, "_PROJECT_ROOT", fake_root)

        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        monkeypatch.chdir(elsewhere)

        db = tmp_path / "empty.db"
        _make_empty_tglc_db(db)

        rc = dd.main(["--sector", "1", "--index", str(db), "--recompute-registry"])
        assert rc == 0
        assert (fake_root / "data" / "discovery" / "sectors.json").exists()
        assert not (elsewhere / "data" / "discovery" / "sectors.json").exists()

    def test_default_registry_write_skipped_off_server_host(self, tmp_path, monkeypatch, capsys):
        """The exact failure mode this fix exists for: --registry left at
        its default, run from a non-server host -- the write must be
        refused, not silently land on a file nobody reads."""
        import discovery_download as dd

        mock_result = MagicMock()
        mock_result.returncode = 3  # "not the server"
        monkeypatch.setattr(dd.subprocess, "run", lambda *a, **k: mock_result)

        fake_root = tmp_path / "fake_repo"
        (fake_root / "data" / "discovery").mkdir(parents=True)
        monkeypatch.setattr(dd, "_PROJECT_ROOT", fake_root)

        db = tmp_path / "empty.db"
        _make_empty_tglc_db(db)

        rc = dd.main(["--sector", "1", "--index", str(db), "--recompute-registry"])
        assert rc == 3
        assert not (fake_root / "data" / "discovery" / "sectors.json").exists()

        captured = capsys.readouterr()
        assert "refusing to write" in captured.err
        last_line = [l for l in captured.out.strip().splitlines() if l][-1]
        assert json.loads(last_line)["status"] == "registry_write_skipped_wrong_host"

    def test_explicit_registry_still_writes_off_server_host(self, tmp_path, monkeypatch, capsys):
        """An explicit --registry (e.g. every existing test, and every
        subprocess the server itself spawns) is unaffected by the host
        check -- this must keep working exactly as before."""
        import discovery_download as dd

        mock_result = MagicMock()
        mock_result.returncode = 3  # "not the server" -- must not matter here
        monkeypatch.setattr(dd.subprocess, "run", lambda *a, **k: mock_result)

        db = tmp_path / "empty.db"
        _make_empty_tglc_db(db)
        registry = tmp_path / "sectors.json"

        rc = dd.main([
            "--sector", "1", "--index", str(db),
            "--registry", str(registry), "--recompute-registry",
        ])
        assert rc == 0
        assert registry.exists()


# ------------------------------------------------------------------ #
# _pack_and_upload_chunk retry+backoff (2026-07-21 MinIO multipart
# read-timeout incident — see the function's own docstring for the full
# story: 56 s7 chunks hit "Read timeout on endpoint URL ...
# ?uploadId=...&partNumber=N" and the pre-fix code recorded pack_failed and
# moved on, leaving 560,000 files indexed with no blob and no tar).
# ------------------------------------------------------------------ #

class _FakeMetadataCacheForPack:
    """Records add_fits_blobs_bulk calls; nothing else is needed by
    _pack_and_upload_chunk."""

    def __init__(self):
        self.blobs: list = []

    def add_fits_blobs_bulk(self, entries):
        self.blobs.extend(list(entries))


class _FlakyOstore:
    """put_file raises *fail_times* times, then succeeds on every call after.

    Tracks reset_client() calls so tests can assert a fresh client is used
    for every RETRY (never the first attempt).
    """

    def __init__(self, fail_times: int):
        self.fail_times = fail_times
        self.calls = 0
        self.reset_calls = 0
        self.put_file_calls: list = []

    def reset_client(self):
        self.reset_calls += 1

    def put_file(self, key, local_path, content_type="application/octet-stream"):
        self.calls += 1
        self.put_file_calls.append(key)
        if self.calls <= self.fail_times:
            raise RuntimeError(
                f"Read timeout on endpoint URL ... "
                f"?uploadId=fake&partNumber={self.calls}"
            )


class _AlwaysFailOstore:
    def __init__(self):
        self.calls = 0
        self.reset_calls = 0

    def reset_client(self):
        self.reset_calls += 1

    def put_file(self, key, local_path, content_type="application/octet-stream"):
        self.calls += 1
        raise RuntimeError(
            f"Read timeout on endpoint URL ... "
            f"?uploadId=fake&partNumber={self.calls}"
        )


class TestPackAndUploadChunkRetry:
    def _make_items(self, tmp_path: Path, n: int):
        items = []
        for i in range(n):
            p = tmp_path / f"hlsp_fake_{i:06d}_s0007-cam1-ccd1_tess_v1_llc.fits"
            p.write_bytes(b"\x00" * 2000)
            items.append((p.name, p))
        return items

    def test_recovers_after_two_failures_no_gap(self, tmp_path, monkeypatch):
        """S3 client fails twice, succeeds on the 3rd attempt: the whole
        chunk still ends up uploaded=10000, failed=0, with NO pack_gap
        manifest written and no pack_failed/pack_gap event emitted."""
        import discovery_download as dd

        monkeypatch.setattr(dd.time, "sleep", lambda s: None)
        events: list = []
        monkeypatch.setattr(dd, "_emit", lambda obj: events.append(obj))

        items = self._make_items(tmp_path, 10_000)
        ostore = _FlakyOstore(fail_times=2)
        mc = _FakeMetadataCacheForPack()

        uploaded = dd._pack_and_upload_chunk(
            items=items, ostore=ostore, container_key="containers/s0007/dl_x.tar",
            metadata_cache=mc, ci=1, progress_log_dir=tmp_path, lot="20260721T000000",
        )

        assert uploaded == 10_000
        assert len(mc.blobs) == 10_000
        assert ostore.calls == 3, "must succeed on the 3rd attempt"
        assert ostore.reset_calls == 2, "reset_client before each of the 2 retries, not the 1st try"
        # Every input file was unlinked (delete_local defaults to True).
        assert not any(p.exists() for _n, p in items)
        # No gap manifest anywhere under tmp_path.
        assert list(tmp_path.glob("pack_gap_*")) == []
        statuses = [e["status"] for e in events]
        assert "pack_failed" not in statuses
        assert "pack_gap" not in statuses
        # Two pack_retry events, one per failed attempt.
        retries = [e for e in events if e["status"] == "pack_retry"]
        assert len(retries) == 2
        assert [r["delay_s"] for r in retries] == [30, 60]

    def test_all_attempts_fail_writes_gap_manifest_with_exact_list(
            self, tmp_path, monkeypatch):
        """Every attempt fails: pack_failed is emitted, uploaded=0, nothing
        is indexed, no local file is deleted, and a pack_gap manifest lists
        EXACTLY the input paths, one per line."""
        import discovery_download as dd

        monkeypatch.setattr(dd.time, "sleep", lambda s: None)
        events: list = []
        monkeypatch.setattr(dd, "_emit", lambda obj: events.append(obj))

        items = self._make_items(tmp_path, 50)
        ostore = _AlwaysFailOstore()
        mc = _FakeMetadataCacheForPack()

        uploaded = dd._pack_and_upload_chunk(
            items=items, ostore=ostore, container_key="containers/s0007/dl_y.tar",
            metadata_cache=mc, ci=3, progress_log_dir=tmp_path, lot="20260721T000000",
        )

        assert uploaded == 0
        assert mc.blobs == []
        # 1 initial attempt + 4 retries = 5 total; reset before each retry.
        assert ostore.calls == 5
        assert ostore.reset_calls == 4
        # Nothing was deleted — the raw files must survive a total failure.
        assert all(p.exists() for _n, p in items)

        gap_path = tmp_path / "pack_gap_20260721T000000_c003.txt"
        assert gap_path.exists()
        lines = gap_path.read_text().splitlines()
        assert lines == [str(p) for _n, p in items]

        statuses = [e["status"] for e in events]
        assert "pack_failed" in statuses
        assert "pack_gap" in statuses
        gap_event = [e for e in events if e["status"] == "pack_gap"][0]
        assert gap_event["manifest"] == str(gap_path)
        assert gap_event["chunk"] == 3
        # 4 pack_retry events (one per retry — not the last, exhausted try).
        retries = [e for e in events if e["status"] == "pack_retry"]
        assert len(retries) == 4
        assert [r["delay_s"] for r in retries] == [30, 60, 120, 240]

    def test_empty_items_is_a_noop(self, tmp_path, monkeypatch):
        """No files to pack -> immediate 0, no S3 call, no gap manifest."""
        import discovery_download as dd

        events: list = []
        monkeypatch.setattr(dd, "_emit", lambda obj: events.append(obj))
        ostore = _AlwaysFailOstore()
        mc = _FakeMetadataCacheForPack()

        uploaded = dd._pack_and_upload_chunk(
            items=[], ostore=ostore, container_key="containers/s0007/dl_z.tar",
            metadata_cache=mc, ci=1, progress_log_dir=tmp_path, lot="20260721T000000",
        )
        assert uploaded == 0
        assert ostore.calls == 0
        assert events == []


# ------------------------------------------------------------------ #
# Chunked download → index → pack (2026-07-14)
#
# The old flow downloaded the WHOLE batch, then indexed it, then packed it, so
# everything after the download was all-or-nothing. Chunking makes progress
# durable: after each chunk the files are indexed, packed and freed, so an
# interruption can lose at most one chunk.
#
# The catch is the LOG: the UI tails it and reads `[n/N] done=…` for the
# progress bar and `[finish]` as "this download completed". A naive chunked
# implementation restarts the counter every chunk (progress bar jumps
# backwards) and emits a [finish] per chunk (a run that dies at chunk 2 of 10
# is reported as a clean success). Both are pinned here.
# ------------------------------------------------------------------ #

class TestChunkedPipeline:

    def _fake_runs(self, dd, monkeypatch, calls):
        def fake_subprocess_run(cmd, **kwargs):
            joined = " ".join(cmd)
            if "download_tglc_batch.py" in joined:
                calls.append(("download", list(cmd)))
                n = len([l for l in Path(cmd[cmd.index("--script") + 1]
                                         ).read_text().splitlines()
                         if l.startswith("curl")])

                class R:
                    returncode = 0
                    stdout = f"done={n} skip=0 fail=0"
                    stderr = ""
                return R()
            if "index_tglc_batch.py" in joined:
                calls.append(("index", list(cmd)))

                class R:
                    returncode = 0
                    stdout = "index: 9 (tic,sector) rows\nfiles_total=9\n"
                    stderr = ""
                return R()
            raise AssertionError(f"unexpected subprocess: {cmd}")
        monkeypatch.setattr(dd.subprocess, "run", fake_subprocess_run)

    def test_batch_is_split_into_chunks_each_indexed_immediately(
            self, tmp_path, monkeypatch, capsys):
        """Each chunk's download must complete before ITS OWN index runs, and
        indexing itself must stay strictly sequential (chunk N+1's index never
        starts before chunk N's finishes).

        Before 2026-07-18 this asserted a strict global alternation
        (download1, index1, download2, index2, ...). That is no longer true
        by design: index+pack of chunk N now runs on a background thread
        while chunk N+1 is already downloading in the foreground (single
        overlap stage — see
        docs/superpowers/specs/2026-07-18-download-index-pipeline.md), so
        whether download(N+1)'s call lands before or after index(N)'s call in
        this fake (which has no artificial delay) is a genuine, harmless race
        — asserting on it would make this test flaky. What must still hold —
        and is what durability actually depends on — are the two invariants
        checked below. TestChunkOverlap (below) proves the overlap itself and
        the strict-sequential-indexing guarantee with real threading.Event
        synchronization instead of relying on scheduling luck.
        """
        import discovery_download as dd

        script = _make_curl_script(tmp_path, n=9)
        dest = tmp_path / "cache"
        dest.mkdir()
        idx = _make_index(tmp_path, [])
        calls: list = []
        self._fake_runs(dd, monkeypatch, calls)

        rc = dd.main([
            "--sector", "2", "--script", str(script), "--dest", str(dest),
            "--registry", str(tmp_path / "sectors.json"), "--index", str(idx),
            "--chunk", "3",
        ])
        assert rc == 0

        # Invariant 1: exactly one index call per chunk, strictly in order —
        # this is what makes recovery lose at most one chunk (indexing is a
        # single SQLite writer; there is only ever one in flight).
        index_order = [_cmd_chunk(k, c) for k, c in calls if k == "index"]
        assert index_order == [1, 2, 3], (
            f"indexing must run exactly once per chunk, strictly in order, "
            f"got {index_order}"
        )
        download_order = [_cmd_chunk(k, c) for k, c in calls if k == "download"]
        assert download_order == [1, 2, 3]

        # Invariant 2: chunk N's download always precedes chunk N's OWN index
        # (a chunk can only be indexed once it exists on disk).
        first_seen = {}
        for pos, (k, c) in enumerate(calls):
            first_seen.setdefault((k, _cmd_chunk(k, c)), pos)
        for n in (1, 2, 3):
            assert first_seen[("download", n)] < first_seen[("index", n)], (
                f"chunk {n} must be downloaded before it is indexed"
            )

    def test_progress_lines_are_global_and_finish_appears_once(
            self, tmp_path, monkeypatch, capsys):
        """The log must read as ONE download: a monotonic [n/GRAND_TOTAL] and a
        single [finish]. Otherwise the UI's bar resets at every chunk, and a run
        that died mid-way is reported as complete."""
        import re

        import discovery_download as dd

        script = _make_curl_script(tmp_path, n=9)
        dest = tmp_path / "cache"
        dest.mkdir()
        idx = _make_index(tmp_path, [])
        log = tmp_path / "dl.log"

        # Real downloader would write the log; our fake must too, honouring the
        # --prior-*/--progress-total contract the chunked caller relies on.
        def fake_subprocess_run(cmd, **kwargs):
            joined = " ".join(cmd)
            if "download_tglc_batch.py" in joined:
                n = len([l for l in Path(cmd[cmd.index("--script") + 1]
                                         ).read_text().splitlines()
                         if l.startswith("curl")])
                total = int(cmd[cmd.index("--progress-total") + 1])
                prior = int(cmd[cmd.index("--prior-done") + 1])
                assert "--suppress-finish" in cmd, (
                    "a per-chunk [finish] would report a run that died at "
                    "chunk 2 as a clean success")
                with open(cmd[cmd.index("--log") + 1], "a") as fh:
                    fh.write(f"[{prior + n}/{total}] done={prior + n} "
                             f"skip=0 fail=0 100.0/s eta=0.0min\n")

                class R:
                    returncode = 0
                    stdout = f"done={n} skip=0 fail=0"
                    stderr = ""
                return R()
            if "index_tglc_batch.py" in joined:
                class R:
                    returncode = 0
                    stdout = "index: 9 (tic,sector) rows\nfiles_total=9\n"
                    stderr = ""
                return R()
            raise AssertionError(f"unexpected subprocess: {cmd}")

        monkeypatch.setattr(dd.subprocess, "run", fake_subprocess_run)

        rc = dd.main([
            "--sector", "2", "--script", str(script), "--dest", str(dest),
            "--registry", str(tmp_path / "sectors.json"), "--index", str(idx),
            "--chunk", "3", "--progress-log", str(log),
        ])
        assert rc == 0

        text = log.read_text()
        progress = re.findall(r"\[(\d+)/(\d+)\]", text)
        assert progress == [("3", "9"), ("6", "9"), ("9", "9")], (
            "progress must run against the GRAND total, monotonically")
        assert text.count("[finish]") == 1
        assert "[finish] done=9 skip=0 fail=0" in text


# ------------------------------------------------------------------ #
# Download↔index overlap (2026-07-18)
#
# docs/superpowers/specs/2026-07-18-download-index-pipeline.md: index+pack of
# chunk N now runs on a background thread while chunk N+1 downloads in the
# foreground (one overlap stage). These tests use real threading.Event
# synchronization — not sleeps or scheduling luck — to pin the three load-
# bearing event orderings: (A) download N+1 starts before index N finishes,
# (B) indexing itself never runs concurrently, (C) an index-N failure leaves
# chunk N+1 undownloaded-into-progress (its download completes, but nothing
# past that — no index, no chunk_done, no registry, no [finish]).
# ------------------------------------------------------------------ #

class TestChunkOverlap:

    def test_download_of_next_chunk_starts_before_index_of_current_finishes(
            self, tmp_path, monkeypatch, capsys):
        """The defining behaviour of the pipeline: chunk 2's download must be
        able to start — and does — while chunk 1's index is still running.

        index(1) blocks on an Event held by the test until the test has
        positively observed download(2) happening; only then does the test
        release it. This is a deterministic proof (via join() semantics), not
        a race won by luck.
        """
        import discovery_download as dd

        script = _make_curl_script(tmp_path, n=6)
        dest = tmp_path / "cache"
        dest.mkdir()
        idx = _make_index(tmp_path, [])

        index1_started = threading.Event()
        index1_release = threading.Event()
        download2_called = threading.Event()

        def fake_subprocess_run(cmd, **kwargs):
            joined = " ".join(cmd)
            if "download_tglc_batch.py" in joined:
                n = _cmd_chunk("download", cmd)
                if n == 2:
                    download2_called.set()

                class R:
                    returncode = 0
                    stdout = "done=3 skip=0 fail=0"
                    stderr = ""
                return R()
            if "index_tglc_batch.py" in joined:
                n = _cmd_chunk("index", cmd)
                if n == 1:
                    index1_started.set()
                    # Hold index(1) open until the test has confirmed
                    # download(2) already happened.
                    assert index1_release.wait(timeout=5), (
                        "test never released index(1) — deadlock"
                    )

                class R:
                    returncode = 0
                    stdout = "index: 3 (tic,sector) rows\nfiles_total=3\n"
                    stderr = ""
                return R()
            raise AssertionError(f"unexpected subprocess: {cmd}")

        monkeypatch.setattr(dd.subprocess, "run", fake_subprocess_run)

        result_holder: dict = {}

        def _run_main():
            result_holder["rc"] = dd.main([
                "--sector", "2", "--script", str(script), "--dest", str(dest),
                "--registry", str(tmp_path / "sectors.json"),
                "--index", str(idx), "--chunk", "3",
            ])

        main_thread = threading.Thread(target=_run_main)
        try:
            main_thread.start()

            assert index1_started.wait(timeout=5), "index(1) never started"
            # download(2) is issued by discovery_download.py's own main
            # thread unconditionally BEFORE it ever joins index(1)'s thread —
            # so with index(1) still blocked, download(2) having already
            # fired is the proof that the two overlapped.
            assert download2_called.wait(timeout=5), (
                "download(2) must start while index(1) is still running"
            )
        finally:
            index1_release.set()
            main_thread.join(timeout=10)

        assert not main_thread.is_alive(), "main() did not finish"
        assert result_holder.get("rc") == 0

    def test_indexing_never_runs_concurrently(self, tmp_path, monkeypatch, capsys):
        """Index N+1 can never start before index N returns — proven with
        per-chunk gates rather than inferred from timing.

        Chunk 1's index is held open by the test; while held, chunk 2's index
        must not have started (deterministic — it can only start after the
        main thread's join() on chunk 1's thread returns, which cannot happen
        before this test releases it). Same for chunk 2 vs chunk 3.
        """
        import discovery_download as dd

        script = _make_curl_script(tmp_path, n=9)
        dest = tmp_path / "cache"
        dest.mkdir()
        idx = _make_index(tmp_path, [])

        started = {1: threading.Event(), 2: threading.Event(), 3: threading.Event()}
        release = {1: threading.Event(), 2: threading.Event()}  # chunk 3 is not gated

        def fake_subprocess_run(cmd, **kwargs):
            joined = " ".join(cmd)
            if "download_tglc_batch.py" in joined:
                class R:
                    returncode = 0
                    stdout = "done=3 skip=0 fail=0"
                    stderr = ""
                return R()
            if "index_tglc_batch.py" in joined:
                n = _cmd_chunk("index", cmd)
                started[n].set()
                if n in release:
                    assert release[n].wait(timeout=5), (
                        f"test never released index({n}) — deadlock"
                    )

                class R:
                    returncode = 0
                    stdout = "index: 3 (tic,sector) rows\nfiles_total=3\n"
                    stderr = ""
                return R()
            raise AssertionError(f"unexpected subprocess: {cmd}")

        monkeypatch.setattr(dd.subprocess, "run", fake_subprocess_run)

        result_holder: dict = {}

        def _run_main():
            result_holder["rc"] = dd.main([
                "--sector", "2", "--script", str(script), "--dest", str(dest),
                "--registry", str(tmp_path / "sectors.json"),
                "--index", str(idx), "--chunk", "3",
            ])

        main_thread = threading.Thread(target=_run_main)
        try:
            main_thread.start()

            assert started[1].wait(timeout=5), "index(1) never started"
            assert not started[2].is_set(), (
                "index(2) must not start before index(1) finishes"
            )
            release[1].set()

            assert started[2].wait(timeout=5), "index(2) never started"
            assert not started[3].is_set(), (
                "index(3) must not start before index(2) finishes"
            )
            release[2].set()

            assert started[3].wait(timeout=5), "index(3) never started"
        finally:
            for ev in release.values():
                ev.set()  # in case an assertion above failed mid-way
            main_thread.join(timeout=10)

        assert not main_thread.is_alive(), "main() did not finish"
        assert result_holder.get("rc") == 0

    def test_index_failure_leaves_next_chunk_without_progress(
            self, tmp_path, monkeypatch, capsys):
        """If chunk 1's index fails, chunk 2's download is left to complete
        (never cancelled) but chunk 2 is never indexed, packed, marked
        chunk_done, or counted — final state is index_failed, exactly as
        before chunking existed.

        index(1) is held open until the test has observed download(2)
        happening, proving the download was not aborted mid-flight — it just
        never leads anywhere once index(1) reports failure.
        """
        import discovery_download as dd

        script = _make_curl_script(tmp_path, n=6)
        dest = tmp_path / "cache"
        dest.mkdir()
        idx = _make_index(tmp_path, [])

        index1_started = threading.Event()
        index1_release = threading.Event()
        download2_called = threading.Event()
        index_calls: list = []

        def fake_subprocess_run(cmd, **kwargs):
            joined = " ".join(cmd)
            if "download_tglc_batch.py" in joined:
                n = _cmd_chunk("download", cmd)
                if n == 2:
                    download2_called.set()

                class R:
                    returncode = 0
                    stdout = "done=3 skip=0 fail=0"
                    stderr = ""
                return R()
            if "index_tglc_batch.py" in joined:
                n = _cmd_chunk("index", cmd)
                index_calls.append(n)
                if n == 1:
                    index1_started.set()
                    assert index1_release.wait(timeout=5), (
                        "test never released index(1) — deadlock"
                    )

                    class R:
                        returncode = 1
                        stdout = ""
                        stderr = "corrupt FITS header"
                    return R()
                raise AssertionError(
                    f"index({n}) must never run — chunk 1's index failed, "
                    f"so chunk {n} must never be indexed"
                )
            raise AssertionError(f"unexpected subprocess: {cmd}")

        monkeypatch.setattr(dd.subprocess, "run", fake_subprocess_run)

        result_holder: dict = {}

        def _run_main():
            result_holder["rc"] = dd.main([
                "--sector", "2", "--script", str(script), "--dest", str(dest),
                "--registry", str(tmp_path / "sectors.json"),
                "--index", str(idx), "--chunk", "3",
            ])

        main_thread = threading.Thread(target=_run_main)
        try:
            main_thread.start()

            assert index1_started.wait(timeout=5), "index(1) never started"
            assert download2_called.wait(timeout=5), (
                "chunk 2's download must be left to complete while index(1) "
                "is still in flight, not cancelled"
            )
        finally:
            index1_release.set()
            main_thread.join(timeout=10)

        assert not main_thread.is_alive(), "main() did not finish"
        assert result_holder.get("rc") == 1
        assert index_calls == [1], (
            "chunk 2 must never be indexed once chunk 1's index has failed"
        )

        raw = capsys.readouterr().out.strip()
        lines = [json.loads(l) for l in raw.splitlines() if l.strip()]
        statuses = [l["status"] for l in lines]
        assert "index_failed" in statuses
        assert "chunk_done" not in statuses, (
            "no chunk may be marked done once the run has aborted on "
            "index_failed"
        )
        assert not (tmp_path / "sectors.json").exists(), (
            "the registry must not be written when the run ends in index_failed"
        )
