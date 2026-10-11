"""Tests for the 2026-07-25 worker telemetry-gaps fixes.

Covers the five gaps from the root-cause report (data/... report, not
committed) as actually implemented in worker/worker.py:
1. Warn-once instead of silent `except Exception: return None` in the
   GPU telemetry probes.
2. macOS macmon lookup (PATH + no-admin-friendly + package-manager
   locations) and its warn-once message.
3. Linux Intel/AMD sysfs power: hwmon variants, then the RAPL
   energy-delta fallback (including the documented PermissionError case
   and the "first sample has no delta" contract).
4. Windows PDH GPU telemetry: pure aggregation/parsing functions, plus
   one end-to-end test against a mocked ctypes.windll layer.
5. The "CPU only (backend)" GPU-name fallback (_friendly_gpu_name).

Constraints (mirrors tests/test_status_fleet.py):
- No real subprocess/filesystem/ctypes calls — everything here is
  monkeypatched at the glob.glob / _safe_read_sysfs / builtins.open /
  ctypes.windll boundary, so results don't depend on what hardware the
  test happens to run on.
- MinimalWorker rebinds ONLY the ExoNitWorker methods each test needs,
  same pattern as test_status_fleet.py's TestWorkerGPUCollector.
"""

from __future__ import annotations

import glob as _real_glob
import io
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from worker.worker import (  # noqa: E402
    ExoNitWorker,
    _friendly_gpu_name,
    _pdh_aggregate_adapter_memory_gb,
    _pdh_aggregate_engine_util,
    _pdh_engine_type,
    _pdh_instance_from_path,
    _warn_telemetry_once,
)


def _bind(*names):
    """Build a MinimalWorker class exposing exactly *names* from
    ExoNitWorker, same rebind trick test_status_fleet.py already uses."""
    class MinimalWorker:
        def __init__(self):
            self.logger = MagicMock()
            for name in names:
                setattr(self, name, getattr(ExoNitWorker, name).__get__(self))
    return MinimalWorker()


# ---------------------------------------------------------------------------
# 1. _warn_telemetry_once
# ---------------------------------------------------------------------------


class TestWarnOnce:

    def test_warns_once_per_key(self):
        w = _bind()
        _warn_telemetry_once(w, "k1", "first message")
        _warn_telemetry_once(w, "k1", "first message again")
        assert w.logger.warning.call_count == 1

    def test_different_keys_each_warn(self):
        w = _bind()
        _warn_telemetry_once(w, "k1", "msg1")
        _warn_telemetry_once(w, "k2", "msg2")
        assert w.logger.warning.call_count == 2

    def test_never_raises_without_logger(self):
        class Bare:
            pass
        b = Bare()
        # No .logger at all -- must not raise.
        _warn_telemetry_once(b, "k1", "msg")
        _warn_telemetry_once(b, "k1", "msg")  # second call, still no raise

    def test_never_raises_on_broken_logger(self):
        w = _bind()
        w.logger.warning.side_effect = RuntimeError("logging is broken too")
        # Must not propagate -- telemetry must not break the heartbeat.
        _warn_telemetry_once(w, "k1", "msg")


# ---------------------------------------------------------------------------
# 2. macOS macmon lookup
# ---------------------------------------------------------------------------


class TestMacmonLookup:

    def test_found_on_path(self):
        w = _bind("_find_macmon")
        with patch("worker.worker.shutil.which", return_value="/usr/bin/macmon"):
            assert w._find_macmon() == "/usr/bin/macmon"

    def test_found_in_no_admin_search_dir(self, tmp_path, monkeypatch):
        """Not on PATH, not via Homebrew/cargo -- but present as a plain
        static binary in ~/bin (the realistic remedy on worker-<host> /
        worker-<host>, neither of which has Homebrew installed)."""
        home = tmp_path / "home"
        bin_dir = home / "bin"
        bin_dir.mkdir(parents=True)
        macmon = bin_dir / "macmon"
        macmon.write_text("#!/bin/sh\necho fake\n")
        macmon.chmod(0o755)
        monkeypatch.setenv("HOME", str(home))

        w = _bind("_find_macmon")
        with patch("worker.worker.shutil.which", return_value=None):
            assert w._find_macmon() == str(macmon)

    def test_missing_everywhere_warns_exactly_once(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path / "empty_home"))
        w = _bind("_gpu_telem_macmon", "_find_macmon")
        with patch("worker.worker.shutil.which", return_value=None):
            r1 = w._gpu_telem_macmon()
            r2 = w._gpu_telem_macmon()
        assert r1 is None
        assert r2 is None
        assert w.logger.warning.call_count == 1
        msg = w.logger.warning.call_args[0][0]
        assert "macmon" in msg
        assert "~/bin" in msg  # no-admin remedy, not just "brew install"


# ---------------------------------------------------------------------------
# 3. Linux Intel/AMD sysfs power (hwmon variants + RAPL delta)
# ---------------------------------------------------------------------------


class _FakeSysfs:
    """A tiny fake /sys tree: glob.glob and _safe_read_sysfs are both
    redirected here so tests don't touch the real filesystem or depend on
    the test-running host's actual hardware."""

    def __init__(self):
        self.dirs = set()       # for glob.glob() to match against
        self.files = {}         # path -> content string
        self.raise_on = {}      # path -> exception instance/class (open())

    def add_dir(self, path):
        self.dirs.add(path)

    def add_file(self, path, content):
        self.files[path] = content
        self.dirs.add(path.rsplit("/", 1)[0])

    def raise_on_open(self, path, exc):
        self.raise_on[path] = exc

    def fake_glob(self, pattern):
        import fnmatch
        candidates = self.dirs | set(self.files)
        return sorted(p for p in candidates if fnmatch.fnmatchcase(p, pattern))

    def fake_read_sysfs(self, path):
        if path in self.raise_on:
            return None  # _safe_read_sysfs swallows everything -> None
        return self.files.get(path)

    def fake_open(self, path, *a, **kw):
        if path in self.raise_on:
            exc = self.raise_on[path]
            raise exc if isinstance(exc, BaseException) else exc()
        if path in self.files:
            return io.StringIO(self.files[path])
        raise FileNotFoundError(path)


@pytest.fixture()
def sysfs(monkeypatch):
    fake = _FakeSysfs()
    monkeypatch.setattr("glob.glob", fake.fake_glob)
    monkeypatch.setattr("worker.worker._safe_read_sysfs", fake.fake_read_sysfs)
    monkeypatch.setattr("builtins.open", fake.fake_open)
    return fake


class TestLinuxSysfsVendorDispatch:

    def test_amd_card_unchanged_behavior(self, sysfs):
        """AMD path (amdgpu) must behave exactly as before this change:
        busy%, VRAM, hwmon temp + power1_average all read directly."""
        base = "/sys/class/drm/card0/device"
        sysfs.add_file(base + "/vendor", "0x1002")
        sysfs.add_file(base + "/gpu_busy_percent", "42")
        sysfs.add_file(base + "/mem_info_vram_used", str(2 * 1024**3))
        sysfs.add_file(base + "/mem_info_vram_total", str(8 * 1024**3))
        sysfs.add_file(base + "/hwmon/hwmon3/temp1_input", "55000")
        sysfs.add_file(base + "/hwmon/hwmon3/power1_average", "15000000")

        w = _bind("_gpu_telem_amd_sysfs", "_hwmon_power_w",
                  "_linux_rapl_power_w", "_warn_intel_util_unavailable")
        result = w._gpu_telem_amd_sysfs()

        assert result["util_pct"] == 42.0
        assert result["mem_used_gb"] == pytest.approx(2.0)
        assert result["mem_total_gb"] == pytest.approx(8.0)
        assert result["temp_c"] == pytest.approx(55.0)
        assert result["power_w"] == pytest.approx(15.0)

    def test_intel_card_no_hwmon_falls_through_to_rapl(self, sysfs):
        """Intel/xe: no gpu_busy_percent, no VRAM files, no hwmon at all
        (confirmed shape on <host>) -- util/mem/temp all None, power comes
        from RAPL instead (tested separately below for the delta math;
        here just confirm the fallthrough happens and nothing crashes)."""
        base = "/sys/class/drm/card0/device"
        sysfs.add_file(base + "/vendor", "0x8086")
        # No hwmon dir at all -- add_dir alone, no files under it.
        sysfs.add_dir(base + "/hwmon")

        w = _bind("_gpu_telem_amd_sysfs", "_hwmon_power_w",
                  "_linux_rapl_power_w", "_warn_intel_util_unavailable")
        with patch.object(w, "_linux_rapl_power_w", return_value=None) as rapl:
            result = w._gpu_telem_amd_sysfs()

        rapl.assert_called_once()
        assert result["util_pct"] is None
        assert result["mem_used_gb"] is None
        assert result["mem_total_gb"] is None
        assert result["temp_c"] is None
        assert result["power_w"] is None

    def test_hwmon_power1_input_variant_recognized(self, sysfs):
        """Not every driver uses amdgpu's power1_average name -- power1_input
        and power2_input must also be tried."""
        base = "/sys/class/drm/card0/device"
        sysfs.add_file(base + "/hwmon/hwmon0/power1_input", "9500000")

        w = _bind("_hwmon_power_w")
        assert w._hwmon_power_w(base) == pytest.approx(9.5)

    def test_hwmon_power2_input_variant_recognized(self, sysfs):
        base = "/sys/class/drm/card0/device"
        sysfs.add_file(base + "/hwmon/hwmon0/power2_input", "3000000")

        w = _bind("_hwmon_power_w")
        assert w._hwmon_power_w(base) == pytest.approx(3.0)

    def test_no_vendor_match_returns_none(self, sysfs):
        """A card whose vendor is neither AMD nor Intel (e.g. a virtual
        display adapter) must not be probed at all."""
        base = "/sys/class/drm/card0/device"
        sysfs.add_file(base + "/vendor", "0x1234")

        w = _bind("_gpu_telem_amd_sysfs")
        assert w._gpu_telem_amd_sysfs() is None


class TestRaplDelta:

    def test_first_sample_returns_none(self, sysfs):
        """The FIRST call after worker start has no previous energy sample
        to diff against -- must return None, not 0.0 or a bogus reading."""
        domain = "/sys/class/powercap/intel-rapl:0"
        sysfs.add_file(domain + "/energy_uj", "1000000")

        w = _bind("_linux_rapl_power_w")
        assert w._linux_rapl_power_w() is None
        assert w._rapl_prev_sample[0] == 1000000

    def test_second_sample_computes_delta_watts(self, sysfs, monkeypatch):
        domain = "/sys/class/powercap/intel-rapl:0"
        sysfs.add_file(domain + "/energy_uj", "1000000")

        times = iter([100.0, 110.0])  # 10s apart
        monkeypatch.setattr("time.time", lambda: next(times))

        w = _bind("_linux_rapl_power_w")
        assert w._linux_rapl_power_w() is None  # primes at t=100

        sysfs.files[domain + "/energy_uj"] = "4000000"  # +3J by t=110
        watts = w._linux_rapl_power_w()
        assert watts == pytest.approx(0.3)  # 3 J / 10 s

    def test_prefers_uncore_child_domain(self, sysfs, monkeypatch):
        """intel-rapl:0 is 'package-0'; intel-rapl:0:1 named 'uncore' is
        the iGPU + memory-controller plane and must be preferred over the
        coarser whole-package total."""
        pkg = "/sys/class/powercap/intel-rapl:0"
        core = "/sys/class/powercap/intel-rapl:0:0"
        uncore = "/sys/class/powercap/intel-rapl:0:1"
        sysfs.add_file(pkg + "/energy_uj", "999999999")
        sysfs.add_file(core + "/name", "core")
        sysfs.add_file(core + "/energy_uj", "555")
        sysfs.add_file(uncore + "/name", "uncore")
        sysfs.add_file(uncore + "/energy_uj", "1000000")

        times = iter([0.0, 1.0])
        monkeypatch.setattr("time.time", lambda: next(times))

        w = _bind("_linux_rapl_power_w")
        w._linux_rapl_power_w()
        assert w._rapl_prev_sample[2] == uncore

    def test_permission_denied_warns_once_and_returns_none(self, sysfs):
        domain = "/sys/class/powercap/intel-rapl:0"
        sysfs.add_dir(domain)
        sysfs.raise_on_open(domain + "/energy_uj", PermissionError())

        w = _bind("_linux_rapl_power_w")
        r1 = w._linux_rapl_power_w()
        r2 = w._linux_rapl_power_w()

        assert r1 is None
        assert r2 is None
        assert w.logger.warning.call_count == 1
        msg = w.logger.warning.call_args[0][0]
        assert "Permission" in msg
        assert "udev" in msg  # names the actual remedy

    def test_no_rapl_domain_at_all_returns_none_quietly(self, sysfs):
        w = _bind("_linux_rapl_power_w")
        assert w._linux_rapl_power_w() is None
        assert w.logger.warning.call_count == 0

    def test_counter_wraparound_returns_none(self, sysfs, monkeypatch):
        domain = "/sys/class/powercap/intel-rapl:0"
        sysfs.files[domain + "/energy_uj"] = "5000000"
        times = iter([0.0, 1.0])
        monkeypatch.setattr("time.time", lambda: next(times))

        w = _bind("_linux_rapl_power_w")
        w._linux_rapl_power_w()  # primes at 5,000,000

        sysfs.files[domain + "/energy_uj"] = "1000"  # wrapped/rolled over
        assert w._linux_rapl_power_w() is None


class TestIntelUtilUnavailableWarning:

    def test_no_pmu_no_warning(self, sysfs):
        """A host with no xe/i915 perf PMU at all (e.g. a VM) has nothing
        more to explain -- must stay quiet."""
        w = _bind("_warn_intel_util_unavailable")
        w._warn_intel_util_unavailable()
        assert w.logger.warning.call_count == 0

    def test_pmu_present_but_paranoid_blocks_it_warns_once(self, sysfs):
        sysfs.add_dir("/sys/devices/xe_0000_00_02.0/events")
        sysfs.add_file("/proc/sys/kernel/perf_event_paranoid", "4")

        w = _bind("_warn_intel_util_unavailable")
        w._warn_intel_util_unavailable()
        w._warn_intel_util_unavailable()

        assert w.logger.warning.call_count == 1
        msg = w.logger.warning.call_args[0][0]
        assert "perf_event_paranoid" in msg
        assert "CAP_PERFMON" in msg


# ---------------------------------------------------------------------------
# 4. Windows PDH — pure functions + one mocked-ctypes integration test
# ---------------------------------------------------------------------------


class TestPdhPureFunctions:

    def test_engine_type_parses_suffix(self):
        name = "pid_1234_luid_0x00000000_0x0001ABCD_phys_0_eng_0_engtype_3D"
        assert _pdh_engine_type(name) == "3D"

    def test_engine_type_none_when_no_marker(self):
        assert _pdh_engine_type("some_other_instance") is None
        assert _pdh_engine_type("") is None
        assert _pdh_engine_type(None) is None

    def test_instance_from_path_extracts_parens(self):
        path = r"\GPU Engine(pid_1_luid_0x0_0x1_phys_0_eng_0_engtype_3D)\Utilization Percentage"
        assert _pdh_instance_from_path(path) == "pid_1_luid_0x0_0x1_phys_0_eng_0_engtype_3D"

    def test_instance_from_path_no_parens_returns_whole_string(self):
        assert _pdh_instance_from_path("nopanrens") == "nopanrens"

    def test_aggregate_engine_util_sums_per_type_then_maxes(self):
        samples = {
            "pid_1_..._engtype_3D": 20.0,
            "pid_2_..._engtype_3D": 30.0,   # 3D total: 50
            "pid_1_..._engtype_Copy": 5.0,  # Copy total: 5
        }
        assert _pdh_aggregate_engine_util(samples) == pytest.approx(50.0)

    def test_aggregate_engine_util_clamps_to_100(self):
        samples = {"pid_1_..._engtype_3D": 80.0, "pid_2_..._engtype_3D": 80.0}
        assert _pdh_aggregate_engine_util(samples) == 100.0

    def test_aggregate_engine_util_empty_is_none(self):
        assert _pdh_aggregate_engine_util({}) is None

    def test_aggregate_engine_util_skips_none_values(self):
        samples = {"pid_1_..._engtype_3D": None, "pid_2_..._engtype_3D": 10.0}
        assert _pdh_aggregate_engine_util(samples) == pytest.approx(10.0)

    def test_aggregate_adapter_memory_sums_and_converts_to_gb(self):
        samples = {"adapter_0": 2 * 1024**3, "adapter_1": 1 * 1024**3}
        assert _pdh_aggregate_adapter_memory_gb(samples) == pytest.approx(3.0)

    def test_aggregate_adapter_memory_empty_is_none(self):
        assert _pdh_aggregate_adapter_memory_gb({}) is None


class _FakePdh:
    """Stand-in for ctypes.windll.pdh. Mirrors the real PDH double-call
    idiom for PdhExpandWildCardPathW (size probe, then fill) and writes
    through ctypes.pointer() out-params exactly like the real DLL would --
    this is only possible because worker.py uses ctypes.pointer() rather
    than ctypes.byref() for its out-params (byref's proxy object can't be
    written through from Python)."""

    ENGINE_PATHS = [
        r"\GPU Engine(pid_1_luid_0x0_0x1_phys_0_eng_0_engtype_3D)\Utilization Percentage",
        r"\GPU Engine(pid_2_luid_0x0_0x1_phys_0_eng_1_engtype_3D)\Utilization Percentage",
    ]
    MEM_PATHS = [
        r"\GPU Adapter Memory(luid_0x0_0x1_phys_0)\Dedicated Usage",
    ]

    def __init__(self):
        self._next_handle = 1000
        self.collect_calls = 0
        # counter_handle -> formatted double value to hand back
        self.values = {}

    def _alloc_handle(self):
        self._next_handle += 1
        return self._next_handle

    def PdhOpenQueryW(self, datasource, flags, phQuery):
        phQuery.contents.value = 1
        return 0

    def PdhExpandWildCardPathW(self, datasource, wildcard, buf, psize, flags):
        paths = self.ENGINE_PATHS if "GPU Engine" in wildcard else self.MEM_PATHS
        blob = "\x00".join(paths) + "\x00\x00"
        if buf is None:
            psize.contents.value = len(blob)
            return 0
        for i, ch in enumerate(blob):
            buf[i] = ch
        return 0

    def PdhAddEnglishCounterW(self, hQuery, path, flags, phCounter):
        handle = self._alloc_handle()
        phCounter.contents.value = handle
        # Deterministic canned values per counter, keyed by handle order.
        if path in self.ENGINE_PATHS:
            self.values[handle] = 30.0 if path == self.ENGINE_PATHS[0] else 25.0
        else:
            self.values[handle] = float(4 * 1024**3)
        return 0

    def PdhCollectQueryData(self, hQuery):
        self.collect_calls += 1
        return 0

    def PdhGetFormattedCounterValue(self, hCounter, fmt, ptype, pvalue):
        handle = hCounter.value if hasattr(hCounter, "value") else hCounter
        # Real PDH: a rate counter reports PDH_CSTATUS_INVALID_DATA on the
        # collect immediately following PdhAddEnglishCounterW -- only the
        # SECOND PdhCollectQueryData onward is meaningful. Mirror that here.
        if self.collect_calls < 2:
            pvalue.contents.CStatus = 0xC0000BC6  # PDH_CSTATUS_INVALID_DATA-ish
            return 0
        pvalue.contents.CStatus = 0
        pvalue.contents.doubleValue = self.values.get(handle, 0.0)
        return 0


class TestWindowsPdhIntegration:

    def test_full_flow_against_mocked_ctypes(self, monkeypatch):
        """End-to-end through _gpu_telem_windows_pdh with a fake pdh.dll --
        exercises query setup, wildcard expansion, counter add, collect,
        and formatted-value read, then the real aggregation functions."""
        monkeypatch.setattr("platform.system", lambda: "Windows")
        fake_pdh = _FakePdh()
        fake_windll = MagicMock()
        fake_windll.pdh = fake_pdh

        w = _bind("_gpu_telem_windows_pdh", "_pdh_ensure_query")

        with patch("ctypes.windll", fake_windll, create=True):
            first = w._gpu_telem_windows_pdh()   # primes the rate counters
            second = w._gpu_telem_windows_pdh()   # real data now

        assert first["util_pct"] is None  # warm-up sample, PDH_CSTATUS invalid
        assert second["util_pct"] == pytest.approx(55.0)  # 30 + 25, both engtype_3D
        assert second["mem_used_gb"] == pytest.approx(4.0)
        assert second["mem_total_gb"] is None  # not available via PDH, by design
        assert second["temp_c"] is None
        assert second["power_w"] is None

    def test_non_windows_platform_returns_none_without_touching_ctypes(self):
        """The one call site guards on platform.system() == 'Windows' --
        confirm the method itself also refuses to run on a non-Windows
        platform.system() report, so it can never touch ctypes.windll
        (which doesn't exist as an attribute on Linux/macOS) by accident."""
        w = _bind("_gpu_telem_windows_pdh")
        with patch("platform.system", return_value="Linux"):
            assert w._gpu_telem_windows_pdh() is None

    def test_setup_failure_warns_once_and_does_not_retry_every_heartbeat(self, monkeypatch):
        monkeypatch.setattr("platform.system", lambda: "Windows")
        fake_windll = MagicMock()
        fake_windll.pdh.PdhOpenQueryW.return_value = 1  # non-zero -> failure

        w = _bind("_gpu_telem_windows_pdh", "_pdh_ensure_query")
        with patch("ctypes.windll", fake_windll, create=True):
            r1 = w._gpu_telem_windows_pdh()
            r2 = w._gpu_telem_windows_pdh()

        assert r1 is None
        assert r2 is None
        assert w.logger.warning.call_count == 1
        # Second call must not even attempt PdhOpenQueryW again.
        assert fake_windll.pdh.PdhOpenQueryW.call_count == 1


# ---------------------------------------------------------------------------
# 5. _friendly_gpu_name — "CPU only (backend)" instead of the bare backend
# ---------------------------------------------------------------------------


class TestFriendlyGpuName:

    def test_real_gpu_name_passthrough(self):
        with patch("worker.worker._bls_gpu_info", return_value="RTX 5060 Ti (16.0 GB)"):
            assert _friendly_gpu_name() == "RTX 5060 Ti (16.0 GB)"

    def test_numba_cpu_only_backend(self):
        with patch("worker.worker._bls_gpu_info", return_value=None):
            with patch("worker.worker._bls_backend", return_value="numba"):
                assert _friendly_gpu_name() == "CPU only (numba)"

    def test_numpy_cpu_only_backend(self):
        with patch("worker.worker._bls_gpu_info", return_value=None):
            with patch("worker.worker._bls_backend", return_value="numpy"):
                assert _friendly_gpu_name() == "CPU only (numpy)"

    def test_no_backend_at_all_falls_back_to_none(self):
        with patch("worker.worker._bls_gpu_info", return_value=None):
            with patch("worker.worker._bls_backend", return_value=None):
                assert _friendly_gpu_name() == "none"
