"""Tests for scripts/lora_macs_driver.py.

Constraints (same spirit as test_discovery_orchestrator.py):
- NEVER real SSH, macwake, farmctl, or network calls.
- Pure functions (parsing, RAM/disk rules, budget formula, endpoints.json
  build, host resolution) are tested directly with plain values.
- Driver.* methods that DO real I/O are tested with a fake SSHRunner /
  fake http_get/http_post/macwake/farmctl_stop injected in the constructor.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import lora_macs_driver as lmd  # noqa: E402


# ---------------------------------------------------------------------------
# vm_stat / swapusage / df parsing
# ---------------------------------------------------------------------------

VM_STAT_SAMPLE = """Mach Virtual Memory Statistics: (page size of 16384 bytes)
Pages free:                              600000.
Pages active:                             120000.
Pages inactive:                           200000.
Pages speculative:                         30000.
Pages throttled:                               0.
Pages wired down:                         150000.
Pages purgeable:                           21968.
"Translation faults":                  900000000.
Pages copy-on-write:                     1200000.
Pages zero filled:                      50000000.
Pages reactivated:                        300000.
Pages purged:                             100000.
File-backed pages:                        200000.
Anonymous pages:                          180000.
Pages stored in compressor:                30000.
Pages occupied by compressor:              10000.
Decompressions:                           400000.
Compressions:                             500000.
Pageouts:                                      0.
Swapins:                                       0.
Swapouts:                                      0.
"""


def test_parse_vm_stat_labels_and_values():
    parsed = lmd.parse_vm_stat(VM_STAT_SAMPLE)
    assert parsed["pages_free"] == 600000
    assert parsed["pages_inactive"] == 200000
    assert parsed["pages_speculative"] == 30000
    assert parsed["pages_purgeable"] == 21968
    # Header line (no trailing digit run) must not appear as a key.
    assert "mach_virtual_memory_statistics" not in parsed


def test_reclaimable_ram_gb_matches_hand_computed_value():
    total_pages = 600000 + 200000 + 30000 + 21968
    expected_gb = (total_pages * lmd.MAC_PAGE_SIZE_BYTES) / (1024 ** 3)
    assert lmd.reclaimable_ram_gb(VM_STAT_SAMPLE) == pytest.approx(expected_gb)
    # Sanity: chosen so the "healthy" fixture clears the >=8 GB RAM-rule band
    # (this VM_STAT_SAMPLE also backs the Driver fake-SSH tests below).
    assert lmd.reclaimable_ram_gb(VM_STAT_SAMPLE) == pytest.approx(13.0)


def test_parse_swap_used_gb_megabytes():
    text = "vm.swapusage: total = 1024.00M  used = 386.00M  free = 638.00M  (encrypted)"
    assert lmd.parse_swap_used_gb(text) == pytest.approx(386.0 / 1024.0)


def test_parse_swap_used_gb_gigabytes():
    text = "vm.swapusage: total = 4.00G  used = 2.30G  free = 1.70G"
    assert lmd.parse_swap_used_gb(text) == pytest.approx(2.30)


def test_parse_swap_used_gb_no_match_is_zero():
    assert lmd.parse_swap_used_gb("garbage output") == 0.0


DF_G_SAMPLE = (
    "Filesystem   1G-blocks  Used Avail Capacity iused    ifree   %iused  Mounted on\n"
    "/dev/disk3s5       460   380    71     85%  611587  4292245      0%   /\n"
)


def test_parse_df_avail_gb_from_header():
    assert lmd.parse_df_avail_gb(DF_G_SAMPLE) == pytest.approx(71.0)


def test_parse_df_avail_gb_fallback_when_header_unmatched():
    # No 'Avail' token at all -> fixed macOS df -g layout, column index 3.
    text = "FS   BLOCKS  USED  FREE  PCT\n/dev/disk3s5  460  380  71  85%\n"
    assert lmd.parse_df_avail_gb(text) == pytest.approx(71.0)


def test_parse_df_avail_gb_raises_on_no_data_line():
    with pytest.raises(ValueError):
        lmd.parse_df_avail_gb("just one line, no data\n")


# ---------------------------------------------------------------------------
# ps worker-process detection
# ---------------------------------------------------------------------------

def test_count_exonit_worker_processes_swift():
    ps = (
        "/usr/sbin/cfprefsd\n"
        "/Users/user/exonit-swift-canary/.build/release/exonit-swift-worker --config x\n"
        "/usr/libexec/trustd\n"
    )
    assert lmd.count_exonit_worker_processes(ps) == 1


def test_count_exonit_worker_processes_python():
    ps = "/Users/user/exonit-worker/venv/bin/python3 /Users/user/exonit-worker/worker.py\n"
    assert lmd.count_exonit_worker_processes(ps) == 1


def test_count_exonit_worker_processes_none():
    ps = "/usr/sbin/cfprefsd\n/usr/libexec/trustd\nmlx_lm.server --model x\n"
    assert lmd.count_exonit_worker_processes(ps) == 0


def test_count_exonit_worker_processes_multiple():
    ps = (
        "exonit-swift-worker instance1\n"
        "exonit-swift-worker instance2\n"
        "/Users/user/exonit-worker/worker.py\n"
    )
    assert lmd.count_exonit_worker_processes(ps) == 3


# ---------------------------------------------------------------------------
# RAM rule (design §2.2)
# ---------------------------------------------------------------------------

def test_ram_rule_healthy_keeps_configured_concurrency():
    v = lmd.apply_ram_rule(8, 13.0, 0.06)
    assert v.concurrency == 8


def test_ram_rule_downgrade_band_never_raises():
    v = lmd.apply_ram_rule(4, 7.0, 0.0)
    assert v.concurrency == 4  # min(4, 4) unchanged


def test_ram_rule_downgrade_band_caps_at_4():
    v = lmd.apply_ram_rule(8, 6.5, 0.0)
    assert v.concurrency == 4


def test_ram_rule_excluded_below_floor():
    v = lmd.apply_ram_rule(8, 5.9, 0.0)
    assert v.concurrency is None
    assert "6 GB floor" in v.reason


def test_ram_rule_excluded_by_swap():
    v = lmd.apply_ram_rule(8, 20.0, 1.6)
    assert v.concurrency is None
    assert "swap" in v.reason


def test_ram_rule_swap_boundary_is_inclusive_pass():
    # exactly 1.5 GB used must NOT exclude (only strictly > 1.5).
    v = lmd.apply_ram_rule(8, 20.0, 1.5)
    assert v.concurrency == 8


# ---------------------------------------------------------------------------
# Foreign LLM process guard (design §2.2's ".60 in use by the operator" oMLX/mlx
# RSS>2GB check, generalized/config-driven to any host)
# ---------------------------------------------------------------------------

# Realistic `ps ax -o rss=,command=` fixtures (RSS in KB, right-justified as
# macOS ps does, then the full command line).
PS_RSS_OMLX = (
    "  1234 /usr/sbin/cfprefsd\n"
    "3145728 /Applications/oMLX.app/Contents/MacOS/oMLX\n"  # 3 GB RSS
)
PS_RSS_MLX_LM_HF_CACHE = (
    "  1234 /usr/sbin/cfprefsd\n"
    "2621440 /opt/homebrew/bin/python3.12 -m mlx_lm.generate --model "
    "/Users/user/.cache/huggingface/hub/models--mlx-community--Qwen3-30B\n"  # 2.5 GB
)
PS_RSS_OUR_OWN_MLX_LM = (
    "  1234 /usr/sbin/cfprefsd\n"
    "4194304 /Users/user/exonit-lora/.venv/bin/mlx_lm.server --model "
    "models/qwen3-4b-lora-fused-q8 --host 127.0.0.1 --port 8503\n"  # 4 GB, but OURS
)
PS_RSS_SMALL_MLX = (
    "  1234 /usr/sbin/cfprefsd\n"
    " 512000 /opt/homebrew/bin/python3.12 -m mlx_lm.generate --model tiny\n"  # 0.49 GB
)
PS_RSS_NO_MATCH = (
    "  1234 /usr/sbin/cfprefsd\n"
    "8388608 /Applications/Xcode.app/Contents/MacOS/Xcode\n"  # 8 GB, doesn't match any pattern
)


def test_find_foreign_llm_processes_detects_omlx_over_threshold():
    hits = lmd.find_foreign_llm_processes(
        PS_RSS_OMLX, lmd.DEFAULT_FOREIGN_LLM_PROCESS_PATTERNS,
        "/Users/user/exonit-lora", 2.0)
    assert len(hits) == 1
    cmd, rss_gb = hits[0]
    assert "oMLX" in cmd
    assert rss_gb == pytest.approx(3.0)


def test_find_foreign_llm_processes_detects_stray_mlx_lm():
    hits = lmd.find_foreign_llm_processes(
        PS_RSS_MLX_LM_HF_CACHE, lmd.DEFAULT_FOREIGN_LLM_PROCESS_PATTERNS,
        "/Users/user/exonit-lora", 2.0)
    assert len(hits) == 1
    assert "huggingface" in hits[0][0]


def test_find_foreign_llm_processes_excludes_our_own_process():
    """Our own about-to-be-started mlx_lm.server (under remote_dir) must
    NEVER self-exclude the host, even though it matches the generic
    'mlx[_-]lm' pattern and is well over the RSS threshold."""
    hits = lmd.find_foreign_llm_processes(
        PS_RSS_OUR_OWN_MLX_LM, lmd.DEFAULT_FOREIGN_LLM_PROCESS_PATTERNS,
        "/Users/user/exonit-lora", 2.0)
    assert hits == []


def test_find_foreign_llm_processes_under_threshold_is_not_a_hit():
    hits = lmd.find_foreign_llm_processes(
        PS_RSS_SMALL_MLX, lmd.DEFAULT_FOREIGN_LLM_PROCESS_PATTERNS,
        "/Users/user/exonit-lora", 2.0)
    assert hits == []


def test_find_foreign_llm_processes_no_pattern_match_is_not_a_hit():
    hits = lmd.find_foreign_llm_processes(
        PS_RSS_NO_MATCH, lmd.DEFAULT_FOREIGN_LLM_PROCESS_PATTERNS,
        "/Users/user/exonit-lora", 2.0)
    assert hits == []


def test_find_foreign_llm_processes_empty_patterns_is_opt_out():
    hits = lmd.find_foreign_llm_processes(
        PS_RSS_OMLX, [], "/Users/user/exonit-lora", 2.0)
    assert hits == []


def test_find_foreign_llm_processes_case_insensitive():
    ps = "3145728 /Applications/OMLX.APP/Contents/MacOS/OMLX\n"
    hits = lmd.find_foreign_llm_processes(
        ps, lmd.DEFAULT_FOREIGN_LLM_PROCESS_PATTERNS, "/Users/user/exonit-lora", 2.0)
    assert len(hits) == 1


def test_find_foreign_llm_processes_custom_threshold():
    # PS_RSS_SMALL_MLX is 0.49 GB — a strict-enough custom threshold catches it.
    hits = lmd.find_foreign_llm_processes(
        PS_RSS_SMALL_MLX, lmd.DEFAULT_FOREIGN_LLM_PROCESS_PATTERNS,
        "/Users/user/exonit-lora", 0.1)
    assert len(hits) == 1


def test_find_foreign_llm_processes_multiple_hits_returns_all():
    combined = PS_RSS_OMLX + PS_RSS_MLX_LM_HF_CACHE
    hits = lmd.find_foreign_llm_processes(
        combined, lmd.DEFAULT_FOREIGN_LLM_PROCESS_PATTERNS,
        "/Users/user/exonit-lora", 2.0)
    assert len(hits) == 2


# ---------------------------------------------------------------------------
# Backlog budget formula (design §2.5)
# ---------------------------------------------------------------------------

def test_backlog_budget_five_healthy_hosts_matches_design_example():
    # design: 4x12.9 + 10.4 = 62 expl/min -> 14h -> 52080, cap 50000 manual.
    rates = [12.9, 12.9, 12.9, 12.9, 10.4]
    assert lmd.compute_backlog_budget(rates, 14.0, 50000) == 50000


def test_backlog_budget_three_healthy_hosts():
    rates = [12.9, 12.9, 12.9]
    l = lmd.compute_backlog_budget(rates, 14.0, 50000)
    # 38.7/min * 60 * 14 = 32508
    assert l == 32508


def test_backlog_budget_cap_zero_disables_backlog():
    assert lmd.compute_backlog_budget([12.9, 10.4], 14.0, 0) == 0


def test_backlog_budget_negative_cap_disables_backlog():
    assert lmd.compute_backlog_budget([12.9], 14.0, -5) == 0


def test_backlog_budget_never_negative():
    assert lmd.compute_backlog_budget([], 14.0, 50000) == 0


# ---------------------------------------------------------------------------
# endpoints.json build
# ---------------------------------------------------------------------------

def test_build_endpoints_basic():
    hosts = [
        {"ip": "<lan-ip>", "name": "m4b", "concurrency": 8},
        {"ip": "<lan-ip>", "name": "m4", "concurrency": 4},
    ]
    out = lmd.build_endpoints(hosts, port=8500)
    assert out == [
        {"url": "http://<lan-ip>:8500/api/explain", "concurrency": 8, "name": "m4b"},
        {"url": "http://<lan-ip>:8500/api/explain", "concurrency": 4, "name": "m4"},
    ]


def test_build_endpoints_per_host_port_override():
    hosts = [{"ip": "1.2.3.4", "name": "x", "concurrency": 1, "port": 9999}]
    out = lmd.build_endpoints(hosts, port=8500)
    assert out[0]["url"] == "http://1.2.3.4:9999/api/explain"


# ---------------------------------------------------------------------------
# Host resolution (--hosts / --exclude / enabled)
# ---------------------------------------------------------------------------

ALL_HOSTS = [
    {"name": "m4", "enabled": True},
    {"name": "m4b", "enabled": True},
    {"name": "m4c", "enabled": False},
]


def test_resolve_hosts_default_respects_enabled():
    out = lmd.resolve_hosts(ALL_HOSTS, None, None)
    assert {h["name"] for h in out} == {"m4", "m4b"}


def test_resolve_hosts_explicit_hosts_overrides_disabled():
    out = lmd.resolve_hosts(ALL_HOSTS, ["m4c"], None)
    assert {h["name"] for h in out} == {"m4c"}


def test_resolve_hosts_exclude_wins_over_explicit_hosts():
    out = lmd.resolve_hosts(ALL_HOSTS, ["m4", "m4b"], ["m4"])
    assert {h["name"] for h in out} == {"m4b"}


def test_resolve_hosts_exclude_only():
    out = lmd.resolve_hosts(ALL_HOSTS, None, ["m4b"])
    assert {h["name"] for h in out} == {"m4"}


# ---------------------------------------------------------------------------
# Manifest integrity
# ---------------------------------------------------------------------------

def test_manifest_integrity_ok():
    manifest = {"files": {"models/m/model.safetensors": {"size": 100}}}
    ok, detail = lmd.check_manifest_integrity(manifest, {"models/m/model.safetensors": 100})
    assert ok and detail == ""


def test_manifest_integrity_missing_file():
    manifest = {"files": {"a": {"size": 100}, "b": {"size": 50}}}
    ok, detail = lmd.check_manifest_integrity(manifest, {"a": 100})
    assert not ok
    assert "b" in detail


def test_manifest_integrity_size_mismatch():
    manifest = {"files": {"a": {"size": 100}}}
    ok, detail = lmd.check_manifest_integrity(manifest, {"a": 99})
    assert not ok
    assert "a" in detail


# ---------------------------------------------------------------------------
# FleetConfig.load (ops/lora_macs.json, real config file shape)
# ---------------------------------------------------------------------------

def test_fleet_config_loads_real_ops_file():
    cfg = lmd.FleetConfig.load(lmd.DEFAULT_CONFIG_PATH)
    names = {h.name for h in cfg.hosts}
    assert names == {"m4", "m4b", "m4c", "m4d", "m4e"}
    m4 = next(h for h in cfg.hosts if h.name == "m4")
    assert m4.concurrency == 4
    assert m4.rate_per_min == pytest.approx(10.4)
    m4b = next(h for h in cfg.hosts if h.name == "m4b")
    assert m4b.concurrency == 8
    assert cfg.min_hosts == 1
    assert cfg.min_free_disk_gb == 10
    assert cfg.budget_hours == 14
    assert cfg.deadline_hours == 20
    # Every real host uses the fleet-wide foreign-LLM-process guard default
    # (no per-host override in ops/lora_macs.json today).
    assert cfg.foreign_llm_process_patterns == lmd.DEFAULT_FOREIGN_LLM_PROCESS_PATTERNS
    assert cfg.foreign_llm_rss_gb_max == pytest.approx(lmd.DEFAULT_FOREIGN_LLM_RSS_GB_MAX)
    assert m4.foreign_llm_process_patterns is None  # -> falls back to fleet default
    assert m4.foreign_llm_rss_gb_max is None


def test_fleet_config_loads_from_tmp_file(tmp_path):
    data = {
        "port": 8500, "mlx_port": 8503, "remote_dir": "/Users/user/exonit-lora",
        "min_hosts": 2, "min_free_disk_gb": 5, "budget_hours": 1, "deadline_hours": 2,
        "model": "test-model",
        "hosts": [
            {"name": "a", "ssh": "user@1.2.3.4", "ip": "1.2.3.4", "concurrency": 4,
             "rate_per_min": 1.0, "workers": ["worker-a"], "enabled": True},
        ],
    }
    p = tmp_path / "cfg.json"
    p.write_text(json.dumps(data))
    cfg = lmd.FleetConfig.load(p)
    assert cfg.min_hosts == 2
    assert cfg.hosts[0].workers == ["worker-a"]
    # No foreign_llm_* keys anywhere in this file -> fleet falls back to the
    # module defaults, and the host has no per-host override.
    assert cfg.foreign_llm_process_patterns == lmd.DEFAULT_FOREIGN_LLM_PROCESS_PATTERNS
    assert cfg.foreign_llm_rss_gb_max == pytest.approx(lmd.DEFAULT_FOREIGN_LLM_RSS_GB_MAX)
    assert cfg.hosts[0].foreign_llm_process_patterns is None
    assert cfg.hosts[0].foreign_llm_rss_gb_max is None


def test_fleet_config_loads_per_host_foreign_llm_override(tmp_path):
    data = {
        "hosts": [
            {"name": "a", "ssh": "user@1.2.3.4", "ip": "1.2.3.4", "concurrency": 4,
             "rate_per_min": 1.0,
             "foreign_llm_process_patterns": ["custom_pattern"],
             "foreign_llm_rss_gb_max": 5.0},
            {"name": "b", "ssh": "user@1.2.3.5", "ip": "1.2.3.5", "concurrency": 4,
             "rate_per_min": 1.0},
        ],
    }
    p = tmp_path / "cfg.json"
    p.write_text(json.dumps(data))
    cfg = lmd.FleetConfig.load(p)
    host_a = next(h for h in cfg.hosts if h.name == "a")
    host_b = next(h for h in cfg.hosts if h.name == "b")
    assert host_a.foreign_llm_process_patterns == ["custom_pattern"]
    assert host_a.foreign_llm_rss_gb_max == pytest.approx(5.0)
    assert host_b.foreign_llm_process_patterns is None
    assert host_b.foreign_llm_rss_gb_max is None


def test_fleet_config_loads_fleet_wide_foreign_llm_override(tmp_path):
    data = {
        "foreign_llm_process_patterns": ["only_this_pattern"],
        "foreign_llm_rss_gb_max": 3.5,
        "hosts": [
            {"name": "a", "ssh": "user@1.2.3.4", "ip": "1.2.3.4", "concurrency": 4,
             "rate_per_min": 1.0},
        ],
    }
    p = tmp_path / "cfg.json"
    p.write_text(json.dumps(data))
    cfg = lmd.FleetConfig.load(p)
    assert cfg.foreign_llm_process_patterns == ["only_this_pattern"]
    assert cfg.foreign_llm_rss_gb_max == pytest.approx(3.5)


# ---------------------------------------------------------------------------
# Driver — fake SSH/http/macwake/farmctl, never touches real hardware
# ---------------------------------------------------------------------------

def _fake_config(n_hosts=1, concurrency=8, rate=12.9):
    hosts = [
        lmd.HostConfig(name=f"h{i}", ssh=f"user@10.0.0.{i}", ip=f"10.0.0.{i}",
                       concurrency=concurrency, rate_per_min=rate,
                       workers=[f"worker-h{i}"], enabled=True)
        for i in range(n_hosts)
    ]
    return lmd.FleetConfig(hosts=hosts, port=8500, mlx_port=8503,
                           remote_dir="/Users/user/exonit-lora", min_hosts=1,
                           min_free_disk_gb=10, budget_hours=14, deadline_hours=20,
                           model="qwen3-4b-lora-fused-q8")


def _manifest_json():
    return json.dumps({"files": {"models/m/model.safetensors": {"size": 100}}})


def _healthy_ssh_runner(ssh_host, remote_cmd, timeout):
    """Fake SSHRunner: every host reachable, no leftover worker, plenty of
    disk/RAM, valid manifest."""
    if remote_cmd == "true":
        return lmd.SSHResult(0, "", "")
    if remote_cmd == "ps ax -o command=":
        return lmd.SSHResult(0, "/usr/sbin/cfprefsd\n", "")
    if remote_cmd == "ps ax -o rss=,command=":
        return lmd.SSHResult(0, "  1234 /usr/sbin/cfprefsd\n", "")
    if remote_cmd.startswith("df -g"):
        return lmd.SSHResult(0, DF_G_SAMPLE, "")
    if remote_cmd == "vm_stat":
        return lmd.SSHResult(0, VM_STAT_SAMPLE, "")
    if remote_cmd == "sysctl vm.swapusage":
        return lmd.SSHResult(0, "vm.swapusage: total = 4.00G used = 0.05G free = 3.95G", "")
    if remote_cmd.startswith("cat") and "MANIFEST.json" in remote_cmd:
        return lmd.SSHResult(0, _manifest_json(), "")
    if remote_cmd.startswith("stat -f"):
        return lmd.SSHResult(0, "models/m/model.safetensors 100\n", "")
    if remote_cmd.startswith("pkill"):
        return lmd.SSHResult(0, "", "")
    return lmd.SSHResult(1, "", f"unhandled test command: {remote_cmd!r}")


def test_preflight_host_passes_when_everything_healthy():
    config = _fake_config()
    driver = lmd.Driver(config, ssh=_healthy_ssh_runner)
    result = driver.preflight_host(config.hosts[0])
    assert result.healthy
    assert result.concurrency == 8
    assert result.excluded_reason == ""


def test_preflight_host_excluded_when_unreachable_and_macwake_fails():
    config = _fake_config()

    def always_unreachable(ssh_host, remote_cmd, timeout):
        return lmd.SSHResult(255, "", "connection refused")

    driver = lmd.Driver(config, ssh=always_unreachable, macwake=lambda alias: False)
    result = driver.preflight_host(config.hosts[0])
    assert not result.healthy
    assert "macwake" in result.excluded_reason


def test_preflight_host_wakes_and_recovers():
    config = _fake_config()
    calls = {"n": 0}

    def flaky_then_ok(ssh_host, remote_cmd, timeout):
        if remote_cmd == "true":
            calls["n"] += 1
            if calls["n"] == 1:
                return lmd.SSHResult(255, "", "down")
            return lmd.SSHResult(0, "", "")
        return _healthy_ssh_runner(ssh_host, remote_cmd, timeout)

    driver = lmd.Driver(config, ssh=flaky_then_ok, macwake=lambda alias: True)
    result = driver.preflight_host(config.hosts[0])
    assert result.healthy


def test_preflight_host_stops_lingering_worker_via_farmctl():
    config = _fake_config()
    stopped = {"called_with": None}
    seen_ps_calls = {"n": 0}

    def ssh_with_worker(ssh_host, remote_cmd, timeout):
        if remote_cmd == "ps ax -o command=":
            seen_ps_calls["n"] += 1
            if seen_ps_calls["n"] == 1:
                return lmd.SSHResult(0, "/Users/user/exonit-worker/worker.py\n", "")
            return lmd.SSHResult(0, "/usr/sbin/cfprefsd\n", "")
        return _healthy_ssh_runner(ssh_host, remote_cmd, timeout)

    def fake_farmctl_stop(names):
        stopped["called_with"] = names
        return True

    driver = lmd.Driver(config, ssh=ssh_with_worker, farmctl_stop=fake_farmctl_stop)
    result = driver.preflight_host(config.hosts[0])
    assert result.healthy
    assert stopped["called_with"] == ["worker-h0"]


def test_preflight_host_excluded_when_worker_persists_after_farmctl_stop():
    config = _fake_config()

    def ssh_always_worker(ssh_host, remote_cmd, timeout):
        if remote_cmd == "ps ax -o command=":
            return lmd.SSHResult(0, "/Users/user/exonit-worker/worker.py\n", "")
        return _healthy_ssh_runner(ssh_host, remote_cmd, timeout)

    driver = lmd.Driver(config, ssh=ssh_always_worker, farmctl_stop=lambda names: True)
    result = driver.preflight_host(config.hosts[0])
    assert not result.healthy
    assert "still running" in result.excluded_reason


def test_preflight_host_excluded_on_low_disk():
    config = _fake_config()

    def low_disk(ssh_host, remote_cmd, timeout):
        if remote_cmd.startswith("df -g"):
            return lmd.SSHResult(0, DF_G_SAMPLE.replace("71", "3"), "")
        return _healthy_ssh_runner(ssh_host, remote_cmd, timeout)

    driver = lmd.Driver(config, ssh=low_disk)
    result = driver.preflight_host(config.hosts[0])
    assert not result.healthy
    assert "disk" in result.excluded_reason


def test_preflight_host_excluded_on_foreign_llm_process():
    config = _fake_config()

    def ssh_with_omlx(ssh_host, remote_cmd, timeout):
        if remote_cmd == "ps ax -o rss=,command=":
            return lmd.SSHResult(0, PS_RSS_OMLX, "")
        return _healthy_ssh_runner(ssh_host, remote_cmd, timeout)

    driver = lmd.Driver(config, ssh=ssh_with_omlx)
    result = driver.preflight_host(config.hosts[0])
    assert not result.healthy
    assert "foreign_llm_loaded" in result.excluded_reason
    assert "oMLX" in result.excluded_reason


def test_preflight_host_not_excluded_by_own_mlx_lm_process():
    """Our own (about-to-be-started) mlx_lm.server under remote_dir must
    never trip the foreign-process guard."""
    config = _fake_config()

    def ssh_with_our_own(ssh_host, remote_cmd, timeout):
        if remote_cmd == "ps ax -o rss=,command=":
            return lmd.SSHResult(0, PS_RSS_OUR_OWN_MLX_LM, "")
        return _healthy_ssh_runner(ssh_host, remote_cmd, timeout)

    driver = lmd.Driver(config, ssh=ssh_with_our_own)
    result = driver.preflight_host(config.hosts[0])
    assert result.healthy


def test_preflight_host_foreign_llm_guard_can_be_disabled_per_host():
    config = _fake_config()
    config.hosts[0].foreign_llm_process_patterns = []  # explicit opt-out

    def ssh_with_omlx(ssh_host, remote_cmd, timeout):
        if remote_cmd == "ps ax -o rss=,command=":
            return lmd.SSHResult(0, PS_RSS_OMLX, "")
        return _healthy_ssh_runner(ssh_host, remote_cmd, timeout)

    driver = lmd.Driver(config, ssh=ssh_with_omlx)
    result = driver.preflight_host(config.hosts[0])
    assert result.healthy


def test_preflight_host_foreign_llm_guard_rss_scan_failure_excludes():
    config = _fake_config()

    def ssh_rss_fails(ssh_host, remote_cmd, timeout):
        if remote_cmd == "ps ax -o rss=,command=":
            return lmd.SSHResult(1, "", "connection reset")
        return _healthy_ssh_runner(ssh_host, remote_cmd, timeout)

    driver = lmd.Driver(config, ssh=ssh_rss_fails)
    result = driver.preflight_host(config.hosts[0])
    assert not result.healthy
    assert "rss" in result.excluded_reason


def test_preflight_host_excluded_on_missing_manifest():
    config = _fake_config()

    def no_manifest(ssh_host, remote_cmd, timeout):
        if "MANIFEST.json" in remote_cmd:
            return lmd.SSHResult(1, "", "no such file")
        return _healthy_ssh_runner(ssh_host, remote_cmd, timeout)

    driver = lmd.Driver(config, ssh=no_manifest)
    result = driver.preflight_host(config.hosts[0])
    assert not result.healthy
    assert "not provisioned" in result.excluded_reason


def test_preflight_all_runs_every_host_in_parallel():
    config = _fake_config(n_hosts=3)
    driver = lmd.Driver(config, ssh=_healthy_ssh_runner)
    results = driver.preflight_all(config.hosts)
    assert len(results) == 3
    assert all(r.healthy for r in results)
    assert set(driver.results) == {"h0", "h1", "h2"}


def test_teardown_host_reports_clean_yes():
    config = _fake_config()
    driver = lmd.Driver(config, ssh=_healthy_ssh_runner)
    result = lmd.HostResult(config=config.hosts[0], started=True)
    driver.teardown_host(result)
    assert result.clean is True


def test_teardown_host_reports_clean_no_when_process_lingers():
    config = _fake_config()

    def ssh_leaves_orphan(ssh_host, remote_cmd, timeout):
        if remote_cmd == "ps ax -o command=":
            return lmd.SSHResult(0, "exonit-lora/.venv/bin/mlx_lm.server --model x\n", "")
        return _healthy_ssh_runner(ssh_host, remote_cmd, timeout)

    driver = lmd.Driver(config, ssh=ssh_leaves_orphan)
    result = lmd.HostResult(config=config.hosts[0], started=True)
    driver.teardown_host(result)
    assert result.clean is False


def test_wait_ready_requires_both_healthz_and_explain_probe():
    config = _fake_config()
    result = lmd.HostResult(config=config.hosts[0], ip="10.0.0.0", port=8500, started=True)

    def http_get(url, timeout):
        return 200, json.dumps({"upstream": "ok"})

    def http_post(url, payload, timeout):
        return 200, json.dumps({"explanation": "x", "model": "m", "has_adapter": True})

    driver = lmd.Driver(config, ssh=_healthy_ssh_runner, http_get=http_get, http_post=http_post)
    driver.wait_ready(result, timeout_s=1.0)
    assert result.ready


def test_wait_ready_fails_when_upstream_never_ok():
    config = _fake_config()
    result = lmd.HostResult(config=config.hosts[0], ip="10.0.0.0", port=8500, started=True)

    def http_get(url, timeout):
        return 200, json.dumps({"upstream": "down"})

    driver = lmd.Driver(config, ssh=_healthy_ssh_runner, http_get=http_get)
    driver.wait_ready(result, timeout_s=0.1)
    assert not result.ready


def test_wait_ready_fails_when_explain_probe_errors():
    config = _fake_config()
    result = lmd.HostResult(config=config.hosts[0], ip="10.0.0.0", port=8500, started=True)

    def http_get(url, timeout):
        return 200, json.dumps({"upstream": "ok"})

    def http_post(url, payload, timeout):
        return 502, "bad gateway"

    driver = lmd.Driver(config, ssh=_healthy_ssh_runner, http_get=http_get, http_post=http_post)
    driver.wait_ready(result, timeout_s=1.0)
    assert not result.ready


# ---------------------------------------------------------------------------
# Lock file
# ---------------------------------------------------------------------------

def test_acquire_lock_blocks_second_instance(tmp_path):
    lock_path = tmp_path / "lora_macs.lock"
    with lmd.acquire_lock(lock_path):
        with pytest.raises(lmd.LockHeldError):
            with lmd.acquire_lock(lock_path):
                pass


def test_acquire_lock_reentrant_after_release(tmp_path):
    lock_path = tmp_path / "lora_macs.lock"
    with lmd.acquire_lock(lock_path):
        pass
    with lmd.acquire_lock(lock_path):
        pass  # must not raise: the first lock released cleanly


# ---------------------------------------------------------------------------
# shim start command: relative remote_dir (public default) must not be doubled
# ---------------------------------------------------------------------------

def _run_start_shim_locally(remote_dir, home):
    """Build the real shim start command, then execute it with bash (no network) against a fake uvicorn."""
    import subprocess
    import time
    config = _fake_config()
    config.remote_dir = remote_dir
    captured = []

    def fake_runner(host, cmd, timeout):
        captured.append(cmd)
        return lmd.SSHResult(0, "", "")

    lmd.Driver(config, ssh=fake_runner).start_host(lmd.HostResult(config=config.hosts[0], started=False))
    shim_cmd = captured[1]
    base = Path(remote_dir) if Path(remote_dir).is_absolute() else home / remote_dir
    (base / "shim").mkdir(parents=True)
    (base / "run").mkdir()
    (base / "logs").mkdir()
    (base / ".venv" / "bin").mkdir(parents=True)
    fake = base / ".venv" / "bin" / "uvicorn"
    fake.write_text('#!/bin/sh\necho "PP=$PYTHONPATH ARGS=$*" > "$PWD/run/uvicorn.args"\n')
    fake.chmod(0o755)
    subprocess.run(["bash", "-c", shim_cmd], cwd=home, env={"HOME": str(home), "PATH": "/usr/bin:/bin"},
                   check=False, timeout=20)   # the trailing `echo > run/shim.pid` runs outside the cd (pre-existing; irrelevant here)
    for _ in range(50):
        if (base / "run" / "uvicorn.args").exists():
            break
        time.sleep(0.1)
    return base, (base / "run" / "uvicorn.args").read_text()


def test_start_shim_relative_remote_dir_resolves_under_home(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    base, args = _run_start_shim_locally("exonit-lora", home)
    assert f"PP={base}/shim" in args and f"--app-dir {base}/shim" in args
    assert "exonit-lora/exonit-lora" not in args


def test_start_shim_absolute_remote_dir_unchanged(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    absdir = str(tmp_path / "abs-lora")
    base, args = _run_start_shim_locally(absdir, home)
    assert f"PP={absdir}/shim" in args and f"--app-dir {absdir}/shim" in args
