from pathlib import Path
from autonomy.config import load_config, AutonomyConfig

def test_defaults_without_file(tmp_path):
    cfg = load_config(tmp_path / "absent.yaml")
    assert cfg.server_url == "http://127.0.0.1:8400"
    assert cfg.armed is False
    assert cfg.mail.enabled is False
    assert cfg.thresholds.disk_free_gb_crit == 10.0

def test_yaml_overrides(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("poll_interval_s: 30\nmail:\n  enabled: true\n  user: x@y.z\n")
    cfg = load_config(p)
    assert cfg.poll_interval_s == 30
    assert cfg.mail.enabled is True and cfg.mail.user == "x@y.z"
    assert cfg.server_url == "http://127.0.0.1:8400"  # default preserved

def test_types_roundtrip():
    from autonomy.types import PlaybookChoice
    c = PlaybookChoice("pause_and_alert", {"reason": "x"}, 1.0, "r", "rules")
    assert c.to_dict()["playbook"] == "pause_and_alert"


def test_alert_defaults_are_noop():
    cfg = load_config(Path("/nonexistent/config.yaml"))
    assert cfg.alert.command is None
    assert cfg.alert.heartbeat_hours == 0.0


def test_alert_yaml_overrides_list_command_and_heartbeat_hours(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text(
        "alert:\n"
        "  command: [\"bin/exonit-alert\"]\n"
        "  heartbeat_hours: 6\n"
    )
    cfg = load_config(p)
    assert cfg.alert.command == ["bin/exonit-alert"]
    assert cfg.alert.heartbeat_hours == 6


def test_alert_yaml_overrides_shell_string_command(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text('alert:\n  command: "bin/exonit-alert"\n')
    cfg = load_config(p)
    assert cfg.alert.command == "bin/exonit-alert"
