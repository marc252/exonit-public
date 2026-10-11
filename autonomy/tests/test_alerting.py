import json
import sys

from autonomy.config import AlertConfig, MailConfig
from autonomy.alerting import Mailer, run_alert_hook


def test_malformed_state_timestamp_does_not_raise(tmp_path):
    # A state file with a non-string value (e.g. written/corrupted by an older
    # version or by hand) must not make send() blow up: an unreadable
    # timestamp is treated as "no previous record" -> not suppressed.
    state_path = tmp_path / "s.json"
    state_path.write_text(json.dumps({"pause": 12345}))
    m = Mailer(MailConfig(enabled=True), state_path, dry_run=True)
    assert m.send("pause", "s", "b") == "dry_run"


def test_disabled(tmp_path):
    m = Mailer(MailConfig(enabled=False), tmp_path / "s.json", dry_run=True)
    assert m.send("pause", "s", "b") == "disabled"


def test_rate_limit_per_type(tmp_path):
    m = Mailer(MailConfig(enabled=True), tmp_path / "s.json", dry_run=True)
    assert m.send("pause", "s", "b") == "dry_run"
    assert m.send("pause", "s2", "b") == "suppressed"
    assert m.send("disk_crit", "s3", "b") == "dry_run"  # different type: passes


# ---------------------------------------------------------------------------
# run_alert_hook (2026-09-19): optional external hook `alert.command`,
# independent of the Mailer (which stays disabled by default). Never raises.
# ---------------------------------------------------------------------------

def test_run_alert_hook_unset_command_is_noop():
    assert run_alert_hook(AlertConfig(command=None), "s", "b") is None


def test_run_alert_hook_list_command_gets_env_and_trailing_argv(tmp_path):
    out_path = tmp_path / "out.json"
    code = (
        "import os, sys, json\n"
        "out = sys.argv[1]\n"
        "data = {'subject_argv': sys.argv[2], 'body_argv': sys.argv[3], "
        "'subject_env': os.environ.get('EXONIT_ALERT_SUBJECT'), "
        "'body_env': os.environ.get('EXONIT_ALERT_BODY')}\n"
        "open(out, 'w').write(json.dumps(data))\n"
    )
    cfg = AlertConfig(command=[sys.executable, "-c", code, str(out_path)])

    result = run_alert_hook(cfg, "test subject", "test body")

    assert result == "rc=0"
    data = json.loads(out_path.read_text())
    assert data == {
        "subject_argv": "test subject",
        "body_argv": "test body",
        "subject_env": "test subject",
        "body_env": "test body",
    }


def test_run_alert_hook_str_command_uses_shell_and_env_only(tmp_path):
    out_path = tmp_path / "out.txt"
    cfg = AlertConfig(command=f'echo "$EXONIT_ALERT_SUBJECT" > "{out_path}"')

    result = run_alert_hook(cfg, "shell subject", "shell body")

    assert result == "rc=0"
    assert out_path.read_text().strip() == "shell subject"


def test_run_alert_hook_command_failure_never_raises():
    cfg = AlertConfig(command=[sys.executable, "-c", "import sys; sys.exit(7)"])

    result = run_alert_hook(cfg, "s", "b")

    assert result == "rc=7"


def test_run_alert_hook_missing_binary_swallows_exception():
    cfg = AlertConfig(command=["/definitely/not/a/real/binary-xyz"])

    result = run_alert_hook(cfg, "s", "b")

    assert result is not None and result.startswith("error=")
