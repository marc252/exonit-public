"""Email alerting with rate limiting (phase 1 — shadow mode).

`Mailer.send()` is the single entry point for notifying events of the
autonomy subsystem. It applies a per-`event_type` rate limit (1 email/hour/type)
using a JSON state file persisted on disk, and supports a `dry_run` mode
that logs what would have been sent without opening any SMTP connection —
useful for phase 1 (shadow mode) and for tests (which must never open real
sockets).

CLI: `python -m autonomy.alerting --event ... --subject ... --body ...`,
used by the Task 9 watchdog.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import smtplib
import subprocess
from email.message import EmailMessage
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from autonomy.config import AlertConfig, MailConfig, load_config

_LOG = logging.getLogger("autonomy.mail")
_SUBJECT_PREFIX = "[ExoNit autonomous] "
_RATE_LIMIT_S = 3600


class Mailer:
    """Sends email alerts with a per-`event_type` rate limit.

    `state_path` is a JSON file `{event_type: iso_ts_last_send}`
    that persists the rate limit across invocations (the CLI runs as a
    separate process each time).
    """

    def __init__(self, cfg: MailConfig, state_path: Path, dry_run: bool = True):
        self.cfg = cfg
        self.state_path = Path(state_path)
        self.dry_run = dry_run

    def _load_state(self) -> dict:
        if not self.state_path.exists():
            return {}
        try:
            return json.loads(self.state_path.read_text())
        except (json.JSONDecodeError, OSError):
            return {}

    def _save_state(self, state: dict) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(json.dumps(state))

    def _is_rate_limited(self, state: dict, event_type: str) -> bool:
        last = state.get(event_type)
        if last is None:
            return False
        try:
            last_dt = datetime.fromisoformat(last)
        except (ValueError, TypeError):
            return False
        elapsed = (datetime.now(timezone.utc) - last_dt).total_seconds()
        return elapsed < _RATE_LIMIT_S

    def _send_smtp(self, subject: str, body: str) -> None:
        password = Path(self.cfg.app_password_file).read_text().strip()
        msg = EmailMessage()
        msg["Subject"] = subject
        msg["From"] = self.cfg.user
        msg["To"] = self.cfg.to
        msg.set_content(body)

        with smtplib.SMTP(self.cfg.smtp_host, self.cfg.smtp_port) as smtp:
            smtp.starttls()
            smtp.login(self.cfg.user, password)
            smtp.send_message(msg)

    def send(self, event_type: str, subject: str, body: str) -> str:
        """Send (or simulate) an alert for `event_type`. Never raises.

        Retorna: `disabled` | `suppressed` | `dry_run` | `sent`.
        """
        if not self.cfg.enabled:
            return "disabled"

        state = self._load_state()
        if self._is_rate_limited(state, event_type):
            return "suppressed"

        full_subject = _SUBJECT_PREFIX + subject

        if self.dry_run:
            _LOG.info("[MAIL DRY-RUN] %s: %s", event_type, full_subject)
            state[event_type] = datetime.now(timezone.utc).isoformat()
            self._save_state(state)
            return "dry_run"

        self._send_smtp(full_subject, body)
        state[event_type] = datetime.now(timezone.utc).isoformat()
        self._save_state(state)
        return "sent"


def run_alert_hook(cfg: AlertConfig, subject: str, body: str) -> Optional[str]:
    """Invoke the optional external hook `alert.command` (2026-09-19, unattended
    run of 1-2 months) with the subject/body of an alert. Independent of the
    `Mailer` (which stays disabled by default, no App Password) --
    this path depends on neither SMTP nor any particular service; the operator
    plugs in whatever they want (`bin/exonit-alert` placeholder by default).

    NEVER raises: any failure (missing command, not executable, timeout)
    is logged and returned as a result, not as an exception -- this hook
    must never bring down the armed loop. Returns `None` if `cfg.command`
    is not configured (no-op, default)."""
    command = cfg.command
    if not command:
        return None

    env = dict(os.environ, EXONIT_ALERT_SUBJECT=subject, EXONIT_ALERT_BODY=body)

    try:
        if isinstance(command, str):
            proc = subprocess.run(command, shell=True, env=env, capture_output=True, text=True, timeout=30)
        else:
            argv = list(command) + [subject, body]
            proc = subprocess.run(argv, env=env, capture_output=True, text=True, timeout=30)
        _LOG.info("alert.command executed (rc=%s): %s", proc.returncode, command)
        return f"rc={proc.returncode}"
    except Exception as exc:  # noqa: BLE001 - the hook must never bring down the armed loop
        _LOG.warning("alert.command failed: %r", exc)
        return f"error={exc!r}"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Send an email alert from the autonomy subsystem.")
    parser.add_argument("--event", required=True, help="event_type (p.ex. supervisor_dead)")
    parser.add_argument("--subject", required=True, help="subject (without the prefix, which is added automatically)")
    parser.add_argument("--body", required=True, help="message body")
    parser.add_argument(
        "--config",
        default=None,
        help="path to the YAML config (default: data/autonomy/config.yaml)",
    )
    return parser


def main(argv: Optional[list] = None) -> str:
    logging.basicConfig(level=logging.INFO)
    parser = _build_parser()
    args = parser.parse_args(argv)

    from autonomy.config import REPO_ROOT

    config_path = Path(args.config) if args.config else REPO_ROOT / "data" / "autonomy" / "config.yaml"
    cfg = load_config(config_path)
    state_path = cfg.data_dir / "mail_state.json"
    dry_run = not cfg.mail.enabled

    mailer = Mailer(cfg.mail, state_path, dry_run=dry_run)
    result = mailer.send(args.event, args.subject, args.body)
    _LOG.info("result: %s", result)
    return result


if __name__ == "__main__":
    main()
