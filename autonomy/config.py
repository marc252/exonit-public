"""Configuration of the autonomy subsystem (phase 1 — shadow mode).

`load_config` reads an optional YAML and shallow-merges it
(field by field, per section) over the defaults. If the file does not exist
the defaults are returned as is.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class MailConfig:
    enabled: bool = False
    to: str = ""
    smtp_host: str = "smtp.gmail.com"
    smtp_port: int = 587
    user: str = ""
    app_password_file: str = ""


@dataclass
class Thresholds:
    disk_free_gb_warn: float = 25.0
    disk_free_gb_crit: float = 10.0
    ram_avail_gb_crit: float = 2.0
    temp_c_warn: float = 95.0


@dataclass
class TriageLLMConfig:
    """On-demand local AI triage server (phase 1.5 — armed mode).

    `enabled=False` (default) makes `autonomy.triage_llm.spawn_and_choose`
    fall straight through to `pause_and_alert` without trying to start anything —
    same as any other startup failure (missing bin, empty model_path,
    timeout)."""

    enabled: bool = False
    model_path: str = ""
    port: int = 8091
    server_bin: str = "llama-server"
    ctx: int = 8192
    startup_timeout_s: int = 240
    call_timeout_s: int = 120


@dataclass
class AlertConfig:
    """Optional external hook (2026-09-19, unattended run of 1-2 months): `Mailer`
    stays disabled by default (no App Password) -- `command` is an
    alternative channel, with no particular network/SMTP dependencies.

    `command`: argv (`list[str]`) or a shell string (`str`), invoked with
    `subprocess.run(..., timeout=30)` from `autonomy.alerting.run_alert_hook`
    -- never raises, the result is only logged. `None`/empty (default) = no-op.
    It always receives `EXONIT_ALERT_SUBJECT`/`EXONIT_ALERT_BODY` via the environment; if it is
    a list, subject and body are ALSO appended as the last two argv
    (a shell string only receives the environment -- subject/body may
    contain any character, they are not interpolated into the shell).

    `heartbeat_hours`: if > 0, the armed loop invokes this same `command`
    at most once every `heartbeat_hours` with subject `heartbeat` and a
    one-line summary (state, current run_id, lots done, live fleet) -- to
    detect that the autonomous chain is still alive with no human watching.
    `0` (default) = disabled."""

    command: Union[list, str, None] = None
    heartbeat_hours: float = 0.0


@dataclass
class FleetConfig:
    """Fleet preflight thresholds before launching a lot (armed mode)."""

    min_alive: int = 30
    start_wait_s: int = 180


@dataclass
class AutonomyConfig:
    server_url: str = "http://127.0.0.1:8400"
    data_dir: Path = field(default_factory=lambda: REPO_ROOT / "data" / "autonomy")
    poll_interval_s: int = 120
    armed: bool = False
    mail: MailConfig = field(default_factory=MailConfig)
    thresholds: Thresholds = field(default_factory=Thresholds)
    triage_llm: TriageLLMConfig = field(default_factory=TriageLLMConfig)
    fleet: FleetConfig = field(default_factory=FleetConfig)
    alert: AlertConfig = field(default_factory=AlertConfig)


def _merge_section(target, section_dict: dict) -> None:
    """Update `target` (dataclass instance) field by field from `section_dict`."""
    for key, value in section_dict.items():
        if hasattr(target, key):
            setattr(target, key, value)


def load_config(path: Optional[Path]) -> AutonomyConfig:
    """Load the autonomy configuration from an optional YAML.

    If `path` is None or the file does not exist, returns the defaults.
    Top-level fields (`server_url`, `data_dir`, `poll_interval_s`,
    `armed`) override directly; the `mail` and
    `thresholds` sections are merged field by field over their respective defaults.
    """
    cfg = AutonomyConfig()

    if path is None:
        return cfg

    path = Path(path)
    if not path.exists():
        return cfg

    with path.open("r") as f:
        raw = yaml.safe_load(f) or {}

    if "server_url" in raw:
        cfg.server_url = raw["server_url"]
    if "data_dir" in raw:
        cfg.data_dir = Path(raw["data_dir"])
    if "poll_interval_s" in raw:
        cfg.poll_interval_s = raw["poll_interval_s"]
    if "armed" in raw:
        cfg.armed = raw["armed"]

    if isinstance(raw.get("mail"), dict):
        _merge_section(cfg.mail, raw["mail"])
    if isinstance(raw.get("thresholds"), dict):
        _merge_section(cfg.thresholds, raw["thresholds"])
    if isinstance(raw.get("triage_llm"), dict):
        _merge_section(cfg.triage_llm, raw["triage_llm"])
    if isinstance(raw.get("fleet"), dict):
        _merge_section(cfg.fleet, raw["fleet"])
    if isinstance(raw.get("alert"), dict):
        _merge_section(cfg.alert, raw["alert"])

    return cfg
