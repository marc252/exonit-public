"""Shared types of the autonomy subsystem (phase 1 — shadow mode).

All types are simple `@dataclass`es with `to_dict()`/`from_dict()`
based on `dataclasses.asdict` to ease JSON serialization in the shadow
logs and email reports.
"""

from dataclasses import dataclass, field, asdict
from typing import Any, Optional


@dataclass
class ErrorContext:
    """Context captured when the monitor detects a run in a problematic state."""

    run_id: str
    phase: str
    state: str
    detail: str
    log_tail: str
    fleet_snapshot: dict = field(default_factory=dict)
    recent_actions: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ErrorContext":
        return cls(**d)


@dataclass
class PlaybookChoice:
    """Decision (rule, LLM or fallback) on which playbook to apply."""

    playbook: str
    params: dict
    confidence: float
    reasoning: str
    source: str  # rules | llm | fallback

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "PlaybookChoice":
        return cls(**d)


@dataclass
class HostHealth:
    """Health snapshot of a host/worker at a given moment."""

    worker_name: str
    host: str
    reachable: bool
    disk_free_gb: Optional[float] = None
    disk_pct_used: Optional[float] = None
    ram_avail_gb: Optional[float] = None
    temp_c: Optional[float] = None
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "HostHealth":
        return cls(**d)


@dataclass
class Advisory:
    """Advisory generated from a HostHealth outside thresholds."""

    worker_name: str
    kind: str
    severity: str
    suggested_playbook: str
    params: dict
    detail: str

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Advisory":
        return cls(**d)
