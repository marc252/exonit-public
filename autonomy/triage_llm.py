"""Local-AI triage client (phase 1 — shadow mode).

`TriageClient.choose(ctx)` sends the `ErrorContext` to a local OpenAI-compatible
server (e.g. `qwen-llm.service` on <host>) and expects a JSON `PlaybookChoice` back.
The client **never raises exceptions**: any failure (HTTP/network error, timeout,
invalid JSON, incomplete schema, playbook that fails `playbooks.validate()`, or
confidence below the threshold of rule 5 of annex A) is turned into a
`pause_and_alert` with `source="fallback"` (or `source="llm"` when the AI chose
`pause_and_alert` itself with a valid answer).

The system prompt is `autonomy/prompts/triage_system.txt`, copied from
annex A of `docs/superpowers/specs/2026-08-08-autonomous-runs-design.md`
(translated to English for the public export).
"""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import Callable
from urllib import request

from autonomy.config import TriageLLMConfig
from autonomy.playbooks import validate
from autonomy.types import ErrorContext, PlaybookChoice

_PROMPT_PATH = Path(__file__).parent / "prompts" / "triage_system.txt"
_SYSTEM = _PROMPT_PATH.read_text()

_LOG_TAIL_MAX_CHARS = 8000
_CONFIDENCE_THRESHOLD = 0.7
_REQUIRED_KEYS = {"playbook", "params", "confidence", "reasoning"}


def _fallback(reason: str, detail: str) -> PlaybookChoice:
    return PlaybookChoice("pause_and_alert", {"reason": reason}, 0.0, detail, "fallback")


class TriageClient:
    """HTTP client (stdlib `urllib`, no new dependencies) for the local AI triage."""

    def __init__(self, base_url: str, model: str, timeout_s: int = 120):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_s = timeout_s

    def _payload(self, ctx: ErrorContext) -> dict:
        user_payload = {
            "phase": ctx.phase,
            "state": ctx.state,
            "detail": ctx.detail,
            "log_tail": ctx.log_tail[-_LOG_TAIL_MAX_CHARS:],
            "fleet": ctx.fleet_snapshot,
            "recent_actions": ctx.recent_actions,
        }
        return {
            "model": self.model,
            "temperature": 0.1,
            "messages": [
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": json.dumps(user_payload)},
            ],
        }

    def choose(self, ctx: ErrorContext) -> PlaybookChoice:
        """Ask the local AI for a playbook choice for `ctx`. Never raises."""
        try:
            body = json.dumps(self._payload(ctx)).encode()
            req = request.Request(
                f"{self.base_url}/v1/chat/completions",
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with request.urlopen(req, timeout=self.timeout_s) as resp:
                raw = resp.read()
        except Exception as exc:  # noqa: BLE001 - any network/HTTP failure
            return _fallback("network/HTTP error with the local AI", repr(exc))

        try:
            content = json.loads(raw)["choices"][0]["message"]["content"]
        except Exception as exc:  # noqa: BLE001 - response with an unexpected shape
            return _fallback("HTTP response with an unexpected shape", repr(exc))

        try:
            start = content.index("{")
            end = content.rindex("}")
            parsed = json.loads(content[start : end + 1])
        except (ValueError, json.JSONDecodeError) as exc:
            return _fallback("invalid JSON in the AI response", repr(exc))

        if not isinstance(parsed, dict) or not _REQUIRED_KEYS.issubset(parsed):
            return _fallback(
                "invalid schema in the AI response",
                f"keys received: {sorted(parsed) if isinstance(parsed, dict) else type(parsed)}",
            )

        try:
            choice = PlaybookChoice(
                playbook=parsed["playbook"],
                params=parsed["params"],
                confidence=float(parsed["confidence"]),
                reasoning=parsed["reasoning"],
                source="llm",
            )
        except (TypeError, ValueError) as exc:
            return _fallback("invalid types in the AI response", repr(exc))

        try:
            reason = validate(choice)
        except Exception as exc:  # noqa: BLE001 - plausible schema but malformed (e.g. params not a dict, playbook not a string)
            return _fallback("validate() failed on the AI response", repr(exc))
        if reason is not None:
            return _fallback(f"invalid playbook: {reason}", reason)

        if choice.confidence < _CONFIDENCE_THRESHOLD and choice.playbook != "pause_and_alert":
            return PlaybookChoice(
                "pause_and_alert",
                {"reason": f"low confidence ({choice.confidence}) for {choice.playbook}"},
                choice.confidence,
                choice.reasoning,
                "llm",
            )

        return choice


# ---------------------------------------------------------------------------
# On-demand triage server (phase 1.5 — armed mode).
# ---------------------------------------------------------------------------

_HEALTH_POLL_S = 2


def spawn_and_choose(
    ctx: ErrorContext,
    cfg: TriageLLMConfig,
    *,
    popen: Callable[..., "subprocess.Popen"] = subprocess.Popen,
    sleep_fn: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
) -> PlaybookChoice:
    """Start `cfg.server_bin` on demand, make the call with `TriageClient`,
    and ALWAYS kill the process (`finally`) — the triage server is never
    left alive between remediations. **Never raises**: any failure (AI disabled,
    empty `model_path`, missing binary, startup timeout) falls back to
    `pause_and_alert` with `source="fallback"`, same as `TriageClient`.

    `popen`/`sleep_fn`/`monotonic` are injectable for tests only —
    in production they are always `subprocess.Popen`/`time.sleep`/`time.monotonic`.
    """
    if not cfg.enabled:
        return _fallback("local AI triage disabled", "TriageLLMConfig.enabled=False")
    if not cfg.model_path:
        return _fallback("empty model_path", "TriageLLMConfig.model_path not configured")

    cmd = [
        cfg.server_bin, "-m", cfg.model_path, "-ngl", "99", "-c", str(cfg.ctx),
        "--jinja", "--host", "127.0.0.1", "--port", str(cfg.port),
    ]

    proc = None
    try:
        try:
            proc = popen(cmd, start_new_session=True)
        except Exception as exc:  # noqa: BLE001 - missing bin, permissions, etc.
            return _fallback(f"could not start {cfg.server_bin}", repr(exc))

        base_url = f"http://127.0.0.1:{cfg.port}"
        deadline = monotonic() + cfg.startup_timeout_s
        healthy = False
        while monotonic() < deadline:
            try:
                with request.urlopen(f"{base_url}/health", timeout=5) as resp:
                    if getattr(resp, "status", 200) < 400:
                        healthy = True
                        break
            except Exception:  # noqa: BLE001 - still starting up, retried until the timeout
                pass
            sleep_fn(_HEALTH_POLL_S)

        if not healthy:
            return _fallback(
                "the triage server did not start in time",
                f"GET {base_url}/health did not respond within {cfg.startup_timeout_s}s",
            )

        return TriageClient(base_url, "triage", cfg.call_timeout_s).choose(ctx)
    finally:
        if proc is not None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except Exception:  # noqa: BLE001 - process hung after terminate
                proc.kill()
