"""Tests of the local AI triage client (`autonomy/triage_llm.py`).

No real HTTP calls: `urllib.request.urlopen` is always mocked via
monkeypatch. The client never raises exceptions -- any failure (HTTP,
timeout, invalid JSON, invalid schema, invalid playbook, low confidence)
returns a `PlaybookChoice("pause_and_alert", ...)`.
"""

import io
import json

import autonomy.triage_llm as T
from autonomy.types import ErrorContext

CTX = ErrorContext("r1", "FPP_BACKFILL", "FAILED", "x", "y", {}, [])


def _fake(content):
    body = json.dumps({"choices": [{"message": {"content": content}}]}).encode()

    class R(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    return lambda *a, **k: R(body)


def test_valid_choice(monkeypatch):
    monkeypatch.setattr(
        T.request,
        "urlopen",
        _fake('{"playbook": "requeue_orphans", "params": {}, "confidence": 0.9, "reasoning": "ok"}'),
    )
    c = T.TriageClient("http://x", "m").choose(CTX)
    assert c.playbook == "requeue_orphans" and c.source == "llm"


def test_low_confidence_escalates(monkeypatch):
    monkeypatch.setattr(
        T.request,
        "urlopen",
        _fake(
            '{"playbook": "restart_worker", "params": {"worker": "worker-<host>"}, '
            '"confidence": 0.5, "reasoning": "meh"}'
        ),
    )
    assert T.TriageClient("http://x", "m").choose(CTX).playbook == "pause_and_alert"


def test_invalid_playbook_escalates(monkeypatch):
    monkeypatch.setattr(
        T.request,
        "urlopen",
        _fake('{"playbook": "rm_rf", "params": {}, "confidence": 0.99, "reasoning": ""}'),
    )
    assert T.TriageClient("http://x", "m").choose(CTX).playbook == "pause_and_alert"


def test_http_error_escalates(monkeypatch):
    def boom(*a, **k):
        raise OSError("connection refused")

    monkeypatch.setattr(T.request, "urlopen", boom)
    c = T.TriageClient("http://x", "m").choose(CTX)
    assert c.playbook == "pause_and_alert" and c.source == "fallback"


def test_invalid_json_escalates(monkeypatch):
    monkeypatch.setattr(T.request, "urlopen", _fake("this is not json"))
    c = T.TriageClient("http://x", "m").choose(CTX)
    assert c.playbook == "pause_and_alert" and c.source == "fallback"


def test_missing_schema_key_escalates(monkeypatch):
    monkeypatch.setattr(
        T.request,
        "urlopen",
        _fake('{"playbook": "requeue_orphans", "params": {}, "confidence": 0.9}'),
    )
    c = T.TriageClient("http://x", "m").choose(CTX)
    assert c.playbook == "pause_and_alert" and c.source == "fallback"


def test_malformed_params_not_dict_escalates_without_raising(monkeypatch):
    # "params": 1 instead of {} -- passes PlaybookChoice construction
    # (no type-checking) but makes validate() blow up internally (set(1)).
    monkeypatch.setattr(
        T.request,
        "urlopen",
        _fake('{"playbook": "requeue_orphans", "params": 1, "confidence": 0.9, "reasoning": "ok"}'),
    )
    c = T.TriageClient("http://x", "m").choose(CTX)
    assert c.playbook == "pause_and_alert" and c.source == "fallback"


def test_malformed_playbook_not_string_escalates_without_raising(monkeypatch):
    # "playbook": ["x"] instead of a string -- CATALOG.get(["x"]) raises
    # TypeError (unhashable type) inside validate().
    monkeypatch.setattr(
        T.request,
        "urlopen",
        _fake('{"playbook": ["x"], "params": {}, "confidence": 0.9, "reasoning": "ok"}'),
    )
    c = T.TriageClient("http://x", "m").choose(CTX)
    assert c.playbook == "pause_and_alert" and c.source == "fallback"


def test_extra_text_around_json_is_stripped(monkeypatch):
    content = (
        "Here is my answer:\n"
        '{"playbook": "wal_checkpoint", "params": {}, "confidence": 0.85, "reasoning": "ok"}'
        "\nThanks."
    )
    monkeypatch.setattr(T.request, "urlopen", _fake(content))
    c = T.TriageClient("http://x", "m").choose(CTX)
    assert c.playbook == "wal_checkpoint" and c.source == "llm"


def test_valid_pause_and_alert_low_confidence_stays_fallback_reason(monkeypatch):
    # pause_and_alert with low confidence: it already is pause_and_alert, no need to
    # "convert" it, but it must remain a valid choice (source="llm").
    monkeypatch.setattr(
        T.request,
        "urlopen",
        _fake(
            '{"playbook": "pause_and_alert", "params": {"reason": "ambiguous"}, '
            '"confidence": 0.3, "reasoning": "unclear"}'
        ),
    )
    c = T.TriageClient("http://x", "m").choose(CTX)
    assert c.playbook == "pause_and_alert" and c.source == "llm"


def test_posts_to_correct_url_and_payload(monkeypatch):
    captured = {}

    def fake_urlopen(req, timeout=None):
        captured["url"] = req.full_url
        captured["timeout"] = timeout
        payload = json.loads(req.data.decode())
        captured["payload"] = payload
        body = json.dumps(
            {
                "choices": [
                    {
                        "message": {
                            "content": '{"playbook": "requeue_orphans", "params": {}, '
                            '"confidence": 0.9, "reasoning": "ok"}'
                        }
                    }
                ]
            }
        ).encode()

        class R(io.BytesIO):
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        return R(body)

    monkeypatch.setattr(T.request, "urlopen", fake_urlopen)
    T.TriageClient("http://x", "m", timeout_s=42).choose(CTX)

    assert captured["url"] == "http://x/v1/chat/completions"
    assert captured["timeout"] == 42
    assert captured["payload"]["model"] == "m"
    assert captured["payload"]["temperature"] == 0.1
    assert captured["payload"]["messages"][0]["role"] == "system"
    user_content = json.loads(captured["payload"]["messages"][1]["content"])
    assert user_content["phase"] == CTX.phase
    assert user_content["state"] == CTX.state
    assert user_content["detail"] == CTX.detail
    assert user_content["log_tail"] == CTX.log_tail
    assert user_content["fleet"] == CTX.fleet_snapshot
    assert user_content["recent_actions"] == CTX.recent_actions
