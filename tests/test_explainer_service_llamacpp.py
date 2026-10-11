"""Tests for scripts/explainer_service_llamacpp.py (Fase 2 LoRA-on-<host> shim).

No network: the upstream llama-server is faked by monkeypatching the module's
httpx.AsyncClient. What matters here:
  - the shim reuses the vLLM service's contract objects (single canonical
    copy — same D9-dedup policy as the system prompt), and
  - /api/explain forwards the exact sampling the vLLM service used
    (temperature 0.7, top_p 0.95, Qwen3 stop tokens) and returns the
    ExplainResponse shape the generator expects.
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS_DIR / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


shim = _load("explainer_service_llamacpp")
vllm_svc = _load("explainer_service_vllm_qwen3")


def test_contract_shared_with_vllm_service():
    """Request model fields, system prompt and stop tokens are the vLLM
    service's own — no second copy that could drift."""
    assert (set(shim.ExplainRequest.model_fields)
            == set(vllm_svc.ExplainRequest.model_fields))
    assert shim.SYSTEM_PROMPT == vllm_svc.SYSTEM_PROMPT
    assert shim.STOP_TOKENS == ["<|im_end|>", "<|endoftext|>"]
    from exonit.explainer_prompts import EXPLAINER_SYSTEM_PROMPT
    assert shim.SYSTEM_PROMPT == EXPLAINER_SYSTEM_PROMPT


class _FakeResponse:
    status_code = 200
    text = ""

    def json(self):
        return {"choices": [{"message": {"content": "  [CA] prova\n\n[EN] test  "}}]}


class _FakeAsyncClient:
    """Stands in for httpx.AsyncClient; records the payload it was sent."""
    last_payload = None

    def __init__(self, timeout=None):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def post(self, url, json=None):
        _FakeAsyncClient.last_payload = json
        return _FakeResponse()

    async def get(self, url):
        return _FakeResponse()


def test_explain_forwards_vllm_sampling_and_strips(monkeypatch):
    monkeypatch.setattr(shim.httpx, "AsyncClient", _FakeAsyncClient)
    client = TestClient(shim.app)
    r = client.post("/api/explain", json={
        "tic_id": "123456789",
        "candidate_num": 1,
        "period_days": 3.4521,
        "depth_ppm": 1250.0,
        "snr": 14.2,
        "cnn_score": 0.912,
        "final_verdict": "strong_candidate",
    })
    assert r.status_code == 200
    body = r.json()
    assert body["explanation"] == "[CA] prova\n\n[EN] test"
    assert body["has_adapter"] is True

    sent = _FakeAsyncClient.last_payload
    assert sent["temperature"] == 0.7
    assert sent["top_p"] == 0.95
    assert sent["stop"] == ["<|im_end|>", "<|endoftext|>"]
    assert sent["messages"][0]["role"] == "system"
    assert sent["messages"][0]["content"] == shim.SYSTEM_PROMPT
    # The user prompt is the canonical builder's output, not a re-implementation.
    req = shim.ExplainRequest(tic_id="123456789", candidate_num=1,
                              period_days=3.4521, depth_ppm=1250.0, snr=14.2,
                              cnn_score=0.912, final_verdict="strong_candidate")
    assert sent["messages"][1]["content"] == shim._build_user_prompt(req)


def test_explain_upstream_down_returns_502(monkeypatch):
    class _DownClient(_FakeAsyncClient):
        async def post(self, url, json=None):
            raise shim.httpx.ConnectError("refused")

    monkeypatch.setattr(shim.httpx, "AsyncClient", _DownClient)
    client = TestClient(shim.app)
    r = client.post("/api/explain", json={"tic_id": "1", "candidate_num": 1})
    assert r.status_code == 502
