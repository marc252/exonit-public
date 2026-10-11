"""Unit tests for scripts/bench/m4_lora_bench_client.py.

Covers the pure helpers (bilingual/truncation heuristics) and run_pool's
stats aggregation, with requests.post monkeypatched — no network, no DB.
See progress/impl-m4-lora-bench-2026-09-25.md for the benchmark this
harness was built for (M4 mini feasibility test, LoRA via llama.cpp).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "bench"))

import m4_lora_bench_client as bench  # noqa: E402


def test_looks_bilingual_true():
    assert bench._looks_bilingual("[CA] hola\n\n[EN] hello")


def test_looks_bilingual_false_missing_en():
    assert not bench._looks_bilingual("[CA] hola nomes")


def test_looks_truncated_clean_ending_not_truncated():
    text = "x" * 2000 + "."
    assert not bench._looks_truncated(text, max_tokens=640)


def test_looks_truncated_no_punctuation_near_budget():
    text = "x" * 2400  # ~0.94 * 640 * 4 chars, no terminal punctuation
    assert bench._looks_truncated(text, max_tokens=640)


def test_looks_truncated_short_text_not_flagged():
    # Short, unfinished text is not "truncated" in the max-tokens sense —
    # it just never got that long.
    assert not bench._looks_truncated("short", max_tokens=640)


def test_looks_truncated_empty_text():
    assert bench._looks_truncated("", max_tokens=640)


def test_run_pool_aggregates_stats(monkeypatch):
    rows = [
        {"id": 1, "tic_id": "1", "candidate_num": 1, "period_days": 1.0,
         "depth_ppm": 100.0, "duration_days": 0.1, "snr": 10.0,
         "cnn_score": 0.8, "triceratops_fpp": None, "triceratops_nfpp": None,
         "triceratops_verdict": None, "exominer_score": None,
         "exominer_ffi_score": None, "claude_verdict": "planet_candidate",
         "tmag": 10.0, "n_sectors": 1, "time_span_days": 27.0,
         "stellar_teff": None, "stellar_logg": None, "stellar_radius": None,
         "stellar_mass": None},
        {"id": 2, "tic_id": "2", "candidate_num": 1, "period_days": 2.0,
         "depth_ppm": 200.0, "duration_days": 0.2, "snr": 12.0,
         "cnn_score": 0.9, "triceratops_fpp": None, "triceratops_nfpp": None,
         "triceratops_verdict": None, "exominer_score": None,
         "exominer_ffi_score": None, "claude_verdict": "planet_candidate",
         "tmag": 11.0, "n_sectors": 2, "time_span_days": 54.0,
         "stellar_teff": None, "stellar_logg": None, "stellar_radius": None,
         "stellar_mass": None},
    ]

    responses = {
        1: {"ok": True, "latency_s": 1.0, "explanation": "[CA] a\n\n[EN] b."},
        2: {"ok": False, "latency_s": 0.5, "error": "http_502: boom"},
    }

    def fake_call_one(url, payload, timeout):
        cid = int(payload["tic_id"])
        return dict(responses[cid])

    monkeypatch.setattr(bench, "call_one", fake_call_one)

    out = bench.run_pool("http://fake/api/explain", rows, concurrency=2,
                          timeout=5.0, max_tokens=640)
    summary = out["summary"]
    assert summary["n_requested"] == 2
    assert summary["n_ok"] == 1
    assert summary["n_failed"] == 1
    assert summary["bilingual_valid_rate"] == 1.0
    assert summary["truncation_rate"] == 0.0
    assert summary["failures"] == ["http_502: boom"]
