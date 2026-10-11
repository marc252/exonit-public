"""Tests for Claude API integration."""

import pytest
from unittest.mock import patch, MagicMock
from exonit.claude_analysis import (
    should_analyze, select_model, _build_analysis_prompt, analyze_candidate
)


class TestShouldAnalyze:
    def test_strong_passing_candidate(self):
        candidate = {"snr": 15.0, "period_days": 5.0}
        vetting = {"passed": True}
        assert should_analyze(candidate, vetting, snr_threshold=10.0) is True

    def test_weak_candidate_rejected(self):
        candidate = {"snr": 8.0, "period_days": 5.0}
        vetting = {"passed": True}
        assert should_analyze(candidate, vetting, snr_threshold=10.0) is False

    def test_failed_vetting_rejected(self):
        candidate = {"snr": 20.0, "period_days": 5.0}
        vetting = {"passed": False}
        assert should_analyze(candidate, vetting) is False

    def test_low_cnn_score_rejected(self):
        candidate = {"snr": 15.0}
        vetting = {"passed": True}
        assert should_analyze(candidate, vetting, cnn_score=0.3, cnn_threshold=0.7) is False

    def test_high_cnn_score_accepted(self):
        candidate = {"snr": 15.0}
        vetting = {"passed": True}
        assert should_analyze(candidate, vetting, cnn_score=0.9, cnn_threshold=0.7) is True


class TestSelectModel:
    def test_strong_candidate_uses_sonnet(self):
        candidate = {"snr": 20.0}
        model = select_model(candidate)
        assert "sonnet" in model

    def test_normal_candidate_uses_haiku(self):
        candidate = {"snr": 12.0}
        model = select_model(candidate)
        assert "haiku" in model


class TestBuildPrompt:
    def test_includes_candidate_data(self):
        candidate = {"period_days": 5.0, "depth_ppm": 500, "snr": 12.0,
                      "duration_days": 0.1, "n_transits_expected": 10}
        vetting = {"is_harmonic": False, "odd_even_consistent": True,
                    "is_v_shaped": False, "passed": True}
        star = {"tic_id": "123456", "tmag": 10.5, "ra": 180.0, "dec": -30.0}

        prompt = _build_analysis_prompt(candidate, vetting, star)
        assert "123456" in prompt
        assert "5.0" in prompt
        assert "500" in prompt


class TestAnalyzeCandidate:
    def test_returns_none_without_api_key(self):
        """Should gracefully return None if no API key."""
        candidate = {"period_days": 5.0, "depth_ppm": 500, "snr": 12.0}
        vetting = {"passed": True}

        with patch.dict("os.environ", {}, clear=True):
            result = analyze_candidate(candidate, vetting, api_key=None)
        assert result is None

    def test_parses_tool_use_response(self):
        """Should extract structured result from Tool Use response."""
        import sys

        mock_anthropic = MagicMock()
        mock_client = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client
        mock_response = MagicMock()
        # Simulate a tool_use content block
        tool_block = MagicMock()
        tool_block.type = "tool_use"
        tool_block.name = "submit_analysis"
        tool_block.input = {
            "verdict": "planet_candidate",
            "confidence": 0.8,
            "analysis": "Strong transit signal consistent with a sub-Neptune.",
            "physical_plausibility": "Plausible",
            "known_matches": "None found",
            "follow_up": ["RV confirmation"],
            "ctoi_summary": "Promising candidate.",
        }
        mock_response.content = [tool_block]
        mock_client.messages.create.return_value = mock_response

        candidate = {"period_days": 5.0, "depth_ppm": 500, "snr": 12.0,
                      "duration_days": 0.1, "n_transits_expected": 10}
        vetting = {"passed": True, "is_harmonic": False,
                    "odd_even_consistent": True, "is_v_shaped": False}

        with patch.dict(sys.modules, {"anthropic": mock_anthropic}):
            result = analyze_candidate(candidate, vetting, api_key="test-key")
        assert result["verdict"] == "planet_candidate"
        assert result["confidence"] == 0.8
        assert "model_used" in result
