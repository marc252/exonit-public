"""Tests for CNN transit classifier."""

import numpy as np
import os
import pytest

# Skip all tests if model doesn't exist
pytestmark = pytest.mark.skipif(
    not os.path.exists(os.path.join(os.path.dirname(__file__),
                                     "..", "exonit", "models", "transit_cnn.onnx")),
    reason="CNN model not trained yet"
)

from cnn import TransitClassifier, N_BINS


@pytest.fixture
def classifier():
    return TransitClassifier()


class TestTransitClassifier:
    def test_planet_gets_high_score(self, classifier):
        """A clear U-shaped transit should get a high planet score."""
        time = np.linspace(0, 60, 5000)
        flux = np.ones(5000)
        # Inject clear transit
        phase = ((time - 5.0) % 10.0) / 10.0
        half = 0.01
        flux[(phase < half) | (phase > 1.0 - half)] -= 0.005
        flux += np.random.normal(0, 0.0005, 5000)

        score = classifier.classify(time, flux, period=10.0, t0=5.0, duration=0.2)
        assert score > 0.5, f"Planet transit should score > 0.5, got {score}"

    def test_noise_gets_low_score(self, classifier):
        """Pure noise should get a low average planet score across multiple realizations."""
        rng = np.random.default_rng(42)
        scores = []
        time = np.linspace(0, 60, 5000)
        for _ in range(10):
            flux = np.ones(5000) + rng.normal(0, 0.001, 5000)
            scores.append(classifier.classify(time, flux, period=10.0, t0=5.0, duration=0.2))
        mean_score = np.mean(scores)
        assert mean_score < 0.5, f"Pure noise mean score should be < 0.5, got {mean_score:.3f}"

    def test_phase_fold_produces_correct_shape(self, classifier):
        """Phase folding should produce N_BINS values."""
        time = np.linspace(0, 60, 5000)
        flux = np.ones(5000)
        result = classifier._phase_fold_and_bin(time, flux, 10.0, 5.0, 0.2)
        assert result.shape == (N_BINS,)
